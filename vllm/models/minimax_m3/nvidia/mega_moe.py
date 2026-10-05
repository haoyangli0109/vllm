# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepGEMM NVFP4 MegaMoE for MiniMax-M3.1 checkpoints.

MiniMax-M3.1 stores routed experts as compressed-tensors NVFP4 and the shared
expert as MXFP8. ``deep_gemm.nvfp4_mega_moe`` runs dispatch, the routed and
shared SwiGLU-OAI MLPs and combine in one persistent kernel, so this module
owns both sets of weights in checkpoint layout and converts them once after
loading.
"""

import torch
from torch import nn

from vllm.distributed import get_ep_group
from vllm.logger import init_logger
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.minimax_m3.nvidia.ops.prepare_nvfp4_megamoe import (
    prepare_nvfp4_megamoe_inputs,
)
from vllm.utils.deep_gemm import _import_deep_gemm

logger = init_logger(__name__)

# MiniMax-M3.1 QAT quantizes the FC2 input with a fixed outer encode scale of
# 16; checkpoints carry a placeholder ``w2.input_global_scale`` of 1.0.
MINIMAX_M3_FC2_INPUT_GLOBAL_SCALE = 16.0


def make_minimax_m3_mega_moe_expert_params_mapping(
    num_experts: int,
) -> list[tuple[str, str, int, str]]:
    """Map ``experts.{e}.w{1,2,3}.*`` to the MegaMoE ``w13_*``/``w2_*`` params."""
    return [
        (
            "experts.w2_" if shard_id == "w2" else "experts.w13_",
            f"experts.{expert_id}.{shard_id}.",
            expert_id,
            shard_id,
        )
        for expert_id in range(num_experts)
        for shard_id in ("w1", "w2", "w3")
    ]


class MiniMaxM3MegaMoESharedLinear(nn.Module):
    """Replicated MXFP8 linear kept in checkpoint layout (E4M3 weight plus
    per-32 UE8M0 scales). No quant method post-processes it; MegaMoE consumes
    it directly."""

    def __init__(self, output_sizes: list[int], input_size: int) -> None:
        super().__init__()
        self.output_sizes = output_sizes
        output_size = sum(output_sizes)
        self.weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.empty(output_size, input_size // 32, dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})
        # 127 is UE8M0 1.0, so dummy loads produce finite scales.
        set_weight_attrs(
            self.weight_scale,
            {"weight_loader": self.weight_loader, "dummy_weight_value": 127},
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: int | None = None,
    ) -> None:
        data = param.data
        if shard_id is not None:
            offset = sum(self.output_sizes[:shard_id])
            data = data.narrow(0, offset, self.output_sizes[shard_id])
        if loaded_weight.dtype != data.dtype:
            # e.g. UE8M0 scales stored as float8_e8m0fnu: same bits as uint8.
            loaded_weight = loaded_weight.view(data.dtype)
        if data.shape != loaded_weight.shape:
            raise ValueError(
                f"MiniMax M3 MegaMoE shared expert shape mismatch: parameter "
                f"{tuple(data.shape)} vs checkpoint {tuple(loaded_weight.shape)}"
            )
        data.copy_(loaded_weight)


class MiniMaxM3MegaMoESharedExpert(nn.Module):
    """Shared-expert weights for MegaMoE, named like ``MiniMaxM3MLP``."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_up_proj = MiniMaxM3MegaMoESharedLinear(
            [intermediate_size] * 2, hidden_size
        )
        self.down_proj = MiniMaxM3MegaMoESharedLinear([hidden_size], intermediate_size)


class MiniMaxM3NvFp4MegaMoEExperts(nn.Module):
    """NVFP4 routed experts (EP-sharded) with a fused MXFP8 shared expert."""

    _symm_buffer_cache: dict[tuple[int, ...], object] = {}

    def __init__(
        self,
        *,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        num_shared_experts: int,
        max_num_tokens: int,
        activation_clamp: float,
        activation_alpha: float,
        activation_beta: float,
    ) -> None:
        super().__init__()
        ep_group = get_ep_group()
        if num_experts % ep_group.world_size != 0:
            raise ValueError(
                f"MiniMax M3 MegaMoE needs num_experts={num_experts} divisible "
                f"by the EP size {ep_group.world_size}."
            )
        self.ep_size = ep_group.world_size
        self.num_experts = num_experts
        self.num_local_experts = num_experts // self.ep_size
        self.experts_start_idx = ep_group.rank_in_group * self.num_local_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_shared_experts = num_shared_experts
        self.max_num_tokens = max_num_tokens
        self.activation_clamp = activation_clamp
        self.activation_alpha = activation_alpha
        self.activation_beta = activation_beta

        e, h, i = self.num_local_experts, hidden_size, intermediate_size
        self._register("w13_weight_packed", (e, 2 * i, h // 2), torch.uint8)
        self._register("w13_weight_scale", (e, 2 * i, h // 16), torch.float8_e4m3fn)
        self._register("w13_weight_global_scale", (e, 2), torch.float32)
        self._register("w2_weight_packed", (e, h, i // 2), torch.uint8)
        self._register("w2_weight_scale", (e, h, i // 16), torch.float8_e4m3fn)
        self._register("w2_weight_global_scale", (e,), torch.float32)
        # Loaded for checkpoint compatibility only: the routed input carries a
        # per-token scale and the FC2 input uses the fixed QAT scale.
        self._register("w13_input_global_scale", (num_experts, 2), torch.float32)
        self._register("w2_input_global_scale", (num_experts,), torch.float32)

        self._l1_weights: tuple[torch.Tensor, torch.Tensor] | None = None
        self._l2_weights: tuple[torch.Tensor, torch.Tensor] | None = None
        self._shared_l1_weights: tuple[torch.Tensor, torch.Tensor] | None = None
        self._shared_l2_weights: tuple[torch.Tensor, torch.Tensor] | None = None
        self._l1_alpha: torch.Tensor | None = None
        self._l2_alpha: torch.Tensor | None = None
        self._l2_activation_scale = 1.0 / MINIMAX_M3_FC2_INPUT_GLOBAL_SCALE

    def _register(self, name: str, shape: tuple[int, ...], dtype: torch.dtype):
        param = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
        attrs: dict[str, object] = {"weight_loader": self.weight_loader}
        if "global_scale" in name:
            attrs["dummy_weight_value"] = 1.0
        set_weight_attrs(param, attrs)
        self.register_parameter(name, param)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"Unsupported expert shard id: {shard_id}")
        if "input_global_scale" in weight_name:
            data = param.data[expert_id]
        else:
            local_expert_id = expert_id - self.experts_start_idx
            if not 0 <= local_expert_id < self.num_local_experts:
                return
            data = param.data[local_expert_id]

        if "global_scale" in weight_name:
            if shard_id != "w2":
                data = data[0 if shard_id == "w1" else 1]
            data.copy_(loaded_weight.reshape(()))
            return
        if shard_id != "w2":
            offset = 0 if shard_id == "w1" else self.intermediate_size
            data = data.narrow(0, offset, self.intermediate_size)
        if data.shape != loaded_weight.shape:
            raise ValueError(
                f"MiniMax M3 MegaMoE expert shape mismatch for {weight_name}: "
                f"parameter {tuple(data.shape)} vs checkpoint "
                f"{tuple(loaded_weight.shape)}"
            )
        data.copy_(loaded_weight)

    def _check_runtime_supported(self) -> None:
        if torch.cuda.get_device_capability(self.w13_weight_packed.device)[0] != 10:
            raise NotImplementedError("DeepGEMM NVFP4 MegaMoE requires SM100 GPUs.")
        if self.hidden_size % 128 != 0 or self.intermediate_size % 128 != 0:
            raise ValueError(
                "DeepGEMM NVFP4 MegaMoE requires hidden and intermediate sizes "
                "to be multiples of 128."
            )

    @staticmethod
    def _nvfp4_sf_for_mega_moe(sf: torch.Tensor) -> torch.Tensor:
        """``[E, N, K/16]`` E4M3 -> int32 ``[E, N, K/64]`` with MN-major strides."""
        return sf.view(torch.int32).transpose(1, 2).contiguous().transpose(1, 2)

    @staticmethod
    def _mxfp8_sf_for_mega_moe(deep_gemm, sf: torch.Tensor) -> torch.Tensor:
        """``[N, K/32]`` UE8M0 bytes -> packed int32 ``[N, K/128]``, MN-major."""
        mn, k = sf.shape[0], sf.shape[1] * 32
        sf_fp32 = (sf.to(torch.int32) << 23).view(torch.float32)
        return deep_gemm.transform_sf_into_required_layout(
            sf_fp32.unsqueeze(0), mn, k, (1, 32), 1
        ).squeeze(0)

    def finalize_weights(
        self, shared_experts: MiniMaxM3MegaMoESharedExpert | None
    ) -> None:
        """Convert the loaded checkpoint tensors into MegaMoE layouts."""
        if self._l1_weights is not None:
            return
        self._check_runtime_supported()
        deep_gemm = _import_deep_gemm()
        if deep_gemm is None or not hasattr(deep_gemm, "nvfp4_mega_moe"):
            raise RuntimeError(
                "MiniMax M3 MegaMoE requires a DeepGEMM build with "
                "nvfp4_mega_moe. Rebuild the vendored _deep_gemm_C extension."
            )
        if (shared_experts is None) != (self.num_shared_experts == 0):
            raise ValueError("MiniMax M3 MegaMoE shared expert weights mismatch.")

        # Compressed-tensors global scales are quantization scales; the
        # kernel wants dequantization scales.
        w13_global_scale = self.w13_weight_global_scale.data
        if not torch.equal(w13_global_scale[:, 0], w13_global_scale[:, 1]):
            logger.warning_once(
                "w1_weight_global_scale must match w3_weight_global_scale. "
                "Accuracy may be affected."
            )
        self._l1_alpha = (1.0 / w13_global_scale[:, 0]).to(torch.float32)
        self._l2_alpha = (
            self._l2_activation_scale / self.w2_weight_global_scale.data
        ).to(torch.float32)

        self._l1_weights, self._l2_weights = deep_gemm.transform_weights_for_mega_moe(
            (
                self.w13_weight_packed.data.view(torch.int8),
                self._nvfp4_sf_for_mega_moe(self.w13_weight_scale.data),
            ),
            (
                self.w2_weight_packed.data.view(torch.int8),
                self._nvfp4_sf_for_mega_moe(self.w2_weight_scale.data),
            ),
        )
        if shared_experts is not None:
            gate_up = shared_experts.gate_up_proj
            down = shared_experts.down_proj
            self._shared_l1_weights, self._shared_l2_weights = (
                deep_gemm.transform_weights_for_mega_moe(
                    (
                        gate_up.weight.data,
                        self._mxfp8_sf_for_mega_moe(
                            deep_gemm, gate_up.weight_scale.data
                        ),
                    ),
                    (
                        down.weight.data,
                        self._mxfp8_sf_for_mega_moe(deep_gemm, down.weight_scale.data),
                    ),
                )
            )

        # The transformed tensors are fresh copies, except the L2 weight which
        # aliases (and keeps alive) the loaded storage.
        for name in (
            "w13_weight_packed",
            "w13_weight_scale",
            "w13_weight_global_scale",
            "w13_input_global_scale",
            "w2_weight_packed",
            "w2_weight_scale",
            "w2_weight_global_scale",
            "w2_input_global_scale",
        ):
            delattr(self, name)

    def get_symm_buffer(self):
        deep_gemm = _import_deep_gemm()
        group = get_ep_group().device_group
        key = (
            id(group),
            torch.accelerator.current_device_index(),
            self.num_experts,
            self.max_num_tokens,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            self.num_shared_experts,
        )
        symm_buffer = self._symm_buffer_cache.get(key)
        if symm_buffer is None:
            symm_buffer = deep_gemm.NVFP4SymmBuffer(
                group,
                self.num_experts,
                self.max_num_tokens,
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
                num_shared_experts=self.num_shared_experts,
                shared_dtype=torch.float8_e4m3fn,
            )
            self._symm_buffer_cache[key] = symm_buffer
        return symm_buffer

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        is_padding: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the fused MoE layer, shared expert included.

        Args:
            hidden_states: BF16 ``[M, H]`` tokens owned by this rank.
            topk_weights: ``[M, top_k]`` routing weights (scaling applied).
            topk_ids: ``[M, top_k]`` global expert ids.
            is_padding: Optional ``[M]`` mask of tokens to route nowhere.
            out: Optional contiguous BF16 ``[M, H]`` output buffer.

        Returns:
            The ``[M, H]`` layer output.

        """
        num_tokens = hidden_states.shape[0]
        if num_tokens > self.max_num_tokens:
            raise ValueError(
                f"MiniMax M3 MegaMoE got {num_tokens} tokens, but the symmetric "
                f"buffer was sized for {self.max_num_tokens}."
            )
        assert self._l1_weights is not None and self._l2_weights is not None, (
            "MegaMoE weights must be finalized after loading."
        )
        if out is None:
            out = torch.empty_like(hidden_states, dtype=torch.bfloat16)

        deep_gemm = _import_deep_gemm()
        symm_buffer = self.get_symm_buffer()
        shared_block_m = deep_gemm.get_block_m_for_nvfp4_mega_moe(
            self.ep_size,
            self.num_experts,
            symm_buffer.num_max_tokens_per_rank,
            num_tokens,
            self.top_k,
        )
        prepare_nvfp4_megamoe_inputs(
            hidden_states,
            topk_weights,
            topk_ids,
            symm_buffer.x[:num_tokens],
            symm_buffer.x_sf[:num_tokens],
            symm_buffer.x_scales[:num_tokens],
            symm_buffer.shared_l1_acts[:num_tokens],
            symm_buffer.shared_l1_acts_sf,
            symm_buffer.topk_idx[:num_tokens],
            symm_buffer.topk_weights[:num_tokens],
            shared_block_m=shared_block_m,
            is_padding=is_padding,
        )
        deep_gemm.nvfp4_mega_moe(
            out,
            self._l1_weights,
            self._l2_weights,
            symm_buffer,
            shared_l1_weights=self._shared_l1_weights,
            shared_l2_weights=self._shared_l2_weights,
            activation_clamp=self.activation_clamp,
            activation_alpha=self.activation_alpha,
            activation_beta=self.activation_beta,
            l1_alpha=self._l1_alpha,
            l2_alpha=self._l2_alpha,
            l2_activation_scale=self._l2_activation_scale,
            use_x_scales=True,
        )
        return out
