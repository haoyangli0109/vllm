# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3.1 DSpark draft model.

The Qwen3 DSpark block drafter (context-KV precompute, non-causal query block,
Markov and confidence heads) built from MiniMax-M3 layers: Gemma RMSNorm
(``x * (1 + w)``) everywhere, partial RoPE, and a dense SwiGLU-OAI MLP. The
checkpoint ships no token embedding or LM head; both are shared with the target.
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3Attention,
    DFlashQwen3DecoderLayer,
    _resolve_layer_attention,
)
from vllm.model_executor.models.qwen3_dspark import (
    Qwen3DSparkForCausalLM,
    Qwen3DSparkModel,
)
from vllm.model_executor.models.utils import WeightsMapper
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3TextConfig

_CHECKPOINT_PREFIX = "language_model.model.dspark."


def _gemma_rms_norm(hidden_size: int, eps: float) -> nn.Module:
    """The target's Gemma RMSNorm (fused FlashInfer kernels on CUDA).

    vLLM's GemmaRMSNorm hands the IR op an fp32 ``1 + w`` weight, which the
    vllm_c kernel rejects for bf16 inputs, so every call ran the unfused native
    path (about ten kernels).
    """
    if current_platform.is_rocm():
        from vllm.models.minimax_m3.amd.model import MiniMAXGemmaRMSNorm
    else:
        from vllm.models.minimax_m3.nvidia.model import MiniMAXGemmaRMSNorm
    return MiniMAXGemmaRMSNorm(hidden_size, eps=eps)


class MiniMaxM3DSparkAttention(DFlashQwen3Attention):
    def __init__(
        self,
        config: MiniMaxM3TextConfig,
        sliding_window: int | None,
        causal: bool,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters={
                "rope_theta": config.rope_theta,
                "partial_rotary_factor": config.partial_rotary_factor,
            },
            max_position=config.max_position_embeddings,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            sliding_window=sliding_window,
            causal=causal,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.q_norm = _gemma_rms_norm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = _gemma_rms_norm(self.head_dim, eps=config.rms_norm_eps)


class MiniMaxM3DSparkMLP(nn.Module):
    """Dense SwiGLU-OAI MLP: gate * sigmoid(alpha * gate) * (up + beta)."""

    def __init__(
        self,
        config: MiniMaxM3TextConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        if config.hidden_act != "swigluoai":
            raise ValueError(
                f"Unsupported MiniMax DSpark activation {config.hidden_act!r}; "
                "only swigluoai is supported."
            )
        intermediate_size = config.dspark_ffn_hidden_size
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMulWithClamp(
            swiglu_limit=config.swiglu_limit,
            alpha=config.swiglu_alpha,
            beta=config.swiglu_beta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x, _ = self.down_proj(self.act_fn(gate_up))
        return x


class MiniMaxM3DSparkDecoderLayer(DFlashQwen3DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config: MiniMaxM3TextConfig,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        if not getattr(config, "dspark_use_dense_ffn", False):
            raise NotImplementedError("MiniMax DSpark supports dense FFN drafts only.")
        self.hidden_size = config.hidden_size
        sliding_window, causal = _resolve_layer_attention(config, layer_idx)
        self.self_attn = MiniMaxM3DSparkAttention(
            config,
            sliding_window=sliding_window,
            causal=causal,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = MiniMaxM3DSparkMLP(config, quant_config, prefix=f"{prefix}.mlp")
        hidden_size, eps = config.hidden_size, config.rms_norm_eps
        self.input_layernorm = _gemma_rms_norm(hidden_size, eps=eps)
        self.post_attention_layernorm = _gemma_rms_norm(hidden_size, eps=eps)


class MiniMaxM3DSparkModel(Qwen3DSparkModel):
    decoder_layer_cls = MiniMaxM3DSparkDecoderLayer
    quant_config_mapper = WeightsMapper(
        orig_to_new_prefix={_CHECKPOINT_PREFIX: "model."}
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        hidden_size, eps = self.config.hidden_size, self.config.rms_norm_eps
        self.hidden_norm = _gemma_rms_norm(hidden_size, eps=eps)
        self.norm = _gemma_rms_norm(hidden_size, eps=eps)

    def _normalize_context_states(self, context_states: torch.Tensor) -> torch.Tensor:
        return self.hidden_norm(context_states)

    def _normalize_context_k(self, all_k: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [layer.self_attn.k_norm(k) for layer, k in zip(self.layers, all_k)]
        )


class MiniMaxM3DSparkForCausalLM(Qwen3DSparkForCausalLM):
    model_cls = MiniMaxM3DSparkModel

    @staticmethod
    def _rename_checkpoint_weights(
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for name, tensor in weights:
            name = name.removeprefix(_CHECKPOINT_PREFIX)
            name = name.replace(".decoder_layer.", ".")
            if name.startswith("final_norm."):
                name = "norm." + name.removeprefix("final_norm.")
            # MiniMax stores the MXFP8 E8M0 block scales as weight_scale_inv.
            name = name.replace(".weight_scale_inv", ".weight_scale")
            yield name, tensor

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        return super().load_weights(self._rename_checkpoint_weights(weights))
