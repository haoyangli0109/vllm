# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MoE on MXFP4 with macro block scaling (MBS), W4A4, in Triton.

Both GEMMs run the scaled MFMA (`tl.dot_scaled`, e2m1 x e2m1) one macro block
at a time. Entering macro block kb the accumulator is multiplied by
f[kb] / f[kb - 1] of each side that carries factors, and the epilogue by
1 / f of the last block, so one accumulator holds the sum of every block's
partial product divided by f_a * f_b
(see `vllm.model_executor.layers.quantization.utils.mxfp4_mbs_utils`).

Weights are row-major per expert: codes [E, N, K/2], E8M0 scales
[E, N, K/32], and, with weight factors, the mantissas transposed to
[E, K/MB, N]. Activations are quantized per row with static factors (MBS-S)
before each GEMM, their mantissas row-major [M, K/MB]. The kernel forms both
sides' ratios from the mantissas with correctly rounded divisions, which
reproduces precomputed fp32 ratios bit for bit at a quarter of their memory.
"""

from dataclasses import dataclass

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import (
    ApplyMoEActivationConfig,
    MoEActivation,
    apply_moe_activation,
    apply_moe_activation_supported,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe import _prepare_expert_assignment
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.mxfp4_mbs_utils import (
    mbs_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    is_mxfp4_mbs_key,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@dataclass
class Mxfp4MbsWeights:
    """One MoE weight (w13 or w2) of all experts in the kernel's layout."""

    q: torch.Tensor  # uint8 [E, N, K/2]
    s: torch.Tensor  # uint8 [E, N, K/32], E8M0
    m: torch.Tensor | None  # uint8 [E, K/MB, N], None without weight factors


def quantize_moe_weight_mxfp4_mbs(
    weight: torch.Tensor, mb: int, ncand: int, use_mbs: bool
) -> Mxfp4MbsWeights:
    """Quantize expert weights [E, N, K] into the kernel layout."""
    E, N, K = weight.shape
    q, s, m = mbs_quantize(
        weight.reshape(E * N, K), mb=mb, ncand=ncand, use_mbs=use_mbs
    )
    q, s = q.view(E, N, K // 2), s.view(E, N, K // 32)
    if m is not None:
        m = m.view(E, N, K // mb).transpose(1, 2).contiguous()
    return Mxfp4MbsWeights(q, s, m)


@triton.jit
def _mbs_factor(m_ptrs):
    """E0M8 mantissas -> f = 1 + m/256 (fp32, exact)."""
    return 1.0 + tl.load(m_ptrs).to(tl.float32) * (1.0 / 256.0)


@triton.jit
def _mxfp4_mbs_moe_kernel(
    a_ptr,
    as_ptr,
    am_ptr,
    b_ptr,
    bs_ptr,
    bm_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_asm,
    stride_amm,
    stride_be,
    stride_bn,
    stride_bse,
    stride_bsn,
    stride_bme,
    stride_bmk,
    stride_cm,
    MB: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    A_MBS: tl.constexpr,
    B_MBS: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    naive_block_assignment: tl.constexpr,
):
    """C[t, :] = A[t // top_k] @ B[expert(t)]^T for the rows t of one expert block.

    K advances BLOCK_K per iteration as BLOCK_K/CH scaled-MFMA dots of
    CH = min(MB, BLOCK_K) k each. Macro blocks longer than BLOCK_K rescale on
    their first iteration only.
    """
    CH: tl.constexpr = MB if MB < BLOCK_K else BLOCK_K
    NCH: tl.constexpr = BLOCK_K // CH
    ITERS_PER_MB: tl.constexpr = MB // BLOCK_K if MB > BLOCK_K else 1

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(EM, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs = tl.arange(0, BLOCK_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return
    if naive_block_assignment:
        offs_token = tl.where(offs == 0, pid_m, num_valid_tokens)
    else:
        offs_token = tl.load(sorted_token_ids_ptr + pid_m * BLOCK_M + offs)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)

    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if expert == -1:
        # The expert lives on another expert-parallel rank.
        tl.store(
            c_ptrs,
            tl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.dtype.element_ty),
            mask=c_mask,
        )
        return

    rows = tl.where(token_mask, offs_token // top_k, 0)
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)) % N

    a_ptrs = a_ptr + rows[:, None] * stride_am + tl.arange(0, CH // 2)[None, :]
    as_ptrs = as_ptr + rows[:, None] * stride_asm + tl.arange(0, CH // 32)[None, :]
    b_ptrs = (
        b_ptr
        + expert * stride_be
        + offs_bn[:, None] * stride_bn
        + tl.arange(0, CH // 2)[None, :]
    )
    bs_ptrs = (
        bs_ptr
        + expert * stride_bse
        + offs_bn[:, None] * stride_bsn
        + tl.arange(0, CH // 32)[None, :]
    )
    am_ptrs = am_ptr + rows * stride_amm
    bm_ptrs = bm_ptr + expert * stride_bme + offs_bn

    one_a = tl.full((BLOCK_M,), 1.0, tl.float32)
    one_b = tl.full((BLOCK_N,), 1.0, tl.float32)
    f_a_prev, f_b_prev = one_a, one_b
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(K // BLOCK_K):
        for j in tl.static_range(NCH):
            a = tl.load(a_ptrs + j * (CH // 2))
            a_s = tl.load(as_ptrs + j * (CH // 32))
            b = tl.load(b_ptrs + j * (CH // 2))
            b_s = tl.load(bs_ptrs + j * (CH // 32))
            if (A_MBS or B_MBS) and ITERS_PER_MB == 1:
                kb = i * NCH + j
                if A_MBS:
                    f_a = _mbs_factor(am_ptrs + kb)
                    r_a = tl.math.div_rn(f_a, f_a_prev)
                    f_a_prev = f_a
                if B_MBS:
                    f_b = _mbs_factor(bm_ptrs + kb * stride_bmk)
                    r_b = tl.math.div_rn(f_b, f_b_prev)
                    f_b_prev = f_b
                if A_MBS and B_MBS:
                    r = r_a[:, None] * r_b[None, :]
                elif A_MBS:
                    r = r_a[:, None]
                else:
                    r = r_b[None, :]
                acc = tl.dot_scaled(a, a_s, "e2m1", b.T, b_s, "e2m1", acc * r)
            else:
                # compile-time condition around a run-time one; they cannot merge
                if (A_MBS or B_MBS) and ITERS_PER_MB > 1:  # noqa: SIM102
                    if i % ITERS_PER_MB == 0:
                        kb = i // ITERS_PER_MB
                        if A_MBS:
                            f_a = _mbs_factor(am_ptrs + kb)
                            acc = acc * tl.math.div_rn(f_a, f_a_prev)[:, None]
                            f_a_prev = f_a
                        if B_MBS:
                            f_b = _mbs_factor(bm_ptrs + kb * stride_bmk)
                            acc = acc * tl.math.div_rn(f_b, f_b_prev)[None, :]
                            f_b_prev = f_b
                acc = tl.dot_scaled(a, a_s, "e2m1", b.T, b_s, "e2m1", acc)
        a_ptrs += BLOCK_K // 2
        as_ptrs += BLOCK_K // 32
        b_ptrs += BLOCK_K // 2
        bs_ptrs += BLOCK_K // 32

    if A_MBS:
        acc = acc * tl.math.div_rn(one_a, f_a_prev)[:, None]
    if B_MBS:
        acc = acc * tl.math.div_rn(one_b, f_b_prev)[None, :]
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        acc = acc * moe_weight[:, None]
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


def get_mxfp4_mbs_moe_config(num_tokens: int, mb: int) -> dict[str, int]:
    """Tile config by token count, measured on MI355X for MiniMax-M3 shapes."""
    if num_tokens <= 32:
        cfg = {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 256, "GROUP_M": 1}
        cfg.update(num_warps=4, num_stages=2, waves_per_eu=2)
    elif num_tokens <= 512:
        cfg = {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_M": 4}
        cfg.update(num_warps=4, num_stages=2, waves_per_eu=2)
    elif num_tokens <= 2048:
        cfg = {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 256, "GROUP_M": 8}
        cfg.update(num_warps=4, num_stages=2, waves_per_eu=2)
    else:
        cfg = {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "GROUP_M": 8}
        cfg.update(num_warps=4, num_stages=2, waves_per_eu=1)
    cfg["matrix_instr_nonkdim"] = 16
    if mb > cfg["BLOCK_K"]:
        assert mb % cfg["BLOCK_K"] == 0
    return cfg


def invoke_mxfp4_mbs_moe_gemm(
    a_q: torch.Tensor,
    a_s: torch.Tensor,
    a_m: torch.Tensor | None,
    w: Mxfp4MbsWeights,
    c: torch.Tensor,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    num_valid_tokens: int,
    mb: int,
    config: dict[str, int],
) -> None:
    """C [num_valid_tokens, N] (rows indexed like `topk_ids.view(-1)`)."""
    N, Kp = w.q.shape[1], w.q.shape[2]
    K = 2 * Kp
    while K % max(mb, config["BLOCK_K"]) and config["BLOCK_K"] > 128:
        config = dict(config, BLOCK_K=config["BLOCK_K"] // 2)
    assert K % max(mb, config["BLOCK_K"]) == 0, (K, mb, config)
    assert c.stride(-1) == 1 and a_q.stride(-1) == 1
    naive = sorted_token_ids is None
    EM = expert_ids.numel() * config["BLOCK_M"] if naive else sorted_token_ids.numel()
    grid = (triton.cdiv(EM, config["BLOCK_M"]) * triton.cdiv(N, config["BLOCK_N"]),)
    a_mbs, b_mbs = a_m is not None, w.m is not None
    _mxfp4_mbs_moe_kernel[grid](
        a_q,
        a_s,
        a_m if a_mbs else a_q,
        w.q,
        w.s,
        w.m if b_mbs else w.q,
        c,
        topk_weights if topk_weights is not None else c,
        sorted_token_ids if not naive else expert_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        num_valid_tokens,
        a_q.stride(0),
        a_s.stride(0),
        a_m.stride(0) if a_mbs else 0,
        w.q.stride(0),
        w.q.stride(1),
        w.s.stride(0),
        w.s.stride(1),
        w.m.stride(0) if b_mbs else 0,
        w.m.stride(1) if b_mbs else 0,
        c.stride(0),
        MB=mb,
        A_MBS=a_mbs,
        B_MBS=b_mbs,
        MUL_ROUTED_WEIGHT=topk_weights is not None,
        top_k=top_k,
        naive_block_assignment=naive,
        **config,
    )


def fused_experts_mxfp4_mbs(
    hidden_states: torch.Tensor,
    w1: Mxfp4MbsWeights,
    w2: Mxfp4MbsWeights,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
    activation_config: ApplyMoEActivationConfig,
    mb: int,
    act_mbs: bool,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    workspace13: torch.Tensor | None = None,
    workspace2: torch.Tensor | None = None,
) -> torch.Tensor:
    """MXFP4-MBS MoE: out [T, K] = sum_k topk_weights * W2 act(W13 x)."""
    assert hidden_states.dim() == 2 and hidden_states.is_contiguous()
    T, K = hidden_states.shape
    E, N, _ = w1.q.shape
    top_k = topk_ids.size(1)
    if global_num_experts == -1:
        global_num_experts = E
    if output is None:
        output = torch.empty_like(hidden_states)
    if T == 0:
        return output

    act_dim = N // 2 if activation.is_gated else N
    if workspace13 is None:
        workspace13 = torch.empty(
            T * top_k * max(act_dim, K), dtype=hidden_states.dtype, device=output.device
        )
    if workspace2 is None:
        workspace2 = torch.empty(
            T * top_k * max(N, K), dtype=hidden_states.dtype, device=output.device
        )
    cache1 = _resize_cache(workspace2, (T * top_k, N))
    cache2 = _resize_cache(workspace13, (T * top_k, act_dim))
    cache3 = _resize_cache(workspace2, (T, top_k, K))

    config = get_mxfp4_mbs_moe_config(T, mb)
    num_align_experts = E if expert_map is None else global_num_experts
    sorted_token_ids, expert_ids, num_tokens_post_padded = _prepare_expert_assignment(
        topk_ids,
        {"BLOCK_SIZE_M": config["BLOCK_M"]},
        T,
        top_k,
        num_align_experts,
        expert_map,
    )

    a_q, a_s, a_m = mbs_quantize(hidden_states, mb=mb, ncand=1, use_mbs=act_mbs)
    invoke_mxfp4_mbs_moe_gemm(
        a_q, a_s, a_m, w1, cache1, None, sorted_token_ids, expert_ids,
        num_tokens_post_padded, top_k, T * top_k, mb, config,
    )  # fmt: skip
    apply_moe_activation(
        activation, cache2, cache1, activation_config=activation_config
    )
    a2_q, a2_s, a2_m = mbs_quantize(cache2, mb=mb, ncand=1, use_mbs=act_mbs)
    invoke_mxfp4_mbs_moe_gemm(
        a2_q, a2_s, a2_m, w2, cache3.view(T * top_k, K),
        topk_weights.to(torch.float32).view(-1), sorted_token_ids, expert_ids,
        num_tokens_post_padded, 1, T * top_k, mb, config,
    )  # fmt: skip
    ops.moe_sum(cache3, output)
    return output


class Mxfp4MbsTritonExperts(mk.FusedMoEExpertsModular):
    """MXFP4 W4A4 experts with macro block scaling (Triton, gfx950 MFMA).

    The macro block size and the sides that carry factors come from the
    weight / activation quant keys; the weight mantissas are read off the
    layer (`w13_weight_mbs`, `w2_weight_mbs`) after loading.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        weight_key: QuantKey,
        activation_key: QuantKey,
    ):
        super().__init__(moe_config, quant_config)
        mb = _macro_block(weight_key) or _macro_block(activation_key)
        assert mb is not None, "MBS experts need a macro block on one side"
        self.mb = mb
        self.act_mbs = _macro_block(activation_key) is not None
        self.w13_mbs: torch.Tensor | None = None
        self.w2_mbs: torch.Tensor | None = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.w13_mbs = getattr(layer, "w13_weight_mbs", None)
        self.w2_mbs = getattr(layer, "w2_weight_mbs", None)

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        if not current_platform.is_rocm():
            return False
        from vllm.platforms.rocm import on_gfx950

        return on_gfx950()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return is_mxfp4_mbs_key(weight_key) or is_mxfp4_mbs_key(activation_key)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return apply_moe_activation_supported(activation)

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not moe_parallel_config.use_all2all_kernels

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (M, topk, max(activation_out_dim, K))
        workspace2 = (M, topk, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        assert not apply_router_weight_on_input, (
            "MXFP4-MBS MoE applies the router weights after the second GEMM"
        )
        assert a1q_scale is None, "MXFP4-MBS MoE quantizes its own activations"
        assert self.w1_scale is not None and self.w2_scale is not None
        fused_experts_mxfp4_mbs(
            hidden_states,
            Mxfp4MbsWeights(w1, self.w1_scale, self.w13_mbs),
            Mxfp4MbsWeights(w2, self.w2_scale, self.w2_mbs),
            topk_weights,
            topk_ids,
            activation,
            self.activation_config,
            mb=self.mb,
            act_mbs=self.act_mbs,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            output=output,
            workspace13=workspace13,
            workspace2=workspace2,
        )
        return None


def make_mxfp4_mbs_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    weight_key: QuantKey,
    activation_key: QuantKey,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> mk.FusedMoEKernel:
    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        maybe_make_prepare_finalize,
    )

    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=moe_quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
    )
    assert prepare_finalize is not None
    experts = Mxfp4MbsTritonExperts(
        moe_config, moe_quant_config, weight_key, activation_key
    )
    return mk.FusedMoEKernel(prepare_finalize, experts)


def _macro_block(key: QuantKey | None) -> int | None:
    if not is_mxfp4_mbs_key(key):
        return None
    assert key is not None and key.scale2 is not None
    return key.scale2.group_shape.col
