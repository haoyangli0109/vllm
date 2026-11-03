# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 macro-block-scaled fused MoE (Triton) against a dequantized reference."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import (
    ApplyMoEActivationConfig,
    MoEActivation,
)
from vllm.model_executor.layers.fused_moe.experts.mxfp4_mbs_moe import (
    Mxfp4MbsWeights,
    fused_experts_mxfp4_mbs,
    get_mxfp4_mbs_moe_config,
    invoke_mxfp4_mbs_moe_gemm,
    quantize_moe_weight_mxfp4_mbs,
)
from vllm.model_executor.layers.fused_moe.fused_moe import _prepare_expert_assignment
from vllm.model_executor.layers.quantization.utils.mxfp4_mbs_utils import (
    mbs_dequant,
    mbs_quant_ref,
    mbs_quantize,
)
from vllm.platforms import current_platform


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


pytestmark = pytest.mark.skipif(not _on_gfx950(), reason="needs the gfx950 MFMA")

LIMIT, ALPHA, BETA = 7.0, 1.702, 1.0
SIDES = [(True, True), (False, True), (True, False), (False, False)]


def _weight_fp32(w: Mxfp4MbsWeights, mb: int) -> torch.Tensor:
    m = w.m.transpose(1, 2) if w.m is not None else None
    return mbs_dequant(w.q, w.s, m, mb=mb)


def _act_qdq(x: torch.Tensor, mb: int, act_mbs: bool) -> torch.Tensor:
    q, s, m = mbs_quant_ref(x, mb=mb, ncand=1, use_mbs=act_mbs)
    return mbs_dequant(q, s, m if act_mbs else None, mb=mb)


def _swiglu_oai(h: torch.Tensor) -> torch.Tensor:
    # The op the kernel path runs: its bf16 rounding of silu(gate) moves values
    # across FP4 boundaries, so the reference has to share it.
    out = torch.empty(*h.shape[:-1], h.shape[-1] // 2, dtype=h.dtype, device=h.device)
    torch.ops._C.silu_and_mul_with_clamp(out, h, LIMIT, ALPHA, BETA)
    return out


def _reference(x, w1, w2, topk_weights, topk_ids, mb, act_mbs):
    """Same quantization points and bf16 roundings as the kernel, fp32 GEMMs."""
    w1f, w2f = _weight_fp32(w1, mb), _weight_fp32(w2, mb)
    xq = _act_qdq(x, mb, act_mbs)
    T, top_k = topk_ids.shape
    out = torch.zeros(T, w2f.shape[1], dtype=torch.float32, device=x.device)
    for e in topk_ids.unique().tolist():
        t, k = (topk_ids == e).nonzero(as_tuple=True)
        h = (xq[t] @ w1f[e].T).to(x.dtype)
        y = (_act_qdq(_swiglu_oai(h), mb, act_mbs) @ w2f[e].T) * topk_weights[
            t, k, None
        ].float()
        out.index_add_(0, t, y.to(x.dtype).float())
    return out


def _routing(T, E, top_k, device):
    logits = torch.randn(T, E, device=device)
    topk_weights, topk_ids = torch.topk(torch.softmax(logits, -1), top_k, dim=-1)
    return topk_weights / topk_weights.sum(-1, keepdim=True), topk_ids.to(torch.int32)


@pytest.mark.parametrize("T", [1, 7, 64, 333, 2049, 4096])
@pytest.mark.parametrize("mb", [128, 256])
@pytest.mark.parametrize("act_mbs,w_mbs", SIDES)
def test_mxfp4_mbs_moe_gemm(T, mb, act_mbs, w_mbs):
    """One grouped GEMM into fp32 matches the dequantized operands' product."""
    torch.manual_seed(0)
    E, K, N, top_k = 16, 1024, 1024, 4
    x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
    w = (torch.randn(E, N, K, device="cuda") * 0.05).to(torch.bfloat16)
    W = quantize_moe_weight_mxfp4_mbs(w, mb=mb, ncand=16, use_mbs=w_mbs)
    topk_ids = torch.stack(
        [torch.randperm(E, device="cuda")[:top_k] for _ in range(T)]
    ).to(torch.int32)
    cfg = get_mxfp4_mbs_moe_config(T, mb)
    sorted_ids, expert_ids, num_post_padded = _prepare_expert_assignment(
        topk_ids, {"BLOCK_SIZE_M": cfg["BLOCK_M"]}, T, top_k, E, None
    )
    a_q, a_s, a_m = mbs_quantize(x, mb=mb, ncand=1, use_mbs=act_mbs)
    c = torch.full((T * top_k, N), float("nan"), device="cuda")
    invoke_mxfp4_mbs_moe_gemm(
        a_q, a_s, a_m, W, c, None, sorted_ids, expert_ids, num_post_padded,
        top_k, T * top_k, mb, cfg,
    )  # fmt: skip
    ref = torch.einsum(
        "tk,tnk->tn",
        _act_qdq(x, mb, act_mbs).repeat_interleave(top_k, 0),
        _weight_fp32(W, mb)[topk_ids.view(-1).long()],
    )
    # Exact products; the factors only add fp32 rounding of the rescaling.
    torch.testing.assert_close(c, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("T", [1, 7, 64, 333])
@pytest.mark.parametrize("mb", [128, 256])
@pytest.mark.parametrize("act_mbs,w_mbs", SIDES)
def test_mxfp4_mbs_moe(T, mb, act_mbs, w_mbs):
    torch.manual_seed(0)
    E, H, inter, top_k = 16, 1024, 512, 4
    dev, dt = "cuda", torch.bfloat16
    x = torch.randn(T, H, device=dev, dtype=dt)
    w13 = (torch.randn(E, 2 * inter, H, device=dev) * 0.05).to(dt)
    w2 = (torch.randn(E, H, inter, device=dev) * 0.05).to(dt)
    W1 = quantize_moe_weight_mxfp4_mbs(w13, mb=mb, ncand=16, use_mbs=w_mbs)
    W2 = quantize_moe_weight_mxfp4_mbs(w2, mb=mb, ncand=16, use_mbs=w_mbs)
    topk_weights, topk_ids = _routing(T, E, top_k, dev)

    out = fused_experts_mxfp4_mbs(
        x,
        W1,
        W2,
        topk_weights,
        topk_ids,
        MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        ApplyMoEActivationConfig(clamp_limit=LIMIT, alpha=ALPHA, beta=BETA),
        mb=mb,
        act_mbs=act_mbs,
    )
    ref = _reference(x, W1, W2, topk_weights, topk_ids, mb, act_mbs)
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    # bf16 output rounding plus rare FP4 code flips of the intermediate.
    assert rel < 3e-3, rel
