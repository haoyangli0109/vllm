# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Zero-copy views (and two converted copies) of one vLLM ``MiniMaxM3DecoderLayer``
for the mono kernels.

The kernels read tensors through raw pointers in fixed layouts: a differently
laid-out weight would not fault, it would produce plausible garbage. So every
layout assumption is asserted once and a model that does not match is refused
(``MonoUnsupported``). The experts already have the kernels' layout (MXFP4, aiter
``shuffle_weight(16, 16)`` + ``e8m0_shuffle`` scales, fused shared expert in slot
128). The attention linears are MXFP8 in vLLM and the router gate is fp32, so
those are converted once when the runner is built: MXFP8 -> bf16 -> per-channel
fp8, preshuffled (16, 16); gate -> bf16.
"""

from dataclasses import dataclass

import torch
from aiter import dtypes
from aiter.ops.quant import pertoken_quant
from aiter.ops.shuffle import shuffle_weight

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequant_mxfp8_to_bf16,
)
from vllm.models.minimax_m3.amd.mono.config import (
    MonoUnsupported,
    HEAD_DIM,
    HIDDEN,
    INTER,
    LOCAL_Q_HEADS,
    N_ROUTED,
    O_K,
    ONE_INDEX_HEAD,
    ROTARY_DIM,
    TOP_K,
    TOPK_BLOCKS,
    IndexHeads,
)


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise MonoUnsupported(what)


def _bf16_vec(t: torch.Tensor, n: int, name: str) -> torch.Tensor:
    _need(
        t.dtype == torch.bfloat16 and t.numel() == n and t.is_contiguous(),
        f"{name}: {t.dtype} {tuple(t.shape)}",
    )
    return t.data


@dataclass(frozen=True)
class SparseMoeLayer:
    """Everything the layer kernel reads for one sparse-attention MoE layer."""

    layer_id: int
    attn: object  # MiniMaxM3SparseAttention: layer name, KV cache views, scales
    index_cache: torch.Tensor  # [blocks, block, head_dim] fp8, shared block ids
    g_in: torch.Tensor
    w_qkv: torch.Tensor
    s_qkv: torch.Tensor
    g_q: torch.Tensor
    g_k: torch.Tensor
    g_iq: torch.Tensor
    g_ik: torch.Tensor
    cos_sin: torch.Tensor
    w_o: torch.Tensor
    s_o: torch.Tensor
    g_post: torch.Tensor
    gate: torch.Tensor
    bias: torch.Tensor
    w13: torch.Tensor
    s13: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor


def _ptpc_fp8_from_mxfp8(
    linear, rows: int, cols: int, name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP8 linear (fp8 [rows, cols], e8m0 [rows, cols/32]) -> per-channel fp8
    preshuffled (16, 16) + fp32 scale [rows], as the kernels' PTPC GEMV reads."""
    w = getattr(linear, "weight", None)
    s = getattr(linear, "weight_scale", None)
    _need(
        w is not None and w.dtype == dtypes.fp8 and tuple(w.shape) == (rows, cols),
        f"{name}: weight {None if w is None else (w.dtype, tuple(w.shape))}",
    )
    _need(
        s is not None and tuple(s.shape) == (rows, cols // 32),
        f"{name}: scale {None if s is None else tuple(s.shape)}",
    )
    bf = dequant_mxfp8_to_bf16(w.data, s.data.view(torch.uint8))
    q, scale = pertoken_quant(bf, quant_dtype=dtypes.fp8)
    return (
        shuffle_weight(q, layout=(16, 16)).contiguous(),
        scale.view(rows).to(torch.float32).contiguous(),
    )


def _cos_sin(attn, layer_id: int) -> torch.Tensor:
    t = attn.rotary_emb.cos_sin_cache
    _need(
        t.dim() == 2 and t.shape[1] == ROTARY_DIM,
        f"layer {layer_id}: cos/sin cache {tuple(t.shape)}",
    )
    if t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    return t.contiguous()


def build_sparse_layer(
    layer, layer_id: int, heads: IndexHeads = ONE_INDEX_HEAD
) -> SparseMoeLayer:
    attn = layer.self_attn
    _need(
        hasattr(attn, "indexer") and layer.is_moe_layer,
        f"layer {layer_id}: not sparse MoE",
    )
    _need(
        (attn.num_heads, attn.num_kv_heads, attn.num_idx_heads, attn.head_dim)
        == (LOCAL_Q_HEADS, 1, heads.count, HEAD_DIM),
        f"layer {layer_id}: heads {attn.num_heads}/{attn.num_kv_heads}/{attn.num_idx_heads}",
    )
    _need(attn.rotary_emb.rotary_dim == ROTARY_DIM, f"layer {layer_id}: rotary_dim")
    sel = attn.indexer.impl  # the selection config lives on the indexer impl
    _need(
        (sel.topk_blocks, sel.init_blocks, sel.local_blocks, attn.skip_index_topk)
        == (TOPK_BLOCKS, 0, 1, False),
        f"layer {layer_id}: sparse selection config",
    )
    _need(
        attn.use_aiter_sparse_pa and str(attn.kv_cache_dtype).startswith("fp8"),
        f"layer {layer_id}: needs the AITER sparse PA path with an fp8 KV cache",
    )
    index_cache = attn.indexer.index_cache.kv_cache
    _need(
        index_cache.numel() > 0
        and index_cache.dtype == dtypes.fp8
        and index_cache.dim() == 3
        and index_cache.shape[2] == HEAD_DIM
        and index_cache.is_contiguous(),
        f"layer {layer_id}: index cache {index_cache.dtype} {tuple(index_cache.shape)}",
    )
    w_qkv, s_qkv = _ptpc_fp8_from_mxfp8(
        attn.qkv_proj, heads.rows, HIDDEN, f"layer {layer_id} qkv_proj"
    )
    w_o, s_o = _ptpc_fp8_from_mxfp8(attn.o_proj, HIDDEN, O_K, f"layer {layer_id} o_proj")

    moe = layer.block_sparse_moe
    # FusedMoEFactory returns a MoERunner wrapping the RoutedExperts module
    experts = getattr(moe.experts, "routed_experts", moe.experts)
    E = N_ROUTED + 1
    w13, w2 = experts.w13_weight, experts.w2_weight
    _need(
        moe.is_fused_shared_expert_enabled and w13.shape[0] == E,
        f"layer {layer_id}: fused shared expert (experts {w13.shape[0]})",
    )
    _need(experts.top_k == TOP_K, f"layer {layer_id}: top_k {experts.top_k}")
    _need(
        getattr(w13, "is_shuffled", False) and getattr(w2, "is_shuffled", False),
        f"layer {layer_id}: experts not in aiter shuffled layout",
    )
    _need(
        w13.numel() * w13.element_size() == E * 2 * INTER * HIDDEN // 2
        and w2.numel() * w2.element_size() == E * HIDDEN * INTER // 2,
        f"layer {layer_id}: expert weight bytes",
    )
    s13, s2 = experts.w13_weight_scale, experts.w2_weight_scale
    _need(
        s13.numel() == E * 2 * INTER * HIDDEN // 32
        and s2.numel() == E * HIDDEN * INTER // 32,
        f"layer {layer_id}: expert scale bytes",
    )
    gate = moe.gate.weight.data
    _need(tuple(gate.shape) == (N_ROUTED, HIDDEN), f"layer {layer_id}: gate")
    bias = moe.e_score_correction_bias
    _need(
        bias is not None and bias.dtype == torch.float32 and bias.numel() == N_ROUTED,
        f"layer {layer_id}: bias",
    )
    return SparseMoeLayer(
        layer_id=layer_id,
        attn=attn,
        index_cache=index_cache,
        g_in=_bf16_vec(layer.input_layernorm.weight, HIDDEN, "input_layernorm"),
        w_qkv=w_qkv,
        s_qkv=s_qkv,
        g_q=_bf16_vec(attn.q_norm.weight, HEAD_DIM, "q_norm"),
        g_k=_bf16_vec(attn.k_norm.weight, HEAD_DIM, "k_norm"),
        g_iq=_bf16_vec(attn.index_q_norm.weight, HEAD_DIM, "index_q_norm"),
        g_ik=_bf16_vec(attn.index_k_norm.weight, HEAD_DIM, "index_k_norm"),
        cos_sin=_cos_sin(attn, layer_id),
        w_o=w_o,
        s_o=s_o,
        g_post=_bf16_vec(
            layer.post_attention_layernorm.weight, HIDDEN, "post_attention_layernorm"
        ),
        gate=gate.to(torch.bfloat16).contiguous(),
        bias=bias.data,
        w13=w13.data,
        s13=s13.data,
        w2=w2.data,
        s2=s2.data,
    )
