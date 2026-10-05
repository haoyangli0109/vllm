# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton input-staging kernel for the DeepGEMM NVFP4 MegaMoE.

In one pass over the BF16 hidden states this writes:
  * the routed input as NVFP4 with a per-token outer scale (MiniMax-M3.1
    recipe, see ``nvfp4_row_quant``): packed E2M1, int32-packed per-16 E4M3
    block scales (row major) and one fp32 dequantization scale per token;
  * the shared-expert input as MXFP8: E4M3 plus int32-packed per-32 UE8M0
    scales, in the MN-major UTCCP row order selected by the MegaMoE BLOCK_M;
  * the int64/float32 top-k tensors, with padding tokens masked out.
"""

import torch

from vllm.model_executor.layers.quantization.utils.nvfp4_row_quant import (
    nvfp4_quant_chunk,
    nvfp4_row_quant_launch_config,
    nvfp4_row_scales,
)
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_tokens"])
def _prepare_nvfp4_megamoe_inputs_kernel(
    hidden_states,
    x_fp4,
    x_sf,
    x_scales,
    shared_x,
    shared_x_sf,
    topk_ids,
    topk_weights,
    is_padding,
    topk_idx_out,
    topk_weights_out,
    hidden_stride_m: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_sf_stride_m: tl.constexpr,
    shared_x_stride_m: tl.constexpr,
    shared_x_sf_stride_m: tl.constexpr,
    shared_x_sf_stride_k: tl.constexpr,
    topk_ids_stride_m: tl.constexpr,
    topk_ids_stride_k: tl.constexpr,
    topk_weights_stride_m: tl.constexpr,
    topk_weights_stride_k: tl.constexpr,
    is_padding_stride_m: tl.constexpr,
    topk_idx_stride_m: tl.constexpr,
    topk_idx_stride_k: tl.constexpr,
    topk_weights_out_stride_m: tl.constexpr,
    topk_weights_out_stride_k: tl.constexpr,
    num_tokens,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    SHARED_BLOCK_M: tl.constexpr,
) -> None:
    # Each program needs its row's amax, then quantizes every SPLIT_K-th chunk.
    token_id = tl.program_id(0)
    split = tl.program_id(1)
    row = hidden_states + token_id.to(tl.int64) * hidden_stride_m
    encode, row_scale = nvfp4_row_scales(row, hidden_size, BLOCK_H)

    # The shared GEMM's TMA loads want the scales MN-major, with rows permuted
    # inside each runtime BLOCK_M tile for UTCCP.
    m_in_block = token_id % SHARED_BLOCK_M
    aligned_block_m: tl.constexpr = triton.cdiv(SHARED_BLOCK_M, 128) * 128
    shared_row = (token_id // SHARED_BLOCK_M) * aligned_block_m + (
        (m_in_block // 128) * 128 + (m_in_block % 32) * 4 + (m_in_block % 128) // 32
    )

    for k0 in tl.range(split * BLOCK_K, hidden_size, SPLIT_K * BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < hidden_size
        hidden = tl.load(row + k_offsets, mask=k_mask, other=0.0).to(tl.float32)

        # Shared input: MXFP8 with power-of-two (UE8M0) scales per 32 elements.
        MX_GROUP: tl.constexpr = 32
        MX_GROUPS: tl.constexpr = BLOCK_K // MX_GROUP
        mx_amax = tl.max(tl.reshape(tl.abs(hidden), [MX_GROUPS, MX_GROUP]), 1)
        mx_scale = tl.maximum(mx_amax, 1.0e-4) / 448.0
        mx_bits = mx_scale.to(tl.uint32, bitcast=True)
        mx_exp = ((mx_bits >> 23) & 0xFF) + ((mx_bits & 0x7FFFFF) != 0).to(tl.uint32)
        mx_exp = tl.minimum(tl.maximum(mx_exp, 1), 254)
        mx_inv_scale = 1.0 / (mx_exp << 23).to(tl.float32, bitcast=True)
        mx_scaled = tl.reshape(hidden, [MX_GROUPS, MX_GROUP]) * mx_inv_scale[:, None]
        tl.store(
            shared_x + token_id * shared_x_stride_m + k_offsets,
            tl.reshape(mx_scaled, [BLOCK_K]).to(tl.float8e4nv),
            mask=k_mask,
        )
        # Four UE8M0 bytes per int32 word, one word per 128 columns.
        mx_words = tl.reshape(mx_exp, [BLOCK_K // 128, 4])
        mx_packed = tl.sum(mx_words << (tl.arange(0, 4)[None, :] * 8), axis=1)
        word_offsets = k0 // 128 + tl.arange(0, BLOCK_K // 128)
        tl.store(
            shared_x_sf
            + shared_row * shared_x_sf_stride_m
            + word_offsets * shared_x_sf_stride_k,
            mx_packed.to(tl.int32, bitcast=True),
            mask=word_offsets < hidden_size // 128,
        )

        # Routed input: NVFP4, x ~= e2m1 * e4m3_block_scale * x_scale.
        packed, nv_sf = nvfp4_quant_chunk(hidden, encode, BLOCK_K)
        packed_offsets = k0 // 2 + tl.arange(0, BLOCK_K // 2)
        tl.store(
            x_fp4 + token_id * x_stride_m + packed_offsets,
            packed.to(tl.int8, bitcast=True),
            mask=packed_offsets < hidden_size // 2,
        )
        sf_offsets = k0 // 16 + tl.arange(0, BLOCK_K // 16)
        tl.store(
            x_sf + token_id * x_sf_stride_m + sf_offsets,
            nv_sf.to(tl.uint8, bitcast=True),
            mask=sf_offsets < hidden_size // 16,
        )

    if split != 0:
        return
    tl.store(x_scales + token_id, row_scale)

    topk_offsets = tl.arange(0, BLOCK_TOPK)
    topk_mask = topk_offsets < top_k
    token_is_padding = False
    if is_padding is not None:
        token_is_padding = tl.load(is_padding + token_id * is_padding_stride_m)

    ids = tl.load(
        topk_ids + token_id * topk_ids_stride_m + topk_offsets * topk_ids_stride_k,
        mask=topk_mask,
        other=0,
    ).to(tl.int64)
    ids = tl.where(token_is_padding, -1, ids)
    tl.store(
        topk_idx_out + token_id * topk_idx_stride_m + topk_offsets * topk_idx_stride_k,
        ids,
        mask=topk_mask,
    )

    weights = tl.load(
        topk_weights
        + token_id * topk_weights_stride_m
        + topk_offsets * topk_weights_stride_k,
        mask=topk_mask,
        other=0.0,
    )
    weights = tl.where(token_is_padding, 0.0, weights)
    tl.store(
        topk_weights_out
        + token_id * topk_weights_out_stride_m
        + topk_offsets * topk_weights_out_stride_k,
        weights,
        mask=topk_mask,
    )


def prepare_nvfp4_megamoe_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_fp4: torch.Tensor,
    x_sf: torch.Tensor,
    x_scales: torch.Tensor,
    shared_x: torch.Tensor,
    shared_x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    *,
    shared_block_m: int,
    is_padding: torch.Tensor | None = None,
) -> None:
    """Stage one MegaMoE launch's inputs into the NVFP4 symmetric buffer.

    Args:
        hidden_states: BF16 ``[M, H]`` layer input.
        topk_weights: ``[M, top_k]`` routing weights.
        topk_ids: ``[M, top_k]`` global expert ids.
        x_fp4: int8 ``[>=M, H/2]`` routed input (two E2M1 per byte, low first).
        x_sf: int32 ``[>=M, H/64]`` routed E4M3 block scales, row major.
        x_scales: float32 ``[>=M]`` routed per-token dequantization scales.
        shared_x: E4M3 ``[>=M, H]`` shared-expert input.
        shared_x_sf: int32 ``[rows, H/128]`` MN-major shared UE8M0 scales.
        topk_idx_out: int64 ``[>=M, top_k]`` staged expert ids.
        topk_weights_out: float32 ``[>=M, top_k]`` staged routing weights.
        shared_block_m: BLOCK_M the MegaMoE kernel selects for ``M`` tokens.
        is_padding: Optional ``[M]`` mask; padding tokens route nowhere.

    """
    num_tokens, hidden_size = hidden_states.shape
    if num_tokens == 0:
        return
    if hidden_size % 128 != 0:
        raise ValueError(
            "NVFP4 MegaMoE input staging requires hidden_size to be a multiple "
            f"of 128, got {hidden_size}."
        )
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "NVFP4 MegaMoE input staging requires topk_weights and topk_ids "
            "to have the same shape."
        )
    if shared_block_m <= 0:
        raise ValueError("NVFP4 MegaMoE shared_block_m must be positive.")
    if (
        hidden_states.stride(1) != 1
        or x_fp4.stride(1) != 1
        or x_sf.stride(1) != 1
        or shared_x.stride(1) != 1
        or x_scales.stride(0) != 1
    ):
        raise ValueError("NVFP4 MegaMoE staging tensors must be row major.")
    aligned_block_m = triton.cdiv(shared_block_m, 128) * 128
    required_rows = triton.cdiv(num_tokens, shared_block_m) * aligned_block_m
    if shared_x_sf.shape[0] < required_rows:
        raise ValueError(
            "NVFP4 MegaMoE shared_x_sf has insufficient rows: requires "
            f"{required_rows}, got {shared_x_sf.shape[0]}."
        )

    top_k = topk_ids.shape[1]
    # Byte view: four E4M3 scales per int32 word, little endian.
    x_sf_bytes = x_sf.view(torch.uint8)
    config = nvfp4_row_quant_launch_config(num_tokens, hidden_size)
    _prepare_nvfp4_megamoe_inputs_kernel[(num_tokens, config["SPLIT_K"])](
        hidden_states,
        x_fp4,
        x_sf_bytes,
        x_scales,
        shared_x,
        shared_x_sf,
        topk_ids,
        topk_weights,
        is_padding,
        topk_idx_out,
        topk_weights_out,
        hidden_states.stride(0),
        x_fp4.stride(0),
        x_sf_bytes.stride(0),
        shared_x.stride(0),
        shared_x_sf.stride(0),
        shared_x_sf.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        is_padding.stride(0) if is_padding is not None else 0,
        topk_idx_out.stride(0),
        topk_idx_out.stride(1),
        topk_weights_out.stride(0),
        topk_weights_out.stride(1),
        num_tokens,
        hidden_size,
        top_k,
        **config,
        BLOCK_TOPK=triton.next_power_of_2(top_k),
        SHARED_BLOCK_M=shared_block_m,
        num_warps=4,
    )
