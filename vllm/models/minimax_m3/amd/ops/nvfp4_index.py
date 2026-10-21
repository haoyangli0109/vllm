# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 index-K cache scorers for the ROCm indexer (gfx950).

The page is NVIDIA's (see ``nvidia/ops/gather_nvfp4_index_k.py``): ``[128, 72]``
bytes holding every token's packed E2M1 codes (64 B, low nibble first), then
their E4M3 16-element block scales (``token * 8 + group``), no global scale.
A key element is ``e4m3_rn_satfinite(e2m1 * scale)``, the key MSA's Q8KV4 /
Q8KV8 indexers score, so these kernels score FP8 keys with FP8 queries like
AITER's fp8 scorer (``pa_sparse_block_score_{decode,prefill}``), bit for bit.

Both scorers are FlyDSL kernels over the NVFP4 pages themselves
(``nvfp4_index_decode`` / ``nvfp4_index_prefill``); the index-K inserts are in
``norm_rope_nvfp4_insert``.
"""

import torch

BLOCK_SIZE = 128
HEAD_DIM = 128
PAGE_BYTES = BLOCK_SIZE * (HEAD_DIM // 2 + HEAD_DIM // 16)  # 9216


def _int32_contiguous(*tensors: torch.Tensor) -> bool:
    return all(t.dtype == torch.int32 and t.is_contiguous() for t in tensors)


def _flat_pages(index_cache: torch.Tensor) -> bool:
    """Page ``p`` at ``p * stride(0)`` bytes, one flat, 16B-aligned [128 x 64B
    data][128 x 8B scales] block (what the FlyDSL scorers read)."""
    return (
        index_cache.dtype == torch.uint8
        and index_cache.stride(2) == 1
        and index_cache.stride(1) == HEAD_DIM // 2 + HEAD_DIM // 16
        and index_cache.stride(0) % 16 == 0
        and index_cache.data_ptr() % 16 == 0
    )


@torch.no_grad()
def minimax_m3_nvfp4_index_score_decode(
    q_idx: torch.Tensor,  # [num_reqs * query_len, H, 128] e4m3
    index_cache: torch.Tensor,  # [num_blocks, 128, 72] uint8
    score: torch.Tensor,  # [H, num_reqs * query_len, S] fp32
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    init_blocks: int,
    local_blocks: int,
    query_len: int,
    max_seq_len: int,
) -> torch.Tensor:
    """``pa_sparse_block_score_decode`` over NVFP4 index pages: per (head, row,
    block) the max over the row's visible tokens of q . k, with the init /
    local sentinels, for blocks below ``cdiv(seq_len, 128)``."""
    total_q = q_idx.shape[0]
    if total_q // query_len == 0:
        return score
    from vllm.models.minimax_m3.amd.ops.nvfp4_index_decode import (
        minimax_m3_nvfp4_index_score_decode_fly,
        supports_nvfp4_index_score_decode_fly,
    )

    assert supports_nvfp4_index_score_decode_fly(
        q_idx, index_cache, score, block_table, seq_lens, query_len
    ), "unsupported NVFP4 decode index scorer layout"
    # A page that several requests share loads once per chunk.
    minimax_m3_nvfp4_index_score_decode_fly(
        q_idx,
        index_cache,
        score,
        block_table,
        seq_lens,
        init_blocks,
        local_blocks,
        query_len,
        max_seq_len,
    )
    return score


@torch.no_grad()
def minimax_m3_nvfp4_index_score_prefill(
    q_idx: torch.Tensor,  # [total_q, H, 128] e4m3
    index_cache: torch.Tensor,  # [num_blocks, 128, 72] uint8
    score: torch.Tensor,  # [H, total_q, S] fp32
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    init_blocks: int,
    local_blocks: int,
    max_query_len: int,
    max_seq_len: int,
) -> torch.Tensor:
    """``pa_sparse_block_score_prefill`` over NVFP4 index pages: per (head,
    row, block below ``cdiv(seq_len, 128)``) the max over the row's visible
    tokens of q . k with the init / local sentinels, each request's chunk
    ending its sequence (prefix = seq_len - q_len)."""
    total_q = q_idx.shape[0]
    if total_q == 0 or block_table.shape[0] == 0:
        return score
    from vllm.models.minimax_m3.amd.ops.nvfp4_index_prefill import (
        minimax_m3_nvfp4_index_score_prefill_fly,
        supports_nvfp4_index_score_prefill_fly,
    )

    assert (
        supports_nvfp4_index_score_prefill_fly(q_idx, score)
        and _flat_pages(index_cache)
        and _int32_contiguous(block_table, cu_seqlens_q, seq_lens)
    ), "unsupported NVFP4 prefill index scorer layout"
    minimax_m3_nvfp4_index_score_prefill_fly(
        q_idx,
        index_cache,
        score,
        block_table,
        cu_seqlens_q,
        seq_lens,
        init_blocks,
        local_blocks,
        max_query_len,
        max_seq_len,
    )
    return score
