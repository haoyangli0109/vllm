# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile cache + launch wrappers of the FlyDSL index top-k (prefill and decode
rows): the logical block ids AITER's ``pa_sparse_block_topk`` selects, in its
order, without its sparse page table (the NVFP4 attends read the block ids)."""

import functools

import torch

from . import prefill_kernel as _kernel

MAX_SLOTS = 128  # AITER's TOPK_MAX_SLOTS: 8192 blocks


def _pow2_ceil(n: int) -> int:
    return 1 << (n - 1).bit_length() if n >= 1 else 1


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the
    cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


@functools.lru_cache(maxsize=len(_kernel.PREFILL_STRIPS))
def _get_prefill_kernel(strips: int):
    return _kernel.compile_topk_prefill(strips)


@functools.lru_cache(maxsize=1)
def _get_decode_kernel():
    return _kernel.compile_topk_prefill(26, decode=True)


def supports_index_topk(topk_blocks: int, block_size: int) -> bool:
    """gfx950 and the shape AITER's top-k is built for (16 blocks of 128
    tokens)."""
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950() and topk_blocks == _kernel.TOPK and block_size == _kernel.BLK


def _check(score: torch.Tensor, topk_idx: torch.Tensor, max_seq_len: int) -> int:
    """Validate the layouts; returns the score columns the kernel walks."""
    heads, rows, width = score.shape
    assert score.dtype == torch.float32 and score.stride(2) == 1
    assert topk_idx.dtype == torch.int32 and topk_idx.stride(2) == 1
    assert topk_idx.shape == (heads, rows, _kernel.TOPK)
    slots = _pow2_ceil((max_seq_len + _kernel.BLK * 64 - 1) // (_kernel.BLK * 64))
    assert slots <= MAX_SLOTS and width >= slots * 64
    return slots * 64


def minimax_m3_index_topk_prefill_indices(
    score: torch.Tensor,  # [H, rows, S] fp32
    topk_idx: torch.Tensor,  # [H, rows, 16] int32
    *,
    max_seq_len: int,
    num_valid_pages: torch.Tensor,  # [>= rows] int32: row n's visible blocks
) -> None:
    """Top-16 blocks of ragged prefill rows."""
    heads, rows, _ = score.shape
    cols = _check(score, topk_idx, max_seq_len)
    assert num_valid_pages.dtype == torch.int32
    assert num_valid_pages.is_contiguous() and num_valid_pages.numel() >= rows
    if rows == 0:
        return
    tiles = _kernel.PREFILL_STRIPS
    strips = next((s for s in tiles if max_seq_len <= s * _kernel.BLK * 64), tiles[-1])
    _run_compiled(
        _get_prefill_kernel(strips),
        score.data_ptr(),
        score.stride(0),
        score.stride(1),
        cols,
        num_valid_pages.data_ptr(),
        1,
        topk_idx.data_ptr(),
        topk_idx.stride(0),
        topk_idx.stride(1),
        rows,
        heads,
        torch.cuda.current_stream(),
    )


def minimax_m3_index_topk_decode_indices(
    score: torch.Tensor,  # [H, num_reqs * query_len, S] fp32
    topk_idx: torch.Tensor,  # [H, num_reqs * query_len, 16] int32
    seq_lens: torch.Tensor,  # [>= num_reqs] int32
    *,
    max_seq_len: int,
    query_len: int,
) -> None:
    """Top-16 blocks of decode rows (``query_len`` rows per request ending its
    sequence), one launch with runtime query and window lengths."""
    heads, rows, _ = score.shape
    cols = _check(score, topk_idx, max_seq_len)
    assert query_len > 0 and rows % query_len == 0
    assert seq_lens.dtype == torch.int32 and seq_lens.is_contiguous()
    assert seq_lens.numel() >= rows // query_len
    if rows == 0:
        return
    _run_compiled(
        _get_decode_kernel(),
        score.data_ptr(),
        score.stride(0),
        score.stride(1),
        cols,
        seq_lens.data_ptr(),
        query_len,
        topk_idx.data_ptr(),
        topk_idx.stride(0),
        topk_idx.stride(1),
        rows,
        heads,
        torch.cuda.current_stream(),
    )
