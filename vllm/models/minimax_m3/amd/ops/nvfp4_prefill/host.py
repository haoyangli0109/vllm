# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validation, compilation and dispatch of the NVFP4 sparse prefill attention."""

import functools

import numpy as np
import torch

from . import kernel as _kernel

D, HG, BLK, TOPK = 128, 16, 128, 16
LOG2E = 1.4426950409


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the
    cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


@functools.cache
def _get_kernel(rounds: int):
    return _kernel.compile_nvfp4_prefill_fp16(False, rounds, False, False, False)


@functools.cache
def _num_cus(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def supports_nvfp4_sparse_prefill(
    num_heads: int, num_kv_heads: int, head_dim: int, topk_blocks: int, block_size: int
) -> bool:
    """Whether the kernel serves this layer: gfx950, 128-dim heads, 16 query heads
    per KV head, 16 selected 128-token blocks per token."""
    from vllm.platforms.rocm import on_gfx950

    return (
        on_gfx950()
        and head_dim == D
        and num_heads == num_kv_heads * HG
        and topk_blocks == TOPK
        and block_size == BLK
    )


@functools.cache
def _unit_scale(device: torch.device) -> torch.Tensor:
    return torch.ones((), dtype=torch.float32, device=device)


def minimax_m3_nvfp4_sparse_prefill(
    q: torch.Tensor,  # [T, H, 128] bf16
    kv_cache: torch.Tensor,  # [pages, 2 * Hkv, 128, 72] uint8 (amd/ops/nvfp4_kv.py)
    topk: torch.Tensor,  # [Hkv, T, 16] int32 logical blocks, -1 = none
    block_table: torch.Tensor,  # [reqs, max_blocks] int32
    query_req_id: torch.Tensor,  # [T] int32, row of block_table per token
    query_abs_pos: torch.Tensor,  # [T] int32
    num_kv_heads: int,
    sm_scale: float,
    out: torch.Tensor,  # [T, H, 128] bf16, written
    k_scale: torch.Tensor | None = None,  # fp32 global scales (None = 1)
    v_scale: torch.Tensor | None = None,
) -> None:
    """NVFP4 sparse prefill attention: token t (request ``query_req_id[t]``, at
    ``query_abs_pos[t]``) attends to the causal tokens of its ``topk`` blocks.
    The K/V products are exact (e2m1 x e4m3 in fp16), Q and P fp16, fp32
    accumulation, bf16 out."""
    n_tok, n_heads, head_dim = q.shape
    if n_tok == 0:
        return
    assert head_dim == D and n_heads == num_kv_heads * HG
    assert q.dtype == torch.bfloat16 and out.dtype == torch.bfloat16
    assert out.shape == q.shape
    assert q.is_contiguous() and out.is_contiguous()
    assert kv_cache.dtype == torch.uint8 and tuple(kv_cache.shape[1:]) == (
        2 * num_kv_heads,
        BLK,
        72,
    )
    assert kv_cache.stride(1) == BLK * 72 and kv_cache.stride(3) == 1
    assert topk.dtype == torch.int32 and tuple(topk.shape) == (
        num_kv_heads,
        n_tok,
        TOPK,
    )
    assert topk.stride(2) == 1 and topk.stride(1) == TOPK
    for t in (block_table, query_req_id, query_abs_pos):
        assert t.dtype == torch.int32 and t.stride(-1) == 1
    assert kv_cache.stride(0) < 2**31
    unit = _unit_scale(q.device)
    ks = unit if k_scale is None else k_scale
    vs = unit if v_scale is None else v_scale
    # Triton's sm_scale * log2(e), rounded the same way (fp32)
    c = float(np.float32(sm_scale) * np.float32(LOG2E))
    # Two workgroups per CU (two-round combine) once the grid has more workgroups than
    # CUs; up to one per CU the one-round combine is ~1 us faster (MI355X, 10-03).
    n_wg = (n_tok + 3) // 4 * num_kv_heads
    rounds = 2 if n_wg > _num_cus(q.device) else 1
    _run_compiled(
        _get_kernel(rounds),
        q.data_ptr(),
        kv_cache.data_ptr(),
        topk.data_ptr(),
        block_table.data_ptr(),
        query_req_id.data_ptr(),
        query_abs_pos.data_ptr(),
        out.data_ptr(),
        ks.data_ptr(),
        vs.data_ptr(),
        0,
        n_tok,
        n_heads,
        num_kv_heads,
        topk.stride(0),
        block_table.stride(0),
        kv_cache.stride(0),
        c,
        torch.cuda.current_stream(q.device),
    )
