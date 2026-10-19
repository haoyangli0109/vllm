# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile cache + launch wrapper of the FlyDSL NVFP4 prefill index scorer."""

import functools
import math

import torch

from . import kernel as _kernel

# Workgroups to aim for (2 resident per CU): a request's query tiles see causal
# windows of different lengths, so more, shorter segments balance the tail.
# Swept 512..4096 (cold L2): 2048 best, e.g. 64K new x 64K 265 -> 230 us and 32K x
# 32K 95 -> 86 us against 1024; long contexts -3..-4%. A segment keeps >= 4 blocks.
TARGET_WORKGROUPS = 2048
# ... but a last round of workgroups that is mostly empty wastes up to a round:
# with 2 workgroups resident per CU, segments are added until the rounds are at
# least this full (e.g. 5 x 13K new on 120K, 4 heads: 1560 workgroups = 3.05
# rounds of 512 -> 4160 = 8.1, 2921 -> 2735 us). Shapes whose count already
# fills its rounds keep the 2048 target.
MIN_ROUND_FILL = 0.9
WORKGROUPS_PER_CU = 2
# Segments per query tile: small query counts on long contexts need many short
# segments to fill the GPU (64 left 64 query rows on 128K at 16 serial blocks per
# workgroup, 2.7x the removed HIP scorer's time); larger shapes stop at the target.
MAX_SEGMENTS = 1024


_NVFP4_E4M3_LUT: dict[torch.device, torch.Tensor] = {}


def nvfp4_e4m3_lut(device: torch.device) -> torch.Tensor:
    """[256] int64: byte j of entry s is e4m3_rn_satfinite(m_j * e4m3(s)) for
    the E2M1 magnitudes m = 0, 0.5, 1, 1.5, 2, 3, 4, 6 -- the fp8 key of code
    magnitude j under block scale byte s (shared by the NVFP4 index scorers and
    the sparse decode)."""
    lut = _NVFP4_E4M3_LUT.get(device)
    if lut is None:
        # Device ops only (no host copy), so a first call under graph capture
        # is fine.
        scales = torch.arange(256, dtype=torch.uint8, device=device)
        scales = scales.view(torch.float8_e4m3fn).float()
        code = torch.arange(8, device=device)
        exp, man = code // 2, (code % 2).float()
        # E2M1 magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6.
        mags = torch.where(
            exp == 0, man * 0.5, torch.exp2((exp - 1).float()) * (1.0 + man * 0.5)
        )
        # e2m1 * e4m3 is exact in fp32; saturate, then round once to E4M3.
        values = (scales[:, None] * mags[None, :]).clamp(-448.0, 448.0)
        lut = values.to(torch.float8_e4m3fn).view(torch.int64).reshape(256)
        _NVFP4_E4M3_LUT[device] = lut
    return lut


@functools.cache
def _num_cus(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


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
def _get_kernel(heads: int, staged: bool = True):
    return _kernel.compile_score_prefill(heads, staged)


def _kernel_shared():
    # imported late: it shares helpers with the decode kernel, whose host
    # imports this module
    from . import kernel_shared

    return kernel_shared


@functools.cache
def _get_kernel_shared(heads: int, mq: int, fp8: bool):
    return _kernel_shared().compile_score_prefill_shared(heads, mq, fp8)


# Small work (<= 128 query rows, or blocks x 256-row tiles x requests x heads
# below SMALL_WORK) takes the shared-dequant kernel with 128-row workgroup tiles,
# on either route; larger work the exact route that kernel with 256-row tiles,
# the fp8 route ``kernel.py``. The shared kernel aims at SMALL_TARGET workgroups
# below MID_WORK (a workgroup of one block exposes its whole prologue; exact,
# 2-16 requests of 128-512 rows over 4K-128K, targets 512 / 1024 / 2048:
# 1.03-1.45x / 0.92-1.30x / 0.78-1.32x the removed HIP scorer) and TARGET_WORKGROUPS
# above (512 there: 0.79x at 32K x 32K).
SMALL_WORK = 4096
MID_WORK = 16384
SMALL_TARGET = 512
# Tests / benchmarks: pin the kernel (True: kernel_shared), its 32-row tiles per
# wave, or the workgroup target.
PIN_SHARED: bool | None = None
PIN_MQ: int | None = None
PIN_TARGET: int | None = None
# kernel.py writes a row's scores 16 blocks (64 B) at a time from LDS
# (``kernel.SPAN``); False stores each 4-block group as it completes. Either way
# the same bytes.
STAGED_STORES = True


def supports_nvfp4_index_score_prefill_fly(
    q_idx: torch.Tensor, score: torch.Tensor
) -> bool:
    """gfx950, 1, 2 or 4 index heads per rank (TP4 / TP2 / TP1), contiguous e4m3
    query rows [total_q, H, 128]."""
    from vllm.platforms.rocm import on_gfx950

    heads = q_idx.shape[1]
    return (
        on_gfx950()
        and q_idx.dtype == torch.float8_e4m3fn
        and heads in (1, 2, 4)
        and q_idx.shape[2] == _kernel.D
        and q_idx.stride(2) == 1
        and q_idx.stride(1) == _kernel.D
        and q_idx.stride(0) == heads * _kernel.D
        and score.shape[0] == heads
        and score.stride(2) == 1
    )


def minimax_m3_nvfp4_index_score_prefill_fly(
    q_idx: torch.Tensor,  # [total_q, H, 128] e4m3
    index_cache: torch.Tensor,  # NVFP4 index pages [num_blocks, 128, 72] uint8
    score: torch.Tensor,  # [H, total_q, >= cdiv(max_seq_len, 128)] fp32
    block_table: torch.Tensor,  # [batch, >= max_blocks] int32
    cu_seqlens_q: torch.Tensor,  # [batch + 1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    init_blocks: int,
    local_blocks: int,
    max_query_len: int,
    max_seq_len: int,
    exact: bool = False,
) -> None:
    """``minimax_m3_nvfp4_index_score_prefill`` for 1, 2 or 4 index heads, fp8
    keys or (``exact``) exact f16 keys. The heads share the index keys: a query
    row is a (token, head) pair, so every dequantized key block serves all of
    them."""
    batch = cu_seqlens_q.shape[0] - 1
    total_q, heads, _ = q_idx.shape
    max_blocks = _cdiv(max_seq_len, _kernel.BLK)
    assert max_blocks <= score.shape[2]
    work = max_blocks * _cdiv(max_query_len, 256) * batch * heads
    small = max_query_len * heads <= 128 or work < SMALL_WORK
    shared = exact or (small if PIN_SHARED is None else PIN_SHARED)
    mq = PIN_MQ or (1 if small else 2)
    target = PIN_TARGET or (
        SMALL_TARGET if shared and work < MID_WORK else TARGET_WORKGROUPS
    )
    tile_q = _kernel_shared().tile_q(mq) if shared else _kernel.TILE_Q
    qt = _cdiv(max_query_len * heads, tile_q)
    # kernel.py: segments of whole 4-block store groups; shared: single blocks
    min_seg = 1 if shared else _kernel.GROUP
    max_nseg = max(1, min(MAX_SEGMENTS, max_blocks // min_seg))
    nseg = max(1, min(max_nseg, target // max(1, qt * batch)))
    slots = WORKGROUPS_PER_CU * _num_cus(q_idx.device)
    while nseg < max_nseg:
        rounds = qt * batch * nseg / slots
        if rounds / math.ceil(rounds) >= MIN_ROUND_FILL:
            break
        nseg += 1
    lut = () if shared else (nvfp4_e4m3_lut(q_idx.device).data_ptr(),)
    _run_compiled(
        (
            _get_kernel_shared(heads, mq, not exact)
            if shared
            else _get_kernel(heads, STAGED_STORES)
        ),
        q_idx.data_ptr(),
        index_cache.data_ptr(),
        *lut,
        score.data_ptr(),
        block_table.data_ptr(),
        cu_seqlens_q.data_ptr(),
        seq_lens.data_ptr(),
        int(index_cache.stride(0)),
        int(block_table.stride(0)),
        int(score.stride(1)),
        int(nseg),
        int(qt),
        int(total_q),
        int(batch),
        int(((batch - 1) * block_table.stride(0) + block_table.shape[1]) * 4),
        int(init_blocks),
        int(local_blocks),
        int(score.stride(0)),
        int(_cdiv(qt * nseg * batch, 8) * 8),
        torch.cuda.current_stream(),
    )
