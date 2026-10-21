# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile cache + launch wrapper of the FlyDSL NVFP4 decode index scorer."""

import functools

import torch

from vllm.models.minimax_m3.amd.ops.nvfp4_index_prefill.host import nvfp4_e4m3_lut

from . import kernel as _kernel

# Waves to aim for: 158 VGPRs = 3 waves / SIMD, i.e. 3 workgroups per CU resident.
# Swept 1024..8192 (cold L2, 16-100 requests): 3072 best or within 1 us of it on
# every layout; 1024 is 10-30% slower, 4096+ adds a second round of workgroups.
TARGET_WAVES = 3072


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


# Scores are buffered in LDS per unit of consecutive blocks and stored by whole
# rows once the batch's score footprint reaches this (scattered dword stores
# into 32 MB cost 17% at 16 rows x 512 requests x 128K; 8 MB cost nothing) and
# the units still balance over the wave slots.
BUFFER_MIN_BYTES = 8 << 20
UNITS_PER_SLOT = 4


def _unit(max_blocks: int, slots: int, umax: int, footprint: int) -> int:
    """Consecutive blocks per unit (a power of two, 4..umax) such that every
    wave slot gets >= UNITS_PER_SLOT units, or 0: single strided positions and
    direct stores."""
    if footprint < BUFFER_MIN_BYTES:
        return 0
    u = umax
    while u > 4 and _cdiv(max_blocks, u) < UNITS_PER_SLOT * slots:
        u //= 2
    return u if _cdiv(max_blocks, u) >= UNITS_PER_SLOT * slots else 0


def _run_compiled(exe, *args, waves_per_eu: int | None = None):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the
    cached CompiledFunction. ``waves_per_eu``: the occupancy the register
    allocator must leave room for (``--amdgpu-waves-per-eu``)."""
    import flydsl.compiler as flyc
    from flydsl.compiler.kernel_function import CompilationContext

    cf = getattr(exe, "_cf", None)
    if cf is None:
        with CompilationContext.compile_hints(
            {"waves_per_eu": waves_per_eu} if waves_per_eu else {}
        ):
            exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


@functools.cache
def _get_kernel(rows: int, heads: int, buffered: bool):
    return _kernel.compile_score_decode(rows, heads, False, buffered)


def supports_nvfp4_index_score_decode_fly(
    q_idx: torch.Tensor,
    index_cache: torch.Tensor,
    score: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_len: int,
) -> bool:
    """gfx950, 1 / 2 / 4 index heads with H * query_len <= 32,
    16 B aligned e4m3 query rows, flat 16 B aligned NVFP4 pages, int32 tables."""
    from vllm.platforms.rocm import on_gfx950

    num_heads = q_idx.shape[1]
    return (
        on_gfx950()
        and num_heads in (1, 2, 4)
        and 1 <= num_heads * query_len <= _kernel.MAX_ROWS
        and q_idx.dtype == torch.float8_e4m3fn
        and q_idx.shape[2] == _kernel.D
        and q_idx.stride(2) == 1
        and q_idx.stride(0) % 16 == 0
        and q_idx.stride(1) % 16 == 0
        and q_idx.data_ptr() % 16 == 0
        and index_cache.dtype == torch.uint8
        and index_cache.stride(2) == 1
        and index_cache.stride(1) == _kernel.D // 2 + _kernel.D // 16
        and index_cache.stride(0) % 16 == 0
        and index_cache.data_ptr() % 16 == 0
        and score.dtype == torch.float32
        and score.stride(2) == 1
        and block_table.dtype == torch.int32
        and block_table.stride(1) == 1
        and seq_lens.dtype == torch.int32
        and seq_lens.is_contiguous()
    )


def minimax_m3_nvfp4_index_score_decode_fly(
    q_idx: torch.Tensor,  # [num_reqs * query_len, H, 128] e4m3
    index_cache: torch.Tensor,  # NVFP4 index pages [num_blocks, 128, 72] uint8
    score: torch.Tensor,  # [H, num_reqs * query_len, >= cdiv(max_seq_len, 128)] fp32
    block_table: torch.Tensor,  # [num_reqs, >= max_blocks] int32
    seq_lens: torch.Tensor,  # [num_reqs] int32
    init_blocks: int,
    local_blocks: int,
    query_len: int,
    max_seq_len: int,
) -> None:
    """``minimax_m3_nvfp4_index_score_decode`` for blocks below ``cdiv(seq_len,
    128)``, fp8 keys (``e4m3_rn_satfinite(e2m1 x scale)``)."""
    total_q, num_heads, _ = q_idx.shape
    num_reqs = total_q // query_len
    if num_reqs == 0:
        return
    rows = num_heads * query_len
    max_blocks = max(1, _cdiv(max_seq_len, _kernel.BLK))
    # Chunk size from the work: num_reqs * max_blocks pages over ~TARGET_WAVES
    # waves. A wave of a shared position gets chunk / NW of its pages, so a chunk
    # of NW * (pages per wave) requests; with at most one page per wave, chunks
    # of one request, each wave on its own positions. (Fixed 64-request chunks
    # left 8K contexts with 64 workgroups: 0.4-0.8x the removed HIP scorer.)
    pages_per_wave = _cdiv(num_reqs * max_blocks, TARGET_WAVES)

    def end_bytes(t: torch.Tensor) -> int:
        """Bytes from the data pointer to the end of the last element"""
        return (
            sum((n - 1) * s for n, s in zip(t.shape, t.stride())) + 1
        ) * t.element_size()

    s_bytes = end_bytes(score)
    assert s_bytes < 2**31 * 4, "score offsets must fit in int32 elements"
    footprint = num_reqs * rows * max_blocks * 4
    chunk = 1 if pages_per_wave <= 1 else _kernel.NW * pages_per_wave
    chunk_cap = _kernel.chunk_max(_kernel.padded_rows(rows))
    chunk = max(1, min(chunk, chunk_cap, num_reqs))
    nch = _cdiv(num_reqs, chunk)
    chunk = _cdiv(num_reqs, nch)
    # A workgroup's waves share its positions, a position's pages going round
    # robin over them. With fewer requests per chunk than waves, a wave takes its
    # own positions instead: one shared position would hold fewer pages than
    # waves, and walking it costs a dependent table load per position.
    per_wave = chunk < _kernel.NW
    wanted = max(1, TARGET_WAVES // (_kernel.NW * nch))
    # past 16 rows: the kernel runs 2 waves per SIMD either way and the
    # buffered variant's flush cost 10% (4 heads x 5 queries, 32K-128K)
    unit = (
        _unit(
            max_blocks,
            wanted * (_kernel.NW if per_wave else 1),
            _kernel.UMAX,
            footprint,
        )
        if _kernel.padded_rows(rows) <= 16
        else 0
    )
    units = _cdiv(max_blocks, max(1, unit))
    positions = _cdiv(units, _kernel.NW) if per_wave else units
    splits = max(1, min(positions, wanted))
    _run_compiled(
        _get_kernel(rows, num_heads, unit > 0),
        q_idx.data_ptr(),
        index_cache.data_ptr(),
        nvfp4_e4m3_lut(q_idx.device).data_ptr(),
        score.data_ptr(),
        block_table.data_ptr(),
        seq_lens.data_ptr(),
        int(index_cache.stride(0)),
        int(q_idx.stride(0)),
        int(q_idx.stride(1)),
        end_bytes(q_idx),
        int(score.stride(0)),
        int(score.stride(1)),
        s_bytes,
        int(block_table.stride(0)),
        end_bytes(block_table),
        int(num_reqs),
        int(chunk),
        int(nch),
        int(splits),
        int(init_blocks),
        int(local_blocks),
        int(per_wave),
        int(unit),
        torch.cuda.current_stream(),
        # the buffered variant: the occupancy of the plain one (154-169 VGPRs);
        # its flush pushed the kernel to 169-171, one wave fewer (no hint past
        # 16 rows: it spilled)
        waves_per_eu=3 if unit > 0 and _kernel.padded_rows(rows) <= 16 else None,
    )


@functools.cache
def _get_kernel_cp_push():
    from . import kernel_cp_push

    return kernel_cp_push.compile_score_push()


def minimax_m3_nvfp4_index_score_push_cp(
    iq: torch.Tensor,  # this rank's index head [num_reqs * 4, 128] e4m3
    index_cache: torch.Tensor,  # NVFP4 index pages [num_blocks, 128, 72] uint8
    dests: list[int],  # head h's scores [rows, row_stride] fp32 (rank h's buffer)
    row_stride: int,
    dest_bytes: int,
    block_table: torch.Tensor,  # [num_reqs, >= max_blocks] int32
    seq_lens: torch.Tensor,  # [num_reqs] int32
    init_blocks: int,
    local_blocks: int,
    rank: int,
    counts: torch.Tensor,  # index_cp's device counters
    flags: list[int],  # rank p's flag words
    iq_bufs: list[int],  # rank p's index-q receive buffer [4, iq_rows, 128] e4m3
    iq_rows: int,
    grid: int = 0,
) -> None:
    """Rank ``rank``'s share of the decode scores of all four heads (the blocks b
    with (b // CP_CH) % 4 == rank, bit for bit the single-head scorer's values),
    stored straight into the heads' buffers ``dests``, the four heads' index-q
    rows exchanged on the way; returns once every rank's scores have arrived
    (``kernel_cp_push``)."""
    from . import kernel_cp_push

    rows = iq.shape[0]
    assert rows % 4 == 0 and rows <= iq_rows
    assert iq.dtype == torch.float8_e4m3fn and iq.shape[1] == kernel_cp_push.D
    assert iq.stride(1) == 1 and iq.stride(0) == kernel_cp_push.D
    num_reqs = rows // 4
    assert num_reqs <= kernel_cp_push.MAX_REQS
    if not grid:
        small = num_reqs <= kernel_cp_push.GRID_SMALL_REQS
        grid = kernel_cp_push.GRID_SMALL if small else kernel_cp_push.GRID
    assert 0 < grid <= kernel_cp_push.SC_MAX_WG
    if num_reqs == 0:
        return

    def end_bytes(t: torch.Tensor) -> int:
        return (
            sum((n - 1) * s for n, s in zip(t.shape, t.stride())) + 1
        ) * t.element_size()

    assert dest_bytes < 2**31 * 4, "score offsets must fit in int32 elements"
    _run_compiled(
        _get_kernel_cp_push(),
        iq.data_ptr(),
        index_cache.data_ptr(),
        nvfp4_e4m3_lut(iq.device).data_ptr(),
        *dests,
        block_table.data_ptr(),
        seq_lens.data_ptr(),
        counts.data_ptr(),
        *flags,
        *iq_bufs,
        int(iq_rows),
        int(index_cache.stride(0)),
        int(row_stride),
        int(dest_bytes),
        int(block_table.stride(0)),
        end_bytes(block_table),
        int(num_reqs),
        int(init_blocks),
        int(local_blocks),
        int(rank),
        int(grid),
        torch.cuda.current_stream(),
    )
