# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact top-p thresholds over bf16 logits for speculative-decoding verification.

For each row, the boundary value v of the top-p nucleus of softmax(x / T); the
rejection kernels keep a token iff its raw logit is >= v and apply the temperature on
load, so no masked fp32 copy of the logits is made. Kept sets equal the fp64
stable-sort reference, except that every token tied with v is kept.
"""

import torch

from vllm.triton_utils import tl, triton

# Order-preserving 16-bit key of a bf16 value: flip all bits of negative values and the
# sign bit of non-negative ones, so key order is value order (-0.0 is keyed as +0.0).
_KEY_BINS_SHIFT = 4  # coarse bins of 16 consecutive keys
_NUM_COARSE_BINS = 1 << (16 - _KEY_BINS_SHIFT)
_WINDOW = 2048  # exact keys histogrammed per refinement pass
_BLOCK = 4096
# Fixed-point fraction bits of a token's mass exp((x - max) / T) <= 1. int64 sums
# hold up to 2**(62 - _MASS_BITS) tokens.
_MASS_BITS = 40
_MAX_VOCAB = 1 << (62 - _MASS_BITS)
# Test hook: send every row with top-p through the fp64 kernel.
_FORCE_FALLBACK = False


@triton.jit
def _bf16_key(x_bf16):
    bits = x_bf16.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    bits = tl.where(bits == 0x8000, 0, bits)
    return tl.where(bits >= 0x8000, bits ^ 0xFFFF, bits ^ 0x8000)


@triton.jit
def _key_value(key):
    bits = tl.where(key >= 0x8000, key ^ 0x8000, key ^ 0xFFFF)
    return bits.to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32)


@triton.jit
def _topp_radix_fallback_kernel(
    # [num_logits]
    thresh_ptr,
    # [num_logits, 3]
    band_ptr,
    # [num_logits, 2]
    stats_ptr,
    # [num_logits, V] bf16
    logits_ptr,
    logits_stride,
    # [num_logits]
    expanded_idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    # [max_num_reqs]
    top_p_ptr,
    vocab_size,
    BLOCK: tl.constexpr,
    NUM_COARSE_BINS: tl.constexpr,
    BINS_SHIFT: tl.constexpr,
    WINDOW: tl.constexpr,
    MAX_WINDOWS: tl.constexpr,
    STATS: tl.constexpr,
):
    """fp64 boundary of the rows the radix2 kernels left flagged (NaN), from the
    candidate key band [band_lo, band_hi] and key of the max in band_ptr."""
    row = tl.program_id(0).to(tl.int64)
    thresh = tl.load(thresh_ptr + row)
    if thresh == thresh:
        return
    req_state_idx = tl.load(expanded_idx_mapping_ptr + row).to(tl.int64)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float64)
    top_p = tl.load(top_p_ptr + req_state_idx).to(tl.float64)
    band_lo = tl.load(band_ptr + 3 * row)
    band_hi = tl.load(band_ptr + 3 * row + 1)
    key_max = tl.load(band_ptr + 3 * row + 2)
    m = _key_value(key_max).to(tl.float64)
    inv_temp = 1.0 / temp
    row_ptr = logits_ptr + row * logits_stride

    # The row total and the mass above the candidate keys, in fp64.
    total = tl.zeros((), dtype=tl.float64)
    above = tl.zeros((), dtype=tl.float64)
    key_min = key_max
    coarse = tl.zeros((NUM_COARSE_BINS,), dtype=tl.int32)
    for start in range(0, vocab_size, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        in_vocab = offs < vocab_size
        raw = tl.load(row_ptr + offs, mask=in_vocab, other=float("-inf"))
        x = raw.to(tl.float32)
        valid = in_vocab & (x == x) & (x > float("-inf"))
        key = _bf16_key(raw)
        x64 = tl.where(valid, x, m.to(tl.float32)).to(tl.float64)
        e = tl.where(valid, tl.exp((x64 - m) * inv_temp), 0.0)
        total += tl.sum(e, axis=0)
        above += tl.sum(tl.where(key > band_hi, e, 0.0), axis=0)
        key_min = tl.minimum(key_min, tl.min(tl.where(valid, key, key_max), axis=0))
        coarse += tl.histogram(key >> BINS_SHIFT, NUM_COARSE_BINS, mask=valid)
    target = top_p * total
    bins = tl.arange(0, NUM_COARSE_BINS)
    occupied = coarse > 0
    band_lo = tl.maximum(band_lo, key_min)

    # Exact per-key counts over the candidate keys, from the top, window by window.
    widx = tl.arange(0, WINDOW)
    win_hi = band_hi
    win_lo = tl.maximum(win_hi - WINDOW + 1, band_lo)
    result = band_lo
    n_passes = 1
    done = False
    for _ in range(MAX_WINDOWS):
        if not done:
            n_passes += 1
            counts = tl.zeros((WINDOW,), dtype=tl.int32)
            for start in range(0, vocab_size, BLOCK):
                offs = start + tl.arange(0, BLOCK)
                in_vocab = offs < vocab_size
                raw = tl.load(row_ptr + offs, mask=in_vocab, other=float("-inf"))
                x = raw.to(tl.float32)
                valid = in_vocab & (x == x) & (x > float("-inf"))
                key = _bf16_key(raw)
                in_win = valid & (key >= win_lo) & (key <= win_hi)
                counts += tl.histogram(key - win_lo, WINDOW, mask=in_win)
            wkey = win_lo + widx
            in_range = (counts > 0) & (wkey <= win_hi)
            wval = tl.where(in_range, _key_value(wkey), m.to(tl.float32)).to(tl.float64)
            wmass = tl.where(
                in_range, counts.to(tl.float64) * tl.exp((wval - m) * inv_temp), 0.0
            )
            mass_from = above + tl.cumsum(wmass, axis=0, reverse=True)
            hit = tl.max(tl.where(in_range & (mass_from >= target), widx, -1), axis=0)
            if hit >= 0:
                result = win_lo + hit
                done = True
            elif win_lo <= band_lo:
                done = True  # band_lo is kept in any case
            else:
                above += tl.sum(wmass, axis=0)
                nxt = tl.max(
                    tl.where(occupied & ((bins << BINS_SHIFT) < win_lo), bins, -1),
                    axis=0,
                )
                win_hi = tl.minimum(
                    (nxt << BINS_SHIFT) + (1 << BINS_SHIFT) - 1, win_lo - 1
                )
                win_lo = tl.maximum(win_hi - WINDOW + 1, band_lo)
                if win_hi < band_lo:
                    done = True
    tl.store(thresh_ptr + row, _key_value(result))
    if STATS:
        tl.store(stats_ptr + 2 * row + 1, n_passes)


# radix2: a two-digit radix select over _R2_SPLITS programs per row, with integer (or
# fixed-order fp64) sums, so the result is exact and deterministic: slice max, per-bin
# fixed-point mass of the high byte, boundary bin, exact low-byte counts, boundary key.
# Rows within the error bound of a bin edge are flagged (NaN) and resolved in fp64;
# _topp_radix_fallback_kernel is the last resort.
_R2_SPLITS = 8
_R2_BLOCK = 2048
_R2_NUM_WARPS = 8
_R2_VEC = 8  # bf16 per 16-byte load: the tokens each thread of the mass kernel sums
# Mass kernel register cap: 3 CTAs of _R2_NUM_WARPS warps per SM (no spills).
_R2_MASS_MAX_REGS = 80
# Tokens in high-byte bins wholly below max - _R2_CUT_LOG2 * T * ln 2 (mass below
# 2**-_R2_CUT_LOG2 of the max's) skip the per-bin atomics.
_R2_CUT_LOG2 = 20.0
_R2_CAND = 4  # candidate high-byte bins of the fp64 path (one spare slot)


@triton.jit
def _r2_params(row, expanded_idx_mapping_ptr, temp_ptr, top_p_ptr):
    req_state_idx = tl.load(expanded_idx_mapping_ptr + row).to(tl.int64)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    top_p = tl.load(top_p_ptr + req_state_idx).to(tl.float32)
    # log2(e) / T as the unevaluated sum s_hi + s_lo
    s = 1.4426950408889634 / temp.to(tl.float64)
    s_hi = s.to(tl.float32)
    s_lo = (s - s_hi.to(tl.float64)).to(tl.float32)
    return temp, top_p, s_hi, s_lo


@triton.jit
def _r2_token_mass(x, m, s_hi, s_lo, MASS_BITS: tl.constexpr):
    """(a, e, q) of exp((x - m) / T) = exp2(a) with an exact argument: x - m is exact
    for bf16 values (up to one rounding when |x| << |m|), and a = (x - m) * (s_hi +
    s_lo) is kept as a_hi + a_lo (FMA), so exp2(a) = exp2(a_hi) * (1 + a_lo * ln 2).
    What is left: the exp2 approximation (<= 2**-21 relative) and |a| * 2**-24 when
    x - m rounds.
    q = floor(e * 2**MASS_BITS)."""
    d = x - m
    a_hi = d * s_hi
    a_lo = tl.fma(d, s_hi, -a_hi) + d * s_lo
    a_hi = tl.minimum(a_hi, 0.0)
    e = tl.minimum(tl.exp2(a_hi) * (1.0 + a_lo * 0.6931471805599453), 1.0)
    return a_hi, e, (e * (2.0**MASS_BITS)).to(tl.int64)


@triton.jit
def _r2_row_max(split_max_ptr, row, SPLITS: tl.constexpr):
    return tl.max(tl.load(split_max_ptr + row * SPLITS + tl.arange(0, SPLITS)), axis=0)


@triton.jit
def _r2_cut_bin(m, temp, CUT_LOG2: tl.constexpr):
    """High-byte key bin of max - CUT_LOG2 * T * ln 2: bins below it are light."""
    lb = m - CUT_LOG2 * 0.6931471805599453 * temp
    return _bf16_key(tl.cast(lb, tl.bfloat16)) >> 8


@triton.jit
def _r2_max_kernel(
    # [num_logits, SPLITS]
    split_max_ptr,
    # [num_logits, SPLITS, 256] int64: zeroed here for _r2_mass_kernel
    bin_mass_ptr,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temp_ptr,
    top_p_ptr,
    vocab_size,
    slice_len,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    temp, top_p, _, _ = _r2_params(row, expanded_idx_mapping_ptr, temp_ptr, top_p_ptr)
    if (temp == 0.0) | (top_p >= 1.0):
        return
    bins = tl.arange(0, 256)
    tl.store(
        bin_mass_ptr + (row * SPLITS + split) * 256 + bins, tl.zeros((256,), tl.int64)
    )
    start = split * slice_len
    end = tl.minimum(start + slice_len, vocab_size)
    row_ptr = logits_ptr + row * logits_stride
    # Elementwise running max (one reduction at the end); the next block's load
    # is issued before this one is consumed.
    run_max = tl.full((BLOCK,), float("-inf"), dtype=tl.float32)
    offs = start + tl.arange(0, BLOCK)
    nxt = tl.load(row_ptr + offs, mask=offs < end, other=float("-inf"))
    for off in range(start, end, BLOCK):
        x = nxt.to(tl.float32)
        offs = off + BLOCK + tl.arange(0, BLOCK)
        nxt = tl.load(row_ptr + offs, mask=offs < end, other=float("-inf"))
        run_max = tl.maximum(run_max, tl.where(x == x, x, float("-inf")))
    tl.store(split_max_ptr + row * SPLITS + split, tl.max(run_max, axis=0))


@triton.jit
def _r2_bin_floor(b):
    """The smallest value of high-byte bin b as fp32, built from its bits (bin 0
    starts at -inf: its lowest keys are NaN patterns). A non-NaN x lies in a bin
    >= b iff x >= it, -0.0 and subnormals included (an exact fp32 compare)."""
    key = b << 8
    bits = tl.where(key >= 0x8000, key ^ 0x8000, key ^ 0xFFFF)
    return tl.where(b > 0, (bits << 16).to(tl.float32, bitcast=True), float("-inf"))


@triton.jit
def _r2_bin_range(
    bins_ptr,
    row_ptr,
    start,
    end,
    m,
    s_hi,
    s_lo,
    lo,
    hi,
    BLOCK: tl.constexpr,
    MASS_BITS: tl.constexpr,
):
    """Adds the mass of the slice's tokens in high-byte bins lo..hi to their bins
    (relaxed int64 atomics)."""
    for off in range(start, end, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        raw = tl.load(row_ptr + offs, mask=offs < end, other=float("-inf"))
        x = raw.to(tl.float32)
        valid = (offs < end) & (x == x) & (x > float("-inf"))
        _, _, q = _r2_token_mass(x, m, s_hi, s_lo, MASS_BITS)
        hb = _bf16_key(raw) >> 8
        add = valid & (hb >= lo) & (hb <= hi)
        tl.atomic_add(bins_ptr + hb, q, mask=add, sem="relaxed")


@triton.jit
def _r2_mass_kernel(
    # [num_logits, SPLITS, 256] int64
    bin_mass_ptr,
    # [num_logits, SPLITS]: int64 mass, int64 light mass, fp32 error bound, int32 count
    part_mass_ptr,
    part_light_ptr,
    part_err_ptr,
    part_n_ptr,
    # [num_logits, 8] int64
    sel_ptr,
    split_max_ptr,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temp_ptr,
    top_p_ptr,
    vocab_size,
    slice_len,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
    VEC: tl.constexpr,
    MASS_BITS: tl.constexpr,
    CUT_LOG2: tl.constexpr,
    DEEP: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    temp, top_p, s_hi, s_lo = _r2_params(
        row, expanded_idx_mapping_ptr, temp_ptr, top_p_ptr
    )
    if (temp == 0.0) | (top_p >= 1.0):
        return
    if DEEP:  # noqa: SIM102 - constexpr branch around a runtime one
        if tl.load(sel_ptr + 8 * row) != -2:
            return
    m = _r2_row_max(split_max_ptr, row, SPLITS)
    if (m == float("-inf")) | (m == float("inf")):
        return
    cut = _r2_cut_bin(m, temp, CUT_LOG2)
    start = split * slice_len
    end = tl.minimum(start + slice_len, vocab_size)
    row_ptr = logits_ptr + row * logits_stride
    bins_ptr = bin_mass_ptr + (row * SPLITS + split) * 256
    if DEEP:
        # The light tokens' bins, which the first pass only summed.
        _r2_bin_range(
            bins_ptr, row_ptr, start, end, m, s_hi, s_lo, 0, cut - 1, BLOCK, MASS_BITS
        )
    else:
        # The top 4 heavy bins are summed in registers (w_j covers bins >= top - j);
        # lower heavy bins get atomics in a second loop.
        top = _bf16_key(tl.cast(m, tl.bfloat16)) >> 8
        nwin = tl.minimum(top - cut + 1, 4)
        wide = cut < top - 3
        v0 = _r2_bin_floor(top)
        v1 = _r2_bin_floor(top - 1)
        v2 = _r2_bin_floor(top - 2)
        v3 = _r2_bin_floor(top - 3)
        vc = _r2_bin_floor(cut)
        G: tl.constexpr = BLOCK // VEC
        tot = tl.zeros((G,), dtype=tl.int64)
        w0 = tl.zeros((G,), dtype=tl.int64)
        w1 = tl.zeros((G,), dtype=tl.int64)
        w2 = tl.zeros((G,), dtype=tl.int64)
        w3 = tl.zeros((G,), dtype=tl.int64)
        lit = tl.zeros((G,), dtype=tl.int64)
        cnt = tl.zeros((G,), dtype=tl.int32)
        err = 0.0
        offs = start + tl.arange(0, BLOCK)
        nxt = tl.load(row_ptr + offs, mask=offs < end, other=float("-inf"))
        for off in range(start, end, BLOCK):
            raw = nxt
            offs = off + tl.arange(0, BLOCK)
            nxt = tl.load(
                row_ptr + offs + BLOCK, mask=offs + BLOCK < end, other=float("-inf")
            )
            x = raw.to(tl.float32)
            valid = (offs < end) & (x == x) & (x > float("-inf"))
            a, e, q = _r2_token_mass(x, m, s_hi, s_lo, MASS_BITS)
            # The same fp32 sum per BLOCK as before: a bit-identical error bound.
            err += tl.sum(tl.where(valid, e * (2.0**-21 - a * 2.0**-24), 0.0), axis=0)
            # Integer sums are order-free: each thread first sums its VEC tokens.
            q = tl.reshape(tl.where(valid, q, 0), (G, VEC))
            xg = tl.reshape(x, (G, VEC))
            tot += tl.sum(q, axis=1)
            w0 += tl.sum(tl.where(xg >= v0, q, 0), axis=1)
            if nwin > 1:
                w1 += tl.sum(tl.where(xg >= v1, q, 0), axis=1)
            if nwin > 2:
                w2 += tl.sum(tl.where(xg >= v2, q, 0), axis=1)
            if nwin > 3:
                w3 += tl.sum(tl.where(xg >= v3, q, 0), axis=1)
                lit += tl.sum(tl.where(xg < vc, q, 0), axis=1)
            cnt += tl.sum(tl.reshape(valid.to(tl.int32), (G, VEC)), axis=1)
        mass = tl.sum(tot, axis=0)
        s0 = tl.sum(w0, axis=0)
        s1 = tl.sum(w1, axis=0)
        s2 = tl.sum(w2, axis=0)
        s3 = tl.sum(w3, axis=0)
        tl.store(bins_ptr + top, s0)
        tl.store(bins_ptr + top - 1, s1 - s0, mask=nwin > 1)
        tl.store(bins_ptr + top - 2, s2 - s1, mask=nwin > 2)
        tl.store(bins_ptr + top - 3, s3 - s2, mask=nwin > 3)
        heavy = tl.where(
            nwin > 1, tl.where(nwin > 2, tl.where(nwin > 3, s3, s2), s1), s0
        )
        light = tl.where(wide, tl.sum(lit, axis=0), mass - heavy)
        tl.store(part_mass_ptr + row * SPLITS + split, mass)
        tl.store(part_light_ptr + row * SPLITS + split, light)
        tl.store(part_err_ptr + row * SPLITS + split, err)
        tl.store(part_n_ptr + row * SPLITS + split, tl.sum(cnt, axis=0))
        if wide:
            _r2_bin_range(
                bins_ptr,
                row_ptr,
                start,
                end,
                m,
                s_hi,
                s_lo,
                cut,
                top - 4,
                BLOCK,
                MASS_BITS,
            )


@triton.jit
def _r2_select_kernel(
    thresh_ptr,
    # [num_logits, 3] int32: the generic fallback's candidate key range and max key
    band_ptr,
    # [num_logits, 8] int64: boundary bin (-1: done, -2: deep), mass above it, target,
    # error band, deep flag
    sel_ptr,
    # [num_logits] int32: 1 for the rows the fp64 path resolves (set by finalize)
    flag_ptr,
    stats_ptr,
    # [num_logits, 256] int32: zeroed here for _r2_count_kernel
    key_count_ptr,
    bin_mass_ptr,
    part_mass_ptr,
    part_light_ptr,
    part_err_ptr,
    part_n_ptr,
    split_max_ptr,
    expanded_idx_mapping_ptr,
    temp_ptr,
    top_p_ptr,
    SPLITS: tl.constexpr,
    MASS_BITS: tl.constexpr,
    DEEP: tl.constexpr,
    STATS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    splits = tl.arange(0, SPLITS)
    bins = tl.arange(0, 256)
    if DEEP:
        if tl.load(sel_ptr + 8 * row) != -2:
            return
        target_q = tl.load(sel_ptr + 8 * row + 2)
        band_q = tl.load(sel_ptr + 8 * row + 3)
        if STATS:
            tl.store(stats_ptr + 2 * row, 4)
    else:
        temp, top_p, _, _ = _r2_params(
            row, expanded_idx_mapping_ptr, temp_ptr, top_p_ptr
        )
        tl.store(sel_ptr + 8 * row, -1)
        tl.store(sel_ptr + 8 * row + 4, 0)
        tl.store(flag_ptr + row, 0)
        if STATS:
            tl.store(stats_ptr + 2 * row, 0)
            tl.store(stats_ptr + 2 * row + 1, 0)
        if (temp == 0.0) | (top_p >= 1.0):
            tl.store(thresh_ptr + row, float("-inf"))
            return
        m = _r2_row_max(split_max_ptr, row, SPLITS)
        if m == float("-inf"):
            tl.store(thresh_ptr + row, float("-inf"))
            return
        if m == float("inf"):
            tl.store(thresh_ptr + row, float("inf"))
            return
        if top_p <= 0.0:
            tl.store(thresh_ptr + row, m)
            return
        # Unresolved until _r2_finalize_kernel says otherwise.
        key_max = _bf16_key(tl.cast(m, tl.bfloat16))
        tl.store(thresh_ptr + row, float("nan"))
        tl.store(band_ptr + 3 * row, 0)
        tl.store(band_ptr + 3 * row + 1, key_max)
        tl.store(band_ptr + 3 * row + 2, key_max)
        if STATS:
            tl.store(stats_ptr + 2 * row, 3)
        total_q = tl.sum(tl.load(part_mass_ptr + row * SPLITS + splits), axis=0)
        light_q = tl.sum(tl.load(part_light_ptr + row * SPLITS + splits), axis=0)
        err = tl.sum(tl.load(part_err_ptr + row * SPLITS + splits), axis=0)
        n_valid = tl.sum(tl.load(part_n_ptr + row * SPLITS + splits), axis=0)
        target_q = tl.math.ceil(top_p.to(tl.float64) * total_q.to(tl.float64)).to(
            tl.int64
        )
        # The exp2 bound of every token on both sides of the comparison, plus flooring.
        band_q = (
            2.0 * (tl.cast(err, tl.float64) * (2.0**MASS_BITS) + n_valid) + 4.0
        ).to(tl.int64)
        tl.store(sel_ptr + 8 * row + 2, target_q)
        tl.store(sel_ptr + 8 * row + 3, band_q)
        if total_q - light_q < target_q:
            tl.store(sel_ptr + 8 * row, -2)  # deep: bin the light tokens too
            tl.store(sel_ptr + 8 * row + 4, 1)
            return
    tile = (row * SPLITS + splits[:, None]) * 256 + bins[None, :]
    bin_mass = tl.sum(tl.load(bin_mass_ptr + tile), axis=0)
    mass_from = tl.cumsum(bin_mass, axis=0, reverse=True)  # bins >= b
    boundary = tl.max(tl.where(mass_from >= target_q, bins, -1), axis=0)
    above = tl.sum(tl.where(bins == boundary, mass_from - bin_mass, 0), axis=0)
    tl.store(key_count_ptr + row * 256 + bins, tl.zeros((256,), tl.int32))
    tl.store(sel_ptr + 8 * row, boundary.to(tl.int64))
    tl.store(sel_ptr + 8 * row + 1, above)


@triton.jit
def _r2_count_kernel(
    # [num_logits, 256] int32
    key_count_ptr,
    sel_ptr,
    logits_ptr,
    logits_stride,
    vocab_size,
    slice_len,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    boundary = tl.load(sel_ptr + 8 * row).to(tl.int32)
    if boundary < 0:
        return
    start = tl.program_id(1) * slice_len
    end = tl.minimum(start + slice_len, vocab_size)
    row_ptr = logits_ptr + row * logits_stride
    ones = tl.full((BLOCK,), 1, dtype=tl.int32)
    for off in range(start, end, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        raw = tl.load(row_ptr + offs, mask=offs < end, other=float("-inf"))
        x = raw.to(tl.float32)
        valid = (offs < end) & (x == x) & (x > float("-inf"))
        key = _bf16_key(raw)
        in_bin = valid & ((key >> 8) == boundary)
        tl.atomic_add(
            key_count_ptr + row * 256 + (key & 255), ones, mask=in_bin, sem="relaxed"
        )


@triton.jit
def _r2_finalize_kernel(
    thresh_ptr,
    band_ptr,
    stats_ptr,
    sel_ptr,
    flag_ptr,
    key_count_ptr,
    bin_mass_ptr,
    # [num_logits, CAND] int32: candidate bins of the fp64 path (-1: none)
    cand_ptr,
    # [num_logits, CAND, 256] int32: zeroed here for _r2_fp64_kernel
    cand_count_ptr,
    split_max_ptr,
    expanded_idx_mapping_ptr,
    temp_ptr,
    top_p_ptr,
    SPLITS: tl.constexpr,
    CAND: tl.constexpr,
    MASS_BITS: tl.constexpr,
    CUT_LOG2: tl.constexpr,
    FORCE_FALLBACK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    boundary = tl.load(sel_ptr + 8 * row).to(tl.int32)
    if boundary < 0:
        return
    above_q = tl.load(sel_ptr + 8 * row + 1)
    target_q = tl.load(sel_ptr + 8 * row + 2)
    band_q = tl.load(sel_ptr + 8 * row + 3)
    temp, top_p, s_hi, s_lo = _r2_params(
        row, expanded_idx_mapping_ptr, temp_ptr, top_p_ptr
    )
    m = _r2_row_max(split_max_ptr, row, SPLITS)
    key_max = _bf16_key(tl.cast(m, tl.bfloat16))
    widx = tl.arange(0, 256)
    counts = tl.load(key_count_ptr + row * 256 + widx)
    wkey = boundary * 256 + widx
    in_range = counts > 0
    _, _, wq1 = _r2_token_mass(_key_value(wkey), m, s_hi, s_lo, MASS_BITS)
    wq = tl.where(in_range, counts.to(tl.int64) * wq1, 0)
    mass_from = above_q + tl.cumsum(wq, axis=0, reverse=True)
    hit = tl.max(tl.where(in_range & (mass_from >= target_q), widx, -1), axis=0)
    mass_k = tl.sum(tl.where(widx == hit, mass_from, 0), axis=0)
    mass_gt = mass_k - tl.sum(tl.where(widx == hit, wq, 0), axis=0)
    sure = (hit >= 0) & (mass_k - target_q > band_q) & (target_q - mass_gt > band_q)
    if FORCE_FALLBACK:
        sure = hit < -1
    if sure:
        tl.store(thresh_ptr + row, _key_value(boundary * 256 + hit))
    else:
        # Within the error bound of a bin edge: the fp64 path resolves the row (thresh
        # stays NaN) over B and the occupied neighbors that may hold the boundary.
        bins = tl.arange(0, 256)
        tile = (row * SPLITS + tl.arange(0, SPLITS)[:, None]) * 256 + bins[None, :]
        occupied = tl.sum(tl.load(bin_mass_ptr + tile), axis=0) > 0
        nxt_above = tl.min(tl.where(occupied & (bins > boundary), bins, 256), axis=0)
        nxt_below = tl.max(tl.where(occupied & (bins < boundary), bins, -1), axis=0)
        cut = _r2_cut_bin(m, temp, CUT_LOG2)
        deep = tl.load(sel_ptr + 8 * row + 4)
        if (deep == 0) & (nxt_below < cut):
            nxt_below = cut - 1
        bottom = tl.max(tl.where(in_range, mass_from, 0), axis=0)  # all of B
        c_above = tl.where(
            (above_q >= target_q - band_q) & (nxt_above < 256), nxt_above, -1
        )
        c_below = tl.where(
            (bottom - target_q <= band_q) & (nxt_below >= 0), nxt_below, -1
        )
        slot = tl.arange(0, CAND)
        cands = tl.where(
            slot == 0,
            c_below,
            tl.where(slot == 1, boundary, tl.where(slot == 2, c_above, -1)),
        )
        tl.store(cand_ptr + row * CAND + slot, cands)
        cslot = tl.arange(0, CAND * 256)
        tl.store(
            cand_count_ptr + row * CAND * 256 + cslot, tl.zeros((CAND * 256,), tl.int32)
        )
        tl.store(flag_ptr + row, 1)
        # The generic kernel's candidate key range (last resort).
        hi_idx = tl.max(
            tl.where(in_range & (mass_from >= target_q - band_q), widx, -1), axis=0
        )
        lo_idx = tl.max(
            tl.where(in_range & (mass_from > target_q + band_q), widx, -1), axis=0
        )
        band_hi = key_max
        band_lo = tl.zeros((), dtype=tl.int32)
        if (above_q < target_q - band_q) & (hi_idx >= 0):
            band_hi = boundary * 256 + hi_idx
        if lo_idx >= 0:
            band_lo = boundary * 256 + lo_idx
        tl.store(band_ptr + 3 * row, band_lo)
        tl.store(band_ptr + 3 * row + 1, band_hi)


@triton.jit
def _r2_fp64_kernel(
    # [num_logits, SPLITS, 2] fp64: the slice's total and its mass above the candidates
    part64_ptr,
    cand_ptr,
    cand_count_ptr,
    flag_ptr,
    split_max_ptr,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temp_ptr,
    top_p_ptr,
    vocab_size,
    slice_len,
    SPLITS: tl.constexpr,
    CAND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    if tl.load(flag_ptr + row) != 1:
        return
    temp = tl.load(temp_ptr + tl.load(expanded_idx_mapping_ptr + row).to(tl.int64))
    inv_temp = 1.0 / temp.to(tl.float64)
    m = _r2_row_max(split_max_ptr, row, SPLITS).to(tl.float64)
    c0 = tl.load(cand_ptr + row * CAND)
    c1 = tl.load(cand_ptr + row * CAND + 1)
    c2 = tl.load(cand_ptr + row * CAND + 2)
    c_top = tl.maximum(tl.maximum(c0, c1), c2)
    start = split * slice_len
    end = tl.minimum(start + slice_len, vocab_size)
    row_ptr = logits_ptr + row * logits_stride
    ones = tl.full((BLOCK,), 1, dtype=tl.int32)
    total = tl.zeros((), dtype=tl.float64)
    above = tl.zeros((), dtype=tl.float64)
    for off in range(start, end, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        raw = tl.load(row_ptr + offs, mask=offs < end, other=float("-inf"))
        x = raw.to(tl.float32)
        valid = (offs < end) & (x == x) & (x > float("-inf"))
        key = _bf16_key(raw)
        hb = key >> 8
        x64 = tl.where(valid, x, 0.0).to(tl.float64)
        e = tl.where(valid, tl.exp((x64 - m) * inv_temp), 0.0)
        total += tl.sum(e, axis=0)
        above += tl.sum(tl.where(hb > c_top, e, 0.0), axis=0)
        slot = tl.where(hb == c0, 0, tl.where(hb == c1, 1, tl.where(hb == c2, 2, -1)))
        tl.atomic_add(
            cand_count_ptr + row * CAND * 256 + slot * 256 + (key & 255),
            ones,
            mask=valid & (slot >= 0),
            sem="relaxed",
        )
    tl.store(part64_ptr + (row * SPLITS + split) * 2, total)
    tl.store(part64_ptr + (row * SPLITS + split) * 2 + 1, above)


@triton.jit
def _r2_fp64_finalize_kernel(
    thresh_ptr,
    stats_ptr,
    part64_ptr,
    cand_ptr,
    cand_count_ptr,
    flag_ptr,
    split_max_ptr,
    expanded_idx_mapping_ptr,
    temp_ptr,
    top_p_ptr,
    SPLITS: tl.constexpr,
    CAND: tl.constexpr,
    STATS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if tl.load(flag_ptr + row) != 1:
        return
    req_state_idx = tl.load(expanded_idx_mapping_ptr + row).to(tl.int64)
    inv_temp = 1.0 / tl.load(temp_ptr + req_state_idx).to(tl.float64)
    top_p = tl.load(top_p_ptr + req_state_idx).to(tl.float64)
    m = _r2_row_max(split_max_ptr, row, SPLITS).to(tl.float64)
    parts = (row * SPLITS + tl.arange(0, SPLITS)) * 2
    total = tl.sum(tl.load(part64_ptr + parts), axis=0)
    above = tl.sum(tl.load(part64_ptr + parts + 1), axis=0)
    target = top_p * total
    # The candidate keys in descending order: slots 3 .. 0, low byte 255 .. 0.
    idx = tl.arange(0, CAND * 256)
    slot = CAND - 1 - idx // 256
    low = 255 - idx % 256
    cbin = tl.load(cand_ptr + row * CAND + slot)
    counts = tl.load(cand_count_ptr + row * CAND * 256 + slot * 256 + low)
    key = cbin * 256 + low
    ok = (cbin >= 0) & (counts > 0)
    val = tl.where(ok, _key_value(key), 0.0).to(tl.float64)
    mass = tl.where(ok, counts.to(tl.float64) * tl.exp((val - m) * inv_temp), 0.0)
    # Candidate bins are distinct and slot order is bin order, so this is a suffix
    # sum over keys from the top.
    cum = above + tl.cumsum(mass, axis=0)
    first = tl.min(tl.where(ok & (cum >= target), idx, CAND * 256), axis=0)
    if first < CAND * 256:
        hit_key = tl.sum(tl.where(idx == first, key, 0), axis=0)
        tl.store(thresh_ptr + row, _key_value(hit_key))
    if STATS:
        tl.store(stats_ptr + 2 * row + 1, 1)


def _launch_topp_radix2(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    thresh: torch.Tensor,
    stats: torch.Tensor | None,
) -> None:
    splits, num_warps = _R2_SPLITS, _R2_NUM_WARPS
    num_logits, vocab_size = logits.shape
    dev = logits.device
    slice_len = triton.cdiv(triton.cdiv(vocab_size, splits), _R2_BLOCK) * _R2_BLOCK
    split_max = torch.empty(num_logits, splits, dtype=torch.float32, device=dev)
    bin_mass = torch.empty(num_logits, splits, 256, dtype=torch.int64, device=dev)
    part_mass = torch.empty(num_logits, splits, dtype=torch.int64, device=dev)
    part_light = torch.empty(num_logits, splits, dtype=torch.int64, device=dev)
    part_err = torch.empty(num_logits, splits, dtype=torch.float32, device=dev)
    part_n = torch.empty(num_logits, splits, dtype=torch.int32, device=dev)
    sel = torch.empty(num_logits, 8, dtype=torch.int64, device=dev)
    flag = torch.empty(num_logits, dtype=torch.int32, device=dev)
    key_count = torch.empty(num_logits, 256, dtype=torch.int32, device=dev)
    band = torch.empty(num_logits, 3, dtype=torch.int32, device=dev)
    cand = torch.empty(num_logits, _R2_CAND, dtype=torch.int32, device=dev)
    cand_count = torch.empty(num_logits, _R2_CAND, 256, dtype=torch.int32, device=dev)
    part64 = torch.empty(num_logits, splits, 2, dtype=torch.float64, device=dev)
    rows = (expanded_idx_mapping, temperature, top_p)
    grid = (num_logits, splits)
    has_stats = stats is not None
    _r2_max_kernel[grid](
        split_max,
        bin_mass,
        logits,
        logits.stride(0),
        *rows,
        vocab_size,
        slice_len,
        SPLITS=splits,
        BLOCK=_R2_BLOCK,
        num_warps=num_warps,
    )
    for deep in (False, True):
        _r2_mass_kernel[grid](
            bin_mass,
            part_mass,
            part_light,
            part_err,
            part_n,
            sel,
            split_max,
            logits,
            logits.stride(0),
            *rows,
            vocab_size,
            slice_len,
            SPLITS=splits,
            BLOCK=_R2_BLOCK,
            VEC=_R2_VEC,
            MASS_BITS=_MASS_BITS,
            CUT_LOG2=_R2_CUT_LOG2,
            DEEP=deep,
            num_warps=num_warps,
            maxnreg=_R2_MASS_MAX_REGS,
        )
        _r2_select_kernel[(num_logits,)](
            thresh,
            band,
            sel,
            flag,
            stats,
            key_count,
            bin_mass,
            part_mass,
            part_light,
            part_err,
            part_n,
            split_max,
            *rows,
            SPLITS=splits,
            MASS_BITS=_MASS_BITS,
            DEEP=deep,
            STATS=has_stats,
            num_warps=4,
        )
    _r2_count_kernel[grid](
        key_count,
        sel,
        logits,
        logits.stride(0),
        vocab_size,
        slice_len,
        BLOCK=_R2_BLOCK,
        num_warps=num_warps,
    )
    _r2_finalize_kernel[(num_logits,)](
        thresh,
        band,
        stats,
        sel,
        flag,
        key_count,
        bin_mass,
        cand,
        cand_count,
        split_max,
        *rows,
        SPLITS=splits,
        CAND=_R2_CAND,
        MASS_BITS=_MASS_BITS,
        CUT_LOG2=_R2_CUT_LOG2,
        FORCE_FALLBACK=_FORCE_FALLBACK,
        num_warps=4,
    )
    _r2_fp64_kernel[grid](
        part64,
        cand,
        cand_count,
        flag,
        split_max,
        logits,
        logits.stride(0),
        *rows,
        vocab_size,
        slice_len,
        SPLITS=splits,
        CAND=_R2_CAND,
        BLOCK=_R2_BLOCK,
        num_warps=num_warps,
    )
    _r2_fp64_finalize_kernel[(num_logits,)](
        thresh,
        stats,
        part64,
        cand,
        cand_count,
        flag,
        split_max,
        *rows,
        SPLITS=splits,
        CAND=_R2_CAND,
        STATS=has_stats,
        num_warps=4,
    )
    _topp_radix_fallback_kernel[(num_logits,)](
        thresh,
        band,
        stats,
        logits,
        logits.stride(0),
        *rows,
        vocab_size,
        BLOCK=_BLOCK,
        NUM_COARSE_BINS=_NUM_COARSE_BINS,
        BINS_SHIFT=_KEY_BINS_SHIFT,
        WINDOW=_WINDOW,
        MAX_WINDOWS=(1 << 16) // _WINDOW + 1,
        STATS=has_stats,
        num_warps=16,
    )


def compute_topp_thresholds_radix2(
    # [num_logits, V] bf16
    logits: torch.Tensor,
    # [num_logits]
    expanded_idx_mapping: torch.Tensor,
    # [max_num_reqs]
    temperature: torch.Tensor,
    # [max_num_reqs]
    top_p: torch.Tensor,
    # [num_logits, 2] int32: fp32 passes and fp64 passes per row (benchmarks)
    stats: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-row top-p threshold in raw logit units: keep a token iff logit >= thresh.

    -inf means no masking (greedy, top_p >= 1 or no finite logit); +inf keeps only
    +inf logits. A row with top-p always gets a finite threshold otherwise.
    """
    assert logits.dtype == torch.bfloat16 and logits.ndim == 2
    assert logits.stride(-1) == 1
    num_logits, vocab_size = logits.shape
    assert vocab_size <= _MAX_VOCAB
    thresh = torch.empty(num_logits, dtype=torch.float32, device=logits.device)
    if num_logits == 0:
        return thresh
    _launch_topp_radix2(logits, expanded_idx_mapping, temperature, top_p, thresh, stats)
    return thresh
