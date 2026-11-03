# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 with Macro Block Scaling (MBS, arXiv:2603.08713).

Format of a tensor with rows R and reduction dim K:
  * q: fp4 e2m1 codes, two per byte (low nibble = even k), uint8 [R, K/2];
  * s: one E8M0 scale per 32 k, chosen with the aiter/Quark "even" rule,
       uint8 [R, K/32] (row-major);
  * m: one E0M8 factor per macro block of MB k, stored as its 8 mantissa
       bits, f = 1 + m/256 in [1, 2), uint8 [R, K/MB].
A macro block is multiplied by f before plain MXFP4 quantization, so
    x ~= fp4(q) * 2^(s - 127) / f.
With m = 0 everywhere this is exactly aiter's MXFP4.

MBS-S picks m from the top 8 mantissa bits of 6/amax(macro block). MBS-D
searches `ncand` candidates for the smallest squared error: spread
geometrically over one octave starting at the MBS-S factor, or all 256 values.

A GEMM over two MBS operands computes
    C[i, j] = sum_kb sigma_a[i, kb] * sigma_b[j, kb] * P_kb[i, j],  sigma = 1/f,
where P_kb is the scaled-MFMA (E8M0) partial sum over macro block kb. It keeps
one accumulator by multiplying it by ratio[kb] = f[kb] / f[kb - 1] of each side
on entering macro block kb, and by sigma of the last block in the epilogue.
"""

import functools

import torch

from vllm.triton_utils import tl, triton

MBS_MACRO_BLOCKS = (128, 256, 512, 1024)
FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# ----------------------------------------------------------------------------
# Reference (torch). Mirrors the kernel arithmetic so results compare exactly.
# ----------------------------------------------------------------------------


def _fp4_code_ref(a: torch.Tensor) -> torch.Tensor:
    """|value| (fp32, divided by the E8M0 scale) -> e2m1 code 0..7: round to
    nearest, ties to even code, saturate at 6."""
    return (
        (a > 0.25).int()
        + (a >= 0.75).int()
        + (a > 1.25).int()
        + (a >= 1.75).int()
        + (a > 2.5).int()
        + (a >= 3.5).int()
        + (a > 5.0).int()
    )


def _e8m0_even_ref(amax: torch.Tensor) -> torch.Tensor:
    """Unbiased E8M0 exponent of amax (fp32 >= 0), aiter's `even` rule."""
    r = (amax.view(torch.int32) + 0x200000) & 0x7F800000
    return (((r >> 23) & 0xFF) - 129).clamp(-127, 127)


def _octave_candidates(ncand: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [2.0 ** (-j / ncand) for j in range(ncand)], dtype=torch.float32, device=device
    )


def _mx_qdq_ref(xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MXFP4 quant-dequant of xs [..., n*32] (fp32): (codes, exponents, deq)."""
    shp = xs.shape
    xb = xs.reshape(*shp[:-1], shp[-1] // 32, 32)
    e = _e8m0_even_ref(xb.abs().amax(-1))
    inv = ((127 - e) << 23).view(torch.float32)
    qx = xb * inv[..., None]
    code = _fp4_code_ref(qx.abs())
    val = torch.tensor(FP4_VALUES, device=xs.device)[code.long()]
    scale = ((e + 127).clamp(min=0) << 23).view(torch.float32)
    deq = torch.where(qx < 0, -val, val) * scale[..., None]
    sign = (qx.view(torch.int32) >> 31) & 1
    return (code | (sign << 3)).reshape(shp), e, deq.reshape(shp)


def mbs_factor_ref(x: torch.Tensor, mb: int = 128, ncand: int = 1) -> torch.Tensor:
    """E0M8 mantissas m [R, K/mb] (int32) that MBS picks for x [R, K]."""
    R, K = x.shape
    xm = x.float().reshape(R, K // mb, mb)
    amax = xm.abs().amax(-1)
    m0 = ((6.0 / amax).view(torch.int32) & 0x7F8000) >> 15
    if ncand == 1:
        return m0
    if ncand == 256:
        cands = [torch.full_like(m0, j) for j in range(256)]
    else:
        f0 = 1.0 + m0.float() / 256.0
        cands = []
        for c in _octave_candidates(ncand, x.device).tolist():
            f = f0 * torch.tensor(c, dtype=torch.float32, device=x.device)
            f = torch.where(f < 1.0, f * 2.0, f)
            cands.append(((f - 1.0) * 256.0).int())
    best_m, best_err = None, None
    for mj in cands:
        f = 1.0 + mj.float() / 256.0
        _, _, deq = _mx_qdq_ref(xm * f[..., None])
        err = (deq / f[..., None] - xm).pow(2).sum(-1)
        if best_m is None:
            best_m, best_err = mj, err
        else:
            better = err < best_err
            best_m = torch.where(better, mj, best_m)
            best_err = torch.where(better, err, best_err)
    return best_m


def mbs_quant_ref(
    x: torch.Tensor, mb: int = 128, ncand: int = 1, use_mbs: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference quantizer: (q uint8 [R, K/2], s uint8 [R, K/32], m uint8 [R, K/mb])."""
    R, K = x.shape
    if use_mbs:
        m = mbs_factor_ref(x, mb, ncand)
    else:
        m = torch.zeros(R, K // mb, dtype=torch.int32, device=x.device)
    f = (1.0 + m.float() / 256.0).repeat_interleave(mb, dim=1)
    code, e, _ = _mx_qdq_ref(x.float() * f)
    q = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)
    return q, (e + 127).to(torch.uint8), m.to(torch.uint8)


def mbs_dequant(
    q: torch.Tensor, s: torch.Tensor, m: torch.Tensor | None, mb: int = 128
) -> torch.Tensor:
    """(q, s, m) -> fp32 [..., K]; s and m row-major, m None for plain MXFP4."""
    q = q.view(torch.uint8)
    codes = torch.stack([q & 0xF, q >> 4], dim=-1).reshape(*q.shape[:-1], -1).long()
    lut = torch.tensor(FP4_VALUES + tuple(-v for v in FP4_VALUES), device=q.device)
    val = lut[codes]
    scale = torch.exp2(s.view(torch.uint8).float() - 127.0).repeat_interleave(32, -1)
    out = val * scale
    if m is not None:
        out = out / (1.0 + m.view(torch.uint8).float() / 256.0).repeat_interleave(
            mb, -1
        )
    return out


# ----------------------------------------------------------------------------
# Quantization kernel
# ----------------------------------------------------------------------------


@triton.jit
def _fp4_code(a):
    """|value| (fp32, divided by the E8M0 scale) -> e2m1 code (int32 0..7):
    round to nearest, ties to even code, saturate at 6."""
    c = (a > 0.25).to(tl.int32) + (a >= 0.75).to(tl.int32)
    c += (a > 1.25).to(tl.int32) + (a >= 1.75).to(tl.int32)
    c += (a > 2.5).to(tl.int32) + (a >= 3.5).to(tl.int32)
    c += (a > 5.0).to(tl.int32)
    return c


@triton.jit
def _fp4_value(c):
    """e2m1 magnitude code 0..7 -> fp32 value."""
    normal = ((((c >> 1) + 126) << 23) | ((c & 1) << 22)).to(tl.float32, bitcast=True)
    return tl.where(c < 2, c.to(tl.float32) * 0.5, normal)


@triton.jit
def _e8m0_even(amax):
    """Unbiased E8M0 exponent (int32) of amax (fp32 >= 0), aiter's `even` rule."""
    r = (amax.to(tl.int32, bitcast=True) + 0x200000) & 0x7F800000
    return tl.minimum(tl.maximum(((r >> 23) & 0xFF) - 129, -127), 127)


@triton.jit
def _block_amax(x3):
    """Per-row absolute maximum of x3 [BLOCK_M, MB/32, 32]."""
    return tl.max(tl.max(tl.abs(x3), axis=2), axis=1)


@triton.jit
def _block_sse(x3, f):
    """Squared error of MXFP4(x * f) / f against x, per row. x3 [BLOCK_M, MB/32, 32]."""
    xs = x3 * f[:, None, None]
    e = _e8m0_even(tl.max(tl.abs(xs), axis=2))
    inv = ((127 - e) << 23).to(tl.float32, bitcast=True)
    scale = (tl.maximum(e + 127, 0) << 23).to(tl.float32, bitcast=True)
    qx = xs * inv[:, :, None]
    val = _fp4_value(_fp4_code(tl.abs(qx))) * scale[:, :, None]
    deq = tl.where(qx < 0, -val, val)
    deq = tl.math.div_rn(deq, f[:, None, None])
    d = deq - x3
    return tl.sum(tl.sum(d * d, axis=2), axis=1)


@triton.jit
def _static_m(amax, BLOCK_M: tl.constexpr):
    """MBS-S: top 8 mantissa bits of 6/amax (6/0 = inf has a zero mantissa)."""
    six = tl.full((BLOCK_M,), 6.0, tl.float32)
    return (tl.math.div_rn(six, amax).to(tl.int32, bitcast=True) & 0x7F8000) >> 15


@triton.jit
def _mbs_quant_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    m_ptr,
    cand_ptr,
    M,
    stride_xm,
    stride_qm,
    stride_sm,
    stride_mm,
    MB: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NCAND: tl.constexpr,
    USE_MBS: tl.constexpr,
):
    """One program: BLOCK_M rows x one macro block of MB k.

    q and s are written through 32-bit views of the uint8 outputs. The block is
    viewed as [BLOCK_M, MB/32, 32] right after loading and every factor is
    applied in that view.
    """
    pid_m = tl.program_id(0)
    kb = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ok = offs_m < M
    offs_k = kb * MB + tl.arange(0, MB)
    xs = (
        tl.load(
            x_ptr + offs_m[:, None].to(tl.int64) * stride_xm + offs_k[None, :],
            mask=row_ok[:, None],
            other=0.0,
        )
        .to(tl.float32)
        .reshape(BLOCK_M, MB // 32, 32)
    )

    if USE_MBS:
        m0 = _static_m(_block_amax(xs), BLOCK_M)
        if NCAND == 1:
            mval = m0
        else:
            f0 = 1.0 + m0.to(tl.float32) * (1.0 / 256.0)
            best_m = m0
            best_err = tl.full((BLOCK_M,), float("inf"), tl.float32)
            for j in range(NCAND):
                if NCAND == 256:
                    mj = tl.full((BLOCK_M,), 0, tl.int32) + j
                else:
                    fj = f0 * tl.load(cand_ptr + j)
                    fj = tl.where(fj < 1.0, fj * 2.0, fj)
                    mj = ((fj - 1.0) * 256.0).to(tl.int32)
                err = _block_sse(xs, 1.0 + mj.to(tl.float32) * (1.0 / 256.0))
                better = err < best_err
                best_err = tl.where(better, err, best_err)
                best_m = tl.where(better, mj, best_m)
            mval = best_m
        f = 1.0 + mval.to(tl.float32) * (1.0 / 256.0)
        xs = xs * f[:, None, None]
        tl.store(
            m_ptr + offs_m.to(tl.int64) * stride_mm + kb,
            mval.to(tl.uint8),
            mask=row_ok,
        )

    e = _e8m0_even(tl.max(tl.abs(xs), axis=2))
    inv = ((127 - e) << 23).to(tl.float32, bitcast=True)
    qx = xs * inv[:, :, None]
    code = _fp4_code(tl.abs(qx)) | (((qx.to(tl.int32, bitcast=True) >> 31) & 1) << 3)
    # Codes and scales leave as 32-bit words (8 codes / 4 scales, little endian,
    # code 2b in the low nibble of byte b): storing the bytes themselves from an
    # fp32 input gave run-to-run different codes on gfx950 (Triton 3.7).
    q_words = tl.sum(
        code.reshape(BLOCK_M, MB // 8, 8).to(tl.uint32)
        << (tl.arange(0, 8) * 4).to(tl.uint32)[None, None, :],
        axis=2,
    )
    offs_qw = kb * (MB // 8) + tl.arange(0, MB // 8)
    tl.store(
        q_ptr + offs_m[:, None].to(tl.int64) * stride_qm + offs_qw[None, :],
        q_words.to(tl.int32, bitcast=True),
        mask=row_ok[:, None],
    )
    s_words = tl.sum(
        (e + 127).reshape(BLOCK_M, MB // 128, 4).to(tl.uint32)
        << (tl.arange(0, 4) * 8).to(tl.uint32)[None, None, :],
        axis=2,
    )
    offs_sw = kb * (MB // 128) + tl.arange(0, MB // 128)
    tl.store(
        s_ptr + offs_m[:, None].to(tl.int64) * stride_sm + offs_sw[None, :],
        s_words.to(tl.int32, bitcast=True),
        mask=row_ok[:, None],
    )


@functools.cache
def _candidates(ncand: int, device: torch.device) -> torch.Tensor:
    return _octave_candidates(max(ncand, 1), device)


def mbs_quantize(
    x: torch.Tensor, mb: int = 128, ncand: int = 16, use_mbs: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Quantize x [M, K] (bf16/fp16/fp32) to MXFP4 (+ MBS factors).

    Returns (q uint8 [M, K/2], s uint8 [M, K/32], m uint8 [M, K/mb] or None).
    `ncand` > 1 searches that many factors per macro block (MBS-D); 1 takes the
    static factor (MBS-S), as activations do."""
    assert x.dim() == 2 and x.stride(1) == 1, "expects a row-major 2D tensor"
    M, K = x.shape
    assert mb in MBS_MACRO_BLOCKS and K % mb == 0, (K, mb)
    ncand = ncand if use_mbs else 1
    dev = x.device
    q = torch.empty(M, K // 2, dtype=torch.uint8, device=dev)
    s = torch.empty(M, K // 32, dtype=torch.uint8, device=dev)
    m = torch.empty(M, K // mb, dtype=torch.uint8, device=dev) if use_mbs else None
    if M == 0:
        return q, s, m
    if ncand > 1:
        block_m = 8
    else:
        # small M: no masked rows, and enough programs to fill the GPU
        block_m = min(32, triton.next_power_of_2(M))
        while block_m > 8 and triton.cdiv(M, block_m) * (K // mb) < 512:
            block_m //= 2
    num_warps = (
        4 if ncand > 1 or block_m * mb >= 4096 else (2 if block_m * mb >= 1024 else 1)
    )
    grid = (triton.cdiv(M, block_m), K // mb)
    q_words, s_words = q.view(torch.int32), s.view(torch.int32)
    _mbs_quant_kernel[grid](
        x,
        q_words,
        s_words,
        m if m is not None else q,
        _candidates(ncand, dev),
        M,
        x.stride(0),
        q_words.stride(0),
        s_words.stride(0),
        m.stride(0) if m is not None else 0,
        MB=mb,
        BLOCK_M=block_m,
        NCAND=ncand,
        USE_MBS=use_mbs,
        num_warps=num_warps,
    )
    return q, s, m
