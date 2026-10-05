# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 activation quantization with a dynamic per-row outer scale.

This is the MiniMax-M3.1 QAT recipe for the routed-expert FC1 input:

    row_scale = BF16(row_amax / 2688)          # 2688 = 6 * 448
    x_norm    = BF16(x * (2688 / row_amax))
    sf        = E4M3(block_amax(x_norm) / 6)   # one scale per 16 values
    code      = E2M1(x_norm / sf)              # round to nearest even

so ``x ~= code * sf * row_scale``. ``row_scale`` multiplies the FC1 GEMM
output, per token, before the activation. The kernel uses the same PTX as the
QAT dispatch quantizer: ``div.full`` for the row scales, ``rcp.approx`` for the
block reciprocals and ``cvt.rn.satfinite.e2m1x2`` for the codes.
"""

import torch

from vllm.triton_utils import tl, triton

NVFP4_ROW_SCALE_MAX = 6.0 * 448.0


@triton.jit
def _div_full(x, y):
    return tl.inline_asm_elementwise(
        "div.full.f32 $0, $1, $2;", "=f,f,f", [x, y], tl.float32, True, 1
    )


@triton.jit
def _rcp_approx(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", "=f,f", [x], tl.float32, True, 1
    )


@triton.jit
def _cvt_e2m1x2(even, odd):
    """Pack E2M1(even) into the low nibble and E2M1(odd) into the high one."""
    return tl.inline_asm_elementwise(
        "{ .reg .b8 v; cvt.rn.satfinite.e2m1x2.f32 v, $2, $1; cvt.u32.u8 $0, v; }",
        "=r,f,f",
        [even, odd],
        tl.int32,
        True,
        1,
    ).to(tl.uint8)


@triton.jit
def nvfp4_row_scales(row, hidden_size: tl.constexpr, BLOCK_H: tl.constexpr):
    """Encode scale ``2688 / amax`` and BF16 dequantization scale ``amax / 2688``
    of the BF16 row at ``row`` (contiguous)."""
    cols = tl.arange(0, BLOCK_H)
    values = tl.load(row + cols, mask=cols < hidden_size, other=0.0)
    row_amax = tl.maximum(tl.max(tl.abs(values), 0).to(tl.float32), 1.0e-12)
    encode = _div_full(row_amax * 0.0 + 2688.0, row_amax)
    row_scale = _div_full(row_amax, 2688.0).to(tl.bfloat16).to(tl.float32)
    return encode, row_scale


@triton.jit
def nvfp4_quant_chunk(x, encode, BLOCK_K: tl.constexpr):
    """Quantize fp32 row values ``[BLOCK_K]`` with the row's encode scale.

    Returns packed bytes ``[BLOCK_K / 2]`` (low nibble first) and E4M3 block
    scales ``[BLOCK_K / 16]``.
    """
    x_norm = (x * encode).to(tl.bfloat16).to(tl.float32)
    blocks = tl.reshape(x_norm, [BLOCK_K // 16, 16])
    block_amax = tl.max(tl.abs(blocks), 1)
    sf = (block_amax * _rcp_approx(block_amax * 0.0 + 6.0)).to(tl.float8e4nv)
    sf_f32 = sf.to(tl.float32)
    inv = tl.where(sf_f32 == 0.0, 0.0, _rcp_approx(sf_f32))
    even, odd = tl.split(tl.reshape(blocks * inv[:, None], [BLOCK_K // 2, 2]))
    return _cvt_e2m1x2(even, odd), sf


def nvfp4_row_quant_launch_config(num_rows: int, hidden_size: int) -> dict:
    """Chunking for one-row programs: small batches split each row's chunks
    across programs, large ones loop over them in one program per row."""
    small = num_rows < 256
    block_k = min(1024 if small else 2048, triton.next_power_of_2(hidden_size))
    return dict(
        BLOCK_H=triton.next_power_of_2(hidden_size),
        BLOCK_K=block_k,
        SPLIT_K=triton.cdiv(hidden_size, block_k) if small else 1,
    )


@triton.jit
def _nvfp4_row_quant_kernel(
    x,
    x_fp4,
    x_sf,
    x_scale,
    x_stride_m,
    x_fp4_stride_m,
    x_sf_stride_m,
    hidden_size: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    encode, row_scale = nvfp4_row_scales(x + row * x_stride_m, hidden_size, BLOCK_H)
    for k0 in tl.range(split * BLOCK_K, hidden_size, SPLIT_K * BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        values = tl.load(
            x + row * x_stride_m + cols, mask=cols < hidden_size, other=0.0
        ).to(tl.float32)
        packed, sf = nvfp4_quant_chunk(values, encode, BLOCK_K)
        byte_cols = k0 // 2 + tl.arange(0, BLOCK_K // 2)
        tl.store(
            x_fp4 + row * x_fp4_stride_m + byte_cols,
            packed,
            mask=byte_cols < hidden_size // 2,
        )
        sf_cols = k0 // 16 + tl.arange(0, BLOCK_K // 16)
        tl.store(
            x_sf + row * x_sf_stride_m + sf_cols, sf, mask=sf_cols < hidden_size // 16
        )
    if split == 0:
        tl.store(x_scale + row, row_scale)


def nvfp4_row_quantize(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NVFP4-quantize BF16 rows with a per-row outer scale.

    Args:
        x: ``[M, H]`` BF16 activations, ``H`` a multiple of 16.

    Returns:
        ``(x_fp4, x_sf, x_scale)``: uint8 ``[M, H/2]`` packed E2M1 (low nibble
        first), E4M3 ``[M, H/16]`` block scales in linear layout, and fp32
        ``[M]`` per-row dequantization scales.

    """
    assert x.dim() == 2 and x.dtype == torch.bfloat16 and x.stride(1) == 1
    num_rows, hidden_size = x.shape
    assert hidden_size % 16 == 0
    x_fp4 = torch.empty(num_rows, hidden_size // 2, dtype=torch.uint8, device=x.device)
    x_sf = torch.empty(
        num_rows, hidden_size // 16, dtype=torch.float8_e4m3fn, device=x.device
    )
    x_scale = torch.empty(num_rows, dtype=torch.float32, device=x.device)
    if num_rows == 0:
        return x_fp4, x_sf, x_scale
    config = nvfp4_row_quant_launch_config(num_rows, hidden_size)
    _nvfp4_row_quant_kernel[(num_rows, config["SPLIT_K"])](
        x,
        x_fp4,
        x_sf,
        x_scale,
        x.stride(0),
        x_fp4.stride(0),
        x_sf.stride(0),
        hidden_size,
        **config,
        num_warps=4,
    )
    return x_fp4, x_sf, x_scale


def nvfp4_row_quantize_ref(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch reference for :func:`nvfp4_row_quantize`, with IEEE division
    and reciprocals where the kernel uses ``div.full`` / ``rcp.approx``.
    Like ``cvt``, values that round to zero keep their sign (-0 is ``0x8``)."""

    # Tensor divisors: PyTorch turns scalar division into a reciprocal multiply.
    def div(a: torch.Tensor, b: torch.Tensor | float) -> torch.Tensor:
        return torch.div(a, torch.as_tensor(b, dtype=a.dtype, device=a.device))

    x = x.float()
    num_rows, hidden_size = x.shape
    row_amax = x.abs().amax(-1).clamp_min(1e-12)
    row_scale = div(row_amax, NVFP4_ROW_SCALE_MAX).bfloat16().float()
    encode = div(torch.full_like(row_amax, NVFP4_ROW_SCALE_MAX), row_amax)
    x_norm = (x * encode[:, None]).bfloat16().float()
    blocks = x_norm.view(num_rows, hidden_size // 16, 16)
    sixth = div(torch.ones((), device=x.device), 6.0)
    sf = (blocks.abs().amax(-1) * sixth).to(torch.float8_e4m3fn)
    inv = div(torch.ones_like(sf, dtype=torch.float32), sf.float())
    scaled = blocks * inv.masked_fill(sf.float() == 0, 0.0)[..., None]
    mag = scaled.abs()
    code = sum(
        (mag > t) if strict else (mag >= t)
        for t, strict in (
            (0.25, True),
            (0.75, False),
            (1.25, True),
            (1.75, False),
            (2.5, True),
            (3.5, False),
            (5.0, True),
        )
    ).to(torch.uint8)
    code |= torch.signbit(scaled).to(torch.uint8) << 3
    code = code.view(num_rows, hidden_size // 2, 2)
    return code[..., 0] | (code[..., 1] << 4), sf.view(num_rows, -1), row_scale
