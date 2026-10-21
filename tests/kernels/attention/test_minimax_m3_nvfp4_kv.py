# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3.1 NVFP4 K/V and index-K pages on ROCm: the Triton insert
(``amd/ops/nvfp4_kv.py``, the reference of the fused FlyDSL insert) against
page bytes computed independently, and the token-major view the attention
tests dequantize through. Also holds the NVFP4 helpers of the other NVFP4
tests."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm NVFP4 K/V cache", allow_module_level=True)

from vllm.platforms.rocm import on_gfx950  # noqa: E402

if not on_gfx950():
    pytest.skip("gfx950 kernels", allow_module_level=True)

from vllm.models.minimax_m3.amd.ops.nvfp4_kv import (  # noqa: E402
    minimax_m3_insert_nvfp4_kv,
    nvfp4_kv_cache_token_major,
)

BLOCK = 128
D = 128
E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _reference_nvfp4(x: torch.Tensor, global_scale: float):
    """E2M1 codes ``[T, 128]`` and E4M3 block-scale bytes ``[T, 8]`` of fp32
    rows: the CUDA encoder's scale math and a nearest-grid E2M1 rounding (ties
    to the even code), written independently of the kernels' bit arithmetic."""
    groups = x.view(-1, 8, 16)
    one_sixth = torch.tensor(1 / 6, dtype=torch.float32)
    scale = (groups.abs().amax(-1) * one_sixth * (1.0 / global_scale)).clamp(max=448)
    scale_e4m3 = scale.to(torch.float8_e4m3fn)
    scale_value = scale_e4m3.float()
    quant = torch.where(scale_value != 0, 1.0 / (scale_value * global_scale), 0.0)
    y = groups * quant[..., None]
    grid = torch.tensor(E2M1_GRID, device=x.device)
    dist = (y.abs()[..., None] - grid).abs()
    is_min = dist == dist.amin(-1, keepdim=True)
    ids = torch.arange(8, device=x.device)
    # Neighbouring codes alternate parity, so every tie has an even candidate.
    even_min = torch.where(is_min & (ids % 2 == 0), ids, 8).amin(-1)
    mag = torch.where(even_min < 8, even_min, dist.argmin(-1))
    codes = mag | (torch.signbit(y).long() << 3)
    return codes.view(-1, D), scale_e4m3.view(torch.uint8).view(-1, 8)


def _dequant(data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Token-major NVFP4 rows -> fp32 e2m1 x e4m3 block scale (no global
    scale): data ``[..., 64]`` uint8 (low nibble the even dim), scales
    ``[..., 8]`` uint8 -> ``[..., 128]``."""
    codes = torch.stack((data & 15, data >> 4), -1).flatten(-2).long()
    grid = torch.tensor(E2M1_GRID, device=data.device)
    mag = grid[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    s = scales.view(torch.float8_e4m3fn).float()
    return (mag.unflatten(-1, (8, 16)) * s[..., None]).flatten(-2)


def _reference_insert(cache, rows, slots, side, global_scale):
    """Place reference NVFP4 rows ``[T, Hkv, 128]`` of one side (0 = K, 1 = V)
    at their bytes of the (CPU) main cache, page layout as ``nvfp4_kv.py``
    documents it; slots < 0 are skipped."""
    num_tokens, heads, _ = rows.shape
    codes, scales = _reference_nvfp4(rows.reshape(-1, D), global_scale)
    keep = (slots >= 0).cpu()
    codes = codes.view(num_tokens, heads, D).cpu()[keep].to(torch.uint8)
    scales = scales.view(num_tokens, heads, 8).cpu()[keep]
    blk, pos = slots.cpu()[keep] // BLOCK, slots.cpu()[keep] % BLOCK
    h, d, g = torch.arange(heads), torch.arange(D), torch.arange(8)
    side0 = blk[:, None] * cache.stride(0) + side * heads * BLOCK * 72  # [n, 1]
    data_base = (side0 + h * BLOCK * 64)[..., None]  # [n, H, 1]
    scale_base = (side0 + heads * BLOCK * 64 + h * BLOCK * 8)[..., None]
    flat = cache.view(-1)
    p = pos[:, None, None]
    if side == 0:  # K data token-major; scales [w][dim chunk][n][st][2 groups]
        flat[data_base + p * 64 + torch.arange(64)] = codes[..., 0::2] | (
            codes[..., 1::2] << 4
        )
        r = p % 32
        n, st = 4 * (r // 8) + r % 4, (r % 8) // 4
        off = ((((p // 32) * 4 + g // 2) * 16 + n) * 2 + st) * 2 + g % 2
    else:  # V data [token octet][dim octet][8 dims][4 bytes]; scales [group][token]
        for parity in (0, 1):  # two tokens share a byte: one nibble per pass
            m = pos % 2 == parity
            pm = p[m]
            byte = (
                data_base[m] + ((pm // 8 * 16 + d // 8) * 8 + d % 8) * 4 + (pm % 8) // 2
            )
            keep_bits = 0x0F << (4 * (1 - parity))
            flat[byte] = (flat[byte] & keep_bits) | (codes[m] << (4 * parity))
        off = g * BLOCK + p
    flat[scale_base + off] = scales


def _reference_index_insert(cache, keys, slots):
    """Reference NVFP4 index keys at their bytes of the (CPU) index pages:
    ``[128 x 64 B data][128 x 8 B scales]``, no global scale."""
    codes, scales = _reference_nvfp4(keys, 1.0)
    keep = (slots >= 0).cpu()
    codes, scales = codes.cpu()[keep].to(torch.uint8), scales.cpu()[keep]
    blk, pos = slots.cpu()[keep] // BLOCK, slots.cpu()[keep] % BLOCK
    base = (blk * cache.stride(0))[:, None]
    flat = cache.view(-1)
    flat[base + pos[:, None] * 64 + torch.arange(64)] = codes[:, 0::2] | (
        codes[:, 1::2] << 4
    )
    flat[base + BLOCK * 64 + pos[:, None] * 8 + torch.arange(8)] = scales


def _request_slots(pages, starts, lens):
    """The slot mapping vLLM builds: request r's ``lens[r]`` tokens at positions
    ``starts[r]..`` of its own pages, requests back to back."""
    slots, page = [], 0
    for start, n in zip(starts, lens):
        pos = torch.arange(start, start + n, device="cuda")
        own = pages[page : page + (start + n + BLOCK - 1) // BLOCK]
        slots.append(own[pos // BLOCK] * BLOCK + pos % BLOCK)
        page += own.numel()
    return torch.cat(slots)


def _scales(k: float, v: float):
    return tuple(torch.tensor(x, dtype=torch.float32, device="cuda") for x in (k, v))


@pytest.mark.parametrize(
    ("starts", "lens"),
    [
        ([5], [1]),  # one decode token
        ([0, 13], [7, 4]),  # a short prefill, a speculative step mid-octet
        ([0, 3, 250, 128], [131, 61, 9, 99]),  # unaligned chunks across pages
    ],
)
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("global_scales", [(1.0, 1.0), (0.5, 2.0)])
def test_nvfp4_insert_matches_reference_bytes(
    starts, lens, num_kv_heads, global_scales
):
    """K/V and index-K bytes, byte for byte; bytes the launch does not write
    (padding slots, the rest of a partly written V octet, other pages) keep the
    stale contents; the token-major view reads the reference values back."""
    torch.manual_seed(0)
    num_pages = sum((a + n + BLOCK - 1) // BLOCK for a, n in zip(starts, lens)) + 1
    pages = torch.randperm(num_pages, device="cuda")
    pad = torch.full((3,), -1, device="cuda")  # CUDA-graph padding
    slots = torch.cat([_request_slots(pages, starts, lens), pad])
    index_slots = torch.cat([_request_slots(pages.flip(0), starts, lens), pad])
    n = slots.numel()
    k = torch.randn(n, num_kv_heads, D, device="cuda").bfloat16()
    v = (torch.randn(n, num_kv_heads, D, device="cuda") * 3).bfloat16()
    index_k = torch.randn(n, D, device="cuda").bfloat16()
    k[0, 0, :16] = 0  # an all-zero group has a zero block scale
    v[-4, -1, 16:32] = 1e4  # saturates the block scale at E4M3's 448
    k_scale, v_scale = _scales(*global_scales)
    shape = (num_pages, 2 * num_kv_heads, BLOCK, 72)
    stale = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
    stale_index = torch.randint(0, 256, (num_pages, BLOCK, 72), device="cuda").to(
        torch.uint8
    )
    kv_cache, index_cache = stale.clone(), stale_index.clone()
    minimax_m3_insert_nvfp4_kv(
        k,
        v,
        kv_cache,
        slots,
        k_scale,
        v_scale,
        index=(index_k, index_cache, index_slots),
    )
    expected, expected_index = stale.cpu(), stale_index.cpu()
    _reference_insert(expected, k.float(), slots, 0, global_scales[0])
    _reference_insert(expected, v.float(), slots, 1, global_scales[1])
    _reference_index_insert(expected_index, index_k.float(), index_slots)
    assert torch.equal(kv_cache.cpu(), expected), (
        f"{int((kv_cache.cpu() != expected).sum())} NVFP4 K/V bytes differ"
    )
    assert torch.equal(index_cache.cpu(), expected_index)

    k_data, k_sf, v_data, v_sf = nvfp4_kv_cache_token_major(kv_cache)
    valid = slots >= 0
    blk, pos = slots[valid] // BLOCK, slots[valid] % BLOCK
    sides = ((k_data, k_sf, k, global_scales[0]), (v_data, v_sf, v, global_scales[1]))
    for data, sf, rows, gs in sides:
        got = _dequant(data[blk, :, pos], sf[blk, :, pos])
        codes, scales = _reference_nvfp4(rows[valid].float().view(-1, D), gs)
        packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
        assert torch.equal(got, _dequant(packed, scales).view(got.shape))
