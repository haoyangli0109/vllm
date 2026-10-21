# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton access to MiniMax M3's NVFP4 main K/V cache pages on ROCm.

Serving writes the pages with the fused FlyDSL insert
(``norm_rope_nvfp4_insert``, the same bytes as this Triton insert, which the
tests use as the reference) and reads them with the FlyDSL attends
(``nvfp4_prefill``, ``nvfp4_sparse_decode``), so the byte layout inside a page is
the ROCm backend's own, laid out for the FlyDSL prefill's loads.

A page (one sparse block of 128 tokens) is ``[K side | V side]``; a side is
every KV head's packed E2M1 data (8192 bytes per head), then every head's E4M3
block scales (1024 bytes per head, one per 16 elements, always of 16 consecutive
dims of one token). A data byte holds two elements adjacent along the
contraction axis of the attention GEMM that reads them (the low nibble the
first), so dequantized code pairs are MFMA operand pairs as they come out of
``v_cvt_scalef32_pk_*_fp4``: K pairs two consecutive dims of one token (Q K^T
sums over dims), V the same dim of two consecutive tokens (P V sums over
tokens). Inside a head:

- K data: token-major, ``[128 tokens][64 bytes]``.
- K scales: in the FlyDSL prefill's lane order, ``[4 token quarters w][4 dim
  chunks c][16 n][2 st][2 groups]`` for token ``32 w + 8 (n // 4) + 4 st +
  n % 4`` and group ``2 c + i``: each lane's two key rows and two groups are one
  dword.
- V data: ``[16 token octets][16 dim octets][8 dims][4 bytes]``: a dword is
  one dim of 8 consecutive tokens, and a (token octet, dim octet) tile is 32
  contiguous bytes, the V^T fragment one lane of the FlyDSL prefill (and of an
  fp8 decode with the same lane map) loads with two 16-byte loads.
- V scales: group-major, ``[8 groups][128 tokens]``: the scales of a lane's 8
  tokens are 8 contiguous bytes, and two consecutive tokens' scales are the
  pair that multiplies one dequantized V byte.

Eight tokens share every V data dword, so the insert writes V a token octet
at a time, with plain stores. When a launch covers a whole octet (prefill), each
of its 8 programs writes one 16-dim group of it: 16 dwords and the group's 8
scales. Otherwise (decode, a chunk edge) the first of the launch's tokens in the
octet writes all 128 dwords, keeping the octet's other tokens' nibbles from the
page. This needs a launch's tokens of one octet to be adjacent in the batch and
in slot order, as vLLM's slot mappings are: a request's tokens are contiguous
with consecutive slots, and a page is written by one request only.

An element is ``e2m1 * block scale * global scale``, where the global scale is
the layer's fp32 ``_k_scale`` / ``_v_scale``. ``nvfp4_kv_cache_token_major``
returns a cache's data and scales token-major (tests, debugging). The
``NVFP4_LAYOUT_LINEAR`` layout (token-major data, ``[token][group]`` scales) is
the NVFP4 index-K cache's (``nvfp4_index.py``).

The insert follows the CUDA encoder's earlier math: block scale
``e4m3(max|x| / 6 / global)``, codes ``e2m1_rn_satfinite(x / (scale * global))``.
The CUDA kernel takes 1/6 and the reciprocals with ``rcp.approx``; this one
rounds them, which can move a code at an exact rounding tie. It does not apply
the M3.1 KV4 spec's 1/512 floor on the block scale, which changes only groups
whose amax is below about 0.006 x the global scale.
"""

import torch

from vllm.triton_utils import tl, triton

NVFP4_HEAD_DIM = 128
NVFP4_GROUP_SIZE = 16
NVFP4_DATA_BYTES = NVFP4_HEAD_DIM // 2  # per (head, token)
NVFP4_SCALE_BYTES = NVFP4_HEAD_DIM // NVFP4_GROUP_SIZE  # per (head, token)
NVFP4_PAGE_TOKENS = 128  # the main-cache layouts address 128-token pages

# Byte layouts inside one head's data / scale region (see the module docstring).
# tl.constexpr so the Triton kernels can read them as globals.
NVFP4_LAYOUT_LINEAR = tl.constexpr(0)  # token-major data, [token][group] scales
NVFP4_LAYOUT_MAIN_K = tl.constexpr(1)
NVFP4_LAYOUT_MAIN_V = tl.constexpr(2)


@triton.jit
def _e2m1_to_f32(code):
    """Value of a 4-bit E2M1 code (sign, 2 exponent bits, 1 mantissa bit)."""
    code = code.to(tl.uint32)
    exp = (code >> 1) & 3
    man = code & 1
    # exp 0 is the subnormal 0 / 0.5; otherwise 2**(exp - 1) * (1 + man / 2).
    mag = tl.where(exp == 0, man * (126 << 23), ((exp + 126) << 23) | (man << 22))
    return (mag | ((code >> 3) << 31)).to(tl.float32, bitcast=True)


@triton.jit
def _f32_to_e2m1(x):
    """Round-to-nearest-even, saturating E2M1 code of ``x`` (the CUDA
    encoder's ``e2m1Code``). Below 1 the grid is the 0.5-step subnormals; from
    1 up it is fp32 rounded to one mantissa bit, whose exponent and mantissa
    bits are the code offset by 252 once the exponent is rebiased."""
    ax = tl.abs(x)
    twice = 2.0 * ax
    small = (twice > 0.5).to(tl.uint32) + (twice >= 1.5).to(tl.uint32)
    bits = ax.to(tl.uint32, bitcast=True)
    rounded = bits + 0x1FFFFF + ((bits >> 22) & 1)
    big = tl.minimum((rounded >> 22) - 252, 7)
    mag = tl.where(ax < 1.0, small, big)
    return mag | ((x.to(tl.uint32, bitcast=True) >> 28) & 8)


@triton.jit
def _nvfp4_data_offset(token, dim, LAYOUT: tl.constexpr):
    """Byte of one head's data region holding ``(token, dim)``."""
    if LAYOUT == 2:
        return ((token // 8 * 16 + dim // 8) * 8 + dim % 8) * 4 + (token % 8) // 2
    else:
        return token * 64 + dim // 2


@triton.jit
def _nvfp4_nibble_shift(token, dim, LAYOUT: tl.constexpr):
    """Bit offset of ``(token, dim)``'s code in its data byte: V bytes pair two
    tokens, the other layouts two dims."""
    if LAYOUT == 2:
        return (token % 2) * 4
    else:
        return (dim % 2) * 4


@triton.jit
def _nvfp4_scale_offset(token, group, LAYOUT: tl.constexpr):
    """Byte of one head's scale region holding the scale of ``(token, group)``."""
    if LAYOUT == 1:
        r = token % 32
        n = 4 * (r // 8) + r % 4
        return ((((token // 32) * 4 + group // 2) * 16 + n) * 2 + (r % 8) // 4) * 2 + (
            group % 2
        )
    elif LAYOUT == 2:
        return group * 128 + token
    else:
        return token * 8 + group


@triton.jit
def nvfp4_kv_load(
    data_ptr,  # this head's packed data, 64 bytes per token
    scale_ptr,  # this head's block scales, 8 bytes per token
    token,
    dim,
    mask,
    LAYOUT: tl.constexpr,  # NVFP4_LAYOUT_*
):
    """Dequantize elements ``(token, dim)`` of one head of one side to fp32,
    without the global scale. ``token`` and ``dim`` broadcast against each
    other, so ``(n[None, :], d[:, None])`` loads K^T and ``(n[:, None],
    d[None, :])`` loads V. Masked elements are 0."""
    byte = tl.load(
        data_ptr + _nvfp4_data_offset(token, dim, LAYOUT), mask=mask, other=0
    )
    shift = _nvfp4_nibble_shift(token, dim, LAYOUT).to(tl.uint32)
    code = (byte.to(tl.uint32) >> shift) & 0xF
    scale = tl.load(
        scale_ptr + _nvfp4_scale_offset(token, dim // 16, LAYOUT), mask=mask, other=0
    )
    scale = scale.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    return _e2m1_to_f32(code) * scale


@triton.jit
def nvfp4_v_tile_load(
    data_ptr,  # this head's V data region (dim-octet-major)
    scale_ptr,  # this head's V scale region (group-major)
    token,  # [N] token positions in the page
    mask,  # [N]
):
    """``[N, 128]`` fp32 V rows (without the global scale) of a main-cache page,
    read as the page stores them: one dword per (dim, 8 tokens)."""
    dim = tl.arange(0, 128)
    words = tl.load(
        data_ptr.to(tl.pointer_type(tl.uint32))
        + (token[:, None] // 8 * 16 + dim[None, :] // 8) * 8
        + dim[None, :] % 8,
        mask=mask[:, None],
        other=0,
    )  # [N, 128]: the dword holding (token, dim)
    codes = (words >> ((token[:, None] % 8) * 4).to(tl.uint32)) & 0xF
    scale = tl.load(
        scale_ptr + (dim[None, :] // 16) * 128 + token[:, None],
        mask=mask[:, None],
        other=0,
    )  # [N, 128]
    scale = scale.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    return _e2m1_to_f32(codes) * scale


@triton.jit
def _quantize_nvfp4(groups, global_scale):
    """E2M1 codes (uint32, same shape) and E4M3 block scales (one per row of the
    last axis) of fp32 16-element groups ``[..., 16]``."""
    group_max = tl.max(tl.abs(groups), axis=-1)
    scale = tl.minimum(group_max * 0.16666667163372040 * (1.0 / global_scale), 448.0)
    scale_e4m3 = scale.to(tl.float8e4nv)
    scale_value = scale_e4m3.to(tl.float32)
    quant = tl.where(scale_value != 0.0, 1.0 / (scale_value * global_scale), 0.0)
    return _f32_to_e2m1(groups * tl.expand_dims(quant, -1)), scale_e4m3


@triton.jit
def _store_nvfp4_row(
    x,  # [128] fp32, one head of one token
    data_ptr,  # this head's packed data, 64 bytes per token
    scale_ptr,  # this head's block scales, 8 bytes per token
    token,
    global_scale,
    LAYOUT: tl.constexpr,  # NVFP4_LAYOUT_* with dim-paired bytes (not MAIN_V)
    mask=None,  # scalar: store at all
):
    codes, scale_e4m3 = _quantize_nvfp4(tl.reshape(x, (8, 16)), global_scale)
    even, odd = tl.split(tl.reshape(codes, (64, 2)))
    packed = (even | (odd << 4)).to(tl.uint8)
    dim = 2 * tl.arange(0, 64)
    tl.store(data_ptr + _nvfp4_data_offset(token, dim, LAYOUT), packed, mask=mask)
    group = tl.arange(0, 8)
    tl.store(
        scale_ptr + _nvfp4_scale_offset(token, group, LAYOUT),
        scale_e4m3.to(tl.uint8, bitcast=True),
        mask=mask,
    )


@triton.jit
def _bit_or(a, b):
    return a | b


@triton.jit
def _nvfp4_v_words(x, shift, global_scale, ROWS: tl.constexpr, DIMS: tl.constexpr):
    """``x`` ``[ROWS, DIMS]`` fp32 V rows (zero rows give code 0) -> the
    ``[DIMS]`` V data dwords (row r's codes at bit ``shift[r]``) and the
    ``[ROWS, DIMS // 16]`` E4M3 scales."""
    codes, scale_e4m3 = _quantize_nvfp4(
        tl.reshape(x, (ROWS, DIMS // 16, 16)), global_scale
    )
    codes = tl.reshape(codes, (ROWS, DIMS)) << shift[:, None]
    return tl.reduce(codes, 0, _bit_or), scale_e4m3


@triton.jit
def _store_nvfp4_v_run(
    v_rows,  # V row of the run's first token (tok)
    stride_v_t,
    n_run,  # the run: tokens tok .. tok + n_run - 1, octet positions p0 ..
    p0,
    v_data,  # the octet's 128 dwords
    v_sf,  # the scale of octet position 0, group 0
    global_scale,
    ROWS: tl.constexpr,  # >= n_run
):
    """Write the run's V into its token octet, keeping the octet's other
    tokens' nibbles (read back from the page)."""
    r = tl.arange(0, ROWS)
    dim = tl.arange(0, 128)
    in_run = r < n_run
    old = tl.load(v_data + dim).to(tl.uint32, bitcast=True)
    x = tl.load(
        v_rows + r[:, None] * stride_v_t + dim[None, :],
        mask=in_run[:, None],
        other=0.0,
    ).to(tl.float32)
    shift = tl.where(in_run, (p0 + r) * 4, 0).to(tl.uint32)
    words, sf = _nvfp4_v_words(x, shift, global_scale, ROWS=ROWS, DIMS=128)
    nibble = tl.full((ROWS,), 0xF, tl.uint32) << shift
    written = tl.reduce(tl.where(in_run, nibble, 0 * nibble), 0, _bit_or)
    tl.store(v_data + dim, (words | (old & ~written)).to(tl.int32, bitcast=True))
    group = tl.arange(0, 8)
    tl.store(
        v_sf + (p0 + r)[:, None] + group[None, :] * 128,
        sf.to(tl.uint8, bitcast=True),
        mask=in_run[:, None],
    )


@triton.jit
def _insert_nvfp4_index_row(
    idx_k_ptr, idx_cache_ptr, idx_slot_ptr, stride_idx_k, stride_idx_blk, tok
):
    """One token's index key into its NVFP4 index page (no global scale), as
    nvfp4_index.minimax_m3_insert_nvfp4_index_cache."""
    slot = tl.load(idx_slot_ptr + tok).to(tl.int64)
    if slot >= 0:
        page = idx_cache_ptr + (slot // 128) * stride_idx_blk
        x = tl.load(idx_k_ptr + tok * stride_idx_k + tl.arange(0, 128)).to(tl.float32)
        _store_nvfp4_row(
            x, page, page + 128 * 64, slot % 128, 1.0, LAYOUT=NVFP4_LAYOUT_LINEAR
        )


@triton.jit
def _insert_nvfp4_kv_kernel(
    k_ptr,  # [num_tokens, num_kv_heads, 128]
    v_ptr,  # [num_tokens, num_kv_heads, 128]
    cache_ptr,  # [num_blocks, 2 * num_kv_heads, BLOCK_SIZE, 72] uint8
    slot_mapping_ptr,  # [num_tokens]
    k_scale_ptr,  # fp32 scalar
    v_scale_ptr,  # fp32 scalar
    stride_k_t,
    stride_k_h,
    stride_v_t,
    stride_v_h,
    stride_c_blk,
    side_bytes,  # num_kv_heads * BLOCK_SIZE * 72
    num_tokens,
    idx_k_ptr,  # HAS_INDEX: [num_tokens, 128] index keys
    idx_cache_ptr,  # HAS_INDEX: [num_blocks, 128, 72] uint8 NVFP4 index pages
    idx_slot_ptr,  # HAS_INDEX: [num_tokens]
    stride_idx_k,
    stride_idx_blk,
    NUM_KV_HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_INDEX: tl.constexpr,
):
    tok = tl.program_id(0)
    head = tl.program_id(1)
    if HAS_INDEX:  # noqa: SIM102 (a constexpr branch around a runtime one)
        # the last program column inserts the index key (one launch for both)
        if head == NUM_KV_HEADS:
            _insert_nvfp4_index_row(
                idx_k_ptr,
                idx_cache_ptr,
                idx_slot_ptr,
                stride_idx_k,
                stride_idx_blk,
                tok,
            )
            return
    # Straight-line until the V branches, so every load not addressed by the
    # slot is in flight with it.
    slot = tl.load(slot_mapping_ptr + tok).to(tl.int64)
    prev = tl.load(slot_mapping_ptr + tok - 1, mask=tok > 0, other=-1).to(tl.int64)
    near = tok - 7 + tl.arange(0, 16)  # every token that can share tok's octet
    near_slot = tl.load(
        slot_mapping_ptr + near, mask=(near >= 0) & (near < num_tokens), other=-1
    ).to(tl.int64)
    offs = tl.arange(0, 128)
    k = tl.load(k_ptr + tok * stride_k_t + head * stride_k_h + offs).to(tl.float32)
    k_scale = tl.load(k_scale_ptr)
    v_scale = tl.load(v_scale_ptr)
    valid = slot >= 0  # else a padded / unscheduled token
    blk = slot // BLOCK_SIZE
    pos = slot % BLOCK_SIZE
    scale_region = NUM_KV_HEADS * BLOCK_SIZE * 64
    k_side = cache_ptr + blk * stride_c_blk
    v_side = k_side + side_bytes
    _store_nvfp4_row(
        k,
        k_side + head * BLOCK_SIZE * 64,
        k_side + scale_region + head * BLOCK_SIZE * 8,
        pos,
        k_scale,
        LAYOUT=NVFP4_LAYOUT_MAIN_K,
        mask=valid,
    )

    # V, one token octet at a time. A launch's tokens of one octet are adjacent
    # in the batch and in slot order (a run).
    j = tl.arange(0, 8)
    match = near_slot[:, None] == slot - slot % 8 + j[None, :]  # [16, 8]
    n_present = tl.sum(match.to(tl.int32))
    v_data = (v_side + head * BLOCK_SIZE * 64).to(tl.pointer_type(tl.int32))
    v_data += pos // 8 * 128  # the octet's 128 dwords, dim d at dword d
    v_sf = v_side + scale_region + head * BLOCK_SIZE * 8 + pos - pos % 8
    v_rows = v_ptr + head * stride_v_h
    first = (prev < 0) | (prev // 8 != slot // 8)
    if valid & (n_present == 8):
        # The whole octet is in this launch: its 8 programs write a 16-dim
        # group each (dwords 16 g .., the group's 8 scales), plain stores.
        g = pos % 8
        d = g * 16 + tl.arange(0, 16)
        x = tl.load(v_rows + (tok - g + j)[:, None] * stride_v_t + d[None, :])
        shift = (j * 4).to(tl.uint32)
        words, sf = _nvfp4_v_words(x.to(tl.float32), shift, v_scale, ROWS=8, DIMS=16)
        tl.store(v_data + d, words.to(tl.int32, bitcast=True))
        tl.store(v_sf + j + g * 128, tl.reshape(sf, (8,)).to(tl.uint8, bitcast=True))
    elif valid & first:
        # Part of the octet (decode, a chunk edge): the run's first token
        # writes all 128 dwords; the tile fits the run.
        run_rows = v_rows + tok * stride_v_t
        if n_present == 1:
            _store_nvfp4_v_run(
                run_rows, stride_v_t, n_present, pos % 8, v_data, v_sf, v_scale, 1
            )
        elif n_present <= 4:
            _store_nvfp4_v_run(
                run_rows, stride_v_t, n_present, pos % 8, v_data, v_sf, v_scale, 4
            )
        else:
            _store_nvfp4_v_run(
                run_rows, stride_v_t, n_present, pos % 8, v_data, v_sf, v_scale, 8
            )


def nvfp4_kv_cache_geometry(kv_cache: torch.Tensor) -> tuple[int, int]:
    """``(num_kv_heads, side_bytes)`` of an NVFP4 ``[B, 2*Hkv, N, 72]`` cache.

    The layer's cache is a view whose block stride may exceed one page (other
    layers can share the block), but each page's sides must be dense.
    """
    _, slots, block_size, width = kv_cache.shape
    if kv_cache.dtype != torch.uint8 or slots % 2 or width != 72:
        raise ValueError(
            "MiniMax-M3 NVFP4 KV cache must be [blocks, 2 * kv_heads, block, 72] "
            f"uint8, got {tuple(kv_cache.shape)} {kv_cache.dtype}"
        )
    if (
        kv_cache.stride(3) != 1
        or kv_cache.stride(2) != 72
        or kv_cache.stride(1) != (block_size * 72)
    ):
        raise ValueError(
            "MiniMax-M3 NVFP4 KV cache pages must be head-major and dense, got "
            f"strides {kv_cache.stride()}"
        )
    num_kv_heads = slots // 2
    return num_kv_heads, num_kv_heads * kv_cache.stride(1)


@torch.no_grad()
def minimax_m3_insert_nvfp4_kv(
    k: torch.Tensor,  # [num_tokens, num_kv_heads, 128]
    v: torch.Tensor,  # [num_tokens, num_kv_heads, 128]
    kv_cache: torch.Tensor,  # [num_blocks, 2 * num_kv_heads, block_size, 72] uint8
    slot_mapping: torch.Tensor,  # [num_tokens], -1 skips the token
    k_scale: torch.Tensor,  # fp32 scalar dequantization (global) scale
    # TODO(zz): k_scale & v_scale can be removed right? since they will be 1.0
    v_scale: torch.Tensor,
    index: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """Quantize post-RoPE K and V into their NVFP4 cache pages; with ``index`` =
    ``(index_k [num_tokens, 128], index_cache [blocks, 128, 72] uint8,
    index_slot_mapping)`` the index keys go into their NVFP4 pages in the same
    launch."""
    num_tokens, num_kv_heads, head_dim = k.shape
    if num_tokens == 0 or kv_cache.numel() == 0:
        return
    cache_heads, side_bytes = nvfp4_kv_cache_geometry(kv_cache)
    if kv_cache.shape[2] != NVFP4_PAGE_TOKENS:
        raise ValueError(
            f"MiniMax-M3 NVFP4 KV pages hold 128 tokens, got {kv_cache.shape}"
        )
    if head_dim != NVFP4_HEAD_DIM or cache_heads != num_kv_heads:
        raise ValueError(
            f"K/V [{num_tokens}, {num_kv_heads}, {head_dim}] do not match the "
            f"NVFP4 cache {tuple(kv_cache.shape)}"
        )
    if k.stride(2) != 1 or v.stride(2) != 1:
        raise ValueError("MiniMax-M3 NVFP4 insert needs a contiguous head dim")
    if index is not None:
        index_k, index_cache, index_slots = index
        assert index_cache.dtype == torch.uint8 and index_cache.shape[1:] == (128, 72)
        assert index_k.shape == (num_tokens, NVFP4_HEAD_DIM) and index_k.stride(1) == 1
        idx_args = (index_k, index_cache, index_slots, index_k.stride(0))
        idx_args += (index_cache.stride(0),)
    else:
        idx_args = (k, kv_cache, slot_mapping, 0, 0)
    _insert_nvfp4_kv_kernel[(num_tokens, num_kv_heads + (index is not None))](
        k,
        v,
        kv_cache,
        slot_mapping,
        k_scale,
        v_scale,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        kv_cache.stride(0),
        side_bytes,
        num_tokens,
        *idx_args,
        NUM_KV_HEADS=num_kv_heads,
        BLOCK_SIZE=kv_cache.shape[2],
        HAS_INDEX=index is not None,
        num_warps=1,
    )


def _layout_offsets(
    layout: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``[128, 128]`` data-byte offsets and nibble shifts of each (token, dim),
    and ``[128, 8]`` scale-byte offsets of each (token, group), inside one head's
    regions (host mirror of ``_nvfp4_data_offset`` / ``_nvfp4_nibble_shift`` /
    ``_nvfp4_scale_offset``)."""
    t = torch.arange(NVFP4_PAGE_TOKENS, device=device)[:, None]
    d = torch.arange(NVFP4_HEAD_DIM, device=device)[None, :]
    g = torch.arange(NVFP4_SCALE_BYTES, device=device)[None, :]
    if layout == NVFP4_LAYOUT_MAIN_V:
        data = ((t // 8 * 16 + d // 8) * 8 + d % 8) * 4 + (t % 8) // 2
        shift = (t % 2) * 4 + 0 * d
        scale = g * 128 + t
    else:
        data = t * 64 + d // 2
        shift = (d % 2) * 4 + 0 * t
        if layout == NVFP4_LAYOUT_MAIN_K:
            r = t % 32
            n = 4 * (r // 8) + r % 4
            scale = ((((t // 32) * 4 + g // 2) * 16 + n) * 2 + (r % 8) // 4) * 2 + g % 2
        else:
            scale = t * 8 + g
    return data, shift, scale


def nvfp4_kv_cache_token_major(
    kv_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(k_data, k_scale, v_data, v_scale)`` of an NVFP4 main cache,
    token-major: data ``[blocks, Hkv, 128, 64]`` uint8 (the low nibble the even
    dim), scales ``[blocks, Hkv, 128, 8]`` uint8 (E4M3, one per 16 dims)."""
    num_kv_heads, side_bytes = nvfp4_kv_cache_geometry(kv_cache)
    blocks = kv_cache.shape[0]
    pages = kv_cache.reshape(blocks, -1)[:, : 2 * side_bytes]
    out = []
    for side, layout in ((0, NVFP4_LAYOUT_MAIN_K), (1, NVFP4_LAYOUT_MAIN_V)):
        region = pages[:, side * side_bytes : (side + 1) * side_bytes]
        data_region = region[:, : num_kv_heads * 8192].reshape(
            blocks, num_kv_heads, 8192
        )
        scale_region = region[:, num_kv_heads * 8192 :].reshape(
            blocks, num_kv_heads, 1024
        )
        data_off, shift, scale_off = _layout_offsets(layout, kv_cache.device)
        data = data_region[:, :, data_off.reshape(-1)]
        data = data.reshape(blocks, num_kv_heads, 128, 128)
        codes = (data >> shift.to(torch.uint8)) & 0xF
        out.append(codes[..., 0::2] | (codes[..., 1::2] << 4))
        out.append(
            scale_region[:, :, scale_off.reshape(-1)].reshape(
                blocks, num_kv_heads, 128, 8
            )
        )
    return out[0], out[1], out[2], out[3]
