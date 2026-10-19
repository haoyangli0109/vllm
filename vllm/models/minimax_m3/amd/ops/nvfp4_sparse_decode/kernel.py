# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 sparse decode attention (MiniMax-M3.1, gfx950, FlyDSL).

The decode of the NVFP4 main K/V cache (``../nvfp4_kv.py``) with one query row per
workgroup, over the blocks the row's top-k selected. Each K / V element is the value
an FP8-page conversion would write and AITER's FP8 sparse PA
decode reads, e4m3_rn_satfinite(e2m1 x block scale), widened to bf16; bf16 Q, fp32
scores and online softmax, P rounded to bf16, fp32 accumulation.
The optional FP16 variant instead uses exact E2M1 times E4M3 K/V and FP16
Q/P MFMAs, retaining FP32 accumulation and BF16 output.

Row r is row r % ql of request r // ql at position seq_len - ql + r % ql. A
selected block counts when it is not past the row's own block; the own block is
masked causally. One workgroup = one row x the 16 query heads of one KV head (the
MFMA N dimension) x one split of the row's blocks; wave w owns keys [32 w, 32 w + 32)
of every block and loads its NVFP4 words straight into the MFMA operand registers,
double-buffered over pairs of blocks, in the NVFP4 prefill kernels' lane map:

  S^T = K Q^T      v_mfma_f32_16x16x32_bf16, A = K (16 keys x 32 dims), B = Q^T
  online softmax   fp32, over the wave's keys
  O^T += V^T P^T   A = V^T (16 dims x 32 keys), B = P^T straight from S^T

The waves' (max, sum, O) are combined through LDS. With ``splits`` > 1 a row's
blocks are split over workgroups (contiguous ranks); each leaves its unnormalized
O, max and sum in a scratch row, and the last split of a row to arrive (a ticket)
combines them in split order, so the result does not depend on arrival order.

No inline assembly: the dequant uses the gfx950 conversions through their
intrinsics (v_perm_b32, v_cvt_scalef32_pk_*), which LLVM schedules with the rest.
The dequant is most of the work (about 260 of the ~480 instructions a wave issues
per block), so the kernel is VALU-bound: a second wave per SIMD does not help.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import rocdl as _rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

BLK = 128  # tokens per block (one page)
D = 128
HG = 16  # query heads per KV head = the MFMA N dimension
NW = 4  # waves per workgroup; wave w owns keys [KPW w, KPW w + KPW) of every block
KPW = BLK // NW
DATA_B = BLK * D // 2  # packed e2m1 bytes per head per side
SCALE_B = BLK * D // 16  # e4m3 block-scale bytes per head per side
NDT = D // 16  # d-tiles of O^T
NKS = D // 32  # k-steps of S^T
RAW = 2 * 4 + 1 + 8 + 2  # NVFP4 dwords a lane loads per block
PART_F = 36  # split partials: fp32 per (split, row, lane): O (NDT x 4), max, sum, pad
# The split partials must reach the combining workgroup, and the XCDs' L2s are not
# coherent. With ``xcds`` (workgroup w runs on XCD w % xcds) the splits of a row run
# on one XCD and plain stores / loads meet in its L2; otherwise they are stored and
# loaded at agent scope (sc1). Never an agent-scope release fence: it writes back the
# whole L2.
AGENT_SCOPE = 0x10
# kernel arguments preloaded into SGPRs (the first 56 bytes) instead of s_load'ed
PRELOAD_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 16,
    }
}
LOG2E = 1.4426950409
TINY_F32 = 1.1754943508222875e-38  # torch.finfo(float32).tiny, the quant's amax floor
SIGN4 = 0x80808080 - (1 << 32)  # the four bytes' sign bits, as an i32
MAG4 = 0x07070707  # the four low-nibble magnitudes


def _global_i32_ptr(addr_i64):
    """Typed global i32 pointer at a raw device address (``p[i]`` loads)."""
    ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=4
    )
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def _bits(f):
    return fx.Float32(f).bitcast(fx.Int32)


def _as_f32(i):
    return fx.Int32(i).bitcast(fx.Float32)


def _permlane_swap(op, d_a, d_b):
    """v_permlane{16,32}_swap (raw rocdl op; fx has no wrapper)."""
    pair_ty = _ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = op(pair_ty, fx.as_ir_value(d_a), fx.as_ir_value(d_b), False, False)
    return fx.Int32(_llvm.extractvalue(T.i32, res, [0])), fx.Int32(
        _llvm.extractvalue(T.i32, res, [1])
    )


def _nnan():
    return _ir.Attribute.parse("#arith.fastmath<nnan>")


def _maxf_nn(a, b):
    """v_max_f32 with nnan (no canonicalizing v_max x, x, x on the inputs)."""
    return fx.Float32(
        _arith.MaxNumFOp(fx.as_ir_value(a), fx.as_ir_value(b), fastmath=_nnan()).result
    )


def _xlane4(x, op):
    """``op`` over lanes L, L^16, L^32, L^48, broadcast to all four (two swaps)."""
    a, b = _permlane_swap(_rocdl.permlane32_swap, _bits(x), _bits(x))
    m = op(_as_f32(a), _as_f32(b))
    a, b = _permlane_swap(_rocdl.permlane16_swap, _bits(m), _bits(m))
    return op(_as_f32(a), _as_f32(b))


def _pk_bf16(a, b):
    """Two fp32 -> one dword of two bf16 (RNE, v_cvt_pk_bf16_f32), ``a`` low."""
    return fx.Int32(
        fx.rocdl.cvt_pk_bf16_f32(
            fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b))
        )
    )


def _pk_fp8(a, b, old, hi):
    """Two fp32 -> two e4m3 bytes (RNE, v_cvt_pk_fp8_f32) in the low (``hi``: high)
    word of ``old``, ``a`` the lower byte."""
    return fx.Int32(
        _rocdl_ops.cvt_pk_fp8_f32(
            T.i32,
            fx.as_ir_value(fx.Float32(a)),
            fx.as_ir_value(fx.Float32(b)),
            fx.as_ir_value(fx.Int32(old)),
            hi,
        )
    )


def _mxfp8_group(words):
    """A lane's 32 output values as 16 packed-bf16 dwords (dims in order) -> their
    MXFP8 quant as AITER's MXFP8 GEMM activation quantizer computes it on the
    bf16 values: 8 dwords of e4m3 codes and the e8m0 scale (fp32,
    0..254)."""
    xs = []
    for wd in words:
        xs.append(_as_f32(wd << 16))
        xs.append(_as_f32(wd & fx.Int32(-65536)))
    amax = fx.Float32(TINY_F32)
    for x in xs:
        amax = fx.arith.maximumf(amax, fx.math.absf(x))
    sb = fx.math.ceil(fx.math.log2(amax / fx.Float32(448.0))) + fx.Float32(127.0)
    sb = fx.arith.minimumf(fx.arith.maximumf(sb, fx.Float32(0.0)), fx.Float32(254.0))
    rs = fx.Float32(fx.math.exp2(fx.Float32(127.0) - sb))
    dws = []
    for b in range_constexpr(8):
        lo = _pk_fp8(xs[4 * b] * rs, xs[4 * b + 1] * rs, fx.Int32(0), False)
        dws.append(_pk_fp8(xs[4 * b + 2] * rs, xs[4 * b + 3] * rs, lo, True))
    return dws, sb


def _mfma(a4, b4, acc, fp16=False):
    """v_mfma_f32_16x16x32_bf16: lane (i, q) gives A[i][8q:8q+8], B[8q:8q+8][i]
    (4 dwords each), gets D[4q + e][i]."""
    dtype = fx.Float16 if fp16 else fx.BFloat16
    a = Vec(a4).bitcast(dtype)
    b = Vec(b4).bitcast(dtype)
    op = fx.rocdl.mfma_f32_16x16x32_f16 if fp16 else fx.rocdl.mfma_f32_16x16x32_bf16
    return Vec(op(Vec.make_type(4, fx.Float32), [a, b, acc, 0, 0, 0]))


def _exp2(x):
    """v_exp_f32 (no denormal range reduction: the P it feeds is rounded to bf16)."""
    return fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, fx.Float32(x).ir_value()))


def _perm(hi, lo, sel):
    """v_perm_b32: byte i of the result is byte sel_i of (hi:lo), 0-3 = lo."""
    return fx.Int32(_rocdl.perm_b32(fx.Int32(hi), fx.Int32(lo), fx.Int32(sel)))


def _bf16x2_e4m3(word, hi):
    """E4M3 bytes 0, 1 (``hi``: 2, 3) of ``word`` -> one dword of two bf16 (exact)."""
    return Vec(
        fx.rocdl.cvt_scalef32_pk_bf16_fp8(
            Vec.make_type(2, fx.BFloat16),
            fx.as_ir_value(fx.Int32(word)),
            fx.as_ir_value(fx.Float32(1.0)),
            hi,
        )
    ).bitcast(fx.Int32)[0]


def _k_bf16x8(codes, tab_lo, tab_hi):
    """Eight K codes (one dword: byte b = dims 2 b / 2 b + 1 in its low / high
    nibble) under one block scale whose e4m3 magnitude table is (tab_lo: magnitudes
    0-3, tab_hi: 4-7; ``nvfp4_index_prefill.host.nvfp4_e4m3_lut``) -> four dwords
    of bf16 pairs (dims 0-1 .. 6-7) of e4m3_rn_satfinite(e2m1 x scale): the
    magnitudes looked up with v_perm, the code's sign merged, the two nibbles
    interleaved, widened."""
    even = _perm(tab_hi, tab_lo, codes & MAG4) ^ ((codes << 4) & SIGN4)
    odd = _perm(tab_hi, tab_lo, (codes >> 4) & MAG4) ^ (codes & SIGN4)
    lo = _perm(odd, even, 0x05010400)  # dims 0, 1, 2, 3
    hi = _perm(odd, even, 0x07030602)  # dims 4, 5, 6, 7
    return [
        _bf16x2_e4m3(lo, False),
        _bf16x2_e4m3(lo, True),
        _bf16x2_e4m3(hi, False),
        _bf16x2_e4m3(hi, True),
    ]


def _f16x2_e4m3(word, hi):
    """E4M3 bytes 0, 1 (``hi``: 2, 3) of ``word`` as an fp16 pair (exact)."""
    return Vec(
        fx.rocdl.cvt_scalef32_pk_f16_fp8(
            Vec.make_type(2, fx.Float16),
            fx.as_ir_value(fx.Int32(word)),
            fx.as_ir_value(fx.Float32(1.0)),
            hi,
        )
    )


def pack_f16(a, b):
    return Vec.from_elements([a, b], fx.Float32).to(fx.Float16).bitcast(fx.Int32)[0]


def dequant_f16(codes, scales):
    """Exact E2M1 times E4M3 products, with no intermediate FP8 rounding."""
    out = []
    for j in range_constexpr(4):
        pair = Vec(
            fx.rocdl.cvt_scalef32_pk_f16_fp4(
                Vec.make_type(2, fx.Float16),
                fx.as_ir_value(fx.Int32(codes)),
                fx.as_ir_value(fx.Float32(1.0)),
                j,
            )
        )
        out.append((pair * scales[j]).bitcast(fx.Int32)[0])
    return out


def _v_bf16x8(codes, scale_pairs):
    """One V dword (byte j = keys 2 j / 2 j + 1 of one dim, low nibble first) and
    the four key pairs' fp16 scale pairs -> four dwords of bf16 pairs (keys 0-1 ..
    6-7) of e4m3_rn_satfinite(e2m1 x scale): each code pair to fp16 (exact), times
    its two scales (exact in fp16: at most 6 significant bits, 2^-10 .. 2688),
    clamped to +-448, rounded once to e4m3, widened."""
    big = Vec.filled(2, 448.0, fx.Float16)
    small = Vec.filled(2, -448.0, fx.Float16)
    f8 = [None, None]
    for j in range_constexpr(4):
        v = Vec(
            fx.rocdl.cvt_scalef32_pk_f16_fp4(
                Vec.make_type(2, fx.Float16),
                fx.as_ir_value(fx.Int32(codes)),
                fx.as_ir_value(fx.Float32(1.0)),
                j,
            )
        )
        v = v * scale_pairs[j]
        v = _arith.MinNumFOp(
            fx.as_ir_value(v), fx.as_ir_value(big), fastmath=_nnan()
        ).result
        v = _arith.MaxNumFOp(v, fx.as_ir_value(small), fastmath=_nnan()).result
        old = Vec.filled(2, 0, fx.Int16) if j % 2 == 0 else f8[j // 2]
        f8[j // 2] = Vec(
            fx.rocdl.cvt_scalef32_pk_fp8_f16(
                Vec.make_type(2, fx.Int16),
                fx.as_ir_value(old),
                v,
                fx.as_ir_value(fx.Float32(1.0)),
                j % 2 == 1,
            )
        )
    out = []
    for h in range_constexpr(2):
        word = f8[h].bitcast(fx.Int32)[0]
        out += [_bf16x2_e4m3(word, False), _bf16x2_e4m3(word, True)]
    return out


def _global_atomic_add(addr_i64, value):
    """``old = *addr; *addr += value`` on a global i32 (agent scope, relaxed)."""
    return fx.Int32(
        _llvm.AtomicRMWOp(
            _llvm.AtomicBinOp.add,
            _global_i32_ptr(addr_i64).llvm_ptr,
            fx.Int32(value).ir_value(),
            _llvm.AtomicOrdering.monotonic,
            syncscope="agent",
            alignment=4,
        ).result
    )


def compile_nvfp4_sparse_decode(
    splits: int = 1,
    topk: int = 16,
    xcds: int = 0,
    quant: bool = False,
    exact_fp16: bool = False,
):
    """``splits``: workgroups per (row, KV head), each over a contiguous share of
    the row's blocks. ``topk``: the top-k width (one lane per slot). ``xcds`` > 0:
    workgroup w runs on XCD w % xcds; a request's rows (and a row's splits) are
    placed on one XCD, where they share blocks in its L2. ``quant``: instead of
    the bf16 output, write its MXFP8 quant as the o_proj's input: e4m3 codes
    row-major or, with ``i32_tile`` 16, in the decode GEMM's (16, 16) tile
    layout, and e8m0 scales in AITER's shuffled layout (the bytes of AITER's
    MXFP8 GEMM quantizers on the bf16 output)."""
    assert 1 <= topk <= 64 and splits >= 1
    SPLITS, TOPK, XCDS = splits, topk, xcds
    PART_SCOPE = 0 if XCDS else AGENT_SCOPE
    name = (
        f"m3_nvfp4_sparse_decode_k{TOPK}_s{SPLITS}"
        + (f"_x{XCDS}" if XCDS else "")
        + ("_q" if quant else "")
        + ("_f16" if exact_fp16 else "")
    )

    @fx.struct
    class Shared:
        # partial O per (wave, d-tile, lane) as 16 B tiles, then (max, sum) per
        # (wave, lane)
        o: fx.Array[fx.Int32, NW * NDT * 64 * 4, 16]
        ml: fx.Array[fx.Float32, NW * 2 * 64]
        lut: fx.Array[fx.Int32, 512]  # the K magnitude table (lo, hi per scale byte)
        ticket: fx.Array[fx.Int32, 4]

    def store_out(arg_out, i32_rows, i32_H, row, kvh, lane, acc, inv):
        """O of ``row`` (``acc`` x ``inv``) in bf16."""
        n, g = lane % 16, lane // 16
        # lane (n, g) holds dims 8 (4 g + e) + dt of head n: per e, 8 consecutive dims
        outr = buffer_ops.create_buffer_resource_from_addr(
            arg_out, num_records_bytes=fx.Int64(i32_rows) * fx.Int64(i32_H) * (D * 2)
        )
        o_row = (row * i32_H + kvh * HG + n) * (D // 2) + g * 16
        for e in range_constexpr(4):
            val = Vec.from_elements(
                [
                    _pk_bf16(acc[2 * j][e] * inv, acc[2 * j + 1][e] * inv)
                    for j in range_constexpr(4)
                ],
                fx.Int32,
            )
            buffer_ops.buffer_store(val, outr, o_row + e * 4, mask=row < i32_rows)

    def store_q(arg_qc, arg_qs, i32_rows, i32_H, i32_tile, row, kvh, lane, acc, inv):
        """The o_proj input of ``row``: MXFP8 codes of O (row-major, or the (16, 16)
        tile layout when ``i32_tile`` is 16) and e8m0 scales in AITER's layout."""
        n, g = lane % 16, lane // 16
        words = [
            _pk_bf16(acc[2 * j][e] * inv, acc[2 * j + 1][e] * inv)
            for e in range_constexpr(4)
            for j in range_constexpr(4)
        ]
        dws, sb = _mxfp8_group(words)
        k_row = i32_H * D  # row width (elements)
        head = kvh * HG + n
        k0 = head * D + g * 32  # the lane's first dim
        rows16 = (i32_rows + 15) // 16 * 16
        qcr = buffer_ops.create_buffer_resource_from_addr(
            arg_qc, num_records_bytes=fx.Int64(rows16) * fx.Int64(k_row)
        )
        shuffled = i32_tile > 0
        # (16, 16) tiles: byte (m, k) at (m//16)*16K + (k//64)*1024 + ((k%64)//16)*256
        # + (m%16)*16 + k%16; the lane's 32 bytes are two 16 B runs 256 B apart
        off_t = (
            (row // 16) * (16 * k_row)
            + (k0 // 64) * 1024
            + ((k0 % 64) // 16) * 256
            + (row % 16) * 16
        ) // 4
        off_l = (row * k_row + k0) // 4
        off = shuffled.select(off_t, off_l)
        step = shuffled.select(fx.Int32(64), fx.Int32(4))
        for h in range_constexpr(2):
            val = Vec.from_elements(dws[4 * h : 4 * h + 4], fx.Int32)
            buffer_ops.buffer_store(val, qcr, off + h * step, mask=row < i32_rows)
        # scale (m, g): [M//32, K//256, klane 4, mlane 16, kpack 2, mpack 2]
        gi = head * 4 + g
        s_off = (
            ((((row // 32) * (k_row // 256) + gi // 8) * 4 + gi % 4) * 16 + row % 16)
            * 2
            + (gi % 8) // 4
        ) * 2 + (row % 32) // 16
        rows32 = (i32_rows + 31) // 32 * 32
        qsr = buffer_ops.create_buffer_resource_from_addr(
            arg_qs, num_records_bytes=fx.Int64(rows32) * fx.Int64(k_row // 32)
        )
        buffer_ops.buffer_store(
            sb.to(fx.Int32).to(fx.Int8), qsr, s_off, mask=row < i32_rows
        )

    # the arguments of the first (dependent) loads lead: the first 56 bytes arrive
    # preloaded in SGPRs
    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        i32_rows: fx.Int32,
        i32_ql: fx.Int32,  # rows per request
        i32_topk_hs: fx.Int32,  # topk stride over KV heads (elements)
        i32_bt_s: fx.Int32,
        i32_H: fx.Int32,
        i32_hkv: fx.Int32,
        arg_seq: fx.Int64,  # [reqs] int32
        arg_topk: fx.Int64,  # [Hkv, >= rows, TOPK] int32 logical blocks, -1 = none
        arg_bt: fx.Int64,  # [reqs, stride] int32
        arg_lut: fx.Int64,  # [256] int64 K magnitude tables
        arg_q: fx.Int64,  # [rows, H, 128] bf16
        arg_cache: fx.Int64,  # [pages, 2 * Hkv, 128, 72] uint8
        arg_ks: fx.Int64,  # fp32 global K scale
        arg_vs: fx.Int64,  # fp32 global V scale
        arg_out: fx.Int64,  # [rows, H, 128] bf16
        arg_part: fx.Int64,  # splits: [grid rows * Hkv, S, 64, PART_F] fp32
        arg_cnt: fx.Int64,  # splits: [grid rows * Hkv] int32 tickets, zero
        arg_qc: fx.Int64,  # quant: [ceil(rows / 16) * 16, H * 128] e4m3 codes
        arg_qs: fx.Int64,  # quant: [ceil(rows / 32) * 32, H * 4] e8m0 scales
        i64_part_bytes: fx.Int64,
        i32_page_byte: fx.Int32,  # cache page stride (bytes)
        i32_tile: fx.Int32,  # quant: 0 row-major codes, 16 the tile layout
        f32_c: fx.Float32,  # sm_scale * log2(e)
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx = fx.thread_idx.x
        bx, kvh = fx.block_idx.x, fx.block_idx.y
        lane = tx % 64
        w = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        n, g = lane % 16, lane // 16
        topk = _global_i32_ptr(arg_topk)
        bt = _global_i32_ptr(arg_bt)
        seq_p = _global_i32_ptr(arg_seq)
        f32p = fx.PointerType.get(
            T.f32, address_space=fx.AddressSpace.Global, alignment=4
        )
        c = f32_c * fx.Float32(fx.inttoptr(f32p, fx.Int64(arg_ks))[0])
        v_scale = fx.Float32(fx.inttoptr(f32p, fx.Int64(arg_vs))[0])
        neg_inf = fx.Float32(float("-inf"))
        if const_expr(not exact_fp16):
            lut_g = _global_i32_ptr(arg_lut)
            smem.lut.ptr[tx] = fx.Int32(lut_g[tx])
            smem.lut.ptr[tx + 256] = fx.Int32(lut_g[tx + 256])

        if const_expr(XCDS > 0):
            # grid.x is padded to XCDS x ql: a permutation that puts each request's
            # ql rows on one XCD (padding rows map past i32_rows)
            xcd = bx % XCDS
            slot = bx // XCDS
            row = ((slot // i32_ql) * XCDS + xcd) * i32_ql + slot % i32_ql
        else:
            row = bx
        row_ok = row < i32_rows
        rowc = row_ok.select(row, fx.Int32(0))
        req = rowc // i32_ql
        pos = fx.Int32(seq_p[req]) - i32_ql + rowc % i32_ql
        # a graph's padding rows (seq_len 0) and the grid's padding see nothing
        pos = (row_ok & (pos >= 0)).select(pos, fx.Int32(-1))
        pos = fx.Int32(fx.rocdl.readfirstlane(T.i32, pos))
        req = fx.Int32(fx.rocdl.readfirstlane(T.i32, req))
        self_blk = (pos >= 0).select(pos // BLK, fx.Int32(-1))

        # lane = top-k slot: its logical block and page (-1: none)
        slot_ok = lane < TOPK
        blk = fx.Int32(
            topk[kvh * i32_topk_hs + rowc * TOPK + slot_ok.select(lane, fx.Int32(0))]
        )
        ok = slot_ok & (blk >= 0) & (blk <= self_blk)
        page = fx.Int32(bt[req * i32_bt_s + ok.select(blk, fx.Int32(0))])
        page = ok.select(page, fx.Int32(-1))
        blk = ok.select(blk, fx.Int32(-1))
        vmask = fx.Int64(fx.rocdl.ballot(T.i64, ok))
        n_v = fx.Int32(fx.math.ctpop(vmask).to(fx.Int32))
        if const_expr(SPLITS > 1):
            # this split's blocks: ranks [lo, lo + chunk) in slot order
            below = (fx.Int64(1) << fx.Int64(lane)) - fx.Int64(1)
            rank = fx.Int32(fx.math.ctpop(vmask & below).to(fx.Int32))
            chunk = (n_v + (SPLITS - 1)) // SPLITS
            lo = fx.block_idx.z * chunk
            vmask = fx.Int64(
                fx.rocdl.ballot(T.i64, ok & (rank >= lo) & (rank < lo + chunk))
            )
            n_v = fx.Int32(fx.math.ctpop(vmask).to(fx.Int32))

        # causal limit of this wave's key offsets in the own block; Q^T
        lim = pos % BLK - w * KPW
        qr = buffer_ops.create_buffer_resource_from_addr(
            arg_q, num_records_bytes=fx.Int64(i32_rows) * fx.Int64(i32_H) * (D * 2)
        )
        q_row = (rowc * i32_H + kvh * HG + n) * (D // 2) + g * 16
        qf = [
            fx.Vector(
                buffer_ops.buffer_load(qr, q_row + ks * 4, vec_width=4, dtype=fx.Int32)
            )
            for ks in range_constexpr(NKS)
        ]

        if const_expr(exact_fp16):
            qf = [x.bitcast(fx.BFloat16).to(fx.Float16).bitcast(fx.Int32) for x in qf]

        # block offsets (bytes) of this KV head's K data / scales, V data / scales
        side_b = i32_hkv * (DATA_B + SCALE_B)
        k_data = kvh * DATA_B
        k_scl = i32_hkv * DATA_B + kvh * SCALE_B
        v_data = side_b + kvh * DATA_B
        v_scl = side_b + i32_hkv * DATA_B + kvh * SCALE_B
        # the bf16 prefill kernel's lane map: K rows 32 w + 8 (n // 4) + 4 st + n % 4,
        # bytes [16 g, 16 g + 16) (dims 32 g..), their scales (groups 2 g, 2 g + 1)
        # one dword; V keys 32 w + 8 g + j of dims 8 n.. one 32-byte tile, their
        # scales (group n // 2) 8 contiguous bytes
        k_rows = [w * KPW + 8 * (n // 4) + 4 * st + n % 4 for st in range_constexpr(2)]
        k_load = [(k_data + r * (D // 2) + g * 16) // 4 for r in k_rows]
        k_scale_load = (k_scl + ((w * 4 + g) * 16 + n) * 4) // 4
        v_load = (v_data + ((w * KPW + 8 * g) // 8 * 16 + n) * 32) // 4
        v_scale_load = (v_scl + (n // 2) * BLK + w * KPW + 8 * g) // 4

        def load_block(pg):
            """This lane's NVFP4 dwords of page ``pg`` (RAW): K rows st = 0, 1 (4 + 4),
            their scales (1), V dims 8 n + dt (8), the scales of their keys (2)."""
            pr = buffer_ops.create_buffer_resource_from_addr(
                arg_cache + fx.Int64(pg) * fx.Int64(i32_page_byte),
                num_records_bytes=i32_page_byte,
            )
            raw = []
            for st in range_constexpr(2):
                kd = fx.Vector(
                    buffer_ops.buffer_load(pr, k_load[st], vec_width=4, dtype=fx.Int32)
                )
                raw += [kd[i] for i in range_constexpr(4)]
            raw.append(
                fx.Int32(
                    buffer_ops.buffer_load(
                        pr, k_scale_load, vec_width=1, dtype=fx.Int32
                    )
                )
            )
            for h in range_constexpr(2):
                vd = fx.Vector(
                    buffer_ops.buffer_load(
                        pr, v_load + 4 * h, vec_width=4, dtype=fx.Int32
                    )
                )
                raw += [vd[i] for i in range_constexpr(4)]
            vs = fx.Vector(
                buffer_ops.buffer_load(pr, v_scale_load, vec_width=2, dtype=fx.Int32)
            )
            raw += [vs[i] for i in range_constexpr(2)]
            return raw

        def dequant_k(raw):
            """K frags kf[st][ks] of a block (4 dwords of bf16 pairs each)."""
            if const_expr(exact_fp16):
                kf = []
                for st in range_constexpr(2):
                    ks = _f16x2_e4m3(raw[8], st == 1)
                    kf.append(
                        [
                            Vec.from_elements(
                                dequant_f16(
                                    raw[st * 4 + j],
                                    [
                                        Vec.from_elements(
                                            [ks[j // 2], ks[j // 2]], fx.Float16
                                        )
                                        for _ in range_constexpr(4)
                                    ],
                                ),
                                fx.Int32,
                            )
                            for j in range_constexpr(NKS)
                        ]
                    )
                return kf
            kf = []
            for st in range_constexpr(2):
                tabs = []
                for grp in range_constexpr(2):
                    sb = raw[8].shrui(fx.Int32(8 * (2 * st + grp))) & fx.Int32(0xFF)
                    tabs.append(
                        (
                            fx.Int32(smem.lut.ptr[sb * 2]),
                            fx.Int32(smem.lut.ptr[sb * 2 + 1]),
                        )
                    )
                kf.append(
                    [
                        Vec.from_elements(
                            _k_bf16x8(raw[st * 4 + ks], *tabs[ks // 2]), fx.Int32
                        )
                        for ks in range_constexpr(NKS)
                    ]
                )
            return kf

        def dequant_v(raw, cur_blk):
            """V^T frags vf[dt] of a block (4 dwords of bf16 pairs each)."""
            if const_expr(exact_fp16):
                sp = []
                for j in range_constexpr(4):
                    pair = _f16x2_e4m3(raw[17 + j // 2], j % 2 == 1)
                    # Unwritten tail scale bytes can be NaN; P=0 cannot mask NaN V.
                    sp.append(
                        Vec.from_elements(
                            [
                                (
                                    cur_blk * BLK + w * KPW + 8 * g + 2 * j + k <= pos
                                ).select(pair[k], fx.Float16(0.0))
                                for k in range_constexpr(2)
                            ],
                            fx.Float16,
                        )
                    )
                return [
                    Vec.from_elements(dequant_f16(raw[9 + dt], sp), fx.Int32)
                    for dt in range_constexpr(NDT)
                ]
            # Each dword holds scales for four consecutive keys.
            sp = [_f16x2_e4m3(raw[17 + j // 2], j % 2 == 1) for j in range_constexpr(4)]
            return [
                Vec.from_elements(_v_bf16x8(raw[9 + dt], sp), fx.Int32)
                for dt in range_constexpr(NDT)
            ]

        if const_expr(not exact_fp16):
            fx.gpu.barrier()  # the LUT is in LDS

        zero4 = Vec.filled(4, 0.0, fx.Float32)
        # offset (within this wave's 32 keys) of S^T tile st, row 4 g + e
        key_off = [
            8 * g + 4 * st + e for st in range_constexpr(2) for e in range_constexpr(4)
        ]
        first = (n_v > 0).select(
            fx.Int32(fx.math.cttz(vmask).to(fx.Int32)), fx.Int32(0)
        )

        def slot_at(mask):
            """The lowest slot left in ``mask`` (the first one again when empty)."""
            return (mask != fx.Int64(0)).select(
                fx.Int32(fx.math.cttz(mask).to(fx.Int32)), first
            )

        def page_of(slot):
            pg = fx.Int32(fx.rocdl.readlane(T.i32, page, slot))
            return (pg >= 0).select(pg, fx.Int32(0))

        def attend(kf, vf, cur_blk, live, m_t, l_t, o_t):
            """One dequantized block into the running (max, sum, O): S^T, the own
            block's causal mask, the online softmax, O^T += V^T P^T. A block that is
            not ``live`` is masked whole: the max does not move, alpha is 1, P is 0."""
            a = []
            for st in range_constexpr(2):
                acc = zero4
                for ks in range_constexpr(NKS):
                    acc = _mfma(kf[st][ks], qf[ks], acc, exact_fp16)
                a += [acc[e] for e in range_constexpr(4)]
            diag = cur_blk == self_blk
            dead = live == fx.Int32(0)
            a = [
                (dead | (diag & (key_off[i] > lim))).select(neg_inf, a[i])
                for i in range_constexpr(8)
            ]
            mx = a[0]
            for i in range_constexpr(1, 8):
                mx = _maxf_nn(mx, a[i])
            # the running max is kept in scaled units: max(a) * c == max(a * c) (c > 0)
            m_new = _maxf_nn(m_t, _xlane4(mx, _maxf_nn) * c)
            m_use = (m_new == neg_inf).select(fx.Float32(0.0), m_new)
            alpha = _exp2(m_t - m_use)
            p = [_exp2(fx.math.fma(x, c, -m_use)) for x in a]
            ps = p[0]
            for i in range_constexpr(1, 8):
                ps = ps + p[i]
            pack = pack_f16 if exact_fp16 else _pk_bf16
            pf = Vec.from_elements(
                [pack(p[2 * j], p[2 * j + 1]) for j in range_constexpr(4)],
                fx.Int32,
            )
            # Multiplication by exactly one leaves the FP32 accumulator unchanged.
            av = Vec.filled(4, alpha, fx.Float32)
            if const_expr(exact_fp16):
                if alpha != fx.Float32(1.0):
                    o_t = [o_t[dt] * av for dt in range_constexpr(NDT)]
                o_t = [_mfma(vf[dt], pf, o_t[dt], True) for dt in range_constexpr(NDT)]
            else:
                o_t = [_mfma(vf[dt], pf, o_t[dt] * av) for dt in range_constexpr(NDT)]
            return m_new, l_t * alpha + ps, o_t

        # Double-buffer two blocks per iteration; an odd final B is masked.
        # FP16 delays V dequantization until after the next load, while BF16
        # retains its original schedule.
        init = [vmask, neg_inf, fx.Float32(0.0)]
        init += [zero4 for _ in range_constexpr(NDT)]
        init += load_block(page_of(slot_at(vmask)))
        n_st = 3 + NDT
        results = init
        for u, st_ in range(  # type: ignore[call-overload]
            fx.Index(0),
            fx.Index((n_v + 1) // 2),
            fx.Index(1),
            init=[fx.as_ir_value(x) for x in init],
        ):
            rem = fx.Int64(st_[0])
            m_t = fx.Float32(st_[1])
            l_t = fx.Float32(st_[2])
            o_t = [Vec(st_[3 + dt]) for dt in range_constexpr(NDT)]
            raw_a = [fx.Int32(st_[n_st + i]) for i in range_constexpr(RAW)]
            blk_a = fx.Int32(fx.rocdl.readlane(T.i32, blk, slot_at(rem)))
            rem_b = rem & (rem - fx.Int64(1))
            raw_b = load_block(page_of(slot_at(rem_b)))
            fx.rocdl.sched_barrier(0)
            kf_a = dequant_k(raw_a)
            if const_expr(not exact_fp16):
                vf_a = dequant_v(raw_a, blk_a)
            rem_n = rem_b & (rem_b - fx.Int64(1))
            fx.rocdl.sched_barrier(0)
            raw_n = load_block(page_of(slot_at(rem_n)))
            fx.rocdl.sched_barrier(0)
            if const_expr(exact_fp16):
                vf_a = dequant_v(raw_a, blk_a)
            res = attend(kf_a, vf_a, blk_a, fx.Int32(1), m_t, l_t, o_t)
            m_t, l_t, o_t = res[0], res[1], res[2]
            blk_b = fx.Int32(fx.rocdl.readlane(T.i32, blk, slot_at(rem_b)))
            live_b = (rem_b != fx.Int64(0)).select(fx.Int32(1), fx.Int32(0))
            kf_b = dequant_k(raw_b)
            vf_b = dequant_v(raw_b, blk_b)
            res = attend(kf_b, vf_b, blk_b, live_b, m_t, l_t, o_t)
            m_t, l_t, o_t = res[0], res[1], res[2]
            state = [rem_n, m_t, l_t] + o_t + raw_n
            results = yield [fx.as_ir_value(x) for x in state]

        # combine the waves' partials (wave 0)
        o_lds = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Int32, smem.o.ptr),
                fx.make_layout(NW * NDT * 64 * 4, 1),
            ),
            fx.make_layout(4, 1),
        )
        lds_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        ml = smem.ml.ptr
        ml[(w * 2) * 64 + lane] = fx.Float32(results[1])
        ml[(w * 2 + 1) * 64 + lane] = fx.Float32(results[2])
        for dt in range_constexpr(NDT):
            r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
            r.store(Vec(results[3 + dt]).bitcast(fx.Int32))
            fx.copy(lds_atom, r, fx.slice(o_lds, (None, (w * NDT + dt) * 64 + lane)))
        fx.gpu.barrier()
        if w == 0:
            ms = [fx.Float32(ml[(ww * 2) * 64 + lane]) for ww in range_constexpr(NW)]
            big = ms[0]
            for ww in range_constexpr(1, NW):
                big = _maxf_nn(big, ms[ww])
            big_raw = big
            big = (big == neg_inf).select(fx.Float32(0.0), big)
            wt = [fx.Float32(fx.math.exp2(ms[ww] - big)) for ww in range_constexpr(NW)]
            l_tot = fx.Float32(0.0)
            for ww in range_constexpr(NW):
                l_tot = l_tot + fx.Float32(ml[(ww * 2 + 1) * 64 + lane]) * wt[ww]
            l_tot = _xlane4(l_tot, lambda x, y: x + y)
            acc = [zero4 for _ in range_constexpr(NDT)]
            for ww in range_constexpr(NW):
                wv = Vec.filled(4, wt[ww], fx.Float32)
                for dt in range_constexpr(NDT):
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    fx.copy(
                        lds_atom,
                        fx.slice(o_lds, (None, (ww * NDT + dt) * 64 + lane)),
                        r,
                    )
                    acc[dt] = acc[dt] + Vec(r.load()).bitcast(fx.Float32) * wv
            if const_expr(SPLITS > 1):
                # this split's partial; the row's last split combines
                tid = row * i32_hkv + kvh
                pr = buffer_ops.create_buffer_resource_from_addr(
                    arg_part, num_records_bytes=i64_part_bytes
                )
                row_f = (tid * SPLITS + fx.block_idx.z) * 64 + lane
                for dt in range_constexpr(NDT):
                    buffer_ops.buffer_store(
                        acc[dt].bitcast(fx.Int32),
                        pr,
                        row_f * PART_F + dt * 4,
                        cache_modifier=PART_SCOPE,
                    )
                buffer_ops.buffer_store(
                    Vec.from_elements([_bits(big_raw), _bits(l_tot)], fx.Int32),
                    pr,
                    row_f * PART_F + NDT * 4,
                    cache_modifier=PART_SCOPE,
                )
                # the partial stores are done before the ticket is taken
                fx.rocdl.s_waitcnt(vmcnt=0)
                if lane == 0:
                    smem.ticket.ptr[0] = _global_atomic_add(
                        arg_cnt + fx.Int64(tid) * 4, 1
                    )
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                if (
                    fx.Int32(
                        fx.rocdl.readfirstlane(T.i32, fx.Int32(smem.ticket.ptr[0]))
                    )
                    == SPLITS - 1
                ):
                    rows_f = [
                        (tid * SPLITS + sp) * 64 + lane
                        for sp in range_constexpr(SPLITS)
                    ]
                    mls = [
                        Vec(
                            buffer_ops.buffer_load(
                                pr,
                                rf * PART_F + NDT * 4,
                                vec_width=2,
                                dtype=fx.Int32,
                                cache_modifier=PART_SCOPE,
                            )
                        )
                        for rf in rows_f
                    ]
                    bigs = [_as_f32(x[0]) for x in mls]
                    top = bigs[0]
                    for sp in range_constexpr(1, SPLITS):
                        top = _maxf_nn(top, bigs[sp])
                    top = (top == neg_inf).select(fx.Float32(0.0), top)
                    ws = [fx.Float32(fx.math.exp2(b - top)) for b in bigs]
                    l_all = fx.Float32(0.0)
                    for sp in range_constexpr(SPLITS):
                        l_all = l_all + _as_f32(mls[sp][1]) * ws[sp]
                    inv_all = (l_all > fx.Float32(0.0)).select(
                        v_scale / l_all, fx.Float32(0.0)
                    )
                    acc_all = [zero4 for _ in range_constexpr(NDT)]
                    for sp in range_constexpr(SPLITS):
                        wv = Vec.filled(4, ws[sp], fx.Float32)
                        for dt in range_constexpr(NDT):
                            part = Vec(
                                buffer_ops.buffer_load(
                                    pr,
                                    rows_f[sp] * PART_F + dt * 4,
                                    vec_width=4,
                                    dtype=fx.Int32,
                                    cache_modifier=PART_SCOPE,
                                )
                            ).bitcast(fx.Float32)
                            acc_all[dt] = acc_all[dt] + part * wv
                    if const_expr(quant):
                        store_q(
                            arg_qc,
                            arg_qs,
                            i32_rows,
                            i32_H,
                            i32_tile,
                            row,
                            kvh,
                            lane,
                            acc_all,
                            inv_all,
                        )
                    else:
                        store_out(
                            arg_out, i32_rows, i32_H, row, kvh, lane, acc_all, inv_all
                        )
                    if lane == 0:
                        _global_i32_ptr(arg_cnt + fx.Int64(tid) * 4)[0] = fx.Int32(0)
            else:
                inv = (l_tot > fx.Float32(0.0)).select(v_scale / l_tot, fx.Float32(0.0))
                if const_expr(quant):
                    store_q(
                        arg_qc,
                        arg_qs,
                        i32_rows,
                        i32_H,
                        i32_tile,
                        row,
                        kvh,
                        lane,
                        acc,
                        inv,
                    )
                else:
                    store_out(arg_out, i32_rows, i32_H, row, kvh, lane, acc, inv)

    @flyc.jit
    def launch(
        arg_q: fx.Int64,
        arg_cache: fx.Int64,
        arg_topk: fx.Int64,
        arg_bt: fx.Int64,
        arg_seq: fx.Int64,
        arg_out: fx.Int64,
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        arg_lut: fx.Int64,
        arg_part: fx.Int64,
        arg_cnt: fx.Int64,
        arg_qc: fx.Int64,
        arg_qs: fx.Int64,
        i64_part_bytes: fx.Int64,
        i32_rows: fx.Int32,
        i32_tile: fx.Int32,
        i32_ql: fx.Int32,
        i32_H: fx.Int32,
        i32_hkv: fx.Int32,
        i32_topk_hs: fx.Int32,
        i32_bt_s: fx.Int32,
        i32_page_b: fx.Int32,
        f32_c: fx.Float32,
        i32_grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_rows,
            i32_ql,
            i32_topk_hs,
            i32_bt_s,
            i32_H,
            i32_hkv,
            arg_seq,
            arg_topk,
            arg_bt,
            arg_lut,
            arg_q,
            arg_cache,
            arg_ks,
            arg_vs,
            arg_out,
            arg_part,
            arg_cnt,
            arg_qc,
            arg_qs,
            i64_part_bytes,
            i32_page_b,
            i32_tile,
            f32_c,
        ).launch(
            grid=(i32_grid_x, i32_hkv, SPLITS),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
