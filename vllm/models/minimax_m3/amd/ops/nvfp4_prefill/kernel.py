# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental FP16 compute with the BF16 prefill layout and epilogue."""

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

BLK = 128  # tokens per page (one M3 sparse block)
D = 128
HG = 16  # query heads per KV head = the MFMA N dimension
NW = 4  # waves per workgroup; wave w owns keys [KPW w, KPW w + KPW) of every page
KPW = BLK // NW
TOK_PER_WG__4 = 4  # query tokens per workgroup
TOPK__16 = 16  # selected pages per token
MAXU__64 = TOK_PER_WG__4 * TOPK__16  # union capacity (one lane per (token, slot))
DATA_B = BLK * D // 2  # packed e2m1 bytes per head per side
SCALE_B = BLK * D // 16  # e4m3 block-scale bytes per head per side
NDT = D // 16  # d-tiles of O^T
NKS = D // 32  # k-steps of S^T
RAW = 2 * 4 + 1 + 8 + 2  # NVFP4 dwords a lane loads per page
LOG2E = 1.4426950409
TINY_F32 = 1.1754943508222875e-38  # torch.finfo(float32).tiny, the quant's amax floor
PACK_ROW_PER_HEAD = 132  # packed row bytes per head: 128 e4m3 codes + 4 e8m0 scales
Q_ROW = D * 2 // 16 + 1  # 16 B LDS tiles per (token, head) row of Q^T, padded by one


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


def _maxf_nn(a, b):
    """v_max_f32 with nnan (no canonicalizing v_max x, x, x on the inputs)."""
    fm = _ir.Attribute.parse("#arith.fastmath<nnan>")
    return fx.Float32(
        _arith.MaxNumFOp(fx.as_ir_value(a), fx.as_ir_value(b), fastmath=fm).result
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


def _mfma(a4, b4, acc):
    """v_mfma_f32_16x16x32_f16: lane (i, q) gives A[i][8q:8q+8], B[8q:8q+8][i]
    (4 dwords each), gets D[4q + e][i]."""
    a = Vec(a4).bitcast(fx.Float16)
    b = Vec(b4).bitcast(fx.Float16)
    return Vec(
        fx.rocdl.mfma_f32_16x16x32_f16(
            Vec.make_type(4, fx.Float32), [a, b, acc, 0, 0, 0]
        )
    )


def _pk_f16(a, b):
    """Two fp32 -> one dword of two fp16 (RNE), ``a`` low."""
    return (
        Vec.from_elements([fx.Float32(a), fx.Float32(b)], fx.Float32)
        .to(fx.Float16)
        .bitcast(fx.Int32)[0]
    )


def _exp2(x):
    """v_exp_f32 (no denormal range reduction: the P it feeds is rounded to fp16)."""
    return fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, fx.Float32(x).ir_value()))


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


def _xlane4_or(x):
    """Bitwise or over lanes L, L^16, L^32, L^48 (Int32), broadcast to all four."""
    a, b = _permlane_swap(_rocdl.permlane32_swap, fx.Int32(x), fx.Int32(x))
    m = a | b
    a, b = _permlane_swap(_rocdl.permlane16_swap, m, m)
    return a | b


def _fp4x2_f16(word, sel):
    """E2M1 codes of byte ``sel`` of ``word`` (low nibble first) as two fp16."""
    return Vec(
        _rocdl_ops.cvt_scalef32_pk_f16_fp4(
            Vec.make_type(2, fx.Float16),
            fx.as_ir_value(word),
            fx.as_ir_value(fx.Float32(1.0)),
            sel,
        )
    )


def _e4m3x2_f16(word, hi):
    """E4M3 bytes 0, 1 (``hi``: 2, 3) of ``word`` as two fp16 (exact)."""
    return Vec(
        _rocdl_ops.cvt_scalef32_pk_f16_fp8(
            Vec.make_type(2, fx.Float16),
            fx.as_ir_value(word),
            fx.as_ir_value(fx.Float32(1.0)),
            hi,
        )
    )


def _q_f16(v):
    return Vec(v).bitcast(fx.BFloat16).to(fx.Float32).to(fx.Float16).bitcast(fx.Int32)


def _q_stage(v, q_f16):
    """A Q dword for the MFMA operand path: bf16 -> f16 conversion, skipped when
    Q already is fp16 (``q_f16``; the producer rounded fp32 -> f16 directly)."""
    return Vec(v) if q_f16 else _q_f16(v)


def compile_nvfp4_prefill_fp16(
    pack: bool = False,
    rounds: int = 2,
    quant: bool = False,
    q_f16: bool = False,
    sched: bool = True,
):
    """``pack``: instead of the bf16 output, write its MXFP8 quant as the
    sequence-parallel o_proj all-to-all's packed rows (``ops/sp_pack.py``: per row
    H x 128 e4m3 codes, then H x 4 e8m0 scales, one per 32 dims). ``rounds``: the
    combine's rounds; 2 halves its LDS so two workgroups share a CU, 1 finalizes all
    tokens at once (for grids of at most one workgroup per CU). ``quant`` writes
    row-major codes to arg_out and AITER-preshuffled scales to arg_pack.
    ``q_f16``: Q already is fp16 (produced as fp16 by norm_rope_nvfp4_insert) —
    the staging conversion bf16 -> fp32 -> fp16 is skipped.
    ``sched`` (kill switch VLLM_M3_PREFILL_MSA_SCHED, default on): the
    scheduling ports — carry the current page id in the loop state (one
    page-union walk per page instead of two), issue the independent prologue
    loads before the Q^T staging writes, and clamp the running max with
    max(m, -8e37) instead of a == -inf select. All bitwise-identical to
    ``sched=False``, which traces the incumbent scheduling.
    """
    assert rounds in (1, 2)
    assert not (pack and quant)
    tpr = TOK_PER_WG__4 // rounds  # tokens per combine round
    wpe = rounds  # waves per SIMD
    name = f"m3_nvfp4_sparse_prefill_fp16matched_tq{TOK_PER_WG__4}_ksplit"
    if sched:
        name += "_sched"
    if q_f16:
        name += "_qf16"
    if pack:
        name += "_pack"
    if quant:
        name += "_quant"
    if rounds == 1:
        name += "_r1"

    @fx.struct
    class Shared:
        # 16 B tiles: the tile's Q^T during the page loop, then the partial O per
        # (wave, token of the combine round, d-tile, lane); (max, sum) per
        # (wave, token, lane)
        o: fx.Array[fx.Int32, NW * tpr * NDT * 64 * 4, 16]
        ml: fx.Array[fx.Float32, NW * TOK_PER_WG__4 * 2 * 64]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_q: fx.Int64,  # [T, 16, 128] bf16
        arg_cache: fx.Int64,  # [pages, 2 * 1, 128, 72] uint8
        arg_topk: fx.Int64,  # [1, T, 16] int32 logical blocks, -1 = none
        arg_bt: fx.Int64,  # [reqs, stride] int32
        arg_req: fx.Int64,  # [T] int32
        arg_pos: fx.Int64,  # [T] int32
        arg_out: fx.Int64,  # [T, 16, 128] bf16
        arg_ks: fx.Int64,  # fp32 global K scale
        arg_vs: fx.Int64,  # fp32 global V scale
        arg_pack: fx.Int64,  # pack: [T, H * 132] uint8 packed rows (else unused)
        i32_T: fx.Int32,
        i32_H: fx.Int32,
        i32_hkv: fx.Int32,
        i32_topk_hs: fx.Int32,  # topk stride over KV heads (elements)
        i32_bt_s: fx.Int32,
        i32_page_byte: fx.Int32,  # cache page stride (bytes)
        f32_c: fx.Float32,  # sm_scale * log2(e)
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx = fx.thread_idx.x
        bidx, kvh = fx.block_idx.x, fx.block_idx.y
        lane = tx % 64
        w = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        n, g = lane % 16, lane // 16
        topk = _global_i32_ptr(arg_topk)
        bt = _global_i32_ptr(arg_bt)
        req_p = _global_i32_ptr(arg_req)
        pos_p = _global_i32_ptr(arg_pos)
        f32p = fx.PointerType.get(
            T.f32, address_space=fx.AddressSpace.Global, alignment=4
        )
        c = f32_c * fx.Float32(fx.inttoptr(f32p, fx.Int64(arg_ks))[0])
        v_scale = fx.Float32(fx.inttoptr(f32p, fx.Int64(arg_vs))[0])
        neg_inf = fx.Float32(float("-inf"))
        topk_base__0 = kvh * i32_topk_hs  # TODO(zz): can remove, will be 0 in tp.

        lds = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Int32, smem.o.ptr),
                fx.make_layout(NW * tpr * NDT * 64 * 4, 1),
            ),
            fx.make_layout(4, 1),
        )
        lds_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        qr = buffer_ops.create_buffer_resource_from_addr(
            arg_q, num_records_bytes=fx.Int64(i32_T) * fx.Int64(i32_H) * (D * 2)
        )
        # sched: issue every independent prologue load before the Q staging
        # writes (memory-level parallelism across the workgroup start); the
        # values are consumed below exactly as in the serial order.
        if const_expr(sched):
            pre_tc, pre_pos, pre_req = [], [], []
            for t in range_constexpr(TOK_PER_WG__4):
                tok_t = bidx * TOK_PER_WG__4 + t
                tc_t = (tok_t < i32_T).select(tok_t, fx.Int32(0))
                pre_tc.append(tc_t)
                pre_pos.append(fx.Int32(pos_p[tc_t]))
                pre_req.append(fx.Int32(req_p[tc_t]))
            lane_tok = bidx * TOK_PER_WG__4 + lane // TOPK__16
            lane_tok_ok = lane_tok < i32_T
            lane_tok_clamped = lane_tok_ok.select(lane_tok, fx.Int32(0))
            pre_blk = fx.Int32(
                topk[topk_base__0 + lane_tok_clamped * TOPK__16 + lane % TOPK__16]
            )
            pre_lane_req = fx.Int32(req_p[lane_tok_clamped])
        # rounds 2: Q^T in LDS (not 64 registers), thread (token, head, quarter)
        # stages 64 B
        if const_expr(rounds == 2):
            tq, hq, qq = tx // 64, (tx // 4) % HG, tx % 4
            tok_q = bidx * TOK_PER_WG__4 + tq
            tok_q = (tok_q < i32_T).select(tok_q, fx.Int32(0))
            q_src = (tok_q * i32_H + kvh * HG + hq) * (D // 2) + qq * 16
            if const_expr(sched):
                pre_q4 = [
                    buffer_ops.buffer_load(
                        qr, q_src + 4 * i, vec_width=4, dtype=fx.Int32
                    )
                    for i in range_constexpr(4)
                ]
        if const_expr(sched):
            # dependent page-table gathers (need the pos/req/topk values)
            pre_diag = [
                fx.Int32(bt[pre_req[t] * i32_bt_s + pre_pos[t] // BLK])
                for t in range_constexpr(TOK_PER_WG__4)
            ]
            lane_blk_ok = lane_tok_ok & (pre_blk >= 0)
            pre_lane_page = fx.Int32(
                bt[pre_lane_req * i32_bt_s + lane_blk_ok.select(pre_blk, fx.Int32(0))]
            )
        if const_expr(rounds == 2):
            for i in range_constexpr(4):
                r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                if const_expr(sched):
                    q4 = pre_q4[i]
                else:
                    q4 = buffer_ops.buffer_load(
                        qr, q_src + 4 * i, vec_width=4, dtype=fx.Int32
                    )
                r.store(_q_stage(q4, q_f16))
                q_tile = (tq * HG + hq) * Q_ROW + qq * 4 + i
                fx.copy(lds_atom, r, fx.slice(lds, (None, q_tile)))

        def q_frags(t):
            """Token t's Q^T: k-step ks's B operand (dims 32 g + 8 ks.. of head n)."""
            if const_expr(rounds == 1):
                return q_regs[t]
            out = []
            for ks in range_constexpr(NKS):
                r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                q_tile = (t * HG + n) * Q_ROW + g * 4 + ks
                fx.copy(lds_atom, fx.slice(lds, (None, q_tile)), r)
                out.append(Vec(r.load()))
            return out

        # per token: its own block's page, the causal limit on this wave's key
        # offsets (rounds 1: Q^T in registers)
        diag, lim, q_regs = [], [], []
        for t in range_constexpr(TOK_PER_WG__4):
            if const_expr(sched):
                # values pre-loaded in the prologue block above
                tc = pre_tc[t]
                pos = fx.Int32(fx.rocdl.readfirstlane(T.i32, pre_pos[t]))
                diag.append(fx.Int32(fx.rocdl.readfirstlane(T.i32, pre_diag[t])))
            else:
                tok = bidx * TOK_PER_WG__4 + t
                ok = tok < i32_T
                tc = ok.select(tok, fx.Int32(0))
                pos = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(pos_p[tc])))
                req = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(req_p[tc])))
                diag.append(
                    fx.Int32(
                        fx.rocdl.readfirstlane(
                            T.i32, fx.Int32(bt[req * i32_bt_s + pos // BLK])
                        )
                    )
                )
            lim.append(pos % BLK - w * KPW)
            if const_expr(rounds == 1):
                q_row = (tc * i32_H + kvh * HG + n) * (D // 2) + g * 16
                q_regs.append(
                    [
                        _q_stage(
                            buffer_ops.buffer_load(
                                qr, q_row + ks * 4, vec_width=4, dtype=fx.Int32
                            ),
                            q_f16,
                        )
                        for ks in range_constexpr(NKS)
                    ]
                )

        # page offsets (bytes) of this KV head's K data / scales, V data / scales
        side_b = i32_hkv * (DATA_B + SCALE_B)
        k_data = kvh * DATA_B
        k_scl = i32_hkv * DATA_B + kvh * SCALE_B
        v_data = side_b + kvh * DATA_B
        v_scl = side_b + i32_hkv * DATA_B + kvh * SCALE_B
        # Keys in page order: S^T tile st row r is key 32 w + 8 (r // 4) + 4 st + r % 4,
        # so lane (n, g)'s eight P^T / V^T key slots are keys 32 w + 8 g + j.
        # K: rows 32 w + 8 (n // 4) + 4 st + n % 4, bytes [16 g, 16 g + 16) (dims
        # 32 g..); the scales of both rows (groups 2 g, 2 g + 1) are one dword.
        k_rows = [w * KPW + 8 * (n // 4) + 4 * st + n % 4 for st in range_constexpr(2)]
        k_load = [(k_data + r * (D // 2) + g * 16) // 4 for r in k_rows]
        k_scale_load = (k_scl + ((w * 4 + g) * 16 + n) * 4) // 4
        # V: keys 32 w + 8 g + j of dims 8 n.. are one 32-byte (token octet, dim
        # octet) tile, dword dt = dim 8 n + dt (a byte = two consecutive keys); their
        # scales (group n // 2) are 8 contiguous bytes (group-major scales).
        v_load = (v_data + ((w * KPW + 8 * g) // 8 * 16 + n) * 32) // 4
        v_scale_load = (v_scl + (n // 2) * BLK + w * KPW + 8 * g) // 4

        def load_page(page):
            """This lane's NVFP4 dwords of ``page`` (RAW): K rows st = 0, 1 (4 + 4),
            their scales (1), V dims 8 n + dt (8), the scales of their keys (2)."""
            # TODO(zz): Check whether one page resource buffer can cover all pages
            # when the total size is below 4 GB.
            pr = buffer_ops.create_buffer_resource_from_addr(
                arg_cache + fx.Int64(page) * fx.Int64(i32_page_byte),
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

        # lane = (token lane // 16, slot lane % 16): the page it selected (-1: none)
        if const_expr(sched):
            # (loads hoisted into the prologue preload block above)
            lane_page = lane_blk_ok.select(pre_lane_page, fx.Int32(-1))
        else:
            lane_tok = bidx * TOK_PER_WG__4 + lane // TOPK__16
            lane_tok_ok = lane_tok < i32_T
            lane_tok_clamped = lane_tok_ok.select(lane_tok, fx.Int32(0))
            blk = fx.Int32(
                topk[topk_base__0 + lane_tok_clamped * TOPK__16 + lane % TOPK__16]
            )  # topk[0, lane_tok, slot]; resolve the block through page_table.
            lane_blk_ok = lane_tok_ok & (blk >= 0)  # <2048时候更safe.
            lane_page = fx.Int32(
                bt[
                    fx.Int32(req_p[lane_tok_clamped]) * i32_bt_s
                    + lane_blk_ok.select(blk, fx.Int32(0))
                ]
            )
            lane_page = lane_blk_ok.select(lane_page, fx.Int32(-1))
        # The union's first page is the lowest valid lane's (lane order): load it
        # before the dedup so the dedup runs under its latency.
        valid = fx.Int64(fx.rocdl.ballot(T.i64, lane_page >= 0))
        first = (valid != fx.Int64(0)).select(
            fx.Int32(fx.math.cttz(valid).to(fx.Int32)), fx.Int32(0)
        )

        def page_at(mask):
            """Page of the lowest union lane left in ``mask`` (the first page again
            when it is empty)."""
            idx = (mask != fx.Int64(0)).select(
                fx.Int32(fx.math.cttz(mask).to(fx.Int32)), first
            )
            pg = fx.Int32(fx.rocdl.readlane(T.i32, lane_page, idx))
            return (pg >= 0).select(pg, fx.Int32(0))

        if const_expr(sched):
            # the lowest valid lane is always a first occurrence: this is also
            # the union's first page; carry it in the loop state instead of
            # re-walking
            page0 = page_at(valid)
            raw0 = load_page(page0)
        else:
            raw0 = load_page(page_at(valid))
        fx.rocdl.sched_barrier(0)
        # union: the first lane of each page, in lane order
        n_dup = fx.Int32(0)  # ==0 是第一次出现.
        # TODO: constassert MAXU__64 == 64, if not 64,
        for j in range_constexpr(MAXU__64 - 1):
            pj = fx.Int32(fx.rocdl.readlane(T.i32, lane_page, j))
            n_dup = n_dup + ((lane_page == pj) & (lane > j)).select(
                fx.Int32(1), fx.Int32(0)
            )
        umask = fx.Int64(
            fx.rocdl.ballot(T.i64, (lane_page >= 0) & (n_dup == 0))
        )  # 有效&第一次出现.
        n_u = fx.Int32(fx.math.ctpop(umask).to(fx.Int32))  # 不同page的个数.

        def dequant(raw):
            """K frags kf[st][ks] and V^T frags vf[dt] of a page (4 dwords of fp16
            pairs each)."""
            kf = []
            for st in range_constexpr(2):
                ksc = _e4m3x2_f16(raw[8], st == 1)  # row st: groups 2 g, 2 g + 1
                row = []
                for ks in range_constexpr(NKS):
                    sv = Vec.from_elements([ksc[ks // 2], ksc[ks // 2]], fx.Float16)
                    row.append(
                        Vec.from_elements(
                            [
                                (_fp4x2_f16(raw[st * 4 + ks], i) * sv).bitcast(
                                    fx.Int32
                                )[0]
                                for i in range_constexpr(4)
                            ],
                            fx.Int32,
                        )
                    )
                kf.append(row)
            # vs[i]: the block scales of key slots 2 i, 2 i + 1
            vs = []
            for h in range_constexpr(2):
                vs += [_e4m3x2_f16(raw[17 + h], False), _e4m3x2_f16(raw[17 + h], True)]
            # V^T frag dt (dim 8 n + dt): data byte i of its dword holds key slots
            # 2 i, 2 i + 1, which is already the operand's fp16 pair: no re-pairing.
            vf = [
                Vec.from_elements(
                    [
                        (_fp4x2_f16(raw[9 + dt], i) * vs[i]).bitcast(fx.Int32)[0]
                        for i in range_constexpr(4)
                    ],
                    fx.Int32,
                )
                for dt in range_constexpr(NDT)
            ]
            return kf, vf

        zero4 = Vec.filled(4, 0.0, fx.Float32)
        # page offset of S^T tile st, row 4 g + e within this wave's 32 keys
        key_off = [
            8 * g + 4 * st + e for st in range_constexpr(2) for e in range_constexpr(4)
        ]
        if const_expr(rounds == 2):
            fx.gpu.barrier()  # Q^T staged
        # iteration u loads page u + 1 and dequantizes page u (loaded one
        # iteration earlier); sched carries the current page id in the loop
        # state so the union walk runs once per page (for the prefetch), not
        # twice
        base = 2 if sched else 1  # loop-state prefix: [rem] (+ [page] if sched)
        init = [umask & (umask - fx.Int64(1)), page0] if sched else [umask]
        init += [neg_inf for _ in range_constexpr(TOK_PER_WG__4)] + [
            fx.Float32(0.0) for _ in range_constexpr(TOK_PER_WG__4)
        ]
        init += [zero4 for _ in range_constexpr(TOK_PER_WG__4 * NDT)]
        init += raw0
        n_st = base + 2 * TOK_PER_WG__4 + TOK_PER_WG__4 * NDT
        results = init
        # FlyDSL rewrites this loop with loop-carried state.
        for u, st_ in range(  # type: ignore[call-overload]
            fx.Index(0),
            fx.Index(n_u),
            fx.Index(1),
            init=[fx.as_ir_value(x) for x in init],
        ):
            rem = fx.Int64(st_[0])
            if const_expr(sched):
                page = fx.Int32(st_[1])
            m = [fx.Float32(st_[base + t]) for t in range_constexpr(TOK_PER_WG__4)]
            l_run = [
                fx.Float32(st_[base + TOK_PER_WG__4 + t])
                for t in range_constexpr(TOK_PER_WG__4)
            ]
            o = [
                [
                    Vec(st_[base + 2 * TOK_PER_WG__4 + t * NDT + dt])
                    for dt in range_constexpr(NDT)
                ]
                for t in range_constexpr(TOK_PER_WG__4)
            ]
            raw = [fx.Int32(st_[n_st + i]) for i in range_constexpr(RAW)]
            if const_expr(sched):
                page_next = page_at(rem)
                rem = rem & (rem - fx.Int64(1))
                raw_next = load_page(page_next)
            else:
                page = page_at(rem)
                rem = rem & (rem - fx.Int64(1))
                raw_next = load_page(page_at(rem))
            kf, vf = dequant(raw)
            hits = fx.Int64(fx.rocdl.ballot(T.i64, lane_page == page))
            for t in range_constexpr(TOK_PER_WG__4):
                m_t, l_t, o_t = m[t], l_run[t], o[t]
                if (hits.shrui(fx.Int64(TOPK__16 * t)) & fx.Int64(0xFFFF)) != fx.Int64(
                    0
                ):
                    qf = q_frags(t)
                    a = []
                    for st in range_constexpr(2):
                        acc = zero4
                        for ks in range_constexpr(NKS):
                            acc = _mfma(kf[st][ks], qf[ks], acc)
                        a += [acc[e] for e in range_constexpr(4)]
                    if page == diag[t]:
                        a = [
                            (key_off[i] <= lim[t]).select(a[i], neg_inf)
                            for i in range_constexpr(8)
                        ]
                    mx = a[0]
                    for i in range_constexpr(1, 8):
                        mx = _maxf_nn(mx, a[i])
                    # the running max is kept in scaled units:
                    # max(a) * c == max(a * c) (c > 0)
                    m_new = _maxf_nn(m_t, _xlane4(mx, _maxf_nn) * c)
                    if const_expr(sched):
                        # clamp instead of the -inf select (identical results:
                        # an all-masked page gives p = exp2(-inf) = 0 and
                        # alpha = exp2(-inf) = 0 either way; a finite scaled
                        # score is always > -8e37)
                        m_use = _maxf_nn(m_new, fx.Float32(-8.0e37))
                    else:
                        m_use = (m_new == neg_inf).select(fx.Float32(0.0), m_new)
                    alpha = _exp2(m_t - m_use)
                    p = [_exp2(fx.math.fma(x, c, -m_use)) for x in a]
                    ps = p[0]
                    for i in range_constexpr(1, 8):
                        ps = ps + p[i]
                    l_t = l_t * alpha + ps
                    pf = Vec.from_elements(
                        [_pk_f16(p[2 * j], p[2 * j + 1]) for j in range_constexpr(4)],
                        fx.Int32,
                    )
                    # alpha is exactly 1 unless some row's max grew: rescale only then
                    if fx.Int64(fx.rocdl.ballot(T.i64, m_new != m_t)) != fx.Int64(0):
                        av = Vec.filled(4, alpha, fx.Float32)
                        o_t = [x * av for x in o_t]
                    o_t = [_mfma(vf[dt], pf, o_t[dt]) for dt in range_constexpr(NDT)]
                    m_t = m_new
                m[t], l_run[t], o[t] = m_t, l_t, o_t
            state = (
                ([rem, page_next] if sched else [rem])
                + m
                + l_run
                + [x for token_out in o for x in token_out]
                + raw_next
            )
            results = yield [fx.as_ir_value(x) for x in state]

        # combine the waves' partials (over Q^T, once every wave left the loop), tpr
        # tokens per round; wave w finalizes token w in round w // tpr
        ml = smem.ml.ptr
        if const_expr(rounds == 2):
            fx.gpu.barrier()
        for t in range_constexpr(TOK_PER_WG__4):
            ml[((w * TOK_PER_WG__4 + t) * 2) * 64 + lane] = fx.Float32(
                results[base + t]
            )
            ml[((w * TOK_PER_WG__4 + t) * 2 + 1) * 64 + lane] = fx.Float32(
                results[base + TOK_PER_WG__4 + t]
            )
        for rnd in range_constexpr(rounds):
            for tt in range_constexpr(tpr):
                for dt in range_constexpr(NDT):
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    ri = base + 2 * TOK_PER_WG__4 + (tpr * rnd + tt) * NDT + dt
                    r.store(Vec(results[ri]).bitcast(fx.Int32))
                    fx.copy(
                        lds_atom,
                        r,
                        fx.slice(lds, (None, ((w * tpr + tt) * NDT + dt) * 64 + lane)),
                    )
            fx.gpu.barrier()
            if w // tpr == rnd:
                t = w
                ms = [
                    fx.Float32(ml[((ww * TOK_PER_WG__4 + t) * 2) * 64 + lane])
                    for ww in range_constexpr(NW)
                ]
                big = ms[0]
                for ww in range_constexpr(1, NW):
                    big = _maxf_nn(big, ms[ww])
                big = (big == neg_inf).select(fx.Float32(0.0), big)
                wt = [
                    fx.Float32(fx.math.exp2(ms[ww] - big)) for ww in range_constexpr(NW)
                ]
                l_tot = fx.Float32(0.0)
                for ww in range_constexpr(NW):
                    l_w = ml[((ww * TOK_PER_WG__4 + t) * 2 + 1) * 64 + lane]
                    l_tot = l_tot + fx.Float32(l_w) * wt[ww]
                l_tot = _xlane4(l_tot, lambda a, b: a + b)
                inv = (l_tot > fx.Float32(0.0)).select(v_scale / l_tot, fx.Float32(0.0))
                acc = [zero4 for _ in range_constexpr(NDT)]
                for ww in range_constexpr(NW):
                    wv = Vec.filled(4, wt[ww], fx.Float32)
                    for dt in range_constexpr(NDT):
                        r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                        o_tile = ((ww * tpr + t - tpr * rnd) * NDT + dt) * 64 + lane
                        fx.copy(lds_atom, fx.slice(lds, (None, o_tile)), r)
                        acc[dt] = acc[dt] + Vec(r.load()).bitcast(fx.Float32) * wv
                # lane (n, g) holds dims 8 (4 g + e) + dt of head n: per e, 8
                # consecutive dims, i.e. dims [32 g, 32 g + 32) of head n in order
                tok = bidx * TOK_PER_WG__4 + t
                words = [
                    [
                        _pk_bf16(acc[2 * j][e] * inv, acc[2 * j + 1][e] * inv)
                        for j in range_constexpr(4)
                    ]
                    for e in range_constexpr(4)
                ]
                if const_expr(pack or quant):
                    # The lane's 32 dims are one MXFP8 group: the quant of the bf16
                    # output as sp_pack's Triton kernel computes it (amax of the bf16
                    # values floored at TINY, scale ceil(log2(amax / 448)) biased and
                    # clamped, codes RNE), into the all-to-all's packed row.
                    xs = []
                    for e in range_constexpr(4):
                        for j in range_constexpr(4):
                            xs.append(_as_f32(words[e][j] << 16))
                            xs.append(_as_f32(words[e][j] & fx.Int32(-65536)))
                    amax = fx.Float32(TINY_F32)
                    for x in xs:
                        amax = fx.arith.maximumf(amax, fx.math.absf(x))
                    lg = fx.math.log2(amax / fx.Float32(448.0))
                    sb = fx.math.ceil(lg) + fx.Float32(127.0)
                    sb = fx.arith.minimumf(
                        fx.arith.maximumf(sb, fx.Float32(0.0)), fx.Float32(254.0)
                    )
                    rs = fx.Float32(fx.math.exp2(fx.Float32(127.0) - sb))
                    dws = []
                    for b in range_constexpr(8):
                        x4 = [x * rs for x in xs[4 * b : 4 * b + 4]]
                        lo = _pk_fp8(x4[0], x4[1], fx.Int32(0), False)
                        dws.append(_pk_fp8(x4[2], x4[3], lo, True))
                    if const_expr(quant):
                        k_row = i32_H * D
                        head = kvh * HG + n
                        qcr = buffer_ops.create_buffer_resource_from_addr(
                            arg_out, num_records_bytes=fx.Int64(i32_T) * fx.Int64(k_row)
                        )
                        q_off = (tok * k_row + head * D + g * 32) // 4
                        for h in range_constexpr(2):
                            val = Vec.from_elements(dws[4 * h : 4 * h + 4], fx.Int32)
                            buffer_ops.buffer_store(
                                val, qcr, q_off + h * 4, mask=tok < i32_T
                            )
                        # AITER scales: [M/32,K/256,4,16,2,2].
                        gi = head * 4 + g
                        s_off = (
                            (
                                (((tok // 32) * (k_row // 256) + gi // 8) * 4 + gi % 4)
                                * 16
                                + tok % 16
                            )
                            * 2
                            + (gi % 8) // 4
                        ) * 2 + (tok % 32) // 16
                        rows32 = (i32_T + 31) // 32 * 32
                        qsr = buffer_ops.create_buffer_resource_from_addr(
                            arg_pack,
                            num_records_bytes=fx.Int64(rows32) * fx.Int64(k_row // 32),
                        )
                        buffer_ops.buffer_store(
                            sb.to(fx.Int32).to(fx.Int8), qsr, s_off, mask=tok < i32_T
                        )
                    else:
                        row_dw = i32_H * (PACK_ROW_PER_HEAD // 4)
                        pack_bytes = (
                            fx.Int64(i32_T) * fx.Int64(i32_H) * PACK_ROW_PER_HEAD
                        )
                        packr = buffer_ops.create_buffer_resource_from_addr(
                            arg_pack, num_records_bytes=pack_bytes
                        )
                        p_row = tok * row_dw + (kvh * HG + n) * (D // 4) + g * 8
                        for h in range_constexpr(2):
                            val = Vec.from_elements(dws[4 * h : 4 * h + 4], fx.Int32)
                            buffer_ops.buffer_store(
                                val, packr, p_row + h * 4, mask=tok < i32_T
                            )
                        # The head's four scale bytes: one dword after the codes.
                        sdw = _xlane4_or(sb.to(fx.Int32) << (g * 8))
                        buffer_ops.buffer_store(
                            sdw,
                            packr,
                            tok * row_dw + i32_H * (D // 4) + kvh * HG + n,
                            mask=(tok < i32_T) & (g == 0),
                        )
                else:
                    outr = buffer_ops.create_buffer_resource_from_addr(
                        arg_out,
                        num_records_bytes=fx.Int64(i32_T) * fx.Int64(i32_H) * (D * 2),
                    )
                    o_row = (tok * i32_H + kvh * HG + n) * (D // 2) + g * 16
                    for e in range_constexpr(4):
                        val = Vec.from_elements(words[e], fx.Int32)
                        buffer_ops.buffer_store(
                            val, outr, o_row + e * 4, mask=tok < i32_T
                        )
            fx.gpu.barrier()

    @flyc.jit
    def launch(
        arg_q: fx.Int64,
        arg_cache: fx.Int64,
        arg_topk: fx.Int64,
        arg_bt: fx.Int64,
        arg_req: fx.Int64,
        arg_pos: fx.Int64,
        arg_out: fx.Int64,
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        arg_pack: fx.Int64,
        i32_T: fx.Int32,
        i32_H: fx.Int32,
        i32_hkv: fx.Int32,
        i32_topk_hs: fx.Int32,
        i32_bt_s: fx.Int32,
        i32_page_b: fx.Int32,
        f32_c: fx.Float32,
        stream: fx.Stream,
    ):
        kernel(
            arg_q,
            arg_cache,
            arg_topk,
            arg_bt,
            arg_req,
            arg_pos,
            arg_out,
            arg_ks,
            arg_vs,
            arg_pack,
            i32_T,
            i32_H,
            i32_hkv,
            i32_topk_hs,
            i32_bt_s,
            i32_page_b,
            f32_c,
            # rounds 2: two workgroups per CU (256 registers, 72 KB of LDS)
            value_attrs={
                "rocdl.waves_per_eu": wpe,
                "rocdl.flat_work_group_size": "256,256",
            },
        ).launch(
            grid=((i32_T + TOK_PER_WG__4 - 1) // TOK_PER_WG__4, i32_hkv, 1),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch.kernel_name = name
    return launch
