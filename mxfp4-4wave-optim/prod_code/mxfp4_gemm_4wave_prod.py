# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MXFP4 dense GEMM — 4-wave (2x2), whole-loop bare-asm INPLACE-DIAG（prod 精简版）。

完整版（含所有实验性分支）见 turbo/mxfp4_gemm_4wave.py。
本文件只保留 prod 路径：ASMMFMA=6（whole-loop hw-loop）+ SC_VGPR=1（scale 直读 VGPR）。

性能（M8192N8192K28672, GPU7/MI355X）：5405T med / 5401T det0。
天花板：LDS 144KB/WG → occ=1 wave/SIMD，MFMA stall 89.5%（operand bubble），物理极限。
  详见 ../08-att-root-cause.md。

拓扑：4 waves，wave_m = wave_id // 2 (0/1)，wave_n = wave_id % 2 (0/1)。
Tile：BM=BN=BK=256，N_SUB=2（每 K-iter 含 2 个 128-deep sub-iter）。
每 wave：M=128（8×16-tile），N=64/slice（4×16-tile × 左/右 2 slice）。
累加器：64 个（32 accL + 32 accR）→ 256 AGPR。
"""

import os as _os
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import buffer_ops, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

from kernels.fp8_gemm_utils import G2SLoader, ceildiv, make_fp8_buffer_tensor, wait_barrier
from turbo.mxfp8_gemm_8wave import StoreCPlain
from turbo.mxfp4_gemm_8wave import (
    MfmaScaleFp4,
    S2RLoaderFp4,
    ScaleS2RPacked,
    fp4_g2s_offsets,
    grouped_xcd_pid,
)


# Prod 配置文档：这些 env var 在 _apply_prod_defaults() 里通过 setdefault 注入，
# 显式设置的 env var 优先级更高（可覆盖做对比实验）。FP4_PROD=0 可禁用全部默认值。
_PROD_DEFAULTS = {
    "FP4_ASMMFMA":         "6",   # whole-loop bare-asm（INPLACE-DIAG）
    "FP4_INPLACE":         "1",
    "FP4_INPLACE_DIAG":    "1",
    "FP4_MMORD":           "5",   # 2x4 N-block 顺序：更好的 B-operand 复用（+3.8%）
    "FP4_SINNER":          "1",
    "FP4_INPLACE_1BAR":    "0",   # per-phase s_barrier（SC_VGPR det0 必须）
    "FP4_INPLACE_ELGK":    "9",
    "FP4_WLVMCN":          "10",
    "FP4_INPLACE_ALT":     "0",   # B-side progressive（配合 MMORD=5）
    "FP4_INPLACE_GAVOID":  "1",   # g2s 避让 refill-free slot → 更好 LDS 带宽（+55T）
    "FP4_WLBARNOP":        "2",   # barrier 后 2 个 s_nop（最优值；1 略差）
    "FP4_SC_VGPR":         "1",   # scale 直读 VGPR，不走 LDS ds_read（det0 关键）
    "FP4_PIN":             "1",
    "FP4_PINSC":           "1",
    "FP4_PINBASE":         "8",
    "FP4_SCV_ILV":         "1",   # scale buffer_load 交织进 MFMA stream（覆盖 vmem slot）
}


def _apply_prod_defaults():
    if int(_os.environ.get("FP4_PROD", "1")):
        for _k, _v in _PROD_DEFAULTS.items():
            _os.environ.setdefault(_k, _v)


def preshuffle_mxfp4_scales_4w(a_e8m0, b_e8m0, K, BLOCK_M=256, BLOCK_N=256):
    """Host-side scale 预处理（lane-contiguous 布局，适配 SC_VGPR 直读路径）。

    lane-contig：每 lane 的 n_sub 个 dwords 连续存放 → 一次 buffer_load_dwordx4 读取。
    n_sub = BK // 128（prod BK=256 → n_sub=2，每 lane 4 个 dwords，对应 dwordx4）。
    """
    _apply_prod_defaults()
    from turbo.mxfp4_gemm_8wave import preshuffle_scale_lane_contig
    _ns = max(int(_os.environ.get("BLOCK_K", "256")) // 128, 1)
    return (
        preshuffle_scale_lane_contig(a_e8m0, K, 4, _ns, "A"),
        preshuffle_scale_lane_contig(b_e8m0, K, 4, _ns, "B"),
    )


def compile_mxfp4_gemm_4w(
    *,
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    block_k: int = 256,
    group_m: int = 4,
    num_xcds: int = 8,
    group_n: int = 16,   # 2D band L2 lever：gate/up 宽 N 时 +3.8%
    swizzle: bool = True,
):
    """编译 prod 4-wave kernel（ASMM=6 whole-loop + SC_VGPR，BM=BN=BK=256）。"""
    assert BLOCK_M == 256 and BLOCK_N == 256 and block_k == 256
    assert K % 256 == 0
    _apply_prod_defaults()

    BLOCK_K = block_k
    K128 = const_expr(K // 128)
    KI   = K // BLOCK_K        # K-iter 数（K=28672, BK=256 → KI=112）
    N_SUB = 2                   # BLOCK_K // 128 = 256 // 128
    BPR   = BLOCK_K // 2       # packed-fp4 每行字节数（128）
    KSTEP = BPR
    K2    = K // 2             # gmem packed-fp4 行步长（bytes）

    N_TILES_A  = 8              # BLOCK_M // 32（每 wave 8 个 M-tile）
    LDS_BN_HALF = 128           # BLOCK_N // 2（每 slice 宽度）
    N_TILES_BH = 4              # LDS_BN_HALF // 32（每 wave 每 slice 4 个 N-tile）

    # FP4_WLPAD：pad LDS 行步长减少 bank conflict（prod=0，暂无收益）
    _WLPAD = const_expr(int(_os.environ.get("FP4_WLPAD", "0")))
    LDS_ROW_STRIDE = BPR + _WLPAD
    a_lds_size  = BLOCK_M    * LDS_ROW_STRIDE   # A：256 rows
    bh_lds_size = LDS_BN_HALF * LDS_ROW_STRIDE  # BL/BR：128 rows per slice

    # G2S 步骤数（每 K-iter 的 buffer_load_lds 指令数）
    _ROWS_PER_STEP = 64 // (BPR // 16) * 4   # 4 waves，BK256 时 = 8
    N_LDS_STEPS_A  = BLOCK_M    // _ROWS_PER_STEP   # 32
    N_LDS_STEPS_BH = LDS_BN_HALF // _ROWS_PER_STEP  # 16

    # Buffer 数量（固定 2 ping-pong）：
    #   NABUF=2 → A 是 1-ahead（NABUF-1=1）
    #   NBB=2   → BL/BR/SC 各 2 个 ping-pong buffer
    NABUF = 2
    NBB   = 2

    # LDS 分配（共 ~144KB/WG → occ=1 锁死）：
    #   2xA (2×32KB=64KB) + 2xBL (2×16KB=32KB) + 2xBR (32KB) + 2xSC (~8KB) ≈ 144KB
    _anns = {f"A_lds{i}":  fx.Array[fx.Float8E4M3FN, a_lds_size,  16] for i in range_constexpr(NABUF)}
    for _b in range_constexpr(NBB):
        _anns[f"BL_lds{_b}"] = fx.Array[fx.Float8E4M3FN, bh_lds_size, 16]
    for _b in range_constexpr(NBB):
        _anns[f"BR_lds{_b}"] = fx.Array[fx.Float8E4M3FN, bh_lds_size, 16]
    _SCBUF = 4 * 4 * N_SUB * 64   # n_waves=4 × 4 groups × n_sub=2 × 64 lanes
    for _b in range_constexpr(NBB):
        _anns[f"SC_lds{_b}"] = fx.Array[fx.Int32, _SCBUF, 16]
    SharedStorageFp4_4w = fx.struct(type("SharedStorageFp4_4w", (), {"__annotations__": _anns}))

    @flyc.kernel(known_block_size=[256, 1, 1])
    def kernel_gemm_4w(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        F8 = fx.Float8E4M3FN.ir_type
        lds    = fx.SharedAllocator().allocate(SharedStorageFp4_4w).peek()
        A_buf  = [getattr(lds, f"A_lds{i}") for i in range_constexpr(NABUF)]
        BL_buf = [getattr(lds, f"BL_lds{b}") for b in range_constexpr(NBB)]
        BR_buf = [getattr(lds, f"BR_lds{b}") for b in range_constexpr(NBB)]
        SC_buf = [getattr(lds, f"SC_lds{b}") for b in range_constexpr(NBB)]

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m  = wave_id // 2   # 0 or 1
        wave_n  = wave_id % 2    # 0 or 1
        block_m, block_n = grouped_xcd_pid(
            fx.block_idx.x, c_m, c_n, BLOCK_M, BLOCK_N,
            group_m=group_m, num_xcds=num_xcds, group_n=group_n,
        )

        A_off  = block_m * BLOCK_M * K2
        BL_off = block_n * BLOCK_N * K2
        BR_off = (block_n * BLOCK_N + LDS_BN_HALF) * K2

        gA    = make_fp8_buffer_tensor(A,   F8)
        gB    = make_fp8_buffer_tensor(B_T, F8)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        mfma   = MfmaScaleFp4(N_TILES_A, N_TILES_BH, packed=True)
        rsrc_a = buffer_ops.create_buffer_resource(A,   max_size=False, num_records_bytes=c_m * K2)
        rsrc_b = buffer_ops.create_buffer_resource(B_T, max_size=False, num_records_bytes=c_n * K2)

        gl_off_a = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_A,  BPR, swizzle=swizzle)
        gl_off_b = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_BH, BPR, swizzle=swizzle)

        a_g2s  = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A,  F8, wave_id)
        bl_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_BH, F8, wave_id)
        br_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_BH, F8, wave_id)

        # ASMM=6 whole-loop：pad=False（手写 asm 路径不用 FlyDSL pad 逻辑）
        a_s2r = S2RLoaderFp4(wave_m, N_TILES_A,  N_SUB, BPR, LDS_ROW_STRIDE, pad=False, swizzle=swizzle)
        b_s2r = S2RLoaderFp4(wave_n, N_TILES_BH, N_SUB, BPR, LDS_ROW_STRIDE, pad=False, swizzle=swizzle)

        # Scale：SC_VGPR=1，lane-contig 布局 → buffer_load 直读 VGPR（不经 LDS ds_read）
        _qm = ((c_m + 63) // 64) * 64
        _qn = ((c_n + 63) // 64) * 64
        sa_s2r  = ScaleS2RPacked(A_scale, _qm, K, 4)
        sb_s2r  = ScaleS2RPacked(B_scale, _qn, K, 4)
        store_c = StoreCPlain(C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_BH)

        wave_m_off = wave_m * (N_TILES_A  * 16)   # 0 or 128
        wave_n_off = wave_n * (N_TILES_BH * 16)   # 0 or 64
        sa_base  = fx.Int32(block_m * BLOCK_M + wave_m_off)
        sbl_base = fx.Int32(block_n * BLOCK_N + wave_n_off)
        sbr_base = fx.Int32(block_n * BLOCK_N + LDS_BN_HALF + wave_n_off)

        accL = [mfma.zero_value] * (N_TILES_A * N_TILES_BH)
        accR = [mfma.zero_value] * (N_TILES_A * N_TILES_BH)

        # --- Prologue ---
        # Stage 1：A[0] + B[0]（k=0），等待落地
        a_g2s.load(A_buf[0], A_off)
        bl_g2s.load(BL_buf[0], BL_off)
        br_g2s.load(BR_buf[0], BR_off)
        wait_barrier(0)

        # Stage 2：A[1] + B[1]（k=1，whole-loop PRELL=2 要求 2 个 buffer 预填）
        a_g2s.load(A_buf[1], A_off + KSTEP)
        bl_g2s.load(BL_buf[1], BL_off + KSTEP)
        br_g2s.load(BR_buf[1], BR_off + KSTEP)
        # SC_VGPR=1：scale 在 whole-loop 内通过 buffer_load 直取，prologue 不需要写 SC_lds。
        # 只等 A/B operand 的 lgkmcnt，然后做全 barrier（跨 wave 可见）。
        _llvm.inline_asm(res=None, operands_=[], asm_string="s_waitcnt lgkmcnt(0)",
                         constraints="", has_side_effects=True)
        wait_barrier(0)   # A[0..1] / B[0..1] 全部落地，loop 可以从 k=2 的 g2s 开始

        # --- LDS base pointers（SGPR-uniform，供 whole-loop asm 内部 g2s 使用）---
        a_base6  = [[a_s2r.base_addr(A_buf[b], s)  for s in range_constexpr(N_SUB)] for b in range_constexpr(NABUF)]
        bl_base6 = [[b_s2r.base_addr(BL_buf[b], s) for s in range_constexpr(N_SUB)] for b in range_constexpr(NBB)]
        br_base6 = [[b_s2r.base_addr(BR_buf[b], s) for s in range_constexpr(N_SUB)] for b in range_constexpr(NBB)]

        def _gbase(buf):
            v = fx.Int32(fx.ptrtoint(buf.ptr)) + fx.Int32(wave_id) * fx.Int32(1024)
            return rocdl.readfirstlane(T.i32, v)
        abase6  = [_gbase(A_buf[b])  for b in range_constexpr(NABUF)]
        blbase6 = [_gbase(BL_buf[b]) for b in range_constexpr(NBB)]
        brbase6 = [_gbase(BR_buf[b]) for b in range_constexpr(NBB)]
        gl_a6 = [fx.Int32(gl_off_a[st]) for st in range_constexpr(N_LDS_STEPS_A)]
        gl_b6 = [fx.Int32(gl_off_b[st]) for st in range_constexpr(N_LDS_STEPS_BH)]

        scv6 = fx.Int32(0x7f7f7f7f)   # scale clamp sentinel（NaN-safe dummy，loop 内覆盖）

        # g2s soffset 初始值（PRELL=2：loop 从 k=2 的 g2s 开始，故 soffset = 2*KSTEP）
        soff6_a  = rocdl.readfirstlane(T.i32, A_off  + fx.Int32(2 * KSTEP))
        soff6_bl = rocdl.readfirstlane(T.i32, BL_off + fx.Int32(2 * KSTEP))
        soff6_br = rocdl.readfirstlane(T.i32, BR_off + fx.Int32(2 * KSTEP))

        # SC LDS 指针（SC_VGPR 路径下 whole-loop asm 内部实际忽略 sc_rb6/sc_gb6，
        # 改走 sa_s2r.rsrc/sb_s2r.rsrc 直读 VGPR，但 API 仍要求这两个参数传入）
        _SCW = const_expr(4 * N_SUB * 64)   # dwords per wave-region per buffer
        sc_rb6 = [
            fx.ptrtoint(fx.add_offset(SC_buf[b].ptr,
                fx.make_int_tuple(fx.Int32(wave_id) * fx.Int32(_SCW) + lane_id)))
            for b in range_constexpr(NBB)
        ]
        sc_gb6 = [
            rocdl.readfirstlane(T.i32, fx.Int32(fx.ptrtoint(fx.add_offset(
                SC_buf[b].ptr, fx.make_int_tuple(fx.Int32(wave_id) * fx.Int32(_SCW))))))
            for b in range_constexpr(NBB)
        ]

        # SC_VGPR scale soffset（4 个 scale stream 的 gmem 起始 soffset）。
        # lane-contig 布局：wi * K128 * 512（kk stride = 512 bytes），lane voffset 另算。
        # _wia：A 所在 128-row 块索引（g0/g1 同属一个 dwordx4 组）
        # _wib：BL 所在 64-col slice 索引
        _wia = sa_base  // fx.Int32(128)
        _wib = (sbl_base // fx.Int32(256)) * fx.Int32(2) + (sbl_base % fx.Int32(256)) // fx.Int32(64)

        # idx1（A-g1）和 idx3（BR）用 LDS-mode 计算（PRELL_B=2, N_SUB=2 → +4 偏移 × 256）；
        # idx0（A-g0）和 idx2（BL）用 SC_VGPR 计算（wi*K128*512）。
        def _scsoff(base, extra):
            grp = (base + fx.Int32(extra)) // fx.Int32(64)
            return rocdl.readfirstlane(T.i32, (grp * fx.Int32(K128) + fx.Int32(2 * N_SUB)) * fx.Int32(256))
        sc_soff06 = [
            rocdl.readfirstlane(T.i32, _wia * fx.Int32(K128) * fx.Int32(512)),   # A-g0 SC_VGPR
            _scsoff(sa_base, 64),                                                   # A-g1 LDS-mode
            rocdl.readfirstlane(T.i32, _wib * fx.Int32(K128) * fx.Int32(512)),   # BL  SC_VGPR
            _scsoff(sbr_base, 0),                                                   # BR  LDS-mode
        ]
        sc_voff6 = lane_id * fx.Int32(8 * N_SUB)   # per-lane voffset：2*N_SUB dwords × 4B

        # --- Whole-loop K-iteration（单个 hw-loop，INPLACE-DIAG 模式）---
        accL, accR = mfma.call_mxfp4_wholeloop(
            a_base6, bl_base6, br_base6,
            a_s2r.tile_stride, b_s2r.tile_stride,
            abase6, blbase6, brbase6,
            gl_a6, gl_b6, rsrc_a, rsrc_b,
            fx.Int32(KSTEP), scv6, accL, accR,
            N_SUB, N_LDS_STEPS_A, N_LDS_STEPS_BH,
            fx.Int32(KI), soff6_a, soff6_bl, soff6_br,
            sc_rb6, sc_gb6, sa_s2r.rsrc, sb_s2r.rsrc, sc_voff6, sc_soff06,
            None, None, None,   # sca_rb6, sca_gb6, sca_voff6（SC_A2=0，不用）
        )

        # --- Store C ---
        base_row   = block_m * BLOCK_M + wave_m_off
        base_col_l = block_n * BLOCK_N + wave_n_off
        base_col_r = block_n * BLOCK_N + LDS_BN_HALF + wave_n_off
        store_c.store(accL, base_row, base_col_l)
        store_c.store(accR, base_row, base_col_r)

    OCC = const_expr(int(_os.environ.get("FP4_OCC", "1")))

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel_gemm_4w(
            A, B_T, C, A_scale, B_scale, c_m, c_n,
            value_attrs={
                "rocdl.flat_work_group_size": "256,256",
                "rocdl.waves_per_eu": OCC,   # prod: occ=1（LDS 144KB/WG 锁死，无法 occ=2）
                "passthrough": [["amdgpu-agpr-alloc", "256"]],
            },
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm
