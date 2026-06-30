# 11 — 上游 AGPR-pin commit 对 mxfp4 4-wave 无帮助（已实测）

> 评估 ROCm/FlyDSL 上游 commit `aeb5afc`（#714, 2026-06-22）"fp8_gemm_4wave: pin MFMA
> accumulator in AGPR (+5~13%)" 对我们 mxfp4 4-wave 的价值。结论：**无帮助，技术已在且更彻底**。
> 日期 2026-06-25。

## 上游 commit 做了什么

给上游 `kernels/fp8_gemm_4wave.py` 的 `Mfma16x16x128` 子类化出 `Mfma16x16x128AGPR`：
单条 MFMA 走内联汇编 `v_mfma_f32_16x16x128_f8f6f4 $0,$1,$2,$0` + 约束 `=a,v,v,0`，
让 f32x4 累加器**原地累加在 AGPR**，避免编译器在 **SSA-lowered 路径**里插
`v_accvgpr_mov/read + s_nop` 来回搬累加器（该路径的主导 stall）。fp8 上 +5~13%。
附带：XCD-swizzle 阈值 `<=`→`<`（边界 tie-break）。

## 为什么对 mxfp4 4-wave 无帮助

1. **技术已在且更彻底**。mxfp4 4-wave 生产路径 = `turbo/mxfp4_gemm_4wave.py`
   `compile_mxfp4_gemm_4w`，走 whole-loop **bareasm**（整个 K-loop = 一个内联汇编 hw-loop）。
   accs 用 `["=a"]` 且 MFMA `... $q, $a, $b, $q, ...`（dst=acc_in=$q，**output tie input = 原地**），
   cons 把 output q tie 到 input q。这正是上游技术，且 bareasm 还额外消了 per-mfma asm 边界
   + per-iter 循环开销。上游的 accvgpr-shuffle 问题在 bareasm 里**根本不存在**。

2. **实测证据（决定性）**：`turbo/test_mxfp4_4w.py` 8192³ K28672 GPU7：
   | accs 位置 | min/med TF |
   |------|------|
   | `agpr1`（默认，AGPR 原地）| 5407/5322 |
   | `agpr0`（VGPR）| 5390/5348 |
   两者**噪声内相等**（agpr0 中位甚至略高）→ 累加器在 AGPR/VGPR 对 mxfp4 4-wave 无差别 →
   上游 commit 消除的 accvgpr-shuffle stall 在我们这里**早已不存在**（bareasm 原地累加，两种都不 shuffle）。

3. XCD-swizzle `<=`→`<`：mxfp4 用自己的 block swizzle（group_m/group_n/num_xcds, GM4=4/GN4=16/NX4=8），
   不是上游那套 `SWIZZLE_THRESHOLD`，不适用。

## 旁注
- 用户 fork 的 fp8 turbo kernel（commit `e3403fdb` "fp8 4wave/8wave dense GEMM with AGPR
  in-place inline-asm MFMA"）也已用同款技术 → 上游对 fp8 turbo 同样无新增。
- 上游 commit 在 `origin/main`，**不在** `kyle/main`（fork main 落后于上游，停在 #685）。
- git：`git fetch kyle main:main` 已更新本地 main = kyle/main；工作分支 `dev/fp8-fused-quant`
  及其未提交工作（FP4_WPE 实验）未动。
