# mxfp4 4-wave 胖形状 ~6% gap：bf16 store 完全暴露，重叠/变宽/atomic 全死路

> 类别: 踩过的坑 · 主题标签: mxfp4, epilogue-store, split-K, ds_bpermute

## 根因：bf16 store 与 compute 0% 重叠
- mxfp4 4-wave 胖形状 ~6% gap 根因 = bf16 输出 epilogue store **完全暴露**（0% 与 compute 重叠）。
- gfx950 LDS=160KB/CU，操作数双缓冲已用 144KB（occ=1）仅 ~16KB 空闲 → 无空间做 store staging。
- store 本质 HBM-write-BW-bound：8192² 128MB ÷ 6.1TB/s ≈ 76% HBM 峰 ≈ **19.7us**。
- WHY 只有 fp4 有 gap：fp4 2× FLOPS/同输出，固定 store 被**快 compute 放大成暴露尾部**；fp8 同样的 store 被 2× 慢的 compute 盖住，无 gap。

## gfx950 计数器约束（贯穿所有死路）
- gfx950 `s_waitcnt` 只有**统一 vmcnt**，无独立 store 计数 `vscnt`（vscnt 是 gfx10/11；per-counter 是 gfx12+）。
- 后果：在途 store 占 vmcnt。任何把 store drip 进主循环的方案，都让每个 g2s 的 vmcnt wait 全卡在 store 上（issue/wait 串行化）。

## ❌ 别再试：store 与 compute 重叠（所有 flavor）
- **STORE_ILV（in-loop 摊 store）** — 仅回收 2.9/19.7us。主循环 VMEM 已被 g2s 打满，无空余 store 发射槽，interleave 后变 issue-bound。
- **寄存器 acc 跨 tile 存活** — 非-persistent 无 next-tile 可存活；persistent 需 256-iter_arg 大改，天花板 <3us。
- **persistent overlap** — store 的 HBM 写与下 tile g2s 的 HBM 读抢同一条总线，不构成真重叠；再叠 per-tile barrier 开销 → 净亏。
- 机制统一为：在途 store 占统一 vmcnt（见上），无法与 g2s 并行。

## ❌ 别再试：store 变宽 / coalescing（所有维度）
- **CShuffle（LDS-stage 128b 宽存）** — neutral 到略差。LDS round-trip 延迟 ~12us（128 ds_write + lgkmcnt 串行 wait + ds_read）吃掉 8× 降发射的收益。
- **DPP / permlane16_swap（转错轴）** — 只换 16-lane 组、转错轴。HW 已把跨 lane store coalesce 成 128B/行（80% HBM 写峰 = 该 pattern 上限），任何寄存器/LDS 重排都不降事务数（masked-off lane 仍发射）。
  - ⚠️ **注意区分**：这里判负的是**转错轴**的 permlane。**转对轴**的 `permlane16_swap` 转置 + `dwordx4` 宽存反而是 store-bound 胖形状的**最优 epilogue**（8192²×4096 从 fly/ait 1.039→0.998），见 `methodology/25`。别因为这条把 permlane 整体当死路。
- **ds_bpermute 正确宽存（dwordx2）** — 全面更慢：LDS-crossbar 延迟 > 省的发射量；dwordx2 仅 32B coalesce 无提升，外加 64 crossbar + drain。

## ds_bpermute 三个坑（从全错救成 maxdiff=0）
1. **in-place dst==src 损坏跨 lane 读** — native rocdl 与 inline-asm `=v,v,v` 都被 RA coalesce 成 in-place → 必须 inline-asm `=&v,v,v`（earlyclobber）。
2. **inline-asm ds_bpermute 是 opaque** — 编译器不插 `s_waitcnt lgkmcnt` → 必须显式 `rocdl.s_waitcnt(0)` drain；但 per-bperm 手写 lgkmcnt(0) 会序列化退化，**必须批量 wait**。
3. **FlyDSL JIT 缓存不 hash 模块级方法** → 需 `FLYDSL_EXTRA_SOURCE_DIRS` bust cache。

## ❌ 别再试：atomic 融合 reduce（dense split-K）
- **f32 逐元素 atomic** — 慢 2-3×：256 scalar atomic/tile 打重叠地址，硬件串行 + 2× 流量。
- **packed bf16 cshuffle-atomic**（AITER 同款 `global.atomic.fadd.v2bf16`）— 慢 0.59×：dense split-K 下 S 个 split 都 atomic-add 到同一批 [M,N] cache line，HBM atomic 争用串行。
- WHY AITER 那个高效：MoE token-scatter 各写不同地址（无争用）；dense GEMM 不适用。
- **split + reduce 是最终方案。**

## 实现级坑
- `buffer_store_short $t` 数据源必须单寄存器：若 `t` 是 `vector<4xi32>`（4 寄存器）汇编器报 `invalid operand for instruction`，须改单 i32 vgpr。
- `flyc.compile(raw, *args)` 会**执行一次 kernel** 进给它的 buffer → atomic 累加 kernel 必须用 throwaway buffer 编译、再跑进真 zeroed 输出，否则编译时那次执行污染结果。
- persistent 路径 CShuffle 损坏：`_do_overlap` 显式假设 store 只碰 VGPR/gmem 不碰 LDS；CShuffle 用 LDS+barrier 破坏 overlap 的 vmcnt/lgkmcnt 记账。
- persist+TACCW 在非默认 swizzle（非 `(4,14,8)`）下 maxdiff 数千（源码 unsafe 警告是真的，别开）。

---
来源: project_mxfp4_epilogue_store.md
