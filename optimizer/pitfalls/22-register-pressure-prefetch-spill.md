# register-bound 墙下预取/double-buffer 全 spill：mxfp4 4w/8w 加 frag 破 occ

> 类别: 踩过的坑 · 主题标签: register-pressure, prefetch, double-buffer, race-correctness

## 4-wave (occ=1) 的 VGPR headroom 只有 88
- occ=1 下 V cap=512 / A cap=256 dwords(两个独立堆)。生产 mxfp4 4-wave：V=424 / A=256 / spill=0。headroom = 512-424 = **88 VGPR**。
- ❌ 别再试 register double-buffer / read-once / RING / ROBUF：2 operand set = 2×192V = **384V ≫ 88 headroom** → RAGreedy crash / `couldn't allocate v[256:259]` / LLVM UNREACHABLE。
- BK128 能 fit(192V)但引入 loop 开销，不划算。read-once 在 fly BK256 物理放不下。

## 8-wave (occ=2 天花板) VGPR 预算 256/wave，零余量
- 8-wave = 2 waves/SIMD → VGPR 预算 = 512/2 = **256/wave**。
- ISA 实测 8w whole-loop 已占满 256 VGPR 零余量：32 vec4 accs(128)+ 24 operand frags(96)+ 6 scale + ~26 地址 temp ≈ 256。
- ❌ 别再试 naive register-prefetch(+96 第2套 frag)：352>256 → occ 掉 → 512线程 WG 无法单 CU 驻留，内核起不来。架构性死路。

## A operand 1-deep 预取 (APF=1) — 双重死
- ❌ 别再试 APF=1：每个 i32x4 A frag=4 VGPR，a0p+a1p+a0n+a1n = **128 VGPR 仅 A** → Scratch spill(332)，且有 K 相关 tail race(SNR 15.3 崩)。
- 被 register-bound 墙挡死：1-deep A 预取需 +64 VGPR，raw 的 VGPR 上限只有 256-128(AGPR)=128，加后 **300>256** 破 1 wg/CU。

## MONOHOIST：把 operand ds_read 上提到迭代顶 — 只提 b1 是唯一干净赢
- ❌ 别再试 FP4_MONOHOIST=1/both/a：同时持有 a0+a1+b0+b1(~96VGPR)+scales/addr/g2s/readout 超 128 预算 → spill(scratch)，perf 掉到 **3735**(baseline ~4477)。
- ❌ 别再试 FP4_MONOHOIST=a2(a1 下移一 barrier)：单独 SNR29.6 数值错 + spill perf **3385**。b2a2 组合正确但仍 spill perf **3350**。
- ✅ 唯一干净赢点 = 只提 b1(b2，+16VGPR，num_vgpr 128→110 不 spill)：稳超 raw-baseline **+0.7%**，微超 intrinsic **+0.13~0.16%**。

## race 安全红线(真 race，非性能)
- 把 operand ds_read 上提到迭代顶部(首 barrier 之前)会读到 8 波协作填充的 LDS 未同步数据：K28672 SNR46.6 / det262400。
- b1 可安全下移一个 barrier(首用在 quad1 有 slack，且 b_cur1 在更早 barrier 已可见)；再往顶提就 race。
- a1 一个都不能提(无 slack，紧贴其首用 quad2)。

## FP4_LDSR：手动提前发射 ds_read 隐藏延迟 — 反退
- ❌ 别再试 FP4_LDSR(手动把 a1/b1 的 ds_read 提到 iter 顶):反退 ~20%(K2048 **2509 vs 3126**)。
- 机制:破坏编译器已做的分段 lgkmcnt 软流水。FP4_RAWSPLIT=1 每-mfma 一块时，编译器已自动做 operand 前置 + staggered s_waitcnt(vmcnt(5)lgkmcnt(7)→lgkmcnt(6)→lgkmcnt(3)…)。LDS 读延迟隐藏编译器已到位，Python 层加不了价值。

## 单 volatile asm 块把 operand 作 SSA 输入 = 强制全 drain
- ❌ 别再试 单个 volatile asm 块把 operand 作 SSA 输入:LLVM 必在块前插 **lgkmcnt(0) 全 drain**(不透明块读所有输入寄存器 → 等所有 ds_read 落地)，反比 per-mfma split 差。
- 正解:真正隐藏 operand 延迟必须把 ds_read 塞进 asm 块内手控 waitcnt(whole-loop bare-asm 就是这么做，staggered lgkmcnt)。

## raw-AGPR 8-wave 追平但不反超 intrinsic 的结构根因
- 1 wg/CU 要求 VGPR+AGPR≤256:raw = 108V+128A = 236；intrinsic = 256V+0A。
- raw 的 VGPR 上限只有 256-128(AGPR)=128，intrinsic 有 256 VGPR 做在飞 frag/软流水。AGPR 累加器虽不占 VGPR 计数但仍吃 256 共享预算 → raw 能腾的 VGPR 反比 intrinsic 少一半。
- 长 K 残余 0.3~1.5% = volatile-asm 每-mfma 硬编码 AGPR 强序阻止跨 s_barrier 全循环软流水 + VGPR 预算少，非 ds_read 冗余。

---
来源: 04-ceiling-analysis.md, 10-8wave-scvgpr.md, agpr_phase5_lds.md, agpr_phase5_ldsr.md, agpr_phase5_mono.md, agpr_rawasm_progress.md
