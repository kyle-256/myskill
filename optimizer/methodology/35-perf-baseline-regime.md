# 性能基线数字与 regime 识别:vs Triton/GB200 倍数、memory-bound 不再调 GEMM

> 类别: 方法论 · 主题标签: perf-baseline, memory-bound-regime, measurement-noise, roofline

## 基线倍数(FlyDSL grouped fp8)
- vs Triton(B=8 balanced,12 MoE shape geomean,2026-06-16):
  - fwd **1.19×**、dgrad **1.14×**、wgrad **1.91×**(wgrad 相对优势最大,单 shape 最高 **2.11×**)
- vs GB200/TE(288 case):
  - fwd geomean **1.80×**、bwd **1.36×**,**286/288 PASS**(int64 解锁所有 shape)

## 本项目净收益分解(2026-06-09 → 06-16)
| kernel | before → after | 增幅 | 来源 |
|---|---|---|---|
| fwd  | 1841 → 2397 | +30% | 非持久 + swizzle-AT |
| wgrad| 1516 → 2163 | +43% | asm_mma + persist/masked M-branch + autotune |
| dgrad| ~2200 → 2307 | +5%  | bm128 小-M + path-J inline-asm |

## regime 识别:memory-bound 就别再调 GEMM
- Grok-2/Mixtral-8x22B fwd ≈ **1.0× vs GB200**:worst case = B=1 小-M 极大-N GateUP(Grok-2 **N=32768**)。
- 这是 op-level small-op/memory-bound regime:GEMM kernel 再快也被 **quant + 固定开销**淹没。
- WHY 停手:一旦识别出 memory-bound / host-overhead 主导的 regime,继续调 GEMM 内核不产生 op-level 收益 → 转去攻 quant/fusion/启动开销。

## roofline 实用判据(tile-size / M 维)
- **M≤512** 通常 memory-bound(关注带宽)。
- **M>512** compute-bound(关注 MFMA 利用率)。
- 算术强度 = flops / bytes_moved,与 roofline crossover 比较判定所在 regime。

## 为什么 grouped wgrad 是"效率优化真能体现"的路径
- grouped wgrad **≠ DVFS 功耗受限**(区别于 dense fp8):实测满频 **2400MHz ~266W**,远低于 dense randn 的 **~1072W 功耗墙**。
- 原因:MfmaUtil 仅 **~54%**,够不到功耗墙 → 指令效率/autotune 优化能真实体现。
- 反例:dense/fwd/dgrad 的指令效率优化被功耗墙掩盖(相同 MFMA → 相同功耗 → 相同频)。

---
来源: 09-perf-numbers.md, flydsl-fp8-gemm-results/SKILL.md, gemm-optimization/SKILL.md
