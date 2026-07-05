# wgrad variable-K 长尾:band-cyclic interleave 负载均衡、masked/persist crossover 1536

> 类别: 方法论 · 主题标签: wgrad, load-balance, prefetch, autotune-dispatch

## variable-K 长尾根因
- wgrad 是 **variable-K GEMM**:output tile 等工作量(固定 OUT_M×OUT_N,MFMA 数相同),但 **K-loop 长度 = per-group M_g**。
- token skew 下大组 tile 串行主导 wall-time,小组 WG 空转 → group-contiguous dispatch 的长尾:
  - 2:1 skew **−20%**
  - 4:1 skew **−34%**
  - 32:1 skew **−59%**

## band-cyclic group interleave(负载均衡)
- cluster = 一个 group_m M-band(`group_m` 行 × N_BLOCKS_N 列)。
- dispatch 顺序:`band0_group0, band0_group1, ..., band0_group(G-1), band1_group0, ...` → 每时刻所有组都在飞(负载均衡)。
- band 内保留 group_m 的 B-stripe **L2 reuse**,因此 balanced 零损失。
- 要求 `N_BLOCKS_M % group_m == 0`,否则 fallback 一行 cluster。
- **balanced 零损失是默认开的关键前提**:band 内 tile 顺序与 group_m swizzle 完全一致。

### skew 鲁棒性(修后实测)
| 场景 | 数字 |
|---|---|
| balanced | 2143(−0.9% 几乎无损) |
| 30:1 skew | 1162 → 1592(−46% → **−26%**) |
| vs Triton skew | 1.18× → 1.62×(FlyDSL band-cyclic 更均衡) |

## masked vs persist(K-loop body 分叉)
- **masked chunked(大-M,per-group m_total/G > 1536)**:outer runtime `scf.for over ceildiv(k_iters, chunk)` × inner `range_constexpr(chunk)` 的 **4-buffer 流水**。over-run 由 per-group SRD `num_records` clamp 到 0(无需 host cap)。`chunk=8`,每 chunk 8 个 K-iter 全展开。
- **persist(小-M,per-group m_total/G <= 1536)**:`_wgrad_loop_body_pipe`,**2-stage prefetch**(prologue prefetch K-tile 0,per-iter prefetch K+1 overlap 当前 MFMA)。短 contraction 下 masked 的 chunk over-run 是废功,persist 精确跑完自己 M_g。
  - 实测 M_g=512:**856 → 1369 TF(+60%)**。

### crossover 阈值 = 1536(per-group M_g)
| M_g 区间 | 赢家 | 幅度 |
|---|---|---|
| M_g <= 1024 | persist 稳赢 | +3 ~ 20% |
| 1024 < M_g < 2048 | wash | ±4%(在 hysteresis 内) |
| M_g >= 2048 | masked 稳赢 | +5 ~ 15% |
- **1536 = wash 区中点偏向真实 MoE(高-G)**。

## 复用范式(共用坐标/寻址,只分叉 loop body)
- masked 和 persist 两个 kernel 共用:
  - `_wgrad_block_mn`:dispatch → `(group_idx, block_m, block_n)`
  - `_wgrad_rebase`:i64 SRD rebase → `(a_div, b_div)`
- 只有 K-loop body 不同(masked = chunked 4-buffer,persist = scf.for 2-stage prefetch)。
- 共用坐标/寻址逻辑、只分叉 loop body 是 FlyDSL 的复用范式。

---
来源: 04-tn-wgrad-kernel.md, 09-perf-numbers.md
