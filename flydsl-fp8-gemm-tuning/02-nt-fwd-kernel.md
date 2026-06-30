# NT fwd kernel 优化

## 核心结论

fwd 用**非持久 nt8w**（non-persistent，one-tile-per-WG），而非 persistent scf.for。
非持久比持久快约 11%（省去 outer tile-loop 的 scf.for 调度惩罚）。
前提：L2 swizzle 必须也移植进非持久 kernel，否则非持久在小-K 上会输。

## 决定性实验

| 路径 | dsv3-up | gpt-down | 备注 |
|---|---|---|---|
| 持久 no-swizzle | 基线 | 基线 | 以前的 prod |
| **非持久 + swizzle-AT** | +8~9% | +3~5% | 全 shape 赢 |
| 非持久 no-swizzle | dsv3 +8% | gpt **-13%** | 小-K 反输 |

结论：非持久优势在大-K（循环惩罚主导），但小-K 要靠 swizzle（L2 reuse）补回。

## L2 swizzle 设计

### 1D M-cluster（num_xcd + group_m）

`group_m` 控制 M 方向的 super-block：连续 group_m 个 M-tile 属同一 M-band，同 band 内的 tile 被 XCD-aware 调度到同一 XCD 的 CU → B[g] 的 N-stripe 在同 XCD L2 保持 resident。

```python
nt_group_m = 4   # 默认，被 autotune 验证
nt_num_xcd = 8   # 固定 = 物理 XCD 数
```

### 2D band swizzle（group_n）

大-N shape（N ≥ 2880）：N 方向加 group_n（N 宽度为 group_n 个 N-tile 一组），每组的 A M-slab 被多组复用。

实测：`group_n = N_BLOCKS_N // 8`（#bands = #XCD）对 big-N 有 +8~9%。

## autotune（_autotune_np_dispatch）

3 个候选（BLOCK_M=256，按 (num_xcd, group_m, group_n) 区分）：
- `(8, 4, 0)` = default，correctness reference
- `(1, 0, 0)` = row-major（down-proj 有时赢）
- `(8, 8, 0)` = wide M-cluster

autotune 协议：
- 在**balanced group_offs**上计时（`_balanced_targs`，不被 skew 第一次 call 带偏）
- **median-of-5 × 50 iter warmup**（短-K shape 冷测 mis-pick 严重）
- **1.5% hysteresis**（防噪声 mis-pick）
- key = `(op, N, K, G, M_total, out_fp16, cbsz, blgp)`，纯静态维度

## 注意事项

- `sched_barrier(0)` before-mfma 是 load-bearing（LDS sync），**不能删**（删了小-K 正确性坏）
- `cshuffle` store 中性偏负（+2%/−2% 噪声），不值得开
- `store_cshuffle` 向量化只对 dgrad 有意义（column-strided 是 dgrad 的瓶颈，不是 fwd）
