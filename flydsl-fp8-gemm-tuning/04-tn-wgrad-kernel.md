# TN wgrad kernel 优化

## wgrad 的特殊性

wgrad = variable-K GEMM：contraction 维度是 **per-group M_g**（每组的 token 数），output = [G, OUT_M, OUT_N]（固定，模型权重形状）。
- 每组的计算量 ∝ M_g，**各组工作量不同**（skew 场景下差异极大）
- Triton 用 persistent round-robin 均衡；FlyDSL 需要专门设计

## 两个 kernel：masked 和 persistent

### masked chunked（大-M 路径）

适用：per-group contraction m_total/G > 1536。

算法：outer runtime scf.for over `ceildiv(k_iters, chunk)` × inner `range_constexpr(chunk)` 的 4-buffer 流水线，over-run 由 per-group SRD num_records clamp 到 0（无需 host cap）。

chunk=8（默认）：每 chunk 内 8 个 K-iter 被完全展开（密集流水）。

### persistent scf.for（小-M 路径）

适用：per-group contraction m_total/G ≤ 1536。

算法：`_wgrad_loop_body_pipe`，2-stage prefetch 流水，prologue prefetch K-tile 0，per-iter prefetch K+1 overlap 当前 MFMA。

为何小-M 用 persist：短 contraction 下 masked 的 chunk over-run 是废功，persist 精确跑完自己的 M_g。实测 M_g=512：856→1369 TF (+60%)。

## gate：m_total/G ≤ 1536（不是 m_total）

**关键**：用 **per-group** contraction m_total/G，不是 m_total。

错误做法（旧）：`m_total <= 2048` → 高-G MoE 被坑（G=8, M_g=512, m_total=4096→旧 gate 误判 masked，persist 反而 +13%）。

crossover sweep（G×Mg×weight）：
- M_g ≤ 1024：persist 稳赢 +3~20%
- 1024 < M_g < 2048：wash（±4%，在 hysteresis 内）
- M_g ≥ 2048：masked 稳赢 +5~15%

阈值 1536 = wash 区中点，偏向真实 MoE（高-G）。

## skew load-balance：band-cyclic group interleave

### 问题

masked grid 是 group-contiguous（`group_idx = pid // TILES_PER_GROUP`），token 倾斜时大组 tile 串行主导 wall-time，小组 WG 空转：

| max/min skew | 损失 |
|---|---|
| 2:1 | −20% |
| 4:1 | −34% |
| 32:1 | −59% |

### 原因与修法

wgrad output tile 等工作量（固定 OUT_M × OUT_N，每 tile MFMA 数相同），但 **K-loop 长度 = M_g（per-group）**，大组的 WG 要跑很多 K-iter，小组跑很少。Group-contiguous dispatch 下，大组 tile 形成长尾。

**修法：band-cyclic**（`_wgrad_block_mn`，interleave=True）：

```
cluster = 一个 group_m M-band（group_m 行 × N_BLOCKS_N 列）
dispatch 顺序：band0_group0, band0_group1, ..., band0_groupG-1, band1_group0, ...
```

效果：
- 每个时刻所有组（各种 M_g）都在飞 → 负载均衡
- band 内保留 group_m 的 B-stripe L2 reuse（balanced 场景零损失）
- `N_BLOCKS_M % group_m == 0` 时精确无越界；否则 fallback 到一行 cluster

```python
if const_expr(interleave and group_m > 0 and N_BLOCKS_M > group_m and N_BLOCKS_M % group_m == 0):
    BAND = group_m * N_BLOCKS_N
    bg = pid // BAND
    return bg % G, (bg // G)*group_m + (pid%BAND)%group_m, (pid%BAND)//group_m
```

### 修后实测（band-cyclic 默认开）

| max/min skew | OFF（旧） | ON（band-cyclic） |
|---|---|---|
| 1（balanced） | 2127 | **2141**（+0.7%）|
| 2:1 | 1693 | **1853**（+9.5%）|
| 4:1 | 1400 | **1898**（+35.6%）|
| 8:1 | 1142 | **1833**（+60.5%）|
| 32:1 | 867 | **1849**（+113%）|

关键：**balanced 不损失**（band 内 tile 顺序与 group_m swizzle 完全一致）。

### 为何 persist 不用 band-cyclic

persist 用 fixed-grid round-robin striding（每个 WG stride 步进跨越所有 tile）：天然混合各组 tile，理论上均衡。但实测 persist 在 skew 下仍塌（小-M store/prologue-bound，改 dispatch 顺序不解决），且 persist 主用途是 B=1 单组（无 skew）。

## 大-G MoE 的 i64 overflow

persist kernel 中 `m_start * OUT_M` 是累积偏移，当 G=256, M_g=1536, OUT_M=8192 时：
`m_total * OUT_M = 3.2e9 > 2^31`，int32 溢出 → 最后几组梯度静默出错。

修法：fold `m_start*OUT_{M,N}` 进 i64 SRD base（见 `05-int64-addressing.md`），num_records = per-group `M_g*OUT`（不用累积 m_end）。

## autotune 候选（masked 大-M）

```python
# 3 candidates (chunk, group_m, num_xcd)
(8, 4, 8)   # prod，group_m=4 + xcd=8（常见最优）
(8, 0, 8)   # no group_m swizzle（大-OUT_M 有时赢）
(4, 4, 8)   # chunk=4（短-M_g shapes）
```

## 代码结构（合并后）

```
_wgrad_block_mn(idx, G, TILES_PER_GROUP, N_BLOCKS_M, N_BLOCKS_N, group_m, group_n, interleave)
    → (group_idx, block_m, block_n)   # masked 和 persist 共用

_wgrad_rebase(A, B, m_start, m_end, OUT_M, OUT_N, F8_IR_t)
    → (a_div, b_div)   # i64 SRD rebase，masked 和 persist 共用
```

两个 kernel 只有 K-loop body 不同：masked 是 chunked 4-buffer；persist 是 scf.for 2-stage prefetch。
