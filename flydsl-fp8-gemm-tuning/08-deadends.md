# 死路全表（别再试）

## fwd/dgrad 死路

| 方向 | 结果 | 原因 |
|---|---|---|
| inter-tile 软流水（SWP prefetch） | 中性（fwd −2%） | readfirstlane codegen 杠杆已穷尽，store_c waterfall 是真瓶颈而非 G2S bound |
| GROUP_M / XCD 扫参（fwd） | 噪声 | autotune 已收敛；over-launch 不是瓶颈 |
| store_c CShuffle（fwd） | 中性（+2%/−2%） | fwd 不是 column-strided store 瓶颈（dgrad 才是，但也只是 milestone） |
| sched_barrier(0) before-mfma 删除 | SNR 坏 | before-mfma s_barrier 是 load-bearing LDS sync，**不能删** |
| bigger tile（256×512, 512×256） | INVALID_ISA 或 CShuffle EPL assert | BLOCK_N=512 撞 CShuffle EPL=16 ≠ 8 |
| 3-stage B ring buffer（fwd） | 中性（+0.05%） | L2 miss latency 来自 8 个不同 expert 权重 thrash 4MB/XCD L2，prefetch distance 无效 |
| per-shape NN num_xcd by c_n（dgrad） | 负（−0.5%） | 无干净阈值，overfitting |
| wgrad B-inline transpose-load（TN） | SNR nan | TN lgkm race，=v 约束不稳 |
| B→VGPR 直载（跳 LDS） | 慢 4× | uncoalesced（相邻 lane 地址差 K），B 走 LDS 就是为 coalesce |

## wgrad 死路

| 方向 | 结果 | 原因 |
|---|---|---|
| chunked K-loop 双缓冲 | 负 | 每 chunk 重启流水线 → fill bubble 更差 |
| 3-stage LDS 双缓冲 | 净负 | LDS 超 160KB/CU（每 stage 64KB × 3 = 192KB > 160KB），无法 2 wave/CU |
| scf.for iter_args carry 多值 | FlyDSL 不支持 | iter_args 目前只能 carry 简单值，无法 carry ping-pong buffer 状态 |
| masked chunk size 扫（8→4→2） | 全中性 | chunk round-up 不是 skew 瓶颈（chunk=2 和 chunk=8 一样慢） |
| persist round-robin（band-cyclic） | 无效 | persist 的 skew 瓶颈是 store/prologue-bound，不是 dispatch 顺序 |
| wgrad cap-fed masked（需 host MAX_K_ITERS） | dropless 不适用 | capacity = MoE token_dispatcher 的 host 已知值；dropless MoE 没有 capacity 约束 |
| register-prefetch wgrad（swpipe net-neg） | 净负已删 | 正确但 swpipe net negative |
| 3-stage persist（pipe3） | 0.87× persist | LDS 超限 |

## dense gemm 死路

| 方向 | 结果 | 原因 |
|---|---|---|
| per-iter SRD base 前进（去 4GB cap） | NN/TN −2% | SRD 每-load 重建有代价（per-load SRD 重建 ~2%，干净 graph-replay 实测） |
| 全 no-cap（与 Triton 一致） | −2% | 同上；bench 里无 >4GB case（最大 3.49e9 < 2^32），foldable+cap 够用 |

## 通用误区

### "是 hipBLASLt 在跑"假设
B=1 grouped fp8 **走 FLYDSL 不走 hipBLASLt**（rocprof 实锤：top_kernels 全是 `kernel_grouped_*`，无 `Cijk_*`）。bf16 路径有 B=1→hipBLASLt 的 trick，**fp8 路径不适用**。

### "quant-bound"假设
op-level bwd 低不是 quant-bound：rocprof 显示量化 kernel 占比 0.0%，真正慢的是 dgrad/wgrad kernel 本身。

### 噪声误判
- timeit.Timer.mean 在 GPU 温度不稳时被 ±5% 噪声掩盖
- 单次 3-trial 对 sub-ms shape 会被热态/冷态差异 ±8% 带偏
- **必须用 interleaved A/B（同进程交替 config A/B × N-trial，win-count 判胜）**

### 并行 bench before/after 对比失效
8-way 并行 bench 所有 kernel 慢 ~10%，after/before 口径不同时无意义（fwd 看起来"回退"其实只是跑在不同负载下）。

### autotune 单次计时 mis-pick
短-K shape 冷/热差异 >20%，warmup=10 会 mis-pick；warmup=250 才稳定。

### readfirstlane 必须 pin
任何 per-group scan 派生的值（m_start、group_idx 等），如果直接用来构建 SRD base → divergence analysis 判 VGPR → waterfall。必须显式 `_readfirstlane_i32(base)` pin 到 SGPR。
