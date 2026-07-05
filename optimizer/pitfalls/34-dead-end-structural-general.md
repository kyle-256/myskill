# 结构性死路杂项：8-wave 达不到 4-wave 长 K、dense SRD/split-K、scf.for iter_args

> 类别: 踩过的坑 · 主题标签: 8-wave, wgrad, dense-gemm, correctness

## 8-wave 结构上限
- 8-wave 每 SIMD 2 waves，顶到 256 寄存器就到头，长 K 结构上无法追平 4-wave 性能。相关实现见下方各条 ❌。

## wgrad 死路清单
- ❌ 别再试 chunked K-loop 双缓冲：净负。每 chunk 重启流水线 → fill bubble 更差。
- ❌ 别再试 masked chunk size 扫参：8→4→2 全中性。chunk round-up 不是 skew 瓶颈。
- ❌ 别再试 register-prefetch wgrad swpipe：净负。
- ❌ 别再试 persist round-robin + band-cyclic：无效。persist skew 瓶颈是 store/prologue-bound，不是 dispatch 顺序。

## fwd 死路清单
- ❌ 别再试 inter-tile 软流水（SWP prefetch）：中性，fwd −2%。store_c waterfall 才是真瓶颈，不是 G2S bound。
- ❌ 别再试 store_c CShuffle：中性。fwd 不是 column-strided store 瓶颈（dgrad 才是）。
- ❌ 别再试 GROUP_M/XCD 扫参：噪声。autotune 已收敛，over-launch 不是瓶颈。
- ❌ 别再试 3-stage B ring buffer（fwd）：中性（+0.05%）。L2 miss latency 来自 8 个不同 expert 权重 thrash 4MB/XCD L2；加 prefetch distance 无效——prefetch 治不了 capacity thrash。

## dense gemm 死路清单
- ❌ 别再试 per-iter SRD base 前进（去 4GB cap）：NN/TN −2%，per-load SRD 重建约 2% 代价。全 no-cap 也 −2%。bench 内最大 3.49e9 < 2^32，foldable + cap 已够用。

## 3-buffer LDS 超限
- ❌ 别再试 3-buffer BK256：LDS 超限。A3+B3 = 192KB > 160KB；A3+B2 = 160KB 恰满；用 SCVGPR 省 scale 16KB 后 A3+B2 = 144KB 可放，但 A3 = SUBSTREAM broken，SNR 1.8。
- ❌ 别再试 8-wave BN512：实现完但墙③ spill-dead（374TF）不值得做；实现时 SNR 24.6，且有 scale layout bug。

## 正确性坑：whole-loop unroll-2 奇数 KI phantom iter
- mxfp4 whole-loop unroll-2 do-while，在奇数 KI（K%512==256）会多算 1 个 phantom iter k=KI：g2s 偏移=下一行起点，不 OOB，但累加下一行垃圾 → SNR 5-16。
- 真实影响：7b-down（K=11008，KI=43）。
- 修法：传 `nval = KI - (KI & 1)` floor-even；循环后对 `INPLACE & SCVGPR & ki&1` 发 MFMA-only phase-A tail，消费 1-ahead 预取的 k=KI-1。
- ❌ 别再试 standalone BLOCK_N=128：也算错，不可用。

---
来源: 08-deadends.md, 05-dead-ends.md, 12-llama-aiter-baseline.md
