# FlyDSL fp8 GEMM 调优经验集

节点 chi2832 / chi2811（gfx950 / MI355X），容器 mlperf_gptoss，rocm 7.2。

## 文件索引

| 文件 | 内容 |
|---|---|
| `01-architecture.md` | FlyDSL fp8 GEMM 核心架构：8-wave 布局、LDS 组织、kernel 三件套 |
| `02-nt-fwd-kernel.md` | NT fwd kernel 优化：非持久路由、L2 swizzle autotune |
| `03-nn-dgrad-kernel.md` | NN dgrad kernel 优化：小-M bm128 autotune、M-branch gate |
| `04-tn-wgrad-kernel.md` | TN wgrad kernel 优化：persist/masked 选择、skew 均衡 band-cyclic |
| `05-int64-addressing.md` | >2^31 / >4GB 寻址：per-tile i64 SRD rebase，readfirstlane pin |
| `06-autotune-design.md` | autotune 设计：balanced 计时、hysteresis、cache key、M-branch |
| `07-benchmarking.md` | 正确的 benchmark 方法：rocprof 冷测、graph-replay、僵尸进程 |
| `08-deadends.md` | 死路全表：测过无用/有害的方向，直接跳过 |
| `09-perf-numbers.md` | 性能数据：vs Triton、vs GB200、vs HipKittens |
