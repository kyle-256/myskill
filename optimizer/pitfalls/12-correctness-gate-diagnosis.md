# "是不是自己改坏"诊断捷径：跨后端比 SNR、rocprof 实锤 quant/hipBLASLt 误区

> 类别: 踩过的坑 · 主题标签: correctness-gate, SNR, rocprof, quant, hipBLASLt

- **判"是不是自己改坏"捷径 = 同 shape 跑两个后端比 SNR**：SNR 一样 → 是上游共用的量化/参考路径精度问题，不是 kernel 问题。实测 E4M3 tensorwise grouped gemm 的 SNR 失败是**预存的、与后端无关**：同 shape 下 `backend=None`(默认 TRITON，非 CK) 与 `FLYDSL` 的 Out-SNR **一致到小数点后 9 位**(`11.976079940795898`)。两后端共用同一条量化/参考路径，SNR 撞在同一个值上就说明问题在上游，不要去改 kernel。

- ❌ **别再试 误区"是 hipBLASLt 在跑"**：B=1 grouped **fp8 走 FLYDSL，不走 hipBLASLt**。rocprof 实锤：top_kernels 全是 `kernel_grouped_*`，**无 `Cijk_*`**(Cijk = hipBLASLt 的 kernel 命名)。注意 bf16 确实有 "B=1 → hipBLASLt" 的 trick，但 **fp8 路径不适用**，别把 bf16 的结论套到 fp8 上去解释性能。

- ❌ **别再试 误区"quant-bound"**：op-level bwd 慢**不是量化瓶颈**。rocprof 显示量化 kernel 占比 **0.0%**，真正慢的是 `dgrad`/`wgrad` kernel 本身。不要去优化 quant/cast，去优化 dgrad/wgrad 主 kernel。

---
来源: remote-sync/SKILL.md, 08-deadends.md
