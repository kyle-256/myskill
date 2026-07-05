# 可比 A-B 基线:两边裸调 kernel 绕 dispatcher、blockwise vs tensorwise 对齐

> 类别: 方法论 · 主题标签: fair-comparison, benchmark-discipline, correctness-gate, timing

## blockwise vs tensorwise 目标线必须两边裸调 kernel
- 对比 blockwise 与 tensorwise 时,tensorwise 目标线**必须也是裸调 Triton kernel**(用 `bench_tensorwise_raw.py` / `bench_tensorwise_raw_bwd.py`),**绝不能用 dispatcher 路径的 tensorwise 数字**。
- WHY: dispatcher + autograd + autotune wrapper 会吃掉 **~30%**,用 dispatcher 数字会让 blockwise 显得比实际更接近目标(目标线被人为拖慢)。
- 两边用**同一套 `time_kernel`**:CUDA event + sort + **trim 20%**。

## 对比改前后必须用同一计时函数
- grouped-gemm autotune dispatch 场景:测"改动前后差多少"反复测出**看似很大、实际非真回退**的差异,根因是两把不同的计时尺子在比。
- 两把尺子量级不同,不可混用:
  - 框架自带 `GK._robust_time`:**250 warmup + 5×50 iters 中位数**,基于 `torch.cuda.Event`。
  - 随手写的 `time.perf_counter()` + 少量 warmup:测的是**绝对值,把 host 端 Python/launch 开销算进去**,和前者不是一个量级。
- 规则:对比改前 vs 改后**直接复用 `GK._robust_time`,不要自造**计时。

## kernel 移植正确性金标准(gate)
- 同进程、同 device、同输入:原版 vs 移植版输出**逐元素比对 outdiff=0**(同源 kernel 应 bit-identical),再比 TF(应在 **±1.5%** 噪声内)。
- SNR 只能证"能跑对",**证不了"和上游同一版本"**;要证同版本必须走 outdiff=0。

---
来源: mi300-blockwise-gg-tuning/SKILL.md, remote-sync/SKILL.md
