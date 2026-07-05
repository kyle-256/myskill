# 可比 A-B 基线:两边裸调 kernel 绕 dispatcher、blockwise vs tensorwise 对齐

> 类别: 方法论 · 主题标签: fair-comparison, benchmark-discipline, correctness-gate, timing

## blockwise vs tensorwise 目标线必须两边裸调 kernel
- 对比 blockwise 与 tensorwise 时,tensorwise 目标线**必须也是裸调 Triton kernel**(用 `bench_tensorwise_raw.py` / `bench_tensorwise_raw_bwd.py`),**绝不能用 dispatcher 路径的 tensorwise 数字**。
- WHY: dispatcher + autograd + autotune wrapper 会吃掉 **~30%**,用 dispatcher 数字会让 blockwise 显得比实际更接近目标(目标线被人为拖慢)。
- 两边用**同一套 `time_kernel`**:CUDA event + sort + **trim 20%**。

## 对比改前后必须用同一计时函数
- 计时尺子必须统一(不能自造 `time.perf_counter`),否则测出假回退:见 methodology/02（计时尺子必须统一）

## kernel 移植正确性金标准(gate)
- 同进程、同 device、同输入:原版 vs 移植版输出**逐元素比对 outdiff=0**(同源 kernel 应 bit-identical),再比 TF(应在 **±1.5%** 噪声内)。
- SNR 只能证"能跑对",**证不了"和上游同一版本"**;要证同版本必须走 outdiff=0。

---
来源: mi300-blockwise-gg-tuning/SKILL.md, remote-sync/SKILL.md
