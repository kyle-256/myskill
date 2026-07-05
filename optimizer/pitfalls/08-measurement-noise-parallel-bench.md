# 并行/跨 GPU bench 口径：8 卡并行慢 10%、before/after 必须同并行度同 GPU

> 类别: 踩过的坑 · 主题标签: measurement-noise, parallel-bench, autotune-dispatch, triton-cache

- **8-way 并行 bench 所有 kernel 约慢 10%**：8 卡同时跑时每卡带宽被共享 L3/HBM 和电源分摊→所有 kernel 一律慢约 10%。并行 bench 只能做**相对比较**(同一次并行跑内互比)，**不能读绝对 TFLOPS**。WHY：绝对值被共享资源压低了。
- **before/after 必须同口径同并行度**：两次跑并行度不同时 A/B 对比无意义。曾出现 fwd "看似回退"其实只是 after 跑在不同负载(不同并行度)下的假象。规则：before/after 要么都单卡、要么都 N-way 同时，口径必须一致。
- **A/B 绝不同 GPU 并行两个计时任务**：'m4096 低于 racing' 曾是同一 GPU 上并行两个计时任务互相干扰造成的假象。跨 GPU 有约 **1% 方差**。正确做法：A/B 对照分开 GPU，或同 GPU 串行；关键结论用**同 GPU 交替多 trial 取中位数**。
- **多 agent 编译撞 Triton cache lock**：多 agent 同时编 kernel 会撞 `.triton.lock`。每个 agent 必须单独 `TRITON_CACHE_DIR=/tmp/triton_cache_<N>` + `HIP_VISIBLE_DEVICES=<N>`。
- **共享 csrc 编译只做一次**：先一次性 `pip install -e . --no-build-isolation`，后续 agent 只跑 Python，避免重复编译争抢。
- **sub-agent 必须 background 跑**：否则会话被锁死。

- ❌ 别再试：用 8-way 并行 bench 的绝对 TFLOPS 下结论——一律被压低约 10%，只有相对值可信。
- ❌ 别再试：before/after 跨不同并行度对比——负载不同，回退/提升都是假象。
- ❌ 别再试：同一 GPU 上并行两个计时任务互比——互相干扰(如 'm4096 低于 racing' 假象)，要串行或分 GPU。
- ❌ 别再试：多 agent 共用同一 `TRITON_CACHE_DIR`——撞 `.triton.lock`。

---
来源: 07-benchmarking.md, 08-deadends.md, 10-grouped-wgrad-4wave-3buf.md, SKILL.md
