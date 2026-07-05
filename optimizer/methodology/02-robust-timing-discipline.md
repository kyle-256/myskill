# Robust timing:_robust_time 计时尺子统一、warmup 长度、buffer 复用防重叠虚高

> 类别: 方法论 · 主题标签: robust-timing, warmup, measurement-noise, benchmark-harness

## 计时尺子必须统一 (同一函数比改前/改后)
- 对比"改动前 vs 改后"必须用**同一个计时函数**,直接复用 `GK._robust_time`,不要自造。两把不同尺子不可比:
  - `_robust_time`:CUDA/HIP `torch.cuda.Event` 计时,warmup=250、reps=5、iters=50,取 **5 次的 median**。
  - 随手写的 `time.perf_counter()` + 少量 warmup 测的是绝对值,把 **host 端 Python/launch 开销**算进去,与 event 计时不是一个量级。
- grouped-gemm autotune dispatch 反复测出"看似很大其实非真回退"的差异,根因就是混用两把尺子。
- 禁用 `timeit.Timer` / `torch.utils.benchmark.Timer.timeit(50).mean`:会被 **CPU scheduler jitter + GPU boost clock 未稳定**污染。

## warmup 长度 (短-K / occ=1 的头号坑)
- 短-K shape **冷/热差异 >20%**,warmup 太短会严重 **mis-pick**(autotune 选错 config)。
- occ=1 的 **4-wave**(计算密集)对 boost 频率敏感:warmup=20 iters 够不到 boost,系统性**低估 5~10%**;occ≥2 的 **8-wave** 不敏感。
- 结论:测 4-wave vs 8-wave 必须 **warmup=250**(= 真实部署 boost 频率),否则会误判 4w 慢。

## 分层计时:op-level vs kernel-only
| 层级 | 计时方式 |
|---|---|
| op-level(含 quant) | graph-replay **min-of-8**(取 8 次 min 免 CPU 抖动) |
| kernel-only | `quantize_fp8_tensorwise_impl` 预量化后,CUDA event **warmup=20 / iter=100**,只包 kernel call |
| autotune 内置 | `_robust_time`:warmup=250 / reps=5 / iters=50 median |
| FlyDSL `@autotune` | `do_bench(fn, warmup=5, rep=25)`,CUDA/HIP events 返回 **median ms**;缓存 `~/.flydsl/autotune/{func}.json` |

## 防 kernel 重叠虚高吞吐
- 连续 launch **写同一 output buffer**(WAW 串行),避免每次新分配 buffer 导致 **kernel 重叠**造成虚高吞吐。
- 但实测:`_robust_time`(复用 buffer 串行)与"轮转 4-buffer 重叠感知计时"对 **4w>8w 结论一致**——**warmup 才是主因**,重叠不是。

## combined-step (fwd+bwd 成对 op)
- fwd+bwd 成对优化的 op 比 **combined step**,不比单方向:
  - `Combined Step Time = Forward + Backward Time`
  - `Combined Step TFLOPS = 6*M*N*K / (Combined Step Time*1e-3) / 1e12`
  - 等价形式:`Combined Step TFLOPS = 6 / (2/Forward_TFLOPS + 4/Backward_TFLOPS)`
- 单一比较分 = 各 shape Combined Step TFLOPS 的**几何平均**;forward/backward TFLOPS 留作诊断。
- 逐-shape 回退判定用 **Combined Step Time**,不看单个 fwd/bwd 分量的符号。保留 per-shape 分向量,防止总分掩盖局部回退。

## correctness 门 + TFLOPS 定义
- `bench_*_turbo.py`:先跑 correctness check 再用 `torch.utils.benchmark.Timer`(**20 warmup + 100 timed iters**)profile 写 CSV。
- Check 列 = PASS/FAIL/ERROR;任一 **FAIL/ERROR → timing 失效**(数字不可信,先修 correctness → score 0)。
- `Forward TFLOPS = 2*M*N*K / time / 1e12`;`Backward TFLOPS = 2*forward FLOPs / time / 1e12`。

## benchmark shape 来自真实 config
- shape 来自真实模型 config:`benchmark/ops/training/config.py`,不要硬编在 bench 脚本里(加新 op shape 扩 config.py)。
  - Dense GEMM:每模型 4 shape(attn QKV / attn out / MLP gate+up / MLP down)× MBS∈{1,2,4}。
  - Grouped GEMM / MoE:`MoEModelConfigs` + `gen_grouped_gemm_group_lens`(balance=True 均匀 / False 倾斜)两种 token 分布。
  - Attention:`gen_attention_test_cases()`。

## FlyDSL benchmark harness
- `bash scripts/run_benchmark.sh`(全 op);`--only softmax,moe` 子集;`--list` 枚举;`--output_csv /tmp/bench.csv` 出 CSV。
- 基线对比:`python3 scripts/compare_benchmark.py base.csv cur.csv` 出 ratio 报告。
- `@autotune(configs=[Config(...)], key=[...], warmup=5, rep=25)` 叠在 `@flyc.jit` 上,首调 bench 所有 config,后续用缓存最优。

---
来源: remote-sync/SKILL.md, flydsl-fp8-gemm-tuning/06-autotune-design.md, flydsl-fp8-gemm-results/SKILL.md, flydsl-fp8-gemm-tuning/07-benchmarking.md, pr-merge-gate/SKILL.md, flydsl-kernel-authoring/SKILL.md, FlyDSL/CLAUDE.md, verify-performance/SKILL.md, optimize-loop.md
