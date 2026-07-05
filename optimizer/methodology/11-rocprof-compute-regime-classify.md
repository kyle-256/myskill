# 先 rocprof-compute 认 regime 再选 lever:memory/compute/stall-bound 信号与阈值

> 类别: 方法论 · 主题标签: profiling, bottleneck-classify, occupancy, rocprof-compute

## 工具分工:先 rocprof-compute 分类,瓶颈模糊再上 rocprofv3
- **rocprof-compute**:Speed-of-Light、roofline、occupancy、MFMA/memory 面板 → 输出 workload 目录。先用它**快速分类瓶颈**。
- **rocprofv3**:timeline trace、原始 counter、API/kernel overlap、Perfetto(CSV/JSON/PFTrace/rocpd)。瓶颈模糊时才上,看精确 trace 时间 / counter / 多 kernel overlap。
- 命令:
  - 采集 `rocprof-compute profile --name workload_name -- python run_kernel.py`(或 `profile -n <tag> --no-roof`)
  - 分析 `rocprof-compute analyze -p workload_name/<GPU_NAME> --cli`,`<GPU_NAME>` 子目录按检测到的 GPU 自动命名(MI300X / MI350X)
  - 发现 metric `rocprof-compute profile --list-metrics`;counter `rocprofv3 --list-counters`
  - flags 随 ROCm 版本变,scripting 前先 `--help` 确认
- analyze 后 grep:MFMA Util / L2 Cache Hit / Dependency Wait / VMEM Util / Wavefront Occ / Insufficient SIMD VGPR / Insufficient CU LDS / Bank Conflict。

## 第 0 步:先看 GPU 利用率,别急着 micro-tune
- **>60%** 才够 GPU-bound,值得 kernel 级优化。
- **<30%** 多半是 launch/CPU/同步/调度问题 → **先看 timeline 和 host 侧 gap**,不要 micro-tune。
- **30-60%** 混合,交叉核对 trace overlap 和 SoL 面板。
- GPU 大部分空闲时做 micro-tuning 是浪费。

## 三类 regime 信号 + 对应 lever

| Regime | 信号 | Lever |
|---|---|---|
| **Memory-bound** | HBM BW 近 roofline/peak + MFMA-issue ratio 低 + 大 K 小 M*N;L2 命中差;LDS bank conflict 或 VMEM latency counter 高 | bigger tiles / async G2S / L2-XCD locality / split-K / preshuffle |
| **Compute-bound** | MFMA-issue ratio 高但吞吐低 + BW slack;指令混合应以 MFMA 为主但 issue 效率差;VALU 相对 GEMM 意图偏高 | wider-K MFMA atom / 减少动态 MFMA 数 / accumulator 放 AGPR / 抬 occupancy |
| **Stall-concurrency-bound** | HBM 和 MFMA **都**低于 roofline 但 GPU 忙;occupancy 因 VGPR/LDS/barrier/wave-limit 压力低;高 s_barrier/s_waitcnt;scratch/spill 非零 | LDS ping-pong overlap / sched hints / bank-conflict swizzle / 修 prefetch depth |

## 健康阈值(起点,非定律)

| 指标 | 好 | 需关注 |
|---|---|---|
| MFMA-issue ratio(compute) | >40%(compute 判据);面板值 >50-70% | <40% |
| HBM BW(memory) | >60% | <30% |
| Occupancy | >50%;≥2 waves/SIMD | <25% |
| LDS bank conflict 比 | <5% | ≥5% |
| arch-VGPR | ≤128 | — |
| Scratch/spill | =0 | 任何非零 |

## fp8 TN big-shape 实测判据(证据)
- **VMEM Util 低(~3%)** = 非 memory/带宽 bound。
- **Dep-Wait 高 + MFMA Util ~34-40% + Occupancy ~1 WG/CU** = latency-bound。
- **L2 Cache Hit**:square/big-K ~66% vs big-N ~51% → L2 复用是**大 N 的关键指标**。

## 下探到 per-line:rocprofv3 ATT(瓶颈确认后)
- FlyDSL kernel 用 rocprofv3 **Advanced Thread Trace (ATT)** 把 per-instruction stall 映射到源码行。
- 采集:`advanced_thread_trace:true` + 单个 target CU + 跳 warmup 的 iteration range + `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`。
- 主产物 `code.json`(per-instruction asm / source-loc / total / stall / issue cycles)。
- **按源码行聚合 stall cycles**,按 opcode 前缀分类:VMEM-load、VMEM-wait(s_waitcnt vmcnt)、LDS/SMEM-wait(s_waitcnt lgkmcnt)、barrier(s_barrier)、MFMA/FMA(v_mfma_*)、LDS(ds_read/ds_write)。
- 映射方向:LDS stall → bank-conflict swizzle;VMEM-wait → 更深 prefetch / async G2S;barrier → ping-pong overlap;**高 MFMA + 低 TFLOPS → 是 barrier/s_waitcnt stall,不是 scheduler 能救的**。

---
来源: flydsl-fp8-gemm-tuning/SKILL.md, gemm/overview.md, programming-model.md, tool-rocprof/SKILL.md
