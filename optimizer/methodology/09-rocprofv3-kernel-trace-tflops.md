# rocprofv3 --kernel-trace 冷测 TFLOPS:排除 host overhead、认 kernel 归属

> 类别: 方法论 · 主题标签: rocprofv3, kernel-trace, TFLOPS 测量, kernel 归属

## 为什么用 rocprofv3 --kernel-trace
- kernel-only 时间,直接排除 host overhead / autotune 污染,是**最可靠**的 kernel TFLOPS 来源,噪声 **±3T**。
- 可信测量方法排序:
  1. `rocprofv3 --kernel-trace` — 最可靠,kernel-only 排除 host overhead,噪声 **±3T**。
  2. Event 500-sample min/p5/p10/med — 分布完整,噪声 **±10T med**。
  3. Event 100-sample — 日常快速对比,噪声 **±25T med(±0.5%)**。
  4. do_bench — 偏差大,不同场景不可比。

## 冷测取时间(SQLite db 路径)
- 命令:`rocprofv3 --kernel-trace -d /tmp/rp_out -o tr -- python bench.py`
- 结果在 `tr_results.db`(SQLite):
  - `top_kernels` 表读 `total_duration`。
  - `kernels` 表取 **min duration = 稳态单次**。

## 冷测取时间(CSV 路径)
- 命令:`rocprofv3 --kernel-trace --output-format csv -d /tmp/rpf -- python ...`
- 解析 `*_kernel_trace.csv`:`End_Timestamp - Start_Timestamp`(ns);按 `Kernel_Name` 过滤含 `gemm` 的行。
- **TF = 2*M*N*K / dur_ns / 1e3**(等价 `2MNK/dur`)。

## 认 kernel 归属(疑似路由到别的 backend)
- `rocprofv3 --kernel-trace` 看 `top_kernels` 里是谁:
  - `kernel_grouped_*` → flydsl
  - `Cijk_*` → hipBLASLt
  - 其他 → 别的 backend

## kernel 名发现
- 命令:`rocprofv3 --stats --kernel-trace -f csv -o /tmp/discover -- python $TEST_SCRIPT`,读 `/tmp/discover_kernel_stats.csv`。
- FlyDSL kernel 名通常含 `pa_decode` / `kernel_0` / 测试脚本里的函数名。

## GEMM profile 一条龙
- 命令:`rocprofv3 --kernel-trace --stats -f csv -- python test_preshuffle_gemm.py --in_dtype fp8 -M -N -K --tile_m --tile_n --tile_k`。
- 若 GPU 时间 **>1.5× 理论**,转 `/kernel-trace-analysis` 跑 ATT 分析定位瓶颈。

---
来源: 07-benchmarking.md, 07-benchmark.md, capture-kernel-trace/SKILL.md, gemm-optimization/SKILL.md
