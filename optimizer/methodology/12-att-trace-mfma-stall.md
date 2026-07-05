# ATT trace 做 stall 根因:MFMA operand bubble 记在 MFMA 头上而非 waitcnt

> 类别: 方法论 · 主题标签: ATT-trace, MFMA-stall, rocprofv3, LDS-vs-VGPR

## 核心洞察:operand bubble 记在 MFMA 头上,不记在 waitcnt 上
- ATT trace 里 MFMA **operand-not-ready** 的等待记在 MFMA 指令头上(计入 MFMA stall%),**不**记在 `s_waitcnt` 上。
- 典型症候:VMEM-wait 只 **1.8%** 但 MFMA stall **89.5%** —— 内存延迟是**隐性**地表现为 MFMA operand bubble,而非显式 waitcnt。别被低 VMEM-wait 骗了。
- 减法探针(减 g2s / ds_read)暴露的成本就是这些 bubble,两视角一致(g2s+ds_read 未就绪 → MFMA 等 operand)。
- ATT 可区分 MFMA stall 到底是等 **operand bubble**(g2s+ds_read 没就绪)还是等**显式 waitcnt**。

## 为什么 ATT 是权威工具
- rocprofv3 汇总 / out_kernel_trace.csv 会**误报 VGPR-bound**;ATT+CSV 才能权威分辨到底是 **LDS-bound** 还是 **VGPR-bound**。
- ATT 把 per-instruction stall 映射到源码行,是定位 MFMA stall 根因的权威工具。

## 采集流程
- 命令:`rocprofv3 -i input.yaml -- python driver.py`(input.yaml = kernel regex + att 配置)。
- 必设 `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` —— 启用 HSACO 里的 DWARF debug info,才能在 code.json 得到 源码 file:line→汇编 映射;没设则 source_loc 全空。
- 需 decoder 库 `/opt/rocm/lib/librocprof-trace-decoder.so`。
- 输出目录 `ui_output_agent_<PID>_dispatch_<N>`(取最新)。
- 后处理:跑 hotspot_analyzer 看 topk stall 分布。
- 复现设施(容器 turbo/):`_trace_kernel.py` + `_trace_input.yaml` + `_hotspot_analyzer.py`,需装 decoder 库。

## input.yaml 关键项
| 项 | 值 | 作用 |
|---|---|---|
| `kernel_include_regex` | 精确名/正则 | 只抓目标 kernel |
| `kernel_iteration_range` | `"[1,[2-4]]"` | 跳过 warmup 迭代 0,只抓 2-4 |
| `advanced_thread_trace` | `true` | 启用 ATT |
| `att_target_cu` | `1` | 单 CU,保持输出可控 |
| `att_shader_engine_mask` | `"0xf"` | |
| `att_simd_select` | `"0xf"` | |
| `att_buffer_size` | `"0x6000000"`(96MB/SE) | 被截断则升到 `0xC000000`=192MB |

## 产物与验证
- 目录含:`code.json` / `occupancy.json` / `filenames.json` / `wstates*.json` / `se*_*.json`。
- 主产物 `code.json`:per-instruction asm / source-loc / total / stall / issue cycles。
- 另需单独下 `out_kernel_trace.csv`(timing + VGPR 信息)。
- 验证下载:读 code.json 数指令数、有源映射的指令占比。

## stall 分类学(按 source line 聚合 stall cycles,按 opcode 前缀分类)
| 类别 | opcode | 优化方向 |
|---|---|---|
| VMEM-load | | |
| VMEM-wait | `s_waitcnt vmcnt` | 更深预取 / async G2S |
| LDS/SMEM-wait | `s_waitcnt lgkmcnt` | |
| barrier | `s_barrier` | ping-pong overlap |
| MFMA/FMA | `v_mfma_*` | (operand bubble,见上) |
| LDS | `ds_read`/`ds_write` | bank-conflict swizzle |
- 判据:**high MFMA + low TFLOPS** → 是 barrier/`s_waitcnt` stall,不是 scheduler 能救的。

---
来源: 08-att-root-cause.md, capture-kernel-trace/SKILL.md, project_mxfp4_k28672_ceiling.md, programming-model.md
