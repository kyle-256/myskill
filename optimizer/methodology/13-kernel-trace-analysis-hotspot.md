# code.json 逐指令 stall 分析:hotspot_analyzer 按源码行聚合、stall 类型分类

> 类别: 方法论 · 主题标签: rocprofv3, ATT-trace, stall-analysis, hotspot

## code.json 每行格式(10 列)
- 列布局:`[asm, _, pc_index, source_loc, codeobj, pc_addr, exec_count, total_cycles, stall_cycles, issue_cycles]`(等价命名:`[ISA, _, LineNum, Source, Codeobj, Vaddr, Hit, Latency, Stall, Idle]`)。
- **col[8] stall_cycles = 首要热点指标**(按此降序找最热指令)。
- col[7] total_cycles:全 wave 总周期。
- col[6] exec_count(Hit):执行该指令的 wave-thread 数。
- col[3] source_loc:`file:LINE`,经 snapshots.json 解析虚拟路径。

## stall 类型分类(按指令模式/waitcnt 归类)
| 类型 | 识别 | 含义 |
|---|---|---|
| VMEM-load | buffer_load/global_load 本身 stall | VMEM 队列满,或无 compute 掩盖 |
| VMEM-wait | s_waitcnt **vmcnt** | 等 load 完成 |
| LDS/SMEM-wait | s_waitcnt **lgkmcnt** | 等 LDS/SMEM |
| barrier | s_barrier | 最慢 wave 主导 |
| MFMA/FMA | v_mfma | RAW 依赖链 |
| LDS | ds_read/ds_write | LDS 延迟 |

## hotspot_analyzer.py 用法
- 常规:`python hotspot_analyzer.py <dir> --topk 15 --mode both`。
- 带源码上下文(最利优化):`--topk 5 --mode src --detail --context 4`。
- 指令级:`--mode asm --topk 20`。
- 全程序化(hotspot_analyzer.py + code.json),不用 GUI。
- **自动检测架构**:见 gfx950 专属指令(`v_mfma_scale_f32_*`、`v_mfma_f32_16x16x128_*`、`v_mfma_f32_32x32x64_*`)=CDNA4;缺失=CDNA3。

## LDS 瓶颈专项诊断
- 筛 `ds_` 开头 或 含 `lgkmcnt` 且 stall>0 的指令;汇总 LDS stall 占总 stall 比例。
- **>15% 值得优化**;按 stall 降序看最热 LDS 指令。

## 下载/验证 ATT trace
- 输出目录名:`ui_output_agent_<PID>_dispatch_<N>`,取最新。
- 目录含:code.json / occupancy.json / filenames.json / wstates*.json / se*_*.json;**另需单独下 out_kernel_trace.csv**(timing + VGPR)。
- 验证下载:读 code.json 数指令数 + 有源映射的指令占比。

## ATT trace 瓶颈对照表 → 改进方向
| 现象 | 根因 | 改进 |
|---|---|---|
| MFMA 前高 s_waitcnt vmcnt(0) | global load 延迟暴露 | 改进预取 / 加大 tile_k |
| 高 lgkmcnt(0) | LDS 延迟暴露 | 增大 write-read 距离 / 查 bank 冲突 |
| 高 s_barrier | 同步开销 | 查 LDS stage / 减 barrier |
| MFMA 利用率 <50% | memory-bound | — |
| MFMA 间多 s_nop | 流水气泡 | 交错 load / 调 scheduler |
| 高 cycle buffer_load | TA 阻塞 | 减并发 load / 查合并 |

---
来源: kernel-trace-analysis/SKILL.md, lds-optimization/SKILL.md, capture-kernel-trace/SKILL.md, gemm-optimization/SKILL.md
