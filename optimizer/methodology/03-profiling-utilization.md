# Profiling 与利用率分析:ISA 权威、rocprofv3 全家桶、regime 分类、ATT stall 根因

> 类别: 方法论 · 主题标签: isa-dump, register-pressure, occupancy, k-loop-stall, rocprofv3, kernel-trace, TFLOPS 测量, kernel 归属, PMC, L2/HBM, LDS-bound, profiling, bottleneck-classify, rocprof-compute, ATT-trace, MFMA-stall, LDS-vs-VGPR, stall-analysis, hotspot

## ISA dump 是寄存器/AGPR/LDS/occupancy 唯一权威:FLYDSL_DUMP_IR + 数指令

### 怎么产出 ISA dump
- env:`FLYDSL_DUMP_IR=1`(必需)、`FLYDSL_DUMP_DIR=/tmp/xx`(可选,不设默认落 `/root/.flydsl/debug/`)、`FLYDSL_RUNTIME_ENABLE_CACHE=0`(避免命中旧缓存)。
- **必须真正 RUN kernel**(`c(*args)`)才触发编译+dump;只 `_compile` 不够。
- 产物路径:`<DUMP_DIR>/kernel_<name>_0/21_final_isa.s`(如 `kernel_dense_tn_0/21_final_isa.s`、`kernel_gemm_0/*_final_isa.s`)。每个 pipeline stage 还产编号 `.mlir`,`21_final_isa.s` 是 lowered 后的最终 ISA。
- `FLYDSL_DEBUG_DUMP_ASM` / `FLYDSL_DUMP_ASM` 不被支持,会报 not-supported —— 只能用 `FLYDSL_DUMP_IR`。
- 相关 env:`FLYDSL_COMPILE_OPT_LEVEL`(默认 2,范围 0-3)、`ARCH`(覆盖架构)、`FLYDSL_DEBUG_ENABLE_DEBUG_INFO`(发 DWARF,验证时查 `final_isa.s` 有无 `.file`/`.loc` directive)。

### 为什么 ISA 是唯一权威(不是 rocprof)
- ISA 元数据是寄存器/LDS/Scratch 占用的**唯一可靠来源**:
  - `.set kernel.num_vgpr` / `.vgpr_count`、`num_agpr` / `.agpr_count`
  - `accum_offset`、`next_free_vgpr`
  - `group_segment_fixed_size`(LDS 字节数)
  - `private_seg_size`(Scratch/spill 字节)、`.vgpr_spill_count`
- rocprof PMC 的 VGPR_Count 误报细节见下方"不可信的计数"一节;扫寄存器/spill 前一律先 dump ISA,别信 PMC。

### 指令直方图(快速体检)
```
grep -c v_mfma 21_final_isa.s          # MFMA 数
grep -c scratch_load 21_final_isa.s    # spill 检测(>0 即 spill)
grep -c s_barrier 21_final_isa.s
grep -c buffer_load / ds_read / ds_write / buffer_store
grep -E "num_vgpr|num_agpr|vgpr_spill" 21_final_isa.s
grep -oE "s_waitcnt.*" 21_final_isa.s | sort | uniq -c | sort -rn   # 数 vmcnt(0)/lgkmcnt(0) drain
```
- 对齐参考实现(aiter / hipBLASLt)目标:**MFMA 数匹配参考、barrier 数 ≤ 参考**。

### 稳态 K-loop stall-free 判据
- 定位 hot loop:label `1:` → `s_cbranch 1b`,即两个 `s_barrier` 之间的循环体。
- stall-free 特征:
  - `v_mfma` 背靠背连发;
  - `buffer_load`(g2s 预取)与 `ds_read`(读 operand)精确塞进 MFMA 间隙;
  - 每迭代只 **1 个 `s_waitcnt vmcnt(0) lgkmcnt(0)`** + **1 个 double-buffer barrier**。
- 满足即 compute 已 **MFMA-bound、零浪费,无 asm 可榨** —— 此时优化方向应转到 occupancy/feed,而非循环体指令调度。

## rocprofv3 --kernel-trace 冷测 TFLOPS:排除 host overhead、认 kernel 归属

### 为什么用 rocprofv3 --kernel-trace
- kernel-only 时间,直接排除 host overhead / autotune 污染,是**最可靠**的 kernel TFLOPS 来源,噪声 **±3T**。
- 可信测量方法排序:
  1. `rocprofv3 --kernel-trace` — 最可靠,kernel-only 排除 host overhead,噪声 **±3T**。
  2. Event 500-sample min/p5/p10/med — 分布完整,噪声 **±10T med**。
  3. Event 100-sample — 日常快速对比,噪声 **±25T med(±0.5%)**。
  4. do_bench — 偏差大,不同场景不可比。

### 冷测取时间(SQLite db 路径)
- 命令:`rocprofv3 --kernel-trace -d /tmp/rp_out -o tr -- python bench.py`
- 结果在 `tr_results.db`(SQLite):
  - `top_kernels` 表读 `total_duration`。
  - `kernels` 表取 **min duration = 稳态单次**。

### 冷测取时间(CSV 路径)
- 命令:`rocprofv3 --kernel-trace --output-format csv -d /tmp/rpf -- python ...`
- 解析 `*_kernel_trace.csv`:`End_Timestamp - Start_Timestamp`(ns);按 `Kernel_Name` 过滤含 `gemm` 的行。
- **TF = 2*M*N*K / dur_ns / 1e3**(等价 `2MNK/dur`)。

### 认 kernel 归属(疑似路由到别的 backend)
- `rocprofv3 --kernel-trace` 看 `top_kernels` 里是谁:
  - `kernel_grouped_*` → flydsl
  - `Cijk_*` → hipBLASLt
  - 其他 → 别的 backend

### kernel 名发现
- 命令:`rocprofv3 --stats --kernel-trace -f csv -o /tmp/discover -- python $TEST_SCRIPT`,读 `/tmp/discover_kernel_stats.csv`。
- FlyDSL kernel 名通常含 `pa_decode` / `kernel_0` / 测试脚本里的函数名。

### GEMM profile 一条龙
- 命令:`rocprofv3 --kernel-trace --stats -f csv -- python test_preshuffle_gemm.py --in_dtype fp8 -M -N -K --tile_m --tile_n --tile_k`。
- 若 GPU 时间 **>1.5× 理论**,转 `/kernel-trace-analysis` 跑 ATT 分析定位瓶颈。

## rocprofv3 PMC counter:L2/HBM 效率、LDS 带宽 vs 延迟、哪些计数不可信

### 命令基本盘
- 新版 rocprofv3 **必须加 `--output-format csv`**,否则输出 `.db`。
- 模板:`rocprofv3 --pmc LDSBankConflict MfmaUtil --output-format csv -d /tmp/rp -- python3 script.py`。
- counter collection 输出在 `<output_directory>/pass_1/<output_file>_counter_collection.csv`。
- 可用 counter 名随 ROCm 版本变(拼写会漂),用 `rocprofv3 --list-avail` 查、以本地输出为准。
- FlyDSL 汇总设施:PMC 文件模板 `turbo/mxfp4_prof_pmc.txt`,汇总 `turbo/prof_summary.py`。

### 关键 counter 家族
| 目的 | counter |
|---|---|
| L2 复用/coalescing | `TCC_HIT_sum`, `TCC_MISS_sum`, `TCC_REQ_sum`(+ TCP/TCC 面板) |
| HBM 读效率 | `TCC_EA0_RDREQ_sum`, `TCC_EA0_RDREQ_32B_sum`, `TCC_EA0_RDREQ_DRAM_sum`, `TCP_TCC_READ_REQ_sum` |
| MFMA 使用/效率 | `SQ_INSTS_MFMA`, `SQ_INSTS_VALU_MFMA_MOPS_*`, `MfmaUtil`(MFMA busy%) |
| VMEM/LDS 带宽浪费 | `SQ_INSTS_VMEM_*`, `SQ_INSTS_LDS`, `SQ_LDS_BANK_CONFLICT` / `LDSBankConflict` |
| Occupancy/资源 | occupancy report, scratch, `hipcc --resource-usage` |

### L2/HBM 三指标三决策(scripts/pmc_l2_analyzer.py)
输入 `pmc_l2` + `pmc_ea` 两个 counter CSV,参数 `--kernel --ideal-gb <每 dispatch GB> --ea-channels 2`。
- **L2 命中率** = TCC_HIT/(TCC_HIT+TCC_MISS) → 有无**时间复用**可挖。
- **32B fraction** = `TCC_EA0_RDREQ_32B / TCC_EA0_RDREQ` → 空间局部性/cache 线浪费。
  - ≈0% = 满 64B cache line、无空间浪费;**高 32B%** 才指向 scatter/misaligned,值得重构。
- **over-fetch** = 实取字节 vs `--ideal-gb` → 有无冗余取数。

### LDS 带宽 bound vs 延迟暴露(必区分)
- 量 `SQ_LDS_IDX_ACTIVE`(LDS 端口忙周期)**:** `SQ_VALU_MFMA_BUSY_CYCLES` 比值。
- 实测 8w wholeloop = **1:9.46**(端口只在 MFMA 忙的 ~10% 活动,≥5× 余量)→ **不是端口带宽 bound**。
- `SQ_WAIT_INST_LDS ≈ SQ_LDS_IDX_ACTIVE` 且占 `SQ_WAIT_ANY` 的 **59%** → 是 **ds_read 延迟暴露**而非带宽打满。
- 长 K 冒烟枪:`SQ_WAIT_INST_LDS/GUI`(LDS-wait 归一化,越低越好)、`MfmaUtil`(MFMA busy%)、`SQ_INSTS_LDS`(=算术强度)、`SQ_INSTS_VMEM`、bank_conflict、`GRBM_GUI_ACTIVE`(cyc/dispatch,时钟无关效率)。

### 不可信的计数(权威判据在别处)
- CSV `Accum_VGPR_Count` **恒报 0**。
- `VGPR_Count` 对 256-VGPR 内核也报 **128 等错值**(如 8-wave wholeloop 内核,真实 num_vgpr=256)。
  ★**2026-07-27 锐化(meta hd64 bwd 实测)**:它报的是**真值的一半**。所以在 occ-2 内核上
  **读到 128 就意味着正好压在悬崖上,读到 132 就已经越界**——实测越界即掉约 **8%**,
  而 **`scratch` 仍是 0(不是 spill)、SNR 仍 bit-identical**,bench 也不报错,**三个常规信号全看不见**。
  ⇒ **任何会动寄存器压力的重构(hoist 地址、双缓冲、融合、加预取),bench 之前先 dump ISA 看 `.vgpr_count`**。
  本例三次失败(−7.9% / −16.5% / 掉占用率)全靠这一条才解释得通,否则只会看到"莫名其妙变慢"。
  - 权威 VGPR/AGPR 必须用 `FLYDSL_DUMP_IR` 的 ISA `num_vgpr/num_agpr`。
- prof_summary 的 "MFMA busy %"(除以 GUI*4)对聚合计数**不成比例(>100%)** → 改用权威派生指标 `MfmaUtil` / `MeanOccupancyPerActiveCU`。
- 查 VGPR 分配的 rocprofv3 SQL(仅供参考,同样不足信):
  ```sql
  SELECT ks.KernelName, ki.arch_vgpr_count, ki.accum_vgpr_count
  FROM rocpd_kernel_dispatch kd
  JOIN rocpd_info_kernel_symbol ks ON kd.kernel_symbol_id=ks.id
  JOIN rocpd_info_kernel ki ON kd.kernel_id=ki.id
  WHERE ks.KernelName LIKE '%target%';
  ```

### PMC 环境彻底不可用的 fallback
- 某些环境 PMC **signal-6 崩**:崩在 torch GPU kernel(randint/contiguous-clone),CPU 生成数据也崩。
- 此时带宽只能**解析估算**:输出字节 / 暴露时间 ÷ HBM 峰值。

## 先 rocprof-compute 认 regime 再选 lever:memory/compute/stall-bound 信号与阈值

### 工具分工:先 rocprof-compute 分类,瓶颈模糊再上 rocprofv3
- **rocprof-compute**:Speed-of-Light、roofline、occupancy、MFMA/memory 面板 → 输出 workload 目录。先用它**快速分类瓶颈**。
- **rocprofv3**:timeline trace、原始 counter、API/kernel overlap、Perfetto(CSV/JSON/PFTrace/rocpd)。瓶颈模糊时才上,看精确 trace 时间 / counter / 多 kernel overlap。
- 命令:
  - 采集 `rocprof-compute profile --name workload_name -- python run_kernel.py`(或 `profile -n <tag> --no-roof`)
  - 分析 `rocprof-compute analyze -p workload_name/<GPU_NAME> --cli`,`<GPU_NAME>` 子目录按检测到的 GPU 自动命名(MI300X / MI350X)
  - 发现 metric `rocprof-compute profile --list-metrics`;counter `rocprofv3 --list-counters`
  - flags 随 ROCm 版本变,scripting 前先 `--help` 确认
- analyze 后 grep:MFMA Util / L2 Cache Hit / Dependency Wait / VMEM Util / Wavefront Occ / Insufficient SIMD VGPR / Insufficient CU LDS / Bank Conflict。
- **先 profile 再调常数**:❌ 别再试 靠实验扫常数(vmcnt_hint/lgkmcnt/barrier_mask 等)——必须先 rocprof-compute + ISA disasm 做理论分析,再决定调什么。扫 tile-blocking 因子(GROUP_M/group_n)时,要测 L2 命中率作机制依据才算 grounded。
- 诊断命令模板:`rocprof-compute profile -n <tag> --no-roof -- python <pmc_run.py>`(pmc_run 跑 kernel 5-10 次)→ `rocprof-compute analyze -p workloads/<tag>/MI* | grep -iE "MFMA Util|L2 Cache Hit|Dependency Wait|VMEM Util|Wavefront Occ|Insufficient SIMD VGPR|Insufficient CU LDS|Bank Conflict"`。

### 第 0 步:先看 GPU 利用率,别急着 micro-tune
- **>60%** 才够 GPU-bound,值得 kernel 级优化。
- **<30%** 多半是 launch/CPU/同步/调度问题 → **先看 timeline 和 host 侧 gap**,不要 micro-tune。
- **30-60%** 混合,交叉核对 trace overlap 和 SoL 面板。
- GPU 大部分空闲时做 micro-tuning 是浪费。

### 三类 regime 信号 + 对应 lever

| Regime | 信号 | Lever |
|---|---|---|
| **Memory-bound** | HBM BW 近 roofline/peak + MFMA-issue ratio 低 + 大 K 小 M*N;L2 命中差;LDS bank conflict 或 VMEM latency counter 高 | bigger tiles / async G2S / L2-XCD locality / split-K / preshuffle |
| **Compute-bound** | MFMA-issue ratio 高但吞吐低 + BW slack;指令混合应以 MFMA 为主但 issue 效率差;VALU 相对 GEMM 意图偏高 | wider-K MFMA atom / 减少动态 MFMA 数 / accumulator 放 AGPR / 抬 occupancy |
| **Stall-concurrency-bound** | HBM 和 MFMA **都**低于 roofline 但 GPU 忙;occupancy 因 VGPR/LDS/barrier/wave-limit 压力低;高 s_barrier/s_waitcnt;scratch/spill 非零 | LDS ping-pong overlap / sched hints / bank-conflict swizzle / 修 prefetch depth |

### 健康阈值(起点,非定律)

| 指标 | 好 | 需关注 |
|---|---|---|
| MFMA-issue ratio(compute) | >40%(compute 判据);面板值 >50-70% | <40% |
| HBM BW(memory) | >60% | <30% |
| Occupancy | >50%;≥2 waves/SIMD | <25% |
| LDS bank conflict 比 | <5% | ≥5% |
| arch-VGPR | ≤128 | — |
| Scratch/spill | =0 | 任何非零 |

### fp8 TN big-shape 实测判据(证据)
- **VMEM Util 低(~3%)** = 非 memory/带宽 bound。
- **Dep-Wait 高 + MFMA Util ~34-40% + Occupancy ~1 WG/CU** = latency-bound。
- **L2 Cache Hit**:square/big-K ~66% vs big-N ~51% → L2 复用是**大 N 的关键指标**。

### 下探到 per-line:rocprofv3 ATT(瓶颈确认后)
- FlyDSL kernel 用 rocprofv3 **Advanced Thread Trace (ATT)** 把 per-instruction stall 映射到源码行。
- 采集:`advanced_thread_trace:true` + 单个 target CU + 跳 warmup 的 iteration range + `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`。
- 主产物 `code.json`(per-instruction asm / source-loc / total / stall / issue cycles)。
- **按源码行聚合 stall cycles**,按 opcode 前缀分类:VMEM-load、VMEM-wait(s_waitcnt vmcnt)、LDS/SMEM-wait(s_waitcnt lgkmcnt)、barrier(s_barrier)、MFMA/FMA(v_mfma_*)、LDS(ds_read/ds_write)。
- 映射方向:LDS stall → bank-conflict swizzle;VMEM-wait → 更深 prefetch / async G2S;barrier → ping-pong overlap。
- 高MFMA低TFLOPS判据见下方"ATT trace 做 stall 根因"一节。

### subtractive/HALF 编译期探针:PMC/ATT 不可用时的 stall 归因 + ★「上界≠可达」铁律
- **技法**:PMC counter 被同集群别的 campaign 占锁、或无 ATT decoder .so 时,用**编译期门控探针**隔离每项成本:`skip_X`(跳过某段计算/store/barrier,故意破坏正确性只测时间)、`HALF_X`(跳一半 MFMA)、`REUSE_X`(某读塌成 1 次)。event-median ×40-60 隔离计时,逐项从 wall 里减出各成本;`skip_both`(同时跳两大计算)剩下的 residual = occ-1 暴露的结构延迟(HBM gather + 循环调度 + 依赖链串行,无第二 wave 掩盖)。
- **★★ 铁律:subtractive/HALF 探针天花板、roofline 峰值率、纸面 op-count 分析给的都是「上界」,不是「可达值」。判负/判正前必须 edit→bench 真实现。** 反复踩证(dsv4 sparse-MLA,见 pitfalls/12):
  - HALF_PV 探针「−8.7% 假想天花板」→ 真 K=32-PV **净负 −11%**(2-tile 批打断 QK→softmax→PV 交织,批结构本身是杀手,非 k 维)。
  - SKIPST「store 占 wall 30-47%」→ 真 DMA / register-transpose / query-blocking **全净负**:store 是「跨-wave 数据共享 + register-prefetch 隐藏 HBM」的必需机制,不是可省浪费。
  - 「gfx950 k=32 双倍 bf16 率」roofline 头 → 落地被数据布局(tr16 转置读 =4bf16/lane=k16)锁死到「必须 2-tile 批」,批税吃光收益。
  - 去-padding「代理测量」判 DMA 负 → 其实测的是 tr16 bank-conflict,不是 DMA 本身;真 per-rank-DMA 保 padding 从没被那个代理覆盖(**代理≠真实现**)。
  - **∴ 探针只告诉你「某成本是否在关键路径」(值不值得投入去攻),不告诉你「去掉它是否可实现」。可达头只有真实现 + bench 能定;纸面 op-count 预判(如 bpermute≫store)与探针天花板一样常错。**

## ATT trace 做 stall 根因:MFMA operand bubble 记在 MFMA 头上而非 waitcnt

### 核心洞察:operand bubble 记在 MFMA 头上,不记在 waitcnt 上
- ATT trace 里 MFMA **operand-not-ready** 的等待记在 MFMA 指令头上(计入 MFMA stall%),**不**记在 `s_waitcnt` 上。
- 典型症候:VMEM-wait 只 **1.8%** 但 MFMA stall **89.5%** —— 内存延迟是**隐性**地表现为 MFMA operand bubble,而非显式 waitcnt。别被低 VMEM-wait 骗了。
- 减法探针(减 g2s / ds_read)暴露的成本就是这些 bubble,两视角一致(g2s+ds_read 未就绪 → MFMA 等 operand)。
- ATT 可区分 MFMA stall 到底是等 **operand bubble**(g2s+ds_read 没就绪)还是等**显式 waitcnt**。

### 为什么 ATT 是权威工具
- 仅靠 ISA 扫描(code.json,不看 CSV)会**误报 VGPR-bound**——把 `v_mfma` 里的 `a[...]` 寄存器引用误判成 agpr-form;`out_kernel_trace.csv` 才是权威(Accum_VGPR_Count=0、vgpr-form、combined≈216),揭示真实瓶颈其实是 **LDS-bound**。ATT(code.json)+ CSV 合参才能权威定位 stall 根因。
- ATT 把 per-instruction stall 映射到源码行,是定位 MFMA stall 根因的权威工具。

### 采集流程
- 命令:`rocprofv3 -i input.yaml -- python driver.py`(input.yaml = kernel regex + att 配置)。
- 必设 `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` —— 启用 HSACO 里的 DWARF debug info,才能在 code.json 得到 源码 file:line→汇编 映射;没设则 source_loc 全空。
- 需 decoder 库 `/opt/rocm/lib/librocprof-trace-decoder.so`。
- 输出目录 `ui_output_agent_<PID>_dispatch_<N>`(取最新)。
- 后处理:跑 hotspot_analyzer 看 topk stall 分布。
- 复现设施(容器 turbo/):`_trace_kernel.py` + `_trace_input.yaml` + `_hotspot_analyzer.py`,需装 decoder 库。

### input.yaml 关键项
| 项 | 值 | 作用 |
|---|---|---|
| `kernel_include_regex` | 精确名/正则 | 只抓目标 kernel |
| `kernel_iteration_range` | `"[1,[2-4]]"` | 跳过 warmup 迭代 0,只抓 2-4 |
| `advanced_thread_trace` | `true` | 启用 ATT |
| `att_target_cu` | `1` | 单 CU,保持输出可控 |
| `att_shader_engine_mask` | `"0xf"` | |
| `att_simd_select` | `"0xf"` | |
| `att_buffer_size` | `"0x6000000"`(96MB/SE) | 被截断则升到 `0xC000000`=192MB |

### 产物与验证
- 主产物 `code.json`:per-instruction asm / source-loc / total / stall / issue cycles。
- 输出目录结构/下载细节(ui_output_agent_* 命名、含哪些文件、out_kernel_trace.csv)见下方"code.json 逐指令 stall 分析"一节。

### stall 分类学
- stall 类型分类表(VMEM-load/VMEM-wait/LDS-SMEM-wait/barrier/MFMA/LDS 及优化方向)见下方"code.json 逐指令 stall 分析"一节。
- 判据:**high MFMA + low TFLOPS** → 是 barrier/`s_waitcnt` stall,不是 scheduler 能救的。

## code.json 逐指令 stall 分析:hotspot_analyzer 按源码行聚合、stall 类型分类

### code.json 每行格式(10 列)
- 列布局:`[asm, _, pc_index, source_loc, codeobj, pc_addr, exec_count, total_cycles, stall_cycles, issue_cycles]`(等价命名:`[ISA, _, LineNum, Source, Codeobj, Vaddr, Hit, Latency, Stall, Idle]`)。
- **col[8] stall_cycles = 首要热点指标**(按此降序找最热指令)。
- col[7] total_cycles:全 wave 总周期。
- col[6] exec_count(Hit):执行该指令的 wave-thread 数。
- col[3] source_loc:`file:LINE`,经 snapshots.json 解析虚拟路径。

### stall 类型分类(按指令模式/waitcnt 归类)
| 类型 | 识别 | 含义 |
|---|---|---|
| VMEM-load | buffer_load/global_load 本身 stall | VMEM 队列满,或无 compute 掩盖 |
| VMEM-wait | s_waitcnt **vmcnt** | 等 load 完成 |
| LDS/SMEM-wait | s_waitcnt **lgkmcnt** | 等 LDS/SMEM |
| barrier | s_barrier | 最慢 wave 主导 |
| MFMA/FMA | v_mfma | RAW 依赖链 |
| LDS | ds_read/ds_write | LDS 延迟 |

### hotspot_analyzer.py 用法
- 常规:`python hotspot_analyzer.py <dir> --topk 15 --mode both`。
- 带源码上下文(最利优化):`--topk 5 --mode src --detail --context 4`。
- 指令级:`--mode asm --topk 20`。
- 全程序化(hotspot_analyzer.py + code.json),不用 GUI。
- **自动检测架构**:见 gfx950 专属指令(`v_mfma_scale_f32_*`、`v_mfma_f32_16x16x128_*`、`v_mfma_f32_32x32x64_*`)=CDNA4;缺失=CDNA3。

### LDS 瓶颈专项诊断
- 筛 `ds_` 开头 或 含 `lgkmcnt` 且 stall>0 的指令;汇总 LDS stall 占总 stall 比例。
- **>15% 值得优化**;按 stall 降序看最热 LDS 指令。

### 下载/验证 ATT trace
- 输出目录名:`ui_output_agent_<PID>_dispatch_<N>`,取最新。
- 目录含:code.json / occupancy.json / filenames.json / wstates*.json / se*_*.json;**另需单独下 out_kernel_trace.csv**(timing + VGPR)。
- 验证下载:读 code.json 数指令数 + 有源映射的指令占比。

### ATT trace 瓶颈对照表 → 改进方向
| 现象 | 根因 | 改进 |
|---|---|---|
| MFMA 前高 s_waitcnt vmcnt(0) | global load 延迟暴露 | 改进预取 / 加大 tile_k |
| 高 lgkmcnt(0) | LDS 延迟暴露 | 增大 write-read 距离 / 查 bank 冲突 |
| 高 s_barrier | 同步开销 | 查 LDS stage / 减 barrier |
| MFMA 利用率 <50% | memory-bound | — |
| MFMA 间多 s_nop | 流水气泡 | 交错 load / 调 scheduler |
| 高 cycle buffer_load | TA 阻塞 | 减并发 load / 查合并 |

---
来源: flydsl-fp8-gemm-tuning/SKILL.md, 10-8wave-scvgpr.md, flydsl-kernel-authoring/SKILL.md, agpr_rawasm_progress.md, project_mxfp4_epilogue_store.md, gemm-optimization/SKILL.md, 07-benchmarking.md, 07-benchmark.md, capture-kernel-trace/SKILL.md, 10-grouped-wgrad-4wave-3buf.md, kernel-trace-analysis/SKILL.md, tool-rocprof/SKILL.md, diag_4w_vs_8w.md, prefetch-data-load/SKILL.md, gemm/overview.md, programming-model.md, 08-att-root-cause.md, project_mxfp4_k28672_ceiling.md, lds-optimization/SKILL.md
