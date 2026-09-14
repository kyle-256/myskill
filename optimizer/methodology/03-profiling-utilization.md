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

### ★ 指令足迹几乎不定价,`.vgpr_spill_count` 才定价(occ=1 whole-loop 核)

ISA 行数/`v_mfma` 总数是**静态**量,而热路径一次只走其中一个 body。把「这颗核 emit 了 N 份重复
body」当成一个可回收的池子之前,先按下面的顺序做,否则会把分配器的抖动记成足迹的收益:

1. **先量,再改。** 用编译期开关把要摘的那份 body 关掉(保留它的 decode),做**固定-config 回文**
   A/B,并带一条逐位相同的不变臂标定槽位偏置。
2. **两边都 dump ISA**,同时记 `v_mfma` / 行数 / `.vgpr_count` / **`.vgpr_spill_count`**。
3. **wall 差先减掉 spill 那一项再归因给足迹。**

实测标定(gpt-oss fp8 per-tensor wgrad,gfx950,4-wave,occ=1,512 VGPR 吃满,两个投影):

| 量 | gate_up 4 body → 2 | down 4 body → 2 |
|---|---|---|
| `v_mfma` | 2464 → **1232** | 2112 → **1056** |
| ISA 行数 | 21103 → 10796 | 18988 → 9766 |
| `.vgpr_count` | 512 → 512 | 512 → 512 |
| `.vgpr_spill_count` | 0 → 0 | **0 → 1** |
| wall(回文,两臂) | **+0.23% / +0.15%**(7/7 draw) | **−0.41% / −0.30%**(0/7 draw) |
| 不变臂(逐位相同的重建) | −0.03% | +0.11% |

读法:**指令足迹减半只值 +0.2%**(gate_up 那一侧,足迹是唯一变量);**一个 spill 槽值 ≈ −0.45%**
(down 那一侧,足迹同样减半却净亏)。⇒ 「删掉死 body 缩小 ISA」不是杠杆;**「别让分配器多出一个
spill 槽」才是**,而且它贵到足以单独解释一个改动的失败签名。⚠ 反直觉方向也成立:**减少代码会
引入 spill** —— 分配器的输入变了,不存在"代码更少所以寄存器更松"。

判「某条派发分支静态死了、可以不 emit」时,枚举**所有**会跳进来的原因,而不只是命名最显眼的那个
(此核 head id 既做 split-K 的 cut,也做深组 tile 的 promote,而 plain 路径把 promote 的 tile 退掉了
⇒ 摘 body = 丢 tile)。若这些条件是**运行期**的(如偏移表 / token 分布),静态证不掉;此时仍可造一条
「只在本 harness 抽样分布下恒假」的**探针臂**来定价(要能写出硬上界,例如 `0.2+0.8*rand` 归一后
G=24 单组上界 `1.0/(1.0+23*0.2)=17.9%` < 门槛 25%,并用 sha 逐位核对),但它**只能量,不能发**。

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

### ★ wave-cycle 预算：把「卡在 barrier/waitcnt」与「想发但管道忙」分开（gfx950 实测配方）
- 三个 counter 恰好把 `SQ_WAVE_CYCLES` 分成 **100%**（全部 quad-cycle，×4 换算成周期）：
  `SQ_ACTIVE_INST_ANY`（正在发射）+ `SQ_WAIT_INST_ANY`（想发但发不出）+ `SQ_WAIT_ANY`（waitcnt/barrier）。
- **只看 `SQ_WAIT_ANY` 会误判**：MFMA-bound 循环里 `SQ_WAIT_INST_ANY` 高是**健康态**（自己的 MFMA 还占着管道，
  sibling 也发不进去），真正的空转来源是 `SQ_WAIT_ANY`。实测一个 8-wave/2-waves-per-SIMD 的 mxfp8 GEMM 主环：
  24.2% / 45.2% / **30.6%**，而管道空转（20% of 主环）≈ 两个 sibling wave 的 `SQ_WAIT_ANY` 交叠（~70%）
  ⇒ 结论是「barrier convoy 偏斜」，而不是 LDS/VMEM 延迟，于是 barrier 数与相位结构成为唯一相关的轴。
- 判据：把 `SQ_WAIT_ANY / SQ_WAVE_CYCLES` 与 `MFMA_BUSY /（SIMD 数 × 时长 × 实测时钟）` 一起读；
  两者之差就是「可归因于 rendezvous」的空转。改动后若 `WAIT_ANY` 占比上升而 MFMA busy 不变，必然更慢
  （实测 30.6%→34.2% 对应 −2%；另一次 **31.7%→50.0% 对应 −26%**，campaign 20260904_151102 r13 的波格双射）。
  ⚠ **符号可靠、斜率不可外推**：两个标定点是 −0.56%/pp 与 −1.42%/pp，差 2.5 倍 ⇒ 用它判方向，别用它折算幅度。
- ★★★ **这套预算是「访存/LDS 计数器全部逐位相同」那类改动的唯一定位手段**。r13 的波格双射把 wall 打掉
  26%，而 `TCP_TCC_READ_REQ`/`TCP_TCC_WRITE_REQ`/`TCC_EA0_WRREQ`（含 64B 与 DRAM 拆分）/`TCC_HIT`/`TCC_MISS`/
  `TCC_EA0_RDREQ`/`SQ_INSTS_LDS`/`SQ_LDS_IDX_ACTIVE`/`SQ_LDS_BANK_CONFLICT`(=0)/`SQ_INSTS_MFMA`/
  `SQ_VALU_MFMA_BUSY_CYCLES` **全部逐位相同**（同一份 kernel、同一组请求、同一份 `|C|`）。
  ⇒ 只要候选是「同样的字节、同样的指令、换个分工」，就**不要**花时间扫访存计数器，直接上三项预算。
- ★★ **`SQ_ACTIVE_INST_ANY / SQ_WAVE_CYCLES` 是「减指令条数」整族候选的立项门**：本核实测 **19.2%**
  （与 00-decision-index 第 54 行独立记下的 20.8% 吻合）⇒ 发射槽有 5× 余量，任何"少发 N 条指令"的
  候选期望值是 0，而它的代价（store 条数、fragment 几何、波格）是实打实的。这一条一次性解释了
  32x32x64 原子为什么在四轮里每次 ISA 门全绿、wall 全负。
  - ★★★ **⚠ 这道门的阈值是 occupancy 的函数，符号会反 —— 19.2% 是 2 waves/SIMD 上的值**
    （2026-09-11，mxfp4 dense NT，gfx950 occ=**1**，campaign 20260910_041317 REPLAN 实测）：
    同一个比值在 occ=1 的核上是 **56.70%**（FC1_fwd）/ **59.48%**（out_proj_wgrad），是参照核的 **3 倍**，
    **发射槽没有余量，「减指令条数」整族在这里是头号杠杆**。机制：occ=1 时没有 sibling wave 去填
    发射空档，同一个 wave 的每一条非 MFMA 指令都直接落在自己的 MFMA 影子外面。
    ⇒ **用这道门之前先报 occupancy**；`MeanOccupancyPerActiveCU = 1.000` 的核不适用 19.2% 那条结论。
    - ★★★ **⚠ 这道门开出来的「删指令」族里,`s_setprio` / `sched_barrier` / `iglp_opt` /
      `sched_group_barrier` 这类**区域标记**必须先剔除 —— 它们的条数收益是假的。** 同一个 occ=1 核
      (`ACTIVE_INST_ANY` 50.05%,gpt-oss D64 fused bwd)删掉 GEMM2 的 4 个 `s_setprio`:指令 4024→3959
      (−65,正是这道门最看好的方向),LLVM 却把 GEMM2 与邻居合并成一个调度区域,回来的排期
      `inloop_lgkm0` **44→88**、`exposed` 4042→5675、`trip` +6.7%,wall **+4.678%(区间不相交)**,
      并且破了 dS 栅栏的竞态(`snr` 读出 `52.3 nan nan`)。⇒ 立项清单第一步是**分类**:这条指令是
      在算东西,还是在给调度器画线?画线的那类只能整段重测,不能按条数定价。
  - ★★★★★ **occ=1 的 MFMA 核有一个两点可标定的闭式模型：`duty = 16 / (16 + c·n)`**
    （`16` = 该 MFMA 原子的管线拍数，PMC 实测 `SQ_VALU_MFMA_BUSY_CYCLES / SQ_INSTS_MFMA`；
    `n` = 每条 MFMA 伴随的**非 MFMA** 指令条数；`duty` = `TF/s ÷ (FLOP/cyc/SIMD × #SIMD × 实测 sclk)`）：

    | 形状 | LDS | SALU | VMEM | VALU(扣 MFMA) | **n** | 实测 duty | 反解 **c** |
    |---|---|---|---|---|---|---|---|
    | FC1_fwd 32768×28672×4096 | 0.2579 | 0.2140 | 0.1720 | 0.2419 | 0.8858 | 68.05% | **8.481** |
    | out_proj_wgrad 4096²×32768 | 0.2509 | 0.1908 | 0.1445 | 0.0352 | 0.6214 | 75.12% | **8.528** |

    两行独立标定一致到 **0.6%**。`c ≈ 8.5` ≈ 4 拍发射（64-lane wave / 16-wide SIMD）+ 4.5 拍依赖/仲裁，
    与同核 `SQ_WAIT_INST_ANY 32%`、`SQ_VALU_MFMA_COEXEC_CYCLES / MFMA_BUSY = **2.07%**` 自洽。
    ⇒ 用法：**每 MFMA 少发 0.1 条指令 ≈ +3.7~4.7% wall**；候选排序直接比 Δn，不必先跑 bench。
    ⚠ 两点标定，**符号与量级可用，斜率别外推到 n 变化 >2× 的改动上**。
    - ★★ **第三个标定点(n 大一倍以上时 `c` 会掉):gpt-oss D64 fused attention bwd,gfx950 occ=1**
      (2026-09-11,campaign 20260910_112555 r11)。ISA census `instr=4024` / `mfma=1280` ⇒ **n = 2.144**;
      PMC 直接量到 `SQ_VALU_MFMA_BUSY_CYCLES / SQ_INSTS_MFMA` = **16.00 拍/条**(闭式里的 `16` 就此独立坐实),
      实测 duty = `MFMA_BUSY /(WAVE_CYCLES×4)` = **54.4%**,而 `c=8.5` 预测只有 **46.8%** ⇒ 反解 **c ≈ 6.25**。
      ⇒ `c` 不是常数,n 从 ~0.75 涨到 2.14 时从 8.5 掉到 6.25(每条非 MFMA 指令的**边际**代价随 n 变小,
      共执行与影子重叠的机会更多)。**用法不变(排序、定方向),但别用任一单点的 `c` 去折算绝对幅度**。
  - ⚠️ **`SQ_INSTS_VALU` 把 MFMA 也算进去了**（实测 out_proj_wgrad VALU 1.737e7 vs MFMA 1.678e7，
    只高 3.5%）。算 `n` 必须先减掉 `SQ_INSTS_MFMA`，否则 n 会大一倍，并且会凑巧拟合出一个
    漂亮但错误的「4 拍/条、零共执行」模型（本项 KB 差点就这么记错了）。
  - ★★★★★ **闭式模型的下一级：把 `c` 拆开,`SQ_WAIT_INST_ANY` 就是「空掉的 MFMA 影子」,可以逐拍记账**
    （2026-09-11，campaign 20260910_112555 r12，gpt-oss D64 fused bwd，occ=1）。
    上面的 `c` 是拟合量;它其实等于 `4 +（未被占用的影子拍数 ÷ 非 MFMA 指令条数）`,所以 **`c` 变小
    就是影子被填得更满** —— 这解释了为什么 n 翻倍时 `c` 从 8.5 掉到 6.25。而右边那一项是可以**直接数**的:
    - **供给**:一条 MFMA 占管道 `P` 拍(PMC 实测 `MFMA_BUSY/INSTS_MFMA`,本例 16)、只占发射 4 拍
      ⇒ 每条留 **`P−4` 拍影子**(本例 12),乘 MFMA 条数 = 每 trip 的影子供给。
    - **需求**:把 ISA 逐条按发射拍数计价并求和。gfx950 实测单价:MFMA 4 / trans(`v_exp` 等,**half-rate**) 8 /
      `ds_*_b128` 8 / 其他 LDS 5 / VMEM 6 / 普通 VALU 4 / `s_*` 1。
    - **归因**:顺着 ISA 走一遍,对每一对相邻 MFMA 之间的间隙 `d`,记
      `used += min(d, P−4)`、`empty += max(0, P−4−d)`、`uncovered += max(0, d−(P−4))`。
      ⚠ **必须按「每个间隙独立」算,不能把没用完的需求攒着留给后面的 MFMA** —— 机器是按序的,
      一段连发 M 条的 MFMA 就是白白扔掉 `(M−1)×(P−4)` 拍,攒着算会把成批改动算成正收益
      (这正是 r10「成批 VALU 判负、静态模型却给它高分」的根因)。
    - **接地**:本例供给 15360 / 需求 13616 / **EMPTY 7424(48.3%)** / FREE 7148,而 PMC 的
      `SQ_WAIT_INST_ANY` = 11809 拍、`15360 − COEXEC 3457 = 11903` —— **1% 内吻合**。
      再把 EMPTY 按合法性拆开:**RAW(读上一条 MFMA 的目的寄存器)= 0**、DRAIN(`s_waitcnt` 隔着)= 364、
      HAZARD(`s_nop` 隔着)= 424 ⇒ 结论是**摆放**问题不是**依赖**问题,而这个区分单看三项预算得不到。
    - ⚠ **`SQ_VALU_MFMA_COEXEC_CYCLES` 只数 VALU 共执行**,不含 LDS/VMEM 骑影子:本例静态"仅 VALU"
      收获 4208 拍 vs 计数器 3457 拍(82% 吻合),而含访存的总收获是 7148 拍。**拿计数器校验静态模型时
      要用同口径的子集**,否则会以为模型高估了一倍。
    - ⚠ 这套账只算**发射占用**,不含依赖延迟:本例静态 trip 26167 vs 实测 36775(低估 28%)。
      **它足以排序**(已与 COEXEC 接地),**不足以预测绝对 wall**;要那 28% 得上 ATT。
    - ★★★★★ **总账只说"浪费了多少",要落到臂上必须再切两刀:按 emitter 切、按调度区域切。**
      ① **成批普查(按 emitter)**:扫出所有**中间不含一条 MFMA 的极大连续实指令段**,对每段算发射拍数,
         再算它 ±8 条 MFMA 内的空影子 = **就地 1:1 交织能吸收的上界**。本例 109 段 / 5922 拍
         (= 影子供给的 38.6%)/ 可吸收 5240 拍,而且 59 段来自**同一条 emitter 链**。
         EMPTY 告诉你浪费了多少,**成批站点告诉你是谁浪费的**——后者才能写成臂。
      ② **区域账(按调度边界)**:按 ISA 里真正存在的区域标记(gfx950 上是 `s_setprio` / `s_barrier`)
         切段,对每段算 `slack = 供给 − 需求`。**slack > 0 的区域自己没有足够的活填满自己的影子**,
         只能跨边界进口;slack < 0 的区域有富余可以出口。于是
         **纯区域内重排的上界 = Σ(EMPTY − max(0, slack))**。本例 49 个区域:EMPTY 7623 拍,
         其中 **4567 拍不跨任何边界就能拿到**,3056 拍是硬地板。⇒ 这一刀直接把"要不要动跨区域的
         大结构"这个问题变成一个可以先算再决定的数,**在开轮前就能判断本轮该不该碰寄存器**。
      ③ ⚠ **别把长连发 MFMA 一律当成"批量取数没交织"**。本例六段 15 连发前面只压着 1-2 条 load
         (要 1:1 填满需要 84 条),因为它们的操作数是**寄存器/AGPR 常驻**的(`K_REG=True`)——
         寄存器常驻的 GEMM 链**自己没有访存可交织**,它的影子只能靠进口填,所以排序要排在
         "有活但排错了"的那些区域后面。开臂前先看**连发前窗口里到底有几条可搬的指令**。
      ④ ★ **同一个核里往往已经存在一段做对了的样板,先找到它**。本例 head-step staging 段
         (ISA 476-501)已经是 `[MFMA][ds_read/buffer_load]` 1:1、**每条 MFMA used 12 / empty 0**,
         证明硬件与调度器在发射序要求时是肯做的;它同时是"改动不能把它弄坏"的回归基准。
- ★★ **三项占比相同不代表等待源相同 —— 必须同时报 `SQ_WAIT_INST_LDS / SQ_WAIT_ANY`**。上面那个 mxfp8 参照案例
  里 `SQ_WAIT_INST_LDS` 占 `WAIT_ANY` 的 **59%**（所以它的结论落在 LDS/相位上）；而 syncv3 grouped fp8
  tensorwise 非持久核实测三项 **21-22 / 45-48 / 31-33%**（与参照几乎重合），`WAIT_INST_LDS` 却只占
  `WAIT_ANY` 的 **22.6-27.3%** ⇒ 同一个三项分布可以来自完全不同的等待源，只看三项会把结论误导到
  「LDS 延迟」这条错路上。**报三项时一律附 `WAIT_INST_LDS` 占比**，剩下的那 ~73% 才是 barrier/vmcnt 会合。
  （campaign 20260904_151102 r6，两个 cell 数字一致。）
- ★★★ **⚠ 但「剩下那 ~73% 是 barrier/vmcnt 会合」≠「减会合条数能拿回来」—— 这一步必须实测，别当推论用。**
  同一个核（syncv3 grouped fp8 tensorwise NT 非持久，`WAIT_ANY` 31-33%，8 条 `s_barrier`/K-iter、489 条/binary）
  在 r7 把 rendezvous/drain 排期**双向**扫了 9 个扰动，**每一个都是负的**（同进程双序配对，xchk 单独判定）：

  | 扰动 | s_barrier | xchk | 双序均值 |
  |---|---|---|---|
  | shipped | 489 | — | 0 |
  | 删 pre-c00 开场会合 | 423 | ❌ race | −3.9…−6.1% |
  | 删 pre-c01 开场会合 | 443 | **0.0 逐位** | −1.1…−2.3% |
  | 删 pre-c10 开场会合 | 420 | ❌ race | −2.1…−4.8% |
  | 删全部 3 条开场会合 | 308 | ❌ race | −2.9…−8.7% |
  | 删只读尾部 K-step 的会合 | 461 | ❌ race | ~−0.3%(对 inert 基准) |
  | **加** 一条只读区间的重锁相会合 | 552 | **0.0 逐位** | −3.9…−6.8% |
  | graded g2s drain 收紧 1 条 | 489 | **0.0 逐位** | −1.9% |
  | 收紧 2 条 | 489 | **0.0 逐位** | −3.7% |
  | 收紧 4 条 | 489 | **0.0 逐位** | **−26%（悬崖）** |

  ⇒ 三条可复用的判据：
  ① **shipped 的会合条数/摆位往往是双向局部最优**：加一条与减一条都掉 1-7%，因为 gfx950 的 `s_barrier`
     前 ISA 里**没有**显式 `s_waitcnt`（region 边界读作 `wait(-)`）⇒ 该会合自带硬件 drain ⇒
     **barrier 条数与 graded drain 排期是同一个变量**，动其一必然打乱其二。
  ② **drain 收紧是单调安全（只可能更强的等待，不违反 pitfalls/04 的"禁放宽"）但单调更慢**，
     且存在悬崖 —— 悬崖位置就是"prefetch cover 刚好等于 HBM 延迟"的那一点，可用它反推 cover 余量。
  ③ 这类核真正的 `WAIT_ANY` 杠杆不在"会合条数"，而在**调度锚点**（见 pitfalls/09 §s_setprio：
     同一个核上拿掉包 mfma group 的 setprio 对 = **−21%**，而 `WAIT_INST` 45-48% 正是发射仲裁等待）。
- `SQ_LDS_IDX_ACTIVE = 4 × SQ_ACTIVE_INST_LDS`（逐位）。**按 CU-cycle 归一才是 LDS 数据利用率**
  （gfx950 LDS 256 B/clk；某 8-wave GEMM 每 K-iter 每 CU 192 KB ⇒ 26.6%，3.8× 富余）；
  按 MFMA busy 归一得到的「10×」是量纲不明的说法，容易被读成结论。
- ⚠️ rocprofv3 counter CSV 的 `VGPR_Count` 是 **granule 计数**（同一 kernel ISA reg-note 246 → CSV 报 124），
  **不能当 spill/寄存器门禁**；门禁只认 ISA reg-note（见 pitfalls/09）。
- ⚠️ 本容器版本的 counter CSV 直接落在 `<dir>/<out>_counter_collection.csv`，**没有 `pass_1/` 子目录**。
- ⚠️ 在 `ssh ... docker exec bash -lc '...'` 里写 `pkill -f rocprofv3` 会**杀掉自己**（模式匹配到自身命令行），
  表现为 exit 143 且无输出；查残留进程用 `ps -ef | grep`。

### LDS 带宽 bound vs 延迟暴露(必区分)
- 量 `SQ_LDS_IDX_ACTIVE`(LDS 端口忙周期)**:** `SQ_VALU_MFMA_BUSY_CYCLES` 比值。
- 实测 8w wholeloop = **1:9.46**(端口只在 MFMA 忙的 ~10% 活动,≥5× 余量)→ **不是端口带宽 bound**。
- `SQ_WAIT_INST_LDS ≈ SQ_LDS_IDX_ACTIVE` 且占 `SQ_WAIT_ANY` 的 **59%** → 是 **ds_read 延迟暴露**而非带宽打满。
- 长 K 冒烟枪:`SQ_WAIT_INST_LDS/GUI`(LDS-wait 归一化,越低越好)、`MfmaUtil`(MFMA busy%)、`SQ_INSTS_LDS`(=算术强度)、`SQ_INSTS_VMEM`、bank_conflict、`GRBM_GUI_ACTIVE`(cyc/dispatch,时钟无关效率)。
- ★**2026-08-04 锐化(gpt-oss fused attn bwd 实测,代价是一整轮)**:`SQ_WAIT_INST_LDS`
  **不能单独当"LDS 是 bound"的证据**,哪怕它高到 **81% 的 CU-cycle 预算**、且 `SQ_LDS_IDX_ACTIVE/CU-cycle`
  高到 **1.20**。该内核在这两个读数下,把**三类 LDS 读各删一半**(lse 广播 `ds_read_b128` 384→320、
  GEMM1 A-frag 384→256、GEMM2 `ds_read_b64_tr_b16` 1536→1280)实测 **全是 0.0%**(交错 A/B 3/7、3/7、2/7)。
  原因:它计的是**依赖等待**(wave 等 LDS 返回的周期),同驻 wave 一掩盖就不构成资源上限;
  而 `IDX_ACTIVE/cycle` 的分母(每 CU 每 cycle 的 LDS 数据通路宽度)在 CDNA4 上比直觉大一倍。
  ⇒ **判 LDS 是否 bound 的唯一可靠手段 = 减法探针(删掉一半读)+ ISA 计数交叉验证**,
  counter 只用来提出假设。**代价警示**:整个 campaign 的第一 bound 结论和由它派生的两个方案
  (换宽 MFMA 减 reads/MFMA、head 间共享 fragment)都建立在这个误读上,白排了两轮。

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
| **Power/clock-bound** | **负载中** sclk 明显低于 boost(实测 hd64 fwd 1713 MHz vs 液冷 2.4 GHz)且 power ≥90% TBP(1346 W / 1400 W) | 降每单位工作的指令数与搬运量;⚠**所有 util 百分比必须按实测 sclk 折算**,否则会低估 util 20%(51% vs 63%)并把「买时钟」记成「省周期」 |

**★ 采 sclk/power 是 profiling 的第 0 步(本项 KB 曾连续 10 轮漏掉,把 util 算错 20%)**:
```
(WARMS=14 REPS=3 python -u <bench>.py >/tmp/b.log 2>&1 &); sleep 30
for i in 1 2 3 4; do rocm-smi -d <gpu> --showclocks --showpower | grep -iE "sclk|socket"; sleep 1; done
```
在-时钟峰值 = `#SIMD × FLOP/cyc/SIMD × 实测 sclk`。**每个候选都要单列 sclk 一栏**:
TF 涨了但 sclk 也涨了,先把时钟贡献扣掉再判「省周期」。
⚠ 反例修正:「dense fwd 在功耗墙上 ⇒ 指令效率优化被掩盖」只对了一半——确实在墙上(96% TBP),
但省周期依然有效(hd64 fwd 的 row-sum 上 MFMA:总 +1.85% 中时钟只占 +0.66%,**省周期 +1.19%**)。

★★ **占用率类改动在功耗墙下会被时钟回吐,符号是负的**(hd64 fwd r14,与上面那条正好互补):
occupancy 2.985 → **3.978** waves/SIMD,功耗纹丝不动(1352 → 1353 W = 96.6% TBP),
但 sclk **1710 → 1654 MHz(−3.3%)**,wall 只 +0.50% ⇒ 反解**省周期 +3.9%**。
机制:并发度上升 ⇒ 每拍同时发射的 MFMA 更多 ⇒ 每拍功耗更高 ⇒ DVFS 压频。
⇒ 判据:**占用率候选必须报 sclk**。只看 TF 会把一个 +3.9% 的周期改动读成 +0.5% 而错误放弃;
  反过来,在**没到功耗墙**的 kernel 上同样的改动应当拿到接近全额的周期收益。

★★★ **MFMA 原子的选择在功耗墙下是「能量」杠杆,不是「周期」杠杆 —— 而且 MFMA-busy 计数器看不见它**
(gptoss bf16 grouped **fwd** campaign 20260826 r15,gfx950 / 1400 W 封顶,**单轮 +9.17%**,是三场 campaign 里最大的一步)

`Mfma32x32x16` → `Mfma16x16x32`(同一块 256×256 tile,每 wave 走两倍数量的小 tile):

| | 32x32x16 | 16x16x32 |
|---|---:|---:|
| `v_mfma` 条数 | 2880 | **3600**(翻倍) |
| `ds_read_b128` / `buffer_load_dwordx4` / `s_barrier` | 1080 / 360 / 361 | **逐字节相同** |
| `SQ_VALU_MFMA_BUSY_CYCLES` | 2.12337e9 | **逐位相同** |
| 每 dispatch 周期数 | — | **+8%(更差)** |
| MFMA 利用率 | 84% | **77%(更差)** |
| sclk @ 1400 W 不变 | 1.40 GHz | **1.667 GHz** |
| **wall** | — | **−10.8%** |

⇒ 机制:**32×32 原子在用累加器寄存器堆的能耗换指令条数优势**,每 MAC 的累加器 RF 字节数是 16×16 的两倍,
功耗封顶的卡上 DVFS 直接为此收费。访存一字未动,周期变多,利用率变低,唯独时钟涨 19%。

**两条判据(都是反直觉的,不写下来必然重犯):**
- ❌ **绝不能用 MFMA util 做门禁** —— 这个 +9.5% 的改动会被 84%→77% 直接否掉。
- ❌ **`SQ_VALU_MFMA_BUSY_CYCLES` 对原子速率是瞎的**,两边逐位相同。判原子改动只能看 **wall + sclk + power**。
⇒ 排查"为什么 mfma 流慢"时,如果周期口径的维度(barrier/ILP/占用/带宽/代码量/AGPR)都排除干净了,
  **换坐标系去看能量**:同工作量下比 sclk,而不是比周期。
  (本项 KB 的前一场 campaign 在周期坐标系里排了二十多个维度、动用完整 ISA + HIP 微基准仍未破,
   就是因为真正的货币是能量;换原子一轮解决。)

⚠ 附带代价与残留:VGPR 220 → **256(触顶)** + 2-dword spill;`TCP_TCC_WRITE_REQ` +5.3%
(ragged-tail body 的非配对 store 从 64 B 掉到 32 B 粒度)——r16 用配对写回把它压回 1.17965e7 的 64 B 地板。

★ **16×16 的配对 store 不需要 permlane**:把一个累加器的两个**行** `cvt_pk` 成 dword 再 `permlane16_swap`
是错的——row-major 的 C 里第 r 与 r+1 行相隔 `c_cols*2` 字节,packed dword 不是一段内存。
可行构造是复用既有的**偶/奇列 16 列交错**,零 `v_permlane16_swap_b32` 就能达到 gfx950 的 64 B 写粒度。

★★ **两个 launch 旋钮(GROUP_M / xcd_band)必须在「tile 成本分布」变化后重扫,不只是 shape 变化后**
(00-decision-index row 40 说「band 赢家不跨 kernel 迁移」,实测**同一 kernel 内部也不迁移**):
- r15:tile 快了约 20%,Down 的 GROUP_M 就从 8 翻到 4(slab 没动)。
- r16:ragged 列块不再是补满的整价 tile,GateUP 的 `xcd_band` 就从 64 翻到 32(−0.58%,回文里每个位置都快)。
⇒ 换过原子/几何/尾块之后,**当轮就要重扫这两个旋钮**,否则带着上一代的最优值跑。

★★ **在 ≥99% TBP 上「省周期」只兑现 20~45%(shape 相关,见下),「省能量」拿满额 —— 排杠杆时按这个折算率排序**
(grouped mxfp8 NT,campaign 20260729 round 11 同 shape 同探针前后对照):
周期口径 tw/mx 1.0315 → **1.0858**(+5.3%),但 wall 只 1.0172 → **1.0392**(+2.2%)⇒ **兑现 45%**。
机制:mx 变快后瞬时功耗更高,DVFS 又把它的 sclk 压低 2.5%(mx 与 tw 的时钟差 1.39% → 4.3%)。
round 12 用同一探针复核,逐点复现:mx 0.9083/0.9115 ms @ **1909.6/1899.3 MHz** @ **1396.0/1397.6 W**,
tw 0.9451/0.9460 ms @ 1992.1/1987.6 MHz @ 1382.1/1383.1 W ⇒ wall 1.0392 / cycle 1.0858,**99.7% TBP**。
⇒ 该折算率在**同一 shape 上跨重复稳定**(非一次性观测),但⚠️**跨 shape 变化——45% 是 `fwd/down balanced` 上的值,收官轮实测 min 配置 `dgrad gate_up heavy` 只兑现 20%**(mx cyc 2905 vs tw 3110 赢 7.1%、wall 1.6014 vs 1.6241 只赢 1.42%,sclk 1814 vs 1915)⇒ **越 skew/越接近 min 的配置、周期类杠杆越不划算,折算系数要按 shape 取,别当全局常数**;排杠杆时:
**「减字节/减能量」按全额计价,「减周期/减指令」按 20~45% 计价(skew 取低端)**。
pJ/FLOP:mx 0.5581~0.5607 vs tw 0.5749~0.5759 ⇒ **功耗高 ≠ 效率低**,mx 只是把同样的工作做得更快。
⇒ 同样纸面收益下,**减访存/减 L2 miss(省能量)优先于减指令/减延迟(省周期)**:前者同时降功耗,
不会被 DVFS 回吞;后者要打对折。pitfalls/13 只写了「省请求=买时钟」的正向,这是它的反向补充。
⚠ 另:**功耗高 ≠ 效率低**。同一对照里 mx 的 pJ/FLOP 0.5601 优于 tw 0.5758 —— 它功耗更高只是因为
把同样的工作做得更快(功率 = 能耗/时间),别拿 W 读数当效率判据。

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
- ★ **技法细化:`drop` 之外再加一个 `alias` 臂,能把 feed 成本拆成「搬运」与「发射」两笔**
  (2026-08-17 gpt-oss padN dgrad NN 实测,主体块 4 条 g2s):
  - `drop`(整条 g2s 不发)= feed 的**全部**成本:865.1 → 703.9 µs ⇒ feed = **155 µs = 18.6%**。
  - `alias`(g2s 照发,但地址全指向 **k-block 0**,指令条数/寻址算式/LDS-write 一条不少,只是永远命中 L2)
    = 865.1 → 774.0 ⇒ **84 µs 是数据搬运**(L2 miss 延迟),`155 − 84 =` **71 µs 是纯发射/寻址/LDS-write**。
  - 再按操作数各来一遍 `alias`,得到每条流的份额(A 28 µs / B 47 µs;两者相加 10.16% vs 合并 10.58% ⇒ 可加、无共享瓶颈)。
  - **为什么值得多写一个臂**:`drop` 的数字会让你去做「更深预取 / 更好局部性」,而那类杠杆**只能碰 84 µs 那笔**;
    71 µs 只有「更少更宽的 g2s」能碰。少了 alias 臂就会把 2 倍于实际的空间算给预取,然后困惑为什么只兑现一半。
  - ⚠️ 只动**主体块**的 g2s,prologue 与 K-tail 保持原样 ⇒ LDS 里始终是合法数据,核不会挂、只是算错;
    这样 wall 才可比。⚠️ 这些包装是**探针专用**,production 树里不留任何 `_dbg` 参数(pitfalls/10)。
- ★ **同一族的第三个臂:「把新增的那条跨 lane 原语 monkeypatch 成恒等函数」= 该改动的零成本上界**
  (2026-08-18 gpt_oss dgrad NN 行合并实测)。改动本身 = ①新的寻址/行分组(减请求)+ ②为此付的跨 lane
  VALU。把 helper 换成 `lambda a, b: [a, b]` 后,**行集合、写字节、store 发射条数、每条 store 的
  lane→地址映射全部与真改动逐位相同**(只有值错了,所以这个臂只能读时间不能读 bit),于是
  `gain(恒等臂)` = ①的全额、`gain(恒等臂) − gain(真臂)` = ②的价钱。本例 **+1.23% 对 +0.83% ⇒ 跨 lane
  那半付掉 0.40 pp**,直接指向下一手(用 `d16_hi` 省掉 64 条 `v_lshrrev`)。
  - **它和 `alias` 臂的区别**:`alias` 改地址(所以会改行集合、有 methodology/07 记的假阳性风险);
    这个臂**只改值**,地址与指令条数全保住 ⇒ 对「请求路径」类改动是安全的定价方式。
  - ⚠️ 恒等替换必须**保持返回结构**(本例是两元素列表)和调用点条数,否则连指令条数都变了就不是上界臂。
- **★★ 铁律:subtractive/HALF 探针天花板、roofline 峰值率、纸面 op-count 分析给的都是「上界」,不是「可达值」。判负/判正前必须 edit→bench 真实现。** 反复踩证(dsv4 sparse-MLA,见 pitfalls/12):
  - HALF_PV 探针「−8.7% 假想天花板」→ 真 K=32-PV **净负 −11%**(2-tile 批打断 QK→softmax→PV 交织,批结构本身是杀手,非 k 维)。
  - SKIPST「store 占 wall 30-47%」→ 真 DMA / register-transpose / query-blocking **全净负**:store 是「跨-wave 数据共享 + register-prefetch 隐藏 HBM」的必需机制,不是可省浪费。
  - 「gfx950 k=32 双倍 bf16 率」roofline 头 → 落地被数据布局(tr16 转置读 =4bf16/lane=k16)锁死到「必须 2-tile 批」,批税吃光收益。
  - 去-padding「代理测量」判 DMA 负 → 其实测的是 tr16 bank-conflict,不是 DMA 本身;真 per-rank-DMA 保 padding 从没被那个代理覆盖(**代理≠真实现**)。
  - **∴ 探针只告诉你「某成本是否在关键路径」(值不值得投入去攻),不告诉你「去掉它是否可实现」。可达头只有真实现 + bench 能定;纸面 op-count 预判(如 bpermute≫store)与探针天花板一样常错。**
- ★ **铁律要精确到「对 *时间* 的上界」——对 *指令数* 纸面数账是可精确兑现的**(2026-07 mxfp8 grouped 实证):
  边界象限跳过的纸面比 12/11.5 = 1.043478,PMC `SQ_INSTS_MFMA` 实测 mx/tw = 36,175,872/34,668,544 = **1.04348,逐位吻合**;
  但换算成**时间**只兑现 ~50~60%(省了 MFMA,barrier / g2s / LDS 往返照旧)。⇒ 用 op-count 预测**指令数**可以当准数,
  预测**时间**必须打 0.5~0.6 折,再 edit→bench 确认。
- ★ **上界也会*低估*:当一处改动顺带消掉了别的成本时,实测可以反向突破纸面预测**。同 campaign 两例:
  ① 按 MfmaUtil 比值 0.956 预测「追平参考只有 +4.6% 余量」,消掉 per-tile O(G) 扫描后实测 **+5.3~19.1%**(因为
  同时消掉了 tile 开头的串行依赖链与 spill);② 按 traffic 上界 2.1% × 50~60% 兑现率预测 +1.0~1.3%,半-N 边界
  tile 删 B 侧 g2s 实测 **+2.0%**(删的是 DMA,而该 tile 恰是 feed-bound)。⇒ **上界是双向不准的,别用它判负。**
- ★ **孤立单核微基准也是一把不可信的尺子——它连符号都能翻**(2026-08-04 融合 flash-bwd r09 实证):
  把 dQ split-K 规约核单独拿出来扫 work-group 形状,`1024×8` 以 **557.2 µs** 拿下第一、`512×2` **563.6 µs** 第二;
  放回完整 backward 后 `1024×8` 反而**输 0.4% 的整体分**(874.9/876.0 vs 878.0~880.7),第二名才是真赢家。
  机制:孤立跑时该核独占整机且工作集冷热状态与真实调用不同,in-situ 它紧跟 4 ms 的主核、承接主核的 ramp-down。
  ⇒ **微基准只用来"筛掉明显差的档",最终选档必须用 end-to-end 打分尺再确认一次**;同一族的
  「探针/roofline/op-count 只给上界」铁律对孤立微基准同样成立。

### K-sweep 拟合 `t_tile = F + n_phase × P`:把「稳态慢」和「固定开销大」一次分开

攻一个 GEMM 之前先回答「稳态 K-loop 到底慢不慢」——否则很容易花几轮去调一个已经满速的主循环。
做法:固定 M/N,**扫 6 个以上的 K 点**,对每个 tile 的时间线性拟合 `t_tile = F + n_phase × P`,
`P` = 每 K-phase 的稳态成本,`F` = 与 K 无关的 per-tile 固定开销(prologue + epilogue + drain)。
把 `P` 和「dense 同精度单体峰值换算出的 per-phase 成本」对比,就知道差距在稳态还是在 F。

mxfp4 grouped 实证:**P=1.565 µs/phase vs dense mxfp4 5405 TF 换算的 1.59 µs —— 只差 1.6%**,
即稳态 K-loop 已经跑在 dense 速度上,**grouped 相对 dense 的全部效率损失都在 F=7.81 µs/tile 里**。
反向核对:令 F=0 反推 5326 TF,对 dense 5405 闭合 1.5%,三方自洽。这一条直接把优化面从「主循环」
整体挪到了「prologue/epilogue」。

★ **拟合有两个会毁掉整个坐标系的系统误差,都踩过**:
1. **时钟按标称算**。用 2.4 GHz(boost 标称)算出 in-loop util 52.8%,用 `GRBM_GUI_ACTIVE/dur/8`
   实测的 **2.09 GHz** 重算是 **62.6%** —— 一整轮的方向建立在错的 util 上。见 connection/common/05。
2. **`P` 里混进了随 K 增长的邻居 kernel**。第一次拟合把 scale preshuffle 一起计进去了(它的耗时也随 K 涨),
   于是 P 被高估、F 被低估。**修法:用 `rocprofv3 --kernel-trace` 只取目标 kernel 那一行,不要用 wall。**

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

## ★铁律:≥2 waves/SIMD 时,per-wave 的「簇再平衡」是吞吐不变量

把一段 VALU 工作从 wave 内"忙"的 cluster 搬到"闲"的 cluster(经典手法:塞进访存 cluster 的 LDS 延迟窗口、
或塞进 MFMA-bound cluster 的影子里)**改变的只是单 wave 的关键路径,不改变每-SIMD 的资源需求总量**。
SIMD 上有 2-3 个 wave 共享同一套 VALU/MFMA 管线时,别的 wave 早就在填那个"空档",搬迁买不到东西。

hd64 fwd 实测(3 waves/SIMD,A=1127.0 TF):把 row-sum 从 issue-bound 的 QK 簇搬走 →
搬到 PV 簇 **−3.08%**(它其实在关键路径上)、搬到访存簇 **−0.10%**;而**删掉**同一批指令 **+6.4%**。

⇒ 判定顺序:①先用减法探针量出"每省 1 个管线周期换多少 wall"的**边际系数**(hd64 fwd = **0.46**;
  旧记 0.49 来自一个 SNR 崩掉的脏探针,已作废——**探针 SNR 不过就不能拿它的 wall 定价**);
②只有能**真减总周期**的改动才排上日程;③"这个 cluster 看起来很闲"不是理由 —— 那是别的 wave 的工作区。
⇒ 推论:cluster/schedule 类改动的正确期望是 **±0**,测到 ±0 不代表实现错了,代表这类杠杆在此 regime 无效。

**★ 2026-07-29 补一条方向性(hd64 fwd r25,4 waves/SIMD)**:不变量说的是「搬迁买不到东西」,
**不是「两个簇等价」**。同族的三次实测(把一个 tile 的整套 softmax = 32 exp + 16 cvt + 4 row-sum MFMA
搬进 PV 簇)= **4 waves −0.33% / 3 waves −0.89%**,而 ISA 上编织是完美的(串行段消失、s_nop_stall 18→28)。
根因:PV 簇的 8 条 MFMA **影子里已经藏了 16 条 `ds_read_b128` + 2 个 DMA 发射**,而 QK 簇的 8 条 MFMA
只藏 8 条 `ds_read`,VALU 槽是空的。⇒ **搬迁前先数目标簇 MFMA 影子里已有多少条访存/VALU**;
往已经饱和的簇搬 = 负,往有空档的簇搬 = ±0(仍然不是收益来源)。
⇒ 同轮另一条:**「完美编织」本身不产生收益**。同一批 VALU 在 wpe=3 上编织成功值 +1.23%(r24),
在 wpe=4 上换个簇编织值 −0.33% ⇒ **编织类结论必须标注测量时的 waves/SIMD,不可跨档外推**。

**★★★ 2026-09-01 补:这条铁律在 ≥99% TBP 下有了机制和一个双向价格(gpt-oss D=64 fwd,analyze-r2)。**
前面说的是"搬迁买不到东西",但没说**为什么**、也没说**反向搬会怎样**。这轮做了一个**指令逐条相同**
的受控对,两边都测:

| | 部署(最大交织) | `sa`(去交织) |
|---|---|---|
| 主循环发射条数 / opcode 直方图 / `s_waitcnt` 组成 / vgpr/spill/LDS | 240 · 64exp+40mfma+… | **逐项相同**(只有 opcode 序列 md5 不同) |
| `SQ_VALU_MFMA_BUSY_CYCLES` / `SQ_ACTIVE_INST_VALU` | 2.454847e9 / 4.584023e8 | **逐位相同** |
| `SQ_VALU_MFMA_COEXEC_CYCLES` | 1.281799e9(占 SIMD 拍 43.64%) | 1.155382e9(37.68%)= **−9.86%** |
| `GRBM_GUI_ACTIVE`(真周期) | 2.294926e7 | **+4.39%** |
| **sclk** | 1602 MHz | **1660 MHz(+3.6%)** |
| **wall** | 1.8364 ms | **1.8707 ms(−1.87%,三批复现到 0.01%)** |

⇒ ①**`SQ_VALU_MFMA_BUSY_CYCLES` 不但对原子速率盲(§B40),对指令顺序也盲**——同一批指令换个序,
它逐位不变。**看调度的那个计数器是 `SQ_VALU_MFMA_COEXEC_CYCLES`**,20 s 一趟,是这一族唯一的仪器。
②**共执行(co-execution)是可以定价的**:每去掉 1 个百分点的 coexec duty = **+9.73 MHz sclk** 但
**+0.74% 真周期**,盈亏平衡在 0.61 %周期/点 ⇒ 部署点(最大交织)赢,但只赢 0.13 点,**是个薄的局部
最大值,不是安全裕度**。
③**机制**:≥99% TBP 时 `wall ≈ 总能量 / TBP`。纯重排**不改总能量**,只在"每拍功耗"和"周期数"之间
互换,于是 sclk 与 `GRBM_GUI_ACTIVE` 反向走、wall 几乎不动。同轮 14 个 arm(两个方向都推到极端:
把 exp 拆碎塞进 MFMA 之间 → −1.46%/−2.44%/−3.39%;把 exp 整段隔离出来 → −1.87%)**全部落在 ±3.4%
带内,部署点在带顶**。而**删掉工作**的 arm 拿到全额(同核 `noexp` +8.7%)。
⇒ ★ **提案前先自问是哪一类:删工作(付 100%)/ 删周期(付 ~15%,见下条)/ 只重排(付 0±2%)。
只重排的不是候选,是旋钮。**
④ 这同时把两个看着矛盾的老观测统一了:r17"MFMA 管线工作 +11.1% 而时钟反而 **升** 3.9%"(MFMA 每拍
便宜)与 r14"占用率上去 sclk **掉** 3.3%"(并发抬高每拍功耗)—— **DVFS 控制器收的是峰值每拍功耗,
不是总工作量**。

**★★ 同轮补:周期→wall 的兑现率是有方向性的,不能两边通用。** 本卡上文记的边际系数 0.46、
pitfalls/13 §r18 记的 91%、以及本卡 §「周期口径 vs wall」记的 45%,都是在**变差方向**上量的。
gpt-oss D=64 fwd 在**变好方向**上实测:`kh1,nf1` arm 让 `GRBM_GUI_ACTIVE` **−1.435%**(coexec duty
43.64%→43.95% 基本不动,spill 4→0,主循环 240→239),wall 只 **+0.12%**(计分尺,带对照)/ **+0.24%**
(持续探针)⇒ **兑现 8–17%**。⇒ 给周期类杠杆定价时,**用改善方向自己量一次**;拿变差方向的系数去
乘,会把一个 1.4% 的周期节省吹成 1.3% 的 wall 期望,然后在噪声里找不到它。

### 边际系数怎么量(比 roofline 可靠)
减法探针要**只删条数、不换语义框架**(例:把 32 元素 fold 改成 2 元素 fold,保留下游 pack/MFMA 链防 DCE),
再按 `Δwall% ÷ (waves_per_simd × Δ指令 × 拍/指令 ÷ 每-SIMD 迭代耗时)` 归一。
系数 ~1 = 纯 issue-bound;~0.5 = 与另一条管线部分重叠;~0 = 该管线不是约束。

---
来源: flydsl-fp8-gemm-tuning/SKILL.md, 10-8wave-scvgpr.md, flydsl-kernel-authoring/SKILL.md, agpr_rawasm_progress.md, project_mxfp4_epilogue_store.md, gemm-optimization/SKILL.md, 07-benchmarking.md, 07-benchmark.md, capture-kernel-trace/SKILL.md, 10-grouped-wgrad-4wave-3buf.md, kernel-trace-analysis/SKILL.md, tool-rocprof/SKILL.md, diag_4w_vs_8w.md, prefetch-data-load/SKILL.md, gemm/overview.md, programming-model.md, 08-att-root-cause.md, project_mxfp4_k28672_ceiling.md, lds-optimization/SKILL.md

## ★ 两个不用 profiler 的判 regime 探针(比 PMC 快、比 PMC 硬)

memory-bound 的核上,PMC 只告诉你「多少字节/多少请求」,不告诉你「少一条会省多少秒」。
这两个探针直接给**边际价**,各 70s,不需要 rocprofv3。

### 1. 计算加法性探针:计算到底被盖住了没有
往核里**纯加计算、零额外访存**,看秒表动不动:把 MFMA 循环由 `range(K)` 改成 `range(K*2)`、
操作数取 `[step % K]`(结果会错,只用于计价;记分尺量不了,要另写同口径非记分谱)。
- 秒表不动(实测 MFMA +76% ⇒ **+0.1%**)⇒ 计算完全被盖住 ⇒ **该核所有「减指令 / 减 VALU / 减 LDS 发射 /
  换占用」的候选全部不必做**,一次性砍掉一整族候选。这比 `SQ_ACTIVE_INST_ANY` 19.2% 那道门更直接:
  门只说「比例低」,这个探针说「边际价 = 0」。
- 秒表按比例涨 ⇒ 计算在关键路径上,减指令是真杠杆。
- **这个探针能分开「同一算子的两个 kernel」**:sparse-MLA 的 prefill 加 MFMA = +0.1%(计算全被 fabric 盖住),
  同一族的 decode producer 加 PV MFMA ×4 = **+7%**(≈6 墙钟 cycle/条 mfma32,约 1/3 透过率)。
  ⇒ 别把「本算子 memory-bound」当成整族的结论,**逐 kernel 各测一次**,70s 就能避免整轮预算排错地方。
- **配套的第二条加法臂:纯同步加法性**(往每 tile 多塞 2 条 `gpu.barrier`)。免费(±0.5%)⇒ 屏障本身与 wave
  收敛不在关键路径,「合并/挪 barrier」那族候选可直接砍掉;此时消融出来的那几个点必然在**被屏障保护的数据
  往返**上(LDS rendezvous),要改的是所有权划分而不是同步原语。
- ⚠ **加法臂的计时不要在 `FLYDSL_DUMP_IR=1` 或 `REPS=1` 下取**:一次实测里 dump 版给出 +15.44%,
  正常复测三次全部是 ±0.2%。dump 走的是另一条编译/落盘路径,会污染同进程的第一次计时。
务必用 ISA 核对指令真的加进去了(本例 `v_mfma` 21→37、VGPR 166→168、spill 0),否则可能是被 DCE/CSE 掉了。

### 2. 「表观字节率 > 可达流带宽」⇒ 你不是带宽限,是缓存命中在帮你
先用 `torch` 标一次本机可达流带宽(实测 MI355X:`copy_` 4.98 TB/s、`add(out=)` 6.25 TB/s、
`sum` 只有 3.74、`amax` 2.01 —— **reduction 不能当带宽尺**,见 pitfalls/03 同名条)。
再把核的字节数除以墙钟:如果**高于**可达流带宽(本例 sparse gather 折算 7.24 TB/s),说明这份"字节"里
有相当比例来自 MALL/L2 命中 ⇒ **不要写"已到 HBM 带宽上限"**,该去量的是「哪一级 miss 才是那笔钱」
(用 pitfalls/03 的 alias 臂:掩 index 缩脚印,分别做出 L2 驻留 / MALL 驻留 / 真实分布三条臂)。
本例三条臂给出:L2 驻留 −52.9%、MALL 驻留 −10.7% ⇒ 贵的是 TCC miss 本身,HBM-vs-MALL 只值 10.7%。

## ★ 共驻两个 kernel 时,用 **full − nored 分解**把"body 变慢"和"共驻核被赶走"分开(2026-08-13)

**场景**:一条流水里主核与辅助核(reduce/epilogue/prefetch 核)**共享同一个 512 dword 寄存器池**并
共驻在同一批 CU 上(判据见 methodology/04 的 `ceil8(body) + n*alloc(sibling) <= 512`)。这时**越过
共驻线的旋钮读起来和一次普通的调度回归一模一样**:wall 慢几个百分点,指令数、spill、LDS 全没变。

**方法**:同一个候选跑两遍 —— 一遍完整流水,一遍**把辅助核整个关掉**(结果算错无所谓,这是定价探针,
按 §「上界≠可达」只用来分账),两个数相减就是该候选下辅助核的 **exposure**。于是
`Δwall = Δbody + Δexposure`,两项各自归因。

**实测(gpt-oss D128 fused bwd,dkdv body + 共驻 dQ reduce)**:

| 臂 | full | nored | exposure |
|---|---|---|---|
| ref(462 dw,线内) | 6.3868 | 5.7149 | 0.657 |
| `g2d=1`(**471 dw,越线**) | 6.9951 | 5.8769 | **1.118** |
| `g3_kreg=0`(415 dw,2 个 reduce WG) | 6.6191 | 5.8376 | 0.782 |

⇒ `g2d=1` 表面是 −0.6 ms 的 ring-depth 回归,**实际是 0.15 body + 0.46 驱逐**;而"共驻得更多"
(415 dw 让第二个 reduce WG 进来)反而把 exposure 从 0.657 抬到 0.782。**没有这个分解,前者会被
错记成"ring 深度很敏感",后者会被错记成"共驻越多越好"。**

**同一把尺还能廉价证伪机理假说**(都是先猜机理、再用能改那个机理的旋钮量它):
* "辅助核共驻时只跑到独占带宽的 1/4,是**每线程 MLP 不足**" → 在装得下更宽 wave 的 body 上把每线程
  in-flight load 翻倍:exposure 0.782 → 0.768(≈0)⇒ **不是延迟,是按字节走的共享 fabric 成本**。
* "让 reduce 按生产者写入顺序读,能吃 MALL 余温" → 反转 grid 顺序:0.004 ms ⇒ 死。
⇒ 判一个共驻核该不该继续调形状(wave 数/WG 宽/向量宽/顺序),先用这两个探针问"exposure 是字节项
还是延迟项";是字节项的话,**所有形状旋钮都无效,只有减字节有效**。

---

## 补丁(r19):sclk 不要用 `rocm-smi` 采,用 `GRBM_GUI_ACTIVE` 反推

本卡「采 sclk/power 是 profiling 第 0 步」的处方是对的,但**采法**在短 kernel 上会骗人:
JIT 缓存命中后 driver 早已跑完,`rocm-smi` 采样窗口落在空闲段上(实测读到 **158 MHz**),
据此算出的 roofline 会把 duty 高估到接近饱和(某核被误判为 "76% roofline",实际 ~56%)。

★ **可靠采法**:`GRBM_GUI_ACTIVE ÷ XCD 数 ÷ dispatch 时长`。
MI355X/gfx950 上 `GRBM_GUI_ACTIVE` 是**跨 8 个 XCD 求和**的,必须先除 8:
`40,318,962 / 8 = 5,039,870 cyc / 2.764 ms ⇒ ~1823 MHz`(profiling 下,未 profiling 约 2.0 GHz)。

★ 顺带校验管线深度:`SQ_VALU_MFMA_BUSY_CYCLES / SQ_INSTS_MFMA` 应等于该 MFMA 的标称拍数
(`v_mfma_f32_16x16x32_bf16` 实测 **16.00 cyc/inst**,整数落点即说明时钟与计数口径自洽)。
两个数都对上之后,`duty = MFMA_BUSY / trip_cycles` 才是可信的。

★ gfx950 发射速率补正:`v_exp_f32`(以及同族 trans)是**半速 8 cyc**,不是四分之一速。
用 `SQ_ACTIVE_INST_VALU` 按类建模来验:各类条数 x 速率之和应与实测在 1% 内(见 pitfalls/13 §r19.1)。

---

## 补丁(r20):PMC 拆不开的等待,用「加法探针」定价

★ **先确认计数器在这台 build 上真的有值**。gfx950 / rocprofv3 1.1.0 实测:
`SQ_LEVEL_WAVES` 与 `SQ_ACCUM_PREV_HIRES` **恒读 0**(不是 0 占用,是不可用)⇒
拿不到「平均在飞 wave 数」;等待类只剩 `SQ_WAIT_ANY` / `SQ_WAIT_INST_ANY` /
`SQ_WAIT_INST_LDS`,**无法把 vmcnt 等待与 barrier 等待分开**。

★ **加法探针(additive probe)**:把怀疑的那段**复制一遍**、再把结果无损折回去,
测「多做一次」的边际成本,作为「少做一次」的**上界**(仍受「上界≠可达」铁律约束)。
关键工程细节,少一条就测不准:
1. **做成编译期参数并写进 kernel 名**,让 baseline 与探针能在**同一进程里 ABBA** 对打。
   跨进程展幅可达 12.9%,1~4% 的边际成本在那种噪声下不可见。
2. **折回要数值精确**:跨 wave 求和后 `×(1/N)`,N 取 2 的幂 ⇒ 二进制浮点精确,
   `relL2` 一位不变,correctness gate 照常过。
3. **探针写到独立 scratch**,避免与原数据构成 WAR ⇒ 只需一个额外 barrier,
   否则要两个,成本被高估。
4. **探针放在原操作之后**(vmcnt/lgkmcnt 已满足)⇒ 测到的是纯粹那一类操作的成本,
   不掺前置等待。
实例(sparse-MLA decode producer):多一次跨 wave rendezvous = +1.0~4.0%;
多一套 8 条 `ds_write_b128`/tile/wave = +3.5~4.9%。见 pitfalls/12 轮 2 补丁。

⚠ **「上界」这个词在这里比想象的弱:加法探针与减法臂可以符号相反**。同一个
sparse-MLA decode producer 上,「多一次 rsum rendezvous」= **+1.0~4.0%**,而真的
把 rsum 的每 tile 折叠**删掉**(合法地推迟到最后一个 tile)= **+1.2/+2.0/+5.9%**,
也就是**删了反而更慢**。原因是那 4 条 `ds_read_b32` 是 barrier 之后第一批 LDS 流量,
正好盖住喂 PV 的 `plds` 读的延迟 —— 它们不是净开销,是**延迟填充**。
⇒ 加法探针量的是「这段在关键路径上的边际价」,只有在核**没有空闲发射位可吸收
额外访存**时才等于「删掉能省的钱」。**决定要不要花几小时写减法重构之前,
先看这个核是访存/调度限还是发射限;凡是「已知有空转槽」的核,加法探针的数
不能当立项依据,必须做减法臂。**(实例见 pitfalls/12 「KB 纠错:rsum」)

★ **bank conflict 的比值单独看没有行动价值**。必须和**手算的宽访问地板**对比:
按 `b128`(8 cyc/指令)/ `tr8_b64`(4 cyc)/ 窄操作逐类累加出 cycle/wave,
再和 `SQ_LDS_IDX_ACTIVE / wave` 比。实测 908(算)vs 900(测)⇒ 已在地板上,
此时**改 stride 动不了**(16B 对齐 ⇒ dword stride 是 4 的倍数 ⇒ `gcd(stride,32) ≥ 4`),
只有 lane→row/col 相位旋转(XOR swizzle)+ 写端配套才有戏。

★ **固定工作量对照**:给「网格形状 / CTA 数」归因时,**不能扫 batch/seq** —— 那同时改了
总工作量。正确做法是固定 seq、扫每 CTA 的 tile 数(`inner_iter` 之类),让
`tile 总数` 不变。实测一例:0.75 → 3.0 CTA/CU(跨过 2 CTA/CU 硬上限)只差 0.4%,
而扫 seq 得到的拟合线曾把同一现象定价为 +11.6%。

★ **本容器的 `rocprofv3` 边界**:`--pmc` 配 `SQ_*`/`GRBM_*` 稳定可用(~13s/次);
`--kernel-trace` 与 **`TCC_*`/`TCP_*` 计数器组**会把 KFD 卡住
(`rocminfo did not return within 60s`,重试撞 300s timeout)⇒ L2 命中率只能算不能量。
