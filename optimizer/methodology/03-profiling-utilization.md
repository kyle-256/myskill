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

### ★ wave-cycle 预算：把「卡在 barrier/waitcnt」与「想发但管道忙」分开（gfx950 实测配方）
- 三个 counter 恰好把 `SQ_WAVE_CYCLES` 分成 **100%**（全部 quad-cycle，×4 换算成周期）：
  `SQ_ACTIVE_INST_ANY`（正在发射）+ `SQ_WAIT_INST_ANY`（想发但发不出）+ `SQ_WAIT_ANY`（waitcnt/barrier）。
- **只看 `SQ_WAIT_ANY` 会误判**：MFMA-bound 循环里 `SQ_WAIT_INST_ANY` 高是**健康态**（自己的 MFMA 还占着管道，
  sibling 也发不进去），真正的空转来源是 `SQ_WAIT_ANY`。实测一个 8-wave/2-waves-per-SIMD 的 mxfp8 GEMM 主环：
  24.2% / 45.2% / **30.6%**，而管道空转（20% of 主环）≈ 两个 sibling wave 的 `SQ_WAIT_ANY` 交叠（~70%）
  ⇒ 结论是「barrier convoy 偏斜」，而不是 LDS/VMEM 延迟，于是 barrier 数与相位结构成为唯一相关的轴。
- 判据：把 `SQ_WAIT_ANY / SQ_WAVE_CYCLES` 与 `MFMA_BUSY /（SIMD 数 × 时长 × 实测时钟）` 一起读；
  两者之差就是「可归因于 rendezvous」的空转。改动后若 `WAIT_ANY` 占比上升而 MFMA busy 不变，必然更慢
  （实测 30.6%→34.2% 对应 −2%）。
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

### 边际系数怎么量(比 roofline 可靠)
减法探针要**只删条数、不换语义框架**(例:把 32 元素 fold 改成 2 元素 fold,保留下游 pack/MFMA 链防 DCE),
再按 `Δwall% ÷ (waves_per_simd × Δ指令 × 拍/指令 ÷ 每-SIMD 迭代耗时)` 归一。
系数 ~1 = 纯 issue-bound;~0.5 = 与另一条管线部分重叠;~0 = 该管线不是约束。

---
来源: flydsl-fp8-gemm-tuning/SKILL.md, 10-8wave-scvgpr.md, flydsl-kernel-authoring/SKILL.md, agpr_rawasm_progress.md, project_mxfp4_epilogue_store.md, gemm-optimization/SKILL.md, 07-benchmarking.md, 07-benchmark.md, capture-kernel-trace/SKILL.md, 10-grouped-wgrad-4wave-3buf.md, kernel-trace-analysis/SKILL.md, tool-rocprof/SKILL.md, diag_4w_vs_8w.md, prefetch-data-load/SKILL.md, gemm/overview.md, programming-model.md, 08-att-root-cause.md, project_mxfp4_k28672_ceiling.md, lds-optimization/SKILL.md

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
