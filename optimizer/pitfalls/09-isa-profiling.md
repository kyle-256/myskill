# ISA 调度提示与 profiling 陷阱：sched_* 计数、s_setprio、FLAT、ATT/PMC 分工、code.json AGPR-blind

> 类别: 踩过的坑 · 主题标签: sched-hints, s_setprio, s_waitcnt, FLAT, occupancy, rocprofv3, ATT, PMC, debug-info

## 调度提示/ISA 陷阱：sched_* 计数必须精确、s_setprio 要归零、别用 blanket s_waitcnt 0

- **sched_* 计数必须精确等于 body emit 的 opcode 数**：FlyDSL 的 `sched_mfma` / `sched_dsrd` / `sched_dswr` / `sched_vmem` hint 计数必须 EXACTLY 等于 body 里对应 opcode 的发射数量。一旦不匹配，backend 会**静默 drop policy**、退回默认调度，**无任何 warning**。改了 body 的 opcode 数一定要同步改 count。
- **sched_barrier(0) 是 load-bearing 且 ordering-only**：删掉它整个手工 schedule 就塌（LLVM 会跨迭代边界 coalesce）。位置错也不行——放进 constexpr loop **里面**（而非 loop 之后）会 pessimize overlap。schedule 必须跑在**拥有它所重排的那个 LDS write 的 `gpu.barrier()` 之后**。
- **s_setprio(1) 后必须归零**：MFMA block 前 `s_setprio(1)`、block 后立即 `s_setprio(0)`，防止 MFMA 被 VMEM/LDS 挤掉。❌ 别再试 忘记把它 DROP 回 0：会饿死 co-resident wave、直接砸 occupancy。
- **s_setprio 的正负是 regime 决定的，别按 kernel 名搬结论**（2026-08-04 融合 hd64 flash bwd 实测；与 pitfalls/12 的 fwd 判负**相反**）。它唯一的作用是**给同一个 SIMD 上的两个共驻 wave 拉开相位**，所以三条同时成立才为正：①**有仲裁对象**(≥2 waves/SIMD；`waves_per_eu=1` 时恒中性，见 pitfalls/02)；②该 region **一个 wave 有 MFMA run 而它的兄弟没有**（GEMM2、carrier 的 GEMM3）；③body 是 **MFMA-pipe-serialized**（MFMA 占 ~47%）。齐了就是强正：拿掉包 GEMM2 的那一对 = **9/9 负 −2.8%**。反例都来自破坏②：把同一对**对称地**推广到 GEMM1（8 个 wave 全都有 MFMA run，没人可错相）= 7/11、6/11 噪声，**即使 hazard `s_nop` 198→102**；包 GEMM3 的 kstep ring = 6/11 且 spill +63 dword（setprio **不切 scheduling region**，却把 ring 的 live range 钉穿它）。
  - **幅度是 no-op，别扫**：全 wave 跑同一份码，比较恒为"in-region(1) vs not-in-region(0)"，1→2 改不了任何序（hd64 fwd 独立实测"提到 2 = 持平"，见 pitfalls/13）。
    ✅ **第二次独立复现**（2026-09-05 syncv3 grouped fp8 tensorwise NT 非持久，gfx950/1400 W）：`s_setprio(1)` → `(2)` / `(3)`，
    ISA 元数据**逐项完全相同**（指令数/VGPR/spill 全等，只差立即数），双序 4 格均值 **−0.09% / −0.02%** = 平。**这条卡准确，别再扫幅度。**
  - ★★★ **⚠ 判据②「一个 wave 有 MFMA run 而兄弟没有」不是必要条件 —— 反例把 −2.8% 放大成 −21%**
    （2026-09-05 syncv3 grouped fp8 tensorwise NT 非持久，8 wave/WG、2 waves/SIMD、MfmaUtil 67-71%）：
    该核**最大程度违反②** —— 8 个 wave 被每 K-iter 8 条 `s_barrier` 锁在同一批 mfma group 上，
    人人都有 MFMA run、没有任何可错相的兄弟，正是本卡记的那个「对称推广到 GEMM1 = 噪声」的形态。
    但**拿掉包 mfma group 的那 22 对 setprio = −20.4…−23.9%，16/16 读数(双序 × 双 draw)、xchk 逐位 0.0**。
    ⇒ **它在这类核上的作用不是（或不只是）硬件仲裁提示，而是 de-facto 调度锚点**：ISA 铁证是删掉后
    编译器整体重排 —— `s_nop` 262→**163**、`v_*` 627→**652**、`vgpr_count` 256→**248**、
    `vgpr_spill_count` 1→**0**、`private_segment` 8→**0**，而 mfma/ds_read/store/s_barrier **逐项不变**。
    即"少 530 条指令 + 消掉唯一的 spill + 降 8 个 VGPR"仍然净亏 21%，代价全在被打乱的软流水上。
    ⇒ 处置：**②只用来预测「加 setprio 能不能赚」，绝不能用来判「拿掉它安不安全」**；
    任何 MFMA-serialized 的手工流水核，setprio 要按"结构件"对待，删它必须与重排后的 schedule 一起定价。
    这也解释了本卡下一条「交织度与吞吐负相关」为什么会发生：setprio 变的是**调度**，不是交织度。
  - **⚠ 交织度指标与吞吐负相关，别拿它当判据**：拿掉 setprio 后 ISA 的 exp↔MFMA 交织**变多**（32 条 exp 里 ±3 指令内有 MFMA 的从 31 变 28），成绩反而 −2.8%。
- **MFMA-issue 高 + TFLOPS 低 ≠ scheduler 没问题**（经典陷阱）：动调度旋钮前先 cross-check ATT stall trace 分类，判据见 methodology/03-profiling-utilization.md。
- **GLOBAL_/SCRATCH_ 别用 FLAT**：地址可证明只落在单一 aperture 时，emit `GLOBAL_*` / `SCRATCH_*` 而非通用 `FLAT_*`。FLAT 付 aperture-decode 税，**且同时 double-count VM_CNT 和 LGKM_CNT**，害了 s_waitcnt 调度。
- ❌ 别再试 blanket `s_waitcnt 0`：只 fence 下一个 consumer 真正需要的那个 counter（vmcnt / lgkmcnt / expcnt），一把清零会白等其他 counter。
- **⚠ `s_waitcnt` 的**条数**不是代价口径，别把"少等几条"当收益去优化**（2026-08-19 gpt-oss D64 fused bwd 实测）。热循环 332 条 `s_waitcnt`（其中仅 16 条 vmcnt，其余全 lgkmcnt）看着像个池子；把 GEMM2 的 dt 预取环从 depth 1 加深到 2，ISA 如预期**少了 26 条 `s_waitcnt`**，wall 却稳定 **+1.9%**（6/6 回文对全负，单对 +1.2~2.7%）。机制：等待被挪走的同时读突发变宽，顶开了 MFMA 发射。⇒ 该看的是**相邻两次等待之间的 MFMA 连跑长度**（issue 密度），不是等待计数；同一个 depth=2 在 D128 上是部署值，说明这是 per-shape 的，别跨 head-dim 搬。
  - **⚠ 订正（2026-08-19 同一 kernel 复测）：上一行"看 MFMA 连跑长度"这个替代判据也不成立，正确的量是 `ds_read` 的发射→退休覆盖距离。** 实测：手搭 `sched_group_barrier` 流水 `"m2,t2,v4"` 把连跑长度从 1.56 抬到 **1.83**、gap 数 820→699、`s_nop` 499→415 cyc/trip、指令 −54、vgpr 同 granule ——**四个静态量全部变好，wall 仍 +0.80%**（6 轮位置均衡轮转，2/6 胜）。真正与 wall 单调对应的是**每条 LDS 读提前于它的 `s_waitcnt` 多少条指令**（按 lgkmcnt 语义模拟队列退休即可算，见 Primus-Turbo `_isa_run.py`）：部署 32.4 → `"m2,t2,v4"` 26.4 → `"r1,m2,t2,v4"` 19.7，wall 依次 0 / +0.80% / **+3.48%**（0/6，散布仅 0.6%），单价约 **0.2~0.3% 每条覆盖指令**。
  - **推论（比上面更硬）：occ=1 时 `s_nop` 不是一个可加的池子。** `"r1,m2,t2,v4"` 抽掉 150 nop cyc = trip 的 2.7%，**一分钱没兑现**（反而 +3.48%）。因为一个 wave/SIMD 上 hazard nop 与它旁边的 lgkmcnt stall 在很大程度上**是同一次 stall**，静态分析把它数了两遍。所以"热循环有 N cyc 的 s_nop"不能当作 N cyc 的可回收量去立项。
  - **在 `sched_group_barrier` 流水里点名一个指令类 = 把它钉住。** 想让读提前，不能写 `r1`（那会每组塞一条读、正好钉在消费点旁边，覆盖掉到 19.7、8 条内退休的比例 8%→29%），要写一个**大于区域读数的领头组**（`"r24,m32"`）。
  - **同族对照：`iglp_opt` 与 `sched_group_barrier` 是同一个 LLVM mutation，换掉它本身就要付钱。** `"r24,m32"` 保住覆盖（32.8）且指令 −31，仍 +0.90%，因为 iglp_opt(2)(MFMAExpInterleave) 还在藏 78% 的 exp 链（恒等替换探针给该链定价 4.05%）。所以任何手搭流水的臂**起手就落后 0.9%**。另：策略 0 在本 body 顶穿寄存器（vgpr 476、nop 630 cyc），策略 1 在带 ds_write 的区域直接触发 `AMDGPUIGroupLP.cpp` 断言（不可达）。
- **waves_per_eu 无法经 `gpu-module-to-binary opts=` 生效**（已知限制）：必须设成 LLVM function attribute，或经 `rocdl-attach-target`。autotune 的 `Config.num_warps` / `waves_per_eu` / `maxnreg` 属编译器级特殊选项。
- **FP8/BF8 正确性依赖**：`SH_MEM_CONFIG` bit[8] 必须为 1（两代都要），否则 FP8/BF8 结果错。
- **CVT_*_F32 up-convert 无 4-cycle forwarding**：两个 convert 写同一目标寄存器的不同 byte/half 之间，必须插一个 NOP 或不相关的 VGPR write，否则读到 stale bytes。
- **FP16_OVFL 是 MODE bit**（saturate vs NaN/Inf），不是 per-instruction：切换它会影响 wave 后续**每一个** convert。

## profiling 坑：ATT 无 cache counter/PMC 多 pass 挂 GPU、code.json AGPR-blind、debug-info 假象

### ATT vs PMC 分工（不能合一个 job）
- ATT (Advanced Thread Trace) 只给**逐指令 stall 时序**，**没有 cache counter**。要问 L2 命中率 / 32B-partial / over-fetch / HBM 效率，必须**单独跑 PMC**——PMC 和 ATT 不能塞进同一个 job。
- PMC 不需要源映射，可以保留 `FLYDSL_RUNTIME_ENABLE_CACHE=1` 提速。

### ❌ 别再试：多 counter 一个 PMC job（gfx942 挂 GPU）
- PMC 每个 job 必须保持**单硬件 pass**（≤ ~4 个 TCC counter）。把多个 counter 塞进一个 job 会强制 **multi-pass 收集**，在 **gfx942 实测触发 GPU Hang (HW Exception)**。
- 正解：拆成多个**单 pass job**（如 L2 组 和 EA 组 分开跑）。

### ATT 常见错误处置
- 空 `ui_output_agent_*` → `kernel_include_regex` 没匹配上，重查 kernel 名。
- `no source mapping` → 确认 `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`。
- trace 截断 → `att_buffer_size` 升到 `0xC000000`。
- `INVALID_SHADER_DATA` → aqlprofile / decoder 版本不匹配，需同时更新。
- `iteration_range` 不匹配 → 试 `"[0,[1-2]]"`。
- ~~❌ 别再试：本容器直接跑 ATT~~ —— **此条已作废（2026-08-04 解决）**。当时判定「基础设施缺失」的依据只是 `find / -iname "*trace-decoder*"` 为空，而 decoder 是可以单独装的：下 `rocprof-trace-decoder` 0.1.6 的 wheel 取出 `.so` 放到 `/opt/rocm/lib/`（装法见 connection/common/05-reference-misc.md），mxfp4 grouped campaign 就是这么把 ATT 打通并拿到逐指令 stall 的。**教训：「库不存在」只说明没装，不等于装不上；把「未探索」写成死路会让后来者退回精度低得多的 PMC 聚合 + 减法探针。**

### ❌ 别再试：靠 code.json 反汇编算占用率（AGPR-blind）
- `code.json` 只含**单 CU、常是 vgpr-form 的反汇编**，无法给出 accum_vgpr / LDS / SGPR / workgroup size。**AGPR-form-blind 的 ISA 扫描会报 `accum=0`**，从而占用率算错。
- 正解：读旁边 staged 的 `out_kernel_trace.csv` 拿权威 `Accum_VGPR_Count` / `LDS_Block_Size` / `SGPR_Count` / `Workgroup_Size`；`arch_vgpr` 取 `max(ISA_scan, CSV)` 防 CSV 低报。
- ⚠️ **`*_counter_collection.csv` 里的 `VGPR_Count` 是 granule 计数，不是寄存器数**（实测同一 kernel：ISA
  reg-note `vgpr_count=246` → 该列报 **124**）。spill / 寄存器预算门禁**只认 ISA reg-note**
  （`vgpr_spill_count` / `private_segment_fixed_size`），别拿 PMC 那一列当门禁或写进报告。

### ❌ 别再试：让 VALU 直接吃 AGPR 以消掉 epilogue 的 `v_accvgpr_read`（gfx950 编不出来）
- 诱人的算术：把累加器钉在 AGPR 的核，epilogue 每个 f32 都要先 `v_accvgpr_read_b32` 搬进 VGPR 才能转换。
  mxfp4 grouped NT 的 `cst_wide` 是 **4 条 read + 2 条 `v_cvt_pk_bf16_f32`** 出 2 个 dword ⇒
  **每 tile 每 thread 256 条 accvgpr_read**，和 r15 刚删掉的 256 条 `accvgpr_write` 一样大（那笔值 +0.99%）。
  若 VALU 能直接读 AGPR，这 256 条整块消失。
- **实测（`llvm-mc -arch=amdgcn -mcpu=gfx950`）：VALU 一律不接受 AGPR 操作数。**
  `v_cvt_pk_bf16_f32 v0, a1, a2` / `v_add_f32 v0, a1, v2` / `v_mov_b32 v0, a1` /
  `v_pk_add_f32 v[0:1], a[2:3], v[4:5]` / `v_pack_b32_f16` **全部 `invalid operand for instruction`**。
- ✅ **但访存指令可以直接吃 AGPR**（同一次 llvm-mc 全部通过）：
  `buffer_store_dwordx4 a[0:3], v0, s[0:3], 0 offen`、`global_store_dwordx2 v[0:1], a[2:3], off`、`ds_write_b64 v0, a[2:3]`。
- ⇒ **判据**：输出**不需要格式转换**（fp32 C、或直接落 LDS/global 的 split-K 工作区）时，
  可以让 store 直接从 AGPR 发，省掉整块 `accvgpr_read`；**输出要 cvt（bf16/fp16/fp8）时这条路是死的**，
  256 条 read 是该 ISA 上的结构下限，别再规划它。

### ❌ 别再试：只加 `-g` flag 想拿 ATT 源码映射
- 光有 `gpu-module-to-binary` 的 `-g` flag 没用：`-g` 只保留 debug info 但**没东西可保留**——`loc()` 元数据在 MLIR→LLVM-IR 翻译时被**静默丢弃**。
- 正解：先跑 `ensure-debug-info-scope-on-llvm-func{emission-kind=LineTablesOnly}` pass（位置在 `reconcile-unrealized-casts` 之后、`gpu-module-to-binary` 之前），把 MLIR `loc()` 转成 LLVM `DISubprogram`/`DICompileUnit`。配好后 PA decode kernel 达 **99.9% 覆盖（1109/1110 指令）**。
- 环境变量要求（`FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`）见 methodology/03-profiling-utilization.md。

### ⚠️ ISA dump 里多个 kernel 首尾相接、没有分隔符（跨核误算，2026-08-12 mxfp4 grouped）
- `21_final_isa.s` 这类 stage dump 把同一 module 的所有 kernel 顺序拼在一起：mxfp4 grouped 的 preshuffle
  `kern_0`（约 324 行处 `s_endpgm` 收尾）后面**直接接** GEMM `kern_1`，中间**没有任何分隔标记**。
- 后果：任何 `grep s_waitcnt` / 数 store / 数 MFMA 的**全文件**脚本都会把前一个 kernel 的收尾指令算到后一个头上
  —— 实际踩到过"报 GEMM 开头有 4 条 store 在排空"，那 4 条属于 preshuffle。
- 正解：先按 `.amdhsa_kernel <name>` / `<name>:` 标签把文件**切成每核一段**再统计；报告里写明统计的是哪一段。

### ❌ 别再试：把 @flyc.kernel 装饰器行当热点（debug-info 假象）
- ATT 中热点若**塌陷到 `@flyc.kernel` 装饰器行**、且 stall 类型是**混合 VMEM-wait + barrier**（Pattern5）——这是 **debug-info 聚合假象**：MLIR/编译器生成指令（地址算术、cndmask、prologue）被映射到最外层 scope 行。
- 正解：忽略此行，只看**有显式用户 op** 的行。

---
来源: optimization-directions.md, gemm/overview.md, gfx950/kernel-implementation-notes.md, flydsl-kernel-authoring/SKILL.md, gfx942/kernel-implementation-notes.md, capture-kernel-trace/SKILL.md, kernel-trace-analysis/SKILL.md, programming-model.md

---

## ★★ rocprofv3 超时挂死:先去**远端**找 rocprofv3 孤儿,不要做 GPU reset

踩证(2026-08-27,n02-29 / 容器 kyle_attn):rocprofv3 对任何树、任何 `HOME` 都 rc=137 挂死,
一度被归因成"内核态 counter session 没释放,要 GPU reset"。**归因是错的。**
容器内 `ps -eo pid,etime,args | grep rocprofv3` 查出两组挂死的实例:
一组是某场 campaign r15 的探针(3h50m),另一组 `--pmc FETCH_SIZE WRITE_SIZE TCC_HIT` **挂了 33.8 小时**,
比当时所有活着的 campaign 都早。`kill -9` 后**立即恢复**(两 counter 小 grid rc=0、CSV 有数据),
未做任何 reset,顺带释放了一块 GPU 的 8.7 GB。

⇒ 判据:**多 counter rocprofv3 挂死 ≠ 硬件/驱动状态坏了**,先按显式 PID 清远端孤儿。
⚠ **清场必须两端都查**:campaign 的 agent 通过 `campaign_remote.py` 发起的 profiling,
本地只剩一个 ssh 驱动进程,**真正的 rocprofv3 在容器里**。只枚举本地 PID 和本地 `/proc/*/cmdline`
(哪怕按 campaign 目录名 grep)**看不到它们** —— 那批孤儿就是这样活了 33 小时,
并在此期间占着 GPU、让后续所有 profiling 挂死、还让另一场 campaign 报 `all pool GPUs busy`。

---

## ★★ rocprofv3 的 CSV 是**每硬件 block 一行、每行都已是 dispatch 全量**——要取均值,不是求和

踩证(2026-09-11,n02-29 / kyle_attn / gfx950,MXFP4 dense NT campaign):
用 `rocprofv3 --pmc SQ_WAVE_CYCLES ...` 做 K-differencing 解每-tile 固定成本,
第一版 parser 按"同一个 dispatch 的多行 = 分片,累加"来聚合,解出来的数**大 ~112 倍**,
而且固定项出现**负值**(物理上不可能)。原因:这台机器上 rocprofv3 给一次 dispatch 吐
**每个硬件 block 一行**(SE/CU 分组),而**每一行携带的已经是整个 dispatch 的聚合值**,
不是该 block 的分片。求和 = 把同一个数乘以行数。

⇒ 聚合规则:**按 `Kernel_Name` + dispatch 过滤后取行的算术均值**。
⇒ 自校验的办法(强烈推荐,比事后怀疑便宜):在同一次解算里放一个**你解析上已知答案的
counter**。这里用 `SQ_INSTS_MFMA`——它只随 k-block 数变、每-tile 固定项**必须为 0**。
均值口径下解出 FIXED = 0.0、PER-PHASE = 512(与 emit 的 128 cell × 4 完全对上),
求和口径下则是一堆脏数。**没有这种"已知零"的探针就不要相信 PMC 解出来的分解。**

### 配套方法:K-differencing 解"每-tile 固定成本 vs 每-k-block 边际成本"

固定 M、N(从而**固定 tile 数**),只改 K,跑两个 KI 值,解
`C = tiles * (F + KI * P)`。因为 tile 数不变,`F` 就是每-tile 的一次性成本
(prologue + head + peel + 该 tile 的 dispatch 摊销),`P` 是每个 k-block 的边际成本。
比"扫 grid 大小"干净,因为不动 tile 数就不动 XCD 分布和尾波形状。

### gfx950 上能用的 SQ 等待类 counter 只有三个

`SQ_WAIT_ANY` / `SQ_WAIT_INST_ANY` / `SQ_WAIT_INST_LDS`(外加 `SQ_ACTIVE_INST_*`、
`SQ_BUSY_CYCLES`、`SQ_WAVE_CYCLES`、`SQ_INSTS_*`)。`SQ_LEVEL_WAVES` **恒读 0**,别拿它算
occupancy。想区分"等 barrier/waitcnt" vs "等发射槽",只能靠
`SQ_WAIT_ANY`(含 waitcnt/barrier)与 `SQ_WAIT_INST_ANY`(等指令发射)两者相减去推。

## ★ 补充(2026-09-11):rocprofv3 `--kernel-trace` 在**单 GPU 的 MoE stage2 driver 上一样挂死**

原卡把"rocprofv3 挂死"记在**4-rank 跨 rank 自旋** workload 上。本轮把 GEMM2 单独摘出来、
只用一张卡、只开 `--kernel-trace`(最轻的模式),仍然:
12 s 后打印 `rocprofv3 caught signal 6`,然后**不退出**;`timeout 600` 发的 SIGTERM 被无视,
22 min 后进程仍 100% CPU。只能按 PID 手工 kill(本次 4 个:驱动 python、rocprofv3、两个子进程)。

⇒ 结论要扩大:**这台机器上 rocprofv3 对 flydsl/aiter 的 MoE kernel 就是不可用**,
与 rank 数、与是否有跨 rank 自旋无关。替代:torch profiler(探针 `--profile N`)或
`probe_isa.py` 从 flydsl runtime cache 的 code object metadata 读静态资源。
⚠ 另外:排队时 `rocm-smi` 看到 GPU3 100% 是**别的租户**的 —— 别把它当自己的信号。

### ⚠ rocprofv3 挂死是**会话级**的,不是装坏了 —— 先查占用再放弃 (GLM-5.2 EP4 r1)

`rocprofv3 --kernel-trace --stats --output-format csv -d DIR -o NAME -- python app.py`
在容器 `kyle_sglang` (smci355-ccs-aus-n02-29) 里 99% CPU 自旋、GPU 全程 idle、
>6 min 无任何输出文件。换 graph-free 驱动一样挂,换成 trivial 的 `a @ a` bf16 matmul 也挂
⇒ 当时判断成"这个容器里 rocprofv3 整体不可用"。

**这个判断是错的。** 同一个容器、同一个 rocprofv3 1.1.0
(`git_revision fc0010cf`),**同一天 14:27–14:28 另一场 sparse-MLA campaign 正常产出了
`/tmp/smla/pmc/r_counter_collection.csv` 等真实 counter 数据**;只有 16:2x 我那几次调用挂住。

⇒ 根因是**会话/占用级**而非安装级。PMC counter 采集要求对硬件计数器的独占访问,
同容器里只要还有一个活着或残留的 rocprof 会话(或邻居在同一张卡上采集),
新会话就会静默地无限等锁 —— 表现完全是"自旋无输出",没有任何报错提示你去找锁。

**进容器要 profile 时的正确顺序**:
1. 先 `ps -eo pid,etime,args | grep "[r]ocprof"` 查残留会话,有就按 pid 清掉
   (用 bracket trick,`pkill -f rocprof` 会连自己的 wrapper 一起杀,见 `pitfalls/02`)。
2. 确认自己被指派的卡上没有别人在采集(本机 2/3/6/7 是别人的)。
3. 拿 trivial matmul 试一次,10 s 内没输出就回到第 1 步,别在真 workload 上耗 10 分钟。
4. 真拿不到再退到下面的替代方案 —— **但不要把"我这次挂了"记成"这台机器不能 profile"**。

替代方案(GLM-5.2 EP4 r1 实际用的,PMC 缺位时仍足够定位):
- 逐 kernel GPU 时间:`torch.profiler` + `ProfilerActivity.CUDA`,按 kernel name 聚合
  (注意 `methodology/03` 的"同名多形状必须求和不能取中位数")。
- 带宽/固定开销拆账:`methodology/03` 的 `t = F + n·P` 拟合 —— 本轮用 7 个 token 档
  拟合 `us = 21.49 + 3.1122×sort_blocks`,截距 21.49 µs 独立复现了 profiler 的
  22.63 µs aux 合计,两路互证后才敢用。
进容器第一件事先拿 trivial matmul 试一次 rocprofv3,别在真 workload 上耗 10 分钟才发现是环境坏的。

### ★★ kernel trace 的**时长**不是该 kernel 的**边际代价**——要定价"少一次 dispatch 值多少",用重复启动法
(2026-09-11 GLM-5.2 EP4 r5 实测,gfx950/MI355X,CUDA graph replay)

上一条那个 `t = F + n·P` 的截距 21.49 µs,本轮拆到了 kernel 级:b64 的 6 个核里 4 个小核
(p0v2 4.56 / p23 4.87 / quant1 6.35 / quant2 5.47 µs)合计 21.3 µs,**与截距逐位对上**。
但这 4.5-6 µs **不是工作量**:
- p0v2 的 T 从 576 翻到 1152(b64→b128),时长 4.56→4.72 µs,**翻倍工作量 +3.5%**;
- 把它的 grid 从 258 降到 65、把 mesh 的 store→load HBM 往返删掉、把串行化 topk_ids 的
  mask 载入移出关键路径,**三次改动对时长的影响都是 0±0.1 µs**;
- 用字节数反推 quant 核:`t ≈ 4.9 µs + bytes/4.5 TBps`(quant1 10.5 MB/6.35 µs vs
  quant2 3.6 MB/5.47 µs 两点定出来的),即**约 4.9 µs 与字节数无关**。

⚠ **但"时长 4.5 µs"≠"删掉它能省 4.5 µs"**。定价要用**重复启动法**:把 p0v2 原地多启动一次
(3 行 dispatch 改动),b64 189.2→191.7 µs、b128 223.3→225.9 µs ⇒ **一次 kernel 边界的边际
代价 = 2.5 µs**,只有 rocprof 报出的时长的 ~55%。差额是 dispatch/drain 与相邻核重叠的部分,
rocprof 把它算进了时长里。⇒ 任何"合并 kernel 省一个核的时间"的估算,**先用重复启动法拿到边际
价,再决定要不要动手**;按 rocprof 时长估会把收益高估约 2 倍。

配套负结果(同轮,值得记住机制):让 p23 在 LDS 里自建 mesh+histogram 从而整核删掉 p0v2,
**功能正确**(rel_l2 不变)但净亏 —— p23 4.87→10.67 µs。原因是 p23 有 E=258 个 block,
每个 block 都要重建 E×T 的 mesh 才能独立推出同一份 prefix sum;**这份复制税(5.8 µs)
超过它省下的 2.5 µs 边界**。⇒ "把生产者核折进消费者核"只在消费者**不需要全局视图**时才划算;
需要全局 prefix sum 的,复制税按 O(E×T)×blocks 记,先算再写。
