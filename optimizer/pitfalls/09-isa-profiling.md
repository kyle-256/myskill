# ISA 调度提示与 profiling 陷阱：sched_* 计数、s_setprio、FLAT、ATT/PMC 分工、code.json AGPR-blind

> 类别: 踩过的坑 · 主题标签: sched-hints, s_setprio, s_waitcnt, FLAT, occupancy, rocprofv3, ATT, PMC, debug-info

## 调度提示/ISA 陷阱：sched_* 计数必须精确、s_setprio 要归零、别用 blanket s_waitcnt 0

- **sched_* 计数必须精确等于 body emit 的 opcode 数**：FlyDSL 的 `sched_mfma` / `sched_dsrd` / `sched_dswr` / `sched_vmem` hint 计数必须 EXACTLY 等于 body 里对应 opcode 的发射数量。一旦不匹配，backend 会**静默 drop policy**、退回默认调度，**无任何 warning**。改了 body 的 opcode 数一定要同步改 count。
- **sched_barrier(0) 是 load-bearing 且 ordering-only**：删掉它整个手工 schedule 就塌（LLVM 会跨迭代边界 coalesce）。位置错也不行——放进 constexpr loop **里面**（而非 loop 之后）会 pessimize overlap。schedule 必须跑在**拥有它所重排的那个 LDS write 的 `gpu.barrier()` 之后**。
- **s_setprio(1) 后必须归零**：MFMA block 前 `s_setprio(1)`、block 后立即 `s_setprio(0)`，防止 MFMA 被 VMEM/LDS 挤掉。❌ 别再试 忘记把它 DROP 回 0：会饿死 co-resident wave、直接砸 occupancy。
- **s_setprio 的正负是 regime 决定的，别按 kernel 名搬结论**（2026-08-04 融合 hd64 flash bwd 实测；与 pitfalls/12 的 fwd 判负**相反**）。它唯一的作用是**给同一个 SIMD 上的两个共驻 wave 拉开相位**，所以三条同时成立才为正：①**有仲裁对象**(≥2 waves/SIMD；`waves_per_eu=1` 时恒中性，见 pitfalls/02)；②该 region **一个 wave 有 MFMA run 而它的兄弟没有**（GEMM2、carrier 的 GEMM3）；③body 是 **MFMA-pipe-serialized**（MFMA 占 ~47%）。齐了就是强正：拿掉包 GEMM2 的那一对 = **9/9 负 −2.8%**。反例都来自破坏②：把同一对**对称地**推广到 GEMM1（8 个 wave 全都有 MFMA run，没人可错相）= 7/11、6/11 噪声，**即使 hazard `s_nop` 198→102**；包 GEMM3 的 kstep ring = 6/11 且 spill +63 dword（setprio **不切 scheduling region**，却把 ring 的 live range 钉穿它）。
  - **幅度是 no-op，别扫**：全 wave 跑同一份码，比较恒为"in-region(1) vs not-in-region(0)"，1→2 改不了任何序（hd64 fwd 独立实测"提到 2 = 持平"，见 pitfalls/13）。
  - **⚠ 交织度指标与吞吐负相关，别拿它当判据**：拿掉 setprio 后 ISA 的 exp↔MFMA 交织**变多**（32 条 exp 里 ±3 指令内有 MFMA 的从 31 变 28），成绩反而 −2.8%。
- **MFMA-issue 高 + TFLOPS 低 ≠ scheduler 没问题**（经典陷阱）：动调度旋钮前先 cross-check ATT stall trace 分类，判据见 methodology/03-profiling-utilization.md。
- **GLOBAL_/SCRATCH_ 别用 FLAT**：地址可证明只落在单一 aperture 时，emit `GLOBAL_*` / `SCRATCH_*` 而非通用 `FLAT_*`。FLAT 付 aperture-decode 税，**且同时 double-count VM_CNT 和 LGKM_CNT**，害了 s_waitcnt 调度。
- ❌ 别再试 blanket `s_waitcnt 0`：只 fence 下一个 consumer 真正需要的那个 counter（vmcnt / lgkmcnt / expcnt），一把清零会白等其他 counter。
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
