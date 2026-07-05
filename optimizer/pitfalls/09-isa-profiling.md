# ISA 调度提示与 profiling 陷阱：sched_* 计数、s_setprio、FLAT、ATT/PMC 分工、code.json AGPR-blind

> 类别: 踩过的坑 · 主题标签: sched-hints, s_setprio, s_waitcnt, FLAT, occupancy, rocprofv3, ATT, PMC, debug-info

## 调度提示/ISA 陷阱：sched_* 计数必须精确、s_setprio 要归零、别用 blanket s_waitcnt 0

- **sched_* 计数必须精确等于 body emit 的 opcode 数**：FlyDSL 的 `sched_mfma` / `sched_dsrd` / `sched_dswr` / `sched_vmem` hint 计数必须 EXACTLY 等于 body 里对应 opcode 的发射数量。一旦不匹配，backend 会**静默 drop policy**、退回默认调度，**无任何 warning**。改了 body 的 opcode 数一定要同步改 count。
- **sched_barrier(0) 是 load-bearing 且 ordering-only**：删掉它整个手工 schedule 就塌（LLVM 会跨迭代边界 coalesce）。位置错也不行——放进 constexpr loop **里面**（而非 loop 之后）会 pessimize overlap。schedule 必须跑在**拥有它所重排的那个 LDS write 的 `gpu.barrier()` 之后**。
- **s_setprio(1) 后必须归零**：MFMA block 前 `s_setprio(1)`、block 后立即 `s_setprio(0)`，防止 MFMA 被 VMEM/LDS 挤掉。❌ 别再试 忘记把它 DROP 回 0：会饿死 co-resident wave、直接砸 occupancy。
- **MFMA-issue 高 + TFLOPS 低 ≠ scheduler 没问题**（经典陷阱）：动调度旋钮前先 cross-check ATT stall trace 分类，判据见 methodology/12-att-trace-mfma-stall。
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

### ❌ 别再试：靠 code.json 反汇编算占用率（AGPR-blind）
- `code.json` 只含**单 CU、常是 vgpr-form 的反汇编**，无法给出 accum_vgpr / LDS / SGPR / workgroup size。**AGPR-form-blind 的 ISA 扫描会报 `accum=0`**，从而占用率算错。
- 正解：读旁边 staged 的 `out_kernel_trace.csv` 拿权威 `Accum_VGPR_Count` / `LDS_Block_Size` / `SGPR_Count` / `Workgroup_Size`；`arch_vgpr` 取 `max(ISA_scan, CSV)` 防 CSV 低报。

### ❌ 别再试：只加 `-g` flag 想拿 ATT 源码映射
- 光有 `gpu-module-to-binary` 的 `-g` flag 没用：`-g` 只保留 debug info 但**没东西可保留**——`loc()` 元数据在 MLIR→LLVM-IR 翻译时被**静默丢弃**。
- 正解：先跑 `ensure-debug-info-scope-on-llvm-func{emission-kind=LineTablesOnly}` pass（位置在 `reconcile-unrealized-casts` 之后、`gpu-module-to-binary` 之前），把 MLIR `loc()` 转成 LLVM `DISubprogram`/`DICompileUnit`。配好后 PA decode kernel 达 **99.9% 覆盖（1109/1110 指令）**。
- 环境变量要求（`FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`）见 methodology/03-profiling-utilization.md。

### ❌ 别再试：把 @flyc.kernel 装饰器行当热点（debug-info 假象）
- ATT 中热点若**塌陷到 `@flyc.kernel` 装饰器行**、且 stall 类型是**混合 VMEM-wait + barrier**（Pattern5）——这是 **debug-info 聚合假象**：MLIR/编译器生成指令（地址算术、cndmask、prologue）被映射到最外层 scope 行。
- 正解：忽略此行，只看**有显式用户 op** 的行。

---
来源: optimization-directions.md, gemm/overview.md, gfx950/kernel-implementation-notes.md, flydsl-kernel-authoring/SKILL.md, gfx942/kernel-implementation-notes.md, capture-kernel-trace/SKILL.md, kernel-trace-analysis/SKILL.md, programming-model.md
