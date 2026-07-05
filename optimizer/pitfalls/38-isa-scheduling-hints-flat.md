# 调度提示/ISA 陷阱：sched_* 计数必须精确、s_setprio 要归零、别用 blanket s_waitcnt 0

> 类别: 踩过的坑 · 主题标签: sched-hints, s_setprio, s_waitcnt, FLAT, occupancy

- **sched_* 计数必须精确等于 body emit 的 opcode 数**：FlyDSL 的 `sched_mfma` / `sched_dsrd` / `sched_dswr` / `sched_vmem` hint 计数必须 EXACTLY 等于 body 里对应 opcode 的发射数量。一旦不匹配，backend 会**静默 drop policy**、退回默认调度，**无任何 warning**。改了 body 的 opcode 数一定要同步改 count。
- **sched_barrier(0) 是 load-bearing 且 ordering-only**：删掉它整个手工 schedule 就塌（LLVM 会跨迭代边界 coalesce）。位置错也不行——放进 constexpr loop **里面**（而非 loop 之后）会 pessimize overlap。schedule 必须跑在**拥有它所重排的那个 LDS write 的 `gpu.barrier()` 之后**。
- **s_setprio(1) 后必须归零**：MFMA block 前 `s_setprio(1)`、block 后立即 `s_setprio(0)`，防止 MFMA 被 VMEM/LDS 挤掉。❌ 别再试 忘记把它 DROP 回 0：会饿死 co-resident wave、直接砸 occupancy。
- **MFMA-issue 高 + TFLOPS 低 ≠ scheduler 没问题**（经典陷阱）：这通常是 barrier 或 s_waitcnt stall，不是调度问题。动调度旋钮前先 cross-check ATT stall trace 分类。**Stall-bound 判据 = HBM BW 与 MFMA ratio 双低 + s_barrier/s_waitcnt 时间高**。此时调 scheduling knob 是白干。
- **GLOBAL_/SCRATCH_ 别用 FLAT**：地址可证明只落在单一 aperture 时，emit `GLOBAL_*` / `SCRATCH_*` 而非通用 `FLAT_*`。FLAT 付 aperture-decode 税，**且同时 double-count VM_CNT 和 LGKM_CNT**，害了 s_waitcnt 调度。
- ❌ 别再试 blanket `s_waitcnt 0`：只 fence 下一个 consumer 真正需要的那个 counter（vmcnt / lgkmcnt / expcnt），一把清零会白等其他 counter。
- **waves_per_eu 无法经 `gpu-module-to-binary opts=` 生效**（已知限制）：必须设成 LLVM function attribute，或经 `rocdl-attach-target`。autotune 的 `Config.num_warps` / `waves_per_eu` / `maxnreg` 属编译器级特殊选项。
- **FP8/BF8 正确性依赖**：`SH_MEM_CONFIG` bit[8] 必须为 1（两代都要），否则 FP8/BF8 结果错。
- **CVT_*_F32 up-convert 无 4-cycle forwarding**：两个 convert 写同一目标寄存器的不同 byte/half 之间，必须插一个 NOP 或不相关的 VGPR write，否则读到 stale bytes。
- **FP16_OVFL 是 MODE bit**（saturate vs NaN/Inf），不是 per-instruction：切换它会影响 wave 后续**每一个** convert。

---
来源: optimization-directions.md, gemm/overview.md, gfx950/kernel-implementation-notes.md, flydsl-kernel-authoring/SKILL.md, gfx942/kernel-implementation-notes.md
