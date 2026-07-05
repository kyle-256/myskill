# AGPR/VGPR-form 是 occupancy-specific：4-wave 用 AGPR、8-wave 用 VGPR-form

> 类别: 踩过的坑 · 主题标签: AGPR, occupancy, vgpr-form, mfma

- **AGPR 是 occupancy-specific（实测核心结论）**：`Mfma16x16x128AGPR`（inline-asm `=a,v,v,0` 把累加器钉到 AGPR）对 **4-wave（occ=1）+5~13%**，但对 **8-wave（occ=2）−30~37%**。WHY：8-wave 下 AGPR 预算减半 → 要么掉 occupancy，要么 RF 互倒（register file 打架）。所以 **8-wave 不要 AGPR**，用编译器默认 VGPR-form（SSA）。
- **fp8 下 4w≈8w（±2.5%）**：4w 赢大 N 的 FFN gate、8w 赢小方阵；不像 mxfp4 那样 4w 碾压。选 wave 数要按 shape 分。

- ❌ **别再试 intrinsic+LLVM scheduler 追平手写 asm**：intrinsic+bundled LLVM scheduler **4500** vs 手写 asm **5401**——FlyDSL bundled LLVM 调度精度不足，手写 asm 打败 bundled scheduler。相关旋钮同时坏：`asm-AGPR`/`FP4_AGPR` 在 asm_mma 路径已坏（AccVGPR 仍 0）；`WPEHINT`/`MAXNREG`/`MaxNReg` hints 不在 FlyDSL JIT 传播（VGPR 不变）。

- ❌ **别再试上游 AGPR-pin commit 救 mxfp4**：上游 ROCm/FlyDSL AGPR-pin commit（**#714 aeb5afc**，作用于 fp8 4wave AGPR 原地累加、消 accvgpr-shuffle +5~13%）对 **mxfp4 4-wave 无帮助**。WHY：mxfp4 走 bareasm whole-loop，accs 已用 `=a` tied 原地累加、更彻底，accvgpr-shuffle 问题根本不存在。实测 **agpr1（5407/5322）≈ agpr0（5390/5348）** 噪声内相等。上游针对 SSA-lowered 路径，bareasm 无此路径。

- ❌ **别再试 `amdgpu-mfma-vgpr-form=false` 在生产 4-wave**：对生产 4-wave 内核**零收益**，ISA 逐字节相同（accvgpr_write=1 / read=256 / agpr=256 / vgpr=432 / scratch=0）。WHY：累加已由 `passthrough amdgpu-agpr-alloc=256` + `waves_per_eu=1` + `maxnreg` 强制进 AGPR，循环内 **0 条 accvgpr shuffle**（256 条 `v_accvgpr_read` 全在 epilogue、只发生一次）。探针"8164/512 per-MFMA accvgpr shuffle 是头号瓶颈"是**误诊**（来自旧版/8wave/agpr=False 变体）。

---
来源: remote-sync/SKILL.md, 05-dead-ends.md, 11-upstream-agpr-pin-moot.md, project_mxfp4_vgprform_deadend.md
