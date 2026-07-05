# 死路：maxnreg 强制 accum_vgpr=0、AGPR 搬移救不了 VGPR 溢出

> 类别: 踩过的坑 · 主题标签: register-pressure, occupancy, MFMA, AGPR

- ❌ 别再试：用 `maxnreg` 强制 `accum_vgpr=0` 来给预取/arch-VGPR 腾寄存器。占用率会翻倍，但 MFMA 累加器被逼经 `v_accvgpr_read` 溢出到 arch_vgpr（arch-VGPR spills），实测 **~4.5× GPU kernel 回退**。MFMA-heavy kernel 绝不能用 `maxnreg`。**AccVGPR 压力只能付在 occupancy 上，无法规避。**

- 占用预算模型（gfx942/gfx950 合并 512-entry/SIMD 池，非 gfx908 的 256/max）：见 methodology/14-occupancy-vgpr-agpr-lds-budget.md

- **死坑：把 accs 搬 AGPR 救不了溢出**。CDNA occ=2 下 `ArchVGPR + AccVGPR` 共享 256 组合预算（`accum_offset 256`）。把累加器搬到 AGPR **不减少总量**，救不了 VGPR 溢出。
  - fp4 MFMA 有 5 个操作数 `(a, b, sa, sb, c)`；`sa`/`sb` 必须留 VGPR，`acc` 可移 AGPR，但 **V 不下降** → `V + A = 384 > cap`。
  - ❌ 别再试：8-wave BN512 BK128 实测 spill 到 Scratch，1132 → 374 TF（**13× 慢**，每个 MFMA 都读写 scratch = 打 HBM）。

---
来源: prefetch-data-load/SKILL.md, kernel-trace-analysis/SKILL.md, programming-model.md, 09-8wave-ceiling.md, lds-optimization/SKILL.md
