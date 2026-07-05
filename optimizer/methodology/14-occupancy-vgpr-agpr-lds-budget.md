# 占用率计算:512 寄存器文件 VGPR+AGPR 共享、LDS/SGPR limit、alloc 粒度

> 类别: 方法论 · 主题标签: occupancy, vgpr-agpr-budget, lds-limit, co-saturation

## 占用率公式

- **occ(waves/SIMD) = min(vgpr_limit, lds_limit, sgpr_limit, hw_max=8)**,取最小的那个资源决定占用。
- `vgpr_limit = 512 // (arch + accum)` — gfx950 每 SIMD **512 个寄存器,VGPR+AGPR 共用同一 512 文件**。每 wave 用 R 个总寄存器 → `512 // R = waves/SIMD`;`×4 SIMD = waves/CU`;`waves/CU ÷ (waves/wg) = wg/CU`。
- `lds_limit = (LDS_total // lds_per_wg) * waves_per_wg // 4`
- `sgpr_limit = 800 // sgpr_alloc`
- **alloc 粒度:VGPR 8 个一档**。

## Worked example (PA decode gfx942)

- arch 144 + accum 136 = **280 合并** → `512 // 280 = 1 wave/SIMD`,**VGPR-bound**(LDS 允许 5,SGPR 允许 7)。
- 要到 2 waves 需合并 **≤256**,即释放 **~24 VGPR**。

## Worked example (gfx950 MI355X LDS-bound)

- LDS 160KB/CU,某内核 144KB/wg → **1 WG/CU** → 4 waves/CU ÷ 4 SIMD = **1 wave/SIMD(occ=1)**,LDS-bound。

## 权威数据源:只信 CSV,不信 analyzer

- 判 occupancy 用**真实 CSV**(`out_kernel_trace.csv`)的 `LDS_Block_Size` / `VGPR_Count` / `Workgroup_Size`,**不信 analyzer 推断**。
- ISA 里 `v_mfma` 引用 `a[...]` 寄存器会被 analyzer 误当 agpr-form → 误报 'bound by VGPR';**只有 CSV `Accum_VGPR_Count=0` 才权威**(证明在 in-AGPR 累加,非误报)。
- 实测占用判据用 rocprofv3 的 **`MeanOccupancyPerActiveCU`**(waves/SIMD),**不用** rocprofv3 的 `VGPR_Count`(不可信)。

## co-saturation(联合饱和)

- **Joint co-saturation**:联合选 tile shape + MFMA 指令宽度 + waves/SIMD,让 **VGPR+AGPR 预算、LDS footprint、matrix-unit issue rate 同时饱和**,而不是某一资源先饿死其他资源。
- 对编译后 AMDGCN 做 **VGPR liveness pass**(找 dead VGPR 窗口、把 boundary 以上的寄存器 remap 进空洞)可在下一个占用边界抬升 waves/SIMD。
- `next_free_vgpr` 步进边界:**64 / 73 / 85 / 102 / 128 / 170 / 256**。

## Occupancy-starved 直接信号:tile 数 < CU 数

- **tile 数 < CU 数** = occupancy-starved 的直接信号。
- 实例(dgrad B=1 grok-up M=512):fwd(N=32768)=**2793TF 满载**;dgrad(N=8192)=**923TF 欠载**。output tile 数 = `G × ceil(M/BM) × ceil(K_fwd/BN)` = **64 tile 只填 64/256 CU**。
- 修法(tile 尺寸选择/BM128 gate 细节):见 methodology/20-tile-size-selection.md。

---
来源: 08-att-root-cause.md, kernel-trace-analysis/SKILL.md, gfx950/kernel-implementation-notes.md, agpr_phase5_lds.md, diag_4w_vs_8w.md, 03-nn-dgrad-kernel.md
