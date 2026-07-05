# gfx950 vmcnt 非 FIFO race：scratch_load 与 buffer_load_lds 混排，只在 spill 时暴露

> 类别: 踩过的坑 · 主题标签: gfx950, vmcnt-race, spill, LDS-direct

## 根因
- **scratch_load（编译器 SGPR spill 回填）和 buffer_load_dwordx4...offen lds（LDS-direct GMEM→SMEM）都走 vmcnt，但两者之间 vmcnt 不保证 FIFO。**
- 编译器对 spill 发的 `s_waitcnt vmcnt(N>0)` 不安全——scratch_load 可能还在飞，`v_readfirstlane` 读进 stale value → `s_mov_b32 m0` 用 stale m0 → 下一条 buffer_load_lds 写到错 LDS 地址 → SMEM 错位 → MFMA 污染 → bit 不一致。
- **只在 SGPR pressure 触发 spill 时暴露**（无 spill 时不出现）。

## disasm 签名（如何确认是这个坑）
- `private_segment_fixed_size > 0`（有 spill）
- 主循环里 `scratch_load_dword` 夹在 `buffer_load_dwordx4...offen lds` 之间
- 紧跟 `s_waitcnt vmcnt(N>0)` + `v_readfirstlane` + `s_mov_b32 m0` + `buffer_load_lds`
- 若 `private_segment == 0` 但仍 race：看 SGPR 是否 spill 到 VGPR lanes（`v_writelane` / `v_readlane`）——lane ops 不走 vmcnt，不会 race。

## 修法（双管齐下，缺一不可）
**A) 降 spill：**
- `-mllvm -sink-insts-to-avoid-spills=true`（mxfp8 grouped 实测 `private_segment` 44→20B）
- CK pattern：把 per-group `readfirstlane` / 指针计算从 inner 提到 outer
- smem packing 传单 `char*` 指针
- 保留 `tile.reserve_pinned_regs()`
- ⚠️ 任一处去掉，race 立即 100% 恢复。

**B) prologue 2-step：**
- buffer_load_lds 拆成 `buffer_load → VGPR → ds_write`（走 lgkmcnt，与 spill 的 vmcnt 错开通道）
- **只在 prologue**，末尾补 `wait_lgkmcnt<0>()` + `barrier`
- 配 `-mllvm -amdgpu-enable-merge-m0=true`
- wgrad 只让 **8 个 data load** 走 2-step（全 16 个 → -25%），scale load 保留 LDS-direct。

## 死胡同（不能同时拿 0 race + PR HEAD perf）
- ❌ 别再试：`__noinline__ compute_tile` → **-25%**（args 通过 scratch 传，反而制造更多 spill）
- ❌ 别再试：全 main loop buffer_load_lds→2-step → **-85%**（破坏 phase_mfma_lds_ldg overlap）
- ❌ 别再试：mid-iter 主循环加 `s_waitcnt vmcnt(0)` → race **不降 0** 且 **-21%**
- ❌ 别再试：FIRST_2STEP（每 phase 第一个 load 2-step）→ race **不降 0** 且 **-30~40%**
- ❌ 别再试：prologue 多个 `vmcnt(0)` drain / mode-switch 藏 spill → 被 reviewer 拒（没修根因）

---
来源: gfx950-vmcnt-race-debug/SKILL.md
