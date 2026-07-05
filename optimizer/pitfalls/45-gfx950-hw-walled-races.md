# gfx950 HW-walled 死路：buffer_load_dwordx2_lds 不支持、SCVGPR prefetch racy

> 类别: 踩过的坑 · 主题标签: gfx950, race-correctness, LDS-load, SCVGPR-prefetch, flydsl-isa

## buffer_load...lds 直写 LDS 的同步死路（HW-walled）

- ❌ 别再试 `buffer_load_dwordx2...lds`：gfx950 **不支持** dwordx2 的 lds 变体，只有 `dword` 和 `dwordx4` 两种宽度。
- ❌ 别再试 用 vmcnt / 隔-phase barrier 同步 `dwordx4-lds` 直写 LDS 的完成：**不被可靠同步** → det≠0。
  - 唯一能 det0 的写法是 `vmcnt(0)` + 紧跟 `s_barrier`，但这会**序列化**，实测 **4803 < 5176**（更慢），得不偿失。
  - WHY：LDS 直写完成不保证在 vmcnt/barrier 观测点前落地。

## SCVGPR prefetch racy（HW-walled）

- ❌ 别再试 SCVGPR 的 `SCV2AHEAD` / `SCPF` prefetch：racy。
  - WHY：**vmcnt 乱序退役**，不保证特定 load 在某个时刻落地。

## BK128 SCVGPR scale VGPR WAR race（同一机制）

- ❌ 别再试 BK128 SCVGPR scale VGPR 写法：存在 WAR race。
  - 证据：K=256（0 main iter）SNR **55.6** 正常，但 K=384（1 iter）SNR **-inf** 崩。
  - 机制：phase-B 的 `emit_sc_vgpr(0)` → 写 `v[8:9]`，覆写了 phase-A mfma **还在读**的 scale VGPR。
  - `vmcnt(0)` **不能修**：vmcnt 乱序退役，不保证该特定 load 落地。
  - 与 **BK256 SCVGPR 同一机制**。

## FlyDSL / gfx950 fp8 相关 ISA 硬约束

- ❌ 别再试 `cvt_scalef32_pk8_fp8_bf16`：gfx950 **Cannot select**，不可用。
- ❌ 别再试 `Vec.to(Float8E4M3FN)`：走 `arith.truncf`，后端**不 lower**。
  - 正确做法：fp8 cast 用 `fx.rocdl.cvt_pk_fp8_f32`（2 f32 → 2 fp8 / op）。
- ❌ 别再试 `create_buffer_resource(max_size=True)`：会 **OOB 读垃圾**。
  - 正确做法：用 `max_size=False, num_records_bytes=...`。

---
来源: 05-dead-ends.md, flydsl-sync/SKILL.md
