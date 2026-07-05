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

## 三类 race 速查表（可修，别当 HW-walled）

区别于上面的 HW-walled 死路，下面三类 race 是同步漏洞，**可修**：

| 类 | 冲突对 | 范围 | 触发 | 修法 | commit |
|---|---|---|---|---|---|
| 1 vmcnt FIFO | `scratch_load` vs `buffer_load_lds` | wave 内 | spill>0 单次即 race | 消 VGPR spill（SGPR 化地址 / outer 解析 ptr） | abd3833 |
| 2 LDS WAR | `ds_read`(lgkmcnt) vs `buffer_load_lds`(vmcnt) | 同 WG 跨 warp | spill=0 单次即 race | `wait_lgkmcnt<0>()` drain | bb48d3f |
| 3 die L2 | 复用 workspace 跨 die 跨调用 | 跨 XCD | 前两类干净、第 2+ 次调用 race | 入口一次 `__threadfence()` + per-tile 退化成 lgkmcnt drain | d7b149a |

### 第二类：LDS WAR race（commit bb48d3f）
- 现象：spill 全 0 但仍 race（~0.004%/tile，多 config 累积 30-50%/session）。
- 机制：`buffer_load_lds`（vmcnt-tracked LDS write）与 `ds_read_pinned`（lgkmcnt-tracked LDS read）目标同一段 LDS 时，`s_barrier` 只同步执行位置**不 drain lgkmcnt** → warp 的 `ds_read` 还在飞、`buffer_load_lds` 已 commit → 读到 post-write 数据污染 MFMA。
- 修法：两处发 `wait_lgkmcnt<0>()`：(1) `phase_mfma_lds_ldg` 的 mid-phase WAR barrier **之前**（原来只有 `s_barrier`）；(2) `compute_tile` prologue 里 `if(k_iters>2)` 块**之前**（原本 drain 在块之后、顺序错了）。

### 第三类：跨 die(XCD) L2 coherence race（commit d7b149a）
- 机制：MI355X 一 node=8 个 die(XCD) 每 die 一块 L2、**die 间不自动 coherent**，persistent kernel 256 个 block 散在 8 die 上跑。复用 workspace 时上次写 GMEM 的副本缓在某 die L2，这次 `buffer_load_lds` 命中 stale L2 行 → 污染。**只在复用 workspace 的第 2+ 次调用出**。
- 修法：kernel 入口发一次 `__threadfence()`（gfx950 lower 成 `buffer_wbl2 sc1` + `buffer_inv sc1` = flush+invalidate 本 die L2）。官方 deterministic ×100 全过。
- ❌ 别再试 per-tile L2 flush+inv：MoE 上 **14-40% tax**。拆两半：per-tile 只保留 `wait_lgkmcnt<0>()`（drain in-flight `buffer_load_lds` 的 GMEM→LDS 写，无 L2 流量）；per-block 一次在 kernel 入口 tile loop 之前发 `__threadfence()`（GEMM 输入只读，一次 invalidate 覆盖该 block 所有 tile）。fwd kernel 改后 dgrad 复用自动受益，wgrad 独立要单独改两处。实测拆分后 vs per-tile-fence baseline：Fwd **+10~85%** / Dgrad **+11~60%** / Wgrad **+33~87%**，det60 bit-exact 不回。

### race 定位法（跨 kernel + kernel 内子结构）
- **pytorch fp32 ref 替换**：python 层加 env-var dispatch，`PRIMUS_TURBO_MXFP8_<STAGE>_REF=1`（`_stage=fwd/dgrad/wgrad`）返回 fp32 dequant+matmul+cast 的 deterministic ref，矩阵化跑 deterministic 测试锁定 race 在哪个 C++ kernel（本 case 锁定 `turbo_grouped_gemm_mxfp8` fwd/dgrad 共享 + `turbo_grouped_gemm_mxfp8_wgrad`）。**commit 前必须删掉 ref 路径+env-var。**
- **补 single-GEMM 覆盖**：single GEMM mxfp8 之前无 deterministic 测试，补最小 deterministic 测试发现 single GEMM 也 race（~30% session fail）→ race 在 shared kernel structure（`phase_mfma_lds_ldg` / 单 GEMM `compute_tile`）而非 grouped 特有的 persistent-loop / per-group 解析；single GEMM 更小迭代更快，后续定位都用它跑。
- **gate 掉嫌疑块**：把 Epi1 LDG 块 gate 掉 `if(k_iters>999999)`，若 `assert_close` 全过（deterministic 恢复）但 SNR 失败（LDG 提供真正参与 MFMA 的数据）→ 确认 race 在该块，下一步是补 `wait_lgkmcnt<0>` 修同步漏洞而非删。
- **每改一处跑 30 runs 统计 session-level fail rate**：无 fix 30 runs ~10 fail(33%)；加 prologue drain ~1-3 fail(7%)；+WAR barrier drain 0 fail；再 50 runs 0 确认。race rate 0.004% 量级 1 run 可能运气过，至少 30 runs、更保险 50。

## FlyDSL / gfx950 fp8 相关 ISA 硬约束

- ❌ 别再试 `cvt_scalef32_pk8_fp8_bf16`：gfx950 **Cannot select**，不可用。
- ❌ 别再试 `Vec.to(Float8E4M3FN)`：走 `arith.truncf`，后端**不 lower**。
  - 正确做法：fp8 cast 用 `fx.rocdl.cvt_pk_fp8_f32`（2 f32 → 2 fp8 / op）。
- ❌ 别再试 `create_buffer_resource(max_size=True)`：会 **OOB 读垃圾**。
  - 正确做法：用 `max_size=False, num_records_bytes=...`。

---
来源: 05-dead-ends.md, flydsl-sync/SKILL.md, gfx950-vmcnt-race-debug/SKILL.md
