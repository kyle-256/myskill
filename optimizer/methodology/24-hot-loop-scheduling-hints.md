# 热循环调度提示:sched_mfma/dsrd/dswr/vmem 交错、gfx942 sync vs gfx950 async 调度

> 类别: 方法论 · 主题标签: scheduling, sched-hints, gfx942-vs-gfx950, hot-loop

## rocdl.sched_* 提示语义
`hot_loop_scheduler()` 在 MFMA 计算段与下轮 load 之间插 `rocdl.sched_*` 提示，引导编译器交错指令:
- `sched_barrier(0)` — 全调度屏障(禁止跨越重排)
- `sched_mfma(N)` — 发 N 条 MFMA
- `sched_dsrd(N)` — 发 N 条 ds_read
- `sched_dswr(N)` — 发 N 条 ds_write
- `sched_vmem(N)` — 发 N 条 buffer_load

这些提示控制 barrier 前发射多少条对应指令(在 barrier 语义之上做发射调度)。

## gfx942 同步 copy — 标准调度序列
序言与主循环显式排定，让 ds_write 与末段 MFMA 重叠:
- **序言**: `sched_dsrd(2)` + 两个 `sched_mfma(1)` 预载 a0
- **主循环每迭代**: `sched_vmem(1)` + `sched_mfma(mfma_group)` + `sched_dsrd(1)` + `sched_mfma(mfma_group)`
- **ds_write 放尾部**: `dswr_start = max(sche_iters - num_a_loads - 2, 0)`,让 ds_write 与末段 MFMA 重叠、赶在 barrier 前落地
- **末尾**: `sched_barrier(0)`
- WHY: 同步 copy 下 load→compute 有依赖，手排序列把 vmem/dsrd 塞进 MFMA 缝里、把 dswr 压到最后掩盖延迟。

## gfx950 async copy — 均匀铺散
用 `_build_scheduler()` 把 ds_read/VMEM 均匀铺到全部 MFMA:
- `dsrd_schedule = _build_scheduler(num_ds_load - dsrd_preload, mfma_total)`
- `vmem_schedule = _build_scheduler(num_gmem_loads, mfma_total)`
- 每个 `sched_mfma(1)` 之后发相应数量提示
- WHY: async copy 解耦了 load 与 compute，不需要手排预载序列，均匀铺散让内存指令覆盖整个 MFMA 段。

## barrier / waitcnt 配套(与调度提示相邻)
- `fx.gpu.barrier()` = `__syncthreads`(workgroup barrier)
- **CDNA3 (gfx942)**: `fx.rocdl.s_waitcnt(0)`(单一 waitcnt)
- **CDNA4 (gfx950)**: 分开 `s_wait_loadcnt(0)` / `s_wait_storecnt(0)` / `s_wait_dscnt(0)`

## hot loop 指令比例判据(ISA dump 复盘)
| 指标 | 好 | 可接受 | 差 |
|---|---|---|---|
| MFMA 比例 = MFMA/total | >40% | 30–40% | <30%(非 MFMA 开销过大) |
| 内存指令比例 = (ds_read+buffer_load+ds_write)/total | <40% | — | >50%(内存主导 → 试更大 tile_k 或减 load) |

- tile **64×256×128 FP8** 典型总指令 **~130–150**。
- WHY: MFMA 占比低 = 计算被非 MFMA 稀释;内存占比高 = feed-bound，加大 tile_k 摊薄或减少 load 次数。

---
来源: gemm-optimization/SKILL.md, flydsl-tile-programming/SKILL.md
