# SRD/寻址坑：readfirstlane 必须 pin SGPR、大-G int32 溢出静默错、sched_barrier load-bearing

> 类别: 踩过的坑 · 主题标签: SRD寻址, int64溢出, waterfall, LDS-sync, OOB

- **readfirstlane 必须 pin SGPR（waterfall 陷阱）**：任何 per-group scan 派生的值（`m_start`、`group_idx` 等）若直接拿去构建 SRD base，divergence analysis 会判为 VGPR → 触发 waterfall（对每个 lane 值串行化循环）。必须显式 `_readfirstlane_i32(base)` 把它 pin 到 SGPR。WHY：SRD base 要求标量，VGPR base 让编译器插 waterfall loop 逐 lane 展开，串行化整个访存。

- **大-G MoE 累积偏移 int32 静默溢出**：persist wgrad 里 `m_start*OUT_M`，当 `G=256 / M_g=1536 / OUT_M=8192` 时 `m_total*OUT_M = 3.2e9 > 2^31` → 最后几组梯度**静默出错（无报错、无 crash）**。修法：把 `m_start*OUT` 折进 **i64 SRD base**；`num_records` 用 per-group `M_g*OUT`（**不要**用累积的 `m_end`，那才是溢出源）。

- **sched_barrier(0) before-mfma 是 load-bearing 的 LDS sync**：它在小-K 场景承担 LDS 同步职责，**删除会导致小-K 正确性坏（SNR 坏）**。❌ 别再试：把它当作纯调度 hint 删掉/移位来"清理"代码——它是隐式 barrier，删了小-K 出错。

- **OOB 修法优先级：修不变量 > 加 mask**。按根因对症，不要一律糊 mask：
  - 循环 trip count 错 → 减循环次数 / 减向量宽
  - per-lane 所有权变了 → 同步更新 layout + LDS store + reader 三处
  - 边界 partial tile → clamp / predicate
  - descriptor 范围太大或 offset i32 溢出 → chunk buffer resource，或在 truncate 前加宽算术（i64）
  - 修后重跑：失败 shape + 一个相邻边界 shape。

- **create_buffer_resource 的 max_size 坑**：`max_size=True` 会 OOB 读垃圾。必须用 `max_size=False, num_records_bytes=...` 精确给范围。WHY：max_size 把 descriptor 范围拉满，越界读进相邻内存。

- **（相关 FlyDSL/gfx950 fp8 硬约束）**：`Vec.to(Float8E4M3FN)` 走 `arith.truncf`，后端不 lower；fp8 cast 用 `fx.rocdl.cvt_pk_fp8_f32`（2 f32→2 fp8/op）。`cvt_scalef32_pk8_fp8_bf16` 在 gfx950 报 `Cannot select` 不可用。

---
来源: 08-deadends.md, 04-tn-wgrad-kernel.md, 02-nt-fwd-kernel.md, oob-detection/SKILL.md, flydsl-sync/SKILL.md
