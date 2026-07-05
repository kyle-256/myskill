# LDS 容量硬约束：gfx950=160KB/gfx942=64KB，3-stage 双缓超限净负

> 类别: 踩过的坑 · 主题标签: LDS容量, prefetch双缓, blockwise-vs-tensorwise, L2-thrash

- **LDS 容量按架构硬上限**：gfx942(MI300X)=64KB/CU；gfx950(MI350/MI355X)=160KB/CU，group_segment 实测 =163840B=160KB/CU。超容量报 LDS overflow → 用 `SmemAllocator` 追踪分配。

- **❌ 别再试：gfx950 3-stage LDS 双缓冲**。每 stage 64KB×3=192KB>160KB → 无法 2 wave/CU，净负。3-stage persist(`pipe3`)=0.87× persist，同样因 LDS 超限。

- **❌ 别再试：3-stage B ring buffer(fwd) 加 prefetch distance**。实测中性(+0.05%)。WHY：L2 miss latency 来自 8 个不同 expert 权重 thrash 4MB/XCD L2，是 **capacity thrash**，不是 latency 可隐藏问题 → 加 prefetch distance 无效，**prefetch 治不了 L2 capacity thrash**。

- **MI300 blockwise 硬约束（❌ 别再试绕过）**：
  - scale block=128 → BK=128 锁死（每 K 块一对 a_s/b_s），**不能像 tensorwise 选 BK=64**，结构性慢。
  - LDS=64KB → BK=128+BM=BN=256 单 stage 刚好，**双 stage 不可能**（LDS 放不下 → 失去 prefetch pipelining）。
  - MFMA nonkdim 实测 32 永远不赢，锁 16。

- **blockwise 打不过 tensorwise 的结构性根因**：
  - tensorwise 能 BM=BN=256 / BK=64 / num_stages=2（LDS=65536=64KB exactly）→ 享受 2-stage prefetch。
  - blockwise 因 BK=128，最大 256×128×128 stages=1 无 prefetch（FMA 后是空泡）。
  - 实测：大 shape blockwise ~0.65-0.75×tensorwise；小 shape 0.85-0.99×tensorwise。

---
来源: 08-deadends.md, flydsl-kernel-authoring/SKILL.md, mi300-blockwise-gg-tuning/SKILL.md
