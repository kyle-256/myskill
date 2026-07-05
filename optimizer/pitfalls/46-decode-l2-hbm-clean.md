# streaming decode L2 命中低是正常：memory 子系统干净无 KV-load 优化空间

> 类别: 踩过的坑 · 主题标签: decode, paged-KV, L2/HBM, gfx1250-TDM

- 独立 per-sequence paged-KV decode 的 **L2 命中率 ~1-3% 是预期正常**，不是 bug。streaming 无复用，每个 KV 字节只读一次。"提高 L2 命中率"是**非目标**——唯一能真实提高 L2 的是 KV 复用=共享前缀服务(workload/调度属性，非 kernel 改动)。
- **memory-bound decode 判断树**（三者都健康即达到该访问模式真实上限，无 KV-load 优化空间）：
  - `L2 < 5%` = 纯 streaming 无复用（预期正确）
  - `32B ≈ 0%` = 满 64B 线无浪费
  - `over-fetch = est_HBM / (ideal_GB × dispatches) ≈ 1.0x` = 只读所需数据
  - 达到的带宽 `= ideal_bytes / kernel_time`；对干净 streaming，**50-60% 理论 HBM 峰值是正常的**，不是可优化的低效。
- ❌ 别再试（在 memory 子系统已干净时找 KV-load 优化）：PA decode gfx942 实测(bs16 ctx131072 batch256) L2命中 **1.7%**、32B **0%**、over-fetch **1.04x**、**2.85TB/s = 54% 峰值** → memory 子系统干净，无 KV-load 优化空间。
  - 旁证1：`block_size 16→64` 回退 **+7.8%**（更大 block 反而更差，说明当前 block_size 已合适，不是 L2/blocking 问题）。
  - 旁证2：`dwordx8` 在 CDNA3 不存在——`dwordx4`/16B 是单向量 load 上限，别指望更宽 load。

## gfx1250 TDM MoE row-gather：必须 addr64，否则硬 hang

- TDM MoE row-gather **必须**用 carry-safe 的 `update_tensor_gather_descriptor_addr64`：它把 lo-add 的进位传播进 `addr_hi`。
- ❌ 别再试短版 addr-lo-only update：会**静默 wrap 一个 4 GiB page**，只在大 tensor 规模下暴露为 **HARD HANG（不是错误结果）**，极难 debug。
- gather descriptor 的构建（以及 B / B-scale descriptors）必须 **hoist 出 K loop**，否则循环 SALU-bound。

---
来源: capture-kernel-trace/SKILL.md, kernel-trace-analysis/SKILL.md, optimization-directions.md
