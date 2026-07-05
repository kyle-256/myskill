# 缓存增益上限规则：id(weight) cache 上限=quant_time/step_time，id(activation) 命中率≈0

> 类别: 踩过的坑 · 主题标签: cache-key, real-training-gain, MoE-skew, benchmark-overfit

## weight-quant cache：keyed(id(weight), weight._version)，增益有硬上限

- **cache 结构**：weight-quant / preshuffle 结果按 `(id(weight), weight._version)` 做 key。这是唯一允许的 cache key 形态。
- **真实训练每 step 增益上限 = quant_time(weight) / step_time**，是低个位数 %，**不是 benchmark headline**。WHY：
  - forward 算一次 quant 写入 entry；backward 复用 forward-time entry → 每 step 只 **1 hit/step**。
  - `optim.step()` 改权重 → `weight._version` bump → 下一个 forward 前 entry 失效。
  - 所以 benchmark 里反复复用同一 tensor 看着"免费"，真实训练里每 step 都要重付一次 quant。
- 上限量化为 `W_real`（对应 W1），REPORT 时必须 capped by `quant_time/step_time`。

## ❌ 别再试：id(activation) / id(grad_out) / id(activation_scale) 作 cache key

- ❌ **别再试**：拿 `id(activation)` / `id(grad_out)` / `id(activation_scale)` 当 key 缓存。真实训练命中率 **≈ 0**。
  - 机制：真实训练每 step 的 activation / grad 都是**新分配的 tensor**，`id(...)` 每次不同 → 永不命中。
  - 只有"复用同一 tensor"的 benchmark 才制造出免费假象。
- 这是 **Rule 11**：这类 W2/W3 cache 必须在 **ANALYZE 阶段直接拒绝**，不许实现出来再测。
- REPORT 时 `R_real` 必须 = 0；若真实命中率 < 50% → 即使 benchmark 已 accept 也**回退**。

## ❌ 别再试：MoE/grouped GEMM 把均匀分布假设烘进内核

真实 routing 给出**倾斜、逐 batch 变化**的 tokens_per_expert 直方图，一般**不是任何 tile 维度的整数倍**，且**部分 expert 拿到 0 token**。因此：

- ❌ **别再试**：把 `M_per_group` 烘成编译期常量。
- ❌ **别再试**：`assert M_per_group % BLOCK_M == 0`。
- ❌ **别再试**：假设固定 per-expert count 的静态 work partitioning。
- ✅ **必须**处理空组：`M_per_group == 0` 时 skip launch / 对 zero rows 分支。
- **验证纪律**：每一轮 MoE 都要在倾斜分布上验证——top-1 + capacity factor，以及近退化情形（某 expert 拿 ≥50% token）。在 skew 下消失的增益 = benchmark over-fit，**必须回退**。

## benchmark 增益 > 结构能产 → 先查，别接受

- 若某轮 benchmark 增益**大于内核改动结构上能产生的量**，accept 之前先排查：
  1. identity-keyed activation cache（id(...) 假命中）
  2. uniform-MoE 假设（均匀分布捷径）
- REPORT 时把 baseline→final delta 重新归因：
  - `S_real`（K1–K4，transfer 1:1）
  - `W_real`（W1，capped by quant_time/step_time）
  - `R_real`（必须 = 0）
- 若 **headline − real 差 > 1%** → 标为 benchmark-loop residual 并建议回退。

## iteration_rules 核心纪律（贯穿以上）

- 一轮一个假设；perf 前先过 correctness gate；accept/rollback 有 lineage。
- 每个 accepted gain **必须能迁移到真实训练 step**：不许 id(...) 作 key 的 activation/grad_out cache；不许只适配均匀分布的 GroupGemm 捷径（真实倾斜 token 分布下无效）。

---
来源: SKILL.md, gemm/optimization-directions.md, optimize-handoff/SKILL.md
