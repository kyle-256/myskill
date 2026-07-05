# 能否共存与 OOM 判据

> 类别: 连接 · 主题标签: cluster-triage, coexistence, OOM, measurement-hygiene

## 能否共存判据（看 VRAM% + CU OCCUPANCY 两个量）

| 邻居状态 | VRAM | CU OCCUPANCY / GPU% | 判定 |
|---|---|---|---|
| idle-waiting worker（vllm/sglang 只锁显存等请求） | ~28%（甚至 200GB+ 被锁） | OCCUPANCY=0 / GPU%=0 | **可共存**：小 GEMM timing 干净；但**大 batch 会 OOM** |
| 真算中 | 96% | OCCUPANCY>0 真算 | **不可共存** |

- GPU%=0 但 `rocm-smi --showpids` 有 vllm/sglang worker 占 200GB+ 显存 → 是 idle waiting、显存被锁：小任务可共存，大 batch OOM。
- 判据核心：**CU OCCUPANCY=0 才是真 idle**；只看 VRAM% 会误判（显存锁着但没算）。

## 立刻换节点的红旗（不可共存）

- `primus-training` / `maxtext-slurm` 是多节点训练 job，会同时铺满一排节点（chi2832/2835/2816/2800...），每节点吃满 8 卡 ~96% VRAM 真算 → 落这种节点容器反复 **OOM 崩（exit 137）**。
- 触发换节点条件：看到 `primus-training` 容器，或某 PID `GPU(s)=8` + 各卡 VRAM 96% → 立刻换。

## 邻卡满载污染测量（即使能共存也要独占单卡测）

- 性能测量必须独占单卡：`HIP_VISIBLE_DEVICES=<n>` 指定一张空闲 GPU。
- WHY：邻卡满载（如 GPU3 常 100%）通过**电源/时钟/ECC 扰动**污染测量；长 kernel（K28672）甚至出现 HW 级间歇非确定（det>0）。
- chi2810 各卡有 ~2~4% 频率差（其他节点未必同样实测过），GPU3 常年被别人占。
- **A/B 性能对比必须同卡同会话进行**（避免卡间频率差 + 跨会话时钟/热态漂移）。

---
来源: claim-mi355x-node/SKILL.md, agpr_phase5_lds.md, agpr_phase5_mono.md, flydsl-fp8-gemm-results/SKILL.md
