# 选空闲卡:rocm-smi 三件套、共存判据与 OOM 判据

> 类别: 连接 · 主题标签: cluster-triage, free-gpu, rocm-smi, vllm-陷阱, coexistence, OOM, measurement-hygiene

## 选空闲卡:rocm-smi 三件套与 sinfo 不可信

### sinfo/squeue 标签不可信
- slurm 的 `sinfo` `idle`/`resv`/`down` 标签和真实 GPU 占用**完全对不上号**:大家都绕过 slurm 直接 `docker run` 抢 GPU。
  - 标 `idle` 的节点可能有 9 个 GPU 进程 + 3 个容器;标 `resv`/`down` 的节点反而真空。
- 唯一可信判据 = 直接 ssh 进节点跑 rocm-smi 三件套,**不能信任单一指标**。

### rocm-smi 三件套
| 命令 | 看什么 | 地位 |
|---|---|---|
| `--showpids` | pid 数 | **金标准** |
| `--showuse` | GPU use% | 辅助 |
| `--showmemuse` | per-GPU VRAM% | 辅助 |
| `--showmeminfo vram` | `Used Memory` 绝对值 | 辅助 |

### 真·可用判据
- 鲁棒批量扫:`rocm-smi --showpids 2>/dev/null | grep -cE '^[0-9]'` == 0 且 `docker ps` 无业务容器。
- VRAM% 要低,但 **pid 数才是金标准**。
- chi2774 实测真空态:8 卡 0% use、每卡仅 ~298MB 底噪、0 pid。
- 空闲 VRAM 参照:`rocm-smi --showmeminfo vram | grep 'Used Memory'` ~300MB = 空;或 `--showuse` GPU%=0 且 `--showmemuse` VRAM%<50。

### vllm 陷阱:gpu_use=0 会骗人
- **绝不能只看 `gpu_use=0` 判空闲**:vllm 用 `--gpu_memory_utilization 0.95` 时锁 95% VRAM,但 idle 等请求时 `gpu_use` 显示 0%。
- 正解:用 `--showmemuse` 看 VRAM%,0% 才真空;或 `--showmeminfo vram` 确认 ~300MB。

### 测量前提 / 杂项
- 测 perf 前必须确认 GPU 0% 占用:被抢占时 perf 波动 **20-30%**;测 race rate 在 noisy GPU 上会被 OOM 干扰。
- 快速查占用:`GPU=7 ./rr.sh sh 'rocm-smi --showuse | grep GPU'`。
- 杀僵尸进程:`ps -eo pid,etime,pcpu,cmd | grep -E python|torch`。

## 能否共存与 OOM 判据

### 能否共存判据（看 VRAM% + CU OCCUPANCY 两个量）

| 邻居状态 | VRAM | CU OCCUPANCY / GPU% | 判定 |
|---|---|---|---|
| idle-waiting worker（vllm/sglang 只锁显存等请求） | ~28%（甚至 200GB+ 被锁） | OCCUPANCY=0 / GPU%=0 | **可共存**：小 GEMM timing 干净；但**大 batch 会 OOM** |
| 真算中 | 96% | OCCUPANCY>0 真算 | **不可共存** |

- GPU%=0 但 `rocm-smi --showpids` 有 vllm/sglang worker 占 200GB+ 显存 → 是 idle waiting、显存被锁：小任务可共存，大 batch OOM。
- 判据核心：**CU OCCUPANCY=0 才是真 idle**；只看 VRAM% 会误判（显存锁着但没算）。

### 立刻换节点的红旗（不可共存）

- `primus-training` / `maxtext-slurm` 是多节点训练 job，会同时铺满一排节点（chi2832/2835/2816/2800...），每节点吃满 8 卡 ~96% VRAM 真算 → 落这种节点容器反复 **OOM 崩（exit 137）**。
- 触发换节点条件：看到 `primus-training` 容器，或某 PID `GPU(s)=8` + 各卡 VRAM 96% → 立刻换。

### 邻卡满载污染测量（即使能共存也要独占单卡测）

- 性能测量必须独占单卡：`HIP_VISIBLE_DEVICES=<n>` 指定一张空闲 GPU。
- WHY：邻卡满载（如 GPU3 常 100%）通过**电源/时钟/ECC 扰动**污染测量；长 kernel（K28672）甚至出现 HW 级间歇非确定（det>0）。
- chi2810 各卡有 ~2~4% 频率差（其他节点未必同样实测过），GPU3 常年被别人占。
- **A/B 性能对比必须同卡同会话进行**（避免卡间频率差 + 跨会话时钟/热态漂移）。

---
来源: claim-mi355x-node/SKILL.md, fp8-gemm-bench/SKILL.md, 13-primus-turbo-prod.md, 07-benchmark.md, agpr_phase5_lds.md, agpr_phase5_mono.md, flydsl-fp8-gemm-results/SKILL.md
