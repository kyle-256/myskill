# 选空闲卡:rocm-smi 三件套与 sinfo 不可信

> 类别: 连接 · 主题标签: cluster-triage, free-gpu, rocm-smi, vllm-陷阱

## sinfo/squeue 标签不可信
- slurm 的 `sinfo` `idle`/`resv`/`down` 标签和真实 GPU 占用**完全对不上号**:大家都绕过 slurm 直接 `docker run` 抢 GPU。
  - 标 `idle` 的节点可能有 9 个 GPU 进程 + 3 个容器;标 `resv`/`down` 的节点反而真空。
- 唯一可信判据 = 直接 ssh 进节点跑 rocm-smi 三件套,**不能信任单一指标**。

## rocm-smi 三件套
| 命令 | 看什么 | 地位 |
|---|---|---|
| `--showpids` | pid 数 | **金标准** |
| `--showuse` | GPU use% | 辅助 |
| `--showmemuse` | per-GPU VRAM% | 辅助 |
| `--showmeminfo vram` | `Used Memory` 绝对值 | 辅助 |

## 真·可用判据
- 鲁棒批量扫:`rocm-smi --showpids 2>/dev/null | grep -cE '^[0-9]'` == 0 且 `docker ps` 无业务容器。
- VRAM% 要低,但 **pid 数才是金标准**。
- chi2811 实测真空态:8 卡 0% use、每卡仅 ~298MB 底噪、0 pid。
- 空闲 VRAM 参照:`rocm-smi --showmeminfo vram | grep 'Used Memory'` ~300MB = 空;或 `--showuse` GPU%=0 且 `--showmemuse` VRAM%<50。

## vllm 陷阱:gpu_use=0 会骗人
- **绝不能只看 `gpu_use=0` 判空闲**:vllm 用 `--gpu_memory_utilization 0.95` 时锁 95% VRAM,但 idle 等请求时 `gpu_use` 显示 0%。
- 正解:用 `--showmemuse` 看 VRAM%,0% 才真空;或 `--showmeminfo vram` 确认 ~300MB。

## 测量前提 / 杂项
- 测 perf 前必须确认 GPU 0% 占用:被抢占时 perf 波动 **20-30%**;测 race rate 在 noisy GPU 上会被 OOM 干扰。
- 快速查占用:`GPU=7 ./rr.sh sh 'rocm-smi --showuse | grep GPU'`。
- 杀僵尸进程:`ps -eo pid,etime,pcpu,cmd | grep -E python|torch`。

---
来源: claim-mi355x-node/SKILL.md, fp8-gemm-bench/SKILL.md, 13-primus-turbo-prod.md, 07-benchmark.md
