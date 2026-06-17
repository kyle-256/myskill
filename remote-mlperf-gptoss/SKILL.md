---
name: remote-mlperf-gptoss
description: 本机已本地化：Primus-Turbo 在 /workspace/code/gpt_oss_docker/sync/Primus-Turbo，命令本地直接跑（无 ssh / 无 docker / 无远程容器）。本机只同步了 Primus-Turbo，HipKittens/FlyDSL/其他 repo 未同步。当用户提到 Primus-Turbo / mlperf_gptoss / FP8 grouped GEMM / flydsl / ROCm/HIP 相关操作时使用此 skill。原"经 login_node2 → chi2811 → docker exec mlperf_gptoss"远程链路仅作历史参考（其他带 GPU 的机器用）。
---

> ⚠️ **本机已本地化**：Primus-Turbo 在 `/workspace/code/gpt_oss_docker/sync/Primus-Turbo`，命令**本地直接跑**，无 ssh / 无 docker / 无远程容器。
> 下方原"经 login_node2 → chi2811 → docker exec mlperf_gptoss"的远程链路仅作历史参考（其他机器用）。

# remote-mlperf-gptoss

本机直接在本地跑命令，工作目录 `/workspace/code/gpt_oss_docker/sync/Primus-Turbo`。

> **注意**：本机只同步了 **Primus-Turbo**。HipKittens / FlyDSL / 其他 repo **未同步到本机**。涉及它们的步骤本机不适用。

## 本地执行命令

进 Primus-Turbo 目录直接跑（环境变量直接前置，无需 ssh / docker exec 包装）：

```bash
cd /workspace/code/gpt_oss_docker/sync/Primus-Turbo && HIP_VISIBLE_DEVICES=6 python scripts2/<probe>.py
```

- 代码本地直接编辑、本地直接 `git`（分支 `dev/kyle/flydsl_grp_gemm`），不需要 rsync / scp。
- 改 flydsl kernel（.py）后 `rm -rf /root/.flydsl/cache` 再跑；改 cpp 才需重装 primus_turbo。
- HipKittens 后端已移除（2026-06-01）；`_C` 只剩 hipblaslt/CK/turbo ops。`torch.ops.primus_turbo_cpp_extension.hipblaslt_gemm_fp8` 仍可用。

---

## 原远程流程（参考，本机不用）

下面整段是原来"经跳板机 ssh 到计算节点再 docker exec"的远程链路，仅供其他机器参考。本机已本地化，不需要这套。

### 连接链路

```
本地 (/wekafs/kyle)              ┐
  └─ ssh login_node2             │  149.28.124.225, root（跳板机）
       └─ ssh compute_node_new   │  chi2811 (见 ~/.ssh/config, 是 SoT), root, ProxyJump=login_node2
            └─ docker exec mlperf_gptoss   ── 镜像: mlperf_gptoss:saved-20260610 (常驻)
                 └─ /workspace/code        ── 默认 cwd
                      ↑
                      └── host bind mount: /mnt/vast/kyle/code2
```

- SSH 配置都在 `~/.ssh/config`，私钥 `~/.ssh/id_ed25519`。
- 容器是常驻的（`docker exec` 而不是 `docker run`），写入文件下次还在。
- Host 路径 `/mnt/vast/kyle/code2` 是 `/workspace/code` 的 bind mount —— rsync 直接走 host 路径就行，参见 `../remote-sync/SKILL.md`。

### 远程环境（已确认 2026-06-11，节点 chi2811）

| 项 | 值 |
|---|---|
| 主机名 | `chi2811`（容器内外一致；当前节点以 ssh config 为准） |
| OS | Linux（容器内 Ubuntu/Debian 系，`/.dockerenv` 存在） |
| GPU | 8 × AMD Instinct MI355X |
| Python | 3.12.3 @ `/opt/venv/bin/python`（无须 activate venv，PATH 已就位） |
| PyTorch | `2.10.0a0+git449b176`，HIP 可用 |
| Triton | 升级到 `3.7.0`（fresh 容器默认 3.6，claim 时升级） |
| 镜像 | `rocm/primus:v26.2` |
| 同节点其他容器 | `mlperf_gptoss2`（别人的，**不要碰**） |
| 外网 | **无**（远程不能 pip install / git clone / curl 外部资源；这是硬约束，编辑必须在本地，参见 remote-sync skill） |

### FlyDSL / primus_turbo 从源码安装（fresh 容器一次性 setup）

> **本机不适用**：这步需要 `/workspace/code/FlyDSL`，而 **FlyDSL 未同步到本机**。本机只有 Primus-Turbo。

**首选别走这条**：换节点时用 claim-mi355x-node §4「从保存的镜像恢复」（`mlperf_gptoss:saved-*`，已含 triton 3.7 + flydsl/primus_turbo），开箱即用，不用下面这步。只有从 fresh `rocm/primus:v26.2` 起才需要。

fresh `rocm/primus:v26.2` 容器里 `flydsl` 没装 → 跑 flydsl kernel 前要 **从源码 editable 安装**
（见 claim-mi355x-node §5.5）：`pip install -e /workspace/code/FlyDSL`（本机无此 repo）+ 重装 primus_turbo。
装完后 `import flydsl` / `import primus_turbo.pytorch` 直接可用，**不需要 PYTHONPATH**。

---

## 本地执行命令（详细）

**单条命令**：

```bash
cd /workspace/code/gpt_oss_docker/sync/Primus-Turbo && <CMD>
```

**指定工作目录**（多 repo 根，本机只有 Primus-Turbo）：

```bash
cd /workspace/code/gpt_oss_docker/sync/Primus-Turbo && python scripts/foo.py
```

**指定环境变量**（直接前置）：

```bash
cd /workspace/code/gpt_oss_docker/sync/Primus-Turbo && HIP_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 python scripts/foo.py
```

**长任务**（Bash 工具用 `run_in_background: true`）：

```bash
# 在 Bash 工具中：run_in_background=true
cd /workspace/code/gpt_oss_docker/sync/Primus-Turbo && python scripts/auto_optimize_gpt_oss_fp8.py 2>&1 | tee /tmp/run.log
```

## 文件传输

本机本地化，无需 scp / rsync / docker cp —— 文件本来就在本地 `/workspace/code/gpt_oss_docker/sync/Primus-Turbo`。

## 约定

- 代码本地直接编辑、本地直接 `git`（分支 `dev/kyle/flydsl_grp_gemm`）。
- 除非用户指定别的路径，先 `cd /workspace/code/gpt_oss_docker/sync/Primus-Turbo`。
- 本机多 repo 根是 `/workspace/code/gpt_oss_docker/sync`，但目前**只有 Primus-Turbo**。
- 改 flydsl kernel（.py）后 `rm -rf /root/.flydsl/cache` 再跑；改 cpp 才需重装 primus_turbo。

## 本机的 repo 目录

```
/workspace/code/gpt_oss_docker/sync/
└── Primus-Turbo/      # FP8 grouped GEMM 优化主工作目录（git 分支 dev/kyle/flydsl_grp_gemm）
```

> **未同步到本机的 repo**（原远程 `/workspace/code` 下有，本机没有）：HipKittens、FlyDSL、code2、triton 等。涉及这些 repo 的步骤本机无法执行。

`Primus-Turbo` 是活跃工作目录，**总是有未提交修改和未跟踪的 probe 脚本/分析笔记**（这是常态，不是问题）。

## 跑命令前的 checklist

1. 涉及代码修改 → 本地直接改、本地直接 git，无需推送。
2. 涉及读 log / 已有产物 → 直接读本地 `/workspace/code/gpt_oss_docker/sync/Primus-Turbo/auto_optimize_logs/`。
3. 长任务 → Bash 工具 `run_in_background: true`，输出写到 `/tmp/<name>.log` 或 `auto_optimize_logs/`。
4. 短查询（`rocm-smi`、`hostname` 等）→ 本地直接跑（本机为 docker 容器，无 GPU / 无 docker CLI，GPU 类命令本机不可用）。

## 故障排查（本地）

| 症状 | 排查 |
|---|---|
| `ImportError: cannot import name '_compile_dense_tn'` | flydsl 没装。**本机 FlyDSL 未同步**，无法 `pip install -e`，需先把 FlyDSL 同步到本机或问用户。 |
| `import primus_turbo.pytorch` 报 `undefined symbol ...hk_gemm_bf16` | HK dense binding 引用了缺失的 .cu；HK 后端已移除，清掉残留 + 重 build。 |
| flydsl kernel 改了不生效 | `rm -rf /root/.flydsl/cache` 再跑。 |
| ROCm/HIP 报错 | 本机为 docker 容器、**无 GPU**，GPU 类命令（`rocm-smi` / `rocminfo`）本机不可用；这类报错说明该步骤需在带 GPU 的机器上跑。 |
