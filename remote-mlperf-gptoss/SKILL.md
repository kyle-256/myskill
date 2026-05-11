---
name: remote-mlperf-gptoss
description: 连接远程计算节点（当前 chi2761，经 login_node2 跳板机）并在 mlperf_gptoss 容器里执行命令。容器默认工作目录 /workspace/code。当用户在 /wekafs/kyle 下提到"远程主机/计算节点/容器/mlperf_gptoss/workspace/code/MI355X/ROCm/HIP"等相关操作时使用此 skill。配合 remote-sync skill：编辑代码在本地、跑命令在远程。节点不可用/容器没了 → 用 claim-mi355x-node skill 换节点。
---

# remote-mlperf-gptoss

通过登录节点跳转到计算节点，再 `docker exec` 进入 `mlperf_gptoss` 容器执行命令。容器内默认工作目录 `/workspace/code`。

## 连接链路

```
本地 (/wekafs/kyle)              ┐
  └─ ssh login_node2             │  149.28.124.225, root（跳板机）
       └─ ssh compute_node_new   │  chi2761, root, ProxyJump=login_node2, ForwardAgent
            └─ docker exec mlperf_gptoss   ── 镜像: rocm/primus:v26.2 (常驻)
                 └─ /workspace/code        ── 默认 cwd
                      ↑
                      └── host bind mount: /mnt/shared/kyle/code2
```

- SSH 配置都在 `~/.ssh/config`，私钥 `~/.ssh/id_ed25519`。
- 容器是常驻的（`docker exec` 而不是 `docker run`），写入文件下次还在。
- Host 路径 `/mnt/shared/kyle/code2` 是 `/workspace/code` 的 bind mount —— rsync 直接走 host 路径就行，参见 `../remote-sync/SKILL.md`。

## 远程环境（已确认 2026-05-11）

| 项 | 值 |
|---|---|
| 主机名 | `chi2761`（容器内外一致） |
| OS | Linux（容器内 Ubuntu/Debian 系，`/.dockerenv` 存在） |
| GPU | 8 × AMD Instinct MI355X |
| Python | 3.12.3 @ `/opt/venv/bin/python`（无须 activate venv，PATH 已就位） |
| PyTorch | `2.10.0a0+git449b176`，HIP 可用 |
| 镜像 | `rocm/primus:v26.2` |
| 同节点其他容器 | `dev_xb_vllm_20`、`amd_dcdit`、`akharida_docker`、`sgl-deepseek-v4-pro-rocm720` —— **不要动** |
| 外网 | **无**（远程不能 pip install / git clone / curl 外部资源；这是硬约束，编辑必须在本地，参见 remote-sync skill） |

## 在容器内执行命令

**单条命令**（默认 cwd `/workspace/code`）：

```bash
ssh compute_node_new 'docker exec mlperf_gptoss bash -lc "cd /workspace/code && <CMD>"'
```

**多行脚本**（heredoc，引号不会乱）：

```bash
ssh compute_node_new 'docker exec mlperf_gptoss bash -lc "$(cat)"' <<'EOF'
cd /workspace/code
<commands>
EOF
```

**指定工作目录**（常用：进具体 repo）：

```bash
ssh compute_node_new 'docker exec mlperf_gptoss bash -lc "cd /workspace/code/Primus-Turbo && python scripts/foo.py"'
```

**指定环境变量**：放在 `bash -lc` 里就行，不要用 `docker exec -e`（要单独转义）：

```bash
ssh compute_node_new 'docker exec mlperf_gptoss bash -lc "cd /workspace/code && HIP_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 python scripts/foo.py"'
```

**长任务**（Bash 工具用 `run_in_background: true`，**不要**在 SSH 命令里加 `&`）：

```bash
# 在 Bash 工具中：run_in_background=true
ssh compute_node_new 'docker exec mlperf_gptoss bash -lc "cd /workspace/code/Primus-Turbo && python scripts/auto_optimize_gpt_oss_fp8.py 2>&1 | tee /tmp/run.log"'
```

**交互式 shell**（仅当用户明确要求时用，**不要**在自动化的 tool call 里开 —— 会卡死）：

```bash
ssh -t compute_node_new 'docker exec -it mlperf_gptoss bash -c "cd /workspace/code && exec bash"'
```

## 文件传输

**首选**：走 `remote-sync` skill 的 `sync.sh`（rsync over ssh，直接走 host 路径，不穿 docker）。

`scp` 也可以，但要传到 host 路径，不是容器内路径：

```bash
scp local_file compute_node_new:/mnt/shared/kyle/code2/Primus-Turbo/scripts/
# 等价于容器内 /workspace/code/Primus-Turbo/scripts/
```

**不要**用 `docker cp`（要先 ssh 到 host 再跑，慢且容易出错）。

## 约定

- 命令一律用 `bash -lc "..."` 包起来 —— 确保 PATH/venv/ROCm env 加载，否则 `python` 都找不到。
- 除非用户指定别的路径，进容器先 `cd /workspace/code`（或具体 repo 子目录）。
- 一条命令能搞定就别拆 —— 链式 `ssh ... 'docker exec ... bash -lc "..."'` 比"先 ssh 再 docker exec"快、不留游离会话。
- 第一次连接慢可加 `-o ConnectTimeout=20`，后续 ControlMaster 热起来不用加。
- **不要在远程跑 git 命令** —— git 在本地 `/wekafs/kyle/remote_sync/<repo>/` 用。
- **不要在远程编辑文件** —— 编辑在本地，rsync 推上去。详见 `remote-sync` skill。
- **不要 `pip install` / `apt install` / `git clone` 外部资源** —— 远程没外网。需要新依赖先问用户怎么搞。

## /workspace/code 下的子目录

```
/workspace/code/
├── Primus-Turbo/      # FP8 grouped GEMM 优化主工作目录（dev/kyle_hipkitten_bf16）
├── HipKittens/         # kernel 侧（save/fp8-progress-20260319-native-layouts）
├── code2/              # 历史/参考
├── triton/             # 参考
└── .codex/             # 工具元数据
```

`Primus-Turbo` 和 `HipKittens` 是用户的活跃工作目录，**总是有未提交修改和未跟踪的 probe 脚本/分析笔记**（这是常态，不是问题）。

## 远程跑命令前的 checklist（自动化场景）

1. 涉及代码修改 → 改本地，`sync.sh push <repo> [path]` 推上去（参见 remote-sync skill）。
2. 涉及读 log / 已有产物 → **优先**读本地 `/wekafs/kyle/remote_sync/<repo>/auto_optimize_logs/`（如果已 sync 过来），不要无谓地 ssh。
3. 长任务 → Bash 工具 `run_in_background: true`，输出写到 `/tmp/<name>.log` 或 `auto_optimize_logs/`。
4. 短查询（`nvidia-smi`、`rocm-smi`、`docker ps`、`hostname`、查容器进程等） → 直接 ssh，不要前置 sync。

## 故障排查

| 症状 | 排查 |
|---|---|
| ssh 卡住 | `ssh -v login_node2 'hostname'` 看是否到跳板机；再 `ssh -v compute_node_new 'hostname'` 看 ProxyJump 是否成功。 |
| `docker exec` 报 `No such container` | `ssh compute_node_new 'docker ps -a --filter name=mlperf_gptoss'` 看是 stopped 还是真没了。**不要擅自 docker run 一个新的** —— 镜像、挂载、env 都可能不对。**先停下来问用户**。 |
| 容器内 `python: command not found` | 没用 `bash -lc`。改成 `bash -lc "..."`。 |
| `chi2811` host key 变了 | 节点被重装的可能。**不要默默 accept**，告诉用户确认。 |
| ROCm/HIP 报错 | 先 `rocm-smi` / `rocminfo` 验证 GPU 可见性 + 容器有 `--device=/dev/kfd --device=/dev/dri` 挂载（用 `docker inspect` 看）。 |
| 同节点其他用户在跑 | `ssh compute_node_new 'docker ps && rocm-smi --showpidgpus'` —— 如果 GPU 被占满，告诉用户，不要无脑跑。 |
| 文件推上去容器看不到 | 检查 rsync 目标是不是 host 路径 `/mnt/shared/kyle/code2/<repo>/...`，**不是**容器内路径 `/workspace/code/...`。 |
