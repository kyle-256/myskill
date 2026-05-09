---
name: remote-mlperf-gptoss
description: 连接远程计算节点（chi2811，经 login_node2 跳板机）并在 mlperf_gptoss 容器的 /workspace/code 下执行命令。当用户在 /wekafs/kyle 下提到"远程主机/计算节点/容器/mlperf_gptoss/workspace/code"等相关操作时使用此 skill。
---

# remote-mlperf-gptoss

通过登录节点跳转到计算节点，再 `docker exec` 进入 `mlperf_gptoss` 容器执行命令。容器内默认工作目录是 `/workspace/code`。

## 连接链路

```
本地 (/wekafs/kyle)
  └─ ssh login_node2          # 149.28.124.225, root
       └─ ssh compute_node_new # chi2811, root, ProxyJump=login_node2, ForwardAgent
            └─ docker exec mlperf_gptoss  # 镜像: rocm/primus:v26.2
                 └─ /workspace/code       # 默认工作目录
```

两段 SSH 配置都在 `~/.ssh/config` 里，使用 `~/.ssh/id_ed25519` 认证。

## 在容器内执行命令

**单条命令**（默认 cwd 为 `/workspace/code`）：

```bash
ssh compute_node_new 'docker exec mlperf_gptoss bash -lc "cd /workspace/code && <CMD>"'
```

**多行脚本**（用 heredoc，引号不会乱）：

```bash
ssh compute_node_new 'docker exec mlperf_gptoss bash -lc "$(cat)"' <<'EOF'
cd /workspace/code
<commands>
EOF
```

**交互式 shell**（仅当用户明确要求时用 —— 不要在自动化的 tool call 里开，会卡死）：

```bash
ssh -t compute_node_new 'docker exec -it mlperf_gptoss bash -c "cd /workspace/code && exec bash"'
```

## 约定

- 命令一律用 `bash -lc "..."` 包起来，确保登录环境（PATH、conda、ROCm modules 等）被加载。
- 除非用户指定别的路径，否则一进容器先 `cd /workspace/code`。
- 第一次连接时如果觉得慢，可以加 `-o ConnectTimeout=20`；ControlMaster 热起来后就不用加了。
- 跑长任务优先用 Bash 工具的 `run_in_background: true`，不要在 SSH 命令里加 `&` —— 这样退出时能拿到干净的通知。
- 一条命令能搞定的话，不要分两步先 `ssh` 再 `docker exec` —— 上面的链式写法更快，也不会留游离会话。
- 容器是常驻的（`docker exec` 而不是 `docker run`），所以一次写入的文件下次还能看到。

## /workspace/code 下的子目录

`HipKittens/`、`Primus-Turbo/`、`code2/`、`triton/`、`.codex/` —— 2026-05-09 确认存在。用户点名某个就直接 `cd` 进去。

## 故障排查

- `ssh login_node2 'hostname'` —— 验证跳板机能不能连。
- `ssh compute_node_new 'docker ps --format "{{.Names}}"'` —— 确认 `mlperf_gptoss` 还在跑。如果挂了，先停下来问用户，不要擅自启动或替换。
- 如果 `chi2811` 的 host key 变了（节点被重装），告诉用户，不要默默接受新 key。
