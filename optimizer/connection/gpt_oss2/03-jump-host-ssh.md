# gpt_oss2 跳板机连接(.ssh-chi.sh)

> 类别: 连接 (gpt_oss2) · 主题标签: ssh, jump-host, ProxyCommand, slurm

- **无 `~/.ssh/config`**：gpt_oss2 环境不依赖 ssh config，连接一律走脚本 `/workspace/code/gpt_oss2_docker/sync/.ssh-chi.sh`。属 common 类跳板拓扑，但脚本路径为 gpt_oss2 专属。
- **默认 key**：`/workspace/code/.ssh_docker/id_ed25519`（脚本内置，无需 `-i` 手动指定）。
- **ProxyCommand 跳板节点**：`root@149.28.124.225`（hostname `chi2866`，登录/跳板节点）。WHY 用它做代理:计算节点不可直连,必须经此跳。
  - 跳板节点自身可跑 `sinfo` / `squeue` 查 Slurm 队列与节点状态。
- **计算节点命令格式**：`./.ssh-chi.sh root@chi2811 '<cmd>'`。
- **节点历史沿革**（避免连到已弃用节点）:
  - `chi2774` — 2026-06-30 弃用
  - `chi2810` — 2026-07-03 换到 `chi2811`
  - 当前计算节点:`chi2811`

---
来源: remote-sync/SKILL.md, project_remote_sync_setup.md
