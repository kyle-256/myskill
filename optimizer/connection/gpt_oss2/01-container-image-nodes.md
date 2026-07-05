# gpt_oss2 容器/镜像/生产节点/连接

> 类别: 连接 (gpt_oss2) · 主题标签: container, image, prod-nodes, chi2811, docker-commit, saved-image, mxfp4-rebuild, bind-mount, ssh, jump-host, ProxyCommand, slurm

## gpt_oss2 目标容器/镜像/生产节点现状

- [目标容器/镜像] gpt_oss2 环境目标容器 = `mlperf_gptoss2`，镜像 `mlperf_gptoss2:flydsl`。所有 gpt_oss2 工作都在此容器内。
- [同节点勿碰] 同节点还有别人的容器，一律不要动：
  - `mlperf_gptoss`（注意无 "2"，挂 code2）——别人的，勿碰。
  - `yambda_smoke` 等——勿碰。
  - WHY: 误操作别人容器会破坏其环境；靠"有无 2 / 挂载 code vs code2"区分归属。
- [当前生产节点] `chi2811`（gfx950 / MI355X）。gpt_oss2 的 mxfp8 实验、grouped wgrad bench 均在此跑；commit/证伪结论都标注 chi2811。
- [节点变更历史] chi2774 于 06-30 弃用；chi2810 于 07-03 更换 → 当前落到 chi2811。WHY: 生产节点会被回收/轮换，认准 chi2811 才是当前有效节点。
- [flydsl 缓存陈旧问题] flydsl 磁盘缓存陈旧（staleness）问题正是在 `mlperf_gptoss2` / chi2811 (gfx950 MI355X) 环境下发现的。

## gpt_oss2 从共享盘 tar 恢复容器 + docker commit 固化

- [跨节点恢复] gpt_oss2 跨节点持久备份 = 共享盘 tar `/mnt/vast/kyle/mlperf_gptoss2_flydsl.tar`，新节点 `docker load -i /mnt/vast/kyle/mlperf_gptoss2_flydsl.tar` 即用。
  - WHY tar 在共享盘：`/mnt/vast` 是跨节点可见的 vast 共享盘，任一新节点直接 load 免重传镜像。
- [tar 内容 = 2026-06-22 快照] 该 tar 打于 **2026-06-22**，只含 `/opt/venv`(mxfp8)，**不含** `/opt/venv-mxfp4`。
  - 后果：新节点 load 后必须**重建 mxfp4 venv**（`/opt/venv-mxfp4`），tar 里没有。
- [容器内改动必须 docker commit 才持久] gpt_oss2 容器内改动（`/opt/venv*`、finder 修改、symlink）必须
  `docker commit mlperf_gptoss2 mlperf_gptoss2:saved-YYYYMMDD` 才持久化。
  - WHY：这些路径在容器可写层，容器删了就没；只有 commit 成新镜像 tag 才留存。
- [bind-mount 本就 host 持久，无需 commit] bind-mount 的 code3 / repo / `.so` 位于 host 文件系统，本就持久，不靠 commit。
  - 推论：区分「容器内可写层」(需 commit) vs「bind-mount host 路径」(自动持久)。
- [最近 saved 版本] 最近 `saved-20260625`；chi2811 于 **2026-07-03** 已 commit。

## gpt_oss2 跳板机连接(.ssh-chi.sh)

- **无 `~/.ssh/config`**：gpt_oss2 环境不依赖 ssh config，连接一律走脚本 `/workspace/code/gpt_oss2_docker/sync/.ssh-chi.sh`。属 common 类跳板拓扑，但脚本路径为 gpt_oss2 专属。
- **默认 key**：`/workspace/code/.ssh_docker/id_ed25519`（脚本内置，无需 `-i` 手动指定）。
- **ProxyCommand 跳板节点**：`root@149.28.124.225`（hostname `chi2866`，登录/跳板节点）。WHY 用它做代理:计算节点不可直连,必须经此跳。
  - 跳板节点自身可跑 `sinfo` / `squeue` 查 Slurm 队列与节点状态。
- **计算节点命令格式**：`./.ssh-chi.sh root@chi2811 '<cmd>'`。
- 节点变更历史（含弃用/切换记录）：见 gpt_oss2/01-container-image-nodes.md

---
来源: remote-sync/SKILL.md, project_remote_sync_setup.md, mxfp8-grouped-gg-devloop/SKILL.md, feedback_flydsl_cache_staleness.md, project_mxfp8_grouped_wgrad_wl.md
