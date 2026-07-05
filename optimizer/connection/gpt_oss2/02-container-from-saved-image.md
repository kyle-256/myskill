# gpt_oss2 从共享盘 tar 恢复容器 + docker commit 固化

> 类别: 连接 (gpt_oss2) · 主题标签: docker-commit, saved-image, mxfp4-rebuild, bind-mount

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

---
来源: remote-sync/SKILL.md
