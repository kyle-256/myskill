# gpt_oss2 磁盘清理边界(overlay 根盘常年紧)

> 类别: 连接 (gpt_oss2) · 主题标签: disk-cleanup, overlay, docker, safety

- **背景**: gpt_oss2 根 overlay 盘常年紧张,需定期清理,但机器为共享环境,必须只清明确属我们且可再生的内容。
- **可清(可再生,安全)**:
  - `rm -rf /root/.cache/comgr` — comgr 编译缓存,可再生
  - `rm -rf /root/.cache/pip` — pip 下载缓存,可再生
  - `rm -rf /tmp/qgolden` — quant golden 临时输出,可再生
  - `rm -rf /tmp/rocprof_out` — rocprof 临时输出,可再生
  - `rm -rf /tmp/*.log` — 临时日志
  - `docker builder prune -f` — 只清 docker build 缓存层,WHY: 不触及镜像/容器本体
- **绝不动(共享资产,不可再生/属别人)**:
  - 别人的镜像 / 容器 / volumes — WHY: 共享机器,误删他人工作不可逆
  - `mlperf_gptoss`(注意无"2"，挂 code2) — **别人的**容器，与本项目容器 `mlperf_gptoss2`(挂 code3) 无关，绝不可碰
  - 禁 `docker volume prune` — WHY: 会连带删掉别人的 volume
  - 禁 `docker rmi` 别人镜像 — WHY: 破坏他人环境
- **本项目自己的容器**: `mlperf_gptoss2`(挂 code3)，勿与上述别人的 `mlperf_gptoss` 混淆

---
来源: remote-sync/SKILL.md (gpt_oss2, 章节: 远程 docker 持久化 / 磁盘清理)
