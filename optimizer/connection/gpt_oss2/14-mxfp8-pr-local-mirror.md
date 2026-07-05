# gpt_oss2 mxfp8 PR 本地镜像(1:1 镜像远端)

> 类别: 连接 (gpt_oss2) · 主题标签: mxfp8, PR镜像, chi2811, gfx950

- **mxfp8 PR 本地镜像路径(实际编辑目录)**: `/workspace/code/gpt_oss2_docker/sync/mxfp8/Primus-Turbo/` —— 与远端 PR 分支逐一对应,可直接在本地改/看。
- `/workspace/code/mxfp8/Primus-Turbo/` 是**远程容器内路径**(`mlperf_gptoss2` 容器,chi2811),对应远程 host 路径 `/mnt/vast/kyle/code3/mxfp8/Primus-Turbo`,不是本地路径。
- **验证容器**: race + perf 验证统一用 `mlperf_gptoss2` 容器(chi2811 / gfx950)。WHY: race(竞态)与 perf(性能)两类验证需在真实 gfx950 硬件 + 该容器环境下跑,本地镜像只做代码同步不做验证。

---
来源: gfx950-vmcnt-race-debug/SKILL.md; remote-sync/SKILL.md(本地/远程路径分层表)
