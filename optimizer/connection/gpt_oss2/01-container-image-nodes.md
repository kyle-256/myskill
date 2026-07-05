# gpt_oss2 目标容器/镜像/生产节点现状

> 类别: 连接 (gpt_oss2) · 主题标签: container, image, prod-nodes, chi2811

- [目标容器/镜像] gpt_oss2 环境目标容器 = `mlperf_gptoss2`，镜像 `mlperf_gptoss2:flydsl`。所有 gpt_oss2 工作都在此容器内。
- [同节点勿碰] 同节点还有别人的容器，一律不要动：
  - `mlperf_gptoss`（注意无 "2"，挂 code2）——别人的，勿碰。
  - `yambda_smoke` 等——勿碰。
  - WHY: 误操作别人容器会破坏其环境；靠"有无 2 / 挂载 code vs code2"区分归属。
- [当前生产节点] `chi2811`（gfx950 / MI355X）。gpt_oss2 的 mxfp8 实验、grouped wgrad bench 均在此跑；commit/证伪结论都标注 chi2811。
- [节点变更历史] chi2774 于 06-30 弃用；chi2810 于 07-03 更换 → 当前落到 chi2811。WHY: 生产节点会被回收/轮换，认准 chi2811 才是当前有效节点。
- [flydsl 缓存陈旧问题] flydsl 磁盘缓存陈旧（staleness）问题正是在 `mlperf_gptoss2` / chi2811 (gfx950 MI355X) 环境下发现的。

---
来源: remote-sync/SKILL.md, project_remote_sync_setup.md, mxfp8-grouped-gg-devloop/SKILL.md, feedback_flydsl_cache_staleness.md, project_mxfp8_grouped_wgrad_wl.md
