# 掉频/热节流假胜：冷 GPU boost、僵尸进程压 10%、autotune 选型赶上坏热态

> 类别: 踩过的坑 · 主题标签: measurement-noise, clock-throttle, autotune-dispatch

## transient 掉频 → 假胜
- 曾把 8w 瞬时掉频测成 4w 赢 **1.42×**，复测仅 **1.01×**。WHY: 那一测正好赶上 8w 侧 transient 掉频/被抢卡。可疑就 `rocm-smi` 看是否掉频/被抢卡并立即复测。

## 僵尸 GPU 进程压 ~10%（掩盖 3 次回退）
- 测速前必须查杀僵尸 GPU 进程，否则带宽/时钟被压 **~10%**，结果严重偏低——曾把 **~2% 回退掩盖 3 次**。
- `rocm-smi --showpidgpus | grep 'using.*DRM'` 找真实 KFD PID。
- `<defunct>` 状态的 python 进程已死不占 GPU；**真占 GPU 的是有 KFD entry 的**。
- 杀完 `rocm-smi --showuse | grep 'GPU use'` 全 **0%** 才算干净。

## 不同 GPU boost 差 10-30%，必须同 GPU 重现
- 死坑：不同 GPU 的 clock boost 状态不同，同一 kernel 差 **10-30%**。GPU0 冷/boost 高 **5536 med** vs GPU7 正常工作温度 **5401 med**。
- 早先看到的 8w **4896/4910** 就是冷 GPU 假象；单 GPU 顺序测才排除并行热降频。
- 判据：GPU 换挡不算达标，目标必须在**同一 GPU 稳定重现**。

## autotune 一次性选型赶上坏热态
- dispatch 只在第一次调用某 shape 时跑候选竞赛并 cache。若 sweep 按固定顺序连测多 shape，某 shape 的选型时刻恰处 GPU 刚从冷启动/低时钟回升阶段，选出的候选可能不是稳态最快的。**不是 dispatch 逻辑错**，是那次选型赶上不具代表性热力状态。
- 短-K shape 冷/热差异 **>20%**，`warmup=10` 会 mis-pick，`warmup=250` 才稳定。**冷测短-K 是 mis-pick 高发区**。

## bench 前 set_auto_tune(False)
- bench 前必须 `set_auto_tune(False)`，否则每个 shape cold-start 跑一遍 autotune 污染 first-iter。
- HK 内部 `_autotune_pick` cache 是 process-local dict，warmup **20 iter** 足够 cache-hit 后才进 timing loop。

## 回归判据（先排噪声再定性）
- NN `16384×4096×4096=1011 TF` 是异常低点（同 shape M=8192 有 **2689**），属小-N 方阵 regime 的 autotune 选到坏配置/timing 抖动；小方阵（oproj 4096×4096）是 dense 最弱 regime。
- 真回归判据：同脚本重测某 shape 掉 **>8%（超噪声）**才算真回归。先排除 autotune 缓存没命中/别的进程抢卡。

---
来源: remote-sync/SKILL.md, flydsl-fp8-gemm-tuning/07-benchmarking.md, pr-merge-gate/SKILL.md, fp8-gemm-bench/SKILL.md, 08-deadends.md, 04-ceiling-analysis.md, flydsl-fp8-gemm-results/SKILL.md
