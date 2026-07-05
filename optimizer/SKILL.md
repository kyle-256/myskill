---
name: flydsl-mi355-optimizer
description: >-
  MI355X (gfx950) 上优化 FlyDSL fp8/mxfp4/mxfp8 GEMM/attention kernel 的知识库：远程连接/选卡/同步
  （两套环境 gpt_oss=mlperf_gptoss/code2、gpt_oss2=mlperf_gptoss2/code3）、通用优化与 profiling 方法论、
  以及大量踩过的坑（tile 尺寸、occupancy、测量噪声、LDS/寄存器、race、autotune、各种死胡同）。
  38 张章节卡（卡内 ## 小节分隔），按需打开对应卡片而不是全量加载。
when_to_use: >-
  优化/调试/benchmark FlyDSL 或 Primus-Turbo 的 GEMM/attention kernel；连接远端 MI355X (chi2811)
  跑测（分清 gpt_oss vs gpt_oss2 两个容器/挂载/venv）；诊断性能回退、掉点、race、SNR 崩；
  查"某个方向是不是已经试过且失败了"；写 autotune dispatch；确认 tile/occupancy/LDS 该怎么选。
  凡是碰 gfx950 FlyDSL kernel 性能/正确性的活都先翻这里。
user-invocable: true
---

# FlyDSL MI355X 内核优化知识库

MI355X (gfx950 / CDNA4) 上 FlyDSL fp8/mxfp4/mxfp8 GEMM 内核优化的沉淀。内容来自 myskill、
`.claude/memory`、FlyDSL 官方 `.claude/skills` 与 Primus-Turbo `agent/skills`，两个工作区
(gpt_oss / gpt_oss2) 合并去重。

## 怎么用

- **章节卡结构**：本文件是导航；每张卡是一个主题章节，卡内用 `## 小节` 分隔子话题。按当前任务打开对应卡（Read），别一次性全读。
- 三个文件夹：`connection/`（怎么连上远端跑）、`methodology/`（怎么优化/debug/看利用率）、`pitfalls/`（踩过的坑——**最重要**，动手前先查）。
- 死胡同用 `❌ 别再试` 标注；卡尾 `来源:` 可回溯；卡间用 `见 <路径>` 或 `见本卡「小节」` 交叉引用。
- 硬件默认 **gfx950 (MI355X)**；gfx942 (MI300) 仅作跨代对照。
- **两套远端环境**（都在 chi2811，同 key/同跳板 149.28.124.225，但容器/挂载/venv 不同）：
  | | gpt_oss | gpt_oss2 |
  |---|---|---|
  | 容器 | `mlperf_gptoss` | `mlperf_gptoss2` |
  | 挂载 | `/mnt/vast/kyle/code2` | `/mnt/vast/kyle/code3` |
  | venv | `/opt/venv`(mxfp4)+`/opt/venv-tw`(tensorwise) | `/opt/venv`(mxfp8)+`/opt/venv-mxfp4` |
  | 重点 | mxfp4 / tensorwise fp8 | mxfp8 |
  ⚠️ 同一台 chi2811 上两容器并存，**互为"别人的、别碰"**（认容器名有无 "2" / 挂 code2 vs code3）。连接共享设施在 `connection/common/`，各自差异在 `connection/gpt_oss/` 与 `connection/gpt_oss2/`。
- 动手前最短路径：① 翻 `pitfalls/` 确认方向没被判负 → ② 翻 `methodology/` 找做法 → ③ 翻 `connection/`（选对环境）把代码弄上卡跑测。

---

## connection/ — 远程连接 / 选卡 / 同步 / 构建

### connection/common/ — 两套环境共享
- [common/01-remote-access-jump-ssh](connection/common/01-remote-access-jump-ssh.md) — 跳板机(149.28.124.225)拓扑与 SSH key 种钥
- [common/02-pick-free-gpu](connection/common/02-pick-free-gpu.md) — rocm-smi 三件套选卡、共存/OOM 判据（sinfo 不可信）
- [common/03-docker-disk-crash-guard](connection/common/03-docker-disk-crash-guard.md) — docker 根盘清理、docker save 暂存坑、大 spill 崩容器防护
- [common/04-build-git-triton](connection/common/04-build-git-triton.md) — 按需 build/清 flydsl cache、git push、triton 版本
- [common/05-reference-misc](connection/common/05-reference-misc.md) — 权限/硬件表/MFMA 延迟/LDS 规格/ISA dump 等参考

### connection/gpt_oss/ — 环境 A（mlperf_gptoss · code2 · mxfp4/tensorwise）
- [gpt_oss/01-container-image-nodes](connection/gpt_oss/01-container-image-nodes.md) — 容器 flag/saved tar/生产节点现状
- [gpt_oss/02-run-and-venv](connection/gpt_oss/02-run-and-venv.md) — docker exec 跑法 + mxfp4/tensorwise venv 隔离
- [gpt_oss/03-sync-and-remote](connection/gpt_oss/03-sync-and-remote.md) — NFS(code2)/rsync 规程/FlyDSL 远端为主

### connection/gpt_oss2/ — 环境 B（mlperf_gptoss2 · code3 · mxfp8/mxfp4）
- [gpt_oss2/01-container-image-nodes](connection/gpt_oss2/01-container-image-nodes.md) — 容器/镜像/生产节点/跳板
- [gpt_oss2/02-mount-run-venv](connection/gpt_oss2/02-mount-run-venv.md) — bind mount + docker exec + venv 隔离(坑A/B)/构建
- [gpt_oss2/03-sync-git-disk](connection/gpt_oss2/03-sync-git-disk.md) — rsync(code3)/git push(origin)/磁盘清理
- [gpt_oss2/04-dev-worktree-scripts](connection/gpt_oss2/04-dev-worktree-scripts.md) — 开发 checklist/工作树布局/fp8 dense legacy sync/PR 镜像

---

## methodology/ — 通用优化 / debug / profiling / FlyDSL 编程

- [01-optimize-loop-benchmark](methodology/01-optimize-loop-benchmark.md) — 优化主循环、robust timing、测量噪声、公平对比、性能基线 regime
- [02-correctness-race-cache](methodology/02-correctness-race-cache.md) — SNR/det 门控、cross-wave LDS-barrier race 诊断、长 K 非确定、JIT 缓存失效
- [03-profiling-utilization](methodology/03-profiling-utilization.md) — ISA dump 权威、rocprofv3(kernel-trace/PMC)、regime 分类、ATT stall 根因、hotspot 分析
- [04-occupancy-and-tile](methodology/04-occupancy-and-tile.md) — 512 寄存器共享池、LDS/SGPR limit、256×256 主 tile、小-M BM128、MFMA 数账
- [05-lds-swizzle-prefetch-sched](methodology/05-lds-swizzle-prefetch-sched.md) — LDS 分配/swizzle、L2 swizzle(group_n/XCD)、双缓冲预取、热循环调度
- [06-register-wholeloop-emit](methodology/06-register-wholeloop-emit.md) — AGPR 累加消 spill、whole-loop LDS-feed bound、mxfp4 生产 emit 杠杆
- [07-epilogue-addressing-transpose](methodology/07-epilogue-addressing-transpose.md) — CShuffle/permlane16_swap epilogue、DS_READ_TR、i64 SRD rebase
- [08-splitk-crosslane-lowprec-wgrad](methodology/08-splitk-crosslane-lowprec-wgrad.md) — Split-K、跨 lane 原语/mxfp4 pack、低精度 dequant 融合、wgrad variable-K
- [09-flydsl-authoring](methodology/09-flydsl-authoring.md) — 骨架选型/layout 代数/tracer 控制流/软件流水/复用范式/debug-OOB/atom 扩展
- [10-code-style](methodology/10-code-style.md) — turbo/FlyDSL 代码风格与工程规范
- [11-mxfp8-dense](methodology/11-mxfp8-dense.md) — dual-cast quant/preshuffle/LDS 转置写/scale_pack 预取/whole-loop 移植/全覆盖
- [12-mxfp8-grouped](methodology/12-mxfp8-grouped.md) — grouped var-K wgrad/quant 融合/公平对标/e2e 计时与节点结果
- [13-autotune-fleet](methodology/13-autotune-fleet.md) — Autotune 五原则/生产四轴 never-regress/N-GPU N-agent 调度
- [99-misc](methodology/99-misc.md) — 其它零散事实

---

## pitfalls/ — 踩过的坑（最重要，动手前先查）

- [01-tile-occupancy-register](pitfalls/01-tile-occupancy-register.md) — **256×256 唯一可行 tile**、512 合并寄存器池、量子边界、maxnreg/预取 spill 死路
- [02-measurement-noise](pitfalls/02-measurement-noise.md) — 噪声地板/DVFS/掉频/并行口径/host 开销/虚高 TF/SNR 掩盖 race
- [03-lds-l2-datapath](pitfalls/03-lds-l2-datapath.md) — LDS 容量/bank/带宽封顶、A/B 直读掉 2.5×、prefetch 何时有害、torch copy_ 假天花板
- [04-race-vmcnt-correctness](pitfalls/04-race-vmcnt-correctness.md) — partial-drain/spill race、SRD 寻址、HW-walled 死路、SNR gate、attention 中性值
- [05-mxfp4-mxfp8-deadends](pitfalls/05-mxfp4-mxfp8-deadends.md) — occ 天花板、epilogue store 暴露、wgrad feed-bound、mxfp8 scale 投递税/WL 死路
- [06-autotune-dispatch](pitfalls/06-autotune-dispatch.md) — 禁 id(tensor) cache、别接不采纳候选、persistent 选择、缓存增益上限
- [07-flydsl-frontend-tracer](pitfalls/07-flydsl-frontend-tracer.md) — JIT 缓存不失效假象、字面 if/for 坑、前端编写陷阱、ThrVal/atom layout 静默错
- [08-sync-build](pitfalls/08-sync-build.md) — rsync/git 远端同步陷阱、build/环境坑（HK undefined symbol/LLVM OOM）
- [09-isa-profiling](pitfalls/09-isa-profiling.md) — sched_* 计数/s_setprio、ATT vs PMC 分工、code.json AGPR-blind
- [10-code-style-deploy](pitfalls/10-code-style-deploy.md) — 源码禁调试痕迹、放行硬门禁+绿测先证伪、backend 签名/cache key
- [11-porting-encoding](pitfalls/11-porting-encoding.md) — FNUZ/OCP 跨代编码门、CDNA4/RDNA 移植、"是不是自己改坏"诊断捷径
- [99-misc](pitfalls/99-misc.md) — 其它零散事实
