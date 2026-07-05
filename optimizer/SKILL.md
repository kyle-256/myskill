---
name: flydsl-mi355-optimizer
description: >-
  MI355X (gfx950) 上优化 FlyDSL fp8/mxfp4/mxfp8 GEMM/attention kernel 的知识库：远程连接/选卡/同步
  （两套环境 gpt_oss=mlperf_gptoss/code2、gpt_oss2=mlperf_gptoss2/code3）、通用优化与 profiling 方法论、
  以及大量踩过的坑（tile 尺寸、occupancy、测量噪声、LDS/寄存器、race、autotune、各种死胡同）。
  三个文件夹共 140+ 张原子卡片，按需打开对应卡片而不是全量加载。
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

- **渐进式披露**：本文件只是导航，**按当前任务打开对应卡片**（Read），别一次性全读。
- 三个文件夹：`connection/`（怎么连上远端跑）、`methodology/`（怎么优化/debug/看利用率）、
  `pitfalls/`（踩过的坑——**最重要**，动手前先查，避免重走死路）。
- 每张卡是一个原子主题，卡尾 `来源:` 可回溯。死胡同用 `❌ 别再试` 标注。
- 硬件默认 **gfx950 (MI355X)**；gfx942 (MI300) 仅作跨代对照。
- **两套远端环境**（都在 chi2811，同 key/同跳板 149.28.124.225，但容器/挂载/venv 不同）：
  | | gpt_oss | gpt_oss2 |
  |---|---|---|
  | 容器 | `mlperf_gptoss` | `mlperf_gptoss2` |
  | 挂载 | `/mnt/vast/kyle/code2` | `/mnt/vast/kyle/code3` |
  | venv | `/opt/venv`(mxfp4)+`/opt/venv-tw`(tensorwise) | `/opt/venv`(mxfp8)+`/opt/venv-mxfp4` |
  | 重点 | mxfp4 / tensorwise fp8 | mxfp8 |
  ⚠️ 同一台 chi2811 上两容器并存，**互为"别人的、别碰"**（认容器名有无 "2" / 挂 code2 vs code3）。
  连接共享设施在 `connection/common/`，各自差异在 `connection/gpt_oss/` 与 `connection/gpt_oss2/`。
- 动手前最短路径：① 翻 `pitfalls/` 确认方向没被判负 → ② 翻 `methodology/` 找做法 →
  ③ 翻 `connection/`（选对环境）把代码弄上卡跑测。

---

## connection/ — 远程连接 / 选卡 / 同步 / 构建

### connection/common/ — 两套环境共享的基础设施
- [common/01-jump-host-topology-ssh-key](connection/common/01-jump-host-topology-ssh-key.md) — 跳板机拓扑(149.28.124.225)与 SSH key 种钥
- [common/02-free-gpu-triage-rocm-smi](connection/common/02-free-gpu-triage-rocm-smi.md) — 选空闲卡：rocm-smi 三件套，sinfo/squeue 不可信
- [common/03-gpu-coexist-oom-judgement](connection/common/03-gpu-coexist-oom-judgement.md) — 能否与他人共存、OOM 判据
- [common/07-docker-disk-cleanup](connection/common/07-docker-disk-cleanup.md) — docker 根盘清理与 docker save 暂存坑
- [common/13-git-push-ssh-override](connection/common/13-git-push-ssh-override.md) — git push 用本机 SSH key、author 覆盖
- [common/14-build-rebuild-rules](connection/common/14-build-rebuild-rules.md) — 按需 build：何时重编 .so vs 清 flydsl cache
- [common/15-triton-version-upgrade](connection/common/15-triton-version-upgrade.md) — triton 升级与 baseline 作废
- [common/16-container-crash-spill-guard](connection/common/16-container-crash-spill-guard.md) — 大 spill 崩容器与 dump ISA 防护
- [common/99-misc](connection/common/99-misc.md) — 权限/硬件表/MFMA 延迟/构建/ISA dump 等零散

### connection/gpt_oss/ — 环境 A（mlperf_gptoss · code2 · mxfp4/tensorwise）
- [gpt_oss/05-container-run-flags](connection/gpt_oss/05-container-run-flags.md) — 起 mlperf_gptoss 容器固定 flag
- [gpt_oss/06-container-from-saved-image](connection/gpt_oss/06-container-from-saved-image.md) — 从共享盘 tar 恢复容器
- [gpt_oss/08-prod-nodes-container-current](connection/gpt_oss/08-prod-nodes-container-current.md) — 生产节点/容器/rocm 版本
- [gpt_oss/09-docker-exec-run-format](connection/gpt_oss/09-docker-exec-run-format.md) — docker exec + HIP_VISIBLE_DEVICES
- [gpt_oss/10-venv-isolation-two-checkout](connection/gpt_oss/10-venv-isolation-two-checkout.md) — mxfp4/tensorwise 两 checkout、venv 隔离
- [gpt_oss/11-rsync-local-remote](connection/gpt_oss/11-rsync-local-remote.md) — 本地→远端 rsync（code2）
- [gpt_oss/12-flydsl-remote-canonical-sync](connection/gpt_oss/12-flydsl-remote-canonical-sync.md) — FlyDSL 远端为主 + JIT 缓存坑
- [gpt_oss/04-nfs-shared-mount](connection/gpt_oss/04-nfs-shared-mount.md) — 共享 NFS 挂载 code2

### connection/gpt_oss2/ — 环境 B（mlperf_gptoss2 · code3 · mxfp8/mxfp4）
- [gpt_oss2/01-container-image-nodes](connection/gpt_oss2/01-container-image-nodes.md) — 目标容器/镜像/生产节点现状
- [gpt_oss2/02-container-from-saved-image](connection/gpt_oss2/02-container-from-saved-image.md) — 从共享盘 tar 恢复 + docker commit
- [gpt_oss2/03-jump-host-ssh](connection/gpt_oss2/03-jump-host-ssh.md) — 跳板机连接 .ssh-chi.sh
- [gpt_oss2/04-bind-mount-paths](connection/gpt_oss2/04-bind-mount-paths.md) — bind mount 与 host↔容器路径映射
- [gpt_oss2/05-docker-exec-run-format](connection/gpt_oss2/05-docker-exec-run-format.md) — docker exec + HIP_VISIBLE_DEVICES（选对 venv）
- [gpt_oss2/06-venv-isolation-two-checkout](connection/gpt_oss2/06-venv-isolation-two-checkout.md) — mxfp8/mxfp4 两 checkout、venv 隔离（坑A/坑B）
- [gpt_oss2/07-venv-build-rebuild](connection/gpt_oss2/07-venv-build-rebuild.md) — venv 新建/重建、build_ext --inplace vs pip install -e
- [gpt_oss2/08-rsync-local-remote](connection/gpt_oss2/08-rsync-local-remote.md) — 本地→远端 rsync（code3）
- [gpt_oss2/09-git-push](connection/gpt_oss2/09-git-push.md) — git push（origin，不推显式 URL）
- [gpt_oss2/10-disk-cleanup](connection/gpt_oss2/10-disk-cleanup.md) — 磁盘清理边界
- [gpt_oss2/11-dev-checklist-bench-scripts](connection/gpt_oss2/11-dev-checklist-bench-scripts.md) — 快速开发 checklist + bench 脚本
- [gpt_oss2/12-mxfp8-kernel-worktree-layout](connection/gpt_oss2/12-mxfp8-kernel-worktree-layout.md) — mxfp8 内核工作树位置与关键文件
- [gpt_oss2/13-fp8-dense-tuning-legacy-sync](connection/gpt_oss2/13-fp8-dense-tuning-legacy-sync.md) — fp8 dense 调优主文件/probe/legacy sync.sh
- [gpt_oss2/14-mxfp8-pr-local-mirror](connection/gpt_oss2/14-mxfp8-pr-local-mirror.md) — mxfp8 PR 本地镜像

---

## methodology/ — 通用优化 / debug / profiling / FlyDSL 编程

### 优化循环 & 测量纪律
- [01-optimize-loop-structure](methodology/01-optimize-loop-structure.md) — 主循环：一假设一改动、正确性先行、accept/rollback
- [02-robust-timing-discipline](methodology/02-robust-timing-discipline.md) — Robust timing：统一 `_robust_time`、compile-once、官方 Timer 口径
- [03-measurement-noise-cross-process](methodology/03-measurement-noise-cross-process.md) — 跨进程 A/B 是幻觉：同进程交错取 min
- [33-fair-comparison-baseline](methodology/33-fair-comparison-baseline.md) — 可比 A/B：两边裸调 kernel 绕 dispatcher
- [35-perf-baseline-regime](methodology/35-perf-baseline-regime.md) — 性能基线与 regime 识别（vs Triton/GB200）

### 正确性 & race 诊断
- [04-correctness-gate-snr-det](methodology/04-correctness-gate-snr-det.md) — SNR 阈值、det=0 bit-exact ≥500-2000 run、outdiff=0
- [05-race-diagnosis-cross-wave-lds](methodology/05-race-diagnosis-cross-wave-lds.md) — cross-wave LDS-barrier race 逐一排除法
- [06-long-k-hw-nondeterminism](methodology/06-long-k-hw-nondeterminism.md) — 长 K HW 级非确定：判 race 用短 K
- [07-jit-cache-invalidation](methodology/07-jit-cache-invalidation.md) — FlyDSL JIT 缓存失效的金标准

### profiling & 利用率
- [08-isa-dump-authority](methodology/08-isa-dump-authority.md) — ISA dump 是寄存器/AGPR/LDS/occupancy 唯一权威
- [09-rocprofv3-kernel-trace-tflops](methodology/09-rocprofv3-kernel-trace-tflops.md) — rocprofv3 --kernel-trace 冷测 TFLOPS
- [10-rocprofv3-pmc-counters](methodology/10-rocprofv3-pmc-counters.md) — PMC counter：哪些可信/不可信
- [11-rocprof-compute-regime-classify](methodology/11-rocprof-compute-regime-classify.md) — 先认 regime 再选 lever（先 profile 再调常数）
- [12-att-trace-mfma-stall](methodology/12-att-trace-mfma-stall.md) — ATT trace 做 stall 根因
- [13-kernel-trace-analysis-hotspot](methodology/13-kernel-trace-analysis-hotspot.md) — code.json 逐指令 stall 分析
- [14-occupancy-vgpr-agpr-lds-budget](methodology/14-occupancy-vgpr-agpr-lds-budget.md) — 占用率：512 寄存器文件 VGPR+AGPR 共享

### FlyDSL 编程模型
- [15-flydsl-authoring-spine-layout](methodology/15-flydsl-authoring-spine-layout.md) — authoring 主线 + 关键 API 速查
- [16-flydsl-tracer-control-flow](methodology/16-flydsl-tracer-control-flow.md) — tracer 编译期-vs-设备控制流
- [17-flydsl-prefetch-software-pipeline](methodology/17-flydsl-prefetch-software-pipeline.md) — 软件流水预取三段
- [18-flydsl-lds-allocator-swizzle](methodology/18-flydsl-lds-allocator-swizzle.md) — LDS 分配与 swizzle、XOR/padding 消 bank
- [30-authoring-pattern-reuse](methodology/30-authoring-pattern-reuse.md) — 复用范式：抄变体、分叉 loop body、融合单发射
- [36-code-style-turbo-conventions](methodology/36-code-style-turbo-conventions.md) — turbo/FlyDSL 代码风格 + lint/CI 同源
- [37-kernel-classify-debug-oob](methodology/37-kernel-classify-debug-oob.md) — 内核分类骨架、debug、OOB 静态区间
- [39-flydsl-atom-extension-pipeline](methodology/39-flydsl-atom-extension-pipeline.md) — atom 两级类型与编译 pipeline

### GEMM 有效杠杆（正向）
- [20-tile-size-selection](methodology/20-tile-size-selection.md) — tile 尺寸：256×256 主 tile、小-M BM128
- [21-l2-swizzle-group-n-xcd](methodology/21-l2-swizzle-group-n-xcd.md) — L2 swizzle：M-cluster / 2D band(group_n) / XCD remap（含 fp8 big-N WIN）
- [22-wgrad-variable-k-dispatch](methodology/22-wgrad-variable-k-dispatch.md) — wgrad variable-K：band-cyclic、masked/persist crossover
- [23-prefetch-double-buffer-async](methodology/23-prefetch-double-buffer-async.md) — LDS ping-pong 双缓冲与 async copy
- [24-hot-loop-scheduling-hints](methodology/24-hot-loop-scheduling-hints.md) — 热循环调度提示 sched_*
- [25-epilogue-store-cshuffle-permlane](methodology/25-epilogue-store-cshuffle-permlane.md) — Epilogue 存：CShuffle vs permlane16_swap
- [26-i64-addressing-srd-rebase](methodology/26-i64-addressing-srd-rebase.md) — >4GB 寻址：per-tile i64 SRD rebase
- [27-register-pressure-agpr-accum](methodology/27-register-pressure-agpr-accum.md) — asm-inplace 把 accum 挪进 AGPR 消 spill
- [28-whole-loop-asm-lds-feed-bound](methodology/28-whole-loop-asm-lds-feed-bound.md) — whole-loop 4-wave 结构与 LDS-feed bound（含 fp8 big-K drain-removal WIN）
- [29-flydsl-emit-knobs-mxfp4](methodology/29-flydsl-emit-knobs-mxfp4.md) — mxfp4 4-wave 生产 emit 杠杆
- [19-gfx950-transpose-read-mfma](methodology/19-gfx950-transpose-read-mfma.md) — gfx950 MFMA transpose load DS_READ_B64_TR
- [32-split-k-few-tile](methodology/32-split-k-few-tile.md) — Split-K 修少-tile 大-K 欠订阅
- [38-cross-lane-mfma-primitives](methodology/38-cross-lane-mfma-primitives.md) — 跨 lane 原语、mxfp4 pack、online-softmax log2
- [40-low-precision-dequant-fuse](methodology/40-low-precision-dequant-fuse.md) — 低精度融合 dequant + scaled-MFMA 对标口径

### MXFP8 专题（dual-cast quant / preshuffle / scale / whole-loop / grouped）
- [42-mxfp8-dualcast-quant-kernel](methodology/42-mxfp8-dualcast-quant-kernel.md) — dual-cast quant kernel 架构/分段计时/集成/数值验证
- [43-mxfp8-preshuffle-scale-layouts](methodology/43-mxfp8-preshuffle-scale-layouts.md) — scale preshuffle：layout-1 / b-comb / element-offset / 逐层隔离
- [44-mxfp8-lds-coalesced-transpose-write](methodology/44-mxfp8-lds-coalesced-transpose-write.md) — 带宽 roofline + LDS-合并转置写（制胜招）+ tile 选配
- [45-mxfp8-scale-pack-opsel-prefetch](methodology/45-mxfp8-scale-pack-opsel-prefetch.md) — scale_pack(op_sel) + 大-K scale group 预取
- [46-mxfp8-wholeloop-port](methodology/46-mxfp8-wholeloop-port.md) — whole-loop 从 fp4 移植：MFMA 格式/fp8 读/swizzle/store
- [47-mxfp8-flydsl-full-coverage-nofallback](methodology/47-mxfp8-flydsl-full-coverage-nofallback.md) — FlyDSL 必须全覆盖 MX case（禁 fallback）+ shape gate
- [48-mxfp8-grouped-wgrad-vark](methodology/48-mxfp8-grouped-wgrad-vark.md) — grouped var-K wgrad：瓶颈/scale-prefetch/软流水/AST-rewrite/occ=2 ROI
- [49-mxfp8-grouped-quant-fusion-batched](methodology/49-mxfp8-grouped-quant-fusion-batched.md) — grouped quant：融合 meta prologue / batched 权重 quant / HIP 契约
- [50-mxfp8-grouped-fair-compare-occ-ceiling](methodology/50-mxfp8-grouped-fair-compare-occ-ceiling.md) — 公平对标/dense 目标/fwd-dgrad autotune/occ=1 上限 PMC
- [51-mxfp8-e2e-timing-node-results](methodology/51-mxfp8-e2e-timing-node-results.md) — e2e 计时口径/节点占用检查/vs GB200 结果

### autotune / dispatch / 调度
- [31-autotune-dispatch-design](methodology/31-autotune-dispatch-design.md) — Autotune 五原则
- [41-primus-turbo-prod-autotune](methodology/41-primus-turbo-prod-autotune.md) — Primus-Turbo 生产 timed autotune 四轴
- [34-fleet-scheduling-multi-agent](methodology/34-fleet-scheduling-multi-agent.md) — N-GPU N-agent 事件驱动调度
- [99-misc](methodology/99-misc.md) — 其它零散事实

---

## pitfalls/ — 踩过的坑（最重要，动手前先查）

### tile 尺寸 / occupancy / 寄存器压力
- [16-tile-size-256-not-128](pitfalls/16-tile-size-256-not-128.md) — **别用小/矩形 tile：256×256 唯一可行点**
- [17-tile-size-bk128-loop-overhead](pitfalls/17-tile-size-bk128-loop-overhead.md) — mxfp4 BK128 死路：loop 固定开销碾压
- [13-occupancy-register-bound-not-lds](pitfalls/13-occupancy-register-bound-not-lds.md) — occupancy 是 register-bound 非 LDS-bound
- [14-occupancy-maxnreg-agpr-deadend](pitfalls/14-occupancy-maxnreg-agpr-deadend.md) — maxnreg/AGPR 搬移救不了 VGPR 溢出
- [15-occupancy-quantum-boundary](pitfalls/15-occupancy-quantum-boundary.md) — 占用率量子边界
- [22-register-pressure-prefetch-spill](pitfalls/22-register-pressure-prefetch-spill.md) — register-bound 墙下预取/double-buffer 全 spill
- [23-agpr-occupancy-specific](pitfalls/23-agpr-occupancy-specific.md) — AGPR/VGPR-form 是 occupancy-specific

### 测量 & benchmark 噪声
- [06-measurement-noise-floor](pitfalls/06-measurement-noise-floor.md) — 噪声地板 ~5%、DVFS、多轮 interleaved、best-of-N
- [07-measurement-noise-clock-throttle](pitfalls/07-measurement-noise-clock-throttle.md) — 掉频/热节流假胜
- [08-measurement-noise-parallel-bench](pitfalls/08-measurement-noise-parallel-bench.md) — 并行/跨 GPU bench 口径一致
- [09-measurement-host-wrapper-overhead](pitfalls/09-measurement-host-wrapper-overhead.md) — 小 shape host/wrapper 开销淹没 kernel（含 tiny-op）
- [10-measurement-fake-tf-snr-gate](pitfalls/10-measurement-fake-tf-snr-gate.md) — 虚高 TFLOPS（含 fp8 TN big-shape 瓶颈画像）
- [11-measurement-race-bit-exact](pitfalls/11-measurement-race-bit-exact.md) — SNR 掩盖低概率 race（含官方 deterministic 口径）
- [12-correctness-gate-diagnosis](pitfalls/12-correctness-gate-diagnosis.md) — "是不是自己改坏"诊断捷径
- [36-snr-tolerance-gate](pitfalls/36-snr-tolerance-gate.md) — 低精度容差假门：SNR 才是 gate

### LDS / L2 / 数据通路
- [18-lds-vs-l2-a-direct-load](pitfalls/18-lds-vs-l2-a-direct-load.md) — 死路：A/B 从 LDS 改 direct global 长 K 掉 2.5×
- [19-lds-capacity-3stage-deadend](pitfalls/19-lds-capacity-3stage-deadend.md) — LDS 容量：3-stage 双缓超限净负
- [20-lds-bank-conflict-swizzle](pitfalls/20-lds-bank-conflict-swizzle.md) — bank 冲突：gfx942 32 vs gfx950 64 banks
- [21-lds-bandwidth-8wave-ceiling](pitfalls/21-lds-bandwidth-8wave-ceiling.md) — 8-wave mxfp4 结构封顶 ~4690-4900T
- [46-decode-l2-hbm-clean](pitfalls/46-decode-l2-hbm-clean.md) — streaming decode L2 命中低是正常
- [50-torch-copy-fake-bw-ceiling](pitfalls/50-torch-copy-fake-bw-ceiling.md) — torch copy_ 5.0TB/s 不是天花板，宽向量 copy 才 6.3TB/s

### 正确性 / race / vmcnt
- [27-race-vmcnt-partial-drain](pitfalls/27-race-vmcnt-partial-drain.md) — partial-drain race：读写距离决定安全 defer
- [28-gfx950-vmcnt-spill-race](pitfalls/28-gfx950-vmcnt-spill-race.md) — gfx950 vmcnt 非 FIFO race（第一类，含 harness）
- [29-race-readfirstlane-int64-addr](pitfalls/29-race-readfirstlane-int64-addr.md) — SRD/寻址：readfirstlane pin SGPR、int32 溢出
- [45-gfx950-hw-walled-races](pitfalls/45-gfx950-hw-walled-races.md) — HW-walled 死路 + 三类 race 速查（LDS WAR / 跨 die XCD L2）
- [35-fp8-encoding-scale-gate](pitfalls/35-fp8-encoding-scale-gate.md) — FP8 编码跨代门 FNUZ/OCP + FlyDSL 编译错误修法清单
- [43-attention-softmax-neutral-values](pitfalls/43-attention-softmax-neutral-values.md) — attention：online-softmax rescale/NaN 守卫

### mxfp4/mxfp8/wgrad 具体死胡同
- [24-mxfp4-k28672-occ2-ceiling](pitfalls/24-mxfp4-k28672-occ2-ceiling.md) — mxfp4 K28672：occ=2 途径净亏 13%
- [25-mxfp4-epilogue-store-exposed](pitfalls/25-mxfp4-epilogue-store-exposed.md) — mxfp4 胖形状 ~6% gap：store 暴露，重叠/变宽/atomic 全死路
- [26-wgrad-feed-bound-deadends](pitfalls/26-wgrad-feed-bound-deadends.md) — wgrad feed-bound 死路全清单（含 persistent-across-groups 判负）
- [30-prefetch-when-useless](pitfalls/30-prefetch-when-useless.md) — prefetch 何时无用/有害
- [34-dead-end-structural-general](pitfalls/34-dead-end-structural-general.md) — 结构性死路杂项
- [51-mxfp8-scale-delivery-tax-not-mfma](pitfalls/51-mxfp8-scale-delivery-tax-not-mfma.md) — mxfp8 残余 gap 是 scale 投递税，非 scaled-MFMA 指令税
- [52-mxfp8-wholeloop-port-deadend](pitfalls/52-mxfp8-wholeloop-port-deadend.md) — mxfp8 whole-loop 移植整体死路（occ=1 天花板 + vendored 严禁）
- [53-mxfp8-grouped-overrun-layout-pollution](pitfalls/53-mxfp8-grouped-overrun-layout-pollution.md) — grouped MX over-run 污染 + BLOCK_M=128 少启动块假象
- [54-mxfp8-quant-cost-e2e-neutral](pitfalls/54-mxfp8-quant-cost-e2e-neutral.md) — grouped quant 占 fwd 35-46% 但 e2e 中性
- [55-mxfp8-layout-scale-pack-plumbing](pitfalls/55-mxfp8-layout-scale-pack-plumbing.md) — B-comb sizing / col 已转置 [K,M] / scale_pack 穿线
- [56-mxfp8-deadend-table](pitfalls/56-mxfp8-deadend-table.md) — mxfp8 死路清单（asm-AGPR/scale via LDS/host 重排/大 BM 不合并）
- [48-mxfp8-e8m0-scale-bcast-int32](pitfalls/48-mxfp8-e8m0-scale-bcast-int32.md) — mxfp8 e8m0 广播必须全程 Int32
- [49-mxfp8-recursion-module-isolation](pitfalls/49-mxfp8-recursion-module-isolation.md) — kernel 与 torch 同模块触发 RecursionError

### autotune / dispatch 陷阱
- [31-autotune-dispatch-cache-rules](pitfalls/31-autotune-dispatch-cache-rules.md) — 禁 id(tensor) cache、can_handle 不 raise、过拟合
- [32-autotune-not-adopted-candidate](pitfalls/32-autotune-not-adopted-candidate.md) — 别接永不被采纳的候选
- [33-persistent-vs-nonpersistent-vmcnt](pitfalls/33-persistent-vs-nonpersistent-vmcnt.md) — persistent vs 非持久 / vmcnt_hint 选择
- [44-cache-key-real-training-gain](pitfalls/44-cache-key-real-training-gain.md) — 缓存增益上限规则

### FlyDSL 前端 / tracer / 同步 / 构建
- [03-flydsl-tracer-literal-if-for](pitfalls/03-flydsl-tracer-literal-if-for.md) — tracer 字面 if/for 坑（含变量名折叠/scf.for 单值 carry）
- [02-flydsl-jit-cache-stale](pitfalls/02-flydsl-jit-cache-stale.md) — JIT 缓存不失效（含 /root/.flydsl/cache 磁盘缓存 env/asm 驱动假阴性）
- [04-flydsl-frontend-authoring-traps](pitfalls/04-flydsl-frontend-authoring-traps.md) — 前端编写坑
- [05-flydsl-thrval-layout-atom](pitfalls/05-flydsl-thrval-layout-atom.md) — ThrVal/atom layout 静默错
- [01-rsync-git-remote-sync-traps](pitfalls/01-rsync-git-remote-sync-traps.md) — 远端同步坑（含显式 URL push 不更新 tracking ref）
- [40-build-etiquette-hipkittens](pitfalls/40-build-etiquette-hipkittens.md) — build/环境坑

### ISA / 调度 / profiling 陷阱
- [38-isa-scheduling-hints-flat](pitfalls/38-isa-scheduling-hints-flat.md) — sched_* 计数精确、s_setprio 归零
- [39-profiling-att-pmc-traps](pitfalls/39-profiling-att-pmc-traps.md) — ATT/PMC/code.json 陷阱

### 代码风格 / 部署 / 跨代移植
- [41-code-style-dead-code-comments](pitfalls/41-code-style-dead-code-comments.md) — 源码禁调试痕迹、探针不 git add（含未用函数 rg 全仓查删）
- [42-authoring-prod-signature-deploy](pitfalls/42-authoring-prod-signature-deploy.md) — 生产化部署坑（含放行硬门禁+绿测先证伪）
- [37-cdna4-no-fallback-porting](pitfalls/37-cdna4-no-fallback-porting.md) — CDNA4-only 路径无 fallback（含 mxfp8 WL MFMA 格式）
- [47-rdna-porting-wmma](pitfalls/47-rdna-porting-wmma.md) — RDNA/WMMA 移植坑（非 MI355，跨代参考）
- [99-misc](pitfalls/99-misc.md) — 其它零散事实
