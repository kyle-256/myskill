---
name: flydsl-mi355-optimizer
description: >-
  MI355X (gfx950) 上优化 FlyDSL fp8/mxfp4 GEMM/attention kernel 的知识库：远程连接/选卡/同步、
  通用优化与 profiling 方法论、以及大量踩过的坑（tile 尺寸、occupancy、测量噪声、LDS/寄存器、
  race、autotune、各种死胡同）。三个文件夹共 100+ 张原子卡片，按需打开对应卡片而不是全量加载。
when_to_use: >-
  优化/调试/benchmark FlyDSL 或 Primus-Turbo 的 GEMM/attention kernel；连接远端 MI355X (chi2811 等)
  跑测；诊断性能回退、掉点、race、SNR 崩；查"某个方向是不是已经试过且失败了"；写 autotune dispatch；
  确认 tile/occupancy/LDS 该怎么选。凡是碰 gfx950 FlyDSL kernel 性能/正确性的活都先翻这里。
user-invocable: true
---

# FlyDSL MI355X 内核优化知识库

MI355X (gfx950 / CDNA4) 上 FlyDSL fp8/mxfp4 GEMM 内核优化的沉淀。内容来自 myskill、
`.claude/memory`、FlyDSL 官方 `.claude/skills` 与 Primus-Turbo `agent/skills` 的合并去重。

## 怎么用

- 这是**渐进式披露**的库：本文件只是导航，**按当前任务打开对应卡片**（用 Read），别一次性全读。
- 三个文件夹：`connection/`（怎么连上远端跑）、`methodology/`（怎么优化/debug/看利用率）、
  `pitfalls/`（踩过的坑——**最重要**，动手前先查这里，避免重走死路）。
- 每张卡是一个原子主题，卡尾有 `来源:` 可回溯。死胡同用 `❌ 别再试` 标注。
- 硬件默认 **gfx950 (MI355X)**；gfx942 (MI300) 仅作跨代对照，gfx1250/RDNA 不在本库讨论重点。
- 动手前的最短路径：① 翻 `pitfalls/` 确认方向没被判负 → ② 翻 `methodology/` 找对应做法 →
  ③ 翻 `connection/` 把代码弄上卡跑测。

---

## connection/ — 远程连接 / 选卡 / 同步 / 构建

- [02-free-gpu-triage-rocm-smi](connection/02-free-gpu-triage-rocm-smi.md) — 选空闲卡：rocm-smi 三件套(showuse/showpids/showmemuse)，sinfo/squeue 不可信
- [03-gpu-coexist-oom-judgement](connection/03-gpu-coexist-oom-judgement.md) — 能否与他人共存、OOM 判据
- [09-docker-exec-run-format](connection/09-docker-exec-run-format.md) — 远端容器内跑法：docker exec + HIP_VISIBLE_DEVICES
- [10-venv-isolation-two-checkout](connection/10-venv-isolation-two-checkout.md) — mxfp4/tensorwise 两份 checkout 与各自 venv 隔离
- [11-rsync-local-remote](connection/11-rsync-local-remote.md) — 本地→远端 rsync 同步规程
- [12-flydsl-remote-canonical-sync](connection/12-flydsl-remote-canonical-sync.md) — FlyDSL 远端为主（编译需 GPU）+ JIT 缓存坑
- [14-build-rebuild-rules](connection/14-build-rebuild-rules.md) — 按需 build：何时重编 .so vs 清 flydsl cache
- [01-jump-host-topology-ssh-key](connection/01-jump-host-topology-ssh-key.md) — 跳板机拓扑与 SSH key
- [04-nfs-shared-mount](connection/04-nfs-shared-mount.md) — 集群共享 NFS /mnt/vast/kyle/code2
- [05-container-run-flags](connection/05-container-run-flags.md) — 起 mlperf_gptoss 容器固定 flag 与验证
- [06-container-from-saved-image](connection/06-container-from-saved-image.md) — 从共享盘 tar 恢复容器（首选）
- [07-docker-disk-cleanup](connection/07-docker-disk-cleanup.md) — docker 根盘清理与 docker save 暂存坑
- [08-prod-nodes-container-current](connection/08-prod-nodes-container-current.md) — 生产节点/容器/rocm 版本现状
- [13-git-push-ssh-override](connection/13-git-push-ssh-override.md) — Primus-Turbo/FlyDSL git push 与 author 覆盖
- [15-triton-version-upgrade](connection/15-triton-version-upgrade.md) — triton 3.6→3.7 升级与 baseline 作废
- [16-container-crash-spill-guard](connection/16-container-crash-spill-guard.md) — 大 spill 崩容器与 dump ISA 防护
- [99-misc](connection/99-misc.md) — 其它零散事实

---

## methodology/ — 通用优化 / debug / profiling / FlyDSL 编程

### 优化循环 & 测量纪律
- [01-optimize-loop-structure](methodology/01-optimize-loop-structure.md) — 主循环：一假设一改动、正确性先行、accept/rollback
- [02-robust-timing-discipline](methodology/02-robust-timing-discipline.md) — Robust timing：统一 `_robust_time`、warmup、buffer 复用防虚高
- [03-measurement-noise-cross-process](methodology/03-measurement-noise-cross-process.md) — 跨进程 A/B 是幻觉：同进程交错 round-robin 取 min
- [33-fair-comparison-baseline](methodology/33-fair-comparison-baseline.md) — 可比 A/B 基线：两边裸调 kernel 绕 dispatcher
- [35-perf-baseline-regime](methodology/35-perf-baseline-regime.md) — 性能基线数字与 regime 识别

### 正确性 & race 诊断
- [04-correctness-gate-snr-det](methodology/04-correctness-gate-snr-det.md) — 正确性门：SNR 阈值、det=0 bit-exact ≥500-2000 run、outdiff=0 证同源
- [05-race-diagnosis-cross-wave-lds](methodology/05-race-diagnosis-cross-wave-lds.md) — cross-wave LDS-barrier race 逐一排除法
- [06-long-k-hw-nondeterminism](methodology/06-long-k-hw-nondeterminism.md) — 长 K HW 级非确定：判 race 用短 K 做干净 det0
- [07-jit-cache-invalidation](methodology/07-jit-cache-invalidation.md) — FlyDSL JIT 缓存失效：改算错的金标准

### profiling & 利用率
- [08-isa-dump-authority](methodology/08-isa-dump-authority.md) — ISA dump 是寄存器/AGPR/LDS/occupancy 唯一权威
- [09-rocprofv3-kernel-trace-tflops](methodology/09-rocprofv3-kernel-trace-tflops.md) — rocprofv3 --kernel-trace 冷测 TFLOPS
- [10-rocprofv3-pmc-counters](methodology/10-rocprofv3-pmc-counters.md) — PMC counter：L2/HBM 效率、LDS 带宽 vs 延迟、哪些不可信
- [11-rocprof-compute-regime-classify](methodology/11-rocprof-compute-regime-classify.md) — 先认 regime(memory/compute/stall-bound)再选 lever
- [12-att-trace-mfma-stall](methodology/12-att-trace-mfma-stall.md) — ATT trace 做 stall 根因
- [13-kernel-trace-analysis-hotspot](methodology/13-kernel-trace-analysis-hotspot.md) — code.json 逐指令 stall 分析
- [14-occupancy-vgpr-agpr-lds-budget](methodology/14-occupancy-vgpr-agpr-lds-budget.md) — 占用率计算：512 寄存器文件 VGPR+AGPR 共享

### FlyDSL 编程模型
- [15-flydsl-authoring-spine-layout](methodology/15-flydsl-authoring-spine-layout.md) — authoring 主线：layout 粘合、MMA atom 锚点、四步 divide/partition
- [16-flydsl-tracer-control-flow](methodology/16-flydsl-tracer-control-flow.md) — tracer 编译期-vs-设备控制流：range_constexpr vs range
- [17-flydsl-prefetch-software-pipeline](methodology/17-flydsl-prefetch-software-pipeline.md) — 软件流水预取：prologue/steady/epilogue 三段
- [18-flydsl-lds-allocator-swizzle](methodology/18-flydsl-lds-allocator-swizzle.md) — LDS 分配与 swizzle layout、XOR/padding 消 bank
- [30-authoring-pattern-reuse](methodology/30-authoring-pattern-reuse.md) — 复用范式：抄既有变体、只分叉 loop body、融合单发射
- [36-code-style-turbo-conventions](methodology/36-code-style-turbo-conventions.md) — turbo/FlyDSL 代码风格：英文注释、命名、复用 helper
- [37-kernel-classify-debug-oob](methodology/37-kernel-classify-debug-oob.md) — 内核分类骨架、错误隔离 debug、OOB 静态区间
- [39-flydsl-atom-extension-pipeline](methodology/39-flydsl-atom-extension-pipeline.md) — atom 两级类型与编译 pipeline、添加 target atom op

### GEMM 有效杠杆（正向）
- [20-tile-size-selection](methodology/20-tile-size-selection.md) — tile 尺寸：256×256 主 tile、小-M BM128、三级 tiling、MFMA 数账
- [21-l2-swizzle-group-n-xcd](methodology/21-l2-swizzle-group-n-xcd.md) — L2 swizzle：1D M-cluster / 2D band(group_n) / XCD remap
- [22-wgrad-variable-k-dispatch](methodology/22-wgrad-variable-k-dispatch.md) — wgrad variable-K：band-cyclic interleave、masked/persist crossover
- [23-prefetch-double-buffer-async](methodology/23-prefetch-double-buffer-async.md) — LDS ping-pong 双缓冲与 async copy
- [24-hot-loop-scheduling-hints](methodology/24-hot-loop-scheduling-hints.md) — 热循环调度提示：sched_* 交错、gfx942 sync vs gfx950 async
- [25-epilogue-store-cshuffle-permlane](methodology/25-epilogue-store-cshuffle-permlane.md) — Epilogue 存：CShuffle vs permlane16_swap 免 LDS 转置宽存
- [26-i64-addressing-srd-rebase](methodology/26-i64-addressing-srd-rebase.md) — >4GB 寻址：per-tile i64 SRD rebase、_readfirstlane pin SGPR
- [27-register-pressure-agpr-accum](methodology/27-register-pressure-agpr-accum.md) — 寄存器压力：asm-inplace 把 accum 挪进 AGPR 消 spill
- [28-whole-loop-asm-lds-feed-bound](methodology/28-whole-loop-asm-lds-feed-bound.md) — whole-loop 4-wave 结构与 LDS-feed bound 账
- [29-flydsl-emit-knobs-mxfp4](methodology/29-flydsl-emit-knobs-mxfp4.md) — mxfp4 4-wave 生产 emit 杠杆清单
- [19-gfx950-transpose-read-mfma](methodology/19-gfx950-transpose-read-mfma.md) — gfx950 MFMA transpose load：DS_READ_B64_TR_*
- [32-split-k-few-tile](methodology/32-split-k-few-tile.md) — Split-K 修少-tile 大-K 欠订阅
- [38-cross-lane-mfma-primitives](methodology/38-cross-lane-mfma-primitives.md) — 跨 lane 原语、mxfp4 pack、online-softmax log2
- [40-low-precision-dequant-fuse](methodology/40-low-precision-dequant-fuse.md) — 低精度融合 dequant 进 MFMA 内循环

### autotune / dispatch / 调度
- [31-autotune-dispatch-design](methodology/31-autotune-dispatch-design.md) — Autotune 五原则：balanced 计时、M-branch、hysteresis、静态 cache key、never-regress
- [41-primus-turbo-prod-autotune](methodology/41-primus-turbo-prod-autotune.md) — Primus-Turbo 生产 timed autotune 四轴
- [34-fleet-scheduling-multi-agent](methodology/34-fleet-scheduling-multi-agent.md) — N-GPU N-agent 事件驱动调度
- [99-misc](methodology/99-misc.md) — 其它零散事实

---

## pitfalls/ — 踩过的坑（最重要，动手前先查）

### tile 尺寸 / occupancy / 寄存器压力
- [16-tile-size-256-not-128](pitfalls/16-tile-size-256-not-128.md) — **别用小/矩形 tile：256×256 唯一可行点**，feed-bound 缩 tile 净负
- [17-tile-size-bk128-loop-overhead](pitfalls/17-tile-size-bk128-loop-overhead.md) — mxfp4 BK128 死路：loop 固定开销碾压收益
- [13-occupancy-register-bound-not-lds](pitfalls/13-occupancy-register-bound-not-lds.md) — occupancy 是 register-bound 非 LDS-bound：512 合并 VGPR 池
- [14-occupancy-maxnreg-agpr-deadend](pitfalls/14-occupancy-maxnreg-agpr-deadend.md) — 死路：maxnreg 强制 accum_vgpr=0、AGPR 搬移救不了溢出
- [15-occupancy-quantum-boundary](pitfalls/15-occupancy-quantum-boundary.md) — 占用率量子边界：只有跨 allocation quantum 才买到 wave
- [22-register-pressure-prefetch-spill](pitfalls/22-register-pressure-prefetch-spill.md) — register-bound 墙下预取/double-buffer 全 spill
- [23-agpr-occupancy-specific](pitfalls/23-agpr-occupancy-specific.md) — AGPR/VGPR-form 是 occupancy-specific：4w 用 AGPR、8w 用 VGPR-form

### 测量 & benchmark 噪声
- [06-measurement-noise-floor](pitfalls/06-measurement-noise-floor.md) — 噪声地板 ~5%、DVFS 功耗受限、多轮 interleaved 才可信
- [07-measurement-noise-clock-throttle](pitfalls/07-measurement-noise-clock-throttle.md) — 掉频/热节流假胜：冷 GPU boost、僵尸进程、autotune 赶上坏热态
- [08-measurement-noise-parallel-bench](pitfalls/08-measurement-noise-parallel-bench.md) — 并行/跨 GPU bench 口径必须一致
- [09-measurement-host-wrapper-overhead](pitfalls/09-measurement-host-wrapper-overhead.md) — 小 shape host/wrapper 开销淹没 kernel：走 raw op
- [10-measurement-fake-tf-snr-gate](pitfalls/10-measurement-fake-tf-snr-gate.md) — 虚高 TFLOPS：SNR<0 跳过计算、超 peak、do_bench 不可靠
- [11-measurement-race-bit-exact](pitfalls/11-measurement-race-bit-exact.md) — SNR 掩盖低概率 race：只有 bit-exact 30000+ 次多跑测得出
- [12-correctness-gate-diagnosis](pitfalls/12-correctness-gate-diagnosis.md) — "是不是自己改坏"诊断捷径：跨后端比 SNR
- [36-snr-tolerance-gate](pitfalls/36-snr-tolerance-gate.md) — 低精度容差假门：element-wise tolerance 无意义，SNR 才是 gate

### LDS / L2 / 数据通路
- [18-lds-vs-l2-a-direct-load](pitfalls/18-lds-vs-l2-a-direct-load.md) — 死路：A/B 从 LDS 改 direct global load，长 K 掉 2.5×
- [19-lds-capacity-3stage-deadend](pitfalls/19-lds-capacity-3stage-deadend.md) — LDS 容量：gfx950=160KB/gfx942=64KB，3-stage 双缓超限净负
- [20-lds-bank-conflict-swizzle](pitfalls/20-lds-bank-conflict-swizzle.md) — bank 冲突：gfx942 32 vs gfx950 64 banks，mask 需重推
- [21-lds-bandwidth-8wave-ceiling](pitfalls/21-lds-bandwidth-8wave-ceiling.md) — 8-wave mxfp4 结构封顶 ~4690-4900T：三道墙皆因 2 waves/SIMD
- [46-decode-l2-hbm-clean](pitfalls/46-decode-l2-hbm-clean.md) — streaming decode L2 命中低是正常，无 KV-load 优化空间

### 正确性 / race / vmcnt
- [27-race-vmcnt-partial-drain](pitfalls/27-race-vmcnt-partial-drain.md) — partial-drain race：读写距离决定安全 defer
- [28-gfx950-vmcnt-spill-race](pitfalls/28-gfx950-vmcnt-spill-race.md) — gfx950 vmcnt 非 FIFO race：只在 spill 时暴露
- [29-race-readfirstlane-int64-addr](pitfalls/29-race-readfirstlane-int64-addr.md) — SRD/寻址：readfirstlane pin SGPR、大-G int32 溢出静默错
- [45-gfx950-hw-walled-races](pitfalls/45-gfx950-hw-walled-races.md) — HW-walled 死路：buffer_load_dwordx2_lds 不支持、SCVGPR prefetch racy
- [35-fp8-encoding-scale-gate](pitfalls/35-fp8-encoding-scale-gate.md) — FP8 编码跨代正确性门：FNUZ(gfx942) vs OCP(gfx950)
- [43-attention-softmax-neutral-values](pitfalls/43-attention-softmax-neutral-values.md) — attention：online-softmax rescale/NaN 守卫、空 partition 写中性值

### mxfp4/wgrad 具体死胡同
- [24-mxfp4-k28672-occ2-ceiling](pitfalls/24-mxfp4-k28672-occ2-ceiling.md) — K28672：occ=2 唯一途径 BK128 让 g2s 翻倍净亏 13%
- [25-mxfp4-epilogue-store-exposed](pitfalls/25-mxfp4-epilogue-store-exposed.md) — 胖形状 ~6% gap：bf16 store 完全暴露，重叠/变宽/atomic 全死路
- [26-wgrad-feed-bound-deadends](pitfalls/26-wgrad-feed-bound-deadends.md) — wgrad feed-bound 死路全清单：占用率非杠杆
- [30-prefetch-when-useless](pitfalls/30-prefetch-when-useless.md) — prefetch 何时无用/有害
- [34-dead-end-structural-general](pitfalls/34-dead-end-structural-general.md) — 结构性死路杂项

### autotune / dispatch 陷阱
- [31-autotune-dispatch-cache-rules](pitfalls/31-autotune-dispatch-cache-rules.md) — 禁 id(tensor) cache、can_handle 不 raise、per-shape 过拟合
- [32-autotune-not-adopted-candidate](pitfalls/32-autotune-not-adopted-candidate.md) — 别接永不被采纳的候选
- [33-persistent-vs-nonpersistent-vmcnt](pitfalls/33-persistent-vs-nonpersistent-vmcnt.md) — persistent vs 非持久 / vmcnt_hint 选择
- [44-cache-key-real-training-gain](pitfalls/44-cache-key-real-training-gain.md) — 缓存增益上限规则：id(weight) 上限=quant/step、id(activation) 命中≈0

### FlyDSL 前端 / tracer / 同步 / 构建
- [03-flydsl-tracer-literal-if-for](pitfalls/03-flydsl-tracer-literal-if-for.md) — tracer 字面 if/for 坑：编译期常量分叉必须在外层 Python 选好
- [02-flydsl-jit-cache-stale](pitfalls/02-flydsl-jit-cache-stale.md) — JIT 缓存不失效假象：key 不 hash 模块级方法/emit 旋钮
- [04-flydsl-frontend-authoring-traps](pitfalls/04-flydsl-frontend-authoring-traps.md) — 前端编写坑：buffer offset 单位/SmemPtr view cache/absf
- [05-flydsl-thrval-layout-atom](pitfalls/05-flydsl-thrval-layout-atom.md) — ThrVal/atom layout 静默错：#1 静默产错结果 bug 源
- [01-rsync-git-remote-sync-traps](pitfalls/01-rsync-git-remote-sync-traps.md) — 远端同步：rsync --delete / 锚定 / git dubious-ownership
- [40-build-etiquette-hipkittens](pitfalls/40-build-etiquette-hipkittens.md) — build/环境：陈旧源移植慢、HK 残留 undefined symbol、节点礼仪

### ISA / 调度 / profiling 陷阱
- [38-isa-scheduling-hints-flat](pitfalls/38-isa-scheduling-hints-flat.md) — sched_* 计数必须精确、s_setprio 归零、别用 blanket s_waitcnt 0
- [39-profiling-att-pmc-traps](pitfalls/39-profiling-att-pmc-traps.md) — ATT 无 cache counter/PMC 多 pass 挂 GPU/code.json AGPR-blind

### 代码风格 / 部署 / 跨代移植
- [41-code-style-dead-code-comments](pitfalls/41-code-style-dead-code-comments.md) — 源码禁调试痕迹/benchmark 数字、探针脚本不 git add
- [42-authoring-prod-signature-deploy](pitfalls/42-authoring-prod-signature-deploy.md) — 生产化：backend 签名一致/setdefault 遮蔽/cache key
- [37-cdna4-no-fallback-porting](pitfalls/37-cdna4-no-fallback-porting.md) — CDNA4-only 路径无 gfx942 fallback（跨代移植参考）
- [47-rdna-porting-wmma](pitfalls/47-rdna-porting-wmma.md) — RDNA/WMMA 移植坑（非 MI355，跨代参考）
- [99-misc](pitfalls/99-misc.md) — 其它零散事实
