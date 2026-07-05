# 内核优化主循环:一假设一改动、正确性先行、accept/rollback 纪律

> 类别: 方法论 · 主题标签: optimize-loop, accept-rollback, scoring, representative-shapes

## 循环骨架
- 完整阶段: `DEFINE_TARGET → PREPARE_ENVIRONMENT → READ_HISTORICAL_TIPS → BASELINE(round-1, full validation) → [ANALYZE → OPTIMIZE → VALIDATE → ACCEPT/ROLLBACK]* → TERMINATION_CHECK → REPORT`。
- round-1 = baseline,优化尝试从 round-2 开始。
- 决策循环细化(每 round 内):1) focused benchmark 抓 baseline profile;2) 写下瓶颈分类 + 一个假设;3) kernel 里只改一件事;4) **先重跑 correctness**;5) 重跑同一 profile/benchmark 路径;6) 数据支持假设才 accept。WHY: 5) 必须走同一路径,否则测量口径漂移。

## 硬规则(不可违背)
- **一 round 一假设 + 一个有意义的 kernel 改动**。多改一起做无法归因。
- **correctness before performance**: 性能重测前先过正确性门,correctness 挂了这一 round 直接废。
- **benchmark 全部 active validation set**,禁止只挑子集报数(no cherry-picking)。
- **accept-or-rollback 干净**: 未达标 `git checkout` 干净丢掉,回到上一个已 accepted baseline。每轮实验落成一个 git commit,达标才留。
- 连续两次 rollback → 触发 stagnation review。
- 判定永远由脚本跑固定 harness 说了算,**不采信 agent 嘴上说变快了**。

## 自动化 dispatch 判定(sync/flydsl_kernel_optimizer.py)
- 无人值守: 每轮"提议改动 → 远端编译 + correctness + benchmark → 达标 commit 否则 revert",最后 claude ultrareview 再 squash。
- `--bench-cmd` 脚本最后一行 stdout 打印 `{"ok":bool,"tflops":number}`;`ok=false` 或没高出 `--min-gain`(默认 1%)就 revert。
- 模式抄 AutoKernel / Meta KernelAgent / AMD AgentKernelArena。

## 执行模式
| 模式 | 何时用 | rebuild |
|---|---|---|
| repo-mode | 小 scope、参数调、快构建;Triton(Python)无需 rebuild | HIP/CK 参数改需 `GPU_ARCHS=<arch> pip install --no-build-isolation -e . -v` |
| workspace-mode | 新 kernel、重度试错、重 build pipeline;最小本地 dev env(src/tests/bench 从 upstream 抽出) | 迭代完 SYNC_BACK **只回核心 kernel 改动,绝不回 scaffolding** |
- workspace-mode 的 VALIDATE 拆成 local gate + integration gate,**只有过 integration gate 才算 accepted**。
- 验收统一回项目跑完整 `pytest tests/pytorch/ -v`。

## representative shapes 与验证粒度
- BASELINE 时从 **Check=PASS 的 row** 里选 3-5 个 representative_shapes,覆盖 small(launch overhead)+ medium + large(compute/memory)两端,**优先高方差 shape**,记进 `manifest.yaml: representative_shapes`。
- GroupGemm/MoE 至少含一个 **SKEWED expert 分布**(如 top_k=1 cf=1.25 及一个近退化 case),不能只测 uniform。
- quick validation(每 VALIDATE round)= 3-5 个 representative_shapes 子集;full validation = 全部 target_shapes,用在 BASELINE、一个方向结束、最终验收、borderline/high-risk 时。
- quick→full 升级条件: improvement <5%,或高风险改动(control flow、data layout)。

## scoring
- 逐 row 取 primary_metric + Check;**任何 Check=FAIL/ERROR → 该候选 score 0 / 直接拒**(Check 是硬门)。
- 单 shape → 该 metric;多 shape → **几何平均(仅 PASS shape)**。
- fwd+bwd(training)用 combined-step 指标: `Combined Step TFLOPS = 6 / (2/Forward_TFLOPS + 4/Backward_TFLOPS)`(per shape),再对 PASS shape 取 geomean。
- 逐 shape 回归看 **Combined Step Time**,不看单独 fwd/bwd 分量的正负。
- 保留 score 向量(per-shape 值),避免总分掩盖局部回归。
- primary_metric 按 campaign 类型: compute-bound forward-only → Forward TFLOPS;compute-bound fwd+bwd → Combined Step TFLOPS;memory-bound(elementwise/quant)→ Forward/Backward GB/s。

## handoff 前置信息
- kernel 源路径(Code Map)、focused test/bench 命令、benchmark 输出格式/metric、quick validation harness、scoring 规则、execution_mode + rebuild 方式。

---
来源: remote-sync/SKILL.md, optimize-loop.md, SKILL.md, tool-rocprof/SKILL.md, optimize-handoff/SKILL.md
