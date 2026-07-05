# N-GPU N-agent 事件驱动调度:GPU 永不空闲、diversify 换维度家族、进步阈值 1%

> 类别: 方法论 · 主题标签: fleet-scheduling, autotune-dispatch, multi-agent, harness

## 事件驱动调度(GPU 永不空闲)
- N-GPU N-agent:每收到一个 agent 完成通知,立刻做三件事——更新 `state.json`(**先读再写**,不丢历史)→ 释放 `gpu_busy` → 按优先级立即 dispatch 新 background agent。GPU 空出即填,永不空闲。
- dispatch 优先级:`queued_next` > `refining`(邻域扫) > `diversifying`(换维度家族) > `done`。
- 收敛判据:进步阈值 **1%**(单轮 gain < 1% 当噪声,不算进步);`no_progress` 连续 **3 轮** 封顶为 `done`;**never regress**(不回退到更差 cfg)。

## tuning harness 要点(避免测量污染)
- 直接调 kernel,**绕开 dispatcher**——dispatcher 自带 autotune,会污染 timing。
- 量化 / 数据准备**只做一次**,循环里只换 cfg。
- CUDA event timing:排序后**掐头尾各 20%**(trim outliers)。
- 写**完整 JSON**:`shape` / `best` / `all` 三段。`all` 必留——diversify 要看次优解分布。
- 备两套 grid:`broad`(粗扫)+ `refine`(邻域细扫)。

## diversify 必须换维度家族(不是同 grid 再扫一遍)
- 只换 BM/BN 反复扫同一 grid 无意义。换维度家族:
  - split-K
  - 不同 `mfma_nonkdim`
  - `chunk_size`
  - cache modifier
  - `waves_per_eu`
  - scale 预取

## 自动化 FlyDSL 优化循环(commit/revert)
- `sync/flydsl_kernel_optimizer.py`:无人值守跑 N 轮 "提议改动 → 远端编译 + correctness + benchmark → 达标 commit 否则 revert";最后 claude ultrareview 再 squash。
- 核心模式(抄 AutoKernel / Meta KernelAgent / AMD AgentKernelArena):每轮实验落成一个 **git commit**,没达标 `git checkout` 干净丢掉。
- correctness + 性能判定**永远脚本自己跑固定 harness 说了算**,不采信 agent 嘴上说"变快了"。
- `--bench-cmd` 脚本最后一行 stdout 打印 `{"ok":bool,"tflops":number}`;`ok=false` 或没高出 `--min-gain`(默认 **1%**)就 revert。

---
来源: gpu-fleet-tuning/SKILL.md, remote-sync/SKILL.md
