---
name: gpu-fleet-tuning
description: 用 N 个 GPU + N 个并行 sub-agent 做 kernel/config 调优时的事件驱动调度模式。每收到 agent 完成通知就立刻判断进步并立刻分发新任务,GPU 永不空闲。当用户说"用 8 个 GPU 调优"、"agent teams"、"每个 GPU 占满"、kernel autotune 类任务时使用。
---

# gpu-fleet-tuning

把 N 个 GPU + N 个 sub-agent 当成一个调度池来用。**核心原则:GPU 永不空闲,事件驱动,不等同步点**。

## 什么时候用

- 有 ≥4 块同型号 GPU
- 任务可分解成"对一个 (shape/op/config) 跑 K 次 timing"这种独立单元
- 每个单元几分钟到几十分钟
- 调优本身有迭代性(round 0 → round 1 → ...),后续 round 依赖前一 round 的结果

不适用:单 GPU 任务、长整体训练任务、调优单元之间有强依赖(必须串行)的场景。

## 状态文件 (你自己维护,每次通知都更新)

写到工作仓库的 `tuning_results/state.json`,**永远先读再写,不要原地覆盖丢历史**。

```json
{
  "round": 1,
  "per_shape_best": {
    "B4_M2048_N3584_K2048": {"ms": 0.237, "tflops": 508.3, "cfg": {...}, "round": 0, "agent_id": "..."}
  },
  "per_shape_status": {
    "B4_M2048_N3584_K2048": "exploring | refining | diversifying | done"
  },
  "no_progress_count": {"B4_M2048_N3584_K2048": 0},
  "tensorwise_ref": {"B4_M2048_N3584_K2048": 457.4},
  "gpu_busy": {"0": "agent_id_xxx", "1": null, ...},
  "queued_next": ["refine SHAPE_X @ anchor", ...]
}
```

## 调度循环

```
Round 0:  8 GPU × 8 (model,op) broad sweep   ← 一次性 dispatch 8 个 background agent
            │
            ▼
等任意 agent 完成通知 (background 自动 ping)─────────┐
            │                                       │
读 JSON → 更新 state → 释放 gpu_busy[N]              │
            │                                       │
对此 agent 覆盖的每个 shape:                          │
  if 进步 ≥ 1%:    update best, status=refining     │
  if 达 target:   status=done                       │
  if 无进步:       no_progress++                    │
                  if cnt < 3: status=diversifying  │
                  if cnt ≥ 3: status=done          │
            │                                       │
按优先级挑下一个任务给 GPU N:                         │
  A. queued_next 非空 → 取一个                       │
  B. status=refining 的 shape → 在 winner 邻域扫     │
  C. status=diversifying → 换探索方向 (split-K /     │
     不同 mfma / 不同 swizzle / scale 预取等)       │
  D. 全部 done → GPU 闲置                           │
            │                                       │
立即 dispatch 新 agent (background) ──────────────────┘
```

终止:全部 shape `status=done` 且 `gpu_busy` 全 None → 写 `best_configs.json` + 汇总表 → 把赢家落到 Python kernel selector。

## 决策规则(不要漏)

- **进步阈值 1%**:小于 1% 当噪声,no_progress++。
- **target = 上界**:有上界就用(同 kernel family 的兄弟 / 厂商 lib / 理论 peak)。摸到 target 直接 done,不浪费 GPU。
- **diversify 必须换方向**:同样的 grid 再扫一遍是浪费,要换"维度家族"——例如 BM/BN 换成 split-K、cache modifier、mfma 维度。
- **never regress**:refine 出来的 best 必须 ≥ 上一轮 best,否则用旧的。
- **no_progress 3 次封顶**:一个 shape 连续 3 轮没进步就 done,留给其他 shape。

## Sub-agent dispatch 规范

每个 sub-agent 的 prompt 里必须有:

```bash
HIP_VISIBLE_DEVICES=<N> TRITON_CACHE_DIR=/tmp/triton_cache_<N> \
  python <harness> --shapes "..." --tag <tag> --out <unique_path>.json
```

- `HIP_VISIBLE_DEVICES=N`:GPU 隔离,每个 agent 只见自己那一块。
- `TRITON_CACHE_DIR=/tmp/triton_cache_<N>`:每 GPU 独立 cache,**避免 8 个 agent 写同一个 .triton.lock 撞车**。
- `--out tuning_results/<unique_tag>.json`:每个 agent 写自己的文件,你后续聚合。
- agent prompt 必须包含"上一轮该 shape 的 best TFLOPS",方便它报告 delta。
- agent 必须 background 跑(`run_in_background=true`),否则你的会话被 8 个 30 分钟任务锁死。

## Harness 设计要点

调优脚本(harness)的几个不能省的特性:

1. **直接调 kernel,绕开 dispatcher**:dispatcher 自带 autotune,会污染 timing。
2. **量化/数据准备只做一次**:在 config sweep 外面建好 a_fp8, b_fp8, scales,循环里只换 cfg。
3. **CUDA event timing + trim outliers**:`start.record() → fn() → end.record() → sync`,排序后掐头尾各 20%。
4. **写完整 JSON**(`shape, best, all` 三段),不止写 best——后续 diversify 要看次优解分布。
5. **broad / refine 两套 grid**:`gen_grid()` 和 `gen_refine_grid(anchor_bm, anchor_bn)`,通过 `--mode {broad,refine} --anchor BM,BN` 切换。

## 常见踩坑

| 现象 | 原因 | 修法 |
|---|---|---|
| 多 agent 同时编同一个 kernel,Triton 报 cache lock | 共享 `~/.triton/cache` | 每 agent `TRITON_CACHE_DIR=/tmp/triton_cache_<N>` |
| Round-0 全部赢家 cfg 一致,refine 没空间 | broad grid 太窄 | 加新维度(mfma_nonkdim / chunk_size / waves_per_eu)再 refine |
| 某 shape refine 连续退步 | 时间噪声 + 进步阈值太松 | iters 增大 + 进步阈值 ≥ 2% |
| 调度器卡住等不到通知 | agent 不是 background | `run_in_background=true` 必须显式传 |
| 报"GPU 3 已占用" | 状态文件没及时写,以为闲实际已 dispatch | 每次 dispatch 前后都 read-modify-write state.json |
| 8 个 agent 同时跑某共享 csrc 编译 | csrc 不是 editable install / setup.py 触发 rebuild | 一次性 `pip install -e . --no-build-isolation`,后续 agent 只跑 Python |

## 最少代码示例(用 Bash 工具发 background)

```bash
# Round 0: 8 个 broad sweep,一次发完
for i in 0 1 2 3 4 5 6 7; do
  HIP_VISIBLE_DEVICES=$i TRITON_CACHE_DIR=/tmp/triton_cache_$i \
    python tools/tune.py --shapes "${SHAPES[$i]}" \
      --tag ${TAGS[$i]} --out tuning_results/${TAGS[$i]}.json \
      > /tmp/tune_$i.log 2>&1 &
done
wait
```

但用 sub-agent 比裸 background bash 好:agent 会读 JSON 并发回**结构化总结**,你直接拿来更新 state.json,不用自己 grep log。

## 与 memory 的关系

调优数据(每轮的 winner、TFLOPS、cfg)放在 `tuning_results/`(项目内),不要写进 memory——这些是会变的中间状态。memory 里只放**调优目标 / 上界来源 / 不变的约定**(比如"target=同 family 的 tensorwise kernel"、"BK=128 是 blockwise 的硬约束"这种)。
