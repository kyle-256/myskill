# FlyDSL 软件流水预取:prologue/steady/epilogue 三段、loop-carried state 携带每个 next

> 类别: 方法论 · 主题标签: prefetch, software-pipeline, loop-carried-state, buffer_load

## 三段结构(software-pipelined prefetch = three regions)
- **prologue(序言)**:发起 iteration 0 的 load(首 tile 数据),`_unwrap` 成 raw `ir.Value` 作 `init_state`。辅助数据(block table、per-block scale、offsets、在线 softmax m/l、accumulator)也必须一并作为 init state。
- **steady(循环体)**:`for iv, state in range(_start, _stop, _step, init=init_state)`,内部构建带 SSA phi 的运行时 `scf.for`(跑 N-1 trips)。每迭代:解包 state→swap → 读上次预取好的值 → 发 `prefetch(iv+1)`(下一 tile 的异步 load)→ `compute(current)` → `yield [acc] + next_tile`。
- **epilogue(尾段)**:循环退出后 `results` 持有最后 yield 的值,消费末次预取(末迭代)。
- **stop 取 N-1**:末迭代放进 epilogue,循环体只跑到 N-1。

## loop-carried state 携带规则
- **携带每个 next**:load 依赖的每一个输入都要进 `init=`(offsets、block-table indices、per-block scales、online-softmax m/l、accumulators)。
- **类型保留**:优先用 FlyDSL 内部类型(`fx.Int32`/`fx.Float32`/`Vector`/`ArithValue`);支持的 state 类型:f32 标量、vector、i32、i64、index。
- **只在边界 unwrap**:仅在 `init=`/`yield` 边界、或底层 helper 明确要求 raw `ir.Value` 处才 unwrap:`v.ir_value() if hasattr(v,'ir_value') else v`;body 内保持 typed。

## buffer_load / 异步语义(WHY)
- `buffer_load`(GPU global load)**异步立即返回**、后台取数,只在消费指令处才需数据。
- 编译器在**首个消费者**处插 `s_waitcnt`——提前发 load 只是给 scheduler slack,它**本身不设 `vmcnt(N)` 也不移除 barrier**。
- prefetch/double-buffer 把延迟藏在 compute 后:总时间从 `N*(load+compute)` 降到约 `load + N*max(load,compute)`。

## A0 跨 tile LDS 预取
- `gpu.barrier()` 完成后 LDS 有效,**立即**把第一个 A pack 从 LDS 读进 VGPR(`lds_load_packs_k64`),让首个 `ds_read` 延迟(~20–40 cycle)藏在随后的 VMEM load 后面。

## 实证:PA decode kernel(112us,0.75× vs Gluon)
- 携带 **15 个 loop-carried 值**:8×`vector<4xi32>` K 数据、1×i32 partition_start、2×i32 block table、2×f32 running_max/sum(在线 softmax)、2×`vector<4xf32>` PV 累加器。
- ISA 结果:8 个 K-prefetch `buffer_load_dwordx4` 出现在 loop body 末尾(PV MFMA 之后),与 MFMA 流水 drain 重叠;序言 8 个 K loads,epilogue 只 8 个 V loads。

---
来源: prefetch-data-load/SKILL.md, flydsl-kernel-authoring/SKILL.md, programming-model.md, debug-flydsl-kernel/SKILL.md, gemm-optimization/SKILL.md
