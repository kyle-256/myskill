# prefetch 何时无用/有害：s_waitcnt 编译器控、memory-bound/compute-bound/1wave 溢出别加

> 类别: 踩过的坑 · 主题标签: prefetch, s_waitcnt, 软流水, latency-hiding

## s_waitcnt 由编译器控，程序员只能"重排"
- FlyDSL kernel 编译到 GCN ISA 时 **s_waitcnt 由编译器插入**，非程序员控制：不能直接消 s_waitcnt、不能强制 vmcnt(N>0)、不能消 barrier（barrier 来自显式 `gpu.barrier()` 或跨 wave reduce 原语）。
- prefetch 唯一能做的：**重排代码**让编译器把 s_waitcnt 放到足够 compute 之后。价值上限就是"改变编译器排布的输入"，不是直接控延迟。

## prefetch 何时无用/别用（先判断再动手）
- **loop body 已 memory-bound**（纯 load 无 compute）→ 无 compute 可掩盖延迟，不帮。
- **单迭代 loop（`range(1)`）** → 无下一迭代可取，无从预取。
- **MFMA 利用率已 >90% compute-bound** → 延迟已被掩盖，不帮。
- **占用率已 1wave/EU 且溢出** → 加预取吃更多寄存器反而更糟。
- 生效条件唯一：**compute(MFMA) 时长 ≥ load 延迟**。

## 正确性约束（重排前必须满足）
- 有数据依赖的 load 不能重排：block table lookup → cache load 必须顺序。
- 条件 load（如 `KV_QUANT_MODE` 下 scale）预取时**必须复制同样条件**。
- prefetch load 放在 **swap 之后、任何消费当前数据的 compute 之前**。
- swap 要简单：只解包不计算。

## 热点 Pattern 1-4 修法（batch load 预取）
- **Pattern1** V/K load 在 MFMA 循环内交替（每 load 只有 1 个 MFMA 的掩盖时间）→ stall_rate 80-95%。修法：把所有 V load **batch 到 QK MFMA 循环之前**预取进寄存器，让整个 QK MFMA 掩盖 VMEM 延迟。实测 **~20% 周期削减**。
- **Pattern2** 连续 load 背靠背无 compute 交织 → VMEM 队列饱和。修法：在当前 tile MFMA 计算期间预取下一 tile 的 K load（double-buffer）。
- **Pattern3** LDS prob 读紧接 PV MFMA → lgkmcnt stall。修法：先把所有 LDS 读 batch 发射，再统一做所有 MFMA，让 LDS 数据先就绪。
- **Pattern4** scale load 和使用只隔 TLOOP 个 MFMA → 用太早 stall。修法：把 scale load 放到 **block 最开头、K load 之前**发射，最大化延迟掩盖距离。

## ❌ 别再试：FP4_LDSR 手动提前发射 ds_read
- 手动把 a1/b1 的 ds_read 提到 iter 顶隐藏延迟 → **反退 ~20%**（K2048：2509 vs 3126）。
- 机制：破坏编译器已做的**分段 lgkmcnt 软流水**。`FP4_RAWSPLIT=1` 每-mfma 一块时，编译器已自动做 operand 前置 + staggered s_waitcnt（`vmcnt(5)lgkmcnt(7)` → `lgkmcnt(6)` → `lgkmcnt(3)` …）。
- LDS 读延迟隐藏**已由编译器做到位**，Python 层加不了价值。

---
来源: prefetch-data-load/SKILL.md, kernel-trace-analysis/SKILL.md, agpr_phase5_ldsr.md, agpr_rawasm_progress.md
