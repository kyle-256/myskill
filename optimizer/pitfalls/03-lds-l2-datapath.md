# LDS/L2 数据通路：容量、bank、带宽、预取与直读的边界

> 类别: 踩过的坑 · 主题标签: LDS-vs-L2, occupancy, direct-load, mxfp4-8wave, LDS容量, prefetch双缓, blockwise-vs-tensorwise, L2-thrash, LDS-swizzle, bank-conflict, gfx950-vs-gfx942, ping-pong, 8-wave, LDS-bandwidth, ds_read, register-ceiling, decode, paged-KV, L2/HBM, gfx1250-TDM, DRAM带宽, copy_, 宽向量, quant天花板, prefetch, s_waitcnt, 软流水, latency-hiding

## 死路：A/B 从 LDS 改 direct global load，长 K 掉 2.5×

- ❌ 别再试：把 A operand 从 LDS 改为 direct global `buffer_load`→VGPR（A-direct）。
  - 机制**正确**：LDS 流量砍到 1/3，消除 8-wave 对 A 的重复读。
  - 但把**低延迟 LDS 读换成高延迟 L2 读**，而 1 wg/CU 藏不住 L2 延迟 → 长 K（8192² K28672）**慢 2.5×：1990 vs intrinsic 5033 TF**。
  - 等价复现历史 X 变体失败模式。
  - raw-AGPR 腾 VGPR 也**救不了**：瓶颈是 **occupancy 不是 VGPR 数**。

- ❌ 别再试：B 操作数直载到 VGPR 跳过 LDS（b_vgpr）→ mxfp4-8wave 实测**慢 ~22%**（4096²×8192 77.8%），已 bit-exact 非坏但慢。
  - 原因：8wave 同 wave_n 的多 wave 本可协作共享 LDS 中的 B，b_vgpr 让每 wave 各自从 global 冗余重载 B → 省了 LDS 读却暴增 global 流量。
  - （另有 fp8 dense/grouped GEMM 调优环境下的说法是"慢 4×，因相邻 lane 地址差 K 步长 uncoalesced"，见 08-deadends.md，但那是不同 kernel/context，非本 mxfp4-8wave 场景，不可混用。）

- ❌ 别再试（实测推翻旧误判）：把 8-wave mxfp4 GEMM 的 occupancy 当 LDS-bound。真相是 **REGISTER-bound**。
  - 128 VGPR + 128 AGPR = 256 共享 512 寄存器文件 → 硬卡 **2 waves/SIMD = 1 wg/CU**。
  - 把 LDS 从 128KB 砍到 64KB（A 直读）occupancy **完全不变（1.98→1.99）**，证明 LDS 不是限制。
  - 要 2 wg/CU 需总寄存器 ≤128，但**累加器单独就 128 AGPR，不可能**。

## LDS 容量硬约束：gfx950=160KB/gfx942=64KB，3-stage 双缓超限净负

- **LDS 容量按架构硬上限**：gfx942(MI300X)=64KB/CU；gfx950(MI350/MI355X)=160KB/CU，group_segment 实测 =163840B=160KB/CU。超容量报 LDS overflow → 用 `SmemAllocator` 追踪分配。

- **❌ 别再试：gfx950 3-stage LDS 双缓冲**。每 stage 64KB×3=192KB>160KB → 无法 2 wave/CU，净负。3-stage persist(`pipe3`)=0.87× persist，同样因 LDS 超限。

- **❌ 别再试：3-stage B ring buffer(fwd) 加 prefetch distance**。实测中性(+0.05%)。WHY：L2 miss latency 来自 8 个不同 expert 权重 thrash 4MB/XCD L2，是 **capacity thrash**，不是 latency 可隐藏问题 → 加 prefetch distance 无效，**prefetch 治不了 L2 capacity thrash**。

- **MI300 blockwise 硬约束（❌ 别再试绕过）**：
  - scale block=128 → BK=128 锁死（每 K 块一对 a_s/b_s），**不能像 tensorwise 选 BK=64**，结构性慢。
  - LDS=64KB → BK=128+BM=BN=256 单 stage 刚好，**双 stage 不可能**（LDS 放不下 → 失去 prefetch pipelining）。
  - MFMA nonkdim 实测 32 永远不赢，锁 16。

- **blockwise 打不过 tensorwise 的结构性根因**：
  - tensorwise 能 BM=BN=256 / BK=64 / num_stages=2（LDS=65536=64KB exactly）→ 享受 2-stage prefetch。
  - blockwise 因 BK=128，最大 256×128×128 stages=1 无 prefetch（FMA 后是空泡）。
  - 实测：大 shape blockwise ~0.65-0.75×tensorwise；小 shape 0.85-0.99×tensorwise。

## LDS bank 冲突/swizzle：gfx942 32 banks vs gfx950 64 banks，mask 需重推

### bank 数按代不同，mask 必须重推
- **gfx942 = 32 banks，512 B 分配块**；**gfx950 = 64 banks，1280 B 分配块**（1280 B 对齐，no wrap）。
- bank 冲突 stride 随 bank 数变：**gfx942(32) stride=128 字节全冲突；gfx950(64) 同样 stride=128 字节只 2-way 冲突**（线程在 2 bank 间交替），要 **256 字节倍数(64*4)** 才全 64-way 冲突。
- WHY：为 32-bank 设计的 XOR swizzle mask 直接搬到 64-bank gfx950 会**留残留 2-way 冲突**，mask 需调宽。
- footprint 陷阱：原本在 512 B 上整除干净的 footprint，在 1280 B 上可能浪费整个块、掉一个 occupancy tier。移植时 swizzle/padding mask 都要重新推导。
- A 型 bank 冲突判据：ds_read/ds_write 自身 stall>100 cycle/hit，read2_b64/write2 的 offset 为 bank 数倍数（gfx942=32 / gfx950=64）。

### swizzle 写读路径必须完全一致
- **每个 LDS write 和每个 LDS read 用完全相同的 XOR swizzle**（XOR 是自逆）。
- ❌ 别再试 不对称 swizzle：写读 mask 不一致会**静默读错行，编译器零信号**——这是最常见的 LDS 数据损坏来源。
- swizzle vs padding 权衡：swizzle 零 LDS 开销但 mask 依赖架构、错了静默冲突；padding(stride+1)简单、无额外寄存器但吃 LDS、超限 kernel fail。**优先 swizzle**；LDS 有余量或 swizzle 难集成时用 padding；gfx950 160KB 给 padding 更多余量。

### LDS 写后读必须同步
- 任何依赖前 ds_write 的 ds_read 之间必须有 **s_waitcnt lgkmcnt(0) 或 s_barrier**。
- 若 **wave A 写、wave B 读，则必须 s_barrier（仅 lgkmcnt 不够）**。
- 写-读距离越长延迟隐藏越好。
- B 型 write-read 延迟暴露判据：ds_write 后紧跟 s_waitcnt lgkmcnt(0) 且 stall>2000，二者间指令太少。
- C 型 跨 wave reduce 串行化：ds_bpermute→lgkmcnt→s_barrier→ds_write→lgkmcnt→s_barrier→ds_read 链，reduce 区 barrier>4。

### ping-pong 每轮必 rotate buffer index
- LDS ping-pong 每次迭代必须旋转 buffer index（以及任何 parity flag / partner register buffer）；**漏一个 swap 下一轮读到 stale LDS，编译期不可见**。
- K loop 要**按对(PAIRS)展开**，使 `write_stage = read_stage ^ 1` 交替对齐；LDS write 之后**恰好保留一个 gpu.barrier()**。

### 无冲突时消冲突无意义（死坑）
- ❌ 别再试 实测本就无冲突时做 padding 消 bank conflict：SQ_LDS_BANK_CONFLICT=0 / ADDR_CONFLICT=0 / UNALIGNED_STALL=0 时消冲突毫无意义。
- 8-wave **SWZ1 已把 bank 冲突清零**：SWZ0=0.75 ratio、MfmaUtil 44% vs **SWZ1 MfmaUtil 60-62%**。
- ❌ 别再试 调度层杠杆：WLDSR 细 staggered lgkmcnt / SS sub-stream / INPLACE 全部 ≤baseline；拆细单一粗同步点只会约束 wave-switching 自由度、暴露更多 stall。

## 8-wave mxfp4 结构封顶 ~4690-4760T：三道墙皆因 2 waves/SIMD

- **结构性封顶**：skill09（2026-06-24）初测给出 **~4900T**（pipe 配置 med/min=4817/4855）；skill10（2026-06-25，PMC+调度实验后）最终裁定修正为 **~4690-4760T**（baseline min/med=4744/4690）。两者是同一课题先后两次迭代的数字，以 skill10 的最终裁定为准。三道墙全部根因 = **2 waves/SIMD**（8-wave = 2 waves/SIMD → 每 wave 硬顶 512/2 = 256 寄存器）。
  - **墙①：LDS 读 A operand 4× 冗余**。8-wave 的 2×4 A frag 被 4 个 N-wave 各读一遍，ds_read/flop = 0.0234 vs 4-wave 0.0156。
  - **墙②：LDS 160KB 装不下大 tile 双缓**。BN512 BK256 = 192KB > 160KB，无法 double-buffer。
  - **墙③：寄存器 256@occ2 装不下 64 accs**。要 128×128 方形 tile 需 256 AGPR 累加器，已占满 256，operand/预取 0 空间。给一个 wave 428 寄存器（172V+256A）的唯一办法 = 降到 1 wave/SIMD = 4 waves/wg = 就是 4-wave kernel 本身。

- **关键修正：瓶颈不是 LDS 带宽 bound**。PMC 推翻带宽假设，LDS 端口 ≥5× 余量。真正瓶颈 = **ds_read 延迟气泡 + MFMA 执行 bound**：occ=2 下第 2 个 wave 用 wave-switching 已尽量盖住 30 读/iter 的深 ds_read 延迟链。
  - ❌ 别再试 SC_VGPR 方向：去掉 20% LDS 访问，对有 5× 余量的端口毫无意义（针对的是不存在的带宽墙）。

- **ds_read 已在理论下限，无冗余可消**（实测推翻"重读冗余"假设）：
  - vraw 8-wave 主循环 A/B frag 已在 Python 层跨 quadrant 完全复用（a0→c00/c01，a1→c10/c11；b0→c00/c10，b1→c01/c11）。
  - ds_read 已在理论下限 **24 b128/iter（A16+B8）**，与 intrinsic 完全相同（均 192@K2048）。
  - 全部 ds_read_b128 已是最大单指令宽度：同 tile s=0/1 隔 64B、tile 间隔 2048B 不连续，无法更宽合并。
  - 所谓"N 子块重读 A 的 1.5× 冗余"**不存在**；phase-5a 看到的 A:B LDS insts = 2:1 是 tile 大小正当读量比，非重复读。
  - phase-4/5 在 8-wave 内追的 0.3~1.5% 残差是"8-wave 局部最优"内部的事，非结构杠杆。

- ❌ **别再试：8-wave 原生 occ=2 藏 store**（换 occ=2 藏 store-bound 形状）。实测全负：28672 −14%、6144³ −20%、8192²×4096 −20%。机制：occ=2 能藏 7-15% store，但 8-wave compute 赤字 14-20%（per-warp tile 减半 → B 复用减半 → ds_read/mfma 翻倍）远大于收益。与 4w+BK128 −13% 同结论。

- **>5200 必须走 4-wave**：occ=1，VGPR 512 能做 register double-buffer，已达 5351。4-wave 是结构性更优工作点（长 K），8-wave 结构上无法达到 4-wave 的长 K 性能。

## streaming decode L2 命中低是正常：memory 子系统干净无 KV-load 优化空间

- 独立 per-sequence paged-KV decode 的 **L2 命中率 ~1-3% 是预期正常**，不是 bug。streaming 无复用，每个 KV 字节只读一次。"提高 L2 命中率"是**非目标**——唯一能真实提高 L2 的是 KV 复用=共享前缀服务(workload/调度属性，非 kernel 改动)。
- **memory-bound decode 判断树**（三者都健康即达到该访问模式真实上限，无 KV-load 优化空间）：
  - `L2 < 5%` = 纯 streaming 无复用（预期正确）
  - `32B ≈ 0%` = 满 64B 线无浪费
  - `over-fetch = est_HBM / (ideal_GB × dispatches) ≈ 1.0x` = 只读所需数据
  - 达到的带宽 `= ideal_bytes / kernel_time`；对干净 streaming，**50-60% 理论 HBM 峰值是正常的**，不是可优化的低效。
- ❌ 别再试（在 memory 子系统已干净时找 KV-load 优化）：PA decode gfx942 实测(bs16 ctx131072 batch256) L2命中 **1.7%**、32B **0%**、over-fetch **1.04x**、**2.85TB/s = 54% 峰值** → memory 子系统干净，无 KV-load 优化空间。
  - 旁证1：`block_size 16→64` 回退 **+7.8%**（更大 block 反而更差，说明当前 block_size 已合适，不是 L2/blocking 问题）。
  - 旁证2：`dwordx8` 在 CDNA3 不存在——`dwordx4`/16B 是单向量 load 上限，别指望更宽 load。

### gfx1250 TDM MoE row-gather：必须 addr64，否则硬 hang

- TDM MoE row-gather **必须**用 carry-safe 的 `update_tensor_gather_descriptor_addr64`：它把 lo-add 的进位传播进 `addr_hi`。
- ❌ 别再试短版 addr-lo-only update：会**静默 wrap 一个 4 GiB page**，只在大 tensor 规模下暴露为 **HARD HANG（不是错误结果）**，极难 debug。
- gather descriptor 的构建（以及 B / B-scale descriptors）必须 **hoist 出 K loop**，否则循环 SALU-bound。

## torch copy_ ~5.0TB/s 不是 DRAM 天花板：自定义宽向量 copy 才达 ~6.3TB/s

- **MI355X 1R:1W DRAM copy 真上限 ≈ 6.3 TB/s**：须用自定义宽向量 copy 测得（`buffer_load`/`buffer_store` i32，vw=2/4 = 8/16B）。torch `copy_` 只有 ~5.0 TB/s（低 25%），**❌ 别再拿 torch copy_ 当 DRAM 天花板**——它自身有 25% 的开销缺口，不反映硬件物理上限。
- **WHY 关键**：曾据 torch `copy_` 的 ~5.0 TB/s 误判 quant "已贴墙 90% / 物理锁死"，**这是错的**。真上限是 ~6.3 TB/s，quant 还有空间，别据假天花板宣布物理无解。
- **vw2 vs vw4（8B vs 16B 事务）几乎无差别**：上限 ~6.3 TB/s 与事务粒度无关，**❌ 别再靠加大事务粒度（16B）指望突破带宽上限**——不是杠杆。
- 呼应 68（别轻易宣布物理无解）：宣布"贴墙/物理锁死"前，先用自定义宽向量 copy 测真上限，不要用带自身开销的 framework 原语（torch copy_）当基准。

## prefetch 何时无用/有害：s_waitcnt 编译器控、memory-bound/compute-bound/1wave 溢出别加

### s_waitcnt 由编译器控，程序员只能"重排"
- FlyDSL kernel 编译到 GCN ISA 时 **s_waitcnt 由编译器插入**，非程序员控制：不能直接消 s_waitcnt、不能强制 vmcnt(N>0)、不能消 barrier（barrier 来自显式 `gpu.barrier()` 或跨 wave reduce 原语）。
- prefetch 唯一能做的：**重排代码**让编译器把 s_waitcnt 放到足够 compute 之后。价值上限就是"改变编译器排布的输入"，不是直接控延迟。

### prefetch 何时无用/别用（先判断再动手）
- **loop body 已 memory-bound**（纯 load 无 compute）→ 无 compute 可掩盖延迟，不帮。
- **单迭代 loop（`range(1)`）** → 无下一迭代可取，无从预取。
- **MFMA 利用率已 >90% compute-bound** → 延迟已被掩盖，不帮。
- **占用率已 1wave/EU 且溢出** → 加预取吃更多寄存器反而更糟。
- 生效条件唯一：**compute(MFMA) 时长 ≥ load 延迟**。

### 正确性约束（重排前必须满足）
- 有数据依赖的 load 不能重排：block table lookup → cache load 必须顺序。
- 条件 load（如 `KV_QUANT_MODE` 下 scale）预取时**必须复制同样条件**。
- prefetch load 放在 **swap 之后、任何消费当前数据的 compute 之前**。
- swap 要简单：只解包不计算。

### 热点 Pattern 1-4 修法（batch load 预取）
- **Pattern1** V/K load 在 MFMA 循环内交替（每 load 只有 1 个 MFMA 的掩盖时间）→ stall_rate 80-95%。修法：把所有 V load **batch 到 QK MFMA 循环之前**预取进寄存器，让整个 QK MFMA 掩盖 VMEM 延迟。实测 **~20% 周期削减**。
- **Pattern2** 连续 load 背靠背无 compute 交织 → VMEM 队列饱和。修法：在当前 tile MFMA 计算期间预取下一 tile 的 K load（double-buffer）。
- **Pattern3** LDS prob 读紧接 PV MFMA → lgkmcnt stall。修法：先把所有 LDS 读 batch 发射，再统一做所有 MFMA，让 LDS 数据先就绪。
- **Pattern4** scale load 和使用只隔 TLOOP 个 MFMA → 用太早 stall。修法：把 scale load 放到 **block 最开头、K load 之前**发射，最大化延迟掩盖距离。

### ❌ 别再试：FP4_LDSR 手动提前发射 ds_read
- 手动把 a1/b1 的 ds_read 提到 iter 顶隐藏延迟 → **反退 ~20%**（K2048：2509 vs 3126）。
- 机制：破坏编译器已做的**分段 lgkmcnt 软流水**。`FP4_RAWSPLIT=1` 每-mfma 一块时，编译器已自动做 operand 前置 + staggered s_waitcnt（`vmcnt(5)lgkmcnt(7)` → `lgkmcnt(6)` → `lgkmcnt(3)` …）。
- LDS 读延迟隐藏**已由编译器做到位**，Python 层加不了价值。

---
来源: 08-deadends.md, agpr_phase5_lds.md, flydsl-kernel-authoring/SKILL.md, mi300-blockwise-gg-tuning/SKILL.md, 10-8wave-scvgpr.md, lds-optimization/SKILL.md, gfx950/kernel-implementation-notes.md, programming-model.md, optimization-directions.md, 09-8wave-ceiling.md, agpr_phase5_ldsr.md, diag_4w_vs_8w.md, project_mxfp4_epilogue_store.md, capture-kernel-trace/SKILL.md, kernel-trace-analysis/SKILL.md, mxfp8-8wave-devloop/SKILL.md, prefetch-data-load/SKILL.md, agpr_rawasm_progress.md
