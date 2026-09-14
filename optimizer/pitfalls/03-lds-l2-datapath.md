# LDS/L2 数据通路：容量、bank、带宽、预取与直读的边界

> 类别: 踩过的坑 · 主题标签: LDS-vs-L2, occupancy, direct-load, mxfp4-8wave, LDS容量, prefetch双缓, blockwise-vs-tensorwise, L2-thrash, LDS-swizzle, bank-conflict, gfx950-vs-gfx942, ping-pong, 8-wave, LDS-bandwidth, ds_read, register-ceiling, decode, paged-KV, L2/HBM, DRAM带宽, copy_, 宽向量, quant天花板, prefetch, s_waitcnt, 软流水, latency-hiding

## 死路：A/B 从 LDS 改 direct global load，长 K 掉 2.5×

- ❌ 别再试：把 A operand 从 LDS 改为 direct global `buffer_load`→VGPR（A-direct）。
  - 机制**正确**：LDS 流量砍到 1/3，消除 8-wave 对 A 的重复读。
  - 但把**低延迟 LDS 读换成高延迟 L2 读**，而 1 wg/CU 藏不住 L2 延迟 → 长 K（8192² K28672）**慢 2.5×：1990 vs intrinsic 5033 TF**。
  - 等价复现历史 X 变体失败模式。
  - raw-AGPR 腾 VGPR 也**救不了**：瓶颈是 **occupancy 不是 VGPR 数**。

- ❌ 别再试：B 操作数直载到 VGPR 跳过 LDS（b_vgpr）→ mxfp4-8wave 实测**慢 ~22%**（4096²×8192，b_vgpr 仅达 intrinsic 的 77.8%），已 bit-exact 非坏但慢。
  - 原因：8wave 同 wave_n 的多 wave 本可协作共享 LDS 中的 B，b_vgpr 让每 wave 各自从 global 冗余重载 B → 省了 LDS 读却暴增 global 流量。
  - （另有 fp8 dense/grouped GEMM 调优环境下的说法是"慢 4×，因相邻 lane 地址差 K 步长 uncoalesced"，但那是不同 kernel/context，非本 mxfp4-8wave 场景，不可混用。）

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
- **gfx942 = 32 banks，LDS 分配粒度 512 B**；**gfx950 = 64 banks，LDS 分配粒度 1280 B**（1280 B 对齐，no wrap）。
- bank 冲突 stride 随 bank 数变：**gfx942(32) stride=128 字节全冲突；gfx950(64) 同样 stride=128 字节只 2-way 冲突**（线程在 2 bank 间交替），要 **256 字节倍数(64*4)** 才全 64-way 冲突。
- WHY：为 32-bank 设计的 XOR swizzle mask 直接搬到 64-bank gfx950 会**留残留 2-way 冲突**，mask 需调宽。
- footprint 陷阱：原本在 512 B 上整除干净的 footprint，在 1280 B 上可能浪费整个块、掉一个 occupancy tier。移植时 swizzle/padding mask 都要重新推导。
- （XOR mask 通式与 padding stride 推导见 methodology/05-lds-swizzle-prefetch-sched.md，本卡只列移植踩坑判据。）
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
- 8-wave **SWZ1 已把 bank 冲突清零**：SWZ0(swizzle 关)=0.75× ratio、MfmaUtil 44% vs **SWZ1(swizzle 开) MfmaUtil 60-62%**。
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
  - ds_read 已在理论下限 **24 b128/iter（A16+B8）**，与 intrinsic 完全相同（均 192@K2048）。（b128 = 单条 128-bit ds_read；192@K2048 = K=2048 配置下每 wave 主循环的 ds_read 总条数，与 intrinsic 逐条相同。）
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

## torch copy_ ~5.0TB/s 不是 DRAM 天花板：自定义宽向量 copy 才达 ~6.3TB/s

- **MI355X 1R:1W DRAM copy 真上限 ≈ 6.3 TB/s**：须用自定义宽向量 copy 测得（`buffer_load`/`buffer_store` i32，vw=2/4 = 8/16B）。torch `copy_` 只有 ~5.0 TB/s（低 25%），**❌ 别再拿 torch copy_ 当 DRAM 天花板**——它自身有 25% 的开销缺口，不反映硬件物理上限。
- **WHY 关键**：曾据 torch `copy_` 的 ~5.0 TB/s 误判 quant "已贴墙 90% / 物理锁死"，**这是错的**。真上限是 ~6.3 TB/s，quant 还有空间，别据假天花板宣布物理无解。
- **vw2 vs vw4（8B vs 16B 事务）几乎无差别**：上限 ~6.3 TB/s 与事务粒度无关，**❌ 别再靠加大事务粒度（16B）指望突破带宽上限**——不是杠杆。
- 呼应 68（别轻易宣布物理无解）：宣布"贴墙/物理锁死"前，先用自定义宽向量 copy 测真上限，不要用带自身开销的 framework 原语（torch copy_）当基准。
- **2026-08-04 融合 flash-bwd r09 复证，并追加一条更隐蔽的假天花板**：同一台 MI355X 上，torch `sum(dim=1)` 折叠 split-K 槽 = **4.48 TB/s**，同样字节的手写 FlyDSL 规约核 = **6.14 TB/s（+37%）**；torch 的 strided transpose+mul（lse 转置预缩放）= **1.1 TB/s**，换 LDS 分块使两侧都 128 B 连续后 = **2.86 TB/s（2.6×）**。⇒ 本卡"torch 原语低 25%"的结论在 reduction / transpose 上更严重（低 27~61%），不止 `copy_`。
  - ⚠️ **假天花板的第二种形态：把"自己当前最快的核"当成机器上限。** 该 campaign 早前用自家 dqred 实测 6.03 TB/s，据此在本地 memory 里写下"6.03 是内存速率封顶、别想手写核超它"，于是把 dk/dv 折叠留给 torch 又跑了 4 轮；本轮三个手写尾核实测 **6.14 / 6.24 / 6.26 TB/s**，与本卡记的真上限 ~6.3 TB/s 吻合。⇒ **"我的核 = 上限"和"torch = 上限"是同一个错误**；只有独立的宽向量 1R:1W 微基准才是上限基准，且任何"已封顶"的记录都要标注它是用什么测的。

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
  ⚠ **动手前先数 ISA,LLVM 在寄存器够用时自己就会做 Pattern3**。sparse-MLA decode producer(gfx950,
  VGPR 162/168 预算)上显式把 QK 的 8 条 `ds_read_b128` 整体前提:`s_waitcnt lgkmcnt(0)` 从 54 条降到 34,
  VGPR 162→174,墙钟 seq48/60 −1~−4.6%、**seq84 +2.7%** ⇒ 符号混杂、判负;而只把占用目标从 3 压到 2
  (`lds_pad`,源码一行不改)同样把 54 降到 33 ⇒ **那 54 条全排空本来就是编译器在占用约束下的取舍,不是漏做**。
  判据:先看 ISA 里下一 tile 的 load 是否已经被提到本 tile 的 MFMA 之前;若已经提了,Pattern3 无钱。
  (完整数据见 pitfalls/12 §轮 7 补丁)
- **Pattern4** scale load 和使用只隔 TLOOP 个 MFMA → 用太早 stall。修法：把 scale load 放到 **block 最开头、K load 之前**发射，最大化延迟掩盖距离。

## 深派发下 L2 命中的守恒量是 **WG 存活时间**，不是 tile 的 MFMA 条数

gpt-oss-20b E=32 wgrad（gfx950，OUT_M=2944，单卡 32 专家全驻，18/34.5 轮派发）实测锚点，
所有数字同一轮 rocprofv3 单遍 5 counter（`GRBM_GUI_ACTIVE / SQ_VALU_MFMA_BUSY_CYCLES /
SQ_INSTS_MFMA / TCC_HIT_sum / TCC_MISS_sum`）：

| 臂 | wall | SQ_INSTS_MFMA | MFMA-busy | L2 命中 | DRAM 读 | 等效带宽 |
|---|---|---|---|---|---|---|
| down (2,6,1,1) 开短-N 余数体 | 14.19 ms | 6.370e8 | 82.98% | 68.0% | 61.7 GB | 4.40 TB/s |
| down 同配置但**齐一 tile**（关余数体）| 14.27 ms | 6.795e8 | **88.35%** | **74.4%** | 50.9 GB | 3.57 TB/s |
| gate_up (1,1,1,1)，本就齐一 | 27.36 ms | 1.3023e9 | **90.66%** | 60.6% | 133.3 GB | 4.87 TB/s |

三条结论，按可迁移性排序：

1. **L2 命中不是直接约束**。gate_up 命中最低（60.6%）却 MFMA-busy 最高（90.66%），且它在
   4.87 TB/s 下照样跑满 —— 所以看到低命中先别去修 L2，先看 MFMA-busy 还剩多少。
   两条 shape 都在 83~91% busy，即**整算子离发射天花板只剩 9~17%**。
2. **余数体（短 M/短 N 体）的定价要打两折**。down 的短-N 体省 6.25% MFMA，实际只兑现
   **+0.6~1.2%**：它同时把 L2 命中打掉 6.4pp、MFMA-busy 打掉 5.4pp。
   经验折扣 ≈ **只有 20% 的余数 MFMA 收益能落到 wall 上**。
3. ❌ **别把余数 tile「配对成等成本 tile」**。让短-M 体（`a_halves=1`）的一个 WG 连做两个
   相邻 N-block，MFMA 精确 −3.89%、逐字节 bit-exact，但 wall **−20%**：
   L2 命中 68.0%→**44.5%**、DRAM +70%（请求总量反而 −1.7%，全部是命中率崩的）。
   机制：`a_halves=1` 的体是 **0.5× MFMA / 0.75× 字节 / 1.0× K 扫描**，两个串起来 = 1× MFMA
   但 **~1.5× 存活时间**。每 XCD 只有 32 个 CU 共驻，WG 活得比一"代"长就会横跨两个 class run，
   活跃 slab 从 12 翻到 ~24。**共驻窗口里要齐一的是存活时间，MFMA 条数齐一没有意义。**

### ❌ 别再试：256×192 tile 消 wgrad 的 N padding（含把 LDS pool 由 96 列补到 128 列）
2880/192=15、5760/192=30，看着是"零 N padding + tile 成本齐一"，但按上表的实测轮次分解
（齐一臂 T_round = 0.700 ms MFMA + 0.092 ms stall）反解为**负**：
- 与今天的短-N 体 **MFMA 完全相同**（两者 N padding 都已是 0），换来的只有齐一性；
- WG 数 +25%（144→180 tile），算术强度 `BM×BN/(BM+BN)` 从 128 掉到 **109.7 B/cell**；
- 真 96 列 pool（要 fork 一整套非 2 的幂地址代数）：22.5 轮 ×(0.526+0.081) = **13.64 ms，+0.9%**，
  落在 `_an_wg_cfg` ±0.7% 的噪声里；
- 补到 128 列 pool（B 每行又回到 256 B）：22.5 轮 ×(0.526+0.092) = **13.90 ms，−1.0%**。

补 pool 宽度这一步本身就是负的：它把省下来的 B 字节又还回去，只剩 WG 数 +25% 的开销。

### ❌ 别再试：小重排核去掉 LDS、改寄存器直接 gather（请求粒度反噬）
mxfp8 scale preshuffle（`_emit_lds_repack`，每 WG 4KB 进 4KB 出、一次 barrier）改成
"每个输出 lane 自己取 4 行、每行一条 dwordx4、零 LDS 零 barrier"：逐位一致，但
**孤立标尺 −12.98%**，K128 越大越差（−34.3 / −24.8 / −16.2%）。
- 机制：LDS 路径每条指令**连续读一行 64 B**，正好压在 gfx950 的读请求粒度上；寄存器
  gather 每条指令只读 16 B 却要跨 16 行 ⇒ 同样字节数换来 **~4×** 的 TCP/L1 请求。
- ISA 也没省（221 vs 209 条）：省掉的 ds_write/ds_read 被 gather 的地址算术吃回去。
- ✅ 反过来成立的是**加宽**：读相位 `vec_width=1→4`（`buffer_load_dwordx4` +
  `ds_write_b128`）叠加 `v_perm_b32` 打包（两条 perm + 一条 or 取代 4 条 shift-mask-or），
  逐位一致、ISA −32.5%、LDS 与占用不变。但见下条的价值口径。
- 🔁 再锚定（2026-09-11，sparse-MLA decode producer 的 Q publish，见 pitfalls/12 decode 节）：
  同一对结论在另一族上重演。去 LDS 改「每 wave 自取一份 Q 进寄存器」= **+1.5~3.5%**（更慢）；
  把 wave0 的 9 条 16B/lane 跨 16 行（整块 144 请求）改成 256 线程走连续 16B chunk
  （整块 72 请求，128B 对齐块的下界）= 最多 **−1.5%**，逐位一致、VGPR 不跨台阶。
  **请求数才是货币**这条判据可跨族直接用。
- ⚠ 价值口径：这类"只减指令不减字节"的改写，**孤立标尺 +2.2~3.9%，进真实 step 后
  只剩 −0.55%（0.13us/池）、折算 step 0.005%**。这些 dispatch 在 step 里是冷的、延迟
  受限（544-880 WG，空核发射地板 1.76us vs 实测 4.8-5.3us，每 CU 仅 ~2 个 WG 可换），
  发射槽不是绑定项。先用 kernel-trace 量池占比再决定要不要动。

### ❌ 别再试：FP4_LDSR 手动提前发射 ds_read
- 手动把 a1/b1 的 ds_read 提到 iter 顶隐藏延迟 → **反退 ~20%**（K2048：2509 vs 3126）。
- 机制：破坏编译器已做的**分段 lgkmcnt 软流水**。`FP4_RAWSPLIT=1` 每-mfma 一块时，编译器已自动做 operand 前置 + staggered s_waitcnt（`vmcnt(5)lgkmcnt(7)` → `lgkmcnt(6)` → `lgkmcnt(3)` …）。
- LDS 读延迟隐藏**已由编译器做到位**，Python 层加不了价值。

## 「请求数才是货币」要再往下修一层：**每行触碰的唯一 128B 行数**才是货币（gfx950 sparse-gather，实测定价谱）

DSV4 sparse-MLA prefill（4096 token × topk2048 × 576B/行，576MB 池均匀随机 index）上做的定价谱。
所有臂**指令流逐条相同**，只改「脚印」或「唯一行数」，同口径 60s ramp + HIP graph + min：

| 臂 | 改了什么 | us | Δ |
|---|---|---|---|
| control | — | 739.7 | — |
| 全 MALL 驻留 | row index 掩到 37MB 脚印 | 660.8 | **−10.7%** |
| 全 L2 驻留 | row index 掩到 1.15MB 脚印 | 348.1 | **−52.9%** |
| 唯一行 5→3 | chunk 偏移改成 (0,1,0,1)，请求数不变 | 537.4 | **−27.3%** |
| out 写脚印 67MB→8.4MB | store 的 dv 偏移去掉 tile 项 | 732.2 | −0.9% |
| PV MFMA 21→37 条 | 纯加计算，零额外访存 | 740.6 | **+0.1%** |

两条可跨族复用的判据：

1. **贵的是 TCC(L2) miss 本身，不是 DRAM 带宽。** 一条「每行多出来的唯一行」边际价 ~78us/(行·全量)，
   而同一条行走 MALL 还是走 HBM 只值 15.6us。⇒ 看到 gather 核 memory-bound，先问「唯一行数能不能少」，
   而不是「能不能让它 MALL 驻留」。后者在本例只值 10.7%，前者线性可兑。
2. **请求数在唯一行数不变时是二阶量。** 同一个核上把 TCP 请求从 7/行压到 5/行（奇数行相位对齐、
   算术逐位相同、VGPR 166→162、occ 不变）实测 **+0.9%（更慢）**；把唯一行数从 5 压到 3 则 −27.3%。
   原有「请求数才是货币」那条是在**请求宽度**语境下说的（64B→128B 合并同时减了行数），
   本条把它收窄：**合并请求只有在同时减少唯一行数时才赚**。

### 定价探针写法（alias 臂，可直接照抄）
让每条臂**发出完全相同的指令**，只改指令背后的地址集合，这样价差是纯粹的数据通路价：
- 缩脚印 → 把 row index `& MASK`（0x7FF ⇒ L2 驻留，0xFFFF ⇒ MALL 驻留）。
- 缩唯一行数 → 把 chunk 偏移序列由 `(0,1,2,3)` 改成 `(0,1,0,1)`。
- 测计算是否被盖住 → 复制 MFMA（`for step in range(K*2)` + `step % K` 取操作数），
  零额外访存。**本例 MFMA +76% 零成本 ⇒ 该核所有减指令/减 VALU/换占用的候选都不必做。**
结果会是错的（checksum 变），所以**记分尺量不了这些臂**（它在计时前先判 relL2 会直接抛），
必须另写一个同口径的非记分谱。

### 推论：不整除的行宽是**必须报备的布局问题**，不是能在 kernel 里绕的
576 不整除 128：偶行相位 0、奇行相位 64，两种相位都必然横跨 **5** 条 128B 行，其中 64B 属于邻行。
邻行同时命中的概率在随机 topk 下是 0.2%，所以 5→4.5 行**只能靠改布局**（拆成 512B + 64B 两个池）。
按上表边际价，−10% 行数 ≈ **−9~−10% 墙钟**。

### ❌ 别再试：靠「把 index 排序」制造跨 CTA 的 L2 co-sweep
理由不是 DRAM row locality（那是另一条），而是**并发窗口不够**：一次调用对 1.05M 行发 8.39M 次引用
（每行约 8 次），但同时在飞的 CTA 只有约 96/XCD、各自流 1.31MB，行早被 4MB L2 挤掉。
用「主机端预排序」这个**免费上界**量出来只有 **−1.5%**（727.8 vs 739.7）⇒ kernel 内做桶排/双调排序
（每 CTA 多数千条 LDS 操作）不可能回本。要吃到 L2 驻留臂那 −52.9%，必须做**跨 token 的 gather 批量化**
（先按 KV 行聚合再散回 token），那是改算法。


## ★★★ M0 下溢：`buffer_load … lds` 的 LDS 基址**不能为负**，所以「预减 k-block」的槽不能是 leaf 0

（2026-09-12 mxfp4 dense NT r50 实测；症状极具辨识度，值得先认症状）

mxfp4 whole-loop 的 g2s 用**同一条 `offset:` 立即数**喂 global 地址和 LDS 目标
（`_GBSK = KSTEP`，buf1 靠这条立即数免费跨一个 k-block），代价是 buf1 的 LDS 基址必须**预先减掉**
一个 k-block：`M0 = ptr(A_buf[b]) + wave*wstride − KSTEP`。基线只让 `A_buf[1]` 当 buf1，永远为正。
一旦把 A 环扩到 3 槽并**轮转**「哪个槽当 buf1」，轮转必然让**声明第一个** LDS leaf（偏移 0）去当 buf1
→ wave 0 的 `M0 = 0 − 128` 下溢 → 该 wave 那批 g2s 的写**被丢弃**（不是回绕到 LDS 顶端）。

- 症状：**正好一个 tile 索引全错**（2048 个 256×256 块里 512 个 = 每个 WG 一个 tile），
  SNR **55.5 → 14.8 dB**，`block_rel_max ≈ 1.25`（错得离谱，不是精度问题）。
- ⚠ 坏块集合别按 `bm` 直接读：`bm` 是 `grouped_xcd_pid` 之后的坐标。本例坏 `bm` = `{4..7, 20..23, …}`
  即 `pid//group_m ≡ 1 (mod 4)`，代回 `r = 256*(bx%8) + bx//8 + 64t` 才看出它就是 **t == 1**。
- ⚠ 写被丢弃 ⇒ LDS 里留着上一个 tile 的确定性数据 ⇒ **多次运行逐位相同**，看起来"不像 race"。
- ❌ 改 `fx.struct` 的字段顺序（"把 A 挪到 B 后面"）**修不了**：`SharedAllocator(static=True)`
  每个 leaf 是独立 `@__shared_alloc_*` 符号，摆放由后端定，源码顺序不决定地址。实测改了以后
  输出**逐位不变**（连错法都一样）。
- ✅ 修法：让轮转**永不把带 skew 的角色放到 0 号 leaf 上**。周期 4 的槽表
  `((0,1),(2,1),(0,2),(1,2))` 同时满足两个约束：k-even 槽永远不属于上一个 tile（跨 tile 预取要的），
  且 k-odd 角色只落在 1/2 槽。改完 SNR 回 **55.535 dB**、`|C|sum` 与基线**逐位相同**。

## ★★★ NT 不是「整个 operand」的属性,而是「每条 load 指令」的属性 —— 按复用度给块打标

(2026-09-12 GLM-5.2 TP4/EP4 a4w4 decode r12 实测,同批交错 3 对)

MoE 权重流式 GEMM 给 B 打 NT(`_bnt2`)整体是 +8% 的大杠杆,理由是「每个 expert slab 只读一次」。
但这个理由**对一部分块是假的**:token 数超过一个 m-block 的 expert(EP 里的共享专家必然如此)
会被它的**每个 m-block 各读一遍整块 slab**,这些块之间有**真实的 L2 复用**,而 NT 恰好把它掐死。
⇒ 正解不是「NT 全开/全关」,而是**同一个核里按块分类**:会被邻块重读的 slab 走 cached,其余保持 NT。

- **先验证 XCD 前提**(不满足则复用物理上不存在,别写代码):N-fastest grid 下同 expert 相邻 m-block
  的线性 WG id 相差 `grid.x`;只要 `grid.x` 是 XCD 数(8)的倍数,这些块就落在**同一个 XCD**、
  共享那 4MB L2。本例 stage1 `gx=32`、stage2 `gx=96` ⇒ 都满足。
- **判据**:kernel 内读 `sorted_expert_ids[blk±1] == expert`(两条标量 buffer_load,可忽略)。
  一个长度 ≥2 的同 expert 连续段里**每一块**都至少有一个同 expert 邻居,所以 prev/next 这一格窗口
  足够,不需要更宽的扫描。
- **主机端必须再 gate 一层**:`token_num > block_size_M` 时才编出这个分支。因为 if/else 会把
  GEMM body **复制两份**,而不可能有重复块的小 batch 只承担代价:b16 **+1.76us(+2.1%)**、b8 +0.35us。
  gate 上以后 b8/b16 逐读回到基线。⇒ 这不是可选优化,是这条路能不能落地的前提。
- **实测**:score 1.11075 → **1.13092(+1.82%)**,3 对逐对为正、区间不重叠;
  b64 176.13→174.31(−1.83us)、b128 211.37→204.71(−6.66us);rel_l2 不变(缓存提示不改结果)。
- ⚠️ **「跳过重复块」的减法上界会低估这条路**:那个探针量出上界 +2.11%(b64 −1.0us),
  而 cached 版 b64 实际 −1.83us,**超过上界**。因为探针只删掉那块的算力/流量,而 cached 版还顺手
  减轻了重复 slab 与其他流在 L2 上的互挤 ⇒ 别用「跳过」的上界去否掉「复用」这条路。
- sc0/GLC(绕 L1、只留 L2)与纯 cached 实测**同分**(1.13120 vs 1.13137)⇒ 跨 CU 的复用与 L1 无关,
  取写法简单的那个即可。
- ❌ **别再试(同动机的两条替代路)**:
  ①`persist_m=2`(让一个 WG 连做 2 个 m-block,把跨 WG 复用变成同 WG 复用)—— WG 数腰斩,
  score **1.016**、b64 194.7us(+20us),并行度损失远大于缓存收益。
  ②`xcd_swizzle=1`(把同 expert 的块重排到同 XCD)—— **0.92692**,b8 116us,灾难性;
  此前只记过 xcd4 死,现在 xcd1(含 group_m=1 变体)也确认死。
- 同理由的反向检查:A/scale 的 load 本来就是 `cache_modifier=0`(cached),**不要**给它们打 NT ——
  32 个 N-block 共享同一个 A tile,这条复用是现役 grid 维序刻意保护的。
- ★ **上一条已被实测坐实(2026-09-12 GLM-5.2 a4w4 EP4 r13)**:给 B-scale 流按「重读块 cached /
  单读块 NT」分类定价(只在非复用块上打 NT,复用块保持 cached),score **1.10968 vs base 1.12915
  = −1.7%**(b64 177.1 vs 174.4、b128 210.2 vs 205.6,只有 b16 反而 −1.8%)⇒ scale 流必须整条
  cached,连"只给单读块打 NT"这个最保守的形态也是负的。scale 只占权重字节 5.9%,却是每 lane 4 B
  的窄 load,NT 在这种宽度上拿不到 streaming 收益、只丢掉行内合并。

## ★★★ CPol(aux)整轴一次扫完:除了部署的 `nt=2`,gfx950 上没有第二个可用值

`buffer_load` 的 `cache_modifier` 直接进 CPol,位是 `sc0=1 / nt=2 / sc1=4`,决策索引里只记过 0/2。
2026-09-12 在 GLM-5.2 a4w4 EP4 stage1+stage2 的 B 权重流上把 8 个值扫完(同 batch、每臂清
`/root/.flydsl/cache`,因为 env 探针不改 kernel 名、共享缓存会把上一臂的二进制喂给下一臂):

| CPol | 含义 | score | 判决 |
|---|---|---|---|
| 2(部署) | nt | 1.12915(base 均值) | — |
| 3 | sc0+nt | 1.12864 | 平 |
| 6 | nt+sc1 | 1.12858 | 平 |
| 7 | sc0+nt+sc1 | 1.12989 | 平(噪声内) |
| **4** | **sc1(device scope,绕 per-XCD L2)** | **1.00268** | ❌ **−11.2%**,b64 195.9 vs 174.4 |

⇒ 两条可用信息:①`nt` 之外的位都不值钱,这条轴**整族封闭**,别再逐位试;②`sc1` 单独打开
= **−11.2%**,说明即便每条 128 B 行只被一个 wave 读一次,**L2 仍在承担约 11% 的量**(请求合并 +
重复 slab 命中),"单读流反正不复用、绕过 L2 更省"这个直觉是错的。判负成本 = 一次 8 臂扫描(80 s)。

### ⛔ 上表的位编号是错的,"整族封闭"的判决作废(2026-09-13 GLM-5.2 sparse-MLA r14 实测)

**skill 说 `sc0=1 / nt=2 / sc1=4`,实测 gfx950 上 `SC1 = CPol::SCC = 16`**(LLVM
`SIDefines.h` 里 `SC1` 就是 `SCC` 这个位)。上表 value **4 是 `DLC`,对跨 XCD 可见性零作用**:

| 判据 | cpol=0 | cpol=4/5(表里的"sc1") | **cpol=16/17(真 SC1)** |
|---|---|---|---|
| 一个 CTA 写、另一 L2 域的 CTA 读 partial(seq60/84) | 坏 1971 / 2940 个元素 | 坏在**完全相同的 1971 个元素**上 | **逐位干净** |
| epilogue 用时 | 1.90us | — | **1.55us(更快 0.3–0.4us)** |

⇒ ①那条 −11.2% 的判决是在**从未真正打开 device-scope 位**的情况下做的,**这条轴解封**:
要 device scope 就写 `0x11`(SC0|SC1),不是 4。②**scope 该按"这条流被复用几次"定,
不按指令族定**:GEMM 的 A/B gather 高复用 ⇒ 留在 L2(打 SC1 −11.2%);sparse-MLA 的
partial 是**写一次、被另一个 CU 读一次** ⇒ 打 `sc0|sc1` 反而**更快**,因为把它留在写者的
L2 里只是给读者买一次 miss。③同族还有一条硬结论:**这个核里永远不要用 agent-scope fence**
(release 5.1us + acquire 1.3us,是整个 epilogue 1.3–2.3us 的 3–4 倍;fence 是整 cache 操作、
每个 CTA 都要发)——**要跨 CU 可见性就逐指令挂 CPol,不要 fence**。

---
来源: 08-deadends.md, agpr_phase5_lds.md, flydsl-kernel-authoring/SKILL.md, mi300-blockwise-gg-tuning/SKILL.md, 10-8wave-scvgpr.md, lds-optimization/SKILL.md, gfx950/kernel-implementation-notes.md, programming-model.md, optimization-directions.md, 09-8wave-ceiling.md, agpr_phase5_ldsr.md, diag_4w_vs_8w.md, project_mxfp4_epilogue_store.md, capture-kernel-trace/SKILL.md, kernel-trace-analysis/SKILL.md, mxfp8-8wave-devloop/SKILL.md, prefetch-data-load/SKILL.md, agpr_rawasm_progress.md

## ★★★ 「co-sweep 判负」要先算**共驻窗口的联合脚印 vs 4 MB**;置换 CTA→tile 是免费的,跟 kernel 内排序不是一回事

本卡上面 §❌「靠把 index 排序制造跨 CTA 的 L2 co-sweep」(prefill,免费上界只有 −1.5%)
的判负理由是**并发窗口不够**:96 CTA/XCD 各流 1.31 MB,行早被 4 MB L2 挤掉。
2026-09-12 在 **GLM-5.2 sparse-MLA decode** 上,同一动机的候选**实测 −33.6%** ——
不矛盾,是窗口条件反过来成立了,而且实现形态完全不同:

- decode 的一个请求 = 6 个 verify token × 8 split = **48 个 CTA**,一个 XCD 有 64 个共驻
  槽位 ⇒ **整个请求同时在飞**;它们的**联合**脚印(live server 实测 topk 复用 4.05-4.29)
  只有 **1.6-1.8 MB < 4 MB** ⇒ 复用物理上存在。
- 实现是**纯置换 `owner → (token, split)`**(输出逐位相同、零指令代价),不是在 kernel 里
  排序 index(每 CTA 数千条 LDS 操作,不可能回本)。
- 反例侧同样成立:记分尺的均匀随机索引下,同一置换是 **−0.1%(平)** ⇒ 这条杠杆的全部
  价值来自**数据的结构**,与调度无关。

⇒ 判据清单(看到 gather 核 memory-bound 时按序问):①**唯一行数**能不能少(本卡主判据);
②共驻窗口里**同时在飞的 CTA 的联合脚印** vs 4 MB —— 小于就去改**派发映射**(免费),
大于就别碰;③别拿"请求数/带宽利用率"当货币。
详见 pitfalls/12 §轮 3 补丁(含 decode 的完整定价谱:1.18 MB −31.4% / 4.1 MB −28.9% /
9.4 MB −18.2% / 56.6 MB −3.4%,以及"同脚印不同结构差 25%"的对照)。
