# LDS 分配/swizzle、L2 swizzle、prefetch 双缓冲与热循环调度

> 类别: 方法论 · 主题标签: LDS, swizzle, bank-conflict, allocator, l2-swizzle, xcd-locality, tile-programming, autotune, prefetch, lds-double-buffer, async-copy, tile-size, scheduling, sched-hints, gfx942-vs-gfx950, hot-loop

## FlyDSL LDS 分配与 swizzle:SmemAllocator.finalize、composed swizzle layout、XOR/padding 消 bank

### LDS 分配(allocator）
- **legacy 路径 SmemAllocator**:`SmemAllocator(None, arch='gfx942', global_sym_name='smem0')` → `allocate_array(fx.T.f16, N)` 分配 typed 数组;kernel 内 `buf = lds_buf(allocator.get_base())` 得 `SmemPtr`,`ptr.store(val,[idx])` / `ptr.load([idx])` 访问。也可 `allocate_array(T.i8, n_bytes)` 分字节再 `buf(allocator.get_base())` 拿 typed ptr。
- **必须 finalize**:在 GPU module body 内(`with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body)`)调 `allocator.finalize()`。**忘则 LDS 符号未解析**(编译期报错/链接失败)。
- **新 kernel 推荐 SharedAllocator**:`SharedAllocator`(`flydsl.expr.gpu`,即 `fx.SharedAllocator`)配 `@fx.struct` 存储布局,优于 legacy `SmemAllocator`/`SmemPtr`。
  - `static=True`(默认):每 leaf 发**静态 LDS global**,由编译器定尺寸,`launch(smem=)` 留空。
  - `static=False`(动态):launch wrapper 从 `allocated_bytes` 推 smem,显式 `smem=` 须 ≥ 该值。
- **对齐硬约束**:`vector.store` 到 LDS 硬编码 **16-byte 对齐**,分配必须满足。

### Composed swizzle layout(多阶段双缓冲）
- swizzled 多阶段 tile:`make_composed_layout(SwizzleType.get(b,m,s), make_ordered_layout((BM,BK,STAGES), order))` 再 `make_view(get_dyn_shared(dtype), layout)`。
  - **STAGES** 表达双缓冲(在 layout 维度里)。
  - **SwizzleType** 在 descriptor 内部消冲突。

### XOR swizzle(消 bank,零 LDS 开销）
- **通式**:`swizzled_col = col ^ (row >> shift)` — 不同行同列命中不同 bank。
- **FlyDSL 经 SmemAllocator 实现**:`swizzled_col = col_idx ^ (row_idx & XOR_MASK)`;`lds_offset = row_idx * PADDED_STRIDE + swizzled_col`。
- **XOR mask 值**:
  - gfx942 = `32/(vec*elem_size/4) - 1`
  - gfx950 = `64/(vec*elem_size/4) - 1`
- **GEMM A tile 专用(16 字节粒度 XOR-with-row)**:`swizzle_xor16(row,col,k_blocks16) = col ^ ((row % k_blocks16)*16)`,其中 `k_blocks16 = tile_k_bytes // a_elem_vec_pack // 16`。零 LDS 开销、约 1 SALU/地址。write/read 必须一致:见 pitfalls/03-lds-l2-datapath.md(swizzle 写读路径必须完全一致,否则静默读错行)。

### 向量化(每 vec 覆盖 4 bank = 16 字节)
| dtype | 推荐 vec |
|---|---|
| fp32 | 4 |
| fp16 / bf16 | 8 |
| fp8 | 16 |

### Padding 消冲突(破 stride 对齐)
- 原理:每行 +1 元素破坏 stride 的 bank 对齐。
- gfx942:stride `HEAD_SIZE=128` 时 bank stride = `128*2/4 = 64`,`64%32=0` 全冲突;**+1 → `129*2/4 = 64.5`** 分数 → 消冲突。
- gfx950 同理:128 仍 `64%64=0` 冲突,**+1 → 64.5** 消。
- 最小 padding = `bank_count/elem_size_bytes`(最坏情形),**通常 1-4 足够**。

### ds_read_tr16 转置读 bank 冲突:width-64 XOR/padding 全无效,唯一解=width-128 + pack-128
attention bwd 的 tr16 转置读(`ds_read_tr16_b64` 喂 GEMM2 A-operand)有顽固 bank 冲突,**通用 XOR/padding 全部无效**,根因与解法(2026-07 gpt_oss meta dkdv 实测):
- **根因数学**:`bank = (row*STRIDE + swz_col)/2 % 32`。bf16 时 **STRIDE=head_dim=64 → row*64/2 = row*32,%32 = 0 → row 项彻底消失** → 一条 tr16 读的 16 行同 col 全撞同一 bank。这就是"XOR/padding address-invariant 修不了"假象的真因——不是 HW 转置固定,是 STRIDE=64(=2×bank数)让行偏移归零。
- **width-64 里 XOR 上限 4-way(~19%)**:col 占 bank 的 bit1-2,要错开 16 行须把 row 的 swizzle 位推到 bit3+,而 XOR mask 值域 <64 只能容 `(row&3)<<4`(4 个不同)→ 4-way。`(row&7)<<3` 带 bit2 与 col 重叠也回落 4-way。**别再在 width-64 里试 XOR/padding/mult-8**。
- ★**唯一解 = 加宽 LDS 行到 128 + `mask=(row&7)<<4`**:值域 0-112 装进 128,`/2` 落 bit3-5,**完全避开 col 的 bit1-2 → 0.00% conflict**(实测,dk/dv 位一致)。mult-16 对 16/8/4-wide 所有读写合法。
- ⚠️ 但**天真加宽废一半 LDS + 2× HBM 读**(DMA 每行读 128 列半数垃圾)→ 反慢 +5%。cache-dedup(垃圾列 `col & (head_dim-1)` 回读真实数据命中 L2)收回到 +3%,仍慢。
- ★★**PACK-128(零浪费终解)**:关键洞察 **`phys_row*128/2 %32 = 0` → bank 与物理行号完全无关**。→ **2 个真实 logical row 塞进一个 128-block,bank 不变(仍 0%),LDS 回到 width-64 大小 + 1× HBM 读**。映射:`_pblk(r)=((r>>3)<<2)|(r&3)`(物理 block),半区由 mask bit6 决定(低行 `r&4=0`→[0,64),高行→[64,128));DMA 每 block 读两个真实行(`logical_row=8*(block>>2)+(block&3)+half*4`,`col=position^mask` 保证 <64)。读端 `_a_idx`/`_read_tr` 按 `_pblk(row)*128` 寻址。
- ★★★**消 conflict ≠ 变快(红鲱鱼判据)**:MFMA-延迟受限的内核(MfmaUtil<40%、VALUBusy 高但非 issue-bound、SQ_WAIT 主要来自 MFMA 依赖链),**把 bank conflict 从 19%→0% 速度纹丝不动**(实测 coop 21.7%→4.4% 同速、pack128 0% 与 baseline 19% schrd 同级)。真正变快来自 **1× HBM + `s_setprio(1)` 包 MFMA**(s_setprio 单独在 baseline 中性,叠 pack-128 上净超 baseline −1.2~1.9%)。**先用 `LDSBankConflict=100*SQ_LDS_BANK_CONFLICT/GUI/CU_NUM` 派生指标确认是否真 bank-bound,再决定要不要碰 swizzle**;latency-bound 的话 swizzle 是纯计数改善、零性能。
- 坑:width-128/pack-128 改的是 **DMA 路径(coop_dma_tile)+ 读端寻址**;`_coop_load`(enable_dma=False 的 VGPR staging 回退)若不同步改会读错。测 fast+DMA 用对格式的脚本(dq/dkdv 默认 fast_exp2 可能不同,喂错 lse 格式 → 全 NaN,见 02-correctness)。

### 藏 LDS 写延迟(增大 write-read 距离)
- 在 `ds_write` 后、`s_waitcnt lgkmcnt(0)` 前插**独立**工作。优先级:
  1. 下一阶段 global load(`buffer_load` 异步 ~300+ cycle)
  2. 地址计算 SALU/VALU(~4-8 cycle)
  3. 独立 MFMA 链(~64 cycle/MFMA)
  4. 标量 load(~20 cycle)
- **禁插**:依赖该 LDS 写结果的操作、更多 LDS 操作(争带宽)、超预算涨寄存器的操作。

### 验证清单(优化后)
- 正确性:fp32 累加须 **bit-for-bit**,fp8/bf16 容差内。
- 重 profile:`ds_read`/`ds_write` 与 ds_write 后 `lgkmcnt(0)` stall 下降、且无新 bank 冲突。
- LDS 容量上限与 gfx950 1280 字节分配粒度:见 pitfalls/03-lds-l2-datapath.md。
- 确认 `waves_per_eu` 占用率没掉。

## L2 swizzle 杠杆:1D M-cluster / 2D band(group_n)/ XCD remap 提 L2 residency

### 本质
- L2 swizzle 候选是纯 **WG→tile 双射,bit-identical**;不改数学,只调 **L2 residency**。增益 ∝ **L2 复用差多少**——复用差的 shape(big-N)增益大,square 无回归。
- **只组合一个 swizzle**:叠两个 locality remap 通常两败俱伤(stacking two locality remaps usually defeats both)。

### 硬件前提(gfx950/gfx942 同)
- 两代都是 **8 个 XCD chiplet,每个自带独立 4 MB L2 slice**(不是一块共享 L2)。
- Dispatcher 把 WG 在 XCD 间 round-robin→连续 blockIdx 落到不同 L2。要靠 remap 把相邻 tile 拉回同一 XCD 才能共暖 L2。

### 三种 swizzle

| 类型 | 做法 | 甜点 | 实测 |
|---|---|---|---|
| **1D M-cluster (group_m)** | 连续 `group_m` 个 M-tile 归同一 M-band,XCD-aware 调度到同 XCD 的 CU→`B[g]` 的 N-stripe 在同 XCD L2 resident | 默认 `nt_group_m=4`, `nt_num_xcd=8`(固定=物理 XCD 数) | 基线 lever |
| **2D band (group_n)** | N 切成宽 `group_n` 的竖带,带内再 `GROUP_M` 1D;把 working set(`GROUP_M·A_slab` + `group_n·B_slab`,各 ~2MB)锁进 L2;A 复用 `group_n`×、B 复用 `GROUP_M`× | `GROUP_M=4`, `group_n = n_blocks/8`(band 数=8=#XCD) | **big-N GM4×GN14 → +12%**(L2 51→57.5%,MFMA 34→40%,det1000=0);big-K 也 +1%;square/ffn_down 无回归。commit `7269aebc` |
| **XCD WG-id remap** | 保持 `chunk_size` 个连续 id 在同 XCD:`chunk_idx*(num_xcds*chunk_size) + xcd*chunk_size + pos`→相邻 tile 共暖 L2 | — | GEMM/attention 有空间局部性时是实打实的 win |

### ★ attention bwd 上的 XCD remap(2026-07-27 meta hd64 实测,单项最大杠杆)
- **形式**:`xcd = block_id % 8`(片上实测确认此式,别猜),让每个 XCD 拿一整块 `(batch, kv_head)`,
  使读同一份 K/V 的 GQA work-group 挨在一起。dq 实测 **L2 hit 86.5→94.8%、miss −62%**;dkdv **+2.81%**、dq **+0.54%**。
- **★★内层最快轴 dq 与 dkdv 必须相反**:**dq 要 kv-head 相邻,dkdv 要 q 位置相邻**(split_idx 最快)。
  同一个 remap 只是内层顺序选错 = **−2.5% 对 +2.4%**。⇒ 迁移这个 lever 时,
  **先问"这个 kernel 的 co-resident WG 之间复用的是哪一份数据"**,让那一维相邻,而不是照抄另一个 kernel 的顺序。
- **门控**:需 `num_kv_heads % num_xcd == 0`,其余 head 数走原解码;双射性**离线穷举验证**(遍历所有部署的 `(B, q_split, tile 数)`)后再上机
  —— 历史上一次 naive nested-division remap 直接 GPU-fault,曾被误记为"方向死"。
- **⚠ TCC hit% 不是判据**:hit 率大涨可能值 0 wall。判这类改动看 **`SQ_WAIT_ANY` / `SQ_VALU_MFMA_COEXEC_CYCLES`**。

### ★ 派发顺序 = list-schedule 顺序(同轮发现,+2.50%)
因果掩码下每个 WG 的工作量 `(q_tile+1)*BLOCK_M/BLOCK_KV` 单调递增,而 dispatcher 按 block_id 顺序发 →
**派发顺序本身就是 list-schedule 顺序**,于是 **LPT(长任务优先=降序 q_tile)** 直接改善尾部平衡。
⚠**别假设 in-order 已最优**:同一份代码里 dkdv 的 in-order 恰好已是 LPT,而 **dq 是反的**。
★可以先用"N slots/XCD 贪心"离线模拟器排序候选再上机(本例模拟预测 +2.30%、实测 +2.9%,模型可信)。

- 2D band 触发条件:大-N shape(N≥2880 / N_BLOCKS_N 够多)才加 `group_n`;`group_n = N_BLOCKS_N//8`(#bands=#XCD)对 big-N 另有 **+8~9%** 口径。big-N 瓶颈是 L2 复用(1D GROUP_M 对每 M-group 重复 stream 整个 B 234MB → L2 51% vs big-K 66%);big-K 的 drain-removal / both-J **❌ 不迁移到 big-N**(短 K 摊不开,both-J 在 big-N 上是噪声)。
- **band det 中性**:纯 tile→CU permutation,满 band 各占 `num_pid_m·GN` 个 pid、余数成最后一个窄 band,恒为 bijection。
- ★★ **1D `GROUP_M` 的最优 band 宽度是收缩维 K 的函数,不能用单一 K 扫出来当全局默认**
  (2026-07-30 grouped mxfp8 NT 实测,见 pitfalls/05 §GROUP_M 按 K 分档)。两项互相拉扯:
  ① **B 流量 ∝ 1/gm**(B 每 band 重 stream 一次;logical B/A 恒 = 1/gm,**与 shape 无关**);
  ② **band 的 A 足迹 = `gm × BLOCK_M × K` 字节**,越过 **4 MB per-XCD slice** 就丢 A 驻留 —— **只有 K 是 shape 项**。
  ⇒ 同一个 gm 在 K=2944 上 band 2.88 MiB(线内,峰在 gm=4)、在 K=5760 上 5.62 MiB(已越线,峰移到 gm=8)。
  实测 min 配置 gm 2/4/8/16 = 1.010 / 1.036 / **1.045** / 1.031,单峰。
  ⚠ **别简化成"A band 必须 ≤ 4 MB"**:按那条规则 K=5760 该选 gm=2,而 gm=2 实测最差(1.010)—— B 流量项压过驻留项。
  ⇒ 做法:autotune 的 cfg_key **必须含 K**,再用 `gm*BLOCK_M*K > 4 MiB` 这个物理阈值给 cand[0] 分档,候选数不变。
- **2D autotune gating**:候选 gated `n_blocks>=32 and M//256>=16`(小 M 的 m-block 太少、banding 不划算,走 1D 防回归);winK 块(K≥28672)也 sweep `group_n ∈ {n_blocks/8, n_blocks/4}`。
- **persistent kernel** 要 remap 的是 **PERSISTENT work-id,不是 `blockIdx.x`**。

### 调参规律(4 轴 autotune)
- L2 swizzle 三参 `(group_m, group_n, num_xcds)`:小-M 要**大 group_n(16/32)**;`num_xcds` **NX8 普遍最优**。
- deep-wl `(16,15)`:只在 **K≥8192** 报(深流水藏高-trip-K g2s 延迟),用 `_WL_MARGIN=1.02` 让步(须超噪声带 2% 才选)。
- split-K:只对 **few-tile 大-K**(一 WG/tile 撑不满 CU)给 2/3/4/6/8/12/16。
- Primus-Turbo 生产用 **timed autotune 替 env**:首调每个 `(M,N,K)` 定时扫候选取 global-min per-shape 缓存。四轴且后三轴 **never-regress**(扫里恒含 baseline + 取全局 min + margin 门槛,只会追平或更快):L2 swizzle / deep-wl(phase-barrier `vmcnt`,`lgkmcnt`)/ 变体轴(COOP scale-load + TACCW wide-store)/ split-K。
- **CUDA-graph capture 内无法定时→回退静态启发式**。

## LDS ping-pong 双缓冲与 async copy:2-stage、同步 vs DMA、跨阶段 load 提进 barrier

### LDS ping-pong 双缓冲 (lds_stage=2)
- A tile 分两块独立 LDS:两个 SmemAllocator,global_sym_name = smem0 / smem1。一块跑 MFMA 时另一块加载下个 K-tile,隐藏 global→LDS 延迟。
- 每次主循环迭代处理 2 个 K-tile(pong + ping)。
- LDS 预算:`lds_tile_bytes = tile_m × tile_k × elem_bytes`。2-stage 需 `2 × lds_tile_bytes`;CShuffle epilogue 另加 `tile_m × tile_n × 2` bytes。
  - 例:64×128 FP8 = 16KB;128×128 FP8 = 32KB。
  - LDS 容量上限:见 pitfalls/03-lds-l2-datapath.md。

### A 矩阵入 LDS 两条路
| 路径 | 机制 | 特点 |
|---|---|---|
| 同步 (默认) | Global→VGPR→LDS:`prefetch_a_tile`(buffer_load_dwordx4)再 `store_a_tile_to_lds`(ds_write) | 走 VGPR |
| 异步 (use_async_copy=True) | Global→LDS 直 DMA:`raw_ptr_buffer_load_lds` | 绕过 VGPR,降寄存器压力,省 arch_vgpr;gfx942/gfx950 均可 |

- async copy 适用条件:`tile_m ≥ 128`(足够 compute 藏 DMA 延迟)。小 tile_m 低寄存器压力用 sync,大 tile_m 用 async。
- granularity:
  - gfx942:sync 16B(dwordx4)/ async 4B(1 dword/DMA)。
  - gfx950:sync 16B / async 16B(4 dwords/DMA)。

### B 矩阵:preshuffle 后直 Global→VGPR
- B 预 shuffle 后直接 Global→VGPR(buffer_load_dwordx4),布局已匹配 MFMA 寄存器排布,无需 VALU shuffle。
- 每 K64 微步 B 需 `2 × num_acc_n` 个 i64(K32×2)。

### 跨阶段 load 提进 barrier-wait 停顿区
- 若某阶段耗在 s_barrier 等待(如 softmax 跨 wave reduce ~96K cycle),把下阶段需要的 global load(如 V-value ~17K cycle)提前发到 barrier 停顿区。
- WHY:barrier 反正要等,期间发 load 基本免费。

### wgrad 两种流水(按 per-group M 分流)
- masked chunked(大-M,per-group `m_total/G > 1536`):outer runtime `scf.for over ceildiv(k_iters, chunk)` × inner `range_constexpr(chunk)` 的 4-buffer 流水。over-run 由 per-group SRD `num_records` clamp 到 0(无需 host cap)。chunk=8:每 chunk 8 个 K-iter 全展开。
- persist(小-M,per-group `m_total/G <= 1536`):`_wgrad_loop_body_pipe`,2-stage prefetch(prologue prefetch K-tile 0,per-iter prefetch K+1 overlap 当前 MFMA)。短 contraction 下 masked 的 chunk over-run 是废功,persist 精确跑完自己 M_g。
  - 实测 M_g=512:856 → 1369 TF(+60%)。

## 热循环调度提示:sched_mfma/dsrd/dswr/vmem 交错、gfx942 sync vs gfx950 async 调度

### rocdl.sched_* 提示语义
`hot_loop_scheduler()` 在 MFMA 计算段与下轮 load 之间插 `rocdl.sched_*` 提示，引导编译器交错指令:
- `sched_barrier(0)` — 全调度屏障(禁止跨越重排)
- `sched_mfma(N)` — 发 N 条 MFMA
- `sched_dsrd(N)` — 发 N 条 ds_read
- `sched_dswr(N)` — 发 N 条 ds_write
- `sched_vmem(N)` — 发 N 条 buffer_load

这些提示控制 barrier 前发射多少条对应指令(在 barrier 语义之上做发射调度)。

### gfx942 同步 copy — 标准调度序列
序言与主循环显式排定，让 ds_write 与末段 MFMA 重叠:
- **序言**: `sched_dsrd(2)` + 两个 `sched_mfma(1)` 预载 a0
- **主循环每迭代**: `sched_vmem(1)` + `sched_mfma(mfma_group)` + `sched_dsrd(1)` + `sched_mfma(mfma_group)`
- **ds_write 放尾部**: `dswr_start = max(sche_iters - num_a_loads - 2, 0)`,让 ds_write 与末段 MFMA 重叠、赶在 barrier 前落地
- **末尾**: `sched_barrier(0)`
- WHY: 同步 copy 下 load→compute 有依赖，手排序列把 vmem/dsrd 塞进 MFMA 缝里、把 dswr 压到最后掩盖延迟。

### gfx950 async copy — 均匀铺散
用 `_build_scheduler()` 把 ds_read/VMEM 均匀铺到全部 MFMA:
- `dsrd_schedule = _build_scheduler(num_ds_load - dsrd_preload, mfma_total)`
- `vmem_schedule = _build_scheduler(num_gmem_loads, mfma_total)`
- 每个 `sched_mfma(1)` 之后发相应数量提示
- WHY: async copy 解耦了 load 与 compute，不需要手排预载序列，均匀铺散让内存指令覆盖整个 MFMA 段。

### barrier / waitcnt 配套(与调度提示相邻)
- `fx.gpu.barrier()` = `__syncthreads`(workgroup barrier)
- **CDNA3 (gfx942)**: `fx.rocdl.s_waitcnt(0)`(单一 waitcnt)
- **CDNA4 (gfx950)**: 仍是**统一 `s_waitcnt vmcnt(0)` + `s_waitcnt lgkmcnt(0)`**，**没有**分离的 load/store/ds 计数。
  - ★**真机勘误(2026-07-28 llvm-mc gfx950 实测)**:早先此处写的「分开 `s_wait_loadcnt/s_wait_storecnt/s_wait_dscnt`」是**错的**——那是 **gfx12(RDNA4)** 的助记符,不是 gfx950(CDNA4)。在 gfx950 上 `s_wait_storecnt/s_wait_loadcnt/s_wait_dscnt/s_wait_kmcnt/s_waitcnt_vscnt` **全部 `instruction not supported`**;只有 `s_waitcnt vmcnt(0)`(编码 `0x70,0x0f,0x8c,0xbf`)与 `s_waitcnt lgkmcnt(0)`(`0x7f,0xc0,0x8c,0xbf`)能汇编。
  - **后果(与 pitfalls/05 一致)**:load 与 store **共用 vmcnt**,在途 store 占 vmcnt → **无法"只等 load 不等 store"**,任何把 store drip 进主循环的方案都被 vmcnt 串行化(见 pitfalls/05 §gfx950 计数器约束、store-overlap 死路)。
  - ⇒ 唯一能与 store/cshuffle 的 **lgkmcnt** 链正交的隐藏媒介是 **MFMA**(既不占 vmcnt 也不占 lgkmcnt),见本卡 §「藏 LDS 写延迟」。

### hot loop 指令比例判据(ISA dump 复盘)
| 指标 | 好 | 可接受 | 差 |
|---|---|---|---|
| MFMA 比例 = MFMA/total | >40% | 30–40% | <30%(非 MFMA 开销过大) |
| 内存指令比例 = (ds_read+buffer_load+ds_write)/total | <40% | — | >50%(内存主导 → 试更大 tile_k 或减 load) |

- tile **64×256×128 FP8** 典型总指令 **~130–150**。
- WHY: MFMA 占比低 = 计算被非 MFMA 稀释;内存占比高 = feed-bound，加大 tile_k 摊薄或减少 load 次数。

---
来源: flydsl-tile-programming/SKILL.md, programming-model.md, gemm-optimization/SKILL.md, lds-optimization/SKILL.md, FlyDSL/CLAUDE.md, flydsl-fp8-gemm-tuning/SKILL.md, 02-nt-fwd-kernel.md, gfx950/kernel-implementation-notes.md, 13-primus-turbo-prod.md, kernel-optimize/knowledge/ops/gemm/optimization-directions.md, prefetch-data-load/SKILL.md, 04-tn-wgrad-kernel.md
