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

- 2D band 触发条件:大-N shape(N≥2880 / N_BLOCKS_N 够多)才加 `group_n`;`group_n = N_BLOCKS_N//8`(#bands=#XCD)对 big-N 另有 **+8~9%** 口径。big-N 瓶颈是 L2 复用(1D GROUP_M 对每 M-group 重复 stream 整个 B 234MB → L2 51% vs big-K 66%);big-K 的 drain-removal / both-J **❌ 不迁移到 big-N**(短 K 摊不开,both-J 在 big-N 上是噪声)。
- **band det 中性**:纯 tile→CU permutation,满 band 各占 `num_pid_m·GN` 个 pid、余数成最后一个窄 band,恒为 bijection。
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
- **CDNA4 (gfx950)**: 分开 `s_wait_loadcnt(0)` / `s_wait_storecnt(0)` / `s_wait_dscnt(0)`

### hot loop 指令比例判据(ISA dump 复盘)
| 指标 | 好 | 可接受 | 差 |
|---|---|---|---|
| MFMA 比例 = MFMA/total | >40% | 30–40% | <30%(非 MFMA 开销过大) |
| 内存指令比例 = (ds_read+buffer_load+ds_write)/total | <40% | — | >50%(内存主导 → 试更大 tile_k 或减 load) |

- tile **64×256×128 FP8** 典型总指令 **~130–150**。
- WHY: MFMA 占比低 = 计算被非 MFMA 稀释;内存占比高 = feed-bound，加大 tile_k 摊薄或减少 load 次数。

---
来源: flydsl-tile-programming/SKILL.md, programming-model.md, gemm-optimization/SKILL.md, lds-optimization/SKILL.md, FlyDSL/CLAUDE.md, flydsl-fp8-gemm-tuning/SKILL.md, 02-nt-fwd-kernel.md, gfx950/kernel-implementation-notes.md, 13-primus-turbo-prod.md, kernel-optimize/knowledge/ops/gemm/optimization-directions.md, prefetch-data-load/SKILL.md, 04-tn-wgrad-kernel.md
