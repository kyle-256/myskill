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
| **XCD WG-id remap** | 保持 `chunk_size` 个连续 id 在同 XCD:`chunk_idx*(num_xcds*chunk_size) + xcd*chunk_size + pos`→相邻 tile 共暖 L2 | — | **dense** GEMM/attention 有均匀空间局部性时是实打实的 win;**grouped/MoE 核在 skew 下相反(见下 ⚠️)** |

⚠️ **`num_xcd>1` 的连续块 remap 在 grouped/MoE 核上是 skew-fragile 的**(2026-07-30 NT 核实测,与 wgrad 同源见 methodology/08):连续 tile 整段绑到同一 XCD ⇒ 把 hot expert 压到一个 XCD。`dgrad/fwd *_down heavy` 在 xcd=8 下掉 **2.8~3.0%**(balanced 中性),全局强制 xcd=8 净 −1.6%(miss −20.4% 但 cyc +5.3%)。**没 ship 的唯一原因是本核顶在 1400 W 功耗墙 ⇒ tile 顺序不改 energy、wall 惰性**;但 NT 候选表里遗留的 `(256,4,8,0)` 对 bench 的 skew 配置是**潜在毒**(竞速当前 3/3 选 base 未触发)。⇒ **`nt_num_xcd=8` 作基线只对 dense 局部性成立;grouped 核最优是 group-major 序 + `xcd=1`(HW round-robin 均分每组 tile),不是照抄 dense 的连续块聚簇**。
★ **反例 / 修正(2026-08-03,mxfp4 grouped 实测)**:「grouped 核最优是 `xcd=1`」**不普适**。同一批
gpt-oss skew 配置上逐配置实测 `xcd=8` 明显优于 `xcd=1`(R2:两个 min 配置 −7.0%/−7.8%;本轮
`fwd/dgrad down balanced` `xcd=1` 反而 +2.3~3.2%),全 18 配置 `xcd=2` gm 1.4366 vs `xcd=8` 1.4486。
**skew 与局部性是一对 trade-off,方向随核而变**:`xcd` 越小越均衡(heavy 组 −4~5%)、越大局部性越好
(`fwd gate_up` 从 1.373 崩到 1.223)。⇒ 该轴必须**当场逐配置扫**,别照搬任何一侧的结论;而且
**快筛配置集必须同时覆盖 (窄N,短K) 与 (宽N,短K)** —— 本轮只筛了 `dgrad gate_up`(长 K)就漏掉了
`fwd gate_up` 的 12% 崩塌。

★★ **第三个数据点 + 可预判的判据(2026-08-17,gpt-oss-20b down-projection padN dgrad NN 实测)**:
这条轴之所以「随核而变」,是因为真正决定方向的是 **`B[g]` 装不装得进一个 XCD 的 4 MB L2 slice**:
- **装得进** ⇒ B[g] 可以整块常驻,tile 顺序只影响均衡 ⇒ 越均衡越好,`xcd=1` / 小 `xcd` 赢(上面 2026-07-30 那组)。
- **装不进**(本例 `B[g]` = 2944×2944 fp8 = **8.67 MB** > 4 MB)⇒ B 必然是**流**,唯一能做的是让
  「共用同一个 B column-block 的 WG 同时在同一个 XCD 上跑」⇒ **`xcd` 必须等于物理 XCD 数 8**,
  且 band 要窄到那批 WG 真的同驻。部署点(4096 tokens/expert)实测、**逐位相同**:
  `(xcd=8,gm=4)` 对 `(xcd=4,gm=8)` **balanced +2.95% / heavy +2.49% / 几何 skew +0.03%**;
  而这张卡为 grouped 核推荐的 **group-major + `xcd=1` 在这里是 −5.72%**。
  band 宽度单峰:`gm` 3/6 = −1.3/−1.7%,`xcd=16`(每 XCD 两条 band)= −6.3%。
- **`heavy` 不再是反方向**:B 是流的时候,hot expert 压在一个 XCD 上的代价被「B 命中」赚回来,
  所以 heavy 也是正的 ⇒ 「xcd 大 = skew 脆」只在 B[g] 能常驻时成立。
- **PMC 判别器(比命中率好用)**:赢的时候 `MemUnitStalled` 12.4%→10.7%(−13.7%),而
  `TCC_REQ` / `TCP_TCC_READ_REQ` / `TCC_HIT`(~76%)**三项都不动** ⇒ 收益来自 **TCP→TCC 请求路径排队**,
  不是少搬字节。**别用 L2 hit-rate 判这条轴有没有生效**——hit-rate 可以一动不动而 wall 快 3%。
  (本例离带宽墙很远:2.6 TB/s vs ~8 TB/s HBM 峰值。)
- ⚠️ 这个 pick 竞速**选不到**,因为竞速打分的 M 不是部署 M ⇒ 落地形态见 pitfalls/06 §静态 lead。

★★★ **判据升级:决定方向的不是"B[g] 装不装得进",而是"这个核每个 tile 流的是私有 slab 还是共享切片"**
(2026-08-17 续测,同一台机、同一个 campaign 的**三个核**同时定档,全部部署 M、多回合 palindrome、逐位相同):

| 核 | 每个 tile 流什么 | 最优 band | 把另一个核的赢家搬过来 |
|---|---|---|---|
| fwd NT / dgrad NN | `B[g]` = per-expert **私有** slab(8.67 MB > 4 MB slice) | `xcd=8, gm=4` | `xcd=1,gm=2` **−2.26%**;`xcd=1,gm=4` −5.72% |
| wgrad TN(变-K,收缩 M) | 两个操作数都是 `[M,*]` 的 token-major 切片,**被该 group 的全部 144 个输出 tile 共享** | **`xcd=1, gm=2`**(group-major 窄带) | XCD-仿射矩形/宽带落后 **0.8~1.0%** |

⇒ 前一条「B[g] 装不进 ⇒ xcd=8」只覆盖了**私有 slab** 那一类。**共享切片类(wgrad/变-K)是反的**:整机本来
就在读同一批行,XCD 分区只会把这份共享**切碎**,所以 `xcd=1`(HW 按 `bid%8` 逐 tile 轮转)+ 最窄的
group-major 带才对。wgrad 实测 balanced **+0.78~0.98% / heavy +0.99% / 几何 +0.76%**,`gm=1` 掉到 −5.06%
(带太窄 ⇒ 同驻的 WG 不再共用一段行)。
⇒ **规矩:同一个 campaign 里也别把一个核的 band 赢家抄到另一个核上**;先按「私有 slab / 共享切片」
预测方向,再扫 3~5 个点验证单峰。两类的落地都用 pitfalls/06 §静态 lead(竞速只在 balanced 上打分,
分不出窄带和"同样在 balanced 上赢但 skew 崩"的臂:`(4,6)` 是 heavy −2.0%/几何 −34.6%,`(8,3)` 是 −3.7%/−39.3%)。

### ★★ 过发射 grid 上做 XCD remap:`total_pids` 必须是**活 tile 数**,不是 grid 上界(2026-08-03 mxfp4 grouped 实测 +1.4% gm)

grouped/MoE 核的 grid 是**上界**(每组 round-up 的最坏情况,`(ceildiv(total_M,BM)+G)*n_blocks`),
真实 tile 数 `total_tiles` 只有片上 O(G) scan 之后才知道。此时 remap 的 `total_pids` 传谁,是
**一个 6% 量级的负载均衡 bug**:

* ❌ `pid = xcd_remap_pid(block_idx, grid_upper, 8)` 然后 `if pid >= total_tiles: s_endpgm`
  —— remap 是 `[0, grid_upper)` 上的双射,死 pid 区间 `[total_tiles, grid_upper)` 整段落在
  **最后一个 XCD 的尾部**(它拥有 remap 值域的最高段)。gpt-oss 形状实测:544 个 m-block 里 32 个是
  死的(5.9%)⇒ XCD0-6 各干 816 tile,**XCD7 只干 432**。XCD 分区是**静态**的(HW: XCD = bid % 8),
  别的 XCD 帮不了它 ⇒ wall 由 816/768 = **+6.25%** 决定。
* ✅ 退出测试打在**原始 bid** 上,remap 只在活 tile 上做:
  `if block_idx >= total_tiles: s_endpgm` → `pid = xcd_remap_pid(block_idx, total_tiles, 8)`。
  这样 remap 是 `[0, total_tiles)` 上的干净双射,过发射的 WG 按 `bid % 8` 被 HW **均摊到 8 个 XCD**。
  实测 gm 1.4292 → 1.4486(两次 1.4467/1.4506),SNR/det 不变,per-tile L2 足迹与遍历序完全没动。
* **别把 remap 的 total_pids 设成 grid 上界再靠 remap 后的 pid 做退出** —— 那两件事顺序反了。
  `mxfp8_grouped_kernel` / `gemm_fp8_grouped_kernel` 本来就是对的(`xcd_remap_pid(t, total_tiles, …)`),
  照抄它们。⚠️ 反过来也别把 remap 的 total_pids 改成 total_tiles 却仍用 remap 后的 pid 做退出:
  `bid >= total_tiles` 时映射会**撞进活区间**(pid 碰撞 ⇒ 有 tile 算两遍、有 tile 没人算)。
* **诊断签名**(不用改代码就能判):同形状不同 skew 下 `SQ_INSTS_MFMA`、`TCC_REQ/HIT` 逐位相同,
  而 `GRBM_GUI_ACTIVE` 差 13%,同时 `SQ_BUSY_CU_CYCLES` 只差 0.7% ⇒ **CU 空转 = 静态分区的负载不均**,
  不是访存、不是 latency(`TCC_EA0_RDREQ_LEVEL/RDREQ` 只差 1.9%)。这三对计数器是判 XCD 分区
  不均的最短路径。

### ★★ grouped 核的 XCD 分区要 **band-cyclic**,不是「连续块」也不是「逐 tile 轮转」(2026-08-03 mxfp4 grouped 实测 +1.6% gm / +4.8% min)

`xcd_remap_pid` 给每个 XCD **一整段连续 pid**;`xcd=1`(identity)让 HW 按 `bid % 8` **逐 tile** 轮转。
这两端都不是最优 —— 它们是同一条轴(**XCD 轮转粒度**)的两个极端:

```python
# band-cyclic: XCD x 拿第 x, x+8, x+16, ... 个「run」,每个 run = band 个连续 tile
span = num_xcd * band
xcd, loc = pid % num_xcd, pid // num_xcd
rnd = loc // band
mapped = (rnd * num_xcd + xcd) * band + (loc - rnd * band)
pid = select(pid < (total_pids // span) * span, mapped, pid)   # 尾部退 identity 保双射
```
双射证明:`bid ↔ (xcd, rnd, q)` 与 `mapped ↔ (rnd*8+xcd, q)` 都是唯一分解;尾部区间 `[full, total)` 不与
`[0, full)` 相交。**退出判定仍打原始 `bid`**(见上一条卡)。

**为什么两端都不对**:grouped 核的**每 tile 成本不均匀**——只有 1 个 M-block 的小 expert 拿不到任何 B 复用。
连续分区把整条小-expert 尾巴压在一个 XCD 上(静态 `bid % 8` 事后无法再平衡,那个 XCD 决定 makespan);
逐 tile 轮转则把一个 band 内的 A/B 复用打散到 8 个 XCD。band-cyclic 两头都保:run 内复用留在本 XCD 的
L2 slice,而每个 XCD 又在整个 token range 上取样。

**run 长度实测(gpt-oss G=32 / M=131072 / BM=256 ⇒ 每组 16 个 M-block;span 单位 = M-block)**:

| span | balanced | moderate | heavy |
|---|---|---|---|
| 2(=一个 gm=2 band) | **+9~11%** | — | −6.3% |
| 8 | +1.4~1.7% | — | −5.8~6.9% |
| **16(=一组的 M-block 数)** | **+0.0~0.8%** | **−1.8~2.7%** | **−4.5~6.1%** |
| 24 | +0.2~2.9% | — | −4.5~5.2% |
| 32 | +0.1~0.4% | — | −3.1~4.4% |

⇒ ① **run 必须装得下同一组的好几个 band**(span=2 只有 1 个 band ⇒ balanced 崩 10%);
② **run 越长均衡越差**(span=32 只有 2 run/XCD,heavy 只剩 −3~4%);
③ **甜点 = run 与组边界对齐**(span=16 时 balanced 每组恰好一个 run ⇒ 零回归;span=14 在 balanced 上
`gate_up` +1.2~1.6%,正是 run 跨组把两个 expert 的 B 同时拉进一个 XCD)。
⇒ 落地形态:**span = 均匀分布下每组的 M-block 数**,并按「run/XCD ≥ 4」核对。
实测把 heavy-vs-balanced 的 skew 罚金从 6.7~7.4% 压到 0~1.8%(fwd `gate_up` 上 heavy 反超 balanced)。
`SQ_INSTS_MFMA` / L2 足迹逐位不变 ⇒ 逐字节确定性天然保住(两次 bench det=True)。

### ⚠️ 同一个核的两条路径可能要**相反**的 XCD 策略:先问 skew 载体是 tile 数还是 contraction 长度

mxfp4 grouped 的 NT(fwd/dgrad)吃 band-cyclic,**wgrad 却必须留在 `xcd=1`**。历史上把 wgrad 一起换成
`(gm=2,xcd=8,gn=0)` 实测崩到 **0.382~0.549x**,当时没归因;机制其实很干脆:
* wgrad 的 tile 数**每组恒等**(`TILES_PER_GROUP` 编译期常量),但**每 tile 的 contraction = 该组的 M_g**,
  heavy 分布下 M_g 相差 **80×**。连续分区 ⇒ XCD x 拿 group `4x..4x+3` ⇒ XCD0 拿到最大的四组
  = 全部工作量的 **86%** ⇒ wall ≈ `8 × 0.86 = 6.9×`(实测 4.3×,同一量级)。
* 而 band-cyclic 也救不了它:group g 的 TPG 个 tile 被按 run 发牌,XCD 之间的差就是 **1 个 run**
  = `R / (TPG/8)` 的组内份额;`TPG=276` 时 R=8 已经是 23% 的组内不均,乘上 group 0 的 62% 成本份额
  = 14% 总不均。**只有 R=1(=identity/HW 轮转)才恰好均衡** ⇒ wgrad 的 `xcd=1` 是最优,它的 L2 复用
  只能从 `group_n` band 里拿。
⇒ **判据**:先量「skew 落在 tile 数上还是落在每 tile 成本上」。落在 tile 数 ⇒ band-cyclic 有解;
落在 per-tile contraction 上且组内 tile 数不够多 ⇒ 任何粗于 1 tile 的分区都会按比例引入不均。

#### ★ 精化:「wgrad xcd=1」可**按-band 放宽**,门控在 device 上判(2026-08-03 tw grouped wgrad campaign,+0.18%)
- 上文与 pitfalls/05 §"XCD gather 在组内做也是死路"(**无条件**给所有 band 上 xcd=8 → −0.6%)说的是
  **一刀切**;它们**没被推翻**。可放宽的是**粒度**:`_wgrad_band_is_xcd_aff` 在 device 上逐 band 判
  「这条 band 的分布够均衡吗」,**只对判定均衡的 band** 做 XCD 聚簇,hot/skew band 仍留 HW round-robin。
- ⇒ 净 **+0.18%(det-safe)** —— 小,但方向明确:blanket-xcd=8 负、blanket-xcd=1 是安全默认、
  **runtime 逐-band 门控**能在不碰 hot band 的前提下把均衡 band 的 L2 复用捡回来。
- **判据**:XCD 亲和不是核级 on/off,是 **band 级**决策;能不能开取决于**该 band 运行时的分布**,
  必须片上判(别引入 host D2H)。与 [[project_wgrad_reach_fwd_campaign]] 的单-window split-K 同源
  (都靠 `group_offs` 的 wave-uniform SALU policy,见 methodology/08 §运行时自适应单-window split-K)。

#### ★★ 第三条路:粗分区的不均可以靠**轮转"类→超块内位置"**修掉,不必退回 identity(2026-08-11 syncv3 wgrad var-K TN 实测 +5.3%)
上面两条把选择写成二选一(留 `xcd=1` / 逐 band 放宽),漏了一条**保留亲和又不吃不均**的形态:
- 场景:`TILES_PER_GROUP < NCU` 的 wgrad(down 2880x2880,TPG=144 < 256)会把 dispatch id 切成
  **gp 组一块的超块**(`gp=2, k=nxcd/gp=4`:类的低 log2 k 位选组内 run、高位选超块里的**哪一组**)。
  这样每个 XCD 的类又回到"一组的 tile",L2 slab 复用保住了。
- 病灶:超块里的位置**跨超块不动**——类 c 永远服务每个超块的同一侧。叠上降序-K(LPT)组序,
  那一侧恒是**重的**那组 ⇒ 静态 `bid % 8` 分区自己给自己造 8~11% 的 per-XCD 不均(离线模型),
  实测 down `strong1.3` CU 时间占用 0.911。
- 解:**每过一个超块把位置也推进一格**(等价于把类的步长从 1 改成 k+1),同一批 tile、同一批 band、
  同样的 run,只是"谁服务重侧"轮着来。落地只有 3~4 条 SALU(`sb=slot>>log2 gp;
  (sb<<log2 gp) | ((slot+sb) & (gp-1))`),ISA 实测 **+1 SGPR、VGPR 不变、0 spill**。
- **步长必须是奇数**:ragged tile(`2880=11.25x256` ⇒ 边界块跑半体)本身更便宜,偶步长会把它们冻在
  固定几个类上,变成下一个静态不均(离线模型:偶步长吐回三分之二收益)。
- 收益/代价(交错 A/B,3 base × 4 cand):`down.strong1.3` 比值 1.1682 → 1.2303(**+5.3%**,两臂分布不相交),
  `unbal1.07`/onehot/balanced 全部不动(前者组间只差 7%,后者被 hot 门排除),balanced 一侧
  **反而 +0.4pp**(bal_dn 0.990 → 0.994)。门控复用降序-K 的 reorder 谓词 ⇒ 均匀负载走原图。
- ⇒ **判据**:粗于 1 tile 的 XCD 分区在 per-tile-contraction skew 下的不均,先问"不均是**分配份额**错了
  还是**同一份额被钉在重侧**";后者只要轮转位置就够,别为它放弃亲和的 L2 收益。
  离线"N slots/XCD 贪心"模拟器在这条上再次可信(预测 +5.0%,实测 +5.3%)。

### ★ attention bwd 上的 XCD remap(2026-07-27 meta hd64 实测,单项最大杠杆)
- **形式**:`xcd = block_id % 8`(片上实测确认此式,别猜),让每个 XCD 拿一整块 `(batch, kv_head)`,
  使读同一份 K/V 的 GQA work-group 挨在一起。dq 实测 **L2 hit 86.5→94.8%、miss −62%**;dkdv **+2.81%**、dq **+0.54%**。
- **★★内层最快轴 dq 与 dkdv 必须相反**:**dq 要 kv-head 相邻,dkdv 要 q 位置相邻**(split_idx 最快)。
  同一个 remap 只是内层顺序选错 = **−2.5% 对 +2.4%**。⇒ 迁移这个 lever 时,
  **先问"这个 kernel 的 co-resident WG 之间复用的是哪一份数据"**,让那一维相邻,而不是照抄另一个 kernel 的顺序。
- **门控**:双射性**离线穷举验证**后再上机 —— 历史上一次 naive nested-division remap 直接 GPU-fault,曾被误记为"方向死"。
  ★**穷举口径是「实际发射的 grid 宽度」,不是部署形状**:同一份二进制常被按 chunk 发出更窄的 grid(nb=1),
  按 `(B, Hkv)` 验证过的取模形式在那条 launch 上就是非双射 ⇒ fault。⇒ 优先用**余数摊派**形式
  `x*(G/8) + min(x, G%8) + bid/8`(`x = bid % 8`),它对**任意** G 恒为置换,只多两条 SALU,于是
  `num_kv_heads % num_xcd == 0` 这个门可以整个去掉 —— 2026-08-15 gfx950 实测 Hkv=6 的窗口 dkdv body **−9.1%**,
  说明"其余 head 数走原解码"是一笔 4% 量级的欠账,不是中性 fallback。
- **★★共驻例外:remap 的符号取决于这条流是否独占 fabric。** 同一套连续切分用在**独占发射**的 reduce 上是
  −5.4% / −1.8%,用在与 body **共驻**的同一个 reduce 上是 **+12%**(2026-08-15 gfx950 实测):把读流固定到一个
  XCD 的 L2 片,正好对准邻居 body work-group 正在写穿的那片,两边争同一批 set。⇒ 判"要不要给某个 kernel 做
  XCD 连续切分"必须先看它是独占 dispatch 还是与另一个 kernel 共驻;共驻时用"同片内错相位"而不是"独占一段"。
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
  ⚠ **这条按 K 分档只在 NT(mxfp8 grouped NT)上成立,别外推**:2026-07-31 在 **tw grouped NN dgrad**
  上照此把 cand[0] 的 gm 在 K<4096 时从 8 收窄到 4(xcd 不变=4),`dgrad down`(K=2944)
  **掉 1.2pp(0.990 → 0.978)**,K≥4096 的 `gate_up` 不受影响 ⇒ 已回滚。
  根因方向:NN 的 B[K,N] 是**沿收缩维 strided**,band 里复用的是 B 的 N-stripe 而不是 A 的 slab,
  A 足迹越 4MB 线的那套算术不适用。**gm 的最优值必须按 kernel(NT/NN)分别实测,不能跨布局搬。**
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
