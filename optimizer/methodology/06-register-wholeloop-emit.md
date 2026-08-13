# 寄存器压力/whole-loop 汇编/emit 旋钮:AGPR 累加、LDS-feed bound 与 mxfp4 生产杠杆

> 类别: 方法论 · 主题标签: register-pressure, agpr-accum, raw-asm, whole-loop-asm, lds-feed-bound, bank-conflict, 算术强度, mxfp4, emit-knob, race-correctness, VGPR-scale

## 寄存器压力:asm-inplace MFMA 把 accum 挪进 AGPR 消 spill、raw-asm 绕 LLVM 拒 AGPR
> ⚠ 标题里的「LLVM 拒 AGPR」是**旧工具链**的情况。ROCm 7.2 上 scaled MFMA 的 tied-operand
> `"=a,v,v,0,v,v"` 已可用,见下面 §「先试 tied-operand」——**动手前先花 60 秒编译一个探针**。

### 核心手法:把累加器从 VGPR 挪进 AGPR
- **asm-inplace MFMA(`asm_mma=2` mode2)**:D 别名 C in AGPR,`agpr_alloc=128`,把 accum 挪进 AGPR,消掉 accvgpr 拷贝 + spill(**18→0**),big-K **+2.3%** 且 **det0**。  (flydsl-fp8-gemm-tuning)
- **whole-loop bare-asm**:整个 K-loop 写成一个内联汇编 hw-loop,消 per-mfma asm 边界 + per-iter 循环开销;是 mxfp4 4-wave 从 **3583(intrinsic)→ 5401** 的关键 lever(**+16%**)。accs 用 `['=a']` 且 MFMA dst=acc_in(`$q,$a,$b,$q`)实现 AGPR 原地累加,天然消除 accvgpr-shuffle。  (11-upstream-agpr-pin-moot)

### gfx950 scaled/fp4 MFMA 的 AGPR 累加器:先试 tied-operand `"=a,v,v,0,v,v"`,不行再 raw-asm

★★★ **2026-08 勘误(这条过时记录直接挡住了一整场 campaign 的四轮)**:**ROCm 7.2 上
`_asm_mma_scale_do` 用 `constraints="=a,v,v,0,v,v"` 的 tied-operand 写法直接编译通过** ——
ISA 里是 `v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[..], v[..], a[0:3]`、`.agpr_count 256`、
spill 0、SNR 逐位不变。在 mx8tw 的 NT 核上它把 **arch VGPR 246 → 184、空闲 arch 寄存器 10 → 72**,
正好是此前四轮跨 tile/persistent 改动一直差的那 23–25 个。
⇒ **动手前先花 60 秒编译一个探针,别按下面的旧记录直接放弃 AGPR。**
下面的 raw-asm 路线是更早工具链才需要的兜底:

- **(旧工具链)问题**:gfx950 scaled MFMA(`mfma_scale`)LLVM 不肯给 AGPR 分累加器 —— `=a` 约束被拒,AccumVGPR 恒 0。  (agpr_rawasm_progress)
- **(旧工具链)绕过**:inline `volatile` asm 文本写死物理 `a[N:N+3]` 做 dst/src,**不用 `=a` 约束**,累加器只放进 clobber `~{aN}`。
  - init:`v_accvgpr_write_b32 aN, 0`
  - readout:`v_accvgpr_read_b32 $k, aN`
- **收益**:32 个 v4f32 累加器从 **256 VGPR 移进 128 AGPR**(`num_vgpr` 256→128);det0 证明跨迭代物理驻留稳定。  (agpr_rawasm_progress)
- **fp4 raw MFMA(`cbsz:4`/`blgp:4`)操作数宽度坑**:要求 A/B operand 是 **4-dword**(`i32x4` / `v[N:N+3]`);官方 intrinsic 用 `i32x8`(高 16B 补零)。用 raw-asm 时 `S2RLoaderFp4` 必须 `pad=False` 返回 `i32x4`,否则汇编器报 `wrong register tuple size for cbsz value 4/blgp value 4`。scale 仍是 `i32`(1 dword)不变。  (agpr_rawasm_progress)

### AGPR 腾出的 VGPR 余量 → 手工预取重叠
- AGPR 累加腾出的 VGPR 余量(如 **128→110**)可用于把 operand `ds_read` 主动提早进 MFMA 窗口做重叠。
- 编译器因 `volatile` + barrier 强序做不到;手工把一个 operand 的 `ds_read` 下移一个 barrier(在可见性安全范围内)能恢复并反转残差。
- **gfx950 barrier 语义**:`wait_barrier(cnt)` = `s_waitcnt vmcnt(cnt)` + `s_barrier`;operand(`ds_read`)的 LDS 可见性依赖 g2s(`buffer_load_lds`,VMEM)vmcnt 落地 + `s_barrier` 跨波同步。读 8 波协作填充的 LDS 必须在某 barrier 之后。  (agpr_phase5_mono)

### wgrad 4-wave whole-loop 结构(AGPR 累加实战)
- 详见本文件「whole-loop 4-wave 结构与 LDS-feed bound」章节的「4-wave whole-loop 结构(wgrad)」小节。

### GEMM VGPR 估算(算 arch_vgpr,判是否 spill)
| 组成 | 公式 |
|---|---|
| 累加器 | `m_repeat × num_acc_n × 4`(→ accum_vgpr) |
| B tile | `k_unroll × 2 × num_acc_n × 2` |
| A 预取 | `2 × 2` |
| A tile regs | `num_a_loads × 4`(同步 copy) |
| 地址 | ~10-20 |

- 例:64×256×128 FP8 ≈ **148 arch_vgpr**。  (gemm-optimization)

## whole-loop 4-wave 结构与 LDS-feed bound:byte-exact LDS 账、bank conflict 与算术强度

### 4-wave whole-loop 结构(wgrad)

> ★★ **这个核就是 gpt-oss mx8tw campaign 的标尺**(`kernel_grouped_tn_wgrad_4wave_0`,3000–3188 TF/s,
> 本机最快的 grouped GEMM)。**2026-08 逐条 ISA 复核为真**:grid 258048 = 1008 WG × 4 wave、
> LDS 163840、256 条 `v_accvgpr_read_b32`、`ds_read_b64_tr_b8`、`buffer_store_short`、`s_setprio` 0 条。
> **教训:要追某个核之前先读这张卡** —— mx8tw 那场十五轮都在猜标尺长什么样,而它一直写在这里。

- occ=1(512 VGPR = 256 操作数 + 256 AGPR 累加器,accum_offset=256),256×256 tile,2×2-wave,两操作数都 transpose-read,whole-loop 裸 asm 主循环,AGPR 累加,CShuffle store。
- 根因瓶颈 = LDS 转置读 feed 带宽(TN 固有税),**非** 占用率/延迟/bank。即使 racing 也只有 fp8 峰值约 44%。
  - ★ 2026-08 给了这条一个新的量化形式:**NT 布局不付这笔税,所以 NT 的稳态每 K-iter 比这个核快
    11–14%**(1.164 vs 1.31–1.35 µs,稳态 MFMA busy 87% vs 81%)。⇒ 追它的时候别去改稳态,
    差距在别处(tile 长度与同步密度,见 methodology/12 §标尺)。
- 主循环 = 两个 ping-pong 相位,相位间一道 `s_waitcnt;s_barrier`。一个 wave 的 transpose-read 要 gather 所有 wave 写的 chunk(`W*chunk_stride`)→ barrier+drain 是跨 wave LDS 一致性硬需求,per-wave 局部 drain 不安全。

### byte-exact LDS 账(2 池 3buf 塞 160KB)

- `_CS=1024`(省 5120B bank-pad) + scalar store(省 8704B C_lds,`PT_WL_2BPOOL` 自动触发)→ 10 buffer × 16384 = 163840 恰好塞满 160KB。
- `_CS` 被双向锁死 = 1024:
  - buffer 需放 16384B → `_CS >= 1024`
  - 10 buffer <= 163840 → `_CS <= 1024`

### LDS bank conflict(_CS=1024 的税)

| 配置 | LDSBankConflict (rocprofv3) | MfmaUtil |
|---|---|---|
| 1 池 @1056 | 0% | — |
| 2 池 @1024 | 14% | 54.3(>50 净胜) |

- `_CS=1024` = bank 周期整数倍 → transpose read 必冲突,但 MfmaUtil 净胜,2 池仍取胜。
- 消 14% 冲突唯一路 = 操作数改 `a_plain`(预转置走 `ds_read_b128`,无 transpose 固定域冲突),但需 upstream 量化产转置副本、跨文件大改。
- `a_plain` 在 kernel 内**零吞吐收益**(gfx950 transpose-read 零 per-op 惩罚),唯一价值是免 pad + 免冲突的 enabler。

#### ⚠️ 上表 14% 已过期:`wswz` wave bank-swizzle 打开后实测 **0%**(2026-07-31)
`kernel_grouped_tn_wgrad_4wave` 现在 `G2SLoader`/`S2RLoaderTr` 都带 `wswz=True`
(`compute_global_swizzle_nn(..., wswz=True)`),rocprofv3 `--pmc LDSBankConflict
MfmaUtil MeanOccupancyPerCU` 实测 tw = **LDSBankConflict 0.0 / MeanOccupancyPerCU 3.6(4 波满)**,
mx 对照 **0.0 / 7.1(8 波满)**。⇒ **别再把 `_CS=1024` 的 bank 冲突当 wgrad 的待优化项**,
也别为消它去做 `a_plain`/padding —— 冲突已被 swizzle 清零,`a_plain` 只剩"免 pad"这一个理由。
- ★ **MfmaUtil 跨 wave 数不可直接比**:同一轮实测 tw(4 波,1 wave/SIMD)32.2% vs
  mx(8 波,2 waves/SIMD)63.8%,而两者 wall 只差 4%、`SQ_INSTS_MFMA` 相等。
  63.8/2 = 31.9 ≈ 32.2 ⇒ 该计数器按 **wave** 累加,occ=2 的 kernel 读数天然 ×2。
  比较不同 wave 数的 kernel 时先除以 waves/SIMD,否则会误判成"mx 的 MFMA 利用率是 tw 的两倍"。

#### ❌ 别再试:直接把 `ds_read_b64_tr_b8` 换成 `ds_read_b128` 当作"半指令数"探针
想验证"DS 指令条数是不是瓶颈"时,把 emit 里的 4×`ds_read_b64_tr_b8` 改成 2×`ds_read_b128`
**读同一组地址** —— 实测 wgrad **慢 80%**(geomean(mx/tw) 0.971 → 0.566,六配置全崩)。
这不是"指令数无关"的证据,是**探针本身失效**:transpose-read 的 per-lane 基址是按转置单元
布的,b128 在同一批基址上按 lane 顺序取 16B 会撞满 bank。`a_plain` 之所以合法,是因为它
同时换了 LDS 布局(配 `a_row_stride` 的 A0/A1 全局偏移),**指令与寻址必须一起换**。

#### ★★ tw fp8 wgrad whole-loop 的 emit 调度是**刀锋级局部最优**(2026-07-31 实测)

`kernel_grouped_tn_wgrad_4wave` 的相位内指令排布(mfma 对角块 + 用后立即 refill +
g2s 均匀撒进非 refill 槽)每一个方向的扰动都是**净负**,四次独立实测:

| 扰动 | wgrad 组 geomean | 机制 |
|---|---|---|
| 基线(refill 紧跟末次使用;g2s 每 3 个 free 槽一发) | **0.9696** | — |
| refill 延后 2 条 mfma(避 mfma-src 写后读冒险) | 0.9604 (**−0.9pp**) | ds_read 越早发越好,冒险不是瓶颈 |
| g2s 全部前置(`fgap=1`) | 0.822~0.860 (**−15%**) | 一串 buffer_load…lds 把 mfma 流整段堵死 |
| g2s 按**指令数**均匀(而非 mfma 槽均匀) | 0.904~0.939 (**−8%**) | 槽均匀 = 跟 mfma 节拍对齐,才是对的口径 |
| m0 写与它的 buffer_load 拆开一个放置槽 | 0.9629 (**−0.7pp**) | 拆开反而更差,别管 m0 冒险 |

⇒ **别再调这个 emit 的排布**。下一个 wgrad 增益必须改**做的功**(tile 形状/象限),
不是改调度。同理 methodology/02 记的 vmcnt 膝点、ELGK 膝点也都已到位。

#### ★ tw wgrad 单 tile 成本模型(三点实测拟合,down balanced/144 tile/32 phase)

强制**所有** tile 走同一变体,直接量出每种变体的整核时间:

| 变体 | MFMA/phase | 池数(ds frag) | 整核 ms | 每 tile 相对成本 |
|---|---|---|---|---|
| (2,2) 全 tile | 64 | 4 (16) | 0.943 | 1.00 |
| (2,1) 半 N | 32 | 4 (16) | 0.773 | 0.82 |
| (1,2) 半 M | 32 | 3 (12) | 0.661 | 0.70 |

线性拟合:**固定 16% / 每池 feed 47% / MFMA 36%**(边际 22 cycle/MFMA)。
- 推论 1:边界半 tile 花 0.70~0.82 的时间做 0.5 的功 ⇒ `down` 全核有 **4.6%** 的边界浪费,
  `gate_up` 约 2.5%。这是 wgrad 目前最大的**已量化**开口。
  - ⚠ **口径修正(2026-08-10, gptoss wgrad campaign)**:上面 4.6%/2.5% 是 **wall 占比**。若 score 是
    small-M/large-M 的**比值**(两个 regime 的 tiles/group 完全相同),这笔浪费大部分相消,regime-aware
    折算后只值 ratio **+0.96%(dn) / +0.57%(gu)**。选杠杆时先把 wall 占比换算到 score 口径再排序。
  - ★ **边界体(`half_bnd`)与 band 深度(`group_m`)必须成对调**,单调任一个都拿不到收益:down deploy
    实测 深 band 单独 +1.00%、精简边界体单独 +0.62%、**两者一起 +3.00%**(均衡)/+1.94%(倾斜)。
    机理:band 里的廉价 tile 是跟同 band 的 h−1 个满 tile 一起走的,不是单独提前释放 CU。
- 推论 2:frag/MFMA 比 = `(ah+bh)/(4·ah·bh)`,(2,2)=0.25 最优,(1,2)=0.375,(1,4)=0.3125。
  128 行的条带要恢复 0.25 **只能**做 128×512(5 池)。
- ⚠ **`all-forced` 探针量的是全局 regime,不是可加的单 tile 成本**:据此预测"半 N tile 少读
  一个池省 0.9%",实做只兑现 ~0.2%(见 pitfalls/05)。

#### ★ tw wgrad 当前(2026-08-01)PMC 分解:缺口 = 2.5% 周期 + 1.5% 时钟

`rocprofv3 --pmc GRBM_GUI_ACTIVE SQ_INSTS_MFMA SQ_INSTS_LDS SQ_INSTS_VALU`,wgrad down balanced,末 20 次 dispatch:

| | SQ_INSTS_MFMA | SQ_INSTS_LDS | SQ_INSTS_VALU | GRBM_GUI_ACTIVE | wall |
|---|---|---|---|---|---|
| tw(4 波) | 34,668,544 | 38,117,376 | 60,316,288 | 13,240,302 | 0.906 ms |
| mx(8 波) | 34,668,544 | 27,131,904 | 53,316,608 | 12,916,310 | 0.871 ms |
| tw/mx | **1.000** | 1.405 | 1.131 | **1.0251** | **1.0402** |

周期只差 2.5%、wall 差 4.0% ⇒ **剩下的 1.5% 是时钟**(tw 每周期能量更高,与 +40% LDS 指令一致)。
LDS 指令差的机制已定量到条:tw 每 wave 每 K-block 64 条 `ds_read_b64_tr_b8`(8B/条,128×128 wave tile),
mx 24 条 `ds_read_b128`/`b64`(16B/条,128×64 wave tile),×波数 = 256 vs 192 = 1.333,
实测 1.405(余量是 prologue/tail)。**这是上游 layout(tw 拿到 token-major 操作数)决定的,不是本核可调项。**

#### ❌ 别去"提前"wgrad 的 256 条累加器清零(`v_accvgpr_write_b32`)

`21_final_isa.s` 里 `v_accvgpr_write_b32` 共 576 条 = 四个边界变体的活累加器数之和
(256+128+128+64),即每 wave-tile 256 条纯清零 VALU。看起来像 occ=1 下无法掩盖的串行开销,
但 ISA 实读:它们已被 LLVM **逐条交织进 prologue 的 9 组 `buffer_load_dwordx4` 之间**
(行 548-620 一条 accvgpr_write 一条 buffer_load),在 `wait_barrier` 之前就发完了 ⇒ 已经藏在
g2s 发射窗口里。把 `acc0` 的构造在 Python 侧往前挪是**无操作**。

#### ❌ `_diag_cells` 的第五次扰动:(bm,bn)=(2,2) 同样净负

上表四个方向之外再补一个:`bm,bn = 2,4`(基线)→ `2,2`,wgrad 组 geomean 0.98486 → 0.98272(−0.2pp)。
`_WL_ELGK` 12→13 则是 wgrad 组 +0.07pp / gm −0.14%(都在 0.2pp 噪声带内,且 13 没有做过 1500-rep
det 压力)⇒ 12 不动。**这个 emit 现在有五个独立方向的负结果,别再来。**

#### ★★ tw wgrad 的 wall 拆解:~89% 计算 + ~11% 输出写(2026-08-01 subtractive 实测)

把 epilogue 整个删掉(`do_store=False`,保留 whole-loop)后直接量:

| 配置 | 带 store | 无 store | 差 | 输出字节 | 推算带宽 |
|---|---|---|---|---|---|
| wgrad down balanced | 0.909ms | **0.812ms** | 97us (10.7%) | 555MB | 5.7 TB/s |
| wgrad gate_up balanced | 1.714ms | **1.549ms** | 165us (9.6%) | 1085MB | 6.6 TB/s |

两个形状都落在 ~6 TB/s(接近 HBM 写峰值)⇒ **这 11% 是带宽地板,不是发射/VALU**,
也解释了为什么 r5 的"正确轴宽存(256 store_short → 64 dwordx2)"只值 +0.2%。
mx 写同样字节数,**这部分两臂均摊,不是差距来源**:把剩余 ~4% 的 wall 缺口换算到
计算段(只占 89%)就是 ~4.5%,与 PMC 的 LDS 1.405x / VALU 1.131x 一致。
⇒ **别再为 wgrad epilogue 建 permlane/CShuffle 宽存**;要动就动计算段。

#### ★ wgrad prologue 的 rendezvous:拆分有微收益,且拆完已到减法上界

`_wholeloop_tile_3buf` 原本在 asm 前发 `s_waitcnt vmcnt(8)+s_barrier`,把 buf0+buf1 primes 全等完。
拆成"只等 buf0(vmcnt(24))→ asm 内 ntmp 条 prologue ds_read → `s_waitcnt vmcnt(8) lgkmcnt(0)` → s_barrier",
让 64 条 ds_read 的发射窗口盖住 buf1 的到达:**gm 0.99616 → 0.9973(三次读数均在基线之上)**。
★ 同轮的减法上界:把两个等待全放到 `vmcnt(63)`(数值错但计时有效)= wgrad_gm 0.99363
vs 拆分后 0.99553 ⇒ **prologue 侧已无剩余**,不要再为"隐藏 tile 开头的 HBM 延迟"去做
跨-tile 软流水(每 WG 2 tile、下一 tile prologue 提前发)—— 它能拿的那部分已经是 0。
同理 whole-loop 结尾那条 `s_waitcnt vmcnt(0)`(排空没人再读的 g2s)推迟到 epilogue 之后
也实测 wgrad_gm 0.99553 → 0.99322(净负/持平),别再做。

#### ❌ 别再把 tw wgrad 当 L2/HBM 带宽 bound(PMC 实测)

`rocprofv3 --pmc TCC_HIT_sum TCC_MISS_sum TCC_EA0_RDREQ_sum`,wgrad down balanced,末 15 次 dispatch:

| | TCC 请求(HIT+MISS) | L2 命中率 | TCC_EA0_RDREQ | wall |
|---|---|---|---|---|
| tw | 89.8M | **66.9%** | 25.5M | 0.918 ms |
| mx | 96.1M | 54.7% | 39.2M | 0.882 ms |

**mx 走了更多 L2 请求、更多 EA 读、更低命中率,却更快** ⇒ tw 不缺带宽,别再为"减 global 流量"
做设计。剩下的差距在 LDS 侧**指令条数**(tw 256 条/CU-phase vs mx 192 条,mx 的操作数 K-连续
可用 `ds_read_b128`),那是上游 layout 差异,不是本核可调项。

### 算术强度:4-wave 长 K 为何领先 8-wave

- 强度定义 = 每 K-step 的 MAC / (A行 + B列 operand-elem)。
  - 4-wave 128×128 方形 = 强度 **64**
  - 8-wave 128×64 长条 = 强度 **42.7**
- 强度高 1.5× → 单位计算的 LDS ds_read 少 1.5×(rocprofv3 实测 8w/4w LDS insts = **1.49×** 精确匹配)→ LDS 延迟更易藏。
- 这是 4-wave 长 K 领先 8-wave 的核心机制,单调随 K 增:**6.4%@K8192 → 10.5%@K28672**。

### fp8 big-K drain removal WIN(both-path-J)

- big-K(K 很大如 8192×8192×28672,长 K-loop,L2 已好)瓶颈 = A-side tr8 的 `s_waitcnt vmcnt(0)` 全排空(LLVM 对 intrinsic `ds_read` 保守插入)。
- **WIN = both-path-J drain removal**:`a_inline_asm=1,b_inline_asm=1`(A、B 都走 inline-asm `ds_read_b64_tr_b8`,opaque 给 SIInsertWaitcnts → 无自动 vmcnt(0) drain)+ `asm_mma=2`(绕过 agpr guard,MMA 仍 intrinsic)+ `agpr_alloc=0` + `vmcnt_hint=3`。配套的 asm-inplace MFMA(`asm_mma=2` wire `_asm_mma_do` mode2,D 别名 C in AGPR,`agpr_alloc=128`,消 accvgpr 拷贝 + spill 18→0):big-K **+2.3%,det0(3×2000 fresh)**,commit **1049c9ee**。
- `vmcnt_hint` 调到 det=0 上限:**vh=3 是 sweet spot**;vh=2 也 det=0 但更慢;big-N 的 det 上限也是 3。(注:此前版本记载的 424fe26b/2580→2756TF/2710-2734TF 数字系与 MXFP8 同名 skill 文件混淆,tensorwise fp8 源文件无此数据,已按 flydsl-fp8-gemm-tuning/SKILL.md 更正)
- **同 FLOPs 不同 shape 对照是金矿定位法**:big-N 慢就拿 big-K(同 FLOPs 同 kernel)对照 profile,差异指标(L2 51 vs 66)直接点出瓶颈。

### wgrad ≠ DVFS 功耗受限(区别于 dense fp8)

- 实测跑核满频 2400MHz ~266W,远低于 dense randn 的 ~1072W 功耗墙。
- 因 MfmaUtil 仅 ~54% 够不到墙 → wgrad 是效率/autotune 优化**真能体现**的路径。
- 反面:dense/fwd/dgrad 的指令效率优化被功耗墙掩盖(相同 MFMA → 相同功耗 → 相同频)。

## mxfp4 4-wave 生产 emit 杠杆:GAVOID/MMORD/INPLACE_ALT/WLBARNOP/ELGK/WLVMCN/SCV_ILV

### 生产成功 emit 杠杆(按 med 增益)

| env var | 值 | 增益 | 机制 / WHY |
|---|---|---|---|
| `FP4_INPLACE_GAVOID` | 1 | +55T | g2s 避开 refill slot |
| `FP4_MMORD` | 9 | Llama 7b-qkv +2% | blocked-diagonal 4×8:块状对角把同-acc 的 2 K-sub 隔开,消累加器 RAW stall |
| `FP4_INPLACE_ALT` | 0 | +27T | B-side progressive,须配 `MMORD=5` |
| `FP4_WLBARNOP` | 1 | +21T | barrier 后插 1 个 s_nop |
| `FP4_INPLACE_ELGK` | 9 | +27T | barrier 处留 9 个 ds_read 在飞 |
| `FP4_WLVMCN` | 10 | +10T | — |
| `FP4_SCV_ILV` | 1 | +20T (min) | scale load 交织进 mfma 流 |

### ★ 移植到 tw fp8(哪些 knob 跨精度成立,哪些不成立)

2026-07 把上表逐条搬到 **tw fp8 TN var-K wgrad 4-wave whole-loop** 上实测:

| knob | mxfp4 上 | tw fp8 上 | 结论 |
|---|---|---|---|
| `INPLACE_ELGK`(barrier 处留 N 条 ds_read 在飞) | +27T @9 | **+0.8% gm @12** | ✅ 跨精度成立,是该轮主要收益 |
| `WLBARNOP`(barrier 后插 1 个 `s_nop`) | +21T | **0.00%**(0.98016 vs 0.98027) | ❌ 不跨精度,别再花轮次 |
| vmcnt 侧再放宽 | — | **0.00%** | ❌ 已在膝点,见 methodology/02 |

- ⇒ **`ELGK` 类(lgkm 侧 partial drain)是这族 kernel 通用杠杆;`WLBARNOP` 类(纯发射
  时序微调)是 mxfp4 专属**。移植 emit knob 表时先做 lgkm 侧,别按表头顺序全试一遍。
- 同族的 **8-wave NN(dgrad)** 上等价杠杆不在 whole-loop asm 里,而在 `S2RLoaderTr` 的
  `vmcnt_hint`:稳态主环里那几条 `vmcnt(2)` 与每迭代的 rendezvous drain **完全重复**,
  删掉主环那份(prelude/epilog 保留)= dgrad 组 **+2.2%**、gm +0.67%。

### race-correctness 边界(稳定 emit)

- `ELGK`:最优 9,`≥15` racy。⚠️ **`15` 是 gfx950 `lgkmcnt` 4-bit 字段的编码上限,不是调参
  边界**(写 16 汇编器报错);racy 阈值 per-kernel,tw fp8 wgrad 上 12 最快、15 仍 det0。
  详见 methodology/02「mxfp4 4-wave 稳定 emit 边界」。
- `WLVMCN`:最优 10,`≥20` racy。
- barrier 是 race-critical:减 barrier(`LEANBAR`)必触发 operand race。
- cross-wave race 根因 = **LDS barrier 不足**,不是 vmcnt 乱序:g2s `buffer_load→LDS` 是 wave 协作完成,barrier 确保所有 wave 的 g2s 全部落地后,跨 wave 的 ds_read 才安全。

### authoring:VGPR-direct scale 补 refill-ahead lag

- 去 LDS ds_read、当场消费 scale 时,soffset 必须补 refill-ahead lag:
  `soffset = o_sca - AH*n_sub*256`,`AH=2`(refill-same,默认)/ `AH=1`(refill-OTHER)。
- WHY:LDS 路径经缓冲,延迟 `AH` 个 K-iter 才消费;VGPR-direct 当场消费须手动对齐这个 lag。
- 8w wholeloop:scale gmem 已是 per-lane 布局,consume-lane 天然对齐,**无需改 host 预处理**。

### authoring:PIN 机制(绕过 LLVM RAGreedy 卡死)

- 症状:`=&v` early-clobber 输出太多(如 2-set ping-pong / register double-buffer 的 +6 或 +96 个 `=&v`)触发贪心 RA 病态卡死(compile >130s 不返回)。
- 解法:把内联汇编的 `=&v` 输出换成 ISA 里显式 `v[pb:pb+3]` 物理寄存器字面量,`pb = PINBASE + 组偏移`。
- 对齐约束:PIN 下 scale VGPR 基址须对齐 —— `PINSC=1`(scale 在前)用 `PINBASE`,否则用 `PINBASE + 4*ntmp`;不对齐触发 SNR21 bug。

---
来源: flydsl-fp8-gemm-tuning/SKILL.md, 10-grouped-wgrad-4wave-3buf.md, gemm-optimization/SKILL.md, agpr_rawasm_progress.md, agpr_phase5_mono.md, 11-upstream-agpr-pin-moot.md, diag_4w_vs_8w.md, flydsl-fp8-gemm-results/SKILL.md, 03-emit-knobs.md, 10-8wave-scvgpr.md
