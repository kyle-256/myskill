# Epilogue 存策略与转置/寻址:CShuffle、permlane16_swap、DS_READ_TR、i64 SRD rebase

> 类别: 方法论 · 主题标签: gfx950, transpose-load, LDS, inline-asm, epilogue-store, cshuffle, permlane16_swap, preshuffle, i64-addressing, SRD-rebase, readfirstlane, buffer-load

## gfx950 MFMA transpose load:DS_READ_B64_TR_*、intrinsic 保守 drain vs inline-asm

### DS_READ_B64_TR_* 指令族(gfx950 专属)
边读 LDS 边转置,免掉 ds_write + 置换地址的 ds_read,直接给 MFMA 准备转置布局的 A/B 操作数。

| 指令 | 数据类型 | 写出 VGPR |
|---|---|---|
| DS_READ_B64_TR_B16 | fp16 / bf16 | 2 VGPR |
| DS_READ_B64_TR_B8 | fp8 / bf8 | 2 VGPR |
| DS_READ_B64_TR_B4 | int4 | 2 VGPR |
| DS_READ_B96_TR_B6 | 6-bit | 3 VGPR |

- 硬性要求:EXEC 全 1;地址按数据大小对齐;≥64-bit 的 DS 访问要求偶对齐 VGPR(B96_TR_B6 例外)。

### intrinsic 保守 drain vs inline-asm(big-K 场景)
- 场景:big-K(长 K-loop、square-ish、L2 已经好)。此时瓶颈 = A-side tr8 的 `s_waitcnt vmcnt(0)` 全排空 —— LLVM 对 intrinsic 版 ds_read 插入保守的 vmcnt(0),把所有 in-flight vmem 全排空。
- 根因:SIInsertWaitcnts 看到 intrinsic ds_read 会保守自动 drain。
- WIN = both-path-J inline-asm drain removal:
  - `a_inline_asm=1` + `b_inline_asm=1`,用 `ds_read_b64_tr_b8` 的 inline-asm 版 → 对 SIInsertWaitcnts 是 opaque,不触发自动 drain。
  - 配套:`asm_mma=2` + `agpr_alloc=0` + `vmcnt_hint=3`。

## Epilogue 存策略:CShuffle LDS 重排 vs permlane16_swap 免 LDS 转置宽存

### 根因:MFMA 输出布局 row-strided → naive store 非合并
- MFMA-native lane layout 是 row-strided:单 lane 拥 4 连续行同列,相邻列由相邻 lane 持有。naive store 逐列走 → uncoalesced。
- 要宽存(`buffer_store_dwordx4`,8 bf16/lane)必须做 lane→col 真转置。

### 两条 epilogue 路线

| 策略 | 机制 | 开销 | 适用 |
|---|---|---|---|
| Direct Store(默认) | 每线程 MFMA 累加器直写 global | 无额外 LDS;某些 tile 非合并写 | 小 tile_n |
| CShuffle | 累加器 row-major 写入 LDS `[tile_m,tile_n]` tile → barrier → 重映射 threads 到 `(MLane,NLane)`(256 线程=8×32)再读,使一行内 lane 持连续列 → 合并成 128-B 事务(`buffer_store_dwordx2`) | 额外 LDS + 2 barrier | `tile_n≥128` |
| permlane16_swap | 寄存器跨 lane 重排完成转置(无 LDS、无 barrier),免费 VALU;转置后 `buffer_store_dwordx4`(16 store/tile,发射量 1/16) | 仅 VALU | store-bound 胖形状最优 |
| **quad_perm DPP 配对列**(最轻的一档) | 只把**相邻两个 n-fragment** 折进一个 dword:`v_cvt_pk_bf16_f32` 打包 + **一条 `quad_perm:[1,0,3,2]` DPP** 与邻 lane 交换 ⇒ 偶 lane 拿低片、奇 lane 拿高片,每 lane 持**相邻两列** | 每对 1 cvt + 1 DPP + 1 perm;无 LDS 无 barrier | 只想把 **32 B 请求并成 64 B、store 数减半**,又不值得上整套转置时 |

**为什么单靠 2 B 标量 store 只有 32 B**:16×16 MFMA 一行的一个 n-fragment 摊在 16 个 lane 上,
每 lane 2 B ⇒ 32 B/请求;fragment **对**覆盖 32 个相邻列,正好是两倍。
配对后一发 64 B。约束:**N 必须是偶数**(这一对是当作一个单元写出去的),
输出 **bf16**(靠的是 `v_cvt_pk_bf16_f32`),边界 clamp 要挪到**这一对的最后一列**上。
2026-08-09 dense fp8 NN 实测计入 **+0.85%**(同 session A/B),
是那一场唯一命中「访存侧(LDS/req per MFMA)」主轴的改动。

### CShuffle 细节
- `e_vec = 4 if tile_n%128==0 else 2`。
- 只在 `tile_n≥128` 值:小 tile_n 时 direct store 胜,两个 barrier 纯属浪费。
- shuffle 数学假设 default MMA row mapping;换 atom 需重写 row iterator + `(MLane,NLane)`。

### permlane16_swap 实战(mxfp4)
- 正确的 permlane16_swap 转置 + dwordx4 是 store-bound 胖形状最优 epilogue:8192²×4096 从 fly/ait `1.039 → 0.998`。
- ⚠️ **转对轴才有效**：`pitfalls/25` 记录过"转错轴"的 permlane 判负(不降 store 事务数)。区别只在转的是不是 MFMA 输出的正确轴——对轴=最优,错轴=白做。
- ⚠️ **"最优 epilogue" 只在 store-bound 胖形状成立,别默认迁移**。同一手法(TACCW 宽存)搬到 **mxfp4 grouped NT**(G=32、N=2944/5760)实测 **+0.7~1.3% 更慢**(fwd down +1.34 / fwd gate_up +0.70 / dgrad gate_up +1.07)。不是矛盾而是不迁移:上面那条来自 dense 8192²×4096(行跨度 16 KB、N%256==0),而 grouped 这边行跨度只有 5888 B。**判据:先量 store 在 epilogue 里的占比,再决定要不要动存策略。**
  这句判据非常正确并且救过整轮(照着"上面那条最优 epilogue"直接做宽存会稳定亏 3.5~4.1%),但**它下面原先引用的
  ATT 归因数字("store 只占 per-tile 固定开销的 8%、42% 在 VALU 链")在 2026-08 的 mxfp4 grouped NT 上被直接消融
  证伪,已删除**:删掉整段 C-store epilogue = **−4.87 µs/tile = 61% of F**(F=7.943 µs,K-sweep 六点同 run 拟合),
  另一独立尺子 = 删 C-store 让 `kern_1` 从 579.2 → 476.5 µs(**−17.7%**);而删掉 254 `v_accvgpr_read` + 254
  `v_cvt_pk_bf16_f32`(256 条 store 一条不留)只值 **−0.5%(噪声内)**。旧数字大概来自更早的核/调度策略。
- ✅ **2026-08-18 补:这一族的真判据不是"宽存有没有效",而是"lane→列的重排能不能落成一个纯地址改动"**
  (gpt_oss-20b down fwd NT grouped fp8,G=32/N=2880/K=2944,tensorwise,占用率 1 WG/CU,**+2.7% 已落盘**)。
  同一个核上此前三条负号(本卡的 TACCW +0.7~1.3% 更慢、`store_cshuffle` −21%、DPP `quad_perm` 配对
  −1.09/−1.9%)**全部是"付跨 lane VALU 或 LDS 中转去换宽 store"**;把重排搬到**喂料侧的地址**上就翻正号:
  - 做法:一个 lane 在同一输出行持有的两个 n-fragment 原本相距 16 列(⇒ 每列一条 `buffer_store_short`)。
    把 B 进 LDS 时的 **operand 行**按对合 `16*t+m ← 2*m+t`(32 行块内)置换,这两个 fragment 就成为
    **相邻两列**,一条 `buffer_store_dword` 覆盖两列。g2s 是 `buffer_load_dwordx4 … lds` 直投、每 lane 的
    global 偏移本就是任意常量 ⇒ **零指令代价**;XOR bank key 仍按 LDS 行算 ⇒ **s2r 读侧逐位不变**。
  - 代价栏(ISA 自证):`buffer_store_short 160→32`(余下的是保持标量的窄边界体)+`dword 0→64`,
    **`v_cvt_pk 160→96` VALU 反而更少**(标量路每条 store 都发一条 pk-cvt 却只用一半结果),
    `v_mfma 920`/`s_barrier 304`/`.vgpr_count 256`/`spill 0`/`LDS 131072` **逐项不变**。
  - **前置条件(决定免费还是要付 VALU)**:输出列必须是那个 operand 的**慢轴**。`trans_b` 的 NT 满足
    (输出列 = B 的行);NN/TN 的输出列是 operand 的**快轴**(一个 lane 的 dwordx4 覆盖 16 个连续列),
    按列配对会打断 16 B 连续性 ⇒ 那两个核要改成"在 `ds_read_b64_tr_b8` 的 lane→列映射上置换"或
    "按 16 B 粒度换 LDS 写",别直搬。**这两条后路 2026-08-18 已被证伪,替代做法见下一条。**
- ✅ **2026-08-18 补:输出列是 operand 快轴时的地板做法 = 行合并(`v_permlane16_swap_b32` 骑在打包好的一对行上)**
  (同一 campaign 的 dgrad NN,grouped fp8,G=32/N=2880/K=2944,1 WG/CU,**+0.83/+0.43/+0.62% 三分布已落盘**,
  逐位相同/det 10/10/M 作用域 1024·2048·4096·8192 四点全正 +2.20/+0.85/+0.83/+0.59%)。
  - 🚨 **先把上一条的两条后路销掉(源码级证明,别再花轮次写探针)**:
    ① **`ds_read_b64_tr_b8` 的 lane→列映射不可寻址**。N 坐标只从
    `j_chunk = (wave_idx*tile_stride + tile_i*16)//16 ^ swz_K//16` 进地址,这是**wave/tile 均匀**的;
    per-lane 那部分只有 `(L//2)`(K 行)与 `(L%2)*8`(16 B chunk 内的字节半)。⇒ 地址置换只能决定
    **整个 16-lane 组读哪个 16 列 chunk**,chunk 内"lane m 拿第 m 列"是硬件转置写死的。
    ② **按 16 B 粒度换 LDS 写也不行**:一个 16 B g2s chunk 就是 16 个**整**输出列,把偶/奇列交错需要
    1 字节粒度,而 g2s 是 `dwordx4`。
  - 做法:两个 n-fragment 相距 16 列这件事**不去改**,改的是**行**:一条
    `v_permlane16_swap_b32` 交换第一操作数的奇行组与第二操作数的偶行组
    (四个 16-lane 行组 r0..r3 ⇒ 返回 `(a.r0,b.r0,a.r2,b.r2)` 与 `(a.r1,b.r1,a.r3,b.r3)`),
    于是**每个结果的每个 32-lane 半区里都同时有两个 fragment 的列**:lane 的 32-lane 半区选行组、
    16-lane 组选它现在持有哪个 fragment 的列 ⇒ 一条 store 的 32 lane 覆盖**一整行 64 B**,
    而不是 16 lane 各覆盖 4 行的 32 B。**存指令条数一条不变**,每条 store 的请求数 4→2。
  - 🔑 **成败全在"swap 骑在什么上"**:同一机制**按 f32 逐值 swap 要 64 条 permlane ⇒ 只剩 +0.13%**;
    改成先按 **(row i, row i+1) 打包**再 swap 打包后的 dword ⇒ permlane 减半到 32 条、
    并且 **`v_cvt_pk 160→96`**(标量路每个值发一条 pk-cvt 却只用一半结果,打包把这一半拿回来)
    ⇒ **+0.83%**。⇒ 跨 lane 原语的定价单位是"每条原语搬几个输出值",不是"用了没用"。
  - 代价栏(ISA 自证,`spill 0`/`LDS 131072`/`accum_offset 96`/`v_mfma 920`/`s_barrier 374`/
    `s_setprio 364`/`s_waitcnt 563`/`ds_read 1196`/`buffer_store_short 160` **逐项不变**):
    `v_cvt_pk 160→96`、`v_permlane16_swap 0→32`、`v_lshrrev_b32 9→73`、`s_nop 223→243`,净 +52 条。
  - ⚠ **`buffer_store_short_d16_hi` 在 flydsl 里拿不到**:`rocdl.RawPtrBufferStoreOp` 走 buffer-store
    intrinsic,而 AMDGPU 的 `d16_hi` 存选择只对**通用 store** 生效 ⇒ 取 packed bf16x2 的高半必须付一条
    `v_lshrrev_b32`(本核 64 条)。同一改动的**零跨 lane 成本上界臂**(同行集同字节同发射条数、只是不做
    swap 所以值是错的)量到 **+1.23%**,而落盘的正确版是 +0.83% ⇒ 剩下的 **0.40 pp 就是这 64 条 lshr
    + 32 条 permlane + 20 条 s_nop**,下一手是用 inline-asm 发 `d16_hi`。
  - 窄边界体不做:它的 `_bnd_ntb` 是**奇数**(本核 1),没有可配对的第二个 fragment;算术上也只值全部写
    请求的 2.2%。
- 🔑 **2026-08-18 补:本机 TCP→TCC 的写请求粒度是 64 B,不是 128 B 的 cache line**(gfx950 实测,
  同一个核三档 store 宽度):bf16 标量 32 B/store ⇒ `755 MB/32 B = 2.36e7`,实测 **2.359e7**;
  成对 64 B ⇒ 预测 1.206e7,实测 **1.206e7 逐位吻合**;再加宽到 dwordx2(128 B)⇒ 发射条数再砍半
  (`TA_BUFFER_WRITE_WAVEFRONTS 3.015e6→1.573e6`)但 **`TCP_TCC_WRITE_REQ` 仍是 1.206e7 一点不动**,
  wall 反而 −0.85 pp。⇒ **"25%→100% line 覆盖"这种以 128 B 为分母的算法会高估一倍**:
  分母是 64 B 的请求,所以标量是 50%、成对就是 100% = 地板。
  ⚠ **"`permlane16_swap` 无处可赚"只对已经站在 64 B 地板上的核成立**:对还在 32 B 的核(输出列 = operand
  快轴,没有免费地址置换)它正是把覆盖抬到地板的那一手,见上面的行合并条 **+0.83%**。
  功耗墙上的兑现率:**−48.9% 写请求 ⇒ wall +2.7%**(KB 既有的读侧案例是 −49.5% ⇒ +2.04%)。
  ⚠ **但别把"减请求 ⇒ 买时钟 ⇒ 稳态也快"当成写侧的通则**:NT 这一笔 6 点重拟合是
  **ΔF=−14.4 µs 与 ΔP=−0.389 µs/iter 同时为负**,而**同机同族的 NN 行合并在写请求同样 −48.9% 的前提下
  是 ΔF=−9.0 µs(−5.1%) 但 ΔP=+0.157 µs/iter(正号)**。两者的差别不在请求数(都减半),而在**存指令/VALU
  条数**:NT 那笔顺带把 `buffer_store` 160→96 与 `v_cvt_pk` 160→96 都砍了,NN 的存指令条数一条没变
  (`TA_BUFFER_WRITE_WAVEFRONTS` 两臂同为 5.898e6)。⇒ **ΔP 的负号要归给指令条数,ΔF 的负号才是请求数的。
  给写侧的臂定价时只承诺 ΔF。**
- ★ **epilogue 分解必须是三件套探针**:①**删整段**(量总量)②**同字节数只改段数**(量请求)③**保段数只删 VALU**(量发射)。
  只有 ① 动分数 ⇒ 成本是字节/带宽。mxfp4 grouped NT 实测 ②=+0.8%、③=−0.5%、①=−20.3% ⇒ **字节限**。
  ⚠ 任何改变**写入字节数或行集合**的"请求"探针都会假阳性:把 4 行塌成 1 行的"1 段"写法量到 −2.58 µs,
  但它同时把写字节掉了 ~6×。做 ② 之前先核对 `M×N×2` 的覆盖面逐项不变。
- inline-asm **不需要写死 `s_nop 1`**:该 hazard 只针对后续 VALU 读结果,而消费者是 `buffer_store`(VMEM 读 vgpr)不触发。
- 降 nop 必须两步都做:
  - 去掉写死 s_nop + 两阶段(先全部 cvt+permlane 进独立 VGPR,再 burst 全部 store,把 permlane 与 store 拉开)→ ISA `s_nop 80→0`,bit-exact。
  - 单去 s_nop 只到 19 nop;单两阶段不去 s_nop 仍 64 nop;**两者都做才 0**。

### ★ peel-last:把 C-store 插进最后一个 K-block 的 MFMA 阴影(occ=1 grouped wgrad 实测 +0.7~1.1%)
- 场景:字节限 + occ=1 ⇒ store 的量既不在 VALU/宽度/顺序上(四项都实测 ≈0),而在"drain 无算力可藏"。
  **tile 内唯一存在的可藏算力 = 最后一个 K-block 的 mfma**;把它从 whole-loop 手写 asm 里整块剥出来,
  以 row-tile major 重发,并把每个 row-tile 的 store 插到**下一个** row-tile 的 mfma 之后(距离 1 最优,2/4 更差)。
- **不需要把 store 搬进手写 asm**:让 asm 额外返回它末相已填好的 srcA/srcB 片段(`return_frags`),剥出的
  block 直接复用寄存器,不重读 LDS。代价 vgpr +20、spill 0、store/cvt/accvgpr_read 条数不变。
- ⚠ **asm mfma 对 backend hazard recognizer 不可见**:每组 store 前 `sched_barrier(0)` 固定发射距离,
  最后一个 row-tile 后补 `s_nop 15`×2,否则 VALU 读到过期累加器(静默错,且只错最后收尾的 row-tile)。
- 上限就是一个 K-block 的 mfma:deploy 每 tile 32 个 K-block,最后一个 ≈1.3 µs 而 drain ≈4.3-4.8 µs
  ⇒ 只回收 store 成本的 ~10%。要吃剩下的必须有**跨 tile** 的 mfma,而统一 in-order vmcnt 封住了它
  (见 pitfalls/05:store 藏进后续 fill/K-loop 整族)。

### B 矩阵 preshuffle(配套,减 load 侧 shuffle)
- CPU 上把 `[N,K]` 预转置重排成 `[N/16, K/kpack, 4, 16, kpack_bytes]`。
- kpack:FP8/INT8 = `64//elem_bytes`(=64);BF16/FP16 = 4。
- 维度含义:`4` = 每 lane 4 dword(`buffer_load_dwordx4`),`16` = MFMA 内 16 lane。
- 收益:global load 直映 MFMA 布局免 VALU shuffle、合并访问;一次性 CPU 成本摊薄。

## >4GB 寻址:per-tile i64 SRD rebase、_readfirstlane pin SGPR 免 waterfall

- **核心技术 = per-tile i64 SRD rebase**：AMD buffer SRD 的 `num_records` 和 per-lane `voffset` 都是 32-bit,上限 4GB。>2^31 元素 / >4GB 寻址时,把每 tile/group 的巨大元素 base(`m_row*K` 等)折进 SRD 的 **i64 base**,in-tile 小偏移留 **i32**。
- **base/num_records 必须 pin SGPR**：i64 rebase 后,SRD base/num_records 必须用 `_readfirstlane_i32` pin 到 SGPR(对 i64 高/低 32-bit 分别 pin)。
  - WHY:不 pin 则 base 来自 `group_scan`,被判为 VGPR → SRD 落 VGPR → 每次 K-loop `buffer_load` 触发 readfirstlane waterfall(16 次/load)。
  - 证据:gateup **-16%** / down **-13%**。
- **i64 entry 传全 rank 张量**：entry 函数必须传全 rank 张量(`a.view(torch.int8)` 保持 2D/3D),**不能 `reshape(-1)`**。
  - WHY:shape-pack `struct 'i'` 在元素数 >2^31 时 crash。
- **i64 K-loop offset 乘法用 `arith.index(k*BLOCK_K)`**：`k` 是 `range_constexpr` 的 python int,必须用 `arith.index(k*BLOCK_K)`,**不能 `arith.index_cast(T.index, python_int)`**。
  - WHY:后者对 python int 会 crash(`_to_raw` 不接受 int,报 `'int' has no _CAPIPtr`)。
- **验证**:检查 SNR 在 `<2^31` 和 `>2^31` threshold 两侧相等(两侧 SNR 相等 = 无溢出断点)。

---
来源: lds-optimization/SKILL.md, flydsl-fp8-gemm-tuning/SKILL.md, gemm-optimization/SKILL.md, gemm/optimization-directions.md, project_mxfp4_epilogue_store.md, 05-int64-addressing.md
