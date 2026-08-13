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
