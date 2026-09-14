# mxfp4/mxfp8 死路总章：occ 天花板、epilogue store 暴露、feed-bound、scale 投递税、authoring 布局坑

> 类别: 踩过的坑 · 主题标签: occupancy, LDS-bound, mxfp4, K28672, epilogue-store, split-K, ds_bpermute, wgrad, LDS-feed, dead-ends, 8-wave, dense-gemm, correctness, mxfp8, scaled-mfma, scale-delivery, whole-loop, E8M0-scale, K-tail, mxfp8-grouped, over-run, SRD-layout, config-sweep, grouped-gemm, quant, authoring-layout, scale-pack, dual-cast, LDS, MFMA, flydsl-bitops, silent-numeric-bug, JIT依赖收集, 模块隔离, RecursionError

## mxfp4 K28672 天花板：occ=2 唯一途径 BK128 让 g2s 翻倍净亏 13%

- **5405T 天花板根因 = occ=1 wave/SIMD，由 LDS 144KB/wg 锁死（非 VGPR）**：160KB/CU ÷ 144KB = 1 WG/CU；2 个 WG 需 288KB > 160KB 物理不可能。即使 VGPR 允许 2 waves，LDS 也只允许 1。单 wave 无法用第二个 wave 的 MFMA 去填 operand bubble → **MFMA 89.5% stall**。

- **减法探针成本（500-sample，4-wave K28672 BK256）**：
  - `g2s`（global→shared，即 buffer_load→LDS 的 DMA 预取，HBM-BW-bound）= **1061T = 主瓶颈**
  - `ds_read`（96 read / 256 mfma）= **292T**
  - `scale-load` 成本已被 GAVOID + SCVGPR（scale 常驻 VGPR、免主循环重复 VMEM load）+ ILV 完全藏住（GAVOID/ILV/CONSTSC 等均为内部探针名；CONSTSC = prod real 5401）
  - 减 g2s 需更大 tile → AGPR 溢出；减 ds_read 需 BK128 → loop 开销更大。K28672 下 5500 med 超出 4-wave BK256 物理顶。

- **K 越长 hiding 越充分**：K8192 = 4603 / K28672 = 5401 / K57344 = 5481。

- ❌ **别再试 occ=2（净亏 -13%）**：BN256 + BK128 达到 occ=2（LDS 72K，VGPR 176），确实把 MFMA stall **89.5%→58.7%**，但 occ=2 的**唯一达成途径就是 BK128**（BK256 时 LDS 都 > 80K）。BK128 让 **g2s 频率翻倍 → VMEM-load stall 2.7%→30%**，完全压倒 MFMA 改善，总速 **5405→4686**。

- ❌ **别再试 BN128 单 slice**：只会让 g2s 更暴露，无收益。

- ❌ **别再试参数扫描（EVENSPREAD / PREFETCH / MMORD / ACC_DIST16 / PINBASE / WLVMCN / BARNOP）**：全落在 **5400±10 噪声内**，因为都不改 occ=1 根约束。

- ❌ **严禁测 K=57344**。

## mxfp4 4-wave 胖形状 ~6% gap：bf16 store 完全暴露，重叠/变宽/atomic 全死路

### 根因：bf16 store 与 compute 0% 重叠
- mxfp4 4-wave 胖形状 ~6% gap 根因 = bf16 输出 epilogue store **完全暴露**（0% 与 compute 重叠）。
- gfx950 LDS=160KB/CU，操作数双缓冲已用 144KB（occ=1）仅 ~16KB 空闲 → 无空间做 store staging。
- store 本质 HBM-write-BW-bound：8192² 128MB ÷ 6.1TB/s ≈ 76% HBM 峰 ≈ **19.7us**。
- ★ **称 store 成本必须在两个并发点各称一次(L≈24 与 L≈240),而且必须再加一个"真网格"点**。三点缺一就会误判主线：
  - 低并发点 = **per-CU 写路地板**(mxfp4 grouped NT 实测 1.66 µs/128KB = 77 GB/s/CU;256 CU 全开需 19.7 TB/s ≫ 纯写上限 7.01 ⇒ 单 CU 写路从来不是限制项)。
  - 满并发锁步点 = 同相突发代价(L=240 → 3.04 µs)。
  - **真网格点(多代、tiles ≫ 256)才是要优化的那个数**,而它**不在前两点之间**：实测 6144 tile/24 代 = **4.59 µs/tile,比全锁步的 3.04 还贵 51%**。
    根因：锁步时"大家一起写、没人在读",store 独占写通路；多代稳态下各 CU 相位散开,store 与别的 CU 的 operand **读**流混跑,
    落到 1R:1W 的 5.33 TB/s 档而不是纯写 7.01 TB/s 档。**⇒ 去同相不是收益,是代价来源**。
  - 判据：`store 字节 ÷ 实测增量时长` 对比纯写上限。真网格实测 805 MB ÷ 110 µs = **7.3 TB/s ≈ 纯写峰**
    ⇒ store 已 100% 暴露且 100% 带宽限,**没有未兑现的重叠折扣可拿**;剩下的开口是**减写字节**或**减读字节以腾出混跑带宽**。
- WHY 只有 fp4 有 gap：fp4 2× FLOPS/同输出，固定 store 被**快 compute 放大成暴露尾部**；fp8 同样的 store 被 2× 慢的 compute 盖住，无 gap。

### gfx950 计数器约束（贯穿所有死路）
- gfx950 `s_waitcnt` 只有**统一 vmcnt**，无独立 store 计数 `vscnt`（vscnt 是 gfx10/11；per-counter 是 gfx12+）。
- 后果：在途 store 占 vmcnt。任何把 store drip 进主循环的方案，都让每个 g2s 的 vmcnt wait 全卡在 store 上（issue/wait 串行化）。

### ❌ 别再试：store 与 compute 重叠（所有 flavor）
- **STORE_ILV（in-loop 摊 store）** — 仅回收 2.9/19.7us。主循环 VMEM 已被 g2s 打满，无空余 store 发射槽，interleave 后变 issue-bound。
- **寄存器 acc 跨 tile 存活** — 非-persistent 无 next-tile 可存活；persistent 需 256-iter_arg 大改，天花板 <3us。
- **persistent overlap** — store 的 HBM 写与下 tile g2s 的 HBM 读抢同一条总线，不构成真重叠；再叠 per-tile barrier 开销 → 净亏。
- 机制统一为：在途 store 占统一 vmcnt（见上），无法与 g2s 并行。

### ❌ 别再试：store 变宽 / coalescing（所有维度）
- **CShuffle（经 LDS 暂存做 128b 宽存的 epilogue 重排手法）** — neutral 到略差。LDS round-trip 延迟 ~12us（128 ds_write + lgkmcnt 串行 wait + ds_read）吃掉 8× 降发射的收益。
- **DPP / permlane16_swap（转错轴）** — 只换 16-lane 组、转错轴。HW 已把跨 lane store coalesce 成 128B/行（80% HBM 写峰 = 该 pattern 上限），任何寄存器/LDS 重排都不降事务数（masked-off lane 仍发射）。
  - ⚠️ **注意区分**：这里判负的是**转错轴**的 permlane。**转对轴**的 `permlane16_swap` 转置 + `dwordx4` 宽存反而是 store-bound 胖形状的**最优 epilogue**（8192²×4096 从 fly/ait 1.039→0.998），见 `methodology/07`（§Epilogue 存策略，permlane16_swap）。别因为这条把 permlane 整体当死路。
- **ds_bpermute 正确宽存（dwordx2）** — 全面更慢：LDS-crossbar 延迟 > 省的发射量；dwordx2 仅 32B coalesce 无提升，外加 64 crossbar + drain。

### ds_bpermute 三个坑（从全错救成 maxdiff=0）
（`ds_bpermute` = 跨 lane 寄存器重排指令，此处用于 epilogue 转置宽存）
1. **in-place dst==src 损坏跨 lane 读** — native rocdl 与 inline-asm `=v,v,v` 都被 RA coalesce 成 in-place → 必须 inline-asm `=&v,v,v`（earlyclobber）。
2. **inline-asm ds_bpermute 是 opaque** — 编译器不插 `s_waitcnt lgkmcnt` → 必须显式 `rocdl.s_waitcnt(0)` drain；但 per-bperm 手写 lgkmcnt(0) 会序列化退化，**必须批量 wait**。
3. **FlyDSL JIT 缓存不 hash 模块级方法** → 需 `FLYDSL_EXTRA_SOURCE_DIRS` bust cache。

### ❌ 别再试：atomic 融合 reduce（dense split-K）
- **f32 逐元素 atomic** — 慢 2-3×：256 scalar atomic/tile 打重叠地址，硬件串行 + 2× 流量。
- **packed bf16 cshuffle-atomic**（AITER 同款 `global.atomic.fadd.v2bf16`）— 慢 0.59×：dense split-K 下 S 个 split 都 atomic-add 到同一批 [M,N] cache line，HBM atomic 争用串行。
- WHY AITER 那个高效：MoE token-scatter 各写不同地址（无争用）；dense GEMM 不适用。
- **split + reduce 是最终方案。**

### 实现级坑
- `buffer_store_short $t` 数据源必须单寄存器：若 `t` 是 `vector<4xi32>`（4 寄存器）汇编器报 `invalid operand for instruction`，须改单 i32 vgpr。
- `flyc.compile(raw, *args)` 会**执行一次 kernel** 进给它的 buffer → atomic 累加 kernel 必须用 throwaway buffer 编译、再跑进真 zeroed 输出，否则编译时那次执行污染结果。
- persistent 路径 CShuffle 损坏：`_do_overlap` 显式假设 store 只碰 VGPR/gmem 不碰 LDS；CShuffle 用 LDS+barrier 破坏 overlap 的 vmcnt/lgkmcnt 记账。
- persist+TACCW 在非默认 swizzle（非 `(4,14,8)`）下 maxdiff 数千（源码 unsafe 警告是真的，别开）。

## wgrad feed-bound 死路全清单：占用率非杠杆，根因 LDS 转置读 feed 带宽

**根因定性**：wgrad 这条 kernel 的瓶颈是 **LDS 转置读 feed 带宽**，不是占用率。occ=1 的 512-VGPR 满载锁死了 prefetch 深度；任何"升 occ"的改动都会引入 feed 争用，净负。真正剩余杠杆 = 在 256 tile 下提 feed 效率（LDS bank / transpose-read / phase-barrier），需 rocprof 定位，别再盲扫参数。

### 占用率不是杠杆（升 occ 全负）

- ❌ 别再试：**8-wave 升 occ(occ≥2)**。反而更慢——occ=1 的 512-VGPR 满载锁死 prefetch，升 occ 引入 feed 争用。
- ❌ 别再试：**4-wave whole-loop occ=2**。实测否决：8-wave 2148 < safe-3buf occ=1 2214，feed 恶化盖过 latency-hiding。
- ❌ 别再试：**缩 tile**（为了升 occ）。净负。

### 结构杠杆本轮全判负（勿重试）

- ❌ 别再试：**store-defer**。LDS 160KB 悬崖：x2 中性，x3 **-18.6%**。
- ❌ 别再试：**免一个操作数转置**（`ds_read_b64_tr_b8` 换 plain `ds_read_b64`）。ISA 等价，**0 收益**——转置读不比 plain 读贵，省不下来。
- ❌ 别再试：**epilogue write128 / tr16**。**-2~7.7%**。
- ❌ 别再试：**跨 tile 重叠**。**-5%**。
- ❌ 别再试：**epilogue 双缓冲去 drain**。**-1~3.5%**。
- ❌ 别再试：**barrier 参数扫**。全平。
- ❌ 别再试：**增大 reg tile**。根因是 occ=1 满载锁死 prefetch，加 reg 只会更挤。

### 旋钮全平（<2%，别再扫）

- ❌ 别再试：worst shape 的 **swizzle{5} × xcd{1,2,4,8} × vmcnt{1..8}** 全平（<2%）。瓶颈是 feed 带宽，不是这些旋钮。

### grid 调度杠杆判负（下为 grouped MXFP8 var-K wgrad 内核，与本卡 fp8-TN wgrad 内核不同，勿混淆）

- ❌ 别再试：**persistent-across-groups**（grid `G*TILES→TILES`，一 WG 顺序做同一 tile 位置的全部 G 组）。此结论来自 **MXFP8 分组变长-K wgrad 内核**（`mxfp8_grouped_kernel.py`，与本卡其余部分讨论的 fp8-tensorwise TN 4-wave whole-loop wgrad 内核是不同内核/不同量化方案/不同 benchmark harness）。该内核中设计正确（SNR 28.14）但性能大幅倒退（MX/TW 从 ~1.02x 恶化到 **1.23-2.00x**）。根因：grid 缩 G× 后 TILES_PER_GROUP 只 **276-448**，在 256 CU 上仅 **1.1-1.75 波**，occ=1 下负载严重不均 + 8× 展开代码 I-cache 抖动。⇒ grid 缩 G× 与 occ=1 的 CU 负载均衡根本冲突，persistent 只在 **TILES_PER_GROUP >> num_CU** 时才有利，本卡的 fp8-TN wgrad 内核未验证此杠杆。

### 测量方法论坑

- ❌ 别再信：**"跳过整条指令测天花板"类探针**（如 `PT_TR_HALF` 跳过读）。跳过读 ≠ 换成更少的等效读。真实替换后（`ds_read_b128` 换 2×`tr-b8`）因带宽受限**收益归零**。测"去掉 X 的天花板"必须用真实替代指令，不能靠删指令——否则天花板虚高、误导方向（见 `methodology/03` §subtractive/HALF「上界≠可达铁律」）。

## 结构性死路杂项：8-wave 达不到 4-wave 长 K、dense SRD/split-K、scf.for iter_args

### 8-wave 结构上限
- 8-wave 每 SIMD 2 waves，顶到 256 寄存器就到头，长 K 结构上无法追平 4-wave 性能。相关实现见下方各条 ❌。

### wgrad 死路清单
- ❌ 别再试 chunked K-loop 双缓冲：净负。每 chunk 重启流水线 → fill bubble 更差。
- ❌ 别再试 masked chunk size 扫参：8→4→2 全中性。chunk round-up 不是 skew 瓶颈。
- ❌ 别再试 register-prefetch wgrad swpipe：净负。
- ❌ 别再试 persist round-robin + band-cyclic：无效。persist skew 瓶颈是 store/prologue-bound，不是 dispatch 顺序。

### fwd 死路清单
- ❌ 别再试 inter-tile 软流水（SWP prefetch）：中性，fwd −2%。store_c waterfall 才是真瓶颈，不是 G2S bound。
- ❌ 别再试 store_c CShuffle：中性。fwd 不是 column-strided store 瓶颈（dgrad 才是）。
- ❌ 别再试 GROUP_M/XCD 扫参：噪声。autotune 已收敛，over-launch 不是瓶颈。
- ❌ 别再试 3-stage B ring buffer（fwd）：中性（+0.05%）。L2 miss latency 来自 8 个不同 expert 权重 thrash 4MB/XCD L2；加 prefetch distance 无效——prefetch 治不了 capacity thrash。

### dense gemm 死路清单
- ❌ 别再试 per-iter SRD base 前进（去 4GB cap）：NN/TN −2%，per-load SRD 重建约 2% 代价。全 no-cap 也 −2%。bench 内最大 3.49e9 < 2^32，foldable + cap 已够用。

### 3-buffer LDS 超限
- ❌ 别再试 3-buffer BK256：LDS 超限。A3+B3 = 192KB > 160KB；A3+B2 = 160KB 恰满；用 SCVGPR 省 scale 16KB 后 A3+B2 = 144KB 可放，但 A3 = SUBSTREAM broken，SNR 1.8。
- ❌ 别再试 8-wave BN512：实现完但墙③ spill-dead（374TF）不值得做；实现时 SNR 24.6，且有 scale layout bug。

### 正确性坑：whole-loop unroll-2 奇数 KI phantom iter
- mxfp4 whole-loop unroll-2 do-while，在奇数 KI（K%512==256）会多算 1 个 phantom iter k=KI：g2s 偏移=下一行起点，不 OOB，但累加下一行垃圾 → SNR 5-16。
- 真实影响：7b-down（K=11008，KI=43）。
- 修法：传 `nval = KI - (KI & 1)` floor-even；循环后对 `INPLACE & SCVGPR & ki&1` 发 MFMA-only phase-A tail，消费 1-ahead 预取的 k=KI-1。
- ❌ 别再试 standalone BLOCK_N=128：也算错，不可用。

## MXFP8 残余 gap 是 scale 投递税不是 scaled-MFMA 指令税（≈0）

核心诊断卡：优化方向是**隐藏 scale load / 减少分散 scale VGPR**，而不是减 MFMA。

### scaled-MFMA 指令本身税 ≈ 0%（此前 5% 结论是错的）
- 探针 NOSCV / NOSCALE_MMA / NOSCMMA 实测 scaled-MFMA 指令本身税 ≈ 0%：
  - dense Down：scaled 3068 vs 非-scaled 3093 = 0.8%
  - grouped wgrad：807 → 807（去掉 scaled-MFMA 不变）
- ❌ 别再试：把残余 5% 归因为「scaled 税」——结论错。真凶=scale VMEM load 未隐藏（big-K 跳 scale load 后达 0.97-0.99）。
- WHY：scaled-MFMA 只是多吃一个 scale 操作数，指令自身无额外 issue 代价；成本全在把 scale 字节从 VMEM 搬进 VGPR。

### ★ MFMA 形状轴（16x16x128 ↔ 32x32x64）关闭：同 FLOP 管道周期逐位守恒（gfx950 实测 2026-08-05）
一整轮的代价换来的三个数，动手前引用它们就能否掉「换更大的 MFMA 形状减指令数」这类立项：
- `SQ_VALU_MFMA_BUSY_CYCLES / SQ_INSTS_MFMA` = **恰好 32.0**（`v_mfma_scale_f32_16x16x128_f8f6f4`，A/B 均 fp8）
  与 **恰好 64.0**（`v_mfma_scale_f32_32x32x64_f8f6f4`）⇒ 与 connection/common/05 的延迟表逐项吻合，
  且**指令本身零附加税**（呼应上一节）。
- 同一个 grouped mxfp8 NT 核，只把主环 MFMA 换成 32x32x64（ISA 除 MFMA 条数 1104→552 外逐项字节相同：
  ds_read/buffer_load/store/cvt/barrier/setprio/v_mov 全等，VGPR 246→244，spill 0）⇒ 聚合
  `SQ_VALU_MFMA_BUSY_CYCLES` **1.10939e9，与 16x16x128 版逐位相同**，e2e **12/12 NT 格慢 0.8-3.0%**。
- WHY 变慢：32×32 tile 让一个相位只剩 **2 条独立累加链 × 每链 2 层 K-step**（16×16 是 8 条全独立），
  管道周期不变而可用于遮蔽 SrcC RAW 与 barrier convoy 偏斜的独立工作减半。
- ★★★ **2026-09-11 补上第二个、可预先算出来的机制：原子形状直接定 `ds_read / MFMA` 比值**
  （mxfp4 dense NT，campaign 20260910_041317 REPLAN；与上面的独立链解释互补，且这个**不用跑就能算**）：
  每 wave 输出块 128×128、原子 16×16 ⇒ 每个 k=128 步 A 片段 128/16=8 条 + B 片段 8 条 = **16 条
  `ds_read_b128`**，MFMA = 8×8 = **64 条** ⇒ **0.25 条/MFMA**，与 PMC 实测 `SQ_INSTS_LDS/SQ_INSTS_MFMA`
  = 0.2509 / 0.2579 逐位吻合。换成 32×32×64：A/B 各 4 片段 = 8 条读、MFMA = 4×4×2 = 32 条
  ⇒ **0.5 条/MFMA，正好翻倍**。按 methodology/03 的 `duty = 16/(16+c·n)`（occ=1 上 c≈8.5），
  这一项单独就够解释 12/12 慢 0.8-3.0%。
  ⇒ **通用判据：`ds_read/MFMA = (BM_wave/atom_M + BN_wave/atom_N) / (BM_wave·BN_wave/(atom_M·atom_N))`
  ——只有把原子的 M、N 一起放大（而不是用更方的原子换更少条数）才降这一项。** 立项前先算这个比值。
- ⇒ **scale 全族至此三向关闭**：VMEM 投递 +0.57%、指令数减半 −2%、释放 16 个常驻 scale VGPR（一相位共享
  单个 sa/sb，指令流逐条不变、VGPR 246→230）**也是 −2%**。曾记的「unscaled 探针 +2.9% ⇒ 指令本身仍开着」
  是 **DCE 复合效应**（连带消掉 scale load、地址链、preshuffle 消费者），不是杠杆；引用前必须重新推导。
- 附带的实现坑：`flydsl/expr/rocdl/__init__.py` **只手包了 `mfma_scale_f32_16x16x128_f8f6f4` 一个** scaled
  MFMA（签名 `(result_type, operands)`，见该文件 §49-50 别名 + §178-190 wrapper）；`..._32x32x64_f8f6f4`
  没有 wrapper，属性查找落到生成的 nanobind OpView 类，用 `(res_ty, [operands])` 调会报
  `TypeError: missing 8 required positional arguments`。正确写法：位置参数直调该类 +
  value 操作数过 `rocdl._unwrap_mfma_operand` + 取 `.result`（或照 §178-190 补一个同形 wrapper）。**形状轴的纯计时上界不需要正确
  layout**：把新形状喂旧 fragment（全部喂到，否则 ds_read 被 DCE、计时失真）、宽累加器在 epilogue 回切
  （vector shuffle 编出零指令），再用 ISA 直方图确认「除 MFMA 条数外逐项相同」即可，~1 小时而不是多小时的
  layout 经验重发现。

### grouped wgrad WL scaled 1.7x 慢的根因（占用率结论作废）
- grouped mxfp8 wgrad WL：scaled 811us（1.7x 慢，另记 817）vs unscaled 445us（反超 baseline 479）。
- ❌ 别再试：把它归因为 occ——4-wave scaled 计算天花板 447 ≈ unscaled，occ 结论作废。真凶=scale 投递机器。
- 两个叠加大头：
  - ① 每 phase **16 条 buffer_load_ubyte**（tiny byte-gather，吞吐/issue-bound；预取隐藏无用——去 load 才 807→448）。WL 全量记为 32 byte-load/iter。
  - ② **16 个分散 scale 目标 VGPR**（SCMIN 砍到 2 就 807→506）。
- 隔离探针：NOSCLOAD=448、SCMIN(16→2 VGPR)=506、全速地板 447（另记 448）。
- scale gap 随 k_iters **线性**（M=1024→1.5x、4096→1.9x、8192→1.94x）→ 是 per-K-iter 开销。

### scale_pack / opsel byte-pack：前提是「裸无预取」，生产不适用
- ❌ 别再试（生产 per-K）：GEMM K-loop 里砍 scale LOAD 数（scale_pack / opsel byte-pack）——探针只加载 k=0 scale 复用测天花板，scale-tax ≈ 0%（4 shape 全 ≤1% 甚至略慢）。
- WHY：现有流水已提前一拍预取 sa/sb（sa0n + s_setprio）把 scale i32 load 藏进 MFMA shadow。外部（AITER 文档 §9.4）的 scale_pack=4/opsel 前提是**没预取的裸状态**，本 codebase 不适用。
- WL(4-wave/occ=1) 上 scale_pack 曾被证伪（寄存器压力↑→spill），但那是 WL 的问题；WL 已整体放弃（手调 2500 行汇编不可维护），per-K(occ=2) 才是生产路径。
- pack2（2 个 E8M0 字节 —— E8M0 = MX 格式的 8-bit 指数、0 尾数 block-scale 编码 —— 合成 1 次 buffer_load_ushort，op_sel:[0,0,0]/[1,1,0] 选低/高字节，载入 32→16/iter）在 WL 吃回 54%（817→620）；但生产 per-K 已被预取藏住，无用。

### scale-128（dwordx4）无提速，WL_SC128 有索引 bug
- dense Down 上 operand:scale 字节比 = 32:1。
- scale-128（LDS-b128 dwordx4 与 VGPR-dwordx4 两条独立路径）perf-neutral：3097 vs 3099。
- 5 探针（无 g2s / 无 ds_read refill / 无两者 / 无 scale / WL_SC128）全 ≈ baseline 3053-3104 → 纯 MFMA-bound，去任何 memory 操作都不提速。
- ❌ WL_SC128 dwordx4 2-K-pack scale 实现完成但 SNR -1.3（确定性索引 bug 未调），速度 3097 vs SC_VGPR-dwordx2 3099 完全中性 → 默认保留 **SC_VGPR dwordx2**（正确同速）。

### ❌ 别再试（2026-07-29 实测，grouped mxfp8 NT 上第二次独立验证）：改 A/B-scale 的投递路径
本卡上面两节（「scale_pack/opsel 生产不适用」「scale-128 dwordx4 无提速」）的结论在 **grouped** NT
(fwd/dgrad) 上同样成立，且此前的 campaign 计划把它当成头号杠杆连发三轮——**先核对 ISA 再动手**：
- ★ **前提本身是假的**：计划书写「ISA 里 `buffer_load_dword`(4B 窄) 111 条就是 `ScaleS2R` 的 A-scale
  投递」。实测 `SA_TILES = BLOCK_M//64 = 4` ⇒ `ScaleS2R.load(vec_width=4)` **早就是
  `buffer_load_dwordx4`**；`kernel_grouped_mxfp8_nt_1` 的**主循环里窄 load 数 = 0**，那 69 条全在
  prologue 的 `_load_go` group-offset 扫描里（而 prologue 指令数已判负，见本卡下文 +0.07%）。
  ⇒ 「把 A-scale 加宽成 dwordx4」这条指令**等于什么都不做**。**引用 ISA 计数前先按 region 分段**
  （`methodology/03`），全 kernel 直方图会把 prologue 的税记到主循环头上。
- 三个投递路径变体，drift-immune 14-config bench 全部落在噪声内（base gm 1.08492/1.08459 两次）：
  | 变体 | gm | NT 八配置几何 |
  |---|---|---|
  | 预取距离 1 iter → 整个 pack-group（照搬 methodology/11 的 dense「大-K scale group 预取」）| 1.08433 | +0.17% |
  | 三条 scale dword 摊到 pack-group 的三个 K-iter（每 iter 恒 9 条 VMEM，而非 8/8/8/11）| 1.08483 | +0.08% |
- ★ **根因（新事实，解释了为什么预取距离没用）**：`wait_barrier` 每个 K-iter 收一次
  `s_waitcnt vmcnt(_NB_DRAIN)`，而 **scale 的 `buffer_load` 到 VGPR 和 g2s 的 `buffer_load...lds`
  共用同一个 vmcnt**。所以无论把 scale load 提前几个 iter 发，它都会被**本迭代的 drain 强制退役**
  ⇒ scale 预取距离的**上限恒为 1 个 K-iter**，加大距离在 ISA 上不可表达。要真正拉长只能抬 drain 的
  tail，而那正是 `pitfalls/04` 实测出 race 的动作 → **这条路封死，别再花轮次**。
- regime 佐证：`fwd down balanced` 每 WG 每 K-iter 从 global 搬 64KB，全核有效读 ≈10.5 TB/s
  （> HBM 峰值）⇒ 该核是 **L2/LLC 带宽 + 复用率**主导，不是 scale-load 延迟主导。下一批杠杆应指向
  「减字节 / 提 L2 复用」，不是 scale 投递。

## MXFP8 whole-loop 移植整体死路：occ=1 结构天花板+vendored 严禁

### 硬约束：BK256 LDS 超容 → 只能 BK128
- fp8 operand 2× 字节（vs fp4）→ **BK256 LDS=256KB ≫ 160KB cap**，只能 **BK128（n_sub=1）**。
- ❌ 别再试照搬 mxfp4 BK256 whole-loop：这是 fp8 whole-loop 无法照搬的**硬约束**，不是调参能绕的。

### WL occ=1 结构天花板（dense）
- dense mxfp8 **4-wave WL：64 acc/wave = 256 VGPR → 必须 AGPR acc → occ=1**；per-tensor occ=2。
- WL **非-scaled 也只到 3093 < per-tensor 3277（-6%）**：gap 是**内核结构(occ)差异，非 scaling**。scale 不是根因。
- ⚠️ **8-wave WL 早期诊断"死路"已被撤回**：Stage-0 初期（2026-06-23）曾判 8-wave WL 用 VGPR accs 无法 pin operand 做 fp8 2×b128 → 死路；但同日晚些时候明确撤回此结论——8-wave 仅 32 acc/wave，VGPR acc 放得下 + 留空间 pin fp8 operand，`occ=2 + 手写调度` 被重新评估为冲 98% 的**真正路径**，此后再无来源重新验证/判死 8-wave；WL 最终移出生产是因用户裁定 vendored 严禁（见下），并非 8-wave 被重新证伪。4-wave WL occ=1 是已验证的结构性天花板。
- ❌ **BM=128 死路**：per-tensor 4096² 也用 BM256，BM128 更慢。

### 补充（不同内核，勿与上文 dense WL 混淆）：分组 wgrad var-K whole-loop（grouped MXFP8 var-K wgrad 内核，与本卡 dense WL 不同）
- 以下数据来自**分组（grouped）GEMM 的 wgrad var-K** 4-wave bare-asm whole-loop 实验（`_build_grouped_mxfp8_wgrad_wl_kernel`，代码已于 2026-07-05 删除、从未提交），是与本卡 dense mxfp8 fwd WL **不同的内核/不同的 GEMM pass**，详见 methodology/12-mxfp8-grouped.md「grouped var-K wgrad」。
- WL-unscaled（无 scale 计算地板）**只在 K∈{4096,2048} 3 个 shape 快 5-7%**；**K=7168 / 2880×2880 反慢 51-59%**。
- 反常：**4096×7168（FLOP 更少）WL=1293 比 8192×4096 WL=911 还慢**；baseline 行为正常（857<946）。
- rocprof 定位 4096×7168 vs 8192×4096：每 tile **逐字节相同、无 WG 级失衡**，但 **MfmaUtil 29.9 vs 54.8 / VALUBusy 5.8 vs 10.6 全线砍半、MemStall≈0** → 气泡在 barrier/依赖：per-phase `s_barrier` + `vmcnt(0)/lgkmcnt(0)` drain 在 **tile-grid 16×28（N=28 非 2 次幂）** 触发调度病态。
- 裁决（分组 wgrad var-K WL）：**WL 放弃**，scale 投递税 > 那 5-7% 余量，occ=1 换结构无用。

### WL 计时假象与 store 选择
- ❌ **端到端计时假象**：把每 call `broadcast_to_wl_a`（activation scale → lane-contig 重排）算进去 → **假象 0.83**（比 per-K 还慢）；重排不该计入 kernel 时间，**broadcast 不计时才是 0.97**。
- ❌ **WL 用 buffer_store 掉 7-9%（0.95→0.91）**：WL(occ=1/4-wave/寄存器紧)**只认 copy-atom store**；per-K(occ=2 有余量)两者持平；**SGPR-pin 救不了**（不是 waterfall，是紧预算下 copy-atom 就是更优）。

### WLPAD：只证瓶颈，破坏正确性
- ❌ **WLPAD=16 给 +10.6% 但破坏正确性**（pad 不兼容协作式连续 G2S）。仅用来证实 identity LDS 的 **16-way bank 冲突（row stride 128B = bank period）** 是瓶颈；**正确修法是 swizzle**，不是 pad。（探针只证瓶颈、非可达值，见 `methodology/03` §subtractive/HALF「上界≠可达铁律」。）

### C++ 后端 raw E8M0 崩 + int32 scale 不可填充
- `test_gemm_fp8_mx_blockwise` 失败根因：quant 吐 **FlyDSL-preshuffled int32 scale**，FlyDSL 处理不了的 case（E5M2/HYBRID、K<256、K%128≠0、N%64≠0、fp16 out）fallback 到 C++ 后端要 **E8M0 → 崩**。
- HIPBLASLT 其实支持 **raw E8M0 MX**；放松 `can_handle` 后 FlyDSL 抢走 unaligned **必须自己全包**，否则 `preshuffle_ab_flydsl` 的 **N%64 崩（回归 180→252）**。
- K-tail：op preshuffle fast-path **必须 gate 到 K/M/N 全对齐（int32 scale 不可填充）** → unaligned 走 **raw 路径**；`execute` 把 **K 零填充到 tile**（padded operand=0 贡献0，**scale pad 127=1.0**）。

### vendored 严禁（用户裁定）
- ❌ **生产环境严禁 vendored dump / `mx_wholeloop` 文件夹存在**。
- WL 的 0.97 精髓 = `call_mxfp4_wholeloop` **2133 行手调 bare-asm 硬件循环（engine.py 914-3046）+ 2500+ 行依赖**，**没有小改能搬进 kernel 的版本** → 最终整个删除，**clean per-K ~0.91**。

## grouped MX gemm over-run 污染 + BLOCK_M=128 少启动块假象

- **deterministic 测试查不出正确性**: deterministic 测试只验 run-to-run 一致性,consistently-wrong 也会"过"。必须跑对拍参考(SNR / bit-exact vs Triton/HIP)才能暴露污染。新增 pipeline 一定要测 `balance=False`(非均衡分布):`balanced=True` 常全过,`unbalanced` 才崩(如 wgrad over-run `b_grad_snr` 崩到 ~6dB)。WHY: balanced 时每组 M 对齐 tile 边界,over-run 不越界;unbalanced 才让越界 tile 读到相邻组。

- **MX wgrad 布局 over-run 污染**: 根因是算子布局差异。
  - tensorwise wgrad 算子 = `[M_total, OUT]`(token 外维),per-group 扁平 SRD `num_records = mg*OUT` 能把 over-read 硬件钳 0。
  - 但 MX wgrad 算子 = `[OUT, M_total]`(token 内维),SRD 用整张量边界 `OUT*mt`,扁平 `num_records` 无法按 `token < m_end` 逐行钳 0 → over-run tile 读到下一组真实数据被 MMA 累加污染。
  - **修法**: body 调用外包运行时守卫 `if k_abs < k_iters:`(`k_iters` 全 WG 一致,分支/barrier 均匀安全),`swap` 留守卫外保持 ping-pong 身份。

- **❌ 别再试 BLOCK_M=128 —— 是少启动块假象**: grouped MX gemm config sweep 里 `BLOCK_M=128` 显示 +9~35% / 反超 TW 是坑。
  - 机制: sweep 时 host `grid_upper` 写死 `/256`,`bm=128` 需 2× 块 → 只启动一半只算一半 tile → 假性变快。
  - 把 `grid_upper` 泛化成 `ceildiv(M_pad, bm) + G` 后,gemm-only `bm=128` 真实 **1.55x 慢**(1666 vs TW 2576)。
  - WHY: 256×256 tile 的 MFMA operand 复用在 compute-bound 形状更优。
  - **教训重申**: sweep/探针任何变快必须先确认 grid/tile 覆盖完整,或跑数值对拍。

- **❌ bm=128 需 nt_a=2 preshuffle 布局(确认死路已回退)**: (`nt` = preshuffle 每 record 的子块数) `bm=128` 还需 A-scale preshuffle 发 `nt_a=2` 布局(32-row group、每 record 2 子块),否则 `ScaleS2R(n_tiles=2)` 只读一半 → 数值全错。已把 `_emit_lds_repack(nt=...)` / preshuffle / workspace / grid 全参数化打通验证,确认 `bm=128` 死路后全回退,dense/wgrad 保持 `nt=4`。

- **输入 band 收紧中性(依赖搬家没净减)**: grouped qa 主 kern 输入 buffer band 从 `[in_rebase, total_M)` 收成 `[in_rebase, RIE)` 靠 `num_records` HW-drop 删 `(grow < real_end)` select,实测中性。WHY: 依赖只是从 RE-load 换成 RIE-load,没净减少;grouped M-remap 的 SRD 本就要 runtime 组信息 gate 住 load,是结构性的。

## grouped MX quant 占 fwd 绝对 35-46%（不是 7%）但 e2e 中性非杠杆

### 绝对占比纠错（核心）
- **grouped MX quant 占 fwd 绝对时间 35-46%（不是 7%!）**。之前的 "7%" 是 **MX/TW 比值差（差分口径）**，不是绝对占比。
- 实测（M=4096，绝对拆分）：
  | shape | 拆分 | quant 占比 |
  |---|---|---|
  | 4096×7168 | qa311 + qb282 / gemm910 | **39%** |
  | 8192×4096 | — | **35%** |
  | 4096×4096 | — | **38%** |
  | 2880×2880 | — | **42%** |
- WHY：要压绝对 fwd 时间追 dense，quant 是大杠杆；**"96.5% 在 gemm" 只对 MX/TW 比值差成立**，不能当绝对占比用。

### ❌ 别再试：grouped FlyDSL dual-cast quant kernel（e2e 中性，非杠杆）
- kernel 已写完 + bit-exact，但 **e2e 中性**：quant 全优化后 fwd 只 **1.2→1.07x**、**bwd 几乎没动**。
- 真相：**bwd 缺口 90% 在 gemm（dgrad+wgrad）、fwd 缺口 60% 在 gemm**。剩余全是 grouped MX gemm 的 **occ=1 结构上限**。
- 结论：下一刀 ROI 是 **gemm（尤其 bwd）**，不是 quant。绝对占比大 ≠ 优化 quant 能撬动 e2e，因为 gemm 侧缺口更大且是结构上限。

### ✅ b-scale 布局担忧多虑（正确性已过门）
- 之前担心 "b-scale 必须匹配 gemm 的 `use_2d_block` 布局" 是多虑：**raw 1×32 E8M0 scale 就能被 gemm 正确消费**（e2e fwd **SNR 28.1dB** 过门）。
- batched FLY 权重 quant 直接产 raw 1×32 E8M0 无碍。

### qa kernel 已高度调优，离峰 gap 是结构性
- qa kernel PMC：**occ 75.4% / MeanOcc 24.16 waves/CU / WriteSize 306MB≈理论 283（放大仅 8%）/ MemStall 1.6% / VALUBusy 30%**。
- 解读：占用率高 → **非 occ-limited**；写高效 → **无半-cache-line 写放大**。
- ❌ 别再试低垂果实（提 occ / 写合并 / scale_pack）：**全证伪，不存在**（scale_pack 同 §scale_pack / opsel byte-pack 结论，已被预取藏住）。
- 离峰 gap 根因：**phased load→barrier→compute→barrier→store 结构**（计算相时 DRAM 空闲，**dense 同样只 ~60% 峰**）。要抬只能**跨-tile 软流水重叠访存与计算**，是大改高风险。
- grouped 独有的 **1.30x（+37us）** 是 M-remap SRD-gating + col-padding + prologue，**全结构性**。

### ❌ 别再试：fwd/dgrad 三条重写路（全证伪）
- grouped MX fwd/dgrad 慢 4-12%，无低风险可落地优化：
  - **scale_pack**：被预取隐藏，scale-tax≈0（见上 §scale_pack / opsel byte-pack）。
  - **BLOCK_M=128**：少启动块假象，真实 **1.55x 慢**（见上 §BLOCK_M=128 少启动块假象，完整机制+数字）。
  - **band-swizzle**：`group_n>0` / `gm=8` 略慢或崩 **1.5x**。
- **preshuffle 只占 3.5%（15.5us / ~3.4 TB/s）非瓶颈**。
- ~~唯一有效 lever 是调度局部性（xcd/gm/gn），已被 autotune 吃掉~~ → **⚠️ 已证伪，见下条**。

### ⚠️ 勘误（2026-07-29 实测）：grouped MX fwd/dgrad 的真 lever 是 per-tile O(G) 索引解码，不是调度局部性
- 上条「唯一有效 lever 是调度局部性」**不成立**。mxfp8 NT 核（`_build_grouped_mxfp8_nt_kernel`）每
  tile 有两处 **O(G)=32** 的向量 group-find 扫描（入口 `total_tiles` + 每 tile 的组解码），实测是隐形大税：
  | 指标 | O(G) 原状 | O(1) 探针 | tw NT 参考 |
  |---|---|---|---|
  | SQ_INSTS_VALU | 225.3M | **109.8M** | 138.1M（**反超参考**）|
  | SQ_INSTS_VMEM | 41.7M | 32.2M | 19.9M |
  | buffer_load（ISA/tile） | 307 | 213 | 228 |
  | v_cmp + v_cndmask | 299 | 74 | ~0 |
  | `private_seg_size` | **16 B（有 spill）** | **0** | 0 |
  | s_barrier | 185 | 185（未动）| 123 |
- 生产落地（单扫描 + SGPR 前缀单调比较 + `_tree_add_i32` log-G 归约 + 两张前缀停 LDS、per-tile 各一次
  `ds_read`）实测 fwd/dgrad 八配置 **+1.3~5.4%**；subtractive 探针上界 +5.3~19.1%（**上界≠可达**）。
- ★ 本卡 §grouped fp8-tensorwise NT 里的「✅ 已交付杠杆：monotonic group-carry（commit 4e15c6ce，+7.7~9.6%）」
  就是同一修复的 tensorwise 版；**mxfp8 NT 长期没拿到它**。⇒ 通用规律：**MFMA 指令数与参考逐条相等时，
  首要嫌疑是索引/解码开销**，不是调度旋钮。
- ★★ **三度实证(2026-08,mxfp4 grouped NT,+9.4% 单笔最大)**:同一个坑在 mxfp4 NT 上还在。ISA 实测
  prologue = 66 条 `buffer_load_dword` + **64 深有符号除法链** + 255 条 `v_accvgpr_mov`(寄存器压力溢出);
  换成 mxfp8 已有的 lane-resident wave scan 后 prologue **2153→761 条、SGPR 97→58**、per-tile 固定开销
  12.46→10.47 µs,而稳态 `P` 一点没动。**新写的 grouped 核别再从零写 O(G) 扫描,直接抄 lane-resident 版。**
- ★★ **诊断法:两个看似无关的症状可能同源,先找能同时解释两者的根因**。同一场 campaign 开局有两条独立
  线索——①「wgrad 已 1.6× 而 NT 只有 1.1×,明明共用同一份 whole-loop 计算体」②「窄 N 的 down 恒为最差格」。
  两条**是同一个根因**:wgrad 的 `TILES_PER_GROUP` 编译期已知 ⇒ prologue 是 O(1),NT 不是;而 down 的
  n-block 少、每 tile 摊到的固定开销占比更高,所以同一份 O(G) 税在窄 N 上更痛。修完 min 直接从 down
  迁到 `fwd gate_up heavy`。⇒ **列症状时把"同一份代码却表现不同"的对照组写进 goal,比泛泛"提升性能"值钱得多。**
- ★ 补充定量（2026-07-29,drift-immune 交替 A/B）：在唯一的弱 shape(`down`)上把 `(gm,xcd,gn)` 共 **14 个 cfg**
  测完,**没有一个**能在 balanced+skew 双点都赢 base 1.5%;`gn>0`(2D N-band)一律差 **1.6~7.2%**;`gm=8` 虽在
  down 双点快 0.6~1.0%,但在 `fwd gate_up` 慢 **2.97%** → 改全局 base 净负,per-shape 硬编码 = 本卡判负的
  overfit。⇒ **调度旋钮在该核已无可采纳增益,真杠杆是索引解码 / scale 投递 / 边界半区。别再扫。**
- 配套坑（2026-07-29 实测）：① `BufferCopy128b`/dwordx4 加宽 `group_offs` 取数**净负 −1.4%**（每次物化
  4 个 VGPR 只用 2 个）；② gfx950 每 wave ~102 SGPR，2×(G+1)=66 个活跃标量就溢出 → 前缀必须停 LDS；
  ③ 别把 A-scale slab 前缀简化成 `m_start_pad>>6`（组起点 32 对齐而非 64 对齐时静默数值错，det 与
  bench 正确性门都抓不到）。

### ⚠️ 勘误（2026-07-29 实测）：grouped **wgrad** 的 `pack=4` 是真杠杆（+5~9%），但有隐式 512 对齐契约
- 本卡 §scale_pack / opsel byte-pack 判负的是 **dense per-K 核**（sa/sb 已提前一拍预取，scale-tax≈0）。
  **grouped 变长-K wgrad 不适用该结论**：其 scale load 没被藏住，`pack=1` 时 `SQ_INSTS_VMEM`/WG **3947**
  vs tw 2944。改 `pack=4`（op_sel 选字节）实测 per-WG VMEM **3947→3387（−14.2%）**、总量 18.19M→15.61M
  （tw 14.80M，基本追平）、MFMA/VALU 不变、SALU +19%（**独立标量端口，不计价**）→ wgrad 六配置
  **+4.7~9.3%**，geomean +2.7%。教科书式「窄标量 load 换 SALU 解码」。
- ★★ **契约雷区**：K 维打包默认要求每组收缩起点是 `pack×128=512` 的倍数；非对齐时 SNR 崩到
  **4.09dB / −2.44dB**，而 **det=True 与 preshuffle=False==full 逐字节一致照样成立** → 这两个门抓不到
  （**确定性门 ≠ 正确性门**）。真实 MoE 的 per-expert token 数不会 512 对齐。
- ✅ 正解（已落地）：**每组从自己的收缩起点独立打包**，闭式基址 `kp0(ks0,g) = ks0//pack + g`、
  行组步长 `k128p = k128//pack + G`（`+g/+G` 就是每组可能留下的那一个残缺 dword；由
  `floor(a/p)+ceil(n/p) <= floor((a+n)/p)+1` 保证各组区域不重叠）。对齐负担全落在**不计时的**
  preshuffle 里，主核只多一次整数加 —— 比写 `[G+1]` 基址表更省（计时区零额外 load）。
  实测 16 个形态（含「128 对齐非 512」「单个奇起点」「空组首/中/尾」「G=32 ragged」×
  {256 对齐 OUT_M, 半边界 OUT_M}）全部 **55.56~55.64dB**，且 `pack=1` 与 `pack=4` 输出**逐字节相同**。

### ✅ 已交付杠杆（2026-07-29 实测）：输出边界半 tile 的象限跳过（wgrad M+N 侧、NT N 侧）
- 适用判据（编译期）：`OUT % BLOCK != 0 and OUT % BLOCK <= BLOCK/2` → 末块的后半区**全是 padding**，
  该 tile 的 acc01/acc10/acc11 象限 MFMA 与其 LDS 读是纯废功。gpt-oss 的 2944 / 5760 都是 `x.5×256`，
  M 与 N 两侧同时命中。
- 落地形态：把 tile 体包成取 `quads=(qm,qn)` 的闭包，用 `_readfirstlane_i32(block_m/n)` 做**标量
  (wave-uniform) 分支**选变体（tw 的 `_HALF_BND` 同一惯用法）。★ **g2s / s_barrier / vmcnt 等待序列在
  各变体间必须逐条相同** —— 只删 ds_read + MFMA，不碰同步，就完全避开 pitfalls/04 的 race 面。
- 纸面 vs 实测（**上界≠可达**再一次）：wgrad down 象限数 529/576 → 省 8.16%，gate_up 1035/1104 → 6.25%；
  **实测 wgrad 六配置 +1.3~4.1%**（down 侧 +3.6~4.1%、gate_up +1.3~3.1%）、geomean +1.27%，兑现率 ~50%
  （省了 MFMA 但 barrier / g2s / LDS 往返照旧）。NT 只 N 侧（2.17~4.35% 纸面）→ min_speedup +1.9%。
- 落地注意：被跳过的象限**照旧无条件 store**（保持零初始化，行/列全落在 `StoreC` 的 clamp 区外），
  比按象限分支 store 更省事且不引入前端作用域坑（见 pitfalls/07 §scf.for 体抽取丢外层名字）。
- ★ **续刀二（2026-07-29 实测）：空相位的 rendezvous 对也能跟着象限一起删,但只在 NT 核有收益**。
  被跳过的象限留下的那对 pre/post-MFMA barrier 夹着一个**空相位**,跟着象限一起门掉(`if const_expr(_full)`
  / `if qn == 2`)是安全的 —— 该象限在这个变体里根本不读 LDS,守 WAR 的是 c00 之后那个 barrier,且整个 WG
  取同一变体。NT 半-N 体 8→5 barrier/K-iter:`fwd/dgrad down balanced` +1.0~1.5%、gm +0.13%(**保留**)。
  同一手术搬到 wgrad(23/144 tile 命中 (1,1)/(1,2)/(2,1) 三个变体)**判负**:gm 1.0850 vs base 1.08492
  (0.00%),wgrad 六配置几何 1.1370 vs 1.1393,`gate_up heavy` 1.320 vs 三次 base 的 1.342/1.347/1.353。
  ⇒ **wgrad 的边界 tile 只有半/四分之一的 MFMA,本来就提前收工、不在关键路径上**;NT 是一 tile 一 WG、
  半-N tile 占 down 的 1/12 且 feed-bound,才吃得到。**判据:先问被优化的 tile 在不在 makespan 上。**
  ★ **修正(2026-08-03,mxfp4 grouped 实测)**:「wgrad 判负」不能按算子写死。把半-N 体搬到
  **mxfp4** wgrad 核是**小幅正收益**(六配置 5/6 上升,gm 两次复测各 +0.2~0.3%,down 三项
  0.617/0.603/0.599ms → 0.590/0.592/0.597ms)。区别在**被删池的发射位次与消费距离**:mxfp4 的
  wgrad 与 NT 共用同一份 whole-loop 体、BR 是**最后发射**的 g2s 组(删了不动前面的 drain);
  mxfp8 wgrad 判负的池是**距离-1 且最先发射**的 A1。⇒ 判据改成按「被删池的距离 + 发射位次」分,
  不是按算子(wgrad vs NT)分。
- ★★ **修正二(2026-08-11, mxfp4 grouped var-K wgrad, G=32 / 每组 M=4096 倾斜 deploy):上面「wgrad 的边界
  tile 本来就提前收工、不在关键路径上」是 **G=4 grid-饱和 regime** 的结论,在 deploy regime 下反号。**
  判据「先问被优化的 tile 在不在 makespan 上」是对的,但要用**运行期的满价 tile 数 vs CU 数**去算,而不是
  按算子写死:gpt-oss gate_up wgrad 每组 `12×23=276` tile > 256 CU ⇒ 胖组被逼出两代派发,而边界 tile 恰恰是
  唯一能变便宜的一批,半价之后满价当量降到 259 ⇒ 边界 tile **就在** makespan 上。
  - **M 侧半 tile 与 N 侧不是同一个手术**:一个 tile 的两个 M 半区落在**不同的 wave**(`wave_m`)上,所以
    「让 `wave_m==1` 空转」根本不缩短 tile —— 留下的那半个 wave 照样走完整条收缩维。正确形态是**四个 wave
    全部保留活行、改分列**:`wave_m==1` 把自己的 A fragment / A scale / L 操作数 / L scale / C 列基址整体
    指到 R 半列,于是整块 tile 走的还是**半-N 已经在发的那个 R-dropped 体**,不新增任何 g2s / s_barrier /
    vmcnt 序列(line 360 的规程照守)。
  - 唯一要动 emit 的地方:半-N 体里「BR 的 g2s 折到 BL 行」这条**只在 R 列是 padding 时才对**。加一个
    `half_g2s` 开关关掉它,改由 host 侧对**末列 N** 的 tile 选 `soff_br := soff_bl`(纯地址替换,指令数不变),
    半-M tile 则保留真实 BR 行(它是 `wave_m==1` 的 L 操作数)。另可顺手在 R-dropped 体的 end-drain 里跳过
    BR 片段的 refill `ds_read`(该变体永不读它们),只删 ds_read 不碰同步。
  - 实测(同卡态,5 次 vs 3 次中位,ref 侧按 long-M cfg 逐位不变):gm **98.50 → 99.10**,min 89.94 → 90.36,
    护栏 99.93 → 100.44。deploy 侧绝对 TF:down wgrad canon/mod/heavy **3921.6/3800.3/3840.3 → 4043.4/3872.6/
    3903.1**(+3.1/+1.9/+1.6%),gate_up wgrad **4078.2/3963.6/3929.4 → 4096.2/4033.5/3956.8**(+0.4/+1.8/+0.7%);
    同期 ref 4367.3→4373.6 与 3684.4→3683.0(不动)⇒ 增益全在 deploy 侧。
  - **兑现率再确认 ~40~50%**:纸面 M 侧省 `22/276 × 50% ≈ 4%` 的 MFMA+store,实测六格 +0.4~3.1%(中位 +1.7%),
    与 2026-07-29 N 侧那次的 ~50% 同量级 —— barrier / g2s / LDS 往返照旧,这一半是拿不到的。
  - ❌ 配套判负,别再走:**试图把半价 tile「摆到更好的位置」这条路**。(a) band 内改成 N 优先遍历、让 22 个
    半-M tile 从只落在 XCD 残差 3/7 变成铺满 8 个残差(每 XCD 负载 max/mean 从 1.050 降下来),实测 gm
    **98.03 vs 99.07**;(b) 把边界 tile 排到组尾去顶溢出位,gm **-0.8**。两条都毁 L2 band 局部性,代价大于
    调度收益。同轮复扫 `group_m=6`(B 重读 3→2 次,少读 377 MB)= gm **97.75**,再次印证 R5 那条
    「别用 DRAM 读字节给本核做正当性论证」。
- ★ **续刀(2026-07-29 实测)：上面「vmcnt 序列必须逐条相同」是保守起手,不是终点**。半区的 g2s 也能删,
  但**只在 NT 核成立**：NT 半-N 变体删掉 b1 半区的 g2s + drain `2*NA+NB`(6)→`2*NA`(4),`fwd/dgrad down
  balanced` 0.985/0.988 → **1.005/1.006**、gm +0.6%、**min 0.985→1.003（14/14 过验收线）**;同一手术在
  **wgrad 核净负 −0.09% gm（六配置持平~−1.6%）已回滚** —— wgrad 的 A1 是 stage `k+1` 的**距离-1池且最先
  发射**,删它必须同步收 drain,省下的 DMA 被更早的等待吃回去。**先看被删池的距离与发射位次,再决定要不要删。**

### ❌ 别再试（2026-07-29 实测）：mxfp8 grouped NT 的 C-store 缓存旋钮与 barrier 削减
- ❌ **plain 标量 C-store 加 `cstore_aux=1`（非临时/write-once 提示）**：8/8 个 NT 配置一致变慢
  **−0.2~−1.7%**（gm −0.46%），同 run 的 wgrad 六配置不动 → 系统性判负而非噪声。
  根因：mxfp8 的 plain store 是**每 lane 2 字节**的 `buffer_store_short`；非临时提示让每个 2B 写绕过
  L2 合并 → partial-line 事务暴增。**该旋钮只在 128b 向量化 store 下才成立**（tw 正是与 CShuffle 同用）。
- ❌ **`store_cshuffle=True` + `cstore_aux=1` 一起开**（照搬 tw 的 `_NT_PERSIST_BIGN=(16,32,True,1)`）：
  gm **−1.08%**，`fwd/dgrad down heavy` 各 −2.7%。→ 源码注释 "NET-NEGATIVE for mxfp8" 在**加了 nt 提示后
  重新成立**；单开 store_cshuffle 仍是中性（见本卡上文）。mxfp8 NT 不是 store-bound，两条都别再花轮次。
- ❌❌ **删掉 NT 主环里 MFMA *之前* 的 rendezvous barrier（8/K-iter → 5/K-iter，正好对齐 tw 的 123）**：
  gm **1.0657→1.0304（−3.3%）**，8 个 NT 配置全跌 **−2.6~−6.9%**（SNR/det 仍过）。
  - 正确性推理是对的（守 LDS WAR 的是 MFMA **之后**那个 barrier —— 此刻各 wave 的 s2r 读已被 MFMA 的
    `lgkmcnt` 等待排空；MFMA 之前那个不提供任何额外顺序），但**pre-MFMA rendezvous 是承重的调度装置**：
    去掉后各 wave 相位漂移，MFMA 突发不再对齐，发射/LDS 争用的损失远大于省下的同步。
  - ★ **勘误 methodology/03 的「barrier 数 ≤ 参考」对标判据**：barrier 计数差**不能**直接当成可回收的
    同步税。mx NT 185 vs tw NT 123 的 62 个差额里，至少 63 个（21 iter × 3）是**负收益可删项**。
    要动 barrier，先问它是「顺序约束」还是「相位对齐装置」——后者删了就掉速。
- ❌❌ **合并 MFMA 相位来减 barrier（4 个象限相位 → 2 个,每 K-iter 8 barrier → 4）**：gm **−1.6%**
  （1.0610→1.0442）,8 个 NT 配置全跌 **2~5%**。这是上一条的**第二次独立验证**——换一种减法（不是删 barrier
  而是合并相位）同样掉速 ⇒ 「一象限一 barrier 对」的粒度本身就是**波间相位对齐装置**,粗化后两个 wave 同时
  争同一资源、ILP 下降。**mxfp8 NT 的 barrier 结构别再动,除了连相位语义一起重设计。**
- ❌ **利用 fwd 里 A/B 同张量把 prologue 的 group-offset 取数减半**（`go_out_div`/`go_pad_div` 别名检测,
  省掉 33×2 次窄 `buffer_load` 中的一半）：gm 1.0603→**1.0610（+0.07%,噪声）**。⇒ NT prologue 的取数条数
  **不在关键路径上**（该核是 latency/同步 bound,不是 VMEM 发射端 bound）;而且它是**只在 bench 的 fwd 形态
  成立的 bench-specific 优化**,已整条移除。**别再往 prologue 指令数上花轮次。**

## MXFP8 B-comb sizing / col 已转置 [K,M] / scale_pack 来源歧义穿线

MXFP8 authoring 布局 / 穿线卡。以下每条都带根因（WHY），改动涉及 C++/kernel/autotune 三层，改一处漏一处后半读 0 或读错转置。

- **[B-comb buffer sizing]** B-comb buffer 大小必须用 `cdiv(dim,256)*4` 组（grp 是块跨步，不是 `dim//64`）。
  - WHY：旧公式 `(M/64)*K128p*64*4` 漏掉 partial 256-block 的 grp → 后半读 0。
  - 正确 C++：`dwords = cdiv(M,256)*4*K128p*64*4`，断言从 `%256` 放宽到 `%64`。
  - **两文件都要改**：`quantization.cpp` 与 `quantization_meta.cpp`（meta 漏改 → shape 推断与实际 alloc 不一致）。
  - kernel 端：`nbytes = ((dim+255)//256)*4*K128*64*4*4`。

- **[col 操作数是已转置 [K,M]]** MXFP8 C++ dual-cast 的 col 输出是 `[K,M]`（已转置，**不是** `[M,K]`）。
  - 所以 `AtQd` 必须是 `[K,M]`。
  - WHY 验证要用 `M≠K` 的 shape 才看得出（M==K 时转置错误被掩盖）。
  - bwd grad_b：`NT(go_col[N,M], at[K,M]) → [N,K]`；`trans_b=True` 时 kernel 把 `at` 当 `[K,M]` 读、输出 `[N,K]`。

- **[scale_pack pack 来源歧义]** kernel 自己分不清 scale 是 packed 还是 broadcast → 由 **caller 按来源指定** `scale_pack`。
  - int32 透传（C++ quant 吐的 packed）→ `mxfp8_scale_pack(K)`。
  - raw 路径（preshuffle 是 broadcast）→ `1`。
  - 穿线链：`gemm_mxfp8_flydsl_kernel(..., scale_pack=1)` → `autotune/_get_mxfp8_launch`（cache key 含 scale_pack）；**无 env 开关**。
  - scale 原语（`MfmaScale16x16x128` / `ScaleS2R` / `ScaleBComb` / `mxfp8_scale_pack`）放共用 `gemm_helper.py`，勿在 kernel 重复。

- **[raw E8M0 K-packing 冲突]** ❌ 别再试：在 raw E8M0 上做 K-packing 免 preshuffle。
  - raw E8M0 布局 `[dim, m//32]` 里同一 microblock 的连续 K-block 是 stride-4（列 = `k*4+mb`）。
  - failing mechanism：dword/dwordx 无法把多 K-block 打进一个 load，跨 tile 又 row-separated → K-packing 必须 preshuffle。
  - 这与 WL 刻意选的 raw E8M0 免 preshuffle 设计**直接冲突**（要 K-packing 就得放弃免 preshuffle）。

- **[dead code review]** Bugbot 按无 dead code 标准要删：
  - `peek_mxfp8_cfg`（从未调用）。
  - `gemm_helper.as_i8_flat`（1D flat 版无调用者）。
  - ⚠️ `as_i8_flat` ≠ tensorwise 的 `_as_i8_flat`（后者保持 2D contiguous 且**在用**，不可互换）。

## MXFP8 死路清单（勿重试）：asm-AGPR MFMA / scale via LDS / host 重排 / 大 BM 不合并 quant

MXFP8 死路速查表——以下方向均已实测判负，勿重试。

| 死路 | 机制/根因 | 实测数字 |
|---|---|---|
| ❌ asm-AGPR MFMA inline_asm | acc-clobber 墙：inline_asm 无法正确声明 accumulator clobber | SNR garbage（结果全错） |
| ❌ scale via LDS 双缓冲 | vmcnt 计数失准 → 流水崩 | 慢 6× |
| ❌ host 侧任何缓存/重排 | 用户硬否决 | —（禁止） |
| ❌ quant 128B 整 cache line 写 (旧 v2, BM=128) | 各 lane 写满 128B，无跨-lane 合并 | 慢 0.58–0.98× |
| ❌ quant wave 跨-lane 合并转置写 (旧 v3) | 双趟 scale 也暂存 LDS（错误做法） | 慢 0.42–0.61× |
| ❌ BLOCK_K=256 移植 MoE grouped GEMM | MoE 是 BW-bound，BLOCK_K=256 回退 | 回退变慢 |
| ❌❌ grouped NT/wgrad **persistent grid**（grid 截到 num_cu + stride loop） | 见下方"两次独立判负" | NT 0.70–0.81×；wgrad +11~18% 更差 |

### 2026-08 mx8tw campaign 新增的死路(gpt-oss grouped mxfp8,18 格 e2e regime)
打法与仍然活着的杠杆见 methodology/12 §「追一个具体 kernel 当标尺」;下面只列判负的。

| 动作 | 根因 | 代价 |
|---|---|---|
| ❌❌ **把 N 列区间切成第二个 dispatch**(尾块 512×128) | **不是** launch 开销(实测只 1.9 µs = 0.23%),是尾块丢掉与相邻 n-block 的 **A-slab L2 共享**、要重新从 HBM 读 A | gm −2.9% / min −4.6% |
| ❌❌ **单 kernel 内塞第二份重 body**(wide tile / paired tail / 2-tile 循环) | 第二份 body 实例把 RA 挤爆,spill 落点是 **MFMA 流里的累加器** fragment;单份 244/0、两份**同样**的 body 仍 244/0、一份 full+一份 paired = 256/**spill 23** | −19…−26% |
| ❌❌ **删稳态 barrier**(每迭代 c11 之后那个) | 该段还有 g2s 在飞 ⇒ 承重的相位对齐装置 | gm −6.1% / min −8.7% |
| ❌❌ **把一组 g2s 提前一个相位** | g2s 发射点与 barrier 同属相位装置,两组 LDS-DMA 挤进同一相位会抢该相位的 ds_read 读口 | gm −1.3% / min −1.6% |
| ❌❌ **epilogue 相位铺开 / 按 fragment 组交错 cvt+store** | 交付态 ISA 已经是 **1:1 的 CVT/ST 交错**,`sched_group_barrier` 只是把编译器排好的序切成 4 段插串行点;相位铺开则 +24 VGPR spill | −0.68% gm / −2.5% min;铺开 −19% |
| ❌ **band-cyclic XCD deal / `group_m=16` 宽 band** | 见下方「符号翻转」 | 前者六分布一律负;后者 bench −1.2% |
| ❌ **C-store cache policy `aux ∈ {0,2,3,16,18}`** | 冷 bench 与热探针两个 regime 都是平的(±0.5% 且符号不一致)。⚠ 源码注释「sc1 costs 2.5%」**不可复现** | 平 |
| ❌ **wave-phase barrier stagger ∈ {0,1,2,3}** | 四臂全在 0.3% 内 | 平 |
| ❌ **派发轮次量化 / LPT 排序** | group allocator 以 **512 行**为单位 ⇒ m-block 恒 512 ⇒ `total_tiles = 512×nb`,6144=24×256、11776=46×256,**每一轮都是满的** | 在计分形状上**恒为 0**(⚠ **限 NT**,见下方修订) |
| ❌ **LLVM 调度 hint**(`max-memory-clause`/`post-misched`/`lsr-drop-solution`) | 本核已被手写 `s_setprio`+`sched_barrier`+显式 waitcnt 钉死,策略空转:24 采样只 +0.06%,ISA 4975 条里**只差 1 条** | 噪声(**同一份 whole-loop 体,NT 与 wgrad 都算判过**) |
| ❌ **加宽 C store 请求宽度**(pack_cols=4 / 128 B 行 store) | PMC 铁证两跳都已 64 B 满宽,加宽只减指令数不减事务数;还会毁掉半-N 体 | 纸面否决 |

★ **判 LLVM 调度 hint 这类「零源码风险」的改动:先 diff ISA 再跑 bench。** pitfalls/13 记的
「换调度策略 +0.6~0.7%」只对**没有**手写 setprio/sched_barrier/显式 waitcnt 的 kernel 成立。
⚠ **别被 goal-plan 骗去重走**:多份 campaign 的 goal 计划仍把「换 LLVM 调度策略」写成「本核从未试过,
预期 +0.6~0.7%」。KB 与 goal 冲突时以本行为准 —— mxfp4 grouped 的 NT 与 wgrad 共用同一份 whole-loop
bare-asm 体,这 24 采样把两条路径一起判掉了,不需要再开一轮。

★ **修订(2026-08-11, mxfp4 grouped var-K wgrad, G=32 倾斜 deploy):上表「派发轮次量化 / LPT 排序恒为 0」
只在 NT 成立,搬到变 K wgrad 是错的。** 该行的论证链是「m-block 恒 512 ⇒ 每一代派发都是满的」,它有两个
NT 专属前提:(a) tile 总数是 256 的整数倍,(b) **每个 tile 等价**。变 K wgrad 两条都不成立:tile 数是
`G×N_BLOCKS_M×N_BLOCKS_N`(gpt-oss gate_up 8832 = 34.5×256,不是整代),而且**一个 tile 的开销正比于它自己
那一组的 token 数**,所以"轮次满不满"根本不是判据,负载均衡才是。实测:按运行期组表判「尾重」并整条反向
遍历 group-major tile 流(升序表如 canon 把最胖的组从尾巴换到前缀),canon 两格 deploy 侧
+1.65% / +2.3% TF,同期 ref 反向动 −0.5%,moderate/heavy/balanced 因前重或半和相等而逐位不变。
⇒ 判据:**先问 tile 是不是等价的**。等价(NT)则派发序无所谓;不等价(变 K wgrad / ragged MoE)则 LPT 是活
lever,而正确形态是"按组表在正/反两个方向里选",不是无条件重排(无条件反序实测 gm 93.04:canon +3.4~6.0%
但 moderate/heavy −7% / −19.8%,因为降序表的 group-major 序**本来**就是 LPT 序)。

### ❌❌ persistent grid：两次独立判负，第二次已排除「per-tile 解码成本」这个混淆项
- 一测（2026-07 round 10，NT）：`num_cu=256`，K=2944 慢 **25.6%**、K=5760 慢 **37.1%**。当时的怀疑是"persistent 把 O(G) 入口扫描从每 tile 一次降到每 CU 一次，应该赚"，结果反向。
- 二测（同 campaign，per-tile 解码已换成 lane-resident 表 ⇒ 每 tile 只剩 1 条 ballot + 5 条 `v_readlane`）：`_probe_nt_persist.py` 单进程交织，3 shape × 2 分布 × 3 swizzle cfg，persistent **全部 0.70–0.81×**，与一测**几乎逐点相同**。
- ⇒ **结论从"入口扫描不是 per-tile 固定成本的大头"硬化为"persistent 的亏损与解码成本无关"**。该核 occ=1（LDS 128 KB/WG），非 persistent 时 CU 空出来立刻换下一个 WG，persistent 并不能多重叠任何东西，却额外背上 `scf.for` tile 循环的活跃值与流水重填。**两个核都别再走截 grid 这条路去摊固定成本。**
- ⚠ **条件收窄(2026-08-10, fp8-tensorwise TN var-K wgrad `grouped_gemm_fp8_kernel.py`)**:上面四次判负的共同前提是
  **截 grid 会新引入 tile 循环**(ISA 每次都指向同一处 spill 23–25)。该 wgrad 核的 walk 循环是**无条件存在**的
  (`for d in range(pid, live, grid_dim.x)`,一 WG 一 tile 时只是循环体跑一次),截 grid 只改 host 侧 grid,
  **ISA 与输出逐字节相同(实测 ndiff=0)**,所以那条根因不迁移。但**结论仍然是别用**,换了个原因:
  生产路径(每种几何各自 race 一遍 autotune)A/B 实测 **−0.03% / −0.24%**(两次,平),而**同几何两臂之间的
  噪声底就有 0.60%**。原因是 autotune 的 `group_m`/`half_bnd` 轴会把这份余量吃掉——one-WG-per-tile 时
  race 选深 band + 精简边界体,resident 时选浅 band,两者落点相同。
  ★ **踩证**:先前"down grid=NCU +1.8~2.7%"的读数是**冻结配置**测的(配置在 resident 几何下 race 出来,再拿去跑
  one-per-tile 臂)。**凡是 A/B 两臂共享一份 autotune 结果的探针都要先问:这份配置是在哪个臂下 race 出来的。**
  中间值的 grid(NCU 与 TOTAL 之间)一律更差(512/768/2304/2208/2944/4416 实测 −1.3~−4.9%):occ=1 下需要第二代
  workgroup,而第二代要等 CU 空出来,于是首尾两代同时在跑相隔一整趟 walk 的 id,live 窗口不再连续。
- ★★ **三测/四测(2026-08 mx8tw campaign)把根因收得更死,而且它不是「重叠不了」而是「寄存器不够」**:
  用**单份 body、grid 减半、stride=grid**(band 窗口与一 tile 一 WG 完全相同,排除了 L2 局部性这个混淆项)
  复现,ISA 直接给出 `vgpr 256 / spill 25`,隔离 −26%。四次判负的 ISA **每一次都指向同一个数字:
  差 23–25 个 arch VGPR**(r7 spill 25 / r12 spill 25 / r14 spill 23)。
  ⇒ **真因 = tile 循环把 lane-resident group 表与 pid/nsms/total_tiles 拉成跨主环存活,撞 246/256 的墙**,
  不是"没东西可重叠"。⚠ **这意味着它在 8-wave(2 waves/SIMD、余量只有 10)上是死的,但降到
  1 wave/SIMD + AGPR 累加器之后空闲 arch 寄存器变成 72** —— 那时它是一条**有 ISA 门可判**的候选
  (先开开关编译一次看 spill),不是死路。**别拿 8-wave 上的四次判负去否定 4-wave 上的同一动作。**
  同一 campaign 的算术:F 的四分项(启动 1.39 / 解码 0.85 / C store 2.37 / 冷填充 0.80 µs)里前两项
  一个 WG 连做 k 个 tile 就只付 1/k,k=8 时 F 5.43 → 3.15 µs ≈ **+6.4%**。
- ✅ 附带复核（同一次探针）：swizzle base `(bm,gm,xcd,gn)=(256,4,4,0)` 在 6 行里 5 行最优或持平，`gm=8` 只在两行 balanced 上快 0.4~1.1% 而在别处输 —— 与 round 5 的 14-cfg 结论一致，别为单 shape 硬编码。

### ❌ 别再试（2026-07-30 实测）：NT prologue 的三条「省一次内存往返」改法 —— prologue 是带宽限而非延迟限
基线 gm 1.10638 / min 1.03563（同 session 两次复测 1.10655 / 1.10638，极差 0.017%），wgrad 六配置作漂移对照。
- ❌ **把 K-iter 0 的三条 scale load 提到 prologue 最前面**（想把首条 MFMA 的 `WAIT[vmcnt(0)]` 换成部分 drain）：
  `WAIT[vmcnt(0) lgkmcnt(6)]` 确实变成 `WAIT[vmcnt(12) lgkmcnt(6)]`，但 **`private_seg_size` 0 → 28、
  `num_vgpr` 254 → 256**，gm **1.0946（−1.08%）**、NT 八配置几何 −1.82%。根因：该核 254/256 VGPR **零余量**，
  拉长 12 个 scale VGPR 的活跃区间就换来每 tile 一次 scratch 往返。**动这个核的任何活跃区间前先看 spill。**
- ⚪ **同一改法收窄到只跨 stage-1**（scale 排在 k=0 与 k=1 两批 g2s 之间）：spill 回到 0、`num_vgpr` 254 保持，
  首条 MFMA 变 `WAIT[vmcnt(1)]`，gm **1.10493**、NT 几何 1.08114（基线 1.08200，wgrad 对照同步 −0.22%）⇒ **噪声内**。
- ⚪ **两段 prologue g2s 合并成一次突发**（k=0 与 k=1 写的是不相交的 LDS 池，先发满 16 条 DMA 再 drain；
  barrier 条数不变，第二个 drain 仍用主环那条已验证的 `_nd` 余量）：ISA 确认
  `G2Sx4 … G2Sx12 BAR WAIT[vmcnt(14)] BAR WAIT[vmcnt(6)] BAR`，spill 0，gm **1.10483**、NT 几何 1.08111
  —— 与上一条**逐位相同**，即少一次串行 drain **完全没有兑现**。
- ⇒ **结论：该核 prologue 的两次 drain 不是两次可省的延迟，而是同一份带宽的两次计费。** 与
  pitfalls/03「prefetch 治不了 L2 capacity thrash」同源。**别再花轮次在 mxfp8 NT prologue 的发射顺序/drain 合并上**；
  剩下的开口在**减字节**（更大 tile / scale 物理布局），不在减延迟。
- ⚠️ **作用域：以上判负是 mxfp8 NT 的，不能搬到 mxfp4 grouped NT**。同一动作（k=0 scale load 提到
  prologue 最前、与操作数首填共用一次 drain）在 mxfp4 grouped NT 上实测 **+0.4%（两样本 3984.7 / 3983.0
  vs 基线带 3963.9~3973.2）**，spill 仍为 0。差别是机制级的：mxfp8 那边 scale VGPR 由 RA 分配，提前发射
  = 拉长活跃区间 ⇒ 254/256 零余量下必然 scratch；mxfp4 whole-loop 的 scale 落在 **pinned VGPR（PINBASE=8）
  且整段在 `has_side_effects` 的 inline_asm 内**，活跃区间由手写调度钉死，RA 碰不到，所以不产生 spill 级联。
  ★ 判据：**「提前发射某个 load」这一族的成败由「它的目的寄存器是 RA 分配的还是手写钉死的」决定，不由核/精度决定**；
  动之前先 dump ISA 看 `vgpr_spill_count / private_segment_fixed_size`（mxfp4 grouped NT 实测
  `agpr 256 / vgpr 460 / accum_offset 204 / spill 0 / private_seg 0`）。

### ⚠️ 作用域修订(2026-08-06 mxfp4 grouped 实测):scale 投递路径的判负是**实现**判负,不是机制判负
上文死路表里的「❌ scale via LDS 双缓冲 / 慢 6×」与本卡别处记的「`coop=True` 判负」读起来像整族已关闭。
mxfp4 grouped 上用 subtractive 探针给这条路重新定价,结论要收窄:
- **scale 路是环内第二大池子**:删掉每相两条 scale `buffer_load_dwordx4` = **+4.69%**(仅次于删全部 operand
  g2s 的 +10.7%);再用**冻结地址**探针(scale 步进置 0,指令数/VGPR 写回/vmcnt 占用逐项不变)拆开 =
  **访存侧 1.79% + 结构侧 2.90%**。⇒ 这一族有真钱,别当死路整族跳过。
- ★ **给 scale 路定价不能只用「删掉两条 load」一个探针**:删 load 的同时把 graded `vmcnt` 的排空量也改了,
  两个效应混在一起。必须配冻结地址探针才能把访存侧与结构侧分开。
- 现存的 `_COOP`(scale 走 LDS)分支自带 `vmcnt(0) lgkmcnt(0)` 的**每相位全排空**并关掉 store fold —— 这一条
  就足以解释它当年为什么输 **⇒ 判负的是那个实现,不是「scale 走 LDS」这条机制**。要重开先把 drain 改成
  graded 并保留 peel 路径;⚠ 但先算 LDS:mxfp4 grouped 的 `group_segment_fixed_size` 已是 163840(满),
  scale 再要 8~16 KB 得先从别处让出来。
- ❌ **别再试「加深 scale 预取」这一族**(2026-08-06,mxfp4 grouped):把 scale 预取从 1 相加深到 2 相
  (发射点移到相位尾、period-2 寄存器轮转,与 unroll-2 的循环体周期对齐,**零额外 VGPR**)实测
  **gm 4070.8 / 4060.7 / 4057.5 vs 基线 4064.8 = 平**(±0.4% 噪声带内),det 12×200 全 0 PASS、
  `next_free_vgpr 464→460`、spill 0。⇒ 上面那 2.90% 的**结构侧不是"预取深度不够"** —— scale 的消费点虽然
  紧贴相位 barrier,但 graded `vmcnt(10)` 已经给了它足够余量,加深只是把同一份余量换个地方花。
  配套反证:同轮把 `wlv` 10→12(给 scale 更多在飞额度)= **−0.2%**。
  ⇒ 结构侧 2.90% 的下一个候选是**发射槽/寄存器写回本身**(每相 2 条 dwordx4 写 8 个 VGPR 并占 2 个 vmcnt 名额),
  形态 = 让相邻两相共用一条更宽的 scale load,而不是让同一批 load 提前更久。
- ⛔ **上一条的"共用一条更宽 scale load"按字面不可实现(2026-08-12 静态判负,别再照着做)**:一相**每操作数需要
  4 个 dword**(2 个 region group × n_sub=2 个子块),而一个 scale dword 靠 `op_sel`+`op_sel_hi` 只能服务 **4 个
  MFMA**(2 bit = 4 种字节选择),8 个 M 行块 × 2 子块正好把 4 个 dword 占满;`buffer_load_dwordx4` 又是最宽的
  buffer load ⇒ **每相 2 条 scale load 是 ISA 地板**,合并两相需要 8 dword。要动这条得先把 **scale slab 物理
  布局改成 2 dword/相**(每 dword 只服务 2 个 MFMA,一半 `op_sel` 组合作废,slab 字节数翻倍),即代价换到
  preshuffle 的搬运量上 —— 这已经不是"改 emit",而是换布局,先掂量 preshuffle 已经是带宽限这件事。

### ⚠️ 判据修订(2026-08-06 mxfp4 grouped 实测):C-store 的「合并到同一条 cache line」在 occ=1 窄存核里**反号**
本卡 §store 变宽/coalescing 一族记的是「加宽没用」。mxfp4 grouped 上多量到一个**方向相反**的现象,要单列:
- 把折叠 C-store 按**输出行**成组发射(连续 4 条 `buffer_store_short` 正好拼满同一个 128B 行)实测
  **−0.76%(`_CRATE=2`)/ −0.52%(`_CRATE=4`)**;而按 accumulator 发(连续 4 条打 **4 个不同行**、每行 32B)更快。
- ⇒ ★ **判据 = 看「连续几条 store 是否打同一条 cache line」,打同一条要避开**:同 line 的连续写在写通路上
  串行,跨 line 才流水。这与「把写攒到一起更好」的直觉相反,而且它与「加宽 store」是两个独立的轴
  (加宽已判负 +3.5~4.1%,这里是**不加宽、只改顺序**也判负)。
- 相关:C-store 的 cache-policy 位整族关闭。`sc0`(mxfp8 grouped NT 实测 gm −0.46%)与 `nt`(mxfp4 grouped 实测
  EA 读 10.030M→8.384M **但 EA 写 12.059M→18.589M(+54%)**、时间不动)机制同源 —— 每条 store 的 4 段 32B
  是**在 TCC 里**才合并成 64B EA 写,标了非临时就合不起来。要靠 `nt` 省掉 C 对 L2 的污染,前提是先把 store
  变成整线宽写,而宽存本身已判负。
- ★★ **续刀(2026-08-07 mxfp4 grouped 实测,md5 门控双样本):折叠 store 之后同一个 `nt` 从"时间不动"
  变成 −4.5%**。`nt` 臂两样本 **3885.2 / 3924.7**(互差 1.0%),基线两样本 **4103.0 / 4109.0**(互差 0.15%)
  ⇒ **−4.49%**(取干净的 md5 门控样本;首个样本 −5.31% 那次环境里有并发 agent,只作旁证)。
  `min_tf` 3828 → 3420,最惨的 wgrad gate_up heavy 4070 → 3420;
  `_probe_det_wide.py` 12×200 全 0、SNR 49.6 ⇒ 纯性能反号。机制 = 上面那条 **+54% EA 写放大**本身没变,
  变的是它**落在哪里**:store 折叠进尾相之后,发射被 `_CRATE` 按"每条 MFMA 清几条 line"定速,窗口是照
  **未放大**的写量算的;放大 54% 之后剩余部分掉出窗口、重新拖在 MFMA 流后面,于是原先被 drain 吸收掉的
  代价全部暴露。⇒ ★ **判据升级:cache-policy 位一旦在"未折叠"形态下量到"时间不动",不能推论折叠形态下也
  免费 —— 折叠把免费的字节放大变成了暴露的发射**。这一族对 mxfp4 grouped **主动有害**,别再重测(round-22
  管理者指令已明写"不要碰任何 store 变宽/`nt` 方案",本轮仍重测了一次,纯属浪费轮次)。
- ⚠️ **作用域修订(2026-08-11 mxfp4 grouped G=32 deploy 实测,两处独立):上面这条 −4.5% 的前提是折叠 store
  是窄形态(`buffer_store_short`,每 lane 2B/4 段 32B),不是"nt 这一族关闭"**。同一个 `nt` 位加在
  **整线宽折叠 store**(`_BILV=4` ⇒ `buffer_store_dwordx2`,16 lane 连续满 128B 行)上,在 32 组瘦组
  deploy regime 是**赢**:wgrad 折叠 store **+2.1~2.4%**(gm 96.1→97.0,L2 hit 59.2→61.2%、DRAM 读 −72~99MB,
  写字节与 `TCP_TCC_READ_REQ` 逐位不变),NT(fwd/dgrad)折叠 store **+0.88 gm**(6+6 交错配对样本中位
  101.42 vs 100.54,deploy 侧 gate_up fwd +1.6% / down fwd +1.1% / heavy gate_up fwd +1.1%,参考路径逐位不变)。
  ⇒ ★ 正确判据是卡尾那条「**每 lane <128B / 16-lane 连续则 L2 写合并承重,不得加非临时提示**」的**逆命题**:
  写满整线时合并不再承重,`nt` 省下的正是被 C 挤掉的 operand 驻留,**而收益大小 ∝ 同一时刻竞争 L2 的
  expert slab 数**(G=32 deploy 拿到 1~2%;G=4 大 M 参考侧同一个位是平的/略负,所以要**按运行期 shape 选**,
  别全局开)。别把"−4.5%"读成整族关闭,先问三件事:store 是不是整线宽、有几个 slab 在竞争、参考侧要不要保持逐位不变。
- ❌ **别再试"把折叠 store 的骑乘窗口拉宽"这一族(2026-08-11 mxfp4 grouped wgrad 实测 −0.47 gm)**:变 K
  wgrad 的运行期 peel 原本是两相(accumulator 只在第二相终态,store 只能骑第二相),改成**按 accumulator 融合
  两个 k-block**(每个 acc 在第一个行块后即终态,store 可骑整段 peel;per-acc MFMA 顺序不变,SNR 55.6/det 通过)
  实测 gm **100.07 vs 100.54**,deploy 侧六个 wgrad 格一致 −1.1~1.4%。根因:两个 k-block 同时在飞 ⇒ B fragment
  要从两个 LDS 槽各读一份(原地 refill 方案只读一份),多出来的 `ds_read` 比拉宽的 store 窗口更贵。
  ⇒ ★ 与 R2/本轮的 `nt` 结果合起来读:折叠 store 的开口在**减少它对 L2 的污染**,不在**给它更多骑乘时间**。
- ❌❌ **整族封板(2026-08-12 mxfp4 grouped wgrad,四条独立判据):折叠 C-store 的宽度 / cache scope / 银行深度 /
  排空速率四个自由度全关,只剩"写字节本身"**。
  - **宽度不可再加**(静态判负,ISA 级):`_BILV=4` 的 B 交错让**一条 lane 恰好拥有 4 个输出列**
    (`_col(tj)=4*(lane%16)+tj`,4 个 N 子块各出一列)= 4×bf16 = **8 B = dwordx2 已是上限**;要 dwordx4 得让
    lane 拥有 8 个连续列,而这需要 **ntb=8 / BLOCK_N=512**(AGPR 翻倍,不可行)或**跨 lane 压缩**(从 lane 2L/2L+1
    收 2 个 dword → stride-2 gather,DPP/`permlane16_swap` 都是错轴,只剩 `ds_bpermute`)。而 16 lane 已经写满
    一整条 128B 行 ⇒ 加宽**不减事务数**,只把 64 条 store 换成 32 条,却要多付 128 条 LDS crossbar 指令。
    ★ 与卡首「别再试:store 变宽」和 §590 的顺序判据合读:本核这一族**已无剩余自由度**,不要再按"tile 内列聚合/
    对齐"去找形态。
  - **cache scope 位是主动有害**(实测):在**整线宽折叠 + `nt`** 的最优态上再加 scope 位,交错 A/B 各 3 次 →
    `nt sc1` **gm 98.32**、`sc1` **98.31**、`sc0 sc1 nt` **99.43**,基线 `nt` **100.49~100.69**;六个 wgrad 格
    一致跌 4~5%(gate_up wgrad canon 0.939→0.893,down wgrad canon 1.107→1.052),同一次运行里不受该旋钮影响的
    NT 格逐位不动(自带对照)。机制:`sc1` 让写**穿过 per-XCD L2**,而 L2 正是这条写流必需的合并/吸收缓冲 ——
    与 §595「4 段 32B 是在 TCC 里才合并」同源,只是这次是"整线宽也一样需要 L2 吸收"。⇒ **`nt`(分配但 evict-first)
    是该轴唯一正解**,`sc0/sc1` 别再碰。
  - **银行深度与排空速率是平的**:`cst_wide` 的数据 VGPR 银行 2→3→4(`s_waitcnt vmcnt(4→8→12)`)与
    `_CRATE` 6→10→14 全部落在噪声带内(bank2 五样本 100.61~101.08 vs bank4 三样本 100.56~100.99,同一分布);
    ⚠ **bank8**(`_NCDV` 18→66)直接触发 LLVM RA **`Cannot decrease cascade number, illegal eviction`** 崩溃 ——
    `_PINBASE` 那套显式钉寄存器的绕法**不覆盖这条**,压力一到就复发。
  - **折叠里的 `v_accvgpr_read` 搬运层不可省**(静态判负,ISA 级,llvm-mc gfx950 实测):`v_cvt_pk_bf16_f32`
    **不接受 AGPR 源**(`v_cvt_pk_bf16_f32 v0, a1, a2` / 混合 `v0, a1, v2` 全部 `invalid operand for instruction`),
    只有 `v_accvgpr_read_b32` 与 MUBUF(`buffer_store_short_d16_hi a{n}` 能直接读 AGPR)。⇒ 每 2 个输出 dword
    的 "4 读 + 2 转" 是**下限**,那 384 条/tile VALU 删不掉;而它本来就藏在尾相 MFMA 影子里(VALU 管道占用
    < MFMA 管道占用),不是开口。
    - ★★★★ **⚠ 2026-09-11 收窄(campaign 20260910_041317 REPLAN,llvm-mc gfx950 五个受控组合)**:
      上面那条结论的**隐含前提是「累加器留在 AGPR」**,而前提本身可以换掉。把 MFMA 的四个寄存器类
      逐格测一遍,真实规则是 **dst 与 src2(累加器)必须同类,srcA/srcB 各自完全自由**:

      | `v_mfma_scale_f32_16x16x128_f8f6f4` (dst, srcA, srcB, src2) | 判定 |
      |---|---|
      | `v, v, v, **a**` / `a, v, v, **v**` — dst 与 src2 跨类 | **REJECTED** |
      | `v, **a**, v, v` / `v, v, **a**, v` / `v, **a**, **a**, v` | **OK** |
      | `a, **a**, **a**, a` | **OK** |
      | scale 操作数放 AGPR(`…, a8, a12`) | **REJECTED** — scale 必须 arch VGPR |

      配套也测了:`ds_read_b128 a[0:3], v2`、`ds_read_b64_tr_b16 a[0:1], v2`、
      `buffer_load_dwordx4 a[4:7], …`、`buffer_store_dwordx2 a[0:1], …` **全部合法**
      ⇒ **LDS / global 可以直接落进 AGPR**。
      ⇒ 于是存在一条没被测过的构造:**把累加器(dst 与 src2 一起)整体搬进 arch VGPR、把 A/B 片段
      搬进 AGPR**,`v_cvt_pk_bf16_f32 v,v,v` 直接读累加器,**256 条 `v_accvgpr_read_b32`/wave/tile
      整族消失且不引入新指令**;片段与"抽干中的那半累加器"都是 128 个寄存器,两个 bank 的占用
      一个字没变(`_CDEF` 已把 AGPR 切成 a[0:127] 抽干 / a[128:255] 累加)。
      ⚠ 之前判死的 `_MDST` 路试的是「累加器留 AGPR、让 MFMA 写 VGPR dst」,那正是唯一被禁的那一格。
      ⚠ `buffer_store_short_d16_hi a{n}` 虽能直接读 AGPR 省掉 `v_cvt`,但它是**截断不是 RNE**,
      会改舍入 ⇒ 在"禁止精度改动"的 campaign 里不许提。
    - ★★ **同处补一条口径**:此前把这 384 条说成"藏在 MFMA 影子里,不是开口"。occ=1 的核上
      实测 `SQ_VALU_MFMA_COEXEC_CYCLES / SQ_VALU_MFMA_BUSY_CYCLES` = **2.07%**
      ⇒ **几乎没有影子可藏**,每条非 MFMA 指令实测值 8.5 拍的 MFMA 管线空转
      (标定见 methodology/03 §`duty = 16/(16+c·n)`)。"藏在影子里"只在 ≥2 waves/SIMD 上成立。
    - ★★★★ **✅ 2026-09-11 r39 上机结果:上面那条构造是真的,但 8.5 拍这个价签是假的。**
      按 `ntmp`(片段组数)封顶把 **40/64 组累加器**换进 arch VGPR、40 组片段换进 AGPR
      (dword 换 dword,`agpr_count` 仍 256、arch 仍 240、`accum_offset 240`、spill 0 全不变),
      `v_accvgpr_read_b32` **512→192 条/模块**(= 160 条/wave/tile),其余每条 opcode 与 `s_nop`
      拍数(10 条/70 拍)逐条不变 ⇒ **"MFMA→VALU 等待态暴涨"这条风险没兑现**。
      正确性:12 形状**逐位相同**(含两条 tail-split)、SNR 55.10~55.61 dB 不变、
      CUDA-graph capture/replay + `beta=1` + fp16 + `trans_c` 全过。
      收益:同进程回文 A/B **+0.65%**(FC1_fwd)/ **+0.05%**(out_proj_wgrad),
      计分 geomean **0.9787 → 0.9831 / 0.9818**(两次采样),12 行无一回退。
      `_NAV`=0/20/40 → +0.00%/+0.23%/+0.55%,**线性单调** ⇒ 满额即最优,无需降粒度。
      ⚠⚠ **价签修正**:按 `c=8.5` 该给 **+2.91%**,实测 +0.65% ⇒ 反解 **c ≈ 1.43 拍**
      (out_proj_wgrad 反解 0.85)。独立交叉验证:每删 1 条指令 `SQ_ACTIVE_INST_ANY` 恰降 1.000
      单位而 `SQ_WAVE_CYCLES` 只降 **0.704 / 1.438** 单位。⇒ **`c=8.5` 是跨不同 K 的两点拟合,
      把 §0.2 那个"每 tile 固定开销(1/K)"重复计了一次**;真实边际成本 0.7~1.4 拍。
      ⚠ **只错在价签,不错在方向**:族的总余量按 `c·n/16` 算 = **+7.9%**(FC1_fwd)/ **+3.3%**
      (out_proj_wgrad),线性外推(dn/n = 8.9% 换 +0.65% ⇒ 全删 +7.3%)自洽 ⇒ **仍是头号杠杆**,
      只是**每个候选的单价除以 6**。⚠ `c=1.43` 不能反推绝对 duty(代进去 93.3%,实测 68%):
      闭式只描述**边际响应**,那 32% 绝对缺口是每 tile 固定开销 + LDS 等待,被 r38 吸进 c 里了。
      ⚠⚠ 施工唯一的坑:约束要从 `"={a[..]}"` 改成 `"=&{v[..]}"`,见 methodology/06 §3 后新增的第 4 条。
    - ★★★★ **✅ 2026-09-11 r40 续刀:稳态 K 循环的 SALU 整族下压(条数模型第二次兑现,单价第三次改)**
      稳态 hw-loop body(unroll-2 一对 = 256 MFMA)**410 → 377 条**,`s_add_u32` **46 → 14**、
      其中 `s_add_u32 m0` **32 → 8**;`ds_read_b128 64` / `buffer_load_dwordx4 36` / `s_waitcnt 2` /
      `s_barrier 2` / `s_nop 2` **逐条不变**,`vgpr 496`/`agpr 256`/spill 0 不变(sgpr 81→87)。
      n(loop) **0.6016 → 0.4727**;PMC 侧 `SQ_INSTS_MFMA/LDS/VMEM` 比值 **1.00000 逐位相同**,
      `SQ_INSTS_SALU` ×0.4457(FC1_fwd)/ ×0.3326(out_proj_wgrad)。
      14 形状**逐位相同**、SNR 54.16~55.61 dB 不变;计分 geomean **0.9825 → 0.9907 / 0.9884**(+0.72%)。
      三件事各自的条数(每对):
      **① wave-major g2s + 12-bit 立即偏移承载 LDS 步进 = −24**。
      **② buf1 的 +KSTEP 也塞进同一个立即数(LDS base 预减 KSTEP)⇒ phase-B 三个 scratch soffset 整族消失 = −6**。
      **③ scale soffset 用立即数区分奇偶相位 + 每对只前进一次、循环出口改用 `s_add_u32` 自己的
      carry-out(计数器初值 = first − bound)⇒ 省掉 `s_cmp_lt_u32` = −3**。
      ⚠ ①②的前提是一条此前**没人测过**的硬件语义(见下一条),③无前提。
      ⚠ 施工唯一的坑:`emit_peel_odd`(odd-KI + half_k)里那个**循环外的 phase A** 原本靠自己的
      `_scv_adv` 前进 scale 指针;把 advance 挪给 phase B 之后它就断了,`padK_1152` 直接掉到
      **14.57 dB**。判据 = correctness gate 必须包含 odd-KI/half-K/half-N/rowsplit 这些**尾块变体**,
      只跑计分表的 12 行会全绿地放过它。
      ⚠ **单价再次下修**:r39 量到"每删 1 条指令 `SQ_WAVE_CYCLES` 降 0.704(FC1_fwd)/1.438
      (out_proj_wgrad)";r40 在**稳态循环里**量到 **0.314 / 0.554**。⇒ **边际单价不是每行一个常数,
      而是分区域的**:r39 删的是 epilogue(MFMA 已停,纯串行)≈0.7~1.4 拍,r40 删的是稳态循环
      (还有 `s_waitcnt`/barrier 可吸)≈0.3~0.55 拍。**提案定价前先说清楚指令在哪个区域**。
      ⚠ **省下来的发射槽去哪了**:`SQ_WAIT_INST_LDS` **纹丝不动**(1.24% / 0.28% of wave cycles),
      涨的是 `SQ_WAIT_ANY` **+11.45% / +11.50%**(11.76→13.20% / 10.83→12.26%)⇒ 是**相位 barrier 上的
      vmcnt/会合等待**吸走的,不是 LDS 管线。FC1_fwd 真周期 −0.67% 但 wall **±0**(1400 W 钉死,
      sclk 回吐);out_proj_wgrad 真周期 −1.49% ⇒ wall **+0.79%**(兑现率 53%)。
      ⇒ **短 K 行的条数杠杆已经落在功耗墙后面,长 K 行还在兑现**。
      ⚠ 循环体现在 121 条非 MFMA 里 **100 条是 ds_read(64)+ buffer_load(36)**,两者都在几何下界上
      (ds_read 0.25/MFMA = 128×128 wave tile ÷ 16×16 atom;g2s 32 条/对 = dwordx4 的字节下界),
      **4 条 m0/相位也是信息论下界**(每 wave 每相位 16 KB ÷ 4096 B 立即数窗口)。
      ⇒ 想再动条数必须先动 wave tile(= 寄存器资本,goal §D),别在这个 body 里继续找。

### ★★★ gfx950 的 MUBUF LDS-DMA:`offset:` **同时**落在 LDS 地址和内存地址上(2026-09-11 r40 实测)
此前没人测过,goal §B 的整条路都压在它上面。三条实测:
1. **合法性**:`buffer_load_dwordx4 vaddr, srsrc, soffset offen offset:N lds` 在 gfx950 上**汇编通过**,
   `offset:1024/3072/4095` 编码分别是 `0x14/0x1c/0xff,0x1f` ⇒ 就是那 12 bit。
   ⚠ **`offset:4096` llvm-mc 不报错,静默截断成 `offset:0`**(编码与 `offset:0` 逐位相同)⇒
   **必须在生成侧自己 assert ≤4095**,别指望汇编器兜底。
   ⚠ 语法上 LDS 形式**没有 vdata 操作数**:`buffer_load_dwordx4 v2, s[8:11], s4 offen lds`(三个操作数)。
   写成 `v1, v2, s[8:11], s4` 会报 `invalid operand for instruction`,别据此以为不支持。
2. **语义**:LDS 地址 = `M0 + inst_offset + TID*16`,内存地址 = `base + soffset + voffset + inst_offset`,
   **同一个 inst_offset 进两条式子**。判据 = 让每步的 gmem voffset **预减**这个立即数:
   若只落 LDS 侧则内存读错、只落内存侧则 LDS 写错,两种都会立刻掉位;实测 14 形状**逐位相同**。
3. **代价**:`s_add_u32 m0` 从"每条 g2s 一条"降到"每 4096 B LDS 窗口一条"。前提是把 LDS 填充从
   **step-major**(一个整 WG 步长里交错四个 wave)改成 **wave-major**(每 wave 一段连续行),
   这样同一 wave 相邻两步只差 `rows_per_step × bytes_per_row` = 1024 B。
   两种排布下 `row % rows_per_step == ph` 都成立 ⇒ **bank swizzle / LDS 镜像 / 下游 ds_read 全不变**,
   这是"可以逐位相同"的原因;变的只是**哪条指令读哪 8 行**,而 g2s 成本 = 覆盖的 128B line 数(见下文
   五探针),line 数不变 ⇒ 访存侧中性(实测 `SQ_INSTS_VMEM` 逐位相同、`TCP` 侧无回退)。
   ⚠⚠ **唯一的正确性约束**:voffset 预减立即数后**不能为负**。A 流(无 ilv)`min_src = r·8`,任意
   K ≥ 256 都够;**B 流带 `ilv=4` 时源行被置换**,`w=0,r=2` 的 `min_src = 1` 而立即数 = 2048
   ⇒ 需要 `row_bytes ≥ 2048`(K ≥ 4096)。**必须在 build 期把 `min_src × row_bytes ≥ imm` 逐流逐步
   算一遍并在不够时回退**,不能靠"看起来单调递增"推。(计分表 12 行 K ∈ [4096, 32768] 全部满足。)

### ❌ 死路(2026-09-11 r40 实测):**SALU 削完之后再调相位 barrier 的 vmcnt 水位线**
PMC 说省下的发射槽变成了 `SQ_WAIT_ANY`(+11.5%),很自然会想"水位线是不是配深一点就能吸掉"。
`_autotune_mxfp4_config` 只对 **K ≥ 8192** 提供 `(wlv, elgk) = (16, 15)` 这一档,K=4096 的行**从来没被
量过**。补测(钉死 swizzle 三元组、只动水位线,90 s 预热 + 回文 + min-of-4):
FC1_fwd `(10,9) / (13,12) / (16,15)` = **+0.00% / +0.02% / −0.01%**;QKV_fwd = **+0.00% / −0.01% / −0.00%**。
⇒ **整条水位线轴在 K=4096 上是平的,别再开轮**。机制:`vmcnt` 水位线只能决定"提前多久停下来等",
决定不了 DMA 本身的延迟和四个 wave 的会合时刻;`SQ_WAIT_ANY` 涨的是后者。
⇒ 想吃掉这块等待要动**环深**(2 buffer → 3)或**每相位字节数**,不是水位线。

### ⚪ 关闭(2026-08-06 mxfp4 grouped 实测):**读侧** cache-policy 位在本核 operand 路上不承重
写侧的非临时/scope 位早已判负(见上),读侧此前没数。补上:operand 的 `buffer_load_dwordx4 … lds` 加 `sc0`
(绕 L1、想给 scale 腾出 L1 驻留)= **4069.9 vs 4071.3,逐点持平**,SNR/det 正常。
⇒ **L1 驻留在本核 operand 路上不承重,读侧 cache-policy 位可以和写侧一起关掉**,别再单开轮次。
- ★ 续刀(2026-08-07):`nt`(非临时,跳过 L1 分配)在同一条 operand g2s 上是**重度负**:**3682.2 vs 4091.6 = −10.0%**。
  ⇒ `sc0` 的"持平"不能读成"L1 对 operand 无所谓" —— 正确读法是 **`sc0` 只改 scope、不改分配策略,而 `nt` 改分配策略**;
  operand 流**重度依赖 L1/L2 的行复用**(group_m band 内同一 B block 被多个 tile 复用)。判据:读侧 cache-policy
  这一族在本核**整族关闭,且 `nt` 方向是主动有害**,别拿"绕 cache 减污染"的直觉去动 operand。

### ★★★ operand g2s 池的五探针分解(2026-08-07 mxfp4 grouped 实测):成本 = **它覆盖多少条 128B line**,字节数 / LDS 写口 / 波间碰撞全是 0
round-16 的 subtractive 探针把"删掉全部 operand g2s"定价在 +10.7%,round-19 用冻结地址探针拆成
访存侧 2.8% + 结构侧 7.9%。本轮用**五个隔离探针**把它钉到唯一一条通路上,并且发现**真值远大于 10.7%**。
每条探针都刻意只放开一个自由度(基线 gm 4076.3 / 4091.6,同 session):

| 探针 | 指令数 | 每条覆盖的 line | 字节 | LDS 写 | gm_tf | vs 基线 |
|---|---|---|---|---|---|---|
| **`buffer_load_dword`**(宽度 ÷4) | 不变 16 | **不变 8** | ÷4 | ÷4 | 4082.0 | **+0.14% = 平** |
| **去 `offen`**(全 lane 同址,`off`) | 不变 16 | **8 → 1** | ÷8 | ÷8 | **4901.6** | **+20.2%** |
| **VGPR-dest**(去掉 `lds` 后缀) | 不变 | 不变 | 不变 | **删光** | 4083.6 | −0.2% = 平 |
| **wave-stagger**(`HW_REG_HW_ID` 逐波插 nop) | +循环 | 不变 | 不变 | 不变 | 3994.9 | −2.4% |
| stagger 的**恒定控制组**(同循环、零错峰) | +同样循环 | 不变 | 不变 | 不变 | 3988.9 | −2.6%(⇒ 错峰净值 **+0.15%**) |

- ★★★ **判据(本卡最重要的新条目):`buffer_load … lds` 的成本 = `覆盖的 128B line 数`,而不是字节数、不是指令数、
  不是 LDS 写口、不是波间同相。** 前两行是配对反证:**宽度 ÷4 而 line 不变 ⇒ 完全不动分;line ÷8 而指令数不变 ⇒ +20.2%。**
- 量纲闭合:本核每条 g2s 是 64 lane × 16 B = 1024 B **跨 8 条 line**,每相每波 16 条 = 128 line,4 波 = **512 line/CU/相**;
  512 line 对 ~3355 cyc 的相位 ≈ **15%**,与实测 20.2%(还含它给 C-store 腾出的混跑带宽)同量级。
- ⚠ **20.2% > round-16 "删光全部 operand g2s" 的 +10.7%**。两者不矛盾但要记住:`off` 版本同时把**每 CU 的
  L1 工作集从 64 KB 压到 8 KB**(compulsory miss 几乎清零),所以它同时买到了 line 数与 L2→L1 流量两项;
  ⇒ **给这一池定价必须用 `off` 探针,`删光 load` 的 subtractive 值是低估。**
- ⇒ **减 line 是唯一入口**,而 line 数 = `(BLOCK_M+BLOCK_N) × BK/2 / 128`,与 MFMA 功(`BM×BN×BK`)之比 ∝ `(BM+BN)/(BM×BN)`
  —— **与 BK 无关**(所以调 BK 永远不动这一项,和 BK128 判负 −13% 一致),只能靠**放大 tile**。
  而 `BM×BN ≤ 4 wave × 64 lane × 256 AGPR = 65536` ⇒ **256×256 正好卡在寄存器池的等号上**(pitfalls/01)。
  下一个要试的形态因此是**把累加器需求降下来**(例如一个 WG 同时算 N 相邻两个 tile 共享 A 的 line,需要 512 AGPR/lane
  ⇒ 先解决累加器容量),而不是继续在发射侧(宽度/顺序/错峰/cache-policy)找。
- ❌ **别再试**:①`buffer_load_dwordx2 … lds`(gfx950 非法编码,MUBUF-LDS 只有 dword 与 dwordx4);
  ②把 g2s 目的地从 LDS 换成 VGPR 来"省 LDS 写口"(写口零成本,换过去还得自己发 ds_write);
  ③任何形式的 wave 级发射错峰(错峰净值 +0.15%,而实现它的循环本身要 −2.4%);
  ④用"窄一点的 load"省字节(字节完全不承重)。
- ★ **读写不对称,别互相外推**:**读侧是 line-数限**(本节),**写侧是字节/带宽限**(见本卡 §store 三点称重)。
  证据 = `store_tacc_wide` 把 store 的 line 请求减 4×(256 条窄存 1024 组 → 32 条宽存 256 组)实测**慢 3.5~4.1%**,
  而读侧同样的 line 减法值 +20.2%。判 epilogue 与判主环必须用两把不同的尺子。

### ❌ 偏移(skewed)operand 环 3 槽 → 2 槽:**LDS 省 32KB 要付 1.66%**,其中只有 0.5% 是同步、1.16% 是预取 aging(2026-08-07 mxfp4 grouped 实测)
`(BM+BN)·320 → (BM+BN)·256 = 131,072B` 是 `BM>256` 整族(BM=288 要 174,080B)的**唯一资金来源**,
所以它到底值多少必须单独称。本轮把它实现并交错称重(base/cand/base/cand,同一 relay/GPU/venv):

| | 基线 | 2 槽候选 | 基线 + **只加**中相同步 |
|---|---|---|---|
| `gm_tf` | 4076.2 / 4082.8 → **4079.5** | 4011.5 / 4012.2 → **4011.9** | **4059.2** |
| vs 基线 | — | **−1.66%** | **−0.50%** |

`min_tf` 3788.8 → 3689.3(−2.6%)。两个候选样本互差 0.02%、两个基线互差 0.16% ⇒ 回归是噪声的 ~10 倍。
SNR 49.6dB / 逐字节确定性全程为真 ⇒ **2 槽环是正确的,只是更慢**。

- **实现要点(想复现别走弯路)**:字面"把 16 条 `ds_read_b128` 前置到相位头"**做不到** —— fragment 得双缓冲
  (+128 VGPR),而 ISA 实测只有 52 条余量。等效且零寄存器成本的做法是:MFMA 流改**子步优先**
  (32 quad 全 s=0,再全 s=1)+ 在 s=1 半程第 8 条 MFMA 处插 `s_waitcnt vmcnt(10) lgkmcnt(0)` + `s_barrier`,
  把 s=0 刚读完的槽在**相位内**交给本相的偏移 g2s。这样偏移环 = 2 个**静态**槽(s=0 读 `buf`、s=1 读 `1-buf`),
  旋转状态(10 个 SGPR、`emit_rot`/`emit_bases`/`mix_g2s`、`q_unit()`、向量 temp 池)全部消失。
  末次迭代那条越过 K 的偏移 g2s 会打烂 peel 要重读的槽,用 `s_cmp_ge_u32 $o_cnt, $o_nlst` + `s_cbranch_scc1` 压掉。
- ★ **减法探针给出唯一有用的分解**:同步本身(2 条指令 + lgkm/vm 排空)只 **−0.50%**;剩下 **−1.16%** 是
  **2 槽固有的** —— 偏移 g2s 的 aging 从"整相"掉到"半相",外加子步优先重排与压制分支。
  ⇒ 同步那 0.5% 可以再优化(挪位置/降 vmcnt),**aging 那 1.16% 这条路上省不掉**。
- ⚠ **判据**:round-5 那个"前置读 ≈ +1.5%"的单独估值**不能**当成 2 槽环的估值 ——
  「前置读」和「缩环」不是同一个改动,而 BM=288 需要的是后者。凡是靠缩环腾 LDS 的方案,
  **必须把 1.7% 记成入场费**、和放大 tile 在**同一轮**里一起称,单独称缩环只会拿到一个负分。
- 参考:本核 128 条 MFMA 的相位里,一次 `lgkmcnt(0)` 只值 0.5% ⇒ **相位内插同步不贵,贵的是预取深度**。
  这和 §"删一组 g2s 不必退回全排空"、pitfalls/04 的"graded vmcnt 首先是结构问题"是同一族判据。
- ★★★ **补一道此前没算的门(2026-08-07 源码判定):`BM` 只能取 128 的倍数,所以 BM=288 就算 LDS/寄存器都
  给够也走不通**。第三道门在 **scale slab 的行组粒度**:`mxfp4_grouped_kernel` 里
  `sa_b = a_pre_g*64 + bm*BLOCK_M + wave_m_off`,而 `_scsoff` 按 `grp = (base+extra)//64` 取组、
  波内区域索引另按 128 行走(`_wia = sa_b//128`)⇒ **`wave_m_off` 必须同时是 64 与 128 的倍数**。
  `BM=288` ⇒ `wave_m_off = 144`,`144 mod 64 = 16 ≠ 0` ⇒ dword 内字节选择错位、A-scale preshuffle 布局失效。
  可达阶梯 = {128, 256, **384**, …};而 `BM=384` 的寄存器账 = 384(acc) + 160(frag) + 76(fixed) = **620 > 512**。
  ⇒ **放大 tile 这一族的真入口是先把 scale slab 重排成与 64 行组解耦的布局**,不是 LDS(2 槽环)也不是寄存器。
  在那之前,`(BM+BN)` 受限于 512 且 `BM·BN ≤ 65536` ⇒ **256×256 是当前 scale 布局下的唯一可行点**,
  round-21 把它"重新打开"时漏掉的就是这道门。
- ★★★ **持久化网格与折叠 store 在 GFX9 上结构互斥(2026-08-07 推导,配 §gfx950 统一 vmcnt)**:折叠 store 之后
  累加器不再是 loop-carried 值(caller 侧 `accL/accR` 已死)⇒ 历史那条"persistent 需要 256 个 iter_arg"的障碍
  **消失了**,所以这条路值得重算 —— 但算下来仍然不通:tile i 的 ~256 条在飞 store 与 tile i+1 的 prologue
  共用统一 vmcnt,prologue 的 `s_waitcnt vmcnt(_NPRE=8)` 会连带等这些 store 退休;非持久化下 `s_endpgm`
  **免费**退休它们、新 WG 从 vmcnt=0 起算。想把 store 排除在计数外需要 `vmcnt(≥264)`,而 **GFX9 的 vmcnt
  字段上限是 63**,根本表达不出来。⇒ 在本核上**非持久化是结构占优**,不是调参问题;要动 persistent 必须先
  放弃 store fold(把 fold 已兑现的收益还回去)。
### ✅ 已交付杠杆（2026-07-30 实测）：NT 的 `GROUP_M` band 宽度必须**按 K 分档**，一个全局 gm 会系统性亏待长-K shape
`GROUP_M` 决定一条 M band 的宽度：B 每条 band 被完整 stream 一次 ⇒ **B 流量 ∝ 1/gm**（与 shape 无关，
logical B/A 恒为 1/gm）；而 band 的 A 足迹 = `gm × BLOCK_M × K` 字节，**只有 K 是 shape 相关项**。
⇒ **同一个 gm 在不同 K 上落在 4 MB per-XCD L2 slice 的两侧**，最优值必然随 K 变。

`_bench_campaign_mxfp8.py` 真尺子实测（每格都是全 14 配置 bench，非竞速点）：

| shape | band@gm4 | gm=2 | gm=4 | gm=8 | gm=16 |
|---|---|---|---|---|---|
| `dgrad gate_up heavy` N=2944 **K=5760**（min 配置） | **5.62 MiB ✗** | 1.010 | 1.036 / 1.041 | **1.045 / 1.045 / 1.047** | 1.031 |
| `dgrad gate_up balanced` 同 cfg_key | 5.62 MiB ✗ | 1.071 | — | **1.089 / 1.093** | 1.064 |
| `fwd gate_up balanced` N=5760 **K=2944** | **2.88 MiB ✓** | 1.089 | **1.122** | 1.086 | — |

* **两条曲线都单峰,但峰不在同一个 gm**:K=2944 峰在 4、K=5760 峰在 8。**四点(2/4/8/16)把 K=5760 的峰夹死在 8。**
* 机制自洽:K=2944 时 gm 4→8 把 band 从 2.88 MiB 推过 4 MiB slice ⇒ 丢 A 驻留,亏 3.0%;
  K=5760 时 gm=4 的 5.62 MiB **已经**越线 ⇒ 再宽也没有更多可丢的驻留,只剩 B 流量减半的净赚,+1.1%;
  gm=16(22.5 MiB)开始亏,说明 band 太宽后 **launch 序的组内局部性**反过来吃掉了 B 的节省。
* ⚠ **别把它读成"A band 必须 ≤ 4 MB"这条简单规则** —— 那条规则预测 K=5760 应当选 gm=2(2.81 MiB 回到线内),
  **实测 gm=2 是全场最差(1.010,比 gm=4 还差 2.5%)**。B 流量项在这里压过驻留项;
  **能同时解释四个点的只有"gm 的最优值是 B 流量与 A 驻留的折中,且折中点随 K 移动"**。
* **落地**:`_gnt_nt_candidates(N, K)` 用 `4*256*K > 4 MiB` 这个**物理阈值**分档 cand[0](另一档留作 cand[1]),
  候选数仍 4;K=2944 三个 shape 的候选表**逐字节不变**(零回归面),只有 K=5760 那个 cfg_key 换基准。
  ✅ 前提是 **`cfg_key = (N, K, G, ...)` 含 K**,所以两档互不干扰;若某核的 cfg_key 不含 K,先补上再分档。
* ⇒ **通用教训:凡是"一条 band 复用一个操作数、另一个操作数按 band 数重 stream"的 swizzle,
  band 宽度的最优值都是 K(收缩维)的函数,拿单一 K 扫出来的 gm 不能当全局默认。**
  round 5 那次 14-cfg 扫描**只在 K=2944 上做**,所以把 K=5760 的 gm=8 漏掉了 —— 判负清单要记扫描时的 K。

### ⚠️ grouped MX NT 的**全局** `gm=8`：对 min 配置是 +1.1%，但全局净负 —— 且 autotune 竞速点复现不出 bench 的判据
2026-07-30 在真 bench 上强制 `gm=8`（单候选，绕过竞速）：gm **1.10404（−0.21%）**、min **1.04659（+1.06%）**。逐配置：
`dgrad gate_up heavy` 1.036 → **1.047（+1.1%，这正是 min 所在配置）**、`fwd gate_up balanced` 1.122 → **1.086（−3.0%）**，
其余六个 NT 配置 ±0.5% 以内。**与 round 5 记的「gm=8 在 down 快 0.6~1.0%、在 fwd gate_up 慢 2.97%」
六轮之后仍逐点吻合到 0.03pp** ⇒ 该 KB 条目经受住了三次内核大改（距离-2 池 / prologue 强度削减 / lane-resident group-find），**可信**。
- 关键结构事实：这两个配置**属于不同的 autotune cfg_key**（`fwd gate_up` 是 N=5760/K=2944，
  `dgrad gate_up` 是 N=2944/K=5760），所以"一个 gm 服务全部"并不是硬约束 —— 硬约束是**同一 cfg_key 要同时服务
  balanced 与 heavy 两个分布**（cfg cache key 不含分布），而 `dgrad gate_up` 恰好 balanced 想要 gm=4（+0.6%）、
  heavy 想要 gm=8（+1.1%）。
- ❌ 别再试：**把竞速采纳门从"每点都赢 1.5%"换成"几何均值赢 1.5% + 单点回退上限 1%"**（gm 1.10648 / min 1.03679，
  与基线同为噪声）—— 因为 `dgrad gate_up` 上 gm=8 的两点几何只有 +0.25%，**够不到任何高于噪声地板的门限**。
- ❌ 别再试：**把竞速 base 播种成 gm=8 再让 gm=4 竞争回来**（gm **1.10217，−0.38%**）：
  `fwd gate_up balanced` 停在 1.086 没被换回 1.122 ⇒ **竞速的 canonical 点（2048 / 8192 tokens per group）
  复现不出 bench 的 4096 tokens/group 在 N=5760 上的判据**。这是本条最有价值的发现：
  **该 autotune 竞速对 N=5760 shape 的采纳判据与生产不同源**，任何指望竞速自己发现 gm 的做法都会落空。
- ⇒ 下一步不是继续调门限，而是**让 cfg cache key 带上一个粗粒度的分布描述子**（如 max_group_tiles/mean_group_tiles 分桶），
  好让同一 shape 的 balanced 与 skew 走不同的 band。⚠ 该描述子必须**在 device 上算**——round 3 已定死"不得引入 host D2H 检查"。

### ★★ 收官轮 L2-miss 分维归因:证伪「scale-thrash」,超额 miss 是**算子回取**,下一刀在 LDS→VGPR 不在 DRAM 字节(2026-07-30 PMC 实测)
本核 NT 上 L2 miss 比 tw 高 +28%(55.66M vs 42.04M ≈ +387 MB/call)。此前假说「两条 E8M0 scale 流拿不到 L2 复用、是超额 miss 主因」被**直接测量证伪**:
- **subtractive pin 探针(只塌某流的 L2 足迹、指令流/grid/请求数逐条不变,ΔTCC_MISS 即该流份额)**,`dgrad gate_up heavy`、7 pin 模式 × 2 counter 组:A-scale **+0.076%** / B-scale **−0.012%** / A+B scale **−0.338%(=超额 miss 的 1.4%)** / A-data **−0.214%(0.9%)** / B-data **−0.401%**。**四条地址流合计 ≤4%** ⇒ 13.57M 超额 miss **不由任何单条流的足迹拥有**;scale 每 band 只要 A 368 KB + B 553 KB,完全驻留得下,根本不 miss。
- 佐证:把两条 scale 流**整体钉成 cache-resident**(P1)拿 **+1.43% 上界**,但拆开看只 ~1/3 是 cache 足迹(discriminator「足迹塌掉但活值全留」+0.54%),另 **~0.89% 是地址算术塌成常量**(死码/live-range 缓解,不是 L2)。非临时提示 scale(H1)反 **−1.69%**——scale 本就享 L2 复用,提前 evict 更亏。
- ❌ **别再做 scale 的 tile-major / cache-line 共置重排**:实测定价 **~0.002% wall**(scale 占 387 MB 超额的 ≤1.4%,而 387 MB 全消掉本身只值 ≤0.18% wall——见下方能量记账),在 0.25% gm 噪声地板以下两个数量级,**未落盘**。
- ❌ **别再在本核上削 DRAM 字节**:超额 miss 全走**本地 HBM**(`TCC_EA0_RDREQ` 恒等 `TCC_EA0_RDREQ_DRAM`、`TCC_EA0_RDREQ_32B=0` ⇒ 无 MALL、**无 32B 半行浪费**,没有 strided/部分行可削);单次调用能量 = 1400 W × 1.6014 ms = **2.242 J**,DRAM 只占 **0.8~1.6%**。⚠ 顺带标定:本核 `TCP_TCC_READ_REQ` 是 **128 B 粒度**(125.3 B/req)、写是 **32 B sector**(31.9 B/req)——与 pitfalls/13 那个 64 B 的形状不同,引流量时按核校准。
- ~~✅ **下一刀 = LDS→VGPR 读放大(满额兑现的能量类)**~~ —— 实测 **0.504 pJ/FLOP vs mxfp8 MFMA 峰值 ~0.28 ⇒ 只 55% 能效**,缺的那一半被归到 LDS→VGPR 通路——每 call 6144 tile × 45 iter × 64 KB = **17.7 GB 进 LDS**,按 wave_n×4 / wave_m×2 **读出 35~70 GB**(~1~2 pJ/B = 35~140 mJ = 能量的 **1.6~6%**)。
  ★★ **2026-08 勘误:这条「下一刀」挂了十几轮,实测两边同档,没有对标差距可拿,已作废**:
  - 闭式 `waves×(tile_m+tile_n)/512` 给 mx 3.0× / 标尺 2.0×,**但按字节实测标尺自己是 2.91×** ——
    它的 TN 布局必须用 8 B 的 `ds_read_b64_tr_b8`,**两条才顶一条 `ds_read_b128`**;而且它每 WG 每
    K-iter 发 **372 条** ds_read(我们只发 192 条,它是我们的 1.94 倍),照样更快。
    ⇒ **闭式只在两边用同宽度读时可比;TN 转置读必须按字节口径重算。**
  - 时间口径上它也不在关键路径:subtractive 实测 32 条 ds_read/iter 只值 ~35 拍
    (reads+MFMA 0.964 vs MFMA-only 0.667 vs 整核 0.965),**LDS 读被 MFMA 完全遮住**;
    `SQ_WAIT_INST_LDS` 折算到每个常驻 wave-slot 是 mx 2.2% / 标尺 2.6%,**同档**(三次独立复现)。
  - 拿它要付 occ 2→1,实测 **−31%**。
  ⇒ **用 `SQ_LDS_IDX_ACTIVE` 立论之前,先用 subtractive 探针确认它在关键路径上。**
    真正的对标差距在**同步指令密度**(每条 MFMA mx 配 0.93 条 waitcnt/barrier/setprio/nop、
    标尺配 0.13 条,7.4×),见 methodology/12 §「追一个具体 kernel 当标尺」。
- ⚠️ **`xcd`/`gn` 轴在 min shape 上已惰性(≠上方 `gm` 轴——`gm` 是已 ship 的 K-band lever,min 上仍值 +1.1%)**:N4 三栏尺子(功耗全钉 1400 W)量 xcd 1/4/8 × gn 0/2/4 全落 ±0.15% wall ⇒ wall=energy/1400 W,这两轴不改 energy。逐条判负:H4 全局 xcd=8 **−1.6%**(miss −20.4% 但 cyc +5.3%)、H5 按 band 宽分档 xcd=8 净负(`*_down heavy` −3%)、xcd=1@NT cyc −0.43% 但 wall 中性、gn>0@K=5760 惰性(K=2944 上 gn 才是 −1.6~7.2% 的活 lever,见上方 2D band 段)。

### ⚠️ 静态 ISA 直方图会被「边界象限变体复制」灌水，只有 steady-state 窗口可比
- 现象：mxfp8 wgrad 的 "mainloop" 段 7581 条指令 / 1152 MFMA = **6.6 条/MFMA**，NT 只有 4546 / 1104 = **4.1 条/MFMA**，看上去 wgrad 主环多背 60% 的地址算术。
- 真相：wgrad 有 **4 个边界象限变体**（M+N 两侧）、NT 有 2 个（半-N），tile 体在 ISA 里被复制了 4 份/2 份，而**每次执行只走其中一份**。取 `mfma[8..40]` 的 steady-state 窗口后：wgrad 106 条/33 MFMA = 3.2，NT 111/33 = 3.4，**两者其实相当**。
- ⇒ round 9 的"引用 ISA 计数前必须按 region 分段"要再收一层：**分段之后还要看 steady-state 窗口**，`[mfma0, mfmaN]` 整段包含所有谓词变体，是静态计数不是动态计数。

关键区分（别把制胜招误判成死路）：
- ❌ 别再试 **大 BM 但不合并** 的 quant 写（旧 v2 BM=128 各 lane 写满 128B 无跨-lane 合并）。但**大 BM 配合 LDS 合并才是制胜招**——死的是"大 BM 不合并"，不是"大 BM"本身。
- ❌ 别再试 **scale 也暂存 LDS**（旧 v3 双趟 scale 暂存 LDS 慢 0.42–0.61×）。教训：**只暂存 fp8 数据、不暂存 scale**，且让两半并发（不串行）才能反超。

## MXFP8 e8m0 scale 广播必须全程 Int32（提前 cast Uint8 高位损坏静默错）

- **最关键的 MXFP8 authoring bug**：e8m0 scale 做 OR/SHIFT 广播时，若提前 cast 到 `Uint8`，高位会损坏——广播出的 4 个字节各不相同，match 仅 **~9%**，结果是数值垃圾（静默错，不报错）。
- **正确做法：全程保持 Int32**，位运算前绝不 cast：
  - `e8_i32 = ep + I32(127)`（范围 0..255，**不 cast**）
  - `bcast = e8_i32 | (e8_i32<<8) | (e8_i32<<16) | (e8_i32<<24)`
- **通用规律**：flydsl 里凡做位运算（OR/SHIFT 广播等）不要提前 cast 到 `Uint8`；Uint8 只有 8 位，左移 8/16/24 会把有效位移出、高字节全 0 或截断，导致每字节不一致。
- ❌ 别再试：先 `Uint8(e8m0)` 再做 `| <<` 广播——high-byte 损坏、match ~9%、纯垃圾。失败机制=8 位容器承不住 <<8/<<16/<<24 的位移。

## FlyDSL kernel 与 torch 同模块触发 JIT 依赖收集 RecursionError

- **症状**: FlyDSL kernel 文件和 torch/测试代码放同一模块 → JIT 依赖收集阶段触发 `RecursionError`。
  - **WHY**: FlyDSL 的 JIT 在编译 kernel 时会递归收集模块级依赖，同模块内的 torch import 让依赖图爆栈。
- **解法**: 把 kernel 放**独立干净模块**（如 `mxfp8_quant_flydsl.py` 只 `import flydsl`，**不** `import torch`），只让 harness 脚本 `import torch`。
  - kernel 模块保持纯净 → JIT 依赖收集不再递归到 torch。
- **区分维度**: 这是**模块隔离**维度的坑，区别于 pitfalls/07-flydsl-frontend-tracer.md（tracer 字面 if/for）与 pitfalls/11-porting-encoding.md（编码）。同为 JIT 相关但根因不同。

## grouped fp8-tensorwise NT(fwd/dgrad)Shape A big-N/short-K:latency-bound 诊断 + 死路/活杠杆

内核 = `gemm_fp8_grouped_kernel.py` 的 `_compile_grouped_nt`(public `grouped_gemm_fp8_tensorwise_flydsl_kernel`)。**Shape A = gpt-oss gate_up fwd K=2880 N=5760**(big-N/short-K,E=32,M_TOTAL=131072);Shape B = down fwd K=5760 N=2880(long-K/small-N,已强)。8-wave(WG=512)、persistent 路径。

### ★ gfx950 统一 vmcnt 真机勘误(2026-07-28 llvm-mc gfx950 实测,推翻别处误记)
- gfx950 **只有** `s_waitcnt vmcnt(0)` + `s_waitcnt lgkmcnt(0)`;`s_wait_storecnt/s_wait_loadcnt/s_wait_dscnt/s_wait_kmcnt`(gfx12 助记符)与 `s_waitcnt_vscnt`(gfx11 助记符)**全部 `instruction not supported`**。
- ⇒ load 与 store 共用 vmcnt,**无法"只等 load 不等 store"**;把 store drip 进主循环必被 vmcnt 串行化。这是"store 藏在 load 后"整类死路的硬件根因(与本卡 §mxfp4 4-wave 胖形状 store 暴露一致)。methodology/05 §waitcnt 配套曾误写 gfx950 有分离计数,已修正。

### ★ Shape A 是 latency/dependency-bound(不是 compute/feed/bank-bound)—— PMC 判据
rocprofv3 派生指标(crsuse2-m2m-328,200 iter balanced,NT kernel):
| 指标 | 值 | 读法 |
|---|---|---|
| MfmaUtil | **44-48%** | MFMA 单元空闲 ~55% → **非 compute-bound**,有大量 MFMA 产能可填 |
| MemUnitStalled | **0.02-0.09%** | 内存管线几乎不 stall → **非 HBM/feed-bound** |
| LDSBankConflict | **0-2%** | 可忽略 → swizzle 是红鲱鱼(铁律),别碰 |
| VALUBusy | 11-13% | 低 → 非 VALU-bound |
| MeanOccupancyPerActiveCU | ~2.0 waves/SIMD | = 8 waves/CU = **1 WG/CU** |
- 资源:VGPR=120-128、SGPR=112、Scratch=**0**(无 spill)、LDS=**131-139KB/WG**。
- **占用率被 LDS 锁死 1 WG/CU**(160KB ÷ 131KB = 1;VGPR 128 本可容 4 waves/SIMD,LDS 才是 binding)。第 2 个 WG 需 LDS≤80KB → 抬 occ 死路。
- 综合:8-wave 协作**同一 tile lockstep** → 全部同时撞 epilog → cshuffle 串行链**完全暴露**;MFMA 空闲一半但内存不 stall = 在等依赖链(operand latency + epilog lgkmcnt)。
- ⚠ **MemUnitBusy 在 gfx950 rocprofv3 1.1.0 不存在**(只有 MemUnitStalled);别请求,会 "Unable to find counter"。

### 死路(勿重试)
- ❌ **PT_NT_PFETCH prefetch-forward**(把下一 tile 的 g2s 提前到本 tile epilog 藏 store):真机 byte-exact 但**净负**(bal −0.78/skew −1.75/empty −1.65)。根因=fill(g2s)与 store 同占 **vmcnt** 无 overlap(见上 gfx950 统一 vmcnt)。已回滚干净。**暴露成本是 cshuffle 的 ds 串行链(lgkmcnt)不是 fill。**
- ❌ store 藏在 load 后(任何 flavor):gfx950 统一 vmcnt,同上。
- ✅ **但"store 藏进 MFMA"是活的,别连坐**:store 与 mfma 不共计数器,把 C-store 插进**最后一个 K-block 的
  mfma 之间**(peel-last,见 methodology/07)在 occ=1 grouped fp8 wgrad 上实测 **+0.7~1.1%**。判负的是
  "藏进 fill/load"(共用 vmcnt)与"store 之间互相重排"(TCC 写合并对发射相邻不敏感),不是"藏进算力"。

### ✅ 已交付杠杆:monotonic group-carry(commit 4e15c6ce),增益 ∝ E/K
- persistent 路径把 per-tile O(G) group-find scan 换成 LDS tiles-prefix-sum + forward-only carry(`PT_NT_CARRY=1`);ds_read 只在**真跨组边界**发生(摊 <1/tile),把 group-find 移出 feed-bound 的 tile 路径。
- **before/after 自比(drift-immune)**:gpt-oss Shape A **+7.7~9.6%** / Shape B +2.0~2.8%(byte-exact SNR 69.4)。跨模型 **增益 ∝ E/K**:Kimi-K2 down **+107.84%**、DeepSeek-V3 down +78%、Qwen3-30B +47~55%;**Mixtral E=8 ≈0%**(专家太少,autotune 选同 config,mism=0 ratio~1.0 是合法而非碰撞)。→ 高-E MoE 的主优化。

### ✅ 剩余活杠杆(目标 Shape A ×1.20 vs ASM,当前 ×1.10):epilog cshuffle lgkmcnt 链 与 下一 tile MFMA overlap
- 天花板≈**+14.6%**(暴露的 epilog),能把 ×1.10 → ×1.26 越过目标。
- 机制:cshuffle = `ds_write → s_waitcnt lgkmcnt(0) → ds_read → buffer_store`。lgkmcnt 链与 **MFMA 正交**(MFMA 既不占 vmcnt 也不占 lgkmcnt),且 PMC 证 MFMA 有 55% 空闲产能 + 内存不 stall → 把下一 tile 的 K-loop MFMA 灌进本 tile epilog 的 lgkmcnt 空窗理论可赢。**这与 PT_NT_PFETCH 死路机制不同**(那个藏的是 store 的 fill/vmcnt,这个藏的是 epilog 的 lgkmcnt-drain 到 MFMA)。
- 难点:tile N+1 的 MFMA 需先做 prelude(g2s→barrier→s2r),g2s 又占 vmcnt 与 store 竞争(chicken-egg);且复用同一 accum VGPR + LDS。需软流水重构 persistent 循环。**未验证,edit→drift-immune A/B 定夺(铁律)**。
- 现状代码坑:K-loop 里每个 MFMA 后有 `rocdl.s_barrier()`(4/iter,8-wave 全 WG 同步)为 LDS ping-pong 正确性;`_ibar()`(MFMA 前)可用 `sched_schedbar=True` 换成编译期 `sched_barrier(0)` 免运行时 sync。

### 度量方法论:drift-immune before/after 版本 A/B(fp8 DVFS ~37% 漂移)
- 提取优化前内核为独立模块:`git show <commit>~1:...gemm_fp8_grouped_kernel.py > _nt_base_kernel.py`(绝对 import 保持,helper 未变即可跑)。
- 同进程 import 两个 public entry(new/old),position-balanced interleave(每 rep 交替 a,b/b,a)。
- **碰撞守卫**:两版定义同名 flydsl 闭包,若 JIT 缓存串了 old 会跑 new 的 kernel → byte-identical **且** ratio~1.0。判据:`mism==0 且 ratio≠1.0` = 真 byte-exact 调度胜;`mism==0 且 ratio~1.0` = 疑碰撞(或 autotune 选同 config,如 Mixtral E8)。
- 探针:`_probe_ver_nt.py`(Shape A/B)+`_probe_ver_models.py`(任意 E,K,N)+`_prof_nt_shapeA.py`(PMC 单核启动器)。高-E(E≥256)`num_cu=-1` autotune 合成探针 M_c=G×16384 会 OOM → `pick_numcu(E)=256` 走 persistent 跳过 autotune(见 [[project_grouped_fp8_autotune_oom_highE]])。


## ❌ tw fp8 TN wgrad:动"变体的 fragment 足迹"必炸 regalloc(2026-07-31 复现两次)

`_wholeloop_asm_3buf` 一个 kernel 里同时存在 4 个变体 ((2,2)/(1,2)/(2,1)/(1,1)),
每个变体用 `=&{v[...]}` 钉死 `ntmp` 个 8-VGPR 片段。**只要新增一种 ntmp 取值,整核 spill 爆炸**:

| 改动 | ntmp 组合 | `vgpr_spill_count` | `private_segment_fixed_size` | 结果 |
|---|---|---|---|---|
| 基线 | {16,12,12,12} | **0** | 0 | gm 0.9864 |
| b_halves=1 整池删除(含 ds_read) | {16,12,12,**8**} | **122** | 464 B | gm 0.9604(**−2.6%**,全 tile 变慢) |
| 只删该池 g2s(足迹不变) | {16,12,12,12} | 9 | 40 B | gm 0.9849,wgrad 组持平 |

- **规则**:边界变体只能删 **MFMA + 累加器 + g2s**,**ds_read 必须保留**(它决定足迹)。
  与 r2 的"只删 MFMA"结论一致,这里补上机制:是 `ntmp` 的取值集合,不是删了什么。
- ⇒ 想做 **128×512(5 池, ntmp=20)** 的边界变体,**不要塞进同一个 kernel** —— 那是第 5 种足迹。
  正确做法:**独立 `@flyc.kernel` + 第二次 dispatch** 只跑 M 边界条带,主核变体集合原样不动。

#### ⚠️ 勘误加强(2026-08-01 实测):不是「ntmp 取值集合」,是「任何变体的池数变化」
上表把根因写成"新增一种 ntmp 取值"。按这个规则做了一次**保证 ntmp 集合不变**的手术:
只给 `(2,1)`(半 N 且 a 满)变体删掉死 b1 池 —— `tiles=[nta,nta,ntb]` ⇒ ntmp=12,
与既有 `(1,2)` 变体**完全同值**,集合仍是 {16,12,12,12}。实测:

| | `vgpr_spill_count` | `private_seg` | gm | wgrad gate_up | wgrad down |
|---|---|---|---|---|---|
| 基线 | **0** | 0 | 0.9847 | 0.965 | 0.955 |
| (2,1) 删死池(ntmp 集合不变) | **16** | 68 B | **0.9573** | **0.854(−14.6%)** | 0.937(−6.3%) |

`gate_up` 的死池 tile 只占 12/276,却掉 14.6% ⇒ **损失是全局的**(spill 打在每个 tile 上),
不是边界变体自己变慢。半 M/N 对拍 SNR 55.1~56.7dB、det=True、ndiff=0,纯性能判负。
- ⇒ **规则改写**:`_wholeloop_asm_3buf` 的四个变体共享一次寄存器分配,**只要任一变体的
  `n_pools` / ds_read 序列变了,整核 spill 就会动**,ntmp 数值是否撞车无关。边界变体能
  安全删的只有 **MFMA + 累加器**(r2 结论),g2s 可删但只值 ~0.2%(r10)。
- ★ **修正(2026-08-03,mxfp4 grouped 实测)**:这条卡里承重的是「**ds_read 必须保留**」那半句
  (足迹 = `ntmp` 取值集合)。「g2s 只值 ~0.2%」是 **tw wgrad 专属,不可外推**:mxfp4 grouped 的
  半-N 体同时删了 MFMA **和**一整组 BR g2s,`vgpr_spill_count` 仍是 **0**(vgpr 428→432、
  sgpr 58→59),而 g2s 那一半值 **好几个百分点**。判据是「变体之间 ds_read 序列有没有变」,
  不是「删了 MFMA 还是删了 g2s」。
- ⇒ 同时把这条钉死:**tw wgrad 的 N 边界"死池"不是开口**。三种删法各自实测 —— 只删
  ds_read(r2)≈0、只删 g2s(r10)≈0.2%、两个连池一起删(本轮)−2.8% gm。别再回来。
- 附带:删掉死池的 global load(占该 tile 1/4 的 LDS-fill 流量)实测只值 ~0.2%,
  远低于按"全 tile 强制"探针线性外推的 0.9% —— 见 methodology/06 的成本模型告警。

## ✅ 已交付杠杆(2026-08-01 实测):边界体「少做功」若逼出更差的同步表,就别少做功

`_compile_grouped_nn` 的半 N 边界体有两种形态(N=2944 ⇒ 最后一个 N-block 只有 128 列有效,
占 1/12 的 tile):
* `nn_halfn_noload=True`(旧默认,`nq==0`):连 b1 的 g2s + s2r 一起删 ⇒ 每迭代只发 3 组 g2s,
  同样的 in-flight 允许量对它就不够(见下一条,`_nd=4/2` 两个值都 racy),只能退回
  **每 K-iter 一次 `vmcnt(0)` 全排空**。
* `nn_halfn_noload=False`(新默认,`nq==1`):**保留** b1 的 g2s/s2r,只删 c01/c11 的 MFMA 与 store。
  g2s 发射序列与满体逐条相同 ⇒ 直接继承满体的 graded 逐消费者 drain(`_nd=2*(NSA+NSB)`)。

实测(scored bench,其余不变):dgrad 组 geomean **0.98897 → 0.99399(+0.5pp)**,gm 0.99372 → 0.99600,
`dgrad down balanced` 0.992→0.998、`gate_up heavy` 0.999→1.000。det:`_ck_nn_det 500` × 4 个宽 shape
= 2000 run,mismatch 0,SNR 55.1~56.3dB。
- ⇒ **规则**:边界体删工作量之前先问「删完之后它还能不能用主体的 drain 计数」。删掉一整组 g2s
  会同时削弱 vmcnt 的保护力,被迫全排空时,省下的 HBM 流量远不够赔。
- 同族反例已在树里:tw wgrad 的 `b_halves=1` 保留死池的 g2s/ds_read(删了炸 regalloc),
  也是「少做功不一定赢」的另一种成因。

### ★ 修正(2026-08-03,mxfp4 grouped 实测):删一组 g2s **不必**退回全排空

上面「删掉一整组 g2s ⇒ in-flight 允许量不够 ⇒ 只能每 K-iter `vmcnt(0)` 全排空」在
`mxfp4_grouped_kernel` 的 NT/wgrad whole-loop 上**不成立**。正确做法是把 phase barrier 的
partial `vmcnt` **按被删 load 组的条数等量收紧**:满体 `vmcnt(_WLV)`,半-N 体
`vmcnt(_WLV - nsb)`(nsb = 被删的 BR g2s 条数),`lgkmcnt` 与 ds_read/barrier 序列逐条不动。
这样 graded drain 完整保住,g2s 的删除立刻兑现 **gm +3.8% / min +4.7%**,det=True。
- ⇒ 规则应写成「**删一组 g2s 必须同时把 vmcnt 减去该组条数**」,而不是「删 g2s 就得全排空」。
  只有在减完之后 vmcnt 仍无法表达「下一个消费者需要的那条已落地」时才退回全排空。

## ✅ 已交付杠杆(2026-08-06 实测,mxfp4 grouped whole-loop):C-store 的 fold 窗口取决于**尾相位有多长**,奇数 K-block 数必须也剥一次

whole-loop 把 C store 折进「不发 g2s 的尾相位」之后(gfx950 统一 vmcnt:只要该相位没有在飞的
g2s,在飞的 store 就不会把某个 g2s 的 wait 拖长),真正决定收益的是**那个相位有多少条 MFMA
供 store 骑**:
* 偶数 trip count(K=2944 ⇒ 23 个 128-K 块 = 11 对 + 半块):剥掉最后一对 ⇒ 尾相位 = 剥出的
  phase A+B 合并 = **192 条 MFMA** 的窗口。
* 奇数 trip count(K=5760 ⇒ 45 个 128-K 块 = 22 对 + 半块):旧写法让那个半块**独立**成一个
  尾相位 ⇒ 窗口只有 **64 条 MFMA**,大半个 store 突发照旧吊在流尾。

修法(单一改动):奇数时同样把最后一对剥出来,但**phase A 保持完整**(它的 g2s 正是把尾部半块
搬进 LDS 的那一次),只把半块并进它后面那个相位 ⇒ 奇数也拿到 192 条窗口。两处 bookkeeping 会
整体错开一格,必须一起翻:
1. **LDS ring slot**:最后一个满块留在哪个 slot 由剥出的 phase A 决定(偶数 sw=0、奇数 sw=1),
   尾部半块在另一个 slot。
2. **scale ping-pong set**:同理整体互换(偶数 满块=set0/半块=set1,奇数反之)。
只翻其中一个 ⇒ 数据串位,SNR 直接负值。

实测(scored bench,18 配置):受影响的 3 个配置(dgrad gate_up,K=5760 是唯一静态奇数 KI 的形状)
mx4 延迟 1.019/1.021/1.023ms → 0.995~0.999/1.009/1.007ms = **−1.1%~−2.0%,两轮复现**;
gm_tf 4023.0 → 4031.7 / 4038.6(+0.22~0.39%,只有 3/18 走到改动的代码)。
det `_probe_det_wide.py 200` × 12 配置 mismatch 0,SNR 49.6dB。ISA:num_vgpr 205→201、
num_agpr 256、spill 0、LDS 163840 不变;代价是静态 MFMA 448→864 条(两个 N 体各多一份 phase A),
kern_1 ≈3.0k 指令仍在 32KB L1I 内。
- ⇒ 规则:**判断 store fold 有没有兑现,要数「fold 进去的那个相位的 MFMA 条数」,不是看
  "store 已经在尾相位里"**。unroll-2 的循环里,trip count 的奇偶会让这个数差 3 倍。
- ⇒ 附带收益:奇数体不再为不存在的下一个块发一整轮 g2s(16 条),半-N 体也不再跑 64 条死的 R 半 MFMA。

### 🛑 补记(2026-08-08):上面这个融合 peel **漏了 trailing 半块的 scale 重发**,错了 19 轮

融合把 phase A+B 合成一个 peel 的时候,**未融合体在 phase A 里做的那次 `emit_sc_vgpr` 一起被删掉了**。
循环交棒时最后一次预取落在 peel **第一个**要消费的那套里,于是 trailing 半块乘的是两个相位前的 E8M0。
* 实测:去掉重发 = fwd/dgrad 计分形状 **10.98~14.00 dB**;补上 = **49.60 dB**。wgrad 走 `emit_peel_rt`,
  自带预取,不受影响。
* 三道门全放行:bench 的 SNR 形状 `K%256==0` 走不到折叠路;det 门看 run-to-run 一致而错是**固定**的;
  SNR 探针当时用 `randn` 量化出 scale ⇒ 每块 E8M0 指数近似恒定 ⇒ 拿错那套数值相同。
  **只有随机 E8M0 指数的探针能看见**(见 `methodology/02` 同名新节)。
* 代价:补发那 2 条 `buffer_load_dwordx4` 的裸露延迟 = **gm −0.49%**(交错双 session,4083.9→4063.8),
  **不是可选项**。位置已扫死:入口排空**之前**发更差(−0.2%,那时还有 ~10 条 g2s 在飞,它排在后面返回);
  barrier 之后发 + peel 内改 **sub-step major**(把 `vmcnt(0)` 推后一整个子步 pass = 16 条 MFMA)最好。
* ⇒ **通用判据**:任何"把两个相位融成一个体"的改动,必须**逐项清点被合并掉的 prefetch**
  (scale 套 / operand 环槽 / 地址前进),融合体不会自动继承未融合体在前一相里做的准备工作。

## ❌ tw fp8 wgrad:XCD gather 在**组内**做也是死路(2026-08-01 实测)

想拿 `num_xcd>1` 的 L2 聚簇又不吃 skew 的亏:把 `xcd_remap_pid` 从全局 tile 序改到
**组内 local 序**(`local = idx % TILES_PER_GROUP` 之后再 remap),这样每组 tile 仍在同一个
launch 窗口里(group-major LPT 完整保留),但每个 XCD 拿到 (m,n) 网格上连续的一条 band。

| | gm | wgrad down balanced | wgrad gate_up heavy |
|---|---|---|---|
| 基线(xcd=1,round-robin 摊开) | **0.99372** | **0.962** | **0.995** |
| 组内 xcd_group=8 | 0.98762(**−0.6%**) | **0.941(−2.1pp)** | 0.965(−3.0pp) |

⇒ **XCD 连续分块对本核在两种形态下都是负的**,不是"全局版被 skew 毁了"那么简单:
round-robin 把 8 个 XCD 同时压在**同一条 band** 上,共享的 MALL 命中率更高;连续分块让 8 个 XCD
各自流一条不同的 band,MALL 工作集×8。与 PMC「tw wgrad 不缺带宽(L2 命中 66.9%)」一致 ——
它缺的是**延迟**,而并发触碰同一份数据正是最好的延迟对策。别再往 XCD 重排上花轮次。

## ❌ tw fp8 wgrad:`group_m=2` 已不再是 balanced 上的赢家(2026-08-01 复测,推翻旧记录)

本卡与 campaign r4 记过「gm2 在 balanced/moderate/heavy 三种分布上都更快,只在 autotune 的
`_canon_skew_offs` 上慢 3.9%,所以被 worst-of-two 判据挡掉」。今天把 `_WGRAD_4WAVE_CANDS`
强制成 `((2,4,1),)` 直接量:

| | gm | wgrad down balanced | wgrad down heavy |
|---|---|---|---|
| gm4(现候选表 cand[0]) | **0.99600** | **0.966** | **0.981** |
| gm2 强制 | 0.99203 | 0.950(**−1.6pp**) | 0.966(−1.5pp) |

⇒ 经过 r2~r7 的边界象限跳过 / K-block 地板 / drain 膝点几轮改造后,**gm2 在 scored 分布上也输了**。
autotune 拒绝 gm2 现在是**凭实力**,不是被 skew 守卫误伤 —— 别再为了"救 gm2"去动 race 的负载集。

## ❌ tw NN dgrad 的 half-N-noload 体(nq==0)不能用 graded drain

r8 把 graded 逐消费者 vmcnt 用在满体上 +0.7pp,但 `nq==0`(丢 b1 载入的边界体)
**每迭代那一次 `wait_barrier(0)` 是承重的**:它每迭代只发 3 组 g2s(满体 6 组),
同样的"after-count 减一个 issue group"推导给出 4,实测 `_ck_nn_det 400`:

| `_nd` | N=2944/K=2944 | N=2944/K=5760 | mg=512 |
|---|---|---|---|
| 4 | 0 | **2** | **12** |
| 2 | 0 | **24** | 0 |

两个值都 racy(且非单调)。**别再放宽它**;g2s 组数少 ⇒ 同样的 vmcnt 计数给的保护更弱。


## ❌ tw fp8:主环 standalone `s_waitcnt vmcnt(N)` —— NN 上能删,NT 上不能

r3 在 **NN dgrad** 主环删掉每 K-iter 那条独立 `s_waitcnt vmcnt(3)`(`nt_vmcnt=-1`),理由是
per-iter rendezvous 已覆盖全部 LDS 读、缓冲是距离-2,实测 dgrad 组不劣且机制干净,已保留。
2026-08-01 把同一个单值改动搬到 **NT fwd**(`_autotune_np_dispatch` 的 `mk()`,`trans_b` 分支):

| | fwd 组 geomean | gm |
|---|---|---|
| `nt_vmcnt=3`(基线) | **1.0119** | 0.98813 |
| `nt_vmcnt=-1` | 1.0099(**−0.2pp**) | 0.98631 |

⇒ NT 主环那条 drain 是**承重的**,NN 的判定不跨核。两个核的 rendezvous 结构不同
(r8 已记:NT 是均匀距离-2、NN 有一个距离-1 的 A1 池),所以"哪条 drain 冗余"必须逐核实测。
同族的 r8 也已证反向不成立:graded 逐消费者 drain 在 NN 上 +0.7pp、搬到 NT 上 −0.3pp。


---
来源: 08-att-root-cause.md, 04-ceiling-analysis.md, project_mxfp4_k28672_ceiling.md, project_mxfp4_epilogue_store.md, flydsl-fp8-gemm-results/SKILL.md, 10-grouped-wgrad-4wave-3buf.md, mxfp8-grouped-gg-devloop/SKILL.md, 08-deadends.md, 05-dead-ends.md, 12-llama-aiter-baseline.md, mxfp8-8wave-devloop/SKILL.md, project_mxfp8_wholeloop_port.md, project_mxfp8_grouped_wgrad_wl.md, project_gptoss_nt_fwd_dgrad_opt(2026-07-28 PMC/gfx950-waitcnt 实测)


## ❌ tw wgrad C store 的非临时/绕 L2 提示:全面大幅净负(2026-08-01 实测)

`StoreCPerTensor(store_aux=)` 在 `kernel_grouped_tn_wgrad_4wave` 上扫 aux in {0,1,2,3,16,17},
两个形状 x 三个分布(均 ndiff=0,纯性能判负):

| aux | 语义 | down bal | gate_up bal | 结论 |
|---|---|---|---|---|
| 0 | 默认 | 1.000 | 1.000 | — |
| 1 | sc0 | 1.002 | 0.999 | 中性 |
| 2 / 3 | nt / sc0+nt | **1.090** | **1.085** | 净负 ~9-11% |
| 16 / 17 | sc1(系统作用域) | **1.179** | **1.177** | 净负 ~18-21% |

根因(比本卡旧条的"partial-line 事务暴增"更具体):MFMA 16x16 累加器的 lane 布局使
每条 `buffer_store_short` 产生 **4 段 32B 的半行写**,靠同一 wave 相邻 4 个 `tj` 的存在 **L2 写合并**
成整行,才能把 555MB 以 ~6 TB/s 发出去。任何 nt/sc1 提示削弱该合并 ⇒ 半行写直达 HBM。
⇒ **凡是"每 lane < 128B/16-lane 连续"的 store,L2 写合并是承重的,不得加非临时提示**;
这也坐实了 goal 里"NN dgrad 的 cstore_aux=1 配标量 store 预期同样负",不必再测。

## ❌ 探针反例:把所有 WG 的 store 指向同一块输出以"消除内存流量"

想把"epilogue 的 10% 到底是字节还是发射"分开,于是去掉 `base_row/base_col` 的 tile 偏移,
让 4608 个 WG 写同一个 256x256 区域 ⇒ wgrad down balanced **0.909 → 1.093ms(慢 20%)**。
量到的是**写竞争/同行串行化**,不是"无流量基准"。这类分解必须保持地址分散,
只能改值(存常量)或改宽度,别改地址。

#### ❌ 2026-08-01：整条"压低 SQ_INSTS_LDS / DS-per-MFMA"路线在 tw grouped fp8 上判负（三次独立实测）

goal 一直把 tw 相对 mx 的 `SQ_INSTS_LDS` 超额（dgrad 1.324×、wgrad 1.405×）当成主开口，
本轮用三个互相独立的实验证明**它根本不在关键路径上**：

1. **DS/MFMA 0.75 的结构已经在树里，而且更慢。** ratio = `32/n + 64/m`（per-wave tile m×n，
   B 侧 8B/lane tr 读）。要到 0.75 必须 `m·n=16384` ⇒ **256 AGPR/wave ⇒ 1 wave/SIMD**。
   8-wave WG 需要 2 waves/SIMD（≤256 reg），所以 **0.75 与 occupancy 直接互斥**，
   `BLOCK_M=512` 的 8-wave 形态还额外撞 LDS（A 4 半区×2 stage=128KB + B 64KB=192KB > 160KB）。
   唯一可行载体 = 已存在的 4-wave `_compile_grouped_nn_4wave`（128×128 wave tile，DS/MFMA 0.75）：
   实测在四个计分点全输 **3.0 / 5.5 / 5.5 / 6.6 %**（最好的 xcd4/gm8）。
   ⇒ **别再为"更方的 wave tile"立项**；DS 少 25% 换不回 occ 从 2→1 的损失。
2. **直接删 25% 的 ds_read = 0.0%。** NN dgrad 半-N 边界体（1/12 tile）的 b1 transpose 读只喂
   被跳过的 c01/c11，是死读；删掉后 dgrad 四点 geomean **1186.5 → 1187.5 us（噪声内）**。
3. **再把那批 g2s 的地址改指向 b0（省掉 25% 的越界 HBM 读流量）= 0.0%**（1186.6 us）。
   ⇒ 这族核既不是 LDS-指令 bound 也不是带宽 bound。与 methodology/05 的
   「消 conflict ≠ 变快（红鲱鱼判据）」同一模式，只是换成了指令**条数**。

#### ★★ tw NN dgrad 的 8 barrier/K-iter 是**双向**局部最优（2026-08-01 实测）

主环每 K-iter 4 phase × 2 barrier。两侧都量过：
- **全删 8 个 barrier**（只留 vmcnt drain，时序探针）→ dgrad geomean **1186 → 1273 us（−7.3%）**。
- **每 phase 在 s2r 读与 g2s 写之间再加 1 个 barrier**（+3 个，wave-uniform、无 race）→ **1234 us（−4.0%）**。
⇒ barrier 在 occ=2 waves/SIMD 下是**波相位对齐装置**（同 SIMD 两波错开 MFMA/LDS），不是开销。
r3 的 `sched_schedbar`（−1.4%）只是这条曲线上的一个点。**别再动 dgrad 的 barrier 数量。**
对照：wgrad（4-wave、occ=1、4 波在 4 个不同 SIMD 上）删掉每-phase barrier 只值 **0.27%**，
加一个 phase 中点 barrier 反而 −0.22% ⇒ **"多同步"不跨占用率迁移**。

#### ★ tw NN dgrad 的 per-tile prologue rendezvous：减法上界只有 0.21%

把 `_w1/_w2` 放宽到 `vmcnt(63)`（数值错，仅计时）→ 1186.5 → **1183.9 us**。
⇒ 非持久化 dgrad 每 tile 一次的 prologue 全局往返已被掩盖，**不值得为它建跨 tile 预取**
（wgrad 侧 r13 已得同结论）。

#### ★★ mxfp4 grouped 的 preshuffle 核是 VALU-issue bound：per-thread O(G) 扫描要换 lane-resident 表（2026-08-03 实测）

`mxfp4_grouped_kernel` 的 scale preshuffle（`kern_0`，A/B 两条 slab 合在一个 grid 里）每个线程只搬
**8 dword**，却曾经跑 **759 VALU/wave**：它为了把 A-slab 的 64 行块映射回源行，在**每个线程**里跑一遍
`G=32` 的串行组扫描（每组一次 group-offset load + 一组 select）。ATT 记账：整核 15.9M VALU insts，
其中扫描占 ~92%。
换成 GEMM prologue 早就在用的 **lane-resident 表**（lane g 拥有组 g，一次 wave 内 inclusive scan +
两次 O(1) 查表；`wi` 由 `gid/nd/16` 结构性 wave-uniform，所以每个 wave 只有 `2*wi`、`2*wi+1` 两个查询）：
**VALU 759 → ~206/wave，kern_0 35.34 → 16.32 µs（−54%），全 18 配置 gm +1.1% / min +2.0%，SNR/det 不变。**
判据(可移植)：**preshuffle/量化这类"每线程搬几个 dword"的核，先用 `SQ_INSTS_VALU/wave ÷ 有用 dword 数`
看比值**；>10 VALU/dword 基本就是索引/扫描逻辑吃掉了核，别去优化搬运路径（宽存、LDS staging 都白搭，
后者在本核还出过 nan）。同一份 `_lane_tbl_*` helper 在 GEMM prologue 与 preshuffle 两处复用。

#### ❌ 别再试：mxfp4 grouped whole-loop 里"拉长 ds_read→MFMA 依赖距离"（含寄存器级 operand 双缓冲）

三条独立证据，方向一致：
1. **物理上装不下。** 4-wave whole-loop 用 VGPR 432（arch 176 + AGPR 256），unified 512 池只剩 **80**；
   一组 operand fragment（A 8 行 + B 8 列 × n_sub，b128）= **128 VGPR** ⇒ 寄存器级双缓冲必然 spill，
   而 spill≠0 会让手写 partial `vmcnt(N)` 语义失效（pitfalls/04）⇒ 直接连带 race。
2. **本来就没有气泡可减。** ATT 逐指令：ds_read stall **0.05–0.12 cycle/条**（整 phase 4.9 cycle
   ≈ in-loop 开销的 0.4%）、in-loop `s_waitcnt` 4.0 cycle；紧跟内存指令的 MFMA stall 0.28。
3. **零寄存器成本的等价探针也判负。** 改 `emit_inplace` 的 4×8 cell 发射序（等价于改每个 fragment
   的 last-use→refill 距离，不多花一个寄存器）：column-major **±0.15%（平局）**、diagonal
   **+0.7~1.2%（更慢）**、cyclic **+0.0~0.3%**。⇒ 现有 blocked-diagonal 序已在最优附近。
**下一个具体杠杆不在 operand feed 上**：ATT 把 in-loop 开销指向 g2s 的**服务速率**背压
（round-4 已判负重排），F 侧则指向 epilogue 的 VALU 链条数（`cvt_pk` 配对）与 preshuffle。

#### ❌ 别再试：mxfp4 grouped NT prologue 的 partial-drain（`wait_barrier(N>0)`）

prologue 发 32 条 `buffer_load ... lds`（2 buffer × (A 8 + BL 4 + BR 4)）后 `vmcnt(0)` 全排空。
把发射序改成 **buffer-major**（buf0 全部 → buf1 的 A → buf1 的 B），再把屏障放宽到只留 buf1 的
B 半区在飞（`vmcnt(8)`，其首个消费者在第一个 phase 第二个 4×8 block，≥1024 cycle 之后，结构上安全）：
**实测 −0.18% / +0.01%，即噪声内的平局**（`vmcnt(4)` 同）。
机制：这段的成本是**issue backpressure 而不是 tail latency** —— ATT 量到 36 条 g2s 的 issue stall
lat/hit **119 cycle**，32 条发完本身就 ~3.8k cycle，等最后一条落地几乎不额外花钱 ⇒ 放宽屏障没有可省的量。
要动只能**不发**（把 buf1 的 g2s 推到第一个 phase 之后），那等于把流水深度砍回 1，与 clean
double-buffer 已判负（0.65–0.82x）撞车。★ round-1 记的 `wait_barrier(16)`「方向不一致
（−1.9%/+1.6%）」也由此结清：那个变体在**旧的 A-major 发射序**下 `vmcnt(16)` 会让 BL0/BR0 还没落地
就被 asm prologue 的 `emit_ds(0,0)` 读走 —— 它是个 **racy 臂**，那两个数字不可用。

## ✅ 已交付杠杆 + ⚠️ 拟合阈值:mxfp4 grouped wgrad 在 token skew 下的塌方与修法(2026-08-12 m4096 UNBALANCE 场)

**现象(修之前)**:G=32、总 token 恒定、只改分布,wgrad 的 deploy/large-M 比值随偏斜单调崩塌 ——
balanced 0.94 → canon(1.07^i,10×) 0.52 → moderate(36×) 0.31 → **heavy(159×) 0.22**;
绝对 TF 从 ~3800 掉到 **~870**,而 large-M 参考稳在 3800–4100。**fwd/dgrad 基本免疫**(heavy 甚至优于
balanced —— 组更少更肥)。三次读数纹丝不动 ⇒ 结构性,不是噪声。

**根因不是"负载不均衡"(我一开始猜错了)**:wgrad 输出是 `[G, OUT_M, OUT_N]`,**字节数只跟 G 有关、
与 token 分布无关**。heavy 下 24 个组各只有 512 token,却照样每组写满一整块 C ⇒ 算力/写出比塌掉,
整个 kernel 变成 store-bound。所以偏斜越深提升越大(见下:heavy +311% > canon +77%)。

**修法(r2,+46.5% 一笔)**:短收缩 regime 换 tile blocking + 非临时 C store,
`_GMXFP4_WGRAD_CFG_SHORT = (4, 1, 6, True, ...)`(关掉 XCD de-interleave、N 方向分带、开 `cst_nt`)。
逐格:canon gu 2133→3971 TF(+76.5%)、mod gu 1288→4002(+194.7%)、**heavy gu 916→3969(+311%)**、
heavy dn 876→3854(+360.6%);**ref 侧不降反升**(4137→4363)⇒ 不是靠拖慢分母。
护栏 balanced 也从 89.1 抬到 99.1。(`nt` 位本身的作用域见本卡上面那条 2026-08-11 的作用域修订。)

### ⚠️ `_GMXFP4_WGRAD_SHORT_MG = 8192` 是**拟合值**,真实交叉点 ∝ 每组 tile 数(2026-08-12 扫描实测)

这个阈值恰好落在 bench 的 deploy(每组 4096)与 ref(32768)之间 —— 两端都判对了,**中间一大段全判错**。
固定 G=32,per-group M 从 2048 扫到 32768,每点强制三档各测一次(TFLOPS):

| M/组 | gate_up SHORT / SPAN / CFG | down SHORT / SPAN / CFG |
|---|---|---|
| 4096 | **4231** / 4163 / 3998 | **3927** / 3913 / 3765 |
| 8192 | 4648 / **4657** / 4648 | **4592** / 4526 / 4410 |
| 12288 | **4737** / 4615 / 4672 | **4743** / 4721 / 4644 |
| 16384 | 4761 / **4823** / 4713 | 4755 / 4789 / **4831** |
| 24576 | 4794 / **4800** / 4679 | 4845 / 4840 / **4937** |
| 32768 | 4433 / 4450 / **4786** | 4837 / 4829 / **4974** |

- SHORT→CFG 的真实交叉:**gate_up 在 24576–32768、down 在 12288–16384**,代码里的 8192 把切换点
  提前了一大截 ⇒ 8192–24576 这段本该走 SHORT 系却被送去 CFG,白掉 1–2%。
- 两个交叉点除以各自每组 tile 数(gate_up `ceil(2944/256)*ceil(5760/256)=276`、down `12*12=144`)
  分别是 **≈119 和 ≈114** ⇒ ★ **分界不是固定 token 数,而是正比于每组 tile 数**:
  `pm / tiles_per_group` 就是每个 tile 摊到多少 K 方向工作量,够大时每 tile 固定成本已摊薄,
  不再需要宽 band 抢 L2 驻留。判据应写成 `(M_total//G) > K * tiles_per_group` 而不是硬比 8192。
- ⚠ **`tiles_per_group > _N_CU` 那道门(SHORT vs SHORT_SPAN)同样缺数据支撑**:两档差距普遍 ±1.5%
  且随 M 来回换手(gate_up 276>256 本该恒走 SPAN,实测 4096/6144/12288 三点是 SHORT 更快;
  down 144≤256 本该恒走 SHORT,2048 与 16384 反而 SPAN/CFG 赢)。四臂交错 A/B(O/G/S/W ×3)里
  **gm 层判不出来**(基线自身三次就散 1.06,大于臂间差异),只有 **min_ratio 层干净**:
  SPAN 分档 +0.9~1.0、配对门单独用 −0.4~−0.9、两者合用 +1.7~1.8。⇒ 结论按 min 取,gm 别信。

### ★ 跨模型泛化:wgrad 这笔泛化,NT(fwd/dgrad)那笔不泛化
同一改动在三个 MoE 几何上对测(详表见 methodology/14):wgrad 六格全正(gpt-oss +7.6/+10.9%、
DeepSeek +5.8/+3.1%、Qwen +6.0/+0.7%);而 NT 的 deploy-only 非临时 store 只在 gpt-oss 拿到
+3.8~4.3%,DeepSeek 剩 +0.2~1.2%、Qwen −0.5~−1.3% —— 它的门 `mb >= num_xcds*xcd_span` 是按
gpt-oss 的 tile 数调的,换 H/I 就落到另一侧(**没生效,不是有害**)。
⇒ 判据只要写成"某个绝对 token 数/tile 数",就只对调它的那个模型成立;要泛化就得写成**比值**。

## ❌ 别再试：用 DPP/`v_perm` 把相邻两列**折进一个 dword** 来砍 epilogue store 发射数

2026-08-17 gpt-oss-20b fp8 grouped(NN dgrad / TN wgrad)实测。动机看着很硬:ISA 普查显示 epilogue
是 `buffer_store_short`(每 lane 2 B、160 条静态),跨 lane 已经完美合并(64 lane × 2 B = 一条 128 B 行),
所以**是发射条数问题不是带宽问题**,比 `dwordx4` 多 8×。实装 `DPP quad_perm` 交换 + 两条 `v_perm_b32`
打包,把相邻列对折成一个 dword(**store 发射直接砍半**),逐位相同 ——
**NN −1.09%(occ=2)、wgrad −1.9%(occ=1)**,已整套回滚。

- **根因**:每折一对要付 1 条 DPP(带 hazard)+ 2 条 VALU,而省下的只是 1 条 store 发射。
  反过来给 store 发射定了价:砍掉一半发射(≈ X/2)打不过 ~2×64 条 VALU ⇒
  **整核 store 发射成本量级只有 ~12 µs(1.5%)**,不是想象中的大头。
- 与本卡 §折叠 store 的**赢**案例(`_BILV=4` ⇒ `buffer_store_dwordx2`,16 lane 铺满 128 B 行,
  wgrad +2.1~2.4% / NT +0.88 gm)**不矛盾,但机制完全不同**:那边的相邻列**本来就在同一个 lane 里**
  (累加器 layout 决定),折叠是**免费**的,赢面还来自少污染 L2;这边的相邻列在**相邻 lane**,
  要折就必须跨 lane 搬,搬的代价超过省下的发射。
- ⇒ **开口只剩一个**:让相邻列**天生落在同一 lane**(改 MFMA→累加器→列的映射,
  或走 methodology/07 的 `permlane16_swap` 寄存器转置),即**不加 VALU 的宽 store**。
  在此之前不要再用任何 shuffle 类原语去砍 store 发射数。
  (LDS 中转的 `store_cshuffle` 在同一族核上是 **−21%**,更早已判负。)

## ❌ 别再试:split-K 三个"省 partials 字节"的方向(r9 全部实测为负)

督导方向是"攻 split-K partials 的字节量而非周期"。三条全部定价完毕,**三条都是负的**,
且负的原因不是字节没省,是**字节根本不是这里的约束项**。探针 `_probe/r9_split.py`:
monkeypatch `_mx_ksplit` 加 cap、复用生产门限、每臂清 autotune cache、所有臂先全部预热再计时、
回文臂序、量真实 `fwd + .backward()` step wall,per-kernel 拆分来自 chrome trace。

**(1) 给 `k_split` 封顶(4→2→1)❌**

| cell | arm | split(`nt_2`) | fold | split+fold | step wall |
|---|---|---|---|---|---|
| QKV | k=2(生产) | 62.5 | 8.9 | 71.4 | 705.2 |
| QKV | cap 2(同一计划) | 61.9 | 8.8 | 70.7 | 706.3(−0.15%) |
| QKV | cap 1(不 split) | — | — | 0 | 717.9(**−1.81%**) |
| MLP-up | k=4(生产) | 110.6 | 17.0 | 127.6 | 900.2 |
| MLP-up | cap 2 | 123.5 | 14.8 | 138.3 | 911.1(**−1.22%**) |
| MLP-up | cap 1 | — | — | 0 | 954.7(**−6.06%**) |
| MLP-down | k=4(生产) | 109.1 | 17.0 | 126.1 | 908.7 |
| MLP-down | cap 2 | 122.8 | 18.7 | 141.5 | 916.3(**−0.83%**) |
| MLP-down | cap 1 | — | — | 0 | 946.8(**−4.19%**) |

QKV 的 `cap 2` **就是**生产计划,所以那 −0.15% 是**探针自己的噪声底**(MLP 的差是它的 6~8 倍,
这是个好用的零对照)。机理:`tail × k_split` 必须等于 `_NCU` 尾轮才填满机器;`k_split` 减半 ⇒
尾轮只占一半 CU ⇒ split 臂 +13 µs,fold 只还回 2 µs。**被补回来的那一轮**才是约束,不是 partial 字节。

**(2) 让 slice 0 直写 C、fold 只读 `k_split−1` 面 ❌ 字节中性,不用花 bench 位**
令 `T = tail·BM·BN·2`。现状 `k·T`(尾臂写)+ `k·T`(fold 读)+ `T`(fold 写 C)= `(2k+1)T`。
slice-0 直写 = `T` + `(k−1)T` + `(k−1)T` + `T`(fold 读 C)+ `T` = **同样的 `(2k+1)T`**。
真正省字节的变体是"最后完成的 slice 就地做 fold"(`(2k−1)T`,k=4 省 fold 的 2/9 ≈ MLP-up 4 µs),
需要跨 workgroup 原子计数 —— **这条没被证伪,只是没做**。

**(3) 给 partials store + fold load 换 cache policy(`nt`/aux=3)让它驻留 LLC ❌**

| cell | aux=0(生产) | aux=2 | aux=3 |
|---|---|---|---|
| QKV | 61.7+9.0 = 70.7 | 62.5+10.7 = 73.2 | 63.0+10.4 = 73.4 |
| MLP-up | 112.4+16.6 = 129.0 | 115.4+20.0 = 135.4 | 114.2+19.4 = 133.6 |
| MLP-down | 111.1+16.7 = 127.8 | 113.0+20.4 = 133.4 | 114.0+20.2 = 134.2 |

坏得最多的是 fold 本身(MLP-up 16.6 → 20.0)。前提就不成立:**MALL 是 memory-side、八个 XCD 共享**,
跨 XCD 的生产者/消费者不需要 streaming hint 就能在那儿碰面;默认 cached 策略已经够,
而 `nt` 额外把 fold 自己的读变成 non-temporal —— 一个 tile 的 `k_split` 张面是**一起消费**的,
这正好打在反面。`gemm_mxfp8_kernel.py` 里原本那句注释是对的,现在它是**对的且有数**。

---

## 折叠 C store 的 WAR bank 深度(2026-09-11 r43,ISA 定位 + 5 次 bench)

### 机理:`s_waitcnt vmcnt(4)` 把每个 wave 的在飞 store 钉死在 8 条

`cst_wide` 每个 store unit 先把 8 个 dword 打包进一条 bank,再发 4 条 `buffer_store_dwordx2`;
bank 只有两条(`b = _CDV + (u & 1) * 8`,v184-191 / v192-199),所以 unit `u` 要写 bank 之前
必须等 unit `u-2` 的 4 条 store 退休 —— emitter 用 unit 头上那条 `s_waitcnt vmcnt(4)` 表达。
FC1_fwd 的 final ISA 实证(`_kyle_r43_isa.py` / `_kyle_r43_peek.py`):

```
s_waitcnt vmcnt(4)          <- WAR 水位线
s_mul_i32 / 4x v_add_u32    <- cst_rows
8x v_cvt_pk_bf16_f32        (+ AGPR unit 另加 16x v_accvgpr_read_b32)
4x buffer_store_dwordx2     <- 4 条连发,一个 MFMA 间隙全塞完
10~12x v_mfma               <- 这段完全空着
```

稳态是"发 4 条 → 等到只剩 4 条",即 **每 wave 最多 8 条在飞,且每个 unit 都显式等最老的 4 条退休**。
一个 unit 的 MFMA 窗口 = 16 条 MFMA = 256 cycle;若 store 完成延迟 L > 256,每 unit 就露 (L−256)。

### 寄存器预算:arch VGPR 还剩 17 dword(gfx950 单波 512 dword/lane 池)

`kernel_gemm_4w_1.num_vgpr = 239`,`num_agpr = 256`,`accum_offset 240`,`next_free_vgpr 496`。
**512 − 496 = 16**,每条 bank 8 个 dword,所以最多再加两条 bank(4 bank 正好 255+256=511)。

| `_MXFP4_CBANK` | `_NCDV` | vmcnt 水位 | 在飞上限/wave | num_vgpr | V+A | spill | mxfp4 geomean |
|---|---|---|---|---|---|---|---|
| 2(原生产) | 18 | `vmcnt(4)` | 8 | 239 | 496 | 0 | 0.9916 / 0.9925(均 **0.99205**) |
| 3 | 26 | `vmcnt(8)` | 12 | 247 | 504 | 0 | 0.9936 / 0.9929 / 0.9920(均 **0.99283**) |
| 4 | 34 | `vmcnt(12)` | 16 | 255 | 512 | 0 | 0.9922 |

发射指令数三者完全相同(6138 条),17 形状 sha **逐位相同**(`KNOBS=_MXFP4_CBANK=2` 对拍)。

**结论:3 bank 比 2 bank 只高 +0.078%,压在噪声底之下;4 bank 反而回落。**
⇒ store 完成延迟在 2~3 bank 处**就已经被 MFMA 流盖住了**,加深在飞队列换不来墙钟。
这条不是"不能做",是**定价做完了**:剩下的 store 成本是发射 + 写端字节,不是 WAR 串行。
下一个入口在写端字节(见本卡"写侧是字节/带宽限"一节)和 epilogue 发射条数,不在 vmcnt 水位。

**安全前提(改水位线必须先验)**:`vmcnt` 是统一计数器,把 load 一起算。
所有 `CstSched` 使用点(`emit_peel_fold` / `emit_runtime_zero` / `emit_runtime_odd_tail` /
`emit_inplace(..., cstq=)`)传的都是 `g2sl=[]`,且入口处都有 `s_waitcnt vmcnt(0)`
(ISA 实证:peel 入口 `vmcnt(0)` → 2 条 scale load → 再 `vmcnt(0)` → 第一条 `vmcnt(4)`)。
**只要有一条 g2s 混进 store 区,`4*(banks-1)` 水位就会放过还没退休的老 store,直接数据错。**
fp16 输出走的是非折叠 store(`_CST` 要求 `not out_fp16`),`_NCDV=0`,这条改动对它是死代码。

### ❌ 别再试(r43 实测,同一 session 同一窗口)

| 试法 | 单变量 | mxfp4 | 备注 |
|---|---|---|---|
| `num_xcds = 16` 推广到短-K 行 | 8 → 16 | **0.9890(−0.46%)** | QKV_dgrad −2.16%、out_proj_fwd −1.41%、out_proj_dgrad −1.23%。xcd=16 只在 `K≥28672` 成立,短-K 行 **必须 8**。 |
| `group_m*2` 加进 `_mxfp4_swizzle_candidates` 让计时赛道选 | 3 → 4 候选 | **0.9927(−0.09%)** | 赛道是 `min` over 8×20,但多一个候选就多一份"取噪声最大值"的选择偏差;FC1_fwd 纹丝不动(0.9480)。**group_m 在这张表上不是活轴。** |
| `_MXFP4_PSP` 16 → 24(放 ki=24 的 QKV_dgrad 进来) | 16 → 24 | 0.9939(+0.03%,噪声内) | 目标行 QKV_dgrad 自己 0.9804 → 0.9797,**没动**。r12 量过 ki∈{16,56,128},现在 ki=24 也量了:PSP 只对 ki=16 有价值。 |

### 顺带量到的固定量(FC1_fwd,`_TPW=4`,32768×28672×4096)

- 模块 6138 条指令;每 tile 640 条 MFMA(head 128 + loop body 256 + peel 256)、
  176 条 `ds_read_b128`、128 条 `v_cvt_pk_bf16_f32`、96 条 `v_accvgpr_read_b32`、64 条 store。
- **tile 边界 = 163 条指令**(36 `buffer_load_dwordx4` = fill 18 + asm prologue 18,
  32 `ds_read_b128`,~95 条标量)+ 3 条 `s_barrier` + 3 条 `s_waitcnt`。
  按发射算 ≈ 367 cycle,只占 FC1_fwd 单 tile 的 ~1%;
  **即"每 tile 固定开销 15.8%"里绝大部分不是边界发射,是等待**。
- `_TPW=4` 让 tile 体被 `range_constexpr` 展开 4 份 ⇒ 模块 ~49 KB,**超过 32 KB L1 I-cache**。
  稳态 K 循环是硬件回跳(~3 KB,放得下),但每 tile 的 head/peel/prologue 走的是不同副本。
  用运行时循环替掉 `range_constexpr` 可以把模块缩到 1/4 —— **这条没被证伪,只是没做**。

### ❌ 别再试:mxfp8 dual-cast 的 pid 走位(`grid_gm` / `xcd_band`)—— 第三次以同样方式死掉(r10)

| 试法 | 单算子 ruler | step ruler | 结论 |
|---|---|---|---|
| 全局 `gm=8, band=64`(r8) | +1.7% 均值 | bench `+0.05%`(105.5581 vs 105.5069) | 不过阈值 |
| **每形状最优方块表**(r10,三次独立 sweep 交叉确认) | **+1.1…+6.5%,10/11 条** | bench **−0.45%**(105.96 vs 106.44),QKV −1.42 | **倒贴** |
| 方块模型 `units=1+1`(两条 scale line 各只归一个 XCD) | 宽 cast 上 **−2.6…−6.0%**(最差档) | — | 模型被证伪 |
| `krun` 加大到吃满整条 scale line | placement 成本 →0.00%(机制确认) | — | grid 缩小,总账 −1.3…−48% |
| `krun∈{2,3,4}` 推广到 r7 留在 1 的窄 cast | — | −0.08…−2.67%,无一为正 | r7 的取值在 step 尺上也对 |
| **★ 第三次确认(2026-09-11 r5,换成 NT/五格打分器)**:放开 `_qdual_tile_cfg` 的 `Mp,Kp>=8192` 门让 `Kp=3072` 那批 cast 也走 `krun=3`(只动 krun,`grid_gm`/`xcd_band` 不动,15 条 cast 里 8 条受影响) | 动机是实测 krun=1 的 cast in-step 只有 **3.97–4.66 TB/s**、krun=3 有 **4.67–5.54**(读/行写连续段短 3 倍) | **step gm −0.50%**(base 107.305 / 惰性 ctl 107.233 / krun 106.774),五格四负:QKV −1.00、linear2 −0.72、linear1 −0.50、MLP-down −0.42,仅 MLP-up +0.17 | 网格从 1536–4032 WG 掉到 **512–1920 WG**,缩水的损失盖过连续段收益 ⇒ **这批 cast 的正确杠杆是"加长连续段但不缩网格"**(例如 `bk` 加宽 + `bm` 减半保住 LDS 与 WG 数),不是加 krun |
| `pad_extra∈{0,8}`、`cm_row/col=0`、`cm_data∈{0,2}` | — | 全部 ≤ 漂移或明显负 | knob 已到位 |

根因写在 `pitfalls/02` 第三个杀手:**写放置的贷方在 step 里已经被 MALL / 消费者付过了,借方却是新增的**。
这条轴上再搜(更大的 band、别的 gm、in-situ race 选走位)都是在同一个错误的尺上加精度。
**要动 scale 平面,就得动它的字节和消费者**(写成 preshuffle 布局、删掉 preshuffle 那 3 次 launch),
不是动它的地址。

### ❌ 【r2 已证伪，勿再照做】"融合 fp4 量化器和独立 quant kernel 舍入约定不同" (GLM-5.2 EP4, r1)

> **r2 更正**: 下面这一节的诊断是错的,照它改**一点用都没有**。
> 实测: 把 `mixed_moe_gemm_2stage_common.py` 里手写的
> `(bits + 0x400000) & 0xFF800000` → `exp - fp_headroom` 换成
> `emit_mx_e8m0_scale(mode=RoundUp, dtype=FP4_E2M1)`,
> rel_l2 **0.0418 → 0.0417**(b8/16/64/128: .041686/.041621/.041846/.039515),
> 一个数量级都没动。
>
> **原因**: 这两个公式在数学上**完全等价**,包括边界。
> RoundUp = `ceil_pow2(amax * inv_max_pos)`, `inv_max_pos = 0x3E2AAAAB ≈ 1/6`。
> 令 `amax = m·2^k, m∈[1,2)`: `m<1.5` → `m/6` 指数 `k-3` 尾数非 0 → 进位 `2^(k-2)`;
> `m≥1.5` → 指数 `k-2` 尾数非 0 → `2^(k-1)`。
> 手写 EVEN(val_to_add = `1<<22`, 即 mbits=0, headroom 2) = `round_pow2(amax)>>2`:
> `m<1.5` → `2^(k-2)`; `m≥1.5` → `2^(k-1)`。逐档相同。
> 连 `m=1.5` 都相同,因为 1/6 在 f32 里不精确(`1.5*0x3E2AAAAB = 0.25000000745`,
> 尾数非 0 仍然进位)。
>
> **真正的机制是输入精度,不是 scale**: 非融合路径是
> `f32 acc → 存 bf16 中间张量 → quant kernel 读 bf16 → fp4`,
> 融合路径是 `f32 acc → 直接 fp4`。fp4 e2m1 相邻档(0,.5,1,1.5,2,3,4,6)相对步长 25–50%,
> 落在量化中点附近的元素会因为这 bf16 一进一出而翻档。
> 在融合量化器里量化前先 RNE 到 bf16
> (`(b + 0x7FFF + ((b>>16)&1)) & 0xFFFF0000`),
> rel_l2 **b64 0.0418 → 0.0161(2.6×)、b8 → 0.0070(5.9×)**,而 score 完全不变
> (1.02832 vs 1.02847,差 1.5e-4 < 0.035% 噪声)。**对齐 golden 的速度代价是零。**
>
> **通用教训**: 比较两条"同一种量化"的路径时,先问**喂进量化网格的输入精度是否一致**,
> 再去查 scale 公式。低精度格式里,前者的影响比后者大一个数量级。
> 另: 判断两个 e8m0 公式等不等价,要**按尾数分段代数推导**,不要看代码长得不一样就下结论。
>
> 残差(bf16 对齐后剩下的 0.0161)**随 token 数单调增长** 0.0070→0.0149→0.0161→0.0186,
> 说明是另一套与 M 相关的机制;当前最可能的候选是融合量化器的 per-1x32 组 amax
> 靠 `shuffle_xor(1/2/4)` 跨 8 条 lane 归约(`e_vec_s1 = min(tile_n//32, 8)`),
> 而这 8 条 lane 在 MFMA C-fragment 里可能**跨行**,导致某行的 amax 被同 tile 其它
> token 的幅度污染(b8 的 tile 多是 padding 所以污染小,b128 塞满真实 token 所以污染大)。
> 未证实,下轮用"逐块对比 e8m0 字节"的脚本判定,不要靠猜 MFMA 布局。

**r3 把这条候选也证伪了,并把搜索空间压到只剩 1 条。** 判决手法都是"换一个只影响
该环节的旋钮,看 rel_l2 动不动",值得复用:

| 环节 | 判决实验 | 结果 | 结论 |
|---|---|---|---|
| 值转换 f32→e2m1 | 读代码 | fp4 路走硬件 `rocdl.cvt_scalef32_pk_fp4_f32`(RNE);同文件 `f32_to_e2m1()` 是**死代码** | 正确 |
| e8m0 幅度 | `fp_headroom` 2→1/3/4 | 0.041845 / **0.272759** / **0.277954** / **0.530839** | hr=2 是锐利最优 ⇒ scale 幅度没偏 |
| e8m0 取整约定 | RNE `(b+0x400000)&0xFF800000` → floor `b&0xFF800000`(OCP 规范写法) | 0.041845 vs **0.141368** | 现有 RNE 更对;**"按规范用 floor"在这里是错的** |
| amax 归约几何 | 用 `tile_n` 换归约树: tn64(`e_vec_s1=2`+16 lane `xor 1/2/4/8`) vs tn128(`e_vec_s1=4`+8 lane `xor 1/2/4`) | **0.041846 vs 0.041845** | 换了树误差 6 位有效数字不变 ⇒ 归约集合就是那 32 个元素,**没有跨行污染**(r2 的猜测错) |

> **`fp_headroom` 扫描是个通用的好探针**: 它把整组值在 fp4 网格上整体平移一档。
> 若当前值是锐利最优(±1 就炸 5~7 倍),说明 scale 幅度与网格对齐是正确的,
> 可以一次性排除所有"scale 偏大/偏小/组太大"类假设 —— 比逐个猜布局快得多。

> **"几何不变性"探针同理**: 让归约跨越的 lane 数变化(这里靠 tile_n),
> 若 rel_l2 到小数点后 6 位都不动,归约集合必然是对的。

**还剩的唯一机制**是 r2 实验 B 指向的输入精度,但它只解释 2.6×(0.0418→0.0161),
残差 0.0161 随 M 单调增长(0.0070/0.0149/0.0161/0.0186)。上面四项都与 M 无关,
所以残差是**第二个、与 M 有关**的机制;与 M 相关的量只剩 padded 行数与 block 数
⇒ 下一步查融合路径用 `torch.empty` 分配的中间 fp4 张量(`moe_kernels.py:1588`)
只写算过的行、而独立 quant kernel 覆盖整个 sorted+padded 张量这个差别。

**r3 另加一个可对比实现**: split-K stage1(`_kb2`)的 rel_l2 也在 0.034–0.042
(b8 0.034499/b16 0.037910/b64 0.041558/b128 0.040639),而它走
"stage1 写 bf16 `tmp_out` → `silu_and_mul_fq` 读 bf16 → fp4",**本身就有 bf16 往返**
却依然 ~0.04 ⇒ 佐证"输入精度不是唯一机制"。字节 dump 时做**三方对比**
(参考 / 融合 epilogue / split-K 后处理),差异落在哪两方之间就直接定位到代码。

<details><summary>r1 的原始(错误)记录,保留以备追溯</summary>

两处都"per_1x32 mxfp4 量化",但产出不同的 e8m0 字节:

| 实现 | 位置 | 公式 | 模式 |
|---|---|---|---|
| 独立 kernel (golden 来源) | `csrc/include/mx_quant_utils.h:130` + `quant_kernels.cu` `fused_mx_quant_moe_sort_kernel` | `scale = ceil_pow2(amax/6)` (`inv_max_pos=1/6`) | **RoundUp / RCEIL** (`kDefaultMxScaleRoundMode`) |
| stage1 融合 (`_fp4` 后缀) | `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py:2558` | `(amax_bits + 0x400000) & 0xFF800000`, 再 `exp - 2` | **EVEN** (`even_val_to_add`, `target_max_pow2=2` ⇒ /4) |

两者都是合法 MX 约定,但中间量只有 2 bit 尾数,换约定就有大量元素跳一格
⇒ **端到端 rel_l2 ≈ 0.042**(GLM-5.2 EP4 全 bucket 恒定),直接顶穿 0.002 的门。

**诊断手法**(值得复用):rel_l2 在 `atomic`/`reduce` epilog、v2-layout/普通 stage2、4 种 stage1 tile 下
**全部恒等于 0.0418** ⇒ 误差源唯一定位在 stage1 输出量化,与 epilog/generator/tile 全部无关。
"各档 rel_l2 完全相同"是"约定不一致"的指纹,"各档不同且随规模变化"才是 race。

**修法**:`aiter/ops/flydsl/kernels/quant_utils.py` 里已有 `emit_mx_e8m0_scale(local_max, mode, dtype)`,
返回值语义与 stage1 现用的 `e8m0_biased` 一致(caller 自己算 `quant_scale = (254-e8m0_biased) << 23`)。
stage1 **没 import 它**,是一份手写的 EVEN 副本 —— 换成 helper 调用即可。
⚠ 该量化器被所有 `_fp4`/`_fp8` stage1 共用(gpt-oss / kimi / dsv4 的 tuned 行都可能命名 `_fp4`),
**不许就地改默认约定**,必须加后缀(如 `_fp4r`)或 `mx_round_mode` 参数让单个模型 opt-in。
fp8 路(`fp_headroom=8`,`FP8_E4M3` 的 `mbits=3`)大概率有同一处分歧,未验。

**为什么值得修**:`_fp4` 消掉层间那个 6.47 µs 的 `fused_mx_quant_moe_sort_kernel<bf16,fp4_t,256,8>`,
GLM-5.2 EP4 上实测 **+3.0% score**(b64 −3.0%,b128 −3.1%,profiler 确认该 kernel 从 trace 消失),
唯一拦路的就是这个舍入约定。

</details>

**r2 保留有效的部分**:「`_fp4` 融合值 +3.0% score」经 r2 独立复测成立
(score 1.0283,b64 197.6→192.3 µs);「rel_l2 在各 epilog/tile 下恒定 ⇒ 误差源唯一在
stage1 输出量化」这条定位手法也成立 —— 只是"恒定"指向的是**输入精度**,不是 scale 公式。
r3 第三次复测: b8 52.194/b16 83.546/b64 190.8/b128 226.647 ⇒ score **1.031**。

#### ❌ r3 实测为负 / 中性的杠杆(GLM-5.2 EP4 a4w4, b64, 别再走)

| 臂 | b64 µs | vs base 197.34 |
|---|---|---|
| `xcd_swizzle=4` stage1 | **436.931** | **+121%** |
| `xcd_swizzle=4` stage2 | **299.521** | **+52%**(且 rel_l2 跳到 0.001353) |
| `block_size_M=128` | 241.156 | +22% |
| `b_nt=0` stage1 | 210.108 | +6.5% |
| `block_size_M=64` | 206.027 | +4.4% |
| oneshot sort(强制单 WG) | 202.827 | +2.9% |
| stage1 grid.y 收紧(275→83,砍 3488/4400 个空 WG) | 196.59 | **0.0%** |
| 全部 wpe/bnt/tn 旋钮里最好的(`wpe=4`) | 196.505 | −0.4%,**4 bucket 复测下消失** |

**两条可复用的反面教训**:

1. **放大 `block_size_M` 是个会反噬的"字节杠杆"**。它确实减少权重字节
   (b64 55→54 blocks @tm64,b128 66→64 @tm64 / 66→63 @tm128),但
   `num_valid_rows` 爆炸(b64 1760→3456→6912;b128 2112→4096→8064),
   padded 中间张量的写+读字节增长远快于权重节省 ⇒ **字节账必须算总量,
   不能只算权重那一项**。
2. **空 workgroup 几乎不要钱**。砍掉 4400 个 WG 里 3488 个空的,时间一点不动。
   MoE 里 `max_num_tokens_padded` 按**全局** expert 数(258)算出的巨大 grid
   看着很浪费,但它不是瓶颈 —— 不要从 dispatch 层找这个收益。

**噪声参照系**: 该 campaign 宣称的 0.035% 是**四 bucket 合并 score** 的噪声;
单 bucket 上 b64 同配置重复 8 次落在 196.14–197.51(**±0.35%**),且每个 session
第一次跑系统性偏慢。⇒ 单 bucket 上 <0.7% 的差异不得判为收益(参见
`pitfalls/02` 的冷热尺子 + 取最小值)。

## ★★★ 折叠 C store 的 WAR 水位:用"被打包掉的累加器"当私有 bank,**免费合法删掉**(2026-09-11 r47)

r43 那一节(本卡 §折叠 C store WAR bank 深度)把 `_MXFP4_CBANK` 2→3→4 扫完,结论是
"每个 unit 一条 `s_waitcnt vmcnt(4*(banks-1))`,深度轴全在噪声带里"。r47 找到了**第四种做法**:
不加 bank,而是**反向构造 bank 来源**。

机制:一个 store unit 把 4 个累加器 tuple 用 `v_cvt_pk_bf16_f32` 打成 8 dword 的 bank,
打完那 4 个 tuple 就**死了**,而且它们**从来没当过 store 的源**(store 只读 pack bank)。
所以每个"已换进 arch VGPR"的 unit 打完之后,可以把自己那 2 段累加器寄存器(各 8 dword)
交给后面的 unit 当 pack bank。可用量 = 2 + 2×(前面已换进 arch 的 unit 数)
= 2,4,6,8,10,12,12,14,14,16,16,18,18,20,20,22 对 unit = 0..15 ⇒ **永不枯竭**,
于是"每个 unit 都拿到一个从未被 store 读过的 bank",WAR 水位整族**合法消失**(不是裸删)。

实测(gfx950 mxfp4 NT,`_NCBK=2`,`_NAV=40`):
- `s_waitcnt vmcnt(4)` **65 → 1 条/module**,少 64 条指令;
- `num_vgpr 239 / num_agpr 256 / accum_offset 240 / next_free_vgpr 496 / spill 0` **一个都没动**
  ⇒ **零新增寄存器**;`v_mfma` 2560 / `buffer_store_dwordx2` 256 / `v_accvgpr_read_b32` 384 /
  `ds_read_b128` 704 / `s_barrier` 27 全部不变;`|C|sum` **逐位相同**,SNR 55.53/55.61 dB 不变。
- 计分台 geomean:基线 **0.9907**,候选 **0.9914 / 0.9914 / 0.9919**(三连读)⇒ **+0.07~0.12%**,
  单次读数在 0.41% 单行噪声底之下,但三读全部 ≥ 基线且机制上不可能亏(指令更少、位相同)。
- 紧尺子(out_proj_fwd,分进程三臂回文,arm0 跨度 0.309%):**判平**。

⇒ 两条结论:①这条改动是**免费的**(少 64 条指令、零寄存器、位相同),留着;
②**折叠 store 区不是发射瓶颈** —— 删 64 条指令时间不动,写端约束在别处(TCP 写请求/完成带宽)。
注意 goal.md 曾把这一臂记成 "+0.79%、兑现率 86%",**不复现**;它那个 +0.79% 落在自己 4 臂
0.73% 的位置跨度里。与 `00-decision-index` §折叠 C store 四自由度已定价 一致。

## ★★★★ 相位 `s_barrier` 值 **0.87%**,相位内存等待 `vmcnt(10)` 值 **0.08%**(2026-09-11 r47 减法定价)

whole-loop 的 `_ipend` = `s_waitcnt vmcnt(_WLV) lgkmcnt(_ELGK)` + `s_barrier` + `s_nop 0`,
每 K-block 执行一次(12 个发射点)。两条减法探针(**都不可发布**,只读时间;一臂一进程、
每臂清 cache、arm0 两端、out_proj_fwd 32768×4096×4096):

| 臂 | us_min | us_med | vs arm0 均值 |
|---|---|---|---|
| arm0 基线 | 225.804 / 226.431 | 228.100 / 228.616 | —(端点跨度 0.28%) |
| `vmcnt(63) lgkmcnt(15)`(等待失效) | 226.166 | 228.293 | **−0.08% / +0.03%** |
| 删 `s_barrier`(27→15) | **224.124** | **226.094** | **+0.87% / +0.99%** |

⇒ **稳态 K 循环不卡访存,卡的是会合**。g2s 在被等到之前早就回来了(等待失效后逐位相同,
说明 `vmcnt(10)` 从来没真的挡住过谁),所以"加深 g2s ring 来盖延迟"这一族**上界只有 0.08%**,
不要投。真正的 0.87% 在**屏障条数**上:64 次/WG × ~26 clk 会合开销。
合法减半的路径 = ring 深度 3(屏障每 2 个相位一次),LDS 账:A 3×32768 + B 3×32768 = 196608 B
> 163840 B,**还差 65536 B**(现有空闲 16384 + 可回收的死 `_NSCBUF` 池 16384 = 32768)。
⇒ 下一条具体杠杆:把 B 的一半改成**不过 LDS**(VGPR-direct,像 scale 那样),B 池腰斩省出
32768 B,凑齐 A/B 双环深度 3 ⇒ 屏障减半 ⇒ 预期 ~+0.45%。

## ★★★★ r50：把跨 tile 的 store drain 彻底移到下一 tile 的 MFMA 之后 = **只值 3 条 barrier**（2026-09-12 实测）

持久 walk（`_TPW=4`）的接缝上，tile t 的 64 条折叠 store 和 tile t+1 的 k=0 填充撞在一起，
而 vmcnt 是**统一、按序**的 6-bit 计数器 ⇒ 想等到填充就必然先等完所有 store。
真做出来的机制（`_MXFP4_SEAM`，本轮**已实现并留在树里但默认关**）：A 环扩到 3 槽 + 轮转，
在 tile t 的 peel 里、**store 簇之前**把 t+1 的整个 k=0 块（A/BL/BR + 两组 scale 进 tied 暂存寄存器）发出去，
用 `s_waitcnt vmcnt(60)` 只等这批 load，然后删掉 prologue 的等待与屏障，并把 k=1 会合点**下沉** 64 条 MFMA。

结构（ISA 实测）：接缝从 `[64 store] vmcnt(16) [34 load] …` 变成
`[34 load][64 store] vmcnt(60) lgkmcnt(0) s_barrier [32 ds_read] … MFMA`；barrier **27→24**；
LDS 恰好 163840 B；VGPR/AGPR/spill **252/256/0 不变**；指令组成逐项相同；输出**逐位相同**
（4 个臂 `|C|sum` 完全一致，SNR 55.535 dB）。

| 计数器（out_proj_fwd，`SQ_INSTS_MFMA` 自检两臂相同） | baseline | seam | 比 |
|---|---|---|---|
| `SQ_WAIT_ANY` | 1.29907e7 | 1.20993e7 | **0.9314** |
| `SQ_BUSY_CYCLES` | 1.13045e7 | 1.11933e7 | **0.9902** |
| `TCP_TCR_TCP_STALL_CYCLES` | 1.62786e6 | 1.80597e6 | **1.1094** |
| `TCP_TCC_WRITE_REQ` | 2048/tile | 2048/tile | 1.0000 |

wall（一臂一进程、每臂清 cache、回文、`W` 全程钉 1391–1400 W）：
**out_proj_fwd +0.31% min / +0.03% med**（`0 7 3 7 0`，端点跨度 0.19%；`SEAM=3`=不下沉会合点 −0.32%），
**FC1_fwd −0.35% min / −0.40% med**（`0 7 7 0`，swizzle 钉 `GM=4 GN=0 XCD=8`；不钉时 −0.22%，两次同号）。

⇒ 三条结论：
1. **drain 本身 ≈ 0**。那 +0.31% 与同时少掉的 3 条 barrier（27→24）按 r47 的 barrier 定价
   （+0.87% / 12 条 ≈ +0.22%）完全自洽，与 r47「让 `vmcnt(10)` 失效 = +0.08%」是同一结论：
   **这颗核的 vm 等待从来不是关键路径**，别再投「让填充更早落地 / 加深 g2s ring / 移 drain」这一族。
2. **周期赢 ≠ wall 赢**：`SQ_BUSY_CYCLES` −0.98% 是真的，但 store 簇一旦和 g2s 簇真重叠，
   `TCP` 反压 +10.94%，宽 N 的行（FC1_fwd，每 tile 的 B/L2 读压最大）净亏。
   ⇒ 下一条具体杠杆是**错峰**：把 34 条预取 load 插进 store 簇的空隙（而不是整簇压在它前面），
   让 TCP 反压不涨的同时保住 3 条 barrier 的收益。
3. 副产品（**已验证可复用**）：A 环深 3 在 mxfp4 dense NT 上**放得下**且逐位相同 ——
   回收死掉的 `_NSCBUF`（16384 B）+ 现有空闲（16384 B），A 3×32768 + B 4×16384 = 恰好 163840 B。
   r47 说的「屏障减半」现在只差 **B 侧**深度 3（还要 32768 B）。
   ⚠ 轮转带来的 M0 下溢坑见 `pitfalls/03 §M0 下溢`（只错一个 tile 索引、SNR 14.8 dB）。
