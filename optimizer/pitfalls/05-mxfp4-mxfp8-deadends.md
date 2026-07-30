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

### ❌❌ persistent grid：两次独立判负，第二次已排除「per-tile 解码成本」这个混淆项
- 一测（2026-07 round 10，NT）：`num_cu=256`，K=2944 慢 **25.6%**、K=5760 慢 **37.1%**。当时的怀疑是"persistent 把 O(G) 入口扫描从每 tile 一次降到每 CU 一次，应该赚"，结果反向。
- 二测（同 campaign，per-tile 解码已换成 lane-resident 表 ⇒ 每 tile 只剩 1 条 ballot + 5 条 `v_readlane`）：`_probe_nt_persist.py` 单进程交织，3 shape × 2 分布 × 3 swizzle cfg，persistent **全部 0.70–0.81×**，与一测**几乎逐点相同**。
- ⇒ **结论从"入口扫描不是 per-tile 固定成本的大头"硬化为"persistent 的亏损与解码成本无关"**。该核 occ=1（LDS 128 KB/WG），非 persistent 时 CU 空出来立刻换下一个 WG，persistent 并不能多重叠任何东西，却额外背上 `scf.for` tile 循环的活跃值与流水重填。**两个核都别再走截 grid 这条路去摊固定成本。**
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
  pitfalls/03「prefetch 治不了 L2 capacity thrash」同源。**别再花轮次在 NT prologue 的发射顺序/drain 合并上**；
  剩下的开口在**减字节**（更大 tile / scale 物理布局），不在减延迟。

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
- ✅ **下一刀 = LDS→VGPR 读放大(满额兑现的能量类)**:实测 **0.504 pJ/FLOP vs mxfp8 MFMA 峰值 ~0.28 ⇒ 只 55% 能效**,缺的那一半在 LDS→VGPR 通路——每 call 6144 tile × 45 iter × 64 KB = **17.7 GB 进 LDS**,按 wave_n×4 / wave_m×2 **读出 35~70 GB**(~1~2 pJ/B = 35~140 mJ = 能量的 **1.6~6%,是整条 DRAM 超额杠杆的 10~30 倍**)。⇒ N10(ISA region 分段量那 254 个 VGPR 的构成、把非累加器活值压到 ≤128)→ N1(256×512/1024 线程,每 LDS 字节喂的 MFMA 翻倍),是唯一同时减读放大与 tile 数的杠杆。
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

---
来源: 08-att-root-cause.md, 04-ceiling-analysis.md, project_mxfp4_k28672_ceiling.md, project_mxfp4_epilogue_store.md, flydsl-fp8-gemm-results/SKILL.md, 10-grouped-wgrad-4wave-3buf.md, mxfp8-grouped-gg-devloop/SKILL.md, 08-deadends.md, 05-dead-ends.md, 12-llama-aiter-baseline.md, mxfp8-8wave-devloop/SKILL.md, project_mxfp8_wholeloop_port.md, project_mxfp8_grouped_wgrad_wl.md, project_gptoss_nt_fwd_dgrad_opt(2026-07-28 PMC/gfx950-waitcnt 实测)
