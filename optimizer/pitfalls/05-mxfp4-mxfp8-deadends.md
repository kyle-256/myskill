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
- 唯一有效 lever 是**调度局部性（xcd/gm/gn），已被 autotune 吃掉**。

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

---
来源: 08-att-root-cause.md, 04-ceiling-analysis.md, project_mxfp4_k28672_ceiling.md, project_mxfp4_epilogue_store.md, flydsl-fp8-gemm-results/SKILL.md, 10-grouped-wgrad-4wave-3buf.md, mxfp8-grouped-gg-devloop/SKILL.md, 08-deadends.md, 05-dead-ends.md, 12-llama-aiter-baseline.md, mxfp8-8wave-devloop/SKILL.md, project_mxfp8_wholeloop_port.md, project_mxfp8_grouped_wgrad_wl.md
