# 00 — 决策索引:动手前先 grep 这里(死路 / 条件 / 开口)

> 类别: 导航 · 主题标签: decision-index, dead-ends, before-you-try, action-lookup, 死路速查, 症状分诊
>
> **这是「动手前」的第一站,不是读物。** agent/人在**准备试某个杠杆前**,先在本表 grep 你要做的动作
> (如 `grep -i "占用率\|maxnreg\|direct\|atomic\|DMA\|scale_pack"`)。命中 = 别重踩,点进详卡看根因。
> 覆盖全库高频被重试的死路(全量 ❌ 散在各详卡,本表是入口)。
>
> **判定图例**:❌**实测DEAD**=真实现+bench 过,别重跑 · ⚠️**分regime/条件**=看形状/占用才定 ·
> 🔬**分析预测**=只探针/纸面判过,可挑战但先读根因 · ✅**开口**=未验证或值得做。
> ★ 铁律(methodology/03):**subtractive/HALF 探针、roofline 峰值率、纸面 op-count 给的都是「上界」不是「可达」——
> 判负/判正前必须 edit→bench 真实现**(反复踩:HALF_PV +8.7% 假想 → 真 K32-PV −11%;SKIPST +30% → DMA/QB 全负)。

## A. 占用率(occupancy)—— 抬 occ 几乎全负,先确认是不是 occ-bound
| 你想试的动作 | 判定 | 一句根因 | 详卡 |
|---|---|---|---|
| `maxnreg`/`waves_per_eu`/`--amdgpu-num-vgpr` 强抬 occ | ❌实测DEAD | 必 spill 到 scratch(灾难);且本容器 external codegen 不可用,flag 对 VGPR 纹丝不动 | pitfalls/01,05 · methodology/03 · [[project_mxfp4_vgprform_deadend]] |
| ★**删掉一个"永远满足不了"的 `waves_per_eu` 请求**(与上一条方向相反) | ⚠️**别与"抬 occ"混为一谈** | **flag 对寄存器分配无效 ≠ flag 是 no-op**:mxfp4 grouped NT 是 256 线程独占 160 KB LDS ⇒ 结构上 1 wave/SIMD,`waves_per_eu=2` 每次编译都 warning "final occupancy is 1"。把这个请求删掉后 vgpr/agpr/sgpr/spill/LDS **逐位相同**,但调度器的 occupancy 目标变了(warning 随开关出现/消失),四个 NT 格隔离回文 **0 ~ +2.7%**(对照格噪声地板 0.3%)。⇒ 评估时把「删一个未满足的请求」与「强抬占用率」分开:前者只改调度,后者要改分配 | pitfalls/01 §198 · methodology/04 |
| `amdgpu-mfma-vgpr-form=false` / inline-asm `=a` 逼 AGPR 抬 occ | ❌实测DEAD | 生产 4-wave ISA 逐字节相同;AGPR 与 VGPR **共用同一 512 池**,搬家不抬 occ | pitfalls/01,03 · methodology/04 |
| 8-wave 原生 occ=2 藏 store / 长 K 追平 4-wave | ❌实测DEAD | 28672 −14%/6144³ −20%;8-wave per-warp tile 减半→B 复用减半→mfma 翻倍 > occ 收益 | pitfalls/03,05,06 |
| ★**降到 4-wave / 1 wave-per-SIMD + AGPR 累加器**(不是抬 occ,是**往下**走) | ⚠️分骨架 | 本机最快的 grouped 核(per-tensor wgrad 标尺 3000+ TF/s)**就是**这个几何。⚠**只换几何不换同步骨架 = −30~40%**(两次独立实测 −31%/−41%,都保留了 6.39 条 barrier/K-iter 与 setprio,标尺只用 ~2.3 条、0 条)。判据 = compile-only 门 **barrier/K-iter ≤3 且 setprio==0**,门不过就不是这条路 | pitfalls/01 §4-wave · methodology/12 §标尺③ |
| 用 tied-operand `"=a,v,v,0,v,v"` 给 **scaled MFMA** 分 AGPR 累加器(目的是腾 **arch VGPR**,不是抬 occ) | ✅ROCm 7.2 可用 | 与上一行不冲突:**2 waves/SIMD 时 arch+acc 共享 256,搬家确实白搭;1 wave/SIMD 时 V/A 是两个独立堆**,mx8tw 实测 arch VGPR 246→**184**(空闲 10→72)。⚠ `amdgpu-agpr-alloc` **属性**路线对 scaled-MFMA 仍无效(AGPR 被当 spill slot,6.2× 慢),别用它的失败否定 tied-operand | methodology/06 · pitfalls/01 |
| BK128 换 occ=2(mxfp4 K28672) | ❌实测DEAD | occ=2 唯一途径 BK128→g2s 频率翻倍 VMEM-stall 2.7→30%,总 5405→4686 | pitfalls/05 |
| 缩 tile / 矩形 tile 为抬 occ | ❌实测DEAD | feed-bound worst shape:占用率非杠杆,根因 LDS-feed 带宽 | pitfalls/01,05 |
| attention 强制 occ(WPEATTR/o_acc 进 AGPR/bf16 o_acc) | ❌实测DEAD | 全 latency-bound,occ-1/2 都填不满;强制必 spill/掉速 | pitfalls/12 |
| bwd(occ-2)dual-wave / 8-wave / warp-spec / bare-asm cross-head 显式并行 GEMM↔softmax | ❌实测DEAD | occ-2 baseline 已靠两独立 WG 共驻**免费享无屏障跨-WG overlap**;8-wave 单-WG 换成带屏障税组内 overlap,天花板<baseline(dkdv 1029<1116)。净胜改走冷-load 寄存器预取(+1.81%) | methodology/15 · pitfalls/13 |
| **先做**:判是不是 occ-bound(而非 register/LDS/feed/latency-bound) | — | occ=`512//(arch+agpr)`;`Accum_VGPR_Count=0`+ISA 才权威;多数 kernel 是 feed/latency-bound | methodology/04 · methodology/03 |
| ✅ **分块读 tile 压活跃度抬 occ 前,先看那条读是不是 inline asm** | ✅开口 | asm 读 waitcnt 不被 `SIInsertWaitcnts` 记分 ⇒ 每切一块就要手写 lgkmcnt + `sched_barrier(0)` 钉住,调度损失正好抵消寄存器收益。换成 rocdl op(如 `ds_read_tr16_b64`)后同一分块从 −0.35% 变 **+0.50%**,occ 2.985→**3.978** | pitfalls/13 §fwd |
| ★★★**换 MFMA 原子** `32x32x16` → `16x16x32`(bf16, 功耗封顶的卡) | ✅**已交付 +9.17% 单轮** | **这是能量杠杆不是周期杠杆**:访存/barrier 逐字节不变、周期 **+8%**、MFMA util **84%→77%(都更差)**,但每 MAC 的累加器 RF 字节数减半 ⇒ sclk **1.40→1.667 GHz @ 1400 W 不变** ⇒ wall **−10.8%**。❌**绝不能用 MFMA util 做门禁**(会否掉这个 +9.5%);❌`SQ_VALU_MFMA_BUSY_CYCLES` **对原子速率是瞎的**(两边逐位相同)。代价:VGPR 220→256 触顶 + 2-dword spill | methodology/03 §Power/clock-bound |
| 周期口径的维度(barrier/ILP/占用/代码量/AGPR/带宽)全排除了,mfma 流仍慢 | ⚠️**换坐标系,别再加维度** | 真正的货币可能是**能量**:同工作量下比 sclk 而不是比周期。踩证:前一场 campaign 在周期坐标系里排了二十余个维度、用完整 ISA + HIP 微基准仍未破;换原子一轮解决 | methodology/03 §Power/clock-bound |
| 16×16 配对 store 用 `cvt_pk` 打包**两行** + `permlane16_swap` | ❌**构造就是错的** | row-major 的 C 里第 r 与 r+1 行相隔 `c_cols*2` 字节,packed dword 不是一段内存。可行构造=复用**偶/奇列 16 列交错**,零 permlane 即达 gfx950 的 64 B 写粒度 | methodology/03 §Power/clock-bound |
| 换过原子/几何/尾块后沿用上一代的 `GROUP_M` / `xcd_band` | ❌**必须当轮重扫** | 本表第 40 行说 band 赢家不跨 kernel 迁移;实测**同一 kernel 内部也不迁移**:tile 快 20% ⇒ GROUP_M 8→4;ragged 块不再补满 ⇒ band 64→32(−0.58%,回文每个位置都快) | methodology/03 §Power/clock-bound |
| ⚠ 抬 occ 成功后**必须报 sclk**:功耗墙下时钟会回吐 | ⚠有条件 | 96.6% TBP 时 occ 2.985→3.978 使 sclk 1710→**1654 MHz(−3.3%)**,wall 只 +0.50%,真实省周期 **+3.9%** | methodology/03 §Power/clock-bound |

## B. LDS / 数据通路 / store
| 你想试的动作 | 判定 | 一句根因 | 详卡 |
|---|---|---|---|
| A/B operand 直载 VGPR 跳过 LDS(a_direct/b_vgpr) | ❌实测DEAD | mxfp4-8wave 慢 ~22%(bit-exact);LDS 复用是必须 | pitfalls/03 |
| 3-stage LDS 双缓冲 / 更深 prefetch distance | ❌实测DEAD | 每 stage×3>160KB 净负;L2 是 capacity-thrash,prefetch 治不了 | pitfalls/03,05 |
| store 与**仍在发 g2s 的相位**重叠(STORE_ILV/persistent/CShuffle 宽存) | ❌实测DEAD | gfx950 统一 vmcnt,在途 store 占 vmcnt→串行化;宽存 LDS round-trip 吃掉收益 | pitfalls/05 |
| ★把**相邻两个 n-fragment 折进一个 dword store**(`v_cvt_pk_bf16_f32` + 一条 `quad_perm` DPP 与邻 lane 交换) | ✅已交付 **+0.85%** | 16×16 MFMA 一行的一个 fragment 摊在 16 lane × 2 B = **只有 32 B/请求**;配对后每 lane 持相邻两列 ⇒ **一发 64 B、store 数减半**,且无 LDS 无 barrier。约束:**N 偶数 + bf16 输出**,边界 clamp 挪到这一对的最后一列。⚠️**作用域=dense fp8 NN**;同一手法搬到 **fp8 grouped 是 −1.09%(NN)/−1.9%(wgrad)**,见下一行 | methodology/07 §epilogue · [[project_dense_fp8_nn_campaign]] |
| ⚠️ 上一行的 DPP 折列**搬到 fp8 grouped 上 = ❌**(2026-08-17 实测,逐位相同) | ❌实测DEAD(跨核不迁移) | 每折一对付 1 DPP(hazard)+2 `v_perm`,只省 1 条 store 发射;反推出**整核 store 发射只值 ~12 µs / 1.5%**,VALU 比它贵。与 `_BILV=4` 的**赢**案例不矛盾:那边相邻列**本来就在同一 lane**(累加器 layout),折叠免费。⇒ 开口只剩「让相邻列天生同 lane」的**不加 VALU 宽 store** | pitfalls/05 §DPP 折列 |
| 写「半深 K-tail body」省掉 K-pad 的 MAC | ⚠️先算指令粒度,多半是 0 | `inst_k` 是硬粒度:`v_mfma_f32_16x16x128_f8f6f4` ⇒ `ceil(2880/128)==ceil(2944/128)==23` **条数相同**,尾体省不下一条 MFMA(纸面 2.2% 被向上取整吃掉),只剩 ~2 µs feed 字节。只改**声明**深度则 −1.70%(丢 128 B 行对齐)。⚠️**唯一 `inst_k` 更小的 fp8 MFMA 是 `32x32x64`,而换形状本身已是本表第 64 行的 12/12 判负** ⇒ 这一族两个分支都有实测读数,别再立项;去打 feed 侧 | methodology/04 §半深 K-tail + §两条 f8f6f4 · 本表第 64 行 |
| 把一个核的 **(num_xcd, group_m) band 赢家抄到同 campaign 的另一个核** | ❌实测不迁移 | 方向由「每 tile 流**私有 slab** 还是**整组共享的 token-major 切片**」决定:NT/NN 要 `xcd=8,gm=4`,wgrad TN 要 `xcd=1,gm=2`;互换各 −2.26% / −0.8~1.0% | methodology/05 §判据升级 |
| 用**背靠背孤立计时**当基线判 in-situ 有没有病 | ❌口径错 | 同一 fwd launch:背靠背 854.7 µs,前面塞一个 quant cast **828.5(−3.07%)**,纯 512 MB fill 843.0 ⇒ 背靠背是**最慢**的排法(DVFS+自己上一轮的输出占满 MALL),生产者相邻值 ~14.5 µs | pitfalls/02 §背靠背偏慢 |
| 拿两个跑在**不同张量**上的核的 TB/s 相减,当成"某条路径每字节有赤字" | ❌口径错 | 跨张量比出 10.7% 赤字/15 µs 可回收,挂了两轮;同一张量只翻开关后真实值是 **1.8% / 4.4 µs**,而且**排序是反的**(激活路径每字节还慢 3.89%) | pitfalls/02 §跨张量 |
| store 折进**结构上不发 g2s 的尾相位** | ✅已交付 | 根因(统一 vmcnt 串行化)在无 g2s 的相位不成立;mxfp4 grouped NT +1.34% 与 wgrad +0.78% 两条独立路径兑现。**上界 = 该相位 MFMA 条数×16cyc,且窗口够用后即闸门**(扩窗/加发射节奏均 0)。需 `_probe_det_wide` 门 | pitfalls/05 |
| ⛔ **折叠 C-store 的四个自由度(宽度/cache scope/银行深度/排空速率)整族封板** | ❌四条独立判据 | ①**宽度**:`_BILV=4` 下一条 lane 恰好拥有 4 个输出列 = 8 B ⇒ `dwordx2` 已是上限,`dwordx4` 需 `ntb=8`(AGPR 翻倍)或跨 lane stride-2 gather(只有 `ds_bpermute` 能做,DPP/`permlane16_swap` 是错轴);且 16 lane 已写满 128B 整线 ⇒ 加宽**不减事务数**。②**scope 位**:在"整线宽 + `nt`"最优态上再加 `sc1`/`sc0 sc1` = **gm −1.2~−2.3**(六个 wgrad 格齐跌 4~5%),L2 是这条写流必需的合并缓冲。③**银行深度** 2→3→4 与 ④**`_CRATE`** 6→10→14 全在噪声带内;bank8 触发 LLVM RA `Cannot decrease cascade number` 崩。⇒ 只剩"写字节本身" | pitfalls/05 §折叠 store |
| ⛔ **`v_cvt_pk_bf16_f32` 在 gfx950 不接受 AGPR 源**(想省掉 epilogue 的 `v_accvgpr_read` 搬运层) | ❌ISA级DEAD | llvm-mc 实测 `v0, a1, a2` 与混合 `v0, a1, v2` 全部 `invalid operand`;能直读 AGPR 的只有 `v_accvgpr_read_b32` 与 MUBUF(`buffer_store_short_d16_hi a{n}`)。⇒ 折叠 store 每 2 个输出 dword 的"4 读 + 2 转"是下限。**动 AGPR→VALU 这类想法前先 llvm-mc 打一行,5 秒判负** | pitfalls/05 |
| ★**成对列 C store(靠喂料侧的行置换把 lane 的两个 n-fragment 变成相邻两列 ⇒ 一条 dword 代替两条 short)** | ✅已交付 **+2.7%**(grouped fp8 NT,G=32/N=2880) | **零 VALU 代价**是它与本表另外三条判负的宽存/配对手法的唯一区别:置换落在 `buffer_load…lds` 的 per-lane global 偏移上(免费),XOR bank key 仍按 LDS 行算 ⇒ s2r 逐位不变,ISA 只有 store 段变(`short 160→32`+`dword 0→64`,**`v_cvt_pk` 160→96 反而更少**,mfma/barrier/VGPR/LDS 逐项不变)。**前置条件:输出列必须是那个 operand 的慢轴**(`trans_b` 的 NT 满足;NN/TN 的输出列是快轴,直搬会打断每 lane 16 B 连续性)。验收计数器是 `TCP_TCC_WRITE_REQ`(−48.9%)而不是 wall | methodology/07 §epilogue |
| 🔑 **写侧的"line 覆盖率"按 128 B 算(想再加宽到 dwordx2/整线)** | ❌**gfx950 的写请求粒度实测是 64 B** | 同核三档实测:标量 32 B/store ⇒ `755MB/32B = 2.36e7`(实测 2.359e7);成对 64 B ⇒ 1.206e7(实测逐位吻合);dwordx2 128 B ⇒ 发射条数再砍半(`TA_BUFFER_WRITE_WAVEFRONTS 3.015e6→1.573e6`)但 **`TCP_TCC_WRITE_REQ` 仍 1.206e7 一点不动**,wall 反而 **−0.85 pp**。⇒ 成对已是地板;以 128 B 为分母的"25%→100% 覆盖"会**高估一倍**。⚠ "`permlane16_swap` 无可赚"只对**已经踩在 64 B 地板上**的核成立,对还在 32 B 的核它正是抬到地板的那一手(见下一行) | methodology/07 §epilogue |
| ★**行合并 C store(输出列是 operand 快轴时的地板做法:`v_permlane16_swap_b32` 把一个 lane 的两个 n-fragment 挪进同一个 32-lane 半区 ⇒ 一条 store 的 32 lane 覆盖一整行 64 B,而不是 16 lane 各覆盖 4 行的 32 B)** | ✅已交付 **+0.83/+0.43/+0.62%**(grouped fp8 NN dgrad,三分布,逐位相同/det 10/10) | 存指令条数**不变**(`TA_BUFFER_WRITE_WAVEFRONTS` 5.898e6 两臂相同),只有每条 store 的请求数从 4 变 2:`TCP_TCC_WRITE_REQ` **2.359e7→1.206e7(−48.9%)**、`TCP_TCC_READ_REQ` 7.163e7 **完全不动**。🔑 **必须让 swap 骑在"打包好的一对行"上**:标量路每个值发一条 `v_cvt_pk` 却只用一半结果,所以按 (row i, row i+1) 打包后 `v_cvt_pk 160→96` 就把跨 lane 的钱赚回来了 —— 同一机制**按 f32 逐值 swap 要 64 条 permlane,只剩 +0.13%**(打包后 32 条 ⇒ +0.83%)。零 LDS 中转、零 DPP hazard、spill/LDS/mfma/barrier 逐项不变 | methodology/07 §epilogue |
| ★**peel-last**:把 tile 最后一个 K-block 整块剥出手写 asm,按 row-tile major 重发,并把每 row-tile 的 C-store 插到下一个 row-tile 的 mfma 后 | ✅已交付 **+0.7~1.1%** | occ=1 下 tile 内唯一可藏算力就是这一个 K-block;asm 额外返回末相已填好的 srcA/srcB 片段即可复用寄存器,不必把 store 搬进 asm。距离 1 最优(2/4 更差);**asm mfma 对 hazard recognizer 不可见 ⇒ 每组 store 前 `sched_barrier(0)` + 收尾 `s_nop 15`×2**,否则静默读到过期累加器 | methodology/07 §peel-last · pitfalls/05 |
| unroll-2 循环里 fold 窗口按 **trip count 奇偶**差 3 倍 | ✅已交付 | 偶数剥最后一对 ⇒ 192 条 MFMA 窗口;奇数若让尾部半块**独立**成相位只有 64 条。修法=奇数也剥,但 phase A 保持完整(它的 g2s 才是搬半块的那次),LDS ring slot 与 scale ping-pong set **必须一起互换**。mxfp4 dgrad K=5760 三配置 −1.1~2.0% 延迟 | pitfalls/05 |
| 把两个相位**融成一个体**(fold peel / 合并 phase A+B / 剥迭代) | ⚠️必须逐项清点被合并掉的 prefetch | 融合体**不会自动继承**未融合体在前一相里做的准备工作。mxfp4 grouped 的 `emit_peel_fold` 就这样丢了 trailing 半块的 scale 重发,**错了 19 轮**:bench 的 SNR 形状 `K%256==0` 走不到折叠路、det 门看 run-to-run 一致而错是固定的、SNR 探针用 `randn` 量化出的 E8M0 近似恒定 ⇒ 三道门全放行。补回来 = gm **−0.49%**(必付)。★ block-scaled 核的 SNR 探针**必须喂随机 E8M0 指数** | pitfalls/05 · methodology/02 |
| 靠错峰/去同相摊平 store 突发(s_sleep skew / XCD 差异化首 tile / tile 序重排) | ⚠️先量再动 | **多代网格的稳态本来就已散相**:实测真网格 25.07 µs/tile 落在锁步曲线 L≈105,而全锁步 L=240 是 34.48;且 24 代不比 3 代更同相。可去的同相度只在**第一代**,skew 已拿走 | pitfalls/05 |
| 缩 operand g2s 的**字节/宽度**(dwordx4→dword、dwordx2、"少搬点") | ❌实测DEAD | `buffer_load…lds` 的成本 = **它覆盖多少条 128B line**,与字节数无关:宽度÷4 而 line 不变 = **+0.14% 平**;line 8→1 而指令数不变 = **+20.2%**。dwordx2 在 gfx950 还是非法编码 | pitfalls/05 |
| 减 operand g2s 的 **line 数**(放大 tile / 提高算术强度) | ⚠️开口·被寄存器挡 | 本核最大的单一池子(≈20% wall)。line/MFMA ∝ `(BM+BN)/(BM×BN)`,**与 BK 无关**。⚠ `BM×BN ≤ 65536` 是**把累加器全放 AGPR 时**的账;gfx950 是 512 合并池,真实约束 = **`BM·BN/256 + frag + 76 ≤ 512`**(frag 双 k-step 驻留 `(BM_w+BN_w)/4`、单驻留 `/8`;256×256 实测 460,spill 0)。但绑定项是累加器,**fragment 减半买不到下一级 tile**(BM=384 单驻留仍 540>512),且波格要求 `wave_m_off` 同为 64 与 128 的倍数 ⇒ `BM ∈ {256,512,…}`,LDS `(BM+BN)·320 ≤ 163840` 也正好卡等号。⇒ 入口是 **scale slab 重排**,不是改发射侧 | pitfalls/05 · pitfalls/01 |
| **缩窄** tile 的 N 边去消 padding(如 wgrad 用 256×192 对齐 2880/5760),含把 LDS pool 由 96 列补到 128 列恢复 2 的幂 swizzle | ❌按实测参数反解为负 | 同一条 `(BM+BN)/(BM×BN)` 定律反着用:算术强度 128→**109.7 B/cell**、WG 数 +25%,而 MFMA 与今天的短-N 体**完全相同**(两者 N padding 都已是 0)。按实测轮次分解:真 96 列 pool +0.9%(在 ±0.7% 噪声内)、**补到 128 列 −1.0%** —— 补 pool 宽度这步本身就把省下的 B 字节还了回去 | pitfalls/03 |
| 把死分支(half-N 的 BR)的 g2s 改成 **wave-uniform(去 `offen`)** | ✅sound·量级小 | 保留指令条数/发射序 ⇒ 主体的 graded vmcnt 原样成立,但 8 line → 1 line。det 18×60 全 0;真 bench 配对 +0.08% = 平(死 line 本就 L1 命中) | pitfalls/05 |
| DMA/`buffer_load_lds` 免 reg→store(attention & GEMM) | ❌实测DEAD | register-prefetch 对 scattered-gather 最优,DMA s_waitcnt 暴露 HBM;attention 版被 flydsl 编译器 bug 挡 | pitfalls/03,04,12 · [[project_dsv4_fwd_dma_experiment]] |
| triple-buffer / prestore 隐藏 exposed store | ❌实测DEAD | store 是 work-bound(工作量不减)/ 占 occ;重排无用 | pitfalls/12 |
| 免一个操作数转置(`ds_read_b64` 换 `_tr`) | ❌实测DEAD | ISA 等价,转置读不比 plain 贵,0 收益 | pitfalls/05 |
| padding 消 bank 冲突(实测本无冲突时) | ❌实测DEAD | BANK_CONFLICT=0 时 pad 白吃 LDS;不对称 swizzle 会静默读错行 | pitfalls/03 |
| register double-buffer/read-once/RING/monohoist(mxfp4) | ❌实测DEAD | 2 operand set 超寄存器预算→RAGreedy crash/spill | pitfalls/01 |
| 把边界块的 g2s 收窄到只载它读的那几列 | ⚠实测 0(±0.3%) | 边界块只占 1/12 的 tile,省下的字节落在 occ≈2 的 sibling wave 已盖住的延迟里;5 组种子 −0.51/−0.62/+0.42/+0.24/−0.44%,逐位相同。要动 feed 就去量**主体块** | pitfalls/04 |

## C. quant / scale(mxfp8/mxfp4)
| 你想试的动作 | 判定 | 一句根因 | 详卡 |
|---|---|---|---|
| 把残余 gap 归因「scaled-MFMA 指令税」 | ❌结论错 | 指令税≈0;真凶=scale VMEM load 未隐藏 | pitfalls/05 |
| ★ 主环 MFMA 换更大形状 `16x16x128_f8f6f4`→`32x32x64_f8f6f4` 减一半指令 | ❌实测DEAD −2%(12/12) | 同 FLOP 下 `MFMA_BUSY` **逐位相同**(实测 32.0 / 64.0 拍/条),换形状不产吞吐余量,只把每相位独立累加链 8→2 ⇒ SrcC RAW 与 barrier convoy 更难遮 | pitfalls/05 · connection/common/05 |
| 一相位共享单个 sa/sb 释放 16 个常驻 scale VGPR | ❌实测DEAD −2% | 指令流逐条不变、VGPR 246→230 也没用 ⇒「unscaled 探针 +2.9%」是 **DCE 复合**(连带消 scale load/地址链),不是杠杆 | pitfalls/05 |
| `scale_pack`/opsel byte-pack(生产 per-K) | ❌实测DEAD | 现有流水已预取藏 scale;前提是「裸无预取」,生产不适用 | pitfalls/05 |
| scale 也暂存 LDS / scale via LDS 双缓冲 | ❌实测DEAD | 慢 6×/0.42-0.61×;只暂存 fp8 数据不暂存 scale | pitfalls/05 |
| 大 BM 但不跨-lane 合并的 quant 写 | ❌实测DEAD | 0.58-0.98×;死的是「不合并」,大 BM+LDS 合并才是制胜招 | pitfalls/05 |
| whole-loop(WL)移植进生产 | ❌用户裁定 | 2500 行 bare-asm 不可维护;occ=1 结构上限,per-K 才是生产路径 | pitfalls/05 |
| 优化 quant kernel 撬 e2e(grouped) | ❌实测DEAD | quant 绝对占 35-46% 但 e2e 中性;缺口在 gemm(occ=1 上限) | pitfalls/05 |
| e8m0 scale 广播前 cast Uint8 | ❌静默错 | 高位损坏 match~9% 垃圾;位运算全程 Int32 | pitfalls/05 |
| scale 流 tile-major/cache-line 共置重排 | ❌实测DEAD | subtractive-pin 证伪「scale-thrash」:四条地址流合计 ≤4% 超额 miss,scale 本就驻留;定价 ~0.002% wall | pitfalls/05 |
| 在功耗墙(≥99% TBP)核上继续削 DRAM 字节 | ❌实测DEAD | 387 MB 超额全消掉只值 ≤0.18% wall(DRAM 占能量 0.8~1.6%);~~★下一刀=LDS→VGPR 读放大~~ **已作废(2026-08 实测两边同档)**:标尺自己就是 2.91× 而非闭式算的 2.0×,且它每 WG 每 K-iter 发的 ds_read 是我们的 1.94 倍;时间口径上 LDS 读被 MFMA 完全遮住。真正的对标差距在**同步指令密度**(7.4×) | pitfalls/05 · methodology/12 §标尺③⑦ |
| atomic 融合 reduce(dense split-K) | ❌实测DEAD | 同地址 HBM atomic 争用串行;split+reduce 是答案 | pitfalls/05 |
| 优化 scale preshuffle 核的**搬运**路径(宽存/LDS staging) | ⚠️先量 VALU/dword | 它常是 **VALU-issue bound**:per-thread O(G) 组扫描 759 VALU/wave 搬 8 dword;换 lane-resident 表 −54% 核时间(gm +1.1%) | pitfalls/05 |
| mxfp4 grouped whole-loop 寄存器级 operand 双缓冲 / 拉长 ds_read→MFMA 距离 | ❌实测DEAD | 只剩 80 VGPR 装不下 128;ATT ds_read stall 0.05-0.12 cy/条;零成本的 cell 重排等价探针 ±0.15~+1.2% | pitfalls/05 |
| mxfp4 grouped NT prologue 放宽 `wait_barrier(N>0)` partial drain | ❌实测DEAD | 该段是 issue backpressure(119 cy/条)不是 tail latency,放宽屏障 ±0.2% 平局 | pitfalls/05 |

## D. attention(fwd + bwd)—— 优化顺序 playbook 见 methodology/15;实测 win/dead 见 pitfalls/12(dsv4)+13(hd64 dense)
> 优化前先读 **methodology/15-attention-optimize**(定 bound→fwd:消/藏 store·_FMAX0·dual-wave / bwd:减 MFMA·融串行核→藏 MFMA 延迟→exp2·occ·确定性)。

### D. attention(dsv4 sparse-MLA)—— 详见 pitfalls/12
| 你想试的动作 | 判定 | 一句根因 | 详卡 |
|---|---|---|---|
| s_setprio(fwd / bwd dQ) | ❌实测DEAD | latency-bound,优先 MFMA 饿死 VALU/read | pitfalls/12,09 |
| s_setprio(1/0) 包 **MFMA-dense 的 GEMM2**(≥2 waves/SIMD 的融合 bwd body) | ⚠️**regime 相反=强正** | **拿掉它 −2.8%(9/9)**。判据不是"哪个 kernel",是 **regime**:①同 SIMD 有 ≥2 个共驻 wave(有仲裁对象),②该 region 一个 wave 有 MFMA run 而兄弟没有(GEMM2 / carrier 的 GEMM3),③body 是 MFMA-pipe-serialized(MFMA≈47%)。三条齐 = 正;fwd 的 VALU+read-bound / `waves_per_eu=1` 无仲裁对象 = 负。**幅度无关**:1→2 持平(全 wave 同码,比较恒为"in-region(1) vs not(0)")。对称推广到 GEMM1(全 8 wave 都有 MFMA run)= 7/11、6/11 噪声,即使 hazard nop 198→102 | pitfalls/12,13,09 |
| cross-tile 软件流水(XPIPE) | ❌实测DEAD | cr4 LDS-port 饱和/few-tile prologue > 重叠 | pitfalls/12 |
| K=32-PV(2-tile batch) | ❌实测DEAD | 批打断 QK→softmax→PV 交织(occ-1 藏延迟唯一机制) | pitfalls/12 |
| dual-layout-KV 消 QK bank 冲突 | ❌实测DEAD | 多写一份 KV 压在 occ-1 关键路径 −7~17%;冲突本 off-critical | pitfalls/12 |
| register-transpose PV(fwd) | ❌实测DEAD | 每 tile 重做转置不摊薄 −77%(bwd rtr 赢是摊在 rank-tile) | pitfalls/12 |
| query-blocking(cr128 减 store) | ❌实测DEAD | 2×o_acc>256 arch-VGPR→强制 occ-1,占用损失>store 省 | pitfalls/12 |
| ✅ pro cr0/cr128 interm(慢 triton ~20%) | ✅开口 | 研究 triton 小 topk tiling/少 launch | pitfalls/12 |
| ✅ banded-SWA dKV 融合 / hybrid scatter | ✅开口 | 减 interm+gather 的 2× tensor HBM | pitfalls/12 |

| ★★★ **动「减 MFMA」这条结构杠杆前,先算它还剩多少可减** | ✅先算再决定 | `issued_atoms ÷ causal-exact flops − 1` = 冗余率。gpt-oss D=64 bwd 实测**只多 3.1%**(残差=对角带在 64×64 粒度的半三角)⇒ 那个 body 上整条杠杆封顶 3.1%,不该再开结构轮。>15% 才按「优先级最高」打,<5% 直接跳 stall | methodology/15 §BWD 减 MFMA |
| ★★★ **PMC 三元组的第二种读法:盯 `neither%`(stall)而不是 coexec,往往是更短的路** | ✅方法 | MFMA/VALU busy 是固定工作量 ⇒ `stall' = 1 − (M+V−C)/X`。同 body(44.0/36.4/8.5,stall 28.1%)要 1.155×:**只需 stall −48%,coexec 不动**;只走 coexec(8.5→17%)**仍要 stall −17%**。踩证:该核 r7–r13 八轮全押 coexec/调度族,合计只值 +0.4~4.0%,而 stall pool 从没被定价 | methodology/15 §coexec |
| ★★ **fwd 和 bwd 可能用不同的 MFMA 形状,别假设统一** | ✅先 dump 事实 | 同一代码库:bwd 五个 GEMM 全 `16x16x32`(40 轮标准化的结果),**fwd 两个 GEMM 是 `32x32x16`**(`utils/attn_helper.py:736`)。bwd 侧实测把 GEMM2 换成 32x32x16 = **+12% 更慢**("gfx950 上该 atom 让 pipeline cycle 翻倍")⇒ fwd 侧大概率有肉。换形状**不改精度**(都 bf16/fp32),但是真移植:fragment 布局/`ds_read_tr16`/寄存器/softmax lane 映射全变 | methodology/15 §MFMA 形状 |

## D2. attention(Meta/gpt_oss hd64 DENSE flash bwd,确定性)—— 详见 pitfalls/13
| 你想试的动作 | 判定 | 一句根因 | 详卡 |
|---|---|---|---|
| ★★✅ **先量 `SQ_VALU_MFMA_COEXEC_CYCLES ÷ SQ_VALU_MFMA_BUSY_CYCLES` 再选杠杆**(occ-1 bwd 的第一诊断量) | ✅**6/6 臂**按 wall 正确排序,灵敏度是 wall 的 5× | gpt_oss d64 dkdv = **19.2%**;MFMA busy 44% + VALU busy 36% − coexec 8.5% = 71.9% busy-union、28.1% 两管线全闲 ⇒ 完全重叠 **1.39×**。同 body 上指令数/arch dword **双向实测为 0**。⚠跨 kernel 比之前先比 `SQ_WAVE_CYCLES÷SQ_BUSY_CYCLES`(fwd 31.5 vs dkdv 7.7 驻留 wave/CU) | methodology/15 §occ-1 第一诊断量 |
| ❌ **把独立 MFMA 塞进 VALU 窗口(靠发射顺序)** | ❌ coexec 19.2%→15.9%,wall −3.0% | LLVM 把它 hoist 到发射区顶端;**放进窗口比放窗口外还差 0.8%**。要做必须配 `sched_barrier(0)` 钉调度区且**不新增 fence** | methodology/15 · pitfalls/09 |
| ❌ **靠"挪消费者"降 softmax 的 P-live 压力** | ❌ −4.0% / −0.9% / −6.3% | softmax 块**就是**寄存器峰值;拆 dP per-tile、把半区 dS/pack 推迟进下一半 GEMM1a、把 dV MFMA 塞进 softmax nt 循环,三向全负(第三向 spill 0→5)。要动先**缩短 softmax 自己持有的位宽/条数** | pitfalls/13 |
| ⚠ **tied `"=a,0"` 把 MFMA operand 钉进 AGPR** | ⚠只在 kv-block-invariant 处 ✅(+0.79/+0.53%) | 摊薄点决定符号:per-head-step 的同类操作数是 **−3.4% / −2.6%** 且更胖。⚠未绑定的 `"=a,v"` 会**静默删掉操作数**并伪造 +4.5%;asm-MFMA 累加器在 NT=2 上错 −0.9dB | methodology/06 §tied-operand |
| ★✅ **XCD-major block_id 解码**(`xcd=bid%8`,每 XCD 一整块 (batch,kv_head)) | ✅dkdv+2.81% / dq+0.54% | per-XCD 私有 L2,让共读同份 K/V 的 GQA WG 相邻(dq L2 hit 86.5→94.8%)。**内层轴 dq 要 kv-head 相邻、dkdv 要 q 相邻,选反=−2.5% vs +2.4%**。⚠推翻旧记的"XCD remap GPU-fault 判负"(那是实现炸了) | pitfalls/13 · methodology/05 |
| ★✅ **dq 派发降序 q_tile(LPT)** | ✅+2.50% | 因果工作量单调递增且**派发序=list-schedule 序**→长任务优先。⚠**别假设 in-order 最优**:dkdv 恰好已是 LPT 但 dq 是反的 | pitfalls/13 · methodology/05 |
| ★✅ **冒险锚点减量 per-slot→per-v4** | ✅dq+1.76% / dkdv+0.77% | clamp 是 MFMA→trans 冒险载体(非数值保护),但只需每 v4 一个;其余 slot 靠 inline-asm dead operand 钉序。`v_min` 72→3 / 256→64 | pitfalls/13 |
| ★✅ **causal-aligned q-tile origin** | ✅+0.96% | tile 数×BLOCK_M 的 padding 别落在因果范围最长的末 tile,下移到 tile 0 → 每 tile 少一个 BLOCK_KV 步(7481→7396 访问) | pitfalls/13 |
| ★✅ **`llvm.amdgcn.exp2.f32` intrinsic 替手写 asm+锚点**(dq) | ✅−0.79% | intrinsic 本身即编译器可见的累加器读→编译器自插等待周期,锚点粒度问题消失。⚠**别推广 dkdv**(那边 +1.06% 更慢) | pitfalls/13 |
| ★❌❌ **锚点放大到"组级"**(per-GEMM1a-block / per-mt-pair / min3-group) | ❌**更快但算错** | 快 2.53%/1.79%/0.31%,但 dk/dv **16.9~19.7dB 且 det=FALSE**,**只在 q_split=4·BLOCK_KV=128 或 BLOCK_KV=64 暴露**。★规则:锚点必须读**它保护的那个 v4**,"读遍组内每个 v4"不充分 | pitfalls/13 |
| ✅ drop 冗余 B-GEMM / rho-R 全局修正(dq) | ✅+9.8% | 真结构性减 MFMA,先审计每核有无可丢项 | pitfalls/13 |
| ✅ odo delta 融合(消独立 delta 核) | ✅+5.3% | 全串行辅助核随 Sq 放大;须 out/dout→q.dtype cast | pitfalls/13,02 |
| ✅ exp2 软流水(GQA head 轴) | ✅+1.4% | 藏 head h+1 exp2 进 head h GEMM2 shadow | pitfalls/13 |
| ✅ q_split 铺满 CU grid(dkdv) | ✅关键 | 旧上限卡 2 CU 空转;qsp≥4 铺满→hw 验收 1/4→2/4(4096 594→702) | pitfalls/13 |
| ✅ 冷 load 寄存器预取(dkdv 下一head lse/delta) | ✅+1.81% | occ-2 latency-bound 上唯一 register-neutral 净胜(dt==DT-1 发射) | pitfalls/13 |
| ★✅ **把上一行的预取延伸过 q-block 边界**(最后一个 head-step 改取下一 q-block 的 head 0:Q/dO+lse/-delta,值走 q-loop 的 iter_args) | ✅融合 bwd +0.29%/+0.55%(2 session);配 GEMM1 **ks-outer 发射** = **15/15 +0.57%** | head 轴预取只覆盖 h→h+1,每 8 个 head-step 就有一个从裸 HBM 延迟起步;单 WG 独占 CU 时没有别的 WG 覆盖这个 prologue。ks-outer 与它互补(前者要操作数已到位,后者保证到位) | pitfalls/13 |
| ⚠️ **跳过对角(masked)q-block 里"kv 行全在因果边之上"的 wave** | ⚠️真做 +0.22/+0.24/+0.33%(3 session,**输出逐位不变**);上界只有 +0.37% | **别按 visit 数估价**:causal kv-outer 里对角块是 6.06% 的 visit 却只有 0.4% 的时间(差 5×)。先跑"所有 wave 都跳"的诊断探针定价再决定实现。代价:16 个 accumulator 穿 scf.if phi ⇒ VGPR 249→256、8 dword spill | pitfalls/13 |
| dQ split-K 的 band-pair 折叠(nb 32→16,省一半 workspace 字节) | ⚠️**钱大(诊断 +6.0% 读侧,合计≈+11%)但需先腾 64 dword** | 一个 WG 的 dK/dV accumulator = (WG 的 kv 行数×D×2)/512 线程 dword,**与 wave 间怎么切无关**(D-split 只换个 wave 放);两个 WG 各拿一半 D 则要各自重算 S/dP(+1.6 ms)。已知腾挪报价:K packs 出寄存器 = 32 dword / −2.0% | pitfalls/13 |
| ★✅ **dQ split-K workspace 的 band 交错**(`[band][B][Sq][Hq][D]` → `[band/ILV][B][Sq][Hq][band%ILV][D]`) | ✅gpt-oss **D128** fused bwd +1.3%(813.2→819 conv TF,ILV=8) | 降序 q 走法让 64 个 band 同时写**同一批 q 行**;slab 布局下这些写分散在 268 MB 间距上 = 一行开 64 个 DRAM page,交错后并发 band 一起填满同一页。ILV=1/2/4/8/16 → 813/812/820/819/820 ⇒ **4 以上就平,机理是 page 局部性不是 cache line**。输出逐位不变、寄存器不变(`band%ILV` 是 wave-uniform) | pitfalls/13 §2026-08-10 |
| ❌ **同族的 head-major 工作区**`[band][B][Hq][Sq][D]`(想让单 WG 的 head-step 连续 16 KB) | ❌实测DEAD **−5.8%** | 与上一行相反方向:**跨 WG 把同一批 q 行填满** 比单 WG 自己连续重要得多。改工作区布局前先想"同一时刻是谁在并发写同一页" | pitfalls/13 §2026-08-10 |
| ⚠ **改 SRD 覆盖范围的探针必须先算 `num_records` 溢出** | ⚠**假收益陷阱** | band 交错扫到 ILV=64 时 slice=`B*Sq*Hq*D*2*ILV`=8.6 GB 溢出 32-bit ⇒ 分块写被硬件**全部丢弃**,测出 **+18%** 纯属虚假。凡是放大 buffer slice 的实验,先验证 <4 GB 再信数字 | pitfalls/13 §2026-08-10 |
| ★ **用"别名/丢弃"探针量字节的边际单价,别拿 roofline 乘** | ✅方法 | D128 fused bwd 上把 64 个 band 全别名到同一 slab(省 ~8.5 GB 写)只值 0.60 ms = **0.071 ms/GB**,而 roofline 是 0.161 ms/GB ⇒ **流水化之后系统性高估 2.3×**。同理减法探针要分层:store-only 1.06 ms / store+GEMM3 1.82 ms,只做后者会把 72% 的计算记成字节 | pitfalls/13 §2026-08-10 |
| ❌ dQ split-K band-pair 折叠 @**D128**(nb 64→32) | ❌**实测DEAD,三种形态全封**(campaign 20260813_154001 r1–r5) | (a) **寄存器折叠**(一个 WG 两 band,唯一不重算的形态):`bkv=256+kv_halves=2+block_q=32` 实测 **510 dword / spill 0 是地板**,`g3d=1`/`g2d=1`/`exp_iglp=0` 三个真供体逐字节不动它(256 dw 累加器塞满 A 堆、arch 堆随后也满 ⇒ **分配器把文件占满**,不是"供体腾错了类"),其余供体 spill 247–1201 或 LDS 溢出;而 dqred 的共驻线最高只有 **488**(4-wave WG×24 dw;部署的 8-wave 是 464)⇒ 必然驱逐 reduce。(b) **驱逐单价** = **0.46 ms**(见下面 full−nored 那行)vs 折叠全部收益 0.58(读 0.38 + 写 0.20)。(c) **重算**(sibling role K/V)r4 实测 **−0.97 ms**。附带:两 band body 还多 **54% ds_read_tr16**(两个 kv half 各自重读 q 侧算子),唯一能修它的 `kv_halves=1` spill 577 | pitfalls/13 §2026-08-13 |
| ★ **split-K partial 往返的两端要相反的 cache policy;判据是 UMC 点数而不是速率** | ✅**方法**(任何「写一次、被另一个 kernel 读一次」的 workspace) | gpt-oss D64 fused bwd 两端各自定价(每读一个独立进程、两臂交替):**读端(dqred 的 partial load)要 non-temporal** —— 换 cached 是 5.8232/5.7802 ms 对 5.7146/5.7309、8.155/8.126 J 对 8.025/8.055,而 **UMC 一点不动** ⇒ 多花的 0.1 J 不是多搬字节,是留住 4.4 GB「读完即死」的 partial 把 body 自己的 Q/dO 驻留挤没了(= MALL 是被**争用**,不是仅仅太小)。**写端(body 的 G3 partial store)要 cached** —— 换 NT 是 5.7364/5.7545 对 5.7021/5.7278、8.109/8.089 J 对 8.016/8.076,**UMC 25→27(+~1.3 GB)**:每条 store 只覆盖 64 B,cached 时两条在 L2 合成一条 128 B 行,NT 之后合并丢失、远端要为半行做 line fill。⇒ 读端按「这批字节还有人读吗」选,写端按「一条 store 覆不覆盖整行」选 | pitfalls/13 §⑩ |
| ⚠ **band 交错(ILV)的收益按「同一 partial 行的并发写者数」缩放,不按 head dim** | ⚠**per-shape**:D128 +1.3%,**D64 full causal 三次全平** | 同一份 `[band/ILV][B][Sq][ILV][D]` 布局在 D128(64 band 并发写同一批 q 行)值 +1.3%;D64 上早期 −0.14%、QDESC 打开后 5.9721 对 5.9479、本轮能量表 5.7127 ms / 8.046 J 对部署 slab 的 5.7035 / 8.023(加 group pad 5.7105 / 8.041)。D64 一条 q 行只有 8 个 band 在写,而这 8 个 band 的 8 个 kv head 的 WG 已经把那一页填满 ⇒ 布局没有散射可收。配 `QDESC_R=1`(把全部共驻 band 压到同一个 q block = 制造最大散射)也救不回来:**+2.1%**,且 sclk 升到 2199 而功耗 1396 W 未贴顶 = 「改动没到达字节」的再定时特征。⇒ 排这条杠杆前先数并发写者数,≤8 就别指望它 | pitfalls/13 §⑪ |
| ★ **split-K partial 字节/flop = D/A(A = 一个 WG 的累加器 float 数),与循环序无关** | ✅**定律**(attention bwd 通用) | kv-outer 持 A=`BLOCK_KV*D*2` ⇒ 写 `(Skv/BLOCK_KV)*|dQ|/2` = **B·Sq·Skv·Hq·D²/A**;q-outer(dQ 进寄存器、dK/dV 出 partial)持 A=`BLOCK_M*G*D` ⇒ 也是 **B·Sq·Skv·Hq·D²/A**(partial 若须 fp32 再 ×2,所以 kv-outer 是较优的一侧)。**换拆法换不掉 partial,只能换 A**。推论:partial 字节/flop **线性于 head dim** —— D128 恰好是 D64 的 2×,这就是 gpt-oss fused bwd 的 D128:D64 分数差本身。能动的只有三样:更大的 A(512 dword 池是硬件)、累加器换 LDS 层(bkv=256 时单是 dV 就 128 KB,而 K tile+Q/dO+dS 已吃掉 96/160 KB)、重算(已定价为负) | pitfalls/13 §2026-08-13 |
| ★ **full − nored 分解:把"旋钮拖慢了 body"与"旋钮把共驻核赶走了"分开** | ✅**方法**(任何两核共享寄存器池的流水) | 越过共驻线的旋钮读起来就是一次普通的调度回归,量级还很像。同一候选各跑一次"带共驻核 / 去掉共驻核",两项就分开了:`g2d=1` 看着是 −0.6 ms 的 ring-depth 回归,实测是 **0.15 body + 0.46 驱逐**。同一把尺还证伪了"共驻 reduce 是每线程 MLP 不足"(在装得下的 body 上开 uc=2 只值 0.014 ms)与"按生产顺序读能吃 MALL 余温"(lpt=0 值 0.004 ms)⇒ 该核的暴露时间是**按字节走的共享 fabric 成本**,形状类旋钮全部无效。★**限定(campaign 20260819 r2/r3 实测)**:「形状类旋钮无效」只管 **fold 自己的速率**;它**加在 body 上的共驻代价**另有一条轴,≈ `fold 字节数 ÷ 与它并行的 body 量`。同样的隐藏字节切成 8 条窄切片投递比一条宽突发少 **1.35%** wall,而把同一批 fold 摊在 3 份 body 上而不是 2 份(不等宽 chunk 计划 [1,2,1])再拿 **1.8%**。**突发宽度是二阶项、到 8 片就饱和**(16 片/按宽度缩放各 8 读数全平),「摊在多少 body 上」才是一阶项 | pitfalls/13 §2026-08-13 |
| ★✅ **把 `-log2e*lse` 折进 GEMM1a 的 C-init —— 掩码 tile 也要折,别只折 unmasked** | ✅ gpt-oss D64 fused bwd **−0.98%** | attention bwd 的对角 q-block 走的是 body 的**另一份**(更贵的)q 循环。零 C-init 让每个被掩码元素额外付一条 identity fma,**更贵的是那条 fma 对 MFMA 累加器的读所拖进来的 hazard nop**:折进 C-init 后掩码循环 6338→**5402** 条指令、`s_nop` 等待 779→**222** cycle/八头 trip,unmasked 循环逐字节不变,+4 vgpr(463,与 459 同属 464 granule ⇒ 共驻线不动)、spill 0。**别按 trip 占比估收益**:对角块只占 6% 的 trip,却兑现了 ~1% wall。窗口形状通常早就折了,**full-causal 那一支容易被漏掉** | pitfalls/13 §2026-08-19 |
| ★✅ **掩码(对角)q-block 按 **wave** 分成 below/diag/dead 三臂,而不是让每个活 wave 都算完整掩码** | ✅ gpt-oss D64 fused bwd **−0.59%**(18 对回文/3 session/14-18 同号/三个 session 均值全负);逐位与旧路径相同 | 当 `ROWS_PER_WAVE_KV == BLOCK_Q` 且形状为方阵时,掩码 q-block 首行与 band 首 kv 行的差恰为 `(m-w)*BLOCK_Q`(causal offset 相消)⇒ **一个 wave 必落在三类之一,没有半对齐**:`w<m` 完全不必掩码、`w==m` 的 16-tile 按 `nt` vs `mt` 判(`nt>mt` 整块 P=dS=0,连 MFMA 都不发)、`w>m` 只发布零。四 wave 被 barrier 锁在同一 head-step ⇒ **代价是各 wave 的 MAX**,所以分类后 diag 臂 MFMA 128→88、该 block 临界路径 675→**560**(= 内部 q-block 原价,below 臂成为地板),`m=0` 时→430。被删的 MFMA 的 B 操作数是**精确零** ⇒ dq/dk/dv 与旧版 `torch.equal`(D64+D128 都验) | pitfalls/13 §2026-08-19 ⑤ |
| ★⚠ **occ=1 body 上做多路特化:用"顺序 if 串 + identity else",别用嵌套 if** | ⚠**结构决定寄存器**:嵌套 vgpr **476 > 464 granule**;顺序 **460** | 同一个三臂特化,嵌套形式(外层 live/dead、内层 below/diag)让两个臂都产出新值,分配器必须同时留住两套 yield ⇒ 掩码循环 219 AGPR、vgpr 476,直接顶穿 co-resident dqred 的共驻线;改成三个**顺序**的 wave-uniform if(每个 else 臂原样 yield 传进来的累加器,可平凡合并)后 vgpr 460、spill 0、LDS 不变,掩码循环还少 468 条指令。代价只是每 head-step 多一条 wave-uniform 分支 | pitfalls/13 §2026-08-19 ⑤ |
| ❌ **GEMM2 的 dt 预取环从 depth 1 加深到 2 @D64**(= "给 LDS 读提前一个 MFMA 组做 double-buffer") | ❌实测DEAD **+1.9%**(6/6 回文全负) | ISA 如预期:未掩码热循环 `s_waitcnt` **少 26 条**。但 wall 稳定变慢 ⇒ **`s_waitcnt` 条数不是代价口径**,等待挪走的同时读突发变宽、顶开 MFMA 发射。同一个 depth=2 在 **D128 上是部署值** ⇒ per-shape,别跨 head-dim 搬。⚠ 同族的"给 GEMM3 的 K^T 做 double-buffer"在 D64 上**没有接线点**:`G3_KREG` 默认已开 = K^T **整 band 驻留寄存器**(比 double-buffer 更强),热循环 K^T 的 `ds_read` 本来就是 0(关掉它反而 +256 条/trip) | pitfalls/09 · pitfalls/13 §2026-08-19 ⑥ |
| ★❌ **热循环的 `s_nop` cyc 不是可回收的池子(occ=1);要动区域调度先量 LDS 覆盖距离** | ❌ 整族判负 **+0.42~+3.48%** | gpt-oss D64 fused bwd,500 `s_nop` cyc/trip(占 trip 8.9%)看着是全库最大单项。把 `iglp_opt(2)` 换成手搭 `sched_group_barrier` 流水(同一 LLVM mutation,互斥):`"m2,t2,v4"` 让**指令 −54、`s_nop` −84 cyc、MFMA 连跑 1.56→1.83、vgpr 同 granule** 四个静态量齐变好,wall 仍 **+0.80%**;`"r1,m2,t2,v4"` 抽掉 150 nop cyc(2.7% of trip)**兑现为 0**,反而 **+3.48%**(0/6)。唯一与 wall 单调对应的量是 **`ds_read` 发射→退休的覆盖距离**(按 lgkmcnt 队列语义模拟,`_isa_run.py`):32.4 / 26.4 / 19.7 ⇒ 0 / +0.80% / +3.48%,单价 **0.2~0.3% 每条覆盖指令**。机制:occ=1 时 hazard nop 与旁边的 lgkmcnt stall **是同一次 stall**,静态分析数了两遍。另:流水里**点名一个类就是钉住它**(要提前读得用大于区域读数的领头组 `"r24,m32"`,不是 `r1`);而**换掉 mutation 本身就要付 0.9%**(`"r24,m32"` 保住覆盖、指令 −31 仍 +0.90%,因为 iglp 还在藏 78% 的 exp 链);策略 0 顶穿寄存器、策略 1 在带 `ds_write` 的区域触发 LLVM 断言 | pitfalls/09 §sched-hints · pitfalls/13 §2026-08-19晚 ⑦ |
| ⚠ **减法探针的定价会随 body 变快而变大,要定期复测;拆混合结论用两因子固定法** | ✅方法 | 同核 dQ fold:`nored` 首次 −4.9%(0.289 ms),body 快 4% 之后复测 **−5.84%(0.333 ms)**,4/4 —— 字节没动,是可藏它的 body 变少了 ⇒ **每次 body 的胜利都让 fold 相对变重**,它的优先级会自己爬上来。拆"边界项 vs 尾部项":q_split=8 有"判负 +1.32%"旧记录但那是配默认 7-chunk 计划测的;固定 3 个 chunk 后 `[1,6,1]`(尾 1/8) +8.01% 与 `[2,4,2]`(尾 1/4) +8.11% **同价** ⇒ 尾部只值 ≤0.1%,+8% 全是每 split 的固定成本(dk/dv slot 翻倍 + 每 band 每 split 多一次 K/V 驻留)。**源码里"每 tail subset 值 X ms"这类系数是在固定 q_split 下拟合的,不能外推到改 q_split** | pitfalls/13 §2026-08-19晚 ⑧ |
| ⚠ **"某资源还有余量" ≠ "有可用杠杆":先找不吃寄存器的喂法** | ✅方法 | 同核 LDS 用 86016 B / 160 KB,占用率由寄存器(460/512)锁死 ⇒ 多花 LDS **零代价**,而该 campaign 花了七轮想**省** LDS。但 `LDS_SLOTS` 钉在 1 是因为 Q/dO 走 VGPR 中转(`Q_PREF`),第二个 slot = **+32 个在飞 dword**,464 granule 只剩 4;`dma_grp>1`/`pf_ring` 另有 builder assert | pitfalls/13 §2026-08-19晚 ⑨ |
| ⚠ **quarter-rate `v_exp_f32` 链:先用 no-exp 探针量"暴露了多少",别按 1024 cycle/trip 记账** | ✅方法 | 同核 ISA:每 trip 64 条 `v_exp_f32` × 16 cycle = **1024 cycle**,占 5589 cycle/trip 的 18%,源码注释也自认 "VALU/exp2-issue-bound"。但把 exp 换成恒等(值错、确定性保留)只快 **4.05%** ⇒ **78% 的 exp 链已经被 `iglp_opt(2)`(MFMAExpInterleave)藏进 MFMA 影子里**,真正暴露的只有 ~226 cycle/trip。⇒ 报"exp 是瓶颈"之前先跑这支探针;藏得好的链再排一次序也拿不回那 18% | pitfalls/13 §2026-08-19 |
| ❌ **把 dQ 的 GEMM3 推迟一个 head-step(g3_defer)去填 exp 窗口 @D64** | ❌实测DEAD | `g3_defer=True` 本身 **+3.5%**(5.931 对 5.731,三轮回文),再把它挪到 q-half 的 exp2 窗口前 **+5.0%**。第二个 dS slot 的寄存器/LDS 在 4-wave body 上没有兄弟 MFMA 段可摊。⇒ 想给 exp 链找独立 MFMA,只能从**未推迟**的发射点想办法 | pitfalls/13 §2026-08-19 |
| swizzle/pad/prefetch/bank(消 tr16 冲突) | ❌实测DEAD | LDS 4-8× port 富余=latency-bound,非 port-bound | pitfalls/13,05 |
| P5 shared-GEMM1 融合(dq+dkdv) | ❌实测DEAD | q-outer=det 陷阱→KV-outer BLOCK_KV=64 但融合核 1.95-2.55× 慢 | pitfalls/13 |
| fp8 GEMM1 operands(FA-3 plain) | ❌实测DEAD | hd64 收缩维短,SNR 28.2<34 破门 | pitfalls/13 |
| dkdv 拆 dV/dK-only 核(冲 occ3) | ❌实测DEAD | 双份 Q/dO 读+拆分开销>占用率收益(-36%) | pitfalls/13 |
| LDS 双缓冲 | ⚠regime | 仅长 Skv 边际 +1.25%,短 shape -3.6% | pitfalls/13 |
| ~~长 Sq(8192/16384)full-causal 达标~~ | ★**已达标(2026-07-27)** | 旧记"确定性杠杆全 measure-closed、只剩放宽 det 或 research 级 exp-overlap"**已被推翻**——真正的开口不在 kernel 体内,在 **grid/派发映射层**(XCD+LPT+padding 对齐 = +7%)与**锚点粒度**(+3.6%)。**没有放宽任何确定性** | pitfalls/13 |
| ~~收官:perf B4 hw 16/20·fast 18/20~~ | ★**19/20 + 4-square 4/4**(turbo 侧) | 4-square hw 565/765/832/**877**(1.44-2.08×)、fast 4/4;20-config full **9/10**(既有 MI350 CK-det 是 0/10)、SWA 10/10(2.12-3.41×)。唯一未过 `newshape full 2048`=1.37×(98%)。**成果在 `sync/mxfp4/Primus-Turbo`,meta-attn 未移植** | pitfalls/13 |
| ★**GB300 对标(gpt-oss FUSED bwd,B3 Hq64,≠上面 Meta-dense-vs-H100)** | full **−2.7%** / SWA **+3%**(反超) | trace `trace/gb300/rank-0.json` B3 S8192:GB300 full 4252.5µs=**969.7 conv-TF** / SWA(128) 445.7µs=**289 eff-TF**(9.54× wall dividend)。我们(964b290a)full ~944 conv-TF、SWA 216→**298**(campaign `20260808_153438`)⇒ wall dividend 7.3→**10.1×**。追 full 那 3%=band-pair 折叠(§3)+occ-3。⚠旧 [[project_gptoss_sbhd_native]] 766 vs B200 812 是 **pre-fusion 不同 HW**,非回归 | pitfalls/13 §2026-08-09 |
| ★✅ **把下一步的预取发射点挪到本步重存储之前**(gpt-oss D128 fused bwd) | ✅**+3.9%** ratio 0.8754→**0.9094** | gfx950 **load 与 store 共用一个按序 vmcnt**:预取发在 4 条 dQ partial store 之后时,消费预取的 graded `vmcnt(7..0)` **必须先让那 4 条 store 退休**;挪到 store 之前,同样的等待变成 `vmcnt(11..4)`,store 留在飞行中。ISA 判据 = 整核 `vmcnt(0)` **16→1 条**(barrier/lgkmcnt/mfma 逐字不变)。**只跨过 store 边界就摆动 5.7%**(前 831.2 / 后 785.0 TF) | pitfalls/13 §2026-08-11 |
| ⚠ **上一行的前置条件 = 那个 wait 前面真的压着 store**(照抄前先查在飞是什么) | ✅判据 | mxfp4 grouped 上按这条去找形态,把 wgrad/NT body 里 5 个 `vmcnt(0)` 全排空点列为候选,ISA 审计后**全部作废**:在飞的清一色是 LDS 填充 load(g2s),**一条 store 都没有**,没有可越过的 store 边界。⇒ 判据两步:①先在 ISA 里数该 wait 之前有几条 store 在飞 ②再看 graded 阈值能不能把它们留在飞行中(`vmcnt` 6 bit,>63 条在飞直接不可表达) | pitfalls/05 §折叠 store · campaign 20260811 R12 |
| ❌ **用「减少 `s_waitcnt` 条数」当访存延迟的优化目标** | ❌实测DEAD | 同一个核:`dma_grp=2` 把 vmcnt 等待 89→**18**、寄存器还更好(v 455→432 spill 0),wall **−5.3%**;上一行把条数留在原地却 +3.9%。**成本是等待的深度(它前面压着多少在飞访存),不是条数**——DMA 分组只是把 graded 等待换成 group 边界上的整条 `vmcnt(0)`。`dma_grp=4` 另因 LDS 200704>163840 build fail | pitfalls/13 §2026-08-11 |
| ❌ **重排派发序换 cache 复用(hd64 dense fwd)** | ❌实测DEAD·两级 | L2 级(XCD 内按 kv-head 分组)=−0.19%,L1 级(把 8 个 GQA sharer 摆到同一 CU,离线证双射、共驻同流 512/768)=**−0.57%**。两次 `TCP_TCC_READ_REQ`/`TCP_PENDING_STALL`/`TCC_HIT` **逐位不动** ⇒ 请求流总量不变、只动时序。根因:**单 WG 自己的 K/V 双缓冲 32 KB = 整个 vL1D**,兄弟 WG 来读时行已被自己挤掉。⇒ 减请求只剩加大 block_m。⚠**作用域仅限 hd64 dense fwd**:同族改动在 gpt-oss D128 fused **bwd** 上**真的移动了请求量**(dkdv DRAM 读 6.98→3.66 GB,纯降序更到 1.77 GB)并兑现到 wall(+2.5%)——别拿这一行一票否决 bwd 侧的遍历序改动。★**但 bwd 侧的开口是 per-shape 的,先用别名探针定价再动 decode**(campaign 20260819 r4 实测):**D64** full-causal fused bwd 上 Q/dO 每 band 重读虽占该核 DRAM 流量 **68%(8.9 GB)**,把每个 band 的 Q/dO 取址钉回自己的首行(指令数/LDS 写/地址链全不动,只让读命中)却是 **+1.6% 更慢**,三读全部不重叠 ⇒ 该形状**没有可回收的重读池**,反而是把 32 条 band 压到同一份 128 KB 上造出 L2 端口争用。同一族的 `qdesc_r` 相位数(2/8/16)也全平。⇒ 「重读字节多」不等于「重读要钱」;**别名探针便宜,先量再决定要不要改 workgroup 解码** | pitfalls/13 §fwd · §2026-08-10 · §2026-08-19 |
| ★✅ **GQA sharer 合并(一个 CTA 跑 2 个 q-head 的同 index q-tile)** | ✅hd64 fwd **+2.04%** | `NUM_WAVES` 4→8 而 `BLOCK_M` 不变(低位 wave 选行组、高位选 sharer),两个 sharer 共用一份 K/V LDS tile 与 barrier ⇒ `TCP_TCC_READ_REQ` **−49.5%**(256→128 sector/WG-迭代)。**收益全在时钟**:sclk +8.7% @ 同功耗(96% TBP),真周期反而 +6.6%。LDS 只装 K/V 与 BLOCK_M 无关 ⇒ vgpr/占用率不掉档。⚠ 改 CTA 波数后所有以**指令数**计的 `vmcnt` 常量都要重算(`dma_wave_reps` 会变) | pitfalls/13 §fwd |
| ⚠ **"省访存还值不值钱"看 `TCP_TCC_READ_REQ`,别看 DRAM 带宽/L2 命中** | ✅方法 | 功耗墙(≥95% TBP)上,能耗在 **TCP→TCC 的请求/sector 路径**:pinned-tile 探针只改命中层级(请求数逐位不动)→ 只买到 +0.94% 时钟;真把请求数减半 → **+8.7% 时钟**。两者结论相反,别用前者否掉后者 | pitfalls/13 §fwd |
| 🚨 **把"减请求 ⇒ 买时钟 ⇒ 稳态也变快"这条从读路径搬到写路径** | ❌**实测不成立** | 同一台 100% TBP 机器、同一个 grouped fp8 核族:**写**请求 2.359e7→1.206e7(−48.9%,读侧 7.163e7 逐位不动)在 NN dgrad 上 6 点 `T=F+n·P` 重拟合出 **ΔF=−9.0 µs(−5.1%) 而 ΔP=+0.157 µs/iter(正号)** ⇒ 收益**全在 tile 的排空(F)**,稳态一点没被时钟抬起来。⇒ 读侧那条 `−49.5% ⇒ +8.7% 时钟` 的传递率**不能**按比例套到写侧;同族 NT 上出现的 `ΔP=−0.389` 要归给它**另一半**(存指令 160→96 + `v_cvt_pk` 160→96),不是请求数。**给写侧的臂定价:只承诺 ΔF,ΔP 另行测** | 本轮实测 · methodology/07 |
| ⚠ **rendezvous / barrier 池随 CTA 波数开出来** | ✅条件 | 同一个"全删 barrier"探针:4-wave CTA = **0%**、8-wave CTA = **+2.4%**。⇒ 任何"同步族已整族关闭"的结论只在当时的 CTA 规模下成立,放大 CTA 必须重测 | pitfalls/13 §fwd |
| ★✅ **把 4 个计算簇 rendezvous 减到 2**(每块 buffer 的 K+V 覆写 DMA 合并进同一个 P·V 簇,barrier 只留两个 QK 簇尾) | ✅hd64 fwd 8-wave **+0.33%** | 让每个 barrier-region「整块读一个 buffer、整块填另一个」,那一个 rendezvous 就同时覆盖它所填 buffer 的两条边 ⇒ 主循环 `s_barrier` 4→2、waitcnt 19→17,vgpr 128/spill 2/LDS 34048 逐项不变、输出逐位相同。**落点唯一**(前后各挪一簇分别破 WAR/RAW);两个 drain 必须收紧到 `vmcnt(0)`(同 region 两个 DMA,in-order vmcnt 放不下"只留一个在飞");循环尾 barrier 撤掉后**epilogue 必须开头补一个** | pitfalls/13 §fwd |
| ❌ **用「池子 ÷ barrier 数」给减 barrier 定价** | ❌实测DEAD | 全删探针:8 barrier +2.59%、4 barrier +1.30%、2 barrier **+1.25%** ⇒ 0.32/0.33/**0.63 %-per-barrier**,越少越贵(剩下的承担全部 WAR 边与全部 drain)。r19 记的"删一半恰好收回一半、线性可外推"被 r20 推翻,每减一档都要重测 | pitfalls/13 §fwd |
| ❌ **插 `sched_barrier(0)` 当占用率手段** | ❌实测DEAD | 主循环四类站点 10 种组合(单/双/三/四点)`.vgpr_count` **全部恰好 150**。r12 那次 168→150 是"切开 inline-asm cvt 与 MFMA 组"这个站点特有的冒险/调度效应,不是普遍规律 | pitfalls/13 §fwd |
| ⚠ **压寄存器前先打逐块 max-VGPR 直方图** | ✅方法 | hd64 fwd 九个基本块同时顶在 142~149(预算 150)⇒ 压任一窗口都拿不到分配,解释了连续三轮"活跃度降了、`.vgpr_count` 不动"(含把 4 宽累加器退出 loop-carry 反而 150→154)。只有某块明显高出才做局部优化,否则改**全局常驻集** | pitfalls/13 §fwd |
| ★**先查 grid/派发层再抠 kernel 体** | ✅方法 | kernel 内部(tile/双缓冲/遍历序)调几周只值 ~1%;XCD-L2 亲和+LPT+padding 对齐 = +7%。清单:①co-resident WG 是否共享同份数据 ②派发序是否=list-schedule 序且工作量单调 ③padding 落在最贵还是最便宜的 tile | pitfalls/13 · methodology/05 |
| ★**bench 必须镜像部署配置** | ⚠**曾优化错配置** | `_bench_*` 直调 builder 用默认参数,而 `_get_bwd` 部署传另一组(dq block_kv/wpe、dkdv fold_lse)→**差 1.4%**。修法:让 **builder 默认值本身=部署值**,别靠每个 bench 记得传参 | pitfalls/13 · methodology/01 |

## D3. attention(Meta/gpt_oss hd64 DENSE flash **fwd**,full-causal S=16384)—— 详见 pitfalls/13 §fwd
| 你想试的动作 | 判定 | 一句根因 | 详卡 |
|---|---|---|---|
| ★✅ **`_FMAX0` 固定参考 max**(零参考,删 per-tile reduce_max + O 的 rescale + `s−m`) | ✅**+13.5%** | softmax 平移不变;hd64 bf16 + Q 预缩放下 score 量级可控,SNR 51.26dB。**不需要卡片写的 first-pair 额外 QK** | methodology/15 · pitfalls/13 |
| ★✅ **LPT 降序 q_tile 派发**(+ loop-carry 拆分抵掉寄存器) | ✅+2.34% | 与 bwd/dq 同型;裸改会 vgpr 162→170 跨悬崖,必须配套省寄存器 | pitfalls/13 · methodology/05 |
| ★❌ **把 softmax 加法打包成 `v_pk_add_f32`** | ❌实测DEAD **−1.45%** | **gfx950 上 `v_pk_add_f32` 对 f32 没有吞吐收益**:46 条 pk_add 替掉 86 条 `v_add_f32`(净减 40 条)、vgpr 不涨(166→166)、SNR/det 全过,wall 却掉 1.45%。⇒ 每条 64-lane f32 VALU op 固定占 4 拍,打包只省代码体积/寄存器。**推翻旧记的「打包判负是因为 +8 寄存器对齐税」** | pitfalls/13 §fwd |
| ★❌❌ **`v_dot2c_f32_bf16` 吃已打包 bf16 P 做 row-sum** | ❌**算错**(SNR 23.7-24.1dB,det=False) | LLVM 选的 **VOP2 `v_dot2c` 形式不遵守 `op_sel_hi`,低 lane 被算两次**;换 intrinsic、换 4 路轮转累加器、加 32 拍 `s_nop` 三种修法全无效(误差稳定在 6.25%)。**推翻 r6 记的「intrinsic 后 51.61dB/det=True」**。⚠2026-07-29 存疑:这个 24 dB + det=False 签名与「inline-asm cvt 前没插 `sched_barrier(0)`」那条完全一样,当时的 `s_nop` 修法没配 `sched_barrier(0)`(调度器仍把 cvt 交织进 dot2 之间)⇒ 重开此条前先按新规程试一次 | pitfalls/13 §fwd |
| ★❌ **per-wave cluster 再平衡**(把 row-sum 从 issue-bound 的 QK 簇搬到 MFMA-bound 的 PV 簇/访存簇) | ❌实测DEAD 0~−3.1% | **≥2 waves/SIMD 时管线是跨 wave 共享的,搬迁不改变每-SIMD 总需求量**⇒wall 不动。搬到 PV 簇 −3.08%(它在关键路径上)、搬到访存簇 −0.10% | methodology/03 · pitfalls/13 §fwd |
| ★❌ **跨-WG 相位错开**(`s_sleep` 按 `bid%3` 给共驻 WG 三种相位,640/1280 拍) | ❌实测DEAD −0.06% | 共驻 WG 本来就异相;"MFMA 空闲一半是因为 3 个 WG 锁步"这个假设不成立 | pitfalls/13 §fwd |
| ✅ 去掉 `_FMAX0` 下已失效的 `_anchor_v_p` + 簇头 `s_nop` 7→3 | ✅+0.31% | 固定 max 后 P 不再被 rescale,锚点无序可钉(省 8 条 `v_mov`/iter + 2 寄存器);簇头不再需要长停顿 | pitfalls/13 §fwd |
| ★❌❌ **减少 barrier 数量**(把访存簇折进计算簇,8→6→4 barrier/iter) | ❌实测DEAD 0%,**且静默破坏 8-wave** | barrier convoy 代价≈0(1114.7/1116.9 vs 1116.3,噪声内);**但 `stagger` 靠"给 group B 多一个 `s_barrier`"错相,总数一变两组配对错位** ⇒ block_m=256 8-wave 4/22 FAIL(cos 0.94),而计分形状 SNR 51.26/det=True 照过。**同步类改动必须多形状+多 wave 数验证** | pitfalls/13 §fwd |
| ★❌ **压 `v_s`/`v_p` 共存窗口冲 4 waves** | ❌只值 **4 个寄存器** | 文本上把 QK 排到 `cast_p` 之后:vgpr 166→**162**、逐位相同、wall 持平;再加硬栅栏彻底禁共存:仍 162 且 wall −0.90%。liveness 由**调度后**顺序决定;寄存器高水位分散在 6+ 个块(138~161),要 ≤128 得砍**全局长命值** | pitfalls/13 §fwd |
| ★★✅ **row-sum 换成 `v_mfma_f32_16x16x32_bf16` + ones A 操作数** | ✅**+1.96%**(1137.1→1159.9,交织 A/B 3v3 不重叠) | 打包好的 P 直接当 B 操作数,ones-mask `(lane%4==0)&&((lane//4)%2==(lane//16)%2)`,自己那行的和落在 **D 元素 0**,顺带折掉 half-wave 伙伴 ⇒ `v_add_f32` 68→**0**、`permlane32_swap` 2→**0**、vgpr 162→**150**、SNR 51.26→**51.61**。★**形状必须 16x16x32**:32x32x16 纸面即净亏(128 MFMA 拍 > 省下的 121 拍) | pitfalls/13 §fwd |
| ★★★ **让 MFMA 直接消费 `cvt_pk_*` 等 inline-asm 产物** | ⚠️**必须自己插 `sched_barrier(0) + s_nop 1`** | `rocdl.cvt_pk_bf16_f32` 在 ISA 里是 ASMSTART 块,`GCNHazardRecognizer` 不记它的 VGPR def ⇒「VALU 写→MFMA 读 SrcA/B 需 2 wait state」一个 `s_nop` 都不补。签名 = **SNR ~25 dB + det=False + 错误只落在一半的行上**。补上后不但正确,vgpr 168→150、还比让它自由交织更快 | pitfalls/13 §fwd |
| ★**先量边际系数再动手**(减法探针 + 每-SIMD 归一) | ✅**0.46 拍 wall / 拍 VALU** | 旧记 0.49 来自 SNR 崩掉的脏探针;干净的 VALU 地板探针(exp 恒等 + row-sum 只折 2 元素,MFMA/cvt 条数不动,−520 发射拍)= **+16.6%** ⇒ 「省 VALU 这一族结构性为 0」**判负作废**(那个 0 来自三个没真减 VALU 周期的实验)。⚠ 减性探针必须保住下游消费链,否则 DCE 连带砍 MFMA 把读数虚高(+8% → +21.6%) | methodology/03 · pitfalls/13 §fwd |
| ★★✅ **把访存簇的 `s_waitcnt` 全排空下沉到消费它的计算簇尾** | ✅**+1.42%** | 「发 `ds_read` → `lgkmcnt(0)` 全排空 → barrier → 计算」把 LDS 延迟完全暴露;而该 buffer 真正被覆写在**两个 barrier 之后**,排空可下沉一个簇 ⇒ 编译器改用**增量 waitcnt**(4→16 条,PV 簇里 7 条与 MFMA 交织)。vgpr/SNR/det 不变。★**必须 `not STAGGER` 门控**:stagger 两组差一个 barrier,吃的正是那一个簇的余量(未门控时 8-wave rel_l2 0.0027→0.0294) | pitfalls/13 §fwd |
| ★★✅ **换 LLVM 调度策略**(`compile_hints["llvm_options"]`:`amdgpu-sched-strategy=max-memory-clause` + `enable-post-misched`) | ✅**+0.6~0.7%**,零源码风险 | flydsl 的 `llvm_options` 是**逐项 save/restore 的 scoped ctx**,不泄漏到同进程其它 kernel ⇒ 可进产线。K/V 从 LDS 分块读散布在各计算簇 ⇒ memory-clause 策略正好把这些读聚簇:主循环 `s_waitcnt` **94→66**,vgpr/spill/LDS 全不变,输出逐位相同,非计分配置(S=8192 / SWA)同向 +1%。`max-ilp` +0.4%、`post-misched` 单独 +0.1%。⚠**`iterative-ilp` 是最快的一臂但 SNR=nan/det=False**(重排掉手工 `sched_barrier(0)`),`iterative-minreg` spill 144 ⇒ 换策略后必须重跑 SNR/det。❌惰性:`amdgpu-use-amdgpu-trackers`/`schedule-relaxed-occupancy`/`misched-postra`/`schedule-metric-bias` | pitfalls/13 §fwd |
| ★**多臂 A/B 用回文序而不是同序重复** | ✅方法 | 同一探针驱动里有**单调位置漂移**(第 1 位比第 9 位低 0.5~1%),同序重复 N 轮会系统性判负靠前的臂;回文序(A B C C B A)让每臂平均位置相同,才分得清 0.1% 级差异 | pitfalls/02 · pitfalls/13 §fwd |
| ❌ **在 region 内挪覆写 DMA 的落点换 slack** | ❌整族已探尽 | region 头 −0.2%(挤掉紧邻 MFMA 的那次 LDS 读)、第二访存簇 中性、K/V 拆两簇 −0.37%、提前两簇 中性、drain 全放开 −0.05%。⇒ 判据是「这个簇的 LDS 读有没有紧邻的 MFMA 消费者」,不是「离 drain 多远」;DMA 完成期限不是馈送池的成因 | pitfalls/13 §fwd |
| ❌ **数 `buffer_load_lds` 条数来判访存压力 / 去 DMA 指令冗余** | ❌**冗余免费** | 16-wave CTA 下两个 wave 取同一条 LDS line,按 wave 半区切分消掉 2× 冗余 = **0%**(1198.2→1199.7)。重复地址会被合并/命中 ⇒ 只看**唯一地址数** `TCP_TCC_READ_REQ` | pitfalls/13 §fwd |
| ❌ **`Q_HEADS_PER_WG=4`(16-wave CTA)** | ❌实测 −0.5~−0.8% | vgpr 128 下 16 波 = **1 WG/CU**,软件流水 prologue/epilogue 失去 WG 间覆盖而完全暴露;时钟 +3.6% @ −26 W 也补不回真周期 −4%。签名 = **小 S 掉得更多**(s8192 −2.1% vs s16384 −0.5%),即固定每-WG 开销 | pitfalls/13 §fwd |
| ★★✅ **K/V 四深 LDS 槽 + 主循环 2× 展开 → 1 barrier / 2 kv-tiles** | ✅**+0.34%**(1217.9 vs 1213.8,回文 A/B 区间不重叠;sclk 1698 vs 1716 MHz ⇒ 真周期 ≈+1.35%) | 一个 body 填的 4 个 buffer 与它读的 4 个不相交 ⇒ body 末尾一个 rendezvous 同时盖住 WAR 与 RAW。槽集每个 body 前进两位 ⇒ 展开两次让 slot id 全是 ds_read 立即数。★**余数必须谓词化而不是尾循环**(见下一行) | pitfalls/13 §fwd |
| ★★★ **展开一个贴着寄存器预算的循环体时,余数用尾循环 = 第三份 body 实例** | ⚠️**必然 spill**(2→35 dword,−19%) | RA 的压力来自 **body 实例数**,不是单个 body 的峰值活跃:2-deep 同样展开也 spill 30,而把尾循环去掉立刻回到 spill 0。正解 = 循环 step=4、第一个 body 无条件、第二个 body 包进 `scf_if_dispatch(j+1 < t_end, ...)`(条件 WG-uniform,body 内 s_barrier 合法),两份实例 ⇒ spill 0。⚠ 与「`scf.if` 包条件 barrier 打爆 RA(spill 97)」不矛盾:包整个 body 时所有 live 值经 scf.if result 一次汇合 | pitfalls/13 §fwd |
| ★**spill 的代价刻度(vgpr 128 / 4 waves)** | ✅标定 | 0→1217.9;5→−0.3%;10→−0.2%;**35→−19%**。个位数值 0.2~0.4%,30+ 是断崖 ⇒ `K_HEAD`/`V_HEAD` 这类"多驻留一点"的旋钮**必须在当前 spill 水位下重扫**(K_HEAD=2 在 2-deep 下 +0.18%,在四深展开下带 5 spill 反而 −0.39%;V_HEAD=2 带 8 spill = −0.9%) | pitfalls/01 · pitfalls/13 §fwd |
| ★**rendezvous 池 ∝ 共驻 WG 数,非单调,且每减一档更贵** | ✅标定 | 4-wave(3 WG/CU)0% ／ 8-wave 合并(2 WG/CU)+2.4% ／ 16-wave(1 WG/CU)**0%**;barrier 数 8/4/2/1 时池子 +2.59/+1.30/+1.25/**+0.91%**。⇒ 零池子结论要标注**共驻 WG 数**,别用「池子 ÷ barrier 数」定价 | pitfalls/13 §fwd |
| ★**给"少做了工作"的诊断臂扣虚高** | ✅方法 | 跳掉余数体的探针分数虚高 **0.8%**(实测 1222.5 vs 1212.2;S=16384/BLOCK_N=64 下半数 q-block 少做 2/129 个 tile = 0.78%,与实测吻合)⇒ SNR 崩掉的上界探针必须先扣掉少做的工作再判正负 | methodology/03 · pitfalls/13 §fwd |
| ★**读 ISA histogram 前先确认 region 是不是真的主循环体** | ✅方法 | 按"最大回边区间"取会把 **epilogue 圈进来**,本 kernel 因此把 VALU 高估 35%、把 `v_mov` 高估 19 倍、还把不执行的掩码块算进发射量。**判据:32 条 `v_mfma_f32_32x32x16` + 零 `buffer_store`**(barrier 数随轮次从 8 变到 1,按 barrier 数挑的老脚本会挑空;2× 展开后主区间是 64 条 mfma32 = 2 个 kv-tile 对)| pitfalls/09 · pitfalls/13 §fwd |
| tile/split 启发式按 **Sq** 选(而 grid 是按 Skv 铺的) | ❌实测DEAD | dkdv 是 KV-outer,grid ∝ Skv;矩形 Sq=2048/Skv=16384 上选反 **−19%**,方阵 Sq==Skv 永远测不出 ⇒ 验收表必须含矩形形状 | pitfalls/13 |
| 引用「fp8 MFMA 是 bf16 的 2×」而不写形状名 | ❌结论错 | legacy `32x32x16_fp8` 是 **32 拍 = 与 bf16 同速率**,只有 native `32x32x64_f8f6f4` 才 2×(且付 120 spill dword) | pitfalls/13 |

## E. autotune / dispatch / grouped-MoE
| 你想试的动作 | 判定 | 一句根因 | 详卡 |
|---|---|---|---|
| `id(tensor)` 作 cache key | ❌实测DEAD | 真实训练命中率≈0 | pitfalls/06 |
| 把 `M_per_group` 烘成编译期常量 / `assert M%BLOCK==0` | ❌实测DEAD | MoE 变长分布,静态划分崩 | pitfalls/06 |
| `BLOCK_M=128`(grouped/dense config sweep) | ❌实测DEAD | grid 写死 /256 → 少启动块假象,真实 1.55× 慢 | pitfalls/05,06 |
| per-shape `num_xcd` / 旧 `m_total<=2048` gate | ❌实测DEAD | overfitting 噪声 / 误判高-G MoE 走 masked | pitfalls/06 |
| grouped/MoE 核用 `num_xcd>1` 连续块 remap(照抄 dense) | ❌实测DEAD·skew **⚠ 但有前提,见下一行** | 连续 tile 整段绑一个 XCD ⇒ hot expert 压到单 XCD;wgrad band-cyclic+xcd=8 是 skew 崩溃主因(修=group-major+xcd=1,+26%/min 0.474→1.005)、NT xcd=8 同源 down-heavy −3% | methodology/08 · methodology/05 |
| ★✅ **`B[g]` 装不进一个 XCD 的 4 MB L2 slice 时,`num_xcd` 反过来必须 = 8**(band 窄到共用同一 B column-block 的 WG 同驻) | ✅gpt-oss padN dgrad NN **balanced +2.9% / heavy +2.5% / 几何 skew ±0**(逐位相同) | 上一行的判负前提是「B[g] 能常驻 ⇒ tile 顺序只影响均衡 ⇒ 越均衡越好」。B[g]=8.67 MB > 4 MB 时 B 必然是**流**,唯一能做的就是让共用 B 列块的 WG 同驻同 XCD;此时 **heavy 也是正的**,而这张卡给 grouped 核推荐的 group-major+`xcd=1` 实测 **−5.7%**。band 宽单峰:`gm` 3/6 = −1.3/−1.7%、`xcd=16` = −6.3%。**判别器用 `MemUnitStalled`(12.4→10.7%),别用 L2 hit-rate**(`TCC_HIT`/`TCC_REQ` 一动不动) | methodology/05 |
| 某候选在部署形状上明显赢,但生产竞速**每次都不选它** | ⚠️先查竞速的**打分 M** | 竞速在 canonical M(如 1024/8192)取几何平均,而算子**部署在 4096**;该臂在两个打分点 −0.94%/+0.54%(几何 +0.2%)过不了 1.5% 采纳门 ⇒ 竞速跑一万次也选不到。落地形态 = **在能唯一识别部署形状的谓词上 reorder 候选表(静态 lead)**,不是加候选(那只占槽位);并且必须用探针复验「真入口的孤立时间贴在预期候选上」。⚠️谓词识别的是**形状族**,族里还有别的 M ⇒ lead 要在**它覆盖的每个 M** 上验非负,不只部署点(同 campaign 两笔:wgrad lead 四点全非负可放心开门;NN lead 在 M=1024 是 −0.94%) | pitfalls/06 |
| ★✅ **一个 WG 走 WGT 个 tile(取自 tile 流两端)的门控 = 配对后的网格宽度,不是每组跨几代** | ✅mxfp4 grouped wgrad gate_up **canon +1.8~2.3pp / moderate +0.7~1.0pp**(gm +0.2) | 原门 `TILES_PER_GROUP ≤ _N_CU` 出自"跨多代的组本来就摊开了、配对只会拉长串行链"这个**推测**,把 276 tile/组的 gate_up 挡在外面;换成 `TOTAL/WGT ≥ _N_CU`(配对后仍铺满一代)后实测三个偏斜全正。真正兑现的是**每 WG 固定成本(launch+decode+group 表加载+流水填充)由 WGT 个 tile 分摊**,与该组跨几代无关。⚠ `WGT=3` 让 heavy 格塌到 0.872、`WGT=4` 更差(串行链吃掉分摊),**2 是甜点**;且 WGT>1 会放大 heavy 格的 run-to-run 方差(≈3×) | pitfalls/05 · campaign 20260811 R13 |
| ⚠ **被判负的旋钮在"周围态变了"之后要重新定价**(别把历史否决当永久) | ✅方法 | 同一个 `wg_tiles=2` 门控:R5/R6 在 gate_up 上实测 canon +1.4% / heavy **−0.7~1.2%** 判负,R13 加了 half_m 半价尾块与整线宽 `nt` 折叠 store 之后再测,同一改动变成三格全正。⇒ 复价的触发条件 = **该旋钮的收益或代价项被后续改动动过**(这里是尾块代价与 store 暴露量),不是"隔了几轮" | pitfalls/02 · campaign 20260811 R13 |
| ★✅ **grouped 核的 XCD 分区改 band-cyclic**(XCD x 拿第 x, x+8, … 个 `span` 个 M-block 的 run) | ✅mxfp4 grouped **+1.6% gm / +4.8% min** | 「连续块」与「逐 tile 轮转」是同一条轴(轮转粒度)的两个极端:连续 ⇒ 小-expert 尾巴全压一个静态 XCD;逐 tile ⇒ band 内 A/B 复用被打散。甜点 = **run 与组边界对齐**(span = 均匀分布下每组 M-block 数),skew 罚金 7%→0~1.8%。⚠**同核的 wgrad 相反**:它的 skew 在 contraction 长度上,只有 `xcd=1` 均衡 | methodology/05 |
| ★✅ **grouped 核 prologue 的 O(G) 串行 group-scan 换 lane-resident wave scan** | ✅三度实证:tw +7.7~9.6% / mxfp8 +1.3~5.4% / **mxfp4 +9.4%** | 2153→761 条 prologue、SGPR 97→58;新写 grouped 核别再从零写 O(G) | pitfalls/05 |
| NT 与 wgrad **共用**一份 `_*_DEFAULT_CFG` / 靠 runtime race 选 | ❌DEAD | race 在合成 balanced 点打分对真实 skew 盲;两路共用会让 wgrad 崩到 0.38–0.55×;拆开写死 gm +1.7% | pitfalls/06 |
| `set_*_backend(BackendType.FLYDSL)` | ❌DEAD | FLYDSL 未注册进 BackendType | pitfalls/06 |
| ★把 grouped 余数 tile **配对成等成本 tile**(短-M 体一个 WG 连做两个相邻 N-block) | ❌实测DEAD **−20%** | MFMA 精确 −3.89% 且逐字节 bit-exact,但 L2 命中 68.0%→**44.5%**、DRAM +70%(请求总量反而 −1.7%)。`a_halves=1` 的体是 0.5× MFMA / 0.75× 字节 / **1.0× K 扫描**,两个串起来 = 1× MFMA 但 ~1.5× **存活时间**;每 XCD 只 32 CU 共驻,WG 活过一"代"就横跨两个 class run,活跃 slab 12→24。**共驻窗口要齐一的是存活时间,不是 MFMA 条数** | pitfalls/03 |
| 用「省下的 MFMA 条数」给余数体(短 M/短 N 体)定价 | ❌方法错·打两折 | E=32 down 短-N 体省 6.25% MFMA 只兑现 **+0.6~1.2%**:同时把 L2 命中打掉 6.4pp、MFMA-busy 打掉 5.4pp。经验折扣 ≈ **只有 20% 落到 wall 上**,先打完折再决定做不做 | pitfalls/03 |
| ★★**要追一个具体 kernel 当标尺**(如 mxfp8 grouped 追 per-tensor wgrad) | ✅**第一轮就去 dump 标尺的 ISA + dispatch 元数据** | mx8tw 那场十五轮只测自己、第十六轮才拆标尺,一拆就换了整个打击面:标尺是 **4-wave/1 wave-per-SIMD/256 AGPR**,而我们一直在用 8-wave 几何打它;且我们**稳态本来就快 11–14%**,缺口全在 tile 太短(23 vs 134 K-iter) | methodology/12 §标尺①② |
| 对标两个 kernel 时用 **MfmaUtil / 占用率** 当主判据 | ⚠️口径陷阱 | 那是 wall 平均、含每 tile prologue/epilogue。剥掉 F 之后 mx 稳态 MFMA busy **87%** vs 标尺 81%,而 wall 口径只有 60% ⇒ 会把「tile 太短」误读成「稳态有 40% 空闲」。改用**同步指令/MFMA** 与 **barrier/K-iter**(逐 opcode 比值不依赖展开系数) | methodology/12 §标尺③ |
| grouped 的 **E8M0 preshuffle** 想加宽访问 / 加大 BLK 提速 | ❌实测DEAD | 它是 **dispatch/latency-bound**(2.6–2.9 TB/s = HBM 的 33–36%),**只吃「砍 WG 数」**:KT 16→64 与 slab 配对(A 侧 WG 2052→1026)全正;BLK 256→512 **+24~28%**、读宽 dword→dwordx4 **净零**。预算是每 WG 份额 `LDS/WG_per_CU`=20480 B 不是 160 KB 总量 | methodology/12 §标尺⑥ |
| 删 grouped NT 主环的 barrier 摊固定成本 | ⚠️分位置 | 判据 = **这一段还有 g2s 在飞吗**:有(稳态)⇒ 承重相位装置,删 = **−8.7%**;没有(尾部两个 K-step)⇒ 纯税,删 12 个 = **min +1.06%**。⚠ 两种情况都过 SNR + 逐字节确定性门 | methodology/12 §标尺④ |
| 加大 grouped wgrad 的 super-block 跨度 `gp`(让一个 XCD 在同一 group 里多待几"代") | ❌实测DEAD·全扫皆负 | E=32 逐点:down gp=2(auto)为基准,gp=4 **−0.52%** / gp=8 **−9.23%** / gp=1 **−5.8%**;gate_up gp=1(auto)为基准,gp=2 **−1.03%** / 4 **−1.81%** / 8 **−1.46%**。私有 L2 的「跨代续用」被**跨 XCD 共享**反噬:gp 越小,同时读同一 group 的 XCD 越多,内存侧这份复用比私有 L2 的连续性值钱。`_wgrad_xcd_span` 的自动值已是最优,别再手调 | methodology/05 · pitfalls/03 |

## F. 正确性 / 数据构造(先读,否则静默错)
| 你想试的动作 | 判定 | 一句根因 | 详卡 |
|---|---|---|---|
| `torch.randn().to(FP8)` 直接造数据 | ❌DEAD | >240 saturate + mma 非线性,误差不可预测;用 quantize helper | pitfalls/04,11 |
| 拿 element-wise tolerance 当 gate | ❌DEAD | 故意松,通过与否不说明正确性;用 SNR/bit-exact | pitfalls/04 |
| 靠 SNR 判断 race 是否存在 | ❌DEAD | 正常 SNR 完全掩盖 1/30000 bit-flip;用 det/多采样 | pitfalls/02,04 |
| deterministic 测试当正确性门 | ❌DEAD | 只验 run-to-run 一致,consistently-wrong 也过;要对拍 + unbalanced | pitfalls/05 |
| `create_buffer_resource(max_size=True)` | ❌DEAD | OOB 读垃圾;用 `max_size=False,num_records_bytes` | pitfalls/04,11 |
| `Vec.to(Float8E4M3FN)` / `cvt_scalef32_pk8_fp8_bf16` | ❌DEAD | 后端不 lower / gfx950 Cannot select;用 `cvt_pk_fp8_f32` | pitfalls/04,11 |
| 用「丢掉一整路 g2s fill」的方式给边界块减发射 | ❌racy(det 0/10) | graded drain 的 `vmcnt(n)` 是按**原本发几条 load** 标定的,发射数变了等待点就错位 → partial-drain race;要省就两半一起窄,或该段改全 `vmcnt(0)` | pitfalls/04 |
| 给 `S2RLoaderTr` 的两个操作数用**不同的 width** | ❌静默算错(maxerr 24) | c0→c2 立即数写死 `round_stride`,真跳距是 `(width//16)*chunk_stride`,只在 `n_waves==width//16` 时相等;一套配置喂两侧时跳错是**同一个 K 置换**故恰好抵消,两侧不同宽就配错 K 行 | pitfalls/07 |

## G. 测量 / bench(别用假数下结论)
| 你想试的动作 | 判定 | 一句根因 | 详卡 |
|---|---|---|---|
| 用「删指令/跳读」测「去掉 X 的天花板」 | ❌方法错 | 跳过≠等效替换;真替换后带宽受限收益归零 | pitfalls/02,05 · methodology/03 |
| ★✅ **给 feed 定价时在 `drop` 之外再加一个 `alias` 臂**(g2s 照发但地址全指向 k-block 0) | ✅方法·把一笔账拆成两笔 | `drop` 只告诉你 feed 总额(NN 实测 155 µs = 18.6%);`alias` 保住指令数/寻址/LDS-write 只让它永远命中 L2 ⇒ 分出 **84 µs 搬运 / 71 µs 纯发射**。预取与局部性类杠杆**只能碰 84 µs 那笔**,少了 alias 臂会把 2 倍空间算给预取。只动主体块(prologue/K-tail 留着 ⇒ 核不挂只算错) | methodology/03 |
| 用 heavy-skew 的**链级比值单次读数**判死一个 ±2% 的核内 knob | ❌方法错 | 链级 heavy 比值自身跨轮有 ~3% 行程,分辨不了 2%;本 campaign 据此把一个 **heavy +2.49%**(孤立核 5 回合 palindrome + 逐位相同)的臂误判成 −5.68% 拒掉了一轮 | pitfalls/06 |
| 拿并行 bench 的绝对 TFLOPS 下结论 | ❌DEAD | 被压低~10%,只有相对值可信;跨并行度对比是假象 | pitfalls/02 |
| `triton.testing.do_bench` / `cuda-event` 隔离小 kernel | ❌DEAD | 测出超 peak 假值;用 `rocprof --kernel-trace` | pitfalls/02 |
| 信裸 mean 差(DVFS 漂移内) | ⚠️ | 用配对窗口 A/B + 逐组数字;清 flydsl+comgr cache | pitfalls/02 · methodology/01 |
| ★用「几个读数取 min」去否决**配对 win-count** | ❌方法错 | 每个读数本身已是 min-of-40 ⇒ 这批样本的散布是**进程间**位移(右偏长尾),从 6 个右偏样本里再取一次 min 方差极大。实测同一批 12 对:配对均值 **−0.83%、11/12 同号、两 session 各自同号**,而两臂的 min 只差 **0.01%**。**min 治进程内高读,配对治进程间位移**,别互相否决 | pitfalls/02 |
| ★某个小 cell「摆动 10%+」时直接加重复次数 | ⚠️**先画分布**:可能是**双模**不是长尾 | 同一份二进制、24 个独立进程重测 0.57 ms 的 cell:低模 **0.567~0.574** / 高模 **0.634~0.642**、**中间无读数**,簇距 +11.4%。双模 ⇒ min-of-N(N≥12)是对的估计量(低模样本充足),mean/median 会按落入高模的比例线性偏移;而 3 个读数分不出双模与长尾。⚠ 不计分的 guard 项 `min(base/cur,1)` 落进高模会被压到 ~0.89,**单独给总分打掉约 1.5%** | pitfalls/02 |
| 直接拿 campaign 自报的 best 当结论 | ⚠️ | `cand_best` 其实是三读**中位数**(名字和 help 都在骗你);单读被邻居打崩就能把候选打成"大幅回归";且 bench 若不 mirror 部署配置就是在优化不出货的配置 | methodology/16 §1,§3,§6.10,§6.28,§7 |
| ★收官只跑**计分的那几格**就 ship | ❌DEAD | 计分口径=campaign 的全部视野,**没被计分的兄弟路径可被改断而全程零告警**:dense fp8 NN 场只跑 NN+TN,NT(fwd)被 `gn` use-before-assign 打断,错误被 `except Exception: continue` 吞成"no working cfg",10 轮全绿。收官必须**每条 layout 各调一次真入口** + AST 查 use-before-assign | methodology/16 §7.5 · pitfalls/06 |
| 硬规矩写成 `git diff <base> HEAD -- <路径> \| wc -l == 0` 却没 `test -f` 那个路径 | ❌DEAD | 路径不存在时**恒为 0**,检查空转一整场(真文件在 `flydsl/utils/`,我 grep 的是 `flydsl/gemm/`) | methodology/16 §7.6 |
| 看 `total_cost` 累计超过用户说的美元数就停机 | ❌**别做**(白杀过两场) | 用户给的预算数字**默认是「每轮」**;`--*-budget` 本来就是 per-turn 上限,`total_cost` 只记账。吃不准就问一句;真要限总额用 `--rounds` 卡轮数 | methodology/16 §5b |
| campaign 换机器后沿用旧 base/best | ❌DEAD | 分数是绝对 TF/s,跨机差 3.5%:慢机全被当回归 revert,快机垃圾改动当 win | methodology/16 §6.5 |
| campaign 收官后直接 merge squash commit | ❌DEAD | squash 会挑错 base(实测删掉整个 helper −2691 行 + 夹带无关改动);且 worktree 一删 commit 变游离 | methodology/16 §6.6,6.7 |
| 两场 campaign 的 `--gpu-pool` 有交集 | ❌DEAD | 自动切卡后两个 bench 抢同一张卡,分数腰斩到约一半 | methodology/16 §6.8 |
| campaign 半程分数不动就以为「方向走不通」 | ❌先查故障 | agent 侧 API 故障被 harness 记成 no-change、`--max-round-crashes` 不触发,20 轮里烧掉 10 轮;判据=`grep -c "cursor-agent failed"` 增量 | methodology/16 §6.14 |
| 用短请求测 API 判断「故障已恢复」 | ❌方法错 | 短请求恒秒回 OK,故障只在 agent 长会话暴露(200-484 秒才断);据此放行又烧五轮 | methodology/16 §6.14 |
| resume 一个已 `campaign done` 的 campaign | ⚠三处状态要改 | phase=REVIEW / action=GO_REVIEW / rnum 已+1,任一没改都秒退;且必须加 `--no-squash`(kept_commits 已不在历史上,会挑错 base) | methodology/16 §6.15 |
| 把标尺 merge 进同一棵树后就不管了 | ⚠必须冻结 | agent 能改到标尺或**两边共用的 helper**,基线中途漂移使前后轮次不可比;goal 里列黑名单+验收查标尺绝对值 | methodology/16 §3.6 |
| campaign 用 `--*-effort max` 求质量 | ❌反效果 | 实测 max 三轮零产出(含撞满 2h 超时报废),xhigh 两轮 25min/轮并刷新 best;Opus 5 两档都保 1M ctx | methodology/16 §6.4 |
| 按 CPU/GPU 占用 0% 判 campaign 卡死 | ❌方法错 | agent 大部分时间等 API;轮内唯一活信号是 kernel 文件 mtime(`deep_raw.jsonl` 只在收轮时落盘) | methodology/16 §4 |
| 改了 kernel 后只清 `/root/.flydsl/cache*` 就重测 | ❌测到旧二进制 | **`/root/.cache/comgr` 也存编译产物**;漏清导致 90 次冷跑零失败的假结论、据此误判"已修复";每轮必 `rm -rf /root/.flydsl /root/.cache/comgr` + 清 `__pycache__` | methodology/16 §6.29 |
| 低频偶发 bug 用"跑 N 次全过"判已修复 | ❌样本不够 | 25% 失败率下 8 次全过概率 10%;我据此误判两次。判据必须**双向**(关掉必好+单独打开必坏)且每组 ≥20 次,定位靠**逐旋钮二分**不靠猜机制 | methodology/16 §6.29 |
| campaign 正确性门只有 SNR | ⚠漏偶发竞态 | 127/47M 元素的偏差对 SNR 毫无影响,整场 ok=true 却带着 25% 的 bitwise 不确定性出货;门里必须含**逐位确定性** | methodology/16 §3.2 |
| campaign 停在 "benchmarking candidate" 不动、进程却活着 | ⚠GPU-hang,超时救不了 | `run_bench(timeout=)` 只 kill 直接子进程,孙子 ssh 持有管道 ⇒ `communicate()` 永不返回,`TimeoutExpired` 都抛不出,主进程死在 `wchan=do_poll`;解法=外挂看门狗只杀那个 bench(**不是停 campaign**) | methodology/16 §6.27b |
| GPU use=100% 就以为 kernel 在干活 | ❌看功耗 | 100% 占用 + **功耗只有满载的一半**(363W vs 600W+)= 少量 CU 空转的死循环;实测由"编译期 trip count ≠ 运行时圈数"引起;主线程 100% CPU 是 `synchronize()` 自旋,不是编译(`ps --ppid` 查有无 clang/comgr) | methodology/16 §6.27a |
| hang 的进程杀掉后 GPU 仍 100% | ⚠孤儿 queue 不自愈 | 容器内 `rocm-smi --gpureset` 报 `Not supported`,**必须 host 上 sudo**;reset 前用 `--showpids` 确认无 KFD 进程(**第 3 列才是 GPU 号**) | methodology/16 §6.27c |
| 候选报大幅回归就当代码写坏了 | ⚠先看三读跨度 | 正常轮跨度 0.15–1.05%(中位 0.72%),被邻居抢卡那轮 15.82% 并把分数拽掉 28%;harness 取中位数 ⇒ 三读坏一读就够打成"回归" | methodology/16 §6.28 |
| 小形状"内核慢"下结论前先量 host enqueue | ⚠必做 | wall 84 / host 85 / GPU 50 µs;判据=扫 S 时**工作量差 64× 而时间不变**;修法=缓存 `flyc.compile` artifact,host 93→7 µs | pitfalls/02 · pitfalls/13 |
| ★★ 同进程 A/B 用**模块级全局**切换两臂(flydsl) | ❌静默同二进制 | JIT cache key = 源码 + 依赖源码 + closure 标量,**全局不进 key** ⇒ 第二臂命中第一臂 .pkl。指纹 = 中位数差 <0.05% 且输出 sha 相同,而 `FLYDSL_DUMP_IR=1`(绕缓存)下 ISA 明显不同。修法 `FLYDSL_RUNTIME_ENABLE_CACHE=0` | pitfalls/02 |
| ★★ 只用 `torch.autograd.grad` 的 bench 上判定"返回 strided view 免了一次拷贝" | ❌看不见成本 | `autograd.grad` 把张量原样交回,**`.backward()` 的 AccumulateGrad 会 clone 非连续梯度**:同一代码两 API 差 **378 µs = 该步 10.2%**(kernel-trace 里多一个 376 µs elementwise)。凡涉及输出 layout 的杠杆必须两个口径都 A/B | pitfalls/02 |
| 拿两次读数估"噪声地板" | ❌方法错 | n=2 估不出分布宽度:两次差 0.13% 被当地板,第三次一测散布是 ~0.5%;单格可达 ±2.5% | pitfalls/02 |
| 用**孤立单核微基准**选最终档位 | ❌方法错 | 只配"筛掉明显差的",不配定档:dQ 规约核 `1024×8` 孤立第一(557.2 µs)、in-situ **输 0.4% 整体分**,第二名 `512×2` 才是真赢家(孤立跑独占整机、工作集冷热与承接主核 ramp-down 都不同) | methodology/03 |
| 拿"自己当前最快的核"当机器带宽上限 | ❌方法错 | 与"拿 torch 当上限"同一个错:自家核 6.03 TB/s 被写成"封顶"→ 把两个 reduction 留给 torch(4.48 TB/s)又跑 4 轮;真上限 ~6.3,手写核实测 6.14~6.26 | pitfalls/03 §torch-copy_ |
| 把 cycle 换成时间时用**标称**时钟 | ❌DEAD | MI355X 标称 2.4 GHz 是 boost,满载实测 **2.09**;用错让 in-loop util 从 62.6% 误报成 52.8%,整轮方向建在错坐标系上 | connection/common/05 |
| 攻 GEMM 前不先分"稳态 vs 固定开销" | ⚠必做 | 6+ 个 K 点拟合 `t_tile=F+n_phase*P`:实测 P 已等于 dense(差 1.6%),全部差距在 F ⇒ 主循环根本不该碰 | methodology/03 |
| 用**代换探针**给一条指令定单价 | ❌方法错 | 差额会被重叠吃掉:exp 代换读出"全速率",PMC 直接分解是 half-rate(8.14 拍),差 1.6× | connection/common/05 · pitfalls/13 |
| 在 e2e 训练里**加 python print** 验证某 kernel 有没有真跑 | ❌方法错 | 派发路径可能不是你以为的那条(MoE 走 ragged,不经 PrimusTurboGroupedLinear/公共 grouped_gemm_fp8)→ print 在死路全为 0,误判"没触发"。**开 torch profiler 看 trace kernel 名才是 ground truth**(pad-quant kernel = K-pad 铁证) | methodology/17 · [[project_kpad_e2e_trace_validated]] |

| ★★★ **campaign 连续多轮「no new best」时,先怀疑尺子,别怀疑想法** | ✅三条命令诊断 | 一场 17 轮里十三轮空转,停场后复测发现 r9–r12 **每轮都有 0.2–1.4% 的真实增益**。三条税叠加把判定线推到 ≥1.5%:①banked best 在**它自己的 commit** 上复现不出(−0.5% 幽灵)②硬 cap guard 是单边噪声税(**逐位相同**的代码跨 8 次跑离散 3.2%,`min(r,1.0)` 只能减)③3-rep max 比 high-water。诊断:回到 banked commit 重测 / 同码连跑看 guard 离散 / `--calibrate` 量 spread | methodology/16 §尺子会撒谎 |
| ★★ **`--repeat` 和「要不要 BUNDLE」应由实测 spread 决定,不是照抄** | ⚠️同 kernel 的 fwd/bwd 差 2–4 倍 | D=64 **bwd** spread ~1.0% ⇒ 最小可测增益 ~1.5% ⇒ 必须 BUNDLE;同 kernel **fwd** spread **0.3–0.8%** ⇒ 0.3% 就能独立计分。**把 BUNDLE 搬到干净尺子上是浪费**(糊成一坨、无法二分)。起场前先 `--calibrate` | methodology/16 §先量尺子 |
| ★★ **guard 的 cap 要留容差,`min(ratio,1.0)` 是单边税** | ⚠️修正旧条目 | 容差要 ≥ 该 guard 腿实测离散的上界。改成 `1.0 if r>=0.98 else r/0.98` 之后,guard 腿**每轮都是 1.0**,而旧口径同样读数会有若干 0.99x 白扣分;它仍抓得住真回退。17 轮里硬 cap **一次真回退都没抓到** | methodology/16 §兄弟盲区 |
| ★★ **共用文件 ≠ 共享收益:给每个不计分的变体测 before/after** | ❌踩证 40 轮零转化 | D=64 与 D=128 bwd 是同一个 `.py`。D=128 打了 40 轮 +19.30%,同尺子测 D=64:**997.2 → 990.0 TF/s**(零转化还微负)。根因是四道 `D == 128` 门 + 计分脚本抬头写着 "LLAMA ONLY … out of the score and out of the guards"。开门那一轮单轮 **+6.36%** | methodology/16 §形态 A |
| ★★ **上下游共用中间量时,加「端到端门」** | ✅代价≈0 | fwd/bwd 共用 **LSE**。改 LSE 布局去加速 fwd ⇒ fwd 自己的 SNR 满分、bwd 静默崩。做法:用**被优化那一侧的真实输出**驱动下游再查 SNR(`o,lse=forward(); dq,dk,dv=backward(...,o,lse)`)。部署实测 fwd 51.7 dB / e2e 49.3-49.0 dB,门设 45 | methodology/16 §形态 B |
| ★ **别用 score 反推 TF/s**(score 含 guard 因子) | ⚠️系统性偏高 ~1.5% | 反推 1057 vs 真值 1033–1043。**在 bench 里就把 `tflops` 算好写进 JSON**,监控端直接读 | methodology/16 §运维坑 |
| ★★ **列 campaign 进程树用 `pgrep -P`,不要 `ps -eo ... --ppid`** | ⚠️`-e` 覆盖过滤 | 容器 PID1 是 `sleep infinity` 不 wait ⇒ 几万个 `<defunct>` 刷爆屏幕,真正要杀的 PID 反而看不见。另:兜孤儿必须兜到远端容器里的 `rocprofv3` —— 停机时兜出过一个挂 **15.7 小时**、一直占卡的 `--kernel-trace` | methodology/16 §运维坑 |

## H. 环境 / 同步 / 构建(违反=破坏别人环境)
| 你想试的动作 | 判定 | 一句根因 | 详卡 |
|---|---|---|---|
| rsync 加 `--delete` | ❌DEAD | 删远端不能删的东西(.so 等) | pitfalls/08 · [[feedback_ssh_key]] |
| 碰其他项目/租户的容器或盘 | 🚨红线 | 共享集群上别人的容器/盘严禁 rsync/docker exec(具体环境边界见 SKILL.md · connection/) | SKILL.md · connection/ |
| 清别人容器/停别人进程/删别人镜像 | 🚨DEAD | 即使 GPU 空也禁;srun/salloc 抢 mi355x 也不行(走 docker) | pitfalls/08 |
| `docker exec '... > /tmp/x'` 外层重定向 | ❌DEAD | 重定向落 host 不落容器 | pitfalls/08 |

## I. flydsl 前端 / tracer / 移植
| 你想试的动作 | 判定 | 一句根因 | 详卡 |
|---|---|---|---|
| 合并 NN/NT 编译函数写 `if trans_b` | ❌DEAD | NameError / 变量只在某分支绑定 | pitfalls/07 |
| `const_expr(lane==0)` / `const_expr(thread_id)` | ❌DEAD | 运行时值不能进 const_expr | pitfalls/07 |
| `scf.for` 的 iter_args carry ping-pong buffer | ❌不支持 | 多值 carry 不支持;分支放外层 python | pitfalls/07,12 · [[reference_flydsl_loop_pitfalls]] |
| 2 字节 buffer_store / bf16 vec1 load | ❌不 lower/NaN | store 须 ≥4B;gather 用 i16 vec1 装配 | pitfalls/12 |
| kernel 与 torch 同模块 | ❌RecursionError | JIT 依赖收集爆栈;kernel 放独立纯模块 | pitfalls/05,07 |
| CDNA 旋钮导 RDNA / TF32 导 gfx950 / transpose-load 导 gfx942 / 硬编码 `gfx*` | ❌DEAD | 编码/调度跨代不通;判据用 `is_rdna_arch()` | pitfalls/11 |

## J. 算子融合(把 activation/SwiGLU 折进 grouped GEMM 的 epilogue)—— 详见 methodology/18
| 你想试的动作 | 判定 | 一句根因 | 详卡 |
|---|---|---|---|
| 拿融合核的 **per-op TF/s** 去比零 epilogue 的 **plain GEMM**、追 parity | ❌**伪目标** | 融合核把 activation 的活也扛了,是"多干活的核"比"没干那活的核";正确口径永远是 **融合-total(fc1+act+fc2)vs 非融合-total**。融合 glu −14%/dglu −28% 的 gap 是结构性的(dglu 强制 2× HBM、glu 第三条 act 流串行写、occ 已顶),**total 折进来是净赚** | methodology/18 §2,§9 |
| 融合 **gated activation** 时把 per-half 维 pad 到 Ip,靠**纯零传播**(只按权重形状推 I=Ip) | ❌**grad_w1 会死** | fwd/dgrad 对,但 w1 梯度真实内容在 `[0,I)∪[Ip,Ip+I)` 夹 pad 行,`m_real` 截前 2I 行截出"整段 padded gate + 半段 up",对不上 tight。**正解=glu/dglu 核内按 pitch(Ip)算、按 real(I)存**(`glu_i` 劈 pitch/real,up-offset 用 pitch、列掩码用 real、pad 段写零) | methodology/18 §4 |
| 用 campaign 的**两核平均**分数评"只动了单核"的改动 | ⚠️**会掩盖真增益** | 单核 `+0.55%`(5 读数无重叠)被另一核 run 的噪声淹掉,平均门 60 轮不动;**只动单核就看那核自己的 ratio/ms,不看聚合 spd_fuse** | methodology/18 §3 · methodology/16 |
| 报融合 MLP 绝对数时传 **bf16 权重** | ❌**部署假象** | bf16 权重每次调用**重量化**(~15% fwd),部署一次性预量化缓存不吃这块;**部署代表数要传 `QuantizedTensor`**(x 仍 bf16=动态 fp8 固有)。bwd 复用 fwd 存的 fp8 权重、从不重量化 ⇒ 预量化只动 fwd | methodology/18 §7 |
| in-GEMM fold(dQ partial / deep-K split partial 折进主 pass)不看**闸门归属** | ⚠️**生产者/消费者自旋** | 恰好一个组/tile 越切分闸门即另一半空转,**结果正确但慢 10–2000× 且抖动**,SNR/det 门抓不到;落地前确认所有组同闸门侧(`k≥2 组越闸`才干净),上线盯尾延迟 | methodology/18 §3.5 |
| 融合 epilogue 的累加器放 AGPR / 融合核追 plain GEMM 的写侧优化 | ✅**可白赚** | 累加器留 VGPR(mma mode 3)省 `v_accvgpr_read`(NT 喂料 S2RLoader、reg 预算不变);plain 的成对列 store 移植进 glu-fwd **+3.7%**(比普通 NT 更肥=三流写压),col_safe 靠 I 偶+pair 落偶列掩码 | methodology/18 §5,§8 · methodology/07 |

---
维护:新判负的杠杆先进对应详卡(带根因+实测数字),再在本表加一行入口。本表只放**高频被重试**的,长尾留详卡。
来源:全库 pitfalls/01-12 的 ❌ 汇总 + memory dsv4/mxfp4/fused-mlp 系列。
