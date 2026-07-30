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
| `amdgpu-mfma-vgpr-form=false` / inline-asm `=a` 逼 AGPR 抬 occ | ❌实测DEAD | 生产 4-wave ISA 逐字节相同;AGPR 与 VGPR **共用同一 512 池**,搬家不抬 occ | pitfalls/01,03 · methodology/04 |
| 8-wave 原生 occ=2 藏 store / 长 K 追平 4-wave | ❌实测DEAD | 28672 −14%/6144³ −20%;8-wave per-warp tile 减半→B 复用减半→mfma 翻倍 > occ 收益 | pitfalls/03,05,06 |
| BK128 换 occ=2(mxfp4 K28672) | ❌实测DEAD | occ=2 唯一途径 BK128→g2s 频率翻倍 VMEM-stall 2.7→30%,总 5405→4686 | pitfalls/05 |
| 缩 tile / 矩形 tile 为抬 occ | ❌实测DEAD | feed-bound worst shape:占用率非杠杆,根因 LDS-feed 带宽 | pitfalls/01,05 |
| attention 强制 occ(WPEATTR/o_acc 进 AGPR/bf16 o_acc) | ❌实测DEAD | 全 latency-bound,occ-1/2 都填不满;强制必 spill/掉速 | pitfalls/12 |
| bwd(occ-2)dual-wave / 8-wave / warp-spec / bare-asm cross-head 显式并行 GEMM↔softmax | ❌实测DEAD | occ-2 baseline 已靠两独立 WG 共驻**免费享无屏障跨-WG overlap**;8-wave 单-WG 换成带屏障税组内 overlap,天花板<baseline(dkdv 1029<1116)。净胜改走冷-load 寄存器预取(+1.81%) | methodology/15 · pitfalls/13 |
| **先做**:判是不是 occ-bound(而非 register/LDS/feed/latency-bound) | — | occ=`512//(arch+agpr)`;`Accum_VGPR_Count=0`+ISA 才权威;多数 kernel 是 feed/latency-bound | methodology/04 · methodology/03 |
| ✅ **分块读 tile 压活跃度抬 occ 前,先看那条读是不是 inline asm** | ✅开口 | asm 读 waitcnt 不被 `SIInsertWaitcnts` 记分 ⇒ 每切一块就要手写 lgkmcnt + `sched_barrier(0)` 钉住,调度损失正好抵消寄存器收益。换成 rocdl op(如 `ds_read_tr16_b64`)后同一分块从 −0.35% 变 **+0.50%**,occ 2.985→**3.978** | pitfalls/13 §fwd |
| ⚠ 抬 occ 成功后**必须报 sclk**:功耗墙下时钟会回吐 | ⚠有条件 | 96.6% TBP 时 occ 2.985→3.978 使 sclk 1710→**1654 MHz(−3.3%)**,wall 只 +0.50%,真实省周期 **+3.9%** | methodology/03 §Power/clock-bound |

## B. LDS / 数据通路 / store
| 你想试的动作 | 判定 | 一句根因 | 详卡 |
|---|---|---|---|
| A/B operand 直载 VGPR 跳过 LDS(a_direct/b_vgpr) | ❌实测DEAD | mxfp4-8wave 慢 ~22%(bit-exact);LDS 复用是必须 | pitfalls/03 |
| 3-stage LDS 双缓冲 / 更深 prefetch distance | ❌实测DEAD | 每 stage×3>160KB 净负;L2 是 capacity-thrash,prefetch 治不了 | pitfalls/03,05 |
| store 与 compute 重叠(STORE_ILV/persistent/CShuffle 宽存) | ❌实测DEAD | gfx950 统一 vmcnt,在途 store 占 vmcnt→串行化;宽存 LDS round-trip 吃掉收益 | pitfalls/05 |
| DMA/`buffer_load_lds` 免 reg→store(attention & GEMM) | ❌实测DEAD | register-prefetch 对 scattered-gather 最优,DMA s_waitcnt 暴露 HBM;attention 版被 flydsl 编译器 bug 挡 | pitfalls/03,04,12 · [[project_dsv4_fwd_dma_experiment]] |
| triple-buffer / prestore 隐藏 exposed store | ❌实测DEAD | store 是 work-bound(工作量不减)/ 占 occ;重排无用 | pitfalls/12 |
| 免一个操作数转置(`ds_read_b64` 换 `_tr`) | ❌实测DEAD | ISA 等价,转置读不比 plain 贵,0 收益 | pitfalls/05 |
| padding 消 bank 冲突(实测本无冲突时) | ❌实测DEAD | BANK_CONFLICT=0 时 pad 白吃 LDS;不对称 swizzle 会静默读错行 | pitfalls/03 |
| register double-buffer/read-once/RING/monohoist(mxfp4) | ❌实测DEAD | 2 operand set 超寄存器预算→RAGreedy crash/spill | pitfalls/01 |

## C. quant / scale(mxfp8/mxfp4)
| 你想试的动作 | 判定 | 一句根因 | 详卡 |
|---|---|---|---|
| 把残余 gap 归因「scaled-MFMA 指令税」 | ❌结论错 | 指令税≈0;真凶=scale VMEM load 未隐藏 | pitfalls/05 |
| `scale_pack`/opsel byte-pack(生产 per-K) | ❌实测DEAD | 现有流水已预取藏 scale;前提是「裸无预取」,生产不适用 | pitfalls/05 |
| scale 也暂存 LDS / scale via LDS 双缓冲 | ❌实测DEAD | 慢 6×/0.42-0.61×;只暂存 fp8 数据不暂存 scale | pitfalls/05 |
| 大 BM 但不跨-lane 合并的 quant 写 | ❌实测DEAD | 0.58-0.98×;死的是「不合并」,大 BM+LDS 合并才是制胜招 | pitfalls/05 |
| whole-loop(WL)移植进生产 | ❌用户裁定 | 2500 行 bare-asm 不可维护;occ=1 结构上限,per-K 才是生产路径 | pitfalls/05 |
| 优化 quant kernel 撬 e2e(grouped) | ❌实测DEAD | quant 绝对占 35-46% 但 e2e 中性;缺口在 gemm(occ=1 上限) | pitfalls/05 |
| e8m0 scale 广播前 cast Uint8 | ❌静默错 | 高位损坏 match~9% 垃圾;位运算全程 Int32 | pitfalls/05 |
| atomic 融合 reduce(dense split-K) | ❌实测DEAD | 同地址 HBM atomic 争用串行;split+reduce 是答案 | pitfalls/05 |

## D. attention(fwd + bwd)—— 优化顺序 playbook 见 methodology/15;实测 win/dead 见 pitfalls/12(dsv4)+13(hd64 dense)
> 优化前先读 **methodology/15-attention-optimize**(定 bound→fwd:消/藏 store·_FMAX0·dual-wave / bwd:减 MFMA·融串行核→藏 MFMA 延迟→exp2·occ·确定性)。

### D. attention(dsv4 sparse-MLA)—— 详见 pitfalls/12
| 你想试的动作 | 判定 | 一句根因 | 详卡 |
|---|---|---|---|
| s_setprio(fwd / bwd dQ) | ❌实测DEAD | latency-bound,优先 MFMA 饿死 VALU/read | pitfalls/12,09 |
| cross-tile 软件流水(XPIPE) | ❌实测DEAD | cr4 LDS-port 饱和/few-tile prologue > 重叠 | pitfalls/12 |
| K=32-PV(2-tile batch) | ❌实测DEAD | 批打断 QK→softmax→PV 交织(occ-1 藏延迟唯一机制) | pitfalls/12 |
| dual-layout-KV 消 QK bank 冲突 | ❌实测DEAD | 多写一份 KV 压在 occ-1 关键路径 −7~17%;冲突本 off-critical | pitfalls/12 |
| register-transpose PV(fwd) | ❌实测DEAD | 每 tile 重做转置不摊薄 −77%(bwd rtr 赢是摊在 rank-tile) | pitfalls/12 |
| query-blocking(cr128 减 store) | ❌实测DEAD | 2×o_acc>256 arch-VGPR→强制 occ-1,占用损失>store 省 | pitfalls/12 |
| ✅ pro cr0/cr128 interm(慢 triton ~20%) | ✅开口 | 研究 triton 小 topk tiling/少 launch | pitfalls/12 |
| ✅ banded-SWA dKV 融合 / hybrid scatter | ✅开口 | 减 interm+gather 的 2× tensor HBM | pitfalls/12 |

## D2. attention(Meta/gpt_oss hd64 DENSE flash bwd,确定性)—— 详见 pitfalls/13
| 你想试的动作 | 判定 | 一句根因 | 详卡 |
|---|---|---|---|
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
| swizzle/pad/prefetch/bank(消 tr16 冲突) | ❌实测DEAD | LDS 4-8× port 富余=latency-bound,非 port-bound | pitfalls/13,05 |
| P5 shared-GEMM1 融合(dq+dkdv) | ❌实测DEAD | q-outer=det 陷阱→KV-outer BLOCK_KV=64 但融合核 1.95-2.55× 慢 | pitfalls/13 |
| fp8 GEMM1 operands(FA-3 plain) | ❌实测DEAD | hd64 收缩维短,SNR 28.2<34 破门 | pitfalls/13 |
| dkdv 拆 dV/dK-only 核(冲 occ3) | ❌实测DEAD | 双份 Q/dO 读+拆分开销>占用率收益(-36%) | pitfalls/13 |
| LDS 双缓冲 | ⚠regime | 仅长 Skv 边际 +1.25%,短 shape -3.6% | pitfalls/13 |
| ~~长 Sq(8192/16384)full-causal 达标~~ | ★**已达标(2026-07-27)** | 旧记"确定性杠杆全 measure-closed、只剩放宽 det 或 research 级 exp-overlap"**已被推翻**——真正的开口不在 kernel 体内,在 **grid/派发映射层**(XCD+LPT+padding 对齐 = +7%)与**锚点粒度**(+3.6%)。**没有放宽任何确定性** | pitfalls/13 |
| ~~收官:perf B4 hw 16/20·fast 18/20~~ | ★**19/20 + 4-square 4/4**(turbo 侧) | 4-square hw 565/765/832/**877**(1.44-2.08×)、fast 4/4;20-config full **9/10**(既有 MI350 CK-det 是 0/10)、SWA 10/10(2.12-3.41×)。唯一未过 `newshape full 2048`=1.37×(98%)。**成果在 `sync/mxfp4/Primus-Turbo`,meta-attn 未移植** | pitfalls/13 |
| ❌ **重排派发序换 cache 复用(hd64 dense fwd)** | ❌实测DEAD·两级 | L2 级(XCD 内按 kv-head 分组)=−0.19%,L1 级(把 8 个 GQA sharer 摆到同一 CU,离线证双射、共驻同流 512/768)=**−0.57%**。两次 `TCP_TCC_READ_REQ`/`TCP_PENDING_STALL`/`TCC_HIT` **逐位不动** ⇒ 请求流总量不变、只动时序。根因:**单 WG 自己的 K/V 双缓冲 32 KB = 整个 vL1D**,兄弟 WG 来读时行已被自己挤掉。⇒ 减请求只剩加大 block_m | pitfalls/13 §fwd |
| ★✅ **GQA sharer 合并(一个 CTA 跑 2 个 q-head 的同 index q-tile)** | ✅hd64 fwd **+2.04%** | `NUM_WAVES` 4→8 而 `BLOCK_M` 不变(低位 wave 选行组、高位选 sharer),两个 sharer 共用一份 K/V LDS tile 与 barrier ⇒ `TCP_TCC_READ_REQ` **−49.5%**(256→128 sector/WG-迭代)。**收益全在时钟**:sclk +8.7% @ 同功耗(96% TBP),真周期反而 +6.6%。LDS 只装 K/V 与 BLOCK_M 无关 ⇒ vgpr/占用率不掉档。⚠ 改 CTA 波数后所有以**指令数**计的 `vmcnt` 常量都要重算(`dma_wave_reps` 会变) | pitfalls/13 §fwd |
| ⚠ **"省访存还值不值钱"看 `TCP_TCC_READ_REQ`,别看 DRAM 带宽/L2 命中** | ✅方法 | 功耗墙(≥95% TBP)上,能耗在 **TCP→TCC 的请求/sector 路径**:pinned-tile 探针只改命中层级(请求数逐位不动)→ 只买到 +0.94% 时钟;真把请求数减半 → **+8.7% 时钟**。两者结论相反,别用前者否掉后者 | pitfalls/13 §fwd |
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
| `set_*_backend(BackendType.FLYDSL)` | ❌DEAD | FLYDSL 未注册进 BackendType | pitfalls/06 |

## F. 正确性 / 数据构造(先读,否则静默错)
| 你想试的动作 | 判定 | 一句根因 | 详卡 |
|---|---|---|---|
| `torch.randn().to(FP8)` 直接造数据 | ❌DEAD | >240 saturate + mma 非线性,误差不可预测;用 quantize helper | pitfalls/04,11 |
| 拿 element-wise tolerance 当 gate | ❌DEAD | 故意松,通过与否不说明正确性;用 SNR/bit-exact | pitfalls/04 |
| 靠 SNR 判断 race 是否存在 | ❌DEAD | 正常 SNR 完全掩盖 1/30000 bit-flip;用 det/多采样 | pitfalls/02,04 |
| deterministic 测试当正确性门 | ❌DEAD | 只验 run-to-run 一致,consistently-wrong 也过;要对拍 + unbalanced | pitfalls/05 |
| `create_buffer_resource(max_size=True)` | ❌DEAD | OOB 读垃圾;用 `max_size=False,num_records_bytes` | pitfalls/04,11 |
| `Vec.to(Float8E4M3FN)` / `cvt_scalef32_pk8_fp8_bf16` | ❌DEAD | 后端不 lower / gfx950 Cannot select;用 `cvt_pk_fp8_f32` | pitfalls/04,11 |

## G. 测量 / bench(别用假数下结论)
| 你想试的动作 | 判定 | 一句根因 | 详卡 |
|---|---|---|---|
| 用「删指令/跳读」测「去掉 X 的天花板」 | ❌方法错 | 跳过≠等效替换;真替换后带宽受限收益归零 | pitfalls/02,05 · methodology/03 |
| 拿并行 bench 的绝对 TFLOPS 下结论 | ❌DEAD | 被压低~10%,只有相对值可信;跨并行度对比是假象 | pitfalls/02 |
| `triton.testing.do_bench` / `cuda-event` 隔离小 kernel | ❌DEAD | 测出超 peak 假值;用 `rocprof --kernel-trace` | pitfalls/02 |
| 信裸 mean 差(DVFS 漂移内) | ⚠️ | 用配对窗口 A/B + 逐组数字;清 flydsl+comgr cache | pitfalls/02 · methodology/01 |
| 直接拿 campaign 自报的 best 当结论 | ⚠️ | `--repeat` 取 MAX 系统偏高;且 bench 若不 mirror 部署配置就是在优化不出货的配置 | methodology/16 §3,§7 |
| campaign 换机器后沿用旧 base/best | ❌DEAD | 分数是绝对 TF/s,跨机差 3.5%:慢机全被当回归 revert,快机垃圾改动当 win | methodology/16 §6.5 |
| campaign 收官后直接 merge squash commit | ❌DEAD | squash 会挑错 base(实测删掉整个 helper −2691 行 + 夹带无关改动);且 worktree 一删 commit 变游离 | methodology/16 §6.6,6.7 |
| 两场 campaign 的 `--gpu-pool` 有交集 | ❌DEAD | 自动切卡后两个 bench 抢同一张卡,分数腰斩到约一半 | methodology/16 §6.8 |
| campaign 用 `--*-effort max` 求质量 | ❌反效果 | 实测 max 三轮零产出(含撞满 2h 超时报废),xhigh 两轮 25min/轮并刷新 best;Opus 5 两档都保 1M ctx | methodology/16 §6.4 |
| 按 CPU/GPU 占用 0% 判 campaign 卡死 | ❌方法错 | agent 大部分时间等 API;轮内唯一活信号是 kernel 文件 mtime(`deep_raw.jsonl` 只在收轮时落盘) | methodology/16 §4 |
| 小形状"内核慢"下结论前先量 host enqueue | ⚠必做 | wall 84 / host 85 / GPU 50 µs;判据=扫 S 时**工作量差 64× 而时间不变**;修法=缓存 `flyc.compile` artifact,host 93→7 µs | pitfalls/02 · pitfalls/13 |
| 用**代换探针**给一条指令定单价 | ❌方法错 | 差额会被重叠吃掉:exp 代换读出"全速率",PMC 直接分解是 half-rate(8.14 拍),差 1.6× | connection/common/05 · pitfalls/13 |

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

---
维护:新判负的杠杆先进对应详卡(带根因+实测数字),再在本表加一行入口。本表只放**高频被重试**的,长尾留详卡。
来源:全库 pitfalls/01-12 的 ❌ 汇总 + memory dsv4/mxfp4 系列。
