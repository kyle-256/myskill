# pitfalls/13 — Meta/gpt_oss hd64 DENSE flash attention bwd(确定性)实测经验 + 死胡同

> 类别: 踩过的坑 · 主题标签: meta, gpt_oss, hd64, D64, dense-flash, attention, bwd, dq, dkdv, odo, deterministic, CK-replacement, 16x16x32, schraudolph, poly-exp2, delta-fusion, drop-b-gemm, exp2-pipeline, shared-gemm1-fusion, fp8-gemm1, occ3-split, pack-128, latency-bound, MFMA-operand-bubble, gfx950, dead-ends

> 任务：`flash_attn_varlen_func_flydsl` —— FlyDSL 算子替代 CK 确定性 hd64 FMHA varlen bwd。
> Meta shape B4 Hq128 Hkv16 **D=64** Skv16384 Sq∈{2048,4096,8192,16384},bottom-right 全因果,bf16,THD。
> 三核:`dq`(Q-outer,identity-center rho/R)+ `dkdv`(KV-outer split-K Q_SPLIT=2)+ `odo`(delta)。16x16x32 MFMA。
> exp2 双模:Schraudolph fast(**35dB**,shipping)/ poly deg-3(**>50dB**)。确定性=构造性(单 WG 独占 tile+无 float atomic;dK/dV split-K workspace + host 固定序 fp32 sum)。
> 文件 `sync/meta-attn/meta_aiter_attn/flydsl/flash_attn_bwd_rect16_kernel.py`(bench 直接 import,无 regen)。基线由 `_apply_pack128.py` 生成(见 [[project_flydsl_meta_ck_replacement]])。

## ★验收口径 + 热稳态测量协议(2026-07-21,老板明确)
- **只验收 4 个 square-causal**:Sq=Skv ∈ {2048,4096,8192,16384},**B=1**,Hq128/Hkv16,D64,**full-causal**;精度**用 hw-exp(52.6dB)对齐 fwd**(fwd 那组用 flash_attn_gfx950 dualwave+hw);det + fp32 dQ 累加。
- **判定**:MI355 实测 conv-TF **÷1.2 → MI350**,比 1.4× H100 target **316/528/652/711**(=MI355 ≥1.68×H100)。conv FLOP=`10·B·Hq·S²·D·(1-(S-1)/2S)`。
- **热稳态协议(务必)**:WARMS=3s **连续满载预热(绝不 sleep)**+ 无 sleep 连续测 REPS 中位数;B=1 时钟衰减致同 kernel 冷 1024/热 727 差 40%,详见 [[pitfalls/02]]。bench=`_bench_square.py`@GPU5(env B/FAST/WARMS/REPS)。H100 全 20-config ref 在 `aiter_bench_reproducer/README_aiter_repro.md §6`(det=min(FA2,FA3)bwdT)。
- **热稳态基线(2026-07-21,÷1.2)**:fast **3/4**(2048/8192 过,4096/16384 差 1% 噪声级);hw **1/4**(2048 刚好,4096/8192/16384 差 5-6%)。hw 天花板=fast。
- **主攻=hw exp-overlap**(藏 quarter-rate v_exp 进 MFMA,+5-6% 拉到 fast:dq gap 最大 −6~8% 因 exp 在 bulk 后没藏,dkdv 已被 exp-before-dP 藏好 −3~4.5%)→ 拿下 8192,逼近 4096/16384。★**融 odo 判负**:full vs sum(odo+dq+dkdv) gap 为负(无 launch 开销),odo 仅占 1-3%,融合无收益(先测省了大改)。

## 瓶颈定性(rocprofv3 + PMC,已接地)
- **MFMA-operand-latency-bound**:bwd 98.9% 时间在 kernel,dq/dkdv 均匀 **MfmaUtil ~47-50% @ occ2**,远未打满 → 延迟受限非吞吐。
- 瓶颈链 = **exp2/pack 的 VALU operand-bubble**(dkdv 更重、是 leader)。
- **LDS 不是瓶颈**:`SQ_LDS_IDX_ACTIVE : MFMA_BUSY` 有 **4-8× 富余**(dq 4×/dkdv 8×)→ 非 port/带宽受限 → **swizzle/prefetch/bank/pad 全 DEAD**(见 05,pack-128 消 0% bank 零收益,bank conflict 是红鲱鱼)。
- **occupancy 锁 occ2**:⚠**纠正**:dq 实测 **VGPR=104**(非旧记的200;drop-B-GEMM 早瘦身),occ 是被 `waves_per_eu=2` 属性锁死,非 VGPR。但**强抬 wpe 实测必 spill**:wpe=3→编译器 schedule 膨胀 live-range→压 VGPR 到 84+scratch180B→慢32%;wpe=4→scratch368→慢5×。占用率-force 死(有数据)。
- **2-head GQA 融合(dq)判负**(2 q-head/WG 共享 K/V,流水 head0-GEMM2‖head1-exp2 填 exp2 bubble):bit-identical 但三变体全净负——BLOCK_M=128 spill(2头packs+累加器破256/wave,慢2.9×);BLOCK_M=64(每头QT半、grid不变)无spill 但慢+17% MfmaUtil42<51(小tile per-iter开销>融合收益)。累加器融合寄存器墙 + BLOCK减半penalty。见 [[project_dq_optimize_vs_dkdv]]。**dq 已 5/6 轴优于 dkdv,MfmaUtil低是 drop-B-GEMM 结构性(更省非缺陷)**。

## ✅ WINS(实测 KEEP,按增益排)
- **drop 冗余 B-GEMM / rho-R 全局修正项(dq)= +9.8%**:dq 里有可丢弃/可简化的第二 GEMM 或全局 renorm(rho/R)修正,drop 之 = 真结构性减 MFMA。**先审计每个 attention bwd 核有无这类项**(dkdv 无、只 +0.17%)。这是本 kernel 最大单杠杆。
- **odo delta 融合 = +5.3%**(长 Sq 最大):把 `delta=rowsum(O·dO)` 融进 bwd,消掉独立 delta 核(它在 wall 上全串行、随 Sq 放大)。★**必坑**:wrapper 必须 `out.to(q.dtype)`/`dout.to(q.dtype)` cast —— 融合核直接读 O/dO,若 harness 传 fp32 O → 崩/NaN(评分 gate raw=[],见 02)。
- **★dq fast-exp 去冗余 rowsum(P~)+renorm = +4%(1202→1245 / 1179→1229 TF,commit 886c5cd,2026-07-21)**:根因诊断=**dq fast 是 VALU-issue-bound**(rocprofv3 GPU5:VALU insts 3.2× MFMA,MfmaUtil 54%,**VMEM-wait 0**、LDS-wait 仅 11%→非内存/LDS 受限,是 VALU 端口饱和堵 MFMA 发射)。fast 路径比 hw 多背一个 per-row `rowsum(P~)` 累加器 + epilogue `dQ=sm/R·A` renorm。**该 renorm 是冗余的**:prescaled-lse 是真 log-sum-exp,Schraudolph P~ 已 sum→1(逐元素近似误差沿 row-sum 抵消)→ **R==1 到 bf16 精度**,与数据无关(只依赖 lse 是真 lse)。实测 SNR 4 seed/shape 全 Δ<0.01dB(34.79→34.79…)仍≥34、det 保持。删 rowsum VALU(连带 r_accs/r_cur/r_finals carry + _hred4/_kg_allreduce)= 把堵在 issue 端口的 VALU 让出来。⚠**推翻旧读法**"dq fast occ2 pipeline 最优、无头room":减 VALU 是 VALU-bound kernel 的真杠杆,exhaustive campaign 之前只盯 sched/tile/occ 全漏了这条。dkdv fast **无此项**(用 lse 归一化 P、本就无 rowsum;VALU 与 hw 几乎一致 3.15 vs 3.14)。
- **exp2 软流水(GQA head 轴)= +1.4%**:把 head h+1 的 `exp2(QK)` 预算藏进 head h 的 GEMM2 MFMA shadow(在 GQA head-loop 轴,**非** dt 轴——dt 轴已判负)。直击 operand-bubble。
- **s_setprio(1/0) 包 GEMM2 MFMA**:单独中性,叠在结构改动上 +1~2%(仅 MFMA-dense 的 GEMM2 有效;dkdv GEMM1 / dq 连续 span 判负)。
- ~~**dq iglp_opt(1) = +0.5%**(bit-identical)~~ ⚠**GPU5 fast-exp 上不复现,实测 -1.2~1.7%**(2026-07-21 A/B iglp 0/1/2,cache-cleared)——旧 +0.5% 是 hw-exp/旧 chi 硬件产物。committed kernel 保持 iglp_opt(0),别再重试。

## ❌ DEAD(实测 do-not-retry)
- **P5 shared-GEMM1 融合(dq+dkdv 共享重算的 GEMM1,-25% 总 MFMA)= 全死**:
  - q-outer grid 融合 = **确定性陷阱**:dK/dV 的 q-归约不用 atomics 无法同时"有界 HBM + 确定" → 必须 KV-outer。
  - KV-outer 融合的 spill:BLOCK_KV=128 spill 187(dK/dV 累加器 NT=8 单独就 256 VGPR);**BLOCK_KV=64 = 218 VGPR/0 spill/24KB/occ2**(round-4 的"636-spill 墙"是 direction-misread,可破)。
  - **但融合核实测 1.95-2.55× 慢**:BLOCK_KV=64 per-tile 惩罚(standalone -24%)+ ~2GB dQ workspace 流量 > -25% MFMA 省。**shared-GEMM1 融合对 hd64 判死**。
- **fp8 GEMM1 operands(FA-3 plain scalar-scaled,非 MX)**:SNR **28.2 < 34** → 破正确性门。hd64 D=64 收缩维太短,fp8 GEMM1 死。
- **★poly-exp2(deg-3 Horner 2^f + exponent-add,想要"fast 速度 + 高精度")= 判负(2026-07-21 dkdv 实测)**:动机=用户嫌 fast Schraudolph 35dB 精度不够、想要更准的 fast exp。调研 DeepSeek FlashMLA/FlashAttention:他们用**硬件 exp2(`ex2.approx` SFU)**,NV 上 fast+准是因为 SFU 是独立单元;AMD gfx950 的 `v_exp_f32` 是 VALU-pipe **quarter-rate**(=我们的 hw path)。**没有可抄的软件魔法**。实测 poly deg-3:**SNR 52.6 = hw 完全一致(非宣称的 60)**——因为**精度天花板是 bf16**(P~/dS 打包成 bf16 喂 GEMM2),不是 exp 近似;任何"准"exp 都卡 52.6。且 poly **比 hw 慢 25%**(dkdv 851 vs hw 1129):VALU-bound 上 ~10 个 full-rate FMA ≫ 1 个 quarter-rate v_exp,transcendental 单元其实很便宜。**结论:要精度就用 hw(52.6dB,只比 fast 慢 ~4-5%,是最快的"准"路径);poly/Schraudolph-变体都被 hw 支配,别再碰**。e865153 当年删 poly 是对的。
  - **A100/H100 老 FlashAttention `softmax.h` 实证**:纯 `exp2f`(→`ex2.approx`/MUFU.EX2 SFU),**零软件多项式**,log2(e) 折进 scale(=我们 hw 做法)。老代码里没有软件 fast-exp 技巧。
  - **唯一的软件-exp 技巧在 FA-4(Blackwell)**:hybrid——把 **10-25%** 的 exp 用软件 poly(Cody-Waite range-reduction + Horner)算在**空闲的 FMA cores** 上,与硬件 MUFU.EX2 **并行**跑,因 Blackwell tensor 快了但 MUFU 没快→SFU 饱和。**前提=SFU 与 FMA 是独立并行单元(NV 专属)**。**AMD gfx950 不成立**:v_exp 是 VALU-pipe quarter-rate、与 FMA **共用同一发射端口**→拆分不并行、只加 VALU 负载。可证明被支配:hybrid 成本 =(1-x)·4 + x·10 slots(v_exp 4 / poly 10,同 VALU),x=0(全 hw)即最小 → hybrid 速度 ≤ 全-hw。所以 FA-4 hybrid 对 AMD 无效,别试。
- **dkdv 拆 dV-only/dK-only 核(为 occ3)= -36%**:双份 Q/dO 读 + 拆分开销 > 占用率收益。
- **★dkdv fast bulk 手动向量化 exp+dS(抄 dq bulk 的 v4 fma+fptosi)= 判负 -2.5%(2026-07-21)**:dkdv 的 exp/dS 是 scalar per-t(`for t: _p_of`),想学 dq 的 unmasked-bulk v4 路径省 VALU。但**编译器早已把 scalar fma 打包成 v_pk_fma**,手写 v4 只多出 Pv 存储 + `[p4[t]]` extract 开销 → 净负;correctness bit-identical(SNR 35 不变)。**教训:先看编译器是否已 auto-vectorize(dkdv 已);dq bulk 向量化有用不代表 dkdv 有用**。⚠flydsl 坑:`range_constexpr` 展开循环里 `continue` 编不过(`SyntaxError: continue not properly in loop`,AST rewriter),用 if/else。
- **★dkdv fast 无 fast-专属头room(与 dq rowsum-drop 对照)**:profile 实测 dkdv fast VALU/MFMA=3.15 **≈ hw 3.14**、VALU count 几乎逐字节相同 → fast 与 hw **结构同一**(只 `_p_of` fptosi vs v_exp 之差,fast 赢在 full-rate 发射非指令数)。dq 的 rowsum-drop 那类"fast 背了 hw 不背的冗余 VALU"在 dkdv **不存在**(dkdv 用 lse 归一化 P、本就无 rowsum renorm)。dkdv fast 已天然"按 hw 流水线"。其余瓶颈=LDS 转置读延迟暴露(LDS_IDX:MFMA_BUSY=0.12→latency 非 bw,swizzle 死;depth-1 预取已满、depth-2 死;b128 转置读 LLVM Cannot select)+ 40B scratch 微 spill(exp-before-dp 的 P-live + r37 carry)。要破需算法级(换 GEMM2 operand 布局免转置读 / 降 P-live 压力)。
- **★dq fast K/V 跨-tile DMA 双缓冲预取 = 判负 -4%(2026-07-21,fast-exp;尽管减法探针显示 8.7% 头room)**:dq 每 kv-tile DMA K+V,减法探针(跳过 per-tile DMA、复用 tile)显示 **+8.7%** 暴露头room。但真做双缓冲预取(2× LDS 16→32KB,occ 不掉仍 occ2,SNR34.8/det 正确)**净负 -4.3~4.7%**:预取的 LDS-write 与 compute 的 LDS-read **争带宽**,加 barrier 开销 > DMA-隐藏收益;预取放 body-顶(撞 GEMM1 K/V 读)或挪到 GEMM2 前都一样负。⚠**教训:减法探针(删掉某成本测天花板)会高估可回收头room——删成本同时也删了它的副作用(此处=LDS 读写争用);"删了+8.7%"≠"藏起来能+8.7%"**。⚠(此条当时下的"dq fast 已 occ2 pipeline 最优"结论**已被 rowsum-drop +4% 推翻**——sched/tile/DMA 层确实闭合,但 VALU 层没查,见 WINS 的 rowsum-drop)。
- **★2026-07-21 dq fast 上"按 hw 流水线"三连全判负(GPU5 profile-grounded;真杠杆是随后的 rowsum-drop)**:①**exp-before-dP hoist**(抄 dkdv:S 后先算 P=exp2 再发 dP GEMM,让 exp VALU 藏 dP MFMA shadow)= dq **-1~2.4%**:dq 故意把 S+dP 打包成 dense MFMA 块(s_setprio(1))再统一发 VALU,拆开反让 exp VALU 抢 dP 的 MFMA 发射掉密度(occ2 姊妹 WG 本就填得比显式 hoist 好);dq kv-tile 结构 ≠ dkdv head 轴结构,同 reorder 不迁移。②**iglp_opt(1)/(2)** on dq fast = **-1.2~1.7%**(旧记 +0.5% 是 hw-exp/旧 chi,GPU5 fast 不复现)。③**BLOCK_KV** 128=-3.8~5.4%(register-walled),96≈噪声(+0.6/-0.3)。dkdv fast(MfmaUtil 60.9,VALU/MFMA 3.15,LDS-wait 27%)近饱和且已含 r37 预取、无 rowsum 可去。
- **★dual-wave / 8-wave / warp-spec / bare-asm cross-head 全 measure-closed 判负(2026-07-21 hw-exp 自主 campaign,决定性)**:8-wave NT=2(BLOCK_KV=256)屏障耦合 MfmaUtil 53→40(-11%);+stagger 错相位=1029(+3.5%,stagger 机制真有效但共享 Q/dO LDS WAR→NaN);**天花板 1029 < 4-wave baseline 1116**。根因=**baseline occ-2 的两个独立 WG 已机会性共驻同 SIMD、免费享无屏障跨-WG overlap**(见通用教训6)→ 8-wave 单-WG 是拿免费的换带屏障税的。bare-asm M6 全 body(手写 GEMM1+exp+pack+GEMM2 塞 256)编不过 occ2(full-width RA assertion 崩)/chunked 装下但 scratch424 spill=940<baseline。**别再走这一系(dual-wave/8-wave/cross-head/bare-asm)**。
- **LDS 双缓冲**:仅 meta 长 Skv=16384 上 **+1.25%**(regime-dependent;gpt_oss 短 shape 上 -3.6%,见 [[project_flydsl_dkdv_bwd_16x16]])—— 边际,不是杠杆。
- 其余判负:dkdv BLOCK_Q=128(-67%)、bigger tiles(register-walled)、q_split=3(-2%,K/V reload 冗余在满 grid)、kv_tile load-balance interleave(-5%,in-order dispatch 已最优)、dkdv GEMM2 depth-2 预取、IGLP dkdv(-4.1%)、per-Sq q_split(qs2 最优)、XCD/L2 remap(dkdv 74% L2 hit vs dq 91%,naive nested-division remap GPU-fault)。

## 结果 & 长 Sq 天花板真相(数据,非"不可达"结论)
- **660 → 783.7 TF/s mean(+18.7%,campaign 进行中可能再动)**,超 CK-det **x1.29-1.59**。
- **短/中 Sq(2048/4096)已达标**(710/794);**长 Sq(8192/16384 ~784/790)仍缺 844/866 约 7-8%**。
- agent PMC-接地判定:**确定性结构杠杆空间(P1-P5/fp8/q_split/occupancy/fusion)已 measure-closed**。残余长 Sq 缺口只剩两条,**均需上层裁决**:①放宽位确定性(atomic-dQ 融合,但确定性=替代 CK 的全部意义,GOAL 禁);②research 级新 exp2(砍 operand-bubble VALU 且守 SNR≥34,无 KB 先例)。
- 目标口径(MI355X conv TF/s)见 `meta_aiter_attn/GOAL_ck_det_1p4x_h100.md`:710/794/844/866 = 1.708× H100(=MI350 ×1.22 后 ≥1.4× H100)。

## 通用教训(可迁移到其它 dense attention bwd)
1. **先跑 `SQ_LDS_IDX_ACTIVE:MFMA_BUSY` 富余测**判 latency vs port-bound;富余大 → 别碰 swizzle/prefetch/bank(纯计数改善零性能)。
2. **减 MFMA 是唯一 sized 杠杆**:审计有无可丢的第二 GEMM / 全局修正项(drop-B-GEMM),再考虑核间 recompute 共享(但注意确定性陷阱 + register 墙 + workspace 流量常吃光收益)。
3. **delta/odo 类全串行辅助核优先融掉**(随 Sq 放大);融合必对齐 O/dO dtype。
4. **occ3 对 latency-bound 是硬墙**:VGPR 降不到阈值(hd64 ~≤168)就别指望;拆核/fp8 降 VGPR 的代价(双读/破 SNR)通常 > 占用率收益。
5. **exp2 是 hd64 bwd 的 operand-bubble 核心**:软流水藏(head 轴)有肉;缩短 exp2(降阶 minimax)是最后的确定性杠杆但量级小。hw v_exp_f32(SNR52.6)比 fast Schraudolph(35dB)慢 ~5%(quarter-rate 发射 + 1-op latency);若门要 SNR≥45 且允许 poly,poly-exp2 可回收大部分,但 full-rate poly 天花板≈fast、超不了。
6. **★occ-2 latency-bound bwd 上,warp-spec/8-wave/dual-wave 常是陷阱——先量 baseline 是否已双-WG 共驻**:occ-2 = 每 SIMD 常驻 2 个**独立** WG,一个卡 barrier 时另一个照发 MFMA → **baseline 已免费享无屏障跨-WG dual-wave overlap**。8-wave 单-WG(合并两 WG)= 把这免费 overlap 换成带 `gpu.barrier` 屏障税的组内 overlap,MfmaUtil 反跌、天花板 < baseline(dkdv 实测 1029<1116)。→ 想显式并行 GEMM↔softmax 前,先确认 baseline occ,occ-2 双-WG 就别做(fwd 常 occ-1 无跨-WG overlap 才是 dual-wave 正场)。真正 register-neutral 的净胜是**冷 load 寄存器预取**(下一 head lse/delta 塞进本 head GEMM2 MFMA shadow,dt==DT-1 发射,+1.81% dkdv hw-exp),不是重构 wave 结构。
7. **VGPR 读 ISA 别读 rocprof**:rocprofv3 --pmc/kernel-trace 的 VGPR_Count 对 256-VGPR 内核**误报 128**(methodology/03);ISA `.vgpr_count` 才权威。dkdv 真值 256(占满 occ-2),据 128 判寄存器余量会全错。
