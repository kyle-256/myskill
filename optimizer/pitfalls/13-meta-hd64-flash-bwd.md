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
## ★★★ 2026-07-27 收官:全部达标(4/4 square + 19/20 config)——成果在 **turbo 侧**
> ⚠**代码位置**:这批优化落在 **`sync/mxfp4/Primus-Turbo/primus_turbo/flydsl/attention/flash_attn_bwd.py`**
> (分支 `dev/kyle/flydsl-bwd16384-campaign`),**meta-attn 的 `flash_attn_bwd_rect16_kernel.py` 未移植**。
> 下面 2026-07-23 那份旧基线是 meta-attn 侧的,读数时别混。campaign 目录
> `sync/flydsl_optimizer/flydsl_campaigns/20260726_152547`(memory.md 有每轮完整 scheme/tried/kb_check)。

- **4-square boss(B=1,÷1.2 比 1.4×H100 target 316/528/652/711)= hw 4/4、fast 4/4**(此前 hw 2/4、fast 3/4):
  - hw:2048=**565**(2.08×)/ 4096=**765**(1.69×)/ 8192=**832**(1.49×)/ 16384=**877**(1.44×)
  - fast:573 / 776 / 841 / **881**;det 全 True。**16384 反超 8192**(曲线方向终于对了)。
- **Meta 全 20-config(B=4,det 口径,ratio=best-H100-det÷(MI355×1.2)≥1.4)= 19/20**(此前 hw 16/20):
  full-causal **9/10**(此前 6/10;既有 MI350 CK-det 在 full 上 **0/10**)、SWA **10/10**(2.12-3.41×)。
  唯一未过:`newshape full 2048` = 1.37×(目标 98%,差 2%)——Sq 最短、并行度最低,且 campaign 计分只覆盖 16384。
  harness `_bench_meta20.py`(turbo 仓根,内置 README 的 H100 det ms 表)。
- **per-kernel(S=16384,B=1,hw,流水内)**:full 12.583ms = 873.9 conv TF;odo 0.139(1.1%)/ **dq 5.085(40.4%)**/ **dkdv 7.358(58.5%)**。
  ★**dq 的"流水税"从 +6.8% 降到 +0.9%**(XCD-major remap 的功劳),dkdv +0.1%。
- **精度**:SNR dq/dk/dv = 50.26/50.36/50.77(hw),det=True。⚠比旧记的 52.x 低约 2dB,**不是回归**:
  是 `fold_lse` 从"bench 建 False / 部署发 True"对齐成两边都 True,50.x 一直是线上真实值(见下 RULER FIX)。

- **★验收基线(2026-07-23 fresh @ node crsuse2-m2m-289 / GPU1,÷1.2→MI350 比 1.4×target 316/528/652/711)**:
  - **hw-exp 2/4**:2048=453✅(2.00×H100)/ 4096=599✅(1.59×)/ 8192=636(−2.5%,1.36×)/ 16384=665(−6.5%,1.31×);det 全 True。
  - **fast-exp 3/4**:2048=474✅/ 4096=635✅/ 8192=671✅(1.44×)/ 16384=703(−1.1%,噪声级)。
  - **★q_split 突破(commit 9aa3be0)是把 hw 从 1/4 拉到 2/4 的关键**:`_qsplit_for` 旧上限卡 2 漏扫,qsp≥4 铺满 CU grid → 2048/4096 翻达标(4096 conv-TF 594→702);per-Sq 最优 2048/4096→4、8192→6、16384→3。
  - 缺口全集中在 **8192/16384 大 shape**(hw −2.5%/−6.5%,fast 0/−1.1%),根因 dkdv 结构层(见下)。hw 天花板=fast。
- **主攻=hw exp-overlap**(藏 v_exp 进 MFMA;⚠此处原写 quarter-rate,实测是 **half-rate ~8 拍**,见本卡「更正…v_exp_f32 是 half-rate」;+5-6% 拉到 fast:dq gap 最大 −6~8% 因 exp 在 bulk 后没藏,dkdv 已被 exp-before-dP 藏好 −3~4.5%)→ 拿下 8192,逼近 4096/16384。★**融 odo 判负**:full vs sum(odo+dq+dkdv) gap 为负(无 launch 开销),odo 仅占 1-3%,融合无收益(先测省了大改)。

## 瓶颈定性(rocprofv3 + PMC,已接地)
- **MFMA-operand-latency-bound**:bwd 98.9% 时间在 kernel,dq/dkdv 均匀 **MfmaUtil ~47-50% @ occ2**,远未打满 → 延迟受限非吞吐。
- 瓶颈链 = **exp2/pack 的 VALU operand-bubble**(dkdv 更重、是 leader)。
- **LDS 不是瓶颈**:`SQ_LDS_IDX_ACTIVE : MFMA_BUSY` 有 **4-8× 富余**(dq 4×/dkdv 8×)→ 非 port/带宽受限 → **swizzle/prefetch/bank/pad 全 DEAD**(见 05,pack-128 消 0% bank 零收益,bank conflict 是红鲱鱼)。
- **occupancy 锁 occ2**:⚠**纠正**:dq 实测 **VGPR=104**(非旧记的200;drop-B-GEMM 早瘦身),occ 是被 `waves_per_eu=2` 属性锁死,非 VGPR。但**强抬 wpe 实测必 spill**:wpe=3→编译器 schedule 膨胀 live-range→压 VGPR 到 84+scratch180B→慢32%;wpe=4→scratch368→慢5×。占用率-force 死(有数据)。
- **2-head GQA 融合(dq)判负**(2 q-head/WG 共享 K/V,流水 head0-GEMM2‖head1-exp2 填 exp2 bubble):bit-identical 但三变体全净负——BLOCK_M=128 spill(2头packs+累加器破256/wave,慢2.9×);BLOCK_M=64(每头QT半、grid不变)无spill 但慢+17% MfmaUtil42<51(小tile per-iter开销>融合收益)。累加器融合寄存器墙 + BLOCK减半penalty。见 [[project_dq_optimize_vs_dkdv]]。**dq 已 5/6 轴优于 dkdv,MfmaUtil低是 drop-B-GEMM 结构性(更省非缺陷)**。

## ✅✅ 2026-07-27 campaign 新增杠杆(turbo 侧,789.6→875.3 conv TF,+10.85%)
> 这批全在**我之前没碰过的两个层面**:grid/派发映射,和冒险锚点粒度。kernel 内部(tile/双缓冲/遍历方向)
> 反复调只值 ~1%,而下面这些是 +10%。**先查这两层再去抠 kernel 体**。

- **★★XCD-major block_id 解码 = dq +0.54% / dkdv +2.81%(单项最大)**:MI355X 有 **8 个 XCD、L2 各自私有**。
  原本 kv-head 最快变化 → 同一 kv-head 的 K/V 被多个 XCD 各读一份。改成 `xcd = block_id % 8`(片上实测确认该式成立),
  让每个 XCD 拿一整块 `(batch, kv-head)`,读同一份 K/V 的 8 个 GQA WG 挨在一起。dq 实测 L2 hit 86.5→94.8%、miss −62%。
  ★**内层最快轴 dq 和 dkdv 必须相反**:dq 要 **kv-head 相邻**、dkdv 要 **q 位置相邻**(split_idx 最快)。
  同一个 remap 用错内层 = **−2.5% 对 +2.4%**。需 `NUM_HEADS_KV % 8 == 0` 门控,其余 head 数走旧解码(双射性离线穷举验证)。
  ⚠**这推翻了本卡旧 DEAD 条目**"XCD/L2 remap … naive nested-division remap GPU-fault"——那次是实现炸了,不是方向错。
- **★dq 派发顺序 = 降序 q_tile(LPT)+2.50%**:因果掩码下每 WG 工作量 = `(q_tile+1)*BLOCK_M/BLOCK_KV`,
  单调递增;而**派发顺序就是 list-schedule 顺序** → 长任务优先(LPT)自然平衡尾部。
  ⚠同样推翻旧 DEAD"kv_tile load-balance interleave(−5%,in-order dispatch 已最优)":in-order 对 **dkdv** 已是 LPT,
  但 **dq 是反的**。离线用 64 slots/XCD 贪心模拟器验证过再上机(模拟预测 +2.30%,实测 +2.9%,模型可信)。
- **★causal-aligned q-tile origin +0.96%**:`num_q_tiles*BLOCK_M` 超出 `seq_len_q` 的 `pad`(16384 时 128 行),
  锚在 row 0 时**全部浪费在最后一个 tile**——恰是因果 kv 范围最长的那个,它那些不存在的行要走完整条范围。
  把每个 tile 原点下移 `floor(pad/BLOCK_KV)*BLOCK_KV`(=96),overshoot 落到 tile 0(最短),
  每个 tile 的因果范围各少一个 BLOCK_KV 步:**7481→7396 次 kv-block 访问 = 每项 per-block 成本(MFMA/v_exp/DMA/LDS)−1.1%**。
  必须是 BLOCK_KV 整数倍且 `BLOCK_M % BLOCK_KV == 0` 才保持对齐;tile 0 夹到 row 0 并加 owned-end store 界(共享行重算但只写一次,det 不变)。
- **★★冒险锚点粒度:per-slot → per-v4 = dq +1.76%(r15,纯 anchor);dkdv 该轮 +0.77% 是 anchor 与标尺修正合计,anchor 本身值 kernel 时间 −1.10%**:接本卡"clamp 不是数值保护而是
  MFMA→trans 冒险载体"的旧发现——**但可以大幅减少它的数量**。每个 v4 只留 slot 0 的 `min(acc,0)` 作锚点,
  slot 1..3 裸读同一 v4,靠 v_exp inline asm 的 **dead input operand**(`"=v,v,v"`,asm 文本只引用 $0/$1,不生成指令)钉在锚点之后。
  dq `v_min` 72→3、热循环 663→596 issue slots;dkdv 256→64。vgpr/scratch/mfma/vexp/ds_read 全不变,输出 bit-identical。
  ★★**正确性规则(血的教训,见下 DEAD)**:锚点必须**读它所保护的那个 v4**。"读遍组内每个 v4"**不充分**。
- **★用 `llvm.amdgcn.exp2.f32` intrinsic 替掉手写 asm+手工锚点(dq)= −0.79%**:intrinsic 本身就是
  编译器可见的累加器读 → 编译器自己插 MFMA→VALU 等待周期,**锚点粒度问题从根上消失**;
  单指令、无 denormal range-reduction(不同于 `math.exp2`,后者展开成 72 v_ldexp+144 v_cndmask+72 v_cmp,bulk 596→953 slots)、
  side-effect free 所以仍能沉进 GEMM2 的 MFMA 气泡。⚠**别推广到 dkdv**:同一改动在 dkdv 是 **+1.06% 更慢**(nop_cycles 205→439)。
- **★上面那条"`math.exp2` 展开很贵"要一路查到 masked 分支(gpt-oss round-8,+0.6%)**:dq 的
  `_p_of` exact 路径一直是 `ArithValue.exp2`,而 dq bulk 与 dkdv `_p_of` 早就用裸 `v_exp_f32`——
  **只有 masked(对角)分支落在展开上**,ISA 实测每个 masked trip 多 48 `v_ldexp`+48 `v_cmp_gt`+
  48 `v_add`+96 `v_cndmask` ≈ **960 cycle**(masked block 3333 cyc vs bulk 1755)。改成裸 `_vexp`
  后 dQ **bit-identical**(SNR 50.25597 逐位相同):masked slot 是 `s_r=-inf`→`v_exp_f32(-inf)=0` 精确,
  而对角 P 是 softmax row 的最大项、**永远不可能 denormal**,denormal range-reduction 纯属白付。
  masked 只占 dq kv-tile 访问的 4.6%(每 q tile 3 个,BLOCK_M=192/BLOCK_KV=64)却值 **+0.6% 整趟**。
  ⚠ round-2 的 `dqmaskexp` 探针把同类改动记成 **−0.46%「无收益」**——那是单窗口 ±0.30% 地板内的读数;
  ABBA 平衡设计重测 = **4/4 win、区间完全不重叠(827.2/827.2/828.1/828.2 vs 818.6/822.1/823.6/824.6)**。
  ⇒ **不平衡单窗口探针的"中性/微负"结论不足以判死一个减指令改动。**

## ✅ WINS(实测 KEEP,按增益排)
- **drop 冗余 B-GEMM / rho-R 全局修正项(dq)= +9.8%**:dq 里有可丢弃/可简化的第二 GEMM 或全局 renorm(rho/R)修正,drop 之 = 真结构性减 MFMA。**先审计每个 attention bwd 核有无这类项**(dkdv 无、只 +0.17%)。这是本 kernel 最大单杠杆。
- **odo delta 融合 = +5.3%**(长 Sq 最大):把 `delta=rowsum(O·dO)` 融进 bwd,消掉独立 delta 核(它在 wall 上全串行、随 Sq 放大)。★**必坑**:wrapper 必须 `out.to(q.dtype)`/`dout.to(q.dtype)` cast —— 融合核直接读 O/dO,若 harness 传 fp32 O → 崩/NaN(评分 gate raw=[],见 02)。
- **★dq fast-exp 去冗余 rowsum(P~)+renorm = +4%(1202→1245 / 1179→1229 TF,commit 886c5cd,2026-07-21)**:根因诊断=**dq fast 是 VALU-issue-bound**(rocprofv3 GPU5:VALU insts 3.2× MFMA,MfmaUtil 54%,**VMEM-wait 0**、LDS-wait 仅 11%→非内存/LDS 受限,是 VALU 端口饱和堵 MFMA 发射)。fast 路径比 hw 多背一个 per-row `rowsum(P~)` 累加器 + epilogue `dQ=sm/R·A` renorm。**该 renorm 是冗余的**:prescaled-lse 是真 log-sum-exp,Schraudolph P~ 已 sum→1(逐元素近似误差沿 row-sum 抵消)→ **R==1 到 bf16 精度**,与数据无关(只依赖 lse 是真 lse)。实测 SNR 4 seed/shape 全 Δ<0.01dB(34.79→34.79…)仍≥34、det 保持。删 rowsum VALU(连带 r_accs/r_cur/r_finals carry + _hred4/_kg_allreduce)= 把堵在 issue 端口的 VALU 让出来。⚠**推翻旧读法**"dq fast occ2 pipeline 最优、无头room":减 VALU 是 VALU-bound kernel 的真杠杆,exhaustive campaign 之前只盯 sched/tile/occ 全漏了这条。dkdv fast **无此项**(用 lse 归一化 P、本就无 rowsum;VALU 与 hw 几乎一致 3.15 vs 3.14)。
- **★q_split 铺满 CU grid = 中间档翻达标(commit 9aa3be0)**:dkdv 的 `_qsplit_for` 旧上限卡 2,大 grid 下 CU 空转;放开到 qsp≥4(2048/4096→4、8192→6、16384→3)铺满 → hw 2048/4096 从不达标翻到达标(4096 conv-TF 594→702)。是"占满 grid"占用率杠杆(非算法),但对中等 Sq 决定性;BKV-up 判死(256 慢 3-4× occ 崩),dq 侧配置已最优(bk64/wpe2 峰值)。
- **★冷 load 寄存器预取(dkdv hw-exp)= +1.81%**(1114→1134,commit c619847):下一 head 的 lse/delta 是冷 load,在本 head GEMM2 的 `dt==DT-1` 处提前发 buffer_load,+32 VGPR carry 只叠最后一个 dt 的 MFMA(spill-neutral、守 occ2、bit-identical)→ 藏掉 consumer latency。是 occ-2 latency-bound bwd 上真正 register-neutral 的净胜(对比 dual-wave 全系判负,见通用教训6)。
- **exp2 软流水(GQA head 轴)= +1.4%**:把 head h+1 的 `exp2(QK)` 预算藏进 head h 的 GEMM2 MFMA shadow(在 GQA head-loop 轴,**非** dt 轴——dt 轴已判负)。直击 operand-bubble。
- **s_setprio(1/0) 包 GEMM2 MFMA**:单独中性,叠在结构改动上 +1~2%(仅 MFMA-dense 的 GEMM2 有效;dkdv GEMM1 / dq 连续 span 判负)。
- ~~**dq iglp_opt(1) = +0.5%**(bit-identical)~~ ⚠**GPU5 fast-exp 上不复现,实测 -1.2~1.7%**(2026-07-21 A/B iglp 0/1/2,cache-cleared)——旧 +0.5% 是 hw-exp/旧 chi 硬件产物。committed kernel 保持 iglp_opt(0),别再重试。

## ❌❌ 2026-07-27 campaign 新增判负 + 新坑
- **★★★锚点放大到"组级"= 更快但算错,且只在特定 grid 暴露(最危险的一条)**:
  把冒险锚点从 per-v4 放大到 per-whole-GEMM1a-block(8 个 `v_min`/kernel)**快 2.53%**,
  放大到 per-mt-pair(32 个)快 1.79% —— 但 dk/dv 掉到 **16.89/17.85 dB 且 det=FALSE**,
  **只在 Sq=4096/8192(q_split=4, BLOCK_KV=128)暴露,主测形状完全测不出来**。
  同类:dkdv min3-group-of-3 快 0.31%,在 **BLOCK_KV=64** 才掉到 19.7 dB det=False。
  ★规则:**锚点必须读它所保护的那个 v4;"读遍组内每个 v4"不充分**。
  ★方法:用 env-gated builder 做三向隔离(fold OFF / fold ON+per-slot / fold ON+组级)才能证明是**锚点**而非 fold 的锅。
  ★**任何锚点/冒险类改动必须跑多形状 grid(至少覆盖 q_split∈{3,4}×BLOCK_KV∈{64,128}),单形状 SNR 会放行错误内核**。
- **non-temporal (NT) CPol 全系判负**:给 dq 的 Q/dO/LSE/DELTA load 打 NT(想让约 820MB 单次读的流量 evict-first、
  不挤占被重读约 370 次的 K/V)= **792.2 vs 791.3 中性**;只 Q/dO 791.8;只 dQ store 793.2;
  **dkdv 的 Q/dO DMA 打 NT = −7.6%**。★由此**推翻**"dq 的 L2 损失来自 dkdv 读 Q/dO"这个猜测:
  那些读是**重度复用**,不是污染。(dkdv workspace store 打 NT 早先也已判负 −0.5%。)
- **★"dq 流水内损失来自 L2 命中率"这个因果不成立**:XCD-remap 把 dq 的 L2 hit 做到 **94.8%(比 solo 的 91% 还高)**,
  solo-vs-pipeline 仍差约 4%。★判内存类想法要看 **`SQ_WAIT_ANY` / `SQ_VALU_MFMA_COEXEC_CYCLES`,不要看 TCC hit%**
  ——大幅 hit 率改善在这类 kernel 上可以值 0 wall。
- **dq K/V 双缓冲(再次判负,根因更新)**:bkv 96/64/32 分别 −2.9%/−3.2%/−2.3%。
  ★**占用率假设被排除**:bkv=32 时双缓冲的 LDS 只有部署版(bkv=96 单缓冲)的 2/3,照样负。
  真根因:减法探针那 6.9% 是 K/V 的**吞吐**成本(HBM/L2 + LDS 写带宽),**不是延迟暴露** —— 双缓冲藏延迟、
  一个字节都不搬走。★**赢的形式是"单缓冲软件流水交接"**:在 `k_tr` 寄存器半数死亡处发下一块 DMA,
  单 LDS buffer 所以没有预取写 vs 计算读的争用,也不动占用率(dq `SQ_WAIT_ANY` −41%,coexec +13%)。
- **`buffer_load_lds` 在飞 → 强制后续**每一个**同 LDS 分配的 `ds_read` 前 `s_waitcnt vmcnt(0)`**(实测 drain 数 16→85/109)。
  ⇒ **DMA-to-LDS 只能与"纯寄存器驻留的工作"重叠**,不能与任何读同一 LDS 的计算重叠。这是 LDS 双缓冲系反复失败的机制根源。
- **LDS 双缓冲在 256-VGPR kernel 上不免费**:它会复制每一份已 hoist 的 lane-dependent LDS 地址 → 本例 **424B scratch**,
  dkdv 实测 −16.5%。★**改 LDS 结构前先 grep ISA 的 `private_seg_size` 和 `s_waitcnt.*vmcnt\(0\)` 计数**,
  这两个失败模式在 SNR(bit-identical)和 bench 报错里**都看不见**。
- **手动 packed(v_pk_mul)dS=P*dP 判负,但旧卡的理由是错的**:本卡旧条目称"dkdv 编译器早已把 scalar fma 打包成 v_pk_fma"
  —— ★**在当前工具链上是 FALSE**:dkdv 的 ISA 有 515 `v_mul_f32` + 450 `v_fmamk_f32`、**零 packed f32**(dq 的 bulk 才有 `v_pk_mul_f32`)。
  手动打包**确实生效**(515→251,VGPR 256→253,甚至 scratch 12→0),但 **wall 纹丝不动、反而 +1.42%**(s_nop 179→270)。
  ⇒ 结论(别做)对,但根因是 **dkdv 是依赖延迟受限,不是 VALU-issue 受限**。★证明一个 kernel 不是 issue-bound 的干净手法:
  砍掉近百条 issue slot 而 wall 不动。
- **rocprof 的 VGPR_Count 报的是一半**:显示 **128 就已经是 occ-2 悬崖**,显示 132 就意味着真值越界、实测掉约 8%,
  而 **scratch 仍是 0、SNR 仍 bit-identical** —— 完全看不出来。★**任何 dq 结构改动,bench 之前先看 ISA 的 VGPR。**
- **把 dkdv 的 idiom 照搬到 dq 会翻车**:把 dq 的 DMA 寻址 hoist 进 soffset/SGPR(在 dkdv 值 +0.4%)
  → VGPR 132→144、掉占用率。★**在寄存器悬崖上的 kernel,每 tile 重算便宜的地址算术优于 hoist**。regime-dependent,不是通用规则。
- **方向 A(dq/dkdv 融合)首次有了硬数据**:dq 的确定性 kv_split=2 split-workspace + 固定序 reduce **做出来了且正确**,
  workspace 只需 **537MB(不是 CK 那个 133GB —— 那个数对本形状是高估)**。但 reduce 净开销 = dq 的 +3.58% ≈ wall 的 1.4%,
  超过 split 本身收益 → REVERTED(实现保留为 `r15_dq_kvsplit_variant.py.txt`)。**融合本身没被否,是 reduce 的代价没摊平**。

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
- **★dkdv fast bulk 手动向量化 exp+dS(抄 dq bulk 的 v4 fma+fptosi)= 判负 -2.5%(2026-07-21)**:dkdv 的 exp/dS 是 scalar per-t(`for t: _p_of`),想学 dq 的 unmasked-bulk v4 路径省 VALU。但~~**编译器早已把 scalar fma 打包成 v_pk_fma**~~ ⚠**2026-07-27 实测此归因为 FALSE**:dkdv ISA 是 515 `v_mul_f32`+450 `v_fmamk_f32`、**零 packed f32**,手动打包确实生效(515→251)但 wall 不动甚至 +1.42%(s_nop 179→270)——**真根因是 dkdv 依赖延迟受限、非 VALU-issue 受限**(结论对、理由错),手写 v4 只多出 Pv 存储 + `[p4[t]]` extract 开销 → 净负;correctness bit-identical(SNR 35 不变)。**教训:先确认 kernel 到底是不是 issue-bound —— 干净判法是砍掉近百条 issue slot 看 wall 动不动;dq bulk 向量化有用不代表 dkdv 有用**。⚠flydsl 坑:`range_constexpr` 展开循环里 `continue` 编不过(`SyntaxError: continue not properly in loop`,AST rewriter),用 if/else。
- **★dkdv fast 无 fast-专属头room(与 dq rowsum-drop 对照)**:profile 实测 dkdv fast VALU/MFMA=3.15 **≈ hw 3.14**、VALU count 几乎逐字节相同 → fast 与 hw **结构同一**(只 `_p_of` fptosi vs v_exp 之差,fast 赢在 full-rate 发射非指令数)。dq 的 rowsum-drop 那类"fast 背了 hw 不背的冗余 VALU"在 dkdv **不存在**(dkdv 用 lse 归一化 P、本就无 rowsum renorm)。dkdv fast 已天然"按 hw 流水线"。其余瓶颈=LDS 转置读延迟暴露(LDS_IDX:MFMA_BUSY=0.12→latency 非 bw,swizzle 死;depth-1 预取已满、depth-2 死;b128 转置读 LLVM Cannot select)+ 40B scratch 微 spill(exp-before-dp 的 P-live + r37 carry)。要破需算法级(换 GEMM2 operand 布局免转置读 / 降 P-live 压力)。
  ★★ **2026-08-29 补:这两条各自的下场,以及一个更好的坐标。** (a) 换 GEMM2 operand 布局**是对的**——
  gpt_oss d64 上把 dK 的 GEMM2b 按 GQA head 配对成 `16x16x128_f8f6f4` = **+0.6%**,把 GEMM1a/GEMM2b
  的操作数换成 HBM 交接来的 E4M3 = 多轮累计 +8%。(b) **"降 P-live 压力"不能靠挪消费者实现**:
  三个方向全部实测为负——把 dP 按 kv 16-tile 拆开让每块 VALU 跟下一块 MFMA 重叠 **−4.0%**;
  把整个半区的 dS/pack 块推迟进下一半的 GEMM1a **−0.9%**;把 dV 的 MFMA 直接塞进 softmax 自己的
  nt 循环(正好在喂它的 P pack 之后)**−6.3%,且 spill 0→5**。原因是 **softmax 块本身就是这个 body
  的寄存器峰值**,任何被迫跨越它存活的东西都要付 spill。⇒ 卡面该写的是**"先缩短 softmax 自己持有的
  东西(S/dP/P/DT 片段的位宽与条数),再谈把消费者搬进来"**,而不是笼统的"降 P-live"。
  (c) 判这一族改动**用 `SQ_VALU_MFMA_COEXEC_CYCLES / SQ_VALU_MFMA_BUSY_CYCLES`**,见
  methodology/15 §occ-1 第一诊断量:同 body 该比值 19.2%,同机同 shape 的 fwd 是 48.1%,
  而指令数/寄存器数在这个 body 上已经**双向测到 0**。
- **★dq fast K/V 跨-tile DMA 双缓冲预取 = 判负 -4%(2026-07-21,fast-exp;尽管减法探针显示 8.7% 头room)**:dq 每 kv-tile DMA K+V,减法探针(跳过 per-tile DMA、复用 tile)显示 **+8.7%** 暴露头room。但真做双缓冲预取(2× LDS 16→32KB,occ 不掉仍 occ2,SNR34.8/det 正确)**净负 -4.3~4.7%**:预取的 LDS-write 与 compute 的 LDS-read **争带宽**,加 barrier 开销 > DMA-隐藏收益;预取放 body-顶(撞 GEMM1 K/V 读)或挪到 GEMM2 前都一样负。⚠**教训:减法探针(删掉某成本测天花板)会高估可回收头room——删成本同时也删了它的副作用(此处=LDS 读写争用);"删了+8.7%"≠"藏起来能+8.7%"**。⚠(此条当时下的"dq fast 已 occ2 pipeline 最优"结论**已被 rowsum-drop +4% 推翻**——sched/tile/DMA 层确实闭合,但 VALU 层没查,见 WINS 的 rowsum-drop)。
- **★2026-07-21 dq fast 上"按 hw 流水线"三连全判负(GPU5 profile-grounded;真杠杆是随后的 rowsum-drop)**:①**exp-before-dP hoist**(抄 dkdv:S 后先算 P=exp2 再发 dP GEMM,让 exp VALU 藏 dP MFMA shadow)= dq **-1~2.4%**:dq 故意把 S+dP 打包成 dense MFMA 块(s_setprio(1))再统一发 VALU,拆开反让 exp VALU 抢 dP 的 MFMA 发射掉密度(occ2 姊妹 WG 本就填得比显式 hoist 好);dq kv-tile 结构 ≠ dkdv head 轴结构,同 reorder 不迁移。②**iglp_opt(1)/(2)** on dq fast = **-1.2~1.7%**(旧记 +0.5% 是 hw-exp/旧 chi,GPU5 fast 不复现)。③**BLOCK_KV** 128=-3.8~5.4%(register-walled),96≈噪声(+0.6/-0.3)。dkdv fast(MfmaUtil 60.9,VALU/MFMA 3.15,LDS-wait 27%)近饱和且已含 r37 预取、无 rowsum 可去。
- **★dual-wave / 8-wave / warp-spec / bare-asm cross-head 全 measure-closed 判负(2026-07-21 hw-exp 自主 campaign,决定性)**:8-wave NT=2(BLOCK_KV=256)屏障耦合 MfmaUtil 53→40(-11%);+stagger 错相位=1029(+3.5%,stagger 机制真有效但共享 Q/dO LDS WAR→NaN);**天花板 1029 < 4-wave baseline 1116**。根因=**baseline occ-2 的两个独立 WG 已机会性共驻同 SIMD、免费享无屏障跨-WG overlap**(见通用教训6)→ 8-wave 单-WG 是拿免费的换带屏障税的。bare-asm M6 全 body(手写 GEMM1+exp+pack+GEMM2 塞 256)编不过 occ2(full-width RA assertion 崩)/chunked 装下但 scratch424 spill=940<baseline。**别再走这一系(dual-wave/8-wave/cross-head/bare-asm)**。
- **LDS 双缓冲**:仅 meta 长 Skv=16384 上 **+1.25%**(regime-dependent;gpt_oss 短 shape 上 -3.6%,见 [[project_flydsl_dkdv_bwd_16x16]])—— 边际,不是杠杆。
- 其余判负:dkdv BLOCK_Q=128(-67%)、bigger tiles(register-walled)、q_split=3(-2%,K/V reload 冗余在满 grid)、~~kv_tile load-balance interleave(-5%,in-order dispatch 已最优)~~ ⚠**2026-07-27 部分推翻**:in-order 对 **dkdv** 恰好已是 LPT,但 **dq 是反的**,改降序 q_tile = **+2.50%**(见上"新增杠杆")、dkdv GEMM2 depth-2 预取、IGLP dkdv(-4.1%)、per-Sq q_split(qs2 最优)、~~XCD/L2 remap(naive nested-division remap GPU-fault)~~ ⚠**2026-07-27 完全推翻**:那次是**实现炸了不是方向错**;正确做法 `xcd=block_id%8` + 每 XCD 一整块 (batch,kv-head),是本任务**单项最大杠杆**(dkdv +2.81%)。

## 收官结果(2026-07-23,meta-attn 侧)——⚠**核心结论已被 2026-07-27 推翻,见本卡开头**
> 下面这段判定"确定性结构杠杆全 measure-closed,剩两条需上层裁决:①放宽位确定性 ②research 级 exp-overlap"。
> ★**两条都没用上**:2026-07-27 在 **turbo 侧**靠 **grid/派发映射层(+7%)+ 冒险锚点粒度(+3.6%)** 达标,
> **确定性一位没放宽,也没做 exp-overlap**。★教训:"杠杆已穷尽"往往意味着**只穷尽了我一直在看的那一层**
> —— 当时全部精力在 kernel 体内(tile/预取/调度/占用率),而 grid 映射与派发顺序这层**从没被审计过**。
> 下结论前先问:**我有没有整整一层没查?**


- **正确性 + 确定性全闭合**:20-config(10 shape × {full-causal,SWA})× B∈{1,4} = **40/40 PASS**;hw-exp dq~52 / dk~50.8 / dv~50.8 dB(全 ≥48 门槛),两次运行 `torch.equal` bitwise det=True(split-K workspace + host 固定序 fp32 sum,无 float atomic)。
- **4-square boss 验收(B=1,÷1.2)**:hw **2/4**、fast **3/4**(数见 ★验收基线;缺口全在 8192/16384)。
- **全 20-config perf(B=4,÷1.2 比 1.4×H100,H100 ref=min(FA2,FA3)bwdT)**:hw **16/20**(SWA 10/10 = 1.77-3.19×;full-causal 6/10)、fast **18/20**(SWA 10/10 = 1.85-3.31×;full-causal 8/10)。**SWA 全线大幅超标**,缺口只在 full-causal 大 shape。既有 MI350 CK-det 基线在 full-causal 仅 0.83-1.16× 且**无 SWA 核** → FlyDSL det bwd 在全 20-config 超既有 MI350 det 基线。
- **残余缺口(8192/16384 full-causal)根因 = dkdv 结构层**:hw 的 `v_exp`(**half-rate**,非此处原写的 quarter-rate)在 bulk 之后未被 MFMA 藏(dq gap 最大,dkdv 已被 exp-before-dP 藏好);16384 叠 dkdv LDS 转置读延迟暴露。**确定性结构杠杆(P1-P5 / fp8-GEMM1 / q_split / occupancy-force / odo-fusion / dual-wave-8wave-warpspec 全系)已 measure-closed 判负**(见 DEAD 段)。剩两条均需上层裁决:①放宽位确定性(atomic-dQ 融合,但确定性=替代 CK 的全部意义,GOAL 禁);②research 级 exp-overlap(藏 v_exp 进 MFMA shadow 且守 SNR≥48,无 KB 先例)。
- **主攻方向(下一步)**:hw exp-overlap 去吃 8192 的 2.5%;16384 的 6.5% 需动 dkdv GEMM2 operand 布局降转置读压力。harness=`_verify_all.py`(40-config 正确性+det)/`_bench_all.py`(20-config perf)/`_bench_square.py`(4-square boss)。H100 全 20-config ref 在 `aiter_bench_reproducer/README_aiter_repro.md`。详见 [[project_meta_bwd_accept_bench]]。

## ★★ 通用教训(2026-07-27 campaign,优先级最高)
8. **★先打 grid/派发映射层,再抠 kernel 体**。我在 kernel 内部(tile 大小、双缓冲、遍历方向)反复调了几周只拿到 ~1%,
   而 XCD-L2 亲和 + LPT 派发序 + padding 对齐这三条加起来 **+7%**,它们全在 kernel 之外。
   ★通用清单:①per-XCD 私有 L2 上,同时驻留的 WG 是否共享同一份权重/KV?②派发顺序是否等于 list-schedule 顺序、
   工作量是否单调(是则 LPT)?③tile 数 × tile 大小超出真实长度的那部分 padding,落在最贵还是最便宜的 tile 上?
9. **★★"标尺"必须镜像部署,否则优化几周的是不发货的配置**。本例 `_bench_*` 直接调 builder 用默认参数,
   而 `_get_bwd` 部署时传的是另一组(dq `block_kv`/`waves_per_eu`、dkdv `fold_lse`)——**差 1.4%,方向可能都是错的**。
   ★修法:让 builder 的**默认值本身**等于部署值(本例 dkdv `fold_lse: False → None`=hw 路径自动开),
   而不是靠每个 bench 记得传参。★改 builder 签名时,grep 所有直调 builder 的 bench 一并更新。
   ★口径修正会让分数跳变,**必须和真实加速分开报告**,否则历史分数不可比。
10. **★正确性门要跨形状 grid,不能单形状**。本轮两次出现"更快但算错"的变体(快 2.53% / 0.31%),
    分别只在 `q_split=4,BLOCK_KV=128` 和 `BLOCK_KV=64` 暴露,主形状一律 PASS。
    ★门至少覆盖 `q_split∈{3,4} × BLOCK_KV∈{64,128}` 且**必须查 det(bitwise)**,det=False 往往比 SNR 先炸。
    ★历史上同类事故:`block_kv∈{80,112,128,160}` 静默算错 dQ(8-14dB)却给出"854 TF 达标"的假数据。
11. **判内存类改动看 `SQ_WAIT_ANY`,不看 TCC hit%**。hit 率大涨可以值 0 wall(本例 86.5→94.8% 而 solo-vs-pipeline 差距没变)。
12. **减法探针给的是"删掉这项省多少",不是"藏起来能赚多少"**(本卡早有此训,本轮再次坐实):
    dq 的 6.9% DMA 探针里,吞吐那半只值约 0.2%、drain 那半约 0.7%。★先分清成本是**吞吐**还是**延迟**:
    延迟才能靠预取/双缓冲藏,吞吐只能靠减字节或提高复用。
13. **一次只改一件事 + 交织 A/B + 强制门,才敢碰"高回报但危险"的改动**。本轮最大的两个收益
    (锚点减量、XCD remap)都属于手工优化时不敢碰的类别 —— 因为单形状测不出错。
    ★配套:每轮报告必须写 `tried:` 全表(含判负项)与 `kb_check:`(KB 说 X / 实测 Y),否则下一轮会重复试。

## 通用教训(可迁移到其它 dense attention bwd)
1. **先跑 `SQ_LDS_IDX_ACTIVE:MFMA_BUSY` 富余测**判 latency vs port-bound;富余大 → 别碰 swizzle/prefetch/bank(纯计数改善零性能)。
2. **减 MFMA 是唯一 sized 杠杆**:审计有无可丢的第二 GEMM / 全局修正项(drop-B-GEMM),再考虑核间 recompute 共享(但注意确定性陷阱 + register 墙 + workspace 流量常吃光收益)。
3. **delta/odo 类全串行辅助核优先融掉**(随 Sq 放大);融合必对齐 O/dO dtype。
4. **occ3 对 latency-bound 是硬墙**:VGPR 降不到阈值(hd64 ~≤168)就别指望;拆核/fp8 降 VGPR 的代价(双读/破 SNR)通常 > 占用率收益。
5. **exp2 是 hd64 bwd 的 operand-bubble 核心**:软流水藏(head 轴)有肉;缩短 exp2(降阶 minimax)是最后的确定性杠杆但量级小。hw v_exp_f32(SNR52.6)比 fast Schraudolph(35dB)慢 ~5%(quarter-rate 发射 + 1-op latency);若门要 SNR≥45 且允许 poly,poly-exp2 可回收大部分,但 full-rate poly 天花板≈fast、超不了。
6. **★occ-2 latency-bound bwd 上,warp-spec/8-wave/dual-wave 常是陷阱——先量 baseline 是否已双-WG 共驻**:occ-2 = 每 SIMD 常驻 2 个**独立** WG,一个卡 barrier 时另一个照发 MFMA → **baseline 已免费享无屏障跨-WG dual-wave overlap**。8-wave 单-WG(合并两 WG)= 把这免费 overlap 换成带 `gpu.barrier` 屏障税的组内 overlap,MfmaUtil 反跌、天花板 < baseline(dkdv 实测 1029<1116)。→ 想显式并行 GEMM↔softmax 前,先确认 baseline occ,occ-2 双-WG 就别做(fwd 常 occ-1 无跨-WG overlap 才是 dual-wave 正场)。真正 register-neutral 的净胜是**冷 load 寄存器预取**(下一 head lse/delta 塞进本 head GEMM2 MFMA shadow,dt==DT-1 发射,+1.81% dkdv hw-exp),不是重构 wave 结构。
7. **VGPR 读 ISA 别读 rocprof**:rocprofv3 --pmc/kernel-trace 的 VGPR_Count 对 256-VGPR 内核**误报 128**(methodology/03);ISA `.vgpr_count` 才权威。dkdv 真值 256(占满 occ-2),据 128 判寄存器余量会全错。

### ★★ 2026-08-04 融合 bwd(dq 折进 dkdv,单趟 5 GEMM)r08:三条可迁移结论
基线 850 → 855~860 conv TF(B=3 S=8192 Hq64/Hkv8 D64 full-causal),同进程交错 A/B ≥11 trials × ≥2 session 同号。
1. **✅ 上面第 6 条的"冷 load 预取"要延伸到 q-block 边界**:head 轴预取只覆盖 h→h+1,**每个 q-block 的最后一个
   head-step 没有下一个 head 可取**,于是每 8 个 head-step 就有一个从裸 HBM 延迟开始。让它改取**下一个 q-block 的
   head 0**(Q/dO + 该 group 的 lse/-delta),值沿 q-loop 的 iter_args 传 ⇒ +0.29%/+0.55%(2 session);
   与"GEMM1 改 ks-outer 发射(同 k-step 的四个 kv tile 连发,相邻 MFMA 写不同 accumulator)"叠加 = **15/15 +0.57%**。
   两者互补:ks-outer 只在操作数已经到位时有用,而跨 q-block 预取正是保证它到位的那一半。单 WG 独占 CU
   (LDS 118,784 B)时没有别的 WG 覆盖这个 prologue,所以这条对"一 CU 一 WG"的 body 普遍成立。
2. **★对角(masked)q-block 的钱要按时间量,不能按 visit 数估**:causal KV-outer 里每个 WG 恰好一个对角 q-block
   = 6.06% 的 visit,直觉给 1.7~3.5%;**让所有 wave 都跳过它(结果算错,纯诊断)实测只有 +0.37%**。
   真做(只让 kv 行全在因果边之上的 wave 跳,输出**逐位不变**)= +0.22/+0.24/+0.33%(3 session)。
   ⇒ 在 makespan 受限的 kv-outer 里,visit 份额 ≠ 时间份额,差了 5 倍;先跑"全跳"探针再决定要不要实现。
   代价:scf.if 要把 16 个 accumulator 当 phi 穿过去,VGPR 249→256、8 dword spill、s_nop 198→370。
3. **★★ 减法探针给融合体定的账**:去掉**所有** q-block 的 GEMM1+GEMM2(S/dP/exp/dS/pack/dV/dK)= 4.845→3.247 ms
   ⇒ 这四个 GEMM 只值 1.60 ms,而它们的 MFMA 管道时间纸面算出来是 1.55 ms(2.08e8 条 ×16 拍 /1024 SIMD /2.1 GHz)
   ⇒ **主 GEMM 已经跑在 97% 的 MFMA 管道效率上,那里没有钱**。对照第五个 GEMM(dQ):1.18 ms 里 MFMA 只有 0.39 ms。
   ⇒ 融合体剩下的钱**全在 dQ split-K 的字节**(nb=Skv/BLOCK_KV=32 ⇒ 写 3.32 GB + dqred 读 3.32 GB)。
   给它定价:让 dqred 只读 16 个 band(dQ 算错,纯诊断)= **+6.0%(0/11)**,写侧同量级 ⇒ band-pair 折叠 ≈ +11%。
   **拦路的是一条与切分方式无关的算术**:一个 WG 的 dK/dV accumulator = (WG 的 kv 行数 × D × 2)/512 线程 dword,
   256 行 = 64 dword、512 行 = 128 dword,**怎么在 8 个 wave 之间分(含 D-split)都不改这个总数**——D-split 只是
   把同样的 dword 换个 wave 放,而让两个 WG 各拿一半 D 就要各自重算一遍 S/dP(= 再花 1.6 ms)。
   ⇒ 想拿这 11% 必须先在别处腾出 64 dword(已知报价:K packs 出寄存器 = 32 dword / −2.0%)。

### ★★★ 2026-08-10 gpt-oss bwd D128 r3:把上面那笔账在 D128 上重测,结论变了三处
形状 Hq64/Hkv8/**D128**、Sq=Skv=8192、B=2、THD、full-causal、BLOCK_KV=128(nb=64)。融合体 6.78 ms。
1. **★★★ 减法探针要一次做完三层,否则会把 DCE 当成收益**:`skip 分块存储` 这一个探针同时删掉了
   GEMM3 的 MFMA——`_g3` 累加器没人读,LLVM 把整条 GEMM3 链 DCE 了。三层拆开量才是真账:
   6.78(基线)→ **5.70**(存储指令照发、地址落到一个越界 SRD ⇒ 只去掉访存)→ **4.91**(存储+GEMM3 一起没了)
   → **4.18**(再去掉 dqred)。⇒ 分块写的 **访存 1.06 ms**、**GEMM3+发射 0.79 ms**、**reduce 0.63 ms**。
   只跑中间那一档会把 1.82 ms 全记到"字节"头上,高估 72%。
2. **★★ D128 融合体的计算相已经和 D64 打平,差距 100% 在 dQ split-K 工作区**:同法量 D64,
   store 0.554 / reduce 0.233 / 其余 2.12 ms。**其余 ×2 = 4.25 ms,D128 是 4.18 ms(还快 1.5%)**;
   而 store 1.82 vs 1.11、reduce 0.63 vs 0.47 ⇒ 0.88 ms 的超额恰好等于实测 0.96 ms 的整体差距。
   ⇒ 在这类"D 翻倍但寄存器封顶 BLOCK_KV"的融合 bwd 上,**别去抠 GEMM/LDS/occ,先量 split-K 字节**。
3. **★ 字节的边际单价远低于 roofline 价**:把 64 个 band 的分块写全部别名到同一个 slab(dQ 算错,纯诊断)
   省掉 ~8.5 GB 的 DRAM 写只值 **0.60 ms ⇒ 0.071 ms/GB**,而 roofline 价是 0.161 ms/GB。原因是这些写
   已经和计算重叠了。**r1 用 0.161 ms/GB 给所有字节杠杆定的价,在流水化之后系统性高估 2.3×**——
   定价前先用"别名/丢弃"探针量一次真实边际价,别拿 roofline 直接乘。
4. **✅ 落地的杠杆:band 交错(band-interleave)**。工作区从 `[band][B][Sq][Hq][D]`(每 band 一个
   268 MB slab)改成 `[band/8][B][Sq][Hq][band%8][D]`,即 8 个相邻 band 在**行内**交错。降序 q 走法
   (QDESC)本来就把并发 band 压在同几个 q block 上,所以同一时刻 64 个 band 在写**同一批 q 行**:
   slab 布局下这 64 笔写分散在 268 MB 间距上(一个 q 行 64 个 DRAM page),交错后落进一段连续
   `ILV*D`,并发 band 一起把整页填满。实测 ILV=1/2/4/8/16 → 813.2/811.5/820.1/818.6/819.7 conv TF,
   **4 以上就平了 ⇒ 机理是 DRAM page 局部性,不是 cache line 共享**。输出**逐位不变**(每个 band 仍
   独占自己的元素,无 atomic),body 寄存器 460 不变(`band%ILV` 是 wave-uniform,进 SGPR)。
   ⚠ **上限是 buffer descriptor 的 32-bit num_records**:reduce 侧的 slice = `B*Sq*Hq*D*2*ILV`,
   ILV=16 时 4.29 GB 越界。**ILV=64 的探针测出 +18% 是假的**——slice 回绕成 0,所有分块写被硬件
   丢弃。凡是改 SRD 覆盖范围的实验,先算 num_records 会不会溢出,再信那个数。
5. **❌ 判负(都别再试)**:(a) 分块存储的 cache policy 全谱——nt 801.0 / sc1 799.6 / nt|sc1 811.4 /
   sc0 816.2,都不如默认 cached 817.0;(b) 把工作区改成 **head-major** `[band][B][Hq][Sq][D]`(想让一个
   head-step 写 16 KB 连续)= **7.17 ms,−5.8%**,反而大输——q-major 下 8 个 kv-head 的 WG 协同把同一批
   q 行填满,比单 WG 自己连续更重要;(c) dqred 速率旋钮在 alloc_body=460 上重扫一遍,现役
   (block512, vec8, uc1, rpw2)=814.4 就是峰值(block256 810.0 / block1024 784.4 / vec4 775.3 /
   uc2 769.8 / rpw4 813.6)⇒ P1 的后续项就此关闭。
6. **★ band-pair 折叠在 D128 的真实报价与真实拦路点(报价已按 2026-08-11 实测改写)**:把 partial 减半 =
   写侧 ~0.9 ms + **实测**读侧 0.34 ms(让 dqred 按 2 倍 band 宽度读)= **~+18%**,是本形状唯一的大钱。
   拦路的是 §216 那条算术,但**那条算术只覆盖代价的一半,过去三轮拿它当全价是错的**:
   - **旧报价(不要再用)**:"accumulator = `BLOCK_KV*D*2/256` = 128 dword ⇒ pair 要 +128 dword,而 body
     只剩 52"。
   - **正确报价 = 两项,分属两个不同的堆**:(i) **累加器 +128 dw —— 这一半可以进 AGPR**(见
     pitfalls/01 §「AccVGPR 不与 arch 竞争分配」);(ii) **与之同量级的一份算子/地址类状态,只能占
     arch**(两条 half 的 kv 行号、两条 dS staging 链、两条 dK/dV store 地址链;见 pitfalls/01
     §「LDS 地址逻辑也吃占用」——LDS 寻址逻辑增长 arch_vgpr,一个 AGPR 都进不去)。occ=1 时
     `V cap=512 / A cap=256` 是**两个独立堆**(pitfalls/01 §「4-wave (occ=1) 的 VGPR headroom」),
     所以判"装不装得下"必须**分堆算**,不能只算合并总量。
   - **实测(gpt-oss D128 fused, Skv=8192, 4 wave)**:供体地板 `k_reg=0,q_pref=0,g3_kreg=0` = v=336
     **agpr=80 arch=256(已满)** spill=0 —— 捐了 119 dw,足够放下 (i) 的 128 dw(80+128=208≤256),
     但 (ii) 无处可去。于是 `bkv=256 + KV_HALVES=2`(dK/dV 跨 256 行、S/dP/dS 仍 128 宽)= v=512
     agpr=256 arch=256 **spill=743**,timed 388.7 TF vs 808.6。
   - **2026-08-11 追加两个数,把这条彻底钉死**:(a) 把每 pass 缩到 64 行(`KV_HALVES=4`,NT 2→1,
     transient 减半)只把 spill 从 743 降到 **655** —— 说明溢出的不是 pass 宽度那部分;(b) 在
     bkv=256 上**所有供体完全无效**:`k_reg=0`、`g3_kreg=0`、两者一起,spill 逐字都是 655。
     ⇒ **一旦累加器吃满 A 堆(256)且 arch 也满(256),供体腾出的那类寄存器不是溢出的那类**,再加供体
     只是在同一个堆里挪。LDS 反而不是拦路(halves=4 时 135168 < 163840)。
   - **2026-08-13 修正上一条的归因,并给这条路收尾**。上面两个"供体无效"的读数有个混淆:`k_reg`/
     `g3_kreg` 在 `_pair` 配置里**本来就是关的**,所以 spill 逐字不动只说明"没改到东西"。换三个
     配置里真正开着的供体重测(`g3d=2→1`、`g2d=2→1`、`exp_iglp=0`),配 `block_q=32`(它把
     transient/dS/dQ patch 全缩一半,是唯一能让 bkv=256 **spill 归零**的旋钮):
     `.vgpr_count` 逐字 **510 / 512 / 510,spill 全 0** —— 真机理是**分配器把整个文件占满**
     (A 堆 256 被累加器吃满、arch 堆随后也满),不是"腾错了类"。**结论比原来更硬:任何大小的供体
     都推不动两 band 的 body,一次 ISA 屏就能封掉整族,不必再做供体扫。**
   - **共驻线才是真正的判据(2026-08-13)**:两 band body 的地板 510 > 该 kernel 对能存在的**最高**
     共驻线 **488**(dqred 最瘦形态 = 4-wave WG × 24 dw/SIMD;部署的 8-wave WG 是 464)⇒ 折叠**必然**
     驱逐共驻的 reduce。驱逐单价用 full−nored 单独量出来 = **0.46 ms**,而折叠全部收益只有 0.58
     (读侧 0.38 + 写侧 0.20)。附带一笔:两 band body 还多 **54% `ds_read_tr16`**(1328→2048,两个
     kv half 各自重读 q 侧算子),唯一能修它的 `kv_halves=1`(dstr 1536)spill **577**。
   ⇒ **三条形态全封**:寄存器折叠(共驻线)、重算(2026-08-13 campaign r4 实测 −0.97 ms)、
   LDS 承接累加器(bkv=256 单 dV 就 128 KB,而 K tile+Q/dO+dS 已占 96/160 KB)。**别再试第四次**;
   更根本的原因见下一条的 D/A 定律 —— 这不是这个核的实现问题。

### ★★★ 2026-08-13 **split-K partial 字节是寄存器文件的函数,不是循环序的函数(D/A 定律)**
attention bwd 里"哪个梯度要走 split-K partial"看起来是拆法选择,实际两种拆法定价完全一样:
* **kv-outer**(WG 持 dK/dV,dQ 出 partial):A = `BLOCK_KV*D*2` 个累加 float ⇒ partial 字节
  = `(Skv/BLOCK_KV)*|dQ|/2` = **B·Sq·Skv·Hq·D²/A**。
* **q-outer**(WG 持 dQ,dK/dV 出 partial):A = `BLOCK_M*G*D`(G = 该 WG 覆盖的 GQA sharer 数)⇒
  partial 字节 = `(Sq/BLOCK_M)*(Hq/G)*(Skv/2)*D*2` = **同一个 B·Sq·Skv·Hq·D²/A**;而且 dK/dV partial
  通常必须 fp32 ⇒ 还要 ×2。**所以 kv-outer 是较优的那一侧,不是随手选的。**

⇒ **partial 字节/flop = D/A**:固定寄存器文件下**线性于 head dim**。gpt-oss fused bwd 的 D128 恰好
是 D64 的 2×(D64 能开 bkv=256、D128 只能开 128,因为 A 一样),**这就是它 D128:D64 分数差的全部**。
可迁移的判断法:遇到"partial/workspace 字节太多"的 bwd 核,先算 D/A,再决定是不是值得开一场
campaign —— 能动它的只有 (a) 更大的 A(硬件的 512 dword 池),(b) 累加器换一层存储(LDS 通常差一个
数量级,且要和 K/V tile、staging 抢同一 160 KB),(c) 重算(边际 MFMA 在 occ=1 上是 1.18× 纸面,
基本必负)。**换个 band 分组方式 / 换个循环序都在同一条 D/A 上,不用试。**

### ★★★ 2026-08-11 gpt-oss D128 fused bwd:**vmcnt 是 load/store 共享且按序的 ⇒ 预取的发射点相对重存储的位置值 +3.9%**
gpt-oss D128 fused(Hq64/Hkv8、Sq=Skv=8192、B=2、full-causal、bkv=128、4 wave/occ=1),score =
convTF(D128 fused)/convTF(D64 fused):**0.8754 → 0.9094**,convtf_d128 804 → 836.4,SBHD 同步 +2%,
门(双 layout SNR≥47 + byte-det)全绿,D64 对照臂 919.4→919.7 平坦。**改动只有一行:把下一个 head 的
Q/dO 预取发射点从"GEMM3 之后"挪到"GEMM3 之前"。**
1. **机理(可迁移到任何 occ=1、既有重存储又有预取的核)**:gfx950 只有**一个 vmcnt**,load 和 store 共用
   且**按发射序退休**。核体每 head-step 顶部要把上一步预取到 VGPR 的 Q/dO 写进 LDS,编译器为 8 条
   `ds_write_b128` 排了递减的 `vmcnt(7..0)`。预取发在 GEMM3 的 4 条 dQ partial store **之后**时,
   "等第 1 条 load 回来"在按序语义下**必须先让那 4 条 store 退休** —— 于是每个 head-step 以"把 16 KB
   partial 排空到 L2"开场,末尾那条还是整条 `vmcnt(0)`。把预取挪到 store **之前**,同样的等待变成
   `vmcnt(11..4)`,store 留在飞行中,唯一会等它的是一整个 head-step 之后对其源寄存器的 WAR。
2. **ISA 判据(比 wall 更早、更便宜)**:整核 `s_waitcnt vmcnt(0)` **16 条 → 1 条**;等待深度分布从
   (7..0) 抬到 (11..4);`s_barrier` 35、`lgkmcnt(0)` 104、`v_mfma` 2560 **逐字不变** —— 说明改的不是
   指令数,是"等待时还有哪些访存在飞"。寄存器代价 v 455→461 agpr 199→205 spill=0、LDS 不变。
   ⇒ **判这类改动看 `vmcnt(0)` 的条数,不要看 `s_waitcnt` 总条数。**
3. **★对照臂把机理钉死**:同一条预取,发射点只挪到 store **后面几条指令**(point 1)= **785.0 TF
   (−2.4%)**,发在前面 = 831.2(+3.4%)。**跨过 store 边界的 5.7% 摆幅**,寄存器读数几乎一样
   (461 vs 463)⇒ 不是寄存器压力,就是 vmcnt 顺序。
4. **❌ 同轮判负的三条(都别再试)**:(a) 管理者点名的 `dma_grp=2`(DMA 路线分组 Q/dO 全局读,目标把
   `s_waitcnt vmcnt` 条数从 104 压向 split pair 的 53)—— 实测**压到 18**、寄存器还更好(v 432 spill 0),
   却 **−5.3%**(0.8291);`dma_grp=4` 直接 LDS 200704 > 163840 build fail。⇒ **等待的"条数"不是成本,
   "深度"才是**:DMA 路线把 graded 等待换成 group 边界上的整条 `vmcnt(0)`。(b) Q/dO 双 LDS slot 把
   head-step 的两个 barrier 减成一个(`s_barrier` 35→19,LDS 102400→135168 仍合法)= **−6.0%**,
   v 455→481 —— 拿掉 barrier 同时拿掉了它当"调度墙"的作用,分配器立刻把 staged tile 拖得更长。
   这是本 body 上**第三个**"用少一次 rendezvous 换调度自由度"的结构判负(另两个是 QDO_TAIL、G3_DEFER
   的反向)。(c) GEMM3 patch 形状 `g3_qt=1/4` = 765.3/752.8、`g3_krt=2` = 791.1、`g2_half=0` = 769.3。
5. **⚠ 判决会随这一行迁移,复扫是必须的**:同一批旋钮在改前/改后读数不同 —— `g3_defer=0` 改前 802.8
   (≈中性)、改后 **789.5(−5.6%)**;`g3d=8` 两次都≈噪声;`g2d=3` 两次都是崖(−5.6%/−6.6%)。
   ⇒ 落地这类"改变访存在飞结构"的改动后,**至少把 defer/ring 深度这一族重扫一遍**。
6. **⚠ 测量纪律再次兑现(别把分母漂移当收益)**:`g1_ks_outer=0` 两次都读到 ratio +0.7~1.0%,但
   **convtf_d128 两次都平坦**(804.2 / 836.9),涨的是 D64 分母掉了(919→913,两次一致)。同进程双臂
   互相有功耗/DVFS 耦合 ⇒ **判 D128 改动只看 convtf_d128,ratio 只在分母平坦时才可信**。

### ★★★ 2026-07-29 gpt-oss bwd r11:odo 融进 dq 兑现(+1.1%),并首次量到 occ-3 的**真实单价**
gpt-oss 形状(Hq64/Hkv8/D64、Sq=Skv=8192、B=2、SBHD、full-causal),基线 785.6 → 本轮 842.4 conv TF。
1. **✅ 上面第 3 条("辅助核优先融掉")在本形状上兑现,给出量级**:dq 是 q-outer 且 GEMM1b 的 B-operand
   就是 dO,所以它能**为自己拥有的 (q_tile,q_head) 行**多读一份 O、复用 round-9 那个 `ds_bpermute`
   xor 蝶形把 `DELTA=-rowsum_d(O·dO)` 直接算出来写回全局给 dkdv ⇒ **odo kernel launch 整个删掉**
   (43.6 µs kernel + 一次 launch),实测 **+1.13%**(ABBA 2/2),dq 自身 1362.7→1344.7 µs(多读 O 只
   花 ~6 µs,因为 dq 已是 53% MFMA-busy、带宽有 8× 余量),输出**逐位相同**(SNR 50.26/50.36/50.77)。
   ⚠ 工程坑:蝶形的 xor 掩码必须是**共享同一 q 行的 lane 间距**(M_TILE=16 ⇒ 掩码 16/32)。误用
   64/128 时每个 lane 与自己配对、值翻倍,dQ SNR 直接掉到 **1.45 dB** —— 这类错只有 SNR 门能抓。
   ⚠ 若 bench harness 的调用序固定为 odo→dq→dkdv 且不许改,可让 odo launcher 退化成"只把 O 的
   引用转交给下一次 dq launch"的 stub(dq 的 K16 空参位接 O),harness 一行不动就完成删核。
2. **★ per-wave 因果分类**(dq 对角带):BLOCK_M=192/BLOCK_KV=64 时对角带 = 3 tile × 4 wave = 12 块,
   其中 3 块整块被 mask(P≡0,整趟 MFMA/exp2/pack 白做)、3 块完全在安全侧(不需要 mask 链)。
   两个判据都是 wave-uniform ⇒ 降成 `s_cbranch_execz`,放在协作 DMA/barrier **之后**(每个 wave 仍要参与
   staging)。实测 **+0.34%**(ABBA 2/2),VGPR 236 不变、spill 0、逐位相同。
   ❌ **同一手法搬到 dkdv 判负**:dkdv 的 8-head 循环是全展开的,分支要把 32 个 dV/dK 累加器 + 8 个
   lse/delta 预取(共 40 个 MLIR 值)做成 phi ⇒ `.vgpr_count` 243→256、`vgpr_spill` 0→**5**,
   实测 **−0.17%**(ABBA 2/2)。省下的 0.77% 工作量被 spill 吃掉。**先看 spill,再想省工作量。**
3. **★★★ occ-3 的单价第一次被量到(此前本卡只说"降不到 168 就别指望",没说到了值多少)**:
   dq 在 BLOCK_M=192 时 236 VGPR(occ-2 满);把 tile 缩到 **128(QT=2)自然落到 190 VGPR / spill 0**,
   再 `waves_per_eu=3` 压到 **168 / spill 19** ⇒ 实测 **815.7 → 836.2 TF(+2.5%)**。
   但 BLOCK_M=128 本身 **−3.0%**(每个 kv tile 要被 1.5 倍数量的 WG 各流一遍),净 −0.8% ⇒ 回滚。
   ⇒ 结论要改写成:**occ-3 在本族 bwd 上是一条真值 ~+2.5% 的杠杆,门槛就是 168 VGPR;
   代价必须来自"不加流量"的地方**(dq 在 192 需要再挤 68 个寄存器:GEMM1 transient 约 32、
   dQ 累加器 48 —— LDS 只用了 16 KB/160 KB,是唯一有余量的地方)。
   ⚠ 配合 fwd 篇那条:≥95% TBP 下占用率收益要被时钟回吐吃掉一半以上,**每个 occ 候选都要同时报 sclk**。
4. **本形状的指令账已到底(两核都 ~90-100% issue-bound,每条都是算法下限)**:dkdv 每 trip
   512 MFMA(= 精确算法 MAC 数,零浪费)+ 256 exp + 256 v_mul(dS)+ 256 cvt_pk;dq 每 trip
   72 MFMA + 48 exp + 24 pk_mul + 24 cvt_pk。LSE 折进 GEMM1a 累加器、−delta 折进 GEMM1b 累加器
   之后,**exp2 前后一条 VALU 都不剩**。⇒ 想再省只能动 MFMA 数(= 因果几何)或换 atom;
   gfx950 无 packed-bf16 ALU、`v_pk_mul_f32` 按写出 dword 计费(8 cyc,不省)⇒ 逐元素段无损压缩到底。
5. **实现真实做 7 个 GEMM-equivalent 而 conv 口径只记 5**(S 与 dP 在 dq/dkdv 各算一遍 = 全部 MAC 的 29%)。
   这 29% 是**确定性 + 累加器寄存器常驻**的代价:让 kv-outer 核也产 dQ 需要 float atomic(破 det)或
   每 trip 128 KB 的 dQ partial store(实测量级 8.7 TB,远超收益)⇒ **别再重新推导这条**,
   两核结构就是本 gate 下的正解。

### ★★ 2026-08-09 GB300 trace 对标(首个 gpt-oss FUSED bwd 外部硬件基准)+ SWA campaign
> ⚠**别与本卡上半的 H100 验收混淆**:那是 **Meta DENSE bwd**(B1 Hq128/Hkv16 square-causal 对 H100 FA-v3,
> "SWA 2.12-3.41× 超标"=对 H100 更慢的 SWA 的比值)。**本条是 gpt-oss FUSED dq+dkdv bwd**(B3 **Hq64/Hkv8** D64
> SBHD full-causal,§198 起那条线)对 **NVIDIA GB300**(Blackwell,cudnn sm100 flash bprop/fprop)。两个 regime、
> 两个参考硬件,数字不可互引。conv-FLOP 口径同本卡 line 13(`10·Hq·S²·D·frac`,B 因子外提)。
- **GB300 权威读数**(trace `trace/gb300/rank-0.json`,gpt-oss-20b B=3 S=8192,各 24 calls):
  full-causal bwd mean **4252.5µs = 969.7 conv-TF/s**;SWA(128) bwd mean **445.7µs = 289 eff-TF/s**;
  full fwd 1310.6µs / SWA fwd 284.0µs。**SWA wall dividend = 9.54×**(bwd)、4.62×(fwd);
  bwd/fwd wall = 2.95×(**不是** 2.5 的 FLOP 比——含 delta/reduce 辅助核 + 核效率差,别当 FLOP 比用)。
- **我们(964b290a,`_bench` 口径,013 实测)**:full-causal median **~944 conv-TF**(区间 925–963;phase-3
  best-config 重测 964.64)⇒ 对 GB300 **落后 ~2.7–3.7%**(≈115–160µs/层)。SWA(128) baseline **215.6 eff-TF**。
- **★SWA 曾是我们的结构短板,现已反超**:NV 靠 window 天然省 kv-band 拿 ~9.5× 的**wall dividend**,而我们旧核
  `MASK_SKIP = FUSE_DQ and window_left < 0` 在 SWA(window_left≥0)下**关掉** band-trim → 每 kv-band 把全 causal
  q-loop 算完再全掩=纯浪费。折算 wall:我们**旧 SWA wall≈596µs**(dividend 仅 **7.3×**,低于 GB300 9.54×);
  SWA campaign(dir `20260808_153438`,base 964b290a,进行中)best **298.1 eff-TF**(wall≈432µs)⇒ dividend 升到
  **10.1×,已反超 GB300**、SWA eff-TF **~+3% 快过 GB300 的 289**。杠杆=window-aware q-loop 上界,全裹
  `const_expr(window_left>=0)` 保 full-causal 字节不变(见 [[project_gptoss_swa_bwd_campaign]] / [[project_fwd_swa_and_latency_floor]])。
- **★与旧记 [[project_gptoss_sbhd_native]] 不矛盾(时点+HW 不同,非回归)**:那条是 **pre-fusion SBHD** 核对 **B200**
  (766 vs 812 conv-TF,−5.7%);本条是 **post-fusion** 核对 **GB300**(944 vs 970)。fusion(786→879→964)与这场
  SWA(216→298)是两段独立提升。B200 亦见同款 ~9× SWA dividend 模式(不同 B/pre-fusion 核,**绝对 µs 不可横比**)。
- **★下一步(full-causal 追 GB300 那 ~3%)**:走 §3(r08 减法探针)的 **dQ split-K band-pair 折叠**(报价 ~+11%,
  但须先在别处腾 64 dword,如 K packs 出寄存器 −2.0%)+ §3(r11)的 **occ-3**(报价 +2.5%,门槛 168 VGPR,代价须来自
  "不加流量"处)。★**别再走 dual-wave/8-wave/warp-spec/shared-GEMM1**——全在本卡 DEAD 段判死。禁天花板结论:
  这 3% 有 sized 报价(band-pair + occ-3),不是"到顶"。

### ❌❌ 2026-07-29 gpt-oss bwd r14:fwd 侧两条"零源码风险"的赢**都不迁移到 bwd**(同基线 843.0,3 臂/1 门)
本轮零收益,但关掉了两个大族,都是**同一把尺子、同一进程、每臂 3 次**测的(当天噪声地板仅 0.1%)。
1. ❌❌ **编译器调度策略族(fwd r15 记 +0.6~0.7%)在 bwd 上一致为负,别再试**:
   `amdgpu-sched-strategy="max-memory-clause"` 在 dkdv = **−0.59%**(837.5,3/3)、在 dq = **−0.13%**;
   `max-ilp` 在 dkdv = **−0.85%**(835.3,3/3)。ISA 方向"正确"却没用:dkdv 的 `s_waitcnt` −40 条、
   `s_nop` +16 条(净 −24 发射位)而 wall 变差 ⇒ **指令数减少不等于 wall 改善**,策略换的是发射优先级,
   在这个 pipe-additive 的核里会把原本被覆盖的延迟暴露出来。
   机理差异:fwd 的 K/V 分块读散布在各计算簇里(memory-clause 有东西可聚簇),bwd 的 LDS 读已经被
   手写 `sched_mfma/sched_dsrd` 1:1 交错定死,策略只能把手工结构打散。⇒ **手工调度已到位的核,
   换编译器策略只有下行空间**。
2. ❌❌ **GQA sharer 合并 / 8-wave CTA(fwd r18 记 +2.04%)在 bwd 上是 −2.7%,且"省请求=买时钟"不成立**:
   dq `Q_HEADS_PER_WG=2`(8-wave、`WAVE_ROW_GROUPS=4`、`BLOCK_M` 仍 192、grid ÷2、两个 sharer 共用
   K/V 的 LDS tile)—— ISA 全绿(vgpr 252→**250**、spill 0、LDS 32 KB 不变,热循环 `buffer_load`
   **8→4 条/wave-trip**,即 K/V 的 L1→L2 请求数确实腰斩),输出**逐位相同**(SNR 三元组不动)
   ⇒ 这是一次**纯结构**测量:**819.6 vs 843.0 = −2.7%(3/3)**,即 dq 自身 −6.5%。
   连同 r10 的 8-wave dkdv(BLOCK_KV=256,−4.3%),**同一失效模式两次独立复现**:8 wave × occ-2
   ⇒ **1 WG/CU**,barrier + DMA drain 再没有共驻 WG 来盖,这个损失比省下的请求值钱。
   ⇒ ★ **修正 fwd r18 那条"能耗在 TCP→TCC 请求路径上、省请求=买时钟"的可迁移性**:它成立的前提是
   *合并后仍不掉 WG/CU*(fwd 是 4 waves/SIMD、LDS 只装 K/V 与 BLOCK_M 无关,合并后仍 ≥2 WG/CU)。
   bwd 两核都是 occ-2 VGPR-locked,任何加宽 CTA 的做法都会撞到 1 WG/CU 这道墙 ⇒
   **bwd 侧想省请求,只能在"不加 CTA 波数"的前提下做**(=加大 tile,而那被寄存器锁死)。
   ⇒ ★ **另加一条 20 秒的前置判据(2026-08 mx8tw 实测):先 PMC 量本核与对标核的
   `TCP_TCC_READ_REQ`,比值 < 1.1 就说明请求路径已经打平,这条杠杆没有可动面,别花轮次。**
   mx8tw 那场量到 73.76 M vs 72.58 M = **1.02×**(HBM 合计 1.90 vs 1.84 GB、功耗 1399.5 vs 1374 W
   也都持平),于是整条"省请求买时钟"的方向当场关掉,改攻同步指令密度。
3. ❌ **dkdv 的 2-tile 分组暂存(r13 在 dq 上 +0.2% 的那个机制)在当前 dkdv 上过不了 ISA 门**:
   dkdv 早就有这个机制(`_PB["dmastage"]`,stage base 是编译期常量、LDS 16 KB/槽,r10 建的),
   `dmastage=2` 现在 = vgpr 243→**256、spill 0→29、scratch 0→120 B**(r10 在旧 body 上是 0 spill / −0.13%)。
   r12 的 per-q-half 重构把 243/0-spill 换来的余量**已经被别的东西占满**,不再容得下第二个 LDS 槽的
   地址/预取活跃区间。⇒ 照 r9 规矩只花 14 s dump 直接判负,不花 bench。
   ⚠ 记账要连 body 版本一起记:同一个开关在 r10 是"中性",在 r12 之后是"spill 29"。
4. ★ **`hfence=1` 其实是 `sched_barrier(0)`(纯编译器栅栏、零硬件成本),不是 `s_barrier`**:所以 r10 记的
   「dmastage=2 + hfence=1 = −0.13%」已经是"硬件 barrier 减半 + 免费栅栏"的最优组合,而它仍然是 0。
   这与 r9 H5(把 dkdv 全部同步删光 = 仅 +1.12%)自洽:**dkdv 的同步池总量就只有 1%,减半的上界是 0.5%,
   不值得为它付任何寄存器**。
5. ★ **grid 层已无余量(离线 list-schedule 模拟,不花 GPU)**:dkdv 的派发顺序 `kv_tile` 升序**就是** LPT
   (因果下 kv_tile 越小 q 范围越长),模拟 imbalance 仅 +1.54%;`Q_SPLIT` 3→4 把 imbalance 降到 +0.77%,
   但每 WG 的 K/V prologue 正好吃掉这点(startup=4 head-block 时两者 makespan **都是 1080**),
   与 r7/r8 实测"3 与 4 平手、3 因少 25% workspace 流量胜出"完全一致 ⇒ **别再扫 Q_SPLIT**。

---

## ★★★ 2026-07-30 出货验收:tile 启发式的自变量选错了轴(矩形形状 −19%,方阵完全测不出)

`_blockkv_for(Sq)` 按 **Sq** 选 BLOCK_KV(`64 if Sq<=2048 else 128`),理由记的是"小 Sq 用 64 才能铺满 CU"。
但 **dkdv 是 KV-outer**,它的 grid 是 `B · Hkv · ⌈Skv/BLOCK_KV⌉ · q_split` —— **铺满 CU 的是 Skv**。
方阵上 Sq==Skv 两者一致所以永远测不出;到了 Meta 的矩形形状 `Sq=2048, Skv=16384` 就选反:
**15.03 → 12.49 ms(−19%)**,那是 20-config 里 bwd 唯一不达标的一格(1.37× → 1.65×)。改成 `_blockkv_for(Skv)` 即可,方阵零影响。

⇒ **通用规程:每个 tile/split 启发式,先写清楚"它调的那一维,grid 是按哪个轴铺的",自变量必须是那个轴。**
   dkdv(KV-outer)→ Skv;dq(Q-outer)→ Sq。**只用方阵验收会让这类错误永久隐形**,必须有矩形形状在验收表里。
   同理:`_dq_block_kv` 调的是 dq 的 kv **循环长度**(∝Skv)而不是它的 grid(∝Sq),该轴上实测 64 vs 96 在矩形形状上打平,暂不改。

## ★★ 2026-07-30 小形状是 **host-bound**,不是 kernel 慢(fwd/bwd 同一个根因)

`Hq=64/Sq=Skv=1024/B=4` wall 84 µs 怎么调都不动;**判据:S=128/256/512/1024 全是 ~80 µs —— 工作量差 64 倍
时间不变,就不可能是内核里的任何东西。**三层量下来 wall 84 / host enqueue 85 / GPU 内核 **50** µs ⇒ GPU 40% 时间在等 host。
根因是 FlyDSL 每次 launch 重解 JIT 签名。**修法 = `flyc.compile` 的 artifact 按标量签名缓存复用**(fwd 一处,
bwd 三处 odo/dq/dkdv 各付一遍):host **93→7 µs**,fwd 1024² 0.084→**0.045 ms**、方阵 S=2048 811→**929 TF**;
长序列本来就 GPU-bound,不变。⚠ 标量会被当 constexpr 烤进 artifact,**cache key 必须含每一个标量参数**。
完整判别配方见 pitfalls/02。

## fwd 篇(同族 hd64 dense flash **forward**,full-causal S=16384 / B=1 / Hq128 Hkv16 / D64)

> 2026-07-28~29 campaign 20260728_082734,chi2798 GPU6。基线 983.2 → **1127 TF(+14.6%)**。
> 收益来源与 bwd 同构:**全部来自 grid/派发层 + 数学层重写,kernel 体内微调合计 <0.5%**。

### ✅ 已部署的赢
1. **`_FMAX0` 固定参考 max = +13.5%**(954.5→1084):softmax 平移不变 ⇒ 用**零参考 max**(不是
   methodology/15 卡片写的 first-pair crossgrp-max,**省掉那次额外 QK**)。一次性删掉 per-kv-tile 的
   `reduce_max`(28 条 `v_max3_f32`/iter,含深 16 的归约链)、O 的 per-tile rescale(23 条 `v_pk_mul_f32`)、
   以及 `s−m`(60 条 `v_sub_f32`)。门控 `causal and window_left<0 and not splitk`,SWA/非因果走原路径。
   D=64 bf16 + Q 预缩放下 score 量级可控,SNR 51.26 dB,22 组形状全 PASS。
2. **LPT 降序 q_tile 派发 = +2.34%**:与 dq 同型。⚠**裸改会 vgpr 162→170 跨 3→2 悬崖**,
   必须同时拆掉 loop-carry 的 pack/unpack 往返把寄存器还回来(本卡"grid 层优先"的又一例)。
3. **`_FMAX0` 之后去掉已失效的 `_anchor_v_p` + 簇头 `s_nop` 7→3 = +0.31%**:固定 max 后 P 不再被
   rescale,那个锚点无序可钉,而它那条 32 宽 tied-operand 空 asm 要搬 16 个 dword(8 条 `v_mov`/iter + 2 寄存器)。
4. **★★把访存簇的 `s_waitcnt` 全排空下沉到消费它的计算簇尾 = +1.42%**(1116.5→1132.4,同会话交织
   A/B 各 4 次区间不重叠;s8192 1090→1103 同向)。软件流水是「访存簇发 `ds_read` → `s_waitcnt lgkmcnt(0)`
   全排空 → barrier → 计算簇」,那次全排空把 LDS 延迟**完全暴露在计算之外**。而该 LDS buffer 真正被
   DMA 覆写要到**两个 barrier 之后**,所以排空只需在**下一个**计算簇尾完成即可。下沉后编译器改用
   **增量 `s_waitcnt`**(ISA:主循环 `s_waitcnt` 4→16 条,PV 计算簇里各出现 **7 条增量等待**与 MFMA 交织),
   延迟藏进 MFMA。`.vgpr_count` 162 不变、spill 0、输出 SNR/det 与基线一致。
   ★**必须用 `not STAGGER` 门控**:stagger 靠给 group B 多发一个 `s_barrier` 错相,两组正好差一个簇 ——
   下沉吃掉的就是那一个簇的余量。未门控时 `_verify_swa_varlen.py` 的 **8-wave(bm=256)组 rel_l2
   从 0.0027 劣化到 0.0112 / 0.0294**(1 组 FAIL),而计分形状 SNR 51.26/det=True 毫无反应。
   门控后 22/22 且 8-wave 的 rel_l2 精确回到 0.0027/0.0028 —— 这个"劣化→门控→数值精确复原"
   是比 SNR 更灵敏的**竞争检测手法**,同步类改动都该这么验。
   ⇒ 通用形态:**任何 `waitcnt` 全排空,先问"它保护的 buffer 真正被覆写是在几个 barrier 之后",
     排空可以下沉到最后一个安全 barrier 之前**;这与 r8「减 barrier 数量 = 0%」不冲突 ——
     赢的不是 barrier,是**排空点的位置**。

### ❌ 别再试(fwd 侧实测,均在 1127 基线上同进程 A/B)
- ★★ **屏障前那条手写的 `s_waitcnt lgkmcnt(0)` 是纯调度税,删掉 = +0.54%**(2026-08-31,
  gpt-oss D=64 fwd,计分尺 6 对回文,均值 1.01398 vs 1.00851,5/6 对同号,两个 SNR 门逐位不变)。
  根因看 ISA:删掉之后**后端自己**在 `s_barrier` 前重新发了 `vmcnt(0)` + `lgkmcnt(0)`
  ——`SIInsertWaitcnts` 本来就给 barrier 建模。手写那条的唯一作用是**把整簇的 LDS 读全钉在
  会合点前面**;去掉后后端改发细粒度 `lgkmcnt(2)/(3)`,在每个真正的消费点等。
  ⇒ 通用判据:**只有当某条读的结果在 barrier 之前没有消费者时,手写 drain 才有意义**;
  否则它只是在削调度自由度。删之前用 ISA 确认后端补了 `lgkmcnt(0)`,别靠猜。
- ★★ **`s_barrier` 两侧手写的 `sched_barrier(0)` 也是调度税,删掉 = +0.96%**(2026-08-31,同一
  gpt-oss D=64 fwd。持续 min-of-30000:1.83025 vs 1.84785 ms;计分尺 2 对回文 1.01834 vs
  1.012795 = +0.55%;筛选尺 n=10 +0.61%)。围栏原来长这样:
  `sched_barrier(0); s_barrier(); sched_barrier(0)`,现在只剩 `s_barrier()`。
  ★ **正确性判据(可复用)**:按 `s_barrier` 把整核切成 region,对每个 region 数
  `ds_read / ds_write / buffer_load`。两臂**21 个 region 逐个相同** ⇒ 没有任何访存跨过会合点;
  `s_barrier` 本来就是后端对访存的硬调度边界,围栏挡住的只是**纯寄存器**的活。
  真正搬家的就是它:16 条 `v_exp_f32`(128 VALU 拍)从 barrier 后面挪到前面去盖等待,
  外加几条 MFMA。指纹其余逐项不变(240 条/vgpr 128/spill 4/LDS 34048/2 barrier/32+8 mfma)。
  ⚠ **两条围栏不可分**:只删前面的 = ISA **逐字节不变**(惰性);只删后面的 = **−0.13%**;
  两条一起才 +0.61~0.96%。别 bisect 一半就下结论。
  ⚠ **这是周期赢不是能量赢**:功耗钉在 1399~1400 W,sclk 反而 **1639→1623 MHz(−0.9%)**,
  真周期 −1.85%。coexec 变多 = 每拍更费电,DVFS 收回一点时钟。不同报 sclk 就会记错坐标系。
  ⚠ 同核里**被折叠**的簇间 `sched_barrier(0)`(`_mem_cluster_sync` / `_pv_cluster_sync`,
  不裹 barrier 的那些)方向相反:删 mem 的 −0.15%、mem+pv 一起 −0.35%,它们在干实事。
- ⚠ **「barrier 处的 vm drain」在本核只值 ≤+0.17%**(减法探针:`vmcnt(0)` → `vmcnt(N≥在飞条数)`,
  指令数不变,ISA 确认编码真的从 `vmcnt(0)` 变成 `vmcnt(2)`;n=10 回文 1.85462 vs 1.85783)。
  根因:DMA 在 region **顶端**发、region **尾部**才等,中间隔着一整个计算簇,延迟早被盖住。
  ⇒ 任何"让覆写 DMA 更早落地"的实现动手前先跑这个探针,它就是那族的天花板。
- ❌ **在 `max-memory-clause` 下从源码挪 DMA 的发射落点 = 源码层不可达**(这解释了本卡早先记的
  "region 内换落点整族已探尽")。把两条 K `buffer_load_dwordx4 … lds` 从计算簇头提到前一个
  访存簇头(WAR/RAW 都重新推过,合法):ISA 里 DMA **两臂都紧贴在 barrier 之后**,只差 1 条指令、
  waitcnt 剖面逐项相同,wall −0.13%。因为该策略会把 region 内所有 `buffer_load … lds`
  **hoist 到 region 顶端**,源码位置根本不过调度器。
- ❌ **四深 KV LDS 环把 barrier 2→1 = −2.2%**(同上 kernel。这条**不推翻** bwd 侧 r22 的 +0.34%,
  它是结构相关的)。减法探针先给了 **+0.97%**(删一个 barrier)与 +1.36%(再加 drain),
  但真实现要付:LDS 34048→**68096**、spill 4→9、+5 条指令、跑一个运行时半环偏移。
  拆解(`四深 + 保留两个 barrier`)= **深化本身 −1.65%**,barrier 只还回 1.30% ⇒ 净负。
  ★ 机制:`_dualwave_sync_barrier` 不只是数据护栏,**它就是 dual-wave 的乒乓**;会合率减半 =
  两个 wave group 交替访存/计算的粒度减半。
  ⚠ 还有一个正确性陷阱:**加深环只放松 WAR,不放松 RAW**。C3 写的 K 由下一轮 C0 **跨 wave** 读,
  中间必须有 barrier;把 barrier 留在簇内(C1 尾)的那一版 SNR 全绿却是真 race。
  唯一无 race 的摆法是把它放在**迭代边界**,让环的奇偶把「本轮所有读」和「本轮所有 DMA」分到两半。
- **`v_pk_add_f32` 打包 row-sum = −1.45%**。这条**推翻了本 campaign 早期的记载**:早期以为打包判负是因为
  "向量值的 +8 寄存器对齐税",实测用 `Vec` 的两元素切片写法 **vgpr 一个不涨(166→166)**、
  ISA 确认发出 46 条 pk_add / `v_add_f32` 86→0(净减 40 条)、SNR 51.26 det=True,**wall 仍掉 1.45%**。
  ⇒ 根因是硬件:**gfx950 的 `v_pk_add_f32` 对 f32 没有吞吐收益**(见 connection/common/05 速率表)。
- **`v_dot2c_f32_bf16` 做 row-sum = 算错**(SNR 23.7~24.1 dB,det=False,误差稳定 6.25%)。
  **这条推翻本 campaign r6 记的「换 intrinsic 后 51.61 dB / det=True」** —— 重测三种写法
  (intrinsic 单累加器 / 4 路轮转累加器 / 末尾补 32 拍 `s_nop`)全部同样错。根因:LLVM 选的
  **VOP2 `v_dot2c` 形式不遵守 `op_sel_hi`**,只把低半 lane 算两次(2×Σ偶数项 = 差 6.25%)。
  r6 那次 51.61 dB 是分子分母共用同一份错 P **互相约掉**的假象(它同时也喂了 O 的分母)。
  ★**2026-08-31 修正:这条只对 VOP2 的 `v_dot2c_` 成立。** 同一 kernel 的 fwd 上用 VOP3
  `v_dot2_f32_bf16`(tied-operand inline asm,`"=v,v,v,0"`)做全部 8 个 pack 的 row-sum,
  **算得对**(两个 SNR 门与部署逐位相同)、`vgpr 126 / spill 0 / row-sum MFMA 归零`,
  但 **wall −4.5%**(1.960 vs 1.871),根因与正确性无关:**tied-operand inline asm 对冒险识别器不透明**,
  热循环 `s_nop` 6→**30** 条、32 条 dot2 串成一条链,总条数 285 × 4 waves = 1140 槽
  打在已经降到 1024 拍的 MFMA 管线预算上 = **111% 发射受限**。
  ⇒ 想再试这族:**发 `llvm.amdgcn.fdot2.f32.bf16` intrinsic,别用 inline asm**;
  且先算发射槽账。另:VALU row-sum 的纸面模型(68 `v_add` + 2 `permlane`)**高估 2×**,
  实测 dot2 路径只要 32 条 + 1 permlane —— 而它**仍然输**,因为模型漏算了那 24 条 `s_nop`。
- **per-wave cluster 再平衡** = 0 ~ −3.1%:把 row-sum 搬进访存簇的 LDS 延迟窗口 −0.10%、
  搬进 PV/MFMA 簇 −3.08%。⇒ 见 methodology/03 新增铁律「≥2 waves/SIMD 时簇再平衡是吞吐不变量」。
- **跨-WG 相位错开**(`s_sleep` 按 `bid%3` 给共驻 WG 三种相位)= −0.06%:共驻 WG 本来就异相。
- **主循环因果掩码删除 = −6.1%**:那 96 条 `v_cmp`/`v_cndmask` 被包在**运行时恒不进入的 `if` 分支**里,
  **发射量为零**,删它 vgpr 一个不省,却让编译器把访存 cluster 重排得更差。
  ⇒ **读 ISA histogram 必须先扣掉 not-taken 分支块**(否则会把 27% 的"VALU 占比"当成可砍量)。
- `sched_group_barrier` 计数重标定五方向全扫(valu_cnt 5→3/5→8、MFMA 覆盖 6→8、
  交织粒度 {1M+NV}→{2M+2NV}、exp_cnt 3→2)= 全部 −0.2~−1.2%,现有标定已是局部最优。
- 关 `s_setprio` = −0.8%;提到 2 = 持平 ⇒ 保持 1。
- row-sum 改 4 累加器树形(依赖深度 32→8)= **零变化**,`_FMAX0` 前后各测一次都是零。
- **把 K/V 的 DMA 发射挪到 V 的 `ds_read` 之后**(想让 `buffer_load ... lds` 落进 LDS 延迟影子)
  = **−0.20%**(1158.4 vs 1161.2,区间不重叠)。ISA 确认顺序真的换了(16 条 tr-read → 7 条 DMA 发射 → `lgkmcnt(12)`)。
- **让 GQA sharer 共驻同一个 CU 以复用 vL1D** = **−0.57%**(s8192 −1.9%),**且访存计数器逐位不动**,见下条。

### ★★★ 馈送池的机制定案:`buffer_load ... lds` 在 vL1D 上**拿不到任何跨-WG 复用**(r13 PMC)
goal P2 留了两个候选机制:(a) vL1D/TA 请求队列满卡在**发射端**,(b) DMA 回填抢 LDS 写端口。r13 用 PMC 定案 **是 (a)**:
- `SQ_LDS_IDX_ACTIVE` 只占 wall **31.8%**、`SQ_LDS_BANK_CONFLICT` = **0**、LDS:MFMA = **1:9.0** ⇒ (b) 出局。
- `TCP_PENDING_STALL_CYCLES` / CU = **24.0% 的 wall**;`TCC_HIT` 93.8%。
- **决定性的一条**:`TCP_TCC_READ_REQ` = 每 WG-迭代 **恰好 256 = 16 KB / 64 B** ⇒ K/V 的每一个 64 B sector
  都下到 L2,**vL1D 复用率为 0**(报表上的"vL1D 命中 52.7%"是计数粒度假象,不是复用)。
根因是容量:**单个 WG 自己的 K/V 双缓冲工作集(2×8 KB K + 2×8 KB V = 32 KB)已经等于整个 vL1D**,
所以**不管把哪些 WG 摆到同一个 CU 上,兄弟 WG 要读时那些行已经被自己挤掉了**。
⇒ 实测佐证:r13 重写派发解码,让同一 (kv_head, q_block) 的 8 个 GQA sharer(它们走**逐位相同**的 K/V tile 序列)
  在每-XCD 派发序里相距 `_CU_PER_XCD`=32 从而落到同一个 CU(离线穷举证双射,共驻 WG 中 512/768 变成
  同一条 K/V 流,原来是 0/768;`.vgpr_count` 150 不变、sgpr 54→50、输出逐位相同)——
  `TCP_TCC_READ_REQ` 2.727e8、`TCP_PENDING_STALL` 4.086e8 vs 4.088e8、`TCC_HIT` 2.637e8 **三个都逐位不动**,
  wall **−0.57%**(纯粹是 LPT 粒度从"逐个 q_block 降序"退化成"32 个一组降序"的损失)。
⇒ **整族关闭**:"重排派发序去换 cache 复用"在 fwd 上现在 L2(r11 Z3 = −0.19%)和 L1(r13 = −0.57%)**两级都判负**,
  两次的计数器都证明请求流的**总量与构成完全不变**,只有时序变了。要减每-CU 请求字节数只剩**加大 block_m**
  一条路(每单位工作的 K/V 字节减半),而它现在被寄存器预算挡着 ⇒ 回到占用率那条账。

### ★ schedule-fragility:「在 exp2 所在 cluster 里紧跟着插 pack」必崩
5 个**互相独立**的改动得到同一签名 **SNR 崩 + `det=False`**(不是"稍微不准"):carry-pack 三变体
(−13.3 / 12.3 dB / NaN)、把 `l_row` 的跨 lane `permlane32_swap` 推迟到 epilogue(−8.9 dB,
**数学上严格等价**)。崩点既不在 loop-carry、也不在 iter_arg 类型(bitcast 成 f32 向量同样崩),
而在**动作本身**。指向一处被当前调度掩盖的缺失同步。**下一轮动 carry-pack 前先 diff 好/坏两版
`21_final_isa.s` 在该 cluster 区间的 `s_waitcnt`/`s_barrier`/`permlane` 相对顺序,别继续猜。**

### ★★ 2026-07-29 r9 重定 bound:之前的 ISA histogram 取错了 region,VALU 被高估 35%
**先纠正取样**:用"最大回边区间"当主循环是错的 —— 那个区间(`.LBB0_5`/`.LBB0_6`,1494 行、
64 MFMA、**16 个 s_barrier**、还含 `buffer_store`)把 **epilogue 一起圈进去了**。
真正的主循环体是 `.LBB0_16`(472 行、**32 MFMA、正好 8 个 s_barrier、0 个 store**)。
★**判据:fwd 主循环体 = 恰好 8 个 `s_barrier` + 32 个 MFMA + 零 `buffer_store`**;
  枚举所有回边并按 barrier 数挑,别按 span 挑。
两份账的差(左=旧污染值/iter,右=真值/iter):
  `v_mov_b32` 37.5 → **2** · `v_permlane32_swap` 10 → **2** · `v_cvt_pk_bf16_f32` 40 → **32** ·
  `v_pk_mul_f32` 8 → **0** · `v_cmp_lt_i32`+`v_cndmask` 96 → **0**(掩码块根本不在循环体里) ·
  `s_nop` 停顿 71.5 拍 → **25 拍**(其中 44 拍原本落在不执行的掩码块内)。
⇒ 真实每 iter:VALU **169 条 ≈ 676 拍** vs MFMA **32 条 × 32 拍 = 1024 拍**,
  **MFMA 才是容量项(1.5×VALU)**,不是旧记的"VALU 1.18× MFMA"。
~~独立校核:MfmaUtil 51.26% ⇒ 1116/0.5126 ≈ 2177 TF,与 2.0 GHz 纸面峰值吻合 ⇒ MFMA 32 拍/条、
51% util 就是全部账 ⇒「省 VALU / 缩短依赖链」这一整类为 0 是结构必然。~~
⚠ **2026-07-29 这两条全部作废,根因是从来没人量过 sclk**(见下面「时钟层」与「省 VALU 族」两节)。
真值:负载中 **sclk 1713 MHz / power 1346 W(板卡 1400 W 的 96%)** ⇒ 在-时钟峰值 =
1024 SIMD × 1024 FLOP/cyc × 1.713 GHz = **1796 TF**,当时的 util 是 **63%** 不是 51%,
每 wave-迭代 **1612 SIMD-cyc** 不是 1941/2023。**MFMA 32 拍/条 ✅ 属实**,其余按 63% 重读。
⇒ 通用铁律:**≥55% util 的 kernel、以及任何写 util 百分比的结论,必须先量负载中的 sclk/power**;
  卡片里所有 util 都要注明按哪个 sclk 折算。
⇒ **缺的是并发(3 waves/SIMD 藏不住延迟)**;按 gfx950 分配粒度 8,`waves/SIMD = floor(512/alloc)`,
**161~168 全是 3 waves,136/144/152/160 降下去一分不涨**,下一档 4 waves 必须**一步到位做到 alloc ≤128**。
r6 实测寄存器构成(D=64 ⇒ D_CHUNKS=2):v_o 32 + 跨 cluster 携带的 v_p 32 +
软件流水让 v_s_0/v_s_1 **同时活着的 64** + Q 16 ≈ 144,余下约 22 是 K/V 与地址。
最大单项就是那 64 个同时活着的 score,其次是 v_p 的 32 个。

### ★★ 更正本卡 bwd 篇的一条硬件记载:`v_exp_f32` 是 **half-rate(~8 拍)**,不是 quarter 也不是全速率

本卡 bwd 篇多处写 "AMD 的 `v_exp_f32` 是 VALU-pipe **quarter-rate**"(收官清单第 5 条)——**quarter-rate 是错的**。
但两版更正之间还有一次反复,结论以**直测**为准:

| 证据 | 方法 | 读数 | 判 |
|---|---|---|---|
| r7 | **代换探针**:64 条 exp2 换成同条数 `v_mul_f32` | wall 只 +1.9% ⇒ 推 "≈全速率(~5 拍)" | ❌ 被 r10 推翻 |
| r10 | **PMC 直接分解**:`SQ_ACTIVE_INST_VALU` 1091.7 拍 / `SQ_INSTS_VALU` 206.7 条(扣掉 32 条 MFMA 按 4 拍) | 64 条 exp 每条 **8.14 拍**,142.7 条普通 VALU 每条 4 拍 | ✅ 采信 |
| **gpt-oss D=64 bwd r12**(**另一具 kernel、另一种拟合**) | 同样 PMC 分解但在 **bwd dkdv** 上做:`SQ_ACTIVE_INST_VALU` 12698 拍 / 2705 条 VALU-类。按"普通 VALU 4 拍 + 512 条 exp 各多 4 拍"预测 18140 拍,实测 18435(含 b128 LDS 的额外拍) | 每条 exp **8 拍** | ✅ **第三个独立标定点,fwd/bwd 一致** |

⇒ **gfx950 `v_exp_f32` ≈ 8 拍 = half-rate,已在 fwd 与 bwd 两具 kernel 上各自独立测到。**
★ 8 拍这个数有一个直接可用的推论(gpt-oss bwd r12 首次用它排臂):一个 MFMA 的阴影是
**16−4 = 12 拍**,所以 **一条 exp(8)+ 一条 `v_cvt_pk_bf16_f32`(4)= 12 = 正好填满一个阴影**。
`iglp_opt(3)`(MFMAExpSimpleInterleave)只认 exp 链,所以它把 exp 塞进阴影后**每个阴影都还剩 4 拍**;
把 pack 与 exp 做 1:1:1 配对是这条硬件常数直接给出的排程处方,不需要再猜。 r7 那 +1.9% 读低的原因是**差额被重叠吃掉了**,
不是指令本身便宜 —— 这正是 methodology/03「代换/减法探针给的是上界不是可达值」的又一个实例:
**代换探针能测"删掉省多少 wall",测不出"这条指令本身占几拍";要拿单价必须用 PMC 分解。**

方向仍不变(exp 不是头号项),但旧的 quarter-rate 算法高估约 2 倍、r7 的全速率结论低估约 1.6 倍。
这也和 bwd 侧 "poly-exp2 天花板≈fast、超不了" 自洽:硬件 exp 半速率,~7 条 FMA 的软件多项式仍然更贵。

### ★★ 2026-07-29 barrier 数量:改对了也不涨,且会**静默**破坏 8-wave 配置
fwd 主循环是 8 个 cluster / 2 个 kv-tile,每个 cluster 尾一个 `s_barrier` = **8 barrier/iter**。
4 个"访存簇"(读 K/读 V)的 barrier 在语义上是**冗余**的:它保护的"读完再让别人覆写"这件事,
由紧跟其后的**计算簇**尾部那个 barrier 同样能保证(覆写发生在再下一个访存簇)。按此把访存簇
折进计算簇实测:
- 8→6 barrier(折 C0→C1、C4→C5)= **1114.7**;8→4(再折 C2→C3、C6→C7)= **1116.9**;
  基线 1116.3(1115.5/1117.0)⇒ **全在 ±0.15% 噪声内,barrier convoy 的代价 ≈ 0**。
  ⇒ "35% 双管线空闲是 8 个 barrier 造成的"这个假设**判负**;3 个共驻 WG 本来就把 convoy 解耦了。
- ★但 `_verify_swa_varlen.py` 22 组里 **8-wave(block_m=256 + stagger on)4 组全 FAIL**
  (cos 0.938~0.968、rel_l2 0.25~0.35),而**计分形状的 SNR 51.26 dB / det=True 完全放行**。
  根因:`stagger` 靠"给 group B 多发一个 `s_barrier`"来错开两个 wave-group 的相位,
  barrier 总数一变,两组的配对关系就错位 ⇒ 跨 group 读到未写完的 LDS。
  ⇒ **铁律:任何 barrier/同步数量改动必须跑多形状 + 多 wave 数验证;单形状 SNR+det 对它是瞎的**
  (det=True 也不保护——4 waves 与 8 waves 走的是不同配对)。

### ★ v_s/v_p 共存窗口只值 4 个寄存器(2026-07-29,推翻"最大单项是 64 个 score"的推断)
"软件流水让两个 tile 的 score 同时活着 = 64 个寄存器"是**从代码结构推的,不是量的**。实测:
- 把 QK 的 MFMA 排到 `cast_p` 之后(P 已从 32 f32 压成 16 bf16 dword 才写新 score):
  `.vgpr_count` **166→162**、spill 0、输出与基线**逐位相同**、wall **持平**(1116.1 vs 1116.3)。
- 再在 pack 与 QK 之间插**硬调度栅栏**彻底禁止共存:vgpr 只再降到 162 一档、wall **−0.90%**。
⇒ 分配器早就把这两个值叠得很好,**完全消除共存总共只值 4 个寄存器**,不是 30 个;
  且 liveness 由**调度后**的顺序决定,不是文本顺序(纯文本重排 −4,加栅栏才再 −0)。
⇒ 想从 166 走到 4 waves/SIMD 需要的 alloc ≤128 必须换**全局长命值**的账(v_o 32 / K,V tile 各 32 /
  Q 16),寄存器高水位实测**分散在 6+ 个基本块**(138~161),没有单一主导窗口可压。

### ★★★ 为什么 hd64 fwd 的 `.vgpr_count` 压不动:高水位是**全kernel摊平的**(r13 逐块实测)
上一条说"分散在 6+ 个基本块"是从活跃度曲线看的;r13 直接按基本块打**它引用到的最大 VGPR 号**,
结论更硬 —— 全 kernel 九个块几乎**同时**顶在预算上(`.vgpr_count` 150 = 最大寄存器号 149 + 1):

| 块 | entry | LBB0_3 | LBB0_6 | LBB0_8 | LBB0_9 | LBB0_12 | LBB0_14 | LBB0_16 | LBB0_18 |
|---|---|---|---|---|---|---|---|---|---|
| max VGPR 号 | 142 | 144 | 147 | 146 | 148 | **149** | 148 | 145 | 144 |

⇒ **压任何一个窗口都拿不到分配**:砍掉一处的 10 个,另外八个块仍然要 148。这一条**定量解释了
连续三轮的"活跃度降了、`.vgpr_count` 纹丝不动"**(r10 K tile 分半读 峰值 142→132 = 162 不变;
r13 V tile 分半读 = 150 不变;r13 把 r12 的 4 宽 `l_row` 累加器退出 loop-carry 只留 1 宽 = 150→**154 反而涨**)。
⇒ 判据入卡:**动手压寄存器前,先按基本块打 max-VGPR 号直方图**;只有当某个块明显高出其它块
  才值得做局部活跃度优化,否则必须换**全局常驻集**的账(v_o 32 + 携带 v_p 16 + Q 16 = 64)。
⇒ ⚠ 顺带算清一条纸面路:把 Q 的 16 个寄存器挪进 LDS 可以**同时**给九个块各减 16(150→约 134),
  但 Q tile = 128×64×2 B = **16 KB/WG**,加上现有 34304 B ⇒ 50 KB/WG,4 WG/CU 需要 200 KB > **160 KB**
  ⇒ 拿到寄存器却被 LDS 卡回 3 WG/CU,占用率净零。**要 4 waves 必须同时缩 K/V 的 LDS 双缓冲。**

### ★★★ 2026-07-29 r14 拿到 3→4 waves/SIMD:堵路的不是寄存器数,是 **inline-asm LDS 读**
上一条推的"要 4 waves 必须先把 Q 挪进 LDS 并缩双缓冲"**没有走通,也不需要**。真正的堵点是
V tile 的转置读写成了 inline asm(`ds_read_b64_tr_b16` + `~{memory}`):`SIInsertWaitcnts`
记不到它的目的寄存器 ⇒ (a) 每个消费者都要**手写** lgkmcnt,(b) 后端会把手写的 graded 值放宽,
所以必须用 `sched_barrier(0)` 把它钉住 ⇒ (c) 这些栅栏把 P·V 簇切成不可调度的碎片。
后果:想靠"把 tile 分成小块、用时按需读"压活跃度时,每分一块就要多付一道栅栏,**register 省下来
的收益被调度损失吃光**(实测:asm 版 K+V 分半读 3 waves = −3.26%,4 waves 只回 +2.99%,净 −0.35%)。

**改法(一行级别):`llvm.inline_asm` → `rocdl.ds_read_tr16_b64`**(deployed flydsl 0.2.2 里
`flydsl.expr.rocdl` 已有这个一等 op,签名 `ds_read_tr16_b64_(res_type, ptr, alias_scopes=, noalias_scopes=)`,
吃 **addrspace(3) 指针**而不是 i32 地址,常量 byte 偏移交给 `get_element_ptr` 让后端折进 offset 域)。
发出的机器指令**逐条相同**(96 条 `ds_read_b64_tr_b16` 不变),但后端开始为它排 lgkmcnt:

| | asm 版 | 已评分 op 版 |
|---|---|---|
| 手写 V 等待 | memory 簇 1 条 + P·V 簇每步 1 条 | **全删** |
| `sched_barrier(0)` 墙 | 每个 tail 读 2~3 道 | **全删** |
| K+V 分半读的代价(3 waves) | −3.26% | **−1.95%** |
| V 可切的粒度 | 2+2 substep(再细就被栅栏吃掉) | **1+1+1+1**(每步读它后面第 V_HEAD 步的那一块) |
| `.vgpr_count` @ wpe=4 | 128 / spill **14~105** | 128 / spill **2**(K_HEAD=1 时 125 / spill **0**) |
| occupancy(实测) | 2.985 | **3.978** |

⇒ **通用教训**:凡是"分块读 + 按需等待"用来压占用率的改动,先确认那条读指令是不是 inline asm。
  asm 读让 waitcnt 与调度都必须手工,手工的代价通常正好抵消寄存器收益 —— 这是本 campaign
  连续四轮(r10/r11/r13/r14 前半)"活跃度降了、分配不动/动了也不赚"的共同根因。
  **MFMA/DOT 一律走 intrinsic** 这条要扩写成:**LDS/访存类专用指令也一律走 rocdl op,别写 asm。**

### ★★ 功耗墙下抬占用率:省下的周期会被**时钟回吐**大半(r14 实测,KB 未记载)
同一次改动的 sclk/power 双臂(口径:负载中 4 个采样点):参考臂 **1710 MHz / 1352 W**,
4-wave 臂 **1654 MHz / 1353 W**,功耗一模一样(都是 1400 W TBP 的 96.6%),**时钟掉了 3.3%**。
分数只涨 +0.50%,反解出"省周期"其实是 **+3.9%**。
⇒ 在 ≥95% TBP 的 kernel 上,**占用率类改动的 wall 收益 ≈ 周期收益 − 时钟回吐**,
  而时钟回吐随并发度上升(更多 wave 同时发 MFMA ⇒ 每拍功耗更高 ⇒ DVFS 压频)。
  **每个占用率候选都必须同时报 sclk**,否则会把一个 +3.9% 的周期改动误判成"只值 0.5%,不值得做"。

### ★★ `sched_barrier(0)` 不是通用的"收窄活跃区间"杠杆(r13 十点扫描全平)
r12 记过"同一份源码加一个 `sched_barrier(0)` 就从 168 掉到 150",容易被读成"多插栅栏能压分配"。
r13 用 env 切臂在主循环四类站点(K 读后 / V 读后 / QK MFMA 组后 / 第二半 exp2 后)扫了
**10 种组合(单点 + 两两 + 三点 + 四点)**,`.vgpr_count` **全部恰好 150、spill 全 0**。
⇒ r12 那次下降是**那个站点特有**的(它把 inline-asm cvt 块与 row-sum MFMA 组切开,改变的是
  冒险/调度结构),不是"栅栏 ⇒ 少寄存器"的普遍规律。**别再把插栅栏当占用率手段。**

### ★★ 「省 VALU 这一族结构性为 0」判负作废(2026-07-29,干净探针 = +16.6%)
上面那条 0 是从三个**没有真的减少 VALU 周期**的实验推出来的:dot2 是 half-rate(32 条 dot2 ≈
64 条 add,收支抵平)、树形归约条数不变、去锚点只省 8 条 `v_mov`。干净的 VALU 地板探针
(exp 全变恒等 + row-sum 32 元素 fold 只留 2 元素,**MFMA/cvt 条数一条不动**,共 −520 发射拍)
= **1320.1 vs 1132.0 = +16.6%** ⇒ 可删 VALU 全族上界 **+16.8%**,
**干净边际系数 = 240/520 = 0.46 拍 wall / 拍 VALU**(旧记 0.49 数值接近但来自 SNR 崩掉的脏探针,合并为这一条)。
⇒ 减性探针必须**保住下游消费链**,否则 DCE 连带砍 MFMA:只删 32 条 exp 却连带删掉 4 MFMA + 16 cvt,
  读数从 +8% 虚高到 +21.6%。每个减性探针配一次 ISA histogram 复核。
**兑现证据(r12)**:按 0.46 折算 row-sum 的 264 拍 = 121 拍 wall,减去 4 条 16x16x32 MFMA
(64 MFMA 拍 × 1.15 = 74 拍)⇒ 预测 +2.9%,实测 **+1.96%**,同号同量级。

### ★★★ row-sum 上 MAI:**16x16x32 + ones A** 的操作数 layout(r12 实测跑通,+1.96%)
把 online-softmax 的 32 元素 VALU fold 换成 MFMA,**形状必须是 16x16x32,不能是 32x32x16**:
32x32x16 要 4 条 × 32 拍 = 128 MFMA 拍(×1.15 = 147 拍 wall)换 121 拍 ⇒ 纸面即净亏;
16x16x32 只要 4 条 × 16 拍 = 64 拍(×1.15 = 74 拍)⇒ 净赚。
**layout 推导(gfx950,已用逐行数值探针证过,不需要任何跨 lane 搬运)**:
- 打包好的 P slice 本来就是 PV 那条 32x32x16 的 B 操作数:`col = lane%32 = q 行`,k = kv。
- 同一批寄存器当 16x16x32 的 B 读:`col = lane%16`、k 按 lane 组四等分 ⇒ 第 n 列里
  lane 组 0/2 装的是 q=n 的两个 kv 半区,lane 组 1/3 装的是 q=n+16 的两个半区。
- A 是 ones-mask,lane 持有 `m = lane%16`、同一 k 组 ⇒ 谓词
  `ones = (lane%4==0) && ((lane//4)%2 == (lane//16)%2)`,四个 dword 全填 `0x3F803F80`(bf16)/`0x3C003C00`(fp16)。
- D 是 4 个 f32,`m = 4*(lane//16)+r` ⇒ **每个 lane 自己那行的 row-sum 恰好落在 D 元素 0**,
  而且这条 MFMA **顺带把 half-wave 伙伴折进去了**,原来的 `permlane32_swap` pair-reduce 整条消失。
- 副作用:分子分母共用同一份 bf16 P,数值上更自洽 ⇒ **SNR 51.26 → 51.61 dB**(比 VALU 路径更好)。
- 实测:`v_add_f32` 68→**0**/iter、`v_permlane32_swap` 2→**0**/iter、新增 8 条
  `v_mfma_f32_16x16x32_bf16`/iter;`.vgpr_count` **162→150** / spill 0。

### ★★★ `v_cvt_pk_bf16_f32` 走 inline asm ⇒ 冒险识别器看不见它(r12 抓到的真 bug,回溯解释 r6 的五次崩)
FlyDSL 的 `rocdl.cvt_pk_bf16_f32` 在 ISA 里是 `;;#ASMSTART / v_cvt_pk_bf16_f32 / ;;#ASMEND`。
`GCNHazardRecognizer` **不把它记成 VALU 的 VGPR 写**,于是「VALU 写 VGPR → MFMA 读它做 SrcA/SrcB
需要 2 个 wait state」这条冒险**一个 s_nop 都不补**(实测只插了 `s_nop 0` = 1 拍)。
症状:第一版 row-sum MFMA 紧跟 cvt 消费刚打包好的 P ⇒ **SNR 24.99 dB + det=False**,
且错误**只落在一半的 q 行上**(`(q//4)%2==0`),用「按 q 行的 scale-free 残差」探针一眼可见:
坏行残差 0.06~0.09、好行 0.0027(= bf16 噪声地板)。
**修法(1 行,零成本)**:在这批 MFMA 前放 `sched_barrier(0)` 把所有 cvt 隔到前面,再补 `s_nop 1`。
修完 det=True、SNR 51.61、22/22 全过,而且因为调度器不再拉长活跃区间,**vgpr 反而从 168 掉到 150**、
wall 比"让它自由交织"的坏版还快(1159.5 vs 1149)。
⇒ **通用铁律:任何让 MFMA 直接消费 `cvt_pk_*` / 其它 inline-asm 产物的改动,都要自己插
  `sched_barrier(0) + s_nop`;`det=False` + 半数行系统性偏差 = 这个签名。**
⇒ 这条**回溯解释了 r6 的五个"exp2 邻域改动必崩"案例**(H3/H3b/H3c/H3d 全是"紧跟 exp2 插 pack"、
  H7 把 `l_row` 归约推迟后让 pack 靠近消费者)——不是 dualwave 握手玄学,是这条没插的 wait state。
  定位规程也随之更新:先看 **cvt-asm 写的寄存器到下一个 MFMA 读它之间隔了几拍**,再去 diff permlane。

### ✅ 编译器调度策略是一条独立的、从未审计过的杠杆(r15 实测 +0.6~0.7%,零源码风险)
FlyDSL 的 `compile_hints["llvm_options"]` 直接改 LLVM 的 cl::opt,机制是
`flydsl/compiler/llvm_options.py` 的 **scoped context manager**(逐项 save/restore)⇒
**不会泄漏到同进程里编译的其它 kernel,可以进产线**。
hd64 fwd 实测(每臂先过 `.vgpr_count`+spill 门,再做位置配平 A/B):
- ✅ **`amdgpu-sched-strategy="max-memory-clause"` + `enable-post-misched=True`** = **+0.6~0.7% wall**,
  主循环 `s_waitcnt` **94→66** 条,`.vgpr_count` 128 / spill 2 / LDS 34048 **全不变**,输出 SNR 逐位相同。
  非计分配置同向:S=8192 1138.5→1152.1、SWA W=512 579.1→585.1、3:1 加权 1011.2→1018.5。
  机理与本 kernel 结构自洽:K/V 从 LDS 分块读散布在各计算簇里,memory-clause 策略正是把这些读聚簇。
- `max-ilp` = +0.4%(次优);`enable-post-misched` 单独 = +0.1%;`lsr-drop-solution=0` = 中性偏负(保持 True)。
- ⚠⚠ **这条只对"调度还没被手工定死"的核成立**:同族 **bwd** 上 `max-memory-clause` = dkdv −0.59% / dq −0.13%、
  `max-ilp` = dkdv −0.85%(各 3/3),因为 bwd 的 LDS 读已被手写 `sched_mfma/sched_dsrd` 1:1 交错锁定,
  策略只能把手工结构打散 ⇒ 见 bwd 篇 r14。**换策略前先看这个核有没有手工调度**。
- ⚠⚠ **`iterative-*` 三个策略必须过 SNR 门再看时间**:`iterative-ilp` 是**最快的一臂(+0.63%)但 SNR=nan / det=False**
  (waitcnt 94→61,它把手工放的 `sched_barrier(0)` / 冒险保护重排掉了);`iterative-minreg` spill **144 dword**;
  `iterative-maxocc` spill 10。⇒ 换调度策略后**必须重跑 SNR/det**,只看 `.vgpr_count`+bench 会放行一个错的 kernel。
- ❌ 本 kernel 上完全惰性(ISA 统计逐项相同,不必再试):`amdgpu-use-amdgpu-trackers`、
  `amdgpu-schedule-relaxed-occupancy`、`misched-postra`(与 `enable-post-misched` 重复)、`amdgpu-schedule-metric-bias`。
  `misched-cluster=0` 把 waitcnt 66→74(与获胜方向相反,未上机)。
- ⚠ **同一次探针驱动里存在单调的位置漂移**(同一臂放在第 1 位比第 9 位低约 0.5~1%,越跑越热/越稳)。
  ⇒ 多臂 A/B 必须用**回文序**(A B C C B A)让每臂平均位置相同,别用"同序重复 N 轮"——
  后者会把靠前的臂系统性判负。这条是本轮能分辨 0.1% 级差异的原因。

### ✅✅ GQA sharer 合并(2026-07-29 r18,+2.04%,1177→1201 conv TF)——**在功耗墙上"省请求=买时钟"**
一个 8-wave CTA 跑同一个 kv-head 下 **2 个 q-head 的同 index q-tile**(`Q_HEADS_PER_WG=2`):
`NUM_WAVES` 4→8 而 **`BLOCK_M` 仍是 128**(低位 wave 选行组、高位 wave 选 sharer),两个 sharer 共用同一份
K/V 的 LDS tile 与 s_barrier。改动量极小:`wave_q_offset = (wave_id % WAVE_ROW_GROUPS) * ROWS_PER_WAVE`、
`q_head_idx += wave_id_uni // WAVE_ROW_GROUPS`、`grid.x /= 2`,其余全部自适应
(`dma_wave_reps = SMEM_N_RPT // NUM_WAVES` 自动 2→1;LDS 只装 K/V,**与 BLOCK_M 无关**,恒 34048 B)。
ISA:vgpr **128** / spill 2 / LDS 34048 / workgroup 512 ⇒ `min(floor(512/128)=4, LDS 4 WG×8/4=8)` = **4 waves/SIMD 未掉档**。
- **三栏记账(必须分开量,否则会把结论记反)**:wall **+2.0%**,但 sclk **1609→1749 MHz(+8.7%)**、功耗不变
  (1342→1339 W,96% TBP)⇒ **真周期反而 +6.6%**。即:收益 100% 来自时钟,8-wave 结构本身**吃掉 6.6% 的周期**。
- PMC 双臂:`TCP_TCC_READ_REQ` **2.734e8 → 1.381e8(−49.5%**,每 WG-迭代 256→128 sector)、
  `TCP_PENDING_STALL` −24.0%。⇒ **能耗在「TCP→TCC 的请求/sector 路径」上,不在 DRAM 回填上**。
  这条**修正 r11 的记载**「pinned 探针省 4.3 TB/s L2 流量只换 +0.94% 时钟 ⇒ 能耗几乎全在 MFMA+VALU」:
  pinned 探针改的是**命中层级**(请求数不变,r13/r16 实测 `TCP_TCC_READ_REQ` 逐位不动),本轮改的是**请求数本身**,
  后者才是功耗墙上的杠杆。⇒ **判断"访存还值不值钱"要看 `TCP_TCC_READ_REQ`,不是看 DRAM 带宽或 L2 命中率**。
- ⚠⚠ **可迁移性有一个硬前提:合并后不能掉 WG/CU**。fwd 这里是 4 waves/SIMD、LDS 只装 K/V(与 BLOCK_M 无关),
  8-wave CTA 之后仍有 ≥2 WG/CU 互相盖 barrier;把同一手法搬到 **bwd**(occ-2、VGPR-locked)上,8-wave ⇒
  **1 WG/CU**,dq 实测 **−2.7%**(请求数确实腰斩、输出逐位相同,纯结构损失),dkdv 亦 −4.3% ⇒ 见 bwd 篇 r14。
- 正确性:输出与 4-wave 路径**逐位相同**(SNR 51.61229610443115、det=True),`_verify_swa_varlen.py` 22/22
  (18 个 bm=128 臂在 Hq128/Hkv16 下**全部走合并路径**,含 SWA / ragged varlen / 短于一个 tile),8-wave rel_l2 0.0027/0.0028。
- ⚠ **`vmcnt` 阈值必须按「在飞的 tile 数」而不是「在飞的指令数」写**:`_waitcnt_vm_n(NUM_DMA_K+NUM_DMA_V)` 的
  常量 2 只在 `dma_wave_reps=2`(4-wave)时等于"一个 tile 的指令数";wave 数一变 reps→1,同一个常量就变成
  **多放一个 tile 在飞**(潜伏 RAW)。已改成 `VM_DRAIN_KV = (NUM_DMA_K+NUM_DMA_V) * dma_wave_reps // 2`
  (4-wave 逐位不变,8-wave 收紧一档),实测收紧**零代价**(1199.0 vs 1199.5,与 r11 Z1「vm-wait 池≈0.25%」自洽)。
  ⇒ 任何改 CTA 波数的改动,都要重算所有以指令数计的 `vmcnt`/`lgkmcnt` 常量。
- ⚠ **`s_barrier` 的 rendezvous 池随 CTA 波数开出来**:r11 Z2 在 4-wave 上测「8 个 barrier 全删 = 0%」,
  同一个探针在 8-wave 合并版上是 **+2.4%**(1229.9,SNR 崩,仅作诊断)。⇒「同步族已整族关闭」这条结论
  **只在 4-wave CTA 下成立**,放大 CTA 必须重测。这也是上面那 6.6% 周期损失的三分之一。
- ⚠ **判负结论要连着测量条件一起记**:r15 记的「block_m=256(同样是 8-wave CTA)= −2.8~−4.1%」在 r17 之后
  用**同一把 bench 尺子**复测是 **1178.7 = 中性**(vgpr 128 / spill 2 / LDS 34048 全部同档)。
  本轮正是因为先花 20 秒复测了这个"已判负"的结构代理臂,才没有按旧结论放弃 GQA 合并。
- 下一步(已量化):合并后 pinned-tile 探针从 +5.4% 降到 **+2.9%**(剩余馈送池),而 rendezvous 池升到 +2.4%
  ⇒ `Q_HEADS_PER_WG=4`(16-wave CTA)最多再拿 ~1.5% 却要付更大的 rendezvous,**期望为负**;
  真正该攻的是那 2.4% 的 rendezvous(减 barrier 数 / 让两个 sharer 组错相位)。
- ★ **rendezvous 池不是线性的,r19 记的「删一半恰好收回一半、可外推」被 r20 推翻**:同一个「barrier 全删」
  探针在 8 barrier 时 +2.59%、4 barrier 时 +1.30%、**2 barrier 时仍有 +1.25%**(0.32 → 0.33 → **0.63 %/barrier**)。
  ⇒ 留下的 barrier 越少,每个越贵(它们承担全部 WAR 边并聚集全部 drain);
  **不要用「池子 ÷ barrier 数」给下一步定价**,每减一档都要重测池子。
- ★ **双缓冲下 2 barriers/iteration 是结构下限,且落点唯一**:8 簇软件流水(mem/compute 交替、K 与 V 各双缓冲)
  把 4 个计算簇 rendezvous 减到 2 的**唯一合法落点是两个 QK 簇尾**(C1/C5),前提是把每块 buffer 的
  K 与 V 覆写 DMA 合并进同一个 P·V 簇(C3/C7),使每个 barrier-region「整块读一个 buffer、整块填另一个」。
  往前挪到 C0/C4 会让 K buf0 的读与覆写落进同一 region(WAR),往后挪到 C2/C6 会让 V buf1 的写与读落进
  同一 region(RAW)。再减到 1 barrier/iteration 时 4 个 buffer 访问里有 3 个塌进同一 region
  ⇒ **必须 ≥3 深缓冲**(LDS 侧免费:三缓冲 51 KB、四缓冲 68 KB,均 ≤ 8-wave CTA 2 WG/CU 的 81920 B),
  代价是 buf 索引要按 tile 轮转(mod 3/mod 4),要么 2× 展开循环体(付余数尾块),要么走
  `_k_buf_base/_v_buf_base` 已支持的 **runtime buf_id**(把 buffer 偏移从 ds_read 立即数搬进地址算术,
  在 vgpr 恰好 128 的悬崖上有掉档风险,先过 ISA 门)。
- ★ **「更深预取」这一族在 2-barrier regime 下仍然是零池子**:两个 barrier 的 drain 都收紧到 `vmcnt(0)`
  (两个 DMA 同 region 发射,in-order vmcnt 下没法只放一个在飞)之后,把它们全放开到 `vmcnt(63)`
  实测 **−0.05%**(1214.7 vs 1215.3)。⇒ 四缓冲的价值**只在 barrier 数**,不在预取深度;
  r11 Z1 的「vm-wait 池 ≈ 0」在 8-wave 合并 + 收紧 drain 之后依然成立。
- ★ **覆写 DMA 在 region 内的落点:重要的不是 slack,而是别挤占「读→紧邻 MFMA」的那个簇**。
  同一对 DMA 放在 region 头(C2/C6,slack 多一个簇)= **−0.2%**(1211.4 vs 1213.2 中位,区间不重叠),
  放在 P·V 簇(C3)与放在第二个访存簇(C4)= **中性**(中位 1213.3 vs 1213.2,完全重叠)。
  C2 之所以差,是因为它自己的 V head 读要立刻喂 C3 的 MFMA,DMA 发射把这条读推后了。
- ★ **rendezvous 池取决于「共驻 WG 数」而不是 CTA 波数,而且是非单调的**(r21 三点实测):
  4-wave(3 WG/CU)= 0%、8-wave 合并(2 WG/CU)= +2.4%/+1.25%、**16-wave(1 WG/CU)= 0%**。
  只有存在第二个共驻 WG 时,barrier convoy 才会与 WG 间调度争用叠加。⇒ 任何"零池子"结论都要标注
  **共驻 WG 数**;`Q_HEADS_PER_WG=4` 判负的真机制是 1 WG/CU 让软件流水 prologue/epilogue 完全暴露
  (s8192 −2.1% vs s16384 −0.5%,固定每-WG 开销的签名),不是"rendezvous 继续涨"。
- ★ **DMA 请求的「指令级冗余」几乎免费**:16 波对 8 条 LDS line 时两个 wave 取同一条 line,
  去冗余后 1198.2→1199.7 = 0%。⇒ 判断访存压力看 `TCP_TCC_READ_REQ`(唯一地址数),
  别数 `buffer_load_lds` 指令条数,重复地址会被合并/命中。
- ★ **覆写 DMA 落点这一族已探尽**(r20/r21 五个读数):region 头 −0.2%、P·V 簇与第二访存簇中性、
  K/V 拆到两个簇 −0.37%、整体提前两个簇中性、drain 全放开 −0.05%。别再扫第六个点。

#### 2026-07-29 r22:1 barrier / 2 kv-tiles(K/V 四深 + 2× 展开 + **谓词化余数体**)
- ★ **展开的代价不是代码体积,是「第三份 body 实例」把 RA 推过 128 VGPR 悬崖**。
  四深 + 2× 展开后主循环 `s_barrier` 2→1(每 2 个 kv-tile),但 `.vgpr_spill_count` 从 2 跳到 **35**、
  wall **981 TF(−19%)**。单变量拆解:2-deep + 同样的 2× 展开 = spill 30 ⇒ **28 个 spill dword 来自展开本身,
  只有 5 个来自四深的槽位**。再拆:把处理奇数余数的**尾循环**(第三份 body 实例)去掉,
  spill 立刻 35→7(K_HEAD=2)、10→**0**(K_HEAD=1)。⇒ 结论:**RA 的压力来自 body 实例数,不是单个 body 的峰值活跃**;
  展开一个已经贴着寄存器预算的 body 时,余数**必须共享同一份代码**。
- ★ **余数的正确写法 = 谓词化第二个 body,不是尾循环**:循环 step=4,第一个 body 无条件跑,
  第二个 body 包在 `scf_if_dispatch(j+1 < split_t_end, ...)` 里(条件是 WG-uniform,body 内的 s_barrier 合法)。
  两份实例 ⇒ spill 0、vgpr 128、LDS 68096(仍 2 WG/CU × 8 波 = 4 waves/SIMD)、22/22 全过。
  ⚠ 与 r19 H5f「`scf.if` 包 barrier 把 RA 打爆(spill 97)」并不矛盾:那次包的是**分支内的条件 barrier**,
  这次包的是**整个 body**(所有 live 值经 scf.if 的 result 一次汇合),实测 RA 完全不受伤。
- ★ **`.vgpr_spill_count` 的代价刻度(本 kernel,vgpr 128 / 4 waves)**:0 → 1217.9;5 → 1214.1(−0.3%);
  10 → 1215.3(−0.2%);35 → 981(−19%)。个位数 spill 值 0.2~0.4%,**30+ 是断崖**。
  推论:`K_HEAD=2` 在 2-deep 下比 `K_HEAD=1` 好 0.18%,但在四深展开下带 5 个 spill dword ⇒ 反而 **−0.39%**;
  **K_HEAD/V_HEAD 这类"多驻留一点"的旋钮必须在当前 spill 水位下重扫**(V_HEAD=2 → spill 8 → −0.9%)。
- ★ **rendezvous 池随 barrier 数继续非线性上升**:8→+2.59%、4→+1.30%、2→+1.25%、**1→+0.91%**
  (每 barrier 0.32 / 0.33 / 0.63 / **0.91**)。2→1 实测兑现 **+0.34%**(bench 1217.9 vs 1213.8 中位,
  sclk 1698 vs 1716 MHz ⇒ 真周期约 +1.35%,被 DVFS 回吐 1%)。再往下要 8 槽(LDS 136 KB ⇒ 1 WG/CU),
  而 1 WG/CU 已被 r21 判负,所以下一步该攻的是别的池子(剩余馈送池 +2.9%)。
- ★ **给「跳过部分工作」的诊断臂定价:本 kernel 跳掉余数体 = 分数虚高 0.8%**(实测 1222.5 vs 1212.2,
  其中 6 个 spill dword 约 0.07%)。full-causal S=16384 / BLOCK_N=64 下,一半 q-block 的 body 数为奇数,
  跳掉 2 个 tile / 平均 129 个 tile ⇒ 0.78%,与实测吻合。**任何 SNR 崩掉的"上界探针"若少做了工作,
  都要先按这个方法把虚高扣掉再判正负。**

#### ★★★ 2026-07-29 r25:「完美编织」不是收益来源,**VALU 落在哪个簇**才是(三种做法全判负)
基线(deployed,fold_pv 已恢复):**1213.7 @ 4 waves / 1118.6 @ 3 waves**。同一件事(消掉 C5 那 36 条
串行 VALU)三种做法,全部有 ISA 佐证,全部判负:
| 做法 | 4 waves | 3 waves | vgpr@wpe3 | spill@wpe4 |
|---|---|---|---|---|
| 基线(两道栅栏在,softmax 跨 3 个簇) | **1213.7** | **1118.6** | 130 | 2 |
| ① 整簇搬去 PV 簇(early_pack:一个 tile 的 softmax 全在读到它 score 的簇里做完,loop-carry 32 f32 → 16 packed dword) | 1209.7(**−0.33%**) | 1108.7(**−0.89%**) | **144** | 6 |
| ② ① + 按 8 个 score 一片交错(每片 8 exp+4 cvt+1 RSUM) | 1201.1(−1.04%) | — | — | 2 |
| ③ r24 原方案:拆两道栅栏,VALU 留在 QK 簇内编织 | 1188.8(−2.05%) | — | — | **24** |
- ① 的 ISA 是**教科书级的完美编织**(`W QKPV dsx2 expx5` ×6 + `cvtx16 RSUMx4`,s_nop_stall 18→28,
  串行段彻底消失),**仍然 −0.33%**。⇒ **r24 记的「完美编织 = +1.23%」不能外推到 deployed 配置**:
  那次是 wpe=3 且 VALU 留在 QK 簇内;本轮把同一批 VALU 搬去 PV 簇,3 waves 上是 **−0.89%**。
- ★ 机制定案:**PV 簇是饱和簇,QK 簇才有空档**。PV 簇的 8 条 MFMA 已经要藏 16 条 `ds_read_b128`(V)
  + 2 个 DMA 发射;QK 簇是 8 条 MFMA + 8 条 `ds_read_b64_tr_b16`,VALU 槽是空的。
  ⇒ **修正 methodology/03「≥2 waves/SIMD 时簇再平衡是吞吐不变量」:方向有价值,两个簇不等价**,
    往饱和簇搬 32 条 VALU = −0.9%。以后写「把 X 挪到 Y 簇」之前,先数 Y 簇 MFMA 影子里已经藏了多少东西。
- ★ ① **不省寄存器,反而多吃 14 个**:3 waves(预算 168)下 RA 从 130 涨到 144。
  根因:bulk 形态要 32 个 exp 后的 f32 同时活着才开始 pack(峰值 48),而原来的跨簇拆分任何时刻只有 32。
  ⇒ 「把 loop-carry 从 32 f32 压成 16 packed dword」在账面上省 16,**实测被簇内峰值吃回去还倒欠**。
- ★ ③ 复现 r24 的 spill 24 预测(1:1 命中),且 ISA 显示它**根本没编织成**——只是把串行块从 MFMA 之后
  挪到了之前。⇒ **P0(拆栅栏 + 腾 20 个寄存器)这条路本轮判负关闭**:即使腾出寄存器,①/② 已经证明
  编织本身在 4 waves 上不值钱。
- ★ **sched_group_barrier 重标定在这里恒等于零**:PV 簇的组只覆盖 8 条 MFMA 里的 6 条、48 条 VALU 里的 27 条,
  改成 8×{1 MFMA + 6 VALU} 全覆盖后 **ISA 逐字节相同**。⇒ 该簇是**依赖约束**(cvt 要等自己那 8 条 exp)
  不是组约束;动 IGroupLP 之前先确认组是不是 binding(改前改后 diff ISA,一次 10 秒)。
- ★ **r6「carry-pack 五次必崩」正式销案**:整个 tile 的 exp2 + `llvm.FPTruncOp` 提前一个簇做,
  SNR **51.61229610443115 逐位不变 / det=True / 22 组全 PASS / 8-wave 组 rel_l2 仍是 0.0027**。
  ⇒ r12 定位 + r17 换 intrinsic 之后,这一族的失败模式只剩性能与 spill,不再是 −13 dB。

#### ★★ 2026-07-29 r25:占用率斜率(同码 wpe A/B)与「5 waves」目标的算术纠错
- **deployed 代码同码 A/B**:wpe=4(alloc 128 / spill 2)= **1213.7**;wpe=3(alloc 130 / spill 0)= **1118.6**
  ⇒ **+8.50%/档**,复现 goal 记的 +9.1%/档量级。这是 wpe 当量具的第二次干净读数(第一次 r24)。
- ★ **纠错:8-wave CTA 下 `alloc ≤102`(5 waves)买不到任何东西**。占用率的粒度是 **WG**:一个 8-wave WG
  给每个 SIMD 放 2 个 wave,所以 waves/SIMD 只能是 **2/4/6/8**。`floor(512/102)=5` 那一档在这个 CTA 形状下
  向下取整回 4。32 rows/wave 这条路的下一档是 **6 waves ⇒ alloc ≤85**(现 128,要省 43 个,而 v_o 32 + Q 16
  + 载荷 P 16 已经 64,不可能),或者换 4-wave CTA(要 LDS ≤32768 且放弃 GQA 合并)。
  ⇒ **给「降到 N 个寄存器」定目标前,先用 CTA 波数把 `floor(512/alloc)` 向下对齐到可达档位**。
- ★ **工作树回归第二次发生**:r20 的 `fold_pv_barriers`(2 barrier/iter)又一次从磁盘上消失(r23 已救过一次)。
  判据是 ISA 主循环 **`s_barrier` 计数(4 = 丢了 / 2 = 在)**,bench 只差 0.3% 看不出来。恢复后回文
  A/B(B A A B B A)= 1214.4/1213.9/1214.9 vs 1210.7/1209.8/1210.7,**+0.31%**,区间不重叠。
  ⇒ **每轮开工第一件事是对 ISA 指纹**(本 kernel:vgpr 128 / spill 2 / sgpr 48 / LDS 34048 / barrier 2 /
    span 244 / s_nop_stall 18 / mfma 32+8),别只信 bench 数字。

### ★★ 2026-07-30 r26 收官:exp/softmax 的簇间落点这一族**已探尽**(五个实测点,两个方向都关上)

r25 定案「P·V 是饱和簇、QK 才有空档」只对一半。r25 只测了"往 P·V 搬"和"在 QK 内编织";r26 补了反方向
——把 P·V 那 16 条 exp **全部搬去 QK 簇头 = −0.27%(中性)**。P·V 若真饱和,卸掉一半 VALU 该有收益,实测没有。

⇒ 正确表述:**QK 簇确有空槽(往里加 16 条 exp 免费),而 P·V 现有的 16 条也已被它的 MFMA 影子完全掩盖
(搬走不赚)。部署的 16/16 拆分位于一个平坦最优点,P·V 的边际容量恰好 ≈16 条 exp —— 不超过免费,
超过(+16 条)= −2.3%。** 别开第七轮。

⚠ **r25 ① 那个 −0.33% 的读数捆绑了第二个变量**:它在搬 exp 的同时把 loop-carry 从 32 f32 压成 16 packed dword
(wpe3 下 vgpr 130→144 / wpe4 spill 6)。纯落点的单变量读数是 r26 的 **−2.3%**;照 ① 定价会低估"往饱和簇搬 VALU"的代价。

### ★ 128 预算是 **prologue/epilogue** 设的,不是主循环(r26 逐块直方图)

逐块 max-VGPR:设预算的两块是 **prologue 与 epilogue(各 127)**,主循环只有 **125** —— 主循环只剩约 2 个裕度。
⇒ 想压分配,要么动 prologue/epilogue 的 Q 载入与 O 写出,要么动 `rows_per_wave` 这种全局常驻集参数;
**压主循环窗口在这里恒等于零**(这也是 r10/r13/r14 三轮"活跃度降了、分配纹丝不动"的最终解释)。

定量刻度(接在 pitfalls/01 的 spill 阶梯后):8-wave CTA 上把 `waves-per-eu` 直接设到 6 档(alloc 80)= **spill 358 dword / −90.6%**。
⇒ 6 waves 与当前常驻集差的不是"再省几个",是**约 48 个寄存器量级**,只能靠 `rows_per_wave` 减半这类结构改动。
★ 这类跳档实验是**定价探针**,不是候选臂(r1 在 4 档、r17 在 5 档、r26 在 6 档各踩一次,曲线已连成)。

### ❌ gfx950 上"让 RA 用 AGPR 分流寄存器压力"是无操作(r26 补 ISA 证据)

三个 LLVM flag(含 `amdgpu-mfma-vgpr-form=False`)产生**逐字节相同的 ISA**,`.agpr_count` 仍是 0。
与 pitfalls/01 记的 mxfp4 结论同源(AGPR 与 VGPR 共用同一 512 池),现在 fwd 侧也有了 ISA 证据。

### ★★ fp8 的「2× 吞吐」只对**一种** MFMA 形状成立(r23 实测,推翻纸面定价)

goal/KB 都按「`v_mfma_*_fp8` 是 bf16 的 2× ⇒ −27% wall」给 fp8-PV 定价。**两种 fp8 形状表现相反**:

| 形状 | 实测 | 结论 |
|---|---|---|
| legacy `32x32x16_fp8_fp8` | **32 拍**,与 bf16 **同速率**(PMC 28.8 cyc/inst @1.69e8 条) | **零吞吐收益** |
| native `32x32x64_f8f6f4` | 64 拍换 4× 的 k ⇒ MFMA-busy **−22.2%** | 唯一能兑现 2× 的路 |

且 −27% 的估算忽略了宽操作数带来的 fold/accumulate 脚手架:实测 **120 spill dword / 439.8 TF**。
⇒ 引用 fp8 吞吐时**必须写明形状名**;只写 "fp8 2×" 会把 legacy 形状也算进去。

**fp8-P 的精度侧(r19)**:`_FMAX0` 下 P ≤ 1 ⇒ tile max ≈ row max,e4m3 的 **per-tile scale 与 per-row scale 同为 33.41 dB**
⇒ **fp8-P 不需要任何 per-row 缩放机构**,代价纯粹是 SNR;e5m2 比 e4m3 差 6 dB。
折算到本 kernel(bf16-P 实测 51.61 dB)≈ **27.6 dB** —— 低于 34 dB 的正确性门,与 bwd 侧 GEMM1-fp8 判负同因。

### fwd 专属 flydsl 坑
- **`bench.sh` 是 rsync 到远端容器里跑的,本地 shell 的环境变量传不过去** ⇒ **env-gated 的 A/B 在
  campaign 里完全无效**(会静默全跑默认臂),切臂只能改源码默认值。曾因此产生一批无意义的"隔离"读数。
- **别用 `for v in vec` 迭代 `Vec`**:tracer 无限生成 IR,进程 22 分钟不退出、无报错、无输出,还占死串行 job 队列。
  取元素一律 `[as_mlir_value(v[r]) for r in range_constexpr(N)]`。
- `llvm.call_intrinsic` 的 `results_` 是**单个 Type 不是列表**,且 `T.f32` 不是 `ir.Type`(要 `ir.F32Type.get()`)。
- `remote.sh` **不清 FlyDSL JIT 缓存**(只有 `bench` 分支清),经它做的任何 ISA/PMC 都要自己
  `rm -rf /root/.flydsl/cache`,否则读到上一次的 `21_final_isa.s` —— 本 campaign 被骗过两次(vgpr 读错档)。

## ★宽 MFMA(32x32x16 换 16x16x32)减不了 LDS 读——算术级结论,别再实现一遍

2026-08-04,gpt-oss fused attn bwd(gfx950,D64)实测/推导:

- **LDS 读的条数由数据量决定,与 MFMA 形状无关**:`ds_read_b64_tr_b16` 恒给 64 lane × 8 B、
  `ds_read_b128` 恒给 64 lane × 16 B。
- **4 条 16x16x32 组成的 2×2 patch 与 1 条 32x32x16 覆盖同一个 32×32 输出块、用同一批 fragment。**
  逐条核对 dQ GEMM:两种形态都是每 wave 每 pass 64 条 tr。换宽 MFMA **唯一**的变化是
  MFMA 指令数减半(1536→768),而指令发射通常远不是瓶颈(该核 0.14 LDS + 0.06 MFMA 每 CU-cycle)。
- ⇒ 想减 LDS 读只有三条真路:**每 wave 摊到更多 MFMA 的 fragment 复用(加大 N-tile)**、
  **跨 head/跨 tile 共享 fragment**、**加宽写(layout 置换)**——前两条都要寄存器,通常就是那道墙。

**软最大值一侧还有一条结构性禁令**:softmax 的 P/dS 若留在寄存器里喂下一个 GEMM 的 B operand,
它就锁在上一个 GEMM 的 C-layout 上(16-wide:lane0-15 持 n=0..3、lane16-31 持 n=4..7)。
而 32x32x16 要求**一个 32-lane 半区内所有 lane 持同一组 k**,k 的置换自由度救不了
(k→列 的映射必须在半区内 lane-uniform)⇒ **该 GEMM 根本换不了宽 MFMA**,除非加 cross-lane
shuffle 或多走一次 LDS。设计 attention bwd 的 tile 形状时先查这条,再排工作量。

## 2026-08-19 gpt-oss D64 full-causal fused bwd:掩码 q 循环是被忽略的那一半

形状 B=4 Sq=Skv=8192 Hq=64 Hkv=8 D=64 bf16,fused dkdv,occ=1(vgpr 459/512)。
本节的四条全部来自同一份 ISA + 同一把回文尺,可直接复用到任何 causal attention bwd。

### ① 每 trip 的 cycle 去向(ISA 静态 + 形状算术,rocprofv3 在该 host 不可用)

body 5.619 ms / 540672 trip / 256 CU ⇒ **5589 cycle/trip**,其中 MFMA 只有 2560(46%)。
未掩码 q 循环(`.LBB0_46`,占 94% 的 trip)一次迭代 = **一个 q block × 8 个 GQA head**,
4456 条指令 / 1280 MFMA。折成每 trip:

| 家族 | 条/trip | cycle | 备注 |
|---|---|---|---|
| `v_mfma_f32_16x16x32_bf16` | 160 | 2560 | 矩阵管线,16 cycle/条 |
| `v_exp_f32` | 64 | **1024** | ⚠ 这 1024 是按 quarter-rate(16 拍)算的,**实测是 half-rate 8 拍 ⇒ 512**(见 §623);1 条/元素,**条数已是下限** |
| `v_cvt_pk_bf16_f32` | 72 | 288 | P 32 + dS 32 + dQ partial 8,**已是下限** |
| `ds_read_b64_tr_b16` + `b128` | 88 | 352 | |
| `v_pk_mul_f32`/`v_mul_f32` | 37 | 150 | dS = P·dP,1 次乘/元素,**下限** |
| `s_nop`(hazard) | — | 70 | |

⇒ scale 早已折进 K 的预缩放、`-log2e*lse` 早已折进 GEMM1a 的 C-init,**算术侧没有可删的指令**。
剩下的全部是**排程**问题。

### ② ★ `v_exp_f32` 看着占 18%,实际只暴露 4% —— 用 no-exp 探针,别用 cycle 账

把 `llvm.amdgcn.exp2.f32` 换成恒等(值错、逐位确定性保留、指令数与地址链不动)实测
**−4.05%**(5.5701 对 5.8051,各 3 轮回文)。1024 cycle/trip 里只有 ~226 是暴露的,
**78% 已被 `iglp_opt(2)`(MFMAExpInterleave)藏进 MFMA 影子**。
⇒ 源码注释写着 "VALU/exp2-issue-bound" 也不能直接当账记;**先量暴露量再决定值不值得排程**。

### ③ ★★ 掩码(对角)q 循环:6% 的 trip,~1% 的 wall,而且是最容易漏的那一支

body 为 causal 生成**两份** q 循环。同样 1280 MFMA,掩码那份 **6338** 条指令、
`s_nop` 等待 **779** cycle,未掩码那份 4456 / 560。差额里最大的两块是
`v_cmp+v_cndmask` 各 448(掩码本身,不可免)和 **`v_add_f32` 512 + 它拖进来的 hazard nop**。

那 512 条加法的来源:窗口形状把 `-log2e*lse` 折进 GEMM1a 的 C-init,而 **full-causal 的掩码 tile
走的是零 C-init + 每元素一条 identity fma** 的老路(`WIN_FOLD = window_left >= 0`)。
把它对 full causal 也打开:

| | 掩码循环指令 | `s_nop` 等待 | 未掩码循环 | vgpr | spill |
|---|---|---|---|---|---|
| 折之前 | 6338 | 779 cyc | 4456 | 459 | 0 |
| 折之后 | **5402** | **222 cyc** | 4455(逐字节同) | 463 | 0 |

`ceil8(459) = ceil8(463) = 464` ⇒ **共驻线不动**,co-resident dqred 照旧装得下。
gptoss_full **14 对回文 / 3 个 session / 14 of 14 同号 / mean −0.98%**;
八格权威尺三读 gptoss_full 5.7108 / 5.6922 / 5.6961(sp 1.0378/1.0411/1.0404),
full-causal 的 5 个 guard 全部 ≥ 冻结基线(d64_hq32_b4 1.005~1.009、d64_hq64_b1 1.009~1.016、
d64_hq128_16k 1.007~1.017、varlen_d64_u 1.017~1.037、d128_hq64_b2 1.002/1.007/0.978),
两个 windowed guard 的二进制**根本没变**(它们早就折了)。SNR 47.8/48.8/48.8 与 48.1/49.0/49.1。
⚠ 折进 C-init 改的是 fp32 累加起点 ⇒ **与旧版不是逐位相同**(det 门是"同输入两次 torch.equal",仍过)。
⇒ **判据**:causal attention bwd 只要有独立的掩码 q 循环,先去对一遍它和 unmasked 那份的指令表;
按 trip 占比(6%)估收益会把这类改动直接扔掉,而 hazard nop 让它的单价远高于占比。

### ④ 已定价为负 / 为零的同族杠杆(别重走)

- ❌ `g3_defer=True` @D64 **+3.5%**;再把推迟的 GEMM3 挪进 q-half 的 exp2 窗口 **+5.0%**。
  第二个 dS slot 的寄存器/LDS 在 4-wave body 上没有兄弟 MFMA 段可摊。
- ❌ Q/dO 每 band 重读虽是该核 68% 的 DRAM 流量(8.9 GB),**别名探针 +1.6% 更慢** ⇒ 无池可收。
- ⭕ `g1_ks_outer` / `g3d=4,8` / `qdesc_r` 2,8,16:n=5~8 回文全部落在 ±0.4%。
- ❌ `qpf_at=0` **+4.2%** / `=1` **+3.1%**(D64 的最优发射点就是 2);`=3` 直接编译不过(支配关系)。
- ⚠ `g3_kreg=True` 与 `g3_at=*` 在 D64 上**编译出逐字节相同的 ISA**,它们在一次 3 轮 sweep 里读出
  −1.05% 的假赢,是最有价值的噪声标定(见 pitfalls/02)。**订正(2026-08-19)**:`g3_kreg` 的
  ISA 不动**不是因为未接线,而是因为它已经是部署值** —— D64 融合分支的 `G3_KREG` 默认就是 True。
  `nokreg` 反向探针把它关掉,未掩码循环**多出 256 条 `ds_read`/q-block trip**,证明它真在生效。
  `g3_at` 才是真的无接线点(只作用于 `g3_defer` 的推迟路径,D64 走未推迟)。

### ⑤ ★★ 掩码 q 循环的第二刀:**按 wave 分类**,而不是把掩码算完再选

④ 把掩码循环从 6338 压到 5402 之后,剩下的大头是 `v_cmp`+`v_cndmask` 各 448/八头 trip。
r4 的出口笔记想**缓存**这批掩码(它 head-invariant,理论只要 64 组)。实测走了另一条更便宜的路:
**根本不用算**。BLOCK_KV=256 / 4 wave ⇒ `ROWS_PER_WAVE_KV = 64 = BLOCK_Q`,而掩码 q-block 的首行
与 band 首 kv 行都从 `kv_start` 出发,差恰好是 `(m-w)*BLOCK_Q`(causal offset 相消)⇒ 一个 wave
必然落在三类之一,**没有半对齐的情况**:

| wave | 状态 | 该 wave 要做的事 |
|---|---|---|
| `w < m` | 每一行都在因果边下 | **完全不掩码**(= 未掩码循环的代价) |
| `w == m` | 对角 wave,16-tile 按 `nt` vs `mt` 分 | `nt>mt` 整块 P=dS=0 直接不发 MFMA;`nt==mt` 掩;`nt<mt` 清 |
| `w > m` | 死 wave | 只发布零(原有的零发布臂) |

四个 wave 被 barrier 锁在同一 head-step ⇒ q-block 的代价是**各 wave 的 MAX**。分类后对角 wave 的
in-branch MFMA 128→88,该 block 的临界路径 675→**560** 条(560 = 内部 q-block 的原价,即 below 臂
成为地板),`m=0`(只有对角 wave 活着)时→**430**。**dq/dk/dv 与旧路径逐位相同**(D64 与 D128 都验),
因为被删的 MFMA 的 B 操作数是精确零。gptoss_full **18 对回文 / 3 session / 14 of 18 同号 /
三个 session 均值全为负 / 合计 −0.59%**;权威尺四读 step_ms 75.079/75.515/75.479/**74.912**,
g8k 1.03918/1.03283/1.03060/**1.04258**。

⚠ **这条路的真实成本是寄存器,而且形式比数量重要**:先写成**嵌套** if(外层 live/dead、内层
below/diag)读出掩码循环 219 AGPR、vgpr **476 > 464 granule** —— 两个臂都产出新值,分配器必须
同时留住两套 yield。改成**三个顺序的 wave-uniform if**(每个 if 的 else 臂原样 yield 传进来的
累加器,能被平凡合并)后 vgpr **460**、spill 0、LDS 不动、掩码循环还少了 468 条指令。
⇒ **判据**:在 occ=1、accumulator 吃掉 128+ dword 的 body 上做多路特化,**用顺序 if 串(identity
else)而不是嵌套 if**;代价只是每 head-step 多一条 wave-uniform 分支,收益是不必为分支结构额外
付 13 个 AGPR dword。副作用:未掩码循环被 LLVM 重排,4455→4478(+23,`s_waitcnt` +31 / `s_nop` −19),
`maxa` 从 187 升到全函数统一的 203 —— 净收益仍为正,但这条漂移是下一轮可去的余量。

### ⑥ ❌ `dkdv_g2d=2`(GEMM2 的 dt 预取环加深一组)@D64 **+1.9%**,6/6 回文全负

方向正是"给 LDS 读提前一个 MFMA 组做 double-buffer",ISA 也如预期:未掩码循环 `s_waitcnt` **少 26 条**。
但它 wall 上稳定 **+1.9%**(六对回文,单对 +1.2%~+2.7%)。D128 上同一个 depth=2 是部署值。
⇒ **`s_waitcnt` 的条数不是代价口径**;把等待挪走的同时加宽了读突发,MFMA 发射被顶开的损失更大。
真正要看的是**两次等待之间的 MFMA 连跑长度**,不是等待计数。
**订正(2026-08-19 晚)**:连跑长度也不是。见下节 ⑦。

## 2026-08-19(晚) 同一 body:LDS 覆盖距离才是判据,`s_nop` 不是池子

### ⑦ 区域调度 mutation 整族判负,以及它给出的新仪器

directive 要求"零指令、零寄存器增量的纯重排,攻 500 cyc/trip 的 `s_nop` 池"。把
`iglp_opt(2)` 换成手搭的 `sched_group_barrier` 流水(同一个 LLVM mutation,互斥),
gptoss_full 上 6 轮位置均衡轮转、每读 min-of-40、独立进程:

| 臂(流水规格) | 指令 | `s_nop` cyc/trip | MFMA 连跑 | **LDS 覆盖** | wall |
|---|---|---|---|---|---|
| 部署 `iglp_opt(2)` | 4478 | 499 | 1.56 | **32.4** | — |
| `"r24,m32"` | 4447 | 512 | 1.59 | 32.8 | +0.90% (1/6) |
| `"m2,t2,v4"` | 4424 | 415 | **1.83** | 26.4 | +0.80% (2/6) |
| `"m2,t2,v6"` | 4425 | 417 | 1.83 | 26.4 | +0.42% (2/6) |
| `"r1,m2,t2,v4"` | 4393 | **349** | 1.91 | **19.7** | **+3.48% (0/6)** |

- **新仪器:LDS 发射→退休覆盖距离。** 按 `s_waitcnt lgkmcnt(N)` 的队列语义模拟退休,记每条
  `ds_read`/`ds_write` 从发射到被退休相隔多少条指令(实现见 Primus-Turbo `_isa_run.py`)。
  它与 wall **单调对应**,单价约 **0.2~0.3% 每条覆盖指令**;而指令数、`s_nop` cyc、MFMA 连跑
  长度三个量在这张表里**全部与 wall 反向**。`"m2,t2,v4"` 四个静态量齐变好仍 +0.80%。
- **`s_nop` 在 occ=1 上不是可加的池子。** `"r1,m2,t2,v4"` 抽掉 150 nop cyc(trip 的 2.7%)
  兑现为 0。一个 wave/SIMD 上 hazard nop 与旁边的 lgkmcnt stall 很大程度是同一次 stall,
  静态分析数了两遍。**别按"热循环有 N cyc 的 s_nop"去立一个 N cyc 的项目。**
- **点名一个指令类 = 钉住它。** 想让读提前不能写 `r1`(每组塞一条读,正好钉在消费点旁,
  8 条内退休的比例 8%→29%),要写一个**大于区域读数的领头组**(`"r24,m32"`)。
- **换掉 mutation 本身要付 0.9%。** `"r24,m32"` 保住覆盖、指令 −31,仍 +0.90%,因为
  `iglp_opt(2)` 同时在藏 78% 的 exp 链。⇒ 手搭流水的臂起手落后 0.9%。策略 0 顶穿寄存器
  (vgpr 476/nop 630),策略 1 在带 `ds_write` 的区域触发 `AMDGPUIGroupLP.cpp` 断言(不可达);
  把调用从"每 q-half 一次"减到"只在一个 half"也不过门(469 顶穿 / nop 541)。
- **源码顺序形式没有余量**:NT=2、每 half 2 个 mt,部署的消费顺序已经是距离最大的排列。

### ⑧ 暴露的 dQ fold 会随 body 变快而**变大**,而"切更细"整族判负

`nored` 减法探针:首次定价 −4.9%(0.289 ms),body 快 4% 之后复测 **−5.84%(0.333 ms)**,4/4 配对。
fold 字节没动,是**可藏它的 body 变少了**。⇒ 每一次 body 的胜利都让这一项相对变重,
所以它的排序会自己往上爬,要定期复测而不是沿用旧定价。

**q_split 的两因子实验(值得复用的做法)**:q_split=8 早有"判负 +1.32%"的记录,但那是配默认计划
(7 个 chunk)测的,把"边界项"和"尾部项"混在一起。固定 3 个 chunk 再比:

- `[1,6,1]`@8(尾 1/8) **+8.01%**、`[2,4,2]`@8(尾 1/4) **+8.11%**,各 0/4。
- 两个尾宽**同价** ⇒ 尾部只值 ≤0.1%,那 +8% **全是每 split 的固定成本**
  (dk/dv slot 数翻倍 + 每个 band 每个 split 多一次 K/V 驻留)。

⇒ 源码里"每个 tail subset 值 0.194 ms"这类系数是**在固定 q_split 下按计划拟合的,不能外推到
改 q_split**。要拆开这类混合结论,就固定 chunk 数、只动尾宽,再加一个尾宽不变的对照臂。

### ⑨ LDS 空着 74 KB 且免费,但喂它的流水不免费

body 用 86016 B / 160 KB,占用率由寄存器(460/512)锁死 ⇒ 多花 LDS **零代价**。这场 campaign
花了七轮想**省** LDS,没人注意到它可以**花**。但 `LDS_SLOTS` 被钉在 1 是因为 Q/dO 走 VGPR 中转
(`Q_PREF`),第二个 slot = +32 个在飞 dword,而 464 granule 只剩 4 个。`dma_grp>1`/`pf_ring`
另有 builder assert。⇒ **"某个资源还有余量"不等于"有可用杠杆",要先找到不吃寄存器的喂法。**

### ⑩ split-K partial 往返的**两端要相反的 cache policy**,而 UMC 表能把这件事量出来

2026-08-20,gpt-oss D64 fused bwd(B=4 S=8192 Hq=64 Hkv=8,`_pw_probe.py bwd 5 <arm> 5`,
每读一个独立进程、两臂交替)。这条卡此前只有 D128 store 侧的一行判负
("分块存储 cache policy 全谱判负",00-decision-index 第 246 行),读 side 从未定价,
**机制**也没写。两端各自定价后是一对反号的结论:

| 端 | 部署 | 换成另一种 | 墙 | J/pass | **UMC 点** |
|---|---|---|---|---|---|
| fold 的 partial **读**(dqred) | non-temporal | cached | 5.8232 / 5.7802 对 5.7146 / 5.7309 | 8.155 / 8.126 对 8.025 / 8.055 | 25 → 25(不变) |
| body 的 partial **写**(dkdv G3) | cached | non-temporal | 5.7364 / 5.7545 对 5.7021 / 5.7278 | 8.109 / 8.089 对 8.016 / 8.076 | **25 → 27(+1.3 GB)** |

- **读端要 NT**:那 4.4 GB 读完即死,留住它只会挤掉 body 自己的 Q/dO 与 lse ⇒ cached 多花
  0.1 J 而 **UMC 一点不动**(不是多搬字节,是把别人的驻留挤没了)。⇒ 这也是本 kernel 第一个
  "MALL 是被**争用**而不是仅仅太小"的直接证据:它装什么是有价值的选择。
- **写端要 cached**:store 每条只覆盖 64 B(D 轴被 permute 过就为了从 32 B 提到 64 B),
  cached 时两条在 L2 合成一条 128 B 行;NT 之后合并没了,远端要为半行做 line fill ⇒
  **DRAM 流量涨 ~1.3 GB(UMC 25→27)**,墙 +0.45%。
- ⇒ 推广:凡是"写一次、被另一个 kernel 读一次"的 split-K workspace,**写端按行合并选 cached、
  读端按不污染选 NT**;判据不是速率而是 UMC 点数,它响应的是 DRAM **事务**而不是 issued 字节,
  所以能直接看见写合并有没有丢。

### ⑪ band 交错(ILV)的收益按"**同一 q 行的并发写者数**"缩放,不是按 head dim

同一份 `[band/ILV][B][Sq][ILV][D]` 交错在 D128(64 band 并发写同一批 q 行)值 +1.3%
(见 ④),在 **D64 full causal 上三次测量全部为平**:早期 −0.14%、QDESC 打开后 mean 5.9721 对
5.9479、本轮能量表 5.7127 ms / 8.046 J 对部署 slab 的 5.7035 / 8.023(加 group pad 5.7105 /
8.041)。原因不在 head dim 而在**并发写者数**:D64 一条 q 行只有 8 个 band 在写,而这 8 个 band
的 8 个 kv head 的 WG 已经把那一页填满了,布局没有散射可收。
**配 QDESC_R=1 也救不回来**(把全部共驻 band 压到同一个 q block = 制造最大散射):5.8236 ms /
8.160 J,且 **sclk 反而升到 2199(功耗 1396 W 未贴顶)**——本 part"改动没到达字节"的再定时特征。
⇒ 排这条杠杆前先数"一条 partial 行有几个并发写者",<= 8 就别指望它。

**⑩ 补一条:cache 驻留在这块卡上是台阶而不是斜坡。** 同一个 fold 用 `band_ring` 把读脚印夹到
n×268 MB(dQ 算错、发的 load 完全相同)测驻留曲线:268 MB **7.662 J / 17 UMC 点**、1.07 GB
8.101 / 26、2.14 GB 8.132 / 27、全量 4.29 GB 8.127 / 27。⇒ **一 GB 以上全部按全 miss 计价**,
只有 <= MALL(256 MB)那一档掉下来。所以"把工作集缩小一点点去换驻留"这类提案在本卡上没有中间态:
要么进 MALL,要么等于没做。定价一条驻留杠杆时先量这条台阶(四个探针、约 5 分钟),
再决定要不要为它付 launch 数/carry 字节。

#### 2026-08-31 r7(hd64 **fwd**):1 WG/CU 这堵墙的两侧读数 + 三个新的 ❌
- ★★ **第二个共驻 WG 值 >4%**。两条独立路线,`vgpr 128 / spill 4 / LDS 34048 / 热循环 240 条`
  **全部不变**,只动占用率:`Q_HEADS_PER_WG=4`(16-wave CTA、4 waves/SIMD、1 WG/CU)= **−4.3%**;
  `waves_per_eu=3`(vgpr 132、spill 0、1 WG/CU @3 waves/SIMD)= **−9.6%**。16-wave 那一臂
  **拿满了 K/V sector 减半的 +1.75%** 仍净亏 4.3% ⇒ r21 记的「`Q_HEADS_PER_WG=4` 判负 = prologue
  完全暴露」在本树复现且更贵。⚠ 让 16-wave 编得出来只要两处:`dma_wave_reps = max(1, …)` 与
  DMA 行号取 `% SMEM_N_RPT`(波数多于 LDS 行时两个 wave 填同一行同样的字节 —— r21 已测该冗余免费)。
- ★ **`bytes/score = 4D / (NUM_WAVES × ROWS_PER_WAVE)`**:与 `BLOCK_N` 无关、与 sharer 数无关。
  所以 `Q_HEADS_PER_WG` 2→4 **配上** `block_m` 128→64(为了保住 8 waves)**买不到任何 sector**,
  ISA 逐位相同。它只买到因果粒度(64 行块 1.0078× vs 128 行块 1.0156× 的精确因果功 = 0.77% 少算),
  实测 screening +0.28% / **计分尺 −0.08%**。
- ✅ **把 r5 的「不手写 lgkm drain」规则推到 prologue + epilogue**:再删 8 行,
  `d64_gptoss_b4` **+0.32%(4/4 回文对为正)**、composite **+0.19%**,两门逐位相同,指纹不变。
  ★ **安全判据是 WAR 侧的「≥2 barrier 余量」,不是「后端会补 lgkmcnt(0)」**——后者只保证可见性;
  真正的风险是本 wave 在飞的 `ds_read` 读的那块 LDS 被别的 wave 的 DMA 覆写。逐 buffer 列表核对
  (本核 epilogue 五处读全部有 2 个 barrier 余量,正是 `sink_drains` 依赖的同一个不变量)。
- ❌ **Q 的 global load 标 `nt`** = **−0.95%**(O 侧 +0.11% 噪声内)。Q 只读一次,但**同一 CTA 的多个
  wave 共享它的 128B 行**;"只读一次" ≠ "没有复用"。`BufferCopy128b(cache_modifier=2)`,`nt` 已在 ISA 核实。
- ❌ **靠源码顺序并起 prologue 的两个 HBM round trip**(Q 与 tile-0 K DMA 今天是严格串行)= **−0.22%**,
  ISA 证实重排真的生效。⚠ 顺带把 tile-1 K / tile-0 V 的 DMA 也提前 = **spill 4→25**,ISA 门拦下。
- **本轮把三个池子钉死了**:epilogue rendezvous 上界 **+0.39%**(13 个 barrier 全换成 `sched_barrier(0)`
  的减法探针),固定每-WG 开销 **≈2.2%**(S=8192 vs 16384 的 TF/s 外推:1072.2 / 1190.5 / 1203.7 / 1255.0
  @ S=4096/8192/16384/32768),IGroupLP 的 MFMA↔exp 交织比(exp/MFMA ∈{1,2,4,6}、VALU∈{2,8}、pairs=12)
  **全部 ±0.15% 以内** —— 48.1% 的 coexec 不是靠 prescription 能动的。
- LLVM 选项复扫:唯一还活着的是 `misched-cluster=0`(243 条 / spill 4,+0.18%/+0.10%,在门槛之下);
  `misched-regpressure=0` / `greedy-regclass-priority-trumps-globalness` / `misched-postra=1` /
  `amdgpu-igrouplp-exact-solver` 指纹逐位不变(惰性);`greedy-reverse-local-assignment=1` spill 8、−1.5%;
  `misched-postra-direction` 与 `amdgpu-disable-power-sched` 是**非法 flag**(编译不出 ISA,别当成 0%)。

### r8 fwd(gpt_oss d64 前向,第八轮)—— MFMA 是寄存器分配的锚;setprio 窗口保护的是 VALU 不是 MFMA

- ❌★★ **把 ones-A 的 row-sum MFMA 从 MFMA 管线搬到 VALU(`v_dot2_f32_bf16`)——整族判负,
  而且真正的代价不是 dot2。** `llvm.amdgcn.fdot2.f32.bf16` 走 `llvm.call_intrinsic` **能用**,
  选出干净的 `v_dot2c_f32_bf16_e32`(无 `s_nop`、无额外寄存器),MFMA 管线周期 1152→1024 **完全符合预期**。
  但 **spill 4 → 94**,热循环里 16 个 loop-invariant 的 Q dword 被逐出、每迭代 `scratch_load_dwordx4` 重载。
  ★ **判据行**:把 row-sum **整个删掉**(值算错,纯诊断)= **spill 118 —— 比 dot2 版还差**。
  ⇒ **不是 dot2 贵,是"少了 8 条 MFMA"贵**:那 8 条 `v_mfma_f32_16x16x32` 在这具 128-VGPR 的
  body 上**是寄存器分配的锚**。换 sched 策略(`minregexcess`/`max-occupancy`/`max-ilp` → 94/94/92)、
  关 IGroupLP(88)、每条 dot2 后加 `sched_barrier(0)`(**ISA 逐字节相同**)、每 pack 一个局部累加器(**191**)
  —— 没有一个能动它。⇒ 想重开这族,必须造一个**仍保留 8 条 MFMA 但让它们做有用功**的臂。
- ⚠ r4 记的"dot2 arm 达到 spill 0 / vgpr 126"只在**带副作用的 inline asm**形式下成立(它顺带是 32 个硬栅栏,
  代价是 30 条 `s_nop`);换成 intrinsic 后 spill 直接 94。**别把 asm 形式的分配结果当成 intrinsic 形式的预测。**
- ★★ **`s_setprio` 窗口保护的是 softmax 的 VALU run,不是 MFMA run。** 同一轮四个方向实测
  (8-wave / 2-WG,screening 中位数):把窗口**提前释放**(P*V 的 4 条 MFMA 之后、softmax 之前)= **−3.5%**;
  给纯 MFMA 的 QK cluster 加优先级 = **−0.70%(prio1)/ −0.45%(prio2)/ −1.9%(prio3)**;
  给 C1/C5 的 `softmax_half` 加 prio2 = **−0.9%**,加 prio1 = screening +0.27% 但**计分尺 7 对回文 = −0.03%**;
  把窗口**延后**到下一个 memory cluster = 0.0%。⇒ 当前放置(整个 P*V compute cluster = prio 2)是局部最优,
  且"该保护谁"的规律是:**VALU 密集区要保护,纯 MFMA 区不要**。
- ★★★ **修正本 KB 与 goal 里的发射预算算术(错了 4 倍)。** r4 记"285×4=1140 槽 vs 1152 MFMA 周期 = 99% 发射受限",
  goal §A3 记 81% —— 两者都拿**每-SIMD 的发射条数**去比**每-WAVE 的 MFMA 管线周期**。
  一条 `v_mfma_f32_32x32x16_bf16` = 16384 MAC,SIMD 矩阵管线 512 MAC/cyc ⇒ 占管线 **32 拍**,而 **4 个 wave 共享一条管线**:
  `MFMA 管线/SIMD/迭代 = 4×1152 = 4608`,`发射槽 = 4×240 = 960` ⇒ **发射占用率 20.8%**。
  (PMC 对得上:`SQ_INSTS_MFMA` 8.52378e7 = 8192 WG × 32.5 迭代 × 8 wave × 40 条,28.8 拍/条 = (32×32+8×16)/40。)
  ⇒ **这具 kernel 上"减指令条数"几乎不值钱**;21.1% 的 MFMA 管线空转不是发射口问题。
  闭环:VALU 2608 拍/SIMD-迭代,coexec = 48.1%×MFMA_busy = 1181 拍 ⇒ **未重叠的 VALU ≈1427 拍,
  而 MFMA 空转 1233 拍** —— 两者在 15% 内吻合,**MFMA 空转就是没盖住的 VALU**,其中 **64 条 `v_exp_f32`
  ≈512 拍 = VALU 的 78%**。下一手是**改 dual-wave 的相位**,让一个 tile 的 exp2 落进**伙伴 wave-group 的 P*V 阴影**里。
- ⚠ **共享节点的噪声会整场吃掉 0.3% 级判据。** 本轮同一 session 内 7 次对照 `fwd64` 跨度
  **1.0118–1.0268(0.9%)**,goal 标定的 0.32–0.78% 不成立。同一个候选前两对读 +0.58%、七对读 −0.03%。
  ⇒ **0.3% 级的候选至少要 6 对回文、分 ≥3 批交替顺序**,两对为正不是证据。
- 再次确认(第五次):`K_HEAD=1`(spill 4→0)在本核 **+0.15%,噪声内**,两次重复彼此差 0.32%;
  `daz=False`(builder 默认是 True,此前从未试过)+0.16%/+0.11%,噪声内;
  `FWD_NOSGB`(删掉整个 IGroupLP prescription)**−0.48%**,prescription 值得保留。

### r9 fwd(gpt_oss d64 前向,第九轮 REPLAN)—— exp 的代价是相位不是速率;计分尺不是另一个 DVFS 世界

- ★★★ **softmax 的 exp 池 = +7.8%,但「让每条 exp 更便宜」整族判负。** 减法定价四臂
  (回文 / 一臂一进程 / 每臂 `rm -rf /root/.flydsl/cache /root/.flydsl/debug`,值算错纯诊断):
  64 条 `v_exp_f32`(部署)1.8363/1.8502 · **同样 64 条换满速 `v_mul_f32` = 1.8874/1.8892(−2.4%)** ·
  32 条 exp 1.7704/1.7685(+4.3%)· **0 条 exp 1.7041/1.6953/1.7109/1.7107(+7.8% = 1293.8 TF/s)**。
  **ISA 门**:0 条与 64 条两臂 `vgpr 128 / agpr 0 / spill 4 / LDS 34048` **逐位相同**、MFMA 条数不变、
  全核 `v_exp_f32` 312→120 ⇒ +7.8% 就是那 192 条指令,不是寄存器分配。
  ⇒ **代价不是 trans 单元的半速率**:把活搬到主 VALU 管线反而 −2.4%,
  **gfx950 上 trans 管线与 MFMA 的共发射优于主 VALU 管线**。这一条封闭多项式 exp2 / 查表 /
  任何把 exp 搬去主 VALU 的构造,并**第一次给出**「NV 风格软件 exp2 判负」的机制
  (~7 条 FMA 正好落在实测更差的那条管线上)。
  ⇒ **线性**(砍一半拿 56% 收益)⇒ 是**占用/重叠**问题不是**关键路径**问题,ILP/拉依赖距离是错的形状。
  ⇒ 闭环:省 0.1346 ms ÷ 520 次串行迭代 @ ~1.69 GHz = **437 拍/SIMD-迭代**,
  而 trans 工作 = 4 wave × 64 条 × 4 拍 = **1024 拍** ⇒ **57% 已被 MFMA 盖住、43% 没盖住**,
  与 `coexec/MFMA_busy = 48.1%` 和 MFMA 空转 21% 三方吻合;trans 只占 MFMA 预算(4608 拍)的 **22%**
  ⇒ **没有吞吐墙,差的是相位**。下一手 = dual-wave 相位偏移 / softmax deferral distance,
  ★ 方向用 `SQ_VALU_MFMA_COEXEC_CYCLES` 筛,不要用 wall 筛(相位改动在 wall 上只有零点几个百分点)。
- ★★ **修正本卡 r4 立的规矩「计分尺与微尺是两个 DVFS regime」。** 固定估计量(min-of-40)、
  只改热身长度:COLD warms=5 → 1.8403/1.8433/1.8435;HOT warms=20000 → 1.8528/1.8540/1.8553。
  **3/3 不重叠,只差 0.63%** ⇒ 5 次热身 DVFS 就落到稳态附近,**计分尺就在功耗墙这一侧**。
  r4 那两个读数(−0.29% vs +0.10%)相差 0.39%,在小 n 的合成噪声之内,不是 regime 分裂。
  ⇒ **`min-of-200` 微尺是合法筛子,别再做双份 bench**;分歧 >0.6% 按噪声处理,用更多回文对解决。
  ⚠ **别用 `rocm-smi` 直接采计分尺**:采到的 **2410 MHz @ 287 W** 是 GPU **boost 但空闲**
  (空闲 1420 MHz / 244 W;真跑该核 ~1390 W),一格只有 ~83 ms GPU,2 s 采样窗几乎永不落在里面。
- ★★ **为收「尾巴」重写调度骨架之前,先扫网格轴。** 固定每-WG 工作量只扫 batch
  (WG 数 = B×Hq/2×S/block_m):B=1..16 → 1146.7/1176.8/1184.1/**1193.6**/1193.3/1194.1/1194.1/1195.7 TF/s,
  **计分形状(B=4)已在渐近线** ⇒ 网格量化 + LPT 尾巴 **≤0.2%,persistent-CTA / work-stealing 整族封闭**。
  而扫每-WG 工作量(S 8192→16384 翻倍)只买到 **+0.7%** ⇒ 可摊薄固定成本 **≈1.4%**。
  ⚠ r7 记的「固定成本 ≈2.2%」是从 **S=32768 一个孤点**读出来的,该点不落在过 16384 的任何摊薄曲线上,
  标为未解释。⇒ **单条 S 扫把「网格」与「每-WG 工作量」两个轴混在一起了**(缩 S 同时缩两者),
  只有 batch 轴能把尾巴单独隔离。**池子大小要从曲线上 ≥2 个点读,别从端点读。**
- ⚠ **环境(新)**:9 个 counter 的多-pass `rocprofv3` 在本节点 **25 分钟没跑完就超时**
  (每个 pass 都重跑一次 JIT)。要 PMC 就**一次最多 4~5 个 counter**,且先预热 JIT 缓存再进 profiler。
- ⚠ **C1 补机制**:16x16x32 原子端口判负(−8.4%)**与 regime 无关**。既然计分尺只比稳态凉 0.63%,
  那份 sclk 收益在这把尺子上**是拿得到的**,它只是**盖不住周期代价**。
  ⇒ 该替换的净值由 **kernel 的 VGPR 余量**决定,不由「用哪把尺」决定。

## r11 fwd(hd64 dualwave forward,21 臂全负,树内零改动)

- ★★★ **LDS 行 stride 必须保住 16 B 对齐 —— 而 `SQ_LDS_BANK_CONFLICT = 0` 看不见这件事。**
  扫 K/V 的 LDS line-stride padding(stride(dword) = `256 + pad/2`):`KPAD=4` 与 `KPAD=12`
  给出**只有 8 B 对齐**的行,K 流的 `ds_read_b128` 被拆开 ⇒ **3.97 ms vs 对照 1.84 ms(−54%)**;
  所有 `pad ≡ 0 mod 8` 的档都健康。**这不是 bank 冲突**,冲突计数器全程为 0。
  ⚠ 健康档之间也不平:V 侧(`ds_read_b64_tr_b16`)部署的 32 是极大值,
  8=**−6.2%** / 16=−0.12% / 24=−0.43% / 40=−0.65% / 48=−0.42% / 64=−0.40%;K 侧 8(部署)最优,
  16=−1.2% / 24=−0.2% / 32=spill 27(门拒)。⇒ **新核上值得一次 ISA-gated 扫描,调过的核上别再扫。**
- ★★★ **「把 `ds_read_b64_tr_b16` 换成更宽的 `ds_read_b128`」这一族在 bf16 上不可构造。**
  三条独立理由:① gfx950 的转置读只有 `B64_TR_B16/B8/B4` 与 `B96_TR_B6`,**bf16 的宽度就是 64 bit**
  (见 methodology/07 的指令表);② 条数 = 字节 ÷ 宽度,而字节已在地板上 —— 每 wave 每迭代
  32 条 tr 读 = 256 B/lane,正好等于两个 V tile 摊到 lane 上的 256 B(K 流同理:16 条 b128 = 256 B),
  因为 q 行是 wave 的划分维度、**每个 wave 都要整块 tile**;③ 写入侧预转置被 DMA 粒度挡死:
  `buffer_load_dwordx4 … lds` 的一个 16 B chunk = 某个 kv 的 8 个连续 d,而 kv-连续的 LDS 像需要
  某个 d 的 8 个连续 kv = **2 字节散射**。per-lane 的 global 偏移是自由的(那是 methodology/07 的
  「+2.7% 免费地址置换」),但它只能决定 chunk **落在哪一行**,不能把 chunk 拆开。
  ⇒ 定价探针给的 **+3.6% V 读流池**只能从「更少的 LDS 条数」以外的方向拿,而条数是结构常数。
- ★★ **行和 MFMA 是「就地」锚定的,不只是「存在」锚定。** r8 已证删掉 8 条 ones-A MFMA = spill 118。
  本轮再证**只是把它们挪位置也要付钱**:`softmax_half` 里 2 条(共 8 条)推到 QK run 之后 ⇒
  spill 4→**8**、−0.3%;把一个 tile 的全部 4 条合并到 C1/C5(前言/尾声一并改线)⇒
  **spill 97**、热循环 240→325 条、22 条 `scratch_load_dwordx4`。**跨一个 rendezvous 多带两个 pack
  的行和状态 = 93 个 spill dword。**
- ★ **QK 簇的 IGroupLP prescription 是惰性的,r8 的 −0.48% 全部属于 P*V 簇那一半。**
  group 1/3/5 的 `(6,3,EXP)+(10,5,VALU)` 请求的是它所在调度区间里**根本没有**的 TRANS/VALU
  (`_sched_barrier(0)` 把 exp 挡在区间外)。整段删掉 = −0.5%、换成 `(2 MFMA,1 DS_READ)×4` = −0.45%,
  两者都落在该批 1.2% 的对照 spread 内且 ISA 相差一条。⇒ **别再把 `FWD_NOSGB` 的 −0.48% 记在 QK 侧。**
- ★ **`spill = 0` 第六次证明在这具核上不是好事,而且分配器在悬崖附近两个方向都不可加。**
  `K_HEAD=0` 释放 16 个 loop-carried dword、达到 **spill 0**,却 **−1.1%**(memory 簇里的 K head 读
  是第一组 QK MFMA 的整簇延迟掩护);而 `K_HEAD=0` 释放 16、`V_HEAD=2` 只要 8,**合起来仍 spill 12**。
  ⇒ 想上 `V_HEAD=2` 不能靠「腾出 8 个 dword」,要靠**改活跃区间的形状**。
- ★ **`softmax_half` 内部那条 `_sched_barrier(0)` 单独也是正贡献。** r10 只测了它与 driver 那条的
  组合(−2.8%),本轮单删它 = **−2.5%**(vgpr 128 / spill 4 / 热循环 240→246)。
  ⇒ r6 的规矩(「先测组合,bisect 只用于归因」)成立,但归因落在**内层**那条。
- **`amdgpu-sched-strategy` 在当前树上重扫**(r6 那批被节点噪声污染,把 max-ilp 留成了未决):
  对照 1.8377/1.8387/1.8489(spread 0.61%),`max-ilp` **−0.10%**、`max-occupancy` −1.1%、
  `minregexcess` −0.8% ⇒ **`max-memory-clause` 保留,max-ilp 结案为中性偏负。**

## r12 fwd(hd64 dualwave forward,11 臂全负或惰性,树内零改动)

★★★ **`sched_group_barrier` 处方要先看它落在哪个 region,而且判负只需一次 opcode md5,不用 bench。**
本核 `softmax_half()` 在两条 row-sum MFMA **之前**有一条内部 `_sched_barrier(0)`,所以驱动里跟在它
后面的 group-2/group-4 处方治的是一个**只有 2 条 MFMA**的 region —— 8 条 P·V MFMA、12 条
`ds_read_b64_tr_b16`、16 条 `v_exp_f32` 全在上一个 region 里。实测:把处方挪进真正的
P·V+exp region(`8x[1 MFMA, 2 TRANS]`)、或删掉旧的、或两者都做,**最终 ISA 的 opcode 序列 md5
与部署逐位相同**。QK 侧同理:`8x[1 MFMA, 1 DS_READ]` 与 `8x[1 MFMA, 2 DS_READ]` 也都逐位相同,
只有 `4x[1 MFMA,3 TRANS] + 4x[1 MFMA,1 DS_READ]` 改了码,benchs **−1.07%**。
⇒ **修正 r11 的 kb_check**:r11 写"`FWD_NOSGB` 的 −0.48% 全归 P·V 组";实测 P·V 组是惰性的,
那 −0.48% 全归 QK 组。⇒ 做法:动 IGroupLP 之前先 dump
`grep -o "^\s*[a-z_0-9]*" 21_final_isa.s | md5sum`,惰性臂零成本出局。

★★★ **ones-A row-sum 的 8 条 MFMA 是一个「分级」的寄存器锚,砍一半就已经 +93 spill。**
r8 只测了 0 与 8 两个端点(删光 = spill 118)。本轮补齐中间点并换了三种替换指令:
| row-sum 构造 | mfma_16x16x32/迭代 | spill |
|---|---|---|
| 部署(ones-A MFMA 吃 bf16 pack) | 8 | **4** |
| 一半 MFMA + 一半 VALU | 4 | **93** |
| 全 VALU,64 条 `v_add_f32` 串行 | 0 | 198 |
| 全 VALU,平衡树(选出 26 条 `v_pk_add_f32`) | 0 | 217 |
| 全 VALU,逐 pack 树(同时只活 8 个 f32) | 0 | 200 |
| dot2 intrinsic / inline asm(r8) | 0 | 94 |
⇒ 墙与**替换指令无关、与活跃区形状无关、与是否删掉 IGroupLP 处方无关**。
★ 顺带一条构造性事实:本核的 S 是**转置**算的(`qk` 传 A=K、B=Q,v_s 即 S^T,m=kv、n=q),
所以一个 lane 的 16 个寄存器是**同一 q 行的 16 个 kv** —— 行和是**片内**归约,只需最后一次
`permlane32_swap` 折半波伙伴(`_lane_pair_reduce` 现成)。**行和根本不需要 MFMA**;拦路的只有寄存器。

★★ **V 的 LDS 读前瞻距离曲线现在两侧都封闭了**:距离 0(读完立刻用)**−0.13%**、距离 1(部署)
**最优**、VSPLIT ≈1.5 **−0.29%**、VPRE ≈2 **−0.66%**、V_HEAD=2 **−2.0% @ spill 11**。
K 侧同向:两个 tail k-step 一起读(`KTAIL`)**−0.62%**。⇒ lgkm FIFO 结论再获两次独立复现。

★ **`misched-cluster=0` 结案**:r7 在 n=2 上读到 +0.18%/+0.10% 并标"under the bar";
本轮 3 对回文实测 **−0.16%**,是噪声不是赢。

★ **tile 形状族(`block_m` x `waves_per_eu` x `gqa_merge` x `fixed_max` x stagger)整族封闭**:
bm64 −4.5% · bm256 −4.3% · bm512 −12% · stagger −6.3% · nomerge spill 28 · `waves_per_eu=3`
**ISA 逐位相同**(trait 里 `waves_per_eu = max(wpe, 4)`,3 根本到不了 codegen)。
⚠ **陷阱**:`block_m=256` + `gqa_merge=True` 会读到 **+3.3%**,那是**坏构建**不是候选 ——
`dma_wave_reps = SMEM_N_RPT // NUM_WAVES = 8 // 16 = 0`,16-wave CTA **一条 K/V DMA 都不发**。
⇒ 任何几何臂都要看 `maxdiff`,不是只看 ms。
★ **「抬 block_m 以提高 MFMA/exp 比值」在算术上不可能**:exps = ROWS_PER_WAVE×BLOCK_N/64,
MFMA 周期 ∝ ROWS_PER_WAVE×BLOCK_N×D ⇒ 比值只由 **D** 决定。ISA 佐证:上面每个臂每迭代都恰好是
64 条 `v_exp_f32` + 32 条 32x32x16 + 8 条 16x16x32。

## r13 fwd(hd64 dualwave forward,软件流水「携带宽度」整轴定价 + iglp_opt 判负,树内零改动)

标尺:同 session 同二进制对照臂,一臂一进程,回文,`warms=5 reps=40`(计分尺口径),
每批 3 个对照;本 session 对照 spread 0.3~0.9%。所有臂先过 ISA 门再 bench。

### ★★★ 软件流水级「携带多少 raw score」是一条有极值的轴,部署点就是极值

一个 kv-tile 的 32 个 f32 分数,可以在**生产它的簇**里 exp2+pack 掉多少、剩多少 raw 传给
下一个簇。四个点全测(D=64,`PV_K_STEPS=2`,一个 pack = 4 dword):

| 臂 | 流水级携带 | ISA 门 | wall vs 同 session 对照 |
|---|---|---|---|
| 全 raw(= r4 之前的形态) | 32 dword | vgpr 128 / spill 4 / 热循环 **238**(比部署少 2 条) | **−3.2%**(4/4) |
| **部署(r4 half-cast)** | 2 pack + 16 raw = **24 dword** | vgpr 128 / spill 4 / 240 | — 极值 |
| 部分(3 pack + 8 raw) | 20 dword | vgpr 128 / spill 4 / 242 | **−1.18%** |
| 全 pack(packs-only) | 4 pack = **16 dword** | vgpr 128 / **spill 0** / private_seg **0** / 246 | **−0.50~0.60%**(9 对 / 3 批) |

⇒ 两个方向都亏,**不是单调的寄存器问题**。机制 = r10 那条「C1/C5 的 16 条 exp 是 head-K
`ds_read_b128` 的承载性延迟掩护」:往前搬(packs-only)把掩护抽走,往后搬(全 raw)把
掩护堆到一个簇里、另一个簇空转。⚠ 顺带**给 r4 补了一个真数字**:r4 当年是靠 +0.08%(计分尺,
噪声内)/ −0.29%(微尺)上船的,今天实测它替换掉的全-raw 形态是 **−3.2%** —— 结论对,当年的
两个读数都不可信。

### ★★ packs-only 腾出来的 8 个 dword **一个都花不出去**(第三次踩同一条)

`packs-only` 把 spill 4→0、scratch 20→0,是本 kernel 迄今唯一把热循环打成**完全无 scratch**
的形态。但把它和三个「当年因寄存器判负」的旋钮组合,分配结果**逐位不变**:

| 组合 | spill(单独) | spill(+packs-only) |
|---|---|---|
| `V_HEAD=2` | 11 | **11** |
| `K_HEAD=6` | 33 | **33** |
| VALU row-sum(免 MFMA) | 246 | **144** |

⇒ 重申 r11 的规矩:**预算的是活跃区间的形状,不是 dword 数**。

### ★★★ `iglp_opt(2)`(MFMAExpInterleave)在这具 forward 上值 **0**,不是正的

姊妹 bwd 核靠 `iglp_opt(2)` 把 **78% 的 exp 链**藏进 MFMA 影子(见本卡 2026-08-19 条),
本 forward 的最大池子正是「43% 的 trans 没被盖住」,所以这条看起来是直球。实测:

* `iglp_opt(2)` + 拆掉 softmax/QK 之间的两条 `sched_barrier(0)`(互斥 mutation,必须同时做):
  spill 4→**27**(热循环里只有 1 条 scratch,寄存器压力全在簇边界),**−6.5%**(3/3)。
* 同一臂叠加 packs-only(spill 回到 **0**,指纹与 packs-only 单独**完全一致**):
  **−0.57%**,即相对 packs-only **±0**。
* 只拆围栏、不上 iglp(让通用调度器自己交织),spill 0:**−0.83%**。

⇒ **手写的 `sched_group_barrier` 处方 + 两条围栏已经等价于 LLVM 的交织 mutation**,换 mutation
拿不到东西;bwd 的 +78% 覆盖率不迁移(那边 exp 链是 quarter-rate 的 1024 拍且没有手写处方)。
⇒ 「让 LLVM 的交织 mutation 去藏 exp」整族封闭。

### r13 fwd 续:★★★ 第一次给这具 forward 做**完整的 stall 普查**(methodology/15 §neither% 的处方)

方法:每个臂只删掉**一个等待源**,保住消费图与 MFMA/exp 条数,先过 ISA 门再回文 bench
(3 个对照 / 3 个臂,一臂一进程)。

| 被删掉的等待源 | ISA delta | wall | 之前的认知 |
|---|---|---|---|
| **稳态 K/V DMA**(4 条 `buffer_load_dwordx4…lds`) | 240→227 指令,spill 4,MFMA/exp 不变 | **+3.95%**(3/3) | 从未定价;与 r6 的 `KVDEDUP=8` 探针(+4.0%)**独立互证** |
| **head-K 的 4 条 `ds_read_b128`**(常驻半个 K tile) | `ds_read_b128` 16→12,spill 4→0 | **+1.54%**(3/3) | LDS 三条流里最后一条没定价的 |
| tail-K 的 8 条 `ds_read_b128` | (r10) | +0.64% | — |
| V 的 24 条 `ds_read_b64_tr_b16` | (r10) | +3.6% | — |
| **两条 `s_barrier` 全删** | `s_barrier` 2→0,240→238,其余逐位不变 | **+0.10%**(3/3) | ★ r5 记「删一条 barrier = +0.97%」 |

★★★ **最重要的一条:会合点现在是免费的。** r5 那个 +0.97% 是**在 r6 删掉 `s_barrier` 两侧的
`sched_barrier(0)` 围栏之前**测的 —— 围栏没了以后 exp/MFMA 可以跨着 barrier 两侧跑,等待被完全盖住。
⇒ **整条「减 barrier」家族(4-deep LDS ring、常量 buf_id 展开、ping-pong 提速)现在值 0.10%,判负封闭**;
这同时解释了 r5 的 4-deep ring 为什么是 −2.2%(它花 1.65% 去买一个值 0.1% 的东西)。
⇒ 教训:**一个减法探针的定价会被后来的改动作废**。会合点相关的数字必须在当前树上重测,别沿用。

- ★★ **补一条更强的:会合点这一族必须"逐条"定价,不能"整族全删"定价 —— 它在条数上是非单调的**
  (2026-09-02,gpt-oss D=64 fwd 第三场 r13,受控)。同一棵树、同一把 sustained 尺(≥14 s / 26 样本 /
  4 个烧臂 / 臂被同二进制对照括起来),每个臂都 register-neutral(vgpr 128 / spill 0 / private 0 /
  occ 512)且 naked-VALU 普查恒为 `[24,24]` sum 48,所以只有 barrier 条数在动:
  **每迭代 2→1 条 = +0.1236% / +0.1841%(两个站点各测,4 个读数全部低于两个对照);2→0 条 = −0.2472%。**
  ⇒ 第一步是正的、第二步是负的,**整族全删的读数(本卡上面那个 +0.10%、以及第二场 r5 的 −0.63%)
  会把这个内部极值完全抹掉**。机理:留一条 rendezvous 仍然把 dualwave 的 ping-pong 相位每 body 钉一次
  (正是 r5 发现 barrier 在这个核里兼任的职能),归零才丢掉相位。
  ⚠ 但**两个站点都是真的 WAR 保护**:单删任一条 `snr_fwd` 从 51.7 掉到 **23.0**(SNR 门当场抓住),
  所以 +0.15% 是**定价上界不是可落地的编辑**;合法形态是把 LDS 从 2 个 buffer 加到 4 个
  (68096 B/WG,2 WG/CU = 136192 ≤ 163840,占用率不掉),让一次 rendezvous 覆盖一个 body 的两个 KV tile。
  ⇒ **通用规则:给任何同步指令族定价,至少测 N、N/2、0 三点;只测 0 会同时丢掉符号和极值。**

**新的池子排序**(互相有重叠,不能相加):exp/trans 7.8%(其中一半是承载性掩护,r10)>
**LDS 读合计 5.8%**(V 3.6 / head-K 1.54 / tail-K 0.64)> **K/V DMA 3.95%** > 每-WG 固定成本 1.4%
> 网格/尾巴 ≤0.2% > **会合点 0.10%**。

### r13 fwd 续:两条「先查事实再动手」省下的轮次

1. **XCD ↔ kv-head 的绑定已经是最优的,是构造出来的**(methodology/15 §0.5 第 1 条在本 kernel 上**天然满足**)。
   `h_kv_idx = h_idx % NUM_HEADS_KV`,grid = (Hq/Q_HEADS_PER_WG = **32**, q_block, batch),
   HIP 线性化 `bid = x + 32·(y + …)`,而 **32 ≡ 0 (mod 8)** ⇒ `bid % 8 == h_idx % 8 == kv_head == XCD`,
   且 `num_kv_heads == num_xcd == 8`。⇒ 每个 XCD 独占一个 kv head,L2 工作集 2 MB / 4 MB。
   这解释了本 campaign 早先两次 remap 探针为什么读到 −0.19% / −0.57% **且访存计数器逐位不动**:
   没有可改进的东西。**审这一层的成本是读三行代码,不是一轮 bench。**
2. **`BufferCopyLDS` 没有 `cache_modifier` 形参**(`BufferCopy` 有:`0=cached, 2=non-temporal`)。
   ⇒ 想给 K/V 的 DMA 打 `nt`/`sc` 提示(理由成立:每条 128 B 行在一个 CU 上**只被读一次**就进 LDS,
   与 r7 判负的 Q-`nt` 不同)在这个 FlyDSL API 上**不可表达**,别再找。
3. **把 head-K 的读从 memory cluster 挪进消费簇**(缩短 lgkm 距离,不是拉长)`kmove`:
   ISA 指纹逐项相同但 opcode md5 不同(确实生效),8 对 / 2 批反序 = **+0.05%,批间变号** ⇒ 噪声。

## r14 fwd (2026-08-31) — occupancy / barrier-domain pricing, and the phase-split family closed

同一 kernel(gpt-oss D=64 attention **fwd**, gfx950, 8-wave CTA / 2 WG per CU / vgpr 128 / spill 4 /
LDS 34048 / 热循环 240 条)。本轮无候选上船;产物是四条带数字的封闭。

### 1. 占用率悬崖:纯探针 −9.9%
把 LDS struct 垫到 90112 B 逼成 1 WG/CU,其余一律不动:vgpr 134 / spill 0 / 237 条 /
MFMA 与 v_exp 条数不变 ⇒ **2.0223, 2.0401 vs 1.8423, 1.8466 = −9.9%**。
另一侧:**删掉 `rocdl.waves_per_eu` 请求**让分配器落到自然点 **vgpr 132 / spill 0**,同样是
3 waves/SIMD = 1 WG/CU ⇒ **ISA 门就判负**。
⇒ 8-wave CTA 上真正的门是 `512 / vgpr >= 4`(即 vgpr <= 128),**不是** "vgpr <= 168 的 3-wave 档"
——3 waves/SIMD 装不下第二个 8-wave WG,第三个槽白扔。
⇒ 由此,「Q 搬进 LDS 再把 `rows_per_wave` 32→64」这条路(逐项算出来是 200 vgpr,Q 全进 LDS 也只降到
**168**)**在拿到任何东西之前先付 9.9%**。重开条件是先找出 **72 个 loop-carried dword**,而 r13 已证明
carry 轴处在内部极值、拿不出来。

### 2. ★ barrier 的「宽度」与「代价」是两个量(修正 r13 的读法)
`Q_HEADS_PER_WG` 2→4 = 16-wave CTA:**vgpr / agpr / spill / LDS / 热循环条数 / MFMA / v_exp /
K-V 字节率逐位相同**,仍是 4 waves/SIMD —— 唯一变量是 CU 上的 16 个 wave 属于**一个** barrier domain。
实测 **1.9011, 1.9113 vs 1.8479, 1.8491 = −3.1%**。
而同一 kernel r13 把两条 `s_barrier` **全删**只值 **+0.10%**。
⇒ 会合点被盖住 ⇒ 留着不要钱;但**要等多少个 wave 到齐**是另一笔钱。任何「合并 WG / 放大 CTA /
persistent / warp-spec」的候选,即使占用率、寄存器、字节全不变,也先扣 ~3%。

### 3. ★ r7 的「16-wave 拿到了 sector 减半」是错的
`_load_kv` 每 wave 填一条 LDS line;16 wave 配 8 条 line 时
`dma_wave_reps = SMEM_N_RPT // NUM_WAVES` = 0,r7 用 `max(1,…)` + line index `mod SMEM_N_RPT` 补救,
结果是**每条 line 填两遍、全局读请求一点没少**。本轮补做真减半版(只让前 8 个 wave 发 DMA,
wave-uniform 标量分支):spill 4→10、+89 条 guarded、**2.0646/2.0657/2.0615 = −10.5%**,
比复制版还差 7.4%(混杂项:spill、分支、8 个 wave 在 DMA 处闲等)。
⇒ 结论只能写成:**迄今没有任何真正减少 K/V 字节的构造测出正值**;两种交付它的构造是 −3.1% 与 −10.5%。

### 4. ★★ P0 相位工程:唯一没试过的形态,以及它为什么造不出来
机制(goal §P0 一直没写对):`_dualwave_sync_barrier` 是普通 CTA-wide `s_barrier`,所以一个 CTA 的
8 个 wave **始终在同一个 cluster 里**;wave `w` 与 `w+4` 是同一 row group 的两个 GQA sharer,落在
**同一个 SIMD** 上、跑同一条指令流,round-robin 仲裁让它们**同时**进 16 条 `v_exp_f32` 块
(trans 管线 2 倍超订、矩阵管线排空)、又**同时**进 8 条 P·V MFMA。r10/r12 的每一个臂都是
**所有 wave 一起重排**,治不了「同相重复」。唯一没试的形态 = **按 wave 反相**:sharer 0 发
[P·V][softmax],sharer 1 发 [softmax][P·V] —— 指令、算术、工作量全同。

三种载具全部死在寄存器(见 00-index 新行);**归因对照臂**(同载具、两分支发同样的码)= **−10.6%**,
⇒ **载具值 −10.6%,相位倒置本身只有 −1.5%**。根因是 `scf.if` 让循环不变量(Q packs、LDS 基址)
的活跃区跨越整个分支而被 spill;**只复制主循环反而最差(−27%)**,因为重载正好落进热循环。

### 5. 其余(带数字,别重走)
* **epilogue 的 8 条手写 `s_waitcnt lgkmcnt(0)`**(r7 曾以 +0.32% 上船、后被回滚,现仍在树里):
  ISA 门干净(热循环逐位相同),9 臂对 8 个对照、两批相反顺序:批 1 **−0.12%**、批 2 **+0.18%**、
  合计 **+0.03%**。⇒ r7 的 +0.32% **不在这棵树上复现**,符合「减法定价随树作废」。
* **按 CTA 奇偶差异化 `s_setprio`**(新想法:既然两个 barrier domain 值 3.1%,偏置它们的仲裁应能进一步错相)
  —— 偶数 WG 保持 `s_setprio 2`,奇数 WG 给 1 或 3,wave-uniform 标量分支。寄存器中性
  (vgpr 128 / spill 4),但 LLVM 把簇头复制了:247 条 issued + 85 条 guarded。
  level 1 **−0.34%**、level 3 **−0.34%**。⇒ 复制出来的簇头比错相赚得多。
* **O store 的写请求粒度**(全库唯一没定过价的 census 项;methodology/07 说 gfx950 写请求是 64 B,
  而本核 lane `l` 与 `l+32` 共一行只覆盖 32 B):把 lane 配对成「四个 lane 覆盖 64 B」的别名探针
  **ISA 门判负** —— spill 4→**26**、private_seg 20→108、热循环 240→247,因为 LLVM 把多出来的
  地址项 hoist 到 kernel 顶端并跨循环常驻。它读到的 −7.2% 是寄存器分配,**记为无效探针**。
  改用算术定界:O 每次计分调用写 268 MB = 146 GB/s ≈ 写带宽的 2%,而 prologue+epilogue+store
  整池只有 1.4% ⇒ store 不可能值超过零点几个百分点。

## r17 fwd (2026-09-01) — the power-domain sign of MFMA vs trans; sclk is not a lever

★★★ **A slower arm clocks HIGHER at a pinned TBP.** gpt-oss d64 fwd, 4/64/8/8192/64, two
register-neutral arms (vgpr 128 / agpr 0 / spill 4 / hot-loop scratch 0 / LDS 34048 / 48 ds_read
/ 4 buffer_load / 2 s_barrier all unchanged), each replicated in reversed order at 1400 W:

| arm | change | wall | sclk |
|---|---|---|---|
| control | — | 1.8346 ms | ~1621 MHz |
| `dup` (row-sum MFMA run twice on the same accumulator) | MFMA pipe 1152 → 1280 cyc (+11.1%) | **−4.4%** | **1684 MHz (+3.9%)** |
| `noexp` (score feeds the pack directly) | −64 `v_exp_f32` | **+8.7%** | **1671 MHz (+3.1%)** |

⇒ **skill 说 "换 MFMA 原子 = 能量杠杆,按 sclk 定价"; 实测 sclk 在功耗封顶下主要是 wall 的倒数** ——
`dup` 用零原子改动复现了那四次 "+6.4/10.3/12.6/17.6% sclk" 的全部特征,而那四条臂的 wall 都更差。
封顶下 `能量/次 = TBP x wall`,所以 **wall 本身就是能量口径**;sclk = cycles/wall,臂一慢它就涨。
**能量信号是 wall 与 clock 同向变好** —— 本核只有两条臂做到:KV dedup 与 `noexp`。

★★★ **MFMA 与 trans 在功耗域符号相反。** 本核 MFMA 管线加 11.1% 的活,卡照样加频 ⇒ MFMA 近乎
免费;`v_exp_f32` 才是功耗大头。⇒ 排序要按 **trans / 访存 指令数**,不按 MFMA 拍数。
`noexp` 读到 **1304 TF/s** —— 距 1300 的那 9.3% 基本全在这 64 条 exp 里。

★★ **别把 cycles→wall 的折算率用在一个已经是 wall 的实测数上**(双重打折)。该错误把本场最大的池
从 8.7% 降到 2.5%,直接导致它被判为"不值得立轮"。

★ **给一个族定价要挑不会 spill 的方向。** row-sum 的 8 条 ones-A MFMA 被"删/换"了六轮,每轮都
spill(93/94/101/118/198/217);**同一累加器上跑两遍**是寄存器中性、DCE-proof 的,给出真价:
**+4.5% wall**(不是管线占比推出来的 11%,因为管线只占 wall 的 83.6%,时钟还退回 3.9%)。

❌ 别再试(本轮实测):
* row-sum 改 VALU **融进 exp 链**(goal 点名的"从没试过的那个形式")→ spill **198**,热循环 240→423,
  71 条 scratch;加显式 `sched_barrier(0)` 锚点仍 198。
* 上式 × `amdgpu-sched-strategy` = max-ilp / minregexcess / max-occupancy / 默认 → spill
  **182/182/182/198**,四档热循环全是 423 条 ⇒ **调度器不是这个族的杠杆**(r16 在端口上 12→5 的
  那次不迁移)。
* **部分 mask 的 `sched_barrier`**(此前 16 轮全是 all-or-nothing):0x008 允许 MFMA 穿越 −0.15%、
  0x408 同 md5、0x002 允许 VALU 穿越 −0.20%、row-sum 后再加一道 fence −0.12%(5 对回文,
  同 session 对照 spread 0.96%)。三者 ISA 均干净,`s_nop` 6 → 14/16。


## r18 fwd（2026-09-01，gpt-oss D=64 forward，campaign 20260831_084030）

**受控对给出的两条硬数字。** 加法方向的 KV 字节探针（每条 tile DMA 再发一次，标量 `soffset`
偏移 2048 行；ISA 门：vgpr 128 / spill 4 / 热循环 scratch 0 / LDS 34048 / mfma 32+8 / v_exp 64 /
ds_read 48 / barrier 2 全部与部署逐项相同，只有 `buffer_load 4→8`）：

| | base | 2× 字节 | 2× 请求(读伙伴张量同 tile) |
|---|---|---|---|
| wall | 1.8353 ms | 1.9030 ms (**−3.69%**) | 1.9309 ms (**−4.95%**) |
| sclk / power | 1625 MHz / 1400 W | 1620 MHz / 1399 W | 1606 MHz / 1400 W |
| `SQ_WAVE_CYCLES` | 2.71096e9 | **2.82048e9 (+4.04%)** | — |
| `SQ_VALU_MFMA_BUSY_CYCLES` | 2.45485e9 | **逐位相同** | — |
| `SQ_WAIT_ANY` | 6.25513e8 | 6.66443e8 (+6.54%) | — |

1. ★★★ **周期→wall 的兑现率在受控对上是 91%,不是 32%。** 3.69/4.04 = 0.91,sclk 持平
   (GRBM 反解 −0.04%)。r15 记的 32% 是 r1 与 r15 两个**相隔十四轮的不同 kernel**之比,
   sclk 的 −4% 与其它所有改动混在一起。那个 32% 把整张 census 打了三折并关掉了周期轴。
   ⚠ 91% 测在劣化方向,是改进方向的上界;32% 也不是下界。**逐杠杆现测。**
2. ★★★ **K/V 字节池是周期类。** MFMA busy 逐位不变、sclk 持平、代价全在 `SQ_WAIT_ANY`。
   r6「dedup 收益 3/4 在时钟里」的读法不成立。而且它在本核**不可达**:
   `bytes/score = 4D/(NUM_WAVES × ROWS_PER_WAVE)`(r7,ISA 逐字节相同),
   共驻的两个 WG 已经是同一个 kv head(r16),L2 已经在替它们去重。
3. ❌ **簇边界围栏的部分掩码放行,整族判负。** 6 个折叠 `_mem/_pv_cluster_sync` 放行
   MFMA(0x8)/VALU+MFMA(0xA) **−0.21%**、再放行 trans(0x40A) **−0.50%**;
   4 个簇首围栏放行(ISA 计数与部署逐项相同、纯重排)**−0.29%**。
   与 r6「裹在 `s_barrier` 两侧的围栏是纯税 +0.96%」并不矛盾:会合点的围栏是税,簇边界的在干活。
4. ❌ **`s_setprio` 抬到访存簇**(C0/C2/C4/C6 头部抬、折叠 sync 处落):level 1 **−0.24%**、
   level 2 **−0.40%**。这是本核第五个被定价的 setprio 放置,部署的那个仍是唯一为正的。
5. ⚪ **零成本判负两条**(opcode md5 与 base 逐位相同 ⇒ 惰性,不必 bench):
   把 row-sum MFMA 推迟到 QK region(`FWD_RSQ`)、给 `_qk` 前那道围栏放行 MFMA(`FWD_QF=0x8`)。
6. ⚪ 把 `softmax_half` 内部围栏从**携带半**(C1/C5)删掉 —— r12 只删过生产半(−1.1%),
   这是它的补集:**−0.31%**,5 对里 1 对为正。整个围栏族到此关闭。
7. ★★ 工具坑三条:地址相同的重复 `buffer_load … lds` 会被 **CSE**;本节点 `rocprofv3` 默认吐
   **SQLite `.db`**,不加 `--output-format csv` 汇总器会读到空表(看起来像 kernel 过滤没命中);
   `remote.sh` **每次都 rsync**,长跑期间再发一条 remote 命令会把后面几条臂换成新树构建的。

## r2-a2 fwd（2026-09-01，gpt-oss D=64 forward，campaign 20260901_123110 analyze-r2）

前一轮（analyze-r1）的头条是「主循环每迭代有两段 24 条无 MFMA 掩盖的 VALU（16 `v_exp_f32` +
8 `v_cvt_pk`），P0 = 把 MFMA 搬进去盖住它」。**这一轮把那个构造做了四遍、把它的反面做了一遍，
五个方向全负。** 14 个 arm，每个先过 6 s ISA 门，两道 SNR 门（含 e2e 竞态检测）**逐位不变**。

### 1. ★★★ `sa` 受控对——这具 kernel 至今最干净的单变量实验，它给整个"重排族"定了价

`sa` = 在 cluster-3/7 的 `softmax_half` 前插**一条** `sched_barrier(0)`，让生产半的 16 条 exp
不能再被吊进 P·V 的 MFMA 段。**除了 opcode 序列的 md5，其余全部逐项相同**：

| | 部署 | `sa` |
|---|---|---|
| 主循环发射条数 | 240 | **240** |
| opcode 直方图 64exp/32mfma32/8mfma16/32cvt/32ds_tr/16ds_b128/6nop/4setprio/2barrier/4buf | — | **逐项相同** |
| `s_waitcnt` 组成 11×(0) 7×(2) 2×(3) 1×(1) 2×vm(0) | — | **逐项相同** |
| vgpr/agpr/sgpr/spill/LDS/private 128/0/50/4/34048/20 | — | **逐项相同** |
| 裸 VALU 段普查 | 2×24 | **4×24** ←唯一的差别 |
| `SQ_VALU_MFMA_BUSY_CYCLES` | 2.454847e9 | **逐位相同** |
| `SQ_ACTIVE_INST_VALU` | 4.584023e8 | **逐位相同** |
| `SQ_VALU_MFMA_COEXEC_CYCLES` | 1.281799e9（43.64% duty） | 1.155382e9（37.68%）**−9.86%** |
| `GRBM_GUI_ACTIVE` | 2.294926e7 | **+4.39%** |
| sclk / wall | 1602 MHz / 1.8364 ms | **1660 MHz（+3.6%）/ 1.8707 ms（−1.87%）** |

★ 复现到 **0.01%**（三批分别 1.8707 / 1.8708 / 1.8709 ms）。

⇒ ①**共执行有价了**：每减 1 点 coexec duty = **+9.73 MHz sclk** 但 **+0.74% 真周期**，
盈亏平衡在 0.61 %周期/点 ⇒ 部署点（最大交织）赢，**但只赢 0.13 点**——它是个**薄的局部最大**，
不是安全裕度，所以别把"我们已经最大交织了"当成结构性优势。
②★★★ **`SQ_VALU_MFMA_BUSY_CYCLES` 对指令顺序也是盲的**（对原子速率盲是 methodology/03 §B40
已知的）。换序后它逐位不变 ⇒ **判重排类改动只能用 `SQ_VALU_MFMA_COEXEC_CYCLES` + `GRBM_GUI_ACTIVE`**。
③**机制**：≥99% TBP 时 `wall ≈ 总能量 / TBP`，纯重排不改总能量、只在"每拍功耗"与"周期数"之间
互换，于是 sclk 与真周期反向走、wall 几乎不动。这同时统一了看着矛盾的 r17（MFMA 工作 +11.1%
而 sclk **升** 3.9%）与 r14（占用率上去 sclk **掉** 3.3%）：**DVFS 收的是峰值每拍功耗，不是总工作量**。

### 2. ★★ 两个方向都推到极端，全落在 ±3.4% 带内，部署点在带顶

| 方向 | arm | 做法 | 裸 VALU 段 | 发射 | spill | wall |
|---|---|---|---|---|---|---|
| 加交织 | `kh1,sp` | 把 exp 八条一组拆到整个 QK 两侧 | 5 段/最长 14 | 246 | 4 | **−3.39%** |
| 加交织 | `kh1,sp,nf2` | 同上 + 两道围栏都开 | 7 段/最长 17 | 248 | 4 | −2.44% |
| 加交织 | (r1) row-sum-in-exps | 8exp→pack→row-sum ×2 | — | 242 | 4 | −1.55% |
| 加交织 | `nf2` | 开 `softmax_half` 内部围栏 | 2×21 | 241 | 4 | −1.46% |
| **部署** | — | — | **2×24** | **240** | **4** | **0** |
| **减**交织 | `sa` | 见上 | 4×24 | 240 | 4 | **−1.87%** |

⇒ **加交织这一族在这具 kernel 上关闭了，而且不是被寄存器关闭的**（`kh1,sp` 已经把 spill 从 25
救回 4，仍然 −3.39%）。**真正的锚是 128-VGPR 预算**：QK 的 MFMA 要往 Phase-B 的 exp 还在读的
寄存器区里写 32 个新 score，源码级强制交织（`sp`）直接 **spill 4→25、scratch 20→104**。

### 3. ★★ r1 说的"围栏是元凶"是误诊——围栏那条是**惰性**的

* 删 `_qk` 外围那道围栏（`nf1`）= 主循环 **逐字节相同**，md5 不变。这独立复现了 r18 第 5 条
  （`FWD_QF=0x8` 惰性）。
* 连 `softmax_half` **内部**那道也删（`nf2`）确实动了调度，但动的只是两条 row-sum `mfma16` 沉进
  cvt 尾巴（`E16 C5 . m C3 . m`），**16 条 exp 依旧连成一段**。wall −1.46%。
* 补 EXP-first 的 `sched_group_barrier` 处方（`ef4`/`ef8`）= **与 `nf2` 逐字节相同** ⇒
  **IGroupLP 处方在 QK 簇同样惰性**，把 A10（P·V 簇惰性）扩到 QK 簇，第二次测量。

### 4. ★★ "把四条转置读改成消费序"这条 ISA 直接判负——而且两种改法是**同一个 arm**

r1 的 P1：barrier 后四条 `ds_read_b64_tr_b16` 发射序 `8512/8384/8320/8448` 而后半程第一个消费者
要第 3、4 条，逼出 `lgkmcnt(0)` 全排空；改成消费序应该降到 `lgkmcnt(2)`。**做了两个方向**——
反转 V 的发射序（`vdc`）和反转 P·V 的消费序（`pvdc`）——**产出的 ISA 完全相同**（md5
`aadf367d8b35`），因为交换两个 `D_CHUNKS` 标号后数据流图是同构的。结果**更差**：
`lgkmcnt(0)` **11→12**、`s_waitcnt` 23→24、发射 240→241，四条转置读**从 `t2…t2` 挤成 `t4`**，
正好破坏了 `lds_load_v` 注释自己描述的性质（k-substep major，让后端第 k 步的 lgkmcnt 不必连带
等后面的 substep）。⇒ **部署序才是对的**；这条不用 bench，ISA 门 6 s 就退了。
★ 通用教训：**改发射序之前先问"这两种改法在数据流上是不是同构"**，否则会把一个 arm 当两个跑。

### 5. ★✅ `K_HEAD` 往**下**扫是新的、已定价的寄存器捐献者

库里此前只记了往上（`K_HEAD=6` spill 33）。往下扫：

| `K_HEAD` | Phase-B 的 head-K `ds_read_b128` | 发射 | spill | scratch | wall |
|---|---|---|---|---|---|
| 4 / 3 | 8 / 6 | 259 / 255 | 34 / 34 | 140 / 140 | ISA 判退 |
| **2（部署）** | 4 | 240 | **4** | 20 | 0 |
| **1** | 2 | 243 | **0** | **0** | −0.47% |
| 0 | 0 | 243 | **0** | **0** | −0.41% |

⇒ 有了一条**约 12 dword、代价 ~0.4% wall 的预算通道**，而且已验证能救人（把 `sp` 的 spill 25
拉回 4）。⚠ 与本卡 §r11/r13「腾出 8 个 loop-carried dword 一个都花不出去」**不矛盾**：那三次腾的
是 loop-carried dword、活跃区形状没变；`K_HEAD` 改的是 **Phase-B 的驻留条数**，直接改形状。
`V_HEAD` 是镜像轴但方向相反（`vh2` spill 11、`vh3` spill 39，都 ISA 判退）。

### 6. ★★ 周期→wall 的兑现率**有方向性**，本卡 r18 记的 91% 不能反着用

`kh1,nf1`（`K_HEAD=1` + 删那道惰性围栏）：`GRBM_GUI_ACTIVE` **2.294926e7 → 2.262003e7 = −1.435%**，
coexec duty 基本不动（43.64%→43.95%）、spill 4→0、主循环 240→**239**（本轮唯一低于部署条数的 arm）、
SNR 逐位不变。wall 只 **+0.12%**（计分尺，带括号对照）/ **+0.24%**（持续探针）⇒ **兑现 8–17%**。
r18 的 91% 是在**变差**方向量的（本卡当时也写明了），methodology/03 的 20–45% 也是。
⇒ **每场自己在改善方向量一次**，否则会把 1.4% 的周期节省吹成 1.3% 的 wall 期望然后找不到它。

### 7. ⚪ 本轮唯一为正的一手，以及它的尺度

`kh1,nf1,dp45`（`K_HEAD=1` + 删惰性围栏 + prologue-only 的 `s_sleep(45)` 给两个共驻 WG 错相）
= 计分尺 **1.0023**，括号对照 0.9982 / 0.9972 ⇒ **+0.46%**，两腿 1.0002 / **1.0044**，两个 guard
1.0000，SNR 逐位不变。b4 的尺子是 1.0%、b2 是 0.5% ⇒ **只有 b2 那条腿是可测的**。
⚠ prologue 错相**单独**是平的（16/32/45/64/90/127 扫出 −0.66% … +0.23%），它只作为 bundle 成员存在。
★ 但这个 arm 的 ISA 门有意义：**主循环 opcode md5 与部署逐字节相同** ⇒ 错相没有漏进循环体，
这正是 r14 那三个 WG **内**错相方案（`scf.if` 把循环不变量的活跃区拖过分支，−10.6% / −27%）过不了的门。

### 8. 工具坑

* `rocm-smi --showgpuclocks` 输出是 `sclk clock level: 1 (1546Mhz)`；正则 `sclk\D*(\d+)\s*Mhz`
  会咬住**档位号 1**。用 `sclk clock level:.*?\((\d+)Mhz\)`。详见 pitfalls/02。
* **功耗封顶下每批的第一个 arm 系统性慢 ~1.5%、sclk 低 ~4%**（DVFS 爬坡）。烧一个丢弃臂，
  或基线批首批尾各测一次只用批尾那个。详见 pitfalls/02。
* `rocprofv3` 在本节点**不挂**：它要在 cwd 建 `.rocprofv3`，cwd 是只读仓库时报 Permission denied
  后**死在自己的信号处理器里**，那就是被记成"15.7 小时挂死占卡"的东西。`cd /tmp/<自己的目录>` 即可，
  8 个 pass 各约 20 s 全部 exit 0。（这条与 connection/smci355 的记载相反，以实测为准。）
* flydsl：`range_constexpr(*args)` 就是 `range(*args)`，`const_expr` 是 trace 期恒等——
  想用它们做"编译期分支"是幻觉。

## §fwd-r12 (2026-09-02) — MFMA 操作数复用 1→2 的完整价目，以及它为什么不是寄存器预算问题

gpt-oss D=64 FORWARD，gfx950。本节把 attention 里最大的一条结构性杠杆量到底了：**每条 K/V LDS
读只喂一条 MFMA**（`ROWS_PER_WAVE` 恰等于 MFMA 的 M 维 32），所以每 MFMA 的 LDS 读指令是 GEMM 的
6×、非 MFMA 指令是 10×。把 `ROWS_PER_WAVE` 32→64（每 wave 两个独立 M-tile，K/V fragment 的读留在
M-tile 循环外）是**唯一**能抬高复用因子的构造。

### 1. ★★★ 能量完全兑现，代价是周期，两者可以分开量

| arm | 官方尺 fwd64 | sustained best ms | **sclk med** | power med |
|---|---|---|---|---|
| 部署（M_TILES=1） | **1.0046**（n=9 同 session，散布 0.65%） | 1.8213 / 1.8219 | **1632 / 1633** | 1380–1388 W |
| M_TILES=2 | 0.9899 | 1.8589 | **1783** | 1394 W |
| M_TILES=2 + `K_HEAD=3` | **0.9960**（n=2） | 1.8603 | 1775 | 1395 W |

* 每单位工作非 MFMA 指令 **194 → 151.5**（`ds_read_b64_tr_b16` 32→16、`ds_read_b128` 16→8、
  per-tile sync/addr 46→23；DMA 字节/单位工作不变，所以 `buffer_load` 那 4 条一条没省）。
* 在硬功耗封顶下这笔能量兑现成 **sclk +9.22%**，代价是 **周期 +11.51%**（wall×sclk 反解）。
* 净 **−0.8571%**。两道 SNR 门在**全部 6 个 M_TILES=2 臂**上逐位不变 ⇒ 双 M-tile 是精确等价变换。

### 2. ★★★ 复用 2 必然 2× 输出态——这不是可以调的预算

VGPR live-range 审计（read-before-write 跨回边）：**loop-carried 100 → 192**，touched 121 → 226，
实测 vgpr 128 → **240**（分配器在两个几何上都是 carried 的 **1.25–1.28 倍**，很稳）。其中
mfma32 首读的 dword 68 → **144**。

64 行几何下不可压的部分：**O 累加器 = ROWS×HEAD_DIM/64 = 64 dword**，**Q = ROWS×HEAD_DIM/128 =
32 dword**，合计 96。加 row-sum 16、至少一个子 tile 的 score 32、packs 16、fragment/地址/计数 12
= carried ≥ 172 ⇒ 实测 ≥ 215。**vgpr ≤170（3 waves/SIMD）靠削 score carry 到不了**，goal 纸面账
里"packs-only carry 出资 32 dword 后 peak ~172"低估了约 48 dword。

★ 而且换任何一种"第二个 M-tile"的来源，register 账**完全一样**：两个 row group、两个 q-head
（`q_heads_per_wg=2` 本来就共享同一份 K/V LDS tile）都是 2× 的 O 和 Q。把 O 放 LDS 要在每个
P·V 步做读改写（每 KV tile 每 M-tile 32 dword 进 + 32 出），比省下的 24 条 LDS 读贵得多；放
AGPR 在 gfx950 的统一 512 池里不改变 waves/SIMD。⇒ **输出态是几何的函数，不是调度余量。**

### 3. 正确的提法：差 2.3 个周期点，不是差 70 个寄存器

+11.51% 周期 vs +9.22% 时钟。MFMA duty 由此反解为 **84.81% → ~72.9%**。本轮扫过的、动不了它的：

* **调度处方**：`sched_group_barrier` 的 `valu_cnt` ×2/×3 产出 **md5 逐字节相同**（处方是上界，
  拉不进比依赖图更多的 VALU）；**对数**翻倍则把 naked-VALU 普查 124→**137**。饱和。
* **围栏位置**：把 M_TILES 个 exp/pack 块合成**一个**调度区（fence 提到全部 pack 之后）是最优点，
  普查 **153→124**；把 row-sum MFMA 插到两个 M-tile 的 exp/pack 之间，最长 run 48→28 但**总和
  124→147**、vgpr +8 ⇒ 再次印证"总和才是判据，最长 run 不是"。
* **`BLOCK_M`**：256 让 `loop_holds_diag`（`BLOCK_M > 2*BLOCK_N`）变真、因果掩码回到主循环，
  每单位工作非 MFMA **273.5**、`s_nop` 50 条；64 把 DMA 摊到 2 个 wave（`buffer_load` 4→16）。
  128 是唯一落点。
* **`waves_per_eu`**：4-wave CTA 下 {1,2,3,4} 可表达，但 WPE=4 逼分配器按 128 dword 压 ⇒
  **spill 926**；WPE=1 与默认 2 产出同一份 md5。

### 4. 工具坑

* 主循环检测器如果把 mfma32 计数**硬编码**成 32，M_TILES=2 下有 64 条，会静默找不到主循环。
  条件写成 `n32 % 32 == 0` 并用 `n32 // 32` 反解 M_TILES，所有计数**除以 M_TILES 报每单位工作**，
  否则两个几何的指令数根本不可比（383 vs 234 看起来是暴涨，实际是 191.5 vs 234）。
* 本轮官方尺一次 **26 s**，比"2–6 分钟"快一个量级 ⇒ 可以在一轮里做 20+ 个带括号对照的臂。
  先量一次 ruler 的单次耗时再规划轮内预算。
* `SPX`/`SVX` 这类只改调度处方的 DEV 旋钮在 M=1 上 **md5 逐字节等同**却读出 1.0084 / 1.0041 —
  这是免费的**同二进制噪声地板**样本，应当并进对照池（本轮对照因此做到 n=9）。

---

## r14(bwd D=64,a16 dQ 路,occ=1 4-wave):MFMA 发射阴影的「空转 EMPTY」是这颗核唯一还在动的静态量

上一节那张表把「指令数 / `s_nop` cyc / MFMA 连跑长度」判成与 wall 反向。r14 在**另一族臂**上
(GEMM1/GEMM3 的读落点)独立复现,并把正面的仪器换成一个可闭合的恒等式。

### 1. 恒等式:`EXCESS = demand_nonmfma + EMPTY − 12 × N_MFMA`

`v_mfma_f32_16x16x32_bf16` 占矩阵管线 16 拍、只占发射 4 拍 ⇒ **每条 MFMA 白送 12 拍发射阴影**。
本体 1280 条 MFMA/q-block ⇒ 阴影供给 **15360 拍**。把非-MFMA 指令按发射拍数计价
(exp 8、`ds_read_b128` 8、`ds_write` 8、`ds_read_b64_tr_b16` 5、`v_cvt_pk`/`v_pk_mul`/VALU 4、
atomic/VMEM 6、标量/`s_nop`/`s_waitcnt` 1),逐个 MFMA 间隙求 `max(0, 需求 − 12)` = **EXCESS**
(真正推迟 MFMA 发射的拍数),求 `max(0, 12 − 需求)` = **EMPTY**(浪费的阴影)。两者恒等相连。

**十一个变体实测:`demand_nonmfma` 恒在 13446–13558(±0.4%)** —— 指令组合已在算术下界,
`v_exp`(512×8=4096)、`v_cvt_pk`(576×4)、`v_pk_mul`(256×4)都动不了。
⇒ **EMPTY 是唯一自由变量,想赢只能让 EMPTY 变小。** 别再为「省几条指令」开轮次。

### 2. 三条上机臂:EMPTY 3/3 定符号,指令数 1/3 定反

同 session、Thue-Morse、`R7_REPS=80`、b4、5 腿(含逐字节相同的对照腿 base2:+0.057%、
faster_in 3/6、区间与 base 重叠 ⇒ 尺子有效)。

| 臂 | 热循环指令 | 发射 trip | **EMPTY** | vgpr/agpr | wall |
|---|---|---|---|---|---|
| 冠军 | 4024 | 18758 | **7424** | 479/223 | — |
| GEMM1 A-fragment 按链下沉 `g1_rd=1` | 4025 | 18759 | 7616 | 483/227 | **−1.565%**(0/6,区间不相交) |
| GEMM1 C-init 按链下沉 `g1_rd=2` | **3997** | **18731** | 7669 | 483/227 | **−2.401%**(0/6,区间不相交) |
| GEMM3 读 1:1 提示 `g3_rd=1` | 4036 | 18770 | 8789 | 479/223 | **−7.695%**(0/6,区间不相交) |

- ★ **`g1_rd=2` 是判决性反例:热循环少 27 条指令、发射 trip 也少 27,却是三条里第二慢。**
  ⇒「热循环指令数与 wall 单调」在这颗核上**证伪**(继 `naked_sum`/`trip`/`tail_sum`/
  `inloop_lgkm0`/`cover_dist`/`Δ(EMPTY-only)` 之后第七个被证伪的静态排序量)。
- ★ EMPTY 的量级也大致对得上:ΔEMPTY 192 / 245 / 1365 → wall −1.57 / −2.40 / −7.70%,
  即 **≈0.0056 %/拍**,跨 4 倍量程外推误差 <2×。**用它做零 GPU 预筛,别用它做验收。**

### 3. ★★★ `sched_group_barrier` 处方会把「没被点名的指令类」赶到 region 末尾

`g3_rd=1` 就是把 GEMM2 那对 `sched_mfma(n)/sched_dsrd(1)` 原样搬到 GEMM3 的 kstep 环上
(marker 不产生指令,Δ指令名义为 0、vgpr/agpr 逐位不变)。**它命中了自己的目标**:
`{LDS_TR:12}` 那族裸读批的 excess 934→**566**(≈ 预测的 416 全额兑现)。
但同 region 里的 exp 链**没有被任何组点名**,LLVM 把未匹配指令一律排到所有组之后 ⇒
EXP excess 566→**2142**,净亏 1365 拍,wall **−7.695%**。

⇒ **判据:给一个 region 写 `sched_group_barrier` 处方前,先普查这个 region 还载着哪些指令类;
凡是靠共发射盖住的类(exp/trans、pack)必须一起点名(trans 掩码 `0x400`、VALU `0x2`),
否则处方会把它们踢出 MFMA 阴影。** 这也解释了 GEMM2 的 `_rd_hints` 为什么能活:
它所在的 region 由 `s_setprio` 切开,里面本来就没有 exp 链。

### 4. 零 GPU 预筛表(全部被 EMPTY 挡下,一次 bench 都没花)

| 变体 | demand | EMPTY | 判 |
|---|---|---|---|
| 冠军 `iglp=3` | 13505 | **7424** | 最小点 |
| `iglp=0` | 13501 | 7987 | 挡 |
| `iglp=2` | 13491 | 8635 | 挡 |
| `iglp=1` | — | — | **编译器崩**:`AMDGPUIGroupLP.cpp:2087` `MFMASmallGemmSingleWaveOpt` 断言 `DSWCounters should be zero in pre-RA scheduling`(与 r13 记的策略 1 同一条,复现) |
| `g1_ks_outer=1` | 13446 | 9195 | 挡(D=64 走 nt-outer 是对的,差 1771 拍) |
| `g3d=4` / `g3d=2` | 13454 / 13496 | 7688 / 7516 | 挡 |
| `g3_dbat=2` / `g3d=6` | 13505 | 7424 | **ISA 逐字节等于冠军**(G3_DBAT 本来就是 2;G3D 6 与 8 在 `G3_KSTEPS=8` 下同调度) |

### 5. EMPTY 7424 的去向 —— 下一手该打哪里

| 来源 | 拍数 |
|---|---|
| MFMA 背靠背游程(游程内间隙需求为 0) | **3756** = 长度-4 游程 43 处×36 + 长度-2 游程 84 处×12 + 长度-15 游程 6 处×168 |
| 部分间隙(需求 1–11 拍) | 3668,摊在约 800 个间隙上 ≈ 4.6 拍/个,属碎屑 |

按「间隙签名」聚合 excess(冠军):

| excess | 处数 | 签名 | 归属 |
|---|---|---|---|
| 688 | 16 | `LDS_TR:3 LDS_WR:1 PACK:4 SCALE:4` | G2_WEAVE 块(43 拍压在 1 条 MFMA 后面) |
| 558+450 | 6+5 | `EXP:5 PACK:12 SCALE:4` / `EXP:2 PACK:17 SCALE:4` | softmax→GEMM2 过渡的 P-pack 爆发 |
| 472 | 8 | `LDS_RD:8 LDS_TR:1` | GEMM1 A-fragment 批(r14 臂 a 打过,−1.57%) |
| 416 | 8 | `LDS_TR:12` | GEMM3 环前言(r14 臂 b 打过,−7.70%) |
| 360+320+252 | 6+16+9 | `PACK:4 SCALE:4` ± `LDS_WR`/`ATOM` | G2_WEAVE 其余块 |

⚠ **两条「把读喂进游程」的臂都亏,而且都在别处把 EMPTY 顶回来了。** 机制假说(未单独验证):
479 vgpr / 223 agpr / spill 0 是紧贴分配上限的点,`g1_rd` 两臂各要 **+4 vgpr +4 agpr**,
调度器为压活跃区间改用更聚簇的 MFMA 排布 —— **读落点类杠杆的前置条件是先腾寄存器**,
不是先找间隙。下一手按此排序:先做寄存器减法(不掉 EMPTY 的前提下),再回来喂游程。

### 6. 寄存器供体:`agpr=N` 旋钮在 D=64 上是反向的,别拿它做「腾 arch VGPR」

`build_flash_attn_bwd_dkdv_module(agpr=N)` 的语义是**用 inline asm 把恰好 N 个 MFMA 累加器钉进
AGPR**,不是「允许多用 AGPR」。冠军 `agpr=0` 让分配器自由使用,实测就已经落在
**vgpr 479 / agpr 223 / spill 0**。强行给 N:`agpr=8` 直接 `error: inline assembly requires more
registers than available` 编不过;`agpr=16` → vgpr 272 / agpr 16 / **spill 4516**;
`agpr=32` → vgpr 288 / agpr 32 / **spill 4115**。⇒ 这个旋钮**封顶**而不是**扩容**,
「腾 arch VGPR 换调度自由度」这条路要另找载体(tied-operand `"=a,v,v,0,v,v"` 见 methodology/06)。

---

## r15 三条「寄存器类 / 排布」腿全部为负:issue-bound 内核的静态代理与覆盖模型都失效

第二场 campaign optimize round 1。尺子 `_bench_attn_bwd2.py`,同 session ABBA 配对,
**逐字节未改动的 drift 腿读数 1.0052**(不是 1.0)——判定任何一臂都必须先减掉这个零点偏移,
而 harness 的 KEEP 阈值 0.8% 已经和尺子自身的偏移同量级了。三个 guard cell
(b4win / b1win / l70b)在这三臂里都是**逐字节相同的代码路径**(旋钮只在 `D==64 && _fuse_wide && a16`
生效),所以它们是天然的 in-run zero control,可用来扣掉批次偏置。

| 臂 | 杠杆 | 静态证据 | score | 扣偏置后 |
|---|---|---|---|---|
| A | 逐字节未改动 | — | **1.0052** | 零点 |
| P1 | `--greedy-regclass-priority-trumps-globalness=True` | 热循环 **−24 指令 / −18 s_nop / −11 vgpr / −11 agpr**,accvgpr 8→0,工作量计数逐条相同 | 0.9959 | **−0.93%** |
| P2 | V pack 组 tied 空 asm `"=a,0"`(32 dword) | agpr **223 → 219**(反向!),GEMM3 的 B 操作数被从 `a` 挤回 `v`,热循环 B=a 的 MFMA 112→256,kernel shuttle 159/156→176/163 | 0.9991 | **−0.49%** |
| D | `g2_ppk`:把 P pack 推进 GEMM2 的 weave,每 kv tile 4 条、夹在两条 MFMA 之间 | PACK burst **13/11/8 全部消失 → 最大 4**;MFMA run 15 → 最大 8;EXP/MFMA/PACK/SCALE/LDS_RD **计数完全相同**;vgpr/agpr/spill 与 base 一致,shuttle 8→7;`s_nop` **72 → 88** | 0.9948 | **−0.67%** |

### 1. 第八个被推翻的静态代理:「更少指令 + 更少 s_nop + 更少寄存器」也不预测 wall

P1 在 ISA 上是**全面**更好的:指令更少、s_nop 更少、两半寄存器都更省、热循环里
`v_accvgpr_*` 从 8 条清到 0 条、而且工作量逐类计数与 base 逐条相同(纯分配器差异)。
实测 **−0.93%,3/3 target cell 同向下降**。⇒ 分配器侧的「省」在这个 kernel 上不是收益来源;
本卡的静态代理黑名单再加一条。

### 2. methodology/06 clause 2 的 +0.79%/+0.53% 在本配置下不复现,且其前提被证伪

clause 2 说「把 kv-block 不变的 MFMA 操作数 tied-pin 进 AGPR」值 +0.79%/+0.53%。
本轮在 V pack 组上原样复现(严格 tied `"=a,0"`,无 `&`),得 **−0.49%**。
ISA 给出了原因:**「AGPR 半区还有 33 个空位可以花」这个前提是错的**——钉进 32 dword 之后
agpr 不是 223→255 而是 **223→219**,分配器把 GEMM3 的 B 操作数从 `a` 挤了出去,
净效果是**换**而不是**填**。⇒ clause 2 应加限定:该定价只在「被钉的值不与其他 AGPR 常驻值竞争」
时成立;钉之前必须先确认 agpr 计数真的**涨**了,涨不上去就是在做零和替换。

### 3. 「给裸露 VALU 盖上 MFMA」这条主线在本 kernel 已经用尽

四个已入账的胜利(含 G2_FILL / G2_WEAVE)共同的机制都是「把无掩护的 VALU 塞进 MFMA run」。
arm D 把这套原样搬给**最大的 excess 类 PACK(1571 cycle)**:P 的 f32 输入本来就因为
dS 乘法而活到 weave 里,所以延后 pack **不增加任何 live range**(vgpr/agpr 与 base 逐位相同,
这一点被 ISA 证实)。burst 结构也确实按设计改造成功了(13/11/8 → 全 4)。
**但实测 −0.67%**,而 `s_nop` 涨了 16 条。

⇒ 机制解释:`flat_wg=256` 即**每 SIMD 只有 1 个 wave**,wave 自己的 VALU 和 MFMA 走同一条
发射流、串行发射,**不存在「MFMA 执行期间免费发射自己的 VALU」这种影子**;
r14 的恒等式 `EXCESS = demand_nonmfma + EMPTY − 12×N_MFMA` 本来就是这个意思。
在 issue-bound 下**纯重排是发射中性的**,而打断 MFMA 连发只会新增寄存器相关性的 s_nop 填充
(MFMA run 15→8,s_nop 72→88)。所以:**剩下的 PACK excess 不可能靠 placement 回收**——
下一步必须冲 issue **数量**(减少 576 条 PACK / 704 条 LDS_RD 本身),而不是它们的**位置**。
这也反向解释了 G2_WEAVE 当初的收益不是「issue 覆盖」而是 LDS 写延迟的掩盖。

### 4. 尺子会为每一臂单独编译一份 artifact(实测确认)

曾怀疑「只改 `_hints` 的臂」会因为 FlyDSL 的 cache key 不含 context compile hints 而
两臂共用一个二进制、从而让 P1 的读数无意义。用 `_r2_key.py`(直接调尺子自己的 `_arms()`,
跑完 ref 再跑 cand,diff cache 里的 `.pkl` 清单)证实:candidate 在同一个 manager 目录下
写出了**自己的** dkdv artifact(`a378ac26…pkl` 与 ref 的 `d3fc56d7…pkl` 并存)。⇒ 臂间确实不共享二进制。

### 5. 尺子自带的 SNR gate 看不见宽 band(methodology/06 clause 3 的具体化)

`_bench_attn_bwd2._snr_gate` 跑 B=1 S=1024,`_fuse_blockkv_for` 在这个 Skv 上返回 **128**,
`_fuse_wide = D*block_kv >= 16384` 为 False ⇒ **宽 band 的 a16 D=64 路径根本没被构建**。
任何只在该路径生效的旋钮(本轮三臂全是)都过不了「真正被验到」这一关。
本轮补了 `_r2_snr.py`(打印实际 `block_kv` / `a16` / `wide` 再算三梯度 SNR):
S=2048 → 48.8/48.9/49.0 dB,S=4096 → 48.2/48.9/49.2 dB(gate 45)。
⇒ 宽 band 旋钮上线前必须跑这个探针,不能只信 bench 的内置 gate。

## r16(bwd D=64,a16 dQ 路,occ=1 4-wave):IGLP 整轴定价完毕 + 「释放 72 个寄存器」买不到任何东西

本轮两臂上尺子、三臂被 40s ISA 门挡在尺子外,树内零改动(md5 与 base 逐位相同)。

### 0. 先把 wall 的账做平(本轮新数据)

- `rocprofv3 --kernel-trace`:`flash_attn_bwd_dkdv_kernel_0` 占含 fwd 全部 GPU 时间的 **91.08%**;
  扣掉 fwd 后 **dkdv ≈ 被计分 backward 的 96.6%**(odo 1.7% / slotred 0.6% / lset 0.15% / torch fill 0.9%)。
  ⇒ **不存在「辅助 kernel 池」**,任何收益只能出自 dkdv 本体。
- LDS 组(r15 没采到、goal 明确要的那一组):`SQ_LDS_BANK_CONFLICT/SQ_LDS_IDX_ACTIVE = 0.23%`,
  LDS 约 **28% busy**,`SQ_WAIT_INST_LDS/SQ_WAVE_CYCLES = 8.37%`。
  ⇒ **bank/swizzle 与 LDS 带宽两条杠杆被数据关闭**;那 8.4% 是纯**延迟**暴露。
- 每 trip ≈ **36,850** 真实 cycle;MFMA 管道地板 1280×16 = **20,480 cycle = 55.6%**(与 PMC duty 54.34% 吻合)。
  非 MFMA 指令 2,744 条 × 4 cycle = **10,976 cycle 的发射占用**,要塞进 1280×12 = **15,360 cycle 的影子**
  ⇒ 需要 **71% 的影子利用率**,而 `COEXEC/MFMA_BUSY` 实测只有 **16.88%**。这就是 46% 非 MFMA 时间的来源。
- 热循环 = **17 个 barrier 段**:8 个大段(441–452 条:MFMA 150 / EXP 64 / PACK 72 / SCALE 32 / TR 32 / RD 24 / WAIT 28 / ATOM 8 / GLD 4)
  与 7 个小段(58 条:TR 32 / MFMA 10 / WR 4 / WAIT 9)交替,尾段 99 条。

### 1. gfx950 只有一条 in-order `vmcnt`,但本 body 没有饱和(排除一条看似很像的杠杆)

`_r3_vm.py` 按发射序重放整个热循环:dQ 的 `buffer_atomic_pk_add_bf16` 是 fire-and-forget,
谁都不读,于是 LLVM 从不为它们插 wait,队列**单调堆积**。看着很像 vmcnt 6-bit(上限 63)饱和,
实测 **peak inflight = 30**,稳态在 8→20 之间循环。Q/dO publish 的 `vmcnt(11/10/9/8)` 阶梯
确实会顺带排掉**上一个** head-step 的 8 条 atomic,但它们此时已经老了整整一个 head-step(≈4,600 cycle)。
⇒ **VMEM 队列 / atomic 耦合不是杠杆**,别再查。

### 2. IGLP 策略轴整轴定价完毕:部署值 3 就是最优(四个值全部有据)

源码注释只比过 2 与 3(「wide band 取 SIMPLE exp-interleave,full strategy 在这个 MFMA:exp 配比下
白花 hazard nop 和架构寄存器」),0 与 1 从未定价。本轮补齐:

| iglp | LLVM strategy | 结果 |
| --- | --- | --- |
| 0 | MFMASmallGemmOpt | 构建通过(vgpr 481 / agpr 225 / spill 0),**score 0.9832(−1.68%)** |
| 1 | MFMASmallGemmSingleWaveOpt | **编译器 assert 崩溃**,不可用 |
| 2 | MFMAExpInterleave | 源码已判负(劣于 3) |
| 3 | MFMAExpSimpleInterleave | **部署值,最优** |

`iglp=1` 的崩溃是上游 LLVM 断言,不是本仓 bug:
`AMDGPUIGroupLP.cpp:2087 MFMASmallGemmSingleWaveOpt::applyIGLPStrategy` 的
`DSWCounters should be zero in pre-RA scheduling!`。名字里的 "SingleWave" 很诱人(本 body 正是 1 wave/SIMD),
**但它在 pre-RA 阶段就 assert,别再试**。
`iglp=0` 的 ISA 确实动了(大段 441→434、小段 58→70、hazard nop 8→4、总指令 4024→4052):
它把每 head-step 约 12 条指令从大段挪进**小段**,而小段正是延迟暴露区 ⇒ 越挪越慢,与 §3 的结论同源。

### 3. ★ `g3_kreg=False` 释放 72 个寄存器,实测 −2.05%:寄存器不是这颗核的总阀门

源码把 `g3_kreg`(整 band 常驻 K^T fragment)的收益写成「**pure read removal**」,
而同文件另一处探针把 GEMM3 读数 1536→1024 定价为 **6/11 = 噪声**并写下「Do not spend a round on read count」。
两条加起来推出一个很强的假设:它在用寄存器买一个已被证明无价值的东西。ISA 完全证实了代价面——
**vgpr 479→407、agpr 223→151(各 −72),spill 仍为 0**,LDS 不变。

**实测 score 0.9795(−2.05%)**(b4 ×0.9770 / b2 ×0.9754 / b1 ×0.9860,guard 全 1.0)。

两条要记的更正:

1. **「skill(methodology/06)说寄存器是总阀门,实测释放 72 个寄存器收益为零(实为 −2.05%)。」**
   occ 不变(407 > 256,仍 1 wave/SIMD),分配器的余量在 issue-bound 下换不到任何东西。
   r15 已经给过 clause 2 一个 caveat,本轮把它推到极端:**寄存器释放本身不是收益,别再为「省寄存器」立臂**。
2. **「源码说 GEMM3 读数免费(6/11),实测反向加 256 条读 = −2.05%。」**
   免费的是**被 MFMA run 覆盖的**读(那次探针靠 CSE 去掉的正是覆盖区内的读);
   `g3_kreg=False` 新增的 32 条/head-step 落在 **barrier 之后的暴露小段**里。
   ⇒ 判据不是读的**条数**,而是读**落在覆盖区还是暴露区**。读数轴与读位置轴必须分开记账。

### 4. GEMM3 prefetch ring 深度:越深越好,而且已经顶到上限(40s ISA 门,省下一次尺子)

调度器本来就把 GEMM3 的 32 条 MFMA 中的一部分吊出暴露小段。ring 深度直接控制吊出多少:

| g3d | 吊进大段的 MFMA | 小段大小 | 小段 WAIT |
| --- | --- | --- | --- |
| 8(部署) | **22 / 32** | **58** | **9** |
| 4 | 18 / 32 | 62 | 9 |
| 2 | 14 / 32 | 71 | 14 |

三者 vgpr/agpr 完全相同(479/223)⇒ 这不是寄存器题。方向是**越深越好**,而
`G3D = min(g3d, G3_KSTEPS)`、`G3_KSTEPS = BLOCK_KV // PV_K_STEP = 256/32 = 8`,**部署值已经就是上限**。
⇒ 想在这条轴上继续走,只能抬高 `G3_KSTEPS`(即 BLOCK_KV > 256),那会改 `_fuse_halves` 与整个几何。

### 5. 下一步该冲的具体杠杆(由本轮数据指名)

暴露小段只有 58 条指令,却含 **32 条 `ds_read_b64_tr_b16`**(GEMM3 的 dS fragment),
而 §3 刚证明**往这个区里加读 = 直接掉 2%**,所以对称地:**从这个区里搬走读 = 应当直接赚**。
两条候选,都不改算术、不碰精度:

- **dS 写侧预 swizzle**:`ds_read_b128_tr_b16` 在 gfx950 不存在,但**普通 `ds_read_b128`(16 B/lane)存在**。
  dS 的 LDS 布局是 softmax 的 `_ds_write_vec` 自己定的 ⇒ 把 transpose 提到**写侧**(它在 MFMA 密集的大段里,有掩护),
  GEMM3 就能用普通宽读,把暴露区的 32 条读降到 **16 条**。这是「把工作从暴露区搬进覆盖区」,
  和本 body 四个已入账胜利同一机制,但从未在 **dS 读**这一家上做过。
- **EXP 家族的条数**:`v_exp_f32` 是半速(8 cycle)且走 trans 口,64 条/head-step = 512 cycle/head-step
  = 4,096 cycle/trip ≈ wall 的 11%。条数 = MT(4)×NT(4)×4,是每 lane 的 P 元素数 ⇒ 只能靠改 tile 几何降低
  (宽 MFMA atom 本轮被禁),但它是 EXP 轴唯一的方向,值得单独一轮。

## ★ r17(bwd D=64,a16 dQ 路,occ=1 4-wave):把**裸 MFMA run 搬进暴露小段**= 这颗核近两轮第一条正收益(+0.9%)

本轮三臂上尺子(两次复测),一条新旋钮 `g2_hs=1` 入账,另一条 `g2_rd_at` 判负后整块删除。
r15/r16 连着两轮全负(寄存器类、排布类、IGLP 类),本轮换了记账口径就直接拿到分:
**不再问"这颗核缺什么资源",只问"哪条 wait 没有 MFMA 掩护"**。

### 1. 机制:大段尾部的裸 MFMA run 是**免费的掩护料**,而它紧邻的小段正是暴露区

每 head-step 的大段(441 条 / 150 MFMA)是这样收尾的:GEMM2 最后一个 d-tile 的 8 条 MFMA
成一串裸 run(`WAIT MFMAx4 WAIT MFMAx4`)——**操作数早在一个 d-tile 前就读完了**,
所以这 8 条只占 8×4 = 32 cycle 的发射,却占 8×16 = 128 cycle 的矩阵管道 ⇒ 约 **96 cycle 的发射空转**。
紧接着 dS fence 之后的小段(58 条 / 只有 10 条 MFMA)开头是 **12 条 `ds_read_b64_tr_b16` 突发**,
它的第一条 `s_waitcnt lgkmcnt(10)` 把 LDS 延迟裸露在外,段内没有任何 MFMA 可以掩护它。

`g2_hs=k`:把 GEMM2 **最后一个 q-half 的最后 k 个 d-tile** 的 MFMA 组以 thunk 形式交给 `_gemm3`,
在它的 read ring 发射完之后立刻 emit。这些 MFMA **既不读 dS 也不读 GEMM3 的任何操作数**,
于是读突发退休在它们的管道影子里。ISA 证实(`FLYDSL_DUMP_IR` + 按发射序重放 in-order lgkm 队列):

| | base(r4 部署面) | `g2_hs=1` | `g2_hs=2` |
| --- | --- | --- | --- |
| 小段 run-length | `TRx12 WAIT …` | **`TRx12 MFMAx8 WAIT MFMA TRx3 …`** | `TRx12 MFMAx16 WAIT …` |
| TR 读的 MFMA 覆盖均值 | 7.30 | **9.28** | 9.92 |
| 热循环指令 | 4010 | 3980 | 3969 |
| vgpr / agpr / arch / spill | 495/239/256/0 | **逐字节相同** | 逐字节相同 |
| score(raw) | 1.0059 | **1.0137 / 1.0151**(2 session) | 1.0054 |
| score(guard 零对照校正后) | 1.0078 | **1.0171 / 1.0134** | 1.0119 |

⇒ `g2_hs=1` 对 r4 部署面 **+0.9% raw / +0.74% 校正**(b2 +1.0~1.4% / b1 +2.6~3.0% / b4 ≈ 中性)。
宽 band 真路径 SNR(`_r2_snr.py 2048`,bench 自带的门跑不到这条路)dq 48.8 / dk 48.9 / dv 49.0 dB,
与 base 逐位同序(每个累加器的 pks 升序不变)⇒ 纯排布,不动数值。
⚠实现两个坑:①只能交**最后一个 q-half**的尾巴,否则会打乱累加器顺序;
②MASK_SKIP 的 `_if_wave` 路径**不能**接 thunk(延迟的 MFMA 会逃到 wave-uniform 分支外),
用一条 `assert not _g2_hs_q` 守住"交出去的 tile 必须被 emit"。

### 2. 掩护量有**饱和点**:搬 8 条赚,搬 16 条不赚

`g2_hs=2` 校正后 1.0119,低于 `g2_hs=1` 的 1.0134~1.0171。机制:`s_barrier` **不等 MFMA 完成**,
搬 8 条时大段尾部管道里还有在飞的 MFMA 跨过 barrier 继续算;搬 16 条把大段尾部的管道也抽空了,
而小段那边买到的掩护已经饱和(覆盖均值 9.28→9.92,只多 0.6 条)。
⇒ **"把工作搬进暴露区"这条机制的正确剂量 = 刚好盖住第一条 wait,多搬只是把暴露从小段搬回大段。**

### 3. ★ 读的**突发化**再次判负,而且很贵:−1.4%

`g2_rd_at=1`:GEMM2 环的 `4*_nk` 条 TR 读改成"1 条 MFMA 引导 + 成组发射 + 其余 MFMA",
替掉部署的 1:1 `sched_mfma(1)/sched_dsrd(1)`。ISA 门全过(495/239/256/spill 0,TR 覆盖 7.30→7.93),
叠在 `g2_hs=1` 之上实测 score 0.9996 / 校正 **1.0031 = 比 hs1 低 1.4%**。

这复核了源码里那条旧结论(「整组读提到 MFMA run 之前会输,因为读突发挡住 MFMA 发射」)并把它推广:
**即便先放引导 MFMA、即便覆盖均值升高,突发化仍然输**。
- 「**skill(pitfalls/09)说 1:1 sched hint 把读钉在退休它的 wait 旁边是失效模式 ⇒ 应当把读拆开;
  实测拆成突发 = −1.4%。**」失效模式的诊断是对的,**处方是反的**:
  不要动读的位置(读一动就挡发射),要给那条 wait **加 MFMA 掩护**(= §1)。
- **`cover_dist` 作为排序代理第三次失效**:at=1 覆盖均值(7.93)高于部署(7.30)却慢 1.4%。
  覆盖距离只能用来**确认机制**(ISA 是否真的动了),永远不能用来**排序臂**。

### 4. 对 methodology/06 的正面更正(r16 给了负面那一半)

`g2_hs=1` 的寄存器头与 base **完全相同**(vgpr 495 / agpr 239 / arch 256 / spill 0 / LDS 86016),
即这 0.9% 与寄存器预算**毫无关系**。配上 r16 的「释放 72 个寄存器 = −2.05%」:
**这颗核的阀门既不是寄存器、也不是读的条数,而是"指令落在覆盖区还是暴露区"。**
本轮供 supervisor 预留的 tied-operand `"=a,0"` 兜底(怕 prefetch 压出 spill)**一次都没用上**——
搬 MFMA 不新增任何 live range,所以它天然 register-neutral。

### 5. 下一步(由本轮 ISA 直接指名)

TR 家族的覆盖直方图仍有很长的低尾:512 条 TR 读里约 **240 条覆盖 ≤7 条 MFMA**
(hist 0:26 / 1:25 / 2:39 / 3:25 / 4:27 / 5:31 / 6:65 / 7:61),RD 家族 192 条里有 **20 条覆盖为 0**。
下一轮的正确做法是**先把这条低尾按「读家族 × barrier 段」归属**(哪一家的哪条 wait 还裸着),
再对每个低尾家族复用 §1 的同一机制——**搬独立 MFMA 过去,不要搬读**(§3)。
r16 §5 提的「dS 写侧预 swizzle」仍然有效且与本轮正交:它减少暴露区的**读条数**,
而 `g2_hs` 增加暴露区的**掩护量**,两者可以叠。

## ★★★ r18(bwd D=64,a16 dQ 路,occ=1 4-wave):量出真时钟,这颗核第一次有了 roofline —— 76%,而且瓶颈不在 MFMA 覆盖

前 4 轮(r14~r17)都在 MFMA 覆盖轴上一次赚 0.8~0.9%,却从没算过"离顶多远"。本轮用
`rocm-smi -c` 在 b4 跑时采样 5 次,拿到**持续 sclk = 1402 MHz**(DPM level 1,不是 2.4 GHz 标称)。

| 口径 | 数 |
|---|---|
| `v_mfma_f32_16x16x32_bf16` | 16384 FLOP / 16 cyc = 1024 FLOP/cyc/SIMD |
| MFMA 管线 roofline @1402 MHz | 256 CU x 4 SIMD x 1024 x 1.402e9 = **1470 TF** |
| 实测 b4 / b2 / b1 | 5.050 / 2.4674 / 1.2235 ms = 1122 / 1149 / 1158 TF(含 3.1% padding 的真 MFMA 量) |
| 离 roofline | **76% / 78% / 79%** |
| 每 q-block trip | 26819 cyc,管线 roofline 20480 cyc ⇒ **多花 6339 cyc(+31%)** |

三个 cell 都落在 76~79% ⇒ **机器级负载均衡已经不是瓶颈**(`_a16_qsplit` 把 causal 三角摊平了),
6339 cyc 是热循环自己的。★ **一个 cell 的 wall 只有配上真时钟才是 roofline;标称 2.4 GHz 会把
76% 读成 45%,得出"还有 2 倍"的错误结论。**

★★★ **这条不是新发现,是本臂欠的账。** `methodology/03` 第 335 行早已写明「**采 sclk/power 是
profiling 的第 0 步(本项 KB 曾连续 10 轮漏掉,把 util 算错 20%)**」,第 484 行连"用 2.4 GHz 标称
算出 in-loop util 52.8%"这个具体错误都点名了,第 333 行的 Power/clock-bound 行给的处方正是
「**降每单位工作的指令数与搬运量**」。
⇒ 「**skill(methodology/03)说先量 sclk、且 sclk 远低于 boost 就按"砍指令条数"走;实测 bwd 这条臂
连做 5 轮(r14~r18)没量过一次时钟,一直按覆盖轴走。**」**KB 是准的,是本臂没执行第 0 步。**
本轮 §1 的发射预算拆解与 §5 的下一步,本质上只是把 `03` 那一行处方在 bwd 上算了出来。

### 1. ★★★ 把 6339 cyc 拆开:发射流(~22132 cyc)比矩阵管线(20480 cyc)还长,而 exp 占它 37%

trip 内 3997 条指令的构成(ISA 普查):MFMA 1280、`v_exp_f32` 512、`v_cvt_pk_bf16` 512、
`v_pk_mul_f32` 256、LDS+atomic 864、wait 230、nop ~70、寻址 ~273。按发射占用折算:

| 类 | 条数 | 发射占用 |
|---|---|---|
| MFMA | 1280 | 1280 x 4 = 5120 |
| **`v_exp_f32`(四分之一速率)** | **512** | **512 x 16 = 8192** |
| 其余 VALU | 768 | 3072 |
| LDS + atomic | 864 | 3456 |
| wait/nop/SALU | 573 | 2292 |
| 合计 | | **22132 > 20480** |

⇒ **occ=1 下这颗核是 issue-bound,不是 pipe-bound**;而发射预算里最大的一项不是 MFMA(5120),
是 **512 条四分之一速率 exp = 8192 cyc(37%)**,比全部 LDS 流量(3456)大 2.4 倍。

★ 这与本文件 `r17 fwd` 那条卡**完全一致**:「MFMA 与 trans 在功耗域符号相反……排序要按 trans /
访存 **指令数**,不按 MFMA 拍数」,fwd 上 `noexp` 删 64 条 exp = **+8.7% wall**。
「**skill(本文件 r17 fwd)早已说清 trans 才是这颗核族的大头;实测 bwd 的 exp 占发射预算 37%,
而 bwd 的 r14~r17 四轮全花在 MFMA 覆盖上,每轮只买到 0.8~0.9%。**」
⇒ **KB 是对的,是 bwd 这条臂没去读 fwd 的卡。** 本轮的 sclk 也印证功耗侧:bwd 1402 MHz vs
fwd 1621~1684 MHz,低 15%,正是 512 exp + 1280 MFMA 的功耗代价。

⚠ 上表有一个**未证的假设**:`v_exp_f32` 是否真的把这个 wave 的发射槽占满 16 cyc。若只占 4 cyc,
发射总量降到 ~16k < 20480,核就回到 pipe-bound,6339 cyc 得另找出处。ISA 里 LLVM 的
`iglp_opt(3)`(MFMAExpSimpleInterleave)把 exp 与 MFMA **严格交替**排(body 48~74),暗示硬件
是能重叠的 ⇒ **下一轮第一件事就是用 PMC 判这个分叉**,别再凭静态模型猜:
`SQ_INSTS_VALU_TRANS_F32` / `SQ_VALU_MFMA_BUSY_CYCLES` / `SQ_BUSY_CYCLES` / `SQ_WAIT_ANY`
(rocprofv3,b4 cell)。这三个数一出来,6339 cyc 就能唯一地分给 trans 占用 / wait 停顿 / 发射串行。

### 2. ★★ 两个静态覆盖模型同时失效 —— 这是第 4、第 5 个失效的静态代理

本轮在 `_r6_cov.py` 里加了两个模型,都**定位有用、排序无用**:
- `pipe`(MFMA 占管线 16 cyc,其余 4 cyc 发射):热循环矩阵管线**空转仅 0.9~1.0%**,而且几乎全在
  seg0 前导。⇒ 「哪里管线空着可以塞 MFMA」这个问题的答案是**没有**,r17 §5 指的低尾不是空转。
- `slack`(每条 wait 处还在飞的 MFMA 拍数):slack **单调累积**到 seg16 的 4590 cyc —— 这是
  "管线需求 20480 > 发射 15988"的必然产物,于是**每条 lgkm wait 的 slack 都是几千拍**,模型给出
  "所有读都免费"的结论,与实测的 0.8~0.9% 收益直接矛盾。

⇒ 与 r15/r16/r17 的 `cover_dist` 三连失效合起来:**这颗核上任何"数拍子"的静态模型都不能排序臂。**
静态模型只有两个合法用途:(a) 确认机制真的落到 ISA 了;(b) **定位**结构性不对称。
本轮 (b) 唯一的产出:**seg1 是全循环唯一低 slack 段**(min 0 / mean 216,其余大段 min 652~4084),
因为上一个 head 的 GEMM3 MFMA 沉不过循环回边。但把它当机会是**错的**:seg1 开头那两条
`lgkmcnt(1)/(3)` 前面已经有 17 条寻址 VALU + 4 条 `buffer_load_dwordx4` 共 68 cyc 在遮 ~120 cyc 的
LDS 延迟,净暴露只剩 ~50 cyc/trip = **0.24%**,不值得为它冒 loop-carried 寄存器的风险。

### 3. ★ `g1_pf` 整轴定价完毕:只有 `G1_PF_AT == DT-1` 赢,别的位置等价且更差

`g1_pf` = 从上一个 q-half 的 GEMM2 第几个 d-tile 发下一个 half 的 GEMM1 A 片段读(r4 部署 =3,+0.8%)。
本轮同 session 配对 + guard 校正(4 臂,零对照臂 = 原封不动的树):

| 臂 | score(raw) | guard bias | 校正后 | vs 零点 |
|---|---|---|---|---|
| 零点(部署 `g1_pf=3`) | 1.0132 | +1.0002 | 1.0130 | — |
| **`g1_pf=1`** | **1.0210** | +1.0048 | **1.0161** | **+0.30%**(raw +0.77%) |
| `g1_pf=2` | 1.0143 | +1.0028 | 1.0115 | −0.15% |

机制(ISA 证实):`_rd_hints()` 每个 d-tile 只分配 `_n_out * _hn = 8` 个 `sched_dsrd(1)` 槽。
`g1_pf=2/3` 落在 `_rd_next == True` 的 d-tile 上,8 条 `ds_read_b128` 预取和 4 条环 TR 读抢同一批槽,
环读被挤到**零 slack 的 `lgkmcnt(0)` 全排空**;`g1_pf=1` 落在 `dt = DT-1 = 3`,那里 `_rd_next == False`
没有环读竞争,预取独占槽位。裸 wait 站点 **41 → 25**,GEMM2 的 `lgkmcnt(0)` 全排空变成
`lgkmcnt(6)/(2)`,指令 3980 → 3997,寄存器头**逐位相同**(vgpr 495 / agpr 239 / arch 256 / spill 0 /
LDS 86016),SNR(真宽带 a16 路)dq 48.8 / dk 48.9 / dv 49.0 dB。
⇒ **赢的不是"覆盖更多",是"消掉零 slack 的全排空"** —— 全排空是唯一在发射之外**额外加停顿**的东西,
这也解释了为什么 r17 的搬 MFMA 只值 0.9%、而 r17 §3 的读突发化会 −1.4%。

### 4. ❌ r17 机制往里再推一层(`g2_hh`)判负:−0.35%

`g2_hh=1`:把上一个 q-half 的 GEMM2 收尾 d-tile(8 条 MFMA,操作数早已读完)交给下一个 half,
站在它的 4 条 `_ld_rd` 累加器初始化读与 GEMM1a 之间。ISA 证明机制落地了(8 处裸 `lgkmcnt(4)`
初始化读站点消失、寄存器头逐位相同、+13 条指令、`s_nop` 每大段 7→9),实测 raw 1.0126 /
校正 **−0.35%**。

⇒ **诊断错在"覆盖只算读之后的 MFMA"。** 那 4 条初始化读**之前**紧挨着 4 条 MFMA,矩阵管线异步
排空,这 ~64 cyc 的**管线残留(carry-over)本来就把这条 wait 遮住了**。
★ **修正机器模型:一条读的 wait 由它前后两侧的 MFMA 共同掩护 —— 读**之前**发射的 MFMA 一样算。**
按这条重新看 base 剩下的两个"裸"家族,两个都是假警报:`RD:v103`(初始化读,前面有 4 条 MFMA)
与 `TR:v87`(GEMM2 dt=0 预读,被故意停在 softmax/pack 那段很长的 VALU 阴影里)。
⇒ r17 §5 让"按家族归属低尾再逐个挂 thunk"的处方**到此收敛**:低尾里已经没有真正暴露的家族了。

### 5. 下一步(按信息量排序,不再是覆盖轴)

1. ★★★ **PMC 定分叉**(§1 末):`SQ_INSTS_VALU_TRANS_F32` / `SQ_VALU_MFMA_BUSY_CYCLES` /
   `SQ_BUSY_CYCLES` / `SQ_WAIT_ANY` on b4。把 6339 cyc/trip 唯一分给 trans / wait / 发射串行。
   在这个数出来之前,任何"再搬一次 MFMA"的臂都是在 0.9% 的老矿脉上继续刮。
2. ★★ 若判为 trans/发射:**目标改成砍指令条数**,而且优先砍 `v_exp_f32` 之外的 VALU
   (512 条 exp 是 `BLOCK_Q x BLOCK_KV / 64 / 4` 的算术下界,删不掉;512 条 `v_cvt_pk_bf16` +
   256 条 `v_pk_mul` 已是 packed 形式)。真正可动的是 864 条 LDS/atomic 与 573 条 wait/nop/寻址,
   r16 §5 的「dS 写侧预 swizzle」正好减 LDS 条数,与本轮正交。
3. ★ 若判为 pipe-bound:那就只剩**每 trip 的 MFMA 条数**。160/head-step 是这套 tiling 的算术下界,
   唯一的结构性冗余是 **GEMM3 对整条带做收缩(掩掉的零 tile 也算)** —— 只在对角 q-block 上有量。

---

## r19 — PMC 判决 §5.1 的分叉,并把 r18 的两个机器数改掉

### 1. ★★★ `v_exp_f32` 在 gfx950 上是**半速(8 cyc 发射)**,不是四分之一速

b4 单次 dispatch:`SQ_INSTS_VALU_TRANS_F32` = 67,252,224 → 508.7 exp/trip ≈ 512 ✓(算术下界对上)。
把 `SQ_ACTIVE_INST_VALU` 按类建模(quad-cycle 单位):
MFMA 1280×1 + EXP 512×**2** + PACK 576×1 + VALU 330×1 + SHUT 8×1 = **3,218**,实测 **3,246**(差 0.9%)。
四分之一速会预测 4,246,全速会预测 2,710 —— 只有半速对得上。
⇒ **推翻 r18 的「exp 占发射预算 37%、kernel 是 issue-bound」。** exp 实际只占 ~16%。

### 2. ★★★ r18 的「76% roofline」和 1402 MHz sclk 是错的;真实 MFMA duty ≈ 52–57%

`rocm-smi` 在这台机器上读不准(用 `_r3_bwd.py` 采样时 kernel 早已因 JIT 缓存跑完,采到 158 MHz 空闲值)。
★ **改用 `GRBM_GUI_ACTIVE ÷ 8 XCD ÷ dispatch 时长** 反推时钟**:40,318,962 / 8 = 5,039,870 cyc / 2.764 ms
⇒ **~1823 MHz(profiling 下)**,未 profiling 约 2.0 GHz。MFMA 管线深度实测 16.00 cyc/inst 确认,
`MFMA_BUSY`/trip = 20,626 cyc,trip ≈ 36,350 cyc ⇒ **duty ≈ 52–57%,还有 ~1.75× 余量。**
三分账(b4):ACTIVE_INST_ANY 51.93% / WAIT_INST_ANY 30.80% / WAIT_ANY 17.18%;
`WAIT_INST_LDS` = WAVE_CYCLES 的 8.06% = WAIT_ANY 的 46.9%。
每 trip 总发射 18,876 cyc < MFMA 管线 20,480;非 MFMA 发射 13,756 < 15,360 影子
⇒ **完美调度装得下,那 ~15,870 cyc 的管线空转是延迟/会合,不是发射带宽。**

### 3. ❌ r16 §5 的「dS 写侧预 swizzle」——**按构造关闭**(这次是读源码确认,不是推导)

dS 的 LDS 布局本来就是 `[kv][qp]` 且带 `qp ^= 8*(kv&7)` 置换,**目的正是**让一个 lane 打包的 8 个
q-连续 dS 值变成一段连续 → 一条 `ds_write_b128`(见 `_g3s_wbase`)。改存 `[q][kv]` 好让 `_g3_tr` 用普通
`ds_read_b128`,每条现有写要拆成 8 条 strided `ds_write_b16`(96 → **768 条 LDS 写/trip**),或 ~1536 次
跨 lane 转置。**减 32 条读要付 672 条写。** 该臂到此关闭,不要再开。

### 4. ❌ 消掉零 slack 全排空**本身**不值钱(`g2_hoist`:机制落地、分数为零)

r18 §3 的诊断「softmax→GEMM2 交接处 6 次/trip 的 `PpUUUUPPPP | lgkmcnt(0) | MTMTTT`,**Mback = 0.0**」
是真的。把 GEMM2 的 TR 环读从 exp 链**之后**提到**之前**(同样的读、同样的顺序、同样的操作数):
ISA 上该家族整族消失,同样 6 个 segment 的 drain 变成 `EMEMEME|ME`,**Mback 0.0 → 6.0**、Mfwd 2→4,
指令 3997→3994,寄存器头逐位相同、spill 0。实测 raw **1.0157 vs 零点 1.0170**,校正 1.0070 vs 1.0117。
⇒ ★ **修正 r18 §3 的结论:「零 slack 全排空」不是一个可以按图索骥去消的成本项。**
r18 认为 `g1_pf` 赢在"消掉全排空"——本轮直接做了这件事,拿到 0。静态 gap/coverage 代理**再次**
失效,黑名单继续有效:裸 wait 条数、Mback 直方图、指令/wait 条数,没有一个能预测 wall。

### 5. ❌ `g2d=2` 在宽带 a16 D=64 上重新定价:仍然负(−0.55% 校正)

r18 的立论是「老结论成于 `g1_pf` 把预取挪走之前,槽位竞争前提已变」,值得一试。
ISA 全面更优:3997→3958 指令、`s_waitcnt` 264→227、vgpr/agpr 495/239→493/237、spill 0、
MFMA/PACK/EXP/LDS_TR/atomic 逐条相同。实测 raw 1.0097 / 校正 1.0062(零点 1.0170 / 1.0117),
**raw 与校正同向为负**,是本轮唯一两个统计量一致判负的臂。⇒ `g2d` 轴定价完毕,不要再开。

### 6. ❌ `g2_hs` 的剂量响应**不是单调的**:r5 在档位 1 的赢不延伸

ISA 筛(零寄存器代价,三档 vgpr 495 / agpr 239 / spill 0 全同):暴露小段(fence 后、32 条 LDS_TR)
MFMA 18 → 26 → 34,大段 142 → 134 → 126,MFMA 总数守恒。看起来是"把 MFMA 从有富余的大段搬进
饥饿的小段",机制与 r5 的赢完全同源。实测(校正):

| g1_pf \ g2_hs | 1 | 2 | 3 |
|---|---|---|---|
| 1 | 1.0117(零点) | — | 1.0195 |
| 3 | **1.0183** | 1.0080 | 1.0059 |

`g1_pf=3` 一行**单调下降**。`(1,3)` 那个 1.0195 是单次、且依赖全轮最大的一次 guard 校正(−0.94%),
没有采信。我曾用 `G1_PF_AT = DT - G1_PF` 落在被 `tail_sink` 交接走的 d-tile 上来解释 (3,3) 的反叠加,
并预测 `G2_HS < G1_PF` 的 `(3,2)` 会叠加 —— **`(3,2)` 实测 1.0080,该机制解释被证伪。**

### 7. ★★★ 方法论:guard 几何均值校正**被跨 session 独立复测验证**,而 raw 分数把配置排反了

同一天 7 次 bench,guard 几何均值(三个 guard 形状代码逐字节相同、本该恒等于 1.000)读数:
+0.52% / +0.35% / +0.87% / −0.26% / −0.94% / −0.20% / +0.15% —— **摆幅 1.8%**,
远大于 harness 宣称的 0.3% 分辨率。三个独立形状**同向**偏移 ⇒ 是 per-session 的 A/B 槽位偏置,不是 per-cell 噪声。

验证:拿校正值去对编排器在**不同 session** 对同一棵树的独立复测:

| 配置 | 本轮 guard 校正值 | 编排器独立复测 |
|---|---|---|
| `g1_pf=3, g2_hs=1` | 1.0183 | 1.0195 (r5) |
| `g1_pf=1, g2_hs=1` | 1.0117 | 1.0143 (r6) |

两者都在 **0.26%** 内吻合;而 raw 分数(1.0158 / 1.0170)把这两个配置**排反了**。
⇒ ★ **决定"留什么在盘上"要用校正值,不要用 raw** —— 编排器复测的期望值 E[raw] = 真值,
而真值的最佳估计是校正值。★ 同时:**单次 bench 的校正值自身带 ±0.5% 噪声**(三个 guard 的 resid
0.4–2.1%,几何均值 ≈ ±0.5%),所以 <0.8% 的差异必须复测,不能单次下结论。
⇒ 这也说明 state.json 里 r5 的 +2.01% 是**高估**:`g1_pf` 3 vs 1 的真实差约 +0.65%,不是 0.51% 那个巧合数。

### 8. 下一步(按信息量排序)

1. ★★★ duty 只有 ~56%、且证明"完美调度装得下" ⇒ 瓶颈是**延迟/会合**,而不是发射或 trans。
   下一个要测的量是 **barrier 会合代价**:8 个 wave 锁在同一 head-step,`s_barrier` 前后的
   `SQ_WAIT_ANY` 分段归属(按 segment 而不是按 wait 站点),看早到的 wave 等了多久。
2. ★★ 若会合是主项,唯一的结构杠杆是**解开 head-step 的 barrier 锁步**(让相邻 head-step 重叠),
   而不是继续在段内搬 MFMA —— 段内搬运的轴(g2_hs / g2_hh / g2d / g1_pf)本轮已全部定价完毕。
3. ★ 所有 <0.8% 的候选一律要求**两次独立 session 的校正值**同向,否则不采信。

## ★★★ r20(bwd D=64,a16 dQ 路,occ=1 4-wave):barrier 轴**双向定价**完毕 —— 边际 barrier 只值 0.10%/个,r19 减法探针的 3.8% 里有 2/3 是"允许 wave 漂移"

本轮四臂全部判负,盘上留回原树(md5 与基线逐字节相同)。但两个方向的定价第一次把
"会合代价"和"barrier 指令代价"分开了,这直接改掉 r19 §8 的下一步描述。

### 1. ★★★ 加法方向:纯加 8 个 barrier(`HS_WAR_BAR` 在 D64 打开),b1 −0.8%

`HS_WAR_BAR = G3_DEFER and ...` 在宽带 a16 D=64 上 `G3_DEFER=False`,所以这个 head-step
首部的 WAR barrier 是关的(WAR 边已被上一 head-step 的 `[lgkmcnt(0) drain, barrier]` 清掉)。
把它强行打开 = **只加 barrier、不动任何别的结构**,ISA 证实:

| | 基线 | +HS_WAR_BAR |
|---|---|---|
| hot-loop `s_barrier` | 16 | **24** |
| vgpr / agpr / arch / spill | 495 / 239 / 256 / 0 | 495 / 239 / 256 / 0 |
| lds | 86016 | 86016 |
| instr / mfma / setprio | 4040 / 1280 / 32 | 3996 / 1280 / 32 |

实测:b1 1.0160 / 1.0151(零点 1.0238)= **−0.78% / −0.87%**;b2 1.0128(零点 1.0145)= −0.17%。
⇒ ★ **边际 barrier ≈ 0.10%/个(b1)、0.02%/个(b2)**,16 个 barrier 的全账 ≈ 1.6%(b1)/ 0.35%(b2)。
附带发现:加 barrier 让 instr 反而 **少 44 条**、`s_nop` 少 16 条(它是调度墙,让编译器忘掉状态),
静态指标全面变好、wall 仍然变差 —— 又一个失效的静态代理(第 6 个)。

### 2. ★★★ 减法方向(合法):`QDO_RING` 把 16 个 barrier 退成 9 个,寄存器**按住不动**,仍然 −1.6%

结构:Q/dO 与 dS 各给第二个 slot,head h 的 dS fence 顺带发布 head h+1 的 tile,
head 边界那对会合消失(`QDO_TAIL`,源码里本来就有、被写死为 `False`)。ISA 理想:

| | 基线 | qdo_ring |
|---|---|---|
| `s_barrier` | 16 | **9** |
| vgpr / agpr | 495 / 239 | **494 / 238** |
| arch / spill / setprio / mfma | 256 / 0 / 32 / 1280 | 256 / 0 / 32 / 1280 |
| lds | 86016 | 135168(仍 1 WG/CU) |
| instr | 4040 | 4042 |

三形状 SNR 48.7–49.2 dB PASS。实测 wall:b4 **−1.20%** / b2 **−1.63%** / b1 **−1.73%**。
把 §1 的边际价代进去:退掉 7 个 barrier 本该 **+0.7%**(b1),所以**双 slot 的"提前发布"本身要 ≈ −2.4%**。
⇒ ★ **更正 2026-08-11 D128 §4(b) 的归因**:那条 −6.0% 当时记成"+26 vgpr 的寄存器代价"。
本轮寄存器按平(−1 vgpr / −1 agpr / spill 0 / 占用不变)后损失照旧复现 ⇒ 代价不是寄存器,
而是**发布点搬家**:h+1 的 `ds_write` 突发从 head-step 首部挪进了 GEMM3 转置读突发的中间。
(与 r17 §3「读的突发化判负 −1.4%」同一族机制:这颗核对 LDS 端口在密集段的争用极敏感。)

### 3. ★★ ❌ 在退役点补一道"零成本"编译器墙(`rocdl.sched_barrier(0)`)不但救不回来,还更差

假设是"退役的会合同时是 `iglp_opt` 交织区的边界"。补墙后:b1 1.0037 / 1.0022、b2 0.9871 / 0.996、
b4 0.9813 ⇒ 三格一致 **≈ −2.1%**,比不补墙(−1.2…−1.7%)更差(instr 4040→4042,lgkm0 64→59)。
⇒ ❌ 别再试"用 `sched_barrier(0)` 替代退役的 `s_barrier`":`sched_barrier` 硬件零成本,
但它**新增一个区边界**,分配器会照它办事;`EXP_IGLP` 的区无论有没有它都在。

### 4. ★★★ 三个计分格的 kernel 级分账(rocprofv3 `--kernel-trace --stats`,首次拿全)

| 格 | dkdv | odo | slotred | lset | odo min→max |
|---|---|---|---|---|---|
| b1(未分块) | **91.54%** (avg 1387 µs) | 1.73% (28.1 µs) | 1.02% | 0.31% | 27.1 → 30.3 µs |
| b2(未分块) | **92.16%** (avg 2757 µs) | 1.75% (56.2 µs) | 0.60% | 0.16% | 55.4 → 57.6 µs |
| b4(2 chunk) | 77.48% (avg 2720 µs) | **15.88%** (avg 600 µs) | 1.39% | 0.09% | **59.8 → 1527 µs** |

三条硬结论:
1. ★ **b1/b2 的 odo 是串行的,而且已经贴着 DRAM 字节地板**:b1 要搬 O+dO(134 MB)+ 图零填(67 MB)
   + delta(2 MB)≈ 203 MB,28.1 µs ⇒ **7.2 TB/s**。⇒ ❌「把 odo 藏起来/压小」这一族在 b1/b2 上
   由实测关闭(dkdv 外的全部尾巴只有 3.1% / 2.5%),杠杆必须落在 dkdv 体内。
2. ★★★ **同一个 odo 并发跑时每次调用慢 25×**(59.8 → 1527 µs,字节数完全相同)⇒ 在 occ=1、
   独占每个 CU 的 body 旁边,"藏"起来的流式 kernel 是按**全价 CU-time** 收费的,并发不免费。
   这解释了 `_A16_MIN_BAT` 4→8(b4 改走不分块、代价换成 ~107 µs 串行 delta+fold)为什么读 **+0.03%**:
   并发损伤 ≈ 串行暴露,分块机器在 b4 今天是个平局。
3. ★ 同一进程内 15 次相同调用,dkdv 的 StdDev = **8.5%**(b1)/ 7.8%(b2),min/max 1263/1627 µs
   ⇒ 这是硬件侧(功耗/时钟)的摆幅,独立佐证 pitfalls/02 的 min-of-7-passes 尺子设计。

### 5. ★ ISA 门再省一次尺子:`g3_defer=True` 在 D64 宽带上

17 barriers(`HS_WAR_BAR` 被它带开)、instr 4030、lgkm0 64、**`vmcnt(0)` 2→8** ⇒ 不花尺子直接排除。
`vmcnt(0)` 计数是这颗核最便宜的坏味道指标(gfx950 只有一条 in-order vmcnt)。

### 6. ❌ dQ 图零填改 `cache_modifier=2`(sc1,绕 MALL 写):未分辨/偏负

b4 −0.05% / b1 +0.05% / b2 −0.87%,l70b 1.0037 未触顶,SNR PASS ⇒ 撤回。
配合 ⑩ 的台阶可解释读数为什么分格反号(**未实测的机制假设,待验**):b1/b2 的 dQ 图是 67/134 MB,
**装得进 256 MB MALL**;b4 的 268 MB 装不进。若成立,它同时解释了本战役一个稳定现象 ——
体内调度类的收益一律 **b1 > b2 > b4**(本轮零点 1.0238 / 1.0145 / 1.0027 就是这个梯度)。

### 7. 两个 harness 坑

1. rocprofv3 `-d DIR -o r` 把统计写在 **`DIR/r_kernel_stats.csv`**(没有 run-id 子目录),
   `DIR/*/…` 这种 glob 一个都匹配不到 —— 用 `find DIR -name "*kernel_stats*"`。
2. `_r2_snr.py` 钉死在 B=1 / Hq=32,**既不会构出 batch 分块的 a16 计划,也构不出 8 头 GQA 组**
   ⇒ 本轮新增 `_r9_snr.py <S> <B> <Hq>`(会打印 `chunks=N`),凡改到分块计划的臂都要用它过 gate。

### 8. 下一步(由本轮双向定价指名)

1. ★★★ barrier **个数**这条轴已双向定价:加 8 个 −0.8%、减 7 个(合法)−1.6%,全账 ≈1.6%(b1)。
   r19 减法探针那 +3.77% 里,只有约 1/3 是 barrier 指令,余下 ≈2% 是**允许 wave 漂移**(探针结果不正确)。
   ⇒ 真正要打的是**到达时刻的方差(convoy)**,而不是 fence 个数;合法手段是让 4 个 wave 的
   每 head-step 延迟更齐,而不是拆掉 fence。
2. ★★ 第一个可测的方差源:对角块 `_if_wave` 的四类 wave(clear/diag/live/dead)工作量不等 ——
   先用 PMC 或按 wave 的 `SQ_WAIT` 分段确认"最后到的是谁",再决定要不要把四类补齐。
3. ★ b1/b2 的 91.5–92.2% 都在 dkdv 一个核里,b4 的 odo 虽占 15.9% 但每个字节都是必须的、
   且膨胀纯粹来自 CU 饥饿 ⇒ 三格的杠杆是同一个:dkdv 体内。

## ★★★ r21(bwd D=64 a16,b4 分块计划):**每 CU 的 WG 深度不是一阶项,每 XCD 同时活着的 (batch,kv-head) 份数才是**

三条臂动的都是"b4 每个 launch 有多少 WG",而 b1/b2/l70b/两个窄带格全部 `_nbc==1`、逐字节不变(天然对照)。
基线同段 `--cell b4` = **1.0100 / 1.0044**,b2 = 1.0117。配对 `R3_B=4 R3_N=9` kernel-trace(µs):

| 臂 | WG/launch | WG/CU | 每 XCD 同时活的 batch | dkdv avg | dkdv min | slotred avg/max | **b4 尺子** |
|---|---|---|---|---|---|---|---|
| `_A16_Q_SPLIT=1`(去掉 slot 折叠) | 512 | 2 | 1 | 2889 | 2553 | **核消失** | 0.9263 / 0.9353 |
| 基线 `q_split=2`,2 chunk 串行 | 1024 | 4 | 1 | 2793 | 2463 | 54.0 / 972 | 1.0100 / 1.0044 |
| `q_split=4`(8 WG/CU,fold 4 slot) | 2048 | 8 | 1 | **2716** | 2504 | **316.9 / 972** | 0.9954 / 0.9945 |
| 2 chunk 各自一条 stream 并发 | 2×1024 | 8 | **2** | **3697** | 2790 | 45.5 / **90** | 0.9853 / 0.9884 |

1. ★★★ **决定性的一行是第四行**:字节、slot 数、q 走法、WG 总数全部与基线逐字节相同,唯一的变化是
   两个 batch-disjoint 的 chunk 同时驻机 ⇒ **dkdv 体自身 +32%**(avg 2793→3697、min 2463→2790)。
   一个 XCD 的 L2 只有 4 MB,而常驻窗口(256 个连续 block_id = 16 个 band × q_split,同一 batch/头)
   的共享操作数 ≈ Q/dO 2 MB + K/V 0.5–2 MB;并发两个 chunk 直接把它翻倍。
   ⇒ **"多给点 ready work"在 occ=1 的 body 上是负收益**;03 卡"L2 命中的守恒量是 WG 存活时间"在这里
   的等价表述是:**别让两份 Q/dO 同时活在一个 XCD 上**。
2. ★★ q 轴双向定价完毕,**再次证实库里"别再扫 Q_SPLIT"**(§r?"离线模拟 makespan 都是 1080"):
   q=1 −7.3%、q=4 −1.2%。但两次读数给出**新的分解**:q=4 的 **body avg 反而快 2.8%**(窗口里 band 数
   16→8,足迹更小),输在 **fold 的每-slot 单价超线性**(2 slot 54 µs → 4 slot 317 µs,字节只 2×)。
   ⇒ q_split 的真实约束不是 makespan,是 **slot workspace 的折叠单价**;把折叠去掉,这 2.8% 就可收。
3. ★ **辅助核的饥饿是结果不是原因**:并发臂里 slotred max 972→90 µs(有空位就不饿了),wall 却更差;
   反过来 q=4 臂 fold 更贵、body 更快,wall 也更差。⇒ 给 odo/slotred 定价必须看 **body 的足迹变化**,
   不能只看这两个核自己的 µs(r20 §3"并发损伤 ≈ 辅助核孤立时长"由此细化:那是 CU 饥饿的量级,不是 wall)。
4. **管理者点名的 R8-P1/P2 本轮不重走**:`_A16_MIN_BAT` 4→8 已在 r20 读 **+0.03%**、dQ 零填
   `cache_modifier=2` 已在 r20 读 b4 −0.05%/b1 +0.05%/b2 −0.87% 并撤回。本轮把同一主题换到未定价的
   q/stream 轴上,并给出了 MIN_BAT=8 为什么必须是 0 的机制:**单个大 launch 的常驻窗口仍然只有一个
   batch**(`_xcd=block_id%8` + q 最快 + band 次快 + batch 最慢),所以它和分块的 L2 足迹完全一样,
   chunk 的全部价值只在藏 odo,而 odo 藏不住。

### 下一步(由本轮足迹模型指名,已 sized)

1. ★★★ **把 dK/dV 的 slot 折叠整个删掉**:a16 的 dQ 路早就用 `buffer_atomic_pk_add_bf16` 直写图、
   每个元素 **32** 份贡献都能过 50 dB;dK/dV 每个元素只有 **q_split 份**(2)贡献,凭同一条先例改成
   原子 epilogue 即可,零填复用 odo 的 `FILL_IMG`。账:省 134 MB 写 + 134 MB 读 + 67 MB fold 写/iter,
   删掉一个核以及它**完全暴露在 `st` 尾部的最后一次折叠**(26 次调用里 13 次是这种),
   并且把 q_split 与 workspace 解耦 ⇒ 上表第三行那 **+2.8% body** 才能收进来。gate 用 `_r9_snr.py`。
2. ★★ 在**深度不变**的前提下压常驻窗口的足迹:窗口里 2 MB 的 Q/dO 是大头,源码在 `QDESC_R` 旁边
   自己点名的"rotate the GQA head order by band"就是直攻它;并发臂给出了灵敏度系数(足迹 2× → body 32%)。
3. ★ 按同一模型给 `_A16_BLOCK_KV` 256→128 定价(窗口里 K/V 那一半减半,band 数翻倍)。

## ★★★ r22(bwd D=64 a16,occ=1 4-wave):r21 点名的 ★★★ 一号杠杆(删 dK/dV slot 折叠)**上尺子判负** —— 原子 epilogue 的代价不在字节,在"暴露的同线 RMW 突发"

本轮两臂全部判负,盘上留回原树(md5 `6102ef3da7cb0d745d2ef4a46cc24f75`,与基线逐字节相同)。
零点(本 session,cheap ruler,清 cache):b4 1.0071 / 1.0036(mean 1.0054),b1 1.0219 / 1.0121(mean 1.0170)。

### 1. ★★★ `dkdv_atom`:dK/dV 用 `buffer_atomic_pk_add_bf16` 直加零填输出 = b4 −1.2% / b1 −4.8%

完全按 r21「下一步 #1」实现:builder 加 `dkdv_atom`,SRD 去掉 `split_idx*Skv*RD_STRIDE_KV` 这一项,
`_store` 把每个 `o_pack`(2 dword)换成 2 条 `raw_ptr_buffer_atomic_fadd`(照抄 `_g3_a16` 的先例),
host 只分配 **1 个零填 slot**、prime 的双次 dispatch 后补零、`_slot_plan=None`,
`_reduce_dkdv_slots(..., n_slots=1)` 退化成零成本直通(fold 核彻底消失)。gate 全过:

| | 基线 | `dkdv_atom` |
|---|---|---|
| vgpr / agpr / arch / spill / lds | 495 / 239 / 256 / 0 / 86016 | **完全相同**(热循环 shuttle 也同为 5) |
| `_r9_snr.py 2048 1 64`(计分 GQA 组 8) | dq 48.8 / dk **48.9** / dv **49.2** | dq 48.8 / dk **47.7** / dv **47.9** |
| `_r9_snr.py 2048 4 32`(分块计划) | — | 47.7 / 47.9,PASS |
| b4(两读) | 1.0071 / 1.0036 | **0.9965 / 0.9894** = mean −1.23% |
| b1 | 1.0219 / 1.0121 | **0.9707 / 0.9653** = mean −4.82% |

1. ★★★ **账全对,结论全反**:省了 134 MB 写 + 134 MB 读 + 67 MB fold 写、删掉一个核和它暴露的尾部折叠,
   wall 仍然掉 1.2%(b4)/4.8%(b1)。⇒ **dK/dV 折叠的成本不是它的字节**;r21 §2 那句「把折叠去掉这 2.8%
   就可收」的前提(折叠贵在字节)本轮被否。
2. ★★★ **机理:同一条指令,摊开的位置决定它的价钱**。dQ 路每 trip 64 条同款原子活得很好,因为它们被
   `g3_st_hs`/`g3_st_g2` 摊进 GEMM2 的 MFMA 流里;dK/dV 的 epilogue 是 WG 末尾**完全暴露**的
   1024 条原子突发(64 KB/WG),而且 `q_split` 的两个 WG(block_id 相邻、同 XCD、同相位)**同时**
   RMW 同一批 cache line。b1 掉得比 b4 狠 4×,正是因为 b1 只有 1 个 batch、两个 split 相位完全对齐。
   ⇒ 一般教训:**把 store 换成 atomic 之前先问它是不是暴露段**;原子的先例不能跨"摊开度"移植。
3. ★★ **顺带证伪 r21 对精度的估算**:r21 说「dQ 每元素 32 份贡献都能过 50 dB,dK/dV 只有 2 份,凭同一
   先例即可」。实测 bf16 原子加**确实**比 fp32 折叠差 **1.2–1.3 dB**(dk 48.9→47.7、dv 49.2→47.9;
   dq 48.8 不动,是干净的对照)。机制:`_reduce_dkdv_slots` 是升 fp32 求和后**只舍入一次**,而
   `0+a` 再 `+b` 的硬件 pk_add_bf16 多一次舍入。虽然仍过 45 dB gate,但这属于"降低中间值精度",
   **按 16-bit 铁律本身就不该上**;wall 只是又给了一票。⇒ 记账:**这一系(dK/dV 原子 epilogue)双重关闭**。

### 2. ❌ `mfma_tie` 减法(管理者点名的寄存器轴):`mfma_tie=1` = b4 −0.80%,b1 +0.48%(噪声内)

先用 `R1_KW` 做零改动 ISA 勘察,拿到三个读数 —— 并**改掉 goal/管理者指令里的两个机器数**:

| `mfma_tie` | 语义 | vgpr / agpr / arch / spill | ISA |
|---|---|---|---|
| 3(**当前部署**,L5241) | dV+dK 两族 accumulator 全 tied | 495 / 239 / 256 / 0 | 与 base **逐字节相同**(md5 `398c4ac2…`) |
| 1 | 只 tie dV(dK 放开) | **481** / 225 / 256 / 0 | instr 4019,s_nop 108 |
| 2 | 只 tie dK | **483** / 227 / 256 / 0 | instr 4030,s_nop 98 |

1. ★★★ **「tied-operand 把 GEMM2 accumulator 钉进 AGPR」不是未试的杠杆,它就是现役配置**
   (`mfma_tie=(3 if D==64 else 2) if a16 else 0` + `mfma_tie_cons=1`)。`grep -c "v_mfma_f32_16x16x32_bf16 a\["`:
   base 1408 / tie1 704 / tie3 1408。⇒ 这条轴上还能动的只有**减法**。
2. ★★★ **「accum_offset 256 以上还有 33 个不花 occupancy 的 dword」被证伪**:实测 agpr **239/256**、
   amax 238、arch 钉死在 256 上限。热循环 MFMA 操作数普查:512 条已是 `dst=a A=a B=v C=a`,
   热循环 accvgpr shuttle 只有 **5** ⇒ GEMM3 的 accumulator LLVM 自己也放进了 AGPR,
   再显式 tie 属于 methodology/06 r15 的"零和替换"(那次实测 −0.49%),不值机器时间。
3. 减法实测:b4 1.0010 / 0.9937(零点 1.0054)= **−0.80%**;b1 1.0219(零点 1.0170)= +0.48%,
   `resid` 0.67–1.41% 说明 b1 那一读在噪声内。⇒ **省 14 个寄存器在 occ=1 上买不到东西**,
   与 methodology/06 r16(放掉 72 vgpr+72 agpr = −2.05%)同向。**tie 轴到此双向定价完毕,别再扫。**

### 3. 本轮顺手关掉的三条(都是"读卡/读源码就省下一次 bench")

- `g3_st_w`(dQ 原子交棒粒度 1/2/4):源码注释自己记了结论 ——「pairs match ONE q-half's d-tile sites;
  going finer puts a vm op in every d-tile, which costs more of GEMM1's exp cover than the spread returns」。
- `QDESC_R=1`(q 相位对齐):§r?(第 1058 行)已把 `qdesc_r` 2/8/16 定价为 ±0.4% 平,且第 1187 行记了
  `QDESC_R=1` = **制造最大散射**(QDESC 本来就把并发 band 压在同几个 q block 上,rotate 是为了散开 dQ 原子冲突)。
- `_A16_ODO_AHEAD`(卡里没有,看着像未定价):**在计分形状上是恒等变换** —— b4 的 `_nbc=2` ⇒
  `_queued=min(2,2)=2`,b1/b2 的 `_nbc=1` ⇒ `_queued=1`,两档在 nbc≤2 时发射序列相同。别去测。
- 另:`img = dq.view(-1) if a16_nat` ⇒ D=64 面上**没有** image→dq 的搬运核(`_unpermute_dq_a16` 只走 D=128),
  这条想象中的 1.3% 不存在。

### 下一步(本轮把 r21 的 #1 换掉,新的 #1 由本轮读数指名)

1. ★★★ **不要删折叠,去修折叠的带宽效率,然后收 q_split=4 那 +2.8% 的 body**。r21 表:2 slot fold
   54 µs / 4 slot fold **317 µs**(字节只 2×,单价超线性)。4 slot 的账是 536 MB 读 + 134 MB 写 = 670 MB,
   317 µs ⇒ **2.1 TB/s,离峰值 8 TB/s 差 4 倍** ⇒ `_reduce_dkdv_slots`(L4607-4681)本身是慢核,
   不是"折叠这件事贵"。把它修到 ~90 µs,q_split=4 就是 +2.8% body − 0.7% fold ≈ **+2.1%**。
   本轮已证:折叠的**字节**不是问题(删掉字节反而 −1.2%),所以这一手方向是**效率**不是**删除**。
2. ★★ r21 #2 仍然有效(rotate GQA head order by band 压常驻窗口足迹),但注意它在源码里是
   "q 反向走"的**使能项**,单独上没有立论;要和 L4016 那条注释一起做。
3. ★ 若要碰 dK/dV epilogue,方向是**把 store 摊进 MFMA 流**(照 `g3_st_g2` 的做法),
   而不是换成原子 —— 本轮已经量出"暴露突发"是那 1.2% 的来源。

## ★★★ r23(bwd D=64 a16,b1/b2/b4 计分面):r22 点名的 ★★★ 一号杠杆(修 dK/dV 折叠的带宽)**在核内被证伪** —— 那 2.1 TB/s 是**并发时长**,不是核的速率;同一轮把 batch 分块**双向**定价完毕

r22 的下一步 #1 写的是「4 slot fold = 670 MB / 317 µs = 2.1 TB/s,离峰值 8 TB/s 差 4 倍 ⇒
`_reduce_dkdv_slots` 本身是慢核,修到 ~90 µs 就能收 q_split=4 的 +2.8% body」。本轮第一件事就是
把这个核**单独**架起来量(新探针 `_r12_fold.py`:三个计分几何 × 12 个 `(block,uc,vec)` 形状,
持续负载预热 + min-of-7×25 次发射,同时有 `R12_CHECK=1` 对 fp32 参考逐位核对)。

| 计分几何 | fold 几何 | 部署形状 µs / TB·s⁻¹ | 全扫最好 | 最好 ÷ 部署 |
|---|---|---|---|---|
| b1(q_split=**4**,整张) | 4 slot × 4.19 M elem | (128,1,4) 15.39 / **5.45** | (512,2,8) 13.63 / **6.16** | **+12.9%** |
| b2(q_split=2,整张) | 2 slot × 8.39 M elem | (128,1,4) 18.12 / **5.55** | (512,2,8) 16.37 / **6.15** | **+10.7%** |
| b4(q_split=2,每 chunk 一条 strided sub) | span=1024 / rstride=2048 | (128,1,4) 18.64 / 5.40 | (128,1,8) 18.55 / 5.43 | +0.5% |

1. ★★★ **核的真实速率是 5.4–6.2 TB/s,不是 2.1 TB/s**;而这台机器(1400 W 功率封顶、sclk 2070)
   实测 HBM 流地板本来就是 **5.95–6.2 TB/s**(methodology/03),**不是 8 TB/s**。⇒ 折叠核在部署形状上
   已经跑在机器流速的 **88–93%**,而最宽 tile 就落在流地板上。r21 表里那个 317 µs 是 b4 **并发**读数:fold 在 side stream 上
   与 occ=1 的 body 抢 CU(同表 slotred max 972 µs 就是同一现象),被当成了核的效率。
   ★ **一般教训:任何从 trace 里读出来的辅助核时长,若它与 body 并发,就不能除以字节当带宽用;
   要给辅助核定价必须把它单独架起来发射**(r21 §3 已经说过"饥饿是结果不是原因",本轮把它变成了硬数)。
2. ★★ 这条杠杆按尺子的真实价钱:fold 孤立时长 = b1 15.4 µs / wall 1.22 ms = **1.26%**、b2 18.1/2.48 ms = 0.73%,
   最宽 tile 省下的 1.76 µs / 1.75 µs ⇒ **b1 0.14% / b2 0.07% wall**(与本轮盘上两条改动的配对读数
   b1 +0.68% / b2 +0.19% 相符:另一半来自下面 §3 那条 dispatch,0.40% / 0.19%)。
   所以**下一个该动的不是折叠核内部,而是 q_split 这条轴本身**(见下一步 #1):
   r22 的账里唯一还站得住的是"q=4 的 body 快 2.8%",而它当年输的那 317 µs **根本不存在**。
3. ★ 管理者点名的 B 臂(让一个 WG 走多行,把 `sub` 的 rows×span 摊成一条线性走法)**用上表省下了一次实现**:
   b4 的 chunk 只有 span=1024 元素,TILE 上限就是 1024,而它在这个上限上离全扫最好只差 **0.5%**
   ⇒ 摊平 rows×span 的全部收益 ≈ **0.02% 的 b4 格**。⇒ 排"让 WG 走更多数据"这类改写之前,
   先用单独架起来的全形状扫描看**同 span 下最好与部署的差**,差不到 2% 就不必写代码。
4. 留在盘上的那一半:`_SLOTRED_CFGS` 由"够窄就行"改成**最宽优先**(`(512,2,8)` 领头,并按
   "TILE 能整除 span 且 WG 数 ≥ `_NUM_CU`"选),b1/b2/l70b 拿到 (512,2,8)、b4 的 chunk 落到 (128,1,8);
   12 个形状全部与 fp32 参考逐位相同(0 mismatch),所以这是纯减法。

### 1. ★★ `_A16_MIN_BAT`(batch 分块)**双向定价完毕**:b4 上是 0,b2 上是 **−2.6%**

r20 记过 `4→8`(= b4 不分块)= +0.03%;本轮用 4 读复测 **0.9831/1.0060/1.0053/0.9968 = 均值 0.9978**
对分块基线 4 读 **0.9925/1.0129/0.9952/1.0014 = 均值 1.0005** ⇒ −0.27%,**同号地确认这条常数在 b4 上是 0**。
新的一半是反方向:`4→2`(= 把 b2 也分块,窗口格与 b1 天然不变,因为 `_dq_a16_for` 要求 `window_left<0`
且 B=1 无法分块)⇒ b2 **0.9830 / 0.9865** 对基线 1.0138/1.0082 = **−2.6%**。trace 给出机制:

| 臂 | body 数 × µs | 合计 body | 暴露辅助 | launch gap |
|---|---|---|---|---|
| b4 分块(部署) | 2557 + 2552 | **5109** | ~100 µs(1.9% wall) | 0 |
| b4 不分块 | 5145(单发) | 5145 | 140 µs(2.6% wall) | 0.00% |
| b2 不分块(部署) | ~2440(单发) | ~2440 | ~95 µs(3.8% wall) | 0 |
| b2 分块 | 1350 + 1324 / 1382 + 1334 | **2674 / 2716** | odo 侧 902 µs 并发 | **999 µs / 11.9%** |

1. ★★★ **每多一次 body launch 的"抽干代价"按这一 chunk 覆盖机器的份数放大,不是常数**:
   b4 的 chunk 是 2048 WG(8/CU)⇒ 两段 body 合计 **比单发还少 36 µs**(抽干≈0);
   b2 的 chunk 是 1024 WG(4/CU)⇒ 两段合计 **比单发多 230–280 µs**(+10%),
   而它藏掉的 odo 只有 ~25 µs。⇒ 判据不是"batch 够不够分",是**分完之后每 CU 还剩几个 WG**;
   `_A16_CHUNK_WGS=512` 这条门槛比实测拐点低 2–4 倍,谁把它当"可以分"的依据都会踩这一脚。
2. ★★ b2 分块还额外冒出 **999 µs 的 launch gap(11.9%)**:side stream 的 odo 被拉成 902 µs、
   fold 前出现 453–526 µs 的空窗 ⇒ event 依赖在 chunk 太小的时候**真的会堵队列**,
   而同样的 event 结构在 b4 上 gap 恒为 0。⇒ **流/event 计划的正确性只能按 gap 验,不能按设计意图信**。
3. ★ **"side stream 上的流式辅助核几乎免费"这条本轮第一次被直接量出来**:b4 的 body0 = 2557 µs
   (身下压着 1359 µs 的 odo+fold),body1 = 2552 µs(独占)⇒ 差 **0.2%**。
   ⇒ 在 occ=1 的 body 旁边跑纯流式核,代价不在带宽争用,只在它自己能不能被完全遮住。

### 2. ★ 记账口径纠正:「odo = backward 的 14.95%」是**核时长之和**的份额,不是 wall 的份额

R8 留下的这个数被后续几轮当成"有 15% 可拿"。本轮按 rocprofv3 timeline 逐 dispatch 算 gap 与暴露段:
b1 的暴露辅助 = **3.9% wall**(odo ~30 µs + lset 5.7 µs + 尾部 fold ~10 µs,wall 1.22 ms),
b4(部署分块)= **1.9% wall**,b4 不分块 = 2.6%。而 odo 自己在 b4 不分块时 103 µs 里搬 ~805 MB
(O + dO 读 + 图零填写)⇒ 表观 7.8 TB/s,**已经在流速之上**(nt 写 + 无读的零填路径)。
⇒ **辅助核这一家在 wall 上的总可取量是 2–4%,不是 15%**;要继续拿必须按"暴露 vs 被遮"而不是按核时长排。

### 3. ★ 白捡的一个 dispatch:`torch.zeros(1)` 占位参数 = 每次 backward 一个 fill 核

`_cu_placeholder` 每次调用 `torch.zeros(1, int32)` 造一个**永远不会被读**的 `cu_seqlens` 占位
(只在 `const_expr(varlen)` 下才读,dense 面恒 False)。torch profiler 把它记成
`aten::zeros/zero_/fill_` 2.0 µs,kernel trace 里是 body 之前一条 **4.6–5.6 µs** 的
`vectorized_elementwise_kernel`。改成 per-device 缓存后这条 dispatch 消失,b1(1.22 ms)上是 **~0.4% wall**。
⇒ ★ **launcher 里任何"给个空张量占位"的写法都是一次完整的核发射**;occ=1 的核在 b1 这种短 wall 上
这一条就够看见。查法:`torch.profiler` 的 op 表里找与 kernel 名对不上的 `fill_`/`zero_`。

### 4. 尺子本身:b4 这条臂的单 session 单读摆幅是 **2.0%**,`cur` 腿比 `ref` 腿噪

同 session 4 读:`ratio` 0.9925–1.0129(2.0%),而各读自报 `resid` 只有 0.49–0.67%;
分解到两腿:`cur` 5.0255–5.1020(1.5%)、`ref` 5.0594–5.0903(0.6%)。
⇒ pitfalls/02 的"单格 0.35%"**在 b4 + a16(side stream + event)这条臂上不成立**;
**b4 的臂至少要 4 读取均值**,2 读只能判 >2% 的东西。b1 相对安稳(resid 0.69%)。

### 5. 工程细节:部署副本是只读的,ABBA 同 session 配对不能靠"跑中间改常数"

想在一次 `remote.sh` 里交替两个常数值取 ABBA,结果 4 次改写全部 `PermissionError`
(容器里的 repo 挂载为只读),4 个读数其实是同一条臂 —— **幸好这恰好变成了上面那张噪声表**。
⇒ 跨臂配对只能靠"本地改 → rsync → 一次 bench",所以**同 session 配对在这套 harness 上只对
"一条臂 + 它自己的 ref 腿"成立**;两条臂之间永远是跨 session,必须按 0.8% 门槛和两次同号来判。

### 下一步(由本轮读数指名)

1. ★★★ **重新给 `q_split=4` 在 b2/b4 上定价** —— r22/r21 把它判负的那笔账(fold 317 µs)已被本轮证伪:
   真实代价是 b1 34–37 µs / b2 16–19 µs 这一档,而收益是 r21 量到的 **body avg +2.8%**(常驻窗口里
   band 数 16→8、足迹更小)。现在 fold 还拿到了最宽 tile。这是库里目前唯一"收益已量出、代价刚被改小"的轴。
2. ★★ **把 `lset`(5.7 µs、与 B 无关)并进 `odo`**:两个核都是按行读 LSE/O/dO、各写一份 per-row fp32,
   b1 上这一条就是 0.47% wall,且不动任何数值路径。它是"暴露辅助 2–4%"里最便宜的一刀。
3. ★★ b1 的暴露辅助(3.9%)想用 b4 那招藏起来就要把 body 按 q 轴切两发,而 §1 的抽干律直接给出报价:
   b1 的 body 是 2048 WG,切两发正好落到**在 b2 上亏了 2.6% 的那个 1024 WG/发**的几何。
   ⇒ 要走这条路必须先改的是"切完还剩几个 WG",也就是 `_A16_BLOCK_KV` 256→128 把 band 数翻倍(r21 下一步 #3),
   两件事得一起做才有意义,单独切 b1 的 body 按现有数据是负的。
4. ★ 折叠核内部若还要动,剩下的量是最宽 tile 的 **6.15–6.16 TB/s** 与流地板 6.2 之间那一点;
   再往上要换的是"读 q_split 条远流"的排布(每 slot 间隔 Skv 个元素),不是 tile 形状 —— 它折到 wall 上
   是 0.0X% 量级,排序上应该排在 #1–#3 之后。
