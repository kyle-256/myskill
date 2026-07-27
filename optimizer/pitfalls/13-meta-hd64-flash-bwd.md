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
- **主攻=hw exp-overlap**(藏 quarter-rate v_exp 进 MFMA,+5-6% 拉到 fast:dq gap 最大 −6~8% 因 exp 在 bulk 后没藏,dkdv 已被 exp-before-dP 藏好 −3~4.5%)→ 拿下 8192,逼近 4096/16384。★**融 odo 判负**:full vs sum(odo+dq+dkdv) gap 为负(无 launch 开销),odo 仅占 1-3%,融合无收益(先测省了大改)。

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
- **残余缺口(8192/16384 full-causal)根因 = dkdv 结构层**:hw 的 quarter-rate `v_exp` 在 bulk 之后未被 MFMA 藏(dq gap 最大,dkdv 已被 exp-before-dP 藏好);16384 叠 dkdv LDS 转置读延迟暴露。**确定性结构杠杆(P1-P5 / fp8-GEMM1 / q_split / occupancy-force / odo-fusion / dual-wave-8wave-warpspec 全系)已 measure-closed 判负**(见 DEAD 段)。剩两条均需上层裁决:①放宽位确定性(atomic-dQ 融合,但确定性=替代 CK 的全部意义,GOAL 禁);②research 级 exp-overlap(藏 v_exp 进 MFMA shadow 且守 SNR≥48,无 KB 先例)。
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
