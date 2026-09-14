# pitfalls/12 — DSV4 sparse-MLA attention (fwd/bwd) 实测经验 + 死胡同

> 类别: 踩过的坑 · 主题标签: dsv4, sparse-mla, attention, fwd, bwd, gfx950, fast-path, BLOCK_H, PV-K32, delta-fusion, register-transpose, K32-MFMA, occupancy, LDS-port, tr16, store-bound, query-blocking, DMA, s_setprio, XPIPE, flydsl-authoring, dead-ends

> 任务：优化 `sync/mxfp4/Primus-Turbo/primus_turbo/flydsl/attention/kernels/sparse_mla_v2/`
> (`dsa_fwd.py` / `dsa_bwd.py`)的 flydsl fwd/bwd kernel。
> 跑分 = **6 组均值**(flash H64 / pro H128 × cr∈{0,4,128})@seq=4096。
> bench 已收敛为单个 `bench_mla.py`(6 fwd + 6 bwd + triton_v2 SNR 门,`python bench_mla.py` 直接跑;
> 旧的 campaign_*/bench_triton/verify_turbo 已删)。commit 在 `dev/kyle/flydsl_attn_deepseekv4`。
>
> ⚠️ 本卡是**当前真相**的整理版。历史上很多结论被反复推翻(尤其"cr128 天花板"用错了硬件数),
> 已按「最后一次真实现 + bench + memory 佐证」拍板。**通用硬件事实(512 共享寄存器池 / 160KB LDS /
> ISA-vs-CSV 权威 / DMA-vs-register-prefetch)见 methodology/04·03 + pitfalls/03,本卡不重复推导。**

## cr 语义(决定 shape/瓶颈,务必先懂)
- **cr=0**:纯 SWA,无 pool,topk=128(kv=latent,num_kv=S)。kv=j 被连续 token 窗口 [j,j+W-1] attend → **banded 结构**。
- **cr=4**:P=S/4,随机 topk,topk=flash640/pro1152。pool-heavy,gather 不规则。**唯一 non-banded 多-tile 组**。
- **cr=128**:HCA,P=S/128=32,topk=160。pool entry 0 被 ~全部 S token 命中(CSR list ~4000,极不均衡)。

## 当前实测状态(2026-07-13,`bench_mla.py`,全 SNR PASS)
| shape | fwd TF | bwd TF |  | shape | fwd TF | bwd TF |
|---|--:|--:|---|---|--:|--:|
| flash cr0 | 282.8 | 256.5 |  | pro cr0 | 368.4 | 255.1 |
| flash cr4 | 671.5 | 340.0 |  | **pro cr4** | **940.2** | **479.0** |
| flash cr128 | 395.3 | 292.1 |  | pro cr128 | 400.3 | 294.4 |
| | | |  | **MEAN** | **509.7** | **319.5** |

- SNR:cr4 路径 ~53dB(fp32-eager 真值:fwd o 52.9/lse 146/dq 52.8dB,就是 bf16 精度,与其他 shape 一致);banded/cr0/cr128 55-78dB。⚠️旧记「cr4 99dB」是 stale,别信。

## ⚠️ cr=4 两个 NaN bug(已修,2026-07-14)——见 [[project_dsv4_cr4_nan_fix]]
- **Bug1(fwd fast_path)**:`_mb[0]=crossgrp_max(lm_f0)` 首 key-pair 全 masked(query t<96)→ -inf → exp2(+inf) → 前 96 token o=NaN/lse=+inf(lse +Inf 行=96×H)。**修**=floor `maxnumf(crossgrp_max,0)`(dsa_fwd.py 两处),speed-neutral。**生产 S=4096 只受此 bug。**
- **Bug2(bwd dq race)**:`build_bwd_dq`(topk<512 路径)在 TOPK=384(cr4 S=1024)有非确定 cross-wave race(NaN 数逐次变)。**修**=pvk32 门槛 512→384(dsa_bwd.py `_get_dq`+`dq_folds_delta` 两处耦合改),补 barrier 无效。**仅 S=1024 小 seq 受影响**(S=4096 topk≥512 本就走 pvk32)。
- **Bug3(bwd dkv misdispatch,小 seq)**:`is_cr128=(num_kv>total_tokens)and(topk<=256)` 裸判在 S≤512 把 cr4(随机 pool topk=256)误判成 cr128→走闭式因果 pool gather→dkv 全错(S=512 dkv 1.0dB)。**修**=加确定性-pool 条件 `total_tokens%npool==0 and total_tokens//npool>=64`(cr4 pool_cr=4<64 落 CSR;真 cr128≥64 仍闭式)。**仅 S≤512;S=4096 topk>256 本就 CSR。** 来自 wenx PR#417 第三修复(我最初 zip 只有前两个),commit 350ec3f5。
- 验证工具=`test_accuracy.py`(fp32 eager,自带 6 shape,import 指向生产包即可);远端跑必 `rm -rf /root/.flydsl/cache`。
- **dsv4 fwd/bwd 结构上 ~2.3× dense flash-attn** = sparse gather/scatter 税(interm+gather 过 2× 的 [T,topk,D] tensor,dense 无此步)。这是当前算法维度的特征,不是"到头"——见下「开口」。

## ✅ 已部署的赢(默认开)
**fwd**(`dsa_fwd.py`,`build_fwd`):
- **fast_path / _FMAX0**(pro cr4/flash cr4,commit 09ea462f):softmax 平移不变 → 用 first-pair 的 crossgrp-max 作固定界(多 1 次 QK,无 pre-pass)→ alpha≡1 编译器**折叠** per-tile `o*=alpha` rescale → nomax 速度 + 纯累加高精。pro cr4 828→**933-940/99dB**,flash cr4 647→668/99dB。banded 无 rescale 头不受益。详见 [[project_dsv4_fwd_dualwave_port]]。
- **BLOCK_H=128**(pro cr4,非 banded H≥128):1 WG/token(8 wave/512 线程)存一次共享 MLA latent,消 BLOCK_H=64 的 2-WG 冗余 KV-store(≈砍 23% 工作)。pro cr4 707→784/827。教训:MQA/MLA 共享 latent 下 store 冗余按 (H/BLOCK_H) 放大,遇 store-bound + 共享 KV 先查 WG/token 冗余度。
- **PV-K32**(many-tile/cr4):PV 从 K=16→2-tile-batch K=32(concat 2 tr16→v8 A、concat p→v8 B,combined softmax over 32 kv),QK/store 仍 per-16 不破交织。+`pv_early`(PVPF=2 个 V32 tr16 读提到 softmax 前藏延迟,必须=2,4/6 寄存器爆)。pro cr4 616→707。**仅 many-tile;few-tile(cr0/cr128)扩它判负**。
- **pstore**(flash cr4):4-buffer parity 预存流水,KV store 与 PV MFMA 重叠 + 3→1 barrier/pair,省 VGPR 后 pf_pv 2→7/pf_qk 3→5。flash cr4 632→667。

**bwd**(`dsa_bwd.py`):
- **delta 融合**(commit 8e83615d,全弱组):delta=rowsum(O·dO) inline 折进 build_bwd_fused/build_bwd_dq,消掉独立 delta 核(ceiling-probe 实测占每组 wall ~12-13% 全串行)。dO 已驻 do_packs → stream O(1 vec8)+ ds_bpermute 跨组和。isolated wall flash −4.6%/pro −2.6~3.3%。详见 [[project_dsv4_bwd_delta_fusion]]。
- **融合 dQ+interm**(build_bwd_fused,门控 num_heads≤64 且 topk≤256):kv-block-batching,Q/dO 只驻 1 个 128-宽 d-block + KB 个 kv-tile 批一次 staging(restage/barrier 4/tile→1/tile),消 chunk_dS/P 的 HBM。flash cr0/cr128 +5-6%。详见 [[project_dsv4_bwd_fusion_win]]。
- **register-transpose interm**(build_interm_rtr,门控 pro cr4):ds_bpermute 手搓 16×16 bf16 转置替 LDS+tr16,pro 上 BD256 + d-tile-owning tiling(转置一次摊全 rank-tile)。pro cr4 interm 1.55×,+11.8%。详见 [[project_dsv4_interm_register_transpose]]。
- **K=32 MFMA**(interm rtr+fused):头-contraction 用 mfma_16x16x32 砍半 mfma 数。bwd_mean +9.9%。★坑:v8 operand 必须转置**直接产 v8**(`from_elements(8 i16).bitcast(bf16)`),别 hold v4 再 shuffle(寄存器翻倍 occ 崩)。
- **KV LDS 双缓冲**(fused):KVBUF=2 去覆写前 barrier(17→13/迭代),flash cr0/cr128 +1.3~3%。
- **few-tile 弱组摊薄**(cr0/cr128):build_bwd_dq `_QK4→0`+`pf6→4`、interm_rtr `GSZ64→128+DBUF1`、fused `KB=NUM_TILES` 单批。

## 当前 bound(权威,已去矛盾)
- **occ 统一到 512 共享寄存器池模型**(见 methodology/04):occ = `512 // (arch_vgpr + agpr)`,VGPR/AGPR **同一 512 文件**;AGPR-accum 只在池内搬家、**不抬 occ**(除非本有 spill)。arch-VGPR 部分硬顶 256,但 AGPR 共享池。★ 本卡历史上「256 硬顶 / 64KB LDS」的推导**是错的**(据此建的"cr128 天花板"作废,见 §fwd 弱组)。ISA `num_vgpr/num_agpr`(FLYDSL_DUMP_IR)是唯一权威,CSV `Accum_VGPR_Count` 恒报 0 会误诊(pitfalls/09、methodology/03)。
- **occ 不是本族杠杆**:全 shape latency-bound(MfmaUtil 12-36%),occ-1/occ-2 都填不满;强制抬 occ(waves_per_eu)必 spill(灾难)。三重证:D-chunk 减半 o_acc 省不了 VGPR;WPEATTR 灾难;AGPR-o_acc/bf16-o_acc 均 occ 不变 + 掉速。
- **fwd = 均衡 latency-bound**(VALU exp2 ≈ MFMA ≈ 35%,per-tile QK→softmax→PV 串行链无第二 WG 掩盖);pro cr4 曾 LDS-port(QK v8-natural + PV tr16-transpose 同数据两布局)+ fast_path 后回均衡。弱组(cr0/cr128)VALU(exp2 transcendental 不可约)+ 均衡。
- **bwd dQ = occ-1 暴露的结构延迟**(subtractive residual **53-61%** = KV-gather HBM 延迟 + per-tile 调度 + softmax 串行链)。**★ 不是 tr16-read bound**(hoist_tr 仅省 2-5%);QK(37-44%)/PV-mfma(11-30%)是主计算但同样 occ-1 暴露;barrier 与 dS/P-store **有益/免费**(去掉更慢)。dq_acc(128)+q/do_packs(128)=256 VGPR 锁 occ-1。
- **bwd interm = MFMA-RAW-latency**(需双-acc 2-chain ILP 藏依赖链),occ-1;occ-2 可达但 per-rank-group gpu.barrier 锁步 wave-switch 失效 → 无收益。BD 必整除 D=512(BD=256 最优,BD=192 是漏算假快)。

## ❌ 已判负杠杆(measured,别再试;"负"= 当前实测负 + 需何条件才可能翻)
**fwd**:
- **s_setprio**:在**本 kernel**全 DEAD(fwd 是 VALU+read bound,优先 MFMA 反饿死 VALU)。⚠**这条不可跨 kernel 搬运**:同一对 `s_setprio(1/0)` 包 MFMA-dense 的 GEMM2 时,在 MFMA-pipe-serialized 的融合 hd64 flash bwd 上**拿掉它 −2.8%(9/9)** ⇒ 正负由 regime 决定(有无共驻仲裁对象 / 该 region 兄弟 wave 有没有 MFMA run / body 是否 MFMA-serialized),判据见 pitfalls/09 §调度提示。
- **cross-tile 软件流水(XPIPE)**:全 shape 死(cr4 LDS-port 饱和填 MFMA 无用;few-tile cr0 prologue 成本 > 重叠)。
- **DMA / buffer_load_to_lds**(所有形态:非流水/per-tile/4-buffer/深预取/banded):register-prefetch+store 对 scattered-gather 最优,DMA 的 s_waitcnt 粗同步暴露 HBM 延迟。**buffer_load_to_lds 已真实现**(非代理):原语全对/单 buffer 587<840,流水被 flydsl 编译器 bug 挡死(同 iter tr16 从 >2 个 LDS base 读就坏),见 [[project_dsv4_fwd_dma_experiment]]。→ 翻它需改 flydsl 编译器(用户禁)。
- **register-transpose PV(fwd RTR)**:实测 **−77%**(97.7TF/SNR PASS)。转置每 tile 重做不摊薄(bwd rtr 赢是转置摊在 rank-tile;fwd PV 每 tile V 不同)。
- **query-blocking(QB,cr128 减 store 冗余)**:实测净负(SNR PASS)。cr128 store 确是 bound(SKIPST +30%),但 QB 的 2×o_acc+Q > 256 arch-VGPR → 强制 occ-1(vs baseline occ-1.9)→ 占用腰斩损失 > store 省的 30%。→ 翻它需可控寄存器分配(本容器 agpr-alloc/waves_per_eu/maxnreg codegen flag 实测全不生效,external codegen 不可用,见 [[project_mxfp4_vgprform_deadend]])或换硬件。
- **triple-buffer / XOR-swizzle / D_LDS≠528 / lazy-O-rescale / post-misched / fast-math / ds_load_tr16_b128**:负或 HW 不可选(D_LDS=528 是 MLA tr16 唯一正确 pad,512 灾难 −66%)。

**bwd**:
- **K=32-PV**:净负(2-tile 批打断 dQ 的 per-tile QK→softmax→PV 交织,该交织是 occ-1 藏延迟唯一机制)。与 interm K=32 赢不同:interm 是纯头-contraction、operand 流式不 hold。
- **dual-layout-KV dQ**(两份 KV-LDS 消 QK bank 冲突):**−7~17%**(多写一份 KV 的 store 压在 occ-1 关键路径 > 消掉的 off-critical-path 冲突)。QK 13% bank 冲突不在关键路径。
- **occ-2-via-prefetch-drop**(interm 去 dS/P 寄存器预取拿 occ-2):−1.4%(丢 prefetch 延迟 > wave-switch)。
- **fused qb/dob 双缓冲**:负(restage barrier 有益地串行化 LDS 流量)。
- **atomic-scatter 消 interm HBM**:慢 10×(pool-kv 数千 token atomic 竞争)。
- **fp8/int8 interm**:SNR 31<35 + 用户硬禁量化。
- **fused+register-transpose**:occ-1(fused 已 256 VGPR,加转置寄存器爆)。
- **cr4 banded+pool-CSR split / side-stream CSR overlap / hold_qo=False dQ**:持平或灾难(hold_qo=False 小topk 0.45×,pro cr4 6× 回退)。
- **gather nw/unroll sweep**:中性(banded 51%BW 是短归约随机访问的实际带宽,nw 与 unroll 两轴都无效)。

## sparse-MLA **decode**(producer partial + combine;campaign 20260911_143318,2026-09-11)
形状 seq48/60/72/84/96,topk2048→ng=32,H=16,KV 行 576B fp8,producer 占 80-87%。记分尺 = geomean(prefill,decode) vs triton;本轮 decode 1.1427→**1.1618~1.1718**(4 次干净跑)、score 1.0909→**1.0995~1.1032**,prefill 未动(路径未碰)。

- ✅ **Q publish 加宽成全 CTA 扁平分块**:9KB 的 [16,576B] Q 块原本由 wave0 按「每 head 一行」发 9 条 16B/lane 的 load(跨 16 行 ⇒ 整块 144 条 line request),改成 576 个 16B chunk 摊到 256 线程(连续线程取连续 chunk ⇒ 每条 load 是一段连续 4KB,整块 72 条 request,即 128B 对齐 9216B 的下界)。全调用实测 seq48 持平 / 60 −0.61% / 72 −0.67% / 84 −1.23% / 96 −1.50%;ISA `global_load` 21→15、`ds_write` 57→51、VGPR 228→230(不跨 occ 台阶)、spill 0、输出逐位一致。**这是 pitfalls/03「反过来成立的是加宽」在本族的再确认**。
- ✅ **发射顺序:Q 的 load 先发、KV gather 后发、最后才落 LDS**。vmcnt 按发射序退休,所以 publish 的 store 只排干自己那两条 Q load,而不是连整个 primed KV tile 一起排干;同时 gather 仍只晚一个 4KB 突发就出门。比「gather 先发」在 seq60/72/84/96 再多 0.6–1.9%,并消掉后者在 seq48 的 +0.99% 反噬。
- ✅ **ownership 要按 XCD fan-out 选,不能只按 grid 饱和**。gfx950 的 WG 轮转 8 个 XCD:token-major(`owner=tok*n_groups+split`)把一个 token 的 CTA 撒到 `min(n_groups,8)` 个 L2 域,每个域都把该 token 整块 9KB Q 从片外拉一遍;split-major(`owner=split*seq+tok`)让 owner 以 seq 为步长,fan-out 降到 `8//gcd(seq,8)`。原来的饱和阈值把 seq48/60 留在了 token-major,切过去 producer 实测 **−3.4% / −5.2%**。常数-Q 消融(所有 CTA 都读 token0 的 Q)把片外 Q 总价定在 seq48 4.2% / seq60 3.5%,且常数-Q 下两种 ownership 只差 0.10% ⇒ 收益确实来自 Q 的 fan-out(seq60 的 5.2% 略超单 Q 的定价,说明那里还叠了 KV 侧的域局部性)。seq96 本就 split-major,常数-Q 在那只值 0.05–0.41%。
- ❌ **Q 常驻寄存器、删掉 LDS 往返与 barrier**:split-major 下 seq48/72/96 = **+3.48 / +1.52 / +2.25%**(4 个 wave 各拉一份 ⇒ 请求数 ×4)。即 pitfalls/03「小重排核去掉 LDS 改寄存器 gather」在本族的再锚定,别再试。
- ❌ **把 576 的尾块(576=2×256+64)改成 4B 全宽一遍以消掉预测掩码**(动机:让任何 wave 都不必在 publish barrier 前排干 KV):**+1.0 ~ +5.9%**,全形状更差。窄 4B load + 256 条 `ds_write_b32` + 运行时 div/mod 的代价远超省下的那一次 wave0 排空。
- **整个 Q stage 的剩余上界只有 0.42%(seq48)/ 2.46%(seq72)/ 0.90%(seq96)**(消融:不载 Q、不进 LDS、不 barrier、不读回)。prologue 这条线已榨到它自身定价之内,下一步该打 per-tile 的 barrier/rendezvous 链(独立消融定价上界 ~8%)或 KV 行的第 5 条 cache line(后者要先裁决)。
  - ⚠ **上面这句的「barrier」指错了对象,已被 R10 的加法探针改正**:往每个 tile 里**多塞 2 条 `gpu.barrier`**(每 CTA 8 条,barrier 数 +67%)实测 **−0.13 / +0.56 / +0.17%**(seq48/72/96)= 免费。⇒ 那 ~8% 不在 `s_barrier` 本身、也不在 wave 收敛上,而在 **rmax/rsum 的 LDS 往返数据依赖**(4 个 wave 各写 1 个 f32、barrier、再全读回)。要吃它只能**改所有权让每个 wave 独占整 head、消掉 cross-wave 归约**,别再去挪 barrier 位置或合并 barrier。

- ✅ **prologue 预取深度 2(`_XPF_PRIME=2`,steady-state 仍 `_XPF_DEPTH=1`)**:CTA 的第 1 个 tile 必须「index load → 回来 → KV load → 回来」两跳串行,而此时 grid 里没有任何别的工作能盖住它;在 Q publish 之前**多发一个 tile 的 gather**,把两条串行链折成一次更深的突发。producer 实测 seq48/60/72/84/96 = **−1.22 / −0.88 / −1.00 / +0.50 / −2.53%**(4 次独立 ABBA 复测,48/72/96 另三组为 −0.71/−2.24/−1.50、−1.29/−1.99/−0.37、−1.22/−1.00/−2.53),输出逐位一致;记分尺 3 跑 score 1.1017/1.1071/1.1036 → **1.1073/1.1040/1.1124**(三个次序统计量同向 +0.3%)。**prime=3/4 反噬 +8~12%**,而 ISA 显示 VGPR 218/224、spill 0 ⇒ 反噬来自调度(prologue 自己的依赖链被拉长),不是寄存器。
- ❌ **让 index load 比它寻址的 KV gather 早一个 tile(`_XPF_ROWS=1`,动机:消掉 tile 体内那跳指针追逐)**:**+1.74 / +4.68 / +7.90%**(seq48/72/96),且叠在 prime=2 上同样更差。索引行要多活一个 tile ⇒ 多 3 个 i32 常驻 + 打断 gather 的紧邻发射。**本族的指针追逐只能靠「多 primed 一个 tile」在 CTA 起点摊掉,不能靠 stage 错位。**

### decode 的 regime:R10 用两条加法探针(methodology/03)重新定性
- **计算**不是自由的:把 PV 的 16 条 `mfma_f32_16x16x32` **×4**(每 wave 每 tile +48 条,纯计算零额外访存)⇒ **+7.03 / +7.75 / +6.79%**。换算 **≈6 个墙钟 cycle / 条 mfma32**(约其管线时间的 1/3 透过率)⇒ 基线 MFMA(16 条 mfma32 + 5 条 mfma128)大致占墙钟 **~16%**。**decode 跟 prefill 完全不同族:prefill 的同类探针是 0%(纯 fabric 限),decode 这里「减指令」是有价的。**
- ISA 指令普查(ii=4 全展开,共 1284 条):`v_pk_mul_f32` **140** / `v_pk_add_f32` 72 / `v_mov_b32` 68 / `mfma32` 64 / `ds_read_b64_tr_b8` 64 / `s_nop` 55 / `ds_read_b128` 49 / `v_cndmask` 41 / `buffer_load_dwordx4` 36 / `ds_write_b128` 35 / `v_exp_f32` 23 / `mfma128` 20 / `s_barrier` 12 / `global_load_dword` 12。**最大单块 = 140 条 `v_pk_mul_f32`,即 online-softmax 每 tile 对 64 个 VGPR 的 PV 累加器无条件 rescale(32 条/tile/wave)**。
  - ❌ **不能靠「固定参考 max、只缩放 P、永不 rescale 累加器」消掉它**:P 要打包成 **fp8 喂 PV MFMA**(e4m3 上限 448),固定参考下 P 会溢出 ⇒ running-max 的 rescale 被 fp8 P 格式**结构性**锁死。剩下的路只有数据相关的 wave-uniform 跳过(`new_max==old_max` 时不缩放),值 ~1%,且随数据分布漂移。

### decode 的两条「非占用」结构量(R10 新定价,均与寄存器/占用无关)
- **每次 launch 有约 3us 的固定填充代价**:用不同 grid size 拟合墙钟得 `wall ≈ 2.95us + 26.96ns × N_CTA`(R² 高)。seq96 的 producer 里它占 12%、seq48 占 ~19%。来源就是上面那条「CTA 起点无人可盖的 index→KV 指针追逐」,`_XPF_PRIME=2` 吃掉的正是它的一部分。
- **per-CU 负载锯齿(与占用无关的 15%)**:CTA/CU 为**半整数**的 grid(seq 48/80/112 ⇒ 每 XCD 每 CU 1.5/2.5/3.5 个 CTA)比拟合线**高 1.3–2.0us**(seq48ate +15%),整数的(32/64/96/128)正落在线上。⇒ seq48 的 15% 不是占用、不是带宽,是**最后一轮只有一半 CU 有活干**。要吃它需要「每 CTA 可变 tile 数 / CTA 跨 token 边界」的重写,是本族目前**最大的已定价未动杠杆**。
- ❌ **占用(occupancy)在 decode producer 上再次证伪**:加一块死 LDS 把 `group_segment_fixed_size` 44800→82048(ISA 已核,常驻 CTA 数腰斩)在 seq96 只值 **0.1–0.2%**(冷/热、两种 ABBA 次序各测)。与 R5-H6(`waves_per_eu=3` 强压到 168 VGPR、带 8 次 spill = **−4.9%**)、R5-H5(Q 出寄存器 224→190 = −1.3%)三条独立证据一致:**本族「压 VGPR 换占用」这条路没有可买的收益,别再排预算**。踩坑提醒:那次 occ 对比的第一版数据(+15.44%)是在 `FLYDSL_DUMP_IR=1` + `REPS=1` 下取的,**IR dump 会污染计时**,复测三次全部回到 ±0.2%。
- **不可靠杠杆**:`inner_iter=8` 的交叉点在 seq48/60 一次测 −0.84/−1.66%、独立复测 +0.54/−0.22% ⇒ 不复现,别据此改 `_pick_inner_iter`。跨进程 spread 12.9%,配对 ABBA + min 才能看清 ≤1% 的差。

### decode 在**生产尺**上的复核(campaign 20260912_134949,GLM-5.2 TP4/EP4,2026-09-12)

换尺:KV 池 **1.53GiB**(< 2GiB ⇒ 走 32 位 buffer gather,不是上一轮量的 64 位回退),verify 形状只有
**seq48/60/84**(= 6×concurrency),draft 是 seq8/10/14。producer 与 combine 分开捕图 ABBA(spread ≤1.7%),
拆分:seq48 = 15.043 + 2.776;seq60 = 16.821 + 2.997;seq84 = 22.127 + 3.414;seq8 = 4.484 + 3.802。
**combine 占 verify 调用 13.4–15.6%、占 draft 调用 46%**,上一轮把它当零头是错的。
seq48 producer 的 roofline:MFMA 241TF/s(fp8 峰值 ~9%)、KV gather 4.18TB/s(HBM ~52%)⇒ 仍在访存+调度侧。

- ❌ **全程加深 gather 预取(`_XPF_DEPTH` 1→2→3)在 decode 上是纯亏**:seq48/60/84 = **+4.56/+6.01/+6.55%**
  (深度 3:+11.98/+12.58/+18.05%),seq8(ii=1)无影响,seq14 +3.48%;输出逐位一致(relL2=0.0)。
  ⚠ **与上面 `_XPF_PRIME=2` 的 ✅ 不矛盾但口径不同**:prime 只加深 prologue、稳态仍是 1;这次量的是
  `min(_XPF_DEPTH,inner_iter)` 起手 + 循环里 `k_i+_XPF_DEPTH` 的**全程深度**。
  ⇒ 结论:**全程深度 ≥2 别再试**;「只加深 prologue」得单独加一个 prime 常数才叫量过,当前 aiter 树里没有这个常数。
- ✅ **per-CU 锯齿在新尺上复现并重新定价**:ii=4 固定、seq 扫 16/32/48/64/80/96(CTA 128…768)得
  7.819/9.803/15.116/17.409/21.751/24.084us。过三个整数点(1.0/2.0/3.0 CTA/CU)的吞吐线是
  `2.83us + 6.81ns × tile`;半整数点全部高出:seq48 **+1.75us(+11.6%)**、seq80 +1.49us(+6.9%)。
  评分形状的锯齿税 = seq48 +11.6% / seq60 +5.3% / seq84 +4.5%。
- 🔧 **「固定成本 + 每 CTA」的拟合式自变量选错了**:同样 1536 个 tile,384 CTA(ii=4)与 768 CTA(ii=2)的
  producer **一样快**(15.043 vs 15.065),而 tile 数一变就线性变。本树上 `2.83us + 6.81ns × tile` 才对;
  截距 2.83us 与旧式的 2.95us 一致,**`26.96ns × N_CTA` 这一项不要再用**。
- ✅ **「256 CTA × T tile」定价线**(挑 `seq×(32//ii)==256`:(16,2)/(32,4)/(64,8)/(128,16)):
  5.789 / 9.803 / 17.703 / 51.546us ⇒ T∈{2,4,8} 拟合 `1.82us + 1.986us × T`(T=4 预测 9.76/实测 9.803)。
  **T=6 ⇒ 13.74us**,与吞吐线预测的 13.29us 互相印证 ⇒ seq48 那个「CTA 跨 token、每 CTA 6 个 tile」的
  均衡调度值 **−9~−12% producer**。T=16 脱线到 51.5us,别把 tile 数堆太高。
- 🔧 **`_pick_inner_iter` 的 docstring(旧尺)把 ii=4 的胜出归给 producer,归因错了**:新尺上 seq48 的
  producer ii=2 与 ii=4 打平(15.065 vs 15.043),**全部差距在 combine**(4.451 vs 2.776)。
  ii=8 在 seq60 上一次测到 full 19.502 vs 19.818(−1.6%),仍在噪声带边缘 —— 与上面「ii=8 不可靠」一致,要动得先复测三次。
- 🔧 **`mla_reduce_kernels.py::_COMBINE_HEADS_PER_CTA` 的原位注释已过期**。注释说 decode48 上
  heads/CTA = 1/4/8 → 2.90/2.69/2.66us、1 会退化;**新尺实测顺序翻转**:
  h1 2.244 / h4 **2.231** / h2 2.478 / h8 2.493(h8 是 {1,2,4,8} 里最差),h16 3.10。
  seq60 最优 h1(2.292 vs ship 2.541);seq84 最优 h2(2.583,基本持平);seq14/ni=16 最优 h1
  (2.610 vs ship 3.267,**−20%**)。原因就是 grid:ship 的 h8 在 seq48 只给 96 个 CTA(0.375 CTA/CU)。
  ⇒ **这个常数应该按 `seq*16//hpc >= 0.75*num_cu` 选,不该是常数**。
- **combine 自身的定价**:ni 扫 4/8/16/32 得 2.004/2.776/4.451/6.278us ⇒ `≈1.23us + 0.772us/(3.1MB)`,
  即 **1.2–1.5us 是纯固定成本**(与 producer 的 2.83us 截距同族)。融进 producer 的上界就是这块加上
  6.3MB 的 partial 往返。

**踩坑(工具侧,非本族专属)**
- 🐞 **同进程里改模块常数 + `lru_cache.cache_clear()` 重编译会把已捕获的 HIP graph 打爆**:`cache_clear()`
  丢掉最后一个 launcher 引用 ⇒ HIP module 被 GC 卸载 ⇒ 之前 `torch.cuda.CUDAGraph` 捕的 replay 直接
  `Memory access fault by GPU node-N`。做「一个进程内比多个编译期常数」的 ABBA 时,**必须把每次编译出来的
  launcher 对象 pin 在一个全局 list 里**(包一层 wrapper 记录返回值,并把 `cache_clear` 转发过去);
  还要注意 `from ... import compile_xxx` 的模块也得一起打补丁。
- 🐞 **`rocprofv3 --kernel-trace` 在这个 sglang 容器里会挂死并把 KFD 卡住**:13 分钟无输出、python 100% CPU,
  杀掉后 `rocminfo` 进 D 态,aiter 的 `_detect_native()` 60s 超时报「KFD is most likely wedged」。
  绕过用 `GPU_ARCHS=gfx950`,根治要清掉残留进程(`kill -TERM`,先核 `/proc/<pid>/environ`)。
  **本轮全部数据改用进程内 HIP-graph + ABBA + min 拿到,比 rocprofv3 又快又稳。**

## sparse-MLA **prefill**(campaign 20260911_143318,2026-09-12;定价谱见 pitfalls/03 同名条)
形状 T=4096/8192,topk2048,H=16,KV 行 576B fp8,一 CTA 一 token / 4 wave / `_BLOCK_N=64`。
基线 prefill4096 = 741.9us、prefill_speedup 1.046、relL2 2.5e-05、ISA VGPR 166 / AGPR 0 / spill 0 / LDS 45568 / occ 3。

**本核是 TCC(L2) miss 服务路径限,且近似线性于「每行触碰的唯一 128B 行数」**(定价谱:全 L2 驻留 −52.9%、
全 MALL 驻留只 −10.7%、唯一行 5→3 −27.3%、PV MFMA +76% **+0.1%**)。
⇒ **在 prefill 上排任何「减指令/减请求/换占用/减 VALU/减 LDS」的预算都是白花**,已用等指令流 alias 臂逐项证死:

- ❌ **`_BLOCK_N=64 → 128`**(动机:每 block 摊薄 barrier/rendezvous):**+1.8%**(755.3 vs 741.9)。
  VGPR 166→212、LDS 45568→79360 ⇒ occ 3→2;relL2 也从 2.6e-05 涨到 5.8e-03(仍在 0.02 门内)。
- ❌ **行对齐 gather,把 TCP 请求从 7/行压到 5/行**:**+0.9%**(748.6 vs 741.9)。算术**逐位相同**
  (relL2 不变 2.6e-05)、VGPR 166→162、spill 0、occ 不变。唯一行数没变 ⇒ 不赚。这是「请求数≠货币」的
  最干净反例,见 pitfalls/03。
- ❌ **靠 index 排序造跨 CTA L2 co-sweep**:主机端免费预排序这个**上界**只有 **−1.5%**,kernel 内排序不可能回本。
- ❌ **out 写端的合并/非临时化**:把 out 唯一行脚印从 67MB 压到 8.4MB 的 alias 臂只有 −0.9%
  ⇒ epilogue 那 32B 粒度的分散 store(每指令 16 片 32B)整族最多值 0.9%,别做 CShuffle 转置 epilogue。
- ✅ **跨 block KV 寄存器预取(XPF)已在位**(R5):`init=`/`yield` 抬 36 个 KV 寄存器过 key 循环 + 128B 宽 gather,
  748.2→741.9us。深度已是 1,再加深在 decode 上实测负。

**唯一 >2% 的已定价杠杆 = 每行唯一行数 5 → 4.5**,即把 576B 行拆成 512B lora 池 + 64B rope 池
(576 不整除 128,偶行相位 0 / 奇行相位 64 **都**横跨 5 条行,邻行同时命中概率 0.2%)⇒ 约 **−9~−10% 墙钟**。
属改布局,**必须先裁决**。次一条 = 跨 token 的 gather 批量化(上界即 L2 驻留臂 −52.9%),同样属改算法。

⚠ **记分尺分布注意**:该尺子的 prefill index 是 `torch.randint(0, 1<<20)`,均匀铺满 576MB 池;
真实 prefill 里 token t 的 topk 来自本序列因果前缀(seq=4096 ⇒ 脚印仅 2.36MB,正落在 L2 驻留臂那侧 349us)。
在这把尺子上 prefill 被钉在 fabric 上、kernel 能控的那半只占 47%;换成因果窗口后今天量到 0% 的计算侧杠杆会全部复活。

## 每内核 flydsl-vs-triton(决定往哪打)
- **dQ = 招牌,处处赢 triton 1.4–2.4×**(flash cr0/4/128=1.42/2.38/1.42×,pro=1.50/1.90/1.50×)。**已最健康,别动**。
- **interm = 大 topk 赢/小 topk 输**:pro cr4 rtr 1.55×;但 **cr0/cr128 输 triton 0.76–0.84×**。
- **gather = 两边 ~2-3TF**(memory-bound reduction,flops 低是本质带宽限,非效率问题)。

## 开口(open,不是"不可达")
1. **pro cr0/cr128 的 interm 慢 triton ~20%** = 当前最明确开口。triton 小 topk interm 更快(299-302 vs 209-254TF)→ 研究其 tiling/少-launch 打法,移植或专调 blk 小 topk。
2. **banded-SWA dKV 融合**(cr0/cr128):kv 被连续 token 窗口 attend → 融成 banded kernel 免 gather/免 interm HBM(gather 已用 banded closed-form,interm 未融)。
3. **hybrid scatter**(低争用 local 走 atomic / 高争用 pool 走物化)= 减 interm+gather 的 2× tensor HBM 的唯一未验证路。
4. **buffer_load_to_lds 流水**:被 flydsl 编译器 bug 挡(需上游修),原语与账都对。

## flydsl 本族专属坑(配合 pitfalls/07 + [[reference_flydsl_loop_pitfalls]])
- **2 字节 buffer_store 不 lower**(单 f32/单 i16 → `ScalarizeVectorOperand`):store 必须向量(≥2 元素/≥4B);f32→bf16 用 `Vec.from_elements([BFloat16(x)..],bf16).bitcast(Int32)`。`cvt_pk_bf16_f32` 打包 store 出垃圾,改 from_elements。
- **bf16 vec1 load 出 NaN**:gather strided operand 用 **i16 vec1 load 装配**。
- **contract-over-head GEMM**(interm):两操作数都 h-strided → 都走 LDS 自然 stage + `ds_read_tr16_b64` 转置读(**tr16 可同时作 A 和 B operand**,实测)。
- **iter_args 累加 loop**:单元素 yield 拆包(≥2 iter_args)、buffer_load raw 作 init 致 trip×2(用 Vec.filled)、字面 if/for 被转 scf(分支放外层 python)。展开循环里 `if t+1<NS` 首个 next 未绑定 → 先兜底赋值。
- **B023 闭包**:flydsl 闭包在同迭代内立即调用是对的,ruff B023 用「默认参数绑定 loop 变量」消除(语义/ISA 不变),别加 noqa。

来源:会话 a17bf3b2 全程实测(2026-07-09 ~ 07-13)。memory:[[project_dsv4_fwd_dualwave_port]] / [[project_dsv4_bwd_delta_fusion]] / [[project_dsv4_bwd_fusion_win]] / [[project_dsv4_interm_register_transpose]] / [[project_dsv4_bwd_dq_occ_truth]] / [[project_dsv4_fwd_dma_experiment]] / [[project_dsv4_turbo_verify_ref]] / [[reference_flydsl_loop_pitfalls]]。通用方法论:methodology/03(subtractive 探针「上界≠可达」铁律)、methodology/04(512 共享池 occ)、pitfalls/03(LDS/DMA)。

---

## 补丁(GLM-5.2 decode 记分线, 轮 2, 2026-09-12):锯齿归因被固定工作量对照证伪

本卡「per-CU 锯齿:半整数 CTA/CU 的网格高出拟合线,seq48 约 +15%」这条**结论层面对、
机制归因错**。原证据是**扫 seq** 得到的吞吐线 —— 改 seq 同时改了 CTA 数**和总工作量**,
这条线不能用来给网格量化归因。

★ **固定工作量对照**(同一个 seq、只扫 `inner_iter`,`tile 总数 = seq×32` 不变,
同进程 ABBA/min-of-4-palindrome/60s ramp,producer 展幅 0.11~0.92%):

| seq | CTA/CU (ii=8 / 4 / 2) | producer us |
|---|---|---|
| 48 | 0.75 / **1.50** / 3.00 | 15.780 / **15.137** / 15.078 |
| 60 | 0.94 / **1.875** / 3.75 | 17.130 / **16.743** / 18.003 |
| 84 | 1.31 / **2.625** / 5.25 | 29.421 / **22.237** / 24.735 |

seq48 从「半整数 1.5、半台机器空转」走到「3.0、超过 2 CTA/CU 硬上限、网格填满」,
**只快 0.4%**,而 combine 涨 0.80us(合计反而变慢)。三种网格总差 4.4%,**最优点在半整数**。
⇒ 「把网格摊平能拿 9~12%」是错的;那 1.8us 超出是 seq48 自身性质,成因仍开着。
⇒ 「为收尾巴重写调度骨架」这类候选,**必须先做固定工作量对照**,不能只看扫 seq 的线。

★ **占用率上限的第四个独立证据**:ISA 侧 VGPR **228** / AGPR 0 / spill 0 / SGPR 32 /
LDS 44800 B。gfx950 是 512 条 VGPR+AGPR 合一池 ⇒ `512/228 = 2 wave/SIMD` ⇒ **2 CTA/CU**;
LDS 侧是 `160K/44800 = 3.6`,**不是**约束项。所以本族「加 LDS 不掉占用率」有余量,
「省寄存器」到 168 以下才换得到第 3 个 wave。

★ **wave cycle 三分**(`SQ_WAVE_CYCLES` 占比):seq48/60/84 = ACTIVE_INST 20.2/17.5/18.4%、
WAIT_INST 25.3/28.0/28.0%、**WAIT_ANY 54.5/54.5/53.6%**、其中 WAIT_INST_LDS 只有 4.1~4.7%。
seq8(ii=1)是 15.6/14.4/**70.0**/2.33。⇒ 池子在 vmcnt+barrier 等待,不在发射、不在 LDS。

★ **MFMA 不是墙,顺手校准两个拍数**:`SQ_INSTS_MFMA` 恰好 21/wave/tile
(1 rope mfma128 + 4 key mfma128 + 16 PV mfma32)。`MFMA_BUSY/INSTS_MFMA = 19.81`
与 `(5×32+16×16)/21 = 19.81` **精确相等** ⇒ mfma128=32cyc、mfma32=16cyc。
MFMA busy 1664 / wave 寿命 26313 = **6.3%**(2 wave/SIMD ⇒ SIMD 占用 ~12.6%)。

★ **bank conflict 已在地板上,改 stride 动不了**:`BANK_CONFLICT/LDS_IDX_ACTIVE` 在
seq48/60/84 上恒 **22.9%**(seq8 30.1%),`LDS_IDX_ACTIVE` 恒 **3600 cyc/CTA**(900/wave)。
手算宽访问地板 ≈908/wave ⇒ 实测就是地板。且 16B 对齐使 dword stride 必为 4 的倍数、
`gcd(stride,32) ≥ 4` 恒成立(PITCH 528/544/560/576/592/624 = 4/8/4/16/4/4 路)⇒
只有 **lane→row/col 相位旋转(XOR swizzle)+ 写端配套**能动,值 ≤0.3% wall。

★ **加法探针给两笔微观开销定价**(做成编译期参数 + 写进 kernel 名,才能同进程 ABBA;
跨进程展幅 12.9% 不能用):
- **多一次跨 wave rsum rendezvous / tile**(写 scratch+barrier+4-way 读回+`×0.25`,
  `4x·0.25` 二进制精确):seq48 **+0.207us(+1.37%)**、seq60 **+0.169(+1.01%)**、
  seq84 **+0.887(+4.00%)**。⇒ 本卡「那 8% 的池子是 rmax/rsum 的 LDS 往返数据依赖」
  第一次被定量:**其中 rsum 这一半值 1~4%**。且 rsum **不是结构性的**
  (`tile_denom` 只喂 `next_denom`;`tile_denom==0` 哨兵等价于 wave-uniform 的
  `tile_max==-inf`),可以整段提到尾声;**rmax 是结构性的**(QK 按 key 切 wave、
  PV 按 DV 切 wave,`plds` 是两者之间的转置,fp8 打包 scale 四 wave 共用)。
- **多一套 8 条 `ds_write_b128` / tile / wave**(写未被读的 scratch,不改数值):
  seq48 **+0.734us(+4.89%)**、seq84 **+0.770(+3.47%)**,展幅 0.16~0.73%。
  对齐 PMC(LDS 管道只用 13.7%)⇒ 这笔钱是 **`ds_write`→`lgkmcnt`→读回 key→`mfma`
  的自身依赖链**,不是 LDS 带宽。⇒ `vlds` 双缓冲(33792→67584B,总 78.6K,
  `160/78.6 = 2.03` 仍 2 CTA/CU)是零算法风险的对策。

★ **直写 LDS 在 gfx950 上的真实约束**:`global_load_lds_dwordx4` 的 LDS 目标地址是
`M0 + inst_offset + lane*16` **按 lane 连续**,给不出带 padding 的 `slot*PITCH` 布局 ⇒
要么紧排(dword stride 128 ⇒ `gcd(128,32)=32`,32 路冲突),要么在**全局地址侧**做
16B 粒度 XOR swizzle 补偿(可行,每 lane 源地址自由)。另外 flydsl **没有**暴露这条指令:
`expr/rocdl/tdm_ops.py` 的 Tensor-Data-Mover(`make_tensor_gather_descriptor` /
`tensor_load_gather`)是 **gfx1250 专属**,gfx950 上不可用;要走
`expr/rocdl/inline_asm.py` 的 inline-asm 逃生口。

★ **`rocprofv3` 的口径要收窄**:`--pmc` 配 `SQ_*`/`GRBM_*` **完全可用**(每次约 13s,
连跑十余次无异常)。会卡 KFD 的是 (a) `--kernel-trace`(已记);(b) **新发现的
`TCC_*`/`TCP_*` 计数器组** —— 一次 rc=1(`rocminfo did not return within 60s.
The KFD is most likely wedged`),重试撞 300s `timeout`(rc=124),单次 shell 调用 367s。
⇒ 本容器里 **L2 命中率只能算,不能量**。另:`SQ_LEVEL_WAVES` 与 `SQ_ACCUM_PREV_HIRES`
在这个 build 上恒读 0 ⇒ **PMC 无法把 vmcnt 等待和 barrier 等待分开**,只能用加法探针。

★ **`_COMBINE_HEADS_PER_CTA` 的原位注释(h8 最好)第二次被证伪**,本轮数字更干净
(ni=8,非 fine 路径,展幅 0.31~1.30%):seq48 h1 **2.2242**(768 CTA)/ h4 2.2362 /
h8(出厂)2.4832 / h16 3.0970;seq84 h1 **2.5297** / h4 2.6280 / h8 2.6129 / h16 3.2281。
⇒ 出厂值比最优慢 **10.4%**(seq48)/ 3.2%(seq84)。

★ **本卡的工具坑「`cache_clear()` + 已捕获 HIP graph 必须钉住 launcher」是对的**,
但更省事的写法是:**直接拿 launcher 对象、用 `_run_compiled` 自己发**(常数写进 kernel 名)。
让闭包在**调用时**去查模块全局常数会静默串味 —— 所有 arm 一起用最后一个值,
而图捕获发生在全部 arm 设置完之后,错得不报警。

★ **XCD/L2 局部性(新开口,当前记分尺看不见)**:一个 token 的 2048 个 topk 行 =
2048×576B = **1.18MB**,装得进一个 XCD 的 4MB L2。生产上 verify 行数 = 6×concurrency,
同请求相邻 6 个位置的 topk 高度重叠 ⇒ 唯一数据只有约 1.18MB。但
`owner = split*seq + tok` 使 `tok=6r..6r+5` 落在 **6 个不同 XCD**,各从 HBM 取一遍。
改成 `r = owner % XCDS; j = owner // XCDS; tok = r*G + j%G; split = j//G` 是**纯置换、
输出逐位不变**,grid/CTA-per-CU 不变。现有 `split_major`/`_split_major_folds_q` 已在用
同一判据,但折叠对象是 **Q 的 9KB**,不是 KV 的 1.18MB。
⚠ 记分尺 `idx = torch.randint(0, rows, (seq, TOPK))` **每 token 独立均匀随机**,
把这件事完全抹掉 ⇒ 该杠杆在尺子上收益恒 0,且尺子对 gather 成本可能偏悲观。
这与本卡 prefill 段记的「记分尺分布注意」是**同一类缺陷的 decode 版本**。

★ **带宽参照系**:用 HBM 峰值当分母会高估余量(gather 算成 52%);换 pitfalls/02 实测的
stream 地板 5.95~6.2TB/s 后是 **67~70%**。访存类结论一律用实测地板做分母。

---

## ★★★★ 补丁(GLM-5.2 decode 记分线, 轮 3, 2026-09-12):XCD 亲和重映射 = 生产 −21~34% producer,记分尺 0

上一段记的「XCD/L2 局部性(新开口,当前记分尺看不见)」本轮**两头都被实测证实**,
并且已经实现、逐位对拍、在计分尺上整跑过。这是本族目前最大的已兑现杠杆。

★ **生产真实重叠率(live server 正面观测,只读 PYTHONPATH 覆盖层包
`flydsl_sparse_mla_decode` 取 `indices` 做 `torch.unique`,没改 aiter 树一行)**:
GLM-5.2-MXFP4 tp4/ep4、EAGLE 6 draft token、`--max-running-requests 8`、
**`--disable-cuda-graph`**(否则 python 侧看到的是捕图时的假索引)、客户端每 worker
一份独立 142k prompt。每请求 6 行 × 2048 = 12288 次行引用里的唯一行数:

| decode 深度 | 唯一行/请求 | 复用 | 相邻 token 重叠 | 距离 5 |
|---|---|---|---|---|
| 刚进 decode | 4094-5175 | 2.64-2.72 | 0.69-0.71 | 0.50-0.54 |
| 中段 | 3331-3592 | 3.54-3.60 | 0.75-0.77 | 0.62-0.64 |
| **稳态** | **2715-3187** | **4.05-4.29** | 0.80-0.92 | 0.68-0.74 |

**跨请求重叠 = 精确 0.000**(距离 6/7/8/12/16/24 全部),请求内 0.5-0.92
⇒「连续 6 行 = 一个请求」的布局是正面证实的,不是推测。
每请求脚印 **1.56-2.98 MB < 4 MB(一个 XCD 的 L2)**;每次调用唯一行 11.5-18.7 MB,
而记分尺的均匀随机是 **56.6 MB**(5 倍)。
★ 生产还会启动 `seq=42/36/30/24/18`(**ii=2, n_groups=16**)与 `seq=1..8`(ii=1) ——
记分尺的 48/60/84 只是满批子集(methodology/14 §部署形状集 ⊋ 计分形状集)。

★ **索引分布定价谱(seq48/ii4/384CTA,producer only,同进程 ABBA/min,展幅 ≤3.5%)**:

| 分布 | 唯一行 | 脚印 | us | vs rand |
|---|---|---|---|---|
| `rand`(记分尺) | 98304 | 56.6 MB | 15.092 | — |
| 56.6 MB 连续跨度内均匀 | 98304 | 56.6 MB | 14.572 | −3.4% |
| 9.4 MB 跨度 | 16384 | 9.4 MB | 12.350 | −18.2% |
| 4.1 MB 跨度 | 7168 | 4.1 MB | 10.729 | −28.9% |
| 1.18 MB 跨度(全 L2 驻留) | 2048 | 1.18 MB | **10.359** | **−31.4%** |
| **生产结构(6 token 共享)** | 16384 | 9.4 MB | **15.441** | **+2.3%(更慢!)** |

⇒ ★★**同样 9.4 MB 脚印,无结构的 12.35 vs 生产结构的 15.44,差 25%**:生产那 4 倍复用
在现役 ownership 下**一分钱没兑现**,还因为同一行被 6 个 XCD 各拉一遍而比随机更慢。
**结论修正**:上一段写的「尺子对 gather 可能偏悲观」是错的 —— 尺子把这条杠杆整条抹掉,
同时给出偏**乐观** 2% 的绝对值。

★ **重映射(纯置换,输出逐位相同)**:`split_major` 是 `owner = split*seq + tok`,
tok 变最快 ⇒ XCD `x` 拿 token `{x, x+8, …}` = 6 个**不同请求**各一个 token ⇒
每 XCD 工作集 12288 个互不相同的行 = 7.08 MB > 4 MB ⇒ 命中 ≈ 0%。改成
```
per_xcd = seq*n_groups/8;  flat = (owner%8)*per_xcd + owner/8
rq = flat/(G*n_groups);    rem = flat%(G*n_groups)
tok = rq*G + rem%G;        split = rem/G          # G = 6
```
seq48 时恰好一个 XCD = 一个请求(8 请求 ↔ 8 XCD);seq60/84 时一个请求最多跨 2 个 XCD。
`seq%G != 0`(draft)或 `seq*n_groups%8 != 0` 时退回现役 ownership。

| 形状(配置元组) | 分布 | ship | **xg6** | Δ | 逐位 |
|---|---|---|---|---|---|
| 48 (48,32,4,8,T,T) | 生产结构(复用6) | 15.369 | **10.198** | **−33.6%** | ✅ |
| 48 | 复用 4.0(稳态) | 15.332 | **10.560** | **−31.1%** | ✅ |
| 48 | 复用 2.67(早期) | 15.395 | **11.158** | **−27.5%** | ✅ |
| 60 (60,32,4,8,T,T) | 复用 4.0 | 17.018 | **11.398** | **−33.0%** | ✅ |
| 84 (84,32,4,8,T,T) | 复用 4.0 | 22.729 | **18.021** | **−20.7%** | ✅ |
| **42 (42,32,2,16,T,T)** | 复用 4.0 | 13.884 | **9.858** | **−29.0%** | ✅ |
| 48 | `rand`(记分尺) | 15.125 | 15.108 | −0.1% | ✅ |
| 60 | `rand` | 16.608 | 16.644 | +0.2% | ✅ |
| 84 | `rand` | 22.222 | 22.784 | **+2.5%** | ✅ |

计分尺整跑:score 1.00454 → **1.00314(−0.14%,噪声内)**、relL2 一位不变。
⇒ **生产 e2e −1.3~1.8%,记分面看不见**。c14 的 +2.5% 是重映射里两次运行期整数除法在
零复用回归下的净成本,把 `per_xcd` 由 host 传参即可消掉。

★ **L2 命中率只能算不能量(本容器 TCC/TCP 会卡 KFD),但有更硬的替代取证**:
xg6 在生产分布下落到 **10.198 us**,而同核全 L2 驻留臂是 **10.359 us**
⇒ 重映射把这条轴上能拿的局部性**基本全拿到了**(贴上该核的 L2 驻留地板)。
预测命中率:现役 ≈0%;xg6 稳态 = 1 − 3000/12288 = **75.6%(74-78%)**,早期 58-67%;
每次调用 DRAM 读 56.6 MB → 13.8-18.3 MB。
⇒ **重映射之后 gather 这条轴已兑现,下一批杠杆换成 per-tile 依赖链与固定成本**
(rsum 归约、`vlds` 双缓冲、combine 融合),而且它们在生产上的相对权重更大
(combine 占 verify 调用从 14% 升到 19%)。

★ **`pitfalls/03` 判负的「靠 index 排序造跨 CTA co-sweep(上界 −1.5%)」不适用于本条**,
但它的**判负理由**可以复用:那里是并发窗口不够(96 CTA/XCD 各流 1.31 MB)。这里
一个请求的 48 个 CTA 全部共驻同一个 XCD(64 个槽)、联合脚印 1.6-1.8 MB < 4 MB
⇒ 窗口成立。**判据:先算「共驻窗口的联合脚印 vs 4 MB」,再决定要不要做 co-sweep 类候选;
而且置换 CTA→tile 是免费的,跟 kernel 内排序不是一回事。**

🐞 **新坑(flydsl 前端):live server 在跑时对本地树做编辑 + 任何会 rsync 的远端调用
= server 必崩**,症状 `flydsl/compiler/ast_rewriter.py AssertionError: unexpected ast
node <ast.ClassDef>`。机制:文件被换掉后 `inspect.getsource` 拿**已加载 code 对象的
旧 `co_firstlineno`** 去读**新文件**,向上找 `@` 时越过 `@flyc.kernel(` 命中上面那个
`@fx.struct class PartialStorage` ⇒ 取到 ClassDef。代价 = 一次 server 启动(~4 min)。
**规矩:server 在跑时,本地树冻结。**

---

## ★★★ combine 的 CTA→行编号要跟生产者的 split-major **同序**;先改编号,再谈 `_COMBINE_HEADS_PER_CTA`

(2026-09-12 GLM-5.2 TP4/EP4 decode,记分尺 `_bench_campaign_attn.py`)

`mla_decode_combine` 原来用 1-D grid + `block % blocks_per_row` / `block // blocks_per_row`
解出 `(row, b)`,即**扁平 WG id = `row * blocks_per_row + b`**(row-major)。生产者的
ownership 是 **split-major `split * seq + tok`**。gfx950 把 WG `i` 派到 XCD `i % 8`
⇒ 生产者把 row `t` 的全部 partial 写进 XCD `t % 8` 的 L2,而 combine 的
`row * blocks_per_row + b` 把同一行的读**摊到 `min(blocks_per_row, 8)` 个域**去。

改法是**纯置换、零指令代价**:换成 2-D grid `grid=(seq, blocks_per_row, 1)`,
`row = block_idx.x`、`block_in_row = block_idx.y` ⇒ 扁平 id = `block_in_row * seq + row`,
**和生产者的 owner 公式逐位同形**。

| 核 | 改前 | 改后 | Δ |
|---|---|---|---|
| `mla_d8`(seq 8,ng 32) | 8.2545 | **6.7213** | **−18.6%** |
| `mla_d14`(seq 14,ng 16) | 10.6077 | **9.8624** | **−7.0%** |
| `mla_v8/v10/v14`(seq 48/60/84) | — | — | ≈ −0.5% |

**为什么 draft 赚、verify 不赚**:seq=8 时生产者的 KV 脚印 ≈10.5 MB,冲不掉 partial;
seq=48 时它在记分尺的均匀随机索引下流 56.6 MB,partial 早被挤出 L2 ⇒ 同序也没东西可命中。
⇒ 这条杠杆的钱同样来自**数据的结构**(和本卡上面的 gather 重映射同源),
生产分布下 verify 侧也应当拿到,记分尺低估它。

★ **它会翻转 `_COMBINE_HEADS_PER_CTA` 的结论**。本卡上面记的
「h1 单测快 8-10%、在位却 +0.9%/+1.7%」是**旧编号下的结论**:h1 把 `blocks_per_row`
从 2 抬到 16,在 row-major 编号里等于把一行的读从 2 个域摊到 8 个域。换成 row-minor
之后 block 数不再左右落域,单测的结论就在位复现了。记分尺两臂各 2 次独立跑:

| 臂 | run1 | run2 | run3 | 均值 | spread |
|---|---|---|---|---|---|
| h8 | 1.57269 | 1.57562 | — | 1.57416 | 0.19% |
| **h1** | 1.58131 | 1.57946 | 1.58358 | **1.58145** | 0.26% |

(区间不重叠,h1 最小值 1.57946 > h8 最大值 1.57562。)
⇒ **规矩:凡是"把 grid 拆得更细"的候选,先确认扁平 WG id 的低 3 位没被顺带改掉;
否则你量到的是落域变化,不是并行度变化。**

## ❌ 别再试:用 ticket 原子把 combine 融进生产者 epilogue(记分尺 −2.5% ~ −62%)

动机是对的:同批量到**空 triton 核在同一张 HIP graph 里 16/192/384/768 CTA 全是
1.93-2.00 us**(平),一步 324 次 launch ⇒ dispatch 约占 `step_c8` 的 31%;
而独立 combine 在位只值 2.24-2.62 us。做法:生产者 epilogue 里
`s_waitcnt(0)` → CTA barrier → tid0 `atomic_add_agent(ticket[tok], +1)` →
barrier → 谁拿到 `n_groups-1` 谁归约该 token(并 `-n_groups` 自复位,graph replay 安全)。
**正确性没问题**:verify 三个形状与分离版**逐位相同**,draft 只有 5.5e-6/3.9e-5 的
bf16 配对顺序差。但两臂都判负:

| 臂 | score | `mla_v8` | `mla_d8` |
|---|---|---|---|
| 分离 combine(部署) | **1.58145** | 17.67-17.79 | 6.72-6.74 |
| 融合 + agent-scope release/acquire(全形状) | **0.60625** | 52.40 | 24.41 |
| 融合 + 无 fence(只在 `split_major && seq%8==0` 时融) | **1.54194** | 19.03 | 16.40 |

两条独立的教训:

1. **别在 gather 核里放 agent-scope fence**。`fence_agent_release/acquire` 要
   writeback-invalidate per-XCD L2,而这个核正靠那 4 MB L2 吃 topk 复用 ⇒ 每个 CTA
   刷一次 = **−62%**。这和 `pitfalls/03` 的「CPol `sc1`(device scope,绕 per-XCD L2)
   = −11.2%」是**同一族**,只是更狠(那里是绕过,这里是主动作废)。
   ⇒ **看到「跨 CTA 发布/订阅」的候选,先问它要的 scope;凡是超出 workgroup 的,
   都要先按 03 那张表定价,不要写完再测。**
2. **「launch floor」不等于「合核能回收的钱」**。空核 1.94 us 是 graph node 的
   **duration**,不是它在依赖链上加的**增量**;`methodology/18` 的定价法
   (「重复启动这个核,看总时间加多少」)才是货币。即便在无 fence、L2 同域的
   最有利形状上,融合仍让 `mla_v8` 从 17.79 涨到 19.03(+7%)、`mla_d8` 从 6.72
   涨到 16.40(+144%):epilogue 的活寄存器压在**整个生产者**头上,而且归约 CTA 要等
   全部 `n_groups` 个兄弟到齐 —— 这段尾巴原来是被下一个核的启动**重叠**掉的。
   ⇒ **合核之前先用 18 的重复启动法给"被合掉的那个核"定价,再减去"尾巴不再重叠"的损失。**
   判负成本 = 2 次记分尺(160 s)+ 一次远端逐位对拍(14 s)。

## 定价:decode 一步的成分与 absorb 的 tuned-config 余量

(同批,`step_c8 = 2038.7` 的分解)`mla_v8*79 = 1406.6`(69.0%)、`mla_d8*5 = 41.3`(2.0%)、
`uk_8*78 = 247.7`(12.1%)、`uv_8*78 = 343.2`(16.8%)。生产者/combine 拆分(同进程
HIP-graph ABBA,spread ≤1.8%):seq48 full 17.79 = prod 15.13 + comb 2.24 + gap 0.43;
seq84 25.67 = 22.15 + 2.57 + 0.96;seq8 6.74 = 4.51 + 2.52 − 0.29。
生产者在 seq48 搬 69.6 MB / 15.13 us ≈ **4.6 TB/s**,对照 `pitfalls/02` 的流式地板 5.95-6.2 TB/s。

**absorb 是固定成本、不是数学**:`uk48 = 3.194` 但 `uk1 = 3.588`、`uv48 = 3.672` 但
`uv1 = 4.454` ⇒ M=1 比 M=48 **更慢**,别指望缩 M。用 wrapper 的 `config=` 覆盖扫了
两张表(各 10 臂),留着没动的现成余量:

- `BATCHED_GEMM-...-B=16-N=512-K=192.json` 的 `M_LEQ_48` 加 `"waves_per_eu": 2`:
  **3.1193 vs 出厂 3.1747(−1.75%)**;`wpe4` 3.1301。其余臂都更差
  (`4_2` 3.327、`8_3` 3.434、`256_4_3` 3.668、`16_64_4_3` 4.267)。
- N=256/K=512 那张的 `uv48`:`16_64_4_3` 3.6463 vs 出厂 `16_64_4_2` 3.6674,
  但该臂 spread 9.7% ⇒ **没判**,要重测。

---

## 补丁(GLM-5.2 decode 记分线, 轮 6, 2026-09-12):在线 softmax 的 rescale 可以整段消失

★ ✅ **把 rescale 折进 PV MFMA 的 C 操作数(`_fusedrescale`)**——本卡 §指令普查
「最大单块 = 140 条 `v_pk_mul_f32` = 每 tile 对 64 个 PV 累加器 VGPR 的无条件 rescale」
第一次被吃掉一半。两步都是恒等变换:
1. **beta 提前折进 P**:`probs = exp2(qk - max(running_max, tile_max))` 而不是
   `exp2(qk - tile_max)` 再乘 `beta`。`tile_denom` 自动带上 beta,递推式
   `next_denom = running_denom*alpha + tile_denom` 与原式逐项相等。
2. **累加器改用 FP8_MAX 单位、拿 `running_acc*alpha` 当 MFMA 的 C 种子**:
   `acc = Vector(running_acc[j]) * alpha` 后直接两条 `mfma32` 累加,尾 tile 用
   `rcp(next_denom * FP8_MAX)` 一次收尾。
⇒ 每 tile 每 wave 省 **16 条 `v_pk_mul_f32` + 16 条 `v_pk_add_f32`**(1284 条里去掉
128 条,−10%),`next_acc` 这个中间量整个消失。
同进程 ABBA(展幅 0.1~0.7%,两次独立复现):producer seq48 **−1.30/−1.65%**、
seq60 −1.22/−1.33%、seq84 **−3.78/−3.39%**、seq14(ii=2)−1.19/−1.58%;
全调用 seq48 −1.07%、seq60 −1.15%、seq84 −3.07%、seq14 −0.66%。
★ **精度反而变好**:`relL2` 0.026227 → **0.024898**(5 次尺子逐位复现同一值)。
beta<1 的 tile 的 fp8 网格虽然变粗,但它对输出的贡献正好按同一比例小;而原式里
`beta` 是在 fp8 量化**之后**乘的,量化误差被 beta 放大后仍留在和里。
★ 附带 ✅ **alpha 提到 rsum 会合之前**(折了 beta 之后 alpha 只依赖 loop-carried 量,
不再等 `tile_denom`):逐位一致,producer −0.37/−0.12/−0.10/−0.03%、
全调用 −0.33/−0.04/−0.20/−0.81%(48/60/84/14)。免费,顺手拿。
记分尺 6 跑(盘面终态):score **1.60338/1.60339/1.59021/1.58981/1.59669/1.60784
⇒ 中位 1.60004**,基线 3 跑 1.58131/1.57946/1.58358(中位 1.58131)⇒ **+1.18%**,
且 6 个数全高于基线最大值。`mla_v10` 中位 19.5148、`mla_v14` 中位 25.0485。
⚠ `mla_v8` 中位 17.633(−0.88% vs 基线 17.79),但它自己 6 跑范围 17.43~18.11
(3.9%)、单跑内 spread 也到 2.4% ⇒ **`mla_v8` 单点不是够用的判据**,要看 score
多跑或同进程 ABBA。

★ ❌ **`vlds` 双缓冲 = 本卡上一条自己开的药方,实测是白药**。本卡 §加法探针
写着「8 条 `ds_write_b128`/tile 值 seq48 +4.89% ⇒ `vlds` 双缓冲(33792→67584B,
`160/78.6=2.03` 仍 2 CTA/CU)是零算法风险的对策」。真做出来(第二块 key buffer,
在 rsum barrier 之后、本 tile PV 之前 publish 下一 tile,tile 尾的 WAR barrier
因此可以删掉,3→2 barrier/tile),ISA 侧完全如预期:**VGPR 228→210、spill 0、
LDS 44800→78848(仍 2 CTA/CU)**、输出逐位一致。但同进程 ABBA:producer
seq48 −0.70%、seq60 −1.77%、seq84 **+1.29%**、seq14 **+2.98%**;全调用
seq48 **+0.53%**、seq60 −1.56%、seq84 +1.02%、seq14 +2.00% ⇒ 记分尺几何均
≈ 0。prologue 只预热 1 个 tile 的变体(`prime=1`)把 seq14 的 +2.98% 压回 +0.33%,
但 seq60 的收益也一起没了(−0.39%)。
⇒ **加法探针的价格 ≠ 可回收的钱**(methodology/03 的「上界≠可达」在本族第二例):
那 4.89% 是**新增指令自己的发射+依赖成本**,不是既有调度里被暴露的 `ds_write`
延迟;pitfalls/03 的 type-B「write-read exposure」判据(`ds_write` 紧跟
`s_waitcnt lgkmcnt(0)`)在这里给出了假阳性。别再为「把 X 挪到别处盖住」立项,
除非能先证明 X 现在就在临界路径上。

★ **512 CTA 悬崖(动态实测,补上占用率的第五个证据)**:固定 ii=4 扫 seq、
per-tile 归一化(us/(seq×32)×1000):
seq 56/60/62/64 = 448/480/496/512 CTA ⇒ 8.891/8.754/8.717/**8.527** ns 单调下降;
seq 66/68/72 = 528/544/576 CTA ⇒ **10.260**/9.925/9.397。
**512→528 CTA(工作量 +3%)时间 +24%**,即刚过 2 CTA/CU 常驻上限就付掉约半个
CTA 波。⇒ 管理者本轮点的「seq48 重写成 512 CTA×3 tile 摊平网格」在算术上不通:
一个 CTA 只服务一个 token 时,32 tile/token 要压到 ≤3 tile 需 ≥⌈32/3⌉=11
CTA/token ⇒ 48×11 = **528 CTA**,正好落在悬崖的坏边;唯一整数 CTA/CU 点是
768(ii=2)与 1536(ii=1),轮 2 的固定工作量对照已给 768 = 只快 0.4% 且
combine +0.80us(`prod+comb` 反而更慢)。**要动这条轴只剩「CTA 跨 token 边界、
partial 记录数可变」一种形态**(每 token 覆盖 11 条记录、combine 读 8→11 条
≈ +0.8us 要先被 producer 的 −10% 盖住才立得住),别再试「每 token 均分 CTA」。

## ★★★★ 补丁(GLM-5.2 decode 记分线, 轮 7, 2026-09-12):记分尺 +2.32% = `inner_iter=8` 的窗口 + 只加深 prologue,两条门都是 **CTA/CU 数**

**留盘改动**(`sparse_mla_decode_kernels.py` 新增 `_decode_inner_iter` / `_pick_xpf_prime`,
`sparse_mla_decode.py` 新增 `xpf_prime`):记分尺 5 跑 score
1.63721/1.63159/1.63217/1.63782/1.63757 ⇒ **中位 1.63721**、展幅 0.38%,
vs 轮 6 留盘的 6 跑中位 1.60004(区间 1.58981~1.60784)⇒ **+2.32%,五个数全部高于
上一轮的最大值**。`relL2` 0.024898 → **0.023468**(5 跑同值;split 数减半 ⇒ 跨 split
合并误差变少)。核级中位:`mla_v8` 17.633 → **16.972**(−3.74%)、`mla_v10`
19.515 → **18.450**(−5.46%)、`mla_v14` 25.049 → 25.199(不动,按设计)。
配置六元组:verify `(48,32,8,4,True,True)`+prime2 / `(60,32,8,4,True,True)`+prime2 /
`(84,32,4,8,True,True)`+prime1(不变);draft `(8,32,1,32)`/`(10,32,1,32)`/`(14,32,2,16)` 全不变。

★ **`inner_iter=8` 不是"不可靠",是本卡 §decode 那句「不复现」的口径错了**。
本卡 line 102 记「ii=8 交叉点 seq48/60 一次 −0.84/−1.66%、复测 +0.54/−0.22% ⇒ 不复现,
别据此改 `_pick_inner_iter`」。轮 7 查明:**`prod+comb` 的绝对值跨进程不可比**
(同一份 ship 在三次探针进程里 seq48 拿到 17.42 / 17.88 / 18.22,差 4.6%),
而 producer-only 跨进程可复现到 0.24%。把不同 `inner_iter` 的 launcher + 各自的
partial buffer **放进同一个 ABBA 块**(一个进程持有全部 launcher,回文序 min-of-5)之后,
ii=8 的符号立刻稳定:`f_` 相对 ii=4 = seq48 −4.15% / seq60 −3.58% / seq64 −4.67%,
再叠 prime2 = **−6.26 / −5.42 / −4.63%**。
⇒ **判据:任何跨 tiling 的比较必须在同一个 ABBA 块里,不同 tiling 的中间 buffer 形状不同不是借口**
(给每个臂配自己的 `partial_output`/`partial_lse` 即可,输出张量共享)。

★ **ii=8 的两道墙都是 CTA/CU,不是 grouping 本身**(同块 ABBA,`f_` vs ii=4):

| seq | ii=8 CTA | CTA/CU | ii=8 | ii=8 + prime2 |
|---|---|---|---|---|
| 32 | 128 | 0.50 | +32.0% | +22.9% |
| 40 | 160 | 0.63 | +6.9% | **−3.3%** |
| 48 | 192 | 0.75 | −4.2% | **−6.3%** |
| 60 | 240 | 0.94 | −3.6% | **−5.4%** |
| 64 | 256 | 1.00 | −4.7% | **−4.6%** |
| 72 | 288 | 1.13 | +5.9% | +4.9% |
| 84 | 336 | 1.31 | +3.7% | +2.0% |

下墙 ≈ 0.6 CTA/CU(gather 在飞数量喂不饱 HBM,seq32 的 producer 9.95→14.32us),
上墙 = 1.0 CTA/CU(胖 CTA 的尾巴,一个 straggler 值 8 个 tile)。窗口内它同时买到两样:
producer 带宽不掉 + combine 的 partial 流量减半(ni 8→4)。
⇒ 本卡 §`_pick_inner_iter` docstring「ii=4 处处胜出或打平」在 seq 48..64 上是**错的**,
错因是它只看 producer 的尺子跑、且用了跨进程数;已在源码里把 ii=8 窗口写成 `_decode_inner_iter`
(`_pick_inner_iter` 本身留作基线策略,因为 op_tests 锁了它的表)。

★ ✅ **`_XPF_PRIME=2`(只加深 prologue,稳态仍 `_XPF_DEPTH=1`)在 aiter 树里补量完成**,
本卡 line 116 的 open 项(「本树没有这个常数、仍未量」)关闭。实现要点:prologue 发 2 个 tile,
循环里用 `len(pipeline)-(k_i+1) < _XPF_DEPTH` 记账,使**多发的那一个在第一次消费时被花掉**、
稳态深度仍为 1(否则就退化成已判负的全程深度 2)。ISA:1148→1151 条、**VGPR 162→166、spill 0、
LDS 不变 ⇒ 仍 3 CTA/CU**,输出逐位一致。但**它有一道门,符号在 2 CTA/CU 翻转**:

| ii=4 的 CTA/CU | 0.99(seq32) | 1.25 | 1.5 | 1.75 | 1.88 | 2.0 | 2.25 | 2.63 | 3.0 |
|---|---|---|---|---|---|---|---|---|---|
| `f_` Δ | +1.4% | −0.0% | **−2.2%** | **−1.7%** | **−2.3%** | **−2.7%** | +2.0% | +2.5% | +0.6% |

> 2 CTA/CU 时第三个共驻 CTA 已经提供了那批 outstanding miss,再加深只多排队。
ii=8 侧(0.63~1.31 CTA/CU)prime2 **处处为正收益**(−9.5 / −2.2 / −1.9 / −0.0 / −1.0 / −1.6%)。
ii<4 一律不开:ii=2 时发 2 个 tile 就是整个 CTA,等于已判负的全程深度 2(seq14 +1.6%)。
⇒ 与本卡 line 112 的 ❌「全程深度 ≥2」不矛盾:**prologue 深度和稳态深度是两个旋钮,
前者✅(带 ≤2 CTA/CU 的门)、后者❌**。

★ **512 CTA 悬崖第二次兑现**:它精确预测了 prime2 的符号翻转点(512→576 CTA 之间),
与本卡 §512 CTA 悬崖的静态测量(512→528 CTA 时间 +24%)是同一个物理。
⇒ **这条悬崖现在可以当先验用**:任何"多发/多驻/更深"的杠杆,先算 CTA/CU;> 2 就默认判负,别浪费一轮。

★ **ISA 普查刷新(轮 6 的 fold 之后)**,本卡旧数(1284 条 / VGPR 228 / 2 CTA/CU)已过期:
**1148 条,VGPR 162 / AGPR 0 / spill 0 / LDS 44800 ⇒ 3 CTA/CU**。前七块:
`s_waitcnt` 105(其中 **`lgkmcnt(0)` 全排空 54 条**)、`v_mov_b32_e32` 68、`v_pk_mul_f32` 68、
`ds_read_b64_tr_b8` 64、`v_mfma_f32_16x16x32_fp8_fp8` 64、`s_nop` 50、`ds_read_b128` 49。
🔧 工具订正:**`FLYDSL_DEBUG_DUMP_ASM=1` 不存在**(轮 6 记的那条无效),正确姿势是
`FLYDSL_DUMP_IR=1` + `FLYDSL_DUMP_DIR=<dir>` ⇒ 目录里除全部 MLIR stage 外有 `21_final_isa.s`;
统计时要按 `.amdgpu_metadata` 切掉尾部 YAML,否则计数为 0。

★ ❌ **`lgkmcnt(0)` 全排空的条数不是货币**(本族第三例"上界≠可达")。三条独立臂都把它砍下来、
墙钟都没动:①`lds_pad=12288`(纯粹把占用目标从 3 压到 2,源码不变)54→**33**、墙钟 ≈0;
②QK 的 8 条 `ds_read_b128` 按 pitfalls/03 Pattern3 显式整体前提(`qkb`)54→**34**、VGPR 174,
墙钟 seq48 −1.05/−4.56、seq60 −1.80/−0.89,但 **seq84 +2.67/+2.17**、seq14 +0.76 ⇒ 符号混杂,
**编译器在寄存器够用时自己就会做这件事**(ISA 里 tile k+1 的 index load 已被提到 tile k 的
`ds_write` 之前、gather 已在 tile k 的 QK 之前);③`rope32`(rope 收缩本来只有 K=64,
用两条 K=32 MFMA 代替一条零填充的 K=128)逐位一致、1148→**1131** 条、`v_mov` 68→**48**、
`lgkmcnt(0)` 54→**30**、VGPR 166(占用不掉),墙钟 producer +0.34/−0.83/+0.82/−0.74、
全调用几何均 **−0.16%**。⇒ **别再为"减少 LDS 排空/前提 `ds_read`"立项**;要动先证明它在临界路径上。

★ ❌ **`running_acc*alpha` 那 16 条 `v_pk_mul_f32`(每 tile 每 wave)的减法上界是负的**。
轮 6 报告里估「值 ~1%」,轮 7 直接做**上界探针**:把整段 `*alpha` 删掉(48 条 `v_pk_mul_f32`,
结果当然是错的,只为定价)⇒ producer **+0.53 / +0.23 / +4.38 / +1.16%**(seq48/60/84/14)。
⇒ `v_pk_fma` 合并、`alpha==1` 的 wave-uniform 快路这一族**整族没有钱**,别做。
(机械解释:删掉它反而让 MFMA 的 C 操作数依赖链变紧,调度器失去了填充空隙的指令。)

★ **为什么指令级杠杆在这条尺子上普遍 ≈0(定量)**:seq48 的 producer 搬
`48×2048×576B = 56.6 MB` 用 14.76us ⇒ **3.83 TB/s 有效**,乘 576B 行恒定的 1.111 行放大
⇒ **4.26 TB/s 线上带宽 = 实测 stream 地板(5.95~6.2 TB/s)的 69~72%**;尺子的索引是
`randint(0, 2.84M)`、48×2048 次抽样几乎不撞车 ⇒ **零复用**,剩下的 28~31% 是随机 128B
粒度的 DRAM row miss,不是指令。拟合的 `2.83us + 6.81ns×tile` 里那 2.83us 截距 = CTA 起点
无人可盖的 index→KV 两跳 + launch,**prime2 吃掉的就是它**,所以 producer 侧真正还剩的
可回收量级 ≈ 截距那 2us(13%),不在 1148 条指令里面。

★ ❌ **那 2.83us 截距不要再从 prologue 的发射顺序上抠**(轮 8 两臂,都退回)。
① 先订正一个**长期记错的前提**:prologue 那 3 条索引 `global_load_dword` **早就不在 barrier 后面**
—— ISA 第 75~85 行,LLVM 已经把 6 条索引 load 全提到 Q load 之前,并配 `vmcnt(7)/(6)/(5)`
分级等待。真正剩下的那条 `vmcnt(0)` 来自 `wave==0` 分支里发的 **Q 尾块 load**(576 = 2×256+64),
它排在 18 条 primed KV load **之后**,而 `vmcnt` 按发射序退休 ⇒ 想等它只能 `vmcnt(0)`。
② 于是把 Q 尾块提到 primed gather **之前**发(`qte`,逐位一致):producer
**+0.37 / 0.00 / +0.20 / +0.19%**(seq48/60/84/8),全调用 +0.04/−0.12/+0.25/+0.05%,
spread 0.09~0.43% ⇒ 省下的那次等待 CTA 几条指令后照样要做,而另外三个 wave 由此要 clamp 的
越界块比它更贵。**教训:先读 ISA 再信"把 load 提前"这类叙事,编译器多半已经做了。**

★ ❌ **`_XPF_DEPTH=2`(稳态 gather 预取深度)在 ii=8 下依然是负的,而且不是占用问题**。
本卡原先把深度 2 的失败归给占用;轮 8 专挑 **ii=8 / 0.75 CTA/CU** 这个"第二块 tile 的 36 个
gather 寄存器完全免费"的点复测:producer **+0.43 / +0.54 / +9.19 / +0.26%**(seq48/60/84/8),
叠在 `qte` 上(`qted2`)= +0.18 / +0.20 / **+9.44** / +0.14%。seq84 那 +9% 说明代价来自
**gather 在途请求挤爆了 L2/HBM 的随机 128B 通路**,与寄存器/占用无关。
`xpf_prime`(只在 prologue 加深)和 `_XPF_DEPTH`(稳态)是两个独立旋钮,**只有前者有钱**。

✅ **已落地(轮 8):combine 的 Dv 切片数 `dv_slices∈{1,2,4}`**。
`_compile_sparse_decode_direct_combine(ni, dv_slices)`,`values_per_lane = 512//(64*dv_slices)`、
`out_lane = (slot%dv)*64 + lane`、行主序编号不变;各片列不相交,**对全部形状逐位相同**
(fp32 golden 交叉验过 dv1/dv2/dv4 三臂 `torch.equal` 全 true,ruler `relL2` 稳在 0.023468)。

现役判据是 `_combine_dv_slices()`:`ni>16`(butterfly LSE)走原来的四片窗口,
**`ni<=16`(scalar LSE)一律两片**。同块 ABBA、放在 producer 后面整调用计时,
按"dv1 会发多少个 CTA(= seq×16)"排开,收益单调衰减但**从不变号**:
224 → −9.72%、288 → −8.00%、384 → −8.06%、512 → −3.74%、672 → −2.08%、
768 → −1.19%、960 → −0.84%、1344 → −0.74%、1536 → −0.58%(后两个用 14-rep 两块独立复现)。

❌ **别再试:给它加"grid 填满 1024 个 SIMD 才切"的门**。轮 8 按 `seq*16 < 4*num_cu` 加过,
把 seq84/96 这两个最重的 verify 形状关在门外白送掉。当初以为 1344 那点要变号,是被
**ruler 跨进程 `mla_v14` 约 0.7% 的漂移**骗了 —— 这条 kernel 的效应必须用同块 A/B 量,
不能拿两次 `bench.sh` 的 `mla_v*` 相减。

四片只在 scalar 路上亏:seq60 比两片慢 0.62%、seq14 慢 1.07%,唯一的赢面 seq84 −0.25%
第二块就不复现。机理很干净:**scalar LSE 每个 CTA 都要把整列 `ni` 个权重重算一遍
(每 split 一次 exp2),多切一片就多付一份;butterfly 的 cross-lane 代价与 split 数无关**,
所以只有 `ni>16` 那边养得起四片。寄存器上切片还是**净赚**的 —— ni13 的 ISA:
dv1 = 72 VGPR / 384 条,dv2 = 56 VGPR / 275 条,dv4 = 41 VGPR / 242 条,均无 spill、无 LDS。

配套结论:**换了更快的 reducer 之后 `inner_iter` 的选择没有变**。带 dv2 重量过
seq48 = ii8 16.73 < ii4 17.89 < ii2 18.86;seq60 = ii8 18.21 < ii4 18.94 < ii2 22.00;
seq84 = ii4 24.66 < ii8 25.28 < ii2 29.69 —— 与 R7 的 `_decode_inner_iter` 完全一致,不必重调。

## ★★★★★ 补丁(GLM-5.2 decode 记分线, 轮 9, 2026-09-12):**参差(ragged)分组** 把 partial 记录数从「2 的幂」放开 = 记分尺 +2.9%,`mla_v14` −8.9%

**留盘改动**(`kernels/sparse_mla_decode.py` 加 `n_groups` 参数 + 最后一轮的 mask;
`sparse_mla_decode_kernels.py` 用 `_decode_partial_groups` 取代 `_decode_inner_iter`/`_partial_groups`):
5 趟 ruler 1.68176 / 1.69272(留盘版)、1.68493 / 1.69165 / 1.69225(另含 uv 表 wpe),
`relL2_sparse_mla` 0.0234(旧 0.023468),`mla_v8` 16.63→16.07(−3.4%)、
`mla_v14` 25.03→22.80(−8.9%)、`mla_v10` 不变(配置没动)。

★ **本卡 line 100 那句「要吃 per-CU 锯齿需要『每 CTA 可变 tile 数』的重写,是本族最大的
已定价未动杠杆」——完全正确,而且它比标价还便宜**。不需要可变 trip count,也不需要
第二个 kernel body:`n_groups` 任意取值、`inner_iter = ceil(ng/n_groups)`,短的那几组
在**最后一轮把 index 行强制成 −1**,现成的 padding mask 就把它打成全 `-inf`:
`probs=0 / tile_denom=0 / alpha=1`,累加器、running max、LSE 全都**逐位不变**。
代价只有一条 `v_cmp` + 两条 `select`,加上那一格 tile 的白跑流量。

★ **一 CTA/CU 是**悬崖**,不是斜坡**(同块 ABBA,min of 5–6,尾部放 ship 复制臂量位置偏差
0.03–1.6%;`ng=32`,num_cu=256):

| seq | n_groups | CTA | CTA/CU | tiles/CTA | Δ vs ship |
|-----|----------|-----|--------|-----------|-----------|
| 11 | 16 | 176 | 0.69 | 2 | **−16.2%** |
| 20 | 11 | 220 | 0.86 | 3 | **−6.8%** |
| 24 | 8 | 192 | 0.75 | 4 | **−3.4%** |
| 24 | 10 | 240 | 0.94 | 4 | −1.1% |
| 32 | 8 | 256 | 1.00 | 4 | **−9.2%** |
| 32 | 9 | 288 | 1.13 | 4 | +9.7% |
| 40 | 6 | 240 | 0.94 | 6 | **−14.9%** |
| 40 | 7 | 280 | 1.09 | 5 | +2.7% |
| 48 | 5 | 240 | 0.94 | 7 | **−3.4%** |
| 48 | 6 | 288 | 1.13 | 6 | +18.7% |
| 60 | 5 | 300 | 1.17 | 7 | +21.6% |
| 72 | 3 | 216 | 0.84 | 11 | **−11.1%** |
| 72 | 4 | 288 | 1.13 | 8 | +4.6% |
| 84 | 3 | 252 | 0.98 | 11 | **−8.1%** |
| 84 | 4 | 336 | 1.31 | 8 | +2.5% |
| 84 | 2 | 168 | 0.66 | 16 | +20.2% |
| 96 | 2 | 192 | 0.75 | 16 | +1.9% |
| 96 | 4 | 384 | 1.50 | 8 | −0.5% |

**≤1.0 CTA/CU 的每一格都赢 3–16%,刚过 1.0 的每一格都亏 2–22%**。所以策略是
「**塞满 CU 阵列恰好一遍、CTA 能多胖就多胖**」= 最大的 `n_groups` s.t. `seq*n ≤ num_cu`,
等胖时取**更小**的 `n`(参差的 padding tile 要付 gather 流量:seq24 的 n=8 与 n=10 都是
4 tile,n=10 有 1/4 是 padding,差值就是那 2.3%)。这条规则在实测的 9 个 seq
(11/20/24/32/40/48/60/72/84)上**每一个都选中了实测最优臂**。

★ **上限是 tiles/CTA,不是 CTA/CU**:`ii=16` 两组时 seq84 +20.2% / seq96 +1.9%,
而 `ii=11` 在 seq72/84 是 −11%/−8%。故留了 `_DECODE_MAX_TILES = 11`;`seq ≥ 86`
(`num_cu//seq ≤ 2`)找不到合法格,回落到 `_pick_inner_iter` 的 2 的幂分组(seq96 = 0%)。

★ **旧口径被订正了两处**:
1. 本卡 §R7「`inner_iter=8` 的窗口是 seq 48..64,两道墙 0.6 和 1.0 CTA/CU」——
   **上墙 1.0 对,下墙 0.6 是假的**:它是「2 的幂只能跳半格」造成的假象。seq11 的
   0.69 CTA/CU、seq24 的 0.75、seq72 的 0.84 全是大赢。真正的下界是 tiles/CTA 太大(≥16)。
2. `_pick_inner_iter` docstring 的 ii=4 表和 `_decode_inner_iter` 的 ii=8 窗口表
   **都只是这条悬崖在 2 的幂网格上的两个采样**,已在源码里被 `_decode_partial_groups`
   取代;`_pick_inner_iter` 只作 `seq ≥ 86` 的回落(op_tests 锁了它的表,不能删)。

★ **参差分组顺手降了误差**:各 seq 对 triton 参考的 relL2(索引里掺 `-1` padding)
seq11 0.0181 / 20 0.0219 / 24 0.0228 / 32 0.0229 / 40 0.0238 / 48 0.0234 /
60 0.0236 / 72 0.0217 / **84 0.0218** / 96 0.0269。最胖的 seq72/84(3 个 bf16 partial)
**误差最低**,没动配置的 seq96(8 个 partial)最高 ⇒ 少一次 partial 落 bf16 比多一格
padding tile 值钱,padding 本身零误差。

## ✅ absorb tuned 表:`uv` 桶(B=16,N=256,K=512)`waves_per_eu 1→3` 是真赢(本轮未留盘)

轮 9 的 directive 只点了 `uk`(N=512,K=192)的 `M_LEQ_64`/`M_LEQ_128`,
全网格(BM 16/32/64 × BN 32/64/128/256 × nw 4/8 × ns 2/3/4 × wpe 1/2/3 × `.cg`/null)
扫下来**出厂值就是实测最优**:M=60 次优 `32_64_w8_s2_e3` −0.49%(位置偏差带 0.11–0.63% 内),
M=84 出厂即最优。**但同一算子的 `uv` 表没人扫过,而 `uv` 在三个并发上都比 `uk` 大**
(占 step 15.7–17.7% vs 12.6–15.5%)。`uv` 只有 `waves_per_eu` 一个轴有货,
每个 wpe 臂放两个位置、重复 8 次 palindrome:

| M | wpe=2 | **wpe=3** | 惰性对照臂(wpe=1 复制) | 位置偏差 |
|---|-------|-----------|--------------------------|----------|
| 48 | +0.05 / +0.11% | **−0.64 / −0.53%** | +0.03 / +0.02% | 0.09% |
| 60 | +0.15 / +0.15% | **−0.43 / −0.39%** | +0.16 / +0.13% | 0.12% |
| 84 | +0.07 / −0.05% | **−0.84 / −0.90%** | +0.20 / −0.06% | 0.07% |

逐位一致、12 个读数同号、全部超出对照带。折到记分尺约 +0.15%,**被 ruler 自身
0.43–0.65% 的趟间噪声盖住**,所以本轮为守「只留一个自洽改动」的规矩没留盘 —— 下轮
一行 JSON 就能落。另注:`ns` 在 `uv` 上**分桶翻脸**(M=48 的 ns=3 是 −0.51%,
M=60 的 ns=3 是 +2.43%),和 pitfalls/06 那条 wpe 分桶警告同源,别跨桶推广。

---

## 轮 10(2026-09-12):生产者已贴住机器地板;真钱在 `uv` 表的 `M_LEQ_128` 桶

★ ✅ **留盘改动:`uv`(B=16,N=256,K=512)的 `M_LEQ_128` 桶整条换掉**
`BM32/BN64/nw8/wpe1` → `BM16/BN64/nw4/wpe4`(同表 `M_LEQ_64` 顺手 wpe 1→3)。
同块 ABBA(对照臂 −0.32…−0.73%,展幅 0.4–1.3%,全臂输出逐位一致):

| M(桶内) | 出厂 | `BM16/nw4` + wpe1 | +wpe2 | +wpe3 | +wpe4 | CTA 数 |
|---|---|---|---|---|---|---|
| 66 | 5.1734 | **+27.93%** | −14.52 | −15.45 | **−15.73** | 320 |
| 84 | 5.2505 | **+33.44%** | −8.79 | −9.40 | **−9.85** | 384 |
| 100 | 5.3341 | **+32.68%** | −8.11 | −8.94 | **−9.26** | 448 |
| 128 | 4.4814 | **+31.69%** | −6.29 | −6.44 | **−6.77** | 512 |

★ **机理:Triton batched GEMM 也吃「>256 CTA 就要两块共驻」这条铡刀。**
grid = `(B, cdiv(M,BM)*cdiv(N,BN))`。`BM16` 把 M 切细 → M>64 时 CTA 数越过 256,
每个 CU 要放两块;`waves_per_eu=1` 让寄存器分配吃满、第二块进不来 → **+28…+33%**;
`wpe≥2` 一放开就变 −6…−16%。**所以 `BM` 和 `wpe` 在这族里不是两个独立轴,
是一个轴**:单独扫 `BM16`(wpe 留 1)会得到「BM16 是灾难」的假结论,
这正是出厂表把 `M_LEQ_128` 定成 `BM32/nw8` 的由来。**扫 tile 轴时必须把 wpe 一起动。**
桶边界 64 恰好压在 CTA=256 上(M≤64 ⇒ ≤256 CTA ⇒ 1 块/CU,hint 无所谓),
所以 `M_LEQ_64` 那半边 wpe 只值 −0.2…−0.6%,`M_LEQ_128` 这半边值 −9%。
与本卡 r9「1 CTA/CU 是悬崖不是斜坡」是同一条物理,**跨到了 Triton 侧**。

⚠️ **kb_check:上一轮本卡写的「`uv` M=84 `wpe 1→3` = −0.84/−0.90%」实测只有 −0.41%**
(在出厂的 `BM32/nw8` 形状上单动 wpe)。上一轮把 `−0.87%` 记成 wpe 的功劳,
其实那一臂多半连 `BM/nw` 一起换了 —— 我这轮第一遍扫也踩了同一个坑:
臂名叫 `wpe3` 但 cfg 写死 `(16,64,4,2,3)`,在 M=84 上同时改了 BM 和 nw,
读出 −8.93% 差点当成 wpe 的功劳。**臂名必须逐字段反映 cfg,否则归因必错。**

⚠️ 记分尺上 `uv_14` 历史 12 趟落在 **5.71–5.86**,本轮 4 趟 5.204/5.307/5.316/5.343
(**−8.4%**);同期 `uv_8`/`uv_10` 比历史高 1.7–2.8%,但**把 `M_LEQ_64` 改回 wpe1
单跑一趟,uv_8/uv_10 一动不动**(4.4932/5.0009)⇒ 那是尺子漂移不是改动的锅。
记分:1.70177 / 1.70179 / 1.69081 / 1.69572(r9 五趟 1.68176–1.69272),relL2 0.0234 不变。

## ✅ 生产者定价模型:`t ≈ 1.8us + bytes / 6.0–6.2 TB/s`(别再在核体里找钱)

把 r9 新放置下五个点的字节数(KV 行按 5×128B 计,含 Q 与 partial 回写)对时间拟合:

| seq | ii | CTA | 计费字节 | 实测 us | 有效 TB/s |
|---|---|---|---|---|---|
| 18 | 3 | 198 | 29.4 MB | 6.84 | 4.30 |
| 42 | 6 | 252 | 68.4 MB | 12.69 | 5.39 |
| 48 | 7 | 240 | 75.0 MB | 13.83 | 5.42 |
| 60 | 8 | 240 | 84.8 MB | 16.10 | 5.27 |
| 84 | 11 | 252 | 120.0 MB | 20.02 | **5.99** |

边际带宽 **6.0–6.2 TB/s** 正好是 `pitfalls/02` 记的流式地板;截距 **1.7–2.4us**
(seq84 只剩 0.66us,因为它最深、摊得最薄)。⇒ **核体里没有可买的带宽了**,
剩下的那 1.8us 是每次调用的派发/首拍延迟。想继续买这段只有两条路:
(a) 少一次 kernel 派发(producer+combine 合核 —— 本卡 ticket-atomic 那条 ❌ 是
「尾巴不再被下一个核启动重叠」判负的,换 persistent/cooperative 路线还没试过);
(b) 换 KV 布局少搬字节(576B 行跨 5 条 128B 线,放大 1.111,补齐到 640B 已判负)。

## ❌ 参差 padding tile 的字节不在关键路径上(别再想把它变便宜)

假设:seq48 的 `n=5/ii=7` 里有 3/5 的 CTA 末拍是死 tile(打 −inf、KV 乘 0),
占全机 9.4% 的 DRAM 流量,把它的行号折到小窗口应该能白赚。
实测(同块 ABBA,对照臂 +0.01…+0.17%,全臂逐位一致)**全是负的**:

| 死 tile 行窗 | seq48 | seq42 | seq84 |
|---|---|---|---|
| 折到 1 行 | +3.23% | +4.04% | +1.06% |
| 折到 32 行 | +3.96% | +4.27% | +1.14% |
| 折到 256 行 | +3.78% | +3.77% | +1.39% |
| 折到 4096 行 | +4.61% | +5.20% | +1.92% |

★ 关键在于**惩罚对窗口大小基本平**(1 行和 4096 行差 1.4%),所以**不是 L2 热行争用**;
ISA 也把占用排除了(两臂 `vgpr_count` 都是 170、0 spill、LDS 44800 全等,
只差 2 条 `v_cndmask` / 6 条 `s_waitcnt`)。真正的原因是**死 tile 根本不定墙**:
`split=0,1` 那两个 CTA 是 7 个**真** tile,把另外 3 个 CTA 的末拍弄便宜
只是让本来就先跑完的人更早收工。**参差分组的 padding 是「机器闲置」不是「浪费带宽」,
要吃它只能让 CTA 跨 token 边界重新均衡(32 tiles / 5 组 = 6.4,整不了)。**
判负成本:3 次远端探针(各 ~60s)+ 一次 ISA 对拍。

## ✅ 两张被 r9 放置改动作废的拟合表,复测后都确认「出厂即最优」

- **`_pick_xpf_prime`(prologue 预取深度)**:先打印 r9 新放置的四元组
  (seq48 → n=5/ii=7/240 CTA/0.938;seq60 → 4/8/240/0.938;seq84 → 3/11/252/0.984;
  seq8 → 32/1/256/1.000;生产态 seq42 → 6/6/252、seq18 → 11/3/198),再逐点扫深度。
  深度 3 = +6.94…+8.76%,深度 4 = +9.83…+11.85%,深度 1(ii≥6 处)= +0.81…+2.28%;
  ii=3 的 seq18 上深度 2 = +2.50%、深度 3 = +4.61%,出厂的 1 就是最优。
  ⇒ 现行规则 `ii<4 → 1;producer_ctas ≤ 2·num_cu → 2` **在每个新点上都复现实测最优,
  一行都不用改**。与本卡「`_XPF_DEPTH≥2` 到处更差」同号:**这族越多在飞的 gather 越慢**。
- **combine 的 `dv_slices` / `heads_per_cta`**:r9 把 ni 从 8..16 压到 5/4/3 后复测,
  出厂规则仍最优(seq48 ni=5:dv4 −0.05%、dv1 +0.11%;seq60 ni=4:dv1 +0.72%、dv4 +0.97%;
  seq84 ni=3:dv1 −0.14% 但对照臂 +0.34%;seq14 ni=16:dv1 +10.31%;seq8 ni=32:dv2 +5.61%)。
  `heads_per_cta>1` 在所有点都是负的(+0.32…+16.04%)。ni 变小只是让 combine 对切片
  **不敏感**(seq84 上五个臂只差 1.5%),没有把最优点挪走。

---

## ★★ 裁决(轮 11, 2026-09-13):XCD 亲和重映射在**新布局**上仍是「生产赚 / 记分尺亏」,但两边都缩水了 ⇒ ❌ 不落

上面轮 3 那段写的「生产 −21~34% producer,记分尺 0」**口径已经过期**。轮 9 的 ragged
placement(`_decode_partial_groups`:取最大的 `n_groups` 使 `seq*n ≤ num_cu`)改变了
baseline 的所有权,必须重量。同进程同块 ABBA + 惰性对照臂(`ship_ctl`,与 ship 逐字节
同一 config),两种索引分布各测,`partial_output`/`partial_lse`/`out` 三者 `torch.equal`
全部为真:

| seq | 布局 | `rand`(记分尺) full / producer | `sh6`(生产结构,复用 5.24–5.27) full / producer |
|---|---|---|---|
| 48 | n_groups=5, ii=7, 240 CTA | **+8.46% / +9.65%** | **−5.60% / −12.17%** |
| 84 | 252 CTA | +5.13% / +5.46% | −8.93% / −13.71% |
| 42 | 252 CTA | +9.18% / +10.65% | −6.97% / −15.05% |
| 18 | 198 CTA | +1.76% / −0.97% | −2.81% / −6.96% |

对照臂只动 −0.46%…+0.58% ⇒ 上面每一个数都在噪声之外。按本轮 ≥5% 才落的门,
**记分尺上是 ❌**。

**「producer −21~33%」与今天的 −12~15% 不矛盾,是两个不同的布局**:轮 3 量的是
`n_groups=8, ii=4, 384 CTA`(1.5 CTA/CU),那时 baseline 的 split-major
(`owner=split*seq+tok`)把一个请求的 6 个 token 撒到 6 个 L2 域;轮 9 之后 seq48 是
`n_groups=5`,而 `48 % 8 == 0` ⇒ **同一个 token 的 5 个 split 本来就已经落在同一个
XCD 上**,请求内的 Q/KV 复用被白捡了一半。剩下能靠 remap 买的只有「请求的 6 个 token
互相聚到一个域」,就是那 −12~15%;同时记分尺侧的代价从 ≈0 涨到 +9.7%
(remap 打乱了均匀随机索引原本自带的跨 CTA 域内条带)。
⇒ **ownership 类杠杆的收益是「布局 × 索引分布」的函数,换了 placement 必须整表重量,
不能沿用上一轮的百分比。**

★ **记分尺侧那 +9.7% 已用 ISA 排除「算术/寄存器」两种解释**(seq48 / ii=7 / n_groups=5,
两臂各编一次):ship `VGPR 170 / SGPR 29 / spill 0 / 1838 条`,remap 臂
`VGPR 164 / SGPR 31 / spill 0 / 1872 条`。remap 多两条运行期整数除法,却**少用 6 条 VGPR、
零 spill、只多 34 条指令**(+1.8%),而且 240 CTA / 256 CU 下每 CU 至多 1 个 CTA,
占用根本不参与。⇒ 那 9.7% 只能是**访存侧**(域内条带被打乱),不是发射或寄存器。
剩下的定价手段是对两臂做 rocprof 的 `FETCH_SIZE` / L2 命中率对拍 —— 这条还没做。

★ 另记一个实现陷阱:若把 host gate 写成 `seq % 6 == 0 && ctas % 8 == 0`(为了让
`per_xcd = seq*n_groups/8` 的分段是双射),则 r9 布局下 **seq84/42/18/8 四个形状全部
落进 no-op**(252/252/198 不被 8 整除,seq8 不被 6 整除),只有 48/60 能量到。
要覆盖 directive 点名的形状,分段必须写成能处理「余数段」的一般形式。

## 🐞 存量补丁要连它的 **gate** 一起复核:轮 9 的 placement 悄悄把轮 3 的补丁变成了空操作

`R2prime_xcd_affinity.patch` 的 host 侧 gate 是
`if (seq * n_groups) % _XCDS: return 0`(旧 segmentation 用
`per_xcd = ctas // 8` 做段基址,要求 8 整除 grid 才是双射)。轮 9 的 ragged grouping
把 grid 变成 240 / 240 / **252**(seq84 与 seq42)/ **198**(seq18)⇒ directive 点名的
4 个形状里有 **3 个** 会直接走 `return 0`,补丁在那里根本没被执行过。
要量它必须先把段基址从**乘积**改成**前缀和**:
`flat = dom*(ctas//8) + min(dom, ctas%8) + owner//8`(8 整除时退化回原式)。

⇒ **凡是跨了几轮的存量补丁,先核它的启用条件在今天的树上还成立不成立**,
不然会拿到一个「实测无差别」的假结论。同理:kernel 半边已经在树上、host 半边没接线的
补丁(`xcd_g` 编译期参数存在但 `_launch_partial` 从不传),`grep` 到 kernel 里有代码
**不等于**它在跑。

## 🔧 KB 纠错:本卡「rsum 非结构性、可整段提到尾声」的建议**已被后续轮次实测否掉**

上面轮 5/R10 那段写的「rsum 这一半值 1~4%,且不是结构性的,可以整段提到尾声」
(依据是**加法**探针:多插一次 rendezvous = +1.37/+1.01/+4.00%)。
**减法**已经做过了,结论相反,而且就记在
`aiter/ops/flydsl/kernels/sparse_mla_decode.py` 的原位注释里:

> Deferring them is exact ... but it is slower: the four ds_read_b32 below are the
> first LDS traffic after the barrier and they cover the latency of the plds read
> that feeds PV. Dropping them cost 1.2% at seq 48, 2.0% at seq 60 and 5.9% at seq 84.

⇒ 那 4 条 `ds_read_b32` 不是净开销,它们**盖住了喂 PV 的 `plds` 读的延迟**。
**加法探针给出的是「这段在关键路径上的边际价」,不是「删掉它能省的钱」** ——
在一个有空闲发射位可以吸收额外访存的核上,两者可以符号相反。
要给「删掉 X」定价,只能做 X 的减法臂,别用加法臂反推。
(⇒ 这条已写进 `methodology/03`;本族目前 producer 侧仍开着的是
per-CU 锯齿 / launch 截距 2.83us,rsum 这条关掉。)

## 补丁(GLM-5.2 decode 记分线, 轮 12, 2026-09-13):死 tile 的**零流量**版也判负 + 两条定价线复核

本轮三个假设全部判负,树留基线(`score 1.71778`,与 r11 的 1.719 同噪声带)。
**两条与存量卡撞车**,先记教训:本卡 922 行,「✅ 出厂即最优」那几节和 ❌ 节
**一样有约束力**,只 grep 关键词不够,提假设前要把**整卡**过一遍。

★ **H1 ❌「把参差 padding tile 的 gather 真正消掉」——比 r10 的「折到小窗口」更强的版本,仍然负**

r10 那节(本卡「参差 padding tile 的字节不在关键路径上」)量的是把死行**重定向**到
1/32/256/4096 行窗口,请求照发。本轮量的是**请求根本不发**:`num_records` 收到 `1<<31`
(`use_buffer` 的门就是池 < 2GiB,所以每个真行都在界内),让死 tile 的 gather 越界,
硬件边界检查直接回零、不进访存系统。两种写法(同块 ABBA、`ship_ctl` 对照 ±0.46%、
`partial_output`/`partial_lse`/`out` 三者 `torch.equal` 全真):

| 写法 | seq48(pad 8.57%) | seq84(pad 3.03%) | seq60(pad 0,空对照) |
|---|---|---|---|
| 每车道 `v_cndmask` 把行号打负 | **+6.69%** | **+2.44%** | +1.50% |
| 标量 `soffset` 整拍越界(零 VALU) | **+3.84%** | **+1.31%** | **+0.11%** |

两行相减 = **2 条 `v_cndmask` 插在「index 取回 → buffer_load 发射」这条地址链上值
seq48 +2.85% / seq84 +1.13%**。这是本核关键路径上**单条 VALU 的价**,以后判断任何
动 gather 地址的改动都可以直接拿这个数比。空对照那 1.50% 也是同一族:只把
`v_cndmask` 的 `0` 立即数换成 `4_000_000`(VOP3 不能带 32 位字面量 ⇒ 多占一个常驻
VGPR)+ 收紧 `num_records`,**一个字节的流量都没动,就值 1.5%**。

剩下那 3.84%/1.31% 与 pad 比例线性(0.448 / 0.432 每 1% pad)⇒ 与 r10 同因:
**死 tile 不定墙,让先跑完的 CTA 更早收工反而扰动了访存节奏**。r10 的机制结论
**在「零流量」这个极端上被确认**,不是例外。⇒ 这条方向到此为止。

★ **技术点(可复用,本族用不上)**:`buffer_ops.buffer_load(..., soffset_bytes=)` 是
指令的 SGPR 字段,配合收紧的 `num_records`,可以用**一条标量 select** 让整拍 gather
越界作废,逐位精确、零逐车道地址算术。`_MISS_SOFFSET = -(1<<31)`(i32 写法,无符号
正好等于 `num_records`)。要用在「整个 CTA/整拍统一跳过」的场合才划算。

★ **H2 ❌ prologue 深度 3/4 —— 与本卡 r9/r10 复测撞车,数字复现**

r9/r10 记的是「深度 3 = +6.94…+8.76%,深度 4 = +9.83…+11.85%」;本轮独立再测
(seq48/60/84,prod-only)= **深度 3 +10.39/+7.28/+5.63%、深度 4 +13.25/+11.22/+8.35%**,
`full` 口径 +4.78…+11.07%,全部逐位一致。⇒ **同号、同量级,跨轮跨进程可复现**,
`_pick_xpf_prime` 封顶 2 是对的。本轮顺手排除了一个诱人的反论证:这三个形状是
**0.94–0.98 CTA/CU(每 CU 就一个 CTA)**,深预取的活寄存器**不花占用率**,
但仍然单调更差 ⇒ **挡住深预取的不是占用率**(与本卡「加 LDS 压占用只值 0.1–0.2%」同向)。

★ ✅ **生产者定价线在第三条轴上复核通过**(本卡 `2.83us + 6.81ns × tile` / `1.8us + bytes/6.0–6.2TB/s`)

前两次拟合都动了 seq 或 ii(同时动了 CTA 数和总工作量)。本轮换一条干净的轴:
**固定 `n_groups`、固定 CTA 数、固定 ii,只扫真 tile 数 `ng`**(`compile_sparse_mla_partial`
校验 `ii == ceil(ng/n_groups)`,所以 ng 30/31/32/33 落在同一个 ii 上):

| seq | n_groups/CTA | ng=8/16/24/32 (us) | 斜率 | 截距 |
|---|---|---|---|---|
| 48 | 5 / 240 | 5.554 / 8.466 / 11.247 / 13.776 | 0.343 us per +1 ng = **7.15 ns/tile** | **2.7 us** |
| 84 | 3 / 252 | 7.171 / 12.935 / 16.303 / 20.202 | 0.543 us per +1 ng = **6.46 ns/tile** | **2.8 us** |

与存量线 `6.81 ns/tile` 差 ±5%,截距与 `2.83us` 一致 ⇒ **三条独立轴同号,这条线可以当钱用**。
顺带:同一 ii 内把一个 pad 槽换成真 tile 只值 0.189us(seq48)⇒ **seq48 那 3 个 pad 槽
总共只值 ~0.6% producer**,H1 的天花板本来就在噪声附近,这个数以后可以先算再动手。

★ ✅ **launch 地板复测 1.85us**(本卡 ticket 那节记的是空 triton 核 1.93–2.00us)

同一套 HIP-graph/20-replay/min-of-5 口径下,1 元素 `torch.add` = **1.8477us**、
1M = 2.52us、16M = 27.78us。⇒ 地板 1.85us,一步 324 次 launch ≈ 599us ≈ `step_c8` 的
**31.5%**,与本卡记的 31% 吻合。**但别再从这个数推「合核能回收多少」** —— 本卡
ticket 那节的教训 2(空核 duration ≠ 依赖链上的增量;尾巴原本被下一个核的启动盖掉)
就是为这条推理写的,融合臂实测 `mla_v8` 17.79→19.03。要定价只能用 `methodology/18`
的重复启动法。

★ **记分尺成分(本轮实测,`step_c8 = 1904.8`,公式逐项对得上)**:
`mla_v8*79 = 1277.3`(67.1%)、`uv_8*78 = 351.5`(18.4%)、`uk_8*78 = 242.6`(12.7%)、
`mla_d8*5 = 33.3`(1.7%)。把 `mla_v8 = 16.169` 再拆(同进程 ABBA):
producer 13.65 + combine 2.32 ≈ 15.97(占步 56.6% / 9.6%)。producer 那 13.65 里
**2.8us 是截距(占步 11.6%)、10.85us 是 tile 流量(占步 45.0%)**;combine 那 2.32us
里 **1.85us 是 launch 地板** ⇒ combine 的「核体」只有 0.47us。

★ 🔧 **工具**:本轮探针(`_probe_oob.py` / `_probe_rounds.py` / `_probe_prime.py`,在
campaign 目录)与记分尺的 `mla_v` 对得上 —— 探针 `full` 15.94/18.25/22.81 对
尺 `mla_v` 16.17/18.22/22.69(差 0.2–1.4%)⇒ **可以用 60s 的探针代替 77s 的尺做筛选**,
且探针能做尺做不了的同块 ABBA + 逐位对拍。`rocminfo` 本轮又被并发任务卡住一次
(`_detect_native` 60s 超时),`GPU_ARCHS=gfx950` 绕过,之后自愈。

---

## ✅ 已验证:producer+combine 的 **persistent/cooperative 合核**(2026-09-13 r14,已落盘)

本卡此前记「ticket 原子融合 ❌(`mla_v8` 17.79→19.03、最坏 +144%),**剩下的路是
persistent/cooperative**」。那条 open 项本轮走通并留盘,记法如下。

**形态**(`kernels/sparse_mla_decode.py` 的 `fuse_combine=True` 分支):
1. partial / LSE 用 `cache_modifier=0x11`(SC0|SC1)写出 —— 见 `pitfalls/03` 的 CPol 勘误;
2. 每 wave `s_waitcnt vmcnt=0` + workgroup barrier(**写穿本身就是 release,不要 fence**);
3. 每 CTA 一个线程在**按 token 分桶**的 sense-reversing 计数器上会合
   (`SENSE = -(1<<31)`;master 加 `SENSE-(n-1)`、其余加 1 ⇒ 每次发射恰好加 `SENSE`,
   **HIP graph replay 不需要重置**,数百次 replay 后仍正确);计数器每槽独占 128 B 行;
4. 归约由该 token 的**全部** `4*n_groups` 个 wave 轮转分担(不是 ticket 那种"最后一个 CTA 全做")。

**实测**:三个 verify 形状 `torch.equal` 逐位相同;fuse vs 两核 −4.95 / −5.46 / −1.02%
(seq48/60/84)、draft seq10 −6.06% / seq14 −7.60%;记分尺 1.7190 → 中位 1.7614(**+2.47%**)。

**开窗判据(两条都必须成立)**:
* **驻留**:`seq * n_groups <= num_cu`。会合只有在一个 token 的全部 split **同时在飞**时才终止;
  网格不超过 CU 阵列 ⇒ 每 CU 至多一个 CTA。仍要给轮询加上限,把"假设被破坏"变成错答案而不是挂设备。
* **列长**:`2 <= n_groups <= 16`。**`n_groups=32` 实测 +3.25%**(seq8):32 个 CTA 抢一个计数器,
  且权重重建从标量列换成跨 lane 蝶形、每 lane 只剩一个 dword。

**为什么值**:一个 graph 节点在本设备上恒定 1.5–1.8us,而 combine 的**核体**用
`methodology/18` 的重复启动法量出来只有 **0.134 / 0.218 / 0.526us**(48/60/84)⇒ 88% 是派发。
合核买的是那个节点,不是归约本身。

**旋钮**:`_BARRIER_SLEEP`(轮询退避)在 {0,1,2,4} 上是噪声,但 **0 会把展幅炸到 10.2%**,保持 1。

## ❌ 别再试:把 epilogue 的 dv 切分由「同 head 切 DV」换成「同 DV 切 head」(2026-09-13 r15)

合核 epilogue 的 slot 现在按 **head-major** 排:slot `j` = head `j/dv`,所以同一个 head 的
`dv` 个 wave 各读一遍该 head 的 LSE 列,且一个 wave **每轮换一个 head** ⇒ 每轮都要重建权重。
直觉上的对策是按 **DV-major** 切:每个 wave 领一整个 head、跨轮走 DV 分片,一个 chunk 的
`dv` 轮**共用一次 LSE 列 + 一次 softmax 权重**。轮数、每轮 store 宽度、负载均衡全不变。

**实测(seq84 = 唯一多轮的计分形状,n_groups=3/units=12/dv=2/3 轮;同块 ABBA + 惰性对照臂)**:

| 臂 | us | vs ship |
|---|---|---|
| ship(head-major) | 22.1313 | — |
| **dvm(DV-major)** | 22.2499 | **+0.536%** |
| ctrl(ship 的惰性副本) | 22.2428 | **+0.504%** |

⇒ `dvm − ctrl = +0.03%`,**逐位相同,净效果为零**。它确实把每 token 的非缓存 LSE dword
读从 96 条降到 60 条(−37.5%),但**省下的那一项从来不在关键路径上**:LSE 列与该轮的
partial 是同时发射的,轮的时延由 partial 那条(每 slot 512 B、同样绕过所有缓存)决定,
LSE 只是躲在它后面的 3 个 dword。

★ **判据**:合核 epilogue 里「减少 metadata 读」不是杠杆,**减少 partial 轮数或 partial
字节**才是。同族已判负的还有:`dv=1`(2 轮)+2.21%、`dv=4`(6 轮)+2.10%、混合宽度 2 轮
+0.96% —— 四条都指向同一件事:**轮数与每轮 store 宽度是唯二的轴**。

## ★ decode producer 的墙 = 最忙 CTA 的**真实** tile 数 + 脚印,不是每 CTA 的 slot 数

`_decode_partial_groups` 选的是「装得下阵列的最短 CTA」,于是 seq48 落在 240 CTA × 7 slot,
而真实工作量只有 1536 tile = 256 CU × 6 ⇒ 看上去有整整一个 slot 的水分。**它值多少,要量。**

**定价法**:固定 `n_groups=5`(网格恒为 240 CTA、ownership 与 epilogue 全不动),只扫
`ng_total`,这样改变的只是「五个 CTA 里有几个扛第七个**真** tile」,而 31 以上每臂都是 7 个 slot:

| ng_total | slot/CTA | 扛 7 真 tile 的 CTA 数 | us |
|---|---|---|---|
| 30 | 6 | 0 | 14.58 |
| 31 | 7 | 1 | 15.07 |
| 32 | 7 | 2 | **15.28(出厂)** |
| 33 | 7 | 3 | 15.43 |

* 关键 CTA 上的**第一个**真 tile = **0.49us**;
* 之后**每多一个 CTA** 把 padding 换成真 tile 只要 **0.18us** —— 而这 0.18us 恰好就是多出来的
  行给 gather 那 2.6us 暴露 DRAM 带来的 +6.5%(见本卡 regime 段)。

⇒ 两条结论:
1. **padding tile 不花流量**(越界行号被 buffer 边界检查干掉),只花骨架;而骨架花在
   **非关键 CTA** 上 ⇒ 240/256 网格的余量把它吸收掉了。这解释了 r10/r12 三次「让死 tile 更便宜」
   全部判负:**要省的那笔钱不在关键路径上**。
2. **「跨 token 持久网格 / 完美打包成 256 CTA × 6 slot」的上界 ≈ 14.75us vs 15.28 = −3.5% producer**
   (≈ 记分尺 +0.5% 几何均),**不是**按「每 CTA slot 数 7→6」纸面算出来的 −8~14%。
   再减去 remainder CTA 要多做 2 次 Q publish,基本抹平。★ **别再按 slot 数给调度重写定价,
   先用这张表量关键 CTA 的边际真 tile。**
