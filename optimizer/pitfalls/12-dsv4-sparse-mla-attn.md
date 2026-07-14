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

- SNR:cr4 路径 99.0dB(fast_path/_FMAX0 纯累加高精);banded/cr0/cr128 55-78dB,均 ≥ 门槛(pro cr4 ≥62.5dB)。
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
- **s_setprio**:全 DEAD(fwd 是 VALU+read bound,优先 MFMA 反饿死 VALU)。
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
