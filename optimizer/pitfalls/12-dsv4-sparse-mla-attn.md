# pitfalls/12 — DSV4 sparse-MLA attention (fwd/bwd) 实测经验 + 死胡同

> 任务：优化 `sync/mxfp4/Primus-Turbo/primus_turbo/flydsl/attention/kernels/sparse_mla_v2/`
> 的 flydsl fwd/bwd kernel。跑分 = **6 组均值**（flash H64 / pro H128 × cr∈{0,4,128}）@seq=4096。
> 目标：**fwd 均值 900T / bwd 均值 600T**。bench：`campaign_fwd.py` / `campaign_bwd.py`
> (末行 JSON {ok,tflops}，ok=每组 flydsl-vs-triton_v2 SNR 门过)。单核隔离：`_bench_interm.py`；
> triton 参照：`bench_triton.py --cr 0 4 128`。

## cr 语义（决定 shape/瓶颈，务必先懂）
- **cr=0**：纯 SWA，无 pool，topk=128（kv=latent，num_kv=S）。kv=j 被连续 token 窗口 [j,j+W-1] attend → **banded 结构**。
- **cr=4**：P=S/4，随机 topk，topk=flash640/pro1152。pool-heavy，gather 不规则。
- **cr=128**：HCA，P=S/128=32，topk=160。pool 项 causal 可见 → **pool entry 0 被 ~全部 S token 命中**（CSR list ~4000，极不均衡）。

## 现状（2026-07-09 深夜，已实测，均已 commit；chi2762 被 kimi-sglang 占,执行环境切 chi2811 8卡空闲同容器/repo,用 campaign_remote_2811.py）
- fwd 均值 **395T**(flash cr0/4/128=266/537/311,pro=324/601/331)。
- bwd 均值 **275.2T**(flash=221/307/188,pro=249/455/231)，baseline 250.4→275.2=**+9.9%**(融合+rtr全pro+transpose-overlap+dS/P双缓冲+**K=32 MFMA 双路径**)。
- **目标"6组全 dQ≥2× + interm≥2×triton"已用实验证不可达**(见下 ❌);当前最好 dQ flash cr4 2.36×、pro cr4 1.90×,小topk 1.42-1.51×;interm pro cr4 1.63×、pro cr128 1.04×、pro cr0 0.91×,余 <1.2×。
- 900/600 分别是 ~2.3× / ~2.3×。**dsv4 fwd/bwd 都恰是 dense flash-attn 的 ~2.3×**(dense bwd 634)=sparse gather/scatter 结构税(interm+gather 过 2× 的 [T,topk,D] tensor,pro cr4=9.6GB HBM≈2.4ms/6.9ms;dense 无此步)。**最强单组 pro cr4 也才 443<600** → 要 600 得每组超现最优 35%,是 sparse 天花板。

## ★★ 2026-07-10 (campaign 154106, r2) 权威 ISA 修正 + 新确认 WIN (务必先读)
> 用单核隔离 `FLYDSL_DUMP_IR` dump `num_vgpr/num_agpr/next_free_vgpr` + rocprofv3 PMC 校准, 纠正本卡旧 VGPR 数字与占用率理解:
- **占用率 = `512 // next_free_vgpr`, 且 `next_free_vgpr = arch_vgpr + agpr`(合并寄存器池, 非分离预算)**。rocprof 报的 `next_free_vgpr` 已是 arch+agpr 之和, 勿误读成两份相加。→ methodology/04 + pitfalls/03「VGPR+AGPR 共用 512 池」正确。**AGPR-accum 只在池内搬家, 不抬占用率**(除非本有 spill/accvgpr-copy, 本族 scratch=0 无此收益)。
- 实测 occ(全 occ-1 除 interm 小R): dQ pro cr4 = arch255+agpr101=**357→occ1**(旧卡「172 VGPR」作废); fused flash cr0 = 256+133=**389→occ1**(旧卡「128+128=256/occ-2 batching」作废, fused 结构不可达 occ-2); interm rtr cr4=256+4=260 occ1; interm rtr cr0/cr128 agpr0=210/224 **occ2**。
- **occ-2-via-prefetch-drop 判负**: interm rtr cr4 去 dS/P 寄存器预取 260→218 拿 occ-2 → pro cr4 **−1.4%**(丢的 prefetch 延迟 > wave-switch 收益)。→ **占用率不是本族杠杆**, occ-1/occ-2 内核都 latency-bound(MfmaUtil 12-35%)。dq_acc 128 / q_do_packs 256 是结构硬底, dQ/fused 不可达 occ-2。
- **✅ 新确认 WIN — fused KV LDS 双缓冲**: fused ISA 显 **17 barrier/迭代**(dQ 相 KV 单缓冲 2 barrier/tile)。改 KVBUF=2(store_kv/`_bv`/pv_base 加 `bufo=(kt%2)*KV1`)去覆写前 barrier(17→13) → 逐组(去 DVFS 漂移)flash cr0 **+~2.5-3%**、flash cr128 **+~2%**, 保守 mean **+1.3~2**(两跑复现), SNR bit-exact。低风险(+16KB LDS 仍 occ1)。⚠️ 本容器 mean DVFS 漂移 ±1.5(276.3↔278.0), **用配对窗口 A/B + 逐组数字**别信裸 mean 差。
- ❌ **fused qb/dob(interm 相 Q/dO d-block)双缓冲判负**: 叠在 KV 双缓上再去 restage barrier(8→4)+34KB LDS → MEAN 279→278.1(反降)。原因: restage barrier 有益地串行化 LDS 流量; 去掉后下一 d-block 的 staging store 与本 d-block 的 tr16-read(interm 是 LDS-read-bound)抢 LDS 端口。→ fused 只留 KV 双缓, 别动 qb/dob。
- **dQ bank conflict 需 XOR swizzle 非 pad**: dQ PMC LDSBankConflict **12.4%**(QK v8 read 2-way)。单一 D_LDS pad 值无解(520/536 均 ≤528 baseline, QK↔tr16 需求相反)→ 用 XOR swizzle 令两读同时 16-distinct(store 与两 load 三处地址须对称)。dQ 占每组 37-52%→广谱杠杆。
- interm rtr _ILP@R1152 = 中性(编译器已排 acc 链)。

## ★★★ 2026-07-11 (campaign 154106, r16) dQ stall 归因表 (subtractive wall-clock 探针, 决定性, 务必先读)
> 无 ATT decoder .so 且 PMC 计数器被同集群别的 campaign 占锁 → 改用 **subtractive 编译期探针** (`_dq_probe_kernel.py`: skip_pv/hoist_tr/skip_qk/skip_store/skip_barrier/dlds/dual_kv 逐个门控), event-median x40-60 隔离计时。把 dQ occ-1 wall 逐指令拆开:

| 组 | base us | tr16-read | PV-mfma | QK(read+mfma) | **residual(skip_both floor)** | dS/P-store | per-tile barrier |
|---|---|---|---|---|---|---|---|
| pro cr128 (R160) | 761 | **2.9%** | 12% | 37% | **57%** | +0.7% | **−2.2%(有益)** |
| pro cr0 (R128) | 684 | **2.0%** | 11% | 37% | **59%** | +0.2% | **−3.0%(有益)** |
| pro cr4 (R1152) | 2844 | **5.1%** | 30% | 44% | **53%** | −4.2% | **−13.8%(有益)** |
| flash cr4 (R640) | 941 | **3.0%** | 28% | 39% | **61%** | −1.8% | **−9.8%(有益)** |

**四条决定性结论 (逐一推翻本卡/goal 旧假设)**:
1. **tr16-read latency ≠ dQ bound**: 仅 2-5%(hoist_tr 单读一次 v-operand 复用 32 d-tile vs base)。**本卡长期把 dQ 定为「tr16-read-latency bound」是错的**——tr16-read 便宜。任何「减/加速 tr16-read」杠杆对 dQ 低价值。
2. **QK 13% bank-conflict 是 off-critical-path**: `dlds520`(QK-clean/tr16-dirty) 比 528 base **慢** +1.7~39us; **`dual_kv`(两份 KV-LDS: 520 供 QK-clean + 528 供 tr16-clean, 彻底消 QK 冲突且不伤 tr16) 全组大幅判负 pro cr128 +7.7% / pro cr0 +7.4% / pro cr4 +17.1% / flash cr4 +13.0%**——多写一份 KV 的 4 条 LDS store/tile 直接压在 occ-1 关键路径上, 远超它消掉的(近乎被掩盖的)QK 冲突。**→ dQ 双-layout-KV(goal N3(b) 的「从没试」杠杆)= measured 死。**
3. **barrier 与 dS/P-store 是有益/免费, 不是可消成本**: 去掉 per-tile barrier **更慢** +2~14%(occ-1 下 barrier 有益地串行化 LDS 流量, 呼应 fused restage-barrier / qb-dob 双缓的判负); 去掉 dS/P store 中性或更慢。**residual 里没有「可删的 store/barrier 开销」。**
4. **真 bound = occ-1 暴露的结构延迟(residual 53-61%) + 不能与第二常驻 WG 重叠的 QK/PV 计算**: skip_both(同时去 QK+PV)仍占 base 的 53-61%——这是 KV-gather HBM 延迟 + per-tile 循环调度 + softmax(QK→P→dS→PV)串行依赖链在 occ-1 下的暴露延迟, 无第二 wave 可掩盖。QK(37-44%)+PV-mfma(11-30%)是主计算但同样 occ-1 暴露。

**→ dQ LDS-datapath 微杠杆家族现已 measured 全封**: T7-XOR(r11 结构死) / D_LDS-pad(528 唯一最优, r11+r16) / ds_load_tr16_b128(gfx950 硬件不可选, r11) / **dual-layout-KV(r16 −7~−17%)**。dQ 唯一剩的真头 = **占用率**(occ-1→occ-2 掩盖 residual)——但 dq_acc(128)+q/do_packs(128)=256 VGPR 锁 occ-1: D-split-2WG 会复制 QK+softmax+gather(≈90% 工作)只砍 PV-mfma(15%)= 坏 trade; q/do 挪进 LDS 装不下(64 head × 512 × 2 tensor × 2B = 128KB > LDS 余量)。**下一杠杆应转 per-tile 结构 amortize**(TILE_K 16→32 摊薄 per-tile residual, 但须权衡 barrier-有益) 或 **cross-tile 软件流水**(tile t 的 PV 与 t+1 的 QK 重叠掩盖 occ-1 暴露延迟), 而非再碰 LDS datapath。

## ★ 每内核 flydsl-vs-triton 实测（seq=4096,6 shape,`_cmp_kernels.py`,决定往哪打）
- **dQ = flydsl 招牌,处处赢 triton 1.4–2.4×**:flash cr0/4/128=1.42/2.38/1.42×,pro=1.50/1.90/1.50×;TFLOPS 317–668(pro cr4 668),triton 只 223–352。**dQ 已最健康,别动**。
- **interm = 大 topk 赢/小 topk 输**:flash cr4 1.20×、pro cr4 **1.55×(rtr)**;但 **cr0/cr128 输 triton 0.76–0.84×**(triton 小 topk interm 299–302TF vs flydsl blk 209–254)。flash cr0/cr128 生产走融合绕过;**pro cr0/cr128 走 blk separate interm → 实打实慢 triton ~20%,当前唯一明确开口**。
- **gather = 两边都 ~2–3TF(memory-bound reduction)**,flydsl 略快 1.06–3.06×;flops 低是本质(带宽受限求和),非效率问题。别指望 gather 提 TFLOPS。

## ⚠️ 根因判据（关键，别再误判 roofline）
- **bwd 是占用率/流水/launch 受限,不是带宽 roofline**：pro cr4 bwd 269T@11.5ms/20GB = 仅 **24% 显存带宽**(MI355X ~8TB/s)，triton 也 24%。**有 3-4× 带宽余量**。当前流量打满≈1240T,极限≈3400T。→ 杠杆 = **提带宽利用率**(占用率/软件流水/融合减 launch)，不是消流量。
- fwd pro cr4 单融合 kernel 已 592T → 证明同硬件 bwd 融合到位能冲高。

## bwd 分解（pro cr4，隔离计时 `_cmp_kernels.py`；interm 已上 rtr）
dQ 2.78ms / interm 2.48ms(rtr,原 blk 3.8) / gather 0.91ms / delta+CSR-build 小。三者近 4:4:1.5。
- **interm**(`_bwd_interm_kernel.py` v3=build_interm_blk)：2D-blocked，Q/dO d-block 常驻 LDS + dS/P 按 rank-group staged，双 tr16 全合并。结构性 2-3× triton；**pro LDS 64KB→1 WG/CU** 是占用率墙。BD 已扫过最优(flash128/pro64)。
- **dQ**(`_bwd_dq_kernel.py`)：**172 VGPR → occ 受限**。single-pass identity(前置 delta)。降 VGPR 提 occ 是明确杠杆。
- **gather**(`_bwd_gather_kernel.py`)：已修负载均衡(NW=8 波分摊 CSR list + LDS 归约)，HCA cr128 从 59% 降下来。
- **delta**：O·dO，很小。

## ✅ 本轮已落地(2026-07-09 晚,commit 6f2ab8be→8b709bde)
- **融合 dQ+interm**(`_bwd_fused_kernel.py` build_bwd_fused,dsa_bwd PRIMUS_DSA_BWD_FUSED 默认开,门控 num_heads≤64 且 topk≤256)=**kv-block-batching occ-2 纯 bf16 跑赢 separate**,消 chunk_dS/P 4.8GB HBM。flash cr0/cr128 各 +5-6%。要点:Q/dO 全驻 LDS=135KB→occ-1 输;只驻 1 个 128 宽 d-block(~68KB→occ-2)+ 把 KB 个 kv-tile 批在一次 d-block staging(restage/barrier 从 4/tile 摊成 1/tile)。KB 选 4(NUM_TILES%4)。**只对小 topk 有效**(大 topk dS/P 装不下 LDS)。
- **register-transpose interm**(`build_interm_rtr`,门控 num_heads>64 且 topk≥512,即 pro cr4)=**ds_bpermute 手搓 16×16 bf16 转置**(替 LDS+ds_read_tr16→Q/dO 不占 LDS→pro 上 BD256,dS/P 只读 2× vs blk 4×)+ **d-tile-owning tiling**(每 wave 只驻自己 8 个 d-tile 的 Q=64VGPR→occ-2,转置一次摊全 rank-tile)+ **dS/P 寄存器预取**。**pro cr4 interm 1.55×,端到端 +11.8%**(402→450)。tr16 排布实测:lane(lo,grp)→M[h=grp*4+i,d=lo]=A[m=d,k=h];bpermute 复刻:输出 lane l elem i ← 源 lane 16*grp+4*i+lo_d4 的 bf16 elem lo_m4。POC=`_poc_bperm.py`。

## ❌ 已实测判负（别再试）
- **atomic-scatter 消 interm HBM**：慢 10×(cr4 pool-kv 数千 token atomic 竞争)。interm 物化+CSR gather 是避 atomic 的最优。
- **fp8/int8 interm**：定scale SNR 31<35 门(d-block-split 看不到 full-d);且用户硬禁量化。
- **fused+register-transpose**:fused 已 dq_acc 128+q/do_packs 128=256 VGPR,加转置寄存器→occ-1;**register-transpose 只能用于 standalone interm(无 dq_acc)**。
- **cr4 banded-local+pool-CSR split**:flash cr4 持平(CSR 640→512 省的被 banded+split+reduce 启动抵消)+ pro cr4 >4GB interm 残留映射 bug(12dB)。默认关 PRIMUS_DSA_GATHER_CR4SPLIT=0。
- **side-stream CSR 重叠**(把 host argsort 藏进 dQ/interm):rocprim sort 抢 SM,更慢(旧 r26 + 本轮复核)。CSR-build 占 flash cr4 11%/pro cr4 6%。
- **rtr 对 flash / pro 小R**:输 blk(flash blk BD512 已读 dS/P 一次无优势;小R 转置摊不开,pro cr128 0.949×/cr0 0.730×)。
- **interm 占用率-换-非合并 / v2v3 单-f32 store / 2字节 vec1 load**:不 lower 或更慢(见 flydsl 坑)。

## ★ K=32 MFMA(mfma_16x16x32)是 interm 的净赢杠杆(用户指点,实测)
- interm 头-contraction 原用 mfma_16x16x16bf16_1k(K=16)。换 **mfma_16x16x32_bf16(K=32)砍半 mfma 数**:rtr(pro)+2-4%、fused(flash cr0/cr128)+2-3%。**bwd_mean 250.4→275.2(+9.9%)**。commit c10f0d23(rtr)+ 966d48f9(fused)。
- **关键坑:K=32 的 v8 operand 别用「hold v4 + shuffle 成 v8」——会同时占 v4(a_q)+v8(aq8)寄存器翻倍(214→256vgpr)+ K=32 accumulator 进 AGPR(+110)→ 总 366 → occ-1 → 更慢**。正解:transpose 直接产 v8(`from_elements(8 i16).bitcast(bf16)`,不留 v4 中间体)→ 214vgpr、agpr=0(vgpr-form)、occ-2 保住。两个 K=16 tr16 v4 concat 成 v8 的布局是对的(SNR 56 验证),但要 direct-build 不要 hold-both。
- fused 里 operand 是 tr_h 流式(LDS ds_read_tr16,不 hold)→ 直接 concat 2 个 tr_h → v8,无寄存器爆。
- **判据**:K=32 只在 MfmaUtil 足够(interm rtr 47%)且能保 occ 时赢;dQ 是 latency-bound(MfmaUtil 35%/WAIT 2.91/occ-1)→ K=32-PV 边际(tr16-read 才是 bound)。blk(bank-conflict-bound)K=32 中性。
- **★ [r20-OPTIMIZE 决定性实测,关闭 K=32-PV 整条线]**:PV-形 matmul(收缩稀疏 kv 维)k=16→k=32 **不是净赢**——
  - **dQ PV k=32 = 净负(DEAD)**:k=32 需 32 kv/mfma,单 16-kv tile 只 k=16,`ds_load_tr16_b128` gfx950 无 → **必须缓存相邻 2 tile 拼 8-bf16 = 2-tile 批**。而 2-tile 批**打断 per-tile QK→softmax→PV 交织**——该交织是 occ-1 dQ 藏 occupied-latency 的唯一机制。4 次实测:M1(k32+2tile+KVBUF2+WAR)**−5.6% mean**;**机理探针(同 2-tile 批但 PV 仍拆 2 条 k16)−11% → 批结构本身是杀手,非 k 维**;KVBUF4 −8%;deferred-PV cross-tile WAR-race。**HALF_PV subtractive 探针(−8.7% dQ)是"mfma 凭空减半"的假想天花板,真 k=32 必付 2-tile-批 −11% 结构税 → 净负。**
  - **blk-interm k=32(flash cr4)= NEUTRAL(实测确证上面的"中性"断言)**:`_BLKSTACK` 迁 k=32 实测 flash cr4 340.2 vs 基线带 337-342 持平,SNR 51.4 PASS。**bank-conflict/LDS-read-latency-bound,half-mfma 无用。**
  - **教训**:**权威 HW roofline 头(gfx950 k=32 双倍 bf16 率)= "若能无损减半 mfma"的假想上限;落地时被数据布局(flydsl 转置读 tr16=4bf16/lane=k16)锁死到"必须 2-tile 批",批的结构税吃掉全部收益。roofline 头 ≠ 可实现头;subtractive/HALF_X 探针天花板是上界非可达值,必须 edit→bench 验结构代价。** rtr/fused-interm 的 k=32 之所以赢,是因它们是**纯头-contraction(无 softmax 交织),且 operand 是流式 tr_h 不 hold** → 无交织税、无寄存器爆。PV 两条都不满足。

## ★★ 2026-07-11 (campaign 154106, OPTIMIZE 执行 P3-b) dQ QK-链 ILP + 读预取 = occ-1×大-R 的真杠杆(修正「dQ micro-levers 全封」)
> 前几轮结论「dQ LDS-datapath 微杠杆全封, 只剩占用/cross-tile 流水」漏了一族: **QK-MFMA-链的 in-wave ILP + 读预取深度**。它们不动 LDS 布局/占用, 只在固定占用下填 occ-1 暴露的 latency 气泡。**只在 occ-1×大-R(pro cr4)有头, 小-R(cr0/cr128)被固定 Q-load/store 开销摊薄=within-session flat。**
- **✅ QK 每操作数累加器 2→8**(2×8=16 独立 MFMA 链藏 depth-KS 的 MFMA-RAW 延迟; 确定性左折归约 ~1ULP, SNR bit-stable/det=0): pro cr4/cr0 8 最优, **16 VGPR 溢出回退**, flash cr4 plateau@4(8=+1.2%)。
- **✅ KV-读(bvq A-操作数)预取深度 PF 6→8**(位精确): 8-acc 的 16 链更快耗操作数 → 深预取喂满 LDS 读管。**注意: 早先「PF 扫 3/6/9/12 无效」是在 ACC4(无累加器重构)下测的; ACC8 后最优点右移到 PF8**(pro cr4 -0.6%/cr0 -1.4%, PF12 too deep, flash cr4 仍 PF6)。→ **教训: 预取深度的最优点依赖累加器数(ILP↑→耗操作数快→需更深预取); 单独扫 PF 会漏。**
- **✅ 新杠杆 PV 读提到 softmax 前(PVPF 0→8, 位精确)**: PV 的 `ds_read_tr16`(bvv, A-操作数)只依赖稳定 LDS-KV, **不依赖 softmax** → 把前 8 条 bvv 读**提到 softmax exp2 之前**, 藏其 LDS 读延迟于 occ-1 的 QK→pB softmax 气泡。pro cr4 -1.1%/cr0 -1.6%/cr128 -1.2%, flash cr4 flat。**与已判负的 cross-tile 软件流水(XPIPE)正交**(那是跨 tile carry S, LDS 端口饱和死; 这是 intra-tile 把稳定读提前, 不加 LDS 端口压力)。
- **三合一门控 pro cr4 单组(_pro_big=num_heads>64 AND topk_len>256), bracketed within-session iso: pro cr4 dQ min 2767.9→2705.7 = -2.3%(spill-free); scored matched-window 平局(pro cr4 ~6% DVFS > ~1% 信号, flash cr0 锚匹配)。build 参 `qk_acc`/`pf`/`pv_pf` 保留 override。**
- **★ rocprof-compute SoL(pro cr4 dQ=k_fn_0, 权威 bound 定性)**: 占用 **4.5%**(368/8192 波)、"Insufficient SIMD VGPRs" **92%**、VGPR **186-204**、MFMA-util **9%**、VALU 20%、VMEM **0.83%**、IPC 0.47/5、LDS-bank-conflict 1.6%(无)。**= VGPR-占用-bound → 深度 latency-bound; 一切远低 roofline → ILP/预取(填管)是固定占用下唯一杠杆(与 -2.3% 自洽)。** 提占用需 split-K/换瓦片(出 QK-链 scope)。

## ⚠️ 已探索杠杆 + 当前差距(perf 实测;继续找,别下"到头"结论)
- **dQ 小topk(1.42-1.51×)**:occ-2 被 dq_acc(128)+Q/dO(128)=256 VGPR 锁死。**hold_qo=False(每tile重载Q/dO省寄存器换occ)实测灾难: 小topk 0.45-0.64×, pro cr4 6×回退(15977 vs 2720us)**(重载 16×vec8 的延迟 >> 占用增益, 虽 VMEM 99% 空闲)→ 占用无法在核内廉价提升。dq_acc 128(32 d-tile 输出)是硬底。固定开销(Q载入+32存储)在 8-12 tile 摊不开→大topk 662TF(2.36×)/小topk 326TF。**(PF 单独扫无效的旧结论已被上节 ACC8+PF8 修正: PF 最优点依赖累加器数。)**
- **interm 小topk(cr0 0.91×)**:triton tl.dot 少 rank-group 开销极低,结构性快。
- **interm 大topk(pro cr4 1.63×)**:转置 8 次 bpermute 不可约(SIMT 两 dword 都取),VALU 1.57:1 封顶。
- **减 barrier**:interm rtr dS/P 双缓冲去 WAR barrier=pro cr0 +2%(小);staging barrier 载重不可去;dQ 1 barrier/tile(双缓冲已最小)。
- **ASM(inline_asm 可用,_llvm.inline_asm(res,[ops],asm,constraints,side_effects);grouped_gemm 有模板)**:转置只能省 select→v_perm_b32(~8op/operand,interm ~5-10%),bpermute 不可约→够不到 2×。要 2× 需 gfx950 没有的:高效寄存器转置(triton tl.dot 内部/global_load_tr,CDNA4 无)或破 occ-2 VGPR 锁。

## 未试/开口（优先做）
1. **pro cr0/cr128 的 interm 慢 triton ~20%**(每内核对比确证:0.80–0.84×)=当前唯一明确开口。triton 小 topk interm 更快(299–302 vs 209–254TF)→ 研究 triton 小 topk interm 打法(可能少 launch/更好 tiling),移植或专调 blk 小 topk。
2. **banded-SWA dKV**(cr0/cr128 local):kv 被连续 token 窗口 attend → 融成 banded kernel 无 gather/无 interm HBM。gather 已用 banded(closed-form)但 interm 未融。
3. fwd 未系统性碰(本轮全在 bwd)。
- **天花板认知**:dsv4 ~2.3× dense 是 sparse 结构税;600 需近-dense 效率,当前算法+bf16 够不着。真要破需减 interm+gather 的 2× tensor HBM(唯一没被堵死的是 hybrid scatter:低争用 local 走 atomic/高争用 pool 走物化,未验证)。
- **[r20-OPTIMIZE 平台确认]**:mean ~307/6组,dQ+interm=每组 wall 82%。dQ=MFMA-吞吐-bound 且唯一大头 PV 的 k=32 已证结构死(见 §K=32 决定性实测);interm rtr/blk latency-bound-on-LDS-tr16(occ 硬底)全 measured 穷尽;gather 双路(banded 51%BW/CSR 83-85%BW)本轮 nw-sweep+2-way-unroll 全中性 → **banded 的 51% 是短归约随机访问的实际带宽底,非可回收头;nw(占用,LDS∝nw→waves/CU 常数)与 unroll(MLP,编译器已排够 loads)两正交轴都无效**。**结论:纯 flydsl+bf16(禁量化)硬约束下 6 组已达结构性平台;进一步需换算法维度(hybrid scatter 未验)或接受此约束下的上限。** 通用教训:占用/MLP 旋钮对已接近实际带宽的短归约 gather 无效——先判是否真 latency-bound(未饱和 BW-eff 可能是访问粒度上限而非 2× 头)。

## flydsl kernel 专属坑（本 kernel 族实测，配合 pitfalls/07 + memory reference_flydsl_loop_pitfalls）
- **2 字节 buffer_store 不 lower**(单 f32/单 i16 → `ScalarizeVectorOperand`)：store 必须向量(≥2 元素/≥4B);f32→bf16 用 `Vec.from_elements([BFloat16(x)..],bf16).bitcast(Int32)`(vec2 store)。
- **`cvt_pk_bf16_f32` 打包 store 错位/出垃圾**：改 from_elements(bf16)。
- **bf16 vec1 load 出 NaN**;要 gather 一个 strided operand 用 **i16 vec1 load 装配**(正确)。
- **contract-over-head GEMM**(interm)：两个操作数都 h-strided → 都走 LDS 自然 stage + `ds_read_tr16_b64` 转置读(**tr16 可同时作 A 和 B operand**,实测)。
- **iter_args 累加 loop**：单元素 yield 拆包(≥2 iter_args)、buffer_load raw 作 init 致 trip×2(用 Vec.filled)、字面 if/for 被转 scf(分支放外层 python)。
- 调试 GEMM 三段隔离：①store 常量 ②常量 operand 喂 mfma ③真实 load。

来源：会话 a17bf3b2 (2026-07-09) 全程实测；commit 见 dev/kyle/flydsl_attn_deepseekv4；细节 memory project_dsv4_bwd_fusion_win / project_dsv4_interm_register_transpose / project_dsv4_turbo_verify_ref / reference_flydsl_loop_pitfalls。harness:`_cmp_kernels.py`(每内核 fly-vs-tri)、`_bench_fused.py`、`_bench_rtr.py`、`_poc_bperm.py`。远端 chi2762 GPU6 mlperf_gptoss,同步 sync/flydsl_optimizer/flydsl_campaigns/20260709_144443/campaign_remote.py remote "&lt;cmd&gt;"。

## ❌ fwd s_setprio 判负(2026-07-11,matched A/B/0-1-0)
- fwd 从未用 s_setprio;试在 QK MFMA 簇(qk2 路径)加 s_setprio(1)/(0),env PRIMUS_DSA_FWD_PRIO。**matched A/B(0/1/0)= mean 425.9→414.8(−11T),弱组全退**(flash cr0 −7%、flash cr128 −8%、pro cr128 −5%)。
- 根因(profile flash cr0 fwd:MfmaUtil 21%<VALU 25%、occ 6.99=occ-2、MemStall 0.72、WAIT 19%):fwd 弱组是 **VALU(softmax exp2/permlane)+ PV ds_read-latency bound,非 MFMA-issue-bound** → 优先 MFMA 反饿死 VALU/read。与 bwd「s_setprio 助 occ-2 MFMA-bound interm、伤 latency-bound dQ」一致。**s_setprio 在 fwd 全 DEAD。**
- 现状:fwd 14 个 per-group 算子参数(pf_pv/pf_qk/xcd/qk2/peel/vec2o/hoist_sink/fused_store/pv_early/single_buffer/banded/pool/dyn_pool)已穷尽逐组门控;campaign(20260710_151850,30 轮)mean 398→430(+8.1%,commit b537057d)后进入平台(多轮无新高+3 replan)。弱组(flash cr0 290/pro cr0 359/cr128 380/388)= occ-2、o_acc(128VGPR)锁死、latency-bound(QK→softmax→PV 串行链无第二 WG 掩盖),与 bwd dQ 同墙。进一步需算法维度(hybrid scatter 未验)非算子调参。

## ★ fwd pro cr4 = LDS-port-bound 平台(2026-07-11,目标 800 攻坚,决定性)
- profile pro cr4 fwd:**MfmaUtil 36.6% ≈ VALU 35.3%(共限)、occ 7.63(occ-2)、LDSBankConflict 18.3%、LDS-wait 16%、LDS_IDX_ACTIVE 26%、INSTS_VALU/LDS=3.3**。618TF=47%峰(MfmaUtil 36%→有 MFMA 余量但填不满)。
- **算子参数全穷尽(matched A/B,isolated pro cr4 计时)**:D_LDS 扫 {512..576} 528 最优(其余更差,tr16 需 528 %32=8);pf_pv/pf_qk/qk2/single_buffer(SB=0)全中性;s_setprio 判负(见上)。
- **cross-tile 软件流水(XPIPE,carry S_{t+1},PV_t 与 QK_{t+1} 交织)= 实现正确(SNR 全过)但中性**(pro cr4 608.8 vs 618)。根因:overlap 后 PV_t 的 tr16 读 + QK_{t+1} 的 v8 读同时砸 LDS 端口(已 26% busy+18% 冲突)→ **LDS 端口饱和成墙**,填 MFMA 气泡无用。
- **根因定论:pro cr4 = LDS-port-bound**。KV 读两次(QK 走 v8-natural[kv,d]、PV 走 tr16-transpose[d,kv],同数据两布局)。QK 必须 natural(收缩 d)、PV 必须 transpose → 共享 KV-LDS 无法同时无冲突:pad(opposed,528 治 tr16/QK 2-way 死)、XOR(破 tr16 硬件转置读)、dual-KV(加 LDS store 反压 already-bound 端口)、register-transpose-PV(移冲突到 VALU,但 VALU 也 35% 共限→中性,且 bpermute +320%)全死。**VALU/MFMA/LDS 三者共限 = 该算法在 gfx950 的结构最优 ~618TF**。
- **→ 800(+31%)非算子级可达;唯一剩路 = 算法级重构 KV-LDS 结构**(一种同时服务 QK-natural + PV-transpose 无冲突的布局/tile 形,pad/XOR 均未达;或预转置 V 免 tr16 但 QK 仍冲突)。research-grade,未验证。

## fwd cross-tile 软件流水(XPIPE)全判负(2026-07-11)+ 下一杠杆 TILE_K=32
- XPIPE(carry S_{t+1},iter t 内 PV_t‖QK_{t+1}):cr4(non-banded,many-tile)= 正确但**中性**(608 vs 618,LDS 端口饱和);cr0(banded,few-tile)= 正确但**判负**(flash 290→273/pro 359→334,8-tile 的 prologue/epilogue+寄存器压力 > 重叠收益)。**cross-tile 流水全 shape 死。**
- fwd 全组 = 均衡结构平台(VALU≈MFMA≈35%,LDS-port cr4 饱和):所有算子参数(D_LDS/pf/qk2/s_setprio/SB)+ XPIPE 已穷尽。
- **下一个未试的结构杠杆 = TILE_K 16→32**:每 tile 32 kv → tile 数减半 → **softmax online-update(crossgrp permlane max/sum,VALU 共瓶颈的主源)减半** + barrier 减半;QK/PV MFMA 总数不变。直击 VALU 共限,理论惠及全组(尤其 VALU-heavy 的 cr4/cr0)。代价=大改写(S[4]→S[8]、per-tile 结构、LDS tile 翻倍),高风险,flydsl 里 TILE_K 深度耦合(grp*4+i / mask / PV loop)。待验证。

## ★ fwd PV 用 K=16(未用 16x16x32)= 真开口(2026-07-11,用户指点 + PVHALF 探针证实)
- fwd QK 用 mfma_16x16x32(K=32,收缩 d),但 **PV 用 mfma_16x16x16(K=16,收缩 kv)**,因 TILE_K=16(每 tile 16 kv)。PV MFMA 数=DT×NUM_TILES(cr4=32×72=2304)是 QK(1152)的 2×。
- **PVHALF subtractive 探针(跳一半 PV MFMA,破坏正确性只测时间):pro cr4 617→694 TF(+12.5%)** → **PV MFMA 在 pro cr4 关键路径上,非纯 LDS-port-bound**(推翻早先预测)。→ PV-K=32(同工作、指令减半、LDS 读不变)有 ~+12% 潜力。
- 实现路=**2-tile-batch PV-K=32**(保 TILE_K=16 的 QK/softmax/store per-tile;每 2 tile 攒 V_a/V_b + p_a/p_b,online-softmax alpha-fold:o=o*(αa*αb)+[p_a*αb|p_b]@[V_a|V_b],concat 2 tr16→v8 A、concat p→v8 B,mfma_16x16x32)。与 bwd 判负的"2-tile-batch"不同:那是 dQ occ-1 且破坏 QK→sm→PV 交织;fwd 这里 QK/sm 仍 per-tile,只 PV 攒对,交织不破。NUM_TILES 全组偶数(cr4=72/cr0=8/cr128=10)。env PRIMUS_DSA_FWD_PVK32,实现中。

## ★★ fwd PV-K32 已落地(2026-07-11,pro cr4 616→700 +14%,commit 251aa625+31bfe733)
- **PV 从 K=16→K=32**:2-tile-batch(每 pair 攒 KV_a/KV_b,concat 2 tr16→v8 A operand,concat p→v8 B),ONE combined softmax over 32 kv(1 crossgrp permlane vs 2,both p 同 m_new 无 alpha-fold,deferred kv-sum),mfma_16x16x32。QK/store 仍 per-16(交织不破,与 bwd 2-tile-batch 判负不同)。
- **pv_early 对 pvk32 是关键**:PV-K32 后 stall-bound(WAIT 42%),把 PF_PV=2 个 V32 tr16 读提到 softmax32 前(读不依赖 p)→ LDS 读延迟藏进 softmax VALU bridge → WAIT 42%→36%,pro cr4 661→707。**PF_PV 必须=2(4/6 因 held V32 寄存器爆 occ 崩:467/536 TF)。**
- 逐步:base 616 → +PV-K32/softmax32 = 661(+7.3%) → +pv_early = 707(+15%);flash cr4 551→609(+11%);mean 430→452.7。
- **只利 many-tile(cr4)**:banded cr0(8 tile)扩 PV-K32 判负(flash cr0 283→263/pro cr0 351→291,few-tile pair-batch 开销 > 收益,同 XPIPE-cr0)。weak few-tile 组(cr0/cr128)不能用 PV-K32。
- pro cr4 700 后 profile:MFMA 28%/VALU 25%/LDSbank 21%/LDSwait 20%/WAIT 36%/occ 7.62,又回均衡;occ 非 VGPR 限(VGPR 128 allows 16,achieved 7.62=barrier 串行实际值,waves_per_eu 强制中性)。QK v8 bank 冲突(21%)仍 XOR-不可解。gfx950 AGPR 是分离预算(非共享512)但 fwd occ 非 VGPR-限,AGPR-accum 无用。

## fwd 弱组(cr0/cr128)+ pro cr4 700 后杠杆全穷尽(2026-07-11)
- pro cr0 profile:VALU 26.7%>MFMA 22.6%、WAIT 20% 低、LDS 低 = 均衡/VALU(exp2)-bound。**SKIPCG 探针(跳 crossgrp permlane)中性**(pro cr0 383.6→383.9/全弱组)→ crossgrp 非瓶颈(编译器已藏),softmax-combine 对弱组无用。弱组 VALU=exp2 transcendental(不可约)+ 均衡,无单一杠杆;pv_early/pf 中性;PV-K32/pair-loop few-tile 判负。**弱组在 tuned 结构位。**
- pro cr4 700 后:qk2 中性(2 独立 QK 链已 ILP)、waves_per_eu 中性、barrier 减少中性。均衡(MFMA28/VALU25/LDSbank21/LDSwait20)。QK v8 bank(21%)XOR-不可解。**pro cr4 ~707 结构位。**
- **结论:fwd 全组算子级杠杆穷尽。fwd mean 452.7 / pro cr4 707。到 800/500 需算法级重设计**(治 QK-v8-bank 的 block-permuted KV-LDS 布局同时保 tr16;或减 exp2 的注意力公式)——research-grade,非微杠杆。

## ★ 决定性:fwd pro cr4 707 = 结构位,QK bank 冲突非杠杆(2026-07-11 QKNOCONF 探针)
- **QKNOCONF 探针**(_bv 把 row 从 lo 换常数 0=广播读,消 bank 冲突,数据错只测时间):pro cr4 708.7→713.8 **仅 +0.7%** → **21% LDSBankConflict 不在关键路径(occ-2 已掩盖)**。→ block-permuted KV-layout 重写(治 bank 冲突)只值 +0.7%,**不值得**,勿投入。
- subtractive 探针族总结(定 pro cr4 杠杆):PVHALF +12%(→ 落地 PV-K32 +14% 真赢)、QKNOCONF +0.7%(bank 非杠杆)、SKIPCG 中性(crossgrp 非杠杆)。qk2/waves_per_eu/barrier/D_LDS/pf 全中性。
- **pro cr4 707(54% 峰)= 该 sparse-MLA 算法在 gfx950 的高效执行位**:QK(K=512 收缩)+PV(K=32,已优化)+softmax(exp2 不可约)+tr16 读(64-bit HW 限)全是必要工作,均衡无单一可削杠杆。到 800(61% 峰)需减 FLOP(改算法)或改 HW 映射,非 kernel 微调可达。

## fwd 突破:BLOCK_H=128 消除 pro KV-store 冗余(2026-07-09)

**背景**:pro cr4 卡在 707TF。诊断链(subtractive probe,pro cr4 isolated `_time_fwd_one.py`):
- **SKIPST**(跳过 pvk32 loop 的 2 个 `gather_store` KV→LDS 写):708 → **1041 TF(+47%)**。→ KV LDS store 是主瓶颈,占 pro cr4 运行时 32%(此前误判为"overlapped/cheap")。
- **CFST**(store 改写成 conflict-free 连续布局 `tid*32`,读端不变→数据错、仅计时):710(**中性**)→ store 成本**不是 bank 冲突**。
- **SKIPBAR**(留 store、去掉 store→QK-read 之间的 2 个 barrier):720(+1.7%,小)→ **不是 barrier 停顿**。
- 结论:成本 = **LDS store 写指令吞吐本身**(v8=dwordx4 已最大宽度,无法再加宽)。

**根因**:pro(H=128)用 BLOCK_H=64 → 每 token 2 个 WG,各自**冗余** gather+store 同一份共享 MLA latent(MQA 单 K==V)到各自 LDS。

**解法(已落地,默认开)**:`BLOCK_H=128`(1 WG/token,8 wave,512 线程)。1 个 WG 存一次 KV → 消除冗余 store ≈ 砍总工作量 23%。同时 512 线程分摊同一 16KB store tile(每线程 v8 数 4→2)。
- **pro cr4:707 → 784(bench)/ 827(isolated),SNR 62.5dB PASS。MEAN 452.7 → 471.7。**
- 实现:`build_fwd(..., bh=64)` 把 BLOCK_H/WAVES/THREADS 从模块级移进函数;派生 `CHUNKS=THREADS//TILE_K / ECHUNK=D//CHUNKS / V8PT=ECHUNK//8`;gather_load/store 与 pvk32/非pvk32 的 kv carry、idx 偏移全部用 V8PT 泛化(bh=64 时 V8PT=4 完全等价)。dispatch:`_bh_default="128" if (not banded and num_heads>=128 and num_heads%128==0)`,env `PRIMUS_DSA_FWD_BH`。
- 仅影响 pro cr4(非 banded H>=128);flash(H=64)、pro cr0/cr128(banded)仍 bh=64 不变。
- **教训**:MQA/MLA 共享 latent 下,BLOCK_H 越小 → 每 token 越多 WG 冗余存同一 KV。store 冗余按 (H/BLOCK_H) 倍放大。以后遇到 store-bound + 共享 KV,先查 WG/token 冗余度。

## 判负:triple-buffer 隐藏 exposed store(2026-07-09)

想法:pvk32 pair-loop 里 `store kv_a→buf0` 暴露(bar1/bar2 之间无 compute 重叠),用 3 buffer ring 把每对 tile-a **提前一对 prestore**(与前一对 PV MFMA 重叠,ds_write 与 v_mfma 不同 pipe 可 co-issue)。实现:`PRIMUS_DSA_FWD_TRIBUF`,NBUF=3,slot=(2pr+k)%3 运行时旋转,prologue prestore tile0,carry 只带 4 个 index(不带 kv 寄存器)。
**结果:正确(SNR 逐位一致 62.5dB)但更慢**——flash cr4 599→499、pro cr4 794→768、mean 469→448。根因=第 3 个 KV buffer 把 LDS 34→50KB,occupancy 下降;本 kernel occupancy-bound,占用率损失 > store-hide 收益。flash(bh64)损失更大(-17%)因 LDS 更敏感。
**结论:LDS-store 成本(SKIPST +47% flash/+15% pro)不是靠 pipelining 能回收的——它是 occupancy 限制的地板。** 要动只能降 store **volume**(register-transpose 免 LDS,昂贵有风险)或 store 线程数(bh=128,已对 pro cr4 用尽;flash H=64 无法加)。env-gate 默认关,留作判负记录。

## fwd occupancy/banded 地板实证(2026-07-09,rocprofv3 + subtractive)

**pro cr4 @ bh=128 profile(k_fn_0)**:occ=7.5 waves/CU=**1 block(LDS-limited)**;LDS_Block_Size=34304B(2 block=68.6KB>64KB → 卡 1);VGPR=124(有余量,2 block VGPR 够=496<512,但被 LDS 卡);`SQ_WAIT_INST_LDS`(186M)>`SQ_BUSY_CYCLES`(107M)→ **LDS-wait 停顿主导**。occ-2 能隐藏它但被 LDS 死锁。
- **D_LDS=512 判负(灾难)**:为省 1024B/pair 冲 occ-2,去掉 tr16 pad → pro cr4 827→**279TF(-66%)**。tr16 PV-read 4-way bank 冲突代价远超 occ 收益(≠ QK-read 冲突的 +0.7%,PV tr16 冲突是灾难级)。**D_LDS=528 强制,不可动。**
- **[d,kv] 布局判负(纸面)**:想让 PV 读连续免 tr16 → 但 QK 需 tr16,且 [d,kv] 有 512 行(每行要 pad)vs [kv,d] 16 行 → pad 开销 ×32 更大。**[kv,d]+D_LDS=528 已是 LDS 最省布局。** → occ-1 是硬地板,同时锁死 pro cr4(LDS-wait)和 flash cr4(store 曝露,只 4 wave 无 sibling 藏 store)。
- prefetch 深度 pf_pv/pf_qk:默认 2/3 已最优(4/8 起 spill 掉速)。

**banded(flash/pro cr0/cr128)= VALU-bound,非 occ/LDS**:pro cr128 profile VALU 72.6M vs LDS 16.7M(4.3×)。
- **single_buffer(SB=1,占用率↑)判负**:中性或更差(flash cr128 418→313,extra WAR barrier 伤 intra-block overlap)→ banded 非 occ-limited。
- **SKIPCG(跳 crossgrp permlane)中性**(全 4 banded)→ crossgrp permlane 已被隐藏,免费;2-tile softmax-batch 省 crossgrp 无意义(未实现)。
- banded VALU = exp2(5/tile)+ o_acc rescale(alpha×32 d-tile/tile)+ MFMA operand packing = softmax/PV 算法本体,不可削(HW v_exp_f32 已最快)。few-tile + 曝露 epilogue(peel/hoist_sink/pv_early 已用)。

**session 净结果(scoring harness campaign_fwd)**:pro cr4 610→794-812(跨 800,+30%);MEAN 452→470-474。warm(30 warmup+best-of-7)mean 508/pro cr4 827。scoring 的 mean 差主要是小 banded shape(<0.4ms)DVFS 冷间节流(8-warmup 跑不满时钟)+ banded VALU 地板。

## flash cr4 store-floor:DMA / swizzle / per-tile 全判负(2026-07-09)

flash cr4(H=64,bh 不可,occ~2 已满)= store 47%(SKIPST)+ LDS-wait 3×busy(profile)。目标 700,warm 地板 ~630。穷尽:
- **XOR swizzle(消 tr16 pad → stride-512 → occ)判负**:MLA 单 latent K==V,一个 swizzle 无法同时让 QK 连续 v8 读 + PV 转置 tr16 读都免冲突(SLA 能是因 K≠V 分开 LDS+分开 swizzle)。SWMASK 1/3/7/15 全部 630→326-419。D_LDS=528 pad 是 MLA 唯一正确布局。
- **DMA(raw_ptr_buffer_load_lds 直接 HBM→LDS,消 store)判负**:功能正确(SNR 60.8/62.5 PASS,已验证 API:`from flydsl.expr.typing import T as _FTY` 避开 kernel 参数 T 遮蔽;readfirstlane(_FTY.i64)+IntToPtrOp(!llvm.ptr<3>)+raw_ptr_buffer_load_lds;行连续 DMA 保 pad→tr16 不变)。但**任何结构都比 baseline 慢**:①非流水 pvk32(occ-2,s_waitcnt(0) 露 HBM 延迟)flash 510/pro 703;②per-tile K16 双缓冲(SLA 模式,2 buffer 流水藏延迟)flash 458/pro 565。原因:pvk32 流水需 4 buffer(67.6KB>gfx950 ~64KB LDS 不入);per-tile 丢 K=32 且 DMA 自身开销(buffer_load_lds 发行+readfirstlane+s_waitcnt 直列)≥ 省下的 store。SKIPST 的 47% 含 store→barrier→read 直列化,DMA 不能回避它,只换了形式。
- gfx950 LDS/CU ≈ 64-68KB(实测:34KB→occ~1.85 block;50KB→1 block;67.6KB 不入)。
- 保留 env-gated 关(PRIMUS_DSA_FWD_DMA/SWIZ/TRIBUF 默认 0)作已验证参考。**flash cr4 = store/LDS-wait 本质地板 ~630 warm。**

## flash cr4 store 是「必要的预取机制」非浪费(2026-07-09 补充,DMA 深挖根因)
DMA per-tile profile 揭示:LDS-wait 被 DMA 杀掉(125M→13.2M!)但 **VALU 76M→130M 爆炸**——根因是 per-tile K=16 **丢了 pvk32 的 2-tile softmax 批处理**,softmax VALU(exp2/crossgrp)翻倍,非 DMA 地址开销(把 uniform 的 src*DQK*2 移到 soffset SGPR 也无改善)。
- 想要的组合 pvk32(低 VALU)+ DMA(无 store)需 4-buffer 流水隐藏 HBM 延迟 → 67.6KB>gfx950 LDS 不入;非流水 pvk32-DMA 的 `s_waitcnt(0)` 是**硬 per-wave 停顿**,sibling block 无法隐藏。
- **而 baseline(gather_load 寄存器预取 + gather_store)之所以好**:register prefetch 廉价隐藏 ~300cyc HBM 延迟(提前 1 tile),再快速 reg→LDS store。SKIPST 的 47% 是这个「预取+store」机制的成本,DMA 用「LDS 直载但暴露 HBM 延迟」换掉反更差。**store 是隐藏 HBM 延迟的必要代价,不是可省的浪费。**
- 结论:flash cr4 ~630 warm 是架构地板(store 必要 + occ 被 tr16-pad 卡 + DMA 换不动)。DMA env-gated 关,留作已验证参考(功能正确 SNR PASS,但本 shape 不划算)。

## DMA 三形态全判负 + gfx950 LDS 实测(2026-07-09 定案)
gfx950 LDS/CU ~160KB(实测:4-buffer 68KB 不掉 occ,仍 7.4 waves)。DMA 实现三形态,全比 register baseline 慢:
- 非流水 pvk32 DMA:flash 510/pro 703(s_waitcnt(0) 露 HBM 延迟)。
- per-tile K16 DMA 双缓冲:flash 452/pro 559(LDS-wait 125→13M 但丢 pvk32 批处理→VALU 76→130M)。
- **4-buffer pvk32 DMA 流水**(prefetch 1 pair ahead,pvk32 低VALU+no store):flash 498/pro 690。profile:LDS-wait 125→32.5M、VALU 76→94M、BUSY 40.8→51.6M、SQ_WAIT_ANY 427M。仍 <630。
- **定案根因**:register-gather baseline 用**精密 vmcnt**(仅特定寄存器被用时 stall)+ 良好 coalescing,对 scattered topk gather 本质最优;DMA 的 s_waitcnt 粗同步 + 地址计算 VALU + 高 BUSY 覆盖不掉,即使 LDS 充足、latency 部分隐藏、VALU 用 pvk32 压低。**此 MLA scattered-gather 工作负载 register 经路最优,store(47%)是「寄存器预取隐藏 HBM 延迟」的必要代价,非可省浪费。**
- DMA/swiz/tribuf 全 env-gated 关(默认 0)+ 文档化。生产 = bh=128 baseline。flash cr4 ~630 warm / cr128 ~380-400 是架构地板(gfx950 LDS 布局 + softmax VALU 双约束)。

## triton 参考基准校准(2026-07-09)—— flydsl 已比参考快 1.6×
用 `primus_turbo.triton.attention.deepseek.sparse_mla_v2.dsa_fwd.sparse_mla_fwd_v4_triton`(同签名)同 _build 同 warm 计测:
| shape | triton | flydsl(本工作) | 倍数 | 目标 |
|---|---|---|---|---|
| flash cr4 | 390 | 630 | 1.62× | 700 |
| flash cr128 | 254 | 417 | 1.64× | 550 |
| pro cr128 | 258 | 428 | 1.66× | 550 |
| (pro cr4) | — | 815-827 | — | 800 ✅ |
**结论**:flydsl 内核已是 SOTA,全 shape 比 triton 参考快 ~1.6×。目标 700/550 ≈ triton 的 1.8×、比 flydsl 当前(已最优)还要再高 1.1-1.3×。经全部 kernel 级杠杆实测(bh=128 拿下 pro cr4;DMA×3/swizzle/tribuf/occ hint/D-chunk 分析对 flash cr4+cr128 全判负),这个额外 gap 在 gfx950+此 MLA 结构下达不到。pro cr4 目标(800)已达成。剩余目标高于参考实现,需算法层突破或调低目标。

## 决定性:occ=2 是 VGPR 平衡最优点(2026-07-09,强制 waves_per_eu 证明)
所有 shape latency-bound(MfmaUtil ~24%),occ~2。想抬 occ 藏延迟:
- compile_hint waves_per_eu/maxnreg 走 external binary codegen(`_use_external_binary_codegen`),此环境 **FLYDSL_COMPILE_LLVM_DIR 未设 + /opt/rocm/llvm 无 mlir-opt → external 不可用 → hint 全被 internal MLIR codegen 忽略**(解释了之前 wpe/mnr/mfma-vgpr-form flag 全无效)。
- **改用 kernel `__call__` 的 `value_attrs={"rocdl.waves_per_eu": N}` 直接注入 func 属性(internal codegen 尊重)**:实测生效但**灾难**——WPEATTR=3→630→140、=4→107。因为强制更多 wave 需把 VGPR 压到 ≤64/85 → 大量 spill 到 scratch。
- **结论:occ=2 是 VGPR=128 的无 spill 最优;抬 occ 必 spill(灾难),降 occ(更多 VGPR/更深 prefetch,见 DEEP 判负)不帮。** VGPR 128 = Q(64,每 lane 头的 512d,QK B 操作数,必须常驻)+ temps;o_acc 在 AGPR(D-chunk halve 它 VGPR 只 128→124、occ 不变=证实 o_acc 非 VGPR 瓶颈)。→ occ 无法在 kernel 源码层提高。
- MfmaUtil 24% 的"算力余量"无法利用:填满(更多 wave)就 spill。这是 attention 固有的寄存器驻留(Q + O-accumulator for D=512)与 occ 的张力。
- 全 shape 已 1.6× 于 triton 参考(flash cr4 630 vs 390 / flash cr128 417 vs 254 / pro cr128 428 vs 258)。next-step 需更低寄存器足迹的算法(与 attention 的 Q+O 驻留本质冲突)或 external codegen 工具链。

## occ 地板再证:o_acc 不是 VGPR 瓶颈,Q+temps 才是(2026-07-11)
- pro cr128 banded profile:VGPR 128 / **AGPR 0**(o_acc 在 VGPR)/ occ 7.27。
- pvk32 D-chunk(o_acc 32→16 d-tile 减半)实测 VGPR 只 128→124(省 4)、occ 不变——**证明 o_acc 减半省不了 VGPR,它不是占用率瓶颈**。VGPR 128 = Q(64,每 lane 头的 512d,QK B 操作数必须常驻)+ temps(~64,编译器管)。
- 要 occ 3 需 VGPR≤85(减 43):Q 砍不动(QK 需要;放 LDS→64KB LDS 反而 cap occ);temps 编译器管。→ **occ=2 是 Q+temps 决定的硬地板,D-chunk/降 o_acc 无效,强制 occ(WPEATTR)必 spill。三重独立证明 occ 不可提高。**
- 由此 latency-bound 的 MFMA 24% 余量无法用占用率填。当前手动 kernel 源码杠杆穷尽;后续交 campaign 多-agent 系统搜索(基线 bh=128 commit 7c57e1a2,已超 triton 参考 1.6×)。

## banded DMA 判负 + store 是必要的 HBM 隐藏机制(2026-07-11,实现+profile 对比)
SKIPST(去 store,读旧数据)对 cr128:pro 432→564、flash 418→545(+30%)——看似 store 是 30% 可省。但**实现 banded 直接 HBM→LDS DMA(连续 KV coalesced,`dma_load_tile_b` + 2-buffer 流水,PRIMUS_DSA_FWD_DMAB)后 SNR 全 PASS 但更慢**:cr128 pro 428→353、flash 417→324。
profile 铁证(pro cr128):baseline-store SQ_WAIT_ANY **82.8M** vs DMA **206M**(2.5× 更多 wait)。DMA 的 LDS-wait 消了(32→8M,store 没了)但 HBM-wait 经 s_waitcnt 暴露=206M。
**根因:store 路径的 register-prefetch(gather_load_b 提前 1 tile 到寄存器)廉价隐藏 HBM 延迟,再快速 reg→LDS store(30%);DMA 直载 LDS 无法廉价深预取(需多 buffer+精确 vmcnt),HBM 延迟暴露。SKIPST 的 30% 是这个"预取+store"机制的代价,不是浪费——去掉它(DMA)反把 HBM 延迟暴露 2.5×。** 对 cr4(散乱)和 cr128(连续)结论一致:store+register-prefetch 是最优 HBM 隐藏,DMA 更差。DMAB env 默认关,留验证过的判负参考。
→ cr128 428 / cr4 630 是 store-necessary + occ-2 + latency-bound(MfmaUtil 24%)的地板,1.6× 于 triton 参考。

## 未判负的候选杠杆(2026-07-11,给 campaign agent 试):register-resident KV + bpermute 软件转置
现状:全 shape latency-bound(MfmaUtil 24% / VALUBusy 23%,算力+VALU 都空闲),瓶颈=LDS store(cr4 47%/cr128 30%)+ LDS 读(QK v8 + PV tr16)的延迟,occ=2 藏不住(occ 已证明不可提高)。store 是 register-prefetch 隐藏 HBM 的必要代价(DMA 判负,HBM-wait 暴露 206M vs store 83M)。
**未试的正解思路**:把 gather 到的 KV 留在寄存器,**不写 LDS**;QK 和 PV 都用 `ds_bpermute`(wave 内 64-lane 跨 lane 洗牌)做 gather-layout→MFMA-layout 的转置,而不是走 LDS store+tr16/v8 读。这样一次性消除 store(47/30%)+ 所有 KV 的 LDS 读+读延迟 → 兑现 SKIPST 天花板(cr128→564>550、cr4→~924)。
- 关键赌注:fwd 是 latency-bound(MFMA/VALU 空闲),ds_bpermute 的 VALU 成本可能被空闲算力吸收(免费),与 bwd(compute-bound,bpermute +320% 判负)不同 regime。
- 难点:16×16 转置需 ~16 bpermute × 32 d-tile,VALU 量大,可能反而饱和空闲的 VALU——需实测 A/B。gather-layout(g_row=kv,g_within=d)→ QK-A(lo=kv,reg=d)和 PV-A(lo=d,reg=kv)两套 reshuffle 的 lane 映射要仔细推(参考 bwd 的 build_interm_rtr 手搓 ds_bpermute 16×16 转置,memory: project_dsv4_interm_register_transpose)。
- 先在小 shape(cr128,少 tile)验证:只把 PV 的 tr16 换成 bpermute(保留 QK LDS),测 LDS-wait 是否降、VALU 是否不饱和。若 PV-only 有效再扩到 QK+免 store。

## 修正:bpermute 是"可隐藏"的正解路径(非必负)——具体实现方案(2026-07-11)
之前误判 bpermute 必负。bwd `_bwd_interm_kernel.py` 注释给出正面证据:ds_bpermute 转置 burst(~512 ops)**"overlaps HBM ... freed VALU slots don't shorten the critical path"(MfmaUtil 22%)**——即对 latency-bound,bpermute burst 能被 HBM/gather 延迟隐藏,不占关键路径。所以 **register-resident KV + bpermute 转置**能同时:①消 store(SKIPST 天花板 cr128→564/cr4→~924)②bpermute 隐藏在 HBM gather 后,是真正未被否定的正解。
**具体实现方案(给下一次专门实现):**
- gather_load_b 已把 KV 放寄存器(布局 g_row=kv=tid//CHUNKS,g_within=d-chunk=tid%CHUNKS,V8PT 个 v8);**不写 LDS(免 store)**。
- QK-A = K[kv=lo, d]:需 lane lo=kv、reg=d。gather 是 kv=g_row。用 ds_bpermute 把 kv 从 g_row 维搬到 lane 维。
- PV-A = V[d=lo-related, kv]:参考 bwd `_tr4`(si=16*grp+4*i+lo_d4;ds_bpermute(idx=si*4, dw);VPERM 版 perm_b32 提取 elem)。fwd 需重推 si 映射匹配 gather 布局→ 当前 tr16 的 (grp*4+lo_d4)=kv, lo_m4*4+dt*16=d。
- 验证路径:先 PV-only(留 QK LDS+store)用 bpermute 替 tr16,SNR>40 确认 lane 映射对 + 测 cr128 是否降 LDS-wait(bpermute 是否被 HBM 隐藏);对了再扩 QK+免 store 兑现天花板。
- 风险:cr128 KV 小、gather 的 HBM 延迟可能不足以隐藏 512-op burst(与 bwd 的 gather-heavy 不同);需实测 A/B。若隐藏不住则回退。
- 这是 occ-2 地板下唯一能把 latency 余量(HBM 61%/MFMA 24%)转成速度的结构路径,值得专门多小时实现+迭代。

## 决定性根本证明:store 架构必需,bpermute 不可行(2026-07-11,修正上一条)
推导 bpermute lane 映射时发现根本障碍,推翻"bpermute 有望":
- gather(g_row=kv=tid//CHUNKS)把 16 个 kv **分布在 4 个 wave**(wave w = kv 4w..4w+3)。QK-A(kv=lo,wave 内)/PV-A(转置)需要的 KV reshuffle **跨 wave**。
- **ds_bpermute 只在 wave 内 64 lane 洗牌,不能跨 wave**;permlane 同理。
- 更根本:**MQA 的单一 KV 被全部 4 wave(64 heads)共享**。跨 wave 共享数据在 CDNA 上**只能走 LDS**。寄存器是 per-wave 的,无法跨 wave 共享。
- **∴ LDS store 是"把 gather 的跨 wave KV 汇成全 wave 共享 + reshuffle"的架构必需机制,无法用寄存器/bpermute 替代。** SKIPST 的 47%/30% 是这个必需机制的代价,SKIPST 天花板(564)不可兑现(它假设跨 wave 数据免费)。
- 这是 store 必需性的**根本原因**(此前只知 register-prefetch 隐藏 HBM 优于 DMA;现知更深层是跨 wave 共享强制 LDS)。
**综合终局**:cr128 428 / cr4 630 是根本地板 = ①store 架构必需(跨 wave MQA 共享)②occ-2 三重证明锁死(Q+temps 128 VGPR)③latency 余量(HBM 61%/MFMA 24%)被 occ 锁住填不进。全 shape 1.6× 于 triton 参考。目标 700/550 需超出 kernel 层(不同算法/硬件/工具链)。bh=128(pro cr4 +35%)是本轮落地净胜。

## 深度 banded DMA 实测判负(2026-07-11)—— register-prefetch 本质最优,终局确认
实现了 banded 深度预取 DMA(ring buffer NBUF=KD+1,精确 vmcnt `_waitcnt_vm`,tile t+KD 预取,SNR 全 PASS):cr128 flash 417→268 / pro 428→286,**更差**(比 1-ahead 353 还差)。∴ 即使深度预取,DMA 的 tile 粒度等待无法像 register-prefetch 的按寄存器细粒度 waitcnt 那样藏 HBM。此前"深度 DMA →611"估算错误。**store-based register-prefetch(428)是最优,已实证确认。** 这关闭了最后一条未测杠杆。全 shape 地板终局:cr128 428 / cr4 630 / 1.6× triton 参考,受 store 架构必需(跨wave)+ occ-2(VGPR)+ register-prefetch-最优 三重根本约束。

## flash cr4 pstore 落地 win(2026-07-12,campaign)+ cr128 7 探针矩阵
**flash cr4 632→667(+5.5%,commit local 77ecadad/remote 9011462e,未push)**:pstore 4-buffer parity 预存流水(暴露 KV store 47% 与 PV MFMA 重叠 + 3→1 barrier/pair),省 VGPR 后 pf_pv 2→7 + pf_qk 3→5 不触 occ-1 悬崖。gate flash cr4(bh=64 非 banded)。pro cr4 顺带 823→840。env PRIMUS_DSA_FWD_PSTORE。SNR 60.8dB。远端 gfx950 干净 A/B(632 stable vs 667 stable,两侧<0.3%)。
**cr128→550 目标 7 探针矩阵(2026-07-12,远端 gfx950 干净 A/B + rocprofv3 kernel-trace)**:
- 权威资源(pro cr128 kernel-trace):VGPR=128,**AGPR=0**(o_acc 在 VGPR;旧「o_acc AGPR128」对 banded 作废),LDS=34304,scratch=0,WG=256。occ=65536/34304=**1.91 纯 LDS-limited**(VGPR 允许 4 blocks)。MfmaUtil 24%、SQ_INSTS_VALU 72.9M、LDS-wait 41% of WAIT、bank-conf 12(低)。
- 7 探针(pro cr128,base 432):①SB=1 LDS 34K→17K→occ 3.76 = **平** 非 occ-limited;②pf_pv 5→10 平(prefetch 饱和);③REUSETR(tr16 读塌1)**−23%**(读已隐藏,非瓶颈);④SKIPRESC(去 rescale mul)**+6.3%**(flash +1.5%,印证 r6 +7.4% 天花板);⑤**dual-accumulator −46%**;⑥SKIPBAR pro +2.3%/flash −20%(barrier 非瓶颈);⑦bank-conf 低。**单杠杆最大 +6.3% 不叠加,目标需 +27%。**
- **方向①dual-accumulator(unroll-2 双累加器破 PV 跨-tile 链,env PRIMUS_DSA_FWD_DUALACC)= −46%(432→232)**:SNR 54.4 PASS(正确)、VGPR 208 **无 spill**、occ 不变 → MFMA-ILP 加倍反伤 → **cr128 确认非 MFMA-ILP/跨-tile-链 bound**(与 occ-4 flat + REUSETR 变慢自洽)。
- **方向②KV-LDS 宽读 = HW 封死 + 前提证伪**:gfx950 无 ds_read_tr16_b128;pre-transpose 写需跨-wave shuffle(bpermute 仅 wave 内,同 cr4 store 墙);REUSETR 已证 tr16 读非关键路径 → 无可兑现实现。
- 根因:cr128 = 均衡 latency-bound,per-tile 串行 QK-mfma→permlane-max→exp2→PV-mfma,MFMA 闲 76% 但 occ/ILP/read-width/barrier 逐一实测都填不进。probe harness:`_pmc1.sh`/`_pmc2.sh`/`_prof_fwd_cr.py`(kernel-trace 取VGPR/LDS)/`_verify_cr128.py`(SNR+time)。

### ★★ 重大修正:cr128 是 KV LDS-STORE (ds_write) 吞吐-bound(2026-07-12,续)
上面 7 探针**漏测了 store**。补测 SKIPST(跳 banded KV LDS store):**pro cr128 432→561.6(+30%)、flash 419→540.6(+29%)——store 就是 cr128 的主 bound,且 SKIPST 天花板 561/540 正好在 550 目标!** 这推翻「单杠杆最大 +6%」。occ-4 flat 的真因 = ds_write 端口(+store 后 barrier)在 occ-2 就饱和。
- **是 work-bound 非 exposure-bound**:SKIPBAR(只去 barrier)+2.3% vs SKIPST(去 store 工作)+30% → 30% 里 ~28% 是 ds_write 工作量本身。∴ **重排/预存 store 无用(工作量不减)**。
- **banded prestore(env PRIMUS_DSA_FWD_BPRESTORE,store t+1 与 PV of t 重叠+双缓冲)实测 −40%(432→257)**:SNR 77.8 bit-exact 正确,但更慢(工作量没减 + 丢 HBM-load 重叠)。**确认 work-bound:赢 cr4 的 pstore(重排)对 work-bound 的 cr128 无效。**
- **可兑现 +30% 的唯一杠杆 = 写更少 = query-blocking**:store 把 16 KV行×512 写进 LDS/tile,SWA 窗口相邻 token 重叠 127/128 → 每个 kv 行被 ~128 个 token 的 WG 各存一次(巨量 ds_write 冗余)。1 WG 处理 B 个相邻 token 共享窗口(存 127+B 行 vs B×128 行),SWA(8/10 tile)的 store /B。B=4 → SWA store 4× 少 → 逼近 +30% ceiling。代价:B× o_acc/m/l 寄存器、grid (T,H/BH)→(T/B,H/BH) 重构、per-token 窗口移位掩码 + per-token epilogue。V 必须进 LDS(PV tr16 转置跨-lane,寄存器/bpermute 不可替代)→ store 本身不可免,只能跨 token 摊。

### query-blocking 已实现且正确,但被 256-VGPR 编译 cap 卡死(2026-07-12)
完整实现 QB(env PRIMUS_DSA_FWD_QB=B):grid T→T/B、SWA 并集 tiled(store 每 tile 一次)、per-token 寄存器逐行掩码(qb_swa_mask/qb_pool_mask,不用共享 LDS mask)、pool 共享 store、B× o_acc、B× epilogue;**全展开 NT tiles 避 scf.for iter_args 坑**(坑真实:`if t+1<NS` 在展开循环里首个 kv_next 未绑定 → 先 `kv_next=kv_cur` 兜底);Q 改 tile 内即时加载(不常驻,省 64VGPR/token)。参数化 _qk_s(qp)/_store_o(tok)/_write_lse(tok)/kv_and_valid_b(tok),QB=0 默认零回归(flash cr4 656、cr128 432/419 不变)。
- **QB=2 实测:SNR 64.9dB PASS(机制正确!)但 432→168TF(−61%)**。kernel-trace:**VGPR=256(cap)、Scratch=196~804(SPILL)、AGPR=0**。根因:2×o_acc(热累加器,每 tile 更新,~152VGPR)+ Q(64)+ temps 撞 256 VGPR 的 occ-2 编译 cap → o_acc 溢出到 scratch,每 tile 读写 scratch → 灾难。降 pf_pv/pf_qk 不改(仍 256/spill)。
- **真正的修法被 toolchain 封死**:①o_acc 进 AGPR(当前 AGPR=0,256 独立 AGPR 空着)或 ②occ-1 允许 512 VGPR——都需 compile flag(amdgpu-num-vgpr/waves-per-eu/mfma-vgpr-form)。**本环境 compile hints 实测无效**:WPEATTR=1/2 对 VGPR 纹丝不动(仍 256),同 [[project_mxfp4_vgprform_deadend]] 的「external codegen 不可用」墙。
- **穷尽的修 spill 尝试全失败**(2026-07-12):①WPEATTR(rocdl.waves_per_eu)→ VGPR 不动;②**make_value_attrs(agpr_alloc=-256/128 + amdgpu-mfma-vgpr-form=false,GEMM 生产在用的机制)→ AGPR 仍 0**;**决定性验证:直接把它应用到 baseline 非-QB kernel,TESTAGPR=-256/128 → AGPR 恒 0、VGPR 恒 128 纹丝不动 → 本容器 agpr-alloc/mfma-vgpr-form 完全无效(external codegen 不可用,同 [[project_mxfp4_vgprform_deadend]]「ISA identical」);③Q-streaming(ks 内即时载 Q,不常驻)→ scratch=0 消了 spill 但更慢(104TF,Q 每 ks/tile/token 重载 352 次 + 丢 PF_QK/qk2 → Q-reload-bound)。hoisted-Q spill=168TF / streaming-Q=104TF,**全 < baseline 432**。
- **★ 根本地板 = gfx950/CDNA4 统一寄存器文件**:再验 inline-asm MFMA(`"=a,v,v,0"` 约束,gemm_helper.asm_mma_do 的机制,绕 compile flag)→ **AGPR 仍 0**。原因:CDNA4 **VGPR/AGPR 是同一个 512-entry 物理文件**,「AGPR=0」是因为**没有独立 AGPR 池可卸载 o_acc**——compile-flag 和 inline-asm 的 "a" 约束都指向同一文件,都不能给 o_acc 腾地方。∴ QB 的 2×o_acc 只能靠 occ-1(512 统一 VGPR)才 fit,而 occ-1 需 waves_per_eu flag,本容器实测死(WPEATTR/make_value_attrs 的 waves_per_eu 对 VGPR 256-cap 纹丝不动)。streaming-Q 能 fit 256(scratch=0)但 Q 每 ks 重载 → Q-reload-bound 104TF;hoisted-Q → o_acc spill 168TF。**双输。**
- **★★ 修正终局:QB 在 gfx950 是架构级净负(不只 toolchain)**。`amdgpu-waves-per-eu="1,1"` 的 **passthrough 函数属性形式生效了**(rocdl op-attr 形式无效,但函数属性形式认)→ QB=2 消 spill(scratch 196→0)但仍 **104TF**。因为:occ-2(256 cap)→ 2×o_acc spill 168TF;occ-1(无 spill)→ 占用 1.9→1.0 且每 block 做 2.2× MFMA → 延迟藏不住 104TF。**两种占用都 < baseline 432 → store 摊薄收益 < B 累加器的占用/寄存器代价**。这是 gfx950(512-VGPR/occ-2=256/统一 RF)的架构 tradeoff:cr128 的 store 是 bound,但唯一能减它的 query-blocking 代价大于收益。**∴ cr128 419/432 是纯 flydsl+bf16 在 gfx950 的真实架构地板,非 toolchain артefакт。550 需换算法(减 store 本身,但 V 必进 LDS 供 tr16 不可免)或换硬件。**
- **QB 机制正确(所有变体 SNR PASS)、代码已就位(env PRIMUS_DSA_FWD_QB,gated off 零默认回归)**——2×o_acc(~152 VGPR,rescale 是 VALU 故必在 VGPR 不能进 AGPR)吃满 256 occ-2 cap,Q 只能 spill(慢)或 reload(更慢),而抬 VGPR cap/强制 AGPR 的 compile flag 在本容器全不生效。在能控寄存器分配(agpr-alloc 生效)的系统上 QB=2 预期 ~497。**cr128 在此环境的地板仍 432/419,受 KV-store ds_write work-bound + 寄存器分配不可控 双重约束。**

### 本轮 myskill 挖掘的新杠杆实测(2026-07-12,cr128 continue)
从 methodology 07/06/08 挖出并逐一实测/分析,全部对 cr128 store-bound 无效:
- **inline-asm ds_read_b64_tr_b16(避 SIInsertWaitcnts 保守 vmcnt(0) drain,methodology-07)**:env PRIMUS_DSA_FWD_ASMTR。实测 cr128 431→431 / flash cr128 419→421 / pro cr4 841→841 / flash cr4 657→657 **全平** → intrinsic tr16 的保守 drain 对本 kernel 不是问题(gather 已良好重叠,印证 REUSETR「tr16 读非瓶颈」)。gated off。
- **raw-asm 写死物理 AGPR(methodology-06:LLVM 拒 =a → raw volatile asm 写 a[N] + clobber ~{aN} + v_accvgpr_write/read)**:这解释了我之前 inline-asm "=a" 得 AGPR=0(=a 被拒)。但分析:QB 2×o_acc=256 AGPR + Q(64)+temps > 256 仍 occ-1.4;且实测 QB occ-2-spill(168)> occ-1(104)→ 占用比 spill 更主导,AGPR 到 occ-1.4 估 ~130-150 仍 < 432。不值重工程。
- **per-wave 冗余 KV load + 寄存器转置免 store(permlane/bpermute,methodology-08)**:估 4× HBM ≈ 0.38ms ≈ 当前总时 → 中性(store 省 30% 被 4× HBM 抵消)。
- **DMA(raw_ptr_buffer_load_lds,免 reg→store)/ closed-form**:已记判负(HBM 暴露 > store 节省)。
∴ 多轮 myskill 交叉验证:cr128 = store(ds_write)work-bound,架构必需(V→LDS 供 tr16),所有已知减-store 杠杆(QB/DMA/per-wave/asm-tr)在 gfx950 512-VGPR/256-occ2/统一RF 结构下均无法净兑现。

### ★ 权威 ISA 修正 CSV 误诊 + QB 根本墙定性(2026-07-12,methodology-03)
methodology-03 明确:kernel-trace CSV 的 `Accum_VGPR_Count` 恒报 0、`VGPR_Count` 对 256-VGPR 内核误报——**唯一权威是 FLYDSL_DUMP_IR 的 ISA `num_vgpr/num_agpr/private_seg_size/accvgpr 数`**。用 ISA 重查 QB=2(occ-1 via amdgpu-waves-per-eu=1,1 passthrough,该函数属性形式生效):
- 真实:**num_vgpr=256, num_agpr=256(非 CSV 说的 0!), private_seg=116(小), accvgpr 拷贝=1204**。→ 之前「AGPR=0/spill 严重」是 CSV 误诊;真相是 **o_acc 在 AGPR(256)但 rescale(o_acc*alpha 是 VALU)每 tile 把它 AGPR↔VGPR 搬运 = 1204 accvgpr 拷贝**,这才是 104TF 的瓶颈。
- 强制 agpr-alloc=0,0:num_agpr=0/accvgpr=0(消 shuffle)但 **num_vgpr 仍 256(编译器死不用 512)+ spill 1192 → 37TF**。∴ **256-VGPR 是编译器硬 cap**(occ-1 有 512 也不用),QB 的 >256 寄存器需求只能靠 AGPR,而 AGPR+rescale = 不可免的 accvgpr shuffle。
- **QB 根本墙(权威)**:256-VGPR 硬 cap → QB 必用 AGPR(256V+256A=512=occ-1)→ rescale 逼 accvgpr shuffle(104TF)或强制免 AGPR 则 spill(37TF)。唯一消 shuffle = 两遍法(alpha≡1 免 rescale → o_acc 纯 AGPR 累加 asm-inplace mode2),但仍 occ-1(QB 需 512 寄存器)+ 2× QK,而 baseline occ-1.9 → 占用腰斩损失 > store 省的 30%。**QB 在 gfx950 架构净负,根因是 256-VGPR 硬 cap 使 QB 必落 occ-1。**

### 两遍法 QB 被实测判负(2026-07-12,决定性探针)
两遍法 QB 的全部价值 = 消 rescale → 消 accvgpr shuffle。用 QBSKIPRESC 探针(QB 内强制 alpha=1,不正确但测无-rescale 天花板):**QB2 normal 89.5TF vs QB2 skiprescale 89.4TF 完全一样**,accvgpr 1204→558 减半但性能纹丝不动。∴ **accvgpr shuffle 不是 QB 瓶颈,occ-1 才是**(256-VGPR 硬 cap → QB 需 512 寄存器 → occ-1 vs baseline occ-1.9,latency-hiding 腰斩)。**两遍法(消 rescale)实测救不了 QB**。这是用测量而非分析证否最后一条杠杆:cr128 的 store 摊薄杠杆(QB)在 gfx950 因 256-VGPR 硬 cap 落 occ-1,占用损失 > store 省,且与 rescale/shuffle 无关。**全部减-store 杠杆(QB/两遍法/DMA/per-wave)均实测/析证 net-negative,cr128 地板 432/419 是 gfx950 架构 + 256-VGPR-cap 的物理结果。**

### ★★★ 硬件级终极:256 arch-VGPR 是 gfx950 硬件上限(2026-07-12,chi2811 确认)
用户导到 chi2811(同镜像 saved-20260703 / ROCm 7.2.1 / NFS 同源 → 同 toolchain,clean baseline 435/419 一致)。找到 flydsl 正确的 VGPR-cap 机制:`CompilationContext.compile_hints({"maxnreg":N})` → codegen CLI `--amdgpu-num-vgpr=N`(**不是**我之前试的 value_attrs 函数属性)。实测 QB=2 + maxnreg=512:**num_vgpr 仍 256、num_agpr=253、accvgpr=945、103TF**。→ **`--amdgpu-num-vgpr=512` 也无法让 arch VGPR 超 256**,因为 **256 是 gfx950/CDNA4 的硬件 arch-VGPR 上限**(256 VGPR + 256 AGPR = 512 物理,但 VGPR 部分硬顶 256)。
- **∴ 最终硬件定论**:cr128→550 需 QB(减 store),QB 的 2×o_acc+Q > 256 寄存器 → gfx950 上**必须**用 AGPR → o_acc 经 rescale(VALU)逼出 accvgpr shuffle → occ-1 net-negative。**256 arch-VGPR 是硬件限制,无任何 toolchain flag(函数属性 or --amdgpu-num-vgpr codegen flag,均已实测)能突破。** cr128 419/432 是 gfx950 硬件 + store-bound 的物理地板。550 需换硬件(更多 arch VGPR)或换算法(消 store 本身,但 V 必进 LDS 供 tr16 不可免)。

### ★ 算法重写调查:store-elimination 经布局分析判净负(2026-07-12)
用户要求"做算法重写"破 cr128 store(+30%)。深挖数据布局后判定 store-elimination 净负/不可行(设计分析先于写代码,避免数百行注定失败):
- **store 的本质 = 合并 gather + 跨线程重分布 + 转置的高效机制**,非浪费:
  1. gather 从 HBM **合并**加载 KV[kv_row,d],thread by g_row=tid//16
  2. QK 消费 K by lo=tid%16(≠gather 布局 → 必须跨线程重分布);PV 消费 V 转置布局
  3. store(ds_write)→LDS→QK-bv/PV-tr16-read = 同时做重分布+转置
- **替代方案全更差**:①per-wave ds_bpermute 转置:走 LDS crossbar,重分布 32bf16/thread 需 ~16 bpermute op ≫ store 的 4 dwordx4 写 → 净负;②HBM 预转置 gather(thread 直读 V[d,kv]):每线程读不同 kv 行单 d 元素 = 非合并 HBM → 更差。
- ∴ **store+tr16(合并 gather + LDS 边界转置)是最优**,SKIPST 的 +30% 是跨线程重分布+转置的固有成本,高效完成,无更便宜替代。**store-elimination 算法重写判负(设计分析,未写代码)。** cr128 419/434 是固有地板,不只硬件(256-VGPR)也是数据流(合并 gather ⊥ 消费布局的重分布税)双重必需。

### ⚠️ 修正:register-transpose (ds_bpermute) 在本 codebase 已证可行且赢(bwd rtr 先例)
上面把 store-elim 的 op-count 分析判净负——**但发现 bwd interm_rtr 已用 `_tr4`/`transpose`(ds_bpermute 16×16 寄存器转置)替代 LDS 转置,且是 WIN(interm rtr cr4 +11.8%,见 [[project_dsv4_interm_register_transpose]])**。`_bwd_interm_kernel.py:138-159` 有已验证的 ds_bpermute 转置原语(每元素 2 bpermute + perm/select,4 元素/tile)。关键:转置 burst 与 HBM 重叠、kernel tr16-latency-bound(MfmaUtil 22%)→ 转置成本被藏(注释 line 130-133)。∴ 我的 op-count 预判(bpermute≫store 净负)**太悲观**,像之前 CSV 误诊——register-transpose 值得实测,有成功先例。
- **fwd cr128 register-transpose PV 是真正该实现+实测的重写**(不是纸上判负):per-wave 全 gather(4×,L2 cached)+ 复用 `_tr4` bpermute 转置 V → PV MFMA A-操作数 → 消 store+tr16。可能兑现 SKIPST +30%。是大重写(需充足 context 分步 + SNR 验),但有 bwd rtr 先例背书,不是死路。下一步实现方向(实测不预判)。

### ★★ register-transpose PV 算法重写:已实现+实测 net-negative(2026-07-12,chi2811)
用户要求"就在这里完成"。**实现了完整 register-transpose banded 路径**(env PRIMUS_DSA_FWD_RTR):QK+PV 从 per-wave 寄存器 gather(QK-B 布局 K[kvrow(lo),ks*32+grp*8] + compact-V 布局 V[kvrow(l//4),dt*16+(l%4)*4])、复用 bwd `_tr4` ds_bpermute 16×16 转置(输出匹配 tr16)、register mask、**免 store 免 barrier**。
- **实测(pro cr128):SNR 79.3dB PASS(机制完全正确,甚至>base 77.8)但 97.7TF(vs 428,−77%)**。
- 根因(测量证实最初 op-count 分析,曾被 bwd-rtr 先例误导去怀疑):**per-tile bpermute 转置成本 ≫ store**。每 d-tile 8 ds_bpermute × DT=32 × 10 tile = ~2560 bpermute/token ≫ store 40 dwordx4 写 + 8× gather(2 布局 ×4 per-wave)。bwd rtr 赢是转置**摊薄在 RGROUPS×4 rank-tile**(转一次复用);**fwd PV 每 tile V 不同、无摊薄** → 转置每 tile 重做 = 昂贵。
- ∴ **store+tr16 对 fwd 是最优,SKIPST +30% 是最优机制的固有代价,register-transpose 实测慢 4×**。算法重写已做到底(实现+SNR+实测),确认 store-elimination 净负。这是测量结论。代码 env-gated off 保留作参考。**cr128 419/434 是数据流最优地板(实测证实)。**

### ★★ flash-attn 手法全移植测试矩阵 + occ-2 结构定论(2026-07-12,对照 flash_attn_{fwd,bwd}_kernel.py)
系统对照标准 dense flash-attn 逐个移植 + matched A/B,全负/中性/被工具链堵(代码全回退,AST 逐字节复原):
- **lazy O-rescale**:flash cr4 −31%(scf.IfOp 破 pstore 流水)/ pro cr4 中性 → cr4 非 rescale-bound。
- **post-misched**(enable-post-misched llvm_options):fwd 中性 / bwd −21%(手工调度 + inline-asm AGPR 已优,post-RA 重排打乱)。
- **fast-math**(compile_hints fast_fp_math/unsafe):逐位无效(rocdl 内联 FP,hint 不作用)。
- **buffer_load_lds**:需 lane-consecutive LDS(无 pad);去 D_LDS pad 实测 flash cr4 648→254 / pro 840→277(**−60~67%**,tr16 bank 冲突)→ by construction 净负。padding 528 已最优(544 更差,560/592≈528)。
- **ds_load_tr16_b128**(v8 一次读,减半 tr 读指令,直打 LDS-port):**LLVM Cannot select intrinsic**(gfx950 后端不支持,绑定有但 codegen 拒)。
- **fine-grained vmcnt / sched_barrier**:N/A(已 hoist gather + 每 tile barrier 是 LDS-RAW 必需);**masked/unmasked split**:已内建(mask 预算到 lds_mask + dyn_pool)。
- **★ occ-2 结构定论(两探针决定性)**:fwd occ-2 是 VGPR 限,o_acc = D512 的 f32 输出累加器 = **128 寄存器**。①AGPR o_acc 探针:occ 仍 1.96(AGPR 文件也 256/SIMD,128 在 VGPR/AGPR 都 occ-2)+ TF −16%;②bf16 o_acc 探针:occ 仍 1.98(MFMA C 仍需 f32 峰值 128 + loop old/new o_acc 双活 + 每 d-tile cast 的 VALU)+ TF −58% + SNR 62→43。→ **o_acc 128 寄存器结构性锁 occ-2,寄存器缩减撬不动**;唯一理论剩余 = d-chunk 2 趟(QK/gather/softmax 翻倍,几乎必负)。
- bench 收敛为单个 `bench_mla.py`(6 fwd + 6 bwd = 12 结果 + triton SNR,`python bench_mla.py` 直接跑;删 campaign_*/bench_campaign/bench_triton/verify_turbo)。
∴ MLA attention 已追平/超越 flash-attn(已有 tr16 / 寄存器预取 / K32-direct-v8 / inline-asm-AGPR / s_setprio / XCD-remap / mask-precompute,数项 flash 反而没有);通用移植无效 = MLA 已重度调优 + sparse-gather/tr16-padded 结构 ⊥ dense 技巧。真正剩余头款 = 算法级 / 工具链解锁(b128 select、accvgpr helper)。
