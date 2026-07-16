# pitfalls/13 — Meta/gpt_oss hd64 DENSE flash attention bwd(确定性)实测经验 + 死胡同

> 类别: 踩过的坑 · 主题标签: meta, gpt_oss, hd64, D64, dense-flash, attention, bwd, dq, dkdv, odo, deterministic, CK-replacement, 16x16x32, schraudolph, poly-exp2, delta-fusion, drop-b-gemm, exp2-pipeline, shared-gemm1-fusion, fp8-gemm1, occ3-split, pack-128, latency-bound, MFMA-operand-bubble, gfx950, dead-ends

> 任务：`flash_attn_varlen_func_flydsl` —— FlyDSL 算子替代 CK 确定性 hd64 FMHA varlen bwd。
> Meta shape B4 Hq128 Hkv16 **D=64** Skv16384 Sq∈{2048,4096,8192,16384},bottom-right 全因果,bf16,THD。
> 三核:`dq`(Q-outer,identity-center rho/R)+ `dkdv`(KV-outer split-K Q_SPLIT=2)+ `odo`(delta)。16x16x32 MFMA。
> exp2 双模:Schraudolph fast(**35dB**,shipping)/ poly deg-3(**>50dB**)。确定性=构造性(单 WG 独占 tile+无 float atomic;dK/dV split-K workspace + host 固定序 fp32 sum)。
> 文件 `sync/meta-attn/meta_aiter_attn/flydsl/flash_attn_bwd_rect16_kernel.py`(bench 直接 import,无 regen)。基线由 `_apply_pack128.py` 生成(见 [[project_flydsl_meta_ck_replacement]])。

## 瓶颈定性(rocprofv3 + PMC,已接地)
- **MFMA-operand-latency-bound**:bwd 98.9% 时间在 kernel,dq/dkdv 均匀 **MfmaUtil ~47-50% @ occ2**,远未打满 → 延迟受限非吞吐。
- 瓶颈链 = **exp2/pack 的 VALU operand-bubble**(dkdv 更重、是 leader)。
- **LDS 不是瓶颈**:`SQ_LDS_IDX_ACTIVE : MFMA_BUSY` 有 **4-8× 富余**(dq 4×/dkdv 8×)→ 非 port/带宽受限 → **swizzle/prefetch/bank/pad 全 DEAD**(见 05,pack-128 消 0% bank 零收益,bank conflict 是红鲱鱼)。
- **occupancy 锁 occ2**:VGPR 是 limiter(dq 200 需 ≤168 才 occ3);occ3 需 AGPR-accum = 外部 codegen(inert),不可达。tile 全 register-walled。

## ✅ WINS(实测 KEEP,按增益排)
- **drop 冗余 B-GEMM / rho-R 全局修正项(dq)= +9.8%**:dq 里有可丢弃/可简化的第二 GEMM 或全局 renorm(rho/R)修正,drop 之 = 真结构性减 MFMA。**先审计每个 attention bwd 核有无这类项**(dkdv 无、只 +0.17%)。这是本 kernel 最大单杠杆。
- **odo delta 融合 = +5.3%**(长 Sq 最大):把 `delta=rowsum(O·dO)` 融进 bwd,消掉独立 delta 核(它在 wall 上全串行、随 Sq 放大)。★**必坑**:wrapper 必须 `out.to(q.dtype)`/`dout.to(q.dtype)` cast —— 融合核直接读 O/dO,若 harness 传 fp32 O → 崩/NaN(评分 gate raw=[],见 02)。
- **exp2 软流水(GQA head 轴)= +1.4%**:把 head h+1 的 `exp2(QK)` 预算藏进 head h 的 GEMM2 MFMA shadow(在 GQA head-loop 轴,**非** dt 轴——dt 轴已判负)。直击 operand-bubble。
- **s_setprio(1/0) 包 GEMM2 MFMA**:单独中性,叠在结构改动上 +1~2%(仅 MFMA-dense 的 GEMM2 有效;dkdv GEMM1 / dq 连续 span 判负)。
- **dq iglp_opt(1) = +0.5%**(bit-identical,全 Sq):藏 dq exp2/C operand-bubble。⚠**别和回归捆一轮**——被 bundled revert 会把这种真 sub-win 一起埋掉(grep committed kernel 查不到,须回退后重隔离)。

## ❌ DEAD(实测 do-not-retry)
- **P5 shared-GEMM1 融合(dq+dkdv 共享重算的 GEMM1,-25% 总 MFMA)= 全死**:
  - q-outer grid 融合 = **确定性陷阱**:dK/dV 的 q-归约不用 atomics 无法同时"有界 HBM + 确定" → 必须 KV-outer。
  - KV-outer 融合的 spill:BLOCK_KV=128 spill 187(dK/dV 累加器 NT=8 单独就 256 VGPR);**BLOCK_KV=64 = 218 VGPR/0 spill/24KB/occ2**(round-4 的"636-spill 墙"是 direction-misread,可破)。
  - **但融合核实测 1.95-2.55× 慢**:BLOCK_KV=64 per-tile 惩罚(standalone -24%)+ ~2GB dQ workspace 流量 > -25% MFMA 省。**shared-GEMM1 融合对 hd64 判死**。
- **fp8 GEMM1 operands(FA-3 plain scalar-scaled,非 MX)**:SNR **28.2 < 34** → 破正确性门。hd64 D=64 收缩维太短,fp8 GEMM1 死。
- **dkdv 拆 dV-only/dK-only 核(为 occ3)= -36%**:双份 Q/dO 读 + 拆分开销 > 占用率收益。
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
5. **exp2 是 hd64 bwd 的 operand-bubble 核心**:软流水藏(head 轴)有肉;缩短 exp2(降阶 minimax)是最后的确定性杠杆但量级小。
