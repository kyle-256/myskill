# attention backward 优化 playbook：定 bound → 减 MFMA → 融串行核 → occ/exp2 → 确定性

> 类别: 方法论 · 主题标签: attention, flash-attention, bwd, dq, dkdv, dkv, MLA, softmax, exp2, online-softmax, delta-fusion, drop-gemm, shared-gemm1, latency-bound, MFMA-operand-bubble, occupancy, occ1-occ2, tr16, ds_read_tr16, determinism, split-K, schraudolph, poly-exp2, gfx950, decision-order

> 面向 attention 反向核(dq / dkdv / dkv / interm / odo-delta)的优化决策顺序。经验来自两条已交付的
> flydsl attention:dsv4 sparse-MLA(pitfalls/12)+ Meta/gpt_oss hd64 dense flash(pitfalls/13)。
> 具体实测的 win/dead 表在那两张 pitfalls 卡;本卡是**通用打法与顺序**(先做什么、别做什么)。

## 步骤 0 — 先定 bound,再动手(最省命的一步)
attention bwd 几乎总是 **latency-bound**(MFMA 依赖链 / operand-bubble),不是吞吐或带宽。动任何杠杆前用 PMC 判:
- **`MfmaUtil`**:~30-50% @ occ2 → 延迟受限(远未打满)。不是 >80% 就别当吞吐问题治。
- **`SQ_LDS_IDX_ACTIVE : MFMA_BUSY` 富余** ≥ 4-5× → **LDS 非 port/带宽受限** → **swizzle / padding / bank-conflict / LDS 预取 全是红鲱鱼,零性能**(见 05;实测 pack-128 消 0% bank 对速度纹丝不动)。
- **subtractive 探针**:把某段(如 tr16 转置读 / 某个 GEMM)替成常量,看 wall 降多少 → 定它占多少、值不值得攻。
- `SQ_WAIT_ANY` 大头来自 MFMA 依赖 vs DMA:决定该攻 ILP 还是攻 load。
> 判据一句:**LDS 有富余 + MFMA 未打满 = latency-bound → 唯一 sized 杠杆是「减 MFMA」或「藏 MFMA 依赖延迟」,不是 feed 优化。**

## 步骤 1 — 减 MFMA(唯一够大的结构杠杆,优先级最高)
1. **审计可丢弃的 GEMM / 全局修正项**:每个 bwd 核里有没有可 drop 的第二 GEMM、或 rho/R 之类全局 renorm 修正(精度代价常 ~0.2dB 非破门)。dq drop-B-GEMM 实测 **+9.8%**(pitfalls/13)。**这一步永远先做**——纯减法、低风险、量级大。
2. **融掉全串行辅助核**:`delta=rowsum(O·dO)`(odo)、interm restage 等在 wall 上串行、且**随 Sq 放大** → 融进主核。odo delta 融合 **+5.3%**(长 Sq 最大)。★坑:融合核直接读 O/dO,wrapper 必须 `out/dout.to(q.dtype)` cast,否则喂 fp32 O → 崩/NaN(见 02)。
3. **核间共享重算的 GEMM1**(dq+dkdv 共用 S=K@Q^T):理论 -25% MFMA,但**三重陷阱常吃光收益,先验证再建**:
   - **确定性陷阱**:q-outer grid 融合的 dK/dV q-归约不用 atomics 无法「有界 HBM + 确定」→ 必须 KV-outer。
   - **register 墙**:融合后两套累加器共驻易 spill(hd64 dkdv dK/dV 累加器 NT=8 单独就 256 VGPR)→ 常被迫 BLOCK 减半。
   - **workspace 流量**:non-owned 输出的 split-K workspace(~GB 级)+ 减半 tile 的 per-tile 惩罚。hd64 实测融合核 **1.95-2.55× 慢**(pitfalls/13)→ 对短收缩维判死。**先 spike 可行性(det+spill+SNR)再决定多轮建核,设硬 abort 门。**

## 步骤 2 — 藏 MFMA 依赖延迟(减不动 MFMA 时)
- **16x16x32 拆链**:32x32 长串行累加链 → 4 条独立 16x16 → MFMA-latency ILP ×4,累加器 VGPR 不变。dkdv 实测 581→632TF(见 [[project_flydsl_dkdv_bwd_16x16]])。
- **operand-bubble 软流水**:把下一迭代的 exp2/pack 预算藏进当前迭代的 GEMM2 MFMA shadow。**选对轴**:GQA head 轴有肉(+1.4%),dt 轴常判负。dq iglp_opt(1) 同理(+0.5% bit-identical)。
- **occ-1 核靠交织藏延迟**:QK→softmax→PV 三段交织是 occ-1 藏延迟的唯一机制,别用「批 2-tile」打断它(dsv4 K=32-PV 判死,pitfalls/12)。
- **s_setprio(1/0) 包 MFMA-dense 的 GEMM2**:叠在结构改动上 +1~2%;单独中性,GEMM1 / 连续 span 常判负。

## 步骤 3 — softmax / exp2 数值(attention 特有,详见 08)
- **online-softmax 在 log2 空间做**,吃硬件 `v_exp2`(见 08 `### online-softmax`)。
- **exp2 三档权衡**:Schraudolph 位操作(最快,~35dB,单魔数,DMA-确定)/ poly deg-3(SNR>50 但 ~8 full-rate op,慢)/ v_exp2(精确,quarter-rate)。**exp2 是 latency-bound attention bwd 的 operand-bubble 核心**;缩短它(降阶 minimax)是最后的确定性杠杆但量级小。
- **lse 预缩放格式必须匹配 exp2 模式**:Schraudolph → `lse_s23 = lse·(-log2e)·2^23 + (127·2^23−486411)`;poly/精确 → 平 `lse·(-log2e)`。**dq/dkdv 默认模式可能不同,喂错格式 → 全 NaN**(见 02,本 session 踩过)。

## 步骤 4 — occupancy 现实检查(常是硬墙,别空转)
- attention **强制拉占用率几乎总 DEAD**:latency-bound 下 occ-1/occ-2 都填不满 wave,强制 waves_per_eu 抬占用率必 spill/掉速(dsv4 we3 实测 3× 慢;pitfalls/01,12)。
- **occ3 对 hd64 是硬墙**:VGPR 降不到阈值(~≤168)免谈;AGPR-accum 冲 occ3 需外部 codegen(inert)。拆核 / fp8 降 VGPR 的代价(双读 / 破 SNR)通常 > 占用率收益(hd64 拆 dV-only -36%、fp8 GEMM1 SNR 28.2<34)。
- **别 co-hold 两套累加器**;transient dV/dK vs loop-carried dQ 的 register 账要分开算(见 09)。

## 步骤 5 — 确定性(若目标要 bit-reproducible)
- **构造性确定**:单 WG 独占一个输出 tile + 无 float atomic。dK/dV 的跨 WG 归约用 **split-K workspace `[B,kv_split,Sq,H,D]` + host 固定序 fp32 sum**,不用 bf16-atomic(atomic 破确定 + 破 SNR)。
- fast 路径要确定 → 只能 Schraudolph(DMA 确定);poly-dq 的 flydsl `buffer_load_lds` 有 WAR race → 非确定(见 pitfalls/04、[[project_dsv4_fwd_dma_experiment]])。
- **确定性把「放宽 det 换 -MFMA」这条路封死** → 长收缩维 / 长 Sq 的最后缺口常只剩「research 级更短 exp2」,那是研究规模、无 KB 先例,要上层裁决。

## 通常 DEAD 清单(latency-bound attention,别重试)
swizzle/pad/bank(port 有富余)、LDS 预取/双缓冲(occ 已藏 DMA;长 Skv 偶尔边际)、强制占用率/AGPR、DMA-免-reg(scattered-gather 用 register-prefetch 最优)、fp8 GEMM1(短收缩维破 SNR)、bigger tile(register-walled)、q_split 过大(K/V reload 冗余)、dual-layout-KV 消 bank、per-tile 重做 register-transpose(不摊薄)。逐条实测出处见 pitfalls/12(dsv4)+ pitfalls/13(hd64 dense)。

## 相关卡
- 经验/死胡同:pitfalls/12(dsv4 sparse-MLA)、pitfalls/13(Meta hd64 dense flash)
- 数值/跨 lane:methodology/08(online-softmax、XOR shuffle、DPP)
- 正确性坑:methodology/02(exp2/lse 格式 NaN、det、JIT 缓存)
- LDS 转置读/bank:methodology/05(ds_read_tr16、pack-128、bank 红鲱鱼)
- profiling:methodology/03(PMC bound 判定、subtractive 探针)
- occupancy:pitfalls/01、methodology/04
