# attention(fwd + bwd)优化 playbook：定 bound → 打对应瓶颈 → exp2/occ/确定性

> 类别: 方法论 · 主题标签: attention, flash-attention, fwd, bwd, dq, dkdv, dkv, MLA, softmax, online-softmax, exp2, lse, rescale, fmax0, dual-wave, store-bound, lds-read-bound, latency-bound, MFMA-operand-bubble, delta-fusion, drop-gemm, shared-gemm1, occupancy, occ1-occ2, tr16, ds_read_tr16, triple-buffer, determinism, split-K, schraudolph, poly-exp2, gfx950, decision-order

> attention 前向 + 反向核(fwd / dq / dkdv / dkv / interm / odo-delta)的优化决策顺序。经验来自两条已交付
> flydsl attention:dsv4 sparse-MLA(fwd+bwd,pitfalls/12)+ Meta/gpt_oss hd64 dense flash(bwd,pitfalls/13)。
> 具体实测 win/dead 在那两张 pitfalls 卡;本卡是**通用打法与顺序**(先做什么、别做什么)。

## 步骤 0 — 先定 bound,再动手(fwd/bwd 通用,最省命一步)
- **`MfmaUtil`** ~30-50% → latency-bound(远未打满,别当吞吐治);>80% 才是吞吐。
- **`SQ_LDS_IDX_ACTIVE : MFMA_BUSY`** 富余 ≥4-5× → **LDS 非 port/带宽受限** → swizzle/pad/bank/LDS 预取全红鲱鱼零性能(见 05)。富余小 → 才是 LDS-read-bound(fwd 常见,PV 转置读)。
- **store 计数 / exposed-store stall**:fwd 常 **store-bound**(epilogue O 写回)—— 用 SKIPST/CFST 探针(注掉 store 看 wall 降多少)判。
- **subtractive 探针**:把某段(tr16 读 / 某 GEMM / rescale / store)替常量,看 wall 降多少,定它占比。
> 一句:attention 几乎总 **latency/store/LDS-read-bound,不是吞吐**。fwd 多为 store 或 PV-tr16-read-bound;bwd 多为 MFMA-operand-bubble(exp2/pack)-bound。**对症下药,别拿 feed 优化治 latency。**

## FWD 专属杠杆(定 bound 后按类打)
- **store-bound → 消冗余 store + 藏 store**:
  - **BLOCK_H 提高**消多-WG 冗余 KV/O-store(dsv4 pro BLOCK_H=128 消 2-WG 冗余 KV-store,707→794);
  - **triple-buffer** 藏 exposed epilogue store;
  - query-blocking 减 store ⚠ 常 DEAD(2×o_acc>256 VGPR 强制 occ-1,占用损失 > store 省)。
- **rescale 折叠(_FMAX0)**:softmax 平移不变 → 用 first-pair 固定 max → `alpha=1`,编译器折叠掉 online-rescale 乘法(dsv4 pro cr4 828→933TF,+13%)。当 max 可安全固定时首选。
- **GEMM↔softmax 并行(dual-wave)**:唯一让 GEMM 与 softmax 真并行的解 = 上游 `flash_attn_gfx950` 两 wave-group **时间复用**(s_barrier 错相位)+ cluster 流水 + `sched_group_barrier`(MFMA 0x8 / VALU 0x2 / **EXP 0x400**,exp2 是 0x400 非 VALU)+ lazy-rescale(见 [[reference_flydsl_flash_gfx950_dualwave]])。重、但 latency-bound fwd 的上限杠杆。
- **PV 的 tr16 转置读常是 fwd 头号 LDS-read 成本**(dsv4 去 pad 掉 60-67%);b128 减半读被 LLVM "Cannot select" 挡。
- ⚠ **lazy O-rescale 单独移植常 net-negative**(dsv4 MLA cr4 -31%,pstore 流水冲突)——它是 dual-wave 套件的一部分,别单拆。

## BWD 专属杠杆:减 MFMA(唯一够大的结构杠杆,优先级最高)
1. **审计可丢弃的 GEMM / 全局修正项**:第二 GEMM、rho/R 全局 renorm(精度代价常 ~0.2dB 非破门)。dq drop-B-GEMM 实测 **+9.8%**。**永远先做**,纯减法量级大。
2. **融掉全串行辅助核**:`delta=rowsum(O·dO)`(odo)、interm restage 等串行且随 Sq 放大 → 融进主核。odo delta 融合 **+5.3%**(长 Sq 最大)。★坑:融合核直读 O/dO,wrapper 必须 `out/dout.to(q.dtype)` cast,否则喂 fp32 O → 崩/NaN(见 02)。
3. **核间共享重算 GEMM1**(dq+dkdv 共用 S=K@Q^T,理论 -25% MFMA):**三重陷阱常吃光收益,先 spike 再建**:①确定性陷阱(q-outer dK/dV q-归约不用 atomics 无法有界+确定→必 KV-outer);②register 墙(两套累加器共驻 spill→常被迫 BLOCK 减半);③workspace 流量(~GB split-K + 减半 tile per-tile 惩罚)。hd64 实测融合核 **1.95-2.55× 慢**→短收缩维判死。设硬 abort 门。

## 步骤 2 — 藏 MFMA 依赖延迟(fwd/bwd 通用,减不动 MFMA 时)
- **16x16x32 拆链**:32x32 长串行累加 → 4 条独立 16x16 → MFMA-latency ILP ×4,累加器 VGPR 不变(dkdv 581→632TF)。
- **operand-bubble 软流水**:下一迭代 exp2/pack 藏进当前 GEMM2 MFMA shadow。**选对轴**(GQA head 轴有肉 +1.4%,dt 轴常负);dq iglp_opt(1) 同理(+0.5% bit-identical,别和回归捆一轮被埋)。
- **occ-1 核交织**:QK→softmax→PV 三段交织是 occ-1 藏延迟唯一机制,别用批-2-tile 打断(K=32-PV 判死)。
- **s_setprio(1/0) 包 MFMA-dense 的 GEMM2**:叠结构改动上 +1~2%;单独/GEMM1/连续 span 常中性或负。

## 步骤 3 — softmax / exp2 数值(attention 特有,详见 08)
- **online-softmax 在 log2 空间**,吃硬件 `v_exp2`(见 08)。
- **exp2 三档**:Schraudolph 位操作(最快 ~35dB,单魔数,DMA-确定)/ poly deg-3(SNR>50 但慢)/ v_exp2(精确 quarter-rate)。exp2 是 latency-bound attention 的 operand-bubble 核心;缩短它(降阶 minimax)量级小。
- **lse 预缩放格式必须匹配 exp2 模式**:Schraudolph→`lse_s23 = lse·(-log2e)·2^23 + (127·2^23−486411)`;poly/精确→平 `lse·(-log2e)`。**dq/dkdv/fwd 默认模式可能不同,喂错格式→全 NaN**(见 02)。

## 步骤 4 — occupancy 现实检查(fwd/bwd 通用,常是硬墙)
- **强制拉占用率几乎总 DEAD**:latency-bound 下 occ-1/occ-2 都填不满 wave,强抬 waves_per_eu 必 spill/掉速(dsv4 we3 实测 3× 慢)。
- **occ-N 硬墙**:VGPR 降不到阈值(hd64 bwd ~≤168 才 occ3)免谈;大 D 的 o_acc(fwd)是 flash 2×,q_packs/o_acc 锁 occ-2。拆核/fp8 降 VGPR 的代价(双读/破 SNR)通常 > 占用率收益(hd64 拆 dV-only -36%、fp8 GEMM1 SNR 28.2<34)。
- 别 co-hold 两套累加器;transient vs loop-carried 的 register 账分开算(见 09)。

## 步骤 5 — 确定性(若目标要 bit-reproducible)
- **构造性确定**:单 WG 独占输出 tile + 无 float atomic。bwd dK/dV 跨 WG 归约用 **split-K workspace `[B,kv_split,Sq,H,D]` + host 固定序 fp32 sum**(不用 bf16-atomic,atomic 破确定+破 SNR)。
- fast 路径要确定 → 只能 Schraudolph(DMA 确定);poly-dq flydsl `buffer_load_lds` 有 WAR race → 非确定(见 pitfalls/04)。
- 确定性把「放宽 det 换 -MFMA」封死 → 长收缩维/长 Sq 最后缺口常只剩「research 级更短 exp2」,需上层裁决。

## 通常 DEAD 清单(latency/store-bound attention,别重试)
swizzle/pad/bank(port 富余)、LDS 预取/双缓冲(occ 已藏 DMA;长 Skv 偶边际)、强制占用率/AGPR、DMA-免-reg(scattered-gather 用 register-prefetch 最优)、fp8 GEMM1(短收缩维破 SNR)、bigger tile(register-walled)、q_split 过大(K/V reload 冗余)、dual-layout-KV 消 bank、per-tile 重做 register-transpose(不摊薄)、query-blocking 减 store(占用损失>store 省)、lazy-rescale 单拆(pstore 冲突)。逐条实测出处见 pitfalls/12(dsv4 fwd+bwd)+ pitfalls/13(hd64 dense bwd)。

## 相关卡
- 经验/死胡同:pitfalls/12(dsv4 sparse-MLA fwd+bwd)、pitfalls/13(Meta hd64 dense flash bwd)
- 数值/跨 lane:methodology/08(online-softmax、XOR shuffle、DPP);dual-wave:[[reference_flydsl_flash_gfx950_dualwave]]
- 正确性坑:methodology/02(exp2/lse 格式 NaN、det、JIT 缓存)
- LDS 转置读/bank:methodology/05(ds_read_tr16、pack-128、bank 红鲱鱼)
- profiling:methodology/03(PMC bound 判定、subtractive/SKIPST 探针)
- occupancy:pitfalls/01、methodology/04
