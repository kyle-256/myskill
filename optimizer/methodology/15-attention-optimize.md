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
- **GEMM↔softmax 并行(dual-wave)**:唯一让 GEMM 与 softmax 真并行的解 = 上游 `flash_attn_gfx950` 两 wave-group **时间复用**(s_barrier 错相位)+ cluster 流水 + `sched_group_barrier`(MFMA 0x8 / VALU 0x2 / **EXP 0x400**,exp2 是 0x400 非 VALU)+ lazy-rescale(见 [[reference_flydsl_flash_gfx950_dualwave]])。重、latency-bound **fwd** 的上限杠杆。
  - ★**"把 exp2 塞进 MFMA 阴影"这条不迁移到 occ-2 的 bwd**(gpt-oss hd64 round-7/8 三次独立实测):
    ① `sched_barrier(0)` 强制交错、ISA 确认 MVVVV 模式 → wall **0.00%**;② 精确计数
    `sched_group_barrier` 把 MFMA:TRANS(0x400)按 1:2 / 2:4 / 4:8 交错、ISA 确认 policy 生效
    (nop_cyc 222→85)→ **+0.26 / −0.04 / −0.07%,全在噪声内**;③ 给 dkdv GEMM1 加
    `s_setprio(1)` 抬 MFMA 发射优先级 → **−3.3%**。根因:occ-2 下同 SIMD 常驻的**另一个 WG**
    已经把阴影填满了(pitfalls/13 §185 同一机制),再排自己的指令不产生新并行,而抬优先级
    等于饿死那个 peer。⇒ **occ-2 bwd 上 wall ≈ 两个 wave 的发射周期之和,只有"减指令"能动,"重排"不能。**
    减法探针定价:砍掉 dkdv GEMM1b 每 trip 128 条 `ds_read_b128` 只值 **+0.85%**(≈1.8 cyc/read
    的纯发射成本)⇒ LDS 延迟本来就藏住了,别再投资"藏 LDS 延迟"。
  - ⚠⚠ **BWD occ-2 上 dual-wave/8-wave/warp-spec 常 net-negative,先证再建(决定性,2026-07-21 dkdv 自主 campaign measure-closed)**:占用率=2 的 baseline **本就有两个独立 WG 机会性共驻同 SIMD → 免费享无屏障的跨-WG dual-wave overlap**(一个 WG 卡 barrier 时另一个照发 MFMA 填气泡)。把 4-wave 两独立 WG 合成一个 8-wave 单-WG(warp-spec/stagger)= 把这**免费 overlap 换成带屏障税的组内 overlap**:8-wave NT=2 屏障耦合 MfmaUtil 53→40(-11%);+stagger 错相位能抢回到 1029(+3.5%,证 stagger 机制真有效)但**天花板 1029 仍 < 4-wave baseline 1116**。→ **occ-2 latency-bound bwd 上,dual-wave 只能逼近 baseline 已免费拥有的、无法超越;别投全套重构**。判据:先量 baseline 是否已 occ-2 双-WG 共驻(是→dual-wave 大概率亏)。fwd 常 occ-1(无跨-WG overlap)才是 dual-wave 的正场。
- **PV 的 tr16 转置读常是 fwd 头号 LDS-read 成本**(dsv4 去 pad 掉 60-67%);b128 减半读被 LLVM "Cannot select" 挡。
- ⚠ **lazy O-rescale 单独移植常 net-negative**(dsv4 MLA cr4 -31%,pstore 流水冲突)——它是 dual-wave 套件的一部分,别单拆。

## ★★ 步骤 0.5 — 先审 grid / 派发映射层(2026-07-27 meta hd64 实测:这层值 +7%,kernel 体内只值 ~1%)
> **这是本 playbook 里被补上的最大缺口。** 我曾在 kernel 内部(tile 大小、双缓冲、遍历方向)反复调几周只拿到约 1%,
> 而下面三条加起来 **+7%**,全在 kernel 之外、纯索引/顺序改动、输出 bit-identical。**动 kernel 体之前先过这三问。**

1. **co-resident 的 WG 之间复用的是哪一份数据?让那一维在 XCD 内相邻。**
   MI355X/MI350 是 **8 个 XCD、L2 各自私有**。`xcd = block_id % 8`(片上实测确认),让每个 XCD 拿一整块
   `(batch, kv_head)` → 共读同份 K/V 的 GQA WG 落在同一 L2 slice。dq L2 hit 86.5→94.8%、miss −62%;dkdv **+2.81%**。
   ★**内层最快轴对不同 kernel 是相反的**:dq 要 **kv-head 相邻**、dkdv 要 **q 位置相邻**。选反 = **−2.5% vs +2.4%**。
   ★门控 `num_kv_heads % num_xcd == 0`,双射性**离线穷举验证**后再上机(naive remap 曾直接 GPU-fault 并被误记为"方向死")。
2. **派发顺序就是 list-schedule 顺序 —— 先取 trip-count 剖面,再按剖面形状选顺序。**
   因果掩码下每 WG 工作量 `(q_tile+1)*BLOCK_M/BLOCK_KV` 单调递增 → 降序 q_tile 派发 = **+2.50%**。
   ★**别假设 in-order 已最优**:同一份代码里 dkdv 的 in-order 恰好已是 LPT,而 dq 是反的。可先用"N slots/XCD 贪心"离线模拟排序候选。
   ★**剖面不单调时"升序 vs 降序"是错的二选一**:矩形因果 + 有限窗口的 dkdv,live band 的 trip 数是**帐篷形**
   (两端 2、中间 16),升序与降序同样差(都以最窄的 band 起手),正确形式是**从帐篷中点向两侧走**
   (`i` → `mid ± i/2`,O(1) 下标算术,输出逐位不变)= **body −4.3% / wall −2.7%**(2026-08-15 gfx950 SWA bwd)。
   ⇒ 顺序选择的输入是剖面形状:单调 → 降序;单峰/帐篷 → 中点向外;恒定 → 动它没有钱(同 campaign 的另外三格 in-order 已最优)。
   ⚠️ **但离线贪心 list-schedule 模型给的是上界,别据它下注**:同一格上,把 8 个 family 的 live 段互相错相位(消掉 live 段
   同时涌入)在贪心模型里预测 body −3.8%,12 次跨进程交错实测 **−0.25%(噪声内)**;另一形式(相邻 family 的 live 段
   首尾相接)模型预测 −3.8%、实测 **+0.4%**。模型把 dispatcher 当成"每 XCD 一条独立队列 + 纯 trip-count 代价",
   而零工作 WG 的掩护、跨 XCD 的 DRAM 争用都不在模型里 ⇒ 剖面形状这一阶效应可信,二阶的相位重排要按实测判。
3. **padding 落在最贵还是最便宜的 tile 上?**
   `num_q_tiles*BLOCK_M` 超出 `seq_len_q` 的部分,若锚在 row 0 则**全部浪费在因果范围最长的末 tile**。
   把原点下移 `floor(pad/BLOCK_KV)*BLOCK_KV` 让 overshoot 落到 tile 0(最短)= **+0.96%**(kv-block 访问 7481→7396)。
   ★必须是 BLOCK_KV 整数倍且 `BLOCK_M % BLOCK_KV == 0` 以保持对齐;首 tile 夹到 row 0 并加 owned-end store 界(共享行重算但只写一次,det 不变)。

4. **想把辅助核(reduce/odo)藏进主核 = 先算 fill quantum,再按"body 税 + 尾巴"计价,别看辅助核自己的时间。**
   主核若是 1 WG/CU(LDS > 一半),**一个 chunk 少于 CU 数就白付一整轮 `T_wg`**(实测 573~690 µs),
   而要藏的 reduce 全长可能只有 300 µs ⇒ 细分永远亏。★但**对齐 fill 也救不了 D128**:
   把 chunk 做成**恰好 2 个整 fill**(512 WG)后,body 仍 **+18%**、共驻 reduce 自己慢 **3×**,
   净 wall −0.6%(噪声内)⇒ r14 记的"亏在 0.75 fill 的额外一轮"是**不充分**解释,真绑定项是
   body 的寄存器/发射位压力(D128 body 498/512 ⇒ 税 18~23%;同代 D64 body 449/512 ⇒ 税只有 3.6%,
   净赢 85 µs)。⇒ **共驻可行性按"主核寄存器余量 + 主核受干扰后的时间"判,不是按 fill 对齐或辅助核字节判。**

★ 判这层改动看 **`SQ_WAIT_ANY` / `SQ_VALU_MFMA_COEXEC_CYCLES`,不要看 TCC hit%** —— hit 率大涨可以值 0 wall。

## BWD 专属杠杆:减 MFMA(唯一够大的结构杠杆,优先级最高)
1. **审计可丢弃的 GEMM / 全局修正项**:第二 GEMM、rho/R 全局 renorm(精度代价常 ~0.2dB 非破门)。dq drop-B-GEMM 实测 **+9.8%**。**永远先做**,纯减法量级大。
2. **融掉全串行辅助核**:`delta=rowsum(O·dO)`(odo)、interm restage 等串行且随 Sq 放大 → 融进主核。odo delta 融合 **+5.3%**(长 Sq 最大)。★坑:融合核直读 O/dO,wrapper 必须 `out/dout.to(q.dtype)` cast,否则喂 fp32 O → 崩/NaN(见 02)。
3. **核间共享重算 GEMM1**(dq+dkdv 共用 S=K@Q^T,理论 -25% MFMA):**三重陷阱常吃光收益,先 spike 再建**:①确定性陷阱(q-outer dK/dV q-归约不用 atomics 无法有界+确定→必 KV-outer);②register 墙(两套累加器共驻 spill→常被迫 BLOCK 减半);③workspace 流量(~GB split-K + 减半 tile per-tile 惩罚)。hd64 实测融合核 **1.95-2.55× 慢**→短收缩维判死。设硬 abort 门。

## 步骤 2 — 藏 MFMA 依赖延迟(fwd/bwd 通用,减不动 MFMA 时)
- **16x16x32 拆链**:32x32 长串行累加 → 4 条独立 16x16 → MFMA-latency ILP ×4,累加器 VGPR 不变(dkdv 581→632TF)。
- **operand-bubble 软流水**:下一迭代 exp2/pack 藏进当前 GEMM2 MFMA shadow。**选对轴**(GQA head 轴有肉 +1.4%,dt 轴常负);dq iglp_opt(1) 同理(+0.5% bit-identical,别和回归捆一轮被埋)。
- **★冷 load 寄存器预取(head 轴,bwd hw-exp 实测 +1.81%)**:每 q-block 的 head-0 lse/delta 是冷 load;把**下一 head 的 lse/delta buffer_load 在本 head GEMM2 的 `dt==DT-1` 处发射**,+32 VGPR carry 只叠最后一个 dt 的 MFMA(spill-neutral、守 occ-2、bit-identical)→ 藏掉 consumer latency。dkdv hw-exp 1114→1134(commit c619847)。⚠**只藏 latency 不消 issue-stall**:若该 load 的 stall 是 VMEM-queue issue-stall(被巨量 Q/dO DMA 占满,ATT 查 col[8]),prefetch/晚发射类结构性无效;cross-q_start 跨 block 提前=neutral(head0 已被自身 Q/dO DMA 的 waitcnt drain 覆盖)。Q/dO DMA 本身 HBM 32B-fraction=0%(全 coalesced)= 纯 latency、occ-2-hidden,别刷 coalescing。
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


## ★ 判"某杠杆对某 regime 无效"前,先确认那个 regime 的其它杠杆状态一致(2026-07-30 实测)

hd64 fwd 收官时踩到:GQA sharer merge 在 full-causal 上 +1.8%,在同形状 SWA 上 **−12%**,于是差点被门控成
"窗口下不划算"。真因是**当时 SWA 还被挡在 `_FMAX0` 固定 max 之外**,仍走 online softmax;把固定 max 放开给 SWA 之后,
同一个 merge 在 10 个形状里 8 个转正,那条窗口门控随即被撤销。

⇒ **regime 之间的杠杆是耦合的**:A 在 regime R 下判负,可能只是 R 缺了 B。正确做法是先把公共杠杆对齐,再测 A。
⇒ 附带教训:一个门控条件如果**注释里给不出理由**(这里 `window_left < 0`),多半是当初圈定在被优化的形状上的
**scope 残留**,不是物理约束 —— 值得当成开口去试。softmax 的平移不变性与掩码无关,SWA 完全可以吃固定 max。
⇒ ★**同一 `window_left<0` scope 残留在 bwd 上再次坐实**(2026-08-09):fused bwd 的 `MASK_SKIP = FUSE_DQ and
window_left < 0` 把 causal band-trim 只留给 full-causal,SWA 下每 kv-band 把全 causal q-loop 算完再全掩=纯浪费,
使我们的 SWA wall dividend 只有 7.3×(GB300 9.54×)。加 window-aware q-loop 上界(裹 `const_expr(window_left>=0)`
保 full 字节不变)把 SWA 216→298 eff-TF、dividend 拉到 10.1×(反超 GB300)。见 pitfalls/13 §2026-08-09 GB300 对标。
