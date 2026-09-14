# 算子融合:把 activation/SwiGLU 折进 grouped GEMM 的 epilogue

> 类别: 方法论 · 主题标签: gfx950, operator-fusion, fused-epilogue, swiglu, glu, dglu, grouped-gemm, mlp, pitch-vs-real, acc_mode, deploy-measurement, padN-padK
>
> 这张卡讲**怎么把一个 elementwise/gated activation 融进它相邻的 grouped GEMM**(gpt-oss-20b MoE MLP 实战:fc1+SwiGLU=glu、fc2+SwiGLU-bwd=dglu)。存策略/转置本身在 [07](07-epilogue-addressing-transpose.md);这里是**融合的决策、结构改动、以及验收口径**——四条纪律里三条都是"测量口径",踩错直接判反。

## 1. 什么值得融、为什么(epilogue fold 免中间张量 HBM 往返)

- **融合的收益不在算力,在访存**:把 activation 折进 GEMM 的 epilogue,`[M, I]` 的中间张量**永不落 HBM**——它只喂给下一步的量化器,就地在 epilogue 里做完 SwiGLU + 量化,省掉一整趟 `[M,I]` 的写回+读入。
- gpt-oss-20b MLP 折成两个 grouped fp8 GEMM:
  - **glu** = fc1(gate_up)+ SwiGLU **前向**:GEMM 出 `[M,2I]`,epilogue 里做 `silu(gate)*up` → `[M,I]`,顺手量化成 fp8 喂 fc2。
  - **dglu** = fc2 的 dgrad + SwiGLU **反向**:dgrad GEMM 的 epilogue 里乘上 SwiGLU 的局部导数。
- **activation quant 该 stage 在哪**:激活的 fp8 量化**放进 GLU 算子内部**(它下游只有量化器一个消费者),让 epilogue 把 cast 折进去、彻底省掉 `[M,I]` 的往返。别在 caller 里先算 activation 再单独量化。

## 2. ★★ 融合的验收口径 = total-vs-total,不是 per-op TF/s(救过整场的纪律)

- **判据**:融合核 vs 非融合的对比,必须是 **融合-total(fc1+act+fc2)vs 非融合-total(fc1 + 独立 act + fc2)**。融合核把 activation 的活也扛了,所以拿它的 per-op TF/s 去比一个**零 epilogue 的 plain GEMM** 是**伪目标**——你在拿"多干了活的核"比"没干那活的核"。
- 实测钉死:融合 glu 比 bare GEMM 慢 ~14%、dglu 慢 ~28%,但这 gap 是**结构性的**(见 §9),不是低效。**把 total 折进来算,融合是净赚**(activation 的那趟 HBM 往返被省了)。
- ❌ **别把 spd_fuse / per-op ratio 当门禁去追 parity**——会把一个真增益判成"还差 5%"而继续瞎调。

## 3. ★ avg-of-two-kernels 指标会掩盖单核的真增益

- 一场 campaign 若把**两个融合核(glu+dglu)的平均**当分数,而你这次只动了**其中一个**核,它的真增益会被**另一个核的噪声**淹掉,聚合门抓不到。
- **判据**:只动单核的改动,看**那个核自己的 ratio/ms**(多读数看重叠),不看聚合 `spd_fuse`。实测踩证:glu 单核 `+0.55%`(5 读数无重叠、真增益)被 dglu run 的噪声掩盖,平均门 60 轮纹丝不动;换成看 glu 自己的 ratio 才认出来。
- 相关:campaign 指标设计见 [16](16-campaign-harness.md);单核判据同理适用于任何"多算子平均"的评分。

## 3.5 dQ 折叠:同一"生产者/消费者自旋"陷阱的另一面

- 融合不止 epilogue。attention 反向把 **dQ 的 partial 累加折进主 pass**、grouped wgrad 把 **deep-K split 的 partial 折进 in-GEMM**,都是"把两趟变一趟"的融合。
- ⚠️ **这类 in-GEMM fold 会引入生产者/消费者自旋**:一个越过切分闸门的组/tile 会让另一半空转等它,**结果仍正确但慢 10–2000× 且抖动**(SNR/det 门抓不到,因为数值没错)。踩证:down wgrad `_FUSE` 恰好一个专家越闸即塌陷;修法=关掉那个 fold,代价 ≤3.2%。
- **判据**:in-GEMM fold 落地前,先确认**所有组都在闸门同一侧**(`k≥2 组越闸`才干净,或 `TILES_PER_GROUP≥NCU` 使 fold 恒 False 而免疫);上线后盯尾延迟抖动,不能只看 SNR。

## 4. ★★★ gated activation 融合的结构核心 = pitch-vs-real 劈分(为什么不能纯零传播)

这是**融合一个"门控"激活(gate/up 在同一维交错)时最容易踩、也最本质**的一条。

- **场景**:w1 输出 `[M, 2I]`,gate 半 `[0,I)` 与 up 半 `[I,2I)` 交错;若把 per-half 的 I pad 到 `Ip`(对齐 128),权重变 `[M, 2Ip]`。
- **纯零传播(只按权重形状推 I=Ip、pad 列填零)对 fwd/dgrad 都对,但 grad_w1 会死**:w1 的梯度 `[G, 2Ip, H]` 真实内容在行 `[0,I) ∪ [Ip, Ip+I)`(gate/up 各半),**中间夹着 pad 行**;plain 的 `m_real` 只能截前 `2I` 行 `[0,2I)`,截出来是"整段 padded gate + 半段 up",对不上 tight 的 `[G,2I,H]`。
- **正解 = glu/dglu 核内部按 pitch(Ip)算、按 real(I)存**:把 `glu_i` 劈成两个量——
  - **pitch = Ip**:用于权重形状、compute 循环上界、**up 半的列偏移**(up 起点 = pitch,不是 real)。
  - **real = I**:用于 **store 宽度 `c_n`、列掩码**、以及 gate/up 存进**紧凑的 2I**。
  - §crux:**up 半所有 offset 用 padded Ip,所有列掩码用 real I,pad 段必须写零**。act/intermediate 存 tight(`act[M,I]`、`l1[M,2I]`),grad_w1 就天然 tight。
- **层次(改哪几层)**:caller(设 `i_real`)→ glu/dglu impl(加 `i_real`/`k_align` 参数)→ flydsl wrapper(放宽 assert、传 pitch/real)→ kernel 几何(`_bn_rows`/`bn_i`/up-offset 用 pitch,`_nb_c`/`_col_safe`/掩码用 real)→ store(pitch vs real 劈分)。
- **干净维不用动核**:H 是连续维,补 Hp 不涉及 gate/up 交错——w2 只补 penult(=dglu 的 K),不碰 last(=dglu 的 N=I),dglu 天然吃 padded-K、输出 I tight,**零核改动**。**只有 ragged 的 per-half 维(这里是 I)才被迫动 glu/dglu 核**。

## 5. acc_mode=vgpr:融合 epilogue 的累加器放 VGPR

- 融合核的 epilogue 是累加器的**唯一消费者**。把 fc1(glu)前向累加器留在 **VGPR(mma mode 3)**,epilogue 就能把它当 **VALU 源**直接读,省掉 AGPR 累加器需要的 `v_accvgpr_read` 搬运层。
- 前提:NT 的喂料是 `S2RLoader`(不是 transpose asm),**寄存器预算不变**,所以这步是白赚。
- ⚠️ 值定了就**内联字面量**(`acc_mode="vgpr"`),别留 `_GLU_ACC_MODE` 之类的 DEV 常量到 ship 版(用户红线:确定的值不搞变量)。

## 6. padN+padK:融合核也要把收缩/特征维补到 128

- 与 [07](07-epilogue-addressing-transpose.md) 的对齐同理,但这里是**为融合核补**:`H=I=2880`,`2880%128==64` ⇒ 未 pad 的 fp8 GEMM 每条 cache line 被劈成两个 L1→L2 请求(**+50% 流量,MFMA 数学逐位相同**)。
- **做法 = pad 下沉进 `QuantizedTensor.quantize`**(`pad_align_last`=K、`pad_align_penultimate`=N),用 `real if real!=pitch else None` 恢复真形状,**存的张量保持 tight(copy-free、梯度 tight)**。对齐 shape 的 pad 是逐位 no-op。
- 融合场景的 6 个 pad 点:x(`pad_align_last`)、w1(`pad_align_last`=H)、w2(`pad_align_penultimate`=H + `pad_align_last`=I)、grad_out(`pad_align_last`)、glu impl 的 `k_align`。
- 实测(gpt-oss-20b MoE,H=I=2880/G=32,pad128 vs pad0,FlyDSL 同进程 A/B,**预量化权重**):fwd −8.2%、bwd −2.6%、fwd+bwd −4.6%(1.05×)。

## 7. ★★ 测量部署代表数:权重量化是 bench 假象,反向不重量化权重

- **陷阱**:bench 里传 bf16 权重 ⇒ **每次调用都重新量化权重**(~15% 的 fwd),部署里权重**一次性预量化缓存**(`QuantizedTensor` / `QuantizedTensorPair`),不吃这块。
- **判据**:报融合 MLP 的部署代表数,要**预量化权重**(传 `QuantizedTensor`,x 仍 bf16——激活每步都变、无法预量化,是动态 fp8 固有开销)。
- **反向不受影响**:backward **复用 fwd 存下的 fp8 权重**,从不重量化权重(只重量化 grad_out)⇒ 预不预量化只动 fwd,bwd 本就是部署代表数。实测:预量化把 fwd 4.28→3.47 ms(证实那 ~15% 就是重量化假象),padnk 的 fwd 相对增益从 3.9%→8.2%(分母去掉固定开销后占比放大)。
- ⚠️ 别把 bf16-leaves 的绝对数当部署数报;别把"重量化那 15%"算进 pad 的收益。

## 8. 把 plain GEMM 的写侧优化移植进融合核(paired store)

- plain grouped GEMM 上验过的**写侧 dword 合并 / 成对列 store**([07](07-epilogue-addressing-transpose.md) §epilogue)**能移植进 SwiGLU 融合核**,而且往往**更肥**——融合核有 gate/up/act 三条流的写压。
- 实战:glu-fwd 移植后 **+3.7%**(比普通 NT 的 +1.5% 更肥),逐位相同。关键点:
  - `col_safe=False` 靠 **I 偶 + pair 落偶列 → pair 粒度掩码** 解决。
  - gate/up 共用同一 `gl_off_b`(up 仅 `+glu_i*KS`),配对守恒。
- dglu 走 cshuffle 通路(`StoreCdSwiGLUCShuffle`),若本就是 128b `dwordx4` 则**无需移植**;`StoreCSwiGLU.store_pair` 才是 glu 的 pair 分支入口。
- **判据**:融合核的普通 NT/NN 子路径(fc2、grad_x)会**自动**吃到 plain 的写侧优化;要手动移植的只有带 SwiGLU epilogue 的那条(glu/dglu)。

## 9. ❌ 别再试:融合核追 per-op parity(对 bare GEMM 的 5% gap 是结构/物理不可达)

- **dglu −29% ≈ 强制 2× HBM 流量**(roofline floor −24% @ 8 TB/s):dgrad 融合必须多流一趟。
- **glu −15% = activation 第三条流串行写**,occ=1 盖不住。
- occ 假说已被否:`waves_per_eu` 2/3/4 + esplit 0 对 glu **零效**(累加器在 AGPR/VGPR、真占用已顶)。
- ⇒ **正确口径永远是 融合-total vs 非融合-total**(§2),单算子 TF 比 plain GEMM 是伪目标。**别立"把 gap 打到 0 / 追 parity"的项**;要立就立"融合-total 净赚多少"。
- ⚠️ 写 goal 时**禁"天花板/不可达"口径**(用户红线):给"结构上多一趟 HBM 流量 + roofline floor + occ 实测已顶"这三条**根因 + 数据**,而不是"到头了"。

## 10. ★★★ 另一类融合:**换并行轴**去掉 topk 倍冗余(MoE quant+sort,2026-09-11 GLM-5.2 EP4 r6,+1.46%)

前 9 节讲的都是"把相邻算子折进 epilogue"。这一节是**同一个 kernel 内换并行轴**,
适用签名:**一个 kernel 的两个输出用不同的索引空间寻址**。

- **发现**:`fused_mx_quant_moe_sort_kernel` 按 **sorted row** 并行,但 stage1 里
  fp4 payload 写在 `token_id` 偏移、只有 e8m0 scale 写在 `sorted_row` 偏移
  ⇒ 一个 token 路由到 topk 个专家就被**重量化 topk 次、把同一份 payload 重写 topk 次**。
  GLM-5.2 EP4 b64:64 token 展开成 1760 行 = **27× 冗余**(10.8M 次转换 + 5.4 MB 重复 store,
  真实需求 393K 次 + 196 KB)。判据:读 kernel 时**逐个输出看它的地址用哪个索引**,
  两个索引空间不同就一定有冗余或一定有 gather。
- **⚠ 先别高兴:把冗余算掉本身可能一分不值。** token-major 重写第一版 rocprof
  **7.12 → 7.12 µs,一模一样**;bench 反而 −0.5%。因为这个 kernel 从来不是 work-bound。
  拆开定价(把子阶段逐个置零、只读时间,`rel_l2` 变 NaN 无所谓)b64:
  | 子阶段 | 代价 |
  |---|---|
  | 纯量化(冗余已消) | **比基线快 3.38 µs** ← 真实收益在这 |
  | 扫 `sorted_ids` 找本 token 的行 | +1.72 µs |
  | e8m0 byte scatter(swizzle) | +1.92 µs |
  三项净额 ≈ 0。**"消冗余"只是把预算腾出来,新引入的两个阶段会全额吃掉它**;
  这两项都不是新算法必需的开销,是**实现细节**,必须单独打掉:
  1. **扫描:动态 `scf.for`(按 `num_valid` 定界)= 每趟一次串行全局 load。**
     7 趟 ≈ 4 µs。改成**按静态分配长度展开的 dwordx4 扫描 + 钳位尾部索引**
     (重读几行是幂等的,因为"把 row 写进它的 slot"可重复执行)⇒ 所有 load 一次发射。
  2. **scatter:每个 slot 一趟、只有 `scale_n_local` 个 lane 活。** 改成
     `(slot 组, 列)` 二维摊到整个 block(`slots_per_pass = block/scale_n_local` 恒为 4),
     并把 `rows[]`/`sorted_weights` 的 load **全部提到循环外一次发射**。
  修完 **7.12 → 4.28 µs**,四个 bucket 全正,配对 A/B **+1.46%**(3/3 同号、零重叠)。
- **★ 别用"拆成两个 kernel"去消冗余**:同一轮先试了仓库里现成的 HIP split 路径
  (`per_1x32_mx_quant_hip` + `mxfp4_moe_sort_hip`),**两个核 3.48 + 5.04 = 8.52 µs,
  比融合的 7.12 还多**,分数只 +0.34%(b8 反而 +2.7 µs)。这台机器上一次 launch 的
  边际价 ≈ 2.5 µs,而 `mxfp4_moe_sort_hip` 那 5 µs 几乎全是"先读 `sorted_ids` 再按它
  gather scale byte"的二级依赖链。**融合版把 scale 留在 LDS,直接消掉这条链**——
  这才是"融合"在这里的真实价值,不是省那趟 HBM。
- **并行度旋钮的方向可能与直觉相反**:给 token 再切列(`nsplit` 个 block 协作一个 token)
  能把 grid 从 64 抬到 384,但**每个 block 都要重扫一遍 `sorted_ids`** ⇒ 扫描代价正比于
  `nsplit`。实测 nsplit ∈ {1,2,3,4,6} 分数 **1.0579 / 1.0571 / 1.0563 / 1.0552 / 1.0521
  单调**,最优是 **nsplit=1 + block=768**(一个 token 一个 block、量化和扫描各一趟)。
  ⇒ 凡是"每个 block 都要读一遍某个全局表"的结构,**先算 `grid × 表长`,别只看 CU 占用**。
- **下一档杠杆(r6 未做,r7 已做见 §11)**:让排序核在 p23 里顺手写一张
  `inv[token*topk + slot] = sorted_row` 的逆表,扫描退化成 topk 次直读(估省 ~1.7 µs);
  再往上是把量化折进 p0v2、把 scale scatter 折进 p23,整个 quant1 launch 消失(~4.3 µs)。

## 11. ★★★ 逆表是**共享产物**,不是单个消费者的私器(GLM-5.2 EP4 r7,+1.62%)

§10 末尾那条"写逆表"的预测在 r7 落地了,但**定价口径和受益者都和预测的不一样**——
两条都是可复用的判据。

- **落法(零改管道)**:`num_valid_ids` 本来就是个多字段元数据张量(`[0]`=padded 行数、
  `[1]`=token 数)且**已经**同时接到排序核和量化核上。把它**超额分配**成
  `2 + M*topk`,逆表就藏在两个标量后面,`has_row_inv = numel() >= 2 + M*topk`
  由消费者自己判定 ⇒ **不动任何函数签名的 5 元组返回契约**。
  ⇒ 判据:要在两个 kernel 间传一张新表时,**先找现成的元数据张量搭车**,
  别急着改返回契约(改契约会牵动所有共用该排序入口的模型)。
- **种 sentinel 不需要新 launch**:p0v2 的 Phase 2 里每个 assignment 恰好被**一个**
  专家 block 认领(`is_mine`),且它手上的 `flat` 就是 `token*topk+slot`
  ⇒ 在那儿写 `-1` 就把全表播种完了,一次 store、零额外 kernel;p23 随后覆盖本地专家的项。
- **顺手把零权重语义搬到生产侧**:p23 写逆表时路由权重**已在寄存器里**,
  把"权重为 0"编进 payload 的高位(`ROW_INV_ZERO_WEIGHT`)⇒ 消费者再也不用为了
  判零而 load 一次 `sorted_weights`(那是一条 **row-dependent** 的二级依赖)。
- **⚠ skill 说 X,实测 Y(定价口径)**:§10 估"省 ~1.7 µs",那是**rocprof duration 口径**。
  实测 duration **4.28 → 4.18 µs(只降 0.1)**,但 bench **+0.61%**(b64 墙钟 −1.2 µs,
  3/3 读数零重叠)。同轮 stage2 换法 duration **4.98 → 4.64(降 0.34)**、墙钟却 **−1.9 µs**。
  ⇒ **这类小核的 duration 几乎不反映它的边际墙钟成本**;缩短**依赖链深度**的改动
  在 duration 上看不见、在分数上看得见。别用 rocprof duration 给依赖链改动定价,用分数。
- **★★ 最大的复用点在"另一个消费者"**:逆表原本是为 stage1 的 token-major 核写的,
  但 **stage2 的量化核(row-major HIP)受益更大**(+0.89% vs +0.61%)。
  stage2 每个输入行 = 一个 (token, slot),本来"无冗余",所以 r6 判它不必换轴——
  **但它仍要回答同一个映射问题**,而 HIP 版是按 sorted row 走的:
  `num_valid_ids[0]` → `sorted_ids[i]` → 输入行 → activation load = **3 级串行依赖**;
  换成 block 直接认领输入行 + 一次逆表读,**依赖链只剩 1 级且与 activation load 重叠**。
  ⇒ 判据:**"无冗余"不等于"无收益"**。看一个核该不该改,别只问它算了多少冗余功,
  要问**它的第一条 load 前面压了几级串行依赖**。
- **同一张表还能省掉一次重扫**:p0v2 的 Phase 3 原本重扫一遍 `topk_ids` 来数本专家的
  assignment,而 Phase 2 的 `is_mine` 已经算出了这件事 ⇒ 把计数并进 scatter 那趟
  (`init=[cnt]` 的循环里带状态),+0.08%(3 读数)。**凡是"第二趟重新推导第一趟已知量"
  都值得并回去**,即使收益贴着噪声。
- **⚠ 并行度旋钮:§10 的结论对,但理由要换**。§10 说 nsplit=1 最优是因为
  "每个 block 都要重扫 `sorted_ids`,扫描代价正比 nsplit"。逆表把扫描消灭后
  这个理由不存在了,于是 r7 重测:grid 64 → 256(nsplit=4)**duration 只降 ~0.2 µs、
  分数落在噪声内**,已回退。⇒ 这些核**不是 CU 饥饿,是 launch/延迟主导**;
  结论仍是 nsplit=1,但今后别再用"扫描成本"当理由。
- **下一档杠杆**:把量化折进 p0v2 + scale scatter 折进 p23,使 quant1 的 launch 整体消失。
  按本轮实测的边际价(单个小核墙钟 ≈ 1.2–1.9 µs)定价 ≈ +1%。
  已知两处待解:(a) `P0V2_BLOCK=512` 不整除 `6144/8=768`,需要 2 block/token 的掩码分工;
  (b) b64 时 p23 的 mesh 扫描只有 ~16/512 线程活,byte scatter 前要先把
  (row, token) 对经 LDS 摊到整个 block。
  → **r8 实测该预测为负,见 §12。**

---

## 12) ❌ 实测负:把 stage1 量化折进 p0v2 + scale scatter 折进 p23(GLM-5.2 EP4, r8)

§11 结尾预测 +1%,**实测 −0.57%**(最好变体 1.07050,baseline 均值 1.07613,7 读数)。
已完整实现并过了正确性门禁(逐 bucket `rel_l2` 0.00028–0.00084,与 baseline 同档)。
两半的定价差了一个数量级,这才是这张卡的价值:

- **载荷量化折进 p0v2 ≈ 免费**。p0v2 本来就以 `E=257` 个 block × 512 线程启动,
  而 `M×cols/8` 个 chunk(b64: 49152)只要 96 个 block 就装下 ⇒ 用**扁平 chunk 号
  `g = bid*512 + tid`** 直接铺,`512 % 4 == 0 且 768 % 4 == 0` 保证 MX 组不跨 block,
  **不需要 §11 说的"2 block/token 掩码分工"**。把 load 放在 Phase-1 clear 的 barrier 之后、
  Phase-2 scatter 之前消费,延迟被 mesh 扫描盖住:p0v2 duration 仅 **4.47 → 4.98 µs**,
  却顶掉了 quant1 的 786 KB 读 + 393 KB 写。
- **swizzled scale scatter 折进 p23 是全部的亏损**:p23 **4.18 → 6.71~6.90 µs(+2.5~2.7)**,
  抵掉了省下的整个 quant1 launch。三种线程映射都试了,结论一致:
  | p23 scatter 映射 | score |
  |---|---|
  | 1 线程 = 1 行 × 48 列(12 次 load + 48 次 byte store) | **1.07050** |
  | 同上但 E8M0 先整表进 LDS(把 HBM 往返挪到 sort 之前) | 1.06959 |
  | 1 wave = 1 条 64 B swizzle line(row 在 lane 内变化,全线合并) | 1.05785 |
  ⇒ (a) **LDS 预存没用** ⇒ 这 2.5 µs **不是 HBM 往返延迟**,是 byte-scatter 本身;
  (b) **合并写更慢** ⇒ 这些核**由动态循环的 trip count 主导,不由 transaction 数主导**:
  line-major 让 shared expert 走 24 趟依赖循环,反而比 1 趟 × 48 条散 store 慢 1.3%。
  判据:**给 512 线程的小核排布工作时,先把 trip count 压到 1,再谈合并。**
- **为什么 scatter 搬到哪都要 ~2.5 µs**:EP4 下 b64 只有 ~208 个真实 sorted row,
  但它们散在 65 个本地专家里(shared 专家独占 64 行连续,其余 64 个专家各 2–3 行)。
  一条 64 B 的 swizzle line = **32 个连续 sorted row × 2 列**,routed 专家每条 line 只填得进
  2–3 个字节 ⇒ **稀疏度是结构性的,换 kernel 换映射都消不掉**。
- **⚠⚠ rocprof duration 在这一轮是直接误导的**,不只是"测不出来":
  baseline eager 合计 190.9 µs vs 折叠后 186.9 µs(**duration 说省了 4.0 µs**),
  同一棵树的 graph 分数却是 **182.41 → 182.65 µs(慢了)**。
  §11 那条"别用 duration 给依赖链改动定价"要升级成:**小核的 duration 连符号都不保证**。
  本轮所有结论都只用 4-bucket score 定的。

**下一档(未试,按本轮数据的优先级)**:
1. 只折载荷、scale scatter 留在独立核(改成读 p0v2 发布的 token-major E8M0,不再重读
   activation):p0v2 那半已证明近乎免费,quant1 的 1.2 MB 收缩到 52 KB。
   但注意 quant1 的 4 µs 里带宽只占 0.19 µs,**它是 launch/延迟主导**,估价只有 +0.3%。
2. 把 scatter 的稀疏度打掉:让 p23 的 **zero-fill block**(`k4_grid = E + n_zero`,
   b64 时有几百个闲 block)也参与 scatter。难点是它们不知道 row→token,
   需要 p23 的 sort block 先把逆表落 HBM,而两者之间没有 grid 同步。

---
来源: project_gptoss_fused_mlp_delta0_campaign.md, project_fused_mlp_padnk_campaign.md, project_fused_mlp_epilogue_5pct_ceiling.md, project_swiglu_epilogue_dword_merge_idea.md, project_attn_a16_port_to_on_main.md, project_gptoss_wgrad_deepk_nb8_collapse.md, 07-epilogue-addressing-transpose.md, campaign 20260911_155754 r6/r7

## 10. ★★ 另一种融合:合并**小核**(省的是 launch,不是 HBM 往返)

前面几节讲的都是"把 elementwise 折进 GEMM 的 epilogue,省中间张量的 HBM 往返"。
还有一类完全不同的融合:**把两个相邻的小核(元数据/排序/计数类)合成一个**,
省的是 **launch + 关键路径上的那段固定延迟**。判据和上面那套完全不同:

- **先定价,再动手。定价方法 = 把那个核再启动一次(必须幂等)**,量 wall 的增量。
  这比 rocprof duration 靠谱得多:MoE 排序的 p0v2 duration 4.87us,但重复启动只加
  **2.29us**(b64)⇒ 有一半 duration 本来就被前后核重叠吃掉,**真正能回收的是 2.3us**。
  如果这个数低于你打算付出的复杂度,直接不做。
  ⚠️ 挑一个**天然对照桶**:本例 b8/b16 走的是 oneshot 路径、根本没有 p0v2,
  重复启动后这两桶一格不动 ⇒ 证明测到的就是那个核,不是漂移。
- **别指望"把核内的活砍短"能赚**。同批减法探针实测:mesh 清零、sentinel 填充置零
  **各 +0.03%(噪声内)**。几百个 block × 512 线程的元数据核是**固定延迟主导**,
  只有整核消失才有钱。
- **合核的数据结构不要照搬"单核 oneshot"的做法**。把 `E × T` 的 mesh 整个搬进 LDS
  会在 T 一大就爆(258×2048 = 528 KB/block),这是历史上合核失败的直接原因。
  正解:每个 block 只保留**全 E 的直方图**(E 个 i32 计数器)+ **自己那一行** mesh(T 字节),
  前者用 LDS 原子 `ds_add` 构建(T*topk 次原子摊到一个 block 的 512 线程上 ≈ 免费)。
  FlyDSL 写法:`llvm.atomicrmw(llvm.AtomicBinOp.add,
  fx.to_llvm_ptr(lds_ptr + fx.Int64(idx)), val, llvm.AtomicOrdering.monotonic,
  syncscope="workgroup")`;**不取返回值**,会降成无返回的 `ds_add`,不 stall。
  需要按字节写而 LDS 数组是 i32 时,若**每个字节只有一个写者**,用 `ds_or` 把字节
  OR 进清零过的 word,可以逐位复现原来的 uint8 布局(输出不变 ⇒ 下游 SNR 不变)。
- **★ 合核会把"跨核天然有序"降级成"核内竞态"**。原来靠 kernel A → kernel B 的隐式屏障
  保证的"A 播种默认值、B 覆盖真值",合进一个核后 `gpu.barrier()` **未必对 global store 排序**。
  别赌屏障语义,**改成让两段写的地址集合不相交**(本例:只给被 mask 掉的专家写 −1,
  本地专家的条目全部交给后面的 scatter 写)。
- **验收**:合核改的是"谁在什么时候写",最容易悄悄改掉**输出顺序**。如果下游有原子累加,
  顺序一变数值就变(实测另一处 grid 维序交换让 rel_l2 从 0.0003 跳到 0.0017,逼近 0.002 门)。
  ⇒ 目标应该是**输出逐位相同**,并且用一份纯 Python 参考把**所有 T 分档 × 有无 mask**
  全对拍一遍 —— 合核往往会顺手暴露旁边那条从没被测过的路径。

### ⚠️ 定价之前,先在依赖链上确认这两个核**相邻**

(2026-09-12 a4w4 r12)定价本身便宜又准:a4w4 那两个 `token_major_quant_sort` 分别值
**+2.44us**(quant1,b64)和 **+2.20us**(quant2,b64),各自都够一次合并的本。但它们**不相邻** ——
quant2 的输入就是 stage1 GEMM 的输出,中间隔着整个 gemm1 ⇒ 合并它们等价于 quant↔GEMM 融合,
不属于"合并小核"这条路(本战役还明令禁止)。唯一合法的相邻对是 (sort, quant1),而 r8 已量出 −0.52%。
⇒ **"每个核值 2.3us"和"这两个核能合"是两件事**:先画依赖链找出真正相邻的对,再花时间定价。
