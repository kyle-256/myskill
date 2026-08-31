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

---
来源: project_gptoss_fused_mlp_delta0_campaign.md, project_fused_mlp_padnk_campaign.md, project_fused_mlp_epilogue_5pct_ceiling.md, project_swiglu_epilogue_dword_merge_idea.md, project_attn_a16_port_to_on_main.md, project_gptoss_wgrad_deepk_nb8_collapse.md, 07-epilogue-addressing-transpose.md
