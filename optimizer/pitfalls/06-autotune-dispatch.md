# autotune / dispatch 纪律与踩坑合集

> 类别: 踩过的坑 · 主题标签: autotune, dispatch, cache, overfit, autotune-dispatch, persistent-vs-nonpersistent, nt-kernel, l2-swizzle, persistent-kernel, vmcnt_hint, tail-cliff, tile-size, cache-key, real-training-gain, MoE-skew, benchmark-overfit

## autotune/dispatch 纪律：禁 id(tensor) cache、can_handle 不 raise、per-shape 过拟合

**cache 纪律**
- ❌ 别再试：autotune 里 result cache(缓存 quantize/transpose/group_offs 的结果)和任何 **id(tensor)-keyed** cache。id(tensor) 会复用/回收——同一地址可指向不同数据,cache 命中即数据错误。(Python 对象被 GC 回收后,新 tensor 可能拿到同一 `id` 值,cache 便把旧 key 命中到新数据。)
- 只能 cache 两种东西：compiled object(`flyc.compile` 产物)和 launch closure(闭包**不含 data**,只含 launch 参数)。
- cache key 用**纯静态维度**：`(op, N, K, G, M_total, out_fp16, cbsz, blgp)`。WHY：这些是唯一决定最优 kernel 的量,不含运行期 buffer 身份。

**per-shape 过拟合(overfit)**
- ❌ 别再试：dgrad NN 按 c_n **用启发式直接算** per-shape `num_xcd`。实测 **−0.5%** 负杠杆——
  当时没有干净物理阈值,属于纯 overfitting。
- 原则：**没有清晰物理阈值的 per-shape 调参一定 overfit**,别做。
- ✅ **但把 `num_xcd`/band 作为候选丢进 per-shape autotune 去 race 是赢的**(2026-08-09 dense fp8 NN,
  同 session A/B 计入 **+1.66%**,是那一场十轮里最大的一笔)。**两条不矛盾**:
  差别在**你是"算"还是"量"**,以及**这次有物理阈值**——
  M-band 复用 B 的 N-stripe(`K × BLOCK_N`),stripe 一旦超过单 XCD 的 L2 slice(gfx950 ≈ 4 MiB),
  把 tile 聚到一个 XCD 就颠簸,硬件 round-robin 反而更好。
  窄-N 与宽-N 要**相反**的 band,两族差距远大于竞速噪声 ⇒ 两族都放进候选表让它自己选。

**wgrad gate 用 per-group m_total/G**
- wgrad gate 必须用 **per-group contraction = m_total/G**,不是裸 m_total。
- ❌ 别再试：旧 gate `m_total<=2048` 坑高-G MoE。反例 G=8 / M_g=512 / m_total=4096 被误判走 masked,而实测 persist 反而 **+13%**。根因:m_total 混了 G 的贡献,per-group 收缩维才是真正决定 kernel 选择的量。

**Constexpr vs Int32**
- `Constexpr[int]` 值被**烘进 IR**——不同值产生不同编译内核(触发 re-compile)。真正动态的值必须用 `Int32`,否则每个取值都重编。
- autotune 里 Config kwargs 注入成 `@jit` 调用的 Constexpr 参数;只有 `key=[...]` 里列的 arg 值变化时才 re-tune。

**★`except Exception: continue` 会把编译期 bug 伪装成"没有可用 config"**
- autotune 候选循环普遍写成 `try: ... except Exception: continue`。**编译错误、NameError、签名不匹配
  统统被降级成"这个候选不行"**,和真的 config 不兼容长得一模一样,最后只剩一句
  `autotune found no working cfg for (M,N,K)`。
- 2026-08-09 实例:NT 循环是 `for bm, gm, xcd, ag in _NT_CANDIDATES`,却被改成
  `cands.append([launch, (bm, gm, gn, xcd, ag), c])` ⇒ `gn` 未定义 ⇒ 每个候选静默丢弃,
  **整条 dense fp8 前向断掉**,而只跑 NN/TN 的 bench 十轮全绿。
- 纪律:**改过候选表或编译参数,必须单独跑一次把异常打印出来的探针**(把 `except` 换成
  `traceback.print_exc()`);收官时再对**每条 layout** 各调一次真入口。
  另配 AST 检查 use-before-assign(`Load 的 Name − Store − 全局`)。

**dispatcher 写法**
- `can_handle` 对不支持的输入必须 **return False,绝不 raise**。WHY:dispatcher 靠返回值做 fallback,raise 会打断 fallback 链。
- `make_key` 必须捕获所有会改变最优 kernel 的东西(shapes / dtypes / layout / 关键标量),但**绝不放 id(tensor)**。
- `register_fake` 必须镜像真实输出的 shape/dtype。

**FLYDSL backend 尚未接入**
- ❌ 别再试：`GlobalBackendManager.set_*_backend(BackendType.FLYDSL)`。FLYDSL **尚未注册进 BackendType**(仅在 roadmap),代码层还没接入。
- 调优知识已在 kernel-optimize KB 的 `knowledge/backend/flydsl/`,但不代表能直接切 FLYDSL backend。

## 别接永不被采纳的候选：4w-persistent NT 对真实 8w-best <4% 净改动为零

- **接生产前必须证明会被采纳**：新内核变体接进 autotune 前，先证它在**真实全量候选池**里能按采纳门槛（>4% 滞后，用来压 DVFS 噪声；比放行门 2-3% 更严，见 pitfalls/02）被选中——分母必须是**真实 dispatch best**，不是手挑的弱子集。否则只白加编译开销，净改动为零。（pr-merge-gate/SKILL.md）
- **grouped NT 4w-persistent 实测被否**：对真实 8w-best 仅 ~3%（<4% 门槛），且在它本该赢的**短-K 区又输给 4w-np**→已撤销，净改动为零。（pr-merge-gate/SKILL.md）
- **fwd/dgrad 上 4-wave ≈ 8-wave-persistent**：打平（±2% 噪声内），autotuner 多数选 8-wave-persistent；dense/grouped 的 4w-persistent NT 候选几乎不被采纳。4-wave 的价值集中在 **wgrad（variable-K）**：早期生产 autotune 对比（auto vs 8w-only）**+6~17%**；后续修复 dispatch bug 并用 `PT_WARMUP=250` 校正后的 A/B（`c846d954` vs main）全 7 shape × 2 m 无一回归，大 contraction / 宽-N **+9~19%**，qwen/synth +3~9%。（flydsl-fp8-gemm-results/SKILL.md）

- **持久 vs 非持久（各形状归属不同）**：见下节「persistent vs 非持久 / vmcnt_hint」

- ❌ **别再试**：非持久 nt kernel 不移植 L2 swizzle。小-K shape 会**反输**（gpt-down −13%）。非持久优势只在大-K（循环调度惩罚主导）；小-K 靠 swizzle（L2 reuse）补回，缺了就崩。（02-nt-fwd-kernel.md）

- ❌ **别再试**：8-wave / BK128 换 occ=2 藏 store。原生 occ=2 的 8-wave kernel 在 store-bound 形状实测**全负**：28672 −14%、6144³ −20%、8192²×4096 −20%。occ=2 能藏 7-15% store，但 8-wave compute 赤字 14-20%（per-warp tile 减半→B 复用减半→ds_read/mfma 翻倍）远大于收益。与 4w+BK128 −13% 同结论。（project_mxfp4_epilogue_store.md）

## ★ 反面情形：候选是好的，但**竞速的打分 M ≠ 部署 M** ⇒ 它永远选不到 ⇒ 落地形态是「静态 lead」

（2026-08-17 gpt-oss-20b down-projection padN dgrad NN 实测；与上一节「别接永不被采纳的候选」是同一枚硬币的两面）

- **症状**：某个 `(num_xcd, group_m)` 在**部署形状**上稳定 +2.5~2.9%（逐位相同、5 回合 palindrome、三分布两正一中性），
  但生产竞速**每次都选另一个**。查竞速口径才发现：它在 `canonical M = (1024, 8192) tokens/expert` 上取几何平均，
  而这条算子**部署在 4096**。该臂在两个打分点是 −0.94% / +0.54%（几何 ≈ +0.2%）⇒ **过不了 1.5% 采纳门**。
  ⇒ 竞速跑一万次也不会选它；这不是噪声问题，是**打分点与部署点错位**。
- **判据**：任何 knob 都要在**部署 M** 上扫一遍再看竞速结论。竞速的 canonical M 是为了「一把钥匙开一族形状」，
  代价就是**部署点可能落在两个打分点之间的凹处**。M-敏感性表（本例 1024 −0.94 / 2048 +0.32 / 4096 +2.5~2.9 / 8192 +0.54）
  是判断「该不该给静态 lead」的必备证据。
- **落地形态 = 静态 lead，不是新候选**：把它做成候选只是白占槽位（上一节）。正确做法是
  **在一个能唯一识别部署形状的谓词上把候选表 reorder**，让它当 base、原 base 退化成「要赢 1.5% 才能顶掉」的挑战者。
  本例谓词 = `n_stride != 0`（只有 N-padded 的 MoE projection 走这条路），因此**其余所有调用者的候选表与顺序逐字节不变**。
- ⚠️ **静态 lead 必须复验「竞速没把它顶掉」**：写个探针，把**真入口**（含竞速）与两个候选的**直接编译产物**
  在一个进程里 palindrome 计时。本例 prod 850.2 µs 贴 cand_x8g4 847.9、离 cand_x4g8 859.8 ⇒ lead 生效。
  只看 bench 总分是看不出来的（差 1.4% 会被链稀释到 0.3%）。
- ⚠️ **别用 heavy-skew 的链级比值单读去判死一个 ±2% 的核内 knob**：本 campaign 早前一轮就是这么把这个臂
  以「heavy −5.68%」拒掉的，回到孤立核 5 回合逐位相同重测是 **heavy +2.49%**（min +2.43 / max +2.70），没复现。
  链级 heavy 比值自身跨轮就有 ~3% 的行程，分辨不了 2%。
- ⚠️ **静态 lead 要在「它的谓词覆盖到的每一个维度」上验非负，不能只验部署点**（2026-08-17 同 campaign 的 wgrad TN 一笔）：
  谓词通常识别的是**形状族**（本例 `m_real or n_real` = padded-operand），而族里还含着别的 M ⇒ lead 会作用到那些 M 上。
  wgrad 的 `(group_m=2, num_xcd=1)` 四点全非负（1024 **+1.47** / 2048 +0.27 / 4096 +0.78~0.98 / 8192 +0.95%），
  可以放心把门开到整族；而同 campaign 的 NN lead（x8g4）在 1024 是 **−0.94%**，门就得更窄或接受那一档的损失。
- ✅ **静态 lead 的正确降级行为**：lead 只是 base，竞速仍照跑。上面 wgrad 在 M=1024 处 `prod` 316.9 µs
  **比 lead 自己（324.3）还快 2.3%** ⇒ 那里竞速用更好的候选把 lead 顶掉了。这是健康的：
  **lead 保证不比原 base 差，竞速保留在别的点上超过 lead 的自由**。若某处 `prod` 比两个候选都慢，才是 bug。
- ⚠️ **写探针时 group_offs 的两种口径别串**：`_balanced_group_offs` / `_skewed_group_offs` 返回的是 int64 offsets 的
  **int32 view**（kernel 要的就是它）；但公共入口 `grouped_gemm_fp8_tensorwise_flydsl_kernel` 对 int32 输入会
  `.to(int64)` 再 `.view(int32)` ⇒ 把 66 个 int32 当 66 个 int64 升位，offsets 全错，**直接 Memory access fault**
  （不是 SNR 掉，是 GPU 挂）。喂公共入口传 `.view(torch.int64)`，喂 `_compile_*` 产物传 int32 view。

## persistent vs 非持久 / vmcnt_hint：fwd/dgrad 非持久优、wgrad 小-M 持久优

- **fwd/dgrad 用非持久 (non-persistent)**：每 WG 做完一个 tile 后直接 `s_endpgm`，不进 `scf.for` 循环。持久内核会吃 `scf.for` 调度惩罚 **~11%**，非持久省掉这部分。WHY：fwd/dgrad tile 数与 CU 匹配良好，无需持久复用。
- **wgrad 小-M 用持久，优于 masked**：per-group contraction ≤ **1536** 时，持久内核胜过 masked 分支——省掉 over-run chunk 的废循环（masked 会跑满整个 chunk 再 mask 掉尾部，浪费循环）。
- ⚠ **别把"wgrad 小-M 持久优"读成"截 grid 到 NCU 优"**：fp8-tensorwise TN var-K wgrad 上截 grid 实测是平的（生产路径 −0.03%/−0.24%），见 pitfalls/05 §persistent grid 的条件收窄。

## ❌ `num_xcd>1`（xcd_remap_pid 分区）：均衡负载 +7%，倾斜负载 −22~−43%

- gpt-oss wgrad var-K deploy（G=32/M=4096）实测，同 band 同边界体单变量：gate_up `num_xcd=2` 在**均衡**负载 **+6.94%**（ratio 0.8922→0.9542）、`num_xcd=8` +3.72%；同两臂在 **skew** 负载上是 **−22.07% / −42.84%**。
- 根因：remap 打散了 group-major 的 LPT 顺序，热组的 tile 被摊到全 launch，尾巴拉长。⇒ **候选表把 xcd>1 全部排除是对的**，race 若只打分均衡负载会被这条骗到（`_score` 必须取 balanced/skew 的**最差**）。
- ✅ 开口：同样的 L2 局部性有**不动派发顺序**的拿法 = 组内 XCD-affine 走位（`xcd_aff`）。它要求 `N_BLOCKS_N` 可分解；**N_BLOCKS_N 为质数时（gpt-oss gate_up 5760/256→23）几何退化成一列宽，等于没有**，这就是 gate_up 拿不到那 7% 的原因。
- ❌ **别再试**：把 big-K 的 drain-removal lever 迁移到 big-N。**不迁移**：big-N 的短 K 摊不开 drain 成本，lever 在 big-N 上无收益。

- **vmcnt_hint 要调到 det=0 的上限**（determinism/正确性边界；与 methodology/13 deep-wl 的 `(vmcnt,lgkmcnt)` 是不同轴：这里是正确性 det=0 上限，非性能 margin）：
  - big-K：`vh=3` 是 sweet spot。`vh=2` 也满足 det=0 但**更慢**。
  - big-N：det 上限**同样是 3**。
  - ❌ **别再试**：`vh > 3`。超过上限会 **race**（越界触发数据竞争，det≠0）。

- **dgrad `bm128` 小-M 分支有 tail cliff**：
  - M ≥ **1280**（N=8192）时，128-row tile 数 = **320 > 256**（CU 数）→ 触发 tail cliff，`bm128` 反输。
  - 对照：M=**4096** 时 `bm256` 反而 **+58%**。
  - gate 公式：`G * ceil(pm/128) * ceil(N/256) <= _num_cus()`，其中 `_num_cus()` = **256**。满足则可用 bm128，否则用 bm256 避开 cliff。

## 缓存增益上限规则：id(weight) cache 上限=quant_time/step_time，id(activation) 命中率≈0

### weight-quant cache：keyed(id(weight), weight._version)，增益有硬上限

- **cache 结构**：weight-quant / preshuffle 结果按 `(id(weight), weight._version)` 做 key。这是唯一允许的 cache key 形态。
- **真实训练每 step 增益上限 = quant_time(weight) / step_time**，是低个位数 %，**不是 benchmark headline**。WHY：
  - forward 算一次 quant 写入 entry；backward 复用 forward-time entry → 每 step 只 **1 hit/step**。
  - `optim.step()` 改权重 → `weight._version` bump → 下一个 forward 前 entry 失效。
  - 所以 benchmark 里反复复用同一 tensor 看着"免费"，真实训练里每 step 都要重付一次 quant。
- 上限量化为 `W_real`（对应 W1），REPORT 时必须 capped by `quant_time/step_time`。

### ❌ 别再试：id(activation) / id(grad_out) / id(activation_scale) 作 cache key

- ❌ **别再试**：拿 `id(activation)` / `id(grad_out)` / `id(activation_scale)` 当 key 缓存。真实训练命中率 **≈ 0**。
  - 机制：真实训练每 step 的 activation / grad 都是**新分配的 tensor**，`id(...)` 每次不同 → 永不命中。
  - 只有"复用同一 tensor"的 benchmark 才制造出免费假象。
- 这是 **Rule 11**：这类 W2/W3 cache 必须在 **ANALYZE 阶段直接拒绝**，不许实现出来再测。
- REPORT 时 `R_real` 必须 = 0；若真实命中率 < 50% → 即使 benchmark 已 accept 也**回退**。

### ❌ 别再试：MoE/grouped GEMM 把均匀分布假设烘进内核

真实 routing 给出**倾斜、逐 batch 变化**的 tokens_per_expert 直方图，一般**不是任何 tile 维度的整数倍**，且**部分 expert 拿到 0 token**。因此：

- ❌ **别再试**：把 `M_per_group` 烘成编译期常量。
- ❌ **别再试**：`assert M_per_group % BLOCK_M == 0`。
- ❌ **别再试**：假设固定 per-expert count 的静态 work partitioning。
- ❌ **别再试**：grouped/dense config sweep 选 `BLOCK_M=128`。少启动块是假象（grid 写死 /256），真实慢 **1.55×**——完整根因见 pitfalls/05。
- ✅ **必须**处理空组：`M_per_group == 0` 时 skip launch / 对 zero rows 分支。
- **验证纪律**：每一轮 MoE 都要在倾斜分布上验证——top-1 + capacity factor，以及近退化情形（某 expert 拿 ≥50% token）。在 skew 下消失的增益 = benchmark over-fit，**必须回退**。

### ⚠️ 候选竞速自身的噪声可以超过采纳裕度 → 逐配置结果双峰（2026-07-29 实测）

- 症状：campaign bench 同一份代码连测两次，多数配置漂 ±0.4~0.9%（正常噪声地板），但**个别配置在两个
  相距 ~3% 的值之间跳**（实测 mxfp8 NT `down heavy` 在 1.010 ↔ 0.976 之间）。别当热噪声查，先查 autotune。
- 根因：首调用的候选竞速（在**合成 balanced** 张量上跑）用 `score < best*0.985`（1.5% 裕度）决定是否
  采纳非 base 候选，而 `_robust_time` 在合成点上的噪声**本身就能超过 1.5%** → 竞速赢家在 run 之间翻转。
  实测清 cfg cache 重跑 3 次：N=2944 选中 `xcd=4 / xcd=8 / xcd=4`（2:1 翻转），N=5760 稳定。
  两个候选在竞速用的 balanced 点上打平，但在 **heavy 分布**上差 ~3%（cfg cache key 不含分布，一个 cfg
  同时服务 balanced 与 heavy）。
- 诊断配方（便宜，不用整跑 bench）：清 `_*_CFG_CACHE` + `_*_AT_CACHE`，同一 shape 连调 N 次，打印
  cache 里选中的 cfg；不稳定就说明裕度不够。
- 处置方向（按代价）：★**① 首选：竞速本身必须交替 A/B**（见 pitfalls/02「interleaved A/B 是唯一可信判胜法」）；
  ② 加大采纳裕度到 >竞速噪声；③ 竞速点里混入倾斜分布，让打分反映真实分布；④ 提高竞速 rep 数（不计时，
  只花首调用时间）。**别靠删候选**——它会伤 off-bench shape。
- ★ **上面 ②③④ 都只是"压噪声",真根因是竞速把 base 与候选放在不同测量窗口里测**（base 先测 = 在更冷的
  GPU 上）→ 偏置随 DVFS 轨迹走，加裕度只是让偏置不够翻门。**pitfalls/02 早已写明 interleaved A/B 是唯一
  可信判胜法，但这条纪律长期只用在 bench 上、没有应用到 autotune 竞速自身** —— 两卡之间的断链。
  修法（2026-07-29 实测）：`_robust_ab_ratio(base, cand, args)` 在**同一测量窗口内逐 rep 交替**计时 base 与
  候选、取比值中位数，再要求**每个竞速点**都 <0.985。实测噪声带 ≤0.5%（最差 0.9%），1.5% 裕度有 1.7~3×
  安全系数；清 cache 连跑 4 次 gm 极差 **0.62% → 0.12%（5×）**，`down heavy` 双峰消失（1.001~1.007）。
- ★★ **"逐 rep 交替"不够,每个 rep 内部要成回文**(2026-09-04 syncv3 grouped fp8 实测补强)。位置项是
  **每个 rep 内部同向复现**的(见 pitfalls/02 §多臂 palindrome 的位置偏置:末位臂独享自热 L2,同一改动
  换到首位就反号,±0.8~1.0pp),所以「交替 + 取 reps 中位数」仍然让固定顺序里的第一个臂系统吃亏。
  修法 = 每个 rep 计时 `(base, cand, cand, base)` 四个窗口、比值取 `(c1+c2)/(b1+b2)`,位置项作为奇函数
  在 rep 内部就抵消掉。**实测**:同一棵未改动的树,原实现(base 单独先测)在 gpt-oss down fwd
  N=2944 上清 cache 连跑 3 次采纳 `(4,8)/(8,4)/(8,4)` —— 而 `(8,4)` 在部署点实测快 2.2%,即
  **1/3 的启动跑的是慢臂**;换成回文配对后 **8 个 cell × 6 次 = 48/48 全稳定**,且每个 cell 采纳的
  都是双序探针实测最快的那个臂。代价:每 (候选, 打分点) 的首调用开销 ×1.7。
- ⚠ **"采纳稳定"这件事本身必须重复量**:同一棵未改动的树,本轮开局 4/4 稳定、两小时后同一探针
  2:1 翻转 —— 单次稳定性检查不构成证据,至少 N≥6 且报出全部采纳元组。
- ★ **第五个处置方向:干掉竞速,直接写死**。2026-08 mxfp4 grouped 实测:那场 race 的候选表里本来就有
  正确答案 `(2,8,0)`,只是因为在合成 balanced 点上打分而选不中它;关掉 race 逐配置实测,`(2,8,0)` 在两个
  最差配置上 **−7.0% / −7.8%**(每个数测两次,重复性 <0.15%)。**改成写死后 gm 1.3529→1.3762、min
  1.1843→1.2271**。当候选表小、且离线能一次量清每个 shape 家族该选谁时,写死比修 race 便宜得多,
  还顺带省掉每 shape 一次的首调用竞速开销。
- ★★ **别让两条派发路径共用同一个默认常量**。同一实证:NT 与 wgrad 共用一个 `_*_DEFAULT_CFG`,于是
  race 为 NT 选的值直接套到 wgrad 上。**两者统一成 NT 的最优 `(2,8,0)` 会让 wgrad 崩到 0.38–0.55×**
  ——拆成两份常量各自写死才对。机制核对过**不是 L2 命中率**(`TCC_HIT` 差 0.006%、`TCC_REQ` 逐位相同、
  `SQ_INSTS_MFMA` 相同),是 **WG→tile 顺序造成的负载不均**:同一个 swizzle 对 M-major 的 NT 和
  对 per-group 变 K 的 wgrad 意味着完全不同的 tile 到达序。

### benchmark 增益 > 结构能产 → 先查，别接受

- 若某轮 benchmark 增益**大于内核改动结构上能产生的量**，accept 之前先排查：
  1. identity-keyed activation cache（id(...) 假命中）
  2. uniform-MoE 假设（均匀分布捷径）
- REPORT 时把 baseline→final delta 重新归因：
  - `S_real`（K1–K4，transfer 1:1）
  - `W_real`（W1，capped by quant_time/step_time）
  - `R_real`（必须 = 0）
- 若 **headline − real 差 > 1%** → 标为 benchmark-loop residual 并建议回退。

### iteration_rules 核心纪律（贯穿以上；通用循环机制见 methodology/13）

- 一轮一个假设；perf 前先过 correctness gate；accept/rollback 有 lineage。
- 每个 accepted gain **必须能迁移到真实训练 step**：不许 id(...) 作 key 的 activation/grad_out cache；不许只适配均匀分布的 GroupGemm 捷径（真实倾斜 token 分布下无效）。

## ★★★★ dispatch 上的 first-call race 是自适应性的最后防线,不许用静态模型替代(2026-09-03 实测)

dense fp8 GEMM campaign 里,agent 为了「省每次 launch 几微秒的 host 时间」,把 TN 的
first-call race(建若干宏 tile → 各测一次 → 取最快)换成了「按解析模型排序、直接取第一个」。
计分面(5 个 P0 形状)一切正常,**因为其中唯一的 TN 形状恰好新旧模型都排对**。

换到计分面外的 llama-2 wgrad 就崩了(A=原 race 版,B=模型排序版,STREAM 三对中位):

| shape | ratio(A/B) |
|---|---|
| 7B gate_up wgrad m=4096 | **0.8599** |
| 70B gate_up wgrad m=4096 | 0.8645 |
| 70B down wgrad m=4096 | 0.8686 |
| TN 层整体 | **0.9145** |

**恢复 race(去掉 `break`、`best=cands[0]` 改回 `_pick_dense_candidate`)后 TN 回到 1.0237,
无一格倒退,llama 48 格整体转正 +0.79%。**

诊断过程里有一条**方法论比结论更重要**:我先做了零 GPU 成本的**静态门控分析**——纯 Python 复刻新旧
排序模型(`rounds` vs `rounds × cells`),算出「只有 70B gate_up/down 会从 SQUARE 翻到 RECT,
模型差 4.5%」。这个分析**圈对了嫌疑面,但定错了罪**:实测显示 7B gate_up 也退 14%,而它新旧模型
都选 RECT ⇒ 差异只能来自「旧代码 race 后按实测选,新代码信模型」。
⇒ **静态分析用来圈嫌疑很划算(几分钟、不占卡),但定罪必须实测;别拿模型推演当判决。**

★ 附带缺陷(次要):新模型 `_tn4_rounds` 只数 CU passes、**丢了每 pass 的代价**——TN 的
SQUARE 是 256×256=65536 cells、RECT 是 384×192=**73728(大 12.5%)**,旧 makespan 是
`rounds × cells`。但只要 race 还在,排序错只影响建构顺序,不影响最终选择。

★ **判据**:任何「把 race / autotune 换成解析模型或硬编码」的改动,验收面**必须**包含计分面之外
的兄弟形状(其它模型、其它 token 数)。省下的是微秒,赔出去的是可迁移性。

## ★★★ aiter Triton tuned-table:先查「这个形状的 grid 有几个 CTA」,再谈任何 kernel 改动(2026-09-12 实测,+57% score)

GLM-5.2 TP4/EP4 decode 的 absorb BMM(`batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant`,
156 次/step)在记分尺上占 c8 的 **55.9%**。它的 tuned 表
`gfx950/triton/gemm/batched_gemm_a8w8_.../...-N=512-K=192.json` 与 `...-N=256-K=512.json` 的
`M_LEQ_64` / `M_LEQ_128` 桶发的是 `BLOCK_SIZE_M=64, BLOCK_SIZE_N=256, num_warps=8`。

host wrapper 的 grid 是 `(B, cdiv(M,BM) * cdiv(N,BN))`,B=16:
- uk(M=48, N=512)→ 16×1×2 = **32 个 CTA**,256 个 CU 上 87% 空转;
- uv(M=48, N=256)→ **16 个 CTA**,94% 空转。

二阶损失:`48 % 64 ≠ 0` ⇒ `EVEN_MN=False`,取模寻址 + masked store;M tile 25% 是 padding。
实测 uk 只跑到 **14.6 TF/s 与 ~256 GB/s** —— 既非算力墙也非带宽墙,**是网格饥饿**。

换成小 tile(ABBA、min-of-6、同进程;`matrix_instr_nonkdim=32` 会触发 LLVM
`ADT/Sequence.h:275 Begin <= End` abort,要从候选集里剔掉):

| 形状 | M | ship us | 最优 | 最优 us | CTA | 加速 |
|---|---|---|---|---|---|---|
| uk N=512 K=192 | 48 | 10.58 | BM16 BN128 nw4 ns2 | 3.336 | 192 | 3.17× |
| uk | 84 | 13.56 | BM32 BN64 nw8 ns2 | 4.980 | 384 | 2.72× |
| uv N=256 K=512 | 48 | 12.68 | BM16 BN64 nw4 ns2 | 3.666 | 192 | 3.46× |
| uv | 84 | 13.56 | BM32 BN64 nw8 ns2 | 5.278 | 192 | 2.84× |

整尺验证:**score 1.00185 → 1.57249,relL2 一位不变**。

★ **可迁移的判据**:decode/小 M 的 batched GEMM,先算 `B × cdiv(M,BM) × cdiv(N,BN)` 对不对得上 CU 数。
`M_LEQ_64` 这种桶是给 M=64 调的,M=48 落进去就是 1 个 M-tile;桶粒度不够时**先改表,别先改 kernel**。

🐞 **配套的机制缺口(已在轮 7/8 补上)**:`_get_config(M, N, K, B=None)` 现在把 `B` 透传给
`get_gemm_config`,`...-B={B}-N={N}-K={K}.json` 可达。原先只按 (N,K) 键时,改 `M_LEQ_*` 桶
会波及同 (N,K) 的所有 B。

★ **`waves_per_eu` 必须逐桶量,它在相邻桶之间能翻脸**(轮 8,B=16 N=512 K=192 = GLM 的 uk)。
同进程 ABBA、`config=` 直传两臂、输出逐位一致:
**M=48(`M_LEQ_48`,BM16/BN128/nw4/ns3)wpe 1→2 = −1.58%**(3.1930→3.1426,spread 0.18~0.23),
整尺 `uk_8` 复现 −1.49%;但同一个旋钮在
**M=60(`M_LEQ_64`,BM32/BN64/nw8/ns3)= +1.10%**、**M=84(`M_LEQ_128`,同配置)= +44.6%**。
⇒ 这类"看起来无害"的占用旋钮**只能落在被量过的那一个桶里**;跨桶推广一次就会赔掉 45%。
(机理侧:BM16/BN128/nw4 的 tile 小、占用本来就低,放宽 wpe 才有 wave 可填;
BM32/BN64/nw8 已经把 EU 填满,wpe=2 只是把 LDS/寄存器预算切碎。)

---
来源: 06-autotune-design.md, 08-deadends.md, 04-tn-wgrad-kernel.md, SKILL.md, pr-merge-gate/SKILL.md, flydsl-fp8-gemm-results/SKILL.md, flydsl-fp8-gemm-tuning/SKILL.md, 02-nt-fwd-kernel.md, project_mxfp4_epilogue_store.md, 01-architecture.md, 03-nn-dgrad-kernel.md, gemm/optimization-directions.md, optimize-handoff/SKILL.md, campaign 20260912_134949

### 轮 9 复核:同一算子的**另一个 shape** 才是没扫过的那格

`uk`(B=16,N=512,K=192)的 `M_LEQ_64`(M=60)/`M_LEQ_128`(M=84)被全网格扫过
(BM 16/32/64 × BN 32/64/128/256 × nw 4/8 × ns 2/3/4 × wpe 1/2/3 × `.cg`/null,
每桶单核 156 次同块 ABBA、头/中/尾三个 ship 复制臂量位置偏差、每臂与 ship 逐位比对):
**出厂值就是实测最优**。M=60 次优臂 `32_64_w8_s2_e3` −0.49%,落在 0.11–0.63% 的
位置偏差带内;M=84 出厂即最优。本条的 wpe 分桶警告再次成立(M=84 的 wpe=2/3 =
+44.6%…+64%)。另外**所有臂输出逐位一致** ⇒ 这族 tile 轴动不了数值,不必挂 SNR 门。

⇒ 教训:**别只盯 directive 点的那张表**。同一个 op 的 `uv`(B=16,N=256,K=512)
从来没人扫,而它在三个并发上都比 `uk` 大(占 step 15.7–17.7% vs 12.6–15.5%)。
一扫就出 `waves_per_eu 1→3`:M=48 −0.58% / M=60 −0.41% / M=84 −0.87%
(逐位一致,每个 wpe 臂放两个位置、8 次 palindrome,12 个读数同号,
惰性对照臂 ±0.2%,位置偏差 0.07–0.12%)。**接 tuned 表的活之前,先按「占 step 的
比重」把同族所有 (B,N,K) 排一遍,再决定扫谁**。

另:`uv` 上 `num_stages` 也**分桶翻脸**(M=48 的 ns=3 是 −0.51%,M=60 的 ns=3 是
+2.43%),与本条 `waves_per_eu` 同源 —— 小 M 的 batched GEMM,**任何**调度轴都要按桶量。

## ⚠️ tile 轴和 `waves_per_eu` 在小 M batched GEMM 上是**一个轴**,不是两个

轮 10 在 `uv`(B=16,N=256,K=512)的 `M_LEQ_128` 桶上实测:把 `BLOCK_SIZE_M` 32→16
(grid = `(B, cdiv(M,BM)*cdiv(N,BN))`,CTA 数 192→384)在 `waves_per_eu=1` 下是
**+28…+33% 的灾难**,同一个形状换 `wpe≥2` 立刻变 **−6…−16%**。原因是 CTA 数越过
`num_cu`(256)之后每个 CU 要放两块,而 `wpe=1` 放任寄存器分配吃满、第二块进不来。

⇒ **扫 tile 轴(BM/BN/nw)时必须把 `wpe` 一起动**,否则会得到「细切 tile 是灾难」的
假结论并把它写进出厂表 —— 这正是该桶原本被定成 `BM32/nw8/wpe1` 的由来,白白压着 9%。
判据:先算每个臂的 CTA 数,**只要跨过 `num_cu` 就必须配 wpe 扫**。

⇒ 另一条更便宜的教训:**臂名必须逐字段反映 config**。同一轮里一个叫 `wpe3` 的臂
其实写死了 `(BM16,BN64,nw4,ns2,wpe3)`,在 `M_LEQ_64` 桶(出厂就是 BM16/nw4)只动 wpe,
到了 `M_LEQ_128` 桶(出厂 BM32/nw8)却同时动了三个字段,差点把 −8.93% 记到 wpe 头上
(单动 wpe 实测只有 −0.41%)。上一轮 KB 里那条「uv M=84 wpe 1→3 = −0.87%」就是这么错的。

## ★ `waves_per_eu` 就是**每 SIMD 可驻留 wave 数的硬上限**,判据要写成不等式

轮 11 在 `uk`(B=16,N=512,K=192)的 `M_LEQ_128` 桶(M=84)上把上一条的定性结论量成了
一条不等式。tile 固定 `BM16/BN64/nw4`(⇒ grid 768 CTA = **3.0 CTA/CU**,每 CTA 4 wave
= 每 CU 12 wave = **每 SIMD 3 wave**),只扫 wpe:

| wpe | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| vs ship | **+16.93%** | −5.92% | −5.93% | −5.73% | −6.32% |

⇒ **`wpe ≥ ⌈CTA/CU⌉ × (nw/4)` 之前是断崖,之后是平台**。上一条写的「CTA 数跨过
`num_cu` 就必须配 wpe 扫」只覆盖了 2 CTA/CU;真正的判据是
**`wpe` 要能装下 `⌈CTA数/num_cu⌉` 块**,3 块就要 wpe≥3,而且**到点就饱和**
(wpe 4/5/6 与 3 在 0.6% 内,别再往上加)。

配套两条:
- **赢的字段是 `BLOCK_SIZE_M` 32→16(网格翻倍),不是 wpe;wpe 只是解锁它**。
  `32_64_w8_s3_e{1,2,3,4}`(384 CTA)分别 +0.00/+45.58/+34.89/+0.94% —— 同一个 tile
  上把 wpe 抬起来一点用都没有。臂名必须带全五元组,否则又会把 BM 的钱记到 wpe 头上。
- **`num_stages` 在同一 tile 形状上按桶翻脸**:`BM16/BN64/nw4` 下 M=60 要 ns2
  (4.032 vs ns3 4.108,−1.9%),M=84 要 ns3(4.593 vs ns2 5.089,**−9.7%**)。
  ⇒ ns 不能全表取一个值,得跟 BM/wpe 一起按桶量。

**轮 11 落表(GLM-5.2 decode absorb,gfx950)**,每臂逐位比对 ship、头/中/尾三个 ship
复制臂量位置偏差(0.23–0.94%)、6–8 次 palindrome 取 min:

| 表 | 桶 | 出厂 | 新 | 实测 |
|---|---|---|---|---|
| `uk` N=512 K=192 | `M_LEQ_64`(M=60) | 32/64/nw8/ns3/wpe1 (256 CTA) | **16/64/nw4/ns2/wpe2** (512 CTA) | **−7.66%** |
| `uk` N=512 K=192 | `M_LEQ_128`(M=84) | 32/64/nw8/ns3/wpe1 (384 CTA) | **16/64/nw4/ns3/wpe4** (768 CTA) | **−6.43%** |
| `uk` N=512 K=192 | `M_LEQ_256`(M=200) | 64/256/nw8/ns2/wpe1 (**128 CTA=0.5/CU**) | **32/128/nw4/ns2/wpe2** (448 CTA) | **−37.4%**(两跑 −37.35/−37.40) |
| `uv` N=256 K=512 | `M_LEQ_256`(M=200) | 64/256/nw8/ns2/wpe1 (**64 CTA=0.25/CU**) | **16/64/nw4/ns3/wpe4** (832 CTA) | **−58.0%** |

`uk` 的 `M_LEQ_48`(M=48)与 `uv` 的 `M_LEQ_64`(M=48 和 M=60 都走它)**出厂即最优**
(最好的挑战臂 −0.29%,在 0.92–1.53% 的位置偏差带内)⇒ 已被前两轮榨干,别再排预算。

★ **例外:`uv` 的 `M_LEQ_128`(M=84)不是出厂最优,但它只在 `ns` 轴上有钱,而 `ns` 只有
配上 wpe≥5 才显形。** 出厂 `16/64/nw4/ns2/wpe4`;把 ns 拨到 3 后 wpe 阶梯变成
e2 −0.31 / e3 −0.14 / e4 −0.33 / **e5 −1.06** / e6 −0.81 / e8 +1.96%,
而 ns 留在 2 时同样的 wpe 阶梯全在 ±0.3% 带内(e5 −0.32 / e6 −0.30 / e8 +1.76)。
两个独立进程复现同序(另一跑 e6 −1.04 / e5 未测 / e4 −0.26),位置偏差 0.06–0.37%,
全臂逐位一致。落 `ns3/wpe5`:记分尺链内 `uv_14` min 5.2335→5.0465(**−3.6%**,
比孤立 ABBA 的 −1.06% 大,链内 L2 被 producer 的 56 MB 冲干净了),
总分 best-of-3 1.70788 → **1.71563**。
⇒ **「桶已榨干」的结论必须写成「在扫过的 (tile × wpe × ns) 网格里榨干」**:
这个桶前两轮都只扫了 tile × wpe,ns 一直钉在出厂值,于是一个 −1% 躲了两轮。

⇒ **大 M 桶的网格饥饿是这族最大的单笔钱(2.4×)**,而它天天躲过计分尺:
`M_LEQ_256` 覆盖 M∈129..256,而本campaign 的尺子只打 M∈{48,60,84}(= 6×concurrency
的 verify 行数)。**接 tuned 表的活,先算出尺子能到达的 M 集合,再分别标注
「能计分」与「只影响生产」的桶** —— 后者照样要修,但不要指望它动分数,
也不要把它混进需要 re-bench 验证的改动里当证据。

## 🐞 tuned 表扫描**必须逐臂带 maxdiff**:一个 config 可以只在某个 M 上算错

轮 11 在 `batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant`
上扫到 `BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, num_warps=4, num_stages=2`:
在 M=48/60/84 上与 ship **逐位一致**,到 M=200 上 maxdiff 突然是
**0.588(N=256,K=512)/ 0.673(N=512,K=192)** —— 输出量级下这是明显错的,不是 fp8 噪声。
同一个 config、同一个 kernel,只换 M 就从正确变成错误(大概率是该组合下的
Triton pipeline/mask 生成 bug)。

⇒ **不要「同族某个 M 上逐位一致 ⇒ 这族 tile 轴不动数值 ⇒ 不必挂门」**(本卡上一段
恰好这么写过)。每个臂、**每个要落表的 M** 都要留 maxdiff,并把非零的臂直接取消资格,
无论它多快。

★ **订正:这个坏 config 的失效域比上一段写的宽,「M=48/60/84 逐位一致」只对 `uv` 成立。**
用「多数票」口径复测(同一 M 上派 ship + 4 个另选 tile,取得票最多的那份答案作基准,
偏离者即误编译;本 op 无 split-K、K 组按序累加,所以合法 tile 形状**必须**逐位相同):

| M | 8 | 16 | 32 | 48 | 60 | 84 | 100 | 128 | 200 | 256 | 600 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `uk` N=512 K=192 | ✓ | ✓ | ✓ | ✓ | ✓ | **.64** | **1.02** | **.54** | **.53** | **.37** | **.60** |
| `uv` N=256 K=512 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **.32** | **.68** | **.60** |

(数字 = maxdiff,输出量级 1.2–2.7。)⇒ `32/64/nw4/ns2` 在 `uk` 上**从 M=84 就已经错了**,
不是 M=200 才错;拐点还跟 (N,K) 有关。**所以「在尺子打的 M 上验过 ⇒ 安全」是假的**:
`uk` 的 M=84 正是尺子打的点。上一段之所以判它在 M=84 通过,大概率是那次的参照臂
本身就是同一个坏 config(自己跟自己比 = 0)。**参照必须是多数票或另一族 tile,不能是单臂。**

★ **顺手把这条做成常规门**:遍历两张表**所有可达桶**(每桶取一个落在桶里的 M)
× 4 个另选 tile 的多数票检查,22 个 (op, M) 点**12 秒**跑完。轮 11 末次结果:
**没有任何在表里的桶被误编译**(22 个点全部 4 臂一致,唯一偏离者就是上面那个
从未进表的 `32/64/nw4/ns2`)。**落表后跑一次,比事后靠 relL2 总分兜底便宜得多。**

⚠ **别手搓这个 op 的 torch 参照**:`group_size=128` 但 `uk` 的 K=192(非整数组)、
A 在核内量化、`w_scale` 逐 batch。手写「按 K 组求 amax → 除 scale → clamp 448 → fp8」
的参照对**每一个**桶都给 relL2 0.25–0.56(连没动过的出厂桶也"挂"),
即参照自己错。数值权威只有两个:**记分尺自己的 relL2 门**、以及上面的**跨 config 多数票**。

## 📐 CTA 饥饿的三条「加并行度」路线,在这个 op 上都被量到是负的

轮 13 的督导指令是:`uv/uk` 在 `M_LEQ_64/M_LEQ_128` 桶上是 BM=16/BN=64 ⇒ 192 CTA
= **0.75 CTA/CU**,看起来是 dispatch 饥饿,应该拿 BN=32 / BM=32 去换更多 CTA。
逐臂 ABBA(同块、min-of-REPS、head/mid/tail 重复 ship 臂定位偏,slot bias 0.11–0.42%,
所有臂 `maxdiff {}` 逐位一致)测下来,**每一条都更慢**:

| 点 | ship | 最好的挑战者 | Δ |
|---|---|---|---|
| `uv` M=48 (N=256,K=512) | BM16/BN64 3.6044us | BN32 / BM32 各臂 | **+12.41%** |
| `uv` M=84 | BM16/BN64 | 同上 | **+17.84%** |
| `uk` M=60 (N=512,K=192) | BM16/BN64 | 同上 | **+8.39%** |
| `uk` M=84 | BM16/BN64 | 同上 | **+4.33%** |
| `uk` M=48 | BM16/BN128 | BN=64 | **+4.44%** |
| `uv` M=48 | — | split-K=2 / =4(双 kernel + f32 partial) | **+50.1% / +64.4%** |
| `uv` M=84 | — | split-K=2 / =4 | **+39.9% / +73.4%** |
| `uk` M=48 | — | split-K=2 | **+87.9%** |
| `uk` M=84 | — | split-K=2 | **+86.0%** |

(split-K 臂的 relL2 vs ship = 4.7e-6…1.75e-5,即算术是对的,只是重新分组了 —— 慢不是因为算错。)

**机制(两条,互相印证)**:
1. **切 N 或切 M ⇒ 字节和 VALU 一起翻倍。**这个 kernel 的激活量化在核内做:
   每个 CTA 对自己的 A tile 跑一遍 `amax → a_scale → clamp → fp8`。多一个 N-tile
   就把**整块 A 的读取 + 整套量化 VALU** 再做一遍。ISA 侧证实开销中心就在这里:
   `uk` M=48 是 **673 条指令喂 6 个 MFMA**,`uv` M=48 是 **460 条喂 2 个 MFMA** ——
   MFMA 不是瓶颈,量化 VALU + 地址算术才是。所以 BN 减半 = 把主要成本乘 2 去买并行度。
2. **split-K 不加字节,但要付跨 kernel 的 f32 中转。**`uv` M=48 的 sk2 多写 3.1MB
   f32 partial、再读回、再加一次 reduce launch = **+1.81us**,而省下的并行度收益远不够。

⇒ **判据更新**:`CTA/CU < 1` 本身**不是**可用的信号。要先问「加 CTA 的代价是什么」:
若每个新 CTA 都要重做一遍 tile 级的前处理(量化/去量化/rescale),或要把 f32 中间结果
落 HBM,则 CTA 数和时间是**同向**的,tuned 表里那个"饥饿"的网格就是最优点。
**下一个该量的是不加字节、也不加 launch 的并行度**:核内 warp 级 K 切分 + LDS reduce
(4 个 warp 各吃 1/4 的 K group),以及让**一个 CTA 覆盖多个 n-tile 并把量化后的 A 与
`a_scale` 存进 LDS 复用** —— 后者正是 BN 减半那条机制的反方向(量化只做一次)。

## ❌ 别再试:在这个 kernel 上「把 K 尾组单独 peel 出来」

`BLOCK_SIZE_K` 被 wrapper 强制等于 `group_size`(128),所以 `uk` 的 K=192 = 1.5 组:
最后半组按全宽跑,**四分之一的算术花在一块零上**,而且 mask 还挂在本来不需要 mask 的
整宽组上。把尾组 peel 成自己的 2 的幂宽度**是位精确的**(用 **zero-pad-K 恒等式**
而不是手搓 torch 参照来验,11 个 shape 全部 bit-exact)。但这条路在当前 codegen 下走不通:

- 第二个不同 K 的 `tl.dot` 让 Triton 把那块窄的 B tile **逐字节过 LDS**:
  M=48 指令数 **673 → 1026**,VGPR **82 → 118**,多出 **32× `buffer_load_ubyte` + 35× `ds_write_b8`**。
- M=84 更夸张:**673 → 3513**,其中 **2388 条是 SALU** —— 64 位索引算术被整个重新物化。
- 实测 `uk` **+43…+55%**,`uv` 不变。两种实现都试过(尾 load 留在原位 / 上提到循环外),
  都救不回来(1.60499)。

⇒ 结论不是"尾组的浪费不存在",而是**必须换一种不引入第二个窄 dot 的方式去拿它**:
比如把 K pad 到 256 让 `EVEN_K` 成立(要在 host 侧改权重布局,成本在 wrapper 不在 kernel),
或让尾组走同宽度 dot 但用 `other=0.0` 之外的方式省掉 mask。**改之前先 dump `.amdgcn`
看那个窄 dot 降成了什么**,这一步 12 秒,比跑尺子便宜两个量级。

## 🔧 订正:`Sequence.h:275 Begin <= End` 不只由 `matrix_instr_nonkdim=32` 触发

本卡原先把这个 LLVM abort 归给 `matrix_instr_nonkdim=32`。轮 13 在 `uk` 扫
`num_warps` 时,**`BLOCK_SIZE_N=256/512` 配 `num_warps=1`** 同样触发,整个扫描进程被带走。
⇒ 扫 `num_warps` 时先排除「BN 很大 + nw 很小」的极端臂,或把每个臂放进子进程隔离。

顺带把 `num_warps` 这根轴关掉:`uv` M=48 上 **nw=2 +15.02%、nw=1 +48.02%** ——
"少 warp 可以省掉 warp 间广播冗余"不成立,**wave 级延迟隐藏的收益远大于广播冗余的成本**。

## 🔧 订正:`wpe` 判据在 `CTA/CU < 1` 的区间失去分辨力

本卡的 `wpe ≥ ⌈CTA/CU⌉ × (num_warps/4)` 判据,在 `uv` 的 `M_LEQ_64` 桶(M=48 与 M=60
共用)上**不再区分**:把 `num_stages` × `waves_per_eu` × `cache_modifier` 的臂全扫一遍,
**两个 M 上所有臂都落在 ship 的 ±0.3% 内**(< slot bias)。
⇒ 在 CTA 数还不够铺满一遍 CU 的桶里,占用相关的旋钮都是平的,**拆桶不会有收益**,
别把预算花在这;该去动的是决定字节数和 VALU 数的那几个(tile 形状、量化的复用)。

## ★★★ 轮 13 勘误:本卡那两个 `num_stages` 数字是**污染仪器**量的,真值差 4~7 倍

上面 §轮 9 复核写「`uv` 上 `num_stages` 分桶翻脸(M=48 的 ns=3 是 **−0.51%**,M=60 的
ns=3 是 **+2.43%**)」。那两个数是**每臂各自分配 x/wq/y** 的探针量的。换成
**一棵树里每个 (M,N,K) 只分配一次、所有臂共用同一组 tensor + 同一个输出**,并在回文
两端各放一个惰性对照臂(`ctrl` 稳定在 ±0.35% 内)之后:

| 形状 | 桶 | ship ns | ns=3 的 `delta − ctrl` | 重复次数 |
|---|---|---|---|---|
| `uv` N=256 K=512 M=48 | `M_LEQ_64` | 2 | **−1.89 / −1.97 / −1.99%** | 3 |
| `uv` M=64 | `M_LEQ_64` | 2 | **−1.28%** | 1 |
| `uv` M=36 / M=42 / M=60 | `M_LEQ_64` | 2 | −0.31 / −0.00 / **+0.34%** | 各 1 |
| `uk` N=512 K=192 M=60 | `M_LEQ_64` | 2 | **−1.02 / −1.11 / −0.85%** | 3 |
| `uk` M=64 | `M_LEQ_64` | 2 | **−2.27%** | 1 |

⇒ 卡片的定性结论(ns 按桶翻脸)对,但**两个数量级都错了**,而且错的方向正好让这 2% 被
压了四轮:−0.51% 在任何采纳裕度之下,没人会为它去拆桶。**逐臂各自分配的 buffer 能把
同一份二进制的两臂拉开 9.6%** ⇒ tuned 表扫描的仪器必须共用 tensor,否则赢家不可信。
本条已落表:`uv` 新增 `M_LEQ_48`(ns=3,M_LEQ_64 保持 ns=2 以避开 M=60 的 +0.34%)、
`uk` 的 `M_LEQ_64` ns 2→3;计分尺三跑 1.77324/1.77418/1.77905 vs 基线三跑
1.7594/1.7618/1.7659(区间不重叠),relL2 恒 0.0234,全臂 maxdiff=0。

**往深处不要再走**:`uv48` 的 ns 4/5/6 = +3.4 / +3.7 / +15.1%,`uk48` = +16.3 / +31.0 /
+47.2%。`uk` 的 K=192 只有 2 次 K 迭代,ns≥4 时流水根本填不满、退化成前导全载。

## ❌ `GROUP_SIZE_M > 1` 在这一族 batched GEMM 上处处判负(轮 13,首次量)

两张 B=16 表出厂全是 `GROUP_SIZE_M: 1`,而这条字段**从来没人动过**。实测
(`delta − ctrl`,共用 tensor,maxdiff=0):

| 形状 | gsm2 | gsm4 |
|---|---|---|
| `uv` M=36/42/48/60/64 | +2.11 / +2.19 / +2.66 / +2.04 / +2.69% | +2.29 / +2.50 / +2.27 / +2.27 / +2.69% |
| `uk` M=60 / M=64 | +3.28 / +5.81% | +3.41 / +6.07% |

机理:grid 是 `(B, mt*nt)`,`program_id(0)` 是 batch 且**变得最快** ⇒ XCD = `linear % 8`
= `batch % 8`,一个 XCD 天然拿到 2 个 batch 的**全部** tile,联合脚印 360 KB ≪ 4 MB L2。
局部性已经满了,swizzle 只是白付一笔 `//`/`%` 的索引算术。⇒ **B 维参与 grid 且变最快的
batched GEMM,不要再扫 `GROUP_SIZE_M`;先算 XCD 拿到的是什么。**

## ★★ `BLOCK_SIZE_M=32` 在这个核上是断崖,而且**与字节数、CTA 数都无关**(轮 13)

轮 10/11 的教训是「扫 tile 轴必须同时动 wpe」。轮 13 把 wpe 按不等式配齐之后重扫,
发现 BM=32 的惩罚**不能用字节或 CTA 数解释**。每个臂的账
(`bytes = A·⌈N/BN⌉ + B·⌈M/BM⌉ + C`,`CTA = 16·⌈M/BM⌉·⌈N/BN⌉`):

| 形状 | 臂 | CTA | 字节 | `delta − ctrl` |
|---|---|---|---|---|
| `uk48` | ship `16_128_w4_s3_e2` | 192 | 6.38 MB | 0 |
| `uk48` | `32_64_w4_s3_e2` | **256**(更多) | **6.00 MB**(更少) | **+39.5%** |
| `uk48` | `32_64_w8_s3_e2` | 256 | 6.00 MB | +33.8% |
| `uk48` | `32_128_w4_s3_e1` | 128 | **4.88 MB**(最少) | **+75.4%** |
| `uk48` | `16_64_w4_s3_e3` | **384** | 7.50 MB | **+0.2%** |
| `uv48` | ship `16_64_w4_s2_e3` | 192 | 9.38 MB | 0 |
| `uv48` | `32_32_w4_s{2,3}_e{1,2}` | 256 | 10.38 MB | +27.5…+27.7% |
| `uv48` | `32_64_w4_s3_e1` | 128 | 7.38 MB | +45.0% |
| `uk60` | `32_64_w4_s3_e{1,2}` | 256 | 6.75 MB | +10.6…+10.8% |

「CTA 192→384、字节 +18%」是 +0.2% 的**持平**,而「CTA 192→256、字节 −6%」是 +39.5%。
⇒ **这个核既不是字节受限也不是 CTA 数受限,`BLOCK_SIZE_M=16` 是硬性前提**。别再用
「字节 vs 并行度」给它挑 tile,那个模型在这里被自己的两个臂同时证伪;判据改成
**先固定 BM=16,再只在 BN / ns / wpe 上动**。
