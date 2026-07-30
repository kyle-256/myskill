# autotune / dispatch 纪律与踩坑合集

> 类别: 踩过的坑 · 主题标签: autotune, dispatch, cache, overfit, autotune-dispatch, persistent-vs-nonpersistent, nt-kernel, l2-swizzle, persistent-kernel, vmcnt_hint, tail-cliff, tile-size, cache-key, real-training-gain, MoE-skew, benchmark-overfit

## autotune/dispatch 纪律：禁 id(tensor) cache、can_handle 不 raise、per-shape 过拟合

**cache 纪律**
- ❌ 别再试：autotune 里 result cache(缓存 quantize/transpose/group_offs 的结果)和任何 **id(tensor)-keyed** cache。id(tensor) 会复用/回收——同一地址可指向不同数据,cache 命中即数据错误。(Python 对象被 GC 回收后,新 tensor 可能拿到同一 `id` 值,cache 便把旧 key 命中到新数据。)
- 只能 cache 两种东西：compiled object(`flyc.compile` 产物)和 launch closure(闭包**不含 data**,只含 launch 参数)。
- cache key 用**纯静态维度**：`(op, N, K, G, M_total, out_fp16, cbsz, blgp)`。WHY：这些是唯一决定最优 kernel 的量,不含运行期 buffer 身份。

**per-shape 过拟合(overfit)**
- ❌ 别再试：dgrad NN 按 c_n 做 per-shape `num_xcd`。实测 **−0.5%** 负杠杆,没有干净物理阈值——属于纯 overfitting。
- 原则：**没有清晰物理阈值的 per-shape 调参一定 overfit**,别做。

**wgrad gate 用 per-group m_total/G**
- wgrad gate 必须用 **per-group contraction = m_total/G**,不是裸 m_total。
- ❌ 别再试：旧 gate `m_total<=2048` 坑高-G MoE。反例 G=8 / M_g=512 / m_total=4096 被误判走 masked,而实测 persist 反而 **+13%**。根因:m_total 混了 G 的贡献,per-group 收缩维才是真正决定 kernel 选择的量。

**Constexpr vs Int32**
- `Constexpr[int]` 值被**烘进 IR**——不同值产生不同编译内核(触发 re-compile)。真正动态的值必须用 `Int32`,否则每个取值都重编。
- autotune 里 Config kwargs 注入成 `@jit` 调用的 Constexpr 参数;只有 `key=[...]` 里列的 arg 值变化时才 re-tune。

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

## persistent vs 非持久 / vmcnt_hint：fwd/dgrad 非持久优、wgrad 小-M 持久优

- **fwd/dgrad 用非持久 (non-persistent)**：每 WG 做完一个 tile 后直接 `s_endpgm`，不进 `scf.for` 循环。持久内核会吃 `scf.for` 调度惩罚 **~11%**，非持久省掉这部分。WHY：fwd/dgrad tile 数与 CU 匹配良好，无需持久复用。
- **wgrad 小-M 用持久，优于 masked**：per-group contraction ≤ **1536** 时，持久内核胜过 masked 分支——省掉 over-run chunk 的废循环（masked 会跑满整个 chunk 再 mask 掉尾部，浪费循环）。
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

---
来源: 06-autotune-design.md, 08-deadends.md, 04-tn-wgrad-kernel.md, SKILL.md, pr-merge-gate/SKILL.md, flydsl-fp8-gemm-results/SKILL.md, flydsl-fp8-gemm-tuning/SKILL.md, 02-nt-fwd-kernel.md, project_mxfp4_epilogue_store.md, 01-architecture.md, 03-nn-dgrad-kernel.md, gemm/optimization-directions.md, optimize-handoff/SKILL.md
