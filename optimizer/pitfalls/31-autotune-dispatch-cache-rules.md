# autotune/dispatch 纪律：禁 id(tensor) cache、can_handle 不 raise、per-shape 过拟合

> 类别: 踩过的坑 · 主题标签: autotune, dispatch, cache, overfit

**cache 纪律**
- ❌ 别再试：autotune 里 result cache(缓存 quantize/transpose/group_offs 的结果)和任何 **id(tensor)-keyed** cache。id(tensor) 会复用/回收——同一地址可指向不同数据,cache 命中即数据错误。
- 只能 cache 两种东西：compiled object(`flyc.compile` 产物)和 launch closure(闭包**不含 data**,只含 launch 参数)。
- cache key 用**纯静态维度**：`(op, N, K, G, M_total, cbsz, blgp)`。WHY：这些是唯一决定最优 kernel 的量,不含运行期 buffer 身份。

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

---
来源: 06-autotune-design.md, 08-deadends.md, 04-tn-wgrad-kernel.md, SKILL.md
