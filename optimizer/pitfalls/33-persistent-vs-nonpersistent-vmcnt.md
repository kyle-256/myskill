# persistent vs 非持久 / vmcnt_hint：fwd/dgrad 非持久优、wgrad 小-M 持久优

> 类别: 踩过的坑 · 主题标签: persistent-kernel, vmcnt_hint, tail-cliff, tile-size

- **fwd/dgrad 用非持久 (non-persistent)**：每 WG 做完一个 tile 后直接 `s_endpgm`，不进 `scf.for` 循环。持久内核会吃 `scf.for` 调度惩罚 **~11%**，非持久省掉这部分。WHY：fwd/dgrad tile 数与 CU 匹配良好，无需持久复用。
- **wgrad 小-M 用持久，优于 masked**：per-group contraction ≤ **1536** 时，持久内核胜过 masked 分支——省掉 over-run chunk 的废循环（masked 会跑满整个 chunk 再 mask 掉尾部，浪费循环）。
- ❌ **别再试**：把 big-K 的 drain-removal lever 迁移到 big-N。**不迁移**：big-N 的短 K 摊不开 drain 成本，lever 在 big-N 上无收益。

- **vmcnt_hint 要调到 det=0 的上限**（determinism/正确性边界）：
  - big-K：`vh=3` 是 sweet spot。`vh=2` 也满足 det=0 但**更慢**。
  - big-N：det 上限**同样是 3**。
  - ❌ **别再试**：`vh > 3`。超过上限会 **race**（越界触发数据竞争，det≠0）。

- **dgrad `bm128` 小-M 分支有 tail cliff**：
  - M ≥ **1280**（N=8192）时，128-row tile 数 = **320 > 256**（CU 数）→ 触发 tail cliff，`bm128` 反输。
  - 对照：M=**4096** 时 `bm256` 反而 **+58%**。
  - gate 公式：`G * ceil(pm/128) * ceil(N/256) <= _num_cus()`，其中 `_num_cus()` = **256**。满足则可用 bm128，否则用 bm256 避开 cliff。

---
来源: 01-architecture.md, SKILL.md, 03-nn-dgrad-kernel.md
