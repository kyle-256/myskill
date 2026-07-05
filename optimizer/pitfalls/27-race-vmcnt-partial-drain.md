# partial-drain race：读写距离决定安全 defer，vmcnt 按所有池写总数算

> 类别: 踩过的坑 · 主题标签: race-correctness, vmcnt, partial-drain, byte-exact

- **读写距离决定安全 defer**：2buf = 距离 1（写完下一相位就读，必须**全 drain**）；3buf = 距离 2（有整整一相位余量，可安全 **partial-drain** 让写跨 barrier）。缓冲深度不足就 defer 写 = 低概率 race。（不是"总能 partial-drain"，取决于 buffer 深度给的距离余量。）

- **多池 partial-drain 的 vmcnt 必须按所有 3buf 池的写总数 `sum(nsa|nsb)` 计算**，不能只算第一个池。`_n_outstanding = nsb` 是 bug：2 池都 3buf 时实际延迟 `2×nsb = 8` 条写，但只 drain 4 条 → 留未 accounted 的 in-flight 写 → **1/30000 race**。
  - 修法靠 emit 顺序 + 正确 vmcnt(sum)：把 distance-2 安全的 B 池排**最后** emit，`vmcnt(sum)` 恰好留 B 池在飞、把 distance-1 的 A 池**全 drain**。

- **byte-exact 排除法必须组合所有省空间手段再算**：曾判"双池 3buf 放不下"而误否决双池路线。错在没把 **scalar store 省 C_lds（8704B）** + **`_CS=1024` 省 bank-pad（5120B）** 组合起来算。单独算每个都不够、组合才够；单独评估会**假阴性**关掉真路。凡涉及 LDS 容量卡点，先把所有省空间手段叠加后再判可行性。

## ❌ 别再试（HW-walled 死坑）

- dwordx4-lds 直写 LDS 同步死路 + SCVGPR prefetch racy 完整版：见 pitfalls/45-gfx950-hw-walled-races.md「❌ 别再试」
- BK128 SCVGPR scale VGPR WAR race 完整机制：见 pitfalls/45-gfx950-hw-walled-races.md「BK128 SCVGPR scale VGPR WAR race」

---
来源: 10-grouped-wgrad-4wave-3buf.md, 05-dead-ends.md
