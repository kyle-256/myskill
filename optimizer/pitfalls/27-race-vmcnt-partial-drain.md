# partial-drain race：读写距离决定安全 defer，vmcnt 按所有池写总数算

> 类别: 踩过的坑 · 主题标签: race-correctness, vmcnt, partial-drain, byte-exact

- **读写距离决定安全 defer**：2buf = 距离 1（写完下一相位就读，必须**全 drain**）；3buf = 距离 2（有整整一相位余量，可安全 **partial-drain** 让写跨 barrier）。缓冲深度不足就 defer 写 = 低概率 race。（不是"总能 partial-drain"，取决于 buffer 深度给的距离余量。）

- **多池 partial-drain 的 vmcnt 必须按所有 3buf 池的写总数 `sum(nsa|nsb)` 计算**，不能只算第一个池。`_n_outstanding = nsb` 是 bug：2 池都 3buf 时实际延迟 `2×nsb = 8` 条写，但只 drain 4 条 → 留未 accounted 的 in-flight 写 → **1/30000 race**。
  - 修法靠 emit 顺序 + 正确 vmcnt(sum)：把 distance-2 安全的 B 池排**最后** emit，`vmcnt(sum)` 恰好留 B 池在飞、把 distance-1 的 A 池**全 drain**。

- **byte-exact 排除法必须组合所有省空间手段再算**：曾判"双池 3buf 放不下"而误否决双池路线。错在没把 **scalar store 省 C_lds（8704B）** + **`_CS=1024` 省 bank-pad（5120B）** 组合起来算。单独算每个都不够、组合才够；单独评估会**假阴性**关掉真路。凡涉及 LDS 容量卡点，先把所有省空间手段叠加后再判可行性。

## ❌ 别再试（HW-walled 死坑）

- ❌ **gfx950 dwordx4-lds 直写 LDS 期望 vmcnt 同步**：`buffer_load_dwordx2...lds` 不支持（gfx950 只有 dword 和 dwordx4）；dwordx4-lds 直写 LDS 的完成**不被 vmcnt / 隔-phase barrier 可靠同步** → det≠0。只有 `vmcnt(0)` + 紧跟 `s_barrier` 能 det0，但序列化后 **4803 < 5176**（更慢）。SCVGPR 的 SCV2AHEAD / SCPF prefetch 也 racy（vmcnt 乱序退役，不保证特定时刻落地）。

- ❌ **BK128 SCVGPR scale VGPR WAR race，靠 vmcnt(0) 修**：K=256（0 main iter）SNR 55.6，但 K=384（1 iter）SNR **-inf**。phase-B 的 `emit_sc_vgpr(0) → v[8:9]` 覆写 phase-A mfma 还在读的 scale VGPR。`vmcnt(0)` **不能修**——vmcnt 乱序退役，不保证特定 load 落地。与 BK256 SCVGPR 同一机制。

---
来源: 10-grouped-wgrad-4wave-3buf.md, 05-dead-ends.md
