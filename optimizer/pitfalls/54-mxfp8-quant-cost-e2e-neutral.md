# grouped MX quant 占 fwd 绝对 35-46%（不是 7%）但 e2e 中性非杠杆

> 类别: 踩过的坑 · 主题标签: mxfp8, grouped-gemm, quant, occupancy

## 绝对占比纠错（核心）
- **grouped MX quant 占 fwd 绝对时间 35-46%（不是 7%!）**。之前的 "7%" 是 **MX/TW 比值差（差分口径）**，不是绝对占比。
- 实测（M=4096，绝对拆分）：
  | shape | 拆分 | quant 占比 |
  |---|---|---|
  | 4096×7168 | qa311 + qb282 / gemm910 | **39%** |
  | 8192×4096 | — | **35%** |
  | 4096×4096 | — | **38%** |
  | 2880×2880 | — | **42%** |
- WHY：要压绝对 fwd 时间追 dense，quant 是大杠杆；**"96.5% 在 gemm" 只对 MX/TW 比值差成立**，不能当绝对占比用。

## ❌ 别再试：grouped FlyDSL dual-cast quant kernel（e2e 中性，非杠杆）
- kernel 已写完 + bit-exact，但 **e2e 中性**：quant 全优化后 fwd 只 **1.2→1.07x**、**bwd 几乎没动**。
- 真相：**bwd 缺口 90% 在 gemm（dgrad+wgrad）、fwd 缺口 60% 在 gemm**。剩余全是 grouped MX gemm 的 **occ=1 结构上限**。
- 结论：下一刀 ROI 是 **gemm（尤其 bwd）**，不是 quant。绝对占比大 ≠ 优化 quant 能撬动 e2e，因为 gemm 侧缺口更大且是结构上限。

## ✅ b-scale 布局担忧多虑（正确性已过门）
- 之前担心 "b-scale 必须匹配 gemm 的 `use_2d_block` 布局" 是多虑：**raw 1×32 E8M0 scale 就能被 gemm 正确消费**（e2e fwd **SNR 28.1dB** 过门）。
- batched FLY 权重 quant 直接产 raw 1×32 E8M0 无碍。

## qa kernel 已高度调优，离峰 gap 是结构性
- qa kernel PMC：**occ 75.4% / MeanOcc 24.16 waves/CU / WriteSize 306MB≈理论 283（放大仅 8%）/ MemStall 1.6% / VALUBusy 30%**。
- 解读：占用率高 → **非 occ-limited**；写高效 → **无半-cache-line 写放大**。
- ❌ 别再试低垂果实（提 occ / 写合并 / scale_pack）：**全证伪，不存在**。
- 离峰 gap 根因：**phased load→barrier→compute→barrier→store 结构**（计算相时 DRAM 空闲，**dense 同样只 ~60% 峰**）。要抬只能**跨-tile 软流水重叠访存与计算**，是大改高风险。
- grouped 独有的 **1.30x（+37us）** 是 M-remap SRD-gating + col-padding + prologue，**全结构性**。

## ❌ 别再试：fwd/dgrad 三条重写路（全证伪）
- grouped MX fwd/dgrad 慢 4-12%，无低风险可落地优化：
  - **scale_pack**：被预取隐藏，scale-tax≈0。
  - **BLOCK_M=128**：少启动块假象，真实 **1.55x 慢**。
  - **band-swizzle**：`group_n>0` / `gm=8` 略慢或崩 **1.5x**。
- **preshuffle 只占 3.5%（15.5us / ~3.4 TB/s）非瓶颈**。
- 唯一有效 lever 是**调度局部性（xcd/gm/gn），已被 autotune 吃掉**。

---
来源: mxfp8-grouped-gg-devloop/SKILL.md
