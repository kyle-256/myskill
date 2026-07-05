# mxfp4 K28672 天花板：occ=2 唯一途径 BK128 让 g2s 翻倍净亏 13%

> 类别: 踩过的坑 · 主题标签: occupancy, LDS-bound, mxfp4, K28672

- **5405T 天花板根因 = occ=1 wave/SIMD，由 LDS 144KB/wg 锁死（非 VGPR）**：160KB/CU ÷ 144KB = 1 WG/CU；2 个 WG 需 288KB > 160KB 物理不可能。即使 VGPR 允许 2 waves，LDS 也只允许 1。单 wave 无法用第二个 wave 的 MFMA 去填 operand bubble → **MFMA 89.5% stall**。

- **减法探针成本（500-sample，4-wave K28672 BK256）**：
  - `g2s`（buffer_load→LDS DMA，HBM-BW-bound）= **1061T = 主瓶颈**
  - `ds_read`（96 read / 256 mfma）= **292T**
  - `scale-load` 成本已被 GAVOID + SCVGPR + ILV 完全藏住（CONSTSC = prod real 5401）
  - 减 g2s 需更大 tile → AGPR 溢出；减 ds_read 需 BK128 → loop 开销更大。K28672 下 5500 med 超出 4-wave BK256 物理顶。

- **K 越长 hiding 越充分**：K8192 = 4603 / K28672 = 5401 / K57344 = 5481。

- ❌ **别再试 occ=2（净亏 -13%）**：BN256 + BK128 达到 occ=2（LDS 72K，VGPR 176），确实把 MFMA stall **89.5%→58.7%**，但 occ=2 的**唯一达成途径就是 BK128**（BK256 时 LDS 都 > 80K）。BK128 让 **g2s 频率翻倍 → VMEM-load stall 2.7%→30%**，完全压倒 MFMA 改善，总速 **5405→4686**。

- ❌ **别再试 BN128 单 slice**：只会让 g2s 更暴露，无收益。

- ❌ **别再试参数扫描（EVENSPREAD / PREFETCH / MMORD / ACC_DIST16 / PINBASE / WLVMCN / BARNOP）**：全落在 **5400±10 噪声内**，因为都不改 occ=1 根约束。

- ❌ **严禁测 K=57344**。

---
来源: 08-att-root-cause.md, 04-ceiling-analysis.md, project_mxfp4_k28672_ceiling.md
