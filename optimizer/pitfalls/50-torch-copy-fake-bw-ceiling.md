# torch copy_ ~5.0TB/s 不是 DRAM 天花板：自定义宽向量 copy 才达 ~6.3TB/s

> 类别: 踩过的坑 · 主题标签: DRAM带宽, copy_, 宽向量, quant天花板

- **MI355X 1R:1W DRAM copy 真上限 ≈ 6.3 TB/s**：须用自定义宽向量 copy 测得（`buffer_load`/`buffer_store` i32，vw=2/4 = 8/16B）。torch `copy_` 只有 ~5.0 TB/s（低 25%），**❌ 别再拿 torch copy_ 当 DRAM 天花板**——它自身有 25% 的开销缺口，不反映硬件物理上限。
- **WHY 关键**：曾据 torch `copy_` 的 ~5.0 TB/s 误判 quant "已贴墙 90% / 物理锁死"，**这是错的**。真上限是 ~6.3 TB/s，quant 还有空间，别据假天花板宣布物理无解。
- **vw2 vs vw4（8B vs 16B 事务）几乎无差别**：上限 ~6.3 TB/s 与事务粒度无关，**❌ 别再靠加大事务粒度（16B）指望突破带宽上限**——不是杠杆。
- 呼应 68（别轻易宣布物理无解）：宣布"贴墙/物理锁死"前，先用自定义宽向量 copy 测真上限，不要用带自身开销的 framework 原语（torch copy_）当基准。

---
来源: mxfp8-8wave-devloop/SKILL.md
