# MXFP8 死路清单（勿重试）：asm-AGPR MFMA / scale via LDS / host 重排 / 大 BM 不合并 quant

> 类别: 踩过的坑 · 主题标签: mxfp8, quant, LDS, MFMA

MXFP8 死路速查表——以下方向均已实测判负，勿重试。

| 死路 | 机制/根因 | 实测数字 |
|---|---|---|
| ❌ asm-AGPR MFMA inline_asm | acc-clobber 墙：inline_asm 无法正确声明 accumulator clobber | SNR garbage（结果全错） |
| ❌ scale via LDS 双缓冲 | vmcnt 计数失准 → 流水崩 | 慢 6× |
| ❌ host 侧任何缓存/重排 | 用户硬否决 | —（禁止） |
| ❌ quant 128B 整 cache line 写 (旧 v2, BM=128) | 各 lane 写满 128B，无跨-lane 合并 | 慢 0.58–0.98× |
| ❌ quant wave 跨-lane 合并转置写 (旧 v3) | 双趟 scale 也暂存 LDS（错误做法） | 慢 0.42–0.61× |
| ❌ BLOCK_K=256 移植 MoE grouped GEMM | MoE 是 BW-bound，BLOCK_K=256 回退 | 回退变慢 |

关键区分（别把制胜招误判成死路）：
- ❌ 别再试 **大 BM 但不合并** 的 quant 写（旧 v2 BM=128 各 lane 写满 128B 无跨-lane 合并）。但**大 BM 配合 LDS 合并才是制胜招**——死的是"大 BM 不合并"，不是"大 BM"本身。
- ❌ 别再试 **scale 也暂存 LDS**（旧 v3 双趟 scale 暂存 LDS 慢 0.42–0.61×）。教训：**只暂存 fp8 数据、不暂存 scale**，且让两半并发（不串行）才能反超。

---
来源: mxfp8-8wave-devloop/SKILL.md
