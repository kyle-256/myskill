# 别接永不被采纳的候选：4w-persistent NT 对真实 8w-best <4% 净改动为零

> 类别: 踩过的坑 · 主题标签: autotune-dispatch, persistent-vs-nonpersistent, nt-kernel, l2-swizzle

- **接生产前必须证明会被采纳**：新内核变体接进 autotune 前，先证它在**真实全量候选池**里能按采纳门槛（>4% 滞后，用来压 DVFS 噪声）被选中——分母必须是**真实 dispatch best**，不是手挑的弱子集。否则只白加编译开销，净改动为零。（pr-merge-gate/SKILL.md）
- **grouped NT 4w-persistent 实测被否**：对真实 8w-best 仅 ~3%（<4% 门槛），且在它本该赢的**短-K 区又输给 4w-np**→已撤销，净改动为零。（pr-merge-gate/SKILL.md）
- **fwd/dgrad 上 4-wave ≈ 8-wave-persistent**：打平（±2% 噪声内），autotuner 多数选 8-wave-persistent；dense/grouped 的 4w-persistent NT 候选几乎不被采纳。4-wave 的价值集中在 **wgrad（variable-K）**：早期生产 autotune 对比（auto vs 8w-only）**+6~17%**；后续修复 dispatch bug 并用 `PT_WARMUP=250` 校正后的 A/B（`c846d954` vs main）全 7 shape × 2 m 无一回归，大 contraction / 宽-N **+9~19%**，qwen/synth +3~9%。（flydsl-fp8-gemm-results/SKILL.md）

- **持久 vs 非持久（各形状归属不同）**：见 pitfalls/33-persistent-vs-nonpersistent-vmcnt.md

- ❌ **别再试**：非持久 nt kernel 不移植 L2 swizzle。小-K shape 会**反输**（gpt-down −13%）。非持久优势只在大-K（循环调度惩罚主导）；小-K 靠 swizzle（L2 reuse）补回，缺了就崩。（02-nt-fwd-kernel.md）

- ❌ **别再试**：8-wave / BK128 换 occ=2 藏 store。原生 occ=2 的 8-wave kernel 在 store-bound 形状实测**全负**：28672 −14%、6144³ −20%、8192²×4096 −20%。occ=2 能藏 7-15% store，但 8-wave compute 赤字 14-20%（per-warp tile 减半→B 复用减半→ds_read/mfma 翻倍）远大于收益。与 4w+BK128 −13% 同结论。（project_mxfp4_epilogue_store.md）

---
来源: pr-merge-gate/SKILL.md, flydsl-fp8-gemm-results/SKILL.md, flydsl-fp8-gemm-tuning/SKILL.md, 02-nt-fwd-kernel.md, project_mxfp4_epilogue_store.md
