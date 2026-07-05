# Split-K 修少-tile 大-K 欠订阅:asm 自包含只改 Python body、bf16 atomic 累加

> 类别: 方法论 · 主题标签: split-k, few-tile, bf16-atomic, occupancy

## 何时用 split-K(候选门)
- 目标形状:few-tile 大-K,即一 WG/tile 撑不满 CU(tiles < ncu/2)且 K≥2048 → tile 数不足以填满 device,K 维长足以切分摊。
- 实证:`1024×1024×8192` fly/ait 从 **2.00 → 1.07**(修少-tile 大-K 欠订阅)。
- Split-K factor 候选集:**2/3/4/6/8/12/16**(仅对 few-tile 大-K 报)。
- Skinny/small-M GEMM(M≤~16–32、N*K 大,如 decode/output-projection)天然配对 split-K,因为 M 填不满 device。
- 实证(源自 gpt_oss2 fp8 项目 2900 TF 下一步,非本环境,见 gpt_oss2_docker/myskill/flydsl-fp8-gemm-tuning/SKILL.md):big-K 只 **1024 tiles** 填不满 CU → **split-K=2 → grid 翻倍**改善 CU 饱和(需加部分和 reduction,big-K 输出仅 **134MB** 可行);big-N tiles 已足(**3584**),split-K 帮助有限。

## asm K-loop 自包含 → split-K 不动一行 asm
- asm K-loop 由「传入 ki 次数 + 初始 soffset + 常量步进」驱动,**不依赖 K**,因此 split-K 只改 Python body:
  - loop 次数(ki)
  - 操作数 soffset 位移
  - scale soffset 位移
  - 输出 workspace 布局 `[ksplit*M, N]` + `base_row` 位移
  - grid × ksplit
- Host 端 reduce:`ws.view(ksplit, M, N).sum(0)`,用 **bf16 累加**(比 fp32-upcast 快 **2×**,误差~**1 ULP**)。

## kernel 内 packed-bf16 atomic 累加(去 intra-CTA 争用)
- 用 `buffer_atomic_pk_add_bf16` reduce bf16 partials(每 32-bit op 打包两个 bf16)。
- 每 warp 的 per-lane byte offset 偏 `warpid*stride`,使并发 warp 打 **disjoint packed slot**,不在同一 RMW word 上碰撞。
- 目标沿 disjoint 轴需 ≥ `num_warps*2*wave_size` 个 bf16 列:**4 warp ≥512**、**8 warp ≥256**,否则 warp aliasing、争用回归。
- 只消除 intra-CTA 争用;跨-CTA 写同一 tile 仍争用。
- bf16 global atomics 仅在 **gfx94+/gfx95+/gfx12+** 存在 → trace 时 arch-gate,否则回退 scaled-f32。

## 与 tile-size / L2 swizzle 的关系
- split-K 仅对 few-tile 大-K 场景补一 WG/tile 撑不满 CU 的欠订阅;不与 L2 swizzle 冲突(后者是纯 WG→tile 双射,bit-identical,只调 L2 residency)。
- skinny-GEMM 一并注意:hard-wire `tile_m=16` + 单 M-warp、把每个 wave 铺满 N;N-tile A-reuse 摊掉 A load,但 loop-carried 寄存器随 N-tile-repeat 倍增可压垮 occupancy → 配 `waves_per_eu` 旋钮。

---
来源: project_mxfp4_epilogue_store.md, gemm/optimization-directions.md, 13-primus-turbo-prod.md, flydsl-fp8-gemm-tuning/SKILL.md
