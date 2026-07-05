# GEMM tile 尺寸选择:256×256 主 tile、小-M BM128、三级 tiling、MFMA 数账

> 类别: 方法论 · 主题标签: tile-size, mfma, arithmetic-intensity, occupancy

## 主 tile 与 dgrad 小-M 修法
- GEMM 用 **256×256 主 tile**(BLOCK_M=256/BLOCK_N=256)。只有 dgrad **小-M occupancy 欠载**才降 BLOCK_M=128。
- dgrad 小-M 修法 = BLOCK_M=128 M-branch:M-tile 数翻倍(M=512:64→128 tile)填满 CU。gate 通过时 **bm128 永远赢(+5.3~30.6%)**,单 config 直接 return,不需 autotune。
- 约束:BLOCK_M >= 128 且 %128==0(kernel 有 assert)。

## 三级 tiling 与派生量
- 三级 tiling:**block_x→M tiles、block_y→N tiles、K 为 reduction**。
- 线程 256 = 4 wave×64 lane:`wave_id=tid//64` 划 N;`lane_div_16=lane//16` 划 M(4 组×16);`lane_mod_16` 划 MFMA 内 N。
- 派生量:
  - `m_repeat = tile_m//16`
  - `n_per_wave = tile_n//4`
  - `num_acc_n = n_per_wave//16`
  - `k_unroll = tile_k_bytes // a_elem_vec_pack // 64`

## 推荐 tile 配置(tile_m / tile_n / tile_k)
| 场景 | 配置 | WHY |
|---|---|---|
| 小 batch M≤32 | 16 / 64-128 / 256-512 | memory-bound,大 tile_k 换复用 |
| 中 batch | 64 / 256 / 128 | 均衡 |
| 大 batch M≥4096 | 128 / 256 / 128 | compute-dense,需 async copy |
| FP4 (gfx950) | 32-64 / 128-256 / 256 | MFMA_SCALE |

## 每 tile MFMA 数账
- **每 tile MFMA 数 = k_unroll × m_repeat × num_acc_n × 2**(末尾 2 = 每 K64 微步做 2 次 K32 MFMA)。
- 例:tile 64×256×128 FP8 → 2×4×4×2 = **64 MFMA**;tile 64×256×512 → k_unroll=8 → **256 MFMA**。
- MFMA 内循环 K64 微步(FP8/INT8)三重循环:
  - 外 `ku`(k_unroll 个 K64 步):取 b_packs0/b_packs1
  - 中 `mi`(m_repeat 个 16 行块):`lds_load_packs_k64` 取 A(2×i64)
  - 内 `ni`(num_acc_n 个 16 列累加器):`mfma_k64_bytes`;每 K64 做 2 次 K32 MFMA

## MFMA 指令选择表(累加器均 f32×4)
| 数据类型 | 指令 | K |
|---|---|---|
| FP8 | `mfma_f32_16x16x32_fp8_fp8` | 32 |
| INT8 | `mfma_i32_16x16x32i8` | 32 |
| BF16 | `mfma_f32_16x16x16bf16_1k` | 16 |
| FP16 | `mfma_f32_16x16x16f16` | 16 |
| FP4 (gfx950) | `mfma_scale_f32_16x16x128_f8f6f4` | 128 |

## Roofline 判据
- M≤512 通常 memory-bound(关注带宽);M>512 compute-bound(关注 MFMA 利用率)。
- 算术强度 = flops / bytes_moved,与 roofline crossover 比较。

## 方形大 tile 提算术强度(4-wave 长 K 领先 8-wave 核心机制)
- 强度定义 = 每 K-step 的 MAC / (A行 + B列 operand-elem)。
- 具体数据(算术强度对比、LDS insts 实测比值、随 K 的领先幅度):见 methodology/28-whole-loop-asm-lds-feed-bound.md（算术强度小节）

---
来源: 03-nn-dgrad-kernel.md, gemm-optimization/SKILL.md, diag_4w_vs_8w.md
