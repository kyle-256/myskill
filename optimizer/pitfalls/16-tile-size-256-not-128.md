# GEMM tile 别用小/矩形：256×256 唯一可行点，feed-bound 缩 tile 净负

> 类别: 踩过的坑 · 主题标签: tile-size, feed-bound, LDS/AGPR-occupancy, mxfp4

## tile 尺寸硬约束（违反=数据错或编译失败）
- tile_m 必须是 **16 的倍数**（MFMA M 维）。
- tile_n 必须是 **64 的倍数**（4 wave × 16 N）。
- tile_k*elem_bytes 必须被 **64 整除**（K64 字节微步）。
- tile_m*tile_k*elem_bytes 要舒服放进 LDS：**gfx942 64KB / gfx950 160KB**。
- B 预 shuffle 成 `(N/16, K/64, 4, 16, kpack_bytes)`，**tile_k 必须整除 K**。
- default first-pass：HIP/CK 256 threads / 4 waves 拥有一个 (tile_m,tile_n)，合理起点 **tile_m=256, tile_n=128, tile_k=64**。
- 占用边界：越过 **128 arch-VGPR** 掉 2 waves→1 wave/SIMD；越过 **256 VGPR** spill。启动足够 work-group 盖满全部 CU（**304 gfx942 / 256 gfx950**），否则小 grid 灌不满器件。

## ❌ 别再试：feed-bound worst shape 缩小/矩形 tile
- 现场：wgrad worst shape gpt_oss-down **2880²**，256×256 只 **~48% peak（MfmaUtil 54%）= feed-bound**。
- 缩 tile 直接砍算术强度 (M·N)/(M+N)：
  - 256² = **128**（基线）
  - 128×256 = **85（-33%）**
  - 128² = **64（-50%）**
- feed-bound 下缩 tile 净负**远超** padding+grid 能救回来的量。**tile 重写不做。**

## ❌ 别再试：tile-menu 多形状 / 大 tile（AITER 那套）
AITER 有 per-shape 选 128×128 / 192×256 / 256×512… 在本架构（gfx950 mxfp4）不可行：
- **256×512**：acc 超 **256 AGPR** → occ=1。
- **128×512**：standalone 落后。
- **512×256**：超 **144KB LDS**。
- 结论：**256×256 是 gfx950 mxfp4 唯一可行 tile 点。**

## ❌ 别再试：BLOCK_N=512（撞 CShuffle EPL assert）
- BLOCK_N=512 撞 **CShuffle EPL assert（EPL=16 ≠ 8）**。
- bigger tile（256×512、512×256）会报 **INVALID_ISA** 或 CShuffle EPL assert。
- **不要扫 bn=512。**

## ❌ 别再试：对 6144 用 split-K
- 6144³ 残差 fly/ait~**1.05** = tile 量化：256×256 → **576 tiles / 256 CU = 2.25 wave 不均**；AITER 192×256 → **768/256 = 3 整除**。
- 但 split-K 切细**反而更慢**：576 tile compute 已高效、无尾波浪费。**别对 6144 用 split-K。**

---
来源: flydsl-fp8-gemm-results/SKILL.md, gemm-optimization/SKILL.md, flydsl-kernel-authoring/SKILL.md, gemm/overview.md, project_mxfp4_epilogue_store.md, 03-nn-dgrad-kernel.md
