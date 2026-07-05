# CDNA4-only 路径无 gfx942 fallback：MX/F8F6F4/transpose-load 移植需重架构

> 类别: 踩过的坑 · 主题标签: porting, gfx950-vs-gfx942, MX-MFMA, F8F6F4, transpose-load

**核心结论：gfx950→gfx942 移植不是重编译，是重架构。** 一批 CDNA4-only 指令/数据类型在 gfx942 上根本不存在，且没有 fallback：block-scaled MX MFMA、F8F6F4 mixed MFMA、DS_READ_*_TR transpose-load、FP6/FP4 dtype、V_PRNG_B32、V_BITOP3、scalar atomics、v_permlane16_swap。移植 = dequantize-then-MFMA / 显式 VGPR transpose / 多指令 bitop 展开。(src: overview.md)

**MX scale 只活一条指令 (block-scaled MX MFMA, gfx950)**
- V_MFMA_SCALE_F32_*_F8F6F4 里的 E8M0 scale **只作用于紧跟的这一条指令**；每个 K-block 都要各自的 SCALE op。后面接一条 plain V_MFMA_*_F8F6F4 就是**无 scale**的 —— 静默丢 scale。
- E8M0 的 **0xFF = NaN**。打包前必须 clamp 量化后的 exponent，否则整个结果被污染 (poisoned)。
- ❌ 别再试 用 {OP_SEL_HI, OP_SEL} 随手选 scale byte：选错会**静默地 scale 到错误的 K-block**，不报错。
- lane→(M, K-block) 布局是**固定且非对称**的，且 16x16x128 与 32x32x64 两种 MFMA **布局不同**。照文档逐字用，❌ 别再试 自己推断布局。(src: gfx950/kernel-implementation-notes.md)

**硬件 stochastic-round FP8 转换必须自己推进 PRNG (gfx950)**
- `V_CVT_SR_*` 须由 kernel 用 per-lane PRNG(Philox/LCG/threefry)驱动 seed 并**每次调用推进它**,否则 'stochastic' rounding 退化成有偏的 deterministic rounding。
- gfx950 上 `V_PRNG_B32` 用一条 VALU op 推进 per-lane LFSR;**multipass SR converters 只读不写 PRNG VGPR**,所以必须自己推进。(这是 gfx950/CDNA4 通用事实,做 SR-FP8 的都要注意,别只在 RDNA 移植时想起。)

**F8F6F4 靠 packing 选格式，不是靠 opcode (gfx950)**
- mixed MFMA 选哪个操作数格式 (FP8/BF8/FP6/BF6/FP4) 取决于 mantissa 在操作数 VGPR 里**怎么 packing**，不是换 opcode。
- ❌ 别再试 靠改 opcode 切格式 —— packing 错了会**静默地算错**，但仍然 type-check 通过、正常跑。host 端的 permutation 与 device 端 descriptor 必须**逐字一致**。
- F8xF8 **拿不到 small-cycle path**：cycle 数是 16 或 32。(src: gfx950/kernel-implementation-notes.md)

**DS_READ_*_TR transpose-load 仅 gfx950 (gfx942 完全没有)**
- 发射前 **EXEC 必须全 1**；LDS 地址要按数据大小对齐。≥64-bit 的 DS op 需要 **偶对齐 VGPR**（B96_TR_B6 例外）。
- gfx942 **没有 transpose-load**。❌ 别再试 把 CDNA4 的 transpose-load tile loader 原样移植 —— 改用 pre-swizzled LDS write 或显式 VGPR transpose。
- 反向：在 gfx950 上跳过 DS_READ_*_TR = 白扔一个 CDNA4-specific 的性能收益。(src: gfx950/kernel-implementation-notes.md)

**TF32 / FP64 的代际反直觉 —— 别假设 newer 更快**
- CDNA4 **没有硬件原生 TF32 matrix path**。❌ 别再试 把 CDNA3 的 TF32 path 导入 gfx950。
- gfx942 的 XF32/TF32 MFMA 会**静默把 mantissa 截到 10 bit** —— 需要完整 FP32 精度处绝不能用。
- FP64 matrix 率：gfx942/CDNA3 是 **256 FLOP/cycle/CU**，gfx950 是 **128**（CDNA4 把 per-CU FP64 **砍半**）。FP64/HPC matmul 用 gfx942（**304 CU**）更强。
- gfx950 赢在 FP16/BF16/FP8（per-CU 低精度约 **2×**），且是**唯一**有 block-scaled MX / FP6 / FP4 的代。(src: gfx942/overview.md, gfx942/kernel-implementation-notes.md)

---
来源: overview.md, gfx950/kernel-implementation-notes.md, gfx942/overview.md, gfx942/kernel-implementation-notes.md
