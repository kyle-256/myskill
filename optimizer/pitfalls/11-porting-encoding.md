# 跨代移植与编码正确性门：CDNA3/CDNA4/RDNA 编码差异、按 arch 重编码 scale（correctness-gate/数据构造详见 pitfalls/04）

> 类别: 踩过的坑 · 主题标签: correctness-gate, SNR, rocprof, quant, hipBLASLt, fp8-encoding, cross-arch, scale, flydsl-isa, porting, gfx950-vs-gfx942, MX-MFMA, F8F6F4, transpose-load, rdna-porting, wmma, autotune-dispatch, stochastic-rounding

## "是不是自己改坏"诊断捷径（→ 详见 pitfalls/04）

- **跨后端比 SNR 判"是不是自己改坏"、rocprof 实锤 quant-bound / hipBLASLt 误区**：这三条属 correctness-gate 主题，完整版见 **pitfalls/04**（同 shape 跑两后端比 SNR，SNR 撞同值 → 问题在上游共用量化/参考路径非 kernel；fp8 grouped 走 FLYDSL 非 hipBLASLt，rocprof 无 `Cijk_*`；op-level bwd 慢是 dgrad/wgrad 本身非 quant，量化 kernel 占比 0.0%）。移植调试先跑这套判据再动 kernel。

## FP8 编码是跨代正确性门：FNUZ(gfx942) vs OCP(gfx950)，scale 不重编码静默错

- **FP8 编码是跨代正确性门，不是性能问题**：
  - gfx942/CDNA3 = **FNUZ**：E4M3 bias 8, **max 240**, no Inf；BF8 E5M2 bias 16。
  - gfx950/CDNA4 = **OCP**：E4M3 bias 7, **max 448**；BF8 E5M2 bias 15, max 57344, **有 Inf**。MXFP8/MXFP4 需要 gfx950。
  - 移植时必须按新 arch 重编码 per-tensor / per-block scale。**复用另一代的 scale 静默产错——不 crash，输出数值直接错**。**recompile 不修**。编码在 `core/low_precision.py` 里按 arch 选。
- **HK fp8 max=240（fnuz）非标准 E4M3（448）、scale_inv 按 448 算偏 ~5x、别用 `randn().to(FP8)` 造数据**：这几条属数据构造主题，完整版（含 `quantize_fp8_tensorwise_impl` 自动算 max、saturate 机制）见 **pitfalls/04**。移植时的 porting 要点已在上一条：scale 必须按目标 arch 的编码（fnuz max=240 vs ocp max=448）重算。
- **多编码可能时绝不能只标 "fp8"**：每个 FP8/FP6/FP4/MXFP round 必须显式记录：目标 arch、精确输入输出格式、accumulator 格式、scale 格式与 scale 粒度、用的 tolerance、失败属于 conversion / scale / math 哪一类。否则无法定位是哪代编码错。
- **`SH_MEM_CONFIG` bit[8]（控制 FP8/BF8 convert 输出编码的使能位）必须为 1** 才能得到正确 FP8/BF8 结果（两代都要）；为 0 时 convert 会按错误编码写出。
- **CVT_*_F32 上转换的目标寄存器无写后转发（write-forwarding）**：两条 convert 写同一目标寄存器的不同 byte/half 时，第二条可能读到第一条尚未落地的 stale byte——中间必须插一个 NOP 或无关 VGPR 写。
- **`FP16_OVFL` 是 MODE bit（saturate vs NaN/Inf），不是 per-instruction**：切换它会影响 wave 后面所有 convert。
- **FlyDSL/gfx950 fp8 cast 三死路（Vec.to(Float8E4M3FN) / cvt_scalef32_pk8_fp8_bf16 / create_buffer_resource(max_size=True)）**：属 flydsl-ISA 约束，完整版见 **pitfalls/04**（正确做法：fp8 cast 用 `fx.rocdl.cvt_pk_fp8_f32`；buffer resource 用 `max_size=False, num_records_bytes=...`）。移植到 gfx950 时用得上，故在此留索引。
- **FlyDSL 常见编译错误修法清单**：
  - `RecursionError` → kernel / torch 分拆成独立文件。
  - `'Float32' has no bitcast`（`.reduce("max")` 返回 Float32）→ 用 `(F32(0.0)>ca).select(F32(0.0), ca)` 包一层。
  - `arith.truncf not lowerable`（`.to(fp8)`）→ 改 `rocdl.cvt_pk_fp8_f32`（同上条 fp8 cast 约束）。
  - `ScalarizeVectorOperand LLVM ERROR`（`buffer_store` 写 bf16 scalar）→ 改写 int32 打包 fp8。
  - `'Int32' has no bitcast` → 用 `fm.exp2(ep.to(F32))`。

## CDNA4-only 路径无 gfx942 fallback：MX/F8F6F4/transpose-load 移植需重架构

**核心结论：gfx950→gfx942 移植不是重编译，是重架构。** 一批 CDNA4-only 指令/数据类型在 gfx942 上根本不存在，且没有 fallback：block-scaled MX MFMA、F8F6F4 mixed MFMA、DS_READ_*_TR transpose-load、FP6/FP4 dtype、V_PRNG_B32、V_BITOP3、scalar atomics、v_permlane16_swap。移植 = dequantize-then-MFMA / 显式 VGPR transpose / 多指令 bitop 展开。(src: overview.md)

**MX scale 只活一条指令 (block-scaled MX MFMA, gfx950)**
- V_MFMA_SCALE_F32_*_F8F6F4 里的 E8M0 scale **只作用于紧跟的这一条指令**；每个 K-block 都要各自的 SCALE op。后面接一条 plain V_MFMA_*_F8F6F4 就是**无 scale**的 —— 静默丢 scale。
- E8M0 的 **0xFF = NaN**。打包前必须 clamp 量化后的 exponent，否则整个结果被污染 (poisoned)。
- ❌ 别再试 用 {OP_SEL_HI, OP_SEL}（选 E8M0 scale 在源 VGPR 里取哪个 byte 的 modifier 字段）随手选 scale byte：选错会**静默地 scale 到错误的 K-block**，不报错。
- lane→(M, K-block) 布局是**固定且非对称**的，且 16x16x128 与 32x32x64 两种 MFMA **布局不同**。照文档逐字用，❌ 别再试 自己推断布局。(src: gfx950/kernel-implementation-notes.md)

**硬件 stochastic-round FP8 转换必须自己推进 PRNG (gfx950)**
- `V_CVT_SR_*` 须由 kernel 用 per-lane PRNG(Philox/LCG/threefry)驱动 seed 并**每次调用推进它**,否则 'stochastic' rounding 退化成有偏的 deterministic rounding。
- gfx950 上 `V_PRNG_B32` 用一条 VALU op 推进 per-lane LFSR;**multipass SR converters 只读不写 PRNG VGPR**,所以必须自己推进。(这是 gfx950/CDNA4 通用事实,做 SR-FP8 的都要注意,别只在 RDNA 移植时想起。)

**F8F6F4 靠 packing 选格式，不是靠 opcode (gfx950)**
- mixed MFMA 选哪个操作数格式 (FP8/BF8/FP6/BF6/FP4) 取决于 mantissa 在操作数 VGPR 里**怎么 packing**，不是换 opcode。
- ❌ 别再试 靠改 opcode 切格式 —— packing 错了会**静默地算错**，但仍然 type-check 通过、正常跑。host 端的 permutation 与 device 端 descriptor 必须**逐字一致**。
- F8×F8 mixed MFMA **走不到 low-cycle 快路径**（该快路径留给更窄格式，如 FP6/FP4）：F8×F8 固定 16 或 32 cycle/MFMA。(src: gfx950/kernel-implementation-notes.md)
- whole-loop MFMA 格式切换：asm 从 fp4 的 `cbsz:4 blgp:4` → mxfp8 的 `cbsz:0 blgp:0`(E4M3)；E5M2/HYBRID 时 cbsz/blgp 按 operand format(**0=E4M3, 1=E5M2**)，**scale 路径不变**。(src: project_mxfp8_wholeloop_port.md)

**DS_READ_*_TR transpose-load 仅 gfx950 (gfx942 完全没有)**
- 发射前 **EXEC 必须全 1**；LDS 地址要按数据大小对齐。≥64-bit 的 DS op 需要 **偶对齐 VGPR**（B96_TR_B6 例外）。
- gfx942 **没有 transpose-load**。❌ 别再试 把 CDNA4 的 transpose-load tile loader 原样移植 —— 改用 pre-swizzled LDS write 或显式 VGPR transpose。
- 反向：在 gfx950 上跳过 DS_READ_*_TR = 白扔一个 CDNA4-specific 的性能收益。(src: gfx950/kernel-implementation-notes.md)

**TF32 / FP64 的代际反直觉 —— 别假设 newer 更快**
- CDNA4 **没有硬件原生 TF32 matrix path**。❌ 别再试 把 CDNA3 的 TF32 path 导入 gfx950。
- gfx942 的 XF32/TF32 MFMA 会**静默把 mantissa 截到 10 bit** —— 需要完整 FP32 精度处绝不能用。
- FP64 matrix 率：gfx942/CDNA3 是 **256 FLOP/cycle/CU**，gfx950 是 **128**（CDNA4 把 per-CU FP64 **砍半**）。FP64/HPC matmul 用 gfx942（**304 CU**）更强。
- gfx950 赢在 FP16/BF16/FP8（per-CU 低精度约 **2×**），且是**唯一**有 block-scaled MX / FP6 / FP4 的代。(src: gfx942/overview.md, gfx942/kernel-implementation-notes.md)

## RDNA/WMMA 移植坑：别导 CDNA 调度旋钮，WMMA 代际差异限于 4 处

- ❌ 别再试：把 CDNA 调度旋钮导入 RDNA。`sched_mfma`/`s_setprio` 的比例是 MFMA/wave64 专用的；WMMA 是 wave32，正确的 interleave 完全不同，照搬 CDNA 比例只会打乱调度。
- **WMMA 代际差异只集中在 4 处**（其余代码路径 CDNA/RDNA 共用，别到处改）：
  1. **`_wmma_op` call**：gfx11 是 v16 operands，lanes 16-31 镜像 lanes 0-15；gfx120x 是 v8 operands。
  2. **LDS-read shape**：读法随代际变。
  3. **accumulator store-back row 公式**：用错公式会**静默地把输出行转置**（不报错，结果错），最难查。
  4. **barrier asm**：gfx11 = `s_waitcnt lgkmcnt(0)` + `s_barrier`；gfx12+ = 拆成 `s_barrier_signal` / `s_barrier_wait` / `s_wait_dscnt` 三条。
- **内层 WMMA 循环保持 'load all B, then 1 A -> reg_n WMMAs'**。❌ 别再试反转成先 load A：反转会膨胀寄存器压力并 spill。
- **CDNA vs RDNA 判定的单一真相是 `is_rdna_arch()`**（`python/flydsl/runtime/device.py`）。❌ 别再试硬编码 `gfx*` 条件。
  - `wave32-true` 前缀匹配只覆盖 `gfx10*`/`gfx11*`/`gfx120*`，**漏了 `gfx1250`**（前缀 `gfx125*` 不在列表里）。
  - wave size 共享逻辑：`get_warp_size(arch) = 32 if is_rdna_arch else 64`，因 gfx1250 不被 `is_rdna_arch` 识别 → **返回 64**。若 gfx1250 实际需 wave32，则 kernel 须自己显式设 wave32，否则 wave size 错。
- 硬件 stochastic-round FP8 转换（`V_CVT_SR_*`）须自己驱动 PRNG：见本卡「CDNA4-only 路径无 gfx942 fallback」小节（硬件 stochastic-round FP8 转换必须自己推进 PRNG）

---
来源: remote-sync/SKILL.md, 08-deadends.md, fp8-gemm-bench/SKILL.md, overview.md, tool-rocprof/SKILL.md, flydsl-sync/SKILL.md, gfx942/kernel-implementation-notes.md, gfx950/kernel-implementation-notes.md, mxfp8-8wave-devloop/SKILL.md, gfx942/overview.md, project_mxfp8_wholeloop_port.md [FLAG: MEMORY.md 索引查无此条,疑似死引用,待核实], optimization-directions.md, FlyDSL/CLAUDE.md
