# 低精度融合 dequant 进 MFMA 内循环、验证策略、NOSTORE 归因

> 类别: 方法论 · 主题标签: dequant-fuse, mxfp4-pack, 低精度验证, NOSTORE归因

## dequant 融进 MFMA 内循环(非 epilogue)
- block-scale dequant 必须融进 MFMA 内循环,不是 epilogue:在 compute tile 内解码当前 K 的 per-block FP8 scale 切片,直接折进 accumulator。
- **tail tile 必须重新(FRESHLY)解码 scale**:沿用 prologue 的 scale state / 用了 stale scale 会污染最后一个 K block。
- FP8 requant 带来的 **1-5% mismatch 是预期容差,不是 bug**。

## 低精度对标要选同类 MFMA 指令
- **mxfp8 是 scaled-MFMA**(`v_mfma_scale_f32` 每条多读 2 个 scale 操作数),合理对标是 **scaled 的 aiter mxfp8**(mxfp4 达 98%),**不是**非-scaled per-tensor(per-tensor 用 `v_mfma_f32` 非-scaled)。别拿非-scaled 基线冤枉 scaled 内核。

## 低精度验证策略(profiling 不够,必须分步验)
1. 单独验 conversion 和 packing。
2. block-scaled 路径单独验 scale 处理。
3. kernel 数学对比高精度 / dequant 参考。
4. 只有前三步都过了才信性能数字。

## NOSTORE 归因 + 回归分离稳态 compute
- CUDA-graph bench 比 rocprof-min 干净:replay 确定性;rocprof-min 跨后端不可信(profiling 扰动曾误报 FLY 反超)。
- gated `NOSTORE` env(epilogue store 直接 return)做归因:**full 时间 − nostore 时间 = 暴露的 store 成本**。
- **斜率/截距回归(us/K-block)**:斜率 = 稳态 compute,截距 = 固定 prologue/epilogue 开销,借此分离两者。

## Branchless f32→E2M1 (MXFP4) pack 要点
- 在 i32 bit-pattern 上做 masked-select:隔离 sign/abs → denormal+normal 谓词 → normal path 用 odd-bit injection 做 RNE 再 shift → saturate 到 `0x7`。
- per-32 E8M0 block scale:`shuffle_xor` butterfly max + `(254-e8m0)` 倒数 trick,**在 conversion 之前**乘进去,再 bit-pack 成 nibble。
- `_fp_headroom` 常量 **FP4 与 FP8 不同**:取错会把整个 tensor 重缩放 `2^6`。
- 该 routine 把 NaN/Inf saturate 到 max(非 IEEE 忠实)。
- ⚠️ **E8M0 `0xFF = NaN`**:量化后的 exponent 在 bit-pack 前**必须 clamp**,否则一个 0xFF 会 poison 整块结果。(另见 `pitfalls/37`)

## W4A16 / W4A8 打包与转换
- preshuffle B,使每 lane 的 MFMA-K micro-step 读到连续 8-byte(16-nibble)pack。
- gfx950:用 `cvt_off_f32_i4`(SDWA `byte_sel` 一次 shift 覆盖全部 8 nibble)+ `cvt_pk_bf16_f32`。
- gfx942:用 shift-based f32→bf16 truncation(对 scaled int 精确,比 `truncf` 省 ~5 VALU)。
- 把唯一的 `>>4` hoist 出循环;若把 x16 correction / groupwise scale 推迟到 epilogue,则 epilogue **必须**施加它。

---
来源: gemm/optimization-directions.md, tool-rocprof/SKILL.md, project_mxfp4_epilogue_store.md, optimization-directions.md, project_mxfp8_wholeloop_port.md
