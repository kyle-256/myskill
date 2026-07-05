# 低精度融合 dequant 进 MFMA 内循环、验证策略、NOSTORE 归因

> 类别: 方法论 · 主题标签: dequant-fuse, mxfp4-pack, 低精度验证, NOSTORE归因

## dequant 融进 MFMA 内循环(非 epilogue)
- block-scale dequant 必须融进 MFMA 内循环,不是 epilogue:在 compute tile 内解码当前 K 的 per-block FP8 scale 切片,直接折进 accumulator。
- **tail tile 必须重新(FRESHLY)解码 scale**:沿用 prologue 的 scale state / 用了 stale scale 会污染最后一个 K block。
- FP8 requant 带来的 **1-5% mismatch 是预期容差,不是 bug**。

## 低精度对标要选同类 MFMA 指令
- 对标口径细节见 methodology/50-mxfp8-grouped-fair-compare-occ-ceiling.md「MX vs TW 唯一公平口径」。

## 低精度验证策略(profiling 不够,必须分步验)
1. 单独验 conversion 和 packing。
2. block-scaled 路径单独验 scale 处理。
3. kernel 数学对比高精度 / dequant 参考。
4. 只有前三步都过了才信性能数字。

## NOSTORE 归因 + 回归分离稳态 compute
- CUDA-graph bench 比 rocprof-min 干净:replay 确定性;rocprof-min 跨后端不可信(profiling 扰动曾误报 FLY 反超)。
- gated `NOSTORE` env(epilogue store 直接 return)做归因:**full 时间 − nostore 时间 = 暴露的 store 成本**。
- **斜率/截距回归(us/K-block)**:斜率 = 稳态 compute,截距 = 固定 prologue/epilogue 开销,借此分离两者。

## Branchless f32→E2M1 (MXFP4) pack / W4A16-W4A8 打包
- 详见 methodology/38-cross-lane-mfma-primitives.md「mxfp4:branchless E2M1 pack」与「W4A16/W4A8 preshuffle」。

---
来源: gemm/optimization-directions.md, tool-rocprof/SKILL.md, project_mxfp4_epilogue_store.md, optimization-directions.md, project_mxfp8_wholeloop_port.md
