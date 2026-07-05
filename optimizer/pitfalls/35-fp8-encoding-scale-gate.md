# FP8 编码是跨代正确性门：FNUZ(gfx942) vs OCP(gfx950)，scale 不重编码静默错

> 类别: 踩过的坑 · 主题标签: fp8-encoding, cross-arch, scale, flydsl-isa

- **FP8 编码是跨代正确性门，不是性能问题**：
  - gfx942/CDNA3 = **FNUZ**：E4M3 bias 8, **max 240**, no Inf；BF8 E5M2 bias 16。
  - gfx950/CDNA4 = **OCP**：E4M3 bias 7, **max 448**；BF8 E5M2 bias 15, max 57344, **有 Inf**。MXFP8/MXFP4 需要 gfx950。
  - 移植时必须按新 arch 重编码 per-tensor / per-block scale。**复用另一代的 scale 静默产错——不 crash，输出数值直接错**。**recompile 不修**。编码在 `core/low_precision.py` 里按 arch 选。
- **HK fp8 是 `float8_e4m3fnuz`(max=240)不是标准 E4M3(max=448)**：手写 `scale_inv=1/112`(按 448 算)会让输出**偏 ~5x**。
  - ✅ 必须用 `quantize_fp8_tensorwise_impl` 自动算正确 max → scale_inv。
  - ❌ 别再试 `torch.randn().to(FP8)` 直接造数据：>240 会 saturate，且 mma 非线性 → 结果不可信。quantize helper 会自动 clamp。
- **多编码可能时绝不能只标 "fp8"**：每个 FP8/FP6/FP4/MXFP round 必须显式记录：目标 arch、精确输入输出格式、accumulator 格式、scale 格式与 scale 粒度、用的 tolerance、失败属于 conversion / scale / math 哪一类。否则无法定位是哪代编码错。
- **`SH_MEM_CONFIG` bit[8] 必须为 1** 才能得到正确 FP8/BF8 结果（两代都要）。
- **CVT_*_F32 上转换无 4-cycle forwarding**：两条 convert 写同一目标寄存器的不同 byte/half 时，中间必须插一个 NOP 或无关 VGPR 写，否则读到 stale byte。
- **`FP16_OVFL` 是 MODE bit（saturate vs NaN/Inf），不是 per-instruction**：切换它会影响 wave 后面所有 convert。
- **FlyDSL/gfx950 fp8 硬约束**：
  - ❌ 别再试 `Vec.to(Float8E4M3FN)`：走 `arith.truncf`，后端**不 lower**。fp8 cast 要用 `fx.rocdl.cvt_pk_fp8_f32`（2 f32 → 2 fp8/op）。
  - ❌ 别再试 `cvt_scalef32_pk8_fp8_bf16`：gfx950 上 **Cannot select，不可用**。
  - ❌ 别再试 `create_buffer_resource(max_size=True)`：会 OOB 读到垃圾。用 `max_size=False, num_records_bytes=...`。

---
来源: fp8-gemm-bench/SKILL.md, overview.md, tool-rocprof/SKILL.md, flydsl-sync/SKILL.md, gfx942/kernel-implementation-notes.md
