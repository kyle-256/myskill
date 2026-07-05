# 低精度容差假门：element-wise tolerance 无意义，SNR 是真 gate

> 类别: 踩过的坑 · 主题标签: 精度验证, SNR, fp8, scale-bug

- **get_tolerances 的 fp8=1e-1 / fp4=0.5 是故意放松的假门，不能当真实 gate。** 量化后 element-wise 容差本身没意义（低精度 round 误差天然大），拿它判过/不过会漏掉真 bug 或误杀正确结果。
- **低精度主 gate 必须用 `compute_snr(ref, actual)`——参数顺序 reference 在前。** 传反了 SNR 数值会错。`relative_error / mean_squared_error / max_abs_error / cosine_similarity / symmetric_similarity_diff` 只作诊断辅助，不作判定门。
- **FP8 PV MFMA 相对 bf16 参考有 ~0.03 max error，是 FP8 数据通路固有的，不是 bug。** 对应场景容差用 `atol=5e-3`（比 element-wise 假门严得多，但仍是辅助判据；主判据还是 SNR）。
- **per-row Q vs per-tensor Q 量化方式不匹配会有 1-3% 差异。** 参考实现和 kernel 的量化粒度要对齐，否则这 1-3% 会被误当成 kernel bug。
- **常见 scale bug：`v_scale` 被应用两次**——prob scaling 阶段一次、PV 之后又一次。表现为输出整体偏大，查 scale 应用点是否重复。
- **HK fp8 是 `float8_e4m3fnuz`(max=240)，不是标准 E4M3(max=448)。** 手写 `scale_inv=1/112`（按 448 算）会让输出偏 ~5x。必须用 `quantize_fp8_tensorwise_impl` 自动按真实 max 算正确 scale_inv。

- ❌ 别再试：**拿 element-wise tolerance（fp8=1e-1/fp4=0.5）当 gate**——它故意松，通过与否都不说明正确性，低精度只认 SNR。
- ❌ 别再试：**`torch.randn().to(FP8)` 直接造数据**——>240 会 saturate，且 mma 非线性，saturate 后误差不可预测。用 quantize helper（自动 clamp）造数据。

---
来源: verify-accuracy/SKILL.md, debug-flydsl-kernel/SKILL.md, fp8-gemm-bench/SKILL.md
