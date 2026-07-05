# 小 shape host/wrapper 开销淹没 kernel：走 raw op 绕过公开入口

> 类别: 踩过的坑 · 主题标签: measurement-noise, wrapper-overhead, quant-e2e, raw-op

- **小 shape 的 kernel-only 测量被 host 端固定开销淹没**：m=1024、单次 kernel 只 0.05~0.1us 时，每次走完整公开入口（`torch.empty`/`.view`/`.reshape`/dispatch-cache 查表）的固定开销比 kernel 本身还大，测出 TFLOPS 波动 **5~15%**，和 GPU 计算吞吐基本无关。
  - **验证"是 kernel 变慢还是 host 波动"的手法**：绕过公开入口，直接编译目标 kernel（如 `GK._compile_grouped_tn_wgrad_persistent`），`targs` 只建一次再反复 `_robust_time(launch, targs)`。若两版本一致 → 之前的差异就是 host/调度噪声。(src: remote-sync/SKILL.md)

- **测纯 kernel TFLOPS 必须走 raw op，不用 PT wrapper**：用 `torch.ops.primus_turbo_cpp_extension.hk_*`，绕开 wrapper。wrapper 把以下固定 overhead 算进 timing：
  - `quantize_fp8_tensorwise_impl` ~50µs + 30µs
  - `grouped_gemm_compute_offs` ~5µs
  - `dispatch` ~10µs
  - 合计 **~80µs 固定 overhead**，对一个 100µs 的 kernel 占 **45%**，会稀释/放大 loss% 对比。(src: fp8-gemm-bench/SKILL.md)

- **所有 4w vs 8w 的对比数都必须是 GEMM-only**：含 quant 的端到端 wgrad 比纯 GEMM 掉 **~15-30%**（悲观口径）。多出来的是一次纯访存 tensorwise 量化（amax 归约 + scale + cast），N/flop 越小占比越大。
  - 但这份 quant 是 **dgrad + wgrad 共享**的，bench 100% 算给 wgrad 是悲观上界，真实分摊约一半。(src: flydsl-fp8-gemm-results/SKILL.md)

- **含 quant 的 e2e 才是用户真实数字，纯 kernel 5500T 只是上限**：含 quant e2e 绝对 TF 大跌 —— fwd **~1700-2200 vs 纯 kernel ~5500**。原因是量化 + 小 M（mbs=1 时 M 仅 4096，算术强度低）双重拉低 FLOP 效率，两边都吃。
  - ❌ 别再试 用 raw-scale 直读核当 aiter baseline：kernel-only baseline 必须用 **shuffled scales**。`bpreshuffle=False` 只是不做权重 B 的离线 preshuffle，**scale 仍必须 shuffle**；raw-scale 直读核 SNR<0 是垃圾，不能当 baseline。(src: 14-fused-preshuffle-e2e.md)

---
来源: remote-sync/SKILL.md, fp8-gemm-bench/SKILL.md, flydsl-fp8-gemm-results/SKILL.md, 14-fused-preshuffle-e2e.md
