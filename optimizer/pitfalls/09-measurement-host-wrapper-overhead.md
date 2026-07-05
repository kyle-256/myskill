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

- **cuda-event 隔离测小 kernel 是假象**：单独测小 kernel（如 preshuffle）用 cuda-event 隔离，会把两个 record 之间 GPU 空等 **~22µs** 的 host/`flyc.jit` 派发算进 elapsed → 得 **37µs / 1.4 TB/s（假象）**，而 rocprof kernel-trace 实测只 **15.5µs / ~3.4 TB/s**。
  - ❌ 别再试 信 cuda-event 隔离值：测小 kernel 必用 `rocprof --kernel-trace`，或用 **both-minus-gemm-only 的差**（host 气泡才抵消）。(src: mxfp8-grouped-gg-devloop/SKILL.md)

- **memory-bound 短 kernel 外的 host per-call 元数据 / tiny-op launch 税常 > kernel 本身的差**：把 grouped quant 的 O(G) 组搜索搬 host（`arange`+`searchsorted`+4×`gather`+`where`×3 算 RB/RO/RE）后 kernel 降到 **153** 但 wrapper 反升到 **236** —— ~10 个 tiny torch 算子每次串行发射 **+80~120µs**（launch-bound，同 stream 无法重叠）。
  - grouped qa wrapper 算 padded lens/offs 的 ~7 个小 torch kernel（`ceil`×2、2×`cumsum`、`fill`、`copy` 各约 4.4µs 独立 launch）≈ **31µs**，dense 完全没有。
  - ❌ 别再试 把 prologue 摊成一串 tiny torch 算子搬 host：正解 = 融合 **on-device prologue kernel**（HIP 用 `compute_padded_layout_gpu <<<1,1>>>`，~9µs，1 线程从 tight int64 offs 直接算 64/128 对齐 lens/offs），塞进 jit stub 与 meta/kern 背靠背发射。(src: mxfp8-grouped-gg-devloop/SKILL.md)

- **per-call scale 转换 Python 循环是 host 瓶颈**：`gemm_mxfp8_flydsl_kernel` 的 per-call scale 转换（broadcast → WL lane-contig 经 `preshuffle_scale_lane_contig` 的 Python 循环）= **8000µs/call**，是端到端 host 开销瓶颈（kernel-only perf 已达标）。解法：向量化 / 缓存。(src: project_mxfp8_wholeloop_port.md)

---
来源: remote-sync/SKILL.md, fp8-gemm-bench/SKILL.md, flydsl-fp8-gemm-results/SKILL.md, 14-fused-preshuffle-e2e.md, mxfp8-grouped-gg-devloop/SKILL.md, project_mxfp8_wholeloop_port.md
