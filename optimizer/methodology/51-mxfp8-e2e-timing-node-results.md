# MXFP8 e2e 计时口径 / 节点占用检查 / vs GB200 结果

> 类别: 方法论 · 主题标签: mxfp8, e2e-timing, compile-once, GB200对标, 节点检查, gfx950

## 测量真实 GPU 时间(compile-once)
- FlyDSL `flyc.jit` launch 每次调用有 **~40us Python 派发开销**,cuda-event 会量成派发延迟。正确:先 `comp=flyc.compile(launch, ...)` 编译一次,再 `comp(...)` 直进 GPU stream 用 cuda-event 计时(warmup 30、iter 300)。

## 官方 e2e benchmark 口径(与 TE GB200 一致)
- `torch.utils.benchmark.Timer(stmt="fn()").timeit(100).mean*1e3`;`tflops=2*M*N*K/(ms*1e-3)/1e12`。
- **禁止手搓 cuda-event 计 bwd**(把 autograd dispatch 算进去,bwd 低估 ~10-18%)。

## 节点 / GPU 占用检查
- gpt_oss2 用 **GPU 4 或 5**(用 `mem_get_info` 现查占用)。检查:
  ```
  docker exec mlperf_gptoss2 bash -c "rocm-smi --showuse | grep 'GPU\[4\]\|GPU\[5\]'"
  ```
- 节点 chi2810 = gfx950 ×8,HBM3e ~8 TB/s 峰值,实测 1R:1W copy 上限 ~6.3 TB/s。

## LDS-合并转置写 vs GB200 结果
- fwd geomean ~0.99×(≈对齐)、bwd ~1.10×(反超)。
- e2e(Timer 口径,9 Llama shape,SNR 全 28dB)新(LDS-合并转置写)vs 旧 BM=32:**fwd 1710→1824 TFLOPS(+6.7%)、bwd 1839→1878(+2%)**。提升集中在 quant-heavy K=11008 fwd(4096×4096×11008 +21%、8192 +13%、16384 +10%)。
- fwd 差距根因 = B200 硬件 MX cast **近免费** vs MI355X **软件 dual-cast**;fwd 稳过 1.0× 仍需 in-gemm fusion(**用户否决**)。

---
来源: mxfp8-8wave-devloop/SKILL.md
