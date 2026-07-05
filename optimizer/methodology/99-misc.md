# 方法论：其它零散事实

> 类别: 方法论 · 主题标签: measurement-noise, preshuffle-gemm, wider-k-mfma, copy-atom-port

## 性能/带宽计算公式
- flops = 2*M*N*K；tflops = flops/(us/1e6)/1e12。
- gfx942 MI300X 单 GCD 峰值参考：FP8 ~653 TFLOPS、BF16 ~326 TFLOPS、INT8 ~653 TOPS。
- 带宽字节数（eb = element bytes）：
  - FP8/INT8 = M*K*eb + N*K*eb + M*N*2 + (M+N)*4
  - INT4 = M*K + N*K/2 + M*N*2 + (M+N)*4
  - MXFP4 = M*K/2 + N*K/2 + M*N*2 + (M+N)*(K//32)
  - tbps = bytes/1e12/(us/1e6)

## 更宽 K 的 MFMA atom（少发指令）
- 选能整除 K-loop 的最宽 K atom，让循环发最少 MFMA。
- gfx942 最宽：F16/BF16 = 32x32x8 (K=8) / 16x16x16 (K=16)，INT8 K=32。
- gfx950 加宽到：32x32x16 / 16x16x32，INT8 K=64 —— 固定 K-loop 的 MFMA 数约减半。
- 例：K=64 的 F16 loop = 4 issues @ 32x32x16 vs 8 issues @ 32x32x8。
- C/D 全程钉在 AccVGPR，只在 prologue/epilogue 搬到 arch-VGPR。WHY：省循环内 shuffle。

## preshuffle GEMM 参考模式 (kernels/preshuffle_gemm.py)
- B 矩阵预置为 (N/16, K/64, 4, 16, kpack_bytes)。
- A tile global→LDS 用 XOR16 swizzle 避 bank conflict。
- K64 字节微步：每步发 2 个 K32 MFMA。
- ping-pong LDS (lds_stage=2) 重叠 load 与 compute。
- epilogue：直接行主序 store，或经 LDS 的 CShuffle 打包。

## copy-atom 移植坑（raw buffer_ops → FlyDSL layout API）
- 移植路径：make_buffer_tensor + logical_divide + copy_atom_call（用 BufferCopy*b atom），替换手写字节运算 / shrui(...,2) / i32→dtype bitcast。
- 坑1：copy_atom_call 没有 mask= 参数 → 必须重新加显式 is_valid.select(...) 或 `if is_valid` 的 OOB guard。
- 坑2：wave-uniform 的 row offset 当前可能折进 voffset (VGPR) 而非 soffset (SGPR) → 移植后重新检查 VGPR 压力。

## trace/code.json 源映射校验
- python 读 data['code']：n=总指令数，has_src=sum(1 for i if i[3])，打印带源映射百分比。
- 典型 PA decode 输出：arch_vgpr=96 accum_vgpr=128 SGPR=80、2692 指令、78% 源映射。

---
来源: gemm-optimization/SKILL.md, flydsl-kernel-authoring/SKILL.md, capture-kernel-trace/SKILL.md, optimization-directions.md, gemm/optimization-directions.md
