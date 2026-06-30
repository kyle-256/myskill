---
name: mi300-blockwise-gg-tuning
description: 在 MI300X (gfx942) 上调优 Triton blockwise FP8 grouped GEMM kernel 的硬约束、已知 priors 和常见坑。当用户在 Primus-Turbo / 类似仓库做 blockwise / scaled-FP8 / per-block-scale 的 grouped GEMM 调优时使用。包括 forward (persistent kernel) 和 backward (variable-K kernel) 两条路径,以及和 tensorwise 的公平对比方法。
---

# mi300-blockwise-gg-tuning

调 MI300X 上 blockwise FP8 grouped GEMM 的速查手册。所有数字都是在 `/wekafs/kyle/Primus-Turbo` 实测得到。

## 底层硬约束(不要试图绕过)

| 约束 | 来源 | 意味着 |
|---|---|---|
| `BK = 128` 锁死 | blockwise scale block size = 128;每 K 块一对 (a_s, b_s) | 不能像 tensorwise 一样选 BK=64,**结构性慢于 tensorwise** |
| LDS = 64KB | gfx942 hardware | `num_stages * (BM+BN)*BK ≤ 65536`;BK=128 + BM=BN=256 → 单 stage 刚好,**双 stage 不可能** → 失去 prefetch pipelining |
| MFMA tile 16/32 | gfx942 mfma_f8 inst | `matrix_instr_nonkdim ∈ {16, 32}`;**实测 32 永远不赢**,锁 16 |
| K%128 ≠ 0 走 Triton | CK kernel hard-asserts K%128==0 | 必须收紧 `GroupedGEMMFP8CKBackend.can_handle` 让 dispatcher 落到 Triton 的 masked-tail 路径 |
| OUT_M / OUT_N % 128 ≠ 0 也走 Triton | variable-K CK 同样 hard-assert | 同上,`GroupedGEMMFP8VariableKCKBackend.can_handle` 也要加 |

## 已经验证的 priors(初始 grid 应该用这些)

来自 8 GPU × 4 模型 × 多轮 sweep 的实测,**有数据支撑**:

### Forward (persistent kernel `_grouped_blockwise_fp8_persistent_gemm_kernel`)

- **`num_warps = 8`、`num_stages = 1`、`mfma = 16`**:几乎所有赢家。num_warps=4 偶尔在小 shape 略好,但差距 <2%;num_stages=2 LDS 不够。
- **两个 BM/BN 家族**:
  - 小 (B, M):`BM=128, BN=256`(2D 平衡 tile)
  - 大 (B, M)、N≥K 的非方形:`BM=256, BN=128`
- **`CHUNK_SIZE = 64` 是非默认但最常赢**(≥3/4 shape);默认 32 仅 1/4 shape 赢;`chunk=16` 几乎从不赢(可从 grid 删掉)
- **`kpack`**:小 shape 偏好 1,大 shape 偏好 2,差距 ~1-3%
- **`waves_per_eu`**:0(auto)和 2 都见过赢家;`waves_per_eu=2 + kpack=1 + 大 BM/BN` 是寄存器溢出地雷,跑出 100-200 TFLOPS
- **`GROUP_SIZE_M`**:小 shape 偏 1-4,大 shape 偏 4-16;影响在 ±5% 之内

### Backward (variable-K kernel `_grouped_blockwise_fp8_variable_k_gemm_kernel`)

- 寄存器压力比 fwd 大(每 K 迭代要 load scale 又要做 partial),所以默认 tile 小:`BM=BN=128, num_warps=4, num_stages=2`
- 实测最优:`BM=256, BN=128, num_warps=8, num_stages=1, waves_per_eu=2, kpack=2, mfma=16`(覆盖 4 模型 × 4 shape 中的多数)
- chunk=32 仍主导(BWD 的 chunk=64 没看到大幅领先)
- 同样的寄存器溢出地雷:`BM=256/BN=128 + waves_per_eu=2 + kpack=1`

## 公平对比 tensorwise(必须做)

**绝不能**用 dispatcher 路径(`turbo.ops.grouped_gemm_fp8`)的 tensorwise 数字当 blockwise 的目标线 — dispatcher + autograd + autotune wrapper 吃掉 ~30% 性能,会让 blockwise 看起来比实际更接近目标。

**正确做法**:两边都裸调 Triton kernel。
- blockwise:`tools/tune_blockwise_gg_fwd.py` / `tune_blockwise_gg_bwd.py`
- tensorwise reference:`tools/bench_tensorwise_raw.py` / `bench_tensorwise_raw_bwd.py`

两边都用同一套 `time_kernel`(CUDA event + sort + trim 20%)。

## 已知 blockwise 结构性差距(为什么打不过 tensorwise)

tensorwise 能 `BM=BN=256, BK=64, num_stages=2`(LDS = 256*64*2 + 64*256*2 = 65536 = 64KB exactly) → 享受 2-stage prefetch pipelining。

blockwise 因 BK=128(scale block 决定),最大组合是 `256x128x128 stages=1` 或 `128x256x128 stages=1`,**没有 prefetch**,FMA 后面就是空泡。

实测差距:大 shape blockwise 大约是 tensorwise 的 0.65-0.75×;小 shape 0.85-0.99×。要进一步缩差距,需要 kernel 级修改:
- **方案 A**:加 BK=64 支持,把 scale 在内层 K loop 里复用 2 次(同一 K=128 块的两个 BK=64 子块共享一对 scale)— 已记到 task 队列
- **方案 B**:hoist scale loads 到 K loop 外、向量化预取 — 收益较小
- **方案 C**:split-K — 对 grouped GEMM 不太适用

## Sub-agent dispatch 模板

每个 sub-agent prompt 必须包含:

1. `HIP_VISIBLE_DEVICES=<N>` 和 `TRITON_CACHE_DIR=/tmp/triton_cache_<N>`(防 Triton cache lock 撞车)
2. 上一轮该 shape 的 best TFLOPS(让 agent 报告 delta)
3. 该 shape 的 tensorwise raw target TFLOPS(让 agent 报告 ratio)
4. 明确"如果大部分 config 报错 → paste first error and stop early"(防止跑 25 分钟才发现 harness 坏了)
5. 结果文件路径要 unique(per agent + per round)

## 常见坑

| 现象 | 原因 | 修法 |
|---|---|---|
| GPT-OSS-20B 报 `k % 128 == 0` assertion | K=2880,CK BLOCKWISE kernel 不支持 | 收紧 CK can_handle,让 dispatcher 落到 Triton |
| 调出来的 blockwise "比 tensorwise 快" | tensorwise 测的是 dispatcher 路径(含 ~30% overhead),blockwise 测的是裸 kernel | 用 `bench_tensorwise_raw{,_bwd}.py` 做公平对比 |
| Refine grid 跑 25 分钟 | gen_refine_grid 没剪枝,生成 1440 configs | 基于 priors 剪枝:锁 mfma=16,删 chunk=16,删 num_stages=2 不可能的组合 |
| 某 config 跑出 100 TFLOPS | `waves_per_eu=2 + kpack=1 + 大 tile` 寄存器溢出 | grid 里把这个组合排掉 |
| Refine 出来比 R0 还慢 | refine grid 把 R0 winner 的 (BM, BN) 漏了 | 按"never regress"原则,refine 不超过 R0 就保留 R0 winner |
| 编译 cache lock 报错 | 多 agent 共享 `~/.triton/cache` | 每 agent 单独 `TRITON_CACHE_DIR` |

## 文件位置(本仓库的入口)

- Forward kernel:`primus_turbo/triton/grouped_gemm/grouped_gemm_fp8_kernel.py:1199` `_grouped_blockwise_fp8_persistent_gemm_kernel`
- Forward wrapper:同文件 `grouped_gemm_fp8_blockwise_triton_kernel`
- Backward variable-K kernel:同文件 `_grouped_blockwise_fp8_variable_k_gemm_kernel`
- Dispatcher:`primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py` `GroupedGEMMFP8KernelDispatcher`
- can_handle 修改点:同文件 `GroupedGEMMFP8CKBackend.can_handle` 和 `GroupedGEMMFP8VariableKCKBackend.can_handle`
- 调优 harness:`tools/tune_blockwise_gg_{fwd,bwd}.py`
- Tensorwise 参考:`tools/bench_tensorwise_raw{,_bwd}.py`
- 状态/结果:`tuning_results/{state.json, *.json, *.csv}`

## 如何把赢家落到 Python 配置选择器

调优结束后,把 per-shape 赢家写到 `_get_gg_blockwise_fwd_config(...)` 风格的 dict-based 选择器(类比同文件的 `_get_gg_fp8_tw_fwd_config`)。**锁 BK=128**,不要把 BK 暴露成可选。

## 配套 skill

调度策略(GPU 永不空闲)见 [`gpu-fleet-tuning/`](../gpu-fleet-tuning/SKILL.md)。本 skill 只讲 MI300 + blockwise 的 domain knowledge,调度模式那边讲。
