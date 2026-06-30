---
name: fp8-gemm-bench
description: Primus-Turbo 的 fp8 GEMM / grouped GEMM kernel TFLOPS 测试方案。用户问 "测一下 fp8 gemm 的 perf / 比较 dense vs grouped / 看 kernel TFLOPS / RCR vs RRR vs CRR 性能" 时使用。覆盖正确的 bench setup（cuda event timing, warmup=20 iter=100, 全部 raw op + quantize_fp8_tensorwise_impl 算正确 scale_inv, 独占 GPU 验证, block-aligned shapes）和已知 pitfalls（triton.do_bench 不可靠 / wrapper 含 quantize overhead 不是纯 kernel TFLOPS / dense 在非对齐 shape silently early-exit / 别的进程占 GPU 出 OOM）。
---

> ⚠️ **本机已本地化**：Primus-Turbo 在 `/workspace/code/gpt_oss_docker/sync/Primus-Turbo`，bench **本地直接跑**，无 ssh / 无 docker。

# fp8-gemm-bench

测 Primus-Turbo 的 fp8 GEMM kernel 纯 kernel-level TFLOPS。MI355X gfx950, fp8 peak ~5 PFLOPS.

## TL;DR

```python
# 全部走 raw op (kernel-only timing), 用 quantize_fp8_tensorwise_impl 算正确 scale_inv
from primus_turbo.pytorch.kernels.quantization.quantization_impl import quantize_fp8_tensorwise_impl
FP8 = torch.float8_e4m3fnuz

a_fp8, a_inv = quantize_fp8_tensorwise_impl(a_bf16, FP8)  # 自动 max=240 正确 scale
b_fp8, b_inv = quantize_fp8_tensorwise_impl(b_bf16, FP8)

# Dense (RCR/RRR/CRR): raw op layout 字符串
out = torch.ops.primus_turbo_cpp_extension.hk_gemm_fp8(
    a_fp8, b_fp8, a_inv, b_inv, "rcr", group_m=4, out_dtype=torch.bfloat16)

# cuda event timing, warmup=20 iter=100
for _ in range(20): fn()
torch.cuda.synchronize()
s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
s.record()
for _ in range(100): fn()
e.record()
torch.cuda.synchronize()
t_us = s.elapsed_time(e) / 100 * 1000

# TFLOPS = 2*M*N*K / (t_us * 1e6)
```

完整 bench: 见 `bench_template.py`（本目录，覆盖 RCR/RRR/CRR 三个 layout 的 dense vs grouped 对比）。

## 必须遵循的约定

### 1. timing 用 `torch.cuda.Event(enable_timing=True)` + `record()/elapsed_time()`

**不要** 用 `triton.testing.do_bench`。原因：
- 在某些 shape 下 `do_bench` 测出来超 peak FLOPS (如 4363 TFLOPS = 87% MI355X peak)，但 kernel 实际跑 garbage。
- `do_bench` 内部 cudagraph capture / cache eviction 行为不可控。
- cuda event 直接读硬件 timestamp，可信。

### 2. 全部走 raw op (`torch.ops.primus_turbo_cpp_extension.hk_*`)，**不要用 PT wrapper** (`gemm_fp8` / `grouped_gemm_fp8`)

测纯 kernel TFLOPS 时 wrapper 把以下都算进 timing 里，掩盖真实 kernel 性能：
- `quantize_fp8_tensorwise_impl` on a, b (~50µs + 30µs)
- `grouped_gemm_compute_offs` (~5µs)
- custom_op dispatch overhead (~10µs)

Wrapper 模式时间 = ~80µs 固定 overhead + kernel time. 对 100µs 的 kernel, overhead 占 45%; wrapper loss% 会被稀释 / 放大 (取决于 dense vs grouped wrapper overhead 是否对称)。

**正确做法**: 用 `quantize_fp8_tensorwise_impl` 预先 quantize，然后 cuda event 只包 kernel call:

```python
from primus_turbo.pytorch.kernels.quantization.quantization_impl import quantize_fp8_tensorwise_impl
FP8 = torch.float8_e4m3fnuz

a_fp8, a_inv = quantize_fp8_tensorwise_impl(a_bf16, FP8)  # 一次性 quantize, 不计入 timing
# 然后 bench 只包 kernel:
fn = lambda: torch.ops.primus_turbo_cpp_extension.hk_gemm_fp8(
    a_fp8, b_fp8, a_inv, b_inv, "rcr", 4, torch.bfloat16)
```

为什么必须用 `quantize_fp8_tensorwise_impl` 而不能手写 scale: HK fp8 是 `float8_e4m3fnuz` (max=240)，不是标准 E4M3 (max=448)。手写 `scale_inv = 1/112` (按 448 算) 输出会偏 ~5x。让 helper 自动算正确 max → scale_inv 唯一可靠.

### 3. 确认 GPU 独占

测之前 `rocm-smi --showpids` 检查 KFD processes 列表，确认 target GPU 没有别的 vllm/sglang/python process。

```bash
rocm-smi --showpids 2>&1 | grep -A20 KFD
# 输出：PID + PROCESS NAME + GPU(s) + VRAM USED
```

如果 default GPU 0 被占，设 `HIP_VISIBLE_DEVICES`:
```bash
HIP_VISIBLE_DEVICES=2 python3 ...
```

`rocm-smi --showmemuse` 看 `VRAM%` per GPU，0% 才算空。**不要看 `gpu_use=0`** —— vllm with `--gpu_memory_utilization 0.95` 占 95% VRAM 但 `gpu_use=0` (idle waiting requests)。

### 4. shape 必须 block-aligned

- M % 256 == 0
- N % 256 == 0  
- K % 128 == 0

否则 **HK dense kernel 在非对齐 shape 上 silently early-exit**（不算 partial tile 的工作就提前返回），测出来超 peak FLOPS（4500-22000 TFLOPS）的不合理数字。

**安全的 N**: 2048, 4096, 8192. **安全的 K**: 128, 256, 512, 1024, 2048, 4096, 7168, 8192 (全 %128=0).

Grouped kernel 有 `m_limit` masked store 处理 partial tile，所以 grouped 在非对齐 shape correctness OK，但比对没意义因为 dense 那端的 baseline 是 bogus.

### 5. del + empty_cache 避免 OOM

bench 每个 shape 后：
```python
del a, b, a_fp8, b_fp8, ...
torch.cuda.empty_cache()
```

不然累积 fp8 tensor 把 300 GB HBM 撑爆 (grouped GEMM 的 b 是 3D 大).

### 6. 设 `set_auto_tune(False)`

bench 之前：
```python
from primus_turbo.pytorch.core.backend import GlobalBackendManager, BackendType
GlobalBackendManager.set_grouped_gemm_backend(BackendType.HIPKITTEN)  # 或 TRITON
GlobalBackendManager.set_auto_tune(False)
```

否则每个 shape cold-start 跑一遍 autotune 把 first-iter 时间污染掉。HK 内部 `_autotune_pick` cache 是 process-local dict，warmup 20 iter 足够 cache-hit 后才进入 timing loop。

## Raw op API 速查

| layout | dense | grouped |
|---|---|---|
| **RCR** (a @ b^T) | `hk_gemm_fp8(a, b, a_inv, b_inv, "rcr", gm, out_dt)` <br/>a [M,K], b [N,K] | `hk_grouped_rcr_fp8(a, b, a_inv, b_inv, g_offs, gm, m_per, xcds, out_dt, bn=0)` <br/>a [M,K], b [G,N,K] |
| **RRR** (a @ b direct) | `hk_gemm_fp8(a, b, a_inv, b_inv, "rrr", gm, out_dt)` <br/>a [M,K], b [K,N] | `hk_grouped_rrr_fp8(a, b, a_inv, b_inv, g_offs, gm, m_per, xcds, out_dt)` <br/>a [M,K], b [G,K,N] |
| **CRR** (a^T @ b) | `hk_gemm_fp8(a, b, a_inv, b_inv, "crr", gm, out_dt)` <br/>a [K,M], b [K,N] | `grouped_gemm_fp8_variable_k_impl(go, x, go_inv, x_inv, glens, g_offs, trans_a=True, trans_b=False, trans_c=False, ...)` <br/>(var_k is the grouped CRR kernel) |

CRR grouped 走 `grouped_gemm_fp8_variable_k_impl` (高级 backend impl), 因为 var_k 是 backward 内部 API. raw op `hk_grouped_var_k_crr_fp8` 也可直接调，但 impl helper 处理 default backend / trans_c 等更省心。

## 实测数据 (Primus-Turbo @ 2026-05-19, MI355X, GPU 2 独占)

Raw op + `quantize_fp8_tensorwise_impl`, cuda event w=20 i=100, 12 个 block-aligned shapes:

| layout | grouped/dense 时间比 | TFLOPS loss | 注 |
|---|---|---|---|
| **RCR (fwd)** | 1.216x | **17.8%** | grouped persistent loop overhead + spill 累加 |
| **RRR (dgrad direct)** | 1.366x | **26.8%** | grouped RRR persistent loop 最贵 |
| **CRR (var_k, wgrad)** | 1.049x | **4.7%** | var_k 已大改造 (per-group SRD + byte-level addressing) |

TFLOPS 范围:
- Dense: 1751-2843 TF (35-57% peak)
- Grouped: 1132-2495 TF (23-50% peak)

**Observations:**
- **CRR grouped 最接近 dense (4.7% loss)** — 部分 shape grouped 反而 faster (e.g., 16384x4096x2048 G=4: grouped 快 15.7%). var_k 设计成熟.
- **RRR grouped 损失最大 (26.8%)** — small K + large G 特别差 (8192x2048x2048 G=4 loss 43%).
- **RCR 居中 (17.8%)** — 主 forward 路径的 overhead 模式.

## 已知 anti-patterns

- ❌ `triton.testing.do_bench` —— RRR 上某些 shape 测出 87% peak (4363 TFLOPS) 但 correctness 烂掉。问题在 do_bench 不可控的 cache 行为。
- ❌ PT wrapper 测 kernel TFLOPS —— wrapper 含 ~80µs quantize/dispatch overhead, 稀释 loss% 数据.
- ❌ 直接 `hk_gemm_fp8(a_fp8, b_fp8, qs, qs, "rcr", ...)` 然后 `qs = torch.full([1], 1.0/112.0)` —— scale 按 E4M3 (max=448) 算，但 HK 是 E4M3FNUZ (max=240)，输出 norm 偏 ~5x. **必须用 `quantize_fp8_tensorwise_impl`**.
- ❌ 用 `torch.randn(...).to(FP8)` 直接造数据 —— 范围超 fp8 表达能力 (>240) 时 saturate, mma 结果非线性. quantize helper 自动 clamp.
- ❌ 假设 `gpu_use=0` 就空闲 —— 看 `VRAM%` 而非 `use%`. vllm 占 95% VRAM 但 use 显示 0% (idle).
- ❌ shape 不对齐就比较 dense vs grouped —— dense 测出来 4500+ TFLOPS 是 bogus (silently skipping partial tile).

## 复现

```bash
cd /workspace/code/gpt_oss_docker/sync/Primus-Turbo && \
  PYTHONPATH=/workspace/code/gpt_oss_docker/sync/Primus-Turbo:$PYTHONPATH \
  HIP_VISIBLE_DEVICES=<free_gpu> python3 /tmp/bench_template.py
```

模板见同目录 `bench_template.py`. 编辑 SHAPES 列表换 shape, 改 layout block 加更多 case.

## 配合的 skill

- [`claim-mi355x-node/`](../claim-mi355x-node/SKILL.md) — 找空 mi355x 节点 + 起容器
- [`remote-mlperf-gptoss/`](../remote-mlperf-gptoss/SKILL.md) — 在容器内操作
- [`remote-sync/`](../remote-sync/SKILL.md) — 本地 ↔ 容器代码同步
