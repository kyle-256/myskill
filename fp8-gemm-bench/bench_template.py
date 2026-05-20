"""fp8 GEMM kernel-only TFLOPS bench template for Primus-Turbo on MI355X.

约定：
- 全 raw op (kernel-only timing), 不走 wrapper
- 用 quantize_fp8_tensorwise_impl 算正确 scale_inv (E4M3FNUZ max=240)
- cuda event timing, warmup=20 iter=100
- shapes block-aligned: M%256=0, N%256=0, K%128=0
- HIP_VISIBLE_DEVICES=<free_gpu> on rocm-smi-verified idle GPU
- set_auto_tune(False)

跑法：
  docker exec -e HIP_VISIBLE_DEVICES=2 mlperf_gptoss bash -lc '
    cd /workspace/code/Primus-Turbo
    PYTHONPATH=/workspace/code/Primus-Turbo:$PYTHONPATH python3 bench_template.py
  '

修改 shapes / 切换 layout 在文件下半部分.
"""
import os, sys, statistics
sys.path.insert(0, "/workspace/code/Primus-Turbo")
import torch
torch.ops.load_library("/workspace/code/Primus-Turbo/primus_turbo/lib/libprimus_turbo_kernels.so")
_pyver = f"_C.cpython-{sys.version_info.major}{sys.version_info.minor}-x86_64-linux-gnu.so"
torch.ops.load_library(f"/workspace/code/Primus-Turbo/primus_turbo/pytorch/{_pyver}")

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import ScalingGranularity
from primus_turbo.pytorch.kernels.quantization.quantization_impl import quantize_fp8_tensorwise_impl
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import grouped_gemm_fp8_variable_k_impl

# ===========================================================================
# Setup
# ===========================================================================

torch.manual_seed(0)
DEV = "cuda"
FP8 = torch.float8_e4m3fnuz   # HK fp8 dtype (max=240, NOT standard E4M3 max=448)
GlobalBackendManager.set_auto_tune(False)
GlobalBackendManager.set_grouped_gemm_backend(BackendType.HIPKITTEN)  # or TRITON / HIPBLASLT

WARMUP, ITER = 20, 100

# Raw ops (kernel-only, no wrapper overhead)
dense_op = torch.ops.primus_turbo_cpp_extension.hk_gemm_fp8
rcr_op   = torch.ops.primus_turbo_cpp_extension.hk_grouped_rcr_fp8
rrr_op   = torch.ops.primus_turbo_cpp_extension.hk_grouped_rrr_fp8
# Grouped CRR (var_k): use the impl helper, it dispatches to hk_grouped_var_k_crr_fp8 raw op
# 但帮你处理 default backend / trans_c 等参数

def bench(fn):
    """cuda event timing, µs per call."""
    try:
        for _ in range(WARMUP): fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(ITER): fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / ITER * 1000  # µs
    except Exception as ex:
        print(f"  bench fail: {ex}")
        return None

def tf(M, N, K, t_us):
    """TFLOPS from µs time. flops = 2*M*N*K."""
    return 2*M*N*K / (t_us * 1e6) if t_us else None

def q(t):
    """Quantize bf16 → fp8 with correct E4M3FNUZ scale."""
    return quantize_fp8_tensorwise_impl(t, FP8)

# ===========================================================================
# Shapes — MUST be block-aligned (M%256=0, N%256=0, K%128=0)
# 非对齐 shape 会让 dense kernel silently early-exit, TFLOPS 数据假.
# ===========================================================================
SHAPES = [
    # (M_total, N, K, G_for_grouped)
    (8192,  2048, 2048, 4),
    (8192,  4096, 2048, 4),
    (16384, 2048, 2048, 4),
    (16384, 4096, 2048, 4),
    (16384, 4096, 4096, 4),
    (16384, 8192, 4096, 4),
    (32768, 2048, 2048, 16),
    (32768, 4096, 4096, 16),
    (65536, 4096, 4096, 16),
    (65536, 4096, 2048, 16),
    (16384, 4096, 7168, 4),
    (65536, 4096, 7168, 16),
]

def print_header(name):
    print(f"\n=== {name} (raw op, cuda event w={WARMUP} i={ITER}) ===")
    print(f"{'M':>6} {'N':>5} {'K':>5} {'G':>3}  {'d_us':>8} {'g_us':>8}  "
          f"{'d_TF':>5} {'g_TF':>5}  {'g/d':>6}  {'loss':>7}")

def print_row(M, N, K, G, t_d, t_g):
    if not (t_d and t_g):
        print(f"{M:>6} {N:>5} {K:>5} {G:>3}  FAIL")
        return None
    d_TF, g_TF = tf(M, N, K, t_d), tf(M, N, K, t_g)
    print(f"{M:>6} {N:>5} {K:>5} {G:>3}  {t_d:8.1f} {t_g:8.1f}  {d_TF:5.0f} {g_TF:5.0f}  "
          f"{t_g/t_d:6.3f}  {(1-g_TF/d_TF)*100:6.1f}%")
    return t_g/t_d

def print_summary(name, ratios):
    if not ratios: return
    g = statistics.geometric_mean(ratios)
    print(f"  {name} geomean grouped/dense: {g:.3f} → grouped TF loss {(1-1/g)*100:.1f}%")

# ===========================================================================
# RCR (forward): a @ b^T
# ===========================================================================
print_header("RCR")
rcr_ratios = []
for M, N, K, G in SHAPES:
    Mg = M // G
    a = torch.randn((M, K), dtype=torch.bfloat16, device=DEV)
    bd = torch.randn((N, K), dtype=torch.bfloat16, device=DEV)
    bg = torch.randn((G, N, K), dtype=torch.bfloat16, device=DEV)
    a_fp8, a_inv = q(a)
    bd_fp8, bd_inv = q(bd)
    bg_fp8, bg_inv = q(bg)
    glens = torch.full((G,), Mg, dtype=torch.int64, device=DEV)
    g_offs = torch.cat([torch.zeros(1, dtype=torch.int64, device=DEV),
                        torch.cumsum(glens, 0)]).to(torch.int64)

    t_d = bench(lambda: dense_op(a_fp8, bd_fp8, a_inv, bd_inv, "rcr", 4, torch.bfloat16))
    t_g = bench(lambda: rcr_op(a_fp8, bg_fp8, a_inv, bg_inv, g_offs, 4, Mg, 4, torch.bfloat16, 0))

    r = print_row(M, N, K, G, t_d, t_g)
    if r is not None: rcr_ratios.append(r)
    del a, bd, bg, a_fp8, bd_fp8, bg_fp8, glens, g_offs
    torch.cuda.empty_cache()
print_summary("RCR", rcr_ratios)

# ===========================================================================
# RRR (dgrad direct): a @ b
# ===========================================================================
print_header("RRR")
rrr_ratios = []
for M, N, K, G in SHAPES:
    Mg = M // G
    a = torch.randn((M, K), dtype=torch.bfloat16, device=DEV)
    bd = torch.randn((K, N), dtype=torch.bfloat16, device=DEV)
    bg = torch.randn((G, K, N), dtype=torch.bfloat16, device=DEV)
    a_fp8, a_inv = q(a)
    bd_fp8, bd_inv = q(bd)
    bg_fp8, bg_inv = q(bg)
    glens = torch.full((G,), Mg, dtype=torch.int64, device=DEV)
    g_offs = torch.cat([torch.zeros(1, dtype=torch.int64, device=DEV),
                        torch.cumsum(glens, 0)]).to(torch.int64)

    t_d = bench(lambda: dense_op(a_fp8, bd_fp8, a_inv, bd_inv, "rrr", 4, torch.bfloat16))
    t_g = bench(lambda: rrr_op(a_fp8, bg_fp8, a_inv, bg_inv, g_offs, 4, Mg, 4, torch.bfloat16))

    r = print_row(M, N, K, G, t_d, t_g)
    if r is not None: rrr_ratios.append(r)
    del a, bd, bg, a_fp8, bd_fp8, bg_fp8, glens, g_offs
    torch.cuda.empty_cache()
print_summary("RRR", rrr_ratios)

# ===========================================================================
# CRR (var_k / wgrad): a^T @ b
# Dense: a [K,M] @ b [K,N] → out [M,N]
# Grouped (var_k): grad_out [M,N] + x [M,K] → grad_b [G,N,K]
# ===========================================================================
print_header("CRR")
crr_ratios = []
for M, N, K, G in SHAPES:
    Mg = M // G

    # Dense CRR
    a_bf = torch.randn((K, M), dtype=torch.bfloat16, device=DEV)
    b_bf = torch.randn((K, N), dtype=torch.bfloat16, device=DEV)
    a_fp8, a_inv = q(a_bf)
    b_fp8, b_inv = q(b_bf)
    t_d = bench(lambda: dense_op(a_fp8, b_fp8, a_inv, b_inv, "crr", 4, torch.bfloat16))

    # Grouped CRR via var_k impl
    grad_out = torch.randn((M, N), dtype=torch.bfloat16, device=DEV)
    x_in = torch.randn((M, K), dtype=torch.bfloat16, device=DEV)
    go_fp8, go_inv = q(grad_out)
    x_fp8,  x_inv  = q(x_in)
    glens = torch.full((G,), Mg, dtype=torch.int64, device=DEV)
    g_offs = torch.cat([torch.zeros(1, dtype=torch.int64, device=DEV),
                        torch.cumsum(glens, 0)]).to(torch.int64)
    t_g = bench(lambda: grouped_gemm_fp8_variable_k_impl(
        go_fp8, x_fp8, go_inv, x_inv, glens, g_offs,
        trans_a=True, trans_b=False, trans_c=False,
        out_dtype=torch.bfloat16,
        granularity=ScalingGranularity.TENSORWISE.value,
        num_cu=None,
        default_backend=BackendType.HIPKITTEN.value))

    r = print_row(M, N, K, G, t_d, t_g)
    if r is not None: crr_ratios.append(r)
    del a_bf, b_bf, a_fp8, b_fp8, grad_out, x_in, go_fp8, x_fp8, glens, g_offs
    torch.cuda.empty_cache()
print_summary("CRR", crr_ratios)

# ===========================================================================
# Final summary
# ===========================================================================
print(f"\n--- SUMMARY (raw op kernel-only TFLOPS) ---")
for name, r in [("RCR", rcr_ratios), ("RRR", rrr_ratios), ("CRR", crr_ratios)]:
    if r:
        g = statistics.geometric_mean(r)
        print(f"  {name}: geomean grouped/dense = {g:.3f}, grouped TF loss = {(1-1/g)*100:.1f}%")

# ===========================================================================
# 切换 backend 对比 (vendor baseline / Triton):
# ===========================================================================
# 把 dense_op / rcr_op / rrr_op 等替换:
#   hipBLASLt: torch.ops.primus_turbo_cpp_extension.hipblaslt_gemm_fp8(...)
#              torch.ops.primus_turbo_cpp_extension.hipblaslt_grouped_gemm_fp8(...)
#   Triton:    via wrapper (GlobalBackendManager.set_grouped_gemm_backend(TRITON))
#              因为 Triton kernel 没暴露 raw op
