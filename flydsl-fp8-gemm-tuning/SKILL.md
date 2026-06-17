---
name: flydsl-fp8-gemm-tuning
description: 优化 FlyDSL fp8 dense GEMM (TN/NN/NT) 在 MI355X (gfx950) 上的性能。当用户要给 Primus-Turbo 的 `primus_turbo/flydsl/gemm/gemm_fp8_kernel.py` 提速、调某个 shape regime (big-K / big-N / square)、或问"为什么这个 case 慢/还能不能优化"时使用。覆盖：profile→定位瓶颈的诊断框架、已验证的 WIN lever、det=0 验证方法、和基础设施。本机已本地化 (代码在 `/workspace/code/gpt_oss_docker/sync/Primus-Turbo`, 本地直接编辑/运行, 无 rsync/ssh/docker)。
--- 

> ⚠️ **本机已本地化**：代码在 `/workspace/code/gpt_oss_docker/sync/Primus-Turbo`，本地直接编辑/运行，无 rsync / 无 ssh / 无 docker。改 flydsl kernel 后清缓存：`rm -rf /root/.flydsl/cache`。

# FlyDSL fp8 dense GEMM 调优 (MI355X / gfx950)

主文件：`/workspace/code/gpt_oss_docker/sync/Primus-Turbo/primus_turbo/flydsl/gemm/gemm_fp8_kernel.py`
公共 API：`gemm_fp8_tensorwise_flydsl_kernel(a, sa, b, sb, trans_a, trans_b, out_dtype)`
内部：`_compile_dense_tn/_nn/_nt(...)` + `_autotune_tn_dispatch(args,M,N,K)` (首调 bench 候选, 按 (M,N,K) 缓存)。
都是模块级 (在 `if flydsl_available():` 下)，probe 脚本可 `from ... import gemm_fp8_kernel as G; G._compile_dense_tn(...)`。

## 0. 工作流 + det 验证

- **det=0 验证方法**：bit-exact diff，**≥500-1000 run**（200-run 会假阳性）。SNR 不掉(55/79/87 dB 看 scale)不代表 det=0。
- **det 验证两个坑**：(1) `_compile_dense_tn` 是 `@functools.lru_cache(128)` — 多 env-variant 单进程对比**必须先 `G._compile_dense_tn.cache_clear()`** 再 compile，否则 env(函数内读)被冻在首次值。(2) race 是 **data-dependent + intermittent** — **每个 det pass 换 fresh `mk()` 随机 a,b**。强测 = cache_clear + ≥2×2000 run，每 pass fresh data。
- **扫寄存器/spill 前先 dump ISA**：`FLYDSL_DUMP_IR=1` 看 `vgpr_spill_count` + V/A，再决定要不要上 GPU（大 spill 配置会把容器搞崩 exit 137）。
- run-to-run 噪声 **~5%**，和很多 lever 的增益同量级 → 用 best-of-3/4 bench + 多次复测区分真假。
- 只用 GPU 6。

### 基础设施 (本地直接跑, 无 sync/ssh/docker)

本机已本地化, FlyDSL+turbo 已源码安装 (无需 PYTHONPATH)。跑脚本的本地等价 (清cache + 指定GPU + 无缓冲输出):
```bash
cd /workspace/code/gpt_oss_docker/sync/Primus-Turbo
rm -rf /root/.flydsl/cache    # 改了 flydsl .py 后必清; 复用已编译则跳过此行
HIP_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 python scripts2/_foo.py [args]
# inline 片段: HIP_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 python -c "<片段>"
# 换 GPU: 改 HIP_VISIBLE_DEVICES (如 =3)
```
输出过滤: `... 2>&1 | grep -vE "^Warning|aiter|total size|speedup"`。已加 `PYTHONUNBUFFERED=1`; 如 grep 缓冲可在管道前加 `stdbuf -oL`。

**`Primus-Turbo/scripts2/_h.py`** — 复用 harness, probe 脚本首行 `from _h import *` 即得:
`mk(*shape)` 造 fp8 输入 · `qargs(a,b,out,M,N)` 造 `_compile_dense_*` 的参数组 · `ref_tn/nt/nn(a,b)` 参考 · `bench(fn)`→best-of-N us · `tflops(M,N,K,us)` · `snr(o,r)` · `detf(fn,runs)`→(max_diff, first_bad) · `detc(c,args,out_view,runs)` · 导出 `G`(kernel 模块)/`FLY`(公共 API)。
- 改 cpp(csrc)后要 `GPU_ARCHS=gfx950 pip install --no-build-isolation -e .` 重装; 改 flydsl .py 只需手动清 cache (`rm -rf /root/.flydsl/cache`)。

## 1. 诊断框架 (先定位瓶颈类，再选 lever)

先 profile 再动手：
```bash
rocprof-compute profile -n <tag> --no-roof -- python <pmc_run.py>   # pmc_run = 跑 kernel 5-10 次
rocprof-compute analyze -p workloads/<tag>/MI* | grep -iE "MFMA Util|L2 Cache Hit|Dependency Wait|VMEM Util|Wavefront Occ|Insufficient SIMD VGPR|Insufficient CU LDS|Bank Conflict"
```
关键判据（实测 fp8 TN big-shape 的典型画像）：
- **VMEM Utilization**：低(~3%) = 非 memory/带宽 bound。
- **Dependency Wait + MFMA Util + Occupancy**：Dep-Wait 高 + MFMA Util ~34-40% + Occupancy ~1 WG/CU → latency-bound。
- **L2 Cache Hit**：square/big-K ~66%；big-N(大 N)~51% → L2 复用是大 N 的关键指标。
- **同 FLOPs 不同 shape 对照是金矿**：big-N 慢就拿 big-K(同 FLOPs 同 kernel)对照 profile，差异指标(如 L2 51 vs 66)直接点出瓶颈。

ISA dump（看 drain / spill / 寄存器）：
```python
os.environ["FLYDSL_DUMP_IR"]="1"; os.environ["FLYDSL_DUMP_DIR"]="/tmp/dscan"; os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"]="0"
# 必须真正 RUN kernel (c(*args)) 才触发编译+dump；只 _compile_dense_tn 不够
# 落在 /tmp/dscan/kernel_dense_tn_0/21_final_isa.s
```
```bash
grep -oE "s_waitcnt.*" 21_final_isa.s | sort | uniq -c | sort -rn   # 数 vmcnt(0)/lgkmcnt(0)
grep -c scratch_load 21_final_isa.s                                  # spill 检测
grep -E "num_vgpr|num_agpr|vgpr_spill" 21_final_isa.s                # 寄存器用量
```

## 2. shape regime → 对应 lever

**瓶颈和 lever 随 shape 变。先认 regime：**

### big-K (K 很大, e.g. 8192×8192×28672, 长 K-loop, square-ish, L2 已好)
- 瓶颈：A-side tr8 的 `s_waitcnt vmcnt(0)` 全排空 (LLVM 对 intrinsic ds_read 保守插入)。
- **WIN = both-path-J drain removal**：`a_inline_asm=1, b_inline_asm=1`（A、B 都走 inline-asm `ds_read_b64_tr_b8`，opaque 给 SIInsertWaitcnts → 无自动 vmcnt(0) drain）+ `asm_mma=2`（绕过 agpr guard）+ `agpr_alloc=0` + `vmcnt_hint=3`。
- **vmcnt_hint 调到 det=0 上限**：vh=3 是 sweet spot；vh=2 也 det=0 但更慢。big-N 的 det 上限也是 3。
- **WIN = asm-inplace MFMA**（`asm_mma=2` wire `_asm_mma_do` mode2 `=a,v,v,0`，D 别名 C in AGPR，`agpr_alloc=128`）：把 accum 挪进 AGPR，消 accvgpr 拷贝 + spill(18→0)。big-K +2.3%，det0 (3×2000 fresh)。commit 1049c9ee。

### big-N (N 很大, e.g. 8192×28672×8192, 短 K-loop, 矩形, L2 差)
- 瓶颈：**L2 复用**。1D GROUP_M 扫法对每个 M-group 重复 stream 整个 B → L2 51% vs big-K 66%。big-N/big-K 同 FLOPs/同 kernel，差距纯是 L2。
- big-K 的 drain-removal 不迁移到 big-N（短 K 摊不开）。
- **WIN = 2D super-block (band) swizzle**：把 N 切成宽 `group_n` 的竖带，带内再 GROUP_M 1D → A 复用 group_n×、B 复用 GROUP_M× → working set (GROUP_M·A_slab + group_n·B_slab, 各 ~2MB) 锁进 L2。
- 甜点 `GROUP_M=4, group_n = n_blocks/8`（band 数 = 8 = #XCD）。big-N GM4×GN14 → +12%, det1000=0, L2 51→57.5%, MFMA 34→40%。
- **2D 通用，增益 ∝ L2 复用差多少**：big-K 也 +1%；square/ffn_down 无回归。commit 7269aebc。

### 实现要点
- `_compile_dense_tn(group_n=N)` 参数 + `_tn_block_mn(pid,...)` helper。**helper 必须是普通 Python 函数，不能用 kernel 内 `if`**：`@flyc.kernel` 把每个 `if` 分支包成独立 fn，分支内定义的 `block_m` 在分支外不可见（`if BR_B1: s_barrier()` 纯 side-effect 可以，定义变量给后面用就废）。helper 在 trace 时被普通 Python 调用，只构建被选中那条路径的表达式图。
- band swizzle 是纯 tile→CU permutation → **det 中性**（满 band 各占 num_pid_m·GN 个 pid，余数成最后一个窄 band，恒为 bijection）。
- autotune 接入：2D 候选 gated `n_blocks>=32 and M//256>=16`；winK 块(K≥28672) 也 sweep `group_n ∈ {n_blocks/8, n_blocks/4}`。

## 3. 教训 (元方法论)

- 先 profile 认 regime → 选对 lever，比盲调快得多。
- 同 FLOPs 不同 shape 的对照(big-N vs big-K)是定位瓶颈最快的工具。
- 提速若伴随"某些 shape race 某些不"，是 root cause 未找到的信号 → 先 ISA 定位再说。
