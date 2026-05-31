---
name: flydsl-fp8-gemm-tuning
description: 优化 FlyDSL fp8 dense GEMM (TN/NN/NT) 在 MI355X (gfx950) 上的性能。当用户要给 Primus-Turbo 的 `primus_turbo/flydsl/gemm/gemm_fp8_kernel.py` 提速、调某个 shape regime (big-K / big-N / square)、或问"为什么这个 case 慢/还能不能优化"时使用。覆盖：profile→定位瓶颈的诊断框架、已验证的 WIN lever (drain removal / 2D L2 swizzle)、已穷尽的死路清单 (省时间)、det=0 红线验证、和 gfx950 fp8 的硬约束。配合 remote-sync + claim-mi355x-node skill。
---

# FlyDSL fp8 dense GEMM 调优 (MI355X / gfx950)

主文件：`/wekafs/kyle/code2/remote_sync/Primus-Turbo/primus_turbo/flydsl/gemm/gemm_fp8_kernel.py`
公共 API：`gemm_fp8_tensorwise_flydsl_kernel(a, sa, b, sb, trans_a, trans_b, out_dtype)`
内部：`_compile_dense_tn/_nn/_nt(...)` + `_autotune_tn_dispatch(args,M,N,K)` (首调 bench 候选, 按 (M,N,K) 缓存)。
都是模块级 (在 `if flydsl_available():` 下)，probe 脚本可 `from ... import gemm_fp8_kernel as G; G._compile_dense_tn(...)`。

## 0. 红线 + 工作流 (违反就白干)

- **det=0 是硬红线**，不能松。验证必须 **≥500-1000 run** 的 bit-exact diff；**200-run 会假阳性**（vmcnt_hint=4 在 200 run 看 det=0 但 500 run 暴露 1.5e-5 race）。SNR 不掉(55/79/87 dB 看 scale)不代表 det=0。
- **严禁靠实验扫常数**(vmcnt_hint/lgkmcnt/barrier_mask 等)。必须先 `rocprof-compute` + ISA disasm 做理论分析，再决定调什么。扫 tile-blocking 因子(GROUP_M/group_n)时要**测 L2 命中率**作为机制依据，才算 grounded。
- **调试期不乱 commit**。只在确认真实进步(超噪声)+ user 认可后合成单个干净 commit。run-to-run 噪声 **~5%**，和很多 lever 的增益同量级 → 用 best-of-3/4 bench + 多次复测区分真假。
- 编辑只动 `code2/remote_sync/<repo>`，改完 `cd /wekafs/kyle/code2/remote_sync && ./sync.sh push Primus-Turbo <path>`（必须在该目录跑 sync.sh）。**改 kernel 后远程必须 `rm -rf /root/.flydsl/cache`** 再跑，否则用旧 .so。
- 只用 GPU 6 (`CUDA_VISIBLE_DEVICES=6 HIP_VISIBLE_DEVICES=6`)。

## 1. 诊断框架 (先定位瓶颈类，再选 lever)

不要凭直觉猜 store/带宽/spill。先 profile：
```bash
rocprof-compute profile -n <tag> --no-roof -- python <pmc_run.py>   # pmc_run = 跑 kernel 5-10 次
rocprof-compute analyze -p workloads/<tag>/MI* | grep -iE "MFMA Util|L2 Cache Hit|Dependency Wait|VMEM Util|Wavefront Occ|Insufficient SIMD VGPR|Insufficient CU LDS|Bank Conflict"
```
关键判据（实测 fp8 TN big-shape 的典型画像）：
- **VMEM Utilization ~3%** → **不是** memory/store/带宽 bound。别浪费时间在 store overlap / persistent kernel 上(被这个证伪过)。
- **Dependency Wait ~63%** + MFMA Util ~34-40% + Occupancy ~1 WG/CU → **latency-bound**：8 waves 喂不饱 MFMA+tr8 延迟链。
- **L2 Cache Hit**：square/big-K ~66%；big-N(大 N)~51% → L2 复用差是大 N 的主瓶颈。
- **Bank Conflict = 0**（已被 `chunk_stride=1056` padding 解掉，别再查这个）。
- **Occupancy 卡 1 WG/CU 的根因 = VGPR**：dense 256×256 tile 的 4 quadrant accumulator = **128 f32 VGPR/lane**，加 A/B frag 共 ~254 VGPR。2 WG/CU 需 V+A≤128/wave → **物理不可能**（128 accumulator 就是地板）。所以"提 occupancy"这条路对 256×256 tile 是死的。

ISA dump（看 drain）：
```python
os.environ["FLYDSL_DUMP_IR"]="1"; os.environ["FLYDSL_DUMP_DIR"]="/tmp/dscan"; os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"]="0"
# 必须真正 RUN kernel (c(*args)) 才触发编译+dump；只 _compile_dense_tn 不够
# 落在 /tmp/dscan/kernel_dense_tn_0/21_final_isa.s
```
```bash
grep -oE "s_waitcnt.*" 21_final_isa.s | sort | uniq -c | sort -rn   # 数 vmcnt(0)/lgkmcnt(0) 全排空
grep -c scratch_load 21_final_isa.s   # spill 检测
```

## 2. shape regime → 对应 lever (核心心法)

**瓶颈和 lever 随 shape 变。先认 regime：**

### big-K (K 很大, e.g. 8192×8192×28672, 长 K-loop, square-ish, L2 已好)
- 瓶颈：A-side tr8 的 `s_waitcnt vmcnt(0)` 全排空 (LLVM 对 intrinsic ds_read 保守插入)。
- **WIN = both-path-J drain removal**：`a_inline_asm=1, b_inline_asm=1`（A、B 都走 inline-asm `ds_read_b64_tr_b8`，opaque 给 SIInsertWaitcnts → 无自动 vmcnt(0) drain）+ `asm_mma=2`（绕过 agpr guard，MMA 仍 intrinsic）+ `agpr_alloc=0` + `vmcnt_hint=3`。
- **vmcnt_hint 调到 det=0 上限**：vh=3 是 sweet spot（2756 TF）；vh=2 也 det=0 但 2710-2734；**vh≥4 RACE**(det 1.5e-5)。big-N 的 det 上限也是 3。
- commit 424fe26b：2580→2756 (+6.8%, det1000=0)。

### big-N (N 很大, e.g. 8192×28672×8192, 短 K-loop, 矩形, L2 差)
- 瓶颈：**L2 复用**。1D GROUP_M 扫法对每个 M-group 重复 stream 整个 B(234MB)→ L2 51% vs big-K 66%。big-N/big-K 同 FLOPs/同 kernel，big-K 2756 而 big-N 卡 2411，差距纯是 L2。
- big-K 的 drain-removal **不迁移**到 big-N（短 K 摊不开；both-J 在 big-N 上是噪声）。
- **WIN = 2D super-block (band) swizzle**：把 N 切成宽 `group_n` 的竖带，带内再 GROUP_M 1D → A 复用 group_n×、B 复用 GROUP_M× → working set (GROUP_M·A_slab + group_n·B_slab, 各 ~2MB) 锁进 L2。
- 甜点 `GROUP_M=4, group_n = n_blocks/8`（band 数 = 8 = #XCD）。big-N GM4×GN14 → 2706-2729 (+12%, det1000=0, L2 51→57.5%, MFMA 34→40%)。
- **2D 通用，增益 ∝ L2 复用差多少**：big-K 也 +1% (winK GM4×GN8 = 2799, 摸 2800)；square/ffn_down 无回归。
- commit 7269aebc。

### 实现要点 (踩过的坑)
- `_compile_dense_tn(group_n=N)` 参数 + `_tn_block_mn(pid,...)` helper。**helper 必须是普通 Python 函数，不能用 kernel 内 `if`**：`@flyc.kernel` 把每个 `if` 分支包成独立 fn，分支内定义的 `block_m` 在分支外不可见（`if BR_B1: s_barrier()` 这种纯 side-effect 可以，定义变量给后面用就废）。helper 在 trace 时被普通 Python 调用，只构建被选中那条路径的表达式图。
- band swizzle 是纯 tile→CU permutation → **det 中性**（满 band 各占 num_pid_m·GN 个 pid，余数成最后一个窄 band，恒为 bijection）。
- autotune 接入：2D 候选 gated `n_blocks>=32 and M//256>=16`（小 M 的 m-block 太少, banding 不划算, 走 1D 防回归）；winK 块(K≥28672) 也 sweep `group_n ∈ {n_blocks/8, n_blocks/4}`。

## 3. 死路清单 (别重试, 全部 ≤ 生产或 racy)

- **single-buffer LDS** (`TN_SINGLE_BUF=1`)：LDS 减半想换 2 WG/CU，但破坏 load/compute 软流水 → **4× 更差** (531 vs 2078)。
- **CRR 4-barrier schedule** (`TN_CRR=1`)：barrier 反而帮调度，去 barrier 更差 (2074 vs 2404)。
- **both-intrinsic** (a=0,b=0)：双 vmcnt(0) drain，更差。
- **swap a-path-J/b-intrinsic** (a=1,b=0)：灾难 510 TF (4.8× 差)。
- **BLOCK_N=512**：LDS 202KB > 160KB 编译失败。
- **BLOCK_M=128**：`a_lds_size` 有 `2*8*1024` G2S-loader 下限钳制 → LDS 不降, 拿不到 2 WG/CU；且小 tile compute 输。
- **XCD-aware pid remap** (un-round-robin `(p%NX)*per+p//NX`)：det=0 但更差，把 GROUP_M reuse 限在单 XCD L2 切片不如全局 Infinity cache。
- **persistent kernel**：user 明确否决("别和我胡搞")。且诊断证明非 store-bound(VMEM 3%)，persistent 主要省 per-tile store/fill 收益有限。
- **降 VGPR 提 occupancy**：128 accumulator 是地板, 256×256 tile 到不了 2 WG/CU。
- **tr8 读优化**：`ds_read_b64_tr_b8` 64-bit 是硬限(无 `b128_tr_b8`)，且已 1读:1frag 最大批，软件无法再减。

## 4. 仍开放 / 下一步候选 (未做)

- **2900 目标**：2D 已榨干 (big-N 2729 / big-K 2799)，都卡 1 WG/CU + 40% MFMA 墙。再上需 **split-K**（尤其 big-K 只 1024 tiles，split-K=2 → grid 翻倍改善 CU 饱和；需加部分和 reduction，big-K 输出仅 134MB 可行）。big-N tiles 已足(3584)，split-K 帮助有限。
- L2 57.5→66 还有空间但 2D 各 GN/GM 组合已扫遍，GM4×GN14 是峰。

## 5. 教训 (元方法论)

- **别轻易宣布 "ceiling / 物理无解"**。我两次错误归因(big-K"HBM-bound"、big-N"store 暴露 + occupancy ceiling")都被 user 坚持"肯定能优化上去"推翻 —— 真因分别是 drain 和 L2，都有解。穷尽一类 lever ≠ 没有别类 lever。
- **同 FLOPs 不同 shape 的对照是金矿**：big-N 慢就拿 big-K(同 FLOPs 同 kernel)对照 profile，差异指标(L2 51 vs 66)直接点出瓶颈。
- 先 profile 认 regime → 选对 lever，比盲调快得多。
