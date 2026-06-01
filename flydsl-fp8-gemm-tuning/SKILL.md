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
- 只用 GPU 6。

### 基础设施 (用这套, 别再手搓 sync/cache/ssh — 反复踩坑的根源)

**`/wekafs/kyle/code2/remote_sync/rr.sh`** — 一条命令搞定 sync+清cache+GPU6+无缓冲输出 (自动 `cd` 到 remote_sync 根, 修了 `./sync.sh` cwd bug; FlyDSL+turbo 已源码安装, **无需 PYTHONPATH**):
```bash
cd /wekafs/kyle/code2/remote_sync
./rr.sh run scripts2/_foo.py [args]   # sync 两 repo + 清 /root/.flydsl/cache + GPU6 跑
./rr.sh py "<python -c 片段>"          # 快速 inline
./rr.sh sh "<容器内 bash 命令>"         # 任意命令 (带 GPU6 env)
./rr.sh sync                           # 只同步
GPU=3 ./rr.sh run ...                  # 换 GPU; NOCLEAN=1 ... 跳过清 cache (复用已编译)
```
输出过滤: `... 2>&1 | grep -vE "^Warning|aiter|total size|speedup"`。注意 grep 经 ssh pipe 会缓冲, rr.sh 已加 `stdbuf -oL` + `PYTHONUNBUFFERED`。

**`Primus-Turbo/scripts2/_h.py`** — 复用 harness, probe 脚本首行 `from _h import *` 即得:
`mk(*shape)` 造 fp8 输入 · `qargs(a,b,out,M,N)` 造 `_compile_dense_*` 的 arg 元组 · `ref_tn/nt/nn(a,b)` 参考 · `bench(fn)`→best-of-3 us · `tflops(M,N,K,us)` · `snr(o,r)` · `detf(fn,runs)`→(max_diff, first_bad) · `detc(c,args,out_view,runs)` · 导出 `G`(kernel 模块)/`FLY`(公共 API)/`BF16/FP8`。
det0 == race-free; **det 必须 ≥1000 run**(边界值如 vmcnt 阈值要 2000)。
- 改 cpp(csrc)后要 `GPU_ARCHS=gfx950 pip install --no-build-isolation -e .` 重装 turbo (见 claim §5.5); 改 flydsl .py 只需 rr.sh 自动清的 cache。

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

## 4. 2900 目标方案 (workflow 2026-06-01 产出, FlyDSL+HK 代码考古)

现状 (chi2774, source-installed, HK 移除后 clean rebuild): big-N ~2730-2750, big-K ~2818。
**user 否决 split-K + persistent。** 真瓶颈 = issue/latency-bound (Dep-Wait 63%, MFMA Util 40%,
VMEM 3% 非带宽; occupancy 1 WG/CU 被 128-accumulator VGPR 地板锁死, 不可破)。

排名 (workflow 综合, 见本 session transcript):
- **#1 (真 lever) 32x32x64 MFMA 替换**: 同 64×128/warp 输出, 用 16× `mfma_scale_f32_32x32x64_f8f6f4`
  替 32× `mfma_16x16x128`/iter → **MFMA 指令数减半** (issue-bound 直接收益, accumulator 不变=不动
  occupancy, 与"32x32 flawed premise"不矛盾)。CDNA4 确认支持 (MmaAtom.cpp:157,251; accVec 32x32→16)。
  估 +2.8% → big-N ~2806 / big-K ~2877。**大改**: 新 `Mfma32x32x64` wrapper (mirror Mfma16x16x128,
  accum=Vec.make_type(16,Float32)) + 32x32 operand frag loader (tr8 读布局变, 是难点+det 风险) +
  2-inner-K loop (BLOCK_K=128 = 2×K64) + StoreC 重写 (32x32 输出 frag→global 映射变)。
  **先做微 probe 复现 +2.8%+det=0 再做 StoreC 大改。** 现有树无 32x32 资产 (RRR 旧的被 supersede 删了)。
- **#2 sched_group_barrier pacing**: 已实现 (`add_sched_hints` 参数 / `TN_SCHED_HINTS=1` env, 在 7-barrier
  loop 每 mfma.call 后插 `sched_mfma(N_ACCUMS)/sched_dsrd(2)/sched_vmem(2)`)。**实测仅边际**:
  big-N +0.7% (2730→2750 det0), big-K -0.3% (噪声)。det 安全 (纯 compiler fence)。留 dormant 待和 #1 叠。
- **#3 B 侧 graded-lgkmcnt 流水**: 把 `S2RLoaderTr_A.load_pipelined` (utils:1469) 的 depth-2 graded
  扩到 B loader `S2RLoaderTr` (现在是 4 read + 1 个 trailing lgkmcnt(0) 全drain)。+1-2% 估, 中 det 风险
  (graded lgkmcnt 是 race-prone 旋钮)。
- **死路确认**: BTR/LDS2LDS-transpose A (软件转置比 HW tr8 慢 3-30×, 且 A-final +33.8KB → 168.9KB 超 160KB);
  wider tr8 (fp8 无 b128_tr, CopyAtom.cpp:148 硬上限); tr16_64b/tr6_96b (fp6-only, 不适 fp8)。
- **诚实天花板**: #1 单独 big-N 到 ~2806 (不够 2900); #1+#2(+#3) 叠加才可能 big-K 破 2900 / big-N 逼近。

## 5. 教训 (元方法论)

- **别轻易宣布 "ceiling / 物理无解"**。我两次错误归因(big-K"HBM-bound"、big-N"store 暴露 + occupancy ceiling")都被 user 坚持"肯定能优化上去"推翻 —— 真因分别是 drain 和 L2，都有解。穷尽一类 lever ≠ 没有别类 lever。
- **同 FLOPs 不同 shape 的对照是金矿**：big-N 慢就拿 big-K(同 FLOPs 同 kernel)对照 profile，差异指标(L2 51 vs 66)直接点出瓶颈。
- 先 profile 认 regime → 选对 lever，比盲调快得多。
