# build/环境坑：陈旧源移植慢、HK 残留 undefined symbol、节点礼仪、LLVM OOM

> 类别: 踩过的坑 · 主题标签: build, 移植, 节点礼仪, pytest/OOM

## 陈旧源移植（stale-source）
- 本地 sync/FlyDSL 比远程容器 /workspace/code/FlyDSL 旧 → 照本地 vendor 会搬到**补丁前**版本，表现为移植版慢 ~15%。
- 实例：fp8 4wave AGPR 提速（ROCm/FlyDSL PR #714，`Mfma16x16x128AGPR` inline-asm `=a,v,v,0` 把 f32x4 累加器钉 AGPR、消 `v_accvgpr_mov`+`s_nop`，+5~13%）**只在远程有**。
- 教训：移植前先 `diff <(本地) <(远程 cat)`，或直接从远程容器取源。host 上没 FlyDSL 时：
  `./.ssh-chi.sh root@chi2811 "docker exec mlperf_gptoss cat <容器内路径>" > 本地文件` 落盘。

## vendor helper 为何不复用 gemm_helper.py（vendor-reuse）
- turbo 产品化的 8-wave 把原语分叉了，4-wave 直接 import 会崩：
  - `Mfma16x16x128` 去掉 `call_one`
  - `G2S`/`S2RLoader` 去掉 `load_one`
  - `StoreC` → `StoreCPerTensor`（per-tensor 标量 scale，非 row/col-wise）
- 因此 4-wave 单独 vendor 一份，保证 bench-identical 且不碰 8-wave。❌ 别再试直接复用 gemm_helper.py。

## HipKittens 残留 undefined symbol（build-hipkittens）
- `import primus_turbo.pytorch` 报 `undefined symbol ...hk_gemm_bf16`：untracked HK dense binding 引用不存在的 kernel .cu。
- HK 后端已于 **2026-06-01** 整体移除（dense+grouped 全 untracked WIP，无 python 引用）。
- 残留导致 build 失败的清理步骤：
  1. 删 `csrc/pytorch/gemm/hk_gemm_hip.cpp`、`csrc/pytorch/grouped_gemm/hk_grouped_gemm_hip*.cpp`、`csrc/kernels/grouped_gemm/HipKittens/`
  2. 从 `bindings_pytorch_hip.cpp` + `extensions_hip.h` 去掉所有 `hk_gemm*`/`hk_grouped*` 的 def/impl/声明
  3. nuke `build/` + `_C*.so` 重 build

## 节点礼仪（etiquette）
- 节点不是我们的，是某 slurm job 拥有者的。
- ❌ 别再试：清别人容器 / 停别人进程 / 删别人镜像（即使 GPU 空）。
- 不要在 idle 节点起容器立刻跑满 8 卡 → 先小任务跑通再上量。
- ❌ 别再试用 `srun`/`salloc` 抢 mi355x：全 ALLOCATED，slurm 不让进；正确路径是 docker。
- 被 primus-training 抢卡（污染环境）时测的**绝对 TFLOPS 不可信**，必须干净节点重测；但结构性的 V/spill/det 来自编译期、正确性仍可信。

## FlyDSL build（build）
- 报 `std::gcd not found` 或 redeclaration = 拿错了 LLVM。解法：`unset MLIR_PATH` 让 build.sh 自动探测。❌ 别再试手动硬设不匹配的 MLIR_PATH。
- MLIR .so 加载报错 = `LD_LIBRARY_PATH` 没含 `build-fly/python_packages/flydsl/_mlir/_mlir_libs/`。
- `No module named flydsl` = 没 `pip install -e .` 或没设 PYTHONPATH。
- LLVM 构建 OOM 时降并行度：`-j64` 而非 `-j128`（128 核并行编译 LLVM 内存压力大）。
- 结果陈旧/不对：清 kernel 缓存 `rm -rf ~/.flydsl/cache`；或 `export FLYDSL_RUNTIME_ENABLE_CACHE=0` 禁磁盘缓存（内存缓存仍生效）。改了内核但结果没变，先怀疑缓存。

## pytest 远端坑（pytest）
- `-u` 必须加，否则 stdout 块缓冲看不到进度；尤其 ❌ 别 `|tail -N`（tail 等 EOF 才输出，像卡死）。
- ssh 中断 ≠ 杀远程：本地 Ctrl-C 只断 ssh，远程 `docker exec` 里 pytest 仍在跑。多次中断重跑 → 僵尸 pytest 抢同块 GPU 越来越慢。重跑前先 `docker exec mlperf_gptoss pkill -9 -f "pytest <file>"`。
- 参数爆炸：`test_grouped_gemm_fp8.py -k FLYDSL` 选中 ~2.3 万 case（大多运行时 skip，但跑到的每个做 per-shape autotune 编译极慢，全量几百分钟）。只跑有界子集：`-x` 停首个失败，或定死 shape（`-k "FLYDSL and Format.E4M3-..."`）。deterministic 版对 flydsl 全 skip，没用。

## bench OOM（oom）
- bench 每个 shape 后必须 `del` + `torch.cuda.empty_cache()`，否则累积 fp8 tensor（grouped 的 b 是 3D 大）把 300GB HBM 撑爆 OOM。

---
来源: remote-sync/SKILL.md, claim-mi355x-node/SKILL.md, build-flydsl/SKILL.md, fp8-gemm-bench/SKILL.md
