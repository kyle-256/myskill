# gpt_oss2 开发流程、工作树布局与 bench/工具脚本

> 类别: 连接 (gpt_oss2) · 主题标签: devloop, bench, mxfp8, rocm-smi, worktree-layout, mxfp8-grouped-gemm, WL-未提交, 关键文件, fp8-dense, flydsl, legacy-sync, probe, PR镜像, chi2811, gfx950

## gpt_oss2 快速开发 checklist + bench/工具脚本

### 快速开发 checklist（改一次 kernel 走一遍）
1. 改 kernel `.py`（**本地**改，别直接在 docker 里改）
2. `rsync -av --checksum` 同步到远端 —— 用 `--checksum` 因为改 py 内容变但 mtime 可能骗过默认比对
3. `docker exec mlperf_gptoss2 rm -rf /root/.flydsl/cache` —— **必清 FlyDSL 缓存**，否则 loader 跑旧编译版
4. `docker exec -e HIP_VISIBLE_DEVICES=4 <测试>` —— 先跑数值
5. 数值通过后 `gms()` 测速，对比 C++ 基线
6. `rocm-smi --showuse` 确认 GPU 干净后再 bench（脏 GPU 测速无意义）
7. `bench` 正式测
8. `git commit` —— **pre-commit 自动格式化可能需两次 `add`+`commit`**（第一次被格式化改文件、第二次才过）

### 干净 GPU 判定（bench / race 前必查）
- `rocm-smi --showuse` → **GPU% = 0**
- `rocm-smi --showmemuse` → **VRAM% < 50**
- 理由：noisy GPU 上 race rate 会被 OOM 干扰，测速也失真

### 并行测试
- `PT_GPUS=4,5,6,7 pytest -n 4` —— conftest 已加 `PT_GPUS` 支持，指定 GPU 集并行

### bench / 工具脚本
| 脚本 | 位置 | 用途 |
|---|---|---|
| `bench_mxfp8_pk.py` | `sync/` | per-tensor vs pk1/2/4 **kernel-only** bench；已重写为生产 API：quant layout 2/4 + scale_pack |
| `correctness_mxfp8_pk.py` | `sync/` | mxfp8 pk **SNR 验证** |
| `probe_gn.py` | `sync/` | gn/gm/xcd 探针 |
| `_wl_perf.py` | `sync/` | grouped wgrad **WL bench** |
| `bench_mxfp8_ratio.py` | `sync/`（chi2774 GPU4） | dense mxfp8 vs 非-scaled per-tensor 的 kernel-only bench |
| `spot_bench_kernel.py` | `scripts/` | kernel-only **TFLOPS**（不含 quant pre-process）|

## gpt_oss2 mxfp8 内核工作树位置(住在 mxfp4 树里)与关键文件

### 反直觉:mxfp8 内核住在 mxfp4 树里
- gpt_oss2 环境的 mxfp8 grouped GEMM 内核实际住在 **mxfp4 树**里,工作树路径 **`sync/mxfp4/Primus-Turbo`**(不是 mxfp8 repo)。mxfp8 内核代码放在 mxfp4 checkout 中——找错 repo 会白费时间。

### 关键文件(均在 sync/mxfp4/Primus-Turbo 下)
- `primus_turbo/flydsl/grouped_gemm/mxfp8_grouped_kernel.py`
- `primus_turbo/flydsl/gemm/mxfp8_quant_flydsl.py`
- `mxfp8_gemm_kernel.py`
- `flydsl.utils.gemm_helper`(即 `gemm_helper.py`)
- `quantization.py`
- `grouped_gemm_fp8.py`

### WL(whole-loop)实验代码留档状态
| 场景 | 是否 commit | 留档位置 | WHY / 陷阱 |
|---|---|---|---|
| dense mxfp8 WL | 有 | 仓库外 `/workspace/code/gpt_oss2_docker/.wl_study/` + git 历史 `fc45f4bb` | 可从留档恢复 |
| grouped wgrad WL | **从未 commit**,已 revert 回 HEAD(git 历史里没有) | 唯一残余 = 删除前的 `/tmp/sp_backup` | `/tmp` ephemeral,容器重启即失,**勿依赖**;重跑得照结论重写 |

- grouped wgrad WL 全套实验代码(`_build_grouped_mxfp8_wgrad_wl_kernel` / `_compile_grouped_mxfp8_wgrad_wl` / `PT_MXGG_WGRAD_WL*` 探针 env / `S2RLoader.base_addr` / `PT_MXGG_*_CFG` autotune 覆盖钩子)全部 revert,不在 git 里。

## gpt_oss2 fp8 dense GEMM 调优:主文件/probe/legacy sync.sh

- **此为 code2/remote_sync legacy 流**,与主 code3 sync 流不同,别搞混两套同步设施。
- **主文件路径**: `/wekafs/kyle/code2/remote_sync/Primus-Turbo/primus_turbo/flydsl/gemm/gemm_fp8_kernel.py`。所有 fp8 dense GEMM 调优改动都落在这里。
- **公共 API**: `gemm_fp8_tensorwise_flydsl_kernel(a, sa, b, sb, trans_a, trans_b, out_dtype)`。
- **内部函数**:
  - `_compile_dense_tn(...)` / `_compile_dense_nn(...)` / `_compile_dense_nt(...)` —— 三种转置组合的 dense 编译入口。
  - `_autotune_tn_dispatch(args, M, N, K)` —— 首次调用 bench 各候选,按 `(M,N,K)` 缓存选中候选。WHY: 缓存后同 shape 不再重 bench,改 kernel 后需清缓存(见下)才会重选。
- **probe 直调方式**: `_compile_dense_tn/_nn/_nt` 都是**模块级**函数(在 `if flydsl_available():` 块下),probe 脚本可 `from ... import gemm_fp8_kernel as G; G._compile_dense_tn(...)` 直接调,无需走公共 API。WHY: 便于单独 bench/调某一转置分支。
- **sync push(必须在指定目录跑)**: 编辑只动 `code2/remote_sync/<repo>`;改完必须**在 `/wekafs/kyle/code2/remote_sync` 目录下**跑 `./sync.sh push Primus-Turbo <path>`。WHY: `sync.sh` 依赖该目录为工作根,不在此目录跑会失败。
- **GPU 固定为 6**: `CUDA_VISIBLE_DEVICES=6 HIP_VISIBLE_DEVICES=6`。flydsl fp8 GEMM 调优只用 GPU 6。
- **改 kernel 必须清 flydsl 缓存**: 远程改完 kernel 后必须 `rm -rf /root/.flydsl/cache` 再跑。WHY: 否则命中旧 `.so`,改动不生效。

## gpt_oss2 mxfp8 PR 本地镜像(1:1 镜像远端)

- **mxfp8 PR 本地镜像路径(实际编辑目录)**: `/workspace/code/gpt_oss2_docker/sync/mxfp8/Primus-Turbo/` —— 与远端 PR 分支逐一对应,可直接在本地改/看。
- `/workspace/code/mxfp8/Primus-Turbo/` 是**远程容器内路径**(`mlperf_gptoss2` 容器,chi2811),对应远程 host 路径 `/mnt/vast/kyle/code3/mxfp8/Primus-Turbo`,不是本地路径。
- **验证容器**: race + perf 验证统一用 `mlperf_gptoss2` 容器(chi2811 / gfx950)。WHY: race(竞态)与 perf(性能)两类验证需在真实 gfx950 硬件 + 该容器环境下跑,本地镜像只做代码同步不做验证。

---
来源: mxfp8-8wave-devloop/SKILL.md, project_mxfp8_wholeloop_port.md, project_mxfp8_grouped_wgrad_wl.md, gfx950-vmcnt-race-debug/SKILL.md, mxfp8-grouped-gg-devloop/SKILL.md, flydsl-fp8-gemm-tuning/SKILL.md, remote-sync/SKILL.md(本地/远程路径分层表)
