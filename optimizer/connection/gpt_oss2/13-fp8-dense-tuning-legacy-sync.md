# gpt_oss2 fp8 dense GEMM 调优:主文件/probe/legacy sync.sh

> 类别: 连接 (gpt_oss2) · 主题标签: fp8-dense, flydsl, legacy-sync, probe

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

---
来源: flydsl-fp8-gemm-tuning/SKILL.md
