# 本地→远端 rsync 同步规程

> 类别: 连接 · 主题标签: rsync, 跳板机, 二进制校验, canonical镜像

- **命令模板**（必须在 `sync/` 目录下跑，保证 `$PWD` 解析正确、trailing `/` 同步目录内容而非目录本身）：
  ```
  rsync -azh --exclude-from=.rsync-exclude -e "$PWD/.ssh-chi.sh" \
    "$PWD/tensorwise/Primus-Turbo/" root@chi2810:/mnt/vast/kyle/code2/Primus-Turbo-tensorwise/
  ```
- `-e "$PWD/.ssh-chi.sh"`：ssh 包装脚本，经跳板机连 chi2810/chi2811（rsync + ssh 跳板）。
- `--exclude-from=.rsync-exclude`：排除 `.git`、`*.so`、`build`、`venv`。
- **canonical 本地镜像**（git）：`sync/mxfp4/Primus-Turbo`（分支 `dev/kyle/flydsl_mxfp4_compute`）、`sync/tensorwise/Primus-Turbo`；也涵盖 FlyDSL/turbo 子树。
- **WHY 必须先 rsync 再测**：改本地 `sync/mxfp4/Primus-Turbo`（或 FlyDSL/turbo）后不推远端，远端（chi2774/chi2810）会一直编译**旧二进制**，所有 ISA/perf/SNR 结论都对着旧代码 → 假象。血泪教训：改完先 rsync 再测。
- **对齐校验**：`md5sum` 本地 == 远程（远程用 `docker exec ... md5sum 容器路径`）必须相等；不等 = 没同步，测的是旧二进制。
- **正常差异**：`.so`/`build` 产物只在远程存在（被 exclude 不同步），属正常，不算未对齐。

---
来源: remote-sync/SKILL.md, flydsl-fp8-gemm-results/SKILL.md, 14-fused-preshuffle-e2e.md, project_mxfp4_epilogue_store.md
