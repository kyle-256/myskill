# 同步与远端协作：NFS 挂载、rsync 规程、FlyDSL 远端编译

> 类别: 连接 · 主题标签: NFS, 集群共享, 挂载, 节点弃用, rsync, 跳板机, 二进制校验, canonical镜像, 远端编译, JIT缓存, git-author

## 集群共享 NFS 挂载 /mnt/vast/kyle/code2

- **主用挂载**: bind-mount `/mnt/vast/kyle/code2` = `10.2.123.177:/aac-8634674/aac/shared/data`,152T 集群共享 NFS。每个节点同源,**写一次全集群可见**,换节点不丢数据/代码/tar。
- **已弃用 /mnt/shared**: 旧路径 `/mnt/shared`(= `login_node2:/mnt/nvmeraid`)在空闲节点上是坏/空 NFS,root 都 `Permission denied`。弃用。
- **节点级弃用坑**: 某些节点(如 **chi2761**)根本路由不到 aac NFS 服务器 → `10.2.123.177 No route to host`。这类节点上没代码、没 tar,直接弃用换节点。WHY: 路由不通,共享挂载失效,无法拉取工作目录。

## 本地→远端 rsync 同步规程

- **命令模板**（必须在 `sync/` 目录下跑，保证 `$PWD` 解析正确、trailing `/` 同步目录内容而非目录本身）：
  ```
  rsync -azh --exclude-from=.rsync-exclude -e "$PWD/.ssh-chi.sh" \
    "$PWD/tensorwise/Primus-Turbo/" root@chi2811:/mnt/vast/kyle/code2/Primus-Turbo-tensorwise/
  ```
- `-e "$PWD/.ssh-chi.sh"`：ssh 包装脚本，经跳板机连 chi2811（rsync + ssh 跳板）。**chi2810 已下线/不可用**（2026-07-03 起被别人的 SGLang 服务 8 卡全占，别去这台跑东西），当前活跃节点是 chi2811。
- `--exclude-from=.rsync-exclude`：排除 `.git`、`*.so`、`build`、`venv`。
- **canonical 本地镜像**（git）：`sync/mxfp4/Primus-Turbo`（分支 `dev/kyle/flydsl_mxfp4_compute`）、`sync/tensorwise/Primus-Turbo`；也涵盖 FlyDSL/turbo 子树。
- **WHY 必须先 rsync 再测**：改本地 `sync/mxfp4/Primus-Turbo`（或 FlyDSL/turbo）后不推远端，远端（chi2774/chi2811）会一直编译**旧二进制**，所有 ISA/perf/SNR 结论都对着旧代码 → 假象。血泪教训：改完先 rsync 再测。
- **对齐校验**：`md5sum` 本地 == 远程（远程用 `docker exec ... md5sum 容器路径`）必须相等；不等 = 没同步，测的是旧二进制。
- **正常差异**：`.so`/`build` 产物只在远程存在（被 exclude 不同步），属正常，不算未对齐。

## FlyDSL 远端为主 + JIT 缓存坑

- **远端为主(编译要 GPU)**: FlyDSL kernel 编译 `flyc.compile` 就要 GPU,不能像 Primus-Turbo 那样纯本地跑。canonical git 在远端 `/mnt/vast/kyle/code2/FlyDSL`(= 容器内 `/workspace/code/FlyDSL`),本地 `sync/FlyDSL` 只是镜像。流程:改完在远端编译/测/commit,再 rsync `.git`+文件回本地对齐。
  - 对比: Primus-Turbo(mxfp4/tensorwise) canonical 在本地;FlyDSL 相反,canonical 在远端。
- **rsync 加 `--no-o --no-g`**: 避开 JuiceFS 的 chown/chmod 报错(cosmetic,但会刷屏)。
- **commit author 必须显式覆盖**: `author=kyle-256 <Kyle.Zhao@amd.com>`,`GIT_AUTHOR_*` / `GIT_COMMITTER_*` 全设;禁止任何 Claude/Cursor coauthor。
- **本地 flydsl egg 可能是坏的**: `sync/` 里的 egg(如 dev409)可能坏/API 太老,本地根本跑不了;所有 kernel 测试必须在远端容器跑(远端有匹配的 flydsl 版本)。
- **live import 源码在远端**: kernel 的 live import 源码是远端 `/workspace/code/Primus-Turbo/`(经 chi2774 ssh+docker),本地 `sync/mxfp4/Primus-Turbo/` 只是镜像。可用 `cat patch.py | ssh ... docker exec -i python -` 管道 patch 远端 live 文件,本地镜像同步改保持一致。

### JIT 缓存坑(最关键)
- **现象**: 改完 FlyDSL kernel 普通跑仍跑旧版二进制。根因: FlyDSL JIT 缓存**不 hash 模块级 class 方法**,clear comgr 缓存也没用。
- **解法**: `FLYDSL_EXTRA_SOURCE_DIRS=<kernel目录>` 强制重编;或 `FLYDSL_DUMP_IR=1` 旁路。
- **判据**: 正确性 maxdiff 怎么改都卡在**同一垃圾值不动** = 多半中缓存,先 bust 再判正确性。

---
来源: claim-mi355x-node/SKILL.md, remote-sync/SKILL.md, flydsl-fp8-gemm-results/SKILL.md, 14-fused-preshuffle-e2e.md, project_mxfp4_epilogue_store.md, flydsl-sync/SKILL.md, pr-merge-gate/SKILL.md
