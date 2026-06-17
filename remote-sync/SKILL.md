---
name: remote-sync
description: 在本机 /workspace/code/gpt_oss_docker/sync/Primus-Turbo 本地直接编辑并运行代码，无 rsync / ssh / docker / 远程节点。当用户要求"修改/编辑/重构 Primus-Turbo 的代码"或要"运行 X"时使用此 skill。需要从 chi2811 刷新代码时见顶部 banner。
---

> ⚠️ **本机已本地化**：Primus-Turbo 代码在 `/workspace/code/gpt_oss_docker/sync/Primus-Turbo`，**本地直接编辑/运行**，无 rsync、无 ssh、无 docker、无远程节点。
> 需要从 chi2811 重新拉取最新代码时，用 `/workspace/code/gpt_oss_docker/sync/.ssh-chi.sh`（rsync transport 包装）+ `.rsync-exclude`：
> `cd /workspace/code/gpt_oss_docker/sync && rsync -azh --exclude-from=.rsync-exclude -e "$PWD/.ssh-chi.sh" root@chi2811:/mnt/vast/kyle/code2/Primus-Turbo/ $PWD/Primus-Turbo/`
> 下方原远程 rsync 流程仅作历史参考（其他机器用）。

# remote-sync

> 注：以下原文档描述的是"本地编辑 → rsync 推远程 → ssh+docker exec 执行"的远程流程。**本机已本地化**（见顶部 banner），所有 Primus-Turbo 操作都在 `/workspace/code/gpt_oss_docker/sync/Primus-Turbo` 本地直接做，无需 push/pull、无需 ssh/docker。下方远程内容仅作历史参考。

## 布局（本机）

```
/workspace/code/gpt_oss_docker/sync/
├── .rsync-exclude              # 从 chi2811 刷新时的排除规则
├── .ssh-chi.sh                 # rsync transport 包装（连 chi2811）
└── Primus-Turbo/              # 本地直接编辑 + 运行（git 分支 dev/kyle/flydsl_grp_gemm）
    └── .git/
```

- 本机是一个本地 docker 容器，**只同步了 Primus-Turbo 这一个仓库**。HipKittens / FlyDSL / 其他 repo 都**没有**同步到本机。
- Primus-Turbo 保留完整 `.git/`，`git status / log / diff / commit / branch` 直接在本地跑。
- 代码就地编辑、就地运行，**没有**本地↔远程两份的概念，无需任何同步即可执行。

## 本地直接编辑，无需 push/pull

本机只有一份 Primus-Turbo，编辑后立即生效，**无需 rsync push、无需 sync.sh、无需双向同步、无需 commit 漂移校验**。原来"两边 commit 必须一致（本地为准）"那套约束在本机不适用 —— 本机就是唯一一份。

唯一需要"同步"的场景：**从 chi2811 拉取最新代码**。用顶部 banner 里的命令：

```bash
cd /workspace/code/gpt_oss_docker/sync && \
  rsync -azh --exclude-from=.rsync-exclude -e "$PWD/.ssh-chi.sh" \
  root@chi2811:/mnt/vast/kyle/code2/Primus-Turbo/ $PWD/Primus-Turbo/
```

注意：从远程拉取时 `.rsync-exclude` 排除了 `.git/`，本地 `.git/` 不会被远程覆盖。拉取的是工作树文件，拉完用本地 git 看 diff 即可。

## 命令速查（本机）

本机直接本地编辑 + 本地运行，没有 push/pull/diff/mirror 这些同步命令。

```bash
# 编辑
$EDITOR /workspace/code/gpt_oss_docker/sync/Primus-Turbo/scripts/foo.py

# 运行（直接本地跑，无 ssh / docker exec）
cd /workspace/code/gpt_oss_docker/sync/Primus-Turbo && python scripts/foo.py

# git 操作（本地）
cd /workspace/code/gpt_oss_docker/sync/Primus-Turbo && git status
```

### 从 chi2811 刷新代码（唯一的 rsync 用法）

```bash
cd /workspace/code/gpt_oss_docker/sync && \
  rsync -azh --exclude-from=.rsync-exclude -e "$PWD/.ssh-chi.sh" \
  root@chi2811:/mnt/vast/kyle/code2/Primus-Turbo/ $PWD/Primus-Turbo/
```

## 执行命令（本机本地）

完整流程：

```bash
# 1. 本地编辑
$EDITOR /workspace/code/gpt_oss_docker/sync/Primus-Turbo/scripts/foo.py

# 2. 直接本地运行（无需 push，无 ssh/docker）
cd /workspace/code/gpt_oss_docker/sync/Primus-Turbo && python scripts/foo.py
```

## HipKittens（未同步到本机）

HipKittens **没有**同步到本机。如需用，照 Primus-Turbo 同样的方式从
`chi2811:/mnt/vast/kyle/code2/HipKittens` 拉取：

```bash
cd /workspace/code/gpt_oss_docker/sync && \
  rsync -azh --exclude-from=.rsync-exclude -e "$PWD/.ssh-chi.sh" \
  root@chi2811:/mnt/vast/kyle/code2/HipKittens/ $PWD/HipKittens/
```

## 排除规则（`.rsync-exclude`）

排除：
- 构建产物：`/build/`、`/dist/`（**根级**，不动 Python 模块里同名目录）、`*.so/o/a`、`*.pyc`、`__pycache__/`、`*.egg-info/`、`.pytest_cache/`、`pytest-of-*/`
- 性能采样输出目录：`.rocprofv3/`
- 大二进制：`*.rpd`、`*.parquet`、`*.npy`、`*.npz`
- env：`venv/`、`.venv/`、`env/`、`.env/`
- 编辑器：`.idea/`、`*.swp`
- 日志：`*.log`

**不排除**（之前排过、后来按用户要求恢复）：
- `auto_optimize_logs/` —— 用户要看 log，不能排掉。Primus-Turbo 这个目录大概 230M。

**关于 `.git/`**：本机从 chi2811 拉取（pull 方向）时，`.rsync-exclude` **必须排除 `.git/`**，否则远程的 stale `.git/refs` 会覆盖本地 `.git/`，把分支 ref detach 到旧位置（commits 还在 `.git/objects` 里，但 HEAD 退回，需要 `git reflog` + `git reset --hard <sha>` 才能找回）。本机刷新只有 pull 这一个方向，所以保持 `.git/` 在排除列表里即可。

**不排除**：`*.csv`、`*.md`、`auto_optimize_logs/`（用户要看）

要改 → 编辑 `/workspace/code/gpt_oss_docker/sync/.rsync-exclude`，新增/删除规则后下次从 chi2811 刷新时生效。

## 常见坑（从 chi2811 刷新时）

- **rsync 排除模式默认匹配任意深度的同名目录**。例如 `dist/` 会误中 `primus_turbo/pytorch/dist/`（这是 Python 模块）。所以 `/build/`、`/dist/` 用 **/ 开头**锚定到 repo 根。
- **trailing slash**：rsync 里 `src/` 表示"目录内容"，`src` 表示"目录本身"。手写 rsync 时小心（顶部 banner 的命令两端都带 `/`，是同步目录内容）。
- **submodule**（Primus-Turbo 有 `3rdparty/composable_kernel` 和 `3rdparty/HipKittens`）：从 chi2811 刷新时 rsync 会同步子模块工作树文件；`.git/` 被排除，子模块 git 元数据不动。

## 故障排查

- 怀疑 rsync 排除规则吃掉了想要的文件 → 看 `/workspace/code/gpt_oss_docker/sync/.rsync-exclude` 并跑 `rsync -nv --exclude-from=...` 做 dry-run 验证。
- 从 chi2811 刷新后想看变了什么 → 在 `/workspace/code/gpt_oss_docker/sync/Primus-Turbo` 跑本地 `git status` / `git diff`。
- 改了 cpp / .cu 这类需要重编的文件 → 改完直接 **重新 build**（本机就地 build，无需任何 sync）。
