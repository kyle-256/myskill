---
name: remote-sync
description: 在本地 /wekafs/kyle/remote_sync/{Primus-Turbo,HipKittens} 编辑代码，按需 rsync 推到远程 host 路径 /mnt/shared/kyle/code2/<repo>（即容器内 /workspace/code/<repo>）后再执行。当用户要求"修改/编辑/重构 Primus-Turbo 或 HipKittens 的代码"、要"同步/上传/推到远程"、或要"运行 X"时使用此 skill。配合 remote-mlperf-gptoss skill 一起使用。
---

# remote-sync

远程主机没有外网。代码编辑、查文档、跑 lint 这些**都在本地**做；要执行/编译/跑实验时再把改动 rsync 到远程。

## 布局

```
本地（有网）                          远程（无网）
/wekafs/kyle/remote_sync/             compute_node_new:/mnt/shared/kyle/code2/
├── .rsync-exclude                       │  ↑↑↑ 这就是容器里的 /workspace/code
├── sync.sh                              │
├── Primus-Turbo/        ─── rsync ──>  ├── Primus-Turbo/   (容器内 /workspace/code/Primus-Turbo)
│   └── .git/                            │
└── HipKittens/          ─── rsync ──>  └── HipKittens/     (容器内 /workspace/code/HipKittens)
    └── .git/
```

- 远程 host 路径 `/mnt/shared/kyle/code2` 是 mlperf_gptoss 容器 `/workspace/code` 的 bind mount —— 推到 host 路径，容器立刻看见，**不需要**任何 docker 操作。
- 本地保留完整 `.git/`，所以 `git status / log / diff / commit / branch` 都在本地跑，**不要**通过 ssh 在远程跑 git 命令。
- 远程那一份只用来执行；远程的工作树可能有未提交修改（已知，参见 `git status`），同步时不要无脑 `--mirror`。

## 同步触发原则（重要）

**默认不主动同步**。只在以下情形 push：

1. 用户明确说"同步"、"推到远程"、"上传"、"sync 一下"。
2. 用户要"运行 X"、"跑 Y"、"benchmark"、"test"等需要在远程执行的动作 —— 在 ssh 跑命令**之前**，自动 `push` 涉及的 repo 一次。
3. 改了多个文件后准备执行前。

**不要**做的事：
- 每次 Edit 之后立刻 push（除非用户明确要求 watch 模式）。
- 默认带 `--mirror`（会删远程文件）。除非用户明确要求"完全同步"或"删除远程多余文件"。
- 用 `ssh ... 'git ...'` 在远程查 git 状态 —— 用本地的 git。

## 命令速查

### sync.sh（推荐入口）

```bash
# 推单个 repo（增量，不删远程文件）
/wekafs/kyle/remote_sync/sync.sh push Primus-Turbo

# 只推一个文件或子目录
/wekafs/kyle/remote_sync/sync.sh push Primus-Turbo primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py
/wekafs/kyle/remote_sync/sync.sh push HipKittens analysis/fp8_gemm/

# 拉远程改动
/wekafs/kyle/remote_sync/sync.sh pull Primus-Turbo

# Dry-run：看看 push 会传什么（不实际传）
/wekafs/kyle/remote_sync/sync.sh diff Primus-Turbo

# 严格镜像（删两边不一致的文件）—— 慎用
/wekafs/kyle/remote_sync/sync.sh push Primus-Turbo --mirror
```

### 不用脚本，直接 rsync

```bash
rsync -avzhP --exclude-from=/wekafs/kyle/remote_sync/.rsync-exclude -e ssh \
  /wekafs/kyle/remote_sync/Primus-Turbo/ \
  compute_node_new:/mnt/shared/kyle/code2/Primus-Turbo/
```

## 执行远程命令（与 remote-mlperf-gptoss skill 配合）

完整流程：

```bash
# 1. 本地编辑
$EDITOR /wekafs/kyle/remote_sync/Primus-Turbo/scripts/foo.py

# 2. 推到远程
/wekafs/kyle/remote_sync/sync.sh push Primus-Turbo scripts/foo.py

# 3. 在容器里跑
ssh compute_node_new 'docker exec mlperf_gptoss bash -lc "cd /workspace/code/Primus-Turbo && python scripts/foo.py"'
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

**不排除**：`*.csv`、`*.md`、`.git/`（要在本地用 git）、`auto_optimize_logs/`（用户要看）

要改 → 编辑 `/wekafs/kyle/remote_sync/.rsync-exclude`，新增/删除规则后下次同步生效。

## 常见坑

- **rsync 排除模式默认匹配任意深度的同名目录**。例如 `dist/` 会误中 `primus_turbo/pytorch/dist/`（这是 Python 模块）。所以 `/build/`、`/dist/` 用 **/ 开头**锚定到 repo 根。
- **trailing slash**：rsync 里 `src/` 表示"目录内容"，`src` 表示"目录本身"。`sync.sh` 已自动处理，手写 rsync 时小心。
- **ControlMaster**：第一条 ssh 慢，后面会快。不用每条命令都加 `-o ConnectTimeout=20`。
- **submodule**（Primus-Turbo 有 `3rdparty/composable_kernel`）：rsync 会同步它的工作树文件，但 git submodule 状态需要本地 `git submodule update` 自己维护。
- **.git 同步了**，所以本地 `git log` 看到的就是远程的提交历史。但本地 `git status` 反映的是**本地**工作树，不一定跟远程工作树一致 —— pull 之后才会同步工作树状态。

## 故障排查

- `sync.sh diff <repo>` —— dry-run，看会传什么、传多少。
- 本地 `git status` 跟同步前的远程 `git status` 对比，确认 rsync 没漏文件。
- 怀疑 rsync 排除规则吃掉了想要的文件 → 看 `.rsync-exclude` 并跑 `rsync -nv --exclude-from=...` 验证。
- 推完之后远程没看到 → 确认推到的是 host 路径 `/mnt/shared/kyle/code2/<repo>`，**不是**容器内路径（容器路径只能通过 docker exec 访问，rsync 不能直接走）。
