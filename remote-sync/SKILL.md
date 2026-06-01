---
name: remote-sync
description: 在本地 /wekafs/kyle/code2/remote_sync/{Primus-Turbo,HipKittens} 编辑代码，按需 rsync 推到远程 host 路径 /mnt/shared/kyle/code2/<repo>（即容器内 /workspace/code/<repo>）后再执行。当用户要求"修改/编辑/重构 Primus-Turbo 或 HipKittens 的代码"、要"同步/上传/推到远程"、或要"运行 X"时使用此 skill。配合 remote-mlperf-gptoss skill 一起使用。
---

# remote-sync

远程主机没有外网。代码编辑、查文档、跑 lint 这些**都在本地**做；要执行/编译/跑实验时再把改动 rsync 到远程。

## 布局

```
本地（有网）                          远程（无网）
/wekafs/kyle/code2/remote_sync/             compute_node_new:/mnt/shared/kyle/code2/
├── .rsync-exclude                       │  ↑↑↑ 这就是容器里的 /workspace/code
├── sync.sh                              │
├── Primus-Turbo/        ─── rsync ──>  ├── Primus-Turbo/   (容器内 /workspace/code/Primus-Turbo)
│   └── .git/                            │
└── HipKittens/          ─── rsync ──>  └── HipKittens/     (容器内 /workspace/code/HipKittens)
    └── .git/
```

- 远程 host 路径 `/mnt/shared/kyle/code2` 是 mlperf_gptoss 容器 `/workspace/code` 的 bind mount —— 推到 host 路径，容器立刻看见，**不需要**任何 docker 操作。
- 本地保留完整 `.git/`，所以 `git status / log / diff / commit / branch` 都在本地跑，**不要**通过 ssh 在远程跑 git 命令（除非验证 commit 一致性，见下）。

> ⚠️ **本地 `/wekafs/kyle/code2/remote_sync/<repo>` 与远程 `/mnt/shared/kyle/code2/<repo>` 是两个独立存储**(本地是 wekafs,远程是节点 host 上的另一个共享盘)。**不要**因为路径相似或 `wc -l` 偶然相等就以为同一份 —— 真相是 rsync 没跑过 / 上次同步是几天前的旧版。每次远程执行前必须 `sync.sh push`,否则你 build / 跑的就是旧 kernel,白白debug 半天发现"perf 没变"。这条踩过,血的教训。

## 两边 commit 必须一致（本地为准）⚠️ 硬约束

**核心规则**：每次 push 后，**本地与远程的 git HEAD 必须指向同一 commit**。本地是**唯一**真理源；远程绝不能"自己有未提交修改"或"卡在旧 commit"。

为什么：
- 之前为了避免 `.git/` 同步覆盖本地分支 ref（pull 方向的隐患），把 `.git/` 设成硬排除，结果代价是 **远程 git 工作树可以与本地分叉**（远程 HEAD 是过去某个时刻的）。这让 ssh 跑出来的行为不可复现："本地 git 显示这是 HEAD,远程 build 的是别的 commit"。
- 用户明确要求："两边的 commit 得一样,按照本地的为准"。

操作规则：
1. **任何 push 之后**，本地工作树 + .git/ 都要镜像到远程 —— 即 `.git/` 也要随 push 推过去，并让远程 HEAD = 本地 HEAD。
2. **pull 方向永远不要拉 `.git/`**（防止远程 stale ref 覆盖本地的新 commit）。`sync.sh pull` 必须继续排除 `.git/`。
3. push 完成后，**强制校验**：
   ```bash
   LOCAL_SHA=$(cd /wekafs/kyle/code2/remote_sync/<repo> && git rev-parse HEAD)
   REMOTE_SHA=$(ssh compute_node_new "cd /mnt/shared/kyle/code2/<repo> && git rev-parse HEAD" 2>/dev/null)
   [ "$LOCAL_SHA" = "$REMOTE_SHA" ] || { echo "❌ commit drift: local=$LOCAL_SHA remote=$REMOTE_SHA"; exit 1; }
   ```
   不一致就重推（`sync.sh push <repo>` 默认应当带 `.git/` 上行）。
4. 如果远程"自己改了东西"导致工作树脏或 HEAD 偏移：**毫不犹豫覆盖**。本地是 canonical，远程没有任何独立提交权。
5. 唯一允许在远程跑的 git 命令是 **只读** 的 `git rev-parse HEAD` / `git log -1` 做校验；任何 `git commit/checkout/reset/stash/pull` 在远程跑都是 bug。

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
/wekafs/kyle/code2/remote_sync/sync.sh push Primus-Turbo

# 只推一个文件或子目录
/wekafs/kyle/code2/remote_sync/sync.sh push Primus-Turbo primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py
/wekafs/kyle/code2/remote_sync/sync.sh push HipKittens analysis/fp8_gemm/

# 拉远程改动
/wekafs/kyle/code2/remote_sync/sync.sh pull Primus-Turbo

# Dry-run：看看 push 会传什么（不实际传）
/wekafs/kyle/code2/remote_sync/sync.sh diff Primus-Turbo

# 严格镜像（删两边不一致的文件）—— 慎用
/wekafs/kyle/code2/remote_sync/sync.sh push Primus-Turbo --mirror
```

### 不用脚本，直接 rsync

```bash
rsync -avzhP --exclude-from=/wekafs/kyle/code2/remote_sync/.rsync-exclude -e ssh \
  /wekafs/kyle/code2/remote_sync/Primus-Turbo/ \
  compute_node_new:/mnt/shared/kyle/code2/Primus-Turbo/
```

## 执行远程命令（与 remote-mlperf-gptoss skill 配合）

完整流程：

```bash
# 1. 本地编辑
$EDITOR /wekafs/kyle/code2/remote_sync/Primus-Turbo/scripts/foo.py

# 2. 推到远程
/wekafs/kyle/code2/remote_sync/sync.sh push Primus-Turbo scripts/foo.py

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

**方向相关**：
- `.git/` —— **push 方向不排除**（保证两边 commit 一致，本地为准）。**pull 方向必须排除**，否则远程的 stale `.git/refs` 会覆盖本地后续的 commit，把分支 ref detach 到旧位置（commits 还在 `.git/objects` 里，但 HEAD 退回，需要 `git reflog` + `git reset --hard <sha>` 才能找回）。
- 也就是说 `.rsync-exclude` 不能再无脑两向通用；`sync.sh push` 不带 `.git/` 排除，`sync.sh pull` 必须带。若手写 rsync：push 命令**不要**用 `--exclude=.git/`，pull 命令**必须**用 `--exclude=.git/`。

**不排除**：`*.csv`、`*.md`、`auto_optimize_logs/`（用户要看）

要改 → 编辑 `/wekafs/kyle/code2/remote_sync/.rsync-exclude`，新增/删除规则后下次同步生效。

## 常见坑

- **rsync 排除模式默认匹配任意深度的同名目录**。例如 `dist/` 会误中 `primus_turbo/pytorch/dist/`（这是 Python 模块）。所以 `/build/`、`/dist/` 用 **/ 开头**锚定到 repo 根。
- **trailing slash**：rsync 里 `src/` 表示"目录内容"，`src` 表示"目录本身"。`sync.sh` 已自动处理，手写 rsync 时小心。
- **ControlMaster**：第一条 ssh 慢，后面会快。不用每条命令都加 `-o ConnectTimeout=20`。
- **submodule**（Primus-Turbo 有 `3rdparty/composable_kernel` 和 `3rdparty/HipKittens`）：rsync 会同步工作树文件 + 子模块 `.git/`，但子模块 HEAD 一致性也要 verify（push 后两边 `git -C 3rdparty/<sub> rev-parse HEAD` 应当相同）。
- **`.git/` 在 push 方向同步，pull 方向不同步**（见上节）。本地 `git log` 是 canonical 历史；远程 `git log` 在每次 push 后会被本地覆盖到一致。

## 故障排查

- `sync.sh diff <repo>` —— dry-run，看会传什么、传多少。
- **commit 漂移检查**：`ssh compute_node_new "cd /mnt/shared/kyle/code2/<repo> && git rev-parse HEAD"` 跟本地 `git -C /wekafs/kyle/code2/remote_sync/<repo> rev-parse HEAD` 比对；不等就重 push 直到等。
- 本地 `git status` 跟同步前的远程 `git status` 对比，确认 rsync 没漏文件。
- 怀疑 rsync 排除规则吃掉了想要的文件 → 看 `.rsync-exclude` 并跑 `rsync -nv --exclude-from=...` 验证。
- 推完之后远程没看到 → 确认推到的是 host 路径 `/mnt/shared/kyle/code2/<repo>`，**不是**容器内路径（容器路径只能通过 docker exec 访问，rsync 不能直接走）。
- "我改了文件,远程 build 出来的 .so 行为没变" → 99% 是没 sync。本地 `wc -l file` 跟 `ssh compute_node_new 'wc -l /mnt/shared/kyle/code2/.../file'` 对比;不一致就是没推。立刻 `sync.sh push`,然后 **重新 build**(改的是 cpp / .cu 这种需要重编的文件时)。
