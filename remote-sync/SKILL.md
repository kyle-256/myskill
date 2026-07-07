---
name: remote-sync
description: 在本地 /workspace/code/conductor_455/sync/<repo> 编辑代码，rsync 推到远程 host 路径 /home/zhuang12/kyle_tmp/<repo>（即容器内 /workspace/code/<repo>）后再执行。当用户要"修改/编辑代码"、"同步/推到远程"、"运行 X"时使用。配合 remote-conductor455 skill。
---

# remote-sync

代码在本地编辑；要执行/测试时再 rsync 到远程。远程**有外网**，但代码改动仍走本地 → rsync，保持 git 历史在本地。

## 布局

```
本地（有网）                                    远程 host（直连）
/workspace/code/conductor_455/sync/             zhuang12@heliosp-1b114-c05-3:/home/zhuang12/kyle_tmp/
├── .ssh-helio.sh                            │  ↑ 即容器内 /workspace/code
└── FlyDSL/              ─── rsync ──>      └── FlyDSL/
    └── .git/
```

- SSH wrapper：`/workspace/code/conductor_455/sync/.ssh-helio.sh`
- 推到 host 路径后，容器立即可见（bind mount），**不需要任何 docker 操作**。
- git 只在本地跑，远程不跑 git 命令。

## 两端 commit 必须一致（本地为准）⚠️

push 后必须校验：
```bash
LOCAL=$(git -C /workspace/code/conductor_455/sync/<repo> rev-parse HEAD)
REMOTE=$(/workspace/code/conductor_455/sync/.ssh-helio.sh \
  zhuang12@heliosp-1b114-c05-3.mnb.dcgpu \
  "git -C /home/zhuang12/kyle_tmp/<repo> rev-parse HEAD" 2>/dev/null | tail -1)
[ "$LOCAL" = "$REMOTE" ] && echo "✓ 一致" || echo "✗ 不一致，需重推"
```

不一致就重推（带 `.git/` 一起推）。

## rsync 命令速查

```bash
SSH_CMD="/workspace/code/conductor_455/sync/.ssh-helio.sh"
LOCAL="/workspace/code/conductor_455/sync"
REMOTE="zhuang12@heliosp-1b114-c05-3.mnb.dcgpu:/home/zhuang12/kyle_tmp"

# 推整个 repo（含 .git/，保证两端 commit 一致）
rsync -avzhP -e "$SSH_CMD" $LOCAL/FlyDSL/ $REMOTE/FlyDSL/

# 推单个文件或子目录
rsync -avzhP -e "$SSH_CMD" $LOCAL/FlyDSL/flydsl/some_file.py $REMOTE/FlyDSL/flydsl/

# dry-run（看会传什么，不实际传）
rsync -avzhPn -e "$SSH_CMD" $LOCAL/FlyDSL/ $REMOTE/FlyDSL/

# 拉远程改动（不带 .git/，防止 stale ref 覆盖本地）
rsync -avzhP -e "$SSH_CMD" --exclude='.git/' $REMOTE/FlyDSL/ $LOCAL/FlyDSL/
```

## 同步触发原则

**默认不主动同步。** 只在以下情形 push：

1. 用户明确说"同步"、"推到远程"、"上传"。
2. 用户要"运行 X"、"跑 Y"等需要在远程执行的动作 —— 先 push 再 ssh 执行。
3. 改了多个文件准备执行前。

**不要做的事：**
- 每次 Edit 后立刻 push。
- 用 `--mirror`（会删远程文件），除非用户明确要求"完全镜像"。
- 用 `ssh ... 'git ...'` 在远程查 git 状态 —— 用本地 git。

## 完整流程示例

```bash
SSH_CMD="/workspace/code/conductor_455/sync/.ssh-helio.sh"
HOST="zhuang12@heliosp-1b114-c05-3.mnb.dcgpu"

# 1. 本地编辑
$EDITOR /workspace/code/conductor_455/sync/FlyDSL/flydsl/some_kernel.py

# 2. 推到远程
rsync -avzhP -e "$SSH_CMD" \
  /workspace/code/conductor_455/sync/FlyDSL/ \
  $HOST:/home/zhuang12/kyle_tmp/FlyDSL/

# 3. 容器内执行
$SSH_CMD $HOST 'docker exec conductor_455 bash -lc "cd /workspace/code/FlyDSL && python3 tests/foo.py"'
```

## 排除建议

push 时可加 `--exclude` 跳过无用文件：
- `__pycache__/`、`*.pyc`、`*.egg-info/`、`.pytest_cache/`
- `build/`、`dist/`（根级）、`*.so`、`*.o`
- `venv/`、`.venv/`

`.git/` **push 方向不排除**（保证两端 commit 一致）；pull 方向**必须排除**。

## 已知 repo

| repo | 本地 | 远程 host | 容器内 |
|------|------|-----------|--------|
| FlyDSL | `/workspace/code/conductor_455/sync/FlyDSL/` | `/home/zhuang12/kyle_tmp/FlyDSL/` | `/workspace/code/FlyDSL/` |

## 常见坑

- **同步后容器看不到** → 确认 rsync 目标是 host 路径 `/home/zhuang12/kyle_tmp/...`，不是容器内路径。
- **改了文件，远程跑出来行为没变** → 99% 是没 sync 或 flydsl 磁盘缓存旧 binary（改 kernel 后 `rm -rf /root/.flydsl/cache`）。
- **两端 commit 不一致** → 重推整个 repo（含 `.git/`）直到校验通过。
