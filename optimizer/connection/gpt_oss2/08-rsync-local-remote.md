# gpt_oss2 本地→远端 rsync 同步规程

> 类别: 连接 (gpt_oss2) · 主题标签: rsync, 同步, checksum, rsync-exclude

- **从 sync/ 目录跑**，`SSH="$(pwd)/.ssh-chi.sh"`（相对 cwd，须在 sync/ 下执行）。
- **推（push）**：`rsync -azh --exclude-from=.rsync-exclude -e "$SSH" ./mxfp8/ root@chi2811:/mnt/vast/kyle/code3/mxfp8/`
  - **含 .git**（.rsync-exclude 保留 .git/），新目录加 `--mkpath` 自动建远端父目录。
  - 注意 **trailing /**（源尾斜杠决定同步的是目录内容而非目录本身）。
- **拉（pull）**：必须额外 `--exclude='.git/'`，WHY = 防远端 stale ref 覆盖本地新 commit。
- **kernel 同步防旧缓存**：用 `rsync -av --checksum`，WHY = mtime 相同会被跳过而 Python 加载旧字节码；同步后 `md5sum` 本地与远程逐一对比确认（不可省）。
- **.rsync-exclude 排除规则**（sync/.rsync-exclude）：
  - 排除：`/build/`、`/dist/`（根级锚定 `/`）、`*.so`/`*.o`/`*.a`、`*.pyc`、`__pycache__/`、`*.egg-info/`、`.pytest_cache/`、`.rocprofv3/`、`*.rpd`/`*.parquet`/`*.npy`/`*.npz`、venv 类、`*.log` 等。
  - 保留：`.git/`（push 含、pull 手动 `--exclude='.git/'`）、`*.csv`、`*.md`、`auto_optimize_logs/`。

---
来源: remote-sync/SKILL.md, mxfp8-8wave-devloop/SKILL.md
