# gpt_oss2 同步/git推送/磁盘清理

> 类别: 连接 (gpt_oss2) · 主题标签: rsync, 同步, checksum, rsync-exclude, git-push, ssh-key, origin, cursor-coauthor, disk-cleanup, overlay, docker, safety

## gpt_oss2 本地→远端 rsync 同步规程

- **从 sync/ 目录跑**，`SSH="$(pwd)/.ssh-chi.sh"`（相对 cwd，须在 sync/ 下执行）。
- **推（push）**：`rsync -azh --exclude-from=.rsync-exclude -e "$SSH" ./mxfp8/ root@chi2811:/mnt/vast/kyle/code3/mxfp8/`
  - **含 .git**（.rsync-exclude 保留 .git/），新目录加 `--mkpath` 自动建远端父目录。
  - 注意 **trailing /**（源尾斜杠决定同步的是目录内容而非目录本身）。
- **拉（pull）**：必须额外 `--exclude='.git/'`，WHY = 防远端 stale ref 覆盖本地新 commit。
- **kernel 同步防旧缓存**：用 `rsync -av --checksum`，WHY = mtime 相同会被跳过而 Python 加载旧字节码；同步后 `md5sum` 本地与远程逐一对比确认（不可省）。
- **.rsync-exclude 排除规则**（sync/.rsync-exclude）：
  - 排除：`/build/`、`/dist/`（根级锚定 `/`）、`*.so`/`*.o`/`*.a`、`*.pyc`、`__pycache__/`、`*.egg-info/`、`.pytest_cache/`、`.rocprofv3/`、`*.rpd`/`*.parquet`/`*.npy`/`*.npz`、venv 类、`*.log` 等。
  - 保留：`.git/`（push 含、pull 手动 `--exclude='.git/'`）、`*.csv`、`*.md`、`auto_optimize_logs/`。

## gpt_oss2 git push 规程(本机 SSH key+origin,不推显式 URL)

- **origin**: `git@github.com:AMD-AGI/Primus-Turbo.git`(容器/本机均无 GitHub HTTPS 凭证 → 必须用 SSH key)。
- **commit(放行时)**: 在 `/workspace/code/gpt_oss2_docker/sync/<mxfp8|mxfp4>/Primus-Turbo` 目录里 `git add` + `git commit`,message 用 HEREDOC 写,**无 coauthor**。
- **push 命令**:
  ```
  GIT_SSH_COMMAND="ssh -i /workspace/code/.ssh_docker/id_ed25519 -o StrictHostKeyChecking=no -o IdentitiesOnly=yes" git push origin <branch>
  ```
- **必须推 `origin <branch>`,禁止推显式 URL**:显式 URL 不更新 tracking ref,会让 Cursor 面板假报 ahead。
- push 后校验命令见 common/13-git-push-ssh-override
- **mxfp8 收尾实例**: 单 commit `dac31090` push 到分支 `dev/kyle/flydsl_mxfp8_compute`,用 key `/workspace/code/.ssh_docker/id_ed25519` + **force-with-lease**。
- **远端只读 git**: 禁止在远程跑 `commit`/`checkout`/`reset`/`pull`;编辑只改本地再 `sync.sh push`(远端直接改会被冲掉)。
- **Cursor co-author 陷阱**: `attributeCommitsToAgent` 会自动加 `Co-authored-by: Cursor` trailer → 须在 cli-config 关掉 + 加规则 `.cursor/rules/no-cursor-coauthor.mdc`(用户硬性要求 commit 不许出现 cursor)。
- 参见 `common/13-git-push-ssh-override`：该文档描述的是 FlyDSL/显式 URL-override 场景（用 `git@` SSH URL 临时覆盖 HTTPS origin + `--force-with-lease`），与本卡片 origin-only 场景不同,不要混用两者规程。

## gpt_oss2 磁盘清理边界(overlay 根盘常年紧)

- **背景**: gpt_oss2 根 overlay 盘常年紧张,需定期清理,但机器为共享环境,必须只清明确属我们且可再生的内容。
- **可清(可再生,安全)**:
  - `rm -rf /root/.cache/comgr` — comgr 编译缓存,可再生
  - `rm -rf /root/.cache/pip` — pip 下载缓存,可再生
  - `rm -rf /tmp/qgolden` — quant golden 临时输出,可再生
  - `rm -rf /tmp/rocprof_out` — rocprof 临时输出,可再生
  - `rm -rf /tmp/*.log` — 临时日志
  - `docker builder prune -f` — 只清 docker build 缓存层,WHY: 不触及镜像/容器本体
- **绝不动(共享资产,不可再生/属别人)**:
  - 别人的镜像 / 容器 / volumes — WHY: 共享机器,误删他人工作不可逆
  - `mlperf_gptoss`(注意无"2"，挂 code2) — **别人的**容器，与本项目容器 `mlperf_gptoss2`(挂 code3) 无关，绝不可碰
  - 禁 `docker volume prune` — WHY: 会连带删掉别人的 volume
  - 禁 `docker rmi` 别人镜像 — WHY: 破坏他人环境
- **本项目自己的容器**: `mlperf_gptoss2`(挂 code3)，勿与上述别人的 `mlperf_gptoss` 混淆

---
来源: remote-sync/SKILL.md, mxfp8-8wave-devloop/SKILL.md, project_mxfp8_wholeloop_port.md, pr-merge-gate/SKILL.md
