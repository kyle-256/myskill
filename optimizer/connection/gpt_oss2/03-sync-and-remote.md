# 同步与远端协作：rsync 规程、git 一致性、push

> 类别: 连接 · 主题标签: rsync, 跳板机, 共享盘, canonical镜像, git-author, git-push, tracking-ref, 切分支重编, 行尾

## 工作模型

**本地编辑(本容器有网,无 GPU)→ rsync 推到当前节点的 host 路径 → 用对应 venv 在 GPU 上跑。** 本地 `sync/` 是 canonical(与 gpt_oss 环境的 FlyDSL 相反,那边 canonical 在远端)。

| | mxfp8 | mxfp4 |
|---|---|---|
| 本地镜像(canonical) | `/workspace/code/gpt_oss2_docker/sync/mxfp8/Primus-Turbo` | `…/sync/mxfp4/Primus-Turbo` |
| 远程 host 路径 | `/mnt/vast/kyle/code3/mxfp8/Primus-Turbo` | `/mnt/vast/kyle/code3/mxfp4/Primus-Turbo` |
| 容器内路径 | `/workspace/code/mxfp8/Primus-Turbo` | `/workspace/code/mxfp4/Primus-Turbo` |

两份都是 git 仓库。host `/mnt/vast/kyle/code3` 是**跨节点共享**的大盘(已验证换节点后内容原样可见),所以换节点**不需要重传代码**,只需 rsync 增量补本地新改动;镜像 tar 也放这里。`/opt/venv*` 在容器 overlay 根盘(紧)。

## rsync 规程

从 `sync/` 目录跑,注意 trailing `/`:

```bash
cd /workspace/code/gpt_oss2_docker/sync
SSH="$(pwd)/.ssh-chi.sh"

# 推(本地→远程,含 .git;新目录加 --mkpath)
rsync -azh --exclude-from=.rsync-exclude -e "$SSH" ./mxfp4/ root@chi2879:/mnt/vast/kyle/code3/mxfp4/
# 拉(远程→本地,**必须额外排 .git/**,防 stale ref 覆盖本地新 commit)
rsync -azh --exclude-from=.rsync-exclude --exclude='.git/' -e "$SSH" \
  root@chi2879:/mnt/vast/kyle/code3/mxfp8/ ./mxfp8/
# 看会传什么
rsync -azhn --stats --exclude-from=.rsync-exclude -e "$SSH" ./mxfp4/ root@chi2879:… | tail -20
```

- **`.rsync-exclude`**:排 `/build/`、`/dist/`(**根级锚定**,否则 rsync 默认匹配任意深度同名目录)、`*.so/o/a`、`*.pyc`、`__pycache__/`、`*.egg-info/`、`.pytest_cache/`、`.rocprofv3/`、`*.rpd/parquet/npy/npz`、`venv/`、`*.log`。**保留** `.git/`(push 含,pull 手动排)、`*.csv`、`*.md`。
- **同步触发原则**:默认不主动同步。只在①用户明说同步/推,②要在远程跑/build/test 前(exec 前自动 push 涉及的 workdir 一次),③改多文件准备执行前。别每次 Edit 后立刻 push。
- **WHY 必须先同步再测**:不推就在远端编旧代码,所有 perf/SNR/ISA 结论都对着旧二进制 → 假象。怀疑没生效时 `wc -l` 本地 vs 远端对比。
- **改了 csrc 远程行为没变** → 99% 没 sync 或没重 build(见 02 铁律 1)。
- **切分支后重编,远端源码树要干净**:hipify 会把 `*.hip`/`*_hip.{h,cuh,cpp}` 生成进 `csrc/`(git 不 track),旧分支残留会导致 stale build。做法:整树 **no-delete** push(保住 `agent/`、`scripts2/`、根级 `_*.py` 探针、`.git`、`3rdparty`、`.so`)后,**对 `csrc/` 单独 `rsync --delete`** 精准清残留,再 `rm -rf build && setup.py build_ext --inplace`。**切忌整树裸 `--delete`**:会抹掉探针脚本 + 远端 `.git` 对象,并用本地未填充的空 `3rdparty` 覆盖远端已建子模块。

## git 两边一致(本地为准)

硬约束:①任何 push 后本地工作树+`.git/` 镜像到远程,远程 HEAD == 本地 HEAD;②pull 永不拉 `.git/`;③push 后校验;④远程工作树脏/HEAD 偏移 → 毫不犹豫覆盖;⑤**远程只允许只读 git**(`rev-parse`/`log -1`),任何 `commit/checkout/reset/pull` 在远程跑都是 bug。

```bash
LOCAL=$(cd mxfp4/Primus-Turbo && git rev-parse HEAD)
REMOTE=$(./.ssh-chi.sh root@chi2879 "cd /mnt/vast/kyle/code3/mxfp4/Primus-Turbo && git rev-parse HEAD")
[ "$LOCAL" = "$REMOTE" ] || echo "❌ drift local=$LOCAL remote=$REMOTE"
```

- **dubious ownership**(vast/JuiceFS uid 错配):远端 git 报错时 `git config --global --add safe.directory '*'`。
- **submodule 指针**:换分支前 `git diff <b1> <b2> -- .gitmodules 3rdparty` 看是否变;变了要重拉/重拷 `3rdparty`(`composable_kernel` 只需 headers ~70M,`hipify_torch` 空也能 build)。

## git push

用本机 SSH key(容器/本机均无 GitHub HTTPS 凭证),**推 `origin <branch>`,不要推显式 URL**:

```bash
GIT_SSH_COMMAND="ssh -i /workspace/code/.ssh_docker/id_ed25519 -o StrictHostKeyChecking=no -o IdentitiesOnly=yes" \
  git push origin <branch>
git rev-list --left-right --count origin/<branch>...HEAD   # 期望 0  0
```

- **坑 — 显式 URL push 不更新 remote-tracking ref**:`git push git@github.com:…` GitHub 端会更新,但本地 `origin/<branch>` 不动,`git status` 仍显示 `[ahead N]`,**Cursor 源代码管理面板误报「待推送」**。已经推过想消除:`git fetch origin <branch>` 刷新。
- **commit author 必须是 `kyle-256 <Kyle.Zhao@amd.com>`;禁止任何 Claude/Cursor coauthor trailer。**
- 工作区根 `/workspace/code/gpt_oss2_docker` 不是 git 仓库,真正的仓库是嵌套的 `sync/<mxfp8|mxfp4>/Primus-Turbo`,Cursor git 面板要对应到子目录。

## ⚠️ 改文件别用整文件重写(行尾陷阱)

仓库里**混有 CRLF 文件**(如 `csrc/pytorch/kernels/grouped_gemm/grouped_gemm_fp4_impl.py`、部分 skill md)。用 Python `read_text()/write_text()` 整文件重写会把 CRLF 静默 normalize 成 LF,**整个文件进 diff**(实测 7 行改动炸成 1135 行)。

- 改动请用**就地字符串替换**的编辑器工具(保留原字节),或改完 `file <path>` 确认行尾未变、`git diff --numstat` 看行数是否合理。
- 已经炸了:`git checkout -- <file>` 还原后重做 —— 但**注意该命令会连同该文件里别人未提交的改动一起丢**。真丢了可从 object store 找回:`git fsck --unreachable | grep blob`,`git cat-file -p <blob>` 认内容,再 `sed 's/$/\r/'` 转回 CRLF。

## 远程磁盘清理

根 overlay 常年紧。**只清明确属我们/可再生的**,绝不动别人镜像/容器/volumes:

```bash
docker exec mlperf_gptoss2 bash -lc 'rm -rf /root/.cache/comgr /root/.cache/pip /tmp/qgolden /tmp/rocprof_out /tmp/*.log'
docker builder prune -f          # 全局 build cache(可再生)
# 不要: docker volume prune / docker rmi 别人的镜像 / 删 mlperf_gptoss
```

---
来源: remote-sync/SKILL.md, pr-merge-gate/SKILL.md, 2026-07-30 实测
