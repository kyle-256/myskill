# 构建、Git 与 Triton 版本管理

> 类别: 连接 · 主题标签: git-push, ssh-override, author, force-with-lease, build, flydsl-cache, csrc-recompile, hipify-pitfall, triton, fp8-grouped, autotune, baseline

## Primus-Turbo/FlyDSL git push 与 author 覆盖

- **无凭证前提**：chi2774 容器与本机对 HTTPS origin（`AMD-AGI/Primus-Turbo`）都无 GitHub 凭证。push 必须用本机 jump-host SSH key，通过 git@ SSH URL 临时覆盖 HTTPS origin（**不改 remote 配置**）。
- **push 命令**（SSH key = `/workspace/code/.ssh_docker/id_ed25519`）：
  ```
  GIT_SSH_COMMAND="ssh -i /workspace/code/.ssh_docker/id_ed25519 -o StrictHostKeyChecking=no -o IdentitiesOnly=yes" \
    git push git@github.com:AMD-AGI/Primus-Turbo.git <branch>
  ```
- **push 后校验**：`git rev-list --left-right --count origin/<branch>...HEAD` 期望 `0 0`（本地与远端同步，无落后/领先）。
- **canonical 位置**：
  | 仓库 | canonical | commit 在哪 |
  |---|---|---|
  | FlyDSL | 远端 `/mnt/vast/kyle/code2/FlyDSL`（本地 `sync/FlyDSL` 是镜像） | 远端 commit |
  | Primus-Turbo (mxfp4/tensorwise) | 本地 | 本地 commit |
- **author 覆盖**：commit 必须显式覆盖 `author=kyle-256 <Kyle.Zhao@amd.com>`，`GIT_AUTHOR_*` / `GIT_COMMITTER_*` 全部设置。**禁止任何 Claude/Cursor coauthor**。WHY：默认身份会带工具署名。
- **URL 推 --force-with-lease**：用 URL（非命名 remote）推时，`--force-with-lease` 无期望值会误报 stale info。须先 `git fetch <url> <branch>` 确认远端仍是旧 hash，再用显式 `--force-with-lease=<branch>:<old-oid>` 推。

## 按需 build:何时重编 .so vs 清 flydsl cache

### Fresh 容器首次安装(flydsl 没装)
- 现象:fresh `rocm/primus:v26.2` 容器里 flydsl 没装 → 落 stub 分支报 `ImportError: cannot import name '_compile_dense_tn'`(`flydsl_available()` 返回 False)。
- 正解:从源码 editable 安装,**不要 PYTHONPATH hack**。PYTHONPATH 只够 `import flydsl`,但编译 kernel 还要 `_mlir` 的 `LD_LIBRARY_PATH`;editable install 会自动 symlink 解决。
  - `pip install -e FlyDSL`:build-fly/ 已存在时只 symlink(~30s),不重 build MLIR。
  - `GPU_ARCHS=gfx950 pip install --no-build-isolation -e Primus-Turbo`:~15-25min。

### 改动后要重编什么(决策表)
| 改了什么 | 动作 | WHY |
|---|---|---|
| flydsl kernel(.py)/ 纯 .py | `rm -rf /root/.flydsl/cache` 再跑 | 无需重编 .so;不清 cache 会跑到旧编译产物 |
| cpp/csrc/.cu/.cpp | clean 重编 .so | 否则测的是旧 .so,读到错误值 |

- csrc 重编命令:`rm -rf build primus_turbo/lib/*.so && GPU_ARCHS=gfx950 pip install --no-build-isolation -e .`(~15-25min);tensorwise 用对应 venv。重编产物是 `libprimus_turbo_kernels.so`。

### 陷阱:增量 pip install -e . 不真重编 csrc
- 改 csrc 后做增量 `pip install -e .` **不会真重编**变动的 hip 对象:
  - `rsync -a` 保留旧 mtime + hipify 中间产物在 `build/temp`,ninja 按时间戳判"过时"判错。
  - 旧 .o 用上个 checkout 的参数序 → 运行时读到错误值(如 mxfp8 误报 `MXFP8 not support RHT`)。
- 5min 完成的"增量重编"没真重编;**必须 clean 重编**(`rm -rf build primus_turbo/lib/*.so`,~15-25min)。

## triton 3.6→3.7 升级与 baseline 作废

- **rocm/primus:v26.2 镜像默认 triton 3.6.0**;fp8 grouped GEMM bench 要求 **triton 3.7**,WHY:3.7 的 autotune cfg 更全(覆盖更多候选配置)。
- 升级命令:`/opt/venv/bin/pip install --upgrade triton`。
- **作废规则**:装完 3.7 后,所有用 triton 3.6 跑出的 bench 数字全部作废,必须重 bench 立新 baseline——不同 autotune cfg 集会选出不同 winner,3.6 与 3.7 的数字不可直接比较。

---
来源: remote-sync/SKILL.md, pr-merge-gate/SKILL.md, 14-fused-preshuffle-e2e.md, claim-mi355x-node/SKILL.md, remote-mlperf-gptoss/SKILL.md, flydsl-fp8-gemm-tuning/SKILL.md, 13-primus-turbo-prod.md, SKILL.md
