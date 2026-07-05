# Primus-Turbo/FlyDSL git push 与 author 覆盖

> 类别: 连接 · 主题标签: git-push, ssh-override, author, force-with-lease

- **无凭证前提**：chi2811 容器与本机对 HTTPS origin（`AMD-AGI/Primus-Turbo`）都无 GitHub 凭证。push 必须用本机 jump-host SSH key，通过 git@ SSH URL 临时覆盖 HTTPS origin（**不改 remote 配置**）。
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

---
来源: remote-sync/SKILL.md, pr-merge-gate/SKILL.md, 14-fused-preshuffle-e2e.md
