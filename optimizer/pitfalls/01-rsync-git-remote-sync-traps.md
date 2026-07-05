# 远端同步陷阱：rsync --delete / 锚定规则 / git dubious-ownership / docker 重定向

> 类别: 踩过的坑 · 主题标签: rsync, git, docker, remote-sync

- **[rsync] ❌ 别再试 给 rsync 加 `--delete`。** 本地是全新 checkout，`--delete` 会删掉远程一堆不能删的东西：
  - `primus_turbo/_build_info.py`（build 生成、运行时被 import，删了直接挂）
  - hipify 生成的中间件 `*_hip.cpp` / `.h` / `.cuh`（远程编译产物）
  - `3rdparty/` 子模块源码（本地是空 gitlink、远程才有实体，build 必需）
  - WHY：本地镜像根本没有这些文件，`--delete` 会以本地为准把远程抹掉 → 远端 build 直接崩。用**增量 rsync（不 delete）**即可。
- **[rsync] 排除规则必须用 `/` 开头锚定到 repo 根。** rsync 排除规则默认匹配**任意深度**同名目录 → 直接写 `build/`、`dist/` 会误伤同名的 Python 模块目录。要写 `/build/`、`/dist/` 锚定到 repo 根。
- **[rsync] `*.so` 被排除是正常的。** 远程 `primus_turbo/lib/libprimus_turbo_kernels.so` 是远程 build 产物，不会被本地空目录覆盖；本地镜像里没有它属正常现象，别去补。
- **[git] `safe.directory '*'` 一劳永逸解 dubious ownership。** JuiceFS 挂载 uid 错配 → git 到处报 dubious ownership；`git clone` 本地路径报 "SSH access rights" 其实也是源 repo `.git` 的 dubious-ownership（**不是真要 SSH**，别去查 key）。解法：`git config --global --add safe.directory '*'`。
- **[git] 显式 URL push 不更新本地 tracking ref**（源自 gpt_oss2 项目，非本环境实测）**。** 用显式 URL（`git push git@github.com:...`）推，GitHub 端会更新，但**不更新本地 `origin/<branch>` tracking ref** → `git status` 仍显示 `[ahead N]`、Cursor 面板误显示待推送。修法：`git fetch origin <branch>` 刷新，之后 `git rev-list --left-right --count origin/<branch>...HEAD` 应为 `0 0`。始终用 `git push origin <branch>`。
- **[git] 本地必设 `core.fileMode false`。** `git config core.fileMode false` —— JuiceFS + rsync 翻动权限位会假报一堆 `M`（modified）。
- **[redirect] ❌ 别再试 在 `docker exec` 外层重定向想写容器内文件。** `docker exec '... > /tmp/x'` 的重定向落在 **host** 不在容器。要写容器内文件必须在 `bash -lc` **内部**重定向（`docker exec ... bash -lc '... > /tmp/x'`）。

---
来源: remote-sync/SKILL.md, 10-grouped-wgrad-4wave-3buf.md, flydsl-sync/SKILL.md, gpt_oss2/myskill/remote-sync/SKILL.md
