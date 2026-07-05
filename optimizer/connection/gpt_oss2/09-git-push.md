# gpt_oss2 git push 规程(本机 SSH key+origin,不推显式 URL)

> 类别: 连接 (gpt_oss2) · 主题标签: git-push, ssh-key, origin, cursor-coauthor

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

---
来源: remote-sync/SKILL.md, project_mxfp8_wholeloop_port.md, pr-merge-gate/SKILL.md
