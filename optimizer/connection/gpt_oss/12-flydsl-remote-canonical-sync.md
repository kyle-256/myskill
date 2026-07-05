# FlyDSL 远端为主 + JIT 缓存坑

> 类别: 连接 · 主题标签: 远端编译, JIT缓存, rsync, git-author

- **远端为主(编译要 GPU)**: FlyDSL kernel 编译 `flyc.compile` 就要 GPU,不能像 Primus-Turbo 那样纯本地跑。canonical git 在远端 `/mnt/vast/kyle/code2/FlyDSL`(= 容器内 `/workspace/code/FlyDSL`),本地 `sync/FlyDSL` 只是镜像。流程:改完在远端编译/测/commit,再 rsync `.git`+文件回本地对齐。
  - 对比: Primus-Turbo(mxfp4/tensorwise) canonical 在本地;FlyDSL 相反,canonical 在远端。
- **rsync 加 `--no-o --no-g`**: 避开 JuiceFS 的 chown/chmod 报错(cosmetic,但会刷屏)。
- **commit author 必须显式覆盖**: `author=kyle-256 <Kyle.Zhao@amd.com>`,`GIT_AUTHOR_*` / `GIT_COMMITTER_*` 全设;禁止任何 Claude/Cursor coauthor。
- **本地 flydsl egg 可能是坏的**: `sync/` 里的 egg(如 dev409)可能坏/API 太老,本地根本跑不了;所有 kernel 测试必须在远端容器跑(远端有匹配的 flydsl 版本)。
- **live import 源码在远端**: kernel 的 live import 源码是远端 `/workspace/code/Primus-Turbo/`(经 chi2774 ssh+docker),本地 `sync/mxfp4/Primus-Turbo/` 只是镜像。可用 `cat patch.py | ssh ... docker exec -i python -` 管道 patch 远端 live 文件,本地镜像同步改保持一致。

### JIT 缓存坑(最关键)
- **现象**: 改完 FlyDSL kernel 普通跑仍跑旧版二进制。根因: FlyDSL JIT 缓存**不 hash 模块级 class 方法**,clear comgr 缓存也没用。
- **解法**: `FLYDSL_EXTRA_SOURCE_DIRS=<kernel目录>` 强制重编;或 `FLYDSL_DUMP_IR=1` 旁路。
- **判据**: 正确性 maxdiff 怎么改都卡在**同一垃圾值不动** = 多半中缓存,先 bust 再判正确性。

---
来源: flydsl-sync/SKILL.md, pr-merge-gate/SKILL.md, project_mxfp4_epilogue_store.md
