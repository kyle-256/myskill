# gpt_oss2 两份 Primus-Turbo checkout 与 venv 隔离

> 类别: 连接 (gpt_oss2) · 主题标签: venv-隔离, editable-install, PEP660, primus-turbo

- **两份并存的 Primus-Turbo checkout**（都是 git 仓库，本地为 canonical）：
  - **mxfp8** → `sync/mxfp8/Primus-Turbo`，分支 `dev/kyle/flydsl_mxfp8_compute`（2026-06-25 @ `dac31090`）；quant 专题分支为 `dev/kyle/flydsl_mxfp8_quant`。
  - **mxfp4** → `sync/mxfp4/Primus-Turbo`，分支 `dev/kyle/mxfp4_triton_gg`（旧名 `triton_gg` 已废弃）。
- **各自独立 venv**（选错 venv 会 import 错 repo）：
  - mxfp8 用 `/opt/venv`；另建 symlink `/opt/venv-mxfp8` → `/opt/venv`（对称命名）。
  - mxfp4 用 `/opt/venv-mxfp4`。
- **editable(PEP 660) 安装机制**：每个 venv 的 `site-packages/__editable___primus_turbo_0_0_0_finder.py` 里的 `MAPPING` dict 指向某 repo 的 `primus_turbo`，共 3 条映射：`libprimus_turbo_kernels` / `primus_turbo` / `tools`。两 venv 各指各 repo → 实现隔离。WHY：editable finder 用 MAPPING 把 import 路由到源码树，改路径即改所指 repo，无需重装。
- **各 repo 自 build 自己的 .so**：`primus_turbo/lib/libprimus_turbo_kernels.so` 在 vast 上各自编译（rsync 排除 `*.so`，故本地不同步二进制）。
- **验证隔离走对 repo**：
  - `/opt/venv/bin/python -c "import primus_turbo; print(primus_turbo.__file__)"` 应指 `…/mxfp8/…`
  - `/opt/venv-mxfp4/bin/python` 同命令应指 `…/mxfp4/…`

（mirrors gpt_oss 的 10-venv-isolation。）

---
来源: remote-sync/SKILL.md, project_remote_sync_setup.md
