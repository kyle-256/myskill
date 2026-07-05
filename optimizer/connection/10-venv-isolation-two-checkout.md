# mxfp4/tensorwise 两份 checkout 与 venv 隔离

> 类别: 连接 · 主题标签: venv隔离, PEP660-editable, checkout映射

- **两份 turbo，两个 venv，按分支选**：
  - mxfp4 分支 `dev/kyle/flydsl_mxfp4_compute` → `/opt/venv`（FlyDSL 也用这个）
  - tensorwise 分支 `dev/kyle/flydsl_grp_gemm` → `/opt/venv-tw`
  - 选错 venv 会跑错代码。跑前必须选对 venv 并验证 `import` 路径。
- **为什么两个 venv 真隔离**：`primus_turbo` 是 editable 安装（PEP 660）。每个 venv 的 `__editable___primus_turbo_0_0_0_finder.py` 里的 `MAPPING` dict 指向某个 repo 的 `primus_turbo`，两 venv 各指各的 repo → 独立。
- **clone venv 的污染坑**：直接 clone 出的 venv，其 `setuptools.pth` 会把源 venv 的 `site-packages` 追加到 `sys.path` → 新 venv fallback 到老 venv（不是真隔离）；`pip install -e` 时会把 editable 写进老 venv，污染另一份。
- **正确建第二份 venv（cp -a clone 后）**：
  1. `sed` 改 `setuptools.pth` 里 `/opt/venv` → `/opt/venv-tw`
  2. 验证 `sys.path` 不含 `/opt/venv`
  3. 再 `pip install -e`
- **如果忘了修复**：直接 `sed` 改两个 venv finder 的 `MAPPING` 路径即可（无需重 build）。

---
来源: remote-sync/SKILL.md, pr-merge-gate/SKILL.md
