# 远端运行方式与 venv 隔离

> 类别: 连接 · 主题标签: docker-exec, HIP_VISIBLE_DEVICES, flydsl-cache, venv, venv隔离, PEP660-editable, checkout映射

> ★★ syncv3(wgrad skew)三-repo/三-venv 环境 + `source activate` 串环境血泪坑 → 见 **04-syncv3-env-pythonpath.md**。规程一句话:**直接 `/opt/venv-<x>/bin/python`,严禁 source activate,跑前先 `import primus_turbo;print(__file__)` 验证 repo**。

## 远端容器内跑法:docker exec + HIP_VISIBLE_DEVICES

- **通用调用格式**:`./.ssh-chi.sh root@chiXXXX "docker exec -e HIP_VISIBLE_DEVICES=$G mlperf_gptoss bash -lc '...'"`。跳板脚本 `.ssh-chi.sh` 进节点 → `docker exec` 进容器 `mlperf_gptoss` → `bash -lc` 跑命令。
- **选卡**:用 `HIP_VISIBLE_DEVICES=$G`(或 `CUDA_VISIBLE_DEVICES`)环境变量指定 GPU,作为 `docker exec -e` 传入。
- **改 kernel 必清缓存**:每次改 kernel 后命令里必须 `rm -rf /root/.flydsl/cache` 清 JIT 缓存,否则跑旧编译产物。
- **venv 按分支选**:
  - `Primus-Turbo-tensorwise` 分支 → `/opt/venv-tw/bin/python -u <script>`
  - 生产 mxfp4 → `/opt/venv`(容器 `mlperf_gptoss` 内默认 venv);mxfp8 属于 gpt_oss2/mlperf_gptoss2/code3 的独立环境(容器/venv 均不同),别混用
  - 完整示例(tensorwise):`./.ssh-chi.sh root@chi2762 "docker exec -e HIP_VISIBLE_DEVICES=$G mlperf_gptoss bash -lc 'cd /workspace/code/Primus-Turbo-tensorwise && /opt/venv-tw/bin/python -u <script>'"`
  - 完整示例(primus_turbo + 清缓存):`docker exec -e HIP_VISIBLE_DEVICES=N -e <env> mlperf_gptoss bash -lc 'cd .../primus_turbo && rm -rf /root/.flydsl/cache && <env> python3 _ow.py'`
- **生产节点/环境**:
  - 当前活跃节点 `chi2798`(2026-07-14,取代 chi2762;容器 `mlperf_gptoss`,gfx950/MI355X,repo 在容器内 `/workspace/code/Primus-Turbo`,分支 dev/kyle/flydsl_attn_deepseekv4)。历史节点 chi2762/chi2811/chi2810/chi2832 用前需按 02-pick-free-gpu 重新核实占用。
  - 仓库 bind mount:宿主 `/mnt/vast/kyle/code2` → 容器内 `/workspace/code`(改宿主文件即容器内可见,无需 rebuild)。

## mxfp4/tensorwise 两份 checkout 与 venv 隔离

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
来源: SKILL.md, 10-grouped-wgrad-4wave-3buf.md, 13-primus-turbo-prod.md, remote-sync/SKILL.md, pr-merge-gate/SKILL.md
