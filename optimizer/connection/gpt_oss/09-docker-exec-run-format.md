# 远端容器内跑法:docker exec + HIP_VISIBLE_DEVICES

> 类别: 连接 · 主题标签: docker-exec, HIP_VISIBLE_DEVICES, flydsl-cache, venv

- **通用调用格式**:`./.ssh-chi.sh root@chiXXXX "docker exec -e HIP_VISIBLE_DEVICES=$G mlperf_gptoss bash -lc '...'"`。跳板脚本 `.ssh-chi.sh` 进节点 → `docker exec` 进容器 `mlperf_gptoss` → `bash -lc` 跑命令。
- **选卡**:用 `HIP_VISIBLE_DEVICES=$G`(或 `CUDA_VISIBLE_DEVICES`)环境变量指定 GPU,作为 `docker exec -e` 传入。
- **改 kernel 必清缓存**:每次改 kernel 后命令里必须 `rm -rf /root/.flydsl/cache` 清 JIT 缓存,否则跑旧编译产物。
- **venv 按分支选**:
  - `Primus-Turbo-tensorwise` 分支 → `/opt/venv-tw/bin/python -u <script>`
  - 生产 mxfp4 → `/opt/venv`(容器 `mlperf_gptoss` 内默认 venv);mxfp8 属于 gpt_oss2/mlperf_gptoss2/code3 的独立环境(容器/venv 均不同),别混用
  - 完整示例(tensorwise):`./.ssh-chi.sh root@chi2811 "docker exec -e HIP_VISIBLE_DEVICES=$G mlperf_gptoss bash -lc 'cd /workspace/code/Primus-Turbo-tensorwise && /opt/venv-tw/bin/python -u <script>'"`
  - 完整示例(primus_turbo + 清缓存):`docker exec -e HIP_VISIBLE_DEVICES=N -e <env> mlperf_gptoss bash -lc 'cd .../primus_turbo && rm -rf /root/.flydsl/cache && <env> python3 _ow.py'`
- **生产节点/环境**:
  - 当前活跃节点 `chi2811`(2026-07-03 从 chi2810 切换),容器 `mlperf_gptoss`(gfx950 / MI355X)。`chi2810` 已废弃(GPU 被别的 SGLang 服务全占,不可用),仅供参考,用前需按 claim-mi355x-node 重新核实。
  - 仓库 bind mount:宿主 `/mnt/vast/kyle/code2` → 容器内 `/workspace/code`(改宿主文件即容器内可见,无需 rebuild)。

---
来源: SKILL.md, 10-grouped-wgrad-4wave-3buf.md, 13-primus-turbo-prod.md
