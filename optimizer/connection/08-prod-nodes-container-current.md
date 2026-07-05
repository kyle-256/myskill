# 生产节点/容器/rocm 版本现状

> 类别: 连接 · 主题标签: prod-node, container, rocm-version, bind-mount

- **当前生产节点**: gfx950/MI355X 当前 = **chi2811**（历史 chi2774/chi2810/chi2832）。2026-06-30 起曾切到 chi2810（8 卡全空）。均为单卡稳态测。
- **容器**: 名 `mlperf_gptoss`（镜像 saved-20260625b），**rocm 7.2**；venv 在 `/opt/venv`（chi2774 亦有 /opt/venv）。
- **仓库挂载**: bind mount 宿主 `/mnt/vast/kyle/code2` → 容器内 `/workspace/code`（chi2810/chi2811/chi2774 一致）。
- **连接**: 经 `sync/.ssh-chi.sh` 跳板，如 `sync/.ssh-chi.sh root@chi2811`。
- **指定 GPU**: 用 `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` 环境变量。
- **运行示例**: `docker exec mlperf_gptoss bash -lc "cd /workspace/code/FlyDSL && HIP_VISIBLE_DEVICES=7 python turbo/test_vmono.py M N K"`。
- **WHY/证据**: 旧绝对 TFLOPS 多在被争用节点测，**仅 ratio 可信**，干净节点须重测。

---
来源: README.md, 09-perf-numbers.md, 13-primus-turbo-prod.md, agpr_phase5_mono.md
