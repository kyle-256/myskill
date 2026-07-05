# gpt_oss2 远端容器内跑法:docker exec + HIP_VISIBLE_DEVICES

> 类别: 连接 (gpt_oss2) · 主题标签: docker-exec, HIP_VISIBLE_DEVICES, flydsl-cache, build

- **GPU 运行格式**（跳板 → chi2811 → 容器内)：
  ```
  ./.ssh-chi.sh root@chi2811 "docker exec -e HIP_VISIBLE_DEVICES=4 mlperf_gptoss2 bash -lc 'cd /workspace/code/mxfp8/Primus-Turbo && /opt/venv/bin/python <script>'"
  ```
  - `-e HIP_VISIBLE_DEVICES=4`：指定容器内可见 GPU（WHY:多卡机器上锁定单卡跑，避免占用他人卡/结果串扰）。
  - 容器名固定 `mlperf_gptoss2`。
- **两套树 + 两套 venv**（WHY:mxfp8 与 mxfp4 依赖/环境隔离，不可混用）：
  - **mxfp8**：树 `/workspace/code/mxfp8/Primus-Turbo`，解释器 `/opt/venv/bin/python`。
  - **mxfp4**：树 `/workspace/code/mxfp4/Primus-Turbo`（即 mxfp4 树），解释器 `/opt/venv-mxfp4/bin/python`。
- **改 kernel 必清 flydsl 缓存**：改 kernel 后远端必须 `rm -rf /root/.flydsl/cache` 再跑（WHY:否则加载旧 .so，测的是旧内核）。
- **改 csrc/triton 必重 build**：改了 csrc/triton kernel 必须远端重 build（`setup.py build_ext --inplace`）后再跑（WHY:否则测的是旧 .so）。
- 镜像 gpt_oss 的 `09-docker-exec-run-format`（同一套 docker exec + HIP_VISIBLE_DEVICES 跑法）。

---
来源: remote-sync/SKILL.md, flydsl-fp8-gemm-tuning/SKILL.md, pr-merge-gate/SKILL.md
