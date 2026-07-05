# gpt_oss2 bind mount 与 host↔容器路径映射

> 类别: 连接 (gpt_oss2) · 主题标签: bind-mount, 路径映射, code3, venv-磁盘

## bind mount 规则
- host bind mount：`/mnt/vast/kyle/code3` → 容器 `/workspace/code`。
- 换算：host 路径去掉前缀 `/mnt/vast/kyle/code3` 换成 `/workspace/code` 即得容器路径（反之亦然）。
- **源/目标一律用 `code3`**：别人的 mlperf_gptoss 挂在 `code2`，勿碰（避免误改他人环境）。

## 两份 turbo host/容器路径
| 分支 | host 路径 | 容器内路径 |
|---|---|---|
| mxfp8 | `/mnt/vast/kyle/code3/mxfp8/Primus-Turbo` | `/workspace/code/mxfp8/Primus-Turbo` |
| mxfp4 | `/mnt/vast/kyle/code3/mxfp4/Primus-Turbo` | `/workspace/code/mxfp4/Primus-Turbo` |

## 磁盘位置（关键：为何别往根盘写）
- `code3` 在 vast 大盘：充裕，含 `.so` build 产物 → 编译产物放这里安全。
- `/opt/venv*` 在容器 overlay 根盘：紧，常年 **97% / 31G** → 别往这里堆东西，会爆盘。

---
来源: remote-sync/SKILL.md, project_remote_sync_setup.md
