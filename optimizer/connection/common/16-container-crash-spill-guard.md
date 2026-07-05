# 大 spill 崩容器与 dump ISA 防护

> 类别: 连接 · 主题标签: spill, dump-ISA, container-crash, sys.path

- **大 spill 配置直接崩容器 (exit 137)**：上 GPU 跑前先用 `FLYDSL_DUMP_IR=1` dump ISA，检查 `vgpr_spill_count`，避免高 spill 配置直接把容器 OOM 搞崩（exit 137）。先看 IR 再上 GPU。
- **探针脚本放包内须剔 `_SELF_DIR`**：探针脚本放在包内（`primus_turbo/`）时，首行必须把 `_SELF_DIR` 从 `sys.path` 剔出去。WHY：否则 `import primus_turbo` 会命中脚本所在目录，造成循环 import / 导入错包。

---
来源: flydsl-fp8-gemm-tuning/SKILL.md, flydsl-fp8-gemm-results/SKILL.md
