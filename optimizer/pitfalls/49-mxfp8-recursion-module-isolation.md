# FlyDSL kernel 与 torch 同模块触发 JIT 依赖收集 RecursionError

> 类别: 踩过的坑 · 主题标签: JIT依赖收集, 模块隔离, RecursionError, mxfp8

- **症状**: FlyDSL kernel 文件和 torch/测试代码放同一模块 → JIT 依赖收集阶段触发 `RecursionError`。
  - **WHY**: FlyDSL 的 JIT 在编译 kernel 时会递归收集模块级依赖，同模块内的 torch import 让依赖图爆栈。
- **解法**: 把 kernel 放**独立干净模块**（如 `mxfp8_quant_flydsl.py` 只 `import flydsl`，**不** `import torch`），只让 harness 脚本 `import torch`。
  - kernel 模块保持纯净 → JIT 依赖收集不再递归到 torch。
- **区分维度**: 这是**模块隔离**维度的坑，区别于 `03-tracer`（tracer 字面 if/for）与 `35-encoding`（编码）。同为 JIT 相关但根因不同。

---
来源: mxfp8-8wave-devloop/SKILL.md
