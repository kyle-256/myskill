# triton 3.6→3.7 升级与 baseline 作废

> 类别: 连接 · 主题标签: triton, fp8-grouped, autotune, baseline

- **rocm/primus:v26.2 镜像默认 triton 3.6.0**;fp8 grouped GEMM bench 要求 **triton 3.7**,WHY:3.7 的 autotune cfg 更全(覆盖更多候选配置)。
- 升级命令:`/opt/venv/bin/pip install --upgrade triton`。
- **作废规则**:装完 3.7 后,所有用 triton 3.6 跑出的 bench 数字全部作废,必须重 bench 立新 baseline——不同 autotune cfg 集会选出不同 winner,3.6 与 3.7 的数字不可直接比较。

---
来源: SKILL.md
