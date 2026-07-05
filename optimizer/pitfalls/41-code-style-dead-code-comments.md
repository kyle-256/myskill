# 代码风格门：源码禁调试痕迹/benchmark 数字、探针脚本不 git add、ruff --fix 查 diff

> 类别: 踩过的坑 · 主题标签: 代码风格, merge-gate, ruff, 死代码

- **源码注释严禁调试/实验痕迹**：跑数结果(如 `58dB 正确`)、调参备忘、`# debug`/`# tmp`/`# 临时`、带日期的过程笔记——这些进 commit message / memory / PR，**不进源码**。放行前必须 `rg -i 'debug|tmp|临时|verified|实验|print\(|TODO'` 得 **0 命中**才放行。WHY：源码是长期资产，调试痕迹是过程噪声，会误导后续读者。
- **探针脚本绝不 git add**：`_g_*.py` / `_d_*.py` / `opt_*.py` 这类临时探针脚本，永远留本地，不入库。
- **tuning 中间态放 `tuning_results/`(项目内)，不写 memory**：每轮 winner / TFLOPS / cfg 属于易变数据，进 `tuning_results/`。memory 只放**不变的约定**——target 来源、`BK=128` 硬约束等。WHY：memory 是稳定索引，塞入易变调参结果会污染。
- **改动对整除-K 必须是编译期 no-op**：加 K-tail / 新分支等改动，对 `K%128==0` 的 shape 必须编译期 no-op，老 shape SNR 不变。回归=不放行。
- **merge-gate 死代码判定**：`留着作校验 harness 用` 若那个 harness 本身是**未入库探针**，则对 merge-ready 而言就是死代码，**必删**。实例：独立 preshuffle 启动链入库了代码但 0 消费者=死代码。
- **生产文件旋钮全 hardcode**：`FP4_*` 全部 hardcode 成生产值；删掉所有 `PT_MX_*` / `PT_` 实验 env、persistent 变体、以及注释里的 benchmark 数字(单形状胜负 / dB / pp / 具体 MxNxK)。

### ruff check --fix 陷阱
- ruff `--fix` 会删 **F401 未用 import** 和 **F841 简单未用变量**并排序 import。
- ❌ 别再试 无脑接受 `--fix` 结果：它可能删掉那些**本意保留**的变量(编译期决策标记 / 局部可读性变量)。自动修复后**必须查 diff**——若改的是行为而非仅格式/import 卫生，恢复行为逻辑再重跑 formatter。
- ruff **不删**未用函数/类，那些需人工审：autoflake/ruff F401/F841 抓不到未用函数/类/私有 helper，自己 `rg '<name>\b'` 全仓搜，**0 命中即删**(实例 `_get_fp4_dtype`，源自 gpt_oss2 pr-merge-gate/SKILL.md，非本环境)。
- **裸 TODO 处理**：无 issue 号的裸 `# TODO` 要么挂号(关联 issue)要么删，不留悬空。注释掉的旧代码/占位一律删。

### CI 报无关文件失败
- 本地风格检查全过，但 PR CI 仍报**无关文件**失败=PR 分支落后 main，**不是 formatter 的锅**。
- 解法：fetch `origin/main` → merge 进 PR 分支 → 重跑检查 → push merge commit。

---
来源: pr-merge-gate/SKILL.md, gpu-fleet-tuning/SKILL.md, 14-fused-preshuffle-e2e.md, format-code/SKILL.md
