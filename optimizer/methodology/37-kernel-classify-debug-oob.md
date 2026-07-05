# 内核分类骨架、错误隔离 debug 流程、OOB 静态区间分析

> 类别: 方法论 · 主题标签: kernel-classify, error-isolation, oob-detection, correctness-verify

## 1. 按 5 类模式选骨架

5类骨架分类表见 methodology/15-flydsl-authoring-spine-layout.md

## 2. debug 症状分类(先看错哪、错多少)

- **全 NaN** → softmax `-inf-(-inf)` 或 `1/0`(用 `.select` 加 guard);或未初始化 buffer。
- **全 0** → 输出地址/stride 错、partition slot 错、buffer 未写。用 sentinel `-999.0` 初始化即可暴露被跳过的写。
- **>50% 错** → partition 数错 / layout 不匹配 / 寻址错。
- **1-5% 小错** → FP8 requant 容差 / scale 被重复施加 / off-by-one mask。可能不是 bug。
- **编译错** → 类型 / loop-carried state / `range` vs `range_constexpr`。

## 3. 系统 debug 流程(最便宜的先做)

1. **无条件清缓存**:`rm -rf ~/.flydsl /tmp/flydsl*`,清 lru_cache。绝大多数"改了没用"都是 stale cache。
2. **全 1 输入**(Q/K/V=1.0)过了 → layout OK,是数据问题。
3. **单 partition**(one_shot)过了 → 多 partition / reduce 的 bug。
4. host 端打印 shape / stride / NaN。
5. 对比中间 buffer(`exp_sums` / `max_logits` / `temp_out`)。
6. 怀疑 layout 时**手动追一个线程地址**:tid=0(lane16id=0, rowid=0, warp_id=0)。
7. 验证 MFMA operand order 和 `range` vs `range_constexpr`。

**合成输入的坑**:Q/K/V=1.0 使 softmax 均匀、PV 塌成已知值,任何偏差都是纯 layout/寻址 bug——**但均匀输入不暴露 swapped V/P MMA operand**,这类 bug 必须对 reference 交叉验证。

## 4. OOB 四类边界

| 类型 | 检出手段 |
|---|---|
| 物理 allocation OOB | HIP illegal address 能测 |
| 逻辑对象 OOB(跨 row/head/tile) | 静态区间分析(物理工具测不出,仍在同一 allocation 内) |
| lane/thread 所有权 OOB(读别 lane 槽) | 静态分析 + printf |
| LDS/共享内存 OOB | 静态分析 |

**静态区间分析法**:每个访存写 `start=base+offset`、`end=start+vec_width-1`、`legal=[obj_base, obj_base+extent-1]`,代入 lane / warp_id / `range_constexpr` 的已知范围:
- `max(end) > legal_end` → 上界 OOB。
- `min(start) < obj_base` → 下界 OOB。

运行时检查用**窄范围** `fx.printf` 只打失败坐标,避免太多 lane 打印淹没信号。

## 5. 新内核正确性验证清单

- 检查 tid 前先 `torch.cuda.synchronize()`。
- `torch.allclose(atol=1e-5)` 对比参考。
- 确认 copy atom 宽度合法(完整规则见 pitfalls/04-flydsl-frontend-authoring-traps.md)。
- 编译期常量用 `Constexpr[int]`,运行时值用 `Int32`。
- GEMM tile size 必须匹配 MFMA 指令形状。
- 加新 atom 后:**先跑 FileCheck,再跑 1-wave 端到端 Python kernel**,才能信 layout。

---
来源: flydsl-tile-programming/SKILL.md, debug-flydsl-kernel/SKILL.md, programming-model.md, oob-detection/SKILL.md, add-target-atom-op/SKILL.md（"先跑 FileCheck 再跑 1-wave 端到端"一条出自此文件 Step 9）
