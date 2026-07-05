# attention 正确性坑：online-softmax rescale/NaN 守卫、空 partition 写中性值、全1测局限

> 类别: 踩过的坑 · 主题标签: attention, online-softmax, correctness, mfma

## online-softmax NaN / 除零守卫
- **全 mask NaN 根因**：某 partition 所有 token 都被 mask（超出 context）时 `qk_max=-inf`，则 `exp(s-qk_max)=exp(-inf-(-inf))=exp(NaN)=NaN`；`exp2(-inf - -inf)` 同样 NaN，会污染整个循环剩余的 l/O。修法：`safe_diff = (qk_max > NEG_INF).select(diff, ZERO_F)`；并对 corr 和 partition rescale 都做 `select`-guard 防 `m==-inf`。
- **归一化除零**：`exp_sum=0` 时 `1/exp_sum=inf`。修法：`safe_sum = (running_sum > ZERO_F).select(running_sum, fx.Float32(1.0)); inv_sum = 1.0/safe_sum`。
- **rescale 顺序**：每个 block 必须在 PV MFMA **之前** 用 corr 对 `O_acc` 做 rescale，不是之后。重构时极易漏掉，漏了直接结果错。
- ❌ **别再试 错 peer-reduction XOR mask**：wave64 MFMA32 的 lane-reduce mask（如 `lane^32`）绑定 tile 几何，mask 写错会在错误的 lane group 上 reduce，静默地污染 softmax 统计量（不报错，结果错）。

## 空 partition / decode block-split 必写中性值
- decode 按 block 拆 partition 时，空 / 越窗 partition **必须仍写中性值**：`sum=0, max=-inf, out=0`。否则 cross-partition reduce kernel 读到未初始化的槽。
- fused 单 partition 快路径**只有 single-visible-tile 前提成立时才安全**，用编译期开关 gate 住，不能默认走。

## 输出全错的地址/网格根因
- **全 0 输出**：输出地址错（`stride_out_seq`/`stride_out_part` 错，打印 `output.stride()` 核对）；multi-partition 必须写到 `part_z` 槽而非绝对 partition index（reduce kernel 从 `part_z=0..grid_z-1` 读）；主 kernel 没写 `exp_sums`/`max_logits` 则 reduce 出 0（启动前 `fill_(-999.0)` sentinel 验证）。
- **>50% 错**：常因 `grid_z < total_partitions` 且 kernel 每 CTA 只处理一个 partition（无循环），大部分 context 被跳过。验证 `total_parts=ceil(context_len/KV_COMPUTE_BLOCK)`，assert `grid_z==total_parts` 或有 multi-partition 循环。

## 测试局限与 MFMA 操作数
- **全 1 隔离测试**：`query/key/value.fill_(1.0)` 下 softmax 概率全相等、PV 输出=1.0，任何偏差暴露 layout/寻址 bug。❌ **别只靠全 1 测**：均匀值无论顺序都对，测不出 V/P 操作数错位，需另测非均匀输入。
- **MFMA 操作数顺序**：`mfma(LHS, RHS, acc)` —— LHS→M 维，RHS→N 维。QK 中 K 是 LHS、Q 是 RHS。搞错就是 layout 大错。
- **FP8 容差**：FP8 PV MFMA 相对 bf16 参考引入 ~0.03 max error，是 FP8 数据通路固有的、不是 bug，容差用 `atol=5e-3`。per-row vs per-tensor Q 量化不匹配会有 1-3% 差异。常见 scale bug：`v_scale` 被应用两次（prob scaling 一次、PV 后一次）。

---
来源: debug-flydsl-kernel/SKILL.md, attention/optimization-directions.md
