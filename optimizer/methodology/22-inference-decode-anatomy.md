# 22 — 推理 decode 一步由什么构成：时间去哪了、有哪些杠杆、哪些已被定价

> 适用：优化 MoE + MLA 类模型（GLM-5.2、DeepSeek 族）的 **decode 侧**吞吐/延迟。
> 先读这张卡建立"一步里钱花在哪"的模型，再去 methodology/21 决定怎么量、pitfalls/15 避坑。
> 数据来自 GLM-5.2 TP4/EP4 MI355X 实测，换模型/并发要重测，但**结构和方法可复用**。

## ★★★★★★ 先决：trace 必须按时间切开 decode 与 prefill

一次采样里两段**完全不重叠**，不切开算出的分块是废的。

**判据 = kernel 名字，不是时间邻近**：decode 走 MXFP4 的 `mfma_moe*_afp4_wfp4`，
prefill 走 bf16 的 `ck kernel_moe_gemm<...DF16b...>`。混着算会得出
"GEMM 41.7%、hipBLASLt 13.4%"这种**对 decode 完全不成立**的结论。

★ 同类翻车：`_gluon_fp8_mqa_logits` 按"时间上离最近的 MoE marker"归到 decode，
打点后发现 22 次**全在 prefill**。⇒ **kernel 归段必须有第二条独立路径验证。**

## decode 一步的实测构成（TP4/EP4，CONC=10，14.4 ms GPU 忙，1985 个 kernel）

⚠️ **逐行分三类用，别整表当真**——另一条线照这张表建 minibench 复现不了 MoE 那行。

| 分块 | us | 占比 | 状态 |
|---|---|---|---|
| **MoE** | 4587 | 31.9% | ❌ **作废**：内部自相矛盾（75层×2×(28.2+16.1)=3323us ≠ 4587us，450 个 kernel 也凑不出 75层×2）⇒ 这桶里混了 routing/permute/量化的碎 kernel |
| **GEMM (a16w16)** | 3022 | **21.0%** | ✅ **可用**：minibench + ITL 反推两条独立路径互证过 |
| 通信 | 2742 | 19.1% | ⚠️ 未验证 |
| attention | 1406 | 9.8% | ⚠️ 未验证 |
| quant/cast | 1209 | 8.4% | ⚠️ 未验证 |
| norm/rope | 503 | 3.5% | ⚠️ 未验证（已 fused_qk_norm_rope）|

★ **默认该是「未验证」而不是「可用」**：没被第二条路径验证过的行都存疑。
拿它当分母去算"我这笔能提升 X%"之前，先把那一行自己量出来（见 methodology/20）。

## 三条固定开销约束（它们决定了哪些方向根本没有空间）

1. **dispatch 有地板**：约 28–35 us/步，和负载无关。低并发下它占比最高——
   **C=4 这类低并发档最弱的不是权重流量，是每步固定开销**。
2. **GEMM 的 85–96% 与 M 无关**：decode 的 M 很小，时间主要不是算出来的。
3. **NCCL 小消息全是延迟**：通信那一块靠"缩小载荷"而不是"加速 collective"——
   实测 logits TP all-gather 占 ITL 7.9–10.8%，**约 90% 的收益来自减少要传的东西**。

## 已被实测定价的杠杆（别重新发明）

| 杠杆 | 实测 | 备注 |
|---|---|---|
| **MoE 权重从 bf16 转 MXFP4**（draft/MTP 层是 dense bf16 的典型情形） | AgentX c8 **+6.09%** | 主干本来就是 MXFP4，只有 draft 那层是 bf16；落在 draft 侧 ⇒ 输出保序 |
| **融合 DSA indexer**（四个 gfx950 kernel） | ITL p50 **7.10 → 5.43 ms** | 上游 #38583；改的是 target 侧选 key ⇒ 两臂输出会分叉 |
| **MoE sorter 重写**（P0 计数折进 mesh 构建） | **+1.5pp** | 8.96 → 5.00 us/次排序 |
| **sparse-MLA flydsl 调度** | +1.9 ~ 6.2% | 比想象中小得多——我一度以为主力在这 |
| **EAGLE draft 融合采样** | 图内 119→13.7 us/次（7~9×） | 但上游对同类改动要求先给精度数字（#38340 被打回并 revert） |
| **`speculative-num-steps` 5→4** | draft 循环占一步 20–27% | 整块定价用 |
| MoE 本身还能压多少 | ❌ **纯带宽受限，verify 已达只读顶 90~92%** | 经八项证伪；别再往"让 MoE 算得更快"上投 |

## ★★★★★★ 小 M 的 GEMV：时间跟 launch 次数走，不跟行数走

decode 侧大量算子是 M 很小的 GEMV，**每次 launch 都要把整套权重重读一遍**。
主机侧按行分块调用 = 把权重流量乘以块数。

**一眼判据：`us/launch` 恒定，总时间与 launch 次数成正比而与行数无关。**

实例（融合 DSA indexer 的 section A，权重对 18.74 MB）：
8/16/24/32/48 行 → 7.59/7.52/7.37/7.44/7.38 us **每次 launch**，总时间线性涨。

**修法优先级**：① 把行循环挪进 kernel、常量（权重）提到循环外常驻寄存器
② 再谈抬块大小——**抬块大小会撞寄存器**（实测 M=8/12/16 的单次成本 7.81/9.82/13.44 us，
M=16 用 1.72 倍成本换 2 倍行数，净收益被吃掉）。

行循环内移后：48 行 53.21 → 32.93 us（**1.62×**），且 rows=1/5/8/12/16/17/24/32/47/48
全部 `torch.equal` 逐位一致。★ 详见 `[[reference_gemv_launch_count_not_rows]]`。

★ **但要清醒**：这个 op 在一步里只占约 5% ⇒ e2e 上限约 −2%，而 agentic replay 的
噪声底是 ~3%（methodology/21）⇒ **孤立 1.62× 在 e2e 上大概率读不出来**。
孤立定价和 e2e 定价要分开报，别把前者当成后者。

## 投机解码（EAGLE）的几个结构事实

* **verify 行数 = 6 × 并发**（num_draft_tokens=6）。这既是性能形状，也是**口径的一眼判据**。
* **draft 提议 / target 验证** ⇒ 改 draft 侧（权重精度、采样融合）**不改输出**，只改接受率；
  改 target 侧（attention 选 key、主干权重）**会改输出**。这条决定了两臂能不能比。
* **`SGLANG_SIMULATE_ACC_*` 把接受率钉成模拟值** ⇒ 性能口径可以用，
  **精度口径绝对不能用**（GSM8K 实测 0.111 vs 0.930）。
* draft 层常常是**整个模型里唯一的 dense bf16 MoE**，它每步搬的权重字节是 MXFP4 邻居的 4 倍
  ⇒ 这是个反复出现的优化点。

## 相关

* methodology/21（怎么量：口径、可比性、守门量）· pitfalls/15（harness 的七个静默错）
* methodology/19（kernel 级部署打分）· methodology/20（先把分母量出来）
* `[[reference_glm52_decode_step_breakdown]]`（原始 trace 与逐行状态）
* `[[reference_glm52_moe_bandwidth_bound]]`（MoE 只读顶的八项证伪）
* `[[reference_gemv_launch_count_not_rows]]`（GEMV launch 次数那条的完整证据）

来源：2026-09-07 trace + 2026-09-11~09-19 多场 GLM-5.2 TP4/EP4 实测。
