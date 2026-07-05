# 基准 shape 定义：dense mxfp4 GEMM（Llama-2 7B/70B）+ grouped GEMM（MoE 三模型）

> 类别: 方法论 · 主题标签: benchmark-shapes, dense-gemm, grouped-gemm, moe, llama, deepseek, qwen, gpt-oss, canonical

**这是 mxfp4 / fp8 GEMM 跑分的 canonical shape 定义。写 bench harness / optimizer 判分口径时用这套，别自己另挑一组。** shape 来自 repo `benchmark/ops/training/config.py`（`DenseModelConfigs` / `MoEModelConfigs`），下表已展开成实际 N/K。

## Dense GEMM（Llama-2 7B + 70B，两个都要跑）

- 每模型 4 个 dense 层：`qkv_proj / attn_out / mlp_gate_up / mlp_down`；`trans_b=True`。
- **M = seq_token × mbs**，mbs ∈ {1, 2, 4}（dense bench 默认三档）。
- N/K 固定如下：

| 模型 | 层 | N | K |
|---|---|---|---|
| **Llama-2-7B**(hidden 4096) | qkv_proj | 12288 | 4096 |
| | attn_out | 4096 | 4096 |
| | mlp_gate_up | 22016 | 4096 |
| | mlp_down | 4096 | 11008 |
| **Llama-2-70B**(hidden 8192) | qkv_proj | 10240 | 8192 |
| | attn_out | 8192 | 8192 |
| | mlp_gate_up | 57344 | 8192 |
| | mlp_down | 8192 | **28672** ← K28672 天花板 shape（见 pitfalls/05） |

- ⚠️ **必须含 70B**：mxfp4 4-wave 的天花板/杠杆研究都在 70B 的 8192² + `mlp_down K=28672` 上（pitfalls/05、方法论各 mxfp4 卡）；只测 7B（K≤11008）会漏掉 KB 主战场。

## Grouped GEMM（MoE，三模型：gpt_oss-20b / Qwen3-235B / DeepSeek-V3）

- **B = 8**（8 个 expert = 8 组）；**per-group M ∈ {1024, 2048, 4096}**；`trans_b=True`。
- 每模型 2 个 GEMM：`GateUP=(N=2·moe_int, K=hidden)`、`Down=(N=hidden, K=moe_int)`。
- balanced 分布是默认，但**验正确性必须另跑 unbalanced（倾斜 token）**——balanced 常全过、over-run 污染只在 unbalanced 暴露（见 pitfalls/05「grouped over-run 污染」）。

| 模型 | hidden | moe_int | GateUP (N,K) | Down (N,K) |
|---|---|---|---|---|
| **gpt_oss-20b** | 2880 | 2880 | (5760, 2880) | (2880, 2880) |
| **Qwen3-235B-A22B** | 4096 | 4096 | (8192, 4096) | (4096, 4096) |
| **DeepSeek-V3** | 7168 | 2048 | (4096, 7168) | (7168, 2048) |

- group_lens：balanced = `torch.full((B,), M)`；unbalanced = `gen_grouped_gemm_group_lens(B, M, balance=False)`（0.2+0.8·rand 归一，末组补差）。

## 判分口径（harness / optimizer）

- 每 shape 用 repo 标准 `profile_gemm_fp4`（含 activation quant 的 fwd+bwd）取 `fwd_tf / bwd_tf`，SNR 门控（fp4 阈值 10）。
- 单一分数 = **combined step TFLOPS 的几何平均**：`Combined = 6/(2/fwd_tf + 4/bwd_tf)`，对所有 shape 取 geomean（见 methodology/01）。
- 噪声：DVFS ~2-3%，判分门槛 ≥2% 才算真收益（见 pitfalls/02）。

---
来源: benchmark/ops/training/config.py（DenseModelConfigs / MoEModelConfigs / gen_gemm_test_cases / gen_grouped_gemm_group_lens）, methodology/12-mxfp8-grouped.md（三模型 grouped shape 表）, 09-perf-numbers.md, mxfp4_4w_oddki_llama_aiter.md
