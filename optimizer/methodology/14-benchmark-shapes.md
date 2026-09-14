# 基准 shape 定义：dense mxfp4 GEMM（Llama-2 7B/70B）+ grouped GEMM（MoE 三模型）

> 类别: 方法论 · 主题标签: benchmark-shapes, dense-gemm, grouped-gemm, moe, llama, deepseek, qwen, gpt-oss, canonical

**这是 mxfp4 / fp8 GEMM 跑分的 canonical shape 定义。写 bench harness / optimizer 判分口径时用这套，别自己另挑一组。** shape 来自 repo `benchmark/ops/training/config.py`（`DenseModelConfigs` / `MoEModelConfigs`），下表已展开成实际 N/K。

## Dense GEMM（Llama-2 7B + 70B，两个都要跑）

- 每模型 4 个 dense 层：`qkv_proj / attn_out / mlp_gate_up / mlp_down`；`trans_b=True`。
- **M ∈ {4096, 8192}**（dense bench 用这两档 token 数；= seq_token × mbs 的常用组合，用户 2026-07-20 明确口径）。
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

### ★★ 单模型 bench = 给 overfit 开门（2026-08-12 mxfp4 m4096 场实证，代价是一整场）

本卡上面那张三模型表**是硬要求，不是参考**。我建 mxfp4 UNBALANCE 场的 bench 时照着 syncv3 的
单模型探针改，把 `KP=2944 / N=5760,2880` 写死，18 个计分格**全是同一个 H/I** —— 结果整场 20 轮的
判据都在往 gpt-oss 的形状上贴，而且贴得毫无察觉，因为 bench 里没有任何东西会因此掉分。

事后拿 campaign 前后两个版本在三个几何上对测（同 token 预算 G=32/Mtot=131072、同 canon skew）：

| 模型 (N/K, gate_up→down) | fwd | dgrad | wgrad |
|---|---|---|---|
| gpt_oss (5760,2944 → 2944,2944) | **+3.8% / +4.0%** | −0.6% / **+4.3%** | **+7.6% / +10.9%** |
| DeepSeek-V3 (4096,7168 → 7168,2048) | +0.4% / +0.9% | +0.2% / +1.2% | **+5.8% / +3.1%** |
| Qwen3-235B* (3072,4096 → 4096,1536) | −0.5% / −0.7% | +0.1% / −1.3% | **+6.0% / +0.7%** |

- **好消息**：没有一个格子被改坏（最差 −1.3%，在噪声带内）。
- **wgrad 那笔是真泛化的**（六格全正）——它抓的是「短收缩下 C 字节相对 FLOP 过大」这个与形状无关的机制。
- **fwd/dgrad 的收益是 gpt-oss 专属**：+3.8~4.3% 到了另外两个模型只剩 +0.2~1.2%、甚至微负。
  根因是它的门 `mb >= num_xcds*xcd_span`（span=16）是按 gpt-oss 的 tile 数调的，换个 H/I 就落到另一侧 —— **不是有害，是没生效**。
⚠ **`*` Qwen 的 `moe_int` 本卡上表写 4096、syncv3 的模型探针
(`CURRENT_gptoss_m4096_wgrad/_kyle_g32_unbal.py`)写 **1536**,两处不一致 —— 上面这组数据是按
**1536** 测的。用之前先去 `benchmark/ops/training/config.py` 的 `MoEModelConfigs` 核一次真值,
两个 I 差 2.7 倍,K 也跟着差,结论不能混用。

- 还顺带暴露了一个单模型 bench 永远看不见的洞：**fwd 在短 K 上塌**（Qwen down K=1536 只有 2555 TF、
  DeepSeek down K=2048 是 3228，而 gpt-oss K=2944 有 3773）。gpt-oss 的 K 恒等于 2944，这个洞在它的
  bench 里不存在。

⇒ **规矩**：MoE grouped GEMM 的 campaign bench **必须模型表驱动**（三个几何 × 分布档），
score 取**跨模型的几何平均**。这样任何往单一形状贴的判据都会立刻在别的模型上掉分，
应试在计分层面就藏不住；单模型 bench 则是把 overfit 的门敞开、还不给任何告警。

## 判分口径（harness / optimizer）

- 每 shape 用 repo 标准 `profile_gemm_fp4`（含 activation quant 的 fwd+bwd）取 `fwd_tf / bwd_tf`，SNR 门控（fp4 阈值 10）。
- 单一分数 = **combined step TFLOPS 的几何平均**：`Combined = 6/(2/fwd_tf + 4/bwd_tf)`，对所有 shape 取 geomean（见 methodology/01）。
- 噪声：DVFS ~2-3%，判分门槛 ≥2% 才算真收益（见 pitfalls/02）。

## ★★★★★ 部署形状集 ⊋ 计分形状集 ⇒ 差集里的代码等于没测过（2026-09-12 GLM-5.2 MoE，代价一整天）

bench 只有 decode 形状 `M∈{8,16,64,128}`，而排序路径 `launch_4k_fused` **只在 M>2048 触发**
⇒ 那条路径**整场 campaign 一次都没执行过**。后果：

- 一个未定义变量 `is_local_expert`（r5 引入、r11 删除）**在树里活了四个 commit**，
  中间任一 commit 部署就是 `NameError`
- 一处越界写把线上 server 打崩，排查花掉大半天

口径差异同样算：bench 用 `num_experts=258`（多一个 fake slot），server 实际是 **257**。

### 修法：加一道**只判通过、没有基线**的覆盖门（关键是别动计分）

改计分口径会让冻结基线失效、历史分数失去可比性（这场的 +13.1% 全靠 `BASELINE_US` 冻结
才有意义）。所以覆盖门必须和计分**分开**：

```python
COVERAGE_M = (3072, 7144)        # 计分形状够不到的尺寸
COVERAGE_EP_IDS = (0, 3)         # ep_id=0 的 mask 前缀和是平凡情况，必须带一个非 0
SERVER_TOTAL_E = ROUTED_E + SHARED_E   # 257，server 口径

def coverage_gate():
    """返回 problem 列表；空=通过。无基线 ⇒ 以后加形状不必重测基线。"""
```

三条设计要点：

1. 只判 raise / 非有限值，**不产生分数** ⇒ 可以随时扩展
2. 配一个 `CFM_SKIP_COVERAGE=1`，**只用于给已知会在这些形状上崩的臂取分做配对对比**；
   判分的臂绝不允许设
3. 被测路径若有"只在大尺寸才走"的门槛，**给门槛加个 env 覆盖**
   （本例 `AITER_P23_FUSED_MAX_T`），让小 M 也能走到那条路径，覆盖门就不必跑昂贵的大形状

### 配套：验证的输入必须能到达被验证的分支

同期 inference2 踩的同型错：改 `MAX_ROWS` 48→96 后做正确性对拍，用的是**单请求**，
而 verify 行数 = 6 × 并发 ⇒ 6 行，两种配置都 ≤48、都走融合路径 ——
**那个对拍是拿一条路径和它自己比**。必须构造到 84 行（14 路并发）才真正跨过分支。
**"输入覆盖到了分支"这件事不会自己成立，要专门构造并用 trace 确认。**

---
来源: benchmark/ops/training/config.py（DenseModelConfigs / MoEModelConfigs / gen_gemm_test_cases / gen_grouped_gemm_group_lens）, methodology/12-mxfp8-grouped.md（三模型 grouped shape 表）, 09-perf-numbers.md, mxfp4_4w_oddki_llama_aiter.md
