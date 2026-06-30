# 性能数据

节点 chi2811/chi2832/chi2774（历史；2026-06-30 起当前 = **chi2810**，8 卡当前全空），gfx950 / MI355X，容器 mlperf_gptoss，单卡稳态。
⚠️ 旧绝对 TFLOPS 多在被争用的节点测的，仅 ratio 趋势可信；干净节点重测见 [[project-flydsl-fwd-opt-baseline]]。

## vs Triton（kernel 级，B=8 balanced，12 MoE shape geomean）

测于 2026-06-16，branch `dev/kyle/flydsl_grp_gemm`（commit e94a30c9）。

| op | FlyDSL | Triton | fly/tri |
|---|---|---|---|
| fwd | 2397 | 2021 | **1.19×** |
| dgrad | 2307 | 2030 | **1.14×** |
| wgrad | 2163 | 1132 | **1.91×** |

### fwd 逐 shape（kernel TF）

| shape | M | FlyDSL | Triton | fly/tri |
|---|---|---|---|---|
| deepseek-down | 2048 | 2365 | 1860 | 1.27× |
| deepseek-down | 4096 | 2317 | 1847 | 1.25× |
| qwen235b-down | 2048 | 2118 | 1847 | 1.15× |
| qwen235b-down | 4096 | 2157 | 1768 | 1.22× |
| deepseek-up | 2048 | 2758 | 2435 | 1.13× |
| deepseek-up | 4096 | 2812 | 2499 | 1.13× |
| qwen235b-up | 2048 | 2716 | 2323 | 1.17× |
| qwen235b-up | 4096 | 2694 | 2243 | 1.20× |
| gpt_oss-up | 2048 | 2210 | 1899 | 1.16× |
| gpt_oss-up | 4096 | 2268 | 1933 | 1.17× |
| gpt_oss-down | 2048 | 2196 | 1898 | 1.16× |
| gpt_oss-down | 4096 | 2302 | 1877 | 1.23× |

### wgrad 逐 shape

| shape | M | FlyDSL | Triton | fly/tri |
|---|---|---|---|---|
| deepseek-up | 2048 | 2123 | 1154 | 1.84× |
| deepseek-up | 4096 | 2498 | 1265 | 1.97× |
| deepseek-down | 2048 | 2159 | 1139 | 1.90× |
| deepseek-down | 4096 | 2424 | 1250 | 1.94× |
| qwen235b-up | 2048 | 2175 | 1150 | 1.89× |
| qwen235b-up | 4096 | 2419 | 1249 | 1.94× |
| qwen235b-down | 2048 | 1803 | 1156 | 1.56× |
| qwen235b-down | 4096 | 2492 | 1257 | 1.98× |
| gpt_oss-up | 2048 | 2000 | 1009 | 1.98× |
| gpt_oss-up | 4096 | 2134 | 1107 | 1.93× |
| gpt_oss-down | 2048 | 1789 | 917 | 1.95× |
| gpt_oss-down | 4096 | 2098 | 994 | 2.11× |

## vs GB200 / TE reference（op 级，288 case，8-way bench ~10% throttle）

测于 2026-06-16，`grouped_gemm_te_fp8_tensorwise_20260330_GB200.csv`。

| 指标 | 值 |
|---|---|
| fwd geomean fly/GB200 | **1.80×** |
| bwd geomean fly/GB200 | **1.36×** |
| fwd 平均（fly/GB200）| 1294 TF / 1000 TF |
| bwd 平均 | 1353 TF / 1232 TF |
| 0 ERROR/FAIL | 286/288 PASS（int64 解锁所有 shape） |

### per-model

| 模型 | fwd fly/GB200 | bwd fly/GB200 |
|---|---|---|
| DeepSeek-V2-Lite | 3.68× | 2.22× |
| Qwen3-30B-A3B | 3.13× | 2.08× |
| DeepSeek-V2 | 2.11× | 1.54× |
| Qwen3-235B | 1.54× | 1.21× |
| Kimi-K2 | 1.42× | 1.08× |
| MoE-1T | 1.40× | 1.12× |
| DeepSeek-V3 | 1.42× | 1.12× |
| Mixtral-8x7B | 1.13× | 1.06× |
| Mixtral-8x22B | 1.00× | 0.95× |
| Grok-2 | 0.99× | 0.94× |

Grok-2 / Mixtral-8x22B fwd ≈ 1.0x：这两个模型的 bwd worst case = B=1 小-M 极大-N GateUP（Grok-2 N=32768），是 op-level small-op / memory-bound regime，GEMM kernel 再快也被 quant 和固定开销淹没。

## wgrad skew 鲁棒性（balanced vs 30:1 skew）

| | balanced | 30:1 skew |
|---|---|---|
| 修前（group-contiguous） | 2163 | 1162（−46%）|
| **修后（band-cyclic）** | **2143**（−0.9%）| **1592**（−26%）|

vs Triton skew：1.18×→**1.62×**（修后）。

## dgrad 小-M 提升（bm128 M-branch）

| M | bm256 baseline | bm128 | gain |
|---|---|---|---|
| 512 | 886 | **1060** | +20% |
| 1024 | 1764 | **2038** | +16% |
| 4096 | 3040 | 1928 | bm256 faster |

## 历史基线对比（vs 本项目起点）

| op | 2026-06-09 起点 | 2026-06-16 | 改进 |
|---|---|---|---|
| fwd kernel geomean | 1841 | 2397 | +30% |
| wgrad kernel geomean | 1516 | 2163 | +43% |
| dgrad kernel geomean | ~2200 | 2307 | +5% |

主要收益：fwd 来自非持久 + swizzle-AT；wgrad 来自 asm_mma + persist/masked M-branch + autotune；dgrad 来自 bm128 小-M + path-J inline-asm。
