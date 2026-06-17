# FlyDSL vs HipKittens fp8 GEMM 基准数据（防遗忘存档）

> **2026-06-12 更新（§6）**：FlyDSL grouped 优化完成 —— **fwd 2314 / dgrad 2242 / wgrad 2149**（kernel级，n=12 真实 MoE，SNR全对）。commit cd215bb8 + efe462dd。
> **fly/hk：fwd 1.063× / wgrad 1.105×**（dgrad HK坏无对照），§5 的「fly/hk≥1.05」目标达成。详见文末 §6。

测于 2026-06-09，节点 **chi2832**（gfx950/MI355X），容器 `mlperf_gptoss`（rocm/primus:v26.2，
triton 3.7），数据盘 `/mnt/vast/kyle/code2`，GPU6（GPU0-3 被别人 vLLM 占，4-7 空）。
warm-mean（`torch.utils.benchmark.Timer`，~10 warmup + timeit(50)）。TFLOPS=2·M·N·K/t。
SNR 全部校验通过（≥20 才计入；50≈FlyDSL/HK，56≈Triton/hipBLASLt 参考精度）。

分支：
- FlyDSL dense+grouped：`dev/kyle/flydsl_grouped_fp8_tensorwise`（HEAD fa8298ce）
- HipKittens grouped：`dev/kyle_hipkitten_gemm_groupedgemm`（HEAD 0d39ab4f，**"Session 13 WIP half-edit"**）
- HK dense：`HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`（build 成 tk_fp8_layouts.so）

脚本（scripts2/）：`_dense_hk_vs_fly.py`（dense 三方）、`_grouped_op_bench.py`（grouped 算子级）、
`_gg_fwd_wgrad.py`（grouped fwd/wgrad kernel级+quant，SNR门槛）、`_grouped_full_bench.py`（含dgrad）。

---

## 一句话结论

- **Dense：FlyDSL 全面快于 HipKittens（geomean +8~13%），也快于 hipBLASLt。**
- **Grouped fwd（算子级，含量化）：FlyDSL ≈ Triton（0.98×），HK ≈ 1.06× Triton → HK 比 FlyDSL 快 ~8%。**
- **Grouped kernel 级（无量化）：HK fwd 1.12×Triton、wgrad 1.74×Triton，很强。**
- **HK grouped dgrad 在此 WIP 分支上是坏的**（RRR race，大-contraction×M=4096 出 SNR≈1 垃圾），未取得有效数。

---

## 1. Dense GEMM（llama7b/70b，FlyDSL vs HipKittens vs hipBLASLt）

shape：7b(h=4096,inter=11008) qkv/out/ffn_up/ffn_down；70b(h=8192,inter=28672) 同；M∈{2048,4096,8192}。
单位 TFLOPS。HK group_m 已扫最优。SNR：HK 50 / FLY 56。

### NT (fwd, A·Bᵀ) geomean: **fly/hk=1.077  fly/hbl=1.015  hk/hbl=0.942**
| tag | M | N | K | HK | FLY | HBL |
|---|---|---|---|---|---|---|
| 7b-qkv | 2048 | 12288 | 4096 | 2249 | 2212 | 1802 |
| 7b-qkv | 4096 | 12288 | 4096 | 2655 | 2865 | 2903 |
| 7b-qkv | 8192 | 12288 | 4096 | 2697 | 2896 | 2992 |
| 7b-out | 2048 | 4096 | 4096 | 1567 | 2310 | 1942 |
| 7b-out | 4096 | 4096 | 4096 | 2578 | 2802 | 2940 |
| 7b-out | 8192 | 4096 | 4096 | 2624 | 2830 | 2902 |
| 7b-ffn_up | 2048 | 11008 | 4096 | 2049 | 2288 | 1862 |
| 7b-ffn_up | 4096 | 11008 | 4096 | 2474 | 2678 | 2789 |
| 7b-ffn_up | 8192 | 11008 | 4096 | 2514 | 2705 | 2593 |
| 7b-ffn_down | 2048 | 4096 | 11008 | 2018 | 2377 | 1865 |
| 7b-ffn_down | 4096 | 4096 | 11008 | 2964 | 3093 | 3249 |
| 7b-ffn_down | 8192 | 4096 | 11008 | 3037 | 3156 | 3289 |
| 70b-qkv | 2048 | 10240 | 8192 | 2160 | 2323 | 2234 |
| 70b-qkv | 4096 | 10240 | 8192 | 2735 | 2877 | 2872 |
| 70b-qkv | 8192 | 10240 | 8192 | 2956 | 3145 | 3230 |
| 70b-out | 2048 | 8192 | 8192 | 2902 | 3046 | 3121 |
| 70b-out | 4096 | 8192 | 8192 | 2955 | 3122 | 3213 |
| 70b-out | 8192 | 8192 | 8192 | 2985 | 3144 | 3287 |
| 70b-ffn_up | 2048 | 28672 | 8192 | 2632 | 2773 | 2835 |
| 70b-ffn_up | 4096 | 28672 | 8192 | 2842 | 3016 | 3247 |
| 70b-ffn_up | 8192 | 28672 | 8192 | 2672 | 3081 | 3205 |
| 70b-ffn_down | 2048 | 8192 | 28672 | 3136 | 3082 | 3177 |
| 70b-ffn_down | 4096 | 8192 | 28672 | 3174 | 3256 | 3277 |
| 70b-ffn_down | 8192 | 8192 | 28672 | 3115 | 3096 | 3147 |

### NN (dgrad, A·B) geomean: **fly/hk=1.071  fly/hbl=1.575  hk/hbl=1.470**
（hipBLASLt 无原生 fp8 NN，走 transpose 回退，故 HBL 低）
| tag | M | N | K | HK | FLY | HBL |
|---|---|---|---|---|---|---|
| 7b-qkv | 2048 | 12288 | 4096 | 2180 | 2155 | 1340 |
| 7b-qkv | 4096 | 12288 | 4096 | 2576 | 2725 | 1447 |
| 7b-qkv | 8192 | 12288 | 4096 | 2641 | 2809 | 1859 |
| 7b-out | 2048 | 4096 | 4096 | 1535 | 2057 | 1385 |
| 7b-out | 4096 | 4096 | 4096 | 2535 | 2734 | 1430 |
| 7b-out | 8192 | 4096 | 4096 | 2544 | 2761 | 1724 |
| 7b-ffn_up | 2048 | 11008 | 4096 | 1988 | 2188 | 1800 |
| 7b-ffn_up | 4096 | 11008 | 4096 | 2452 | 2612 | 1736 |
| 7b-ffn_up | 8192 | 11008 | 4096 | 2467 | 2676 | 1874 |
| 7b-ffn_down | 2048 | 4096 | 11008 | 1861 | 2109 | 1408 |
| 7b-ffn_down | 4096 | 4096 | 11008 | 2869 | 3041 | 1513 |
| 7b-ffn_down | 8192 | 4096 | 11008 | 2942 | 3102 | 1984 |
| 70b-qkv | 2048 | 10240 | 8192 | 2059 | 2164 | 1815 |
| 70b-qkv | 4096 | 10240 | 8192 | 2621 | 2764 | 1580 |
| 70b-qkv | 8192 | 10240 | 8192 | 2853 | 3051 | 2071 |
| 70b-out | 2048 | 8192 | 8192 | 2828 | 2985 | 1473 |
| 70b-out | 4096 | 8192 | 8192 | 2850 | 3018 | 1894 |
| 70b-out | 8192 | 8192 | 8192 | 2856 | 3054 | 2005 |
| 70b-ffn_up | 2048 | 28672 | 8192 | 2528 | 2646 | 1966 |
| 70b-ffn_up | 4096 | 28672 | 8192 | 2779 | 2920 | 1994 |
| 70b-ffn_up | 8192 | 28672 | 8192 | 2812 | 2941 | 1976 |
| 70b-ffn_down | 2048 | 8192 | 28672 | 2911 | 3044 | 1511 |
| 70b-ffn_down | 4096 | 8192 | 28672 | 2978 | 3160 | 1984 |
| 70b-ffn_down | 8192 | 8192 | 28672 | 2918 | 2989 | 2007 |

### TN (wgrad, Aᵀ·B) geomean: **fly/hk=1.091  fly/hbl=1.826  hk/hbl=1.673**
| tag | M | N | K | HK | FLY | HBL |
|---|---|---|---|---|---|---|
| 7b-qkv | 2048 | 12288 | 4096 | 1977 | 2066 | 1230 |
| 7b-qkv | 4096 | 12288 | 4096 | 2388 | 2608 | 1380 |
| 7b-qkv | 8192 | 12288 | 4096 | 2384 | 2685 | 1418 |
| 7b-out | 2048 | 4096 | 4096 | 1362 | 1625 | 1219 |
| 7b-out | 4096 | 4096 | 4096 | 2265 | 2602 | 1274 |
| 7b-out | 8192 | 4096 | 4096 | 2357 | 2599 | 1346 |
| 7b-ffn_up | 2048 | 11008 | 4096 | 1782 | 2080 | 1363 |
| 7b-ffn_up | 4096 | 11008 | 4096 | 2241 | 2514 | 1335 |
| 7b-ffn_up | 8192 | 11008 | 4096 | 2247 | 2581 | 1403 |
| 7b-ffn_down | 2048 | 4096 | 11008 | 1731 | 1925 | 1427 |
| 7b-ffn_down | 4096 | 4096 | 11008 | 2725 | 2732 | 1472 |
| 7b-ffn_down | 8192 | 4096 | 11008 | 2772 | 2947 | 1516 |
| 70b-qkv | 2048 | 10240 | 8192 | 1907 | 2040 | 1398 |
| 70b-qkv | 4096 | 10240 | 8192 | 2437 | 2664 | 1485 |
| 70b-qkv | 8192 | 10240 | 8192 | 2670 | 2935 | 1478 |
| 70b-out | 2048 | 8192 | 8192 | 2607 | 2809 | 1469 |
| 70b-out | 4096 | 8192 | 8192 | 2689 | 2846 | 1484 |
| 70b-out | 8192 | 8192 | 8192 | 2658 | 2918 | 1479 |
| 70b-ffn_up | 2048 | 28672 | 8192 | 2413 | 2590 | 1436 |
| 70b-ffn_up | 4096 | 28672 | 8192 | 2609 | 2858 | 1424 |
| 70b-ffn_up | 8192 | 28672 | 8192 | 2650 | 2881 | 1400 |
| 70b-ffn_down | 2048 | 8192 | 28672 | 2803 | 2913 | 1465 |
| 70b-ffn_down | 4096 | 8192 | 28672 | 2863 | 3020 | 1460 |
| 70b-ffn_down | 8192 | 8192 | 28672 | 2792 | 2917 | 1453 |

---

## 2. Grouped GEMM — fwd(RCR) + wgrad(CRR var-K)，balanced B=8

MoE shape：deepseekv3(h7168,i2048)/qwen235b(h4096,i1536)/gpt_oss(2880,2880)，up/down，M∈{2048,4096}。
两列：**kern**=kernel级（预量化，无 quant 开销）；**full**=含 tensorwise 量化（每次重量化 a/b，算子真实开销）。

### geomean TFLOPS
| op | 量 | **HIPKITTEN** | TRITON | HIPBLASLT | HK/TRI | HK/HBL |
|---|---|---|---|---|---|---|
| **fwd** | kern | **2178** | 1945 | 1480 | 1.12 | 1.47 |
| fwd | full(quant) | 374 | 368 | 313 | 1.02 | 1.19 |
| **wgrad** | kern | **1918** | 1100 | 1048 | 1.74 | 1.83 |
| wgrad | full(quant) | 354 | 310 | 287 | 1.14 | 1.23 |

→ kernel 级 HK fwd/wgrad 都明显强于 Triton/hipBLASLt。含量化后差异被量化开销压平（每次 quant 摊到所有 backend）。

### fwd 逐行 kern TFLOPS（SNR 50/56 全对）
| tag | M | N | K | HK | TRI | HBL |
|---|---|---|---|---|---|---|
| deepseek-up | 2048 | 2048 | 7168 | 2388 | 2098 | 1379 |
| deepseek-up | 4096 | 2048 | 7168 | 2652 | 2404 | 1433 |
| deepseek-down | 2048 | 7168 | 2048 | 2059 | 1905 | 2189 |
| deepseek-down | 4096 | 7168 | 2048 | 2162 | 1986 | 2383 |
| qwen235b-up | 2048 | 1536 | 4096 | 2081 | 1733 | 918 |
| qwen235b-up | 4096 | 1536 | 4096 | 2489 | 2170 | 1017 |
| qwen235b-down | 2048 | 4096 | 1536 | 1945 | 1823 | 1252 |
| qwen235b-down | 4096 | 4096 | 1536 | 1810 | 1763 | 1956 |
| gpt_oss-up | 2048 | 2880 | 2880 | 2230 | 1920 | 1454 |
| gpt_oss-up | 4096 | 2880 | 2880 | 2112 | 1832 | 1502 |
| gpt_oss-down | 2048 | 2880 | 2880 | 2248 | 1942 | 1474 |
| gpt_oss-down | 4096 | 2880 | 2880 | 2090 | 1858 | 1454 |

### wgrad 逐行 kern TFLOPS（SNR 50/56 全对）
| tag | M | N | K | HK | TRI | HBL |
|---|---|---|---|---|---|---|
| deepseek-up | 2048 | 2048 | 7168 | 1785 | 1084 | 1110 |
| deepseek-up | 4096 | 2048 | 7168 | 2255 | 1250 | 1269 |
| deepseek-down | 2048 | 7168 | 2048 | 1776 | 1132 | 1161 |
| deepseek-down | 4096 | 7168 | 2048 | 2237 | 1249 | 1280 |
| qwen235b-up | 2048 | 1536 | 4096 | 1903 | 1140 | 1053 |
| qwen235b-up | 4096 | 1536 | 4096 | 2272 | 1241 | 1245 |
| qwen235b-down | 2048 | 4096 | 1536 | 1898 | 1140 | 1054 |
| qwen235b-down | 4096 | 4096 | 1536 | 2227 | 1253 | 1204 |
| gpt_oss-up | 2048 | 2880 | 2880 | 1606 | 911 | 845 |
| gpt_oss-up | 4096 | 2880 | 2880 | 1815 | 989 | 846 |
| gpt_oss-down | 2048 | 2880 | 2880 | 1620 | 911 | 836 |
| gpt_oss-down | 4096 | 2880 | 2880 | 1793 | 988 | 845 |

### FlyDSL grouped（算子级 fwd，含量化，B=8，来自 `_grouped_op_bench.py`，flydsl build）
geomean TF：**FLYDSL 1104 / TRITON 1126 / HIPBLASLT 907** → fly/tri 0.98、fly/hbl 1.22。
（与 HK 同 build 锚点 TRITON 1136 对齐：HK fwd 算子级 1200 → **HK ≈ 1.06×Triton、FlyDSL ≈ 0.98×Triton，即 grouped fwd 上 fly/hk ≈ 0.92，HK 快 ~8%**。）
逐 shape（算子级 fwd TF）fly / hk：dsv3-up M2048 1072/1164、M4096 1344/1429；dsv3-down 1108/1235、1409/1514；
qwen-up 834/946、1110/1168；qwen-down 947/1085、1146/1221；gpt-up 968/1097、1211/1276；gpt-down 1020/1100、1208/1272。

**FlyDSL grouped 的 kernel 级 fwd/wgrad（无量化）尚未测**——需切回 flydsl build 跑 `_gg_fwd_wgrad.py GG_BACKEND=FLYDSL`。

---

## 3. ⚠️ HK grouped dgrad（RRR）在此 WIP 分支上是坏的

`dev/kyle_hipkitten_gemm_groupedgemm @ 0d39ab4f`（"Session 13 WIP half-edit"）的 **RRR(a·b) kernel 有 race**：
大 contraction（K 或 N ≥ ~2880）× M=4096 时输出 **SNR≈1（垃圾）**。生产 autograd 算子复现：
- dsv3-up trans_b=False fwd(RRR,contraction K=7168) → SNR 1；trans_b=True dgrad(RRR,contraction N=2048) → SNR 28 OK。
- gpt_oss(N=K=2880) trans_b=True dgrad(RRR) 和 trans_b=False fwd(RRR) 都 SNR 1。
- RCR/CRR 全对（fwd 用 trans_b=True、wgrad CRR 都正确）。

即 memory 记的 **bn128 RRR race**（K≥384 main-iter + 大问题触发）。修复在 **HK `6c7327bc` / PT 3rdparty `54527db7`（2026-05-21，"378/378 RRR no regression"）**，不在此 WIP HEAD 上。
→ 要取有效 HK dgrad，必须切到 RRR 已修的 commit 重建。

**坑**：测 grouped 时 trans_b 约定决定哪个 op 走 RRR：fp8 算子默认 trans_b=True（fwd=RCR、dgrad=RRR）；
bf16 默认 trans_b=False（fwd=RRR、dgrad=RCR）。在坏 RRR 分支上，用错约定就会测到垃圾 kernel（慢且错）。

---

## 4. 环境重建要点
- 节点claim/容器/triton/flydsl安装：见 `claim-mi355x-node/SKILL.md`（数据盘已改 `/mnt/vast`）。
- HK dense build：容器内 `cd /workspace/code/HipKittens/analysis/fp8_gemm/mi350x && THUNDERKITTENS_ROOT=/workspace/code/HipKittens ROCM_PATH=/opt/rocm PATH=/opt/venv/bin:$PATH make -j`。
- HK grouped：切 `dev/kyle_hipkitten_gemm_groupedgemm` → `sync.sh push Primus-Turbo --mirror`（必须 --mirror 删 flydsl 残留 moe_permute，否则撞 get_max_shmem_per_block）→ 容器内 `rm -rf build && GPU_ARCHS=gfx950 pip install --no-build-isolation -e .`。
- FlyDSL grouped：切 `dev/kyle/flydsl_grouped_fp8_tensorwise` → sync --mirror → 同样重建。
- 强制 backend：`GlobalBackendManager.set_grouped_gemm_backend(BackendType.X, PrecisionType.FP8)`；务必 `set_auto_tune(True)`。

---

## 5. FlyDSL grouped 基线（kernel级，待优化起点，2026-06-09）

`_gg_fwd_wgrad.py GG_BACKEND=FLYDSL`，同 shape/协议，SNR 56 全对。

| op | FLYDSL kern | HK kern | **fly/hk** | 目标(≥1.05×HK) |
|---|---|---|---|---|
| fwd | 1841 | 2178 | 0.845 | ≥2287 (+24%) |
| wgrad | 1516 | 1918 | 0.79 | ≥2014 (+33%) |

逐 shape fwd kern（fly / hk）：
dsv3-up 2088/2388, 2450/2652; dsv3-down 1713/2059, 1926/2162;
qwen-up 1577/2081, 2075/2489; qwen-down 1525/1945, 1673/1810;
gpt-up 1665/2230, 1797/2112; gpt-down 1911/2248, 1875/2090.
wgrad kern fly: dsv3 1447/1691/1516/1776, qwen 1487/1814/1588/1775, gpt 1273/1291/1383/1291.

**任务（user 2026-06-09）：优化 FlyDSL grouped 至 fly/hk ≥ 1.05（绝对快 5%）。** FlyDSL dense fwd 已 2500-3000（>HK），
说明 grouped body 开销是主攻点。优化驱动器：`scripts2/run_grouped_opt_sessions.sh` + `GROUPED_OPT_PLAN.md`（基线/目标需改为 HK-relative）。

---

## 6. FlyDSL grouped 优化结果（2026-06-12，分支 dev/kyle/flydsl_grp_fwd_swpipe）

`_gg_fwd_wgrad.py GG_BACKEND=FLYDSL BENCH_COOLDOWN=2`，同 §2/§5 协议（kernel级 no-quant，B=8 balanced，SNR56全对），chi2832 GPU6。

### geomean kern TFLOPS 进步链
| op | §5 原始基线(2026-06-09) | HEAD 231e0f09(autotuned persistent) | **现在(非持久+swizzle-AT)** | vs HEAD | HK(§2) | **fly/hk** |
|---|---|---|---|---|---|---|
| **fwd** | 1841 | 2232 | **2315** | **+3.7%** | 2178 | **1.063** |
| **dgrad** | (HK坏,未测) | 2157 | **2238** | **+3.8%** | (坏) | — |
| **wgrad** | 1516 | 2138 | **2119** | -0.9% | 1918 | **1.105** |

fwd 比原始基线 **+25.7%**，wgrad **+39.7%**。**§5 目标「fly/hk≥1.05」: fwd(1.063)+wgrad(1.105) 达成。**

### 关键架构（怎么做到的）
1. **num_cu 路由**：`num_cu=-1`(默认,全device,不留CU给comm) → **非持久 nt8w(fwd)/nn8w(dgrad)**；`num_cu>0`(comm-overlap,留CU) → persistent grid capped 到 num_cu。wgrad 始终 persistent。
2. **非持久 kernel(nt8w/nn8w)**：官方 8wave 直线 body(无 scf.for tile loop → 省 ~11% 后端调度惩罚)，+G 上界 grid + s_endpgm over-launch guard + per-group m_end store clamp(变组正确)+ K-tail mask(gpt K=2880)。
3. **决定性教训**：光直线化(非持久)**不够**——全 shape 测它比 persistent 略亏(-1%)，因为赢大K(dsv3 +8%,循环惩罚主导)但**输小K(gpt -13%)**,小K靠的是 persistent 的 L2 调优。**必须把 L2 swizzle(XCD remap + group_m/band)移植进非持久 kernel**,非持久才全面超 persistent。
4. **非持久 per-shape autotune(≤4候选)**：默认 `(num_xcd=8,group_m=4)` + alts `{(1,0,0)行优先, (8,8,0)宽簇, (8,4,band)2D}`，鲁棒计时(100 warmup+median5)+ 1.5% hysteresis(防噪声 mis-pick)。逐 shape 选最优。
5. fwd 逐 shape:11/12 清晰超 persistent(dsv3 +8~9%/qwen-up +6~7%/gpt +2~5%),qwen-down M4096 噪声级平手(2010 vs 2014)。

### fwd 逐 shape kern（非持久 swizzle-AT / HEAD persist）
dsv3-up 2788/2559(+9%),2794/2585; dsv3-down 2266/2254,2285/2268; qwen-up 2657/2495(+6.5%),2681/2497;
qwen-down 1679/1650,2010/2014(平); gpt-up 2219/2174,2235/2222; gpt-down 2183/2103(+3.8%),2268/2153(+5.3%).

### 坑/注意
- **非持久 cherry-pick 陷阱**：单看 deepseek(大K)非持久 +7%,会误以为全面赢；必须全 shape 测——小K是反的。
- **autotune 单次计时噪声**会 mis-pick(naive pick-min 把 gpt-up M2048 选到 -10%);必须强默认+hysteresis+鲁棒计时。
- nt8w 初版漏了 K-tail(assert K%128==0)→ gpt K=2880 AssertionError;已加 mask_a_tail。
- e5m2/e4m3 唯一差别=MFMA atom;nt8w/nn8w 都接 cbsz/blgp,走同一快路径。
- 改动全在 Primus-Turbo(gemm_fp8_grouped_kernel.py + grouped_gemm_fp8_impl.py + 新 gemm_fp8_grouped_nt8w.py/nn8w.py),未 commit。


### 6b. source-level 平台期 + 到竞品 B200 2800T 的路（2026-06-12 续）
竞品 NV **B200** grouped fp8 平均 **2800T**；我们 ~2300（fwd 2314）。MI355X fp8(BF8) **dense 天花板 ~3014-3217**（native 16×16，非 MXFP4 的 5253）。big-K(dsv3 2761)已 ~92% 天花板；小 K(gpt/qwen-down)更低（overhead/load-bound）。
- **autotune 鲁棒计时(250warmup/5x50)= 真修复**：旧 5-iter-warmup 在短 K mis-pick；修后 qwen-down M2048 fwd 1679→1956(+16.5%)；wgrad 2119→2149。**短 K kernel 计时必须重 warmup 到 boost clock。**
- **死路（别再试，nt8w 非持久不 transfer persistent 的杠杆）**：sched_barrier(0) before-mfma → -5.5%+SNR掉（非持久 before-mfma s_barrier 是 load-bearing LDS 同步）；cshuffle store → 中性（qwen-down +2.2%/gpt -2.3%）。
- **2800 只剩两条重路**：①全循环裸汇编抬 BF8 dense 天花板（mxfp4-campaign 式，几 session）②换 MXFP4 microscaling 格式（峰值 5253）——取决于竞品用啥格式。BF8 source-level 已榨干(~2300 平台)。
