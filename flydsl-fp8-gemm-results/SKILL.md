---
name: flydsl-fp8-gemm-results
description: Primus-Turbo (tensorwise) FlyDSL fp8 dense + grouped GEMM 在 MI355X(gfx950) 上的生产实测 TFLOPS / SNR 基线 + 复现方法。当用户问"现在 fp8 gemm/grouped gemm 跑多少 TFLOPS""4-wave 对 grouped 有多少提升""某个 shape 该有的性能是多少""回归了吗(和基线比)"时使用。数据由公开 wrapper 的 per-shape autotune 得出(= 生产实际选中的 kernel)。区别于 fp8-gemm-bench(那个测 hipkitten raw op)、flydsl-fp8-gemm-tuning(那个是调优方法论)。
---

# flydsl-fp8-gemm-results (MI355X 生产基线)

Primus-Turbo **tensorwise** 分支的 FlyDSL fp8 GEMM/grouped GEMM 实测基线。数据走**公开 wrapper**(内部 per-shape autotune，候选网格含 4-wave/8-wave/persistent) → 即**生产实际选中的 kernel 性能**。MI355X gfx950, fp8 dense peak ~5 PFLOPS。

- 测量日期：2026-07-01，chi2810 容器 `mlperf_gptoss`，GPU4 独占，venv `/opt/venv-tw`。
- 配置：cuda event, warmup=20, iter=50, per-tensor scale=1.0, E4M3(`float8_e4m3fn`), out bf16, block-aligned shapes。
- ⚠️ run-to-run 噪声 **~5%**；<1.5% 的差异是噪声。fp8 DVFS 功耗墙 → **绝对 TFLOPS 只在同脚本/同次会话内可比**，跨天/跨脚本对比要重测基线。

## 复现

```bash
cd /workspace/code/gpt_oss_docker/sync
# 同步(本地→远端)
rsync -azh --exclude-from=.rsync-exclude -e "$PWD/.ssh-chi.sh" \
  "$PWD/tensorwise/Primus-Turbo/" root@chi2810:/mnt/vast/kyle/code2/Primus-Turbo-tensorwise/
# 挑空闲卡: rocm-smi --showmeminfo vram | grep 'Used Memory' (找 ~300MB 的卡)
G=4   # 空闲 GPU id
./.ssh-chi.sh root@chi2810 "docker exec -e HIP_VISIBLE_DEVICES=$G mlperf_gptoss bash -lc \
  'cd /workspace/code/Primus-Turbo-tensorwise && /opt/venv-tw/bin/python -u primus_turbo/_d_dense_bench.py'"
# grouped: 真实 MoE 模型 shape (deepseek/qwen235b/gpt_oss, G=8, m=2048/4096)
./.ssh-chi.sh root@chi2810 "docker exec -e HIP_VISIBLE_DEVICES=$G mlperf_gptoss bash -lc \
  'cd /workspace/code/Primus-Turbo-tensorwise && /opt/venv-tw/bin/python -u primus_turbo/_g_models2.py'"
# grouped 8-wave-only 基线(量化 4-wave 增益; 开关是 import-time 读的 -> 单独进程):
./.ssh-chi.sh root@chi2810 "docker exec -e HIP_VISIBLE_DEVICES=$G -e PT_NO_4WAVE=1 mlperf_gptoss bash -lc \
  'cd /workspace/code/Primus-Turbo-tensorwise && PT_NO_4WAVE=1 /opt/venv-tw/bin/python -u primus_turbo/_g_models2.py'"
```

- 探针脚本(仓库内, untracked)：`primus_turbo/_d_dense_bench.py`(dense NT/NN/TN 公开 wrapper) · **`primus_turbo/_g_models2.py`(grouped fwd/dgrad/wgrad，真实 MoE 模型 shape，`PT_NO_4WAVE` 切 4w/8w，canonical)**。(`_d_grouped_bench.py` 是早期通用-shape 版，已废弃：MoE 要用真实模型 shape 而非 square/big-N 通用格。)
- 公开入口：dense `gemm_fp8_tensorwise_flydsl_kernel(a,sa_inv,b,sb_inv,trans_a,trans_b,out_dtype)`；grouped fwd/dgrad `grouped_gemm_fp8_tensorwise_flydsl_kernel(a,b,sa,sb,offs,trans_b,...)`；grouped wgrad `grouped_gemm_fp8_variable_k_tensorwise_flydsl_kernel(lhs,rhs,ls,rs,offs,...)`。
- 首调每个 shape 做 autotune 编译(慢)，按 (M,N,K,...) 缓存 → warmup 内已 cache-hit。全量 dense 72 shape ~13min；grouped 7 shape × 2 M × 3 op 冷跑 ~7min，热缓存(第二模式)~1min。
- 布局：NT=forward(a[M,K]@b[N,K]^T) · NN=dgrad(a[M,K]@b[K,N]) · TN=wgrad(a[K,M]^T@b[K,N])。

## Dense fp8 GEMM (autotuned = 生产) — Llama 7B/70B 权重 shape

per-shape 生产 autotune 选中 kernel 的 TFLOPS，SNR 全 = **56**(fp8 E4M3 健康)。

| shape (N×K) | M=4096 | M=8192 | M=16384 |
|---|---|---|---|
| **NT (forward)** | | | |
| 7b-qkv 12288×4096 | 2381 | 2763 | 2802 |
| 7b-oproj 4096×4096 | 2284 | 2426 | 2671 |
| 7b-gateup 22016×4096 | 2657 | 2664 | 2665 |
| 7b-down 4096×11008 | 2762 | 2974 | 3001 |
| 70b-qkv 10240×8192 | 2726 | 3018 | 3001 |
| 70b-oproj 8192×8192 | 2883 | 2990 | 2999 |
| 70b-gateup 57344×8192 | 2984 | 3052 | 2965 |
| 70b-down 8192×28672 | 3147 | 3070 | 3089 |
| **NN (dgrad)** | | | |
| 7b-qkv 12288×4096 | 2522 | 2793 | 2699 |
| 7b-oproj 4096×4096 | 2236 | 2689 | **1011** ⚠ |
| 7b-gateup 22016×4096 | 2543 | 2639 | 2635 |
| 7b-down 4096×11008 | 2691 | 2974 | 2976 |
| 70b-qkv 10240×8192 | 2562 | 2963 | 2940 |
| 70b-oproj 8192×8192 | 2799 | 2942 | 2902 |
| 70b-gateup 57344×8192 | 2881 | 2934 | 2846 |
| 70b-down 8192×28672 | 3053 | 3069 | 2994 |
| **TN (wgrad)** | | | |
| 7b-qkv 12288×4096 | 2521 | 2611 | 2488 |
| 7b-oproj 4096×4096 | 2106 | 2396 | 1976 |
| 7b-gateup 22016×4096 | 2282 | 2457 | 2503 |
| 7b-down 4096×11008 | 2678 | 2766 | 2459 |
| 70b-qkv 10240×8192 | 2227 | 2811 | 2738 |
| 70b-oproj 8192×8192 | 2333 | 2820 | 2740 |
| 70b-gateup 57344×8192 | 2912 | 2926 | 2932 |
| 70b-down 8192×28672 | 3084 | 3021 | 3030 |

**读法：**
- 稳态区间 **~2200–3150 TF**(≈ 44–63% dense peak)。big-K/大方阵最高(3000+)；小 N 方阵(oproj 4096×4096)最弱。
- M 越大越好(4096→8192 普遍 +5~15%)，8192↔16384 已饱和。
- ⚠ **NN 16384×4096×4096 = 1011 TF 是异常低点**(同 shape M=8192 有 2689)。属该小-N 方阵 regime 的 autotune 选到坏配置/timing 抖动 → 复现如稳定复现即为可优化点(dense skill 里的小方阵 regime)。TN 同 shape 也偏弱(1976)。
- **回归判据**：同脚本重测，某 shape 掉 >8%(超噪声)才算真回归；先排除是不是 autotune 缓存没命中/别的进程抢卡。

## Grouped fp8 GEMM (真实 MoE 模型 shape, G=8)

**测法**：`_g_models2.py`，三个真实模型的 per-expert GateUP/Down GEMM，**G=8**、**M=2048/4096(每专家 token 数，Mt=G×M=16384/32768)**、平衡分组、走公开生产 wrapper。SNR 全 = **56**(健康)。fwd/dgrad/wgrad 三个 op 的生产 autotuned TFLOPS：

模型 forward (N,K)：`deepseek-up 4096×7168 · deepseek-down 7168×2048 · qwen235b-up 3072×4096 · qwen235b-down 4096×1536 · gpt_oss-up 5760×2880 · gpt_oss-down 2880×2880`（另 `synth-oddtail 3008×3008` 覆盖 odd-K tail）。

| model-gemm | N | K | fwd (m2048/m4096) | dgrad | wgrad |
|---|---|---|---|---|---|
| deepseek-up | 4096 | 7168 | 2690 / 2728 | 2404 / 2500 | 2271 / 2593 |
| deepseek-down | 7168 | 2048 | 2227 / 2248 | 2531 / 2628 | 2182 / 2495 |
| qwen235b-up | 3072 | 4096 | 2476 / 2561 | 2321 / 2336 | 2045 / 2431 |
| qwen235b-down | 4096 | 1536 | 1834 / 1981 | 2065 / 2390 | 2101 / 2482 |
| gpt_oss-up | 5760 | 2880 | 2184 / 2230 | 2349 / 2434 | 2014 / 2299 |
| gpt_oss-down | 2880 | 2880 | 2011 / 2143 | 1975 / 2107 | 1821 / 2146 |
| synth-oddtail | 3008 | 3008 | 2157 / 2267 | 2096 / 2172 | 1839 / 2311 |

**读法：**
- 稳态 **~1800–2730 TF**。m4096(Mt=32768) 普遍比 m2048 高(更满的 grid)，尤其 wgrad(+10~20%)；小 K 的 qwen235b-down(K=1536) 最弱(fwd ~1900)。
- **fwd/dgrad：4-wave 与 8-wave-persistent 打平**(auto vs `PT_NO_4WAVE=1` 差异 ±2% 全在噪声内) → autotuner 多数选 8-wave-persistent。印证会话结论"dense/grouped 4w-persistent NT 候选几乎不被采纳"。
- **4-wave 的价值集中在 wgrad(variable-K)**：deepseek-up/down、gpt_oss-up 稳定 **+6~17%**(auto vs 8w-only)；其余 shape 在 ±噪声内。
  - ⚠️ 两次跑(8w+4w 冷 autotune 441s / 8w-only 热缓存 51s)**时长差 8×→热态不同**，个别 shape auto 反而略低于 8w 是**跨进程 DVFS 噪声**(进程内 autotuner 绝不会选更慢的 4w)。要精确 4w 增益须**同进程同热态**对比。
- 同量级 shape grouped 比 dense 低 ~10-20%(per-group offset/mask/persistent 循环开销)。

## 2026-07-02 更新：grouped wgrad autotune 扩展 + 精简（当前生产 4-wave 网格）

本轮把 4-wave wgrad 的 per-shape autotune 候选从「group swizzle × xcd{1,8}」扩到「× xcd{1,2,4,8}」，再按真实采纳率精简到 **8 个候选**（`_autotune_wgrad_dispatch._w4`，字段 = `(group_m, group_n, num_xcd)`，phase-barrier waitcnt 固定 16/10/0）：

    (0,0,8) (0,0,2) (0,0,4) (8,4,8) (8,4,2) (8,4,4) (4,4,8) (4,4,2)

- **xcd 2/4 确实被采纳**（非仅 {1,8}）；row-major 与两 band 族 (8,4)/(4,4) 各有 shape 命中。删掉的「从不被采纳」候选：`(0,0,1)`、`(4,4,4)`、孤儿 waitcnt `(0,0,8,20,8,0)`（11→8，实测 best TF 无回归、SNR 50-56）。
- 采纳率实测法：`_ow.py`（直编各候选取 best = 等价 autotune 选择），chi2810 MI355X GPU7 独占、每档 `rm -rf /root/.flydsl` 重编。

### wgrad per-shape best（`_ow.py` 直编 4-wave，SNR 54-56，实测 2026-07-01）
| model wgrad | OUT (M×N) | m2048 | m4096 |
|---|---|---|---|
| deepseek-up | 7168×4096 | 2331 | 2644 |
| deepseek-down | 2048×7168 | 2212 | 2612 |
| qwen235b-up | 4096×3072 | 2367 | 2680 |
| qwen235b-down | 1536×4096 | 2302 | 2661 |
| gpt_oss-up | 2880×5760 | 2090 | 2404 |
| gpt_oss-down | 2880×2880 | 1914 | 2238 |

- OUT 维度是 wgrad 输出 `[K_fwd, N_fwd]`，与上文 forward `(N,K)` 命名相反。
- 点测（非全集）：deepseek-up **m8192 ≈ 2665**；synth-oddtail(3008×3008) **m2048 ≈ 2069**。
- **swizzle×xcd autotune 相对「无 swizzle」基线 +4.9~5.9%**（`_ab.py` A/B，m2048：deepseek g84x8/g44x8 +5.5%、qwen g84x8 +5.9%、gpt_oss g44x8 +4.9%）；xcd{2,4} 扩展在 m2048 未超已有 xcd{1,8}+swizzle，其边际增益(+0.5~2.9%)出现在其它 m。

### 关键结论（优化到头了否？详见 memory `project_wgrad_occ_feed_bound`）
- **grouped wgrad ≠ DVFS 功耗受限**（区别于 dense fp8）：实测跑核满频 **2400MHz、~266W**（远低于 dense randn 的 ~1072W 功耗墙），因 MfmaUtil 仅 ~54%、功耗够不到墙。**故 wgrad 是效率/autotune 优化真能体现的路径**；而 dense/fwd/dgrad 的指令效率优化被功耗墙掩盖（`project_flydsl_fp8_power_limited`）。
- **autotune 之外的结构杠杆本轮全部实测判负**：store-defer（LDS 160KB 悬崖，x2 中性 / x3 -18.6%，≤+5% 天花板且仅小 m）、免一个操作数转置（`ds_read_b64_tr_b8`==plain `ds_read_b64`，0 收益）、epilogue write128/tr16（-2~7.7%）、跨 tile 重叠（-5%）、epilogue 双缓冲去 drain（-1~3.5%）、barrier 参数扫、增大 reg tile。根因 = occ=1 的 512-VGPR 满载锁死 prefetch 深度，8-wave 升 occ 反更慢（feed 争用）。

## 2026-07-02（续）：4-wave 合入分支 + dispatch 修复 + 全 shape 战胜 8-wave-only + tile 路线证伪

分支 `feat/kyle/grouped-wgrad-4wave`（AMD-AGI/Primus-Turbo，author kyle-256），两 commit：
- `b2dc5a35` feat：新增 occ=1 4-wave whole-loop wgrad（基于 main `3a5f485e`；main 的 grouped wgrad **只有 8-wave** masked+persistent，无 4-wave）。
- `c846d954` opt：大 contraction dispatch 原来「任一 4-wave 数值通过就无条件返回」（既不与 8-wave 比速，也只拿单个 masked 作 fallback）→ 改为 **masked 8-wave 池 `(8,4,8)/(8,0,8)/(4,4,8)` + 3 个 4-wave 一起串行 `_robust_time` 计时取最快**，best-8w 作 fallback，1.5% 滞后（与全 dispatch 一致）。

### ⚠️ 计时坑（重要）：`_g_models2` 的 `tus` warmup=20 太短，系统性低估 occ=1 的 4-wave 5~10%
- 20 iters 够不到 boost 频率；**occ=1 的 4-wave（计算密集）对频率敏感 → 被低估 5~10%，occ≥2 的 8-wave 不敏感**。曾据此误判 qwen「4-wave 回归 -2~4%」——纯假象。
- 修法：`_g_models2` 加了 `PT_WARMUP` env（默认 20）。**测 4-wave vs 8-wave 必须 `PT_WARMUP=250`**（= autotune `_robust_time` 同款 warmup = 真实部署 boost 频率）。
- 佐证：warmup 20→250，main（8w，occ≥2）qwen 几乎不动（2263/2281→2281 等），HEAD（4w）qwen 大涨（2212→2374、2134→2308）。差值全被短 warmup 吃掉。
- 注：`_robust_time`（复用 buffer 串行）与重叠感知计时（轮转 4 buffer）对 qwen 结论一致（4w 均 +5~10%）→ 之前怀疑的「8-wave 借新 buffer 重叠虚高」**不是**主因，warmup 才是。

### 4-wave(HEAD `c846d954`) vs 8-wave-only(main `3a5f485e`)：同 GPU、`PT_WARMUP=250`、wgrad TFLOPS
| shape | OUT M×N | m2048 8w→4w (Δ) | m4096 8w→4w (Δ) |
|---|---|---|---|
| deepseek-up | 7168×4096 | 2072→2368 (+14.3%) | 2504→2677 (+6.9%) |
| deepseek-down | 2048×7168 | 2050→2385 (+16.3%) | 2449→2609 (+6.5%) |
| qwen235b-up | 4096×3072 | 2281→2374 (+4.1%) | 2505→2684 (+7.1%) |
| qwen235b-down | 1536×4096 | 2233→2308 (+3.4%) | 2547→2731 (+7.2%) |
| gpt_oss-up | 2880×5760 | 1778→2121 (+19.3%) | 2085→2342 (+12.3%) |
| gpt_oss-down | 2880×2880 | 1762→1928 (+9.4%) | 2171→2324 (+7.0%) |
| synth-oddtail | 3008×3008 | 2002→2080 (+3.9%) | 2303→2517 (+9.3%) |
- **全 7 shape × 2 m 无一回归**；大 contraction/宽-N +9~19%，qwen/synth +3~9%。SNR 52-56（gpt_oss-up 偶 48，上游 E4M3 抖动，同 shape fwd/dgrad=56）。

### tile 路线证伪（不做，用数据 + 一阶原理省下 asm 重写）
- worst shape 是 **feed-bound**（gpt_oss-down/up/synth）。当前 whole-loop 手调 asm 是**写死 4-pool/4-quadrant/2×2-wave**（`QUADS` 硬编码、`_diag_cells` 要 nta≥2 ntb≥2）→ 最小 BLOCK 就是 256×256，缩到 128 = 重写 asm 结构，非改参数。
- gpt_oss-down 2880² 损失拆解（256×256 kernel，useful-TF）：满速 ~2400 →(grid 尾巴 4.5 波/90% util) -5% →(padding，边缘 tile 仅 64/256=25% useful) -14%→ 1956。可回收头顶 ~+23%。
- **但更小/矩形 tile 是错药**：现 256×256 只 ~48% peak（MfmaUtil 54%）= feed-bound；算术强度 (M·N)/(M+N) 随 tile 缩水（256²=128 → 128×256=85=-33% → 128²=64=-50%）。feed-bound 下缩 tile 净负，远超 padding+grid 能救的量。→ **tile 重写不做**。
- swizzle{5}×xcd{1,2,4,8}×vmcnt{1..8} 对 worst shape 全平(<2%)：瓶颈是 feed 带宽，非这些旋钮。真正剩余杠杆 = 256 tile 下提 feed 效率（LDS bank/transpose-read/phase-barrier），需 rocprof，未做。

### 端到端(含 quant)wgrad：`_g_models2.py PT_QUANT=1`
- 默认 `_g_models2` 计的是**纯 GEMM**（输入已预量化为 fp8，`s=1.0`）。加 `PT_QUANT=1` 把 `grad_out(dY, [Mt,K])` 的 bf16→fp8 **tensorwise 量化算进 wgrad 计时区**（`quantize_fp8(x, FP8, ScalingGranularity.TENSORWISE)`，返回 `(qdata, scale_inv)`）。
- **符合真实 backend**：激活 `a` 在 forward 量化一次并缓存、反向复用（不计）；只有 `grad_out` 在反向重量化。**且这份 quant 是 dgrad+wgrad 共享的** → 本 bench 把它 100% 算给 wgrad = **悲观上界**，真实分摊约一半。
- 实测（chi2810 gfx950 GPU7，`PT_WARMUP=250`，HEAD `c846d954` 8w+4w，SNR 全 56）wgrad GEMM-only→含 quant：

| shape | m2048 gemm→e2e (Δ) | m4096 gemm→e2e (Δ) |
|---|---|---|
| deepseek-up | 2340→1793 (−23%) | 2663→2052 (−23%) |
| deepseek-down | 2406→1833 (−24%) | 2583→2202 (−15%) |
| qwen235b-up | 2354→1685 (−28%) | 2650→1886 (−29%) |
| qwen235b-down | 2281→1729 (−24%) | 2714→1999 (−26%) |
| gpt_oss-up | 2113→1728 (−18%) | 2330→1948 (−16%) |
| gpt_oss-down | 1906→1430 (−25%) | 2273→1599 (−30%) |
| synth-oddtail | 2077→1529 (−26%) | 2433→1712 (−30%) |
- **含 quant 掉 ~15–30%**（悲观口径）：多的是一次纯访存的 tensorwise 量化（amax 归约+scale+cast），N/flop 越小占比越大（gpt_oss/synth 最重）。fwd/dgrad 两版几乎不变（无额外 quant）。
- **所有 4-wave vs 8-wave 的对比数（上文表）都是 GEMM-only**——反映 kernel 本身；含 quant 是端到端视角，不改变 4w>8w 的结论（quant 是二者共同的固定 overhead）。

## 已知坑 / 注意

- **grouped wgrad SNR 偶尔跌到 51-64 不是 kernel bug**：E4M3 tensorwise grouped 的低 SNR 是**上游量化/参考路径**精度(默认 TRITON 后端同 shape SNR 一致到小数点后 9 位)。判断法：同 shape 比两个后端 SNR，一样就是上游问题。fp8 下 50+ dB 视为通过(本次真实模型 shape 基本全 56)。
- 探针脚本首行剔除 `_SELF_DIR` 出 `sys.path`：否则 `import primus_turbo` 命中脚本所在的 `primus_turbo/` 目录(脚本就放在包内)造成循环/错包。
- dense 探针的旧版(`_d_llama_nt.py` 等)引用了 i64 重构前已删的 `_nt_4wave_args` → 已废弃，用 `_d_dense_bench.py`(走公开 wrapper，抗 API 变更)。
- 别看 `gpu_use=0` 判空闲，看 `--showmeminfo vram` 的 Used(~300MB=空)。GPU3 常年被别人占。
- **`_g_models2` 默认 warmup=20 太短，坑 occ=1 的 4-wave（低估 5~10%）**：测 4-wave/8-wave 对比必须 `PT_WARMUP=250`（详见「2026-07-02（续）」计时坑）。跨卡也要注意 chi2810 各卡有 ~2~4% 频率差，A/B 必须同卡同会话。

## 配合的 skill
- [`flydsl-fp8-gemm-tuning`](../flydsl-fp8-gemm-tuning/SKILL.md) — 调优方法论(profile→regime→lever)，想**提升**某 shape 时看它。
- [`fp8-gemm-bench`](../fp8-gemm-bench/SKILL.md) — hipkitten raw-op bench(另一套 baseline)。
- [`remote-sync`](../remote-sync/SKILL.md) — tensorwise 本地↔chi2810 同步 + venv-tw 运行。
