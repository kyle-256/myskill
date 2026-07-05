# Split-K、跨 lane 原语、低精度 dequant 融合与 wgrad variable-K 调度

> 类别: 方法论 · 主题标签: wgrad, load-balance, prefetch, autotune-dispatch, split-k, few-tile, bf16-atomic, occupancy, cross-lane-reduce, attention-numerics, flash-attention, mxfp4-pack, dequant-fuse, 低精度验证, NOSTORE归因

## wgrad variable-K 长尾:band-cyclic interleave 负载均衡、masked/persist crossover 1536

### variable-K 长尾根因
- wgrad 是 **variable-K GEMM**:output tile 等工作量(固定 OUT_M×OUT_N,MFMA 数相同),但 **K-loop 长度 = per-group M_g**。
- token skew 下大组 tile 串行主导 wall-time,小组 WG 空转 → group-contiguous dispatch 的长尾:
  - 2:1 skew **−20%**
  - 4:1 skew **−34%**
  - 32:1 skew **−59%**

### band-cyclic group interleave(负载均衡)
- cluster = 一个 group_m M-band(`group_m` 行 × N_BLOCKS_N 列)。
- dispatch 顺序:`band0_group0, band0_group1, ..., band0_group(G-1), band1_group0, ...` → 每时刻所有组都在飞(负载均衡)。
- band 内保留 group_m 的 B-stripe **L2 reuse**,因此 balanced 零损失。
- 要求 `N_BLOCKS_M % group_m == 0`,否则 fallback 一行 cluster。
- **balanced 零损失是默认开的关键前提**:band 内 tile 顺序与 group_m swizzle 完全一致。

#### skew 鲁棒性(修后实测)
| 场景 | 数字 |
|---|---|
| balanced | 2143(−0.9% 几乎无损) |
| 30:1 skew | 1162 → 1592(−46% → **−26%**) |
| vs Triton skew | 1.18× → 1.62×(FlyDSL band-cyclic 更均衡) |

### masked vs persist(K-loop body 分叉)
- **masked chunked(大-M,per-group m_total/G > 1536)**:outer runtime `scf.for over ceildiv(k_iters, chunk)` × inner `range_constexpr(chunk)` 的 **4-buffer 流水**。over-run 由 per-group SRD `num_records` clamp 到 0(无需 host cap)。`chunk=8`,每 chunk 8 个 K-iter 全展开。
- **persist(小-M,per-group m_total/G <= 1536)**:`_wgrad_loop_body_pipe`,**2-stage prefetch**(prologue prefetch K-tile 0,per-iter prefetch K+1 overlap 当前 MFMA)。短 contraction 下 masked 的 chunk over-run 是废功,persist 精确跑完自己 M_g。
  - 实测 M_g=512:**856 → 1369 TF(+60%)**。

#### crossover 阈值 = 1536(per-group M_g)
| M_g 区间 | 赢家 | 幅度 |
|---|---|---|
| M_g <= 1024 | persist 稳赢 | +3 ~ 20% |
| 1024 < M_g < 2048 | wash | ±4%(在 hysteresis 内) |
| M_g >= 2048 | masked 稳赢 | +5 ~ 15% |
- **1536 = wash 区中点偏向真实 MoE(高-G)**。

### 复用范式(共用坐标/寻址,只分叉 loop body)
- masked 和 persist 两个 kernel 共用:
  - `_wgrad_block_mn`:dispatch → `(group_idx, block_m, block_n)`
  - `_wgrad_rebase`:i64 SRD rebase → `(a_div, b_div)`
- 只有 K-loop body 不同(masked = chunked 4-buffer,persist = scf.for 2-stage prefetch)。
- 共用坐标/寻址逻辑、只分叉 loop body 是 FlyDSL 的复用范式。

## Split-K 修少-tile 大-K 欠订阅:asm 自包含只改 Python body、bf16 atomic 累加

### 何时用 split-K(候选门)
- 目标形状:few-tile 大-K,即一 WG/tile 撑不满 CU(tiles < ncu/2)且 K≥2048 → tile 数不足以填满 device,K 维长足以切分摊。
- 实证:`1024×1024×8192` fly/ait 从 **2.00 → 1.07**(修少-tile 大-K 欠订阅)。
- Split-K factor 候选集:**2/3/4/6/8/12/16**(仅对 few-tile 大-K 报)。
- Skinny/small-M GEMM(M≤~16–32、N*K 大,如 decode/output-projection)天然配对 split-K,因为 M 填不满 device。
- 实证(源自 gpt_oss2 fp8 项目 2900 TF 下一步,非本环境,见 gpt_oss2_docker/myskill/flydsl-fp8-gemm-tuning/SKILL.md):big-K 只 **1024 tiles** 填不满 CU → **split-K=2 → grid 翻倍**改善 CU 饱和(需加部分和 reduction,big-K 输出仅 **134MB** 可行);big-N tiles 已足(**3584**),split-K 帮助有限。

### asm K-loop 自包含 → split-K 不动一行 asm
- asm K-loop 由「传入 ki 次数 + 初始 soffset + 常量步进」驱动,**不依赖 K**,因此 split-K 只改 Python body:
  - loop 次数(ki)
  - 操作数 soffset 位移
  - scale soffset 位移
  - 输出 workspace 布局 `[ksplit*M, N]` + `base_row` 位移
  - grid × ksplit
- Host 端 reduce:`ws.view(ksplit, M, N).sum(0)`,用 **bf16 累加**(比 fp32-upcast 快 **2×**,误差~**1 ULP**)。

### kernel 内 packed-bf16 atomic 累加(去 intra-CTA 争用)
- 用 `buffer_atomic_pk_add_bf16` reduce bf16 partials(每 32-bit op 打包两个 bf16)。
- 每 warp 的 per-lane byte offset 偏 `warpid*stride`,使并发 warp 打 **disjoint packed slot**,不在同一 RMW word 上碰撞。
- 目标沿 disjoint 轴需 ≥ `num_warps*2*wave_size` 个 bf16 列:**4 warp ≥512**、**8 warp ≥256**,否则 warp aliasing、争用回归。
- 只消除 intra-CTA 争用;跨-CTA 写同一 tile 仍争用。
- bf16 global atomics 仅在 **gfx94+/gfx95+/gfx12+** 存在 → trace 时 arch-gate,否则回退 scaled-f32。

### 与 tile-size / L2 swizzle 的关系
- split-K 仅对 few-tile 大-K 场景补一 WG/tile 撑不满 CU 的欠订阅;不与 L2 swizzle 冲突(后者是纯 WG→tile 双射,bit-identical,只调 L2 residency)。
- skinny-GEMM 一并注意:hard-wire `tile_m=16` + 单 M-warp、把每个 wave 铺满 N;N-tile A-reuse 摊掉 A load,但 loop-carried 寄存器随 N-tile-repeat 倍增可压垮 occupancy → 配 `waves_per_eu` 旋钮。

## 跨 lane 原语与 MFMA/attention 数值:wave64 XOR shuffle、DPP butterfly、online-softmax log2

### wave64 cross-lane reduce（所有 gfx9xx 固定 64 线程）
- 全 wave reduction 用 **XOR shuffle 移位序列 `[32,16,8,4,2,1]`**（6 步）。
- API：`gpu.ShuffleOp(val, off, width_i32=64, mode='xor').shuffleResult` 做 peer 交换。
- block reduction 两级：intra-wave XOR shuffle → lane0 写每 wave partial 到 LDS → barrier → wave0 读并规约 `NUM_WAVES` 个 partial。
- 通用版（任意 `BLOCK_THREADS/WARP_SIZE`，wave64/wave32 都成立）：`shuffle_xor` wave reduce + per-wave LDS slot + 第二轮 wave reduce，越界槽用 `select(in_range, v, identity)` mask 掉。

### 免 LDS 的 cross-lane 原语
- **DPP butterfly**：`update_dpp` row-xor/shift，offsets `1,2,4,8` 配 associative op = wave reduce；Kogge-Stone 变体 = prefix sum。
  - control/mask immediate **必须编译期常量、仅 32-bit**（f32↔i32 用 bitcast 过渡）。
  - DPP 界在 **16-lane row 内**；跨 row 需 `ds_bpermute` 或 `permlane`。
- **`shuffle_xor` sub-warp reduce**：`width` 必须等于 group size 而**非** `WARP_SIZE`，否则 reduction 跨 token 边界。
- argmax tie 要确定性打破：`greater | (equal & lower_idx)`。

### online-softmax 数值（log2 空间 + 硬件 exp2）
- 全程 log2 空间用 `exp2`：`log2e = 1/ln2` 预乘。
  - `corr = exp2((m_old-m_new)*log2e)`
  - `p = exp2(fma(s, log2e, -log2e*m_new))`
  - `l_new = corr*l_running + sum(p)`
- exp 用 **single-issue `V_EXP`**（~1-ULP 足够）；最终 normalize 用 `rcp`。
- 计算 **`K@Q^T`** 使 S 落在 PV-aligned 寄存器布局，省掉两个 GEMM 之间的 transpose。
- S/P 全程留在寄存器（QK 与 PV 之间不落 LDS）；P 用 packed cast 构造，无 LDS round-trip。
- row max/sum 的 peer-reduce 用 **一次 `ds_bpermute`**（wave64 MFMA32 用 `lane^32`）over MFMA partner lane。

### FlashAttention 骨架 + 默认 config
- fwd：running softmax state 全在寄存器，**永不 materialize 完整 `S=QK^T`**。一个 work-group 拥有一段 `BLOCK_M` query slice；内层循环遍历 `BLOCK_N` 的 KV block，携带 `(m_running, l_running, O_acc)`。
- bwd 更重：重算 S/P + 产出 dQ/dK/dV；**dQ 是 KV block 上的 split-K 式 reduction** → reduction 策略是一等公民。
- decode：`BLOCK_M` 塌成几个 query token，主导成本是流式 KV（常 FP8）over paged block table。
- **默认 fwd config**：`BLOCK_M ∈ {128,256}`，`BLOCK_N=64`，head_dim 为 32 的倍数（≥64），**4 waves / 256 threads**，atom **32x32x16 (gfx950) / 32x32x8 (gfx942)**。
- wave size = 64，**禁止 import wave32 (RDNA/WMMA) 的 peer-reduction mask**。

### mxfp4：branchless E2M1 pack + E8M0 block scale + preshuffle
- **branchless f32→E2M1 (MXFP4) pack**：对 i32 bit-pattern masked-select（隔离 sign/abs，denormal+normal 谓词，normal 路径靠 odd-bit injection 做 RNE 再 shift，饱和到 `0x7`）。
  - per-32 E8M0 block scale：`shuffle_xor` butterfly max 求块内 max + `(254-e8m0)` reciprocal trick；**在 conversion 之前**乘入，然后 bit-pack nibble。
  - `_fp_headroom` 常量 **FP4 与 FP8 不同**，取错会把整张 tensor rescale `2^6`。
  - 该 routine 把 NaN/Inf 饱和到 max（**非 IEEE-faithful**）。
  - ⚠️ **E8M0 `0xFF = NaN`**：量化后的 exponent 在 bit-pack 前**必须 clamp**，否则一个 0xFF 会 poison 整块结果。（另见 `pitfalls/37`）
- **W4A16/W4A8 preshuffle**：把 B preshuffle 成每 lane 的 MFMA-K micro-step 读连续 8-byte（16-nibble）pack。
  - gfx950：`cvt_off_f32_i4`（SDWA `byte_sel` 一次 shift 覆盖全 8 nibble）+ `cvt_pk_bf16_f32`。
  - gfx942：shift-based f32→bf16 truncation（对 scaled int 精确，比 `truncf` 省 ~5 VALU）。
  - 把单个 `>>4` 提出循环；若把 x16 correction/groupwise scale 推迟到 epilogue，则 **epilogue 必须应用它**。

## 低精度融合 dequant 进 MFMA 内循环、验证策略、NOSTORE 归因

### dequant 融进 MFMA 内循环(非 epilogue)
- block-scale dequant 必须融进 MFMA 内循环,不是 epilogue:在 compute tile 内解码当前 K 的 per-block FP8 scale 切片,直接折进 accumulator。
- **tail tile 必须重新(FRESHLY)解码 scale**:沿用 prologue 的 scale state / 用了 stale scale 会污染最后一个 K block。
- FP8 requant 带来的 **1-5% mismatch 是预期容差,不是 bug**。

### 低精度对标要选同类 MFMA 指令
- 对标口径细节见 methodology/12-mxfp8-grouped.md「MX vs TW 唯一公平口径」。

### 低精度验证策略(profiling 不够,必须分步验)
1. 单独验 conversion 和 packing。
2. block-scaled 路径单独验 scale 处理。
3. kernel 数学对比高精度 / dequant 参考。
4. 只有前三步都过了才信性能数字。

### NOSTORE 归因 + 回归分离稳态 compute
- CUDA-graph bench 比 rocprof-min 干净:replay 确定性;rocprof-min 跨后端不可信(profiling 扰动曾误报 FLY 反超)。
- gated `NOSTORE` env(epilogue store 直接 return)做归因:**full 时间 − nostore 时间 = 暴露的 store 成本**。
- **斜率/截距回归(us/K-block)**:斜率 = 稳态 compute,截距 = 固定 prologue/epilogue 开销,借此分离两者。

### Branchless f32→E2M1 (MXFP4) pack / W4A16-W4A8 打包
- 详见本卡「跨 lane 原语与 MFMA/attention 数值」小节（branchless E2M1 pack / W4A16-W4A8 preshuffle）。

---
来源: 04-tn-wgrad-kernel.md, 09-perf-numbers.md, project_mxfp4_epilogue_store.md, gemm/optimization-directions.md, 13-primus-turbo-prod.md, flydsl-fp8-gemm-tuning/SKILL.md, flydsl-kernel-authoring/SKILL.md, optimization-directions.md, attention/optimization-directions.md, attention/overview.md, tool-rocprof/SKILL.md, project_mxfp8_wholeloop_port.md
