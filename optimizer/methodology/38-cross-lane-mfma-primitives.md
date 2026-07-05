# 跨 lane 原语与 MFMA/attention 数值:wave64 XOR shuffle、DPP butterfly、online-softmax log2

> 类别: 方法论 · 主题标签: cross-lane-reduce, attention-numerics, flash-attention, mxfp4-pack

## wave64 cross-lane reduce（所有 gfx9xx 固定 64 线程）
- 全 wave reduction 用 **XOR shuffle 移位序列 `[32,16,8,4,2,1]`**（6 步）。
- API：`gpu.ShuffleOp(val, off, width_i32=64, mode='xor').shuffleResult` 做 peer 交换。
- block reduction 两级：intra-wave XOR shuffle → lane0 写每 wave partial 到 LDS → barrier → wave0 读并规约 `NUM_WAVES` 个 partial。
- 通用版（任意 `BLOCK_THREADS/WARP_SIZE`，wave64/wave32 都成立）：`shuffle_xor` wave reduce + per-wave LDS slot + 第二轮 wave reduce，越界槽用 `select(in_range, v, identity)` mask 掉。

## 免 LDS 的 cross-lane 原语
- **DPP butterfly**：`update_dpp` row-xor/shift，offsets `1,2,4,8` 配 associative op = wave reduce；Kogge-Stone 变体 = prefix sum。
  - control/mask immediate **必须编译期常量、仅 32-bit**（f32↔i32 用 bitcast 过渡）。
  - DPP 界在 **16-lane row 内**；跨 row 需 `ds_bpermute` 或 `permlane`。
- **`shuffle_xor` sub-warp reduce**：`width` 必须等于 group size 而**非** `WARP_SIZE`，否则 reduction 跨 token 边界。
- argmax tie 要确定性打破：`greater | (equal & lower_idx)`。

## online-softmax 数值（log2 空间 + 硬件 exp2）
- 全程 log2 空间用 `exp2`：`log2e = 1/ln2` 预乘。
  - `corr = exp2((m_old-m_new)*log2e)`
  - `p = exp2(fma(s, log2e, -log2e*m_new))`
  - `l_new = corr*l_running + sum(p)`
- exp 用 **single-issue `V_EXP`**（~1-ULP 足够）；最终 normalize 用 `rcp`。
- 计算 **`K@Q^T`** 使 S 落在 PV-aligned 寄存器布局，省掉两个 GEMM 之间的 transpose。
- S/P 全程留在寄存器（QK 与 PV 之间不落 LDS）；P 用 packed cast 构造，无 LDS round-trip。
- row max/sum 的 peer-reduce 用 **一次 `ds_bpermute`**（wave64 MFMA32 用 `lane^32`）over MFMA partner lane。

## FlashAttention 骨架 + 默认 config
- fwd：running softmax state 全在寄存器，**永不 materialize 完整 `S=QK^T`**。一个 work-group 拥有一段 `BLOCK_M` query slice；内层循环遍历 `BLOCK_N` 的 KV block，携带 `(m_running, l_running, O_acc)`。
- bwd 更重：重算 S/P + 产出 dQ/dK/dV；**dQ 是 KV block 上的 split-K 式 reduction** → reduction 策略是一等公民。
- decode：`BLOCK_M` 塌成几个 query token，主导成本是流式 KV（常 FP8）over paged block table。
- **默认 fwd config**：`BLOCK_M ∈ {128,256}`，`BLOCK_N=64`，head_dim 为 32 的倍数（≥64），**4 waves / 256 threads**，atom **32x32x16 (gfx950) / 32x32x8 (gfx942)**。
- wave size = 64，**禁止 import wave32 (RDNA/WMMA) 的 peer-reduction mask**。

## mxfp4：branchless E2M1 pack + E8M0 block scale + preshuffle
- **branchless f32→E2M1 (MXFP4) pack**：对 i32 bit-pattern masked-select（隔离 sign/abs，denormal+normal 谓词，normal 路径靠 odd-bit injection 做 RNE 再 shift，饱和到 `0x7`）。
  - per-32 E8M0 block scale：`shuffle_xor` butterfly max 求块内 max + `(254-e8m0)` reciprocal trick；**在 conversion 之前**乘入，然后 bit-pack nibble。
  - `_fp_headroom` 常量 **FP4 与 FP8 不同**，取错会把整张 tensor rescale `2^6`。
  - 该 routine 把 NaN/Inf 饱和到 max（**非 IEEE-faithful**）。
  - ⚠️ **E8M0 `0xFF = NaN`**：量化后的 exponent 在 bit-pack 前**必须 clamp**，否则一个 0xFF 会 poison 整块结果。（另见 `pitfalls/37`）
- **W4A16/W4A8 preshuffle**：把 B preshuffle 成每 lane 的 MFMA-K micro-step 读连续 8-byte（16-nibble）pack。
  - gfx950：`cvt_off_f32_i4`（SDWA `byte_sel` 一次 shift 覆盖全 8 nibble）+ `cvt_pk_bf16_f32`。
  - gfx942：shift-based f32→bf16 truncation（对 scaled int 精确，比 `truncf` 省 ~5 VALU）。
  - 把单个 `>>4` 提出循环；若把 x16 correction/groupwise scale 推迟到 epilogue，则 **epilogue 必须应用它**。

---
来源: flydsl-kernel-authoring/SKILL.md, optimization-directions.md, attention/optimization-directions.md, attention/overview.md
