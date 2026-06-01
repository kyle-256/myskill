---
name: gfx950-vmcnt-race-debug
description: 调试 / 修复 AMD MI355X (gfx950) 上 grouped / persistent kernel 里 buffer_load_lds 跟 scratch_load 的 vmcnt race，以及 buffer_load_lds 跟 ds_read 之间的 LDS WAR race，以及跨 die (XCD) L2 coherence race —— 症状是低概率 (0.001-3%) bit-non-deterministic 输出。当用户在 Primus-Turbo / HipKittens 等 gfx950 kernel 上看到 "race / non-deterministic / bit-不一致 / rerun 不稳定 / lds 写错地方 / 不同 cu 不同 die 数据不一致" 时使用此 skill。覆盖三类 race：(1) vmcnt 不 FIFO（scratch_load vs buffer_load_lds，需要消除 VGPR spill）；(2) LDS WAR（ds_read 在飞时 buffer_load_lds 覆盖写，需要 wait_lgkmcnt drain）；(3) 跨 die L2 coherence（复用 workspace 留下 stale L2 行，需要 __threadfence，但要拆掉 per-tile fence tax）。配合 remote-sync / remote-mlperf-gptoss skill。
---

# gfx950 vmcnt race 调试

## TL;DR

gfx950 上 race 的硬件机制是 **`scratch_load` 和 `buffer_load_dwordx? ... offen lds` 共享 vmcnt 但不保证 FIFO 完成顺序**。编译器对 spill 回填发的 `s_waitcnt vmcnt(N)` 假设 FIFO，N 假设错了 → `v_readfirstlane vXX, ...` 读到 stale value → `s_mov_b32 m0, sNN` 用了 stale m0 → `buffer_load_lds` 写到错的 LDS 地址 → SMEM 数据错位 → MFMA 输入污染 → output bit 不一致。

**真正的修法**（已在 Primus-Turbo `dev/kyle_mxfp8_gg_pr` 验证）：从源头消除 VGPR spill —— 不让 scratch_load / scratch_store 出现在 disasm 里。**任何走 vmcnt 通道的 spill 都不可接受**。

具体技术（按重要性排序）：
1. SMEM 地址全 SGPR：`__builtin_amdgcn_readfirstlane(warp_id * STRIDE)` + 删 `sts_offsets` 里的 lane_id 部分（`buffer_load_lds` 硬件 auto-stride）
2. Outer kernel 解析 per-group 基址，inner 只接收 resolved ptr（i64 offset 链不进 inner）
3. 输出类型用硬件转换或 raw-bit truncation，不用 `hip_bfloat16(float)` 这种带软件分支的

**reviewer 硬约束**：VGPR spill = 0、scratch / private_segment = 0、不动单 GEMM 的 aggressive pipeline、不能改 vmcnt/lgkmcnt 通道分工、pinned 区 v[112:255] 不能踩。

## 症状识别

低概率（0.001-1%）的 bit-non-deterministic：相同输入跑 N 次，rerun 之间结果有 bit 差异。**不**是数值漂移（不是 NaN / Inf / 显著 SNR 下降），是少数 element 偶尔不一致。在 grouped / persistent kernel 上特别常见；dense single-GEMM 通常不会有。

典型测试方式（`repro_race_*.py`）：
```python
ref = kernel(inputs)
torch.cuda.synchronize()
races = 0
for _ in range(ITERS):
    out = kernel(inputs)
    torch.cuda.synchronize()
    try:
        torch.testing.assert_close(out, ref, rtol=0, atol=0)  # 严格 bit-exact
    except AssertionError:
        races += 1
print(f"{races}/{ITERS}")
```

`ITERS` 用 50000+ 才有统计意义。0.005% 量级的 race 在 10K iters 看起来像 0/N，要 50K-200K 才稳定可见；多 GPU 跑确认不是单卡 fluke。

## 根因机制

GFX950 上：
- `scratch_load`（编译器 VGPR/SGPR 溢出的回填）走 **vmcnt** 计数
- `buffer_load_dwordx? ... offen lds`（LDS-direct GMEM→SMEM 路径）也走 **vmcnt**
- **两者之间 vmcnt 不保证 FIFO 完成顺序**

所以编译器对 spill 回填发的 `s_waitcnt vmcnt(N)` 不安全：N 假设按发射顺序 FIFO 完成，scratch_load 可能还在飞、`v_readfirstlane vXX` 已经读了 stale 的 vXX 进 SGPR，`s_mov_b32 m0, sNN` 用了 stale m0，下一条 `buffer_load_dwordx? ... offen lds` 写到错的 LDS 地址。

**前提**：必须有 spill 走 scratch（即 disasm 里有 `scratch_store` / `scratch_load`）。SGPR-to-VGPR-lane spill (`v_writelane` / `v_readlane`) 不走任何 counter，**不会** race。Race-prone 的就是 `scratch_*` 这一条。

## Disasm 验证

提取 GPU code + reg notes：

```bash
SO=build/lib/libprimus_turbo_kernels.so
LINE=$(/opt/rocm/bin/roc-obj-ls $SO 2>/dev/null | awk '/gfx950/{print; exit}')
OFFSET=$(echo "$LINE" | sed -n 's/.*offset=\([0-9]*\).*/\1/p')
SIZE=$(echo "$LINE" | sed -n 's/.*size=\([0-9]*\).*/\1/p')
dd if=$SO bs=1 skip=$OFFSET count=$SIZE 2>/dev/null > /tmp/gpu.hsaco

/opt/rocm/lib/llvm/bin/llvm-objdump -d --triple=amdgcn-amd-amdhsa --mcpu=gfx950 \
    /tmp/gpu.hsaco > /tmp/gpu.disasm
/opt/rocm/lib/llvm/bin/llvm-readelf --notes /tmp/gpu.hsaco > /tmp/notes.txt

# reg counts per kernel symbol
grep -B1 -A6 "<kernel-symbol-substring>" /tmp/notes.txt | \
    grep -E "\.name|\.private_segment_fixed_size|\.sgpr_count|\.sgpr_spill_count|\.vgpr_spill_count"
```

Race 的 disasm 签名：
- `private_segment_fixed_size > 0` → **race 风险，必须修**
- `vgpr_spill_count > 0` → 同上
- 主循环里 `scratch_load_dword vXX, off, off offset:N` 夹在多个 `buffer_load_dwordx? ... offen lds` 之间
- 紧跟 `s_waitcnt vmcnt(N>0)` + `v_readfirstlane sNN, vXX` + `s_mov_b32 m0, sNN` + `buffer_load_dwordx? ... offen lds`

判断单个 kernel：
```bash
grep -nE "scratch_(load|store)" /tmp/<kernel>.disasm | head
grep -nE "buffer_load_dwordx?[0-9]* .* offen lds" /tmp/<kernel>.disasm | head
# 两者行号相邻或交错 = race signature
```

`v_writelane`/`v_readlane` 出现 OK，那是 SGPR→VGPR-lane spill，不走 vmcnt。

## 修法（消除 VGPR spill 到 scratch）

目标 reg notes：
```
.private_segment_fixed_size: 0
.sgpr_spill_count:           0  (理想；如有也只能是 lane-spill 到 v111 这种)
.vgpr_spill_count:           0  (硬约束 — 不能踩 pinned 区)
```

### 1. SMEM 地址用 SGPR，不让走 VGPR

最大头的 spill 来源：persistent kernel 里 8 个 SMEM subtile 地址（`0x2000/0x4000/0x6000/0x8000 | warp_id<<11 | lane_id<<4`）的 VGPR 计算。这些地址要进 m0，最终 SGPR-only。

源码里通常写成：
```cpp
const uint32_t sts_warp_base = warp_id * MFMA_SIZE_M * MFMA_SIZE_K;  // VGPR
sts_offsets[i] = i * 1024 + lane_id * 16;                             // VGPR
load_gmem_to_smem_srd<16>(srd, ldg_off, base + sts_warp_base + sts_offsets[i], soff);
```

编译器：算 VGPR sum → readfirstlane（取 lane 0 = base + warp_base + i*1024）→ m0。计算过程占 VGPR，pressure 一高就 spill。

**关键观察**：`buffer_load_lds` 硬件按 data_size 自动 per-lane stride（b128 = 16 B/lane，b32 = 4 B/lane），**所以源码里加的 `lane_id*16` 是冗余的**——readfirstlane 拿 lane 0 = 0，硬件自己 stride。

修法两步：
```cpp
// (a) 删 sts_offsets 里的 lane_id 部分（保留 i*1024 这种 wave-offset）
sts_offsets[0] = 0;
sts_offsets[1] = 1024;  // = 16 B/lane × 64 lanes = 一个 wave 的 LDS 写跨度

// (b) warp-id 算术 readfirstlane 强制 SGPR
const uint32_t sts_warp_base =
    __builtin_amdgcn_readfirstlane(warp_id * MFMA_SIZE_M * MFMA_SIZE_K);

// 结果：base + sts_warp_base + sts_offsets[i] 全 SGPR/immediate，无 VGPR 中间
```

scale 路径同理：`smem_byte_offset = (warp_id * 64 + lane_id) * 4` → `readfirstlane(warp_id * 64 * 4)`，删掉 `+ lane_id`。

实测：grouped fwd 17 vgpr_spill / 72 scratch → **0 / 0**。

### 2. Outer 解析 per-group 基址，inner 只接收 resolved ptr

Persistent kernel 里 outer dispatcher 调 inner `compute_tile`（必须 `__forceinline__`，`__noinline__` -25% perf）。如果 inner 自己做 i64 group offset 计算：

```cpp
// inner 里：
const int64_t group_offset = ...split-load + recombine...;
const int64_t m_global = group_offset + pid_m;
const int64_t b_group_off = (int64_t)group_id * n * k;          // i64 mul
const AType *a_base = a_ptr + m_global * k;
const BType *b_base = b_ptr + b_group_off + (int64_t)pid_n * k;
// ... 一堆 i64 临时值 live 直到 SRD 构造完
```

i64 中间值 + 8 个 GMEM ptr live 时间长，加上 outer 持有的 dispatcher 状态，VGPR 顶不住。

修法：把 per-group 解析挪到 outer，inner 只接 resolved 的 5 个 base ptr：
```cpp
// outer 里（每 tile 一次）：
const int64_t g_off = readfirstlane_split_recombine(group_offs_ptr[group_id]);
const AType    *a_grp_ptr   = a_ptr + g_off * k;
const BType    *b_grp_ptr   = b_ptr + (int64_t)group_id * n * k;
const uint32_t *a_s_grp_ptr = a_s_ptr + g_off * scale_cols;
// ... 等等
compute_tile(tile, ..., a_grp_ptr, b_grp_ptr, a_s_grp_ptr, b_s_grp_ptr, c_grp_ptr,
             pid_m, pid_n, M_g, M_g_in, n, k);

// inner 里：
const AType *a_base = a_grp_ptr + (int64_t)pid_m * k;  // 一步加法，无 i64 mul / split-load
```

Inner 的 SGPR/VGPR live set 立刻接近 dense single-GEMM 水平。

### 3. 输出类型转换：避开有软件分支的 ctor

```cpp
// hip_bfloat16(float f) 默认 ctor —— round-to-nearest-even：
//   if (~u.int32 & 0x7f800000) { u.int32 += 0x7fff + ((u.int32>>16)&1); }
//   else if (u.int32 & 0xffff)  { u.int32 |= 0x10000; }
// 软件分支 → SCC 比较结果被 lane-spill 到 v111 → 7 个 v_writelane/v_readlane
```

`__half(float f)` 是单条硬件指令 `v_cvt_f16_f32`，bf16 缺这个对应。

修法：specialize `float → CType` 转换。bf16 用 raw-bit truncation：
```cpp
template <typename C> static __device__ __forceinline__ C float_to_ctype(float f) {
    if constexpr (std::is_same_v<C, dtype::bfloat16>) {
        C r;
        r.data = static_cast<uint16_t>(__builtin_bit_cast(uint32_t, f) >> 16);
        return r;  // truncation：跟 v_cvt_f32_bf16 truncate mode 一致
    } else {
        return C(f);  // half 走默认硬件转换
    }
}
```

GEMM 输出已经是 lossy（fp32 → 16-bit），truncate vs round-to-even 半位精度差，SNR 不变。

或者 gfx950 有 `v_cvt_pk_bf16_f32`（pack 2 个 fp32 → 2 个 bf16，硬件 round），更精确但需要 pair 处理。

### 4. 分支去掉

`if (some_ptr != nullptr)` 这种 launcher 决定的分支，挪到 outer 一次。或者干脆让 launcher always 传有效 ptr，加一个 mask 参数走 branchless：

```cpp
// 而不是 if (c_group_offs_ptr != nullptr) { ... padding ... } else { ... }
// 用 mask：
const int32_t M_g_in = (M_g + c_padding_align_mask) & ~c_padding_align_mask;
// mask = 127 → 对齐到 128；mask = 0 → M_g_in = M_g
```

少一个分支 = 少一对 SCC 比较 → 少 lane-spill 候选。

## 验证清单

```bash
# 1) reg counts: 全 0 才行
grep -B1 -A6 <kernel> /tmp/notes.txt | \
    grep -E "private_segment_fixed_size|sgpr_spill_count|vgpr_spill_count"
# 期望：
#     .private_segment_fixed_size: 0
#     .sgpr_spill_count:           0  (理想；非零必须是 v_writelane 落 v[0:111] 内的 lane spill)
#     .vgpr_spill_count:           0

# 2) disasm: 确认主循环里没有 scratch_*
grep -cE "scratch_(load|store)" /tmp/<kernel>.disasm
# 期望：0

# 3) race rate: fwd/dgrad/wgrad 各 50K-200K iters, 3-4 GPU 跑
for g in 0 1 2 3; do
    HIP_VISIBLE_DEVICES=$g python repro_race_grouped_fwd.py 50000
done
# 期望：每个都 0/N
```

## 反例（**不要**走的路 — reviewer rejected 列表）

| 方案 | 为什么 reject |
|---|---|
| `c_group_offs == nullptr` mode-switch + `clobber_vgpr_one` 占 pinned VGPR | "藏 spill"，不修根因；后续小改易复现 |
| `__noinline__ compute_tile` | 0/N race 但 -25% perf（args 通过 scratch 传） |
| 全 main loop `buffer_load_lds → 2-step (buffer_load → VGPR → ds_write)` | 把 ds_write 塞进 lgkmcnt → 跟 ds_read 共享 counter，aggressive pipeline 必须降级；新增 VGPR 可能踩到 pinned [112:255]；buffer_load_lds 本来就是零 VGPR + 全异步 |
| Prologue 2-step（即使只在 prologue） | 同上根因，counter 通道分工被破坏 |
| `-mllvm -amdgpu-enable-merge-m0=true` | 工程 flag，不修根因 |
| `-mllvm -sink-insts-to-avoid-spills=true` | 同上 |
| `tile.reserve_pinned_regs()` 多处调用 | 是占用 pinned 区的 hack，不解决"为什么编译器要 spill" |
| Mid-iter 主循环加 `s_waitcnt vmcnt(0)` | race 不归 0、-21% perf |
| 多个 `s_waitcnt vmcnt(0)` 在 prologue 4× drain | race ~0.01% 不归 0，且没修根因 |

reviewer 核心观点：**"硬件 vmcnt 行为是约定，编译器在合理 pressure 下不该 spill。出现 spill 是 kernel 写法逼出来的，去 kernel 源头改。绕通道、加 flag、占 pinned 区都是把症状藏起来。"**

## Pinned 区约束（来自 reviewer 的硬规则）

mxfp8 GEMM kernel 里 v[112:255] 是 pinned A/B data + scale double-buffer（layout 见下）。**不能踩**：
- 任何新增 VGPR 不能落到 [112:255]
- VGPR spill **绝对不允许**（编译器 spill 候选 v[0:111]，但 spill 路径走 scratch 就 race；任何 spill 等于失败）
- `clobber_vgpr_one<N>()` 用在 [112:255] 范围是反模式（占 pinned）

```
v[0:111]    compiler-managed
v[112:115]  AS0    v[116:119]  AS1     A scale buf 0/1  (4 + 4)
v[120:123]  BS0    v[124:127]  BS1     B scale buf 0/1
v[128:159]  A0     v[160:191]  A1      A data buf 0/1   (32 + 32)
v[192:223]  B0     v[224:255]  B1      B data buf 0/1
a[0:255]    C accumulator (256 fp32)
```

## 测试 / 容器约定

- gfx950 节点：参见 `claim-mi355x-node` skill 找真闲节点
- Race + perf 验证：用 `mlperf_gptoss2` 容器（见 `remote-mlperf-gptoss` skill）
- mxfp8 PR 在 `/workspace/code/code3/Primus-Turbo/`（见 `remote-sync` skill；本地 `mxfp8/Primus-Turbo` 自动映射）
- GPU 选**真闲**的（`rocm-smi --showuse` GPU%=0、`--showmemuse` VRAM% < 50）；race rate 在 noisy GPU 上会被 OOM 干扰
- bench 用 `scripts/spot_bench_kernel.py`（kernel-only TFLOPS，不算 quant pre-process）

## 真实案例 (Primus-Turbo `dev/kyle_mxfp8_gg_pr`, 2026-05)

gpt_oss-20B Expert shape (B=4 M=2048 N=5760 K=2880)：

| Kernel | reg notes (sgpr/sgpr_spill/vgpr_spill/private_seg) | race |
|---|---|---|
| Single GEMM (reference) | 81 / 0 / 0 / 0 | 0 |
| Grouped fwd bf16 (PR base 847872c) | 106 / 7 / 10 / 44 | ~1% |
| Grouped fwd half (PR base) | 106 / 0 / 17 / 72 | ~1% |
| Grouped wgrad (PR base) | 106 / 0 / 17 / 72 | — |
| **修后**：grouped fwd bf16 | **106 / 0 / 0 / 0** | **0/200K** |
| **修后**：grouped fwd half | **106 / 0 / 0 / 0** | **0/N** |
| **修后**：grouped wgrad | **104-105 / 0 / 0 / 0** | — |

Perf（B=16 M=2048 N=4096 K=7168）: Fwd 1435 / Dgrad 1416 / Wgrad 2013 TFLOPS。SNR 28.23 / 28.23 / 28.08 dB。

修复 commit: `abd3833` (主修) + `2c890b0` (refactor: 删 compute_sts_offsets) on `dev/kyle_mxfp8_gg_pr`。

---

# 第二类 race: LDS WAR (ds_read vs buffer_load_lds)

(2026-05 修复，commit `bb48d3f` on `dev/kyle_mxfp8_gg_pr`)

## TL;DR

跟第一类 vmcnt FIFO race 完全不同的另一类。spill 已经修干净（reg notes 全 0）但 race 仍然存在（~0.004% per-tile，多 config 累积 30-50% per-test-session）。

**机制**：`buffer_load_lds`（vmcnt-tracked LDS write）和 `ds_read_pinned`（lgkmcnt-tracked LDS read）目标同一段 LDS 时，`s_barrier` **只同步 wavefront 执行位置，不强制 drain lgkmcnt**。所以 warp X 的 `ds_read` 还在飞、`buffer_load_lds` 已经 commit 到 LDS → warp X 拿到 post-write 数据 → MFMA 输入被未来 K-tile 数据污染。

**修法**：在两个具体位置发 `wait_lgkmcnt<0>()` 把所有 in-flight ds_reads drain 掉，**再**让任何 warp 发 LDG：
1. `phase_mfma_lds_ldg` 的 mid-phase WAR barrier（原来只有 `s_barrier`）
2. `compute_tile` prologue 里 `if (k_iters > 2)` 块**之前**（原本 `wait_lgkmcnt<0>` 在块之后，顺序错了）

修后：single-GEMM 50/50 + grouped (balance=[True,False], full fwd+bwd) 50/50 都 0 race；perf 回归 < 1%（Fwd −0.67%, Bwd −0.92% on grouped MoE bench）。

## 症状区分

跟第一类的关键差别：
- **第一类（vmcnt FIFO）**：disasm 里有 `scratch_load` / `scratch_store`，reg notes `vgpr_spill > 0` 或 `private_segment > 0`。修法是消 spill。
- **第二类（LDS WAR）**：reg notes 已经全 0，但 race 仍然存在。disasm 里 `buffer_load_lds` 紧跟 `s_barrier`，没有 `s_waitcnt lgkmcnt(0)`。修法是补 lgkmcnt drain。

如果 spill 已经修干净仍然 race，要怀疑是第二类。

## 根因机制

LDS bus 上：
- `ds_read_b32/b64/b128` (LDS → VGPR)：lgkmcnt-tracked，issue → 数据回 VGPR 之间有 latency
- `buffer_load_dwordx? ... offen lds` (GMEM → LDS)：vmcnt-tracked，issue → 数据回到 LDS 之间有 GMEM RTT 的 latency

`s_barrier` 是 wavefront-level 的同步，只保证所有 warp **到达** barrier 指令，**不**保证：
- 各 warp 已经 issue 的 ds_read 已经回 VGPR（lgkmcnt > 0 时 ds_read 还在飞）
- 各 warp 已经 issue 的 buffer_load_lds 已经 commit 到 LDS（vmcnt > 0 时 LDS-write 还没到）

危险序列（warp X 和 warp Z 同 WG）：
```
warp X: ds_read_pinned ... addr=Y       (lgkmcnt++，read 在飞)
warp X: ... 算 MFMA ...
warp Z: ... 算 MFMA ...
warp Z: s_waitcnt vmcnt(0)               (drain 之前的 LDG)
warp Z: s_barrier                         (warp X 也已到达，lgkmcnt 仍 > 0)
warp Z: buffer_load_dwordx4 ... offen lds  addr=Y  (LDS write 飞向 Y)
                                          ↑
                          warp X 的 ds_read 此时如果还没 dispatch，
                          LDS bus 仲裁可能让 LDG 先 commit，
                          warp X 读到 post-write 数据 → race
```

实际触发频率取决于 LDS bank 排队、GMEM 命中率等运行时因素，所以**每 tile ~0.004%** —— 看起来 0/N 但 50K iters + 多卡能稳定见到。

## 出现位置

只要 source 里出现 "几个 warp 的 ds_read 后面紧跟着同 WG 别的 warp 的 buffer_load_lds 写同一段 LDS" 都是嫌疑：

### A. `phase_mfma_lds_ldg` 内部（MXFP8 主循环 / Epi1）

structure：
```cpp
mfma; ds_read; ds_read; mfma; ds_read; ds_read; ... (前 6 个 ds_read 写 PIN_NEXT)
mfma; ds_read; ds_read;
mfma; ds_read; ds_read; (scale 4 个)
__builtin_amdgcn_s_barrier();          // 原来只有这个
mfma; buffer_load_lds;                  // 这些 LDG 跟前面的 ds_read 抢同一段 LDS
mfma; buffer_load_lds;
...
```

各 warp 的 ds_read 写 PIN_NEXT_D/S，但**地址来源**是 `a/b_smem_tile[cur][warp_n+2]` —— 跟同 WG 别的 warp（warp 2/3 vs 0/1）的 LDG `a/b_smem_tile[cur][2,3]` 落在**同一对 sub-tile** 上。s_barrier 后 lgkmcnt 仍 > 0 → race。

修法：
```cpp
- // WAR barrier
+ // WAR barrier: drain ds_reads before any warp issues LDG to the
+ // same LDS regions; s_barrier alone doesn't sync lgkmcnt.
+ wait_lgkmcnt<0>();
  __builtin_amdgcn_s_barrier();
```

### B. `compute_tile` prologue 的 "if k_iters > 2" LDG 块

原来的写法（**bug**）：
```cpp
// prologue: 起 ds_reads 把 K-tile 0 的 PIN_A0/B0 prefetch 到 VGPR
load_data_subtile_pinned<PIN_A0>(a_smem_tile[cur][warp_m], lds_offsets);  // ds_read
load_data_subtile_pinned<PIN_B0>(b_smem_tile[cur][warp_n], lds_offsets);  // ds_read
load_scale_subtile_pinned<PIN_AS0>(...);                                   // ds_read
load_scale_subtile_pinned<PIN_BS0>(...);                                   // ds_read

if (k_iters > 2) {
    // LDG 写到 a/b_smem_tile[cur][0,1] 的 H=0 区域 —— 跟上面 warp_m=0,1 的 ds_read 读位置重叠
    load_a_gmem_to_smem_half_srd<0>(a_srd, ldg_off, a_smem_tile[cur], 2*DATA_STRIDE);
    ...
}
wait_lgkmcnt<0>();   // ← 顺序错了，drain 应该在 LDG 之前
```

**race**：warp 0/1 (`warp_m ∈ {0,1}`) 的 ds_read 还在飞，warp 0/1 的 LDG（写 `cur[0,1]`，sub-tile 0 和 1）已经 issue，LDG 先 commit → warp 0/1 的 ds_read 读到 K-tile 2 数据而不是 K-tile 0 → MFMA 算错。

修法：drain 调到 LDG **之前**：
```cpp
load_*_subtile_pinned<PIN_*0>(...);  // 4 个 ds_read

+ // Drain prologue ds_reads before LDG below — both target cur[0,1].
+ wait_lgkmcnt<0>();
  if (k_iters > 2) {
      load_a_gmem_to_smem_half_srd<0>(a_srd, ldg_off, a_smem_tile[cur], 2*DATA_STRIDE);
      ...
  }
- wait_lgkmcnt<0>();
```

两处都改完才会归零。Fwd / dgrad / wgrad 三个 kernel 都要改（共享 `phase_mfma_lds_ldg` 一次性覆盖；prologue 那处每个 compute_tile 都要单独加）。

## 定位方法学（二分逐层缩小）

整套定位流程花了大半天，记下来下次同类问题省时间：

### Step 1: 跨 kernel 定位 —— pytorch ref 替换

每个 autograd `apply` 里 fwd / dgrad / wgrad 各自调一个 C++ kernel。先在 python 层加 env-var dispatch：

```python
def _ref_grouped_gemm_mxfp8(...):
    # 用 fp32 dequant + matmul + cast 写个 deterministic ref
    ...

def _turbo_grouped_gemm_mxfp8(..., _stage="fwd"):
    if os.environ.get(f"PRIMUS_TURBO_MXFP8_{_stage.upper()}_REF") == "1":
        return _ref_grouped_gemm_mxfp8(...)
    return torch.ops.primus_turbo_cpp_extension.turbo_grouped_gemm_fp8(...)

# fwd call:    _turbo_grouped_gemm_mxfp8(..., _stage="fwd")
# dgrad call:  _turbo_grouped_gemm_mxfp8(..., _stage="dgrad")
```

然后矩阵化跑 deterministic 测试：

| FWD_REF | DGRAD_REF | balance=True | balance=False | 结论 |
|---|---|---|---|---|
| 1 | 1 | pass | pass | ref 自身 deterministic ✓ |
| 1 | 1 | pass | pass | wgrad（唯一真 kernel）clean on balance=True，race on balance=False |
| 1 | 0 | r1 fail | — | dgrad 真 → race ⇒ fwd/dgrad 共享 C++ kernel 有 race |
| 0 | 1 | r1 fail | — | fwd 真 → race ⇒ 同上 |

这个矩阵能锁定 race 在哪个 C++ kernel 里。本 case 锁定到两个 kernel：`turbo_grouped_gemm_mxfp8`（fwd/dgrad 共享）和 `turbo_grouped_gemm_mxfp8_wgrad`。

**调试完之后必须 commit 之前删掉 ref 路径 + env-var**（debug-only 代码不该入 prod 库）。

### Step 2: 加 single-GEMM deterministic 覆盖

只有 grouped GEMM 之前有 deterministic 测试，single GEMM mxfp8 之前**没人测过**。补一个最小 deterministic 测试发现 single GEMM 也 race（~30% session fail）。这说明 race 不在 grouped 特有的 persistent-loop / per-group 解析里，而是在 **shared kernel structure**（phase_mfma_lds_ldg / 单 GEMM compute_tile）里。

复杂度立刻下降：single GEMM 更小、迭代更快、没有 grouped 特有的 boundary store / per-group 复杂度。后续所有定位用 single GEMM 跑。

### Step 3: kernel 内部子结构定位 —— gate 掉嫌疑代码块

在 wgrad 上观察到一个细节：`balance=True` 时 `k_iters=1`，`if (k_iters > 2)` 那个 LDG 被 skip → 不 race；`balance=False` 出现 `k_iters > 2` 的 group → race。

进一步验证：把 Epi1 LDG 块整个 gate 掉：
```cpp
- if (k_iters > 2) {
+ if (k_iters > 999999) {  // DEBUG: bisect — skip Epi1 LDG path
```

跑测试。**新现象**：assert_close 全过（deterministic 恢复），但 SNR 失败（output 数值错，因为 LDG 提供了真正参与 MFMA 的数据）。

→ 确认 race 在 LDG 块里。下一步是修而非删 —— 用 `wait_lgkmcnt<0>` 实际 fix 那个块的同步漏洞。

### Step 4: 增量验证

每改一处 wait_lgkmcnt 跑 30 runs（一次跑只跑 deterministic 测试，~6-10s/run），统计 session-level fail rate：
- 无 fix: 30 runs ~10 fail (~33%)
- 加 prologue drain: 30 runs ~1-3 fail (~7%)
- + WAR barrier drain: 30 runs 0 fail
- 再 50 runs 0 fail 确认

不要单测 1 次就声明修好 —— race rate 在 0.004% 量级时 1 run 有概率运气过。**至少 30 runs**，要更保险 50。

## 反例（这次又试过的、**不要**走的路）

| 方案 | 为什么不 work |
|---|---|
| 把 `wait_vmcnt<12>` (main loop end) tighten 到 `wait_vmcnt<0>` | -23% perf，race 没归零（race 不是 vmcnt 问题） |
| 把 Epi1 `wait_vmcnt<6>` (split-drain) tighten 到 `wait_vmcnt<0>` | race rate 从 ~33% 降到 ~30%，几乎没用 |
| 在 main loop 每个 iter 末尾加 `wait_lgkmcnt<0>` | race 反而 **变多**（53% pass vs 93% pass）—— 改了 timing 暴露了别的窗口 |
| compute_tile 开头加 `wait_lgkmcnt<0> + s_barrier` 作为持久 loop hygiene | catastrophic：所有测试全 fail。原因未完全弄清，跟 outer kernel 早 return 时 barrier 配对有关，规避 |
| 在 MFMA 后加 `clobber_agpr_one<PIN_ACC + 0..3>` | race rate 改善但不归零（0.19% → 0.05%）—— 不是这条路 |

reviewer 硬规则（沿用第一类的约束）：
- **`asm volatile : : : "memory"` 不允许**（用户明确禁止）
- AGPR clobber **加了必须把 race 修到零**，否则不接受
- 测试**有错立即停**（pytest `-x`），别一直跑

## 验证清单（第二类专用）

```bash
# 1) single GEMM mxfp8 deterministic: 30+ runs 0 fail
for i in $(seq 1 30); do
    pytest -x tests/pytorch/ops/test_gemm_fp8.py::test_gemm_fp8_mx_blockwise_deterministic \
        --deterministic-only 2>&1 | grep -E "FAILED|passed" | tail -1
done

# 2) grouped GEMM mxfp8 (balance=[True,False], no env-var refs, full fwd+bwd)
#    30+ runs 0 fail
for i in $(seq 1 30); do
    pytest -x tests/pytorch/ops/test_grouped_gemm_fp8.py::test_grouped_gemm_fp8_mx_blockwise_deterministic \
        --deterministic-only 2>&1 | grep -E "FAILED|passed" | tail -1
done

# 3) perf 回归 < 1%（grouped MoE bench）
python benchmark/ops/bench_grouped_gemm_turbo.py --dtype fp8 --granularity mxfp8
# 跟未修版本的 avg Fwd/Bwd TFLOPS 对比
```

session-level fail rate 计算：每个 pytest 命令是一个 session，覆盖 N 个 config × 10 reps each。session fail = 至少 1 个 config 跑出 mismatch。

## 真实案例 (Primus-Turbo `dev/kyle_mxfp8_gg_pr`, 2026-05)

测试 config：
- single GEMM mxfp8: m=[255,257,512,1024], n=[256,512,1024,4096], k=[256,1024,4096], TURBO backend (skip 不满足 m/n%128, k%128, k≥384 的)
- grouped GEMM mxfp8: B=[1,8], M=[256,1024], NK=[(2048,1536),(4096,7168)], balance=[True,False], 全 fwd+bwd

| 状态 | single 30 runs | grouped 30 runs |
|---|---|---|
| 修前 | 9 fail (30%) | ~10 fail (~33%) |
| 加 prologue `wait_lgkmcnt<0>` | — | ~1-3 fail (~7%) |
| 加 WAR barrier `wait_lgkmcnt<0>` | 0/30 | 0/30 |
| 50 runs 复确认 | 0/50 | 0/50 |

Perf（grouped MoE bench, avg 跨所有 case）：
- Fwd: 941.02 → 934.67 TFLOPS (−0.67%)
- Bwd: 1109.91 → 1099.66 TFLOPS (−0.92%)

修复 commit: `bb48d3f` on `dev/kyle_mxfp8_gg_pr`（+ `17df201` style pre-commit pass）。

**回顾**：第二类 race 跟第一类完全独立，第一类修 spill / SGPR 化地址，第二类修 lgkmcnt drain。两类都修了才 0 race。下次遇到 spill 全 0 但还是有 race，第一反应是查 `phase_mfma_lds_ldg` / prologue 这种 "ds_read + buffer_load_lds 同地址" 模式。

---

# 第三类 race: 跨 die (XCD) L2 coherence

(2026-06，commit `d7b149a` on `dev/kyle_mxfp8_gg_pr`)

## TL;DR

又一类独立 race。spill 全 0、LDS WAR 也修了（前两类干净），但**第二次调用** kernel 偶发 bit 不一致。

**机制**：MI355X 一个 node = **8 个 die (XCD)，每个 die 自己一块 L2**，die 间不自动 coherent。persistent kernel 的 256 个 block 散在 8 个 die 上跑。当 **scale workspace（或任何中间 buffer）跨调用复用**：上一次调用把数据写进 GMEM，留下的副本还缓在某些 die 的 L2 里；这一次 `buffer_load_lds` 命中那个 die 的 stale L2 行 → 读到上一次的数据 → MFMA 污染 → bit 不一致。第一次调用（L2 冷）不出，复用 workspace 的后续调用才出。

用户原话点破：**"race 和数据分布在不同 cu 有关，不同 cu-group / die 间同步要 fence；数据预分配好就不会 race。"**

**修法**：kernel 入口发一次 `__threadfence()`（gfx950 lower 成 `buffer_wbl2 sc1` + `buffer_inv sc1` = flush + invalidate 本 die L2）→ 强制这次的 load 从 GMEM 取新数据，不吃 stale L2。

## 症状区分（vs 前两类）

| | 第一类 vmcnt FIFO | 第二类 LDS WAR | 第三类 die L2 |
|---|---|---|---|
| reg notes | spill > 0 | 全 0 | 全 0 |
| disasm 标志 | `scratch_*` | `buffer_load_lds` 紧跟裸 `s_barrier` | 入口/prologue 无 `buffer_inv` |
| 触发条件 | 单次调用即出 | 单次调用即出 | **复用 workspace 的第 2+ 次调用**才出 |
| 修法 | 消 spill | `wait_lgkmcnt<0>` drain | `__threadfence()`（L2 inv） |

前两类干净了还 race，且 race 只在"连续调用 / 复用 buffer"时出 → 怀疑第三类。

## ⚠️ 性能：别用 per-tile threadfence（核心优化）

最朴素的修法是在 `compute_tile` 每个 tile 的 prologue 发 `__threadfence()` —— **正确但巨慢**：per-tile L2 flush+inv 在 MoE 上 **14-40% tax**。

拆成两半（committed 修法）：
1. **per-tile** 只保留 `wait_lgkmcnt<0>()` —— drain in-flight `buffer_load_lds` 的 GMEM→LDS 写（这才是 per-tile 真正需要的，无 L2 流量）。
2. **per-block 一次** 在 kernel 入口（tile loop 之前）发 `__threadfence()` —— GEMM 输入对 kernel 生命周期是只读，一次 invalidate 覆盖这个 block 处理的所有 tile。

```cpp
// kernel 入口，tile loop 之前：
// Invalidate this XCD's L2 once so buffer_load_lds reads fresh inputs, not
// stale lines from a prior call's reused scale workspace. Inputs are read-only,
// so one invalidate per block covers all its tiles (vs. the old per-tile fence).
__threadfence();
for (int32_t tile_id = blockIdx.x; tile_id < total_tiles; tile_id += gridDim.x) { ... }

// compute_tile prologue（每 tile）：__threadfence() → wait_lgkmcnt<0>()
wait_vmcnt<0>();
wait_lgkmcnt<0>();   // 只 drain LDS 写，不碰 L2
__builtin_amdgcn_s_barrier();
```

fwd kernel 改了，dgrad 复用 fwd kernel 自动受益；wgrad 是独立 kernel 要单独改同样两处。

实测（DeepSeek-V3 / gpt_oss MoE, kernel-only TFLOPS）：拆分后 vs per-tile-fence baseline，Fwd +10~85% / Dgrad +11~60% / Wgrad +33~87%，且 det60 bit-exact 不回。

## 验证

```bash
# det60：bit-exact，atol=0，连续多次调用复用 workspace（关键：要复用才暴露第三类）
GPU=N scripts/dev.sh det      # 0 non-OK
# 反复跑十几次确认稳定 0（race rate 低，单次 0 不算数）
for i in $(seq 1 12); do GPU=N scripts/dev.sh det; done

# 全量 pytest（idle GPU）
pytest tests/pytorch/ops/test_grouped_gemm_fp8.py -k mx_blockwise   # 3840/3840
```

**铁律（沿用 memory `feedback_no_excuses_for_race`）**：det60 atol=0 失败 = race，**不准说"噪声"**，必须诊断。GPU 必须空闲（noisy GPU 会干扰）。单次 0 不够，多跑确认。

## 反例

| 方案 | 为什么不对 |
|---|---|
| per-tile `__threadfence()` | 正确但 14-40% tax，白交 L2 flush 税 |
| per-tile 只换 `wait_lgkmcnt<0>`、删掉 fence（不补 per-block） | 跨 die race（用户当场抓出："这个 race 和数据分布在不同 cu 有关"） |
| 8-wave compute 内部加 `__threadfence()` | 用户明确禁止（8-wave compute 路径内不许 fence；只有 4-wave 入口的跨 die fence 是对的/需要的） |

reviewer / 用户约束：编辑只改本地再 `sync.sh push`（别在远端改会被冲掉）；`commit 前给看`；race 测试 GPU 必须空闲。

---

# 三类 race 速查

| 类 | 共享通道 / 范围 | 症状 | 修法 | commit |
|---|---|---|---|---|
| 1. vmcnt FIFO | scratch_load vs buffer_load_lds（一个 wave 内） | spill>0，单次即 race | 消 VGPR spill（SGPR 化地址 / outer 解析 ptr） | `abd3833` |
| 2. LDS WAR | ds_read vs buffer_load_lds（同 WG 跨 warp） | spill=0，单次即 race | `wait_lgkmcnt<0>` drain（WAR barrier + prologue） | `bb48d3f` |
| 3. die L2 | 复用 workspace 跨 die（跨调用） | 前两类干净，第 2+ 次调用 race | `__threadfence()` 入口一次 + per-tile 退化成 lgkmcnt drain | `d7b149a` |
