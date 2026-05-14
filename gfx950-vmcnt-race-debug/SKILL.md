---
name: gfx950-vmcnt-race-debug
description: 调试 / 修复 AMD MI355X (gfx950) 上 grouped / persistent kernel 里 buffer_load_lds 跟编译器 SGPR spill 的 vmcnt race —— 症状是低概率 (0.001-1%) bit-non-deterministic 输出。当用户在 Primus-Turbo / HipKittens 等 gfx950 kernel 上看到 "race / non-deterministic / bit-不一致 / rerun 不稳定 / lds 写错地方" 时使用此 skill。也覆盖一般的 vmcnt 不 FIFO 类问题（gfx950 在 scratch_load 和 buffer_load_lds 之间不保证 FIFO 顺序）。配合 remote-sync / remote-mlperf-gptoss skill。
---

# gfx950 vmcnt race 调试

## 症状识别

低概率（0.001-1%）的 bit-non-deterministic：相同输入跑 N 次，rerun 之间结果有 bit 差异，但**不**是数值漂移（不是 NaN / Inf / 显著 SNR 下降），是少数 element 偶尔不一致。在 grouped GEMM / persistent kernel 上特别常见；dense single-GEMM 通常不会有。

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

`ITERS` 用 50000+ 才有统计意义。0.005% 量级的 race 在 10K iters 看起来像 0/N，要 50K-150K 才稳定可见。

## 根因机制

GFX950 上：
- `scratch_load`（编译器溢出的 SGPR 回填）走 vmcnt 计数
- `buffer_load_dwordx4 ... offen lds`（LDS-direct GMEM→SMEM 路径）也走 vmcnt
- **两者之间 vmcnt 不保证 FIFO 完成顺序**

所以编译器对 spill 回填发的 `s_waitcnt vmcnt(N>0)` 不安全：N 假设了 FIFO，scratch_load 可能还在飞、`v_readfirstlane` 已经把 stale value 读进 SGPR，然后 `s_mov_b32 m0, sNN` 用了 stale m0，下一条 `buffer_load_dwordx4 ... offen lds` 写到错的 LDS 地址 → SMEM 数据错位 → MFMA 输入污染 → 最终 output bit 不一致。

只在 SGPR pressure 触发 spill 时暴露。Grouped GEMM 因为 outer + compute_tile 的 SGPR 总量大、且 outer 的 VGPR 也可能 spill 到 scratch，更容易踩。

## Disasm 验证

提取 GPU code 并 disasm：

```bash
# 1) extract gfx950 code object from .so
ROCMOBJ=$(/opt/rocm/bin/roc-obj-ls build/lib/libprimus_turbo_kernels.so 2>&1 | awk '/gfx950/{print $NF; exit}')
OFFSET=$(echo "$ROCMOBJ" | sed -n 's/.*offset=\([0-9]*\).*/\1/p')
SIZE=$(echo "$ROCMOBJ" | sed -n 's/.*size=\([0-9]*\).*/\1/p')
dd if=build/lib/libprimus_turbo_kernels.so bs=1 skip=$OFFSET count=$SIZE 2>/dev/null > /tmp/gpu.hsaco

# 2) disasm
/opt/rocm/lib/llvm/bin/llvm-objdump -d --triple=amdgcn-amd-amdhsa --mcpu=gfx950 /tmp/gpu.hsaco > /tmp/gpu.disasm

# 3) reg metadata（看 sgpr_count / private_segment_fixed_size / vgpr_spill_count）
/opt/rocm/lib/llvm/bin/llvm-readelf --notes /tmp/gpu.hsaco > /tmp/notes.txt
grep -B1 -A8 "<your kernel symbol substring>" /tmp/notes.txt | grep -E "sgpr|private|vgpr"
```

Race 的 disasm 签名：
- `private_segment_fixed_size > 0`（说明有 spill 到 scratch）
- 主循环里看到 `scratch_load_dword vXX, off, off offset:N` 夹在多个 `buffer_load_dwordx4 ... offen lds` 之间
- 紧跟着 `s_waitcnt vmcnt(N>0)` + `v_readfirstlane_b32 sNN, vXX` + `s_mov_b32 m0, sNN` + `buffer_load_dwordx4 ... offen lds`

如果 `private_segment_fixed_size == 0` 但还有 race，去看 SGPR 是否 spill 到 VGPR lanes（`v_writelane_b32` / `v_readlane_b32`）—— 这条**不会**race（lane ops 不走 vmcnt）。

## 修法（双管齐下）

### A) 降低 spill

**`-mllvm -sink-insts-to-avoid-spills=true`**：把定值指令下沉到 use 点，缩短 live range。在 setup.py 的 nvcc/hipcc flags 里加：
```python
"-mllvm",
"-sink-insts-to-avoid-spills=true",
```
对 mxfp8 grouped GEMM 实测把 `private_segment` 从 44 → 20 字节。

源码侧的 SGPR pressure 削减：
- **CK pattern**：把 per-group readfirstlane / pointer 计算从 inner kernel 提到 outer kernel，inner 只接收已 resolved 的 base 指针 + per-tile bookkeeping。inner SGPR live set 立刻接近 dense single-GEMM 水平。
- **smem packing**：inner kernel 接 `char *smem_buf` 一个指针，自己 reinterpret_cast 出各 typed view，省掉 4 个 typed pointer 参数。
- **保留 `tile.reserve_pinned_regs()`**：在 compute_tile 入口、prologue 前、main loop 前各一次。**实证**任何一处去掉，race 立即恢复 / 100%。

### B) 把 LDS commit 从 vmcnt 改到 lgkmcnt

**2-step prologue**：把 `buffer_load_lds`（一条指令，写 LDS 走 vmcnt）拆成 `buffer_load → VGPR → ds_write → LDS`。`ds_write` 走 lgkmcnt，跟 spill 的 vmcnt 错开通道。**只在 prologue 这么做**，main loop 的 `phase_mfma_lds_ldg` 保留 LDS-direct（替换会破坏 LDG/MFMA overlap，实测 -85% perf）。

helper 模板：
```cpp
template <int Bytes>
__device__ __forceinline__ void
load_gmem_to_smem_srd_two_step(const BufferSRD &srd, uint32_t ldg_offset,
                               uint32_t lds_addr, int32_t soffset) {
    if constexpr (Bytes == 4) {
        uint32_t v = llvm_amdgcn_raw_buffer_load_b32(srd.srd, ldg_offset, soffset, 0);
        *reinterpret_cast<as3_uint32_ptr>((uintptr_t) lds_addr) = v;
    } else {  // Bytes == 16
        int32x4_t v = llvm_amdgcn_raw_buffer_load_b128(srd.srd, ldg_offset, soffset, 0);
        *reinterpret_cast<as3_int32x4_ptr>((uintptr_t) lds_addr) = v;
    }
}
```
prologue load 改用 `<bool TwoStep = false>` 默认参数，prologue 调用点传 `, true>`。

**`-mllvm -amdgpu-enable-merge-m0=true`**：合并相邻 m0 init，减少 m0 写次数 → 减少 stale m0 暴露窗口。配合 2-step 一起加。

prologue 末尾 drain 两个 counter：
```cpp
wait_vmcnt<0>();
wait_lgkmcnt<0>();  // ← 2-step 后必须加这条，否则 ds_write 还没完成主 loop 就开 ds_read
__builtin_amdgcn_s_barrier();
```

### Wgrad 特殊处理

Wgrad prologue 全 16 个 load 都改 2-step → -25% wgrad perf。只让 8 个 **data load** 走 2-step，scale load（4 字节）保留 LDS-direct，race 仍归 0、perf 不退。

### 验证清单

```bash
# 1) reg counts: private_segment_fixed_size 应该明显下降（理想 <24 字节）
grep -B1 -A6 <kernel> /tmp/notes.txt | grep -E "sgpr_count|private_segment|sgpr_spill"

# 2) disasm: 确认主循环里 scratch_load 跟 buffer_load_lds 不再交错
sed -n '<kernel_start>,<+15000>p' /tmp/gpu.disasm | grep -nE 'buffer_load_dwordx4.*lds|scratch_load' | head -50

# 3) race rate: fwd/dgrad/wgrad 各 50K-150K iters, 多 GPU 跑确认不是单卡 fluke
for i in 1 2 3; do python repro_race_grouped_fwd.py 50000; done
```

## 反例（**不要**走的路）

试过的 workaround，都不能同时拿到 0/N race + PR HEAD perf：

| 方案 | 结果 | 为什么不行 |
|---|---|---|
| `c_group_offs == nullptr` mode-switch + `clobber_vgpr_one`（PR 之前的 workaround） | 0/N race + perf 持平 | reviewer 拒：本质是把 spill 藏起来，没修根因；后续小改易复现 race |
| `__noinline__ compute_tile` | 0/N race | -25% perf（args 通过 scratch 传，外部跨 call live VGPR 也 spill） |
| 全 main loop `buffer_load_lds → 2-step` | 0/N race | -85% perf（破坏 phase_mfma_lds_ldg overlap） |
| Mid-iter 主循环加 `s_waitcnt vmcnt(0)` | race 不降 0 | -21% perf |
| FIRST_2STEP（每 phase 第一个 load 走 2-step） | race 不降 0 | -30%~-40% perf |
| 多个 `s_waitcnt vmcnt(0)` 在 prologue（PR HEAD 的 4× drain） | race ~0.01% | reviewer 拒，没修根因 |

## 测试 / 容器约定

- Race + perf 验证用 mlperf_gptoss2 容器 (gfx950 节点)，参见 remote-mlperf-gptoss skill
- mxfp8 PR 在 `/workspace/code/code3/Primus-Turbo/`，参见 remote-sync skill
- GPU 选**真闲**的（`rocm-smi --showuse` GPU%=0、`--showmemuse` VRAM% < 50）；race rate 在 noisy GPU 上会被 OOM 干扰
- bench 用 `scripts/spot_bench_kernel.py`（kernel-only TFLOPS，不算 quant pre-process），跟 PR HEAD baseline 比

## 真实案例

`Primus-Turbo` PR `dev/kyle_mxfp8_gg_pr` (commits `567e578` + `07e9940`)，gpt_oss-20B Expert shape (B=4 M=2048 N=5760 K=2880)：
- PR HEAD `0df887d`: fwd race ~3/30K, perf 1322 / 1233 / 1749 TFLOPS
- 修后: **0/450K race**（fwd+dgrad+wgrad 各 0/150K）、perf 1337 / 1250 / 1753 TFLOPS

仓库 commit log 里的 commit message 给了完整的 before/after 数据可参考。
