---
name: mxfp8-8wave-devloop
description: MXFP8 dense GEMM 在 MI355X/gfx950 上的完整调试与优化手册（FlyDSL kernel 开发、preshuffle/dual-cast 融合、tail 处理、vs GB200 benchmark）。分支 dev/kyle/flydsl_mxfp8_fused_test，2026-06。
---

# MXFP8 Dense GEMM：调试与优化完整手册

> MI355X / gfx950，Primus-Turbo `dev/kyle/flydsl_mxfp8_fused_test`，2026-06。
> 覆盖 kernel 开发、数值调试、preshuffle/dual-cast 融合、tail 处理、到 vs GB200 benchmark 的全流程。

---

## 1. 环境速览

```
节点:      chi2811  (gfx950 x8, HBM3e ~8 TB/s 峰值，实测混合 BW ~5257 GB/s)
容器:      mlperf_gptoss2
代码挂载:  /mnt/vast/kyle/code3/mxfp8/Primus-Turbo → /workspace/code/mxfp8/Primus-Turbo
本地镜像:  /wekafs/kyle/remote_sync/mxfp8/Primus-Turbo  (rsync 同步)
使用 GPU:  4 或 5 (mem_get_info 现查占用)
flydsl 磁盘缓存: /root/.flydsl/cache  ← 改 kernel 后必须 rm -rf
```

### 1.1 检查 GPU 占用

```bash
ssh chi2811 'docker exec mlperf_gptoss2 bash -c \
  "rocm-smi --showuse 2>/dev/null | grep \"GPU\[4\]\|GPU\[5\]\""'
```

### 1.2 rsync 同步（防旧缓存）

```bash
# --checksum 防止 mtime 相同被跳过（Python 会加载旧字节码）
rsync -av --checksum primus_turbo/flydsl/gemm/mxfp8_quant_flydsl.py \
  chi2811:/mnt/vast/kyle/code3/mxfp8/Primus-Turbo/primus_turbo/flydsl/gemm/

# 同步后 md5 确认（这步不可省）
ssh chi2811 'md5sum /mnt/vast/kyle/code3/mxfp8/Primus-Turbo/primus_turbo/flydsl/gemm/mxfp8_quant_flydsl.py'
md5sum primus_turbo/flydsl/gemm/mxfp8_quant_flydsl.py
```

---

## 2. FlyDSL Kernel 开发要点

FlyDSL 是 JIT 框架，纯 Python 改 `.py` 无需 C++ rebuild，迭代极快。

### 2.1 模块隔离（RecursionError 陷阱）

```python
# ❌ 错误：kernel 文件和 torch/测试代码同一模块
# → JIT 依赖收集触发 RecursionError

# ✅ 正确：kernel 放独立干净模块
# mxfp8_quant_flydsl.py — 只 import flydsl，不 import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl, range_constexpr, buffer_ops as bo
# harness 脚本才 import torch，分开两个文件
```

### 2.2 flydsl 磁盘缓存陷阱（最常踩）

```bash
# ❌ 改了 kernel 但忘清缓存 → 测的是旧 binary！
# 症状：数值没变、速度没变，改动完全没生效

rm -rf /root/.flydsl/cache   # 每次改 kernel 后必做
```

### 2.3 测量真实 GPU 时间

```python
# ❌ 错误：flyc.jit launch 每次调用有 ~40us Python 派发开销
# → cuda-event 量出的是派发延迟，不是 kernel 时间

# ✅ 正确：先 flyc.compile，再用 cuda-event 量
comp = flyc.compile(launch, x, Qr, ASp, AtQd, AtSp, st)  # 编译一次
comp(x, Qr, ASp, AtQd, AtSp, st)  # 直接进 GPU stream

def gms(fn, it=300, w=30):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(it): fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / it  # ms
```

### 2.4 关键 API 速查

```python
from flydsl.expr import buffer_ops as bo, rocdl, range_constexpr
from flydsl.expr.typing import Vector as Vec
from flydsl.expr import math as fm
from primus_turbo.flydsl.utils.gemm_helper import make_row_band_resource

# Global memory read (vec4 bf16)
v = bo.buffer_load(rsrc, offset_elem, vec_width=4, dtype=fx.BFloat16.ir_type)

# Global memory write (int32, byte offset)
bo.buffer_store(word, rsrc, byte_offset, cache_modifier=1, offset_is_bytes=True)

# create_buffer_resource (flat buffer, element-offset 默认)
rsrc = bo.create_buffer_resource(tensor, max_size=False, num_records_bytes=I32(nbytes))
bo.buffer_store(val, rsrc, elem_idx)  # offset_is_bytes=False (default)

# make_row_band_resource (2D [rows,cols] buffer, int64 rebasing)
rsrc = make_row_band_resource(bo.extract_base_index(T), base_row, c_rows, c_cols, elem_bytes)

# LDS alloc
@fx.struct
class Smem:
    tile: fx.Array[fx.BFloat16, SIZE, ALIGN]
sm = fx.SharedAllocator().allocate(Smem).peek()
tile = sm.tile

# LDS write (strided view store)
p = fx.add_offset(tile.ptr, fx.make_int_tuple(offset))
fx.make_view(p, fx.make_layout(n_elem, stride)).store(Vec(v))

# LDS strided read (32 bf16, stride=PAD)
base = fx.add_offset(tile.ptr, fx.make_int_tuple(col))
cv = Vec(fx.make_view(base, fx.make_layout(32, PAD)).load()).to(fx.Float32)
# 注：make_view(base, layout(32,PAD)).load() 正确生成 32 次 ds_read_b16

# f32 → fp8 打包 (2个f32 → i32的2字节)
IRI = fx.Int32.ir_type
word = fx.Int32(rocdl.cvt_pk_fp8_f32(IRI, a, b, fx.Int32(0), 0))  # bytes 0,1
word = fx.Int32(rocdl.cvt_pk_fp8_f32(IRI, c, d, word, 1))          # bytes 2,3

# e8m0 exponent（round-even，匹配 C++ compute_tile_scale）
def _ep(amax):
    I32 = fx.Int32
    ai = I32(amax.bitcast(fx.Int32.ir_type)) + I32(1 << 19)
    ep = ((ai >> I32(23)) & I32(0x1FF)) - I32(135)
    ep = (ep < I32(-127)).select(I32(-127), ep)
    ep = (ep > I32(128)).select(I32(128), ep)
    return I32(ep)

# barrier
rocdl.s_barrier()
```

### 2.5 常见编译错误与修法

| 错误 | 原因 | 修法 |
|------|------|------|
| `RecursionError` | kernel 和 torch 同模块 | 独立 `.py` 文件 |
| `AttributeError: 'Float32' has no 'bitcast'` | `.reduce("max")` 返回 Float32 | `(F32(0.0)>ca).select(F32(0.0),ca)` 包一层 |
| `arith.truncf` not lowerable | `.to(fp8)` | 改用 `rocdl.cvt_pk_fp8_f32` |
| `ScalarizeVectorOperand` LLVM ERROR | `buffer_store` 写 bf16 scalar | 改写 int32（打包 fp8） |
| `'Int32' has no attribute 'bitcast'` | 想做 int→float bitcast | 用 `fm.exp2(ep.to(F32))` |
| 数值全 0，改动无效 | 旧 flydsl 缓存 binary | `rm -rf /root/.flydsl/cache` |
| scale 全 0 / bcast 错 | Uint8 bcast 高位损坏 | **全程保持 Int32**（核心 bug，见下） |

---

## 3. 最关键 Bug：e8m0 bcast 必须全程 Int32

```python
# ❌ 错误：cast 到 Uint8 再做 OR/shift → 高位损坏
e8 = (ep + I32(127)).to(fx.Uint8)
bcast = e8 | (e8 << I32(8)) | (e8 << I32(16)) | (e8 << I32(24))
# 结果：bcast 各字节不同，match 仅 ~9%，数值垃圾

# ✅ 正确：全程保持 Int32，不提前 cast
e8_i32 = ep + I32(127)   # Int32，范围 0..255，不 cast 到 Uint8
bcast = e8_i32 | (e8_i32 << I32(8)) | (e8_i32 << I32(16)) | (e8_i32 << I32(24))
```

**规律**：flydsl 里凡是要做 OR/SHIFT 位运算，**不要提前 cast 到 Uint8**，保持 Int32 全程。

---

## 4. MXFP8 Preshuffle 架构

### 4.1 为什么要 preshuffle

MXFP8 GEMM 的 `mfma_scale_f32_16x16x128_f8f6f4` 通过 `opsel` 字节选择器读 E8M0 scale。每个 wave 64 条 lane 各需对应 scale 字节，原始布局 `[DIM, K//32]` 是散点 gather，实测比 coalesced **慢 ~2.6×**。Preshuffle 把 scale 重排成 wave/lane 连续布局。

### 4.2 Layout-1（A-scale broadcast）

```
输入: e8m0 [DIM, K//32]
输出: i32  [DIM//16, K//128, 64, n_tiles]

SP[grp, k, lane, s] = broadcast_u8_to_u32(scale[grp*16*n_tiles + s*16 + lane%16, 4k + lane//16])

n_tiles = BLOCK_M // 64  (BLOCK_M=256 时固定 = 4)
```

**索引公式（Python 验证 100% 正确）**：

```python
def mydword(row, kcol, K128):
    g    = kcol & 3
    kg   = kcol >> 2
    grp  = row // 64
    sub  = (row % 64) // 16   # 注意：在 kernel 里必须命名 sub，不能叫 s（变量名冲突）
    r    = row % 16
    lane = g * 16 + r
    return ((grp * K128 + kg) * 64 + lane) * 4 + sub
```

**写侧用 create_buffer_resource + element-offset**（与 ScaleS2R 读侧完全一致）：

```python
rasp = bo.create_buffer_resource(ASp, max_size=False, num_records_bytes=I32(SP_A_BYTES))
bo.buffer_store(bcast, rasp, dword, cache_modifier=1)  # element offset（默认非字节）
```

**变量名冲突陷阱**：加载循环 `for s in range_constexpr(4)` 之后，Python 的 `s=3`。再写 `s=(grow>>4)&3`，JIT trace 里可能折叠为常量 3。**必须改名为 `sub`**。

### 4.3 B-comb（B-scale combined）

```
用于: fwd B 操作数 (dim=N, contract=K) 以及 bwd at 操作数 (dim=K, contract=M)

block_n = col // 256
wn      = (col % 256) // 32
s_bc    = ((col >> 7) << 1) | (((col & 127) & 31) >> 4)
r_bc    = col % 16
lane_bc = g_bc * 16 + r_bc   (g_bc = row_block & 3)
grp_bc  = block_n * 4 + wn
dword   = ((grp_bc * K128p + kg_bc) * 64 + lane_bc) * 4 + s_bc
```

**Buffer 大小：用 `cdiv(dim,256)*4` 组**（grp 是块跨步，不是 `dim//64`）：

```python
# ❌ 旧：断言 %256，大小公式错
dwords = (M / 64) * K128p * 64 * 4       # 漏掉 partial 256-block 的 grp

# ✅ 正确
dwords = cdiv(M, 256) * 4 * K128p * 64 * 4   # C++ 端
nbytes = ((dim + 255) // 256) * 4 * K128 * 64 * 4 * 4  # kernel ScaleBComb
```

---

## 5. Dual-Cast Quant Kernel（mxfp8_quant_flydsl.py）

### 5.1 架构

```
输入: X[M, K] bf16
输出:
  Qr[M, K]     fp8  ← row cast (fwd gemm A)
  ASp[SP_A]   int32  ← layout-1 A-preshuffle scale (ScaleS2R element-offset)
  AtQd[K, M]   fp8  ← col cast 已转置 (bwd at B 操作数)
  AtSp[SP_AT] int32  ← b-comb preshuffle col scale (ScaleBComb element-offset)

Tile: [BM=32 × BK=256]，Block: 512 线程
  所有 512 线程：coalesced load [BM×BK] → LDS，barrier
  前 256 (half==0)：ROW cast → Qr + ASp
  后 256 (half==1)：COL cast → AtQd + AtSp   ← 两半并发，共享 LDS
```

### 5.2 SP_A / SP_AT 大小

```python
SP_A_ELEMS  = (M // 64) * (K // 128) * 64 * 4         # layout-1 dwords
SP_AT_ELEMS = (K // 256) * 4 * (M // 128) * 64 * 4    # b-comb dwords (K free, M contract)
```

### 5.3 Col 操作数形状

```python
# C++ dual-cast 的 col 输出是 [K, M]（已转置！不是 [M, K]）
# AtQd 必须是 [K, M]

# 验证方法
_, _, cq_ref, _ = quantize_mxfp8_impl(x, e4m3, None, 32, True, ...)
print(cq_ref.shape)  # → [K, M]，M≠K 时可以验证

# bwd grad_b: NT(go_col[N,M], at[K,M]) → [N,K]
# at_qd.shape = [K,M]，trans_b=True 时 kernel 把它当 [K,M] 读，NT 输出 [N,K] ✓
```

---

## 6. 调试方法论：逐层隔离

遇到数值错误时按此顺序排查，**找到最小复现单元**：

```
Step 1: Python 公式验证
  → mydword(row,kcol) 枚举所有 (row,kcol)，match=2048/2048
  → "formula match 100%" = 公式没问题

Step 2: GPU 地址路径验证
  → Sp[dword] = dword（写 dword index 作为 value）
  → "address-correct: 1.0000" = SRD 和 offset 正确

Step 3: scalar 值验证
  → 写 e8 byte 到 uint8 scratch，比 C++ plain scale
  → "e8 vs C++ plain: 1.0000" = 量化数学正确

Step 4: bcast 验证（全程 Int32 路径）
  → 写 bcast 到 flat int32 buffer，比 C++ preshuffle 输出
  → 定位到"Uint8 cast 后 OR/shift"的问题

Step 5: 行隔离（只跑 col-half，row-half 空 pass）
  → 99.69% = col 单独正确

Step 6: 两半并存时干扰测试（加 row fp8 写）
  → 还是 99.69% = 无干扰

Step 7: 完整 kernel import 仍错
  → 必是 flydsl 磁盘缓存或 Python 字节码旧版本
  → rm -rf /root/.flydsl/cache + rsync --checksum
```

### SKIP 法：kernel 内分段计时

```python
import os; _SK = int(os.environ.get('SK', '0'))
# 在 kernel 里各阶段 if _SK==N: return
# SK=0: 完整；SK=1: 只 row；SK=2: row+col 计算无写；SK=3: 跳 LDS 写
# 实测：barrier/占用/amax+exp2 全移除均无效 → 是 dual-cast DRAM 散写的固有代价
```

---

## 7. Partial-Tile Tail（非 256 倍数 M/N 支持）

### 7.1 机制

| 组件 | 实现 | 效果 |
|------|------|------|
| 数据 G2S | buffer resource (SRD) | OOB → 0 |
| scale 读 | create_buffer_resource(num_records) | OOB → 0 |
| 输出写 | make_row_band_resource + col < c_n mask | 越界被钳 |

只需 M/N 是 **64 的倍数**即可安全。256 倍数约束已解除。

### 7.2 Shape Gate

```python
def _flydsl_mxfp8_nt_ok(M, N, K):
    # NT-fold 使 M/N/K 都当过 contract 维；K-loop 无 K-tail 且预取 2 层
    return (
        M % 128 == 0 and N % 128 == 0 and K % 128 == 0
        and M >= 256 and N >= 256 and K >= 256
        and M * K < 2**31 and N * K < 2**31
    )
```

### 7.3 Dispatcher 禁用

```python
# gemm_fp8_impl.py GEMMFP8FlyDSLBackend.can_handle()
if granularity == MX_BLOCKWISE:
    return False  # 只接 preshuffled，dispatcher 喂 raw → 禁
```

---

## 8. 算子集成

### 8.1 FP8GemmMXFunction._qdual 缓存机制

```python
_QDUAL_CACHE: dict = {}   # (M, K) → compiled callable

def _qdual(x_bf16, M, K):
    key = (M, K)
    comp = _QDUAL_CACHE.get(key)
    SP_A  = (M // 64) * (K // 128) * 64 * 4
    SP_AT = (K // BK) * 4 * (M // 128) * 64 * 4
    xq   = torch.empty(M, K, dtype=float8_e4m3, device=x_bf16.device)
    asp  = torch.empty(SP_A,  dtype=torch.int32, device=x_bf16.device)
    atqd = torch.empty(K, M,  dtype=float8_e4m3, device=x_bf16.device)
    atsp = torch.empty(SP_AT, dtype=torch.int32, device=x_bf16.device)
    st = torch.cuda.current_stream()
    if comp is None:
        comp = _flyc.compile(_compile_qdual(M, K), x_bf16, xq, asp, atqd, atsp, st)
        _QDUAL_CACHE[key] = comp
    comp(x_bf16, xq, asp, atqd, atsp, st)
    return xq, asp, atqd, atsp   # 完全替换 C++ dual-cast
```

### 8.2 fwd/bwd 数据流

```
FWD: _qdual(a) → a_qd,a_sc(layout-1),at_qd,at_sc(b-comb)
     _qdual(b) → b_qd,b_sc(layout-1),bt_qd,bt_sc(b-comb)
     gemm(a_qd, a_sc, b_qd, b_sc)

BWD: go_col = _qdual_row(grad_out) for col
     grad_a = NT(go_row, go_row_sc, bt_qd, bt_sc)
     grad_b = NT(go_col, go_col_sc, at_qd, at_sc)
```

---

## 9. 性能分析

### 9.1 正确的 benchmark 口径

```python
# ✅ 官方口径（和 TE GB200 benchmark 一致，用 Timer.timeit）
import torch.utils.benchmark as bm
ms = bm.Timer(stmt="fn()", globals={"fn": fwd}).timeit(100).mean * 1e3
tflops = 2 * M * N * K / (ms * 1e-3) / 1e12

# ❌ 手搓 cuda-event 计时 bwd → 把 autograd dispatch 算进 GPU 时间，bwd 低估 ~10-18%
```

### 9.2 带宽分析

```python
# 实测 MI355X HBM 混合 BW 天花板（非标称 8 TB/s）
# 512MB copy → ~5257 GB/s （比 128MB 低，因 L2 装不下）
# dual-cast quant 流量 = 4.0625 * M * K bytes (读 2MK + 写 2MK + scale)
# 已达 copy 峰值 90% = 贴墙，不是 kernel 问题

# 判断瓶颈：移除 barrier/占用/计算均无效 → 固有 DRAM 散写代价
```

### 9.3 vs GB200 结论

| 路径 | fwd geomean | bwd geomean |
|------|-------------|-------------|
| 旧 C++ dual-cast | ~0.84× | ~0.97× |
| **flydsl dual-cast（集成后）** | **~0.93×** | **~1.08×** |

关键 shape (8192×4096×4096)：fwd 1.28×，bwd 1.29×（反超 B200）。

fwd 差距根因：B200 硬件 MX cast（近免费）vs MI355X 软件 dual-cast。  
**突破 fwd 的唯一路**：in-gemm fusion（量化融进 NT gemm prologue）。

---

## 10. 数值验证 SOP

```python
# 1. self-consistent check（不依赖 C++ 参考）
rowok = (Qr.view(torch.uint8) == (x.float() / torch.exp2((Sr.float()-127))
         .repeat_interleave(32,dim=1)).to(e4m3).view(torch.uint8)).float().mean()
# 期望 rowok = 1.0000（0.31% 偏差是 cvt round tie-break，正常）

# 2. vs C++ 参考
_, _, cq_ref, cs_ref = quantize_mxfp8_impl(x, e4m3, None, 32, True,
    ScalingRecipe(preshuffle_layout=1, preshuffle_n_tiles=4),
    ScalingRecipe(preshuffle_layout=3))
# cq_ref.shape = [K, M]（已转置！M≠K 时才能看出来）
col_fp8 = (AtQd.view(uint8) == cq_ref.view(uint8)).float().mean()
col_sp  = (AtSp == cs_ref.view(int32).reshape(-1)).float().mean()

# 3. e2e SNR（最终验收）
fwd_snr  > 25 dB    # E4M3 阈值
gradA_snr > 25 dB
gradB_snr > 25 dB
```

---

## 11. C++ 改动规范

### 11.1 B-comb buffer sizing（两个文件都要改）

```cpp
// quantization.cpp + quantization_meta.cpp
// ❌ 旧：断言 %256，dwords 用 free/64（少 2 组导致 partial block 后半读 0）
PRIMUS_TURBO_CHECK(M % 256 == 0, "...");
dwords = (M / 64) * K128p * 64 * 4;

// ✅ 正确：%64 即可，cdiv(free,256)*4 覆盖所有 block
PRIMUS_TURBO_CHECK(M % 64 == 0, "...");
dwords = cdiv(M, 256) * 4 * K128p * 64 * 4;
```

### 11.2 C++ 改动后 rebuild

```bash
# 远端容器内
MAX_JOBS=64 python setup.py build_ext --inplace > /tmp/build.log 2>&1
echo "exit=$?"; tail -3 /tmp/build.log
```

---

## 12. 死路记录（勿重试）

| 方向 | 死路原因 |
|------|---------|
| asm-AGPR MFMA inline_asm | acc-clobber 墙，SNR garbage |
| scale via LDS 双缓冲 | vmcnt 计数失准→流水崩，慢 6× |
| host 侧任何缓存/重排 | 用户硬否决 |
| 单独 quant 2× 目标 | 固有 DRAM 带宽墙（已达 copy 峰值 90%，比值物理锁死 ~1.9×） |
| BLOCK_K=256 移植 MoE grouped GEMM | MoE BW-bound，实测回退 |

---

## 13. 快速开发 Checklist

```bash
# 1. 改 kernel .py（本地）
# 2. 同步（强制检验）
rsync -av --checksum FILE chi2811:/mnt/vast/kyle/code3/mxfp8/Primus-Turbo/FILE
# 3. 清缓存
ssh chi2811 'docker exec mlperf_gptoss2 rm -rf /root/.flydsl/cache'
# 4. 测试
ssh chi2811 'docker exec -e HIP_VISIBLE_DEVICES=4 mlperf_gptoss2 bash -c \
  "cd /workspace/code/mxfp8/Primus-Turbo && python test.py 2>&1 | tail -5"'
# 5. 数值通过 → gms() 测速 → 和 C++ 对比
# 6. GPU 干净确认（rocm-smi --showuse）→ bench
# 7. git add / commit（pre-commit 自动格式化，可能需两次 add+commit）
```

---

## 14. 分支结构

```
main
  └── 21e235e  flydsl dense MXFP8 GEMM (NT/NN/TN)
        └── 9905d5f  scale preshuffle 进 quant kernel
              └── e7b9c20  TE-style dual-cast quant
                    └── ef8695c  preshuffle 融合进 dual-cast，opsel byte-pack
                          └── 0234575  tail + flydsl dual-cast quant 集成 ← HEAD
                                       (dev/kyle/flydsl_mxfp8_fused_test)

dev/kyle/flydsl_mxfp8_gemm  ← 停在 e7b9c20，已过时
```
