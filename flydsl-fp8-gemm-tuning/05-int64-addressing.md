# >2^31 / >4GB 寻址：per-tile i64 SRD rebase

## 为何需要 i64

AMD buffer SRD（标量资源描述符）的结构：
- base address：64-bit（存在 SRD 内）
- num_records：**32-bit**（上限 0xFFFFFFFF = 4GB）
- per-lane voffset：**32-bit**

两类越界（同时存在）：
1. flat `reshape(-1)` / 1D 张量在 2^31 元素处 `pack_layout_buffer`（struct 'i' format）溢出
2. 单个 buffer SRD 的 per-lane voffset + num_records 32-bit → 4GB 上限

## 核心技术：per-tile i64 SRD rebase

把每 tile/每 group 的巨大元素 base（`m_row*K`, `m_start*OUT_M` 等）折进 **SRD 的 i64 base**，in-tile 的小偏移留 i32。

```python
# fp8_gemm_helper.py: make_fp8_buffer_tensor_rebased
base = extract_base_index(arg_i8)          # 张量真实 base ptr (i64)
base = base + index_cast(i64, base_elems)  # + per-tile 大偏移（i64 运算）
nr   = min(index_cast(i64, num_records_bytes), 0xFFFFFFFF)  # clamp 到 32-bit
base = _readfirstlane_i32(base)            # ★ 必须 pin 到 SGPR
nr   = _readfirstlane_i32(nr)
# 组装 SRD: [base_ptr(64b), stride/swizzle, num_records(32b), flags]
```

## readfirstlane pin（CRITICAL）

若不 pin，base 来自 group_scan（每组不同）→ divergence 分析判为 VGPR → SRD 落 VGPR → 每次 K-loop `buffer_load` 都触发 readfirstlane waterfall（16 次/load）→ **gateup -16%，down -13%**。

`_readfirstlane_i32` 对 i64 同样有效（高低 32-bit 分别 pin）。

## 两类算子的不同策略

### Foldable（NT A + B_T，NN A：contraction 在内/连续维）

per-load offset 只在 tile 内部（小），巨大 base 完全 fold 进 i64 SRD。**无上限**。

```python
# NT fwd: A[M,K] - 按 m_row 做 rebase，tile 内只走 K 方向（小 offset）
a_base = index_cast(i64, m_row) * index(K)
gA = make_fp8_buffer_tensor_rebased(A, fp8_t, a_base, tile_K_bytes)
```

### Traversal-spanning（NN B[K,N]，TN A[K,M]+B[K,N]：contraction 跨整个 tensor）

K-loop 每步的 offset = `k * BLOCK_K * stride`，单次 load 要覆盖全程。i32 在 >2^31 wrap → 地址错。

策略：fold 列基址进 i64 base，i64 的 `k*BLOCK_K*stride` 乘法（**必须用 `arith.index(k*BLOCK_K)` 而非 `arith.index_cast`**），单描述符封顶 4GB（`can_handle CAP=2^32`，超了 decline 回 fallback）。

```python
# TN 每个 load site：
arith.index((k+2)*BLOCK_K) * cn_i   # 注意：k 是 range_constexpr 的 python int
# ★ 不能用 arith.index_cast(T.index, (k+2)*BLOCK_K) -- 对 python int 会 crash
# (_to_raw 不接受 int，报 'int' has no _CAPIPtr)
```

### wgrad（grouped）

contraction = M_g（per-group），fold `m_start*OUT_{M,N}` 进 i64 base，num_records = `M_g*OUT`（per-group，不是累积）：

```python
a_base = index_cast(T.index, m_start) * index(OUT_M)
mg     = index_cast(T.index, m_end) - index_cast(T.index, m_start)
gA = make_fp8_buffer_tensor_rebased(A, fp8_t, a_base, mg * index(OUT_M))
```

残留：per-group `M_g * OUT < 2^31`（M_g < ~300K，现实安全）。

## 输出 C 的 i64（StoreCPerTensor）

output 地址 = `group_idx * OUT_M * OUT_N + row * OUT_N + col`，对大-G 也会溢。

`StoreCPerTensor` 按 64-row band 重新 rebase（`extract_base_index` + `create_buffer_resource_from_addr`），每 band 的 num_records = 64行量 = 小 i32。CShuffle 向量化 128b store 用同样机制。

## entry 函数改动

传 **全 rank** 张量（不 `reshape(-1)`），否则 shape-pack struct 'i' 在元素数 >2^31 时 crash：

```python
a_i8 = a.view(torch.int8)    # 2D [M_total, K]，保持全 rank
b_i8 = b.view(torch.int8)    # 3D [G, N, K]，保持全 rank
# ❌ 不能 reshape(-1)：shape-pack 会溢出
```

## 验证方法

```bash
python scripts2/_grouped_biginput_int64.py  # A+B > 2^31
python scripts2/_grouped_bigc_int64.py      # C > 2^31
# 检查 SNR 在 <2^31 和 >2^31 threshold 两侧相等（无溢出断点）
```
