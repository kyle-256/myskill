# FlyDSL LDS 分配与 swizzle:SmemAllocator.finalize、composed swizzle layout、XOR/padding 消 bank

> 类别: 方法论 · 主题标签: LDS, swizzle, bank-conflict, allocator

## LDS 分配(allocator）
- **legacy 路径 SmemAllocator**:`SmemAllocator(None, arch='gfx942', global_sym_name='smem0')` → `allocate_array(fx.T.f16, N)` 分配 typed 数组;kernel 内 `buf = lds_buf(allocator.get_base())` 得 `SmemPtr`,`ptr.store(val,[idx])` / `ptr.load([idx])` 访问。也可 `allocate_array(T.i8, n_bytes)` 分字节再 `buf(allocator.get_base())` 拿 typed ptr。
- **必须 finalize**:在 GPU module body 内(`with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body)`)调 `allocator.finalize()`。**忘则 LDS 符号未解析**(编译期报错/链接失败)。
- **新 kernel 推荐 SharedAllocator**:`SharedAllocator`(`flydsl.expr.gpu`,即 `fx.SharedAllocator`)配 `@fx.struct` 存储布局,优于 legacy `SmemAllocator`/`SmemPtr`。
  - `static=True`(默认):每 leaf 发**静态 LDS global**,由编译器定尺寸,`launch(smem=)` 留空。
  - `static=False`(动态):launch wrapper 从 `allocated_bytes` 推 smem,显式 `smem=` 须 ≥ 该值。
- **对齐硬约束**:`vector.store` 到 LDS 硬编码 **16-byte 对齐**,分配必须满足。

## Composed swizzle layout(多阶段双缓冲）
- swizzled 多阶段 tile:`make_composed_layout(SwizzleType.get(b,m,s), make_ordered_layout((BM,BK,STAGES), order))` 再 `make_view(get_dyn_shared(dtype), layout)`。
  - **STAGES** 表达双缓冲(在 layout 维度里)。
  - **SwizzleType** 在 descriptor 内部消冲突。

## XOR swizzle(消 bank,零 LDS 开销）
- **通式**:`swizzled_col = col ^ (row >> shift)` — 不同行同列命中不同 bank。
- **FlyDSL 经 SmemAllocator 实现**:`swizzled_col = col_idx ^ (row_idx & XOR_MASK)`;`lds_offset = row_idx * PADDED_STRIDE + swizzled_col`。
- **XOR mask 值**:
  - gfx942 = `32/(vec*elem_size/4) - 1`
  - gfx950 = `64/(vec*elem_size/4) - 1`
- **GEMM A tile 专用(16 字节粒度 XOR-with-row)**:`swizzle_xor16(row,col,k_blocks16) = col ^ ((row % k_blocks16)*16)`,其中 `k_blocks16 = tile_k_bytes // a_elem_vec_pack // 16`。**write 和 read 两路径都要用**,零 LDS 开销、约 1 SALU/地址。

## 向量化(每 vec 覆盖 4 bank = 16 字节)
| dtype | 推荐 vec |
|---|---|
| fp32 | 4 |
| fp16 / bf16 | 8 |
| fp8 | 16 |

## Padding 消冲突(破 stride 对齐)
- 原理:每行 +1 元素破坏 stride 的 bank 对齐。
- gfx942:stride `HEAD_SIZE=128` 时 bank stride = `128*2/4 = 64`,`64%32=0` 全冲突;**+1 → `129*2/4 = 64.5`** 分数 → 消冲突。
- gfx950 同理:128 仍 `64%64=0` 冲突,**+1 → 64.5** 消。
- 最小 padding = `bank_count/elem_size_bytes`(最坏情形),**通常 1-4 足够**。

## 藏 LDS 写延迟(增大 write-read 距离)
- 在 `ds_write` 后、`s_waitcnt lgkmcnt(0)` 前插**独立**工作。优先级:
  1. 下一阶段 global load(`buffer_load` 异步 ~300+ cycle)
  2. 地址计算 SALU/VALU(~4-8 cycle)
  3. 独立 MFMA 链(~64 cycle/MFMA)
  4. 标量 load(~20 cycle)
- **禁插**:依赖该 LDS 写结果的操作、更多 LDS 操作(争带宽)、超预算涨寄存器的操作。

## 验证清单(优化后)
- 正确性:fp32 累加须 **bit-for-bit**,fp8/bf16 容差内。
- 重 profile:`ds_read`/`ds_write` 与 ds_write 后 `lgkmcnt(0)` stall 下降、且无新 bank 冲突。
- LDS 用量上限:**gfx942 ≤ 65536 bytes**,**gfx950 ≤ 163840 bytes**(gfx950 分配 **1280 字节粒度**)。
- 确认 `waves_per_eu` 占用率没掉。

---
来源: flydsl-tile-programming/SKILL.md, programming-model.md, gemm-optimization/SKILL.md, lds-optimization/SKILL.md, FlyDSL/CLAUDE.md
