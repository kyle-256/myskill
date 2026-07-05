# FlyDSL 前端编写坑：buffer offset 单位/SmemPtr view cache/absf 缺失/DLTensorAdaptor

> 类别: 踩过的坑 · 主题标签: frontend-authoring, mlir-dominance, buffer-ops, autotune

**buffer_load/buffer_store offset 单位是 ELEMENTS 不是 bytes**
- `buffer_ops.buffer_load(rsrc, offset, vec_width, dtype)` 的 offset 单位是 dtype 个元素，内部会转字节。写成字节偏移会越界读错数据。
- 症状是垃圾数据不是 crash（off-by-sizeof(elem) 地址 bug）。例：FP8 数据按字节寻址、但用 `dtype=T.i32` 加载时要把字节地址除以 4：`buffer_load(k_rsrc, k_addr_bytes//4, vec_width=4, dtype=T.i32)`。

**进 epilogue 前必须清 `SmemPtr._view_cache = None`**
- `SmemPtr.get()` 会缓存它创建的 view 到 `SmemPtr._view_cache`。若在运行时循环体内调用，缓存的 view 定义在循环 scope；epilogue（循环外/退出 scf.for 后）复用它会触发 MLIR SSA dominance 报错。
- 解法：循环后、epilogue 前手动 `my_smem_ptr._view_cache = None`。已验证坑。
- 配套规矩：raw memref 在 block 顶部一次性取好，让它 dominate 所有子 scf.for/scf.if region。if 分支变量不外泄的完整规则见 03-flydsl-tracer-literal-if-for。

**run-time loop 边界必须用 `fx.Index(...)` 不能用 Python int**
- 细节/后果见 03-flydsl-tracer-literal-if-for。

**`arith.absf` 在 FlyDSL 不存在**
- 求绝对值必须用组合：`neg = -v; is_neg = v < zero; out = is_neg.select(neg, v)`（Vector/ArithValue 运算符）。

**DLTensorAdaptor 缓存旧 context → segfault** ❌ 别再试
- 调 `@jit` 且 Constexpr 值变化时，不要用 `flyc.from_dlpack()` 预包装 tensor。DLTensorAdaptor 缓存首个 `ir.Context` 的 MLIR 类型；新 context 创建后旧类型失效导致 segfault。
- 应直接传原始 `torch.Tensor`。

**Vector.store 要求存 vector 不是标量**
- 错：`Vec(scalar_i32).store(...)`
- 对：`vec = Vec.from_elements([scalar_i32], fx.Int32); vec.store(lds_ptr, [idx])`

**copy/atom width 约束：`vec_width * sizeof(elem) <= 128b`**
- atom 上限 128b，没有 BufferCopy256b。mismatch 会静默不向量化或损坏数据。
- 按 `VEC*elem_bits` 选 atom：f16x8→128b，i8x8→64b，f32x1→32b。

**嵌套 helper 不要 mutate 捕获的外层变量**
- `@flyc.kernel`/`@flyc.jit` 内的嵌套 helper 可读捕获值，但不应 mutate 捕获的外层变量——显式传值并返回更新后的 state。

---
来源: prefetch-data-load/SKILL.md, flydsl-kernel-authoring/SKILL.md, debug-flydsl-kernel/SKILL.md, FlyDSL/CLAUDE.md, programming-model.md
