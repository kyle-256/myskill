# FlyDSL authoring 主线:layout 是粘合剂、MMA atom 是锚点、四步 divide/partition

> 类别: 方法论 · 主题标签: flydsl-authoring, layout-algebra, mma-atom, tile-programming

## 内核骨架:先按 5 类模式选
每个 kernel 两部分:`@flyc.kernel` 定义内核体(用 `fx.thread_idx.x`/`fx.block_idx.x`),`@flyc.jit` 定义 launch wrapper(内部 `kernel(...).launch(grid=(x,y,z), block=(bx,1,1), stream=stream)`,从 `kernels.*` 导入)。host 边界 `torch.Tensor` 经 DLPack 自动转 `fx.Tensor`。

| 模式 | 例 | 骨架 |
|---|---|---|
| Elementwise | vecadd/scale/relu | `logical_divide` + copy_atom_call |
| Reduction | sum/max/softmax/layernorm | `buffer_load` + warp shuffle + LDS |
| Tiled Copy | transpose/permute/gather | `zipped_divide` + TiledCopy |
| GEMM | matmul | TiledMma + TiledCopy + LDS |
| Fused | fused attention / GEMM+epilogue | 组合 GEMM + elementwise |

## Layout 代数 = 90% 工作量(核心粘合剂)
- `make_layout(shape, stride)` 定义 coord→index:`Index = sum(coord_i * stride_i)`。
- 例 `(8,16):(1,8)` 列主序,`crd2idx((3,5)) = 3*1 + 5*8 = 43`。
- **`getThrValLayout` 对错会静默产出垃圾结果** → 把 layout 弄对是 FlyDSL 90% 的工作量。
- retile 只重写 view,**不搬数据**;`None` 在 `slice` 里表示保留该 mode。

## Authoring spine(顺序固定,"layout is the glue")
`make_buffer_tensor(T)` → divide(1D 用 `logical_divide` / 2D 用 `zipped_divide`)+ slice 到本 block tile → `make_copy_atom(...)` + `make_mma_atom(MFMA(M,N,K,acc))` → `make_tiled_copy`/`make_tiled_mma`(`make_tiled_copy_A/B/C` 从 MMA 派生 copy partition) → `get_slice`/`thr_slice` + `partition_S/D` 或 `partition_A/B/C` → `make_fragment_like`/`make_fragment_A/B/C` + `retile` → `copy(atom,src,dst)` 和 `gemm(atom,D,A,B,C)`。

## 数据流四步 divide/partition
1. `zipped_divide(Tensor, Tile)` → `(tile_interior, tile_id)`
2. `slice(divided, (None, bid))` 取本 block 的 tile
3. `ThrCopy.partition_S/D` 或 `ThrMma.partition_A/B/C` 取本线程数据
4. `make_fragment_like(partition)` 分配寄存器 tile → `fx.copy`/`fx.gemm` 执行

## MMA atom 是锚点:必须先选
`make_mma_atom(fx.rocdl.MFMA(M,N,K,AccType))` 定硬件指令形状,`make_tiled_mma(mma_atom, make_layout((M_rep,N_rep,K_rep), strides))` 跨线程平铺(如 `(2,2,1)` = 4 个 MMA atom/warp)。
- **MMA atom 固定 A/B/C/D 寄存器布局 → 进而定 LDS tile layout、swizzle 公式、epilogue re-tile。**
- **CHOOSE IT BEFORE designing staging;事后重选会 invalidate 上述全部。**
- copy atom 要和 MMA layout 匹配:`make_tiled_copy_A/B/C(copy_atom, tiled_mma)` → partition → `make_fragment_A/B/C` → `thr_copy.retile(frag)` 让 fragment 与 copy 兼容。

## MFMA 指令
- 直接 intrinsic:`rocdl.mfma_f32_16x16x16_f16`(fp16/bf16)、`rocdl.mfma_f32_16x16x32_fp8`(fp8)、`rocdl.mfma_i32_16x16x32i8`(int8)。
- atom 形式:`make_mma_atom(fx.rocdl.MFMA(16,16,4,Float32))` 等。
- W4A8:A 是 int8,B 打包 int4(2值/字节),kernel 内解包成 int8。

## Copy atom 类型
- `fx.UniversalCopy32b()`(1×f32)、`fx.UniversalCopy(64)`(2×f32)、`fx.UniversalCopy(128)`(4×f32)。
- `fx.rocdl.BufferCopy128b()`(AMD buffer load 4×f32,包 tensor 先 `fx.rocdl.make_buffer_tensor(A)`);spine 中 `make_copy_atom` 支持 `BufferCopy{16,32,64,128}b`/`UniversalCopy`。
- TiledCopy 用 `raked_product(thr_layout, val_layout)` 建交错访问 layout。

## 三级抽象梯子:选能表达该 pattern 的最高层
1. **Layout API**(`make_buffer_tensor` + layout 代数 + atoms + TiledCopy/TiledMma)= 新 kernel 默认。
2. **raw buffer_ops**(`create_buffer_resource` + 手动 byte/element-offset `buffer_load/store`)= 最大微控(手排 `s_waitcnt`、scattered stores、masked OOB)= legacy。
3. **TensorView shim** = 轻量 torch-agnostic 临时 launch / host 单测,但仅 row-major + per-axis tiling,dtype 表只覆盖 `{f32,f16,bf16}`(超出静默返回 `None`)。
- 三层正交可共存(如 A 走 layout API + LDS,预 shuffle 的 B 走手搭 descriptor 直入寄存器)。

## 内部类型 & 其他
- 优先 FlyDSL 内部类型而非裸 MLIR op:`Vec=fx.Vector` 包 `vector<NxTy>`;索引 `Vec(v)[i]`、bitcast `Vec(v).bitcast(fx.Float32)`、转换 `v.to(fx.BFloat16)`、算术 `a*b`/`a+b`、splat `Vec.filled(N,val,dtype)`、组向量 `Vec.from_elements`。常量 `fx.Index`/`fx.Int32`/`fx.Int64`/`fx.Float32`,index cast 用 `fx.Int32(v)` 而非 `arith.index_cast`。仅需显式 fastmath flags 时才用裸 `arith.*FOp`。
- 动态 shape / 对齐:`flyc.from_dlpack(tensor).mark_layout_dynamic(leading_dim=0, divisibility=4)`。

---
来源: flydsl-tile-programming/SKILL.md, flydsl-kernel-authoring/SKILL.md, programming-model.md, overview.md, FlyDSL/CLAUDE.md
