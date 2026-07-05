# FlyDSL 内核编写全流程:骨架选型、layout 代数、tracer 控制流、软件流水、复用范式、debug/OOB、atom 扩展 pipeline

> 类别: 方法论 · 主题标签: flydsl-authoring, layout-algebra, mma-atom, tile-programming, tracer, control-flow, authoring-pattern, code-gen, prefetch, software-pipeline, loop-carried-state, buffer_load, code-style, host-dispatch, fusion, kernel-classify, error-isolation, oob-detection, correctness-verify, atom-types, compile-pipeline, mlir-passes, graph-capture

## FlyDSL authoring 主线:layout 是粘合剂、MMA atom 是锚点、四步 divide/partition

### 内核骨架:先按 5 类模式选
每个 kernel 两部分:`@flyc.kernel` 定义内核体(用 `fx.thread_idx.x`/`fx.block_idx.x`),`@flyc.jit` 定义 launch wrapper(内部 `kernel(...).launch(grid=(x,y,z), block=(bx,1,1), stream=stream)`,从 `kernels.*` 导入)。host 边界 `torch.Tensor` 经 DLPack 自动转 `fx.Tensor`。

| 模式 | 例 | 骨架 |
|---|---|---|
| Elementwise | vecadd/scale/relu | `logical_divide` + copy_atom_call |
| Reduction | sum/max/softmax/layernorm | `buffer_load` + warp shuffle + LDS |
| Tiled Copy | transpose/permute/gather | `zipped_divide` + TiledCopy |
| GEMM | matmul | TiledMma + TiledCopy + LDS |
| Fused | fused attention / GEMM+epilogue | 组合 GEMM + elementwise |

### Layout 代数 = 90% 工作量(核心粘合剂)
- `make_layout(shape, stride)` 定义 coord→index:`Index = sum(coord_i * stride_i)`。
- 例 `(8,16):(1,8)` 列主序,`crd2idx((3,5)) = 3*1 + 5*8 = 43`。
- **`getThrValLayout` 对错会静默产出垃圾结果** → 把 layout 弄对是 FlyDSL 90% 的工作量。
- retile 只重写 view,**不搬数据**;`None` 在 `slice` 里表示保留该 mode。

### Authoring spine(顺序固定,"layout is the glue")
`make_buffer_tensor(T)` → divide(1D 用 `logical_divide` / 2D 用 `zipped_divide`)+ slice 到本 block tile → `make_copy_atom(...)` + `make_mma_atom(MFMA(M,N,K,acc))` → `make_tiled_copy`/`make_tiled_mma`(`make_tiled_copy_A/B/C` 从 MMA 派生 copy partition) → `get_slice`/`thr_slice` + `partition_S/D` 或 `partition_A/B/C` → `make_fragment_like`/`make_fragment_A/B/C` + `retile` → `copy(atom,src,dst)` 和 `gemm(atom,D,A,B,C)`。

### 数据流四步 divide/partition
1. `zipped_divide(Tensor, Tile)` → `(tile_interior, tile_id)`
2. `slice(divided, (None, bid))` 取本 block 的 tile
3. `ThrCopy.partition_S/D` 或 `ThrMma.partition_A/B/C` 取本线程数据
4. `make_fragment_like(partition)` 分配寄存器 tile → `fx.copy`/`fx.gemm` 执行

### MMA atom 是锚点:必须先选
`make_mma_atom(fx.rocdl.MFMA(M,N,K,AccType))` 定硬件指令形状,`make_tiled_mma(mma_atom, make_layout((M_rep,N_rep,K_rep), strides))` 跨线程平铺(如 `(2,2,1)` = 4 个 MMA atom/warp)。
- **MMA atom 固定 A/B/C/D 寄存器布局 → 进而定 LDS tile layout、swizzle 公式、epilogue re-tile。**
- **CHOOSE IT BEFORE designing staging;事后重选会 invalidate 上述全部。**
- copy atom 要和 MMA layout 匹配:`make_tiled_copy_A/B/C(copy_atom, tiled_mma)` → partition → `make_fragment_A/B/C` → `thr_copy.retile(frag)` 让 fragment 与 copy 兼容。

### MFMA 指令
- 直接 intrinsic:`rocdl.mfma_f32_16x16x16_f16`(fp16/bf16)、`rocdl.mfma_f32_16x16x32_fp8`(fp8)、`rocdl.mfma_i32_16x16x32i8`(int8)。
- atom 形式:`make_mma_atom(fx.rocdl.MFMA(16,16,4,Float32))` 等。
- W4A8:A 是 int8,B 打包 int4(2值/字节),kernel 内解包成 int8。

### Copy atom 类型
- `fx.UniversalCopy32b()`(1×f32)、`fx.UniversalCopy(64)`(2×f32)、`fx.UniversalCopy(128)`(4×f32)。
- `fx.rocdl.BufferCopy128b()`(AMD buffer load 4×f32,包 tensor 先 `fx.rocdl.make_buffer_tensor(A)`);spine 中 `make_copy_atom` 支持 `BufferCopy{16,32,64,128}b`/`UniversalCopy`。
- TiledCopy 用 `raked_product(thr_layout, val_layout)` 建交错访问 layout。

### 三级抽象梯子:选能表达该 pattern 的最高层
1. **Layout API**(`make_buffer_tensor` + layout 代数 + atoms + TiledCopy/TiledMma)= 新 kernel 默认。
2. **raw buffer_ops**(`create_buffer_resource` + 手动 byte/element-offset `buffer_load/store`)= 最大微控(手排 `s_waitcnt`、scattered stores、masked OOB)= legacy。
3. **TensorView shim** = 轻量 torch-agnostic 临时 launch / host 单测,但仅 row-major + per-axis tiling,dtype 表只覆盖 `{f32,f16,bf16}`(超出静默返回 `None`)。
- 三层正交可共存(如 A 走 layout API + LDS,预 shuffle 的 B 走手搭 descriptor 直入寄存器)。

### 关键 API 速查(逐字签名)
- `buffer_load(rsrc, offset, vec_width=4, dtype=BFloat16.ir_type)`
- `buffer_store(word, rsrc, byte_offset, cache_modifier=1, offset_is_bytes=True)`
- `create_buffer_resource(tensor, max_size=False, num_records_bytes=I32(nbytes))` — **element-offset 默认**
- `make_row_band_resource(bo.extract_base_index(T), base_row, c_rows, c_cols, elem_bytes)` — 2D int64 rebasing
- LDS 分配:`fx.SharedAllocator().allocate(Smem).peek()`
- LDS strided read:`make_view(base, make_layout(32, PAD)).load()` 正确生成 32 次 `ds_read_b16`
- barrier:`rocdl.s_barrier()`

### 内部类型 & 其他
- 优先 FlyDSL 内部类型而非裸 MLIR op:`Vec=fx.Vector` 包 `vector<NxTy>`;索引 `Vec(v)[i]`、bitcast `Vec(v).bitcast(fx.Float32)`、转换 `v.to(fx.BFloat16)`、算术 `a*b`/`a+b`、splat `Vec.filled(N,val,dtype)`、组向量 `Vec.from_elements`。常量 `fx.Index`/`fx.Int32`/`fx.Int64`/`fx.Float32`,index cast 用 `fx.Int32(v)` 而非 `arith.index_cast`。仅需显式 fastmath flags 时才用裸 `arith.*FOp`。
- 动态 shape / 对齐:`flyc.from_dlpack(tensor).mark_layout_dynamic(leading_dim=0, divisibility=4)`。

## FlyDSL tracer 编译期-vs-设备控制流:range_constexpr vs range、if 分支变量不外泄

### if 分支变量不外泄 → 用普通 Python helper
- if 分支内定义的变量在分支外不可见、运行时分支写法的完整规则见 pitfalls/07-flydsl-frontend-tracer.md。
- 纯 side-effect 的 if 可以:`if BR_B1: s_barrier()`(不外泄变量)。

### 控制流四种形态
| 写法 | 语义 | 产物 |
|---|---|---|
| `range_constexpr(K)` | 编译期展开循环(常量边界) | 全展开,无 scf.for |
| `range(runtime_N)` / `range(start,stop,step,init=[...])` | 运行时循环 | scf.for,带 phi/loop-carried state |
| `if const_expr(FLAG)` | 编译期静态 if | 不产 MLIR |
| 普通 `if bid==0` | 运行时 if | scf.IfOp;`== < >=` 等生成 MLIR predicate |

- 低层手动构造:`arith.cmpi` + `arith.unwrap` 传给 `scf.IfOp`。

### range_constexpr vs range 的选择(关键)
- `range_constexpr(...)`:编译期展开的 Python 循环,用于固定内层步数(MFMA cluster、tile repeat、`sched_*` emission)。**在其内部构建 register fragment 的 list 合法,正因为循环被展开**。
- 若把这种循环改成运行时 `scf.for` → fragment list 会被打散、数据落内存,破坏寄存器驻留。
- `range(start,stop,step,init=[...])`(bound 用 `fx.Index`) 是**唯一**能跨迭代携带 loop state 的方式;bound 必须是 `fx.Index` 否则静默 unroll 丢 init= 见 pitfalls/07-flydsl-frontend-tracer.md。
- loop-carried state 支持类型与 unwrap 规则见本文"FlyDSL 软件流水预取"一节。

### tracer literal-if 要 unwrap
- 条件直接传给 `scf.IfOp` 时须 unwrap DSL 布尔:
  ```
  cond = arith.unwrap(partition_idx >= visible_tile_count)
  if_op = scf.IfOp(cond, has_else=False)
  ```
- 简单整数比较优先用 DSL 运算符(`lane < c_limit`),而非手写 `arith.cmpi`。

### 裸标量 store 别包 1-wide Vec
- 单 dword store 用 scalar `buffer_store(packed, rout, gid, mask=)` 传**裸标量**。
- 包成 `1-wide Vec.from_elements` 会触发 LLVM `'Do not know how to scalarize'` 崩溃。

### i64 地址偏移
- i64 K-loop offset 乘法必须用 `arith.index(k*BLOCK_K)`(k 是 `range_constexpr` 的 python int)。
- 不能用 `arith.index_cast(T.index, python_int)`——对 python int 会 crash(`_to_raw` 不接受 int,报 `'int' has no _CAPIPtr`)。

## FlyDSL 软件流水预取:prologue/steady/epilogue 三段、loop-carried state 携带每个 next

### 三段结构(software-pipelined prefetch = three regions)
- **prologue(序言)**:发起 iteration 0 的 load(首 tile 数据),`_unwrap` 成 raw `ir.Value` 作 `init_state`。辅助数据(block table、per-block scale、offsets、在线 softmax m/l、accumulator)也必须一并作为 init state。
- **steady(循环体)**:`for iv, state in range(_start, _stop, _step, init=init_state)`,内部构建带 SSA phi 的运行时 `scf.for`(跑 N-1 trips)。每迭代:解包 state→swap → 读上次预取好的值 → 发 `prefetch(iv+1)`(下一 tile 的异步 load)→ `compute(current)` → `yield [acc] + next_tile`。
- **epilogue(尾段)**:循环退出后 `results` 持有最后 yield 的值,消费末次预取(末迭代)。
- **stop 取 N-1**:末迭代放进 epilogue,循环体只跑到 N-1。

### loop-carried state 携带规则
- **携带每个 next**:load 依赖的每一个输入都要进 `init=`(offsets、block-table indices、per-block scales、online-softmax m/l、accumulators)。
- **类型保留**:优先用 FlyDSL 内部类型(`fx.Int32`/`fx.Float32`/`Vector`/`ArithValue`);支持的 state 类型:f32 标量、vector、i32、i64、index。
- **只在边界 unwrap**:仅在 `init=`/`yield` 边界、或底层 helper 明确要求 raw `ir.Value` 处才 unwrap:`v.ir_value() if hasattr(v,'ir_value') else v`;body 内保持 typed。

### buffer_load / 异步语义(WHY)
- `buffer_load`(GPU global load)**异步立即返回**、后台取数,只在消费指令处才需数据。
- 编译器在**首个消费者**处插 `s_waitcnt`——提前发 load 只是给 scheduler slack,它**本身不设 `vmcnt(N)` 也不移除 barrier**。
- prefetch/double-buffer 把延迟藏在 compute 后:总时间从 `N*(load+compute)` 降到约 `load + N*max(load,compute)`。

### A0 跨 tile LDS 预取
- `gpu.barrier()` 完成后 LDS 有效,**立即**把第一个 A pack 从 LDS 读进 VGPR(`lds_load_packs_k64`),让首个 `ds_read` 延迟(~20–40 cycle)藏在随后的 VMEM load 后面。

### 实证:PA decode kernel(112us,0.75× vs Gluon)
- 携带 **15 个 loop-carried 值**:8×`vector<4xi32>` K 数据、1×i32 partition_start、2×i32 block table、2×f32 running_max/sum(在线 softmax)、2×`vector<4xf32>` PV 累加器。
- ISA 结果:8 个 K-prefetch `buffer_load_dwordx4` 出现在 loop body 末尾(PV MFMA 之后),与 MFMA 流水 drain 重叠;序言 8 个 K loads,epilogue 只 8 个 V loads。

## FlyDSL 复用范式:抄既有变体、共享坐标/寻址只分叉 loop body、融合单发射省 host dispatch

### 抄既有变体(最大化复用原语、禁魔术数)
- 新 kernel 变体逐字段照抄既有模式:4-wave↔8-wave、NN↔NT/TN、tensorwise↔mxfp4。命名/错误处理/dispatch 候选 grid/group_n band 都跟着抄。
- 必须复用既有原语:`mask_a_tail` / `emit_wholeloop_tile` / `grouped_block_mn` / `S2RLoader` / `G2SLoader`。
- 禁魔术数:tile/BLOCK 从 shape 推导,不写死。

### turbo 代码风格约定
- 注释一律英文,禁中文/日文。
- 函数命名跟 turbo 现有风格:`_grouped_<noun>`(如 `_grouped_block_mn`)、`_wgrad_<verb_or_noun>_<variant>`(如 `_wgrad_wholeloop_asm_3buf`)。别自造缩写:`_wl3buf_fused_tail_split` 错,应写 `_wholeloop_tail_split_3buf`。
- 能复用 `gemm_helper.py` 就必须复用:`ceildiv`、`_readfirstlane_i32`、`xcd_remap_pid`、`make_fp8_buffer_tensor_rebased`、`S2RLoader`/`S2RLoaderTr`、`_robust_time` 都在里面。先 grep 确认没有再写。

### 共享坐标/寻址、只分叉 loop body
- masked 和 persist 两个 wgrad kernel 共用:
  - `_wgrad_block_mn`:dispatch → `(group_idx, block_m, block_n)`
  - `_wgrad_rebase`:i64 SRD rebase → `(a_div, b_div)`
- 唯一分叉的是 K-loop body:masked = chunked 4-buffer;persist = `scf.for` 2-stage prefetch。
- 共用坐标/寻址逻辑、只把 loop body 分叉,是 FlyDSL 的标准复用范式。

### 融合单发射(省 host dispatch)
- 模式:GEMM/preshuffle 工厂返回裸 `@flyc.kernel`,由一个 `@flyc.jit` stub 在同一 stream 上依次 `.launch()` 发射 preA→preB→GEMM → 一次 Python dispatch(stub 内多 kernel enqueue 是廉价 C++ 调用)。
- WHY:eager 下每次 `@flyc.jit` dispatch ~38µs,compiled 直调 ~6µs;小 shape 被 host 开销主导 → 融合是 eager 小 shape 的关键。
- preshuffle v2 优化:一 thread 出全部 NG=4 个 output dword,共享的 4 个源 int32 只读一次(v1 是一 thread 一 output dword,4 个 thread 各重读同样 4 行 → 4× cross-thread 读放大)。grid 缩到 1/NG,输入 HBM 读流量 ~4×↓,device 时长减半(~16→8µs),bit-exact。

### 非持久 vs 持久
- fwd 用非持久(non-persistent, one-tile-per-WG)比持久 `scf.for` 快约 11%(省 outer tile-loop `scf.for` 调度惩罚)。
- 前提:L2 swizzle 也移植进非持久 kernel。

### helper 放置规则(先搜索复用,勿散落勿重复)
- 共享 kernel helper → `kernels/kernels_common.py`:`get_warp_size`/`dtype_to_elem_type`/`validate_moe_dtypes`/`_if_then`(SCF context manager)。
- 领域专用 → `moe_common.py`/`layout_utils.py`/`pipeline_utils.py`/`fp8_gemm_utils.py`/`dpp_utils.py`/`mfma_epilogues.py`/`mfma_preshuffle_pipeline.py`。
- DSL 级 numeric/arith/type → `expr/utils/arith.py` 或 `expr/numeric.py`。
- 编译/运行时级 → `utils/`。

## 内核分类骨架、错误隔离 debug 流程、OOB 静态区间分析

### 1. 按 5 类模式选骨架
5类骨架分类表见本文"FlyDSL authoring 主线"一节。

### 2. debug 症状分类(先看错哪、错多少)
- **全 NaN** → softmax `-inf-(-inf)` 或 `1/0`(用 `.select` 加 guard);或未初始化 buffer。
- **全 0** → 输出地址/stride 错、partition slot 错、buffer 未写。用 sentinel `-999.0` 初始化即可暴露被跳过的写。
- **>50% 错** → partition 数错 / layout 不匹配 / 寻址错。
- **1-5% 小错** → FP8 requant 容差 / scale 被重复施加 / off-by-one mask。可能不是 bug。
- **编译错** → 类型 / loop-carried state / `range` vs `range_constexpr`。

### 3. 系统 debug 流程(最便宜的先做)
1. **无条件清缓存**:`rm -rf ~/.flydsl /tmp/flydsl*`,清 lru_cache。绝大多数"改了没用"都是 stale cache。
2. **全 1 输入**(Q/K/V=1.0)过了 → layout OK,是数据问题。
3. **单 partition**(one_shot)过了 → 多 partition / reduce 的 bug。
4. host 端打印 shape / stride / NaN。
5. 对比中间 buffer(`exp_sums` / `max_logits` / `temp_out`)。
6. 怀疑 layout 时**手动追一个线程地址**:tid=0(lane16id=0, rowid=0, warp_id=0)。
7. 验证 MFMA operand order 和 `range` vs `range_constexpr`。

**合成输入的坑**:Q/K/V=1.0 使 softmax 均匀、PV 塌成已知值,任何偏差都是纯 layout/寻址 bug——**但均匀输入不暴露 swapped V/P MMA operand**,这类 bug 必须对 reference 交叉验证。

### 4. OOB 四类边界
| 类型 | 检出手段 |
|---|---|
| 物理 allocation OOB | HIP illegal address 能测 |
| 逻辑对象 OOB(跨 row/head/tile) | 静态区间分析(物理工具测不出,仍在同一 allocation 内) |
| lane/thread 所有权 OOB(读别 lane 槽) | 静态分析 + printf |
| LDS/共享内存 OOB | 静态分析 |

**静态区间分析法**:每个访存写 `start=base+offset`、`end=start+vec_width-1`、`legal=[obj_base, obj_base+extent-1]`,代入 lane / warp_id / `range_constexpr` 的已知范围:
- `max(end) > legal_end` → 上界 OOB。
- `min(start) < obj_base` → 下界 OOB。

运行时检查用**窄范围** `fx.printf` 只打失败坐标,避免太多 lane 打印淹没信号。

### 5. 新内核正确性验证清单
- 检查 tid 前先 `torch.cuda.synchronize()`。
- `torch.allclose(atol=1e-5)` 对比参考。
- 确认 copy atom 宽度合法(完整规则见 pitfalls/07-flydsl-frontend-tracer.md)。
- 编译期常量用 `Constexpr[int]`,运行时值用 `Int32`。
- GEMM tile size 必须匹配 MFMA 指令形状。
- 加新 atom 后:**先跑 FileCheck,再跑 1-wave 端到端 Python kernel**,才能信 layout。

## FlyDSL atom 两级类型与编译 pipeline:添加 target atom op、emitAtomCall、graph capture

### atom 两级类型
- **generic wrapper**(fly 方言,target 无关):`!fly.mma_atom<...>` / `!fly.copy_atom<...,bits>`,IR 里到处出现。wrapper 上每个方法都是 **trampoline**,转发到 payload。
- **target payload**(后端方言如 fly_rocdl):`!fly_rocdl.cdna3.mfma<...>`,知道发哪条 intrinsic。
- **加新 Op 只需定义 payload 类型 + 实现接口方法**,wrapper 和 kernel 级 op 自动工作。

### 4 个接口(只 emitAtomCall 强制)
| 接口 | 适用 | 需实现方法 |
|---|---|---|
| `Fly_MayStaticTypeInterface` | 无状态 atom | `isStatic`/`rebuildStaticValue` |
| `Fly_CopyOpTypeInterface` | 所有 CopyOp | `getThrLayout`/`getThrBitLayoutSrc-Dst-Ref`/`emitAtomCall` |
| `Fly_MmaOpTypeInterface` | 所有 MmaOp | `getThrLayout`/`getShapeMNK`/`getValTypeA-B-C-D`/`getThrValLayoutA-B-C`/`emitAtomCall` |
| `Fly_StatefulOpTypeInterface` | 带 per-call 可变状态 | `getConvertedType`/`getDefaultState`/`setAtomState` |
- 助记:**stateful => 无 MayStatic 接口**。

### emitAtomCall(唯一强制)vs emitAtomCallSSA(可选)
- `emitAtomCall`:处理 **memref 形式**——收寄存器内存指针,自己发 `llvm.load`/`store` 读写寄存器,中间发后端 intrinsic。**足够跑完整编译到二进制**。
- `emitAtomCallSSA`:可选,**仅当 pipeline 含 `fly-convert-atom-call-to-ssa-form` pass** 时需要。该 pass 对 coalescable layout 的 register 操作数用 `PtrLoadOp` 拉成单 SSA 值;此实现只需 intrinsic + 必要的 `LLVM::BitcastOp`。
- **参考实现把 emitAtomCall 写成 emitAtomCallSSA 的 ~15 行 shim**(load -> 调 SSA -> store),保持两路同步。

### stateful atom 状态 = !llvm.struct
- 运行时状态是 `!llvm.struct<(i32,i32,...)>`。
- `getConvertedType` 返回 struct layout;`getDefaultState` 建零初始值(`UndefOp`+`InsertValueOp`)。
- `setAtomState` 写字段:`fieldAttr` 是 `StringAttr`,**必须是 `AtomStateField` 枚举 mnemonic**(如 `'soffset'`/`'imm_offset'`);**未识别字段返回 nullptr,不能静默 success**。
- lowering 时 `LLVM::ExtractValueOp` 按 `getFieldIndex` 读回。**新字段种类要扩后端 `Atom.td` 的 `AtomStateField` 枚举**。

### getThrLayout(发一条指令的线程组线程数 layout)
- 各硬件协作线程数表见 pitfalls/07-flydsl-frontend-tracer.md
- CopyOp 的 `getThrLayout` 是**一次 atom call 参与线程数**:per-thread load=1,AMD `ds_read_tr16_b64`=16。
- 用 `FxLayout/FxShape/FxStride/FxThr/FxVal/FxC` 宏(`ThrValLayoutMacro.h.inc`)构造 `LayoutAttr`。

### 编译 pipeline
```
Python(@flyc.kernel/@flyc.jit)
  -> AST rewriting(for/if -> scf.for/scf.if)
  -> MLIR tracing(生成 Fly dialect + gpu/arith/scf/memref/vector)
  -> MlirCompiler.compile(Fly -> ROCDL -> LLVM -> HSACO)
  -> JITCFunction
```
关键 pass(顺序):
- `fly-rewrite-func-signature`
- `fly-layout-lowering`(layout 代数降为算术)
- `fly-convert-atom-call-to-ssa-form` + `fly-promote-regmem-to-vectorssa`
- `convert-fly-to-rocdl`
- `gpu-module-to-binary{format=fatbin}`

### buffer_tensor 包 SRD
- 用 layout 代数做 GEMM/2D copy 前,tensor 要先 `fx.rocdl.make_buffer_tensor(A)` 包成 AMD **buffer descriptor(SRD)**。
- 低层直接 intrinsic:`buffer_ops.create_buffer_resource(A, max_size=True)` 得 `rsrc`,再 `buffer_load/store`,**绕过 layout 代数最大控制**。

### graph capture 须先 warmup
- FlyDSL JIT kernel **不能直接进 CUDA/HIP graph**——首次 launch 会发不可 capture 的 host work(JIT 触发)。
- 流程:先在普通 stream 上用 **相同 shapes/dtypes/constexpr key** warmup 一次 -> `torch.cuda.synchronize()` -> 在专用 capture stream(`wait_streams` 当前 stream)内 `torch.cuda.graph` 里重新 launch,传 `stream=capture_stream`。
- **warmup 与 capture 之间任何 Constexpr key 变化都会在 capture region 里重新触发 JIT**。

---
来源: flydsl-tile-programming/SKILL.md, flydsl-kernel-authoring/SKILL.md, programming-model.md, overview.md, FlyDSL/CLAUDE.md, mxfp8-8wave-devloop/SKILL.md, add-target-atom-op/SKILL.md(ThrVal/ThrBit 静默垃圾结果), SKILL.md(flydsl-fp8-gemm-tuning), 13-primus-turbo-prod.md, SKILL.md(debug-flydsl-kernel), 05-int64-addressing.md, prefetch-data-load/SKILL.md, gemm-optimization/SKILL.md, pr-merge-gate/SKILL.md, remote-sync/SKILL.md, 04-tn-wgrad-kernel.md, 14-fused-preshuffle-e2e.md, 02-nt-fwd-kernel.md, oob-detection/SKILL.md, add-target-atom-op/SKILL.md（"先跑 FileCheck 再跑 1-wave 端到端"一条出自此文件 Step 9）, optimization-directions.md
