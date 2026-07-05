# FlyDSL atom 两级类型与编译 pipeline:添加 target atom op、emitAtomCall、graph capture

> 类别: 方法论 · 主题标签: atom-types, compile-pipeline, mlir-passes, graph-capture

## atom 两级类型
- **generic wrapper**(fly 方言,target 无关):`!fly.mma_atom<...>` / `!fly.copy_atom<...,bits>`,IR 里到处出现。wrapper 上每个方法都是 **trampoline**,转发到 payload。
- **target payload**(后端方言如 fly_rocdl):`!fly_rocdl.cdna3.mfma<...>`,知道发哪条 intrinsic。
- **加新 Op 只需定义 payload 类型 + 实现接口方法**,wrapper 和 kernel 级 op 自动工作。

## 4 个接口(只 emitAtomCall 强制)
| 接口 | 适用 | 需实现方法 |
|---|---|---|
| `Fly_MayStaticTypeInterface` | 无状态 atom | `isStatic`/`rebuildStaticValue` |
| `Fly_CopyOpTypeInterface` | 所有 CopyOp | `getThrLayout`/`getThrBitLayoutSrc-Dst-Ref`/`emitAtomCall` |
| `Fly_MmaOpTypeInterface` | 所有 MmaOp | `getThrLayout`/`getShapeMNK`/`getValTypeA-B-C-D`/`getThrValLayoutA-B-C`/`emitAtomCall` |
| `Fly_StatefulOpTypeInterface` | 带 per-call 可变状态 | `getConvertedType`/`getDefaultState`/`setAtomState` |
- 助记:**stateful => 无 MayStatic 接口**。

## emitAtomCall(唯一强制)vs emitAtomCallSSA(可选)
- `emitAtomCall`:处理 **memref 形式**——收寄存器内存指针,自己发 `llvm.load`/`store` 读写寄存器,中间发后端 intrinsic。**足够跑完整编译到二进制**。
- `emitAtomCallSSA`:可选,**仅当 pipeline 含 `fly-convert-atom-call-to-ssa-form` pass** 时需要。该 pass 对 coalescable layout 的 register 操作数用 `PtrLoadOp` 拉成单 SSA 值;此实现只需 intrinsic + 必要的 `LLVM::BitcastOp`。
- **参考实现把 emitAtomCall 写成 emitAtomCallSSA 的 ~15 行 shim**(load -> 调 SSA -> store),保持两路同步。

## stateful atom 状态 = !llvm.struct
- 运行时状态是 `!llvm.struct<(i32,i32,...)>`。
- `getConvertedType` 返回 struct layout;`getDefaultState` 建零初始值(`UndefOp`+`InsertValueOp`)。
- `setAtomState` 写字段:`fieldAttr` 是 `StringAttr`,**必须是 `AtomStateField` 枚举 mnemonic**(如 `'soffset'`/`'imm_offset'`);**未识别字段返回 nullptr,不能静默 success**。
- lowering 时 `LLVM::ExtractValueOp` 按 `getFieldIndex` 读回。**新字段种类要扩后端 `Atom.td` 的 `AtomStateField` 枚举**。

## getThrLayout(发一条指令的线程组线程数 layout)
| 场景 | ThrLayout |
|---|---|
| AMD wave64 MFMA | `(64):(1)` |
| AMD wave32 WMMA | `(32):(1)` |
| 单线程 | `(1):(1)` |
| NVIDIA WGMMA warpgroup | `(128):(1)` |
- CopyOp 的 `getThrLayout` 是**一次 atom call 参与线程数**:per-thread load=1,AMD `ds_read_tr16_b64`=16。
- 用 `FxLayout/FxShape/FxStride/FxThr/FxVal/FxC` 宏(`ThrValLayoutMacro.h.inc`)构造 `LayoutAttr`。

## 编译 pipeline
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

## buffer_tensor 包 SRD
- 用 layout 代数做 GEMM/2D copy 前,tensor 要先 `fx.rocdl.make_buffer_tensor(A)` 包成 AMD **buffer descriptor(SRD)**。
- 低层直接 intrinsic:`buffer_ops.create_buffer_resource(A, max_size=True)` 得 `rsrc`,再 `buffer_load/store`,**绕过 layout 代数最大控制**。

## graph capture 须先 warmup
- FlyDSL JIT kernel **不能直接进 CUDA/HIP graph**——首次 launch 会发不可 capture 的 host work(JIT 触发)。
- 流程:先在普通 stream 上用 **相同 shapes/dtypes/constexpr key** warmup 一次 -> `torch.cuda.synchronize()` -> 在专用 capture stream(`wait_streams` 当前 stream)内 `torch.cuda.graph` 里重新 launch,传 `stream=capture_stream`。
- **warmup 与 capture 之间任何 Constexpr key 变化都会在 capture region 里重新触发 JIT**。

---
来源: add-target-atom-op/SKILL.md, flydsl-kernel-authoring/SKILL.md, flydsl-tile-programming/SKILL.md, optimization-directions.md
