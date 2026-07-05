# FlyDSL tracer 编译期-vs-设备控制流:range_constexpr vs range、if 分支变量不外泄

> 类别: 方法论 · 主题标签: tracer, control-flow, authoring-pattern, code-gen

## if 分支变量不外泄 → 用普通 Python helper
- if 分支内定义的变量在分支外不可见、运行时分支写法的完整规则见 pitfalls/03-flydsl-tracer-literal-if-for。
- 纯 side-effect 的 if 可以:`if BR_B1: s_barrier()`(不外泄变量)。

## 控制流四种形态
| 写法 | 语义 | 产物 |
|---|---|---|
| `range_constexpr(K)` | 编译期展开循环(常量边界) | 全展开,无 scf.for |
| `range(runtime_N)` / `range(start,stop,step,init=[...])` | 运行时循环 | scf.for,带 phi/loop-carried state |
| `if const_expr(FLAG)` | 编译期静态 if | 不产 MLIR |
| 普通 `if bid==0` | 运行时 if | scf.IfOp;`== < >=` 等生成 MLIR predicate |

- 低层手动构造:`arith.cmpi` + `arith.unwrap` 传给 `scf.IfOp`。

## range_constexpr vs range 的选择(关键)
- `range_constexpr(...)`:编译期展开的 Python 循环,用于固定内层步数(MFMA cluster、tile repeat、`sched_*` emission)。**在其内部构建 register fragment 的 list 合法,正因为循环被展开**。
- 若把这种循环改成运行时 `scf.for` → fragment list 会被打散、数据落内存,破坏寄存器驻留。
- `range(start,stop,step,init=[...])`(bound 用 `fx.Index`) 是**唯一**能跨迭代携带 loop state 的方式;bound 必须是 `fx.Index` 否则静默 unroll 丢 init= 见 pitfalls/03-flydsl-tracer-literal-if-for。
- loop-carried state 支持类型与 unwrap 规则见 methodology/17-flydsl-prefetch-software-pipeline。

## tracer literal-if 要 unwrap
- 条件直接传给 `scf.IfOp` 时须 unwrap DSL 布尔:
  ```
  cond = arith.unwrap(partition_idx >= visible_tile_count)
  if_op = scf.IfOp(cond, has_else=False)
  ```
- 简单整数比较优先用 DSL 运算符(`lane < c_limit`),而非手写 `arith.cmpi`。

## 裸标量 store 别包 1-wide Vec
- 单 dword store 用 scalar `buffer_store(packed, rout, gid, mask=)` 传**裸标量**。
- 包成 `1-wide Vec.from_elements` 会触发 LLVM `'Do not know how to scalarize'` 崩溃。

## i64 地址偏移
- i64 K-loop offset 乘法必须用 `arith.index(k*BLOCK_K)`(k 是 `range_constexpr` 的 python int)。
- 不能用 `arith.index_cast(T.index, python_int)`——对 python int 会 crash(`_to_raw` 不接受 int,报 `'int' has no _CAPIPtr`)。

---
来源: SKILL.md(flydsl-fp8-gemm-tuning), 13-primus-turbo-prod.md, SKILL.md(flydsl-tile-programming), SKILL.md(debug-flydsl-kernel), programming-model.md, 05-int64-addressing.md
