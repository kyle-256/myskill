# FlyDSL tracer 字面 if/for 坑：编译期常量分叉必须在外层 Python 作用域选好

> 类别: 踩过的坑 · 主题标签: flydsl-tracer, const_expr, loop-carried-phi, frontend-semantics

## 字面 if 被 rewrite 成 scf.if，即使条件是编译期常量
- `@flyc.kernel`/`@flyc.jit` 装饰的函数体（含内部嵌套 `def` 如 `_do_tile`）源码里字面出现的 `if`/`for` 会被 AST-rewriter 转成设备侧控制流（scf.if/scf.for），**哪怕分支条件是纯 Python 编译期常量**（如 `trans_b`）。想按编译期常量分叉行为，必须在外层未被追踪的 Python 作用域先选好分支（生成不同闭包 / 调用不同模块级函数），内核体内只做函数调用，**不能写字面 `if trans_b`**。
- 裸 python `if _flag:` 会让**两个分支都被 trace**（例如两个 SharedAllocator → `Only one SharedAllocator` 报错）。编译期分支必须写 `if const_expr(_flag):` 才走单一分支。
- ❌ 别再试：把 NN/NT 两个 4-wave 编译函数硬合并、在 `_do_tile` 里写 `if trans_b`。先 NameError（变量只在某个 scf.if 分支可见）；改闭包写法后，重复调用同一已编译 `@flyc.jit` 函数时又触发 FlyDSL 全局漂移检测报错（`_WL_ASM_CACHE_3BUF` 在 trace 期间被 mutate，和首次编译快照对不上）。**最终定案：拆成两个独立编译函数**，只共享跟 `trans_b` 无关的辅助函数（如 `_grouped_4wave_tile_scan`）。

## const_expr 只包编译期已知值，绝不包运行时 GPU 值
- ❌ 别再试：`if const_expr(lane==c_zero)` / `const_expr(gpu.thread_id('x'))` / `const_expr(warp_id...)`。即使 `@flyc.kernel(known_block_size=...)`，`gpu.thread_id('x')`、`lane`、`warp_id` 都是**运行时 SSA 值**（编译器只知取值范围，不知当前 lane/thread 实例）。正确写法是运行时 `if lane == c_zero`（AST rewriter 下沉为 scf.IfOp）或 `.select()`。
- `const_expr` 只用于 trace/编译期已知值：Python bool、constexpr 参数、循环展开 / 类型 / layout 分支（如 `if const_expr(trans_v)`）。

## 字面 for + Python int 边界 = 静默展开丢 init=
- FlyDSL loop bounds **必须是 `fx.Index(...)`** 才能 lower 成真正的 scf.for 并带 loop-carried operands（phi 节点）。普通 Python int 会让 trace 把它当 Python range **UNROLL 展开，并静默丢掉 `init=`** —— 任何 prefetch/double-buffer 对 MLIR 都变不可见，**无报错**。这是流水线被静默禁用的头号原因。
- `@flyc.kernel` 内 `range()` 被 AST rewriter 转成 MLIR 运行时循环，循环变量 `i` 变成 ArithValue，**无法索引 Python list**。要编译期展开必须用 `range_constexpr(N)`，此时 `i` 是 Python int。
- 真正的 loop-carried 预取用运行时 `range(start,stop,step,init=...)` 制造 SSA phi。

## Python `data = next_data` 不是 phi
- FlyDSL 里 Python 级 `data = next_data` 重绑定**不产生 loop phi**：两个名字别名同一 SSA 值，load 被当循环不变量外提（hoisted）。同理 Python 级 `for _pi in range(N)` 被 trace 成 N 份平铺副本再由 LLVM 重卷，交换对 MLIR 不可见。
- 预取只在预取值**穿过 `init=` 和 `yield`** 时才生效。**Python swap 不是 phi。**
- ❌ 局限：scf.for 的 `iter_args` 目前只能 carry 简单值，**无法 carry ping-pong buffer 状态**（多值 carry 不支持）。

## 前端 if/for 语义限制（纯 Python 合法但和 MLIR 构造冲突）
- 分支局部定义：值只在某个 `if`/`else` arm 内定义、arm 外使用 → **静默破坏 MLIR result typing**。必须把定义提到分支上方，或 yield 单一 merged 值。
- 前端会把 if 分支体抽成 `__then_*`/`__else_*` 函数；分支内引用的名字若定义在外层可能触发 **NameError**。标量/向量类型不匹配、Python int 出现在需 DSL 值处也是常见 wizard 报错。
- 不要在嵌套 helper 里 mutate 捕获的外部变量（只读闭包 OK，写要走显式参数 + 返回值）。
- 避免 early return：不要把 `return`/`yield` 放进 if/else 分支（前端靠单一显式出口确定结果类型）。
- 有副作用 / 循环携带值 / 分支局部定义的**运行时分支**，应拆成 local helper 并把 `if` 包进 local `@flyc.jit` dispatch 函数里调用，不能直接写在 kernel 体内。
- 深度内核调试（all-1s 测试、单 partition 隔离、MFMA 操作数 layout 检查）用 debug-flydsl-kernel skill。

## 循环变量名残留被 JIT 折叠成常量
- `for s in range_constexpr(4)` 结束后 Python 的 `s=3` 仍留在作用域，紧接着写 `s=(grow>>4)&3` **可能被折叠成常量 3**（变量名冲突导致 trace 折叠）。preshuffle 索引里的子块变量**必须改名为 `sub`** 而非复用 `s`。

## dynamic scf.for loop-carry LDS 触发 lowering bug + 单值 carry 限制
- 两条链式 dynamic `scf.for` loop-carry LDS shared-ptr 会触发 lowering bug：`unrealized_conversion_cast fly.ptr→llvm.ptr<3>` remained live（只能 balanced/无 tail 才跑）。❌ 别再试。
- 动态 `scf.for` 只能 loop-carry **单个 MLIR 值**，carry 不了 32 个 vec 的 list（同上 iter_args 多值 carry 限制）。
- module-level fn 里的 `if`（如 `if wave_m==1`）不被 AST 改写，会 `bool()` dynamic 报错；**conditional-barrier 必须 inline 在 kernel body**。

---
来源: remote-sync/SKILL.md, 08-deadends.md, prefetch-data-load/SKILL.md, flydsl-kernel-authoring/SKILL.md, flydsl-tile-programming/SKILL.md, debug-flydsl-kernel/SKILL.md, programming-model.md, agpr_phase5_lds.md, mxfp8-8wave-devloop/SKILL.md, mxfp8-grouped-gg-devloop/SKILL.md
