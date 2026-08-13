# FlyDSL 前端/tracer 踩坑合集：JIT 缓存失效假象、字面 if/for 语义、编写陷阱、ThrVal/atom layout 静默错

> 类别: 踩过的坑 · 主题标签: jit-cache, cache-invalidation, emit-knobs, false-negative, flydsl-tracer, const_expr, loop-carried-phi, frontend-semantics, frontend-authoring, mlir-dominance, buffer-ops, autotune, ThrVal-layout, atom-op, MmaOp/CopyOp, 静默错

## FlyDSL JIT 缓存不失效假象：key 不 hash 模块级方法/emit 旋钮/env-gate

"改了代码没生效"的第一大原因。FlyDSL 激进缓存已编译 kernel（磁盘 `~/.flydsl/cache`、`/tmp/flydsl*`；`@functools.lru_cache` 还有内存层），cache-key 只覆盖很窄的一部分源码。改动落在 key 之外 → 命中 stale kernel → 改坏了仍出旧正确 SNR，是最阴险的假象。

**cache-key 到底 hash 什么（正常失效的范围）**
- ① flydsl 自身包源码
- ② `@kernel` 函数及其**嵌套 def / 嵌套 class** 的源码
- ③ `FLYDSL_EXTRA_SOURCE_DIRS` 目录下的 `.py`
- ④ dtype（注意：**不总是 tensor shape**）

**key 之外的东西改了都不失效（假象来源）**
- **模块级辅助类的方法源码**：`_collect_dependency_sources` 不跟进被实例化类的方法。改一个模块级 class 的方法（如 `FusedQuantG2SLoader`）缓存不失效，跑出旧版。
- **env-gated 探针** `const_expr(env)`：不改 `@kernel` 源码 hash → 命中旧核，改动根本没生效但 SNR 仍正确。换 env-gate 必 `rm -rf /root/.flydsl/cache`。
- **FP4_* emit 旋钮**：cache-key tuple **不含** `FP4_MMORD` / `FP4_SINNER` / `FP4_ACC_DIST16` / `FP4_DSRD` / block-dict。扫这些旋钮必须每次 `rm -rf /root/.flydsl/cache`，否则复用 stale kernel 看似无差异 → 误判。
  - 反证：本仓早期 "MMORD 变体全判负"、"6=1×8/7=4×4 racy" 都是没清 cache 的假测；清缓存后实为 **mm9 +2%**、新 **6-10 全 det0**。
  - 反例（key 里，安全）：`SS_UNROLL` / `WLVMCN` / `INPLACE` / `PIN` 在 key 里，不受此坑影响。
- **C++ lowering / pass**：不在 traced 闭包内，磁盘缓存不失效。
- **不在 traced 闭包内的 helper 代码**：同理不失效。
- **应 re-specialize 的 shape 但共享 dtype key**：被 stale cache 掩盖。相关坑：`make_buffer_tensor(t)` 要保持默认 `max_size=True`——因为 key on dtype not shape，首个 shape 捕获的 `num_records` 会在后续更大 tensor 上静默截断。

- **comgr 缓存**：除 `~/.flydsl/cache` 外还有 `/root/.cache/comgr`。
- **in-memory ASM 缓存 `_MX_WL_ASM_CACHE`**：**任何**改变 asm 发射的编译期变体都必须进 key —— 不只是 env flag，**发射函数自己的参数同样算**。实证：`call_mxfp4_wholeloop` 新增 `half_n` 参数分岔出第二份 K-loop，但 key 没跟着加，两个臂跑的是同一个二进制，A/B 报「半-N 慢 1.6%」——**假负，整次测量作废**；key 补上 `half_n` 后实测是正收益并最终 ship。判据同「金标准验证」：一组本该不同的变体给出逐字节相同的时间就是 key 漏了。in-process 扫变体时用 `FLYDSL_RUNTIME_ENABLE_CACHE=0` 旁路磁盘层（内存层仍需 key 正确）。

**金标准验证（唯一可信的判缓存是否失效的办法）**
- 故意把核**改算错**（如 scale×2）**不清缓存**跑：出 NaN/错 = 缓存正常失效；仍出旧正确 SNR = 缓存陈旧，你之前所有结果都不可信。
- **rocprofv3 寄存器 ground truth**：一组本该不同的探针若给出逐字节相同的时间/寄存器（如都 807us、VGPR=256/SGPR=112，或 WL 恒 ~843us），八成是 stale 命中。rocprofv3 kernel-trace CSV 列 `VGPR_Count` / `Accum_VGPR_Count` / `SGPR_Count` / `Scratch_Size` 是判内核到底变没变的可靠 ground truth。

**被此坑污染的历史假阴性（清缓存重测）**
- MXFP8 "scale 双缓冲预取无收益"、"scale-128 perf-neutral" 极可能是 stale-cache 污染的假 negative，需 `rm -rf` 清缓存重测。grouped wgrad 已翻案：清缓存后根因完全不同。

**解法（优先级从高到低）**
1. 把可编辑 kernel 逻辑放进 `@kernel` 函数体 / 嵌套 def / 嵌套 class —— 首选，零踩坑。
2. 测试加 `FLYDSL_EXTRA_SOURCE_DIRS` hash 整目录。
3. `FLYDSL_RUNTIME_ENABLE_CACHE=0`（`0/false` 只禁**磁盘**缓存，内存缓存仍在）——迭代 lowering 时旁路磁盘。
4. `rm -rf ~/.flydsl /tmp/flydsl*`（最后手段）；用 lru_cache 时还要 `compile_my_kernel.cache_clear()`。
- 铁律：任何 "改动没效果" 的结论，**先无条件 `rm -rf ~/.flydsl /tmp/flydsl*` + 清 lru_cache 重跑**，再谈结果。

## FlyDSL tracer 字面 if/for 坑：编译期常量分叉必须在外层 Python 作用域选好

### 字面 if 被 rewrite 成 scf.if，即使条件是编译期常量
- （`@flyc.kernel`=设备内核体、`@flyc.jit`=host launch wrapper，定义见 methodology/09）
- `@flyc.kernel`/`@flyc.jit` 装饰的函数体（含内部嵌套 `def` 如 `_do_tile`）源码里字面出现的 `if`/`for` 会被 AST-rewriter 转成设备侧控制流（MLIR scf dialect 的 `scf.if`/`scf.for`，即在 GPU 上运行的分支/循环，而非编译期 Python 分叉），**哪怕分支条件是纯 Python 编译期常量**（如 `trans_b`）。想按编译期常量分叉行为，必须在外层未被追踪的 Python 作用域先选好分支（生成不同闭包 / 调用不同模块级函数），内核体内只做函数调用，**不能写字面 `if trans_b`**。
- 裸 python `if _flag:` 会让**两个分支都被 trace**（例如两个 SharedAllocator → `Only one SharedAllocator` 报错）。编译期分支必须写 `if const_expr(_flag):` 才走单一分支。
- ❌ 别再试：把 NN/NT 两个 4-wave 编译函数硬合并、在 `_do_tile` 里写 `if trans_b`。先 NameError（变量只在某个 scf.if 分支可见）；改闭包写法后，重复调用同一已编译 `@flyc.jit` 函数时又触发 FlyDSL 全局漂移检测报错（`_WL_ASM_CACHE_3BUF` 在 trace 期间被 mutate，和首次编译快照对不上）。**最终定案：拆成两个独立编译函数**，只共享跟 `trans_b` 无关的辅助函数（如 `_grouped_4wave_tile_scan`）。

### const_expr 只包编译期已知值，绝不包运行时 GPU 值
- ❌ 别再试：`if const_expr(lane==c_zero)` / `const_expr(gpu.thread_id('x'))` / `const_expr(warp_id...)`。即使 `@flyc.kernel(known_block_size=...)`，`gpu.thread_id('x')`、`lane`、`warp_id` 都是**运行时 SSA 值**（编译器只知取值范围，不知当前 lane/thread 实例）。正确写法是运行时 `if lane == c_zero`（AST rewriter 下沉为 scf.IfOp）或 `.select()`。
- `const_expr` 只用于 trace/编译期已知值：Python bool、constexpr 参数、循环展开 / 类型 / layout 分支（如 `if const_expr(trans_v)`）。

### 字面 for + Python int 边界 = 静默展开丢 init=
- FlyDSL loop bounds **必须是 `fx.Index(...)`** 才能 lower 成真正的 scf.for 并带 loop-carried operands（phi 节点）。普通 Python int 会让 trace 把它当 Python range **UNROLL 展开，并静默丢掉 `init=`** —— 任何 prefetch/double-buffer 对 MLIR 都变不可见，**无报错**。这是流水线被静默禁用的头号原因。（run-time loop 边界必须用 `fx.Index(...)` 不能用 Python int 这一条同样适用于前端编写场景。）
- `@flyc.kernel` 内 `range()` 被 AST rewriter 转成 MLIR 运行时循环，循环变量 `i` 变成 ArithValue，**无法索引 Python list**。要编译期展开必须用 `range_constexpr(N)`，此时 `i` 是 Python int。
- 真正的 loop-carried 预取用运行时 `range(start,stop,step,init=...)` 制造 SSA phi。

### Python `data = next_data` 不是 phi
- FlyDSL 里 Python 级 `data = next_data` 重绑定**不产生 loop phi**：两个名字别名同一 SSA 值，load 被当循环不变量外提（hoisted）。同理 Python 级 `for _pi in range(N)` 被 trace 成 N 份平铺副本再由 LLVM 重卷，交换对 MLIR 不可见。
- 预取只在预取值**穿过 `init=` 和 `yield`** 时才生效。**Python swap 不是 phi。**
- ❌ 局限：`iter_args` 可同时 carry **多个**简单 SSA 值（标量 / vector / i32 / i64 / index），不支持的是把 ping-pong buffer 对象 / 32-vec 的 list 作为**单个** iter_arg carry（见 methodology/09 "loop-carried state 携带规则"，其 PA decode 范例即携带 15 个 loop-carried 值）。

### 前端 if/for 语义限制（纯 Python 合法但和 MLIR 构造冲突）
- 分支局部定义：值只在某个 `if`/`else` arm 内定义、arm 外使用 → **静默破坏 MLIR result typing**。必须把定义提到分支上方，或 yield 单一 merged 值。
- 前端会把 if 分支体抽成 `__then_*`/`__else_*` 函数；分支内引用的名字若定义在外层可能触发 **NameError**。标量/向量类型不匹配、Python int 出现在需 DSL 值处也是常见 wizard 报错。
- ★ **同一机制的 `scf.for` 版（2026-07-29 实测）**：嵌套 `def` 里若含**运行期** `for _c in range(<runtime>)`，
  前端抽取循环体时只捕获**循环自身引用到的**外层名字；**只在循环之后**才用到的外层名字会掉出闭包，
  变成该 `def` 的局部名 → 运行到那行报 `UnboundLocalError: cannot access local variable 'X'`
  （注意报的是 *local*，不是 free variable，很容易误以为自己写了赋值）。
  - 踩证：把 mxfp8 wgrad 的 chunk 主环包进 `_run(quads)` 闭包，`store_c`（只在环后用）当场炸；
    同一闭包里 `k_iters` / `acc00` / `mfma` / `a_g2s`（环内也用）全部正常捕获。
  - 解法：把**环后代码留在父作用域**（或显式传参 + 返回值，同本卡「不要 mutate 捕获变量」）。
    只把「含运行期 for 的循环本体」放进嵌套 def，epilogue/store 别跟进去。
- 不要在嵌套 helper 里 mutate 捕获的外部变量（只读闭包 OK，写要走显式参数 + 返回值）。
- 避免 early return：不要把 `return`/`yield` 放进 if/else 分支（前端靠单一显式出口确定结果类型）。
- 有副作用 / 循环携带值 / 分支局部定义的**运行时分支**，应拆成 local helper 并把 `if` 包进 local `@flyc.jit` dispatch 函数里调用，不能直接写在 kernel 体内。
- 深度内核调试（all-1s 测试、单 partition 隔离、MFMA 操作数 layout 检查）用 debug-flydsl-kernel skill。

### 循环变量名残留被 JIT 折叠成常量
- `for s in range_constexpr(4)` 结束后 Python 的 `s=3` 仍留在作用域，紧接着写 `s=(grow>>4)&3` **可能被折叠成常量 3**（变量名冲突导致 trace 折叠）。preshuffle 索引里的子块变量**必须改名为 `sub`** 而非复用 `s`。

### dynamic scf.for loop-carry LDS 触发 lowering bug + 单 iter_arg 复合值限制
- 两条链式 dynamic `scf.for` loop-carry LDS shared-ptr 会触发 lowering bug：`unrealized_conversion_cast fly.ptr→llvm.ptr<3>` remained live（只能 balanced/无 tail 才跑）。❌ 别再试（如遇需按当前 flydsl 版本复验）。
- 动态 `scf.for` 每个 iter_arg 只能是**单个 MLIR 值**（可有多个 iter_arg），carry 不了 32 个 vec 的 list 塞进一个 iter_arg（同上 iter_args 复合值限制）。
- module-level fn 里的 `if`（如 `if wave_m==1`）不被 AST 改写，会 `bool()` dynamic 报错；**conditional-barrier 必须 inline 在 kernel body**。

## FlyDSL 前端编写坑：buffer offset 单位/SmemPtr view cache/absf 缺失/DLTensorAdaptor

**buffer_load/buffer_store offset 单位是 ELEMENTS 不是 bytes**
- `buffer_ops.buffer_load(rsrc, offset, vec_width, dtype)` 的 offset 单位是 dtype 个元素，内部会转字节。写成字节偏移会越界读错数据。
- 症状是垃圾数据不是 crash（off-by-sizeof(elem) 地址 bug）。例：FP8 数据按字节寻址、但用 `dtype=T.i32` 加载时要把字节地址除以 4：`buffer_load(k_rsrc, k_addr_bytes//4, vec_width=4, dtype=T.i32)`。

**进 epilogue 前必须清 `SmemPtr._view_cache = None`**
- `SmemPtr.get()` 会缓存它创建的 view 到 `SmemPtr._view_cache`。若在运行时循环体内调用，缓存的 view 定义在循环 scope；epilogue（循环外/退出 scf.for 后）复用它会触发 MLIR SSA dominance 报错。
- 解法：循环后、epilogue 前手动 `my_smem_ptr._view_cache = None`。已验证坑。
- 配套规矩：raw memref 在 block 顶部一次性取好，让它 dominate 所有子 scf.for/scf.if region。if 分支变量不外泄的完整规则见本卡「FlyDSL tracer 字面 if/for 坑」小节。

**run-time loop 边界必须用 `fx.Index(...)` 不能用 Python int**
- 细节/后果见本卡「FlyDSL tracer 字面 if/for 坑」小节。

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

**跨 lane 原语可以直接调，但别用 `llvm.TruncOp` 收 i64**
- `rocdl.update_dpp / ballot / readlane / readfirstlane`、`llvm.intr_ctpop` 在 flydsl 里都能直接 emit（wave64 的 `ballot` 必须 `res=i64`；`readlane` 的 lane 参数允许传 Python int）。
- ⚠️ 该 build 的 `llvm.TruncOp.__init__(self, res, arg, overflowFlags, *, loc, ip)` 是**三个位置参数**，按 `(res, arg)` 两参调用直接 `TypeError: missing 1 required positional argument`。**用 `arith.trunci(target_type, value)`**（`flydsl.expr.arith`，签名 `(out, in_, *, overflow_flags=None)`，稳定）把 `ctpop` 的 i64 收成 i32。
- DPP 前缀扫描要求**满 EXEC**（放 kernel 入口、任何分叉之前），`bound_ctrl=True` 让移入的 lane 贡献 0，就不用再修 bank_mask。

## FlyDSL ThrVal/atom layout 静默错：#1 静默产错结果 bug 源

- **#1 静默产错结果源**：错误的 ThrVal/ThrBit layout 是 FlyDSL 头号静默错 bug。编译器接受、内核正常运行、输出垃圾，**无任何运行时诊断**。定位极难，因为没有报错。
- **顶层 shape 必须 rank-2 `((thr...),(val...))`**：mode0=线程轴、mode1=值轴。`TiledOpUtils.h` 无条件做 `shape.at(0)`/`shape.at(1)` 切 thr/val。多套一层嵌套或压成 rank-1 会编译通过但静默产出错误 tile 划分。

- **MmaOp ThrVal 乘积不变式**（违反仍编译，结果 UB）：
  - `|thr|` == 协作线程数：AMD wave64 MFMA = **64**，wave32 WMMA = **32**，NVIDIA WGMMA = **128**。
  - A: `|thr|*|val| == M*K`；B: `|thr|*|val| == N*K`；C/D: `|thr|*|val| == M*N`。
  - 各 ThrValLayout 的 `|thr|` 要和 `getThrLayout` 一致；`|val|` 要和 `emitAtomCallSSA` 里线程寄存器向量宽度一致。
  - ⚠️ **线程数不匹配最阴险**：wave64 MFMA 误注册成 FxC(32)，会照常发出指令，但一半线程在旧（陈旧）寄存器上算 → 结果错但无报错。

- **参考坐标系是列主序（非行主序）**：写 ThrValLayout 时必须按列主序算 stride：
  - MmaOp A=(M,K) 基线 stride `(1,M)`；B=(N,K) 基线 `(1,N)`；C/D=(M,N) 基线 `(1,M)`。
  - CopyOp src/dst=(M,N) 基线 `(1,M)`。

- **CopyOp：bit 粒度 vs value 粒度 recast**：
  - CopyOp 发布的是 **bit 粒度** layout（`getThrBitLayout*`，shape 恒为 `(|thr|, bitSize)`）。
  - `CopyAtomType` wrapper 经 `layoutRecast(bitLayout, oldBits=1, newBits=valBits)` 算出 value-layout。
  - 例：32b buffer copy 写 `FxShape(FxC(1), FxC(32))`，valBits=32 recast 成 1 个 f32/线程；**同一 Op 复用**于 16b f16 pair 时自动得 2 个 f16/线程。
  - ❌ 别再试：让 bitSize 不能整除 valBits（如 96b/64b = 1.5）。`layoutRecast` 会**静默产垃圾**，无报错，程序员必须自己发现。

- **verify 必须拒绝非法元组**：新 MmaOp/CopyOp 的 verify（`genVerifyDecl=1`）必须拒绝不支持的 `(m,n,k,elemTy)` 元组并给清晰 `emitError`。否则非法配置会静默命中 `emitAtomCallSSA` 的 `return failure()`，且无诊断信息 → 又一条静默错路径。

---
来源: flydsl-sync/SKILL.md, pr-merge-gate/SKILL.md, 03-emit-knobs.md, debug-flydsl-kernel/SKILL.md, FlyDSL/CLAUDE.md, programming-model.md, mxfp8-8wave-devloop/SKILL.md, mxfp8-grouped-gg-devloop/SKILL.md, feedback_flydsl_cache_staleness.md, project_mxfp8_grouped_wgrad_wl.md, remote-sync/SKILL.md, 08-deadends.md, prefetch-data-load/SKILL.md, flydsl-kernel-authoring/SKILL.md, flydsl-tile-programming/SKILL.md, agpr_phase5_lds.md, add-target-atom-op/SKILL.md
