# Primus-Turbo 生产 timed autotune:四轴 never-regress、L2/deep-wl/变体/split-K

> 类别: 方法论 · 主题标签: autotune, dispatch, L2-swizzle, split-K

**核心机制:timed autotune 替 env**
- 生产用 timed autotune 替代 env 配置:首调每个 `(M,N,K)` 定时扫候选,取 global-min,per-shape 缓存。
- 四轴,且后三轴 **never-regress**:扫里恒含 baseline + 取全局 min + margin 门槛 → 只会追平或更快。
  1. **L2 swizzle**:`(group_m, group_n, num_xcds)`
  2. **deep-wl**:phase-barrier `(vmcnt, lgkmcnt)`
  3. **变体轴**:COOP scale-load + TACCW wide-store
  4. **split-K**
- **CUDA-graph capture 内无法定时** → 回退静态启发式。

**四轴规律与门槛**

| 轴 | 性质 / 触发条件 | 具体值 / WHY |
|---|---|---|
| L2 swizzle | 纯 WG→tile 双射,bit-identical,只调 L2 residency | 小-M 要大 `group_n` (16/32);`num_xcds` **NX8 普遍最优** |
| deep-wl | 只在 **K≥8192** 报 | `(16,15)` 深流水藏高-trip-K g2s 延迟;`_WL_MARGIN=1.02`(须超噪声带 2% 才选) |
| split-K | 只对 few-tile 大-K(一 WG/tile 撑不满 CU) | 给 `2/3/4/6/8/12/16` |
| 变体 | COOP scale-load + TACCW wide-store | never-regress,恒含 baseline |

**FlyDSL authoring 坑(实现 autotune 分支时)**
- `@flyc.kernel` 体内**不能写** `if mode==...` 分支:AST 变换转成 traced conditional,分支内赋值不外泄 → `NameError`。须放普通 Python helper,在 trace 时求值。
- 单 dword store 用 scalar `buffer_store`(packed,`rout`,`gid`,`mask=` 传裸标量)。包成 1-wide `Vec.from_elements` 会触发 LLVM `'Do not know how to scalarize'` 崩。

---
来源: 13-primus-turbo-prod.md
