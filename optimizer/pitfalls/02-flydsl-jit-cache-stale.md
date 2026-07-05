# FlyDSL JIT 缓存不失效假象：key 不 hash 模块级方法/emit 旋钮/env-gate

> 类别: 踩过的坑 · 主题标签: jit-cache, cache-invalidation, emit-knobs, false-negative

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
- **in-memory ASM 缓存 `_MX_WL_ASM_CACHE`**：env 驱动改 asm 发射时，须把 env flag **全部加进 key**，否则内存层也复用旧 asm。

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

---
来源: flydsl-sync/SKILL.md, pr-merge-gate/SKILL.md, 03-emit-knobs.md, debug-flydsl-kernel/SKILL.md, FlyDSL/CLAUDE.md, programming-model.md, mxfp8-8wave-devloop/SKILL.md, mxfp8-grouped-gg-devloop/SKILL.md, feedback_flydsl_cache_staleness.md, project_mxfp8_grouped_wgrad_wl.md
