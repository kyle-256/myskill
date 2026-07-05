# FlyDSL JIT 缓存失效:改算错验金标准、清缓存时机、lru_cache.cache_clear

> 类别: 方法论 · 主题标签: jit-cache, correctness-gate, isa-dump, cache-invalidation

## 金标准:验缓存是否真失效
- 故意把 kernel 改算错(如 scale×2),**不清缓存**直接跑:
  - 出 nan / 结果错 = 缓存正常失效,新代码生效。
  - 仍跑出**旧的正确 SNR** = 缓存陈旧,你验的是旧内核(踩坑)。

## disk cache 何时自动失效、何时必须手动清
- FlyDSL JIT disk cache 在 `~/.flydsl/cache`,`FLYDSL_RUNTIME_ENABLE_CACHE` 默认 true。
- **kernel 源码或闭包值变化 → 自动失效**;in-memory cache 始终生效。
- 只有以下两种情况需要 `FLYDSL_RUNTIME_ENABLE_CACHE=0` 或 `rm -rf ~/.flydsl/cache`:
  1. 改了 C++ passes;
  2. 改了**非闭包** helper 函数(改动不进 hash,disk cache 不失效)。

## 单进程多 env-variant 对比:必须 cache_clear
- `_compile_dense_tn` 是 `@functools.lru_cache(128)`。
- 多 env-variant 单进程对比时,若不清 lru_cache,函数内读的 env 被**冻在首次调用值**,后续 variant 全用第一次的 env。
- 正确姿势:每换 env 前 `G._compile_dense_tn.cache_clear()` 再 compile。

## dump ISA 前的完整清缓存 + 触发流程
- 扫寄存器/spill 前 dump ISA,设:
  - `FLYDSL_DUMP_IR=1`
  - `FLYDSL_DUMP_DIR=/tmp/dscan`
  - `FLYDSL_RUNTIME_ENABLE_CACHE=0`
- **必须清 comgr 缓存** `~/.cache/comgr`,否则 comgr 缓存 codegen,不重编不 dump。
- **必须真正 RUN kernel**(`c(*args)`)才触发编译+dump;只 `_compile` 不够。
- 产物落在 `/tmp/dscan/kernel_dense_tn_0/21_final_isa.s`。

---
来源: flydsl-sync/SKILL.md, flydsl-fp8-gemm-tuning/SKILL.md, flydsl-kernel-authoring/SKILL.md, project_mxfp4_epilogue_store.md
