# Autotune 设计五原则:balanced 计时、M-branch、hysteresis 1.5%、纯静态 cache key、never-regress

> 类别: 方法论 · 主题标签: autotune, dispatch, backend-selection, never-regress

## 五原则
1. **balanced 计时**:M_total 均分到 G 组(`_balanced_targs` / balanced group_offs)计时,不被首次真实调用的 skew 倾斜分布带偏。
2. **M-branch**:不同 M size 用**完全不同**的 candidate set。NN small-M gate(`bm128_tiles <= num_cus`)时 bm128 永远赢,直接 return 单 config,不进 autotune。
3. **hysteresis 1.5%**:只在候选比当前快 ≥1.5% 才切,防噪声 mis-pick。
4. **cache key 纯静态维度**:`(op, N, K, G, M_total, out_fp16, cbsz, blgp)`,**永不含 tensor id/version**。
5. **warmup 要长**:NT fwd 用 median-of-5 × 50 iter warmup;短-K 冷测 mis-pick 严重。

## correctness reference + 切换门槛
- candidate 第一个 `cands[0]` = correctness reference;后面只有 ≥1.5% 更快才切换。

## never-regress(追平或更快,永不回退)
- **twin 交错取 global-min**(通用模式,细节因轴而异):候选 config 作为多路 twin **同进程交错计时**取全局 min(交错验证 twin 正确性等价且稳定才敢用,避免噪声误选)。
  - **变体轴(COOP/TACCW)**:四路 twin `{df,T,C,CT}`,`margin` 由 0.99 演进到 **0.995**,`VREPS=16`(2026-07-01)。
  - **split-K**:缓存 key = `(M,N,K)`(未标注 margin 数值,不等同于变体轴的 0.995);`config tuple` 的 8-tuple 形式 `(gm,gn,xcd,persist,wlv,elgk,taccw,coop)` 也只出现在变体轴段。
- **Primus-Turbo 生产 timed autotune 替代 env**:首调每个 (M,N,K) 定时扫候选取 global-min per-shape 缓存。四轴,后三轴 never-regress(扫里恒含 baseline + 取全局 min + margin 门槛):
  - L2 swizzle `(group_m, group_n, num_xcds)`
  - deep-wl(phase-barrier `vmcnt`, `lgkmcnt`)
  - 变体轴(COOP scale-load + TACCW wide-store)
  - split-K
- **split-K 候选加入条件**:`tiles < ncu/2` 且 `K ≥ 2048` 且 `s` 整除 `K//256`;实测端到端(含 reduce)取 min → 永不回退。
- CUDA-graph capture 内无法定时,回退静态启发式。

## mode-split launch
- graph capture 用 raw `@flyc.jit`(graph-friendly);eager 用 `flyc.compile` 一次性编译的 compiled 对象(跳过 per-call drift-check overhead)。
- 小 shape eager per-call **18.4 → 17.2us(~7%)**。

## Backend 选择优先级(高→低)
代码 setter → 环境变量 → autotune → 代码内 default → fallback(试所有 `can_handle` 的)。
- env 支持按精度:`"fp8:CK,other:TRITON"`。
- env:`PRIMUS_TURBO_GEMM_BACKEND`、`PRIMUS_TURBO_GROUPED_GEMM_BACKEND`、`PRIMUS_TURBO_ATTENTION_BACKEND`、`PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND`、`PRIMUS_TURBO_AUTO_TUNE`。

## 强制/切 backend API
- `GlobalBackendManager.set_gemm_backend(BackendType.CK)` 强制。
- `GlobalBackendManager.set_auto_tune(True)`(或 `PRIMUS_TURBO_AUTO_TUNE=1`)。
- `GlobalBackendManager.reset()` 清设置 + autotune 缓存。测试里强制 backend 后 teardown 必须调 `reset()`。

## BackendType → 用途映射
| Backend | 用途 |
|---|---|
| HIPBLASLT | GEMM bf16/fp8 tensorwise;**dense GEMM 默认** |
| TRITON | GEMM/GroupedGEMM/Attention;可调无需 rebuild |
| CK | GEMM/GroupedGEMM FP8 row/block |
| TURBO | MXFP8/MXFP4 GEMM、Attention;自研 gfx950 |
| AITER | Attention 默认 |
| DEEP_EP | MoE dispatch/combine |

---
来源: 06-autotune-design.md, 02-nt-fwd-kernel.md, 13-primus-turbo-prod.md, project_mxfp4_epilogue_store.md, SKILL.md
