# FlyDSL 复用范式:抄既有变体、共享坐标/寻址只分叉 loop body、融合单发射省 host dispatch

> 类别: 方法论 · 主题标签: authoring-pattern, code-style, host-dispatch, fusion

## 抄既有变体(最大化复用原语、禁魔术数)
- 新 kernel 变体逐字段照抄既有模式:4-wave↔8-wave、NN↔NT/TN、tensorwise↔mxfp4。命名/错误处理/dispatch 候选 grid/group_n band 都跟着抄。
- 必须复用既有原语:`mask_a_tail` / `emit_wholeloop_tile` / `grouped_block_mn` / `S2RLoader` / `G2SLoader`。
- 禁魔术数:tile/BLOCK 从 shape 推导,不写死。

## turbo 代码风格约定
- 注释一律英文,禁中文/日文。
- 函数命名跟 turbo 现有风格:`_grouped_<noun>`(如 `_grouped_block_mn`)、`_wgrad_<verb_or_noun>_<variant>`(如 `_wgrad_wholeloop_asm_3buf`)。别自造缩写:`_wl3buf_fused_tail_split` 错,应写 `_wholeloop_tail_split_3buf`。
- 能复用 `gemm_helper.py` 就必须复用:`ceildiv`、`_readfirstlane_i32`、`xcd_remap_pid`、`make_fp8_buffer_tensor_rebased`、`S2RLoader`/`S2RLoaderTr`、`_robust_time` 都在里面。先 grep 确认没有再写。

## 共享坐标/寻址、只分叉 loop body
- masked 和 persist 两个 wgrad kernel 共用:
  - `_wgrad_block_mn`:dispatch → `(group_idx, block_m, block_n)`
  - `_wgrad_rebase`:i64 SRD rebase → `(a_div, b_div)`
- 唯一分叉的是 K-loop body:masked = chunked 4-buffer;persist = `scf.for` 2-stage prefetch。
- 共用坐标/寻址逻辑、只把 loop body 分叉,是 FlyDSL 的标准复用范式。

## 融合单发射(省 host dispatch)
- 模式:GEMM/preshuffle 工厂返回裸 `@flyc.kernel`,由一个 `@flyc.jit` stub 在同一 stream 上依次 `.launch()` 发射 preA→preB→GEMM → 一次 Python dispatch(stub 内多 kernel enqueue 是廉价 C++ 调用)。
- WHY:eager 下每次 `@flyc.jit` dispatch ~38µs,compiled 直调 ~6µs;小 shape 被 host 开销主导 → 融合是 eager 小 shape 的关键。
- preshuffle v2 优化:一 thread 出全部 NG=4 个 output dword,共享的 4 个源 int32 只读一次(v1 是一 thread 一 output dword,4 个 thread 各重读同样 4 行 → 4× cross-thread 读放大)。grid 缩到 1/NG,输入 HBM 读流量 ~4×↓,device 时长减半(~16→8µs),bit-exact。

## 非持久 vs 持久
- fwd 用非持久(non-persistent, one-tile-per-WG)比持久 `scf.for` 快约 11%(省 outer tile-loop `scf.for` 调度惩罚)。
- 前提:L2 swizzle 也移植进非持久 kernel。

## helper 放置规则(先搜索复用,勿散落勿重复)
- 共享 kernel helper → `kernels/kernels_common.py`:`get_warp_size`/`dtype_to_elem_type`/`validate_moe_dtypes`/`_if_then`(SCF context manager)。
- 领域专用 → `moe_common.py`/`layout_utils.py`/`pipeline_utils.py`/`fp8_gemm_utils.py`/`dpp_utils.py`/`mfma_epilogues.py`/`mfma_preshuffle_pipeline.py`。
- DSL 级 numeric/arith/type → `expr/utils/arith.py` 或 `expr/numeric.py`。
- 编译/运行时级 → `utils/`。

---
来源: pr-merge-gate/SKILL.md, remote-sync/SKILL.md, 04-tn-wgrad-kernel.md, 14-fused-preshuffle-e2e.md, 02-nt-fwd-kernel.md, FlyDSL/CLAUDE.md
