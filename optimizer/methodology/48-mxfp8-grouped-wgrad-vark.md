# grouped MXFP8 var-K wgrad:瓶颈定位 / scale-prefetch 真因 / 软流水 / AST-rewrite 绕坑 / occ=2 ROI

> 类别: 方法论 · 主题标签: mxfp8, grouped, wgrad, variable-K, scale-prefetch, chunk-SSA, AST-rewrite, occ2

## bwd 追 dense 的主战场 = var-K wgrad
- 逐组件对标(total_M=32768,dense→grouped):**dgrad(grad_a)已追平** dense 1.05-1.18×(同一 NT kernel);**wgrad(grad_b)最大结构性缺口 1.16-1.35×**(绝对 +200~270us),根因 dense wgrad 是普通 NT gemm,grouped wgrad 是 **variable-K kernel**(按组收缩 M);grad_out 量化 grouped 慢 1.2-1.6× 但绝对占比 <10% ROI 低。

## PMC 根因(非访存、非占用)
- var-K vs dense NT wgrad(4096×7168)`rocprofv3 --pmc`:两者 **MemStall≈0**、**occ 相同**(~20-23%,LDS=128KB→1WG/CU occ=1 结构上限)、VGPR/LDS 几乎相同;缺口纯在 **MFMA 利用率 dense 61% vs var-K 45.5%(-16pp)**。var-K grid 恰好 8×(每 group 一套 blocks 448×8),per-group prologue/epilogue + 4-buf 软流水在组边界重启占比大 → MMA 间空隙(组边界依赖/barrier bubble,不是等访存)。

## 提速真因 = scale 载入时机(不是 memref 累加器)
- baseline body 顶部**当拍**载入 sa0/sa1/sb → scale ds_read 延迟压在 MMA 关键路径;dense NT **预取下一拍** scale(sa0n)用上拍已到的 scale 做 MMA。
- 只加 scale 软流水吃到全部 ~9%(单独 SSA 累加=0 提速,单独 scale-prefetch ≈ 全部收益)。
- 落地 = **chunk-local 结构**(通用零判断):`for _c in range(k_iters//chunk)` 调 module-level `_wgrad_ssa_chunk`(chunk 内累加器进 SSA、scale 逐拍预取,只在 chunk 边界读写一次 memref),尾部 `range_constexpr(chunk)+k_abs<k_iters` guard 收余数。成绩 930/1047/530/307 → 851/936/478/274,方阵追平 grad_a NT 兄弟核。

## 4-buffer 软流水移植(优化1,前置)
- MX wgrad 单缓冲串行 → chunked 4-buffer distance-2 软流水(移植 tensorwise `_wgrad_body_4buf`)。LDS 2→4 缓冲(cur0/1+next0/1,A/B 各一套);外层运行时 `scf.for over ceildiv(k_iters,chunk)`,内层 `range_constexpr(chunk=8)` 展开 distance-2,trace 期 swap cur↔next;**偶数 chunk 在边界重置 ping-pong 身份**(scf.for 不能 loop-carry Python 变量);accum 用 rmem in-place(`_wgrad_mx_accum`)。结果:M=2048 全反超快 4-26%。

## 绕开 AST-rewrite state-variable 报错(动态 scf.for)
- 主循环体不能有 `obj.method()` 调用、不能有 list 状态变量。做法:body 是**单个 module-level fn 调用** + 只对 buffer 引用做 tuple-unpack 重赋值,动态 `scf.for` 只 loop-carry buffer 引用。

## 用户硬约束:禁 balance 判断
- 严禁任何地方判断是否 balance(host per-call 判 uniform、once-cache、核内 per-WG 分派全禁);不能特化 uniform-K,只能让**单一变长-K kernel** 对所有分布都用上 scale 预取。host wrapper 里彻底没有 D2H/offs 读取,eager/capture 同一条 kernel 无 sync 税。

## 生产内核 vs WL 对象
- 生产 = `_build_grouped_mxfp8_wgrad_kernel`(8-wave / occ=2 / packed-preshuffle scale / chunk-SSA 软流水)。
- 对象 WL = `_build_grouped_mxfp8_wgrad_wl_kernel`(occ=1 / 4-wave 单块 bare-asm hw-loop,NT 布局 `[OUT,m_total]`,per-1x32 E8M0 折进 `v_mfma_scale_f32_16x16x128_f8f6f4`)。
- 基准(chi2811 gfx950 B=8 M=4096 N×K=4096×4096 清缓存):baseline 479us / WL-scaled 811-813us(1.7× 慢)/ WL-unscaled 445us(0.93× 反超);SNR 28.1 全对。**WL scale 折进 MMA 是 1.7× 慢的真凶**。
- WL scale-prefetch 修法(收益仅 ~4%,845→811):删 phase-top emit_scale + 阻塞 vmcnt(0),改成在 emit_inplace 里每个 scale VGPR 最后消费 MFMA 之后重载下一 phase scale(藏进 MFMA shadow),prologue 只留一次 emit_scale。瓶颈是吞吐非延迟。真正修复(未做)= packed/preshuffle scale(4 个 E8M0 打进 1 i32 op_sel 选),预期边际 ~5%。

## 下一刀 ROI
- 生产 var-K wgrad 上 **occ=2**(LDS 128KB→≤80KB:**削流水缓冲 4→2 而非缩 tile**,让第二 wave 填 barrier/依赖气泡,唯一能真正抬 MfmaUtil 45→61% 的 lever)。不是 WL、不是 cross-group persistent(均已证伪)。风险:fwd/dgrad occ=2 曾证伪(bm=128 坏),wgrad 削缓冲是新尝试。dense NT 自己也只 61%@occ=1,现实目标是逼近 61% 吃掉 16pp。

---
来源: mxfp8-grouped-gg-devloop/SKILL.md(优化1/优化6/var-K PMC/真因&已修复); project_mxfp8_grouped_wgrad_wl.md
