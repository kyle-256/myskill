# 踩过的坑：其它零散事实

> 类别: 踩过的坑 · 主题标签: autotune-dispatch, barrier-divergence, register-pressure, correctness

**寄存器压力 / 溢出地雷**
- MI300 blockwise 溢出组合：`waves_per_eu=2 + kpack=1 + 大 BM/BN(256/128)` → 只跑 100-200 TFLOPS，从 grid 里排掉。fwd/bwd 都有此雷。
- 每个 `buffer_load_dwordx4` 使 arch-VGPR +~4；一个 `buffer_load_lds`(async) 反而回收 ~32(A-tile 本来占的)。

**MI300 blockwise autotune priors / grid 剪枝**
- fwd 几乎所有赢家：`num_warps=8 / num_stages=1 / mfma=16`。
- `CHUNK_SIZE=64` 最常赢；默认 32 仅 1/4 情况赢；`chunk=16` 几乎从不赢，可删。
- 两个 BM/BN 家族：小 (B,M) 用 128×256；大 (B,M) 或 N≥K 非方形用 256×128。
- refine grid 剪枝：锁 `mfma=16`、删 `chunk=16`、删 `num_stages=2` 的不可能组合。WHY：否则 1440 configs 跑 25 分钟。

**dispatch / container**
- `K%128≠0`(如 GPT-OSS K=2880)触发 CK BLOCKWISE kernel 的 `k % 128 == 0` assertion。解法：收紧 `GroupedGEMMFP8CKBackend.can_handle`(和 VariableK 版)让 dispatcher 落到 Triton 的 masked-tail 路径。

**Preshuffle-B / kpack / skip-LDS**
- Preshuffle-B 的 kpack 与 dtype 耦合：FP8/INT8 用 `kpack = 64/elem_bytes`，BF16/FP16 用 `kpack = 4`。❌ kpack 错值不会报错——loads 不再对齐 MFMA atom，仍 type-check、仍能跑，但静默产出 garbage。同时核对 MFMA 操作数顺序：`gemm(LHS,RHS,acc)`，LHS→M、RHS→N。
- Preshuffle 且 skip-LDS 的 B 操作数会同时省掉 B 的 barrier；但块内 B 的**第二个消费者**会重新引入 LDS copy 需求。Preshuffle 不适用于：B 每次 launch 变化(activations/动态权重，offline 成本不摊销)、或 B 真正靠跨多 fragment 的 LDS 复用获益。

**async copy 粒度按 arch 不同**
- global→LDS async copy(`buffer_load_lds`)：gfx942 每 op 搬 4 B，gfx950 每 op 搬 16 B。gfx942 上小 tile 的 per-issue 成本可能超过收益 → 只对大 `tile_m (>=128)` 用 async copy；gfx950 更普遍地划算。

**LDS swizzle 正确性**
- swizzle 必须 write 与 read 路径一致：一边 swizzle 另一边不 swizzle → 从错误位置读出数据(静默错误)。改 swizzle 后必须跑正确性测试。

**barrier / GPU hang**
- ❌ 别再试：`gpu.barrier()`/`s_barrier` 放在 divergent 控制流(运行时 if 让部分线程走不同分支)下 → workgroup 内非全部线程到达 → GPU 死锁。FlyDSL 不支持 divergent barrier。MFMA lane 屏蔽要用 tile selection，绝不用 EXEC(MFMA 忽略 EXEC/MODE——强制 RNE、保留 denorms、无 FP 异常)。
- GPU hang 也可能来自错误循环边界：`stop<start` 的无符号比较问题，或 `step=0`。

**死路（❌ 别再试）**
- ❌ wgrad B-inline transpose-load (TN) → SNR nan。WHY：TN 有 lgkm race，inline-asm 的 `=v` 输出约束不稳。
- ❌ mxfp4 emit 失败旋钮：
  - `FP4_EVENSPREAD=1` -207T(ds_read 均匀分散破坏 INPLACE last-use overlap)
  - `FP4_FINELGK=1` -50T(per-mfma drain 序列化)
  - `FP4_MFMANOP`(s_nop pacing) 无帮(LDS-bw bound 非 latency)
  - `FP4_CHUNKBAR` 细粒度 barrier -130T(单块 BK256 内加纯增成本)
  - `FP4_ACCD16` racy 且慢(cross-bank acc 杀 A-operand 复用)
  - `FP4_PREFETCH` racy(nxt_buf g2s 未落地)
  - `FP4_WLDSR` 对 INPLACE 路径无效(只在 else 分支)

**code-style / 模块边界**
- `expr/` 直接子模块(typing/primitive/gpu/derived/struct/arith/math/vector/numeric/meta/extern/utils)必须 backend-agnostic，不得 import ROCDL/HIP 绑定；`import flydsl.expr` 须在无 FlyROCDL 绑定时成功(`tests/unit/test_expr_optional_rocdl.py` CI 强制)。目标专用(ROCDL/HIP、MFMA/WMMA、buffer/TDM/cluster)代码进 `expr/rocdl/` 包(cdna4/cluster/inline_asm/tdm_ops/universal)，不放新 top-level `expr/*.py`。buffer_ops/rocdl/tdm_ops 从 `expr/__init__.py` 经 `__getattr__(_LAZY_MODULES)` 懒加载。

**bench 命令纪律**
- ❌ 别 pipe bench/test 命令过 `| tail / | head / | grep / 任何 filter`——piping 强制 stdout 全缓冲，隐藏中间输出、让运行中的命令看起来 hang。先 `> out.txt 2>&1` 重定向，完成后再读/搜。backgrounded(超时)命令不算完成：只有进程退出 **且** 预期产物落盘才算一轮完成。

---
来源: mi300-blockwise-gg-tuning/SKILL.md, 08-deadends.md, 03-emit-knobs.md, gemm-optimization/SKILL.md, debug-flydsl-kernel/SKILL.md, FlyDSL/CLAUDE.md, programming-model.md, hardware/gfx950(或gfx942)/kernel-implementation-notes.md, optimize-loop.md
