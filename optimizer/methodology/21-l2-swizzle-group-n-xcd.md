# L2 swizzle 杠杆:1D M-cluster / 2D band(group_n)/ XCD remap 提 L2 residency

> 类别: 方法论 · 主题标签: l2-swizzle, xcd-locality, tile-programming, autotune

## 本质
- L2 swizzle 候选是纯 **WG→tile 双射,bit-identical**;不改数学,只调 **L2 residency**。增益 ∝ **L2 复用差多少**——复用差的 shape(big-N)增益大,square 无回归。
- **只组合一个 swizzle**:叠两个 locality remap 通常两败俱伤(stacking two locality remaps usually defeats both)。

## 硬件前提(gfx950/gfx942 同)
- 两代都是 **8 个 XCD chiplet,每个自带独立 4 MB L2 slice**(不是一块共享 L2)。
- Dispatcher 把 WG 在 XCD 间 round-robin→连续 blockIdx 落到不同 L2。要靠 remap 把相邻 tile 拉回同一 XCD 才能共暖 L2。

## 三种 swizzle

| 类型 | 做法 | 甜点 | 实测 |
|---|---|---|---|
| **1D M-cluster (group_m)** | 连续 `group_m` 个 M-tile 归同一 M-band,XCD-aware 调度到同 XCD 的 CU→`B[g]` 的 N-stripe 在同 XCD L2 resident | 默认 `nt_group_m=4`, `nt_num_xcd=8`(固定=物理 XCD 数) | 基线 lever |
| **2D band (group_n)** | N 切成宽 `group_n` 的竖带,带内再 `GROUP_M` 1D;把 working set(`GROUP_M·A_slab` + `group_n·B_slab`,各 ~2MB)锁进 L2;每组 A 的 M-slab 被多组复用 | `GROUP_M=4`, `group_n = n_blocks/8`(band 数=8=#XCD) | **big-N +12%**(L2 51→57.5%,MFMA 34→40%);big-K +1%;square 无回归 |
| **XCD WG-id remap** | 保持 `chunk_size` 个连续 id 在同 XCD:`chunk_idx*(num_xcds*chunk_size) + xcd*chunk_size + pos`→相邻 tile 共暖 L2 | — | GEMM/attention 有空间局部性时是实打实的 win |

- 2D band 触发条件:大-N shape(N≥2880 / N_BLOCKS_N 够多)才加 `group_n`;`group_n = N_BLOCKS_N//8`(#bands=#XCD)对 big-N 另有 **+8~9%** 口径。
- **persistent kernel** 要 remap 的是 **PERSISTENT work-id,不是 `blockIdx.x`**。

## 调参规律(4 轴 autotune)
- L2 swizzle 三参 `(group_m, group_n, num_xcds)`:小-M 要**大 group_n(16/32)**;`num_xcds` **NX8 普遍最优**。
- deep-wl `(16,15)`:只在 **K≥8192** 报(深流水藏高-trip-K g2s 延迟),用 `_WL_MARGIN=1.02` 让步(须超噪声带 2% 才选)。
- split-K:只对 **few-tile 大-K**(一 WG/tile 撑不满 CU)给 2/3/4/6/8/12/16。
- Primus-Turbo 生产用 **timed autotune 替 env**:首调每个 `(M,N,K)` 定时扫候选取 global-min per-shape 缓存。四轴且后三轴 **never-regress**(扫里恒含 baseline + 取全局 min + margin 门槛,只会追平或更快):L2 swizzle / deep-wl(phase-barrier `vmcnt`,`lgkmcnt`)/ 变体轴(COOP scale-load + TACCW wide-store)/ split-K。
- **CUDA-graph capture 内无法定时→回退静态启发式**。

---
来源: flydsl-fp8-gemm-tuning/SKILL.md, 02-nt-fwd-kernel.md, gfx950/kernel-implementation-notes.md, 13-primus-turbo-prod.md
