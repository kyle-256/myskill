# Autotune 设计与调度:五原则、生产四轴 never-regress、N-GPU N-agent 事件驱动调度

> 类别: 方法论 · 主题标签: autotune, dispatch, backend-selection, never-regress, L2-swizzle, split-K, fleet-scheduling, multi-agent, harness

## Autotune 设计五原则:balanced 计时、M-branch、hysteresis 1.5%、纯静态 cache key、never-regress

### 五原则
1. **balanced 计时**:M_total 均分到 G 组(`_balanced_targs` / balanced group_offs)计时,不被首次真实调用的 skew 倾斜分布带偏。
2. **M-branch**:不同 M size 用**完全不同**的 candidate set。NN small-M gate(`bm128_tiles <= num_cus`)时 bm128 永远赢,直接 return 单 config,不进 autotune。
3. **hysteresis 1.5%**:只在候选比当前快 ≥1.5% 才切,防噪声 mis-pick。
4. **cache key 纯静态维度**:见 pitfalls/06-autotune-dispatch.md
5. **warmup 要长**:NT fwd 用 median-of-5 × 50 iter warmup;短-K 冷测 mis-pick 严重。

### correctness reference + 切换门槛
- candidate 第一个 `cands[0]` = correctness reference;后面只有 ≥1.5% 更快才切换。

### ★ hysteresis 的副作用:候选表要"沿轴排梯子",cands[0] 的座位本身是一个可调参数
1.5% 迟滞意味着**赢不到 1.5% 的候选永远落不了地**,于是:
- **cands[0] 就是决策**(见 pitfalls/06);想换赢家,改座位比加候选有效。
- **更进一步(2026-07-31 实测)**:当同一条轴上没有"到处最优"的点时,把 cands[0] 坐在
  **轴的一端而不是中间**,能让迟滞链条一步步走到另一端。tw grouped NN dgrad 的
  `num_xcd` 轴实测:xcd2 只在 skew 方 square-K 最优、xcd4 在 deep-K 最优、xcd8 在
  balanced square-K 最优、xcd1/xcd3 全面落后。候选表 `[(xcd4,gm8),(xcd8,gm4),(gm2)]`
  (中间起步)gm=0.98628;改成沿轴的梯子 `[(xcd2,gm8),(xcd4,gm8),(xcd8,gm4)]`
  → gm **0.98770/0.98738**(dgrad 组 +0.35pp)。从 xcd4 起步时两个邻居都够不到 1.5%,
  梯子就断在原地。
- ⚠️ 但**别越过端点**:同一轮把 dgrad 基座推到 xcd1、把 NT 短-K 基座推到 xcd2,
  gm 掉到 0.98564(dgrad 组 −0.25pp、fwd 组 −0.42pp)。梯子只能铺到实测的最优端。
- **先数一遍每个 shape 分支实际有几个候选**:同一个 dispatch 函数按 `N vs K` 分叉时,
  很容易出现某一分支只装了 3 个(浪费一个名额)。gpt-oss fwd `down`(N==K==2944)就是
  这种"空位"分支,补一个 `(xcd4,gm4)` 座位实测 fwd 组 +0.30pp(两次读数一致)。

### never-regress(追平或更快,永不回退)
- **twin 交错取 global-min**(通用模式,细节因轴而异):候选 config 作为多路 twin **同进程交错计时**取全局 min(交错验证 twin 正确性等价且稳定才敢用,避免噪声误选)。
  - **变体轴(COOP/TACCW)**:四路 twin `{df,T,C,CT}`,`margin` 由 0.99 演进到 **0.995**,`VREPS=16`(2026-07-01)。
  - **split-K**:缓存 key = `(M,N,K)`(未标注 margin 数值,不等同于变体轴的 0.995);`config tuple` 的 8-tuple 形式 `(gm,gn,xcd,persist,wlv,elgk,taccw,coop)` 也只出现在变体轴段。
- **Primus-Turbo 生产 timed autotune 四轴机制**:见下节「Primus-Turbo 生产 timed autotune」。
- **split-K 候选加入条件**:`tiles < ncu/2` 且 `K ≥ 2048` 且 `s` 整除 `K//256`;实测端到端(含 reduce)取 min → 永不回退。

### mode-split launch
- graph capture 用 raw `@flyc.jit`(graph-friendly);eager 用 `flyc.compile` 一次性编译的 compiled 对象(跳过 per-call drift-check overhead)。
- 小 shape eager per-call **18.4 → 17.2us(~7%)**。

### Backend 选择优先级(高→低)
代码 setter → 环境变量 → autotune → 代码内 default → fallback(试所有 `can_handle` 的)。
- env 支持按精度:`"fp8:CK,other:TRITON"`。
- env:`PRIMUS_TURBO_GEMM_BACKEND`、`PRIMUS_TURBO_GROUPED_GEMM_BACKEND`、`PRIMUS_TURBO_ATTENTION_BACKEND`、`PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND`、`PRIMUS_TURBO_AUTO_TUNE`。

### 强制/切 backend API
- `GlobalBackendManager.set_gemm_backend(BackendType.CK)` 强制。
- `GlobalBackendManager.set_auto_tune(True)`(或 `PRIMUS_TURBO_AUTO_TUNE=1`)。
- `GlobalBackendManager.reset()` 清设置 + autotune 缓存。测试里强制 backend 后 teardown 必须调 `reset()`。

### BackendType → 用途映射
| Backend | 用途 |
|---|---|
| HIPBLASLT | GEMM bf16/fp8 tensorwise;**dense GEMM 默认** |
| TRITON | GEMM/GroupedGEMM/Attention;可调无需 rebuild |
| CK | GEMM/GroupedGEMM FP8 row/block |
| TURBO | MXFP8/MXFP4 GEMM、Attention;自研 gfx950 |
| AITER | Attention 默认 |
| DEEP_EP | MoE dispatch/combine |

## Primus-Turbo 生产 timed autotune:四轴 never-regress、L2/deep-wl/变体/split-K

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

## N-GPU N-agent 事件驱动调度:GPU 永不空闲、diversify 换维度家族、进步阈值 1%

### 事件驱动调度(GPU 永不空闲)
- N-GPU N-agent:每收到一个 agent 完成通知,立刻做三件事——更新 `state.json`(**先读再写**,不丢历史)→ 释放 `gpu_busy` → 按优先级立即 dispatch 新 background agent。GPU 空出即填,永不空闲。
- dispatch 优先级:`queued_next` > `refining`(邻域扫) > `diversifying`(换维度家族) > `done`。
- 收敛判据:进步阈值 **1%**(单轮 gain < 1% 当噪声,不算进步);`no_progress` 连续 **3 轮** 封顶为 `done`;**never regress**(不回退到更差 cfg)。

### tuning harness 要点(避免测量污染)
- 直接调 kernel,**绕开 dispatcher**——dispatcher 自带 autotune,会污染 timing。
- 量化 / 数据准备**只做一次**,循环里只换 cfg。
- CUDA event timing:排序后**掐头尾各 20%**(trim outliers)。
- 写**完整 JSON**:`shape` / `best` / `all` 三段。`all` 必留——diversify 要看次优解分布。
- 备两套 grid:`broad`(粗扫)+ `refine`(邻域细扫)。

### diversify 必须换维度家族(不是同 grid 再扫一遍)
- 只换 BM/BN 反复扫同一 grid 无意义。换维度家族:
  - split-K
  - 不同 `mfma_nonkdim`
  - `chunk_size`
  - cache modifier
  - `waves_per_eu`
  - scale 预取

### 自动化 FlyDSL 优化循环(commit/revert)
- `sync/flydsl_kernel_optimizer.py`:无人值守跑 N 轮 "提议改动 → 远端编译 + correctness + benchmark → 达标 commit 否则 revert";最后 claude ultrareview 再 squash。
- 核心模式(抄 AutoKernel / Meta KernelAgent / AMD AgentKernelArena):每轮实验落成一个 **git commit**,没达标 `git checkout` 干净丢掉。
- correctness + 性能判定**永远脚本自己跑固定 harness 说了算**,不采信 agent 嘴上说"变快了"。
- `--bench-cmd` 脚本最后一行 stdout 打印 `{"ok":bool,"tflops":number}`;`ok=false` 或没高出 `--min-gain`(默认 **1%**)就 revert。

---
来源: 06-autotune-design.md, 02-nt-fwd-kernel.md, 13-primus-turbo-prod.md, project_mxfp4_epilogue_store.md, SKILL.md, gpu-fleet-tuning/SKILL.md, remote-sync/SKILL.md
