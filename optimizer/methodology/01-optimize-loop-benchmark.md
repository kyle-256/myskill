# 优化主循环与基准测量:结构纪律、计时尺子、测量噪声、公平对比、性能基线 regime

> 类别: 方法论 · 主题标签: optimize-loop, accept-rollback, scoring, representative-shapes, robust-timing, warmup, measurement-noise, benchmark-harness, round-robin, bisect, 瓶颈定位, fair-comparison, benchmark-discipline, correctness-gate, timing, perf-baseline, memory-bound-regime, roofline

## 内核优化主循环:一假设一改动、正确性先行、accept/rollback 纪律

### 循环骨架
- 完整阶段: `DEFINE_TARGET → PREPARE_ENVIRONMENT → READ_HISTORICAL_TIPS → BASELINE(round-1, full validation) → [ANALYZE → OPTIMIZE → VALIDATE → ACCEPT/ROLLBACK]* → TERMINATION_CHECK → REPORT`。
- round-1 = baseline,优化尝试从 round-2 开始。
- 决策循环细化(每 round 内):1) focused benchmark 抓 baseline profile;2) 写下瓶颈分类 + 一个假设;3) kernel 里只改一件事;4) **先重跑 correctness**;5) 重跑同一 profile/benchmark 路径;6) 数据支持假设才 accept。WHY: 5) 必须走同一路径,否则测量口径漂移。

### 硬规则(不可违背)
- **一 round 一假设 + 一个有意义的 kernel 改动**。多改一起做无法归因。
- **correctness before performance**: 性能重测前先过正确性门,correctness 挂了这一 round 直接废。
- **benchmark 全部 active validation set**,禁止只挑子集报数(no cherry-picking)。
- **accept-or-rollback 干净**: 未达标 `git checkout` 干净丢掉,回到上一个已 accepted baseline。每轮实验落成一个 git commit,达标才留。
- 连续两次 rollback → 触发 stagnation review。
- 判定永远由脚本跑固定 harness 说了算,**不采信 agent 嘴上说变快了**。

### 自动化 dispatch 判定(sync/flydsl_kernel_optimizer.py)
- 无人值守: 每轮"提议改动 → 远端编译 + correctness + benchmark → 达标 commit 否则 revert",最后 claude ultrareview 再 squash。
- `--bench-cmd` 脚本最后一行 stdout 打印 `{"ok":bool,"tflops":number}`;`ok=false` 或没高出 `--min-gain`(默认 1%)就 revert。
- 模式抄 AutoKernel / Meta KernelAgent / AMD AgentKernelArena。

### 执行模式
| 模式 | 何时用 | rebuild |
|---|---|---|
| repo-mode | 小 scope、参数调、快构建;Triton(Python)无需 rebuild | HIP/CK 参数改需 `GPU_ARCHS=<arch> pip install --no-build-isolation -e . -v` |
| workspace-mode | 新 kernel、重度试错、重 build pipeline;最小本地 dev env(src/tests/bench 从 upstream 抽出) | 迭代完 SYNC_BACK **只回核心 kernel 改动,绝不回 scaffolding** |
- workspace-mode 的 VALIDATE 拆成 local gate + integration gate,**只有过 integration gate 才算 accepted**。
- 验收统一回项目跑完整 `pytest tests/pytorch/ -v`。

### representative shapes 与验证粒度
- BASELINE 时从 **Check=PASS 的 row** 里选 3-5 个 representative_shapes,覆盖 small(launch overhead)+ medium + large(compute/memory)两端,**优先高方差 shape**,记进 `manifest.yaml: representative_shapes`。
- GroupGemm/MoE 至少含一个 **SKEWED expert 分布**(如 top_k=1 cf=1.25 及一个近退化 case),不能只测 uniform。
- quick validation(每 VALIDATE round)= 3-5 个 representative_shapes 子集;full validation = 全部 target_shapes,用在 BASELINE、一个方向结束、最终验收、borderline/high-risk 时。
- quick→full 升级条件: improvement <5%,或高风险改动(control flow、data layout)。

### scoring
- 逐 row 取 primary_metric + Check;**任何 Check=FAIL/ERROR → 该候选 score 0 / 直接拒**(Check 是硬门)。
- 单 shape → 该 metric;多 shape → **几何平均(仅 PASS shape)**。
- fwd+bwd(training)用 combined-step 指标: `Combined Step TFLOPS = 6 / (2/Forward_TFLOPS + 4/Backward_TFLOPS)`(per shape),再对 PASS shape 取 geomean。
- 逐 shape 回归看 **Combined Step Time**,不看单独 fwd/bwd 分量的正负。
- 保留 score 向量(per-shape 值),避免总分掩盖局部回归。
- primary_metric 按 campaign 类型: compute-bound forward-only → Forward TFLOPS;compute-bound fwd+bwd → Combined Step TFLOPS;memory-bound(elementwise/quant)→ Forward/Backward GB/s。

### handoff 前置信息
- kernel 源路径(Code Map)、focused test/bench 命令、benchmark 输出格式/metric、quick validation harness、scoring 规则、execution_mode + rebuild 方式。

## Robust timing:_robust_time 计时尺子统一、warmup 长度、buffer 复用防重叠虚高

### 计时尺子必须统一 (同一函数比改前/改后)
- 对比"改动前 vs 改后"必须用**同一个计时函数**,直接复用 `GK._robust_time`,不要自造。两把不同尺子不可比:
  - `_robust_time`:CUDA/HIP `torch.cuda.Event` 计时,warmup=250、reps=5、iters=50,取 **5 次的 median**。
  - 随手写的 `time.perf_counter()` + 少量 warmup 测的是绝对值,把 **host 端 Python/launch 开销**算进去,与 event 计时不是一个量级。
- grouped-gemm autotune dispatch 反复测出"看似很大其实非真回退"的差异,根因就是混用两把尺子。
- 禁用 `timeit.Timer` / `torch.utils.benchmark.Timer.timeit(50).mean`:会被 **CPU scheduler jitter + GPU boost clock 未稳定**污染。

### FlyDSL compile-once:避 jit 派发被计进 event(下述为 MXFP8 8-wave devloop 计时口径,与本节 tensorwise fp8 harness 不同,勿混淆)
- `flyc.jit` 的 launch **每次调用有 ~40us Python 派发开销**,cuda-event 会把它量成派发延迟(而非真实 GPU 时间)。
- 正确:先 `comp = flyc.compile(launch, ...)` **编译一次**,再 `comp(...)` 直进 GPU stream 用 cuda-event 计时(**warmup=30 / iter=300**)。

### 官方 benchmark 口径 (e2e / bwd 必用,禁手搓 event)(下述为 MXFP8 8-wave devloop 口径,与 tensorwise fp8 harness 不同,勿混淆)
- MXFP8 e2e benchmark 用官方口径(与 **TE GB200 一致**):`torch.utils.benchmark.Timer(stmt="fn()").timeit(100).mean*1e3`;`tflops = 2*M*N*K/(ms*1e-3)/1e12`。
- ❌ 别再试 手搓 cuda-event 计 bwd:会把 **autograd dispatch** 算进去,**bwd 低估 ~10-18%**。
- (注:上文"禁 `timeit(50).mean`"针对 autotune/kernel 微基准;e2e op-level 口径反用 `timeit(100).mean` 与官方对齐,但两条均为 MXFP8 devloop 的口径,与 tensorwise Primus-Turbo FlyDSL-fp8-gemm 流程不是同一套 harness,引用前需确认适用。)

### warmup 长度 (短-K / occ=1 的头号坑)
- 短-K shape **冷/热差异 >20%**,warmup 太短会严重 **mis-pick**(autotune 选错 config)。
- occ=1 的 **4-wave**(计算密集)对 boost 频率敏感:warmup=20 iters 够不到 boost,系统性**低估 5~10%**;occ≥2 的 **8-wave** 不敏感。
- 结论:测 4-wave vs 8-wave 必须 **warmup=250**(= 真实部署 boost 频率),否则会误判 4w 慢。

### 分层计时:op-level vs kernel-only
| 层级 | 计时方式 |
|---|---|
| op-level(含 quant) | graph-replay **min-of-8**(取 8 次 min 免 CPU 抖动) |
| kernel-only | `quantize_fp8_tensorwise_impl` 预量化后,CUDA event **warmup=20 / iter=100**,只包 kernel call |
| autotune 内置 | `_robust_time`:warmup=250 / reps=5 / iters=50 median |
| FlyDSL `@autotune` | `do_bench(fn, warmup=5, rep=25)`,CUDA/HIP events 返回 **median ms**;缓存 `~/.flydsl/autotune/{func}.json` |

### 防 kernel 重叠虚高吞吐
- 连续 launch **写同一 output buffer**(WAW 串行),避免每次新分配 buffer 导致 **kernel 重叠**造成虚高吞吐。
- 但实测:`_robust_time`(复用 buffer 串行)与"轮转 4-buffer 重叠感知计时"对 **4w>8w 结论一致**——**warmup 才是主因**,重叠不是。

### combined-step (fwd+bwd 成对 op)
- fwd+bwd 成对优化的 op 比 **combined step**,不比单方向:
  - `Combined Step Time = Forward + Backward Time`
  - `Combined Step TFLOPS = 6*M*N*K / (Combined Step Time*1e-3) / 1e12`
  - 等价形式:`Combined Step TFLOPS = 6 / (2/Forward_TFLOPS + 4/Backward_TFLOPS)`
- 单一比较分 = 各 shape Combined Step TFLOPS 的**几何平均**;forward/backward TFLOPS 留作诊断。
- 逐-shape 回退判定用 **Combined Step Time**,不看单个 fwd/bwd 分量的符号。保留 per-shape 分向量,防止总分掩盖局部回退。

### correctness 门 + TFLOPS 定义
- `bench_*_turbo.py`:先跑 correctness check 再用 `torch.utils.benchmark.Timer`(**20 warmup + 100 timed iters**)profile 写 CSV。
- Check 列 = PASS/FAIL/ERROR;任一 **FAIL/ERROR → timing 失效**(数字不可信,先修 correctness → score 0)。
- `Forward TFLOPS = 2*M*N*K / time / 1e12`;`Backward TFLOPS = 2*forward FLOPs / time / 1e12`。

### benchmark shape 来自真实 config
- shape 来自真实模型 config:`benchmark/ops/training/config.py`,不要硬编在 bench 脚本里(加新 op shape 扩 config.py)。
  - Dense GEMM:每模型 4 shape(attn QKV / attn out / MLP gate+up / MLP down)× MBS∈{1,2,4}。
  - Grouped GEMM / MoE:`MoEModelConfigs` + `gen_grouped_gemm_group_lens`(balance=True 均匀 / False 倾斜)两种 token 分布。
  - Attention:`gen_attention_test_cases()`。

### FlyDSL benchmark harness
- `bash scripts/run_benchmark.sh`(全 op);`--only softmax,moe` 子集;`--list` 枚举;`--output_csv /tmp/bench.csv` 出 CSV。
- 基线对比:`python3 scripts/compare_benchmark.py base.csv cur.csv` 出 ratio 报告。
- `@autotune(configs=[Config(...)], key=[...], warmup=5, rep=25)` 叠在 `@flyc.jit` 上,首调 bench 所有 config,后续用缓存最优。

## 跨进程 A/B 是幻觉:同进程交错 round-robin 取 min 才可信

### 核心:跨进程 A/B 全是幻觉
- **跨进程/跨 session 的 A/B 对比不可信**——GPU boost/throttle 时钟漂移 ~1.5-3%,大于绝大多数微调赢面。曾两次测出 **+1.66%** 和 **-2.2%** 完全矛盾的结果,都是漂移 artifact。
- **唯一可信 = 同进程内交错(round-robin)**:一个进程里对多 config 用**固定 swizzle** 编译好、交错计时、取 **min(reps30)**。漂移对所有 config 同相,交错采样使其抵消。

### 测量纪律(每次改动)
- 每改一次 env 先 `rm -rf /root/.flydsl/cache`。
- 每形状跑 3 轮,取跨轮最佳 **min-TF**(min-TF = 峰值时钟事件,抗节流)。
- 每轮**同 session 现测对照基线**(如 intrinsic `RAWAGPR=0`)对标,不用历史数字。
- 可选:每 K `rm cache` + `sleep8` 冷却。

### 可信测量方法排序(噪声量级)
| 方法 | 噪声 | 用途 |
|---|---|---|
| rocprofv3 `--kernel-trace` | ±3T | 最可靠,kernel-only 排除 host overhead |
| Event 500-sample (min/p5/p10/med) | ±10T med | 决策级比较,分布完整 |
| Event 100-sample | ±25T med (±0.5%) | 日常快速对比 |
| do_bench | 偏差大 | 不同场景不可比 |

- **决策级比较必须用 500-sample**;20-sample 噪声 ±20-25T。
- **只信 med 不信 min**:min 是瞬时最优 dispatch。
- 换算:`rocprof best ≈ Event min × 0.98`(Event 含 launch overhead)。
- **CUDA-graph bench 比 rocprof-min 干净**且 replay 确定性;rocprof-min 跨后端不可信(profiling 扰动曾误报 FLY 反超)。

### 瓶颈定位:同 FLOPs 不同 shape 对照
- 最快的定位工具 = **同 FLOPs 不同 shape 对照**(big-N vs big-K,同 kernel)。big-N 慢就拿 big-K 对照 profile,差异指标直接点出瓶颈。
- 例:L2 **51 vs 66** 直接指出瓶颈是 L2 复用。

### 归因技巧
- **gated `NOSTORE` env**(epilogue store 直接 return):`full 时间 − nostore 时间 = 暴露的 store 成本`。
- **斜率/截距回归**(us/K-block):斜率=稳态 compute,截距=固定 prologue/epilogue 开销,二者分离。

### 长 K 的 HW 级非确定(别用来判 race)
- K28672 在高负载机上有 **HW 级间歇非确定**:官方 intrinsic K28672 `DETRUNS=15` 也出 det 15233/152750;raw-baseline 出 det 39926。幅度比自研改动还大。
- K8192/K16384 三者均 **det0 干净**。非确定阈值在 **K16384~K28672 之间**,纯 HW/长-kernel 效应,与 kernel 逻辑无关。
- 教训:分辨 <1% 真信号必须用**短 K(K8192/K16384)做干净 det0 + 交错对标**,不能用 K28672 的 det 指标判 race。

### bisect 性能回归纪律
- 每个 commit 跑 **3 次取中位数**。
- 回归 <5% 警告可能是噪声;回归 <0% 说明 good/bad 搞反。
- bad 阈值 = `good + (bad-good)*0.3`(容噪声和渐变);接近阈值 10% 内跑 **5 次**。
- build 失败跳到相邻 commit,连续 3 个失败问用户。
- **全 good 时回归可能是环境性的**(driver/library/硬件热节流),重跑 bad commit 确认。
- 默认 **first-parent-only**,跳过 merge 内部。

## 可比 A-B 基线:两边裸调 kernel 绕 dispatcher、blockwise vs tensorwise 对齐

### blockwise vs tensorwise 目标线必须两边裸调 kernel
- 对比 blockwise 与 tensorwise 时,tensorwise 目标线**必须也是裸调 Triton kernel**(用 `bench_tensorwise_raw.py` / `bench_tensorwise_raw_bwd.py`),**绝不能用 dispatcher 路径的 tensorwise 数字**。
- WHY: dispatcher + autograd + autotune wrapper 会吃掉 **~30%**,用 dispatcher 数字会让 blockwise 显得比实际更接近目标(目标线被人为拖慢)。
- 两边用**同一套 `time_kernel`**:CUDA event + sort + **trim 20%**。

### 对比改前后必须用同一计时函数
- 计时尺子必须统一(不能自造 `time.perf_counter`),否则测出假回退:见本卡「Robust timing」小节（计时尺子必须统一）

### kernel 移植正确性金标准(gate)
- 同进程、同 device、同输入:原版 vs 移植版输出**逐元素比对 outdiff=0**(同源 kernel 应 bit-identical),再比 TF(应在 **±1.5%** 噪声内)。
- SNR 只能证"能跑对",**证不了"和上游同一版本"**;要证同版本必须走 outdiff=0。

## 性能基线数字与 regime 识别:vs Triton/GB200 倍数、memory-bound 不再调 GEMM

### 基线倍数(FlyDSL grouped fp8)
- vs Triton(B=8 balanced,12 MoE shape geomean,2026-06-16):
  - fwd **1.19×**、dgrad **1.14×**、wgrad **1.91×**(wgrad 相对优势最大,单 shape 最高 **2.11×**)
- vs GB200/TE(288 case):
  - fwd geomean **1.80×**、bwd **1.36×**,**286/288 PASS**(int64 解锁所有 shape)
- **MXFP8**(LDS-合并转置写)vs GB200(9 Llama shape,SNR 全 28dB):
  - fwd geomean **~0.99×**(≈对齐)、bwd **~1.10×**(反超)。
  - e2e(Timer 口径)新 vs 旧 BM=32:fwd **1710→1824 TFLOPS**(+6.7%)、bwd **1839→1878**(+2%)。
  - 提升集中在 quant-heavy K=11008 fwd:4096×4096×11008 **+21%**、8192 **+13%**、16384 **+10%**。
  - fwd 差距根因 = B200 硬件 MX cast 近免费 vs MI355X 软件 dual-cast(详见 mxfp8 卡 51)。
  - fwd 稳过 1.0× 仍需 in-gemm fusion ❌ 别再试(用户否决)。

### 本项目净收益分解(2026-06-09 → 06-16)
| kernel | before → after | 增幅 | 来源 |
|---|---|---|---|
| fwd  | 1841 → 2397 | +30% | 非持久 + swizzle-AT |
| wgrad| 1516 → 2163 | +43% | asm_mma + persist/masked M-branch + autotune |
| dgrad| ~2200 → 2307 | +5%  | bm128 小-M + path-J inline-asm |

### regime 识别:memory-bound 就别再调 GEMM
- Grok-2/Mixtral-8x22B fwd ≈ **1.0× vs GB200**:worst case = B=1 小-M 极大-N GateUP(Grok-2 **N=32768**)。
- 这是 op-level small-op/memory-bound regime:GEMM kernel 再快也被 **quant + 固定开销**淹没。
- WHY 停手:一旦识别出 memory-bound / host-overhead 主导的 regime,继续调 GEMM 内核不产生 op-level 收益 → 转去攻 quant/fusion/启动开销。

### roofline 实用判据(tile-size / M 维)
- **M≤512** 通常 memory-bound(关注带宽)。
- **M>512** compute-bound(关注 MFMA 利用率)。
- 算术强度 = flops / bytes_moved,与 roofline crossover 比较判定所在 regime。

### 为什么 grouped wgrad 是"效率优化真能体现"的路径
- grouped wgrad **≠ DVFS 功耗受限**(区别于 dense fp8):实测满频 **2400MHz ~266W**,远低于 dense randn 的 **~1072W 功耗墙**。
- 原因:MfmaUtil 仅 **~54%**,够不到功耗墙 → 指令效率/autotune 优化能真实体现。
- 反例:dense/fwd/dgrad 的指令效率优化被功耗墙掩盖(相同 MFMA → 相同功耗 → 相同频)。

---
来源: remote-sync/SKILL.md, optimize-loop.md, SKILL.md, tool-rocprof/SKILL.md, optimize-handoff/SKILL.md, flydsl-fp8-gemm-tuning/06-autotune-design.md, flydsl-fp8-gemm-results/SKILL.md, flydsl-fp8-gemm-tuning/07-benchmarking.md, pr-merge-gate/SKILL.md, flydsl-kernel-authoring/SKILL.md, FlyDSL/CLAUDE.md, verify-performance/SKILL.md, mxfp8-8wave-devloop/SKILL.md, flydsl-fp8-gemm-tuning/SKILL.md, agpr_rawasm_progress.md, agpr_phase5_ldsr.md, project_mxfp4_epilogue_store.md, agpr_phase5_mono.md, 07-benchmark.md, bisect-perf-regression/SKILL.md, mi300-blockwise-gg-tuning/SKILL.md, 09-perf-numbers.md, gemm-optimization/SKILL.md

## ★★★ 造尺子之前先 grep 有没有现成的(2026-09-03,同一个坑犯第二次)

要测某个算子时,**先在 memory / `_bench_campaign_*.py` 里搜这个算子有没有被量过**。自己现搭的探针
几乎必然会重踩已经记录在案的坑。

实例:测「gpt-oss 稠密投影 FlyDSL vs scaled_mm」,我现写了个探针,同时犯了两条已记录的错:
1. **把 bf16 喂给公共 op** ⇒ 每次调用都在量化。`project_gptoss_fp8_nopad_campaign` 的原话是
   「走公共 op 的话量化占 ~40%,把比值稀释(我第一版就错在这)」——**同一句话适用于我这次**。
   正确做法:`QuantizedTensor.quantize` **预量化一次**,闭包里只调 GEMM(部署里权重本来就是预量化的)。
2. **单个 GEMM 紧循环计时** ⇒ operands 一直待在 L2,量的是热数据。dense fp8 campaign 专门弃掉了
   这种计时,并因此**撤回过一个伪证**。正确做法 = **STREAM**:把该算子的若干形状排成链背靠背连跑,
   同一形状两次调用之间其它几个自然把它的 operand 逐出 L2/MALL,链内逐个计时、median over ITERS。

代价:第一版数字(1.078-1.099×)碰巧和正确尺子(1.091-1.095×)接近,**但逐格结构完全不同**——
正确尺子才看得出「FlyDSL 不是全面更好,是更稳:scaled_mm 在两格塌到 1700 TF/s」。
**数字对不代表方法对;方法错的时候,对也是碰巧。**
