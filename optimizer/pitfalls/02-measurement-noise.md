# 测量噪声全谱：地板/DVFS/掉频/并行口径/host 开销/虚高 TF/race 掩盖

> 类别: 踩过的坑 · 主题标签: measurement-noise, dvfs, interleaved-ab, regression-gate, clock-throttle, autotune-dispatch, parallel-bench, triton-cache, wrapper-overhead, quant-e2e, raw-op, snr-gate, do-bench, shape-alignment, race, correctness, wgrad

## ★优化口径：EAGER 优先，禁用 cuda-graph 掩盖 host/launch 开销

- **用户硬规则**：grouped GEMM(及同类)性能优化**只看 eager 模式**。不允许用 cuda-graph 的数字来"达标"——cuda-graph 会隐藏 kernel launch 延迟 + inter-kernel gap + host wrapper 开销,把 eager 下真实存在的 overhead 抹掉,是自欺欺人。WHY：真实训练/推理里这些 overhead 对小/短-K shape 是实打实的瓶颈,graph 只是掩盖不是消除。
- 推论：eager 下的杠杆是**减 kernel 数 / 减 host torch op / 减 launch**(合并 preshuffle、消 F.pad、融合),而不是"反正 graph 会摊掉"。cuda-graph 只用于旁证"某开销是 launch 而非 kernel-exec"(诊断用),**绝不用于报达标数**。

## ★小-workload(B=1)热稳态协议：连续满载预热,绝不 sleep 冷却

- **现象(2026-07-21 meta hd64 bwd B=1)**:同一个 kernel、同 config,冷启动 1024TF / 带 `sleep(0.3)` 半冷 727-969 / 连续满载稳态中间值 —— **差 40%**,把 5-14% 的真实缺口完全淹没。根因=B=1 workload 太小,GPU 时钟 boost/衰减 + pipeline 填充状态的瞬态占比巨大。
- **协议(反直觉但正确)**:①**连续满载预热数秒(WARMS≈3s,循环跑 kernel + sync,绝不 sleep)** → 让时钟 settle 到持续满载稳态、pipeline 填满;②预热后**无 sleep 连续测 REPS≥7 取中位数**。
- **严禁 sleep 冷却**:`sleep` 让 GPU 掉出满载 pipeline + 时钟回 boost,测量**既不稳又偏低**(连续满载反而更高更稳)。这跟"控热靠 sleep 散热"的直觉相反 —— 反映真实部署(持续满载)的口径就是**连续满载稳态**,不是间歇。
- **验收/优化前必须先定死这个协议**,否则改一版分不清是优化还是热漂移。参考 [[project_meta_bwd_accept_bench]]。

## 测量噪声地板：run-to-run ~5% / DVFS 功耗受限 / 多轮 interleaved 才可信

**噪声地板的量级（不认清就会把噪声当收益放行）**
- 同代码同 shape 连跑 3 遍完整 sweep，数值本身有 **4%** 上下波动（GPU 时钟/温度每次不一样）。只看一次对比就下"回归 X%"不可靠。WHY: 新旧版本波动范围重叠 → 大概率是噪声。
- fp8 GEMM run-to-run 噪声 **~5%**，和很多 lever 增益同量级 → 单次比对无法分辨真收益。
- `timeit.Timer.mean` 在 GPU 温度不稳时被 **±5%** 噪声掩盖；单次 3-trial 对 sub-ms shape 被热/冷态差异 **±8%** 带偏。

**fp8 是 DVFS 功耗受限（跨脚本绝对 TFLOPS 不可信）**
- 同一 kernel，输入 `zeros`→`randn` 速度差 **21-37%**（数据幅度改变功耗 → DVFS 降频）。
- 跨脚本绝对 TFLOPS 方差 **~8-12%**。WHY: 不同脚本/session 的时钟状态、数据分布不同。
- 结论：**跨脚本绝对 TFLOPS 不可信**，只认同脚本同数据的相对比较（只信比值，如 4w/8w），配多轮均值。

**判胜/放行纪律**
- A/B 谁更快最稳办法：两份代码同一远端（只要 csrc/cmake 没变，直接互换 python 文件原地跑）、同脚本同 GPU **紧挨着**跑，而非不同时间/session。
- 至少跑 **3 轮**取范围；波动范围重叠即噪声。
- 增幅 **< DVFS 噪声带（~2-3%）** 视为噪声，不放行。
- Benchmark acceptance discipline: improvement **<2%** 属近噪声 → 重测 **>=3x**，只在 mean improvement **>1%** 且 **stddev < 收益幅度的一半** 时接受，否则判噪声 reject。
- 正确性是硬门：任何行 Check=FAIL → aggregate score = 0，立即 reject。
- 任一 core shape 在主接受指标上回归 **>=5%** → 默认 reject。
- best-of-3/4 bench + 多次复测区分真假：只在确认真实进步（超噪声）**且 user 认可**后，才把改动合成**单个干净 commit**。

**interleaved A/B 是唯一可信判胜法**
- 正解 = interleaved A/B：**同进程**交替 config A/B × N-trial，**win-count** 判胜（不是比均值绝对数）。WHY: 交替执行让两者共享同一时钟/温度轨迹，抵消 DVFS 与热漂移。

**timing 引擎别跨比（工具边界）**
- （归属：`quick_test_bench.py` / `bench_<op>_turbo.py` 是 **fp8-gemm turbo harness**；dsv4 attention 另用单个 `bench_mla.py`。）
- `quick_test_bench.py` 用手写 `time.perf_counter` 循环；full `bench_<op>_turbo.py` 用 `torch.utils.benchmark.Timer` → **不同 timing 引擎，绝不能跨这两者比绝对数字**。
- `quick_test_bench.py` 是 round-to-round 每-shape 回归门的**权威源**（BASELINE 和每个 VALIDATE 都跑它，比 `--summary-csv`）。
- full bench 只用于：(i) 从全 shape 集挑 `representative_shapes`；(ii) campaign 后最终验收。

**--summary-csv schema 必须逐字节冻结**
- GEMM 类列（严格顺序）：`label,B,M,N,K,Check,Forward TFLOPS,Forward TFLOPS_stddev,Backward TFLOPS,Backward TFLOPS_stddev,Forward Time (ms),Backward Time (ms),out_snr,da_snr,db_snr`。
- `Check` 用 `PASS/FAIL`；`*_stddev` 是该 metric 单位下的**绝对** stddev。
- ❌ 别再试 改列名/调列序：任何 schema drift 会**静默关掉**每-shape 回归门（gate 按精确列名 key）。
- 非 GEMM op：保留 `label/Check/*_stddev` 约定，只换掉 GEMM 专用列。

## 掉频/热节流假胜：冷 GPU boost、僵尸进程压 10%、autotune 选型赶上坏热态

### transient 掉频 → 假胜
- 曾把 8w 瞬时掉频测成 4w 赢 **1.42×**，复测仅 **1.01×**。WHY: 那一测正好赶上 8w 侧 transient 掉频/被抢卡。可疑就 `rocm-smi` 看是否掉频/被抢卡并立即复测。

### 僵尸 GPU 进程压 ~10%（掩盖 3 次回退）
- 测速前必须查杀僵尸 GPU 进程，否则带宽/时钟被压 **~10%**，结果严重偏低——曾把 **~2% 回退掩盖 3 次**。
- `rocm-smi --showpidgpus | grep 'using.*DRM'` 找真实 KFD PID。
- `<defunct>` 状态的 python 进程已死不占 GPU；**真占 GPU 的是有 KFD entry 的**。
- 杀完 `rocm-smi --showuse | grep 'GPU use'` 全 **0%** 才算干净。

### 不同 GPU boost 差 10-30%，必须同 GPU 重现
- 死坑：不同 GPU 的 clock boost 状态不同，同一 kernel 差 **10-30%**。GPU0 冷/boost 高 **5536 med** vs GPU7 正常工作温度 **5401 med**。
- 早先看到的 8w **4896/4910** 就是冷 GPU 假象；单 GPU 顺序测才排除并行热降频。
- 判据：GPU 换挡不算达标，目标必须在**同一 GPU 稳定重现**。

### autotune 一次性选型赶上坏热态
- dispatch 只在第一次调用某 shape 时跑候选竞赛并 cache。若 sweep 按固定顺序连测多 shape，某 shape 的选型时刻恰处 GPU 刚从冷启动/低时钟回升阶段，选出的候选可能不是稳态最快的。**不是 dispatch 逻辑错**，是那次选型赶上不具代表性热力状态。
- warmup 长度（短-K/occ=1 的头号坑）：见 methodology/01-optimize-loop-benchmark.md

### bench 前 set_auto_tune(False)
- bench 前必须 `set_auto_tune(False)`，否则每个 shape cold-start 跑一遍 autotune 污染 first-iter。
- HK 内部 `_autotune_pick` cache 是 process-local dict，warmup **20 iter** 足够 cache-hit 后才进 timing loop。

### 回归判据（先排噪声再定性）
- NN `16384×4096×4096=1011 TF` 是异常低点（同 shape M=8192 有 **2689**），属小-N 方阵 regime 的 autotune 选到坏配置/timing 抖动；小方阵（oproj 4096×4096）是 dense 最弱 regime。
- 真回归判据：同脚本重测某 shape 掉 **>8%（超噪声）**才算真回归。先排除 autotune 缓存没命中/别的进程抢卡。

## 并行/跨 GPU bench 口径：8 卡并行慢 10%、before/after 必须同并行度同 GPU

- **8-way 并行 bench 所有 kernel 约慢 10%**：8 卡同时跑时每卡带宽被共享 L3/HBM 和电源分摊→所有 kernel 一律慢约 10%。并行 bench 只能做**相对比较**(同一次并行跑内互比)，**不能读绝对 TFLOPS**。WHY：绝对值被共享资源压低了。
- **before/after 必须同口径同并行度**：两次跑并行度不同时 A/B 对比无意义。曾出现 fwd "看似回退"其实只是 after 跑在不同负载(不同并行度)下的假象。规则：before/after 要么都单卡、要么都 N-way 同时，口径必须一致。
- **A/B 绝不同 GPU 并行两个计时任务**：'m4096 低于 racing' 曾是同一 GPU 上并行两个计时任务互相干扰造成的假象。跨 GPU 有约 **1% 方差**。正确做法：A/B 对照分开 GPU，或同 GPU 串行；关键结论用**同 GPU 交替多 trial 取中位数**。
- **多 agent 编译撞 Triton cache lock**：多 agent 同时编 kernel 会撞 `.triton.lock`。每个 agent 必须单独 `TRITON_CACHE_DIR=/tmp/triton_cache_<N>` + `HIP_VISIBLE_DEVICES=<N>`。
- **共享 csrc 编译只做一次**：先一次性 `pip install -e . --no-build-isolation`，后续 agent 只跑 Python，避免重复编译争抢。
- **sub-agent 必须 background 跑**：否则会话被锁死。

- ❌ 别再试：用 8-way 并行 bench 的绝对 TFLOPS 下结论——一律被压低约 10%，只有相对值可信。
- ❌ 别再试：before/after 跨不同并行度对比——负载不同，回退/提升都是假象。
- ❌ 别再试：同一 GPU 上并行两个计时任务互比——互相干扰(如 'm4096 低于 racing' 假象)，要串行或分 GPU。
- ❌ 别再试：多 agent 共用同一 `TRITON_CACHE_DIR`——撞 `.triton.lock`。

## 小 shape host/wrapper 开销淹没 kernel：走 raw op 绕过公开入口

- **小 shape 的 kernel-only 测量被 host 端固定开销淹没**：m=1024、单次 kernel 只 0.05~0.1us 时，每次走完整公开入口（`torch.empty`/`.view`/`.reshape`/dispatch-cache 查表）的固定开销比 kernel 本身还大，测出 TFLOPS 波动 **5~15%**，和 GPU 计算吞吐基本无关。
  - **验证"是 kernel 变慢还是 host 波动"的手法**：绕过公开入口，直接编译目标 kernel（如 `GK._compile_grouped_tn_wgrad_persistent`），`targs` 只建一次再反复 `_robust_time(launch, targs)`。若两版本一致 → 之前的差异就是 host/调度噪声。(src: remote-sync/SKILL.md)

- **测纯 kernel TFLOPS 必须走 raw op，不用 PT wrapper**：用 `torch.ops.primus_turbo_cpp_extension.hk_*`，绕开 wrapper。wrapper 把以下固定 overhead 算进 timing：
  - `quantize_fp8_tensorwise_impl` ~50µs + 30µs
  - `grouped_gemm_compute_offs` ~5µs
  - `dispatch` ~10µs
  - 合计 **~80µs 固定 overhead**，对一个 100µs 的 kernel 占 **45%**，会稀释/放大 loss% 对比。(src: fp8-gemm-bench/SKILL.md)

- **所有 4w vs 8w 的对比数都必须是 GEMM-only**：含 quant 的端到端 wgrad 比纯 GEMM 掉 **~15-30%**（悲观口径）。多出来的是一次纯访存 tensorwise 量化（amax 归约 + scale + cast），N/flop 越小占比越大。
  - 但这份 quant 是 **dgrad + wgrad 共享**的，bench 100% 算给 wgrad 是悲观上界，真实分摊约一半。(src: flydsl-fp8-gemm-results/SKILL.md)

- **含 quant 的 e2e 才是用户真实数字，纯 kernel 5500T 只是上限**：含 quant e2e 绝对 TF 大跌 —— fwd **~1700-2200 vs 纯 kernel ~5500**。原因是量化 + 小 M（mbs=1 时 M 仅 4096，算术强度低）双重拉低 FLOP 效率，两边都吃。
  - ❌ 别再试 用 raw-scale 直读核当 aiter baseline：kernel-only baseline 必须用 **shuffled scales**。`bpreshuffle=False` 只是不做权重 B 的离线 preshuffle，**scale 仍必须 shuffle**；raw-scale 直读核 SNR<0 是垃圾，不能当 baseline。(src: 14-fused-preshuffle-e2e.md)

- **cuda-event 隔离测小 kernel 是假象**：单独测小 kernel（如 preshuffle）用 cuda-event 隔离，会把两个 record 之间 GPU 空等 **~22µs** 的 host/`flyc.jit` 派发算进 elapsed → 得 **37µs / 1.4 TB/s（假象）**，而 rocprof kernel-trace 实测只 **15.5µs / ~3.4 TB/s**。
  - ❌ 别再试 信 cuda-event 隔离值：测小 kernel 必用 `rocprof --kernel-trace`，或用 **both-minus-gemm-only 的差**（host 气泡才抵消）。(src: mxfp8-grouped-gg-devloop/SKILL.md)

- **memory-bound 短 kernel 外的 host per-call 元数据 / tiny-op launch 税常 > kernel 本身的差**：把 grouped quant 的 O(G) 组搜索搬 host（`arange`+`searchsorted`+4×`gather`+`where`×3 算 RB/RO/RE）后 kernel 降到 **153** 但 wrapper 反升到 **236** —— ~10 个 tiny torch 算子每次串行发射 **+80~120µs**（launch-bound，同 stream 无法重叠）。
  - grouped qa wrapper 算 padded lens/offs 的 ~7 个小 torch kernel（`ceil`×2、2×`cumsum`、`fill`、`copy` 各约 4.4µs 独立 launch）≈ **31µs**，dense 完全没有。
  - ❌ 别再试 把 prologue 摊成一串 tiny torch 算子搬 host：正解 = 融合 **on-device prologue kernel**（HIP 用 `compute_padded_layout_gpu <<<1,1>>>`，~9µs，1 线程从 tight int64 offs 直接算 64/128 对齐 lens/offs），塞进 jit stub 与 meta/kern 背靠背发射。(src: mxfp8-grouped-gg-devloop/SKILL.md)

- **per-call scale 转换 Python 循环是 host 瓶颈**：`gemm_mxfp8_flydsl_kernel` 的 per-call scale 转换（broadcast → WL lane-contig 经 `preshuffle_scale_lane_contig` 的 Python 循环）= **8000µs/call**，是端到端 host 开销瓶颈（kernel-only perf 已达标）。解法：向量化 / 缓存。(src: project_mxfp8_wholeloop_port.md)
  - （术语：**WL** = whole-loop mxfp8 移植路径；**lane-contig** = 把 scale 重排成 lane-contiguous 布局供该核直读。）

## 虚高 TFLOPS 假象：SNR<0 跳过计算 TF 虚高、超 peak 数字、do_bench 不可靠

**核心铁律：高 TF 数字必须先过 SNR/det gate 才算数。** 任何超 peak 或异常高的 TFLOPS 在过 gate 前一律当 bogus。

### SNR<0 编译器跳过真算 → 时间短 → TF 虚高（mirror 死坑）
- SNR<0 时编译器把真实计算优化掉 → kernel 时间短 → TF 虚高。
- **FEWOP=1**（用单 reg 测 operand 多样性天花板）得 **5648 TFLOPS**，但 SNR garbage。
- **SCDWX4 / TRB8 的 '5640'** 同样是 SNR<0 假象；一旦计算正确，实际 **<5176**。
- ❌ 别再试：把这些 5648/5640 当天花板参考——它们是跳过计算的产物，不是可达性能。

### '跳过整条指令测天花板'类探针不可信
- **PT_TR_HALF**（跳过读）之类探针不可信：跳过读 ≠ 换成更少的等效读。
- 真实替换后（`ds_read_b128` 换 2×`tr-b8`）因带宽受限，收益归零。
- ❌ 别再试：靠删指令测'去掉 X 的天花板'。测'去掉 X'必须用**真实替代指令**，不能靠删指令。（呼应 methodology/03「★★ 上界≠可达铁律」：subtractive/HALF/roofline/纸面 op-count 都只给上界，判正/判负前必须 edit→bench 真实现。）

### proxy 测量的赢点常是 artifact
- **STORE_PLW / BPERM** proxy 用 contiguous 地址（数据故意错，footprint 变小）显得快 **+8%**；正确 row-strided 数据拿不到——coalescing 受 tile 列宽限，最大 ~64-128B。
- **COALADDR** 把数据写飞进别的行才显快（地址错 → artifact）。
- （以下两条非 proxy artifact，是真实测的中性/判负结论，仅同处此小节）
- `s_setprio` 对 mxfp4 neutral（`waves_per_eu=1` 无跨 wave 仲裁对象 + 稳态已 stall-free）。
- **wl-depth** 轴对 Llama shape 全在 ±0.5% 噪声内（"best" 随机跳）。❌ 别再试：扩 wl autotune。

### do_bench 不可靠 → 用 cuda.Event
- ❌ 别再试：`triton.testing.do_bench`。某些 shape 测出**超 peak（4363 TFLOPS = 87% MI355X peak，甚至 4500-22000）**但 kernel 跑 garbage；内部 cudagraph capture / cache eviction 行为不可控。
- 正解：`torch.cuda.Event(enable_timing=True)` + `record()` / `elapsed_time()` 直读硬件 timestamp。

### 非对齐 shape silently early-exit → bogus 超 peak
- HK dense kernel 在非对齐 shape 上 **silently early-exit**（不算 partial tile 直接返回），测出**超 peak（4500-22000 TFLOPS）**的 bogus 数字。
- 必须 **M%256==0 N%256==0 K%128==0**。
  - 安全 N：2048 / 4096 / 8192。
  - 安全 K：128 / 256 / 512 / 1024 / 2048 / 4096 / 7168 / 8192。
- ❌ 别再试：shape 不对齐时比 dense vs grouped——dense baseline 是 bogus。

### SNR gate 正确用法
- `get_tolerances` 返回的 **fp8=1e-1 / fp4=0.5** 容差故意很松，**不能当真实 gate**——量化后 element-wise 容差没意义。
- 低精度主 gate 必须用 **SNR 门**：`compute_snr(ref, actual)`，参数 **reference 在前**。
- 只用于诊断（非 gate）：`relative_error` / `mean_squared_error` / `max_abs_error` / `cosine_similarity` / `symmetric_similarity_diff`。

### fp8 TN big-shape 典型瓶颈画像（profile 解读）
- **VMEM Utilization ~3%** → 不是 memory/store/带宽 bound。
- **Dependency Wait 高（source 未给具体数字）+ MFMA Util ~34-40% + Occupancy ~1 WG/CU** → latency-bound（8 waves 喂不饱 MFMA+tr8 延迟链）。
- **L2 Cache Hit**：square/big-K ~66%，big-N ~51%（L2 复用差是大 N 的主瓶颈）。

### （另一内核）wgrad 4-wave 3buf 的 bank conflict / chunk_stride
- 这是**不同内核**（grouped wgrad 4-wave 3-buffer transpose-read 内核，非上面的 dense-TN autotune 内核）：`chunk_stride=1056` padding 消掉了转置读的 bank conflict，`1 池@1056 = 0%，2 池@1024 = 14%`（`_CS` 越界会导致 LDS 超限编译报错）。
- 来源：10-grouped-wgrad-4wave-3buf.md, project_wgrad_occ_feed_bound.md（不属于 fp8 TN big-shape 画像）。

## SNR 掩盖低概率 race：只有 bit-exact 30000+ 次多跑能测出

- **SNR 会掩盖低概率 race**：0.17% ~ 1/30000 级别的 bit-flip 在 SNR 里几乎看不出来，SNR 数字正常不等于输出干净。唯一可靠的检测是 **bit-exact 多次跑**（`_race_wg.py`，需 **30000+ 次**）。低于这个量级根本采不到那个 flip。
- **任何改缓冲布局 / vmcnt 都必须重跑 race 测**：这两类改动直接影响跨 barrier 的写窗口，SNR 过了也可能已经在腐蚀输出。改完不重跑 `_race_wg.py` = 没验证。
- **racing 优势本质是不安全的跨 barrier 写**：`PT_WL_2BPOOL` 2 池 3buf 时开 racing（`PT_RACE_VM=1`），在 m4096 上 SNR 掉到 **53-54**，就是输出正在被腐蚀的信号。其机制是让 `vmcnt(16)` 把 4 个 pool 的全部 G2S 写放到跨 barrier 之外（约 **0.17% bit-flip**），这是不安全的加速。
- **安全 3buf 才是正解**：不要为 racing 的速度收益牺牲正确性；racing 的"优势"是拿正确性换来的假象。
- （下为 grouped MXFP8 GEMM 内核，分支 `dev/kyle_mxfp8_gg_pr`，与本卡 wgrad 4wave 内核无关，跨内核参考）**官方 deterministic pytest 是该内核的 race 验收口径**：以 `test_grouped_gemm_fp8_mx_blockwise_deterministic` 为准（`rtol=0`/`atol=0` + `empty_cache` churn、`repeats=10`）；d7b149a ×100 全过。全量 `pytest tests/pytorch/ops/test_grouped_gemm_fp8.py -k mx_blockwise` 期望 **3840/3840**。
- （同上 grouped MXFP8 内核，与本卡 wgrad 4wave 无关，跨内核参考）**mxfp8 真实性能数字（安全实现，供参考）**：gpt_oss-20B Expert shape(B=4 M=2048 N=5760 K=2880) reg notes 修后 grouped fwd/wgrad 达 **106/0/0/0**；Perf(B=16 M=2048 N=4096 K=7168) **Fwd 1435 / Dgrad 1416 / Wgrad 2013 TFLOPS**，SNR **28.23 / 28.23 / 28.08 dB**。grouped MoE bench 第二类修前后基本持平：Fwd 941.02→934.67(**−0.67%**)、Bwd 1109.91→1099.66(**−0.92%**)。

❌ 别再试：靠 SNR 判断 race 是否存在。SNR=53-54 才暴露、正常 SNR 完全掩盖 1/30000 级 bit-flip，采样量不到 30000+ 次时假阴性。
❌ 别再试：`PT_RACE_VM=1` + 2 池 3buf 的跨 barrier G2S 写。m4096 实测 SNR 掉到 53-54，约 0.17% bit-flip，速度收益是以腐蚀输出为代价的。

---
来源: remote-sync/SKILL.md, pr-merge-gate/SKILL.md, 08-deadends.md, optimize-handoff/SKILL.md, optimize-loop.md, flydsl-fp8-gemm-tuning/SKILL.md, flydsl-fp8-gemm-tuning/07-benchmarking.md, fp8-gemm-bench/SKILL.md, 04-ceiling-analysis.md, flydsl-fp8-gemm-results/SKILL.md, 07-benchmarking.md, 10-grouped-wgrad-4wave-3buf.md, gpu-fleet-tuning/SKILL.md, 14-fused-preshuffle-e2e.md, mxfp8-grouped-gg-devloop/SKILL.md, project_mxfp8_wholeloop_port.md, 05-dead-ends.md, project_mxfp4_epilogue_store.md, verify-accuracy/SKILL.md, project_wgrad_occ_feed_bound.md, gfx950-vmcnt-race-debug/SKILL.md
