# 测量噪声地板：run-to-run ~5% / DVFS 功耗受限 / 多轮 interleaved 才可信

> 类别: 踩过的坑 · 主题标签: measurement-noise, dvfs, interleaved-ab, regression-gate

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
- `quick_test_bench.py` 用手写 `time.perf_counter` 循环；full `bench_<op>_turbo.py` 用 `torch.utils.benchmark.Timer` → **不同 timing 引擎，绝不能跨这两者比绝对数字**。
- `quick_test_bench.py` 是 round-to-round 每-shape 回归门的**权威源**（BASELINE 和每个 VALIDATE 都跑它，比 `--summary-csv`）。
- full bench 只用于：(i) 从全 shape 集挑 `representative_shapes`；(ii) campaign 后最终验收。

**--summary-csv schema 必须逐字节冻结**
- GEMM 类列（严格顺序）：`label,B,M,N,K,Check,Forward TFLOPS,Forward TFLOPS_stddev,Backward TFLOPS,Backward TFLOPS_stddev,Forward Time (ms),Backward Time (ms),out_snr,da_snr,db_snr`。
- `Check` 用 `PASS/FAIL`；`*_stddev` 是该 metric 单位下的**绝对** stddev。
- ❌ 别再试 改列名/调列序：任何 schema drift 会**静默关掉**每-shape 回归门（gate 按精确列名 key）。
- 非 GEMM op：保留 `label/Check/*_stddev` 约定，只换掉 GEMM 专用列。

---
来源: remote-sync/SKILL.md, pr-merge-gate/SKILL.md, 08-deadends.md, optimize-handoff/SKILL.md, optimize-loop.md, flydsl-fp8-gemm-tuning/SKILL.md
