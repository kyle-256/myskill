# 跨进程 A/B 是幻觉:同进程交错 round-robin 取 min 才可信

> 类别: 方法论 · 主题标签: 测量噪声, round-robin, bisect, 瓶颈定位

## 核心:跨进程 A/B 全是幻觉
- **跨进程/跨 session 的 A/B 对比不可信**——GPU boost/throttle 时钟漂移 ~1.5-3%,大于绝大多数微调赢面。曾两次测出 **+1.66%** 和 **-2.2%** 完全矛盾的结果,都是漂移 artifact。
- **唯一可信 = 同进程内交错(round-robin)**:一个进程里对多 config 用**固定 swizzle** 编译好、交错计时、取 **min(reps30)**。漂移对所有 config 同相,交错采样使其抵消。

## 测量纪律(每次改动)
- 每改一次 env 先 `rm -rf /root/.flydsl/cache`。
- 每形状跑 3 轮,取跨轮最佳 **min-TF**(min-TF = 峰值时钟事件,抗节流)。
- 每轮**同 session 现测对照基线**(如 intrinsic `RAWAGPR=0`)对标,不用历史数字。
- 可选:每 K `rm cache` + `sleep8` 冷却。

## 可信测量方法排序(噪声量级)
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

## 瓶颈定位:同 FLOPs 不同 shape 对照
- 最快的定位工具 = **同 FLOPs 不同 shape 对照**(big-N vs big-K,同 kernel)。big-N 慢就拿 big-K 对照 profile,差异指标直接点出瓶颈。
- 例:L2 **51 vs 66** 直接指出瓶颈是 L2 复用。

## 归因技巧
- **gated `NOSTORE` env**(epilogue store 直接 return):`full 时间 − nostore 时间 = 暴露的 store 成本`。
- **斜率/截距回归**(us/K-block):斜率=稳态 compute,截距=固定 prologue/epilogue 开销,二者分离。

## 长 K 的 HW 级非确定(别用来判 race)
- K28672 在高负载机上有 **HW 级间歇非确定**:官方 intrinsic K28672 `DETRUNS=15` 也出 det 15233/152750;raw-baseline 出 det 39926。幅度比自研改动还大。
- K8192/K16384 三者均 **det0 干净**。非确定阈值在 **K16384~K28672 之间**,纯 HW/长-kernel 效应,与 kernel 逻辑无关。
- 教训:分辨 <1% 真信号必须用**短 K(K8192/K16384)做干净 det0 + 交错对标**,不能用 K28672 的 det 指标判 race。

## bisect 性能回归纪律
- 每个 commit 跑 **3 次取中位数**。
- 回归 <5% 警告可能是噪声;回归 <0% 说明 good/bad 搞反。
- bad 阈值 = `good + (bad-good)*0.3`(容噪声和渐变);接近阈值 10% 内跑 **5 次**。
- build 失败跳到相邻 commit,连续 3 个失败问用户。
- **全 good 时回归可能是环境性的**(driver/library/硬件热节流),重跑 bad commit 确认。
- 默认 **first-parent-only**,跳过 merge 内部。

---
来源: flydsl-fp8-gemm-tuning/SKILL.md, agpr_rawasm_progress.md, agpr_phase5_ldsr.md, project_mxfp4_epilogue_store.md, agpr_phase5_mono.md, 07-benchmark.md, bisect-perf-regression/SKILL.md
