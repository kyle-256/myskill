# 21 — 推理服务端 e2e A/B：口径、可比性、以及七道非要不可的守门量

> 适用：起 sglang server 跑 GLM-5.2 / MoE 类模型的端到端对照（AgentX agentic replay、
> 自建 bench、榜单复现）。**与 17（训练 trace 验 kernel 有没有跑）、19（kernel 级部署打分）
> 是不同层**：这张卡讲的是「请求级指标能不能用来判一个改动的去留」。

## ★★★★★★ 第一件事：确认口径，而不是照抄脚本默认值

同一个 repo 里会并存**两套完全不同的口径**，脚本默认值往往是前者：

| | 已提交榜单 | **优化目标**（用户裁定） |
|---|---|---|
| TP / EP | 8 / 1 | **4 / 4** |
| 并发 | 1, 2, 4 | **8, 10, 14** |
| **verify 行数 = 6 × 并发** | 6 / 12 / 24 | **48 / 60 / 84** |
| 每卡专家数 | 256 | 64 |

`rocm-llm-bench/glm52/launch_sglang.sh` 默认 `TP=8 EP=1 DP=1 CONC=1`，
`agentx_bench.sh` 默认 `CONC=1`。**照抄默认 = 自动落进榜单口径。**

★ **一眼判据：verify 行数**。起跑后看日志里的行数（=6×并发），48/60/84 才是 TP4/EP4 口径。
我按默认值起过一整个矩阵，跑了一个点才被用户拦下（"你是 tp4ep4 c8 c10 c14 啊"）。
**写下 TP/EP/并发三个数字、和裁定逐个对一遍，再去读脚本默认值。**

## ★★★★★★ agentic replay 是闭环：能不能比，取决于改动落在哪条路径

AgentX 的 replay 用真实 agent 轨迹，**模型自己的输出决定后续轮次**。所以一个改动如果
改变了输出 token，两臂重放的请求集就会分叉，**吞吐/延迟都不再可比**。

实测分叉（同一固定种子 `--random-seed 42`）：

| | base | dsa0（#38583 融合 indexer） |
|---|---|---|
| 请求总数 | 1488 | 1123 |
| 每请求输出 token 均值 | 331.7 | **1068.1（3.2×）** |
| 输出吞吐 | 129.24 | 306.30（**看着 +137%，其实是负载不同**）|

**判断会不会分叉，看改动落在哪条路径，不是笼统看"是不是有损量化"：**

| 改动落点 | 输出是否保序 | 可比性 |
|---|---|---|
| **kernel 内部、逐位一致**（tile 配置、launch 合并） | ✅ 完全相同 | **严格可比**，请求数应逐个对上 |
| **draft / MTP 侧**（投机解码的提议者） | ✅ 保序 | **可比**。EAGLE 是 draft 提议 / target 验证，**被接受的 token 来自 target** ⇒ draft 精度只影响接受率与速度 |
| **target 侧**（主干权重、attention 选 key、indexer top-k） | ❌ 会变 | **不可比**，轨迹会分叉 |

★ 我曾把"draft 层 bf16→MXFP4"说成"有损⇒输出分叉⇒只能近似对照"，**方向说反了**：
它落在 draft 侧且有精确验证，输出是保序的（实测请求数 1490 vs 1483，差 0.5%）。

★ **请求总数就是现成的分叉探测器**：两臂差 >2% 就别比吞吐了。

## ★★★★★★ 这把尺子的噪声底（实测，不是估计）

**同一份代码跑三次**（因为一个 bug，三臂其实都在跑镜像自带的 sglang）：

| 指标 | 三次读数 | 极差 |
|---|---|---|
| 输出吞吐 | 129.24 / 306.30 / 306.84 | **2.4 倍** |
| ITL p50 | — | 0.51% |
| ITL p90 | — | 2.17% |
| TTFT p90 | — | 3.53% |

⇒ **agentic replay 的吞吐数不能做 A/B**（时间窗固定 + 闭环放大调度抖动）；
**延迟分位数可用，但分辨率只有 ~3%**。小于 3% 的效应在这把尺子上不可分辨，
要么换同进程配对尺子（见 methodology/01、[[reference_paired_inprocess_ruler]]），
要么做重复读定当天的噪声底。

## ★★★★★★ 七道守门量（每一道都是被真实事故逼出来的）

| # | 守门量 | 不设会怎样 |
|---|---|---|
| 1 | **server 进程的 `PYTHONPATH` 必须含被测树**（读 `/proc/<pid>/environ`） | server 静默 import 镜像自带的 sglang，三个"成功"的点全在测别人的代码 |
| 2 | **readiness 要自证**：本轮自己的 server 日志出现 `Load weight begin` **且**端口应答 | 上一轮正在排空的 server 还在监听，curl 秒过，实际连的是旧进程 |
| 3 | **功能 flag 要有物理计数**（如转换层数 `converted_layers`，应 = TP rank 数） | flag 没生效也能跑完一小时，读数看着完全正常 |
| 4 | **投递前清臂日志** | 上一轮残留的 `ARM ... DONE` 被当成本轮完成，点在 30 秒内"跑完" |
| 5 | **停 server 按所有权（带 PID 的锁），并轮询确认到进程数=0** | 被取消的旧臂睡满轮询后醒来，把下一轮正在跑的 server 杀掉 |
| 6 | **精度测量的 server environ 里 `SIMULATE_ACC` 计数必须为 0** | 接受率被钉成模拟值，GSM8K 量的是模拟器（实测 0.111 vs 0.930） |
| 7 | **请求总数两臂对齐**（见上一节） | 把负载差异当成性能收益 |

## 起 server 的正确姿势

**环境变量走 `docker exec -e`，不要写成引号串里的 shell 赋值。**

```bash
docker exec -d \
  -e HIP_VISIBLE_DEVICES=0,1,4,5 -e PYTHONPATH="$TREE/python" \
  -e PYTHONNOUSERSITE=1 -e SGLANG_AITER_BF16_MOE_MXFP4="${FLAG:-0}" \
  -w / "$C" bash -c "exec /opt/venv/bin/python -m sglang.launch_server ... > $SRV 2>&1"
```

理由见 pitfalls/15 §注释截断：那串命令要同时活过**本脚本的解析器**和**容器里的解析器**，
一旦出错就是「server 安静地少了几个环境变量」。`-e` 由 docker 解析，没有 shell 参与。

★ **server 日志文件名带时间戳**：它由容器里的 root 创建，宿主脚本**截断不了**
（Permission denied），而残留日志会让守门量读到上一轮的结果。

## upstream / 镜像差异（起不来时先查这几条）

| 症状 | 原因 |
|---|---|
| `argument --cuda-graph-max-bs: ambiguous` 或 exit 2 | upstream 拆成了 `-decode`/`-prefill`；旧名成了**有歧义前缀**，`--help` 里 grep 得到所以存在性检查会误判 |
| `--dsa-topk-backend aiter: invalid choice` | upstream 只有 `{sgl-kernel,torch,flashinfer}`，`aiter` 是 fork 专有 |
| `--dsa-decode-backend flydsl: invalid choice` | 同上，flydsl 路径是 fork 里加的 |
| 启动即 `ImportError: sgl_kernel.kvcacheio.get_device_accessible_ptr is missing` | **hicache 路径**要的符号，镜像的 sglang-kernel 没有 ⇒ 关掉 `--enable-hierarchical-cache`（全臂一致则对照仍成立，但绝对值不对标榜单） |
| ROCm 上 EAGLE 建树崩 `Device value [rocm:1] not in the allowed options: [cuda]` | upstream #40033 把投机 kernel 挪到 JIT 后 `tree.cuh` 写死 `kDLCUDA`，改 3 行即修 |
| `DistStoreError: Timed out ... 1/4 clients joined` | **僵尸 rank 占着显存/端口**。`docker restart` 无效，只能 `docker rm -f` + `docker run` 重建容器 |

**秒级参数校验**：把完整 argv 后面加 `--help` 丢给 argparse，rc=0 就是全部参数被接受，
几秒出结果，不用等 8 分钟的 server 启动去试错。

## 验参数前先 grep，别直接加

给 `REPLAY_CMD` 加 `--random-seed` 之前没 grep，结果和上游已有的 `--random-seed 42`
撞车，`Parameter --random-seed specified multiple times`，两个点各白跑 5 分钟。
**改别人的 harness 脚本前先 `grep -n <你要加的参数>`。**

## ★★★★★★ 报数之前先确认：你用的是不是榜单那两个轴

Pareto 图（所有人都在用的那张）的两个轴，JSON 里的字段是：

| 轴 | 字段 | c8 base 实测 |
|---|---|---|
| **Y = (输入+输出) tok/s / 物理 GPU** | `request_metrics/throughput/per_gpu/total_tput_tps` | 15678.0 |
| **X = P90 interactivity (tok/s/user)** | `request_metrics/latency/intvty/p90` | 89.79 |

★ `intvty` 的百分位是**反向**的（越高越好）：p50=140.81 > p90=89.79。别当成延迟读。

**不要用 `throughput/output/tokens_per_second`**（整机纯输出速率）：它既没除 GPU 数也不含输入，
而这个 agentic 负载里 **输出只占总 token 的 0.21%**（62712 = 输入 62577 + 输出 134）。
我用它报过一轮，**方向是反的**：

| 臂 | 纯输出速率（错的口径） | Y per-GPU total | X ITV p90 |
|---|---|---|---|
| mxfp4 vs base | **+6.09%** | **−2.42%** | **−3.82%** |
| dsa0 vs base | −1.58% | +4.17% | **+17.39%** |

## ★★★★★ 闭环让两个轴同向，但 X 的杠杆大得多

Y 虽然 99.8% 是输入 token，却**不是对 decode 不敏感**：decode 变快 ⇒ 请求完成得快 ⇒
固定窗口内重放更多轨迹 ⇒ 输入 token 跟着涨。实测 #38583：X +17.39% 带动 Y +4.17%。

⇒ **判 decode 侧优化看 X（ITV p90），Y 是顺带涨的**；只看 Y 会把一个 17% 的改善读成 4%。
★ 这也是 `[[reference_glm52_decode_step_breakdown]]` 里"对账该比 Median ITL"那条的由来。

## ★★★★★★ Y 轴的构成：它几乎全是「重放了多长的对话历史」

```
per-GPU total = (ISL 均值 × 完成请求数 ÷ 窗口秒数) ÷ GPU 数
实测 base c8 = (160,740 × 1403 ÷ 3604) ÷ 4 = 15,678
```

两件事要同时记住：

1. **输出只占 0.21%**，所以这个轴 99.8% 是输入侧；
2. **输入里 ~97% 是前缀缓存命中**（日志 `prefix_cache_hit=96.9%`），**不是真算出来的 token**。

⇒ 这个轴实质上是「窗口内重放了多长的对话历史 ÷ GPU 数」。

**ISL 是极度长尾的**：实测 p50=108k、p75=183k、**p90=384k**、p95=503k，均值被拉到 160k。
**少数超长轨迹就能把这个数抬三成** —— 这也是「同一份代码三次读出 129/306/307」那个
2.4 倍"噪声"的真正来源：**不是机器抖动，是重放到的轨迹长度不同**。

⇒ 想让 Y 轴可比，两臂的 **ISL 均值/分位必须对齐**，不只是请求数对齐。

## ★★★★★★ 绝对值不可比：三项配置差就能吃掉 30%

我的 c8 base 读 15,678，而公开 Pareto 图上同为 c8/TP4-EP4 的点约 11.8k。**不是跑得快，是口径不同**：

| | 公开图（0907 曲线） | 我的 |
|---|---|---|
| sglang | `402df1e4`（他们的 fork 分支） | upstream `7ccbf5f` |
| aiter | `2c71811b32` | 镜像自带 |
| **hicache** | 开 | **关**（镜像 sglang-kernel 缺 `kvcacheio.get_device_accessible_ptr`，开着必崩） |

hicache 直接决定 KV 池容量与前缀命中率，而 Y 轴几乎全由"命中的输入 token"构成
⇒ **这一项就足以解释三成的绝对值差**。

⇒ **要出能放进公开图的点，必须先对齐配置**（修 hicache + 用同一个 fork commit 当 base），
否则扫再多并发点也只是自成一体的一套数。臂间百分比仍然有效，绝对值不要往图上放。

### ★★★★★★ 09-19 更新：上面这件事做成了，而且「hicache 开着必崩」是镜像特定的

**`kyle_cam` 的 `/opt/venv` 有那个符号，hicache 开着跑通了两臂**（`server_info.
enable_hierarchical_cache = True`，日志里 `hicache_attached=True`、
`cpu_kv_usage` 从 6.6% 一路爬到 38%）。所以上表那行是**某个镜像的**事实，不是普遍结论——
换环境要重测，别照抄「关掉」。

对齐配置后的同机配对（`crsuse2-m2m-043`，榜单配方逐条照抄，TP8/EP1，c1，3600s，
aiter 干净 `2c71811b32`）：

| | X `intvty/p90` | Y `per_gpu/total_tput_tps` |
|---|---|---|
| 基线 `402df1e1e4`（图上那个 sglang） | 253.28 | 2155.72 |
| 我们的分支 | 264.57 | 2199.40 |
| | **+4.46%** | **+2.03%** |

★ **尺子自检**：在**另一台机器**上复现同一份基线代码，落点 (253.28, 2155.72) 对公开图的
`c1/TP8` ≈ (260, 2.1k) 只差 x −2.6% / y +2.7% ⇒ 配方搭对了。**这一步值得每次都做**：
它把"我们比图上好"从口头断言变成可验证的东西。

### ★★★★★★ 600s 不够：前缀缓存要 ~30 分钟才走完爬坡

同一批代码，600s 与 3600s 给出**符号相反**的结论：

| 窗口 | c1 | c2 |
|---|---|---|
| 600s（hicache 关） | x +2.03% | x **−6.19%** |
| 3600s（hicache 开，配方） | x **+4.46%** | 未跑 |

根因不是噪声，是**窗口整个落在爬坡段**：`prefix_cache_hit` 从 99.6% 降到 95.5%、
`tput_in` 从 31k 降到 16k，要 **30 分钟**才进平台期。600s 只看到前期"高命中的便宜 token"。

**指纹**：600s 那批点测出过 **c1 的 Y 高于 c2**（3,274 vs 2,446），而图上是 c2 高于 c1。
**并发低的点吞吐反而更高 = 窗口没出爬坡段**，看到这个形状就别再往下算了。

### ★ hicache 开着的代价：拆机把 HBM 漏在驱动里

跑完 hicache-ON 的臂、server 正常停止后，**6/8 张卡仍占 96–98%**，而容器里 0 个 python、
0 个僵尸、PID 1 是 `docker-init`、**`/sys/class/kfd/kfd/proc/` 是空的** ⇒ 没有主人的显存。
`docker restart` 无效；`rocm-smi --gpureset` **要 sudo**（Crusoe 上普通用户做不了，
卡在密码提示，不是报错）。**等约 20 分钟会逐张自己释放，最后一次性归零**。
⇒ 排 3600s 的点要给排空留 20 分钟。

### ★★★★★ 报增益前先查「基线到你的分支之间都是谁的 commit」

差值里混进同事的工作是这条线反复出现的错误。两次实例：

* `--dsa-decode-backend flydsl` 是**同事的 PR #31**，不是我们的。开着它去对公开图那条
  tilelang 的线，等于把他的收益算进我们的增益 ⇒ 对图必须用 **tilelang**。
* 基线 `402df1e1e4` 到我们 fork base 之间有 5 笔别人的 commit，**逐个 diff 确认全是格式改动**
  （换行重排、import 顺序、去引号、codespell 词表）之后，这个差值才敢说"只含我们的"。

**做法**：`git log --format="%h | %an | %s" <base>..<yours>`，非你的 commit 逐个
`git diff --ignore-all-space` 看是不是零语义。

### ★ 核对配方要对「活进程的 argv」，不是对自己的脚本

grep 自己的启动脚本只能证明"我写的和我想写的一致"。真正的证据是
`tr '\0' ' ' < /proc/<pid>/cmdline` 和 `/proc/<pid>/environ`，逐条对榜单脚本。
我这样做时发现了两处自己漏掉的：**缺 hicache 那五个 flag**、
**缺 `AITER_USE_FLYDSL_MOE_SORTING=1`**（干净 aiter 里它默认是 0，不显式设就走 Opus 后端，
和配方不是一回事）。同时把"刻意偏差"列出来核对——我的只应有 5 条，多一条就是污染。

## 画 Pareto 图需要并发扫描

图上每条线是 **c1/c2/c4/c8/c10/c14 六个点**连起来的（低并发在右下、高并发在左上）。
**单个并发点画不出曲线**，只能落一个散点——想和别人的图对话，就得把并发扫满。
每点 3600s ⇒ 一臂六点 ≈ 7 小时。

## 相关

* pitfalls/15（本轮七个「不报错只给错数」的 bug 与它们的识别指纹）
* methodology/19（kernel 级部署打分：冷/热尺子、取 min）
* methodology/20（先把分母在本机量出来）
* `[[project_glm52_agentx_bench_config]]`（口径裁定原文）、
  `[[reference_glm52_upstream_ab_harness]]`、`[[reference_sglang_e2e_ab_pitfalls]]`

来源：2026-09-18/19 GLM-5.2 TP4/EP4 AgentX 四臂对照（base / mxfp4 / dsa0 / dsa），
14 小时里前 9 小时零有效数据，七个 bug 全部「不报错、只给错数」。
