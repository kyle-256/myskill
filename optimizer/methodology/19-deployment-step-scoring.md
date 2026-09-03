# 19 — 拿真实部署的一步(decode step)当分数:热/冷尺子、取最小、部署权重

> 类别: 方法论 · 主题标签: deployment-scoring, cold-vs-hot, decode-step, sglang, GLM-5.2, aiter,
> min-of-N, contention, profile_decode, trace_step, ruler-failure

**场景**:优化的 kernel 服务于一个**在线推理部署**(sglang / vLLM 起模型、每步回放一张 cuda graph)。
问题不是"kernel 快不快",是"**这一步的墙钟短没短**"。这张卡记的是 2026-08-31~09-01 那场
aiter bf16 a16w16 skinny GEMM 的全部尺子教训 —— **三把尺子连续排错序**,每一把都在微基准上赢、
在部署上输。

---

## ★★★ 头号教训:重放同一个权重 = 权重在 cache 里,和部署根本不是一回事

decode 一步会摸**每个权重各一次**,而且在该层再次执行之前已经流过几十 GB;所以**部署里 B 永远是冷的**。
微基准如果反复回放同一个矩阵,量的是一个模型永远遇不到的 cache 状态。

同一个 kernel、同一个形状 `(8, 6144, 16384)`,两种量法:

| 量法 | 读数 |
|---|---|
| 热(单权重反复回放) | **30.7 us** |
| 冷(真实 step 里 trace 出来) | **45.1 us** |

**47% 的差,而且排序会翻。** hipBLASLt 的 kernel 对 cache 状态没这么敏感,于是它**输掉热尺子、赢下部署**。
第一版按热尺子选型 ship 出去,真机 decode **慢了 3.3%**(22.35 → 23.15 ms),kernel 一点毛病没有,**尺子错了**。

⇒ **任何微基准必须让权重在 ≥512 MB 的副本上轮转**,否则它的排序没有意义。

## ★★ 冷微基准也**不是**分数:部署权重才是

56 个形状均摊求和读 **+10%**,而按部署权重算的总时间是 **−1%** —— 因为 decode 一步只发四个形状,
调用数 78/78/78/21。**用冷微基准做候选筛选,用 step 时间做判决。**

同理:**比值的几何平均**更糟 —— 在 56 个形状上读 1.27,而总时间落后 10%(赢一堆便宜形状掩盖输几个贵的)。
要用就用**时间求和**,不要用比值平均。

## ★★★ 节点被抢时,估计量要用 min 不是 median

同节点有别的 campaign 时,**抢占只会让时间变长,不会变短** ⇒ 多次取**最小**才是无抢占真值,
中位/均值会把被抢的读数算进分数。

真实翻车:原 bench 跑 2 次取"中位"(两个数的中位 = 平均),同一 commit 四次读数
**22.70 / 23.89 / 24.57 / 28.52 ms(25% 跨度)**,基线被记成 26.55 ms(虚高 15%)。
KEPT 门槛是 0.5% —— 于是**第一个候选随便撞上一次干净读数就 +17% 被 KEPT**,best 棘轮到离群值,整晚在噪声上打转。

改成 **跑 4 次取最小**之后,同一 commit 四次读数变成 **21.789/21.798/21.816/21.834(spread 0.21%)**。
底层信号本来就极稳,问题只在估计量。

**min 的失败方向还是安全的**:全被抢时只会低估候选(误 REVERT),不会误 KEEP。

```python
# 好:被抢只会让某几次变慢,取最小把它们丢掉
step = min(steps)          # 跑满 4 个有效读数
# 坏:两次取"中位"= 平均,一次被抢就毁掉整轮判决
step = statistics.median(steps[:2])
```

## ★ 跨尺子的数字不能相减

同一份代码在不同尺子/不同时段下的读数差好几个百分点。收官**必须同一把尺子、同一台空节点、同一段时间**
把 before/after 都重测一遍。本场例子:上游 stock 早先记的是 22.343 ms(有邻居、单次 ABBA),
干净节点 min-of-4 重测是 **22.631 ms** —— 1.3% 的差,足以让"我们快多少"这个数错。

## ★★★ aiter GEMM tuner 的选型口径会把 flydsl vs hipBLASLt 判反(2026-09-03)

`csrc/gemm_a16w16/gemm_a16w16_tune.py` 用 `run_perftest(gemm_a16w16, ...)` 计时 —— 量的是
**整个 python 包装**,不是 kernel。`gemm_a16w16` 的 dispatch(查表 / kernel name 解析 / JIT lookup)
对 **flydsl 比 hipBLASLt 慢约 7 us**,而 decode 走 cuda graph replay 时这段**根本不执行**。

实证(GLM-5.2 o_proj `6144x16384 M=1`,干净卡):

| 口径 | flydsl | hipBLASLt | 判决 |
|---|---|---|---|
| tuner(`gemm_a16w16` 整包装) | 37.30 | 34.32 | flydsl 输 8.7% |
| **直调 kernel(同为 `run_perftest`)** | **30.10** | **32.34** | **flydsl 赢 6.9%** |

⇒ 那批 tuned CSV 里 `6144x16384` **九个 M 全选 hipBLASLt**,是口径造成的假象。
★**判后端胜负必须直调 kernel**;`gemm_a16w16` 的读数只能用来比"同后端不同 config"。

### ★★ 冷/热尺子:B 权重 201 MB vs MALL 256 MB

`aiter/test_common.py: perftest` 用 **`rotate_args` 轮换多份输入**强制冷缓存
(`cache_size = min(L2_cache_size*64*128, ...)`,再按 inputSize 算轮换份数)。
自己写 `for _ in range(N): fn()` 连发、或用 cuda graph replay,**同一份 B 会留在 256 MB MALL 里**
—— 而这个形状的 B 正好 201 MB,整块装得下,读数直接失真。
★**所有 GEMM 计时一律走 `run_perftest`**(用户硬令);判增益再叠**回文次序 + 同二进制对照臂**
(本场地板 0.13%,`xcd_band=2` 的 −2.83% 是地板的 21 倍,才敢说成立)。

### ★★★ 四个编译轴从来没进过 tuner 的搜索空间

`get_flydsl_splitk_hgemm_kernels` 只 `product(tile_m, tile_n, tile_k, stages, KERNEL_CONFIG_VARIANTS)`
再套 `split_k`;**`xcd_band` / `k_rot` / `m_rows` / `b_cpol` 一个都不枚举**,全走默认
`xb=1,kr=0,mr=0,cp=0`;`run_flydsl_gemm_bf16` 也不传这四个参数。
⇒ decode campaign 赢来 −6.7% 的那四个轴,**tuner 结构上搜不到**,只有 CSV 的 kernelName
带 `_xb/_kr/_mr/_cp` 后缀时才生效。已接 `xcd_band`(`HGEMM_XCD_BAND_OPTIONS=(1,2)`,
名字加 `_xb2`,catalog 6571→13142;tuner 侧 `run_flydsl_gemm_bf16` 传 `xcd_band`)。

### 判负的两个"想当然"(别再试)

- **照抄 hipBLASLt 的窄 tile**:它赢在 `MT16x16` → `6144/16=384` 个 WG 铺满 256 CU 不切 K,
  而 flydsl `tile_n` 最小 64 → 96 个 tile,被迫 `split_k=2`。给 flydsl 补 `tile_n=16/32` 后
  干净卡实测 42.39 / 38.45 us,**都不如原有的 64(30.10)**。已回滚。
- **`b_to_lds=False`**(对应 hipBLASLt 的 `LDSB0`):M=1 时 B 是全部流量、LDS 无复用可摊,
  看似该关,实测**全面更差**且候选成功率 79.6%→14%。已回滚。

---

## GLM-5.2 decode 的两把尺子(SIKL,`sikl/benchmarks/models/glm52/`)

两个脚本量的**不是同一个东西**,别混用:

| 脚本 | 量什么 | 用途 |
|---|---|---|
| `trace_step.py` | **整步装一张 graph**,量总时间 | **计分**。唯一能写进结论的数 |
| `profile_decode.py` | **每个 module 各装一张 graph** 单独量,父项减子项得 `<rest>` | 归因/看分解。**总和是真的,单行拆分是歪的** |

`profile_decode` 的每个子项在自己的 graph 里反复回放 ⇒ 权重是热的、偏快,差额全被挤进 `<rest>`。
**`<rest>` 是减法行,不是测量行** —— 曾经据此误判"attn.`<rest>` 变差 1.98 ms",其实是子项偏低顶上去的假象。

固定参数(改一个字都不可比):

```
--tp-size 8 --enable-dp-attention --dp-size 8 --kv-cache-dtype fp8_e4m3
--enable-aiter-allreduce-fusion --enable-fused-qk-norm-rope
--dsa-prefill-backend tilelang --dsa-decode-backend tilelang
--ISL 100000 --CONC 64 --reps 20
```

★ `trace_step.py` 里必须在 capture 之前先 `mr.forward(fb)`,否则 DSA 后端报
`'DeepseekSparseAttnBackend' object has no attribute 'forward_metadata'`。
★ `--trace-reps 1` 时 ROCm profiler 一个 CUDA 事件都不吐(`kernels 0`),要 kernel 分解得用 `--trace-reps 3`。
★ ⚠ `trace_step.py` 的 kernel 表把 `device_time_total` 和 `count` **都除了 `trace_reps`,这个除法是多余的**
(profiler 只记了一次回放)。**每次调用的 us 不受影响,每步总量小 3 倍**。识别方法:看调用数 ——
78 层的模型 hgemm 应该是 x78,表里显示 x26 就是被除了 3。

## ★★ 一步里到底有哪些 GEMM(GLM-5.2 / TP8+DP8 / CONC 64)

只有四个形状进 `gemm_a16w16`,M 全是 8:

| 形状 (M,N,K) | 调用/步 | 角色 |
|---|---|---|
| (8, 6144, 16384) | 78 | `o_proj` ← 最贵,B 有 201 MB,近乎纯流式 |
| (8, 16384, 2048) | 78 | `q_b_proj` |
| (8, 2624, 6144) | 78 | `fused_qkv_a_proj` |
| (8, 4096, 2048) | 21 | DSA indexer |

⇒ **M ≥ 256 在这个 context 长度下根本到不了**(ISL 100k 时 KV pool 把并发卡在 252,DP8 ⇒ 单 rank ≤ ~32 行)。
拿 M=512/1024 调优对这个部署毫无意义。

---

## ★★ 运行期的四个坑(全都真的把一整轮/一晚上毁过)

**1. `pkill -f trace_step.py` 杀不干净。** 八个 worker 的命令行是
`python -c from multiprocessing.spawn import ...`,**匹配不上**,父进程一死它们全变孤儿(PPID=1)挂在卡上。
后果:下一次启动自检报 `The memory capacity is unbalanced`,或者**卡死在 NCCL 建通信子**(日志停在
`sglang is using nccl==2.27.7`、显存全 0)。清理要**同时按显式 PID 杀 `multiprocessing.spawn` 的 worker**。

**2. `/dev/shm` 会攒死。** 反复 `kill -9` 之后攒了 **4533 个残留段**(469 个 `sgl_shm_mq` 死队列 + 73 个
`sglang_loads`),`lsof` 显示零进程在用。新进程按名字 attach 一个没有对端的死队列会永久阻塞。
清理:`find /dev/shm -maxdepth 1 -type f ! -name 'rocm_smi_*' -delete`。

**3. 别的容器的孤儿会让你以为是自己的 bug。** 曾经四次 bench 全部卡死在 NCCL,**容器重启也没用、
换成已知能跑的旧 commit 对照也一样卡** —— 真凶是**另一个容器**里 17 个 KFD python 孤儿(最老 3.8 天,
一个正吃 936W)。宿主上 `kill` 会 EPERM(容器内是 root、ssh 用户不是),**必须 `docker exec <那个容器>` 进去杀**。
诊断动作:`rocm-smi --showpids` 拿 PID → `cat /proc/<pid>/cgroup` 找容器 → `docker ps` 对名字。

**4. `docker exec ... bash -lc 'pkill/grep ...'` 会杀掉自己。** 命令行里含有关键字 ⇒ 匹配到自身的 shell。
用 `[t]race_step` 这种写法,或者干脆按显式 PID 杀。

---

## 收官清单(这场用到的完整口径)

1. 节点清空:`rocm-smi --showpids` 必须是 **0 个**,八卡功耗回落到 idle(~230W,不是 900W)。
2. `/dev/shm` 清一遍,确认没有 `trace_step` / `multiprocessing.spawn` 残留。
3. **同一把尺子**在**同一段时间**测 before / after,各 4 次取最小,报 spread。
4. 微基准(冷、轮转 ≥512 MB)作为旁证:报总时间、赢参照实现的形状数、**逐形状回退清单**。
5. ⚠ **冷微基准单遍噪声 5–6%** —— 任何"这行退化了要回退"的决定**必须两遍都复现**。
   本场靠这条筛掉 6 个假阳性(`64x6144x16384` 第一遍 +15.2%、第二遍 +1.7%)。

---

## ★★★ 尺子必须走部署真正走的那条 kernel(gptoss 整步 campaign,09-02 尸检)

一场 20 轮的 campaign 用「trace 加权的模型化整步」计分,收官 bench 说 **−5.2%**,
真训练只有 **−0.84%**。逐 commit 在同形状同机器上复测,**只差一个 `num_cu` 参数**:

| commit | 部署 grid(`num_cu=None`) | bench grid(`num_cu=256`) |
|---|---|---|
| base | 1.0000 | 1.0000 |
| r2 | **1.0107** | 0.9945 |
| r5 | **1.0156** | 0.9782 |
| r9 | **1.0165** | **0.9656** |

**每一轮被 bench 判「赢」的改动,在部署真走的那条体上都是「输」。**
根因:部署的调用点(`experts.py` 的 `grouped_mlp_fp8`)**一个 `num_cu` 都不传**
→ 非 persistent 体;bench 传 `num_cu=256` → persistent 体。两条是不同的 kernel。

### 开场必做的三件事
1. **把部署的调用点读出来,逐个参数抄进 bench**(不是抄形状,是抄**参数**)。
   一个默认值就能把你送到另一条 kernel 上。
2. **grep bench 自己的注释里有没有「as X would run it」这类假设句**。
   那场 22% 的计分面注释就明写着「部署跑的是 TE norm + hipBLASLt,接到 turbo 才归我们」
   —— 也就是说那一整块改了等于没改,而这行字是作者(我)自己写的。
3. **别把「kernel 省下的时间」当「整步省下的时间」**。实测兑现率 81%(计算 −7.78 →
   墙钟 −6.31),不是 1:1,也不是「NCCL 吞一半」。

### ★★ 即使 grid 对上了,孤立单算子探针仍不能预测 e2e
把 grouped GEMM 退回 campaign 之前:单卡部署 grid 快 0.5%,**真训练慢 0.71%**。
⇒ 单卡探针只配做**筛选**,判定一律回到真步。

### ★★ 拿真训练当 bench 的四个实现坑(gptoss/Primus)
- **每步时间在日志里拿不到**:Primus 的 training_log 用 `print_rank_last` 打到 rank-last,
  转发到 rank0 是 **DEBUG 级**(默认 INFO 吞掉),而 primus-cli 硬编 `--local-ranks-filter 0`。
  → 走 **profiler trace 的 ProfilerStep**。tensorboard 有 `lm loss` 但没有 iteration-time。
- **冷 JIT 缓存下 autotune 会落进 profile 窗口**:实测一次 **2423 ms**(真值 737),
  而且**所有**被 profile 的步一起被污染,「≤最快步 1.10×」这种相对过滤器识别不出来。
  → 必须加**绝对**守卫(超过基线 1.25× 判为编译时间)+ 重试。
- **loss 不能当紧门**:同一份代码三次 9.841 / 10.136 / 9.822,**抖 3.2%**。
  2% 的门稳定误杀 1/3 的好候选。loss 只配做发散检查,正确性要**算子级前置门**
  (对 fp32 的 SNR + **部署宽形状**的逐字节确定性)。
- **读 trace 两坑**:`ProfilerStep` **每步发两次**(按窗口重叠折叠,别数事件数);
  **冷臂第一个 profiled step 带 JIT**(1189 ms vs 稳态 723.6),平均进去会让 NCCL
  动得比被测效应还大。
