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

### ★★★ 探针也能拿绝对值 —— 前提是**每个 `synchronize` 跨几十次调用**(2026-09-11 mxfp4 dense 实测)
本项 KB 与多份 campaign goal 反复记过「带 host 空隙的探针一律掉出 boost 档,**绝对 TF/s 不可用,
绝对值只认计分 bench**」(某轮探针低 24-30%,功耗从 827 W 塌到 320-373 W)。**那条结论的真正变量是
批次结构,不是"探针"这个身份。** 把驱动改成 `for _ in range(40): fn()` 之后再 `synchronize`
(GPU 在批与批之间不空转),同一份探针与计分 bench 逐行对齐:

| row | 探针 vs_b200(两趟) | bench vs_b200 | 差 |
|---|---|---|---|
| FC1_fwd | 0.9287 / 0.9241 | 0.9283 | +0.04% / −0.45% |
| FC2_fwd | 0.9523 / 0.9504 | 0.9508 | +0.16% / −0.04% |
| FC1_dgrad | 0.9909 / 0.9892 | 0.9925 | −0.16% / −0.33% |

精度:6 个独立 25 s 窗口 TF/s 散布 **0.36%**,socket power 稳在 1398.3–1400.3 W。
⇒ **价值**:30 s/形状就能拿到**带 sclk/W 的绝对读数**(计分 bench 跑全表要 139 s,而且不报 sclk),
于是 methodology/03 要求的"每个候选单列 sclk/W"第一次变得便宜。**排序仍然只认同进程回文 A/B。**
⚠ 两个会毁掉它的做法:①每批只跑 1 次调用就 `synchronize`;②在计时循环里起 `rocm-smi` 子进程
(要起就另开进程并行采样,按 epoch 时间戳对齐)。

## 测量噪声地板：run-to-run ~5% / DVFS 功耗受限 / 多轮 interleaved 才可信

**噪声地板的量级（不认清就会把噪声当收益放行）**
- 同代码同 shape 连跑 3 遍完整 sweep，数值本身有 **4%** 上下波动（GPU 时钟/温度每次不一样）。只看一次对比就下"回归 X%"不可靠。WHY: 新旧版本波动范围重叠 → 大概率是噪声。
- fp8 GEMM run-to-run 噪声 **~5%**，和很多 lever 增益同量级 → 单次比对无法分辨真收益。
- `timeit.Timer.mean` 在 GPU 温度不稳时被 **±5%** 噪声掩盖；单次 3-trial 对 sub-ms shape 被热/冷态差异 **±8%** 带偏。

**fp8-data-magnitude-ban ★严禁用"喂 zeros/随机字节看 TF 随数据幅度变"论证任何东西**
- 🚫 **禁令(2026-08-09 用户斥"极其不靠谱,以后严禁出现")**:同一 config 把输入 fp8 换 `zeros` / `view(uint8).random_(lo,hi)` 随机字节 / 不同幅度,测出 TF 从 ~3100 掉到 ~2500,**据此下"fp8 DVFS 数据依赖/功耗墙/内核在真实数据下不兑现/kernel-only 只能复现到数据带"结论——全是伪证,禁用**。WHY:zeros 无真实 bit switching→功耗假低→时钟假高→假快;随机字节是 NaN/Inf/denormal 位模式→时序功耗全垃圾。旧卡里"zeros→randn 21-37%""±37% 数据幅度"同源,一并作废,别再引([[project_flydsl_fp8_power_limited]] 亦作废此点)。
- **跨脚本绝对 TFLOPS 方差 ~8-12% 仍成立**(时钟/session 状态不同)→ **跨脚本绝对值不可信,只认同脚本同数据的相对比值 + 多轮均值**。
- 判 config/派发:同一份**固定真实数据**横扫 config 比相对快慢(合法)。判**内核结构性回退/负优化**:旧版 vs 当前版**同一份真实数据、同进程 drift-immune A/B**,别拿数据幅度伪证替代。见 [[feedback_no_data_magnitude_dvfs_test]]。

**★ 别用两次读数去"测"噪声地板 —— 会得到假紧的地板**
- 踩证（mxfp4gg campaign）：一个候选连测两次得 1.50954 / 1.51151（差 **0.13%**），据此认定"噪声地板 0.13%"，
  于是把另一个候选的两次读数判为"与它不重叠 = 真差异"。**再多测一次就崩了**：同一候选三次是
  1.51058 / 1.51587 / 1.51773，散布 **~0.5%**。两次读数落得近只是运气，n=2 估不出分布宽度。
- 单格更野：`wgrad down balanced` 这一格 run-to-run 达 **±2.5%**（1.563/1.606/1.644/1.650/1.657）。
  **多配置 bench 要看聚合量（geomean/median）的稳定性，不要拿单格的涨落下结论。**

**★★ A/B 交错里「谁跑在前」本身是一个系统偏置 —— 必须两个次序各跑一半(2026-08-12 mxfp4 grouped 实测)**
- 踩证:同一个只动 wgrad 的旋钮,`base→cand` 五对给出 gate_up wgrad canon **+1.70pp**、`cand→base` 三对给出
  **+2.34pp**;更硬的证据是**该旋钮根本不碰的 NT 格**(`gate_up fwd`)在两个次序里都是"先跑的那个偏低 ~0.5pp"。
  ⇒ 每对里第二次运行的卡态与第一次不同(热/时钟/L2 余温),固定次序会把这个偏置整包记到候选头上。
- 协议:**A/B/A/B 之外还要做 B/A/B/A,取两个次序的平均**;并且在输出里带一个**该旋钮按构造不影响的格**当
  自带对照 —— 它的漂移量就是该批次的偏置尺度。本轮正是靠这个对照才没把 −0.5pp 的 NT 漂移当成真回归。
- **一个批次里没有交错对照 = 这个批次不能用来判决**:同一份代码,交错批里 heavy 格均值 0.9057,
  30 分钟后的无对照批里同一格 0.8996(差 0.6pp,比效应本身还大)。收尾复测只能证"落盘态没写错",
  **判决必须回到配对差**。
- **每个批次的第一次运行是冷的**(本轮两次踩到:gm 99.62 / 99.54,而同批后续 100.5~100.8;冷跑的特征是
  **参考侧也一起变快**,如 ref_gu_wg 4456→4555)⇒ 丢弃或先跑一次热身。

**★★ 隔离/graph-replay 探针不能给「cache 状态」或「launch 开销」类改动排序 —— 会符号翻转**
- 踩证(2026-08 mx8tw campaign):同一个 `group_m=16`,在 graph-replay 探针上 **−0.65%(更好)**、
  在 18 格 bench 上 **−1.2%(更差)**,**同一个旋钮符号相反**。
- 根因是两种载体的 regime 不同:bench 交错 tw/mx 且每个算子 `record→fn→record→synchronize`,
  所以**每个 mx launch 都是 L2-cold**(tw 刚把 772 MB 冲过 32 MB 的 L2)、host enqueue 也在测量窗口内;
  graph-replay 连发 10 次同一 launch,是 **warm-L2 且摊薄了 launch**。
- ⇒ **凡是机制落在 cache 局部性 / swizzle / 派发顺序 / launch 次数上的改动,只在计分 bench 上排序;**
  隔离探针只能用来判 >5% 的结构性变化(spill、几何错误)。
- 配套两个量级:隔离 wall 探针**同臂散布 ~1.3%**(远大于常想分辨的 0.4%),而且**首次读数系统性偏高 3%**
  (回文序里第一臂必须丢弃,否则第一臂恒被判负)。

**★★ 多配置 bench 上 gm 和 min 的稳定性不一样,别用同一套判据**
- 实测(同一 campaign 反复三次):同 session 内标尺那一侧会**单调漂**,单格比值能摆 **2.8%**,
  足以把 18 格 gm 摆 **0.15%** —— 而 **min 格反而稳**(它的 mx 绝对时间全程只在 0.811–0.822 ms 之间)。
- ⇒ **gm 只能靠 X/C 交替夹心 + 逐块配对判**(相邻块两两比,看每一对是否同向);
  **min 可以直接比区间**。只看总均值会同时犯两种错:把真效应淹掉、把漂移当成效应。

**★★ flydsl JIT 磁盘缓存会让同进程 A/B 的两臂跑同一个二进制(2026-08-11 gptoss m4096 wgrad 实测,代价:把 +1% 的真收益误判为 0 并差点回退)**
- cache key = `函数源码 + 依赖函数源码 + closure 标量`,**模块级全局不进 key**。用 `KMOD._MY_KNOB = 0/1`
  在同一进程里编两臂时,第二臂直接命中第一臂的 `.pkl`,两臂**跑的是同一个 kernel**。
- 指纹:两臂中位数差 **<0.05%** 且输出 sha 完全相同;而 `FLYDSL_DUMP_IR=1`(它绕过缓存)下重编两臂
  **ISA 明显不同**(本例 v_mfma 1320 vs 1440、vgpr 488 vs 508)。「ISA 有差、计时无差」= 这个坑。
- 修法:探针一律 `FLYDSL_RUNTIME_ENABLE_CACHE=0`(或把旋钮做成 factory kwarg/closure 标量,让它进 key)。
  注意 `bench.sh` 那种**每 rep 起新进程 + 清 cache** 的路径不受影响,所以"bench 说 0、探针说 0"里
  探针那半可能是假的。

**★ `FLYDSL_DUMP_IR=1` 的计时**(2026-08-31 gpt-oss D=64 attention fwd,代价:整轮的收益投影作废)
- 它绕过 JIT 缓存、每次重编,且这类探针脚本通常只跑 3 rep,读数比真基线**慢约 5%**。
- 踩证:一个探针臂 1.798 ms 对着一条 **DUMP_IR 3-rep 的"对照" 1.9695 ms**,被记成 **−9% wall**;
  换成同会话正常对照(~1.85)重测,真值是 **+2.4%(更慢)**。
- ⇒ **DUMP_IR 的进程只用来读 ISA,永远不要拿它的时间当任何一侧的基线**;ISA 与计时分两次跑。

**★ 同一改动在「scorer 口径」与「热循环微标尺」上读出相反符号**(同上,gfx950 功耗墙下)
- `WARMS=5 / REPS=40` + 每格新进程的计分尺全程停在**冷、高时钟**状态;`min-of-200` 的循环坐死在
  1400 W 上限。实测同一改动:min-of-200 **−0.29%**、scorer **+0.08%**;某个臂 scorer **−2.4%**、
  持续态 **+1.2%**。⇒ **微标尺必须照抄计分尺的 warm/rep 数**,否则量的是另一个 DVFS regime。

**★★ 地板的正确量法 = 在 sweep 里塞一支「按构造应当同二进制」的臂,别靠重复同一个臂名(2026-08-19 gpt-oss D64 fused bwd 实测,−1.05% 的假赢)**
- 踩证:4 臂回文 sweep(每臂独立进程、min-of-40、3 轮)读出 `kreg3` 对 `B` **mean −1.05%,3 个读数全部低于 B 的 3 个读数**,
  看上去是教科书级的"分布不重叠 ⇒ 真效应"。事后 ISA 对拍才发现该旋钮在这个 head dim 上**根本没接线**:
  两臂三个循环的指令数、`s_nop` 等待周期、vgpr/lds **逐项相同** ⇒ 那 −1.05% 是**同一份代码**读出来的。
- 为什么比"重复同一个臂"更管用:同臂重复会被自己当成同一次测量对待(常共用编译产物、也不占回文里的另一个位置),
  **同二进制的异名臂**则完整走一遍"独立进程 + 独立 cache key + 回文中的另一个时隙",
  测到的正是**真实判决所处的**噪声通道。代价只是多一个臂的机时。
- ⇒ 口径:①每场 campaign 的 sweep **常驻一支同二进制对照臂**(取一个在本形状上确定失效的 kwarg);
  ②任何 <1.5% 的结论,先看对照臂这一轮漂了多少,再看候选;
  ③**候选与对照臂的 ISA 必须真的不同**——本轮同一批里另外两个臂(`g3at1`/`g3at4`)也是逐字节相同,
  它们"看起来的收益"同样是这条通道。
- 呼应本卡 §「两组样本不重叠不构成证据」:那里的机制是**单调漂移**,这里的机制是**随机通道**,
  两者都能造出 3/3 不重叠。真效应的签名是**跨 session 配对同号**:本轮真正 KEPT 的那个改动是
  **14 对回文、3 个 session、14/14 同号、mean −0.98%**,而假赢只在一个 session 里出现过一次。

**★★ 别用「几个读数的 min」去否决配对 win-count(2026-08-19 gpt-oss D64 fused bwd 实测)**
- 同一批 12 对回文:配对均值 B 5.6701 对 N 5.7174 = **−0.83%,11/12 同号,两个 session 各自独立同号**;
  但两臂的 **min 分别是 5.6497 / 5.6504,只差 0.01%**。两个口径指向完全不同的结论。
- 谁对:**配对 win-count**。每个读数**本身已经是 min-of-40**,所以这批样本的散布是**进程间**
  (allocator / 时钟态)的位移,右偏且长尾;从 6 个右偏样本里再取一次 min,是个方差极大的统计量,
  而配对差把两臂放在相邻时隙上、逐对相消掉进程间位移。
- ⇒ 口径:**min 用来对抗进程内的偶发高读(min-of-40),不用来对抗进程间位移**;进程间位移只能靠
  **回文配对 + 跨 session 同号**。看到「配对说赢、min 说平」时不要判平——那正是配对方法要解决的情形。
- 反过来也成立:若 **min 说赢而配对不同号**,那才是应当判噪声的形态(见本节同二进制对照臂)。

**★★ 「11.5% 摆动」可能不是长尾而是**双模**——24 进程才看得出来(2026-08-19 gptoss_swa 实测)**
- 一个 0.57 ms 的 cell,goal 里记的是"3 个独立进程取 min 仍读出 0.6331 与 0.5677,11.5% 摆动"。
  用**同一份二进制、24 个独立进程**重测:读数**不是**连续长尾,而是干净地分成两簇 ——
  低模 **0.567~0.574**、高模 **0.634~0.642**,中间**一个读数都没有**,簇间距 +11.4%。
- 判据价值:①双模 ⇒ **min-of-N 是对的估计量**(低模样本充足,N≥12 时几乎必然采到),而 mean/median
  会按落进高模的比例线性偏移;②同一份二进制在两臂上分别取 min 得 0.5668 / 0.5680(差 0.2%)⇒
  **这个 cell 的双模与代码无关**,任何"它回退了"的读数在 <12 进程时都不能当回退;
  ③ guard 项 `min(base/cur, 1)` 落进高模时被压到 ~0.89,单独给总分打掉约 1.5% ——
  **不计分的 guard 仍会用噪声扣分**,所以要 pin 的是"取 min 的进程数",不是"重复次数"。
- ⇒ 遇到"某个小 cell 摆动特别大"时,先用 ≥12 进程画一次分布再决定用什么估计量;
  **双模和长尾要用不同的处置**,而 3 个读数分不出这两者。

**★ 每场 campaign 开工先量自己那支探针的地板(2026-08-07 gpt-oss E=32 wgrad 实测)**
- `_an_wg_cfg.py` 的 palindrome 交错探针,同一配置(down `(2,6,1,1)`)在一个 session 内四次独立提交测得
  **2829.0 / 2841.7 / 2844.3 / 2848.3 TF**,极差 **0.68%** —— 这已经是"同进程交错、每轮清 flydsl cache"
  之后的地板。**验收线定在 >1.5%**,+0.9% 这种量级不许当 win。
- 代价对照:同一轮里被判死的两个改动分别是 **−20%** 和 **−9.2%**,远在地板之外 —— 真正的结构性错误
  从来不需要精细统计就能看出来。**要靠统计才能看出来的收益,基本都不是收益。**

**★★★ 链式(多 cell)尺子上,必须用「本改动逐字节碰不到的 cell」做同 run 批次偏置对照**
(2026-09-01 dense per-tensor fp8 campaign 20260901 r3 实测)
- 场景:5 个 cell 背靠背成链、单进程、每 cell 取 median-of-21、8 臂回文、band 钉死、cfg 逐 run 核对无漂移
  —— 也就是这把尺子能做到的最严设计。臂 = 只改 NT 方形宏 tile 的 mfma 发射序(纯置换,ISA 指令直方图逐项相同)。
- 读到的 raw 逐格 delta:被改动的 `nt_lin2` **+0.274%**、`nn_lin1` +0.173%;
  **而同一批 run 里逐字节未变的 `nt_mlpdn` +0.227%、`tn_lin1` +0.214%、`nn_mlpup` +0.366%。**
  ⇒ 未改动格与被改动格**同号同幅**,整批一起位移。
- ⇒ **判据:先算「本改动在 ISA 上碰不到的那些 cell」的平均 delta = 该 session 的批次偏置,从每个臂 delta 里减掉。**
  本轮三个"候选赢"全部因此塌回 0:`mstep=4` 在 A session 读 `nt_lin2` −0.28%(该 session 未改动的 `tn_lin1`
  是 −0.41% ⇒ 校正后 **+0.13%**)、在 B session 读 +0.274%(偏置 +0.22% ⇒ **+0.05%**)。两个 session 校正后同号且为负,
  与"raw 一正一负"的结论完全相反。
- 与本卡 §同二进制对照臂互补:那条是**跨 run** 的地板(需要额外的臂),这条是**同 run 内**的地板(不花额外样本)。
  链式尺子天然自带若干"同二进制"的 cell,**不用它们做对照就是在浪费尺子自带的对照组**。
- ⚠ 注意与"链式耦合"区分:功耗/L2 耦合会让未改动格**反向**动(改动格快 ⇒ 邻居抢到功耗 ⇒ 邻居也快);
  **批次偏置是同向同幅**。两者用同一套读数就能分开:看未改动格与改动格的符号关系。

**★★★ 用 monkey-patch 造对照臂时,patch 到共享 helper = 会静默改掉「别的 cell」的 autotune 选型;
唯一的检出手段是每臂 dump 逐格 cfg**(2026-09-01 dense per-tensor fp8 campaign 20260901 r4 实测)
- 场景:候选是「把 per-tensor scale 的两条标量 load 从 store 处提到 K 循环之前」,只动 TN。
  基线臂图省事,直接 `K.load_per_tensor_scale = lambda a, b: None`。
- 结果:`nn_lin1` 在基线臂上**换了一个内核**(cfg 从 `('nn4',256,2,0,2)` 变成 8-wave 的 `(256,1,0,4,32)`),
  该格 **+3.3%**。根因:同一个 helper 也被 NT/NN 的 `line_n` store 用来接它自己 hoist 过的 scale;
  返回 None 让那条路径输出**未乘 scale**,autotune 的正确性门把整个 `nn4` 候选拒掉、回退到别的家族。
  ⇒ 基线臂与候选臂之间差的不止一个变量,该对比整批作废(被改动格 tn_lin1 的读数也不可用,链式耦合已被污染)。
- 改正:patch **候选自己新加的那个接口**(这里是 store 基类新加的 `scale=` kwarg,`kw.pop("scale", None)`),
  不要 patch 任何被别的 layout 共用的函数。改正后六臂 cfg 逐格逐臂相同,四个对照格重新可用。
- ⇒ **口径:任何 monkey-patch 臂都必须在同一次输出里打印五格 cfg;cfg 有一格不同 = 这一批不是单变量,直接作废。**
  这条与本卡 §惰性臂互为镜像:惰性臂是"以为改了其实没改",本条是"以为只改了一处其实改了两处",
  两者都只能靠**每臂的机器可读指纹**(asm md5 / 逐格 cfg)排除,靠读代码推断会漏。

**★★ 共享节点污染的检测:`vs_sm` 不可靠,要用「绝对值上限」**(2026-09-01 同上,勘误本卡上文)
- 本卡此前记的识别签名是 `vs_sm > 1.5`(参照实现被一起拖慢 ⇒ 整卡争用)。本轮抓到一整批 8 个 run 里 5 个被
  邻居租户打坏:**五格全部 2~4 倍膨胀**(`nt_lin2` 0.4944 → 0.9445,`fp8gemm` 1.020 → 0.484),
  而 **`vs_sm` 全程停在 0.93~1.02**——因为参照实现在**同一个进程、同一条链**里测,被等比例拖慢,比值不动。
- ⇒ `vs_sm` 只能抓**瞬态/不对称**争用;**稳态争用要靠每格的绝对上限**(本核:`tn_lin1 > 0.73 ms`
  或 `nt_lin2 > 0.52 ms` 即判污染)或"总分远低于任何历史读数"。把这个筛子写进 parse 脚本,别靠眼睛。
- 污染源识别:`rocm-smi --showpidgpus` 只报 DRM device 序号,**不报是不是你的卡**;要配合 `ps -eo args`
  看别的 venv(本轮是同节点另一场 campaign 的 `--calibrate`)。GPU3 自己 0% 也可能被同 socket 的邻居影响功耗预算。

**★★ drift-corrected 的聚合分数**也会漂 —— 而且是**单调**漂,不是随机抖(2026-08-06 mxfp4 grouped 实测)
- 该 campaign 的计分量 `tf4_corr = r × REF`,`r = t_mxfp8/t_mxfp4` **同进程交错 A/B** 测得,设计上"免 DVFS 漂移",
  官方噪声带写 **±0.4%**。实测:**同一份未改动代码**在一个 session 内四次 `bench.sh` 得
  **4066.4 → 4084.9 → 4094.4 → 4100.7**,**单调上升,极差 0.84%** —— 是所声称噪声带的 **2.1 倍**,
  而且 `gm_ratio` 本身同向上升(1.6613 → 1.6746 → 1.6772)⇒ **不是 REF 常数的问题,比值 r 自己在漂**
  (交错 A/B 只消掉了"同一次调用内"的漂移,消不掉"mxfp4 与 mxfp8 对机器热态的敏感度不同"这一项)。
- ⇒ ★ **判据:交错 A/B 只保证 numerator 与 denominator 同时刻,不保证"比值"跨 run 稳定。**
  凡是两个候选**不在相邻两次 run 里测**的比较,0.3~0.5% 级别的差异一律不可信 —— 它可能整个是 session drift。
- ⇒ ★ **正确做法 = 把 A/B 提到 bench 层做配对**:候选与基线**交替连跑**(base,cand,base,cand…),
  用**配对差**判胜,而不是"候选三样本 vs 上一轮记录的基线三样本"。本轮踩证:一个改动早段测 4057~4071、
  看着与当时基线(4066.4)持平,而同一份基线晚段测到 4100.7 —— 若拿晚段基线去比早段候选,会**误判为 −1.0% 回归**;
  反过来若拿早段基线比晚段候选,会**误判为 +0.8% 收益**。两个方向的误判都只是时间顺序造成的。
- ⚠ 连带后果:任何以"+0.27~0.40%,两组样本不重叠"为由 KEPT 的轮次都值得复核 —— 样本不重叠在**单调漂移**下
  是必然现象,不构成证据。「两组无交叠」这个判据只在漂移是**随机**时才成立。

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
- **★ paired win-count 判 <0.5% 的效应时，9 trials 不够：必须 ≥11 trials 且在 ≥2 个独立 session 里同号**（2026-08-04 融合 hd64 flash bwd 实测）。踩证：同一改动首轮 **8/9 胜 +0.49%**、复测 **6/11、3/15 = 翻号**（另一个是 7/9 +0.31% → 2/11、5/15），两个都是假阳性，按 9-trial 收下就会把噪声写进 baseline。**判负同样要复现**：真效应在两个 session 里稳定同号且幅度一致（本轮 −2.9%/−2.8%、11/11 + 11/11）。
  ⇒ 口径：**|Δ|>2% 一个 9-trial session 够；0.5~2% 要 ≥11 trials；<0.5% 要 ≥11 trials × ≥2 session 同号**，否则记"噪声"而不是"赢"。
- **★★★ baseline run 和候选 run 要过同一道可比性门 —— 只查候选是最容易漏的一步**（2026-09-05 wgrad r14 实测）。
  本库已有「哨兵干净 ≠ 这一 run 可比，必须 grep 全部四条 race winner 行」（r8(d)/r10(a)），但那条通常
  只被用在**候选**上：候选 run 一旦选出非 canonical config，改动效果就被配置变化掩盖，这个人人都记得。
  漏的是**分母**：baseline run 自己也在 race，也会选出非 canonical 的臂。r14 的 baseline 首测 99.4067，
  其 gate_up G=24 选中 `(1,4,1,1,3,True)`（r11-H6 已量到 0.9958x），14 格里有 7 格拿了折扣价 ⇒ baseline
  被压低约 0.2pp，把改动的收益虚记成 **+0.49pp**。把改动摘出、在 HEAD 上重测到四条 winner 全 canonical
  得 **99.5882**，真实收益 **+0.31pp**（与两条独立证据一致：r5 高分辨率探针 +0.35%/+0.37% 单边 ≈ +0.18pp、
  r12 三次 bench +0.585pp）。
  ⇒ 判据：**baseline 数字和候选数字必须来自同一套 winner 校验**；baseline 脏了就把改动 `cp` 出去、
  `git show HEAD:<file> >` 回来重测一次再放回（4 min，比带着虚高数字进下一轮便宜得多）。
  ⇒ 方向性：脏 baseline 只会让收益**虚高**（分母被打折），所以「收益比预期大」永远比「比预期小」更该先去查分母。
- **别把跨进程双峰当 DVFS —— 先查 host 侧有没有 per-iteration 设备同步**（同上 campaign 实测）：同一 tree 跨进程分数曾稳定双峰（~810 / ~825，跨度 1.8%），归因 DVFS 是错的。根因是每次 backward 5 次 `.item()`，host 提交下一批 launch 前先等 GPU 排空，**host-enqueue 4.57 ms 与 GPU 4.90 ms 串起来而非重叠**；删掉后 host-enqueue → 0.064 ms，wall 与 kernel 时间和只差 5 µs，跨进程重复收敛到 ±0.4%（双峰消失）。⇒ 见到双峰先测 host-enqueue，别直接归因时钟。

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

### ★ campaign 指定的「对照组」就是最便宜的抢卡探测器（不用起 rocm-smi）
- 凡是 harness 每跑都打印、而本轮**一行没碰**的那个 metric（mxfp4 joint quant+GEMM campaign 里 = `quant_ms`），
  就是免费的 co-tenant 探针：它有稳定的历史带（本场 30+ 跑只在 **0.470~0.481** 摆），一旦跳出带就说明卡上多了别人。
  r15 实测：另一租户以 `HIP_VISIBLE_DEVICES=2`（与本 campaign 同卡）跑 `pytest tests/pytorch/...` 20+ 分钟，
  `quant_ms` 0.463 → **0.548**、`gemm_ms` 4.7 → 5.9~6.9、护栏 → 7~11 ms。**比 rocm-smi 更早、更确定**，
  因为它和被测量的东西跑在同一条流上、同一把尺子里。
- ⚠ **补充（r16）：对照组只在污染是「持续」时才灵，突发争用会漏报。** 同一租户的另一段窗口里
  `quant_ms` 0.468（干净带内）而 `gemm_ms` 4.68 → **5.79（+23%）**——短 kernel 恰好挤进空档，长 kernel 被打满。
  ⇒ **两条门一起用**：①对照组跳出历史带；②被测 metric 跳出历史带 >8%（`pitfalls/02` §回归判据）。任一触发就整段作废。
- 处置：整段读数作废（**不要**拿它跟别的臂比），等窗口干净再复测；实在等不到就退回**同 session 回文配对**
  ——r15 在被抢卡的窗口里仍用逐格 min/med 隔离 + 回文在 **11σ** 上分辨出 50 µs。
- 查证：`ps -eo pid,etimes,pcpu,args --sort=-pcpu | grep -v defunct`，看有没有别人的 pytest/bench；
  `<defunct>` 与长期 199% CPU 的老进程通常不占 GPU，别误杀（环境红线：不属于你的进程严禁 kill）。
- ⚠ **别指望 kernel-trace 的 min-of-N 能「绕过」被抢卡（r16 实测证伪）。** 直觉是"取 min 就能挑到干净窗口"，
  实测：8 臂回文 × 每臂 10 次派发，同一个 kernel 的 per-cell min 在臂间摆 **1084~2058 µs（±40%）**，
  某一臂整体和 3515 µs 而其余 4650~5240 —— **是那一臂恰好赶上安静窗口，不是那一臂的代码更快**。
  ⇒ 争用是**突发**的，min-of-N 只在 N 大到覆盖一个完整安静窗口时才收敛；N=10 远远不够。
  被抢卡时**唯一**可信的还是同 session 回文配对（且要看配对差的符号是否 4/4 一致），不是换个 min 口径。

### ★★ 逐字节指纹门（NDIFF=0）本身可能是不确定的 —— 先跑「自己 vs 自己」的对照
- 场景：campaign 红线要求"动 GEMM 必须自带 24 格逐字节指纹对拍，NDIFF=0 才算通过"。r16 实测**这个门有假阳性**：
  **同一份二进制、同一个 `torch.manual_seed(0)`、连跑 6 次，有 5~6 个格的指纹在变**
  （`wgrad {balanced,moderate,heavy,extreme} down`、`wgrad moderate gate_up`，换一组 6 跑还会冒出 `dgrad heavy down`）。
  其中 **`wgrad balanced down` 是计分格**。
- 机制：指纹算的是 kernel 返回的**整个** output 张量，而 output 是 `torch.empty` 分配的；
  skew 分布下有整组 `lens=0`／尾部 pad 区**根本没被写过** ⇒ 指纹把上一次分配残留的内存也算了进去。
  （所以不确定的格集合**每次都不一样**，还会跨到没有 split-K 的 NT 格上——这正是"未初始化内存"而不是"原子累加"的签名。）
- ⇒ **判据**：改动前先把 baseline 自己对拍 ≥4 次，得到「不稳定格集合」；
  只有**稳定格**上的 diff 才算真 diff。r16 的改动 diff 落在 2 个格，二者都在不稳定集合里，
  且 6 个 balanced（计分）格全部逐位一致 ⇒ 判通过。
- ⇒ **修法**（下一轮该做）：指纹只覆盖 kernel 合约要写的区域（按 `offs[G]` 截断、按有效 N 截断），
  或把 output 预先填成常量再调 kernel。**在修好之前，"NDIFF=0" 不能单跑一次就当证据。**

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

## ★★ kernel-only bench 必须**锁死 e2e regime**：G、K-pad、M_g 分布都要对齐真实派发(2026-08-03 tw grouped wgrad campaign 头号教训)

kernel-only 测量哪怕**口径干净**(raw op、rocprofv3、interleaved),只要**操作点(regime)**跟 e2e 训练不一致,
就是在优化一个训练里**根本不出现**的 shape。本 campaign(gpt-oss-20b MoE grouped wgrad)踩实的三条:

- **组数 G**:EP8 部署 ⇒ 每卡 **G=4 个本地 expert**,不是整模型 E=32。用 G=32 扫出来的
  L2 swizzle / XCD / band-cyclic 最优参数**不迁移到 G=4**(band 数、组内 tile 数、skew 结构全变)。
  bench 的 `_alloc` 必须按部署的 G 生成 group 直方图。
- **K-pad**:fwd 的 K=2880 `%128=64` 非对齐,e2e 里 fwd 走 **K-pad 到 2944**(见 [[project_kpad_e2e_trace_validated]]);
  拿 K=2880 未 pad 去 bench,量的是一个 trace 里**不存在**的 leading-dim 对齐画像
  (K%128 拆行 penalty,见 [[project_tw_fwd_gateup_kalign]])。**wgrad 的 free 维 2880 补到 2944 slice 回
  = 白捡 +3%,已烘进 baseline**(见 [[project_wgrad_reach_fwd_campaign]]);别再把 2880/2944 混着比。
- **M_g 分布**:真实 MoE routing 是 skew 的 ⇒ bench 必须覆盖 **balanced/moderate/heavy** 三档
  (heavy 是 min_ratio 所在,也是 skew 杠杆的验收点),单跑 balanced 会把 skew 尾部的坑全测漏。

- **比值口径的 yardstick 必须同窗测**:score = `r = t_fwd/t_wgrad`(wgrad 追平**冻结** fwd)时,fwd 这个
  分母**每个 config 都要在同一 interleaved 窗口里重测**——fp8 DVFS ~37% 漂移会同时动分子分母,
  隔次/隔 session 测的 fwd 当分母 = 把它的时钟漂移记进 wgrad 的账。**冻结的是 fwd 的代码,不是它的计时。**

⇒ **通用规则:定 kernel-only harness 前,先从 e2e trace 抄回真实 regime(G、每个操作数走不走 pad、
M_g 直方图),再锁死;口径干净 ≠ 操作点对**。这条比本卡其它测量纪律更靠上游——操作点错了,
后面 interleaved/多轮/SNR 门做得再干净也是在优化错的 shape。

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

## ★★ PMC 取「最后 N 次 dispatch」会静默读到**错误的那个臂**（多臂 palindrome 探针 + 同名 kernel）

2026-08-18 gpt_oss down fwd NT 实测，差点把一个真实的 **−48.9% 写请求**记成 0：

- 多臂 A/B 探针为了共享时钟通常按 **palindrome** 发射（`names + names[::-1]`），于是**最后一次 dispatch
  永远是 `ref`**；而两个臂编译出的 kernel **同名**（同一个 `@flyc.jit` 函数），所以按 kernel 名聚合、
  再取「最后 N 次」的 PMC 汇总脚本会把整个窗口落在 ref 的最后一轮里。
- **指纹**：待测臂与 ref 的**每一个** counter 都逐位相同（本例两臂都读 `TCP_TCC_WRITE_REQ = 2.359e7`）。
  和本卡 §同进程 A/B 静默同二进制是同一个家族的错觉，但根因完全不同（这次两份二进制确实不同，
  只是**没有一次 dispatch 属于待测臂**）。
- **正确入口**：探针要提供「单臂背靠背 N 次」的模式（`PROBE_SUSTAIN=N`：200 次预热 + N 次计时，
  且不进 palindrome），PMC 只在这个模式下采。换过去立刻读到 1.206e7。
- 顺带一条：**store 侧的臂必须同时报 `TCP_TCC_WRITE_REQ`（请求路径 = 功耗墙上付钱的那个）与
  `TA_BUFFER_WRITE_WAVEFRONTS`（发射条数）**。同一轮里 dwordx2 让 wavefronts 再砍半而 requests
  一点不动 ⇒ 只看任何一个都会误判「还有一半可减」。**这两个计数分道扬镳的第二种形态**（2026-08-18
  NN 行合并）是反过来的：**wavefronts 两臂完全相同 5.898e6，而 requests −48.9%** —— 存指令一条没减、
  只是每条 store 的 64 lane 从「覆盖 4 行各 32 B」变成「覆盖 2 行各 64 B」。⇒ 报了两个数才能说清
  「减的是请求还是指令」，而这决定了 ΔP 会不会跟着动（见 methodology/07 的 ΔF/ΔP 定价）。

## ★★ 计分尺的 autograd API 口径决定它**能不能看见**成本：`autograd.grad` 对非连续梯度免费，`.backward()` 要 378 µs

2026-08-17 gpt-oss-20b down-proj padN campaign 实测。bench 用 `torch.autograd.grad(out, [a,b], ...)`,
它把 kernel 的输出张量**原样交回调用者**;真实训练用 `.backward()`,`AccumulateGrad` 必须把梯度落进
`param.grad` 并满足参数自己的 layout 契约 —— **非连续梯度在这里被 clone,落在关键路径上**。

同一份代码、同一进程、两个 API 交错 min-of-12(gpt-oss-20b down, G=32/M=131072/N=K=2880):

| API | grad_b 是 [G,Np,Kp] 的 view | grad_b 原生连续 [G,N,K] |
|---|---|---|
| `.grad`(bench 用的) | padN 3.3438 ms | padN 3.3643 ms |
| `.backward()`(真实训练) | padN **3.7218 ms** | padN 3.3775 ms |
| 两 API 差 | **378 µs = 该步的 10.2%** | 13 µs |

kernel-trace 双证:view 版 `.backward()` 15 次 dispatch/3832.9 µs,含一个 **376 µs 的
`elementwise_kernel_manual_unroll`**;连续版 14 次/3456.9 µs,该核消失。

- **判据**:「返回 strided view 省掉一次 `.contiguous()`」这类杠杆,**必须同时用 `.grad` 和
  `.backward()` 两个口径 A/B**。只测 `.grad` 会把成本挪到尺看不见的地方,读数还很好看。
- **同族陷阱**:`autograd.grad` 也不会触发 optimizer / allreduce 对连续性的要求。凡是把
  "un-pad 拷贝"换成 view 的优化,真实收益都要按下游第一个**要求连续**的消费者来记账。
- 反过来用:如果一个 campaign 的 bench 就是 `.grad`,那么让 kernel **原生吐连续输出**这件事在尺上
  是噪声(本例 1.3254→1.3310),但在真实训练上是 **+11.5%**(1.1955→1.3327)。报告要把两个数都给。


### ★★★★ 自家孤儿探针能把整场 campaign 的尺子污染到失真(2026-09-03 实测)

dense fp8 GEMM campaign 收官复测时,同一份代码的配对 A/B 给出 **+0.48%、4W/2L**,噪声地板 >2%,
我据此报告「卡在 keep 门上、判不了」。用户说「我要啊」。

真凶:**我们自己 campaign 某一轮留下的探针 `_kyle_a2_ref.py fly 8 tn_lin1`,在计分卡 GPU3 上
以 200% CPU 空转了 18 小时**(容器内 PPID=1 的孤儿)。清掉之后:

| | 尺子噪声地板 | 同一对比的结论 |
|---|---|---|
| 孤儿在跑 | >2%(r10 跨度 2.2%) | +0.48%,4W/2L |
| 清掉之后 | **0.29%** | **−0.65%,0W/4L** |

**结论直接翻转**,而且清干净后的 r10 四读(1.0207-1.0237)与 `state.json` 记的 best=1.0205
严丝合缝——尺子本来是准的,是被污染了。harness 自己对同轴前几轮判的 −0.22%~−0.32% 一直是对的。

⇒ ★★**纪律:收官 / 任何关键 A-B 之前,先扫孤儿再测。**
```bash
docker exec <容器> ps -eo pid,ppid,etime,args | grep -E '_kyle_|_bench_|_probe_'   # 自家探针
rocm-smi --showpidgpus | grep -A1 'is using' | paste - -                            # 目标卡上几个 PID
```
目标卡上 PID > 1 就先查清楚再测。**注意 `rocm-smi --showuse` 显示 0% 也不代表干净**——
孤儿可能正好在两次 kernel 之间。

#### ★★★ 复现(2026-09-11,gpt-oss D64 bwd campaign r12):这次污染的是**两轮的 banked best**

上面那条卡写完之后又原样中了一次,而且这次的代价更隐蔽:被污染的不是某个候选的判决,是
**campaign 记在 `state.json` 里的冠军分数**。

- r10 与 r11 两轮**都以"树与 r9 冠军逐字节相同"收尾**(自己的轮次笔记里记了 `file_md5` 未变),
  banked best 却一路爬 **1.0339 → 1.0357 → 1.0403**,即 **+0.62% 纯尺子漂移被记成了进展**。
- r12 开轮扫孤儿,抓到**两个 `python3 _r11_prof.py` 99% CPU 空转,`etime` 分别 2h45m 与 1h54m**
  (05:18 / 05:44 启动)——**正好覆盖 r11 取计分读数的时段**。杀掉后同一个二进制官方尺读 **1.0351**,
  比 banked 的 1.0403 低 **0.52%**。
- 孤儿的来源是 **rocprofv3 自身**:一次给它 ≥6 个 counter 会以 signal 6 abort,**但它派生的 python
  子进程留在 100%**。4 个 counter 一趟是稳的。

⇒ 两条新纪律:
1. **扫孤儿是每轮的第 0 步**(不只是"收官前"),并把扫描结果写进轮次报告——
   否则一个零改动的轮次也能把 best 抬上去,而后续每一条真收益都要先还这笔虚账。
2. **别用 `pkill -f rocprofv3`**:在 `docker exec bash -lc '...'` 里它会匹配到自己的命令行,
   返回 exit 143、没有输出、孤儿还在。按 PID 杀,并连它的 `bash -lc` 父进程一起杀。

### ★★ 本地调用被打断,远端派发的活不会跟着死(同日,一天犯两次)

同一天第二次:我用 `bash remote.sh 'pytest ...'` 前台跑测试,**工具 10 分钟超时切断了本地这一端**,
远端容器里的 pytest 毫发无伤继续跑;二十多分钟后我又起了一轮 A/B,于是**两个 pytest 在同一张卡上
互抢**,两边都慢、都不可信。

⇒ ★**凡是「本地进程 → ssh → 容器」这条链,本地一被中断(超时/Ctrl-C/kill),必须主动去容器里收尸**,
不能假定信号会传过去。收尸时按 §pgrep 自杀那条:先把 PID 打印出来肉眼确认,再按显式 PID kill,
**别用会匹配到自己命令行的 `pkill -f`**(我今天也在这上面自杀过一次 shell,exit 144)。

---
来源: remote-sync/SKILL.md, pr-merge-gate/SKILL.md, 08-deadends.md, optimize-handoff/SKILL.md, optimize-loop.md, flydsl-fp8-gemm-tuning/SKILL.md, flydsl-fp8-gemm-tuning/07-benchmarking.md, fp8-gemm-bench/SKILL.md, 04-ceiling-analysis.md, flydsl-fp8-gemm-results/SKILL.md, 07-benchmarking.md, 10-grouped-wgrad-4wave-3buf.md, gpu-fleet-tuning/SKILL.md, 14-fused-preshuffle-e2e.md, mxfp8-grouped-gg-devloop/SKILL.md, project_mxfp8_wholeloop_port.md, 05-dead-ends.md, project_mxfp4_epilogue_store.md, verify-accuracy/SKILL.md, project_wgrad_occ_feed_bound.md, gfx950-vmcnt-race-debug/SKILL.md


## ★★ 小形状先证伪「内核慢」:很可能是 host enqueue 慢(2026-07-30 实测)

hd64 fwd 在 Hq=64/Sq=Skv=1024/B=4 上 wall 84 µs,怎么调都不动。**判据一句话:把序列长度扫一遍,
S=128/256/512/1024 全是 ~80 µs —— 工作量差 64 倍而时间不变,就不可能是内核里的任何东西。**

分三层量:
```python
a=torch.cuda.Event(True); b=torch.cuda.Event(True)          # ① wall(含等 GPU)
a.record(); [r() for _ in range(60)]; b.record(); torch.cuda.synchronize()
t0=time.perf_counter(); [r() for _ in range(60)]; t1=time.perf_counter()   # ② host enqueue(不等 GPU)
rocprofv3 --kernel-trace ...                                 # ③ GPU 内核真实时长
```
实测 ①84 / ②85 / ③50 µs ⇒ **host 是瓶颈,GPU 40% 时间在等**。②≈① 就是 host-bound 的签名。

根因是 FlyDSL 每次 launch 重解 JIT 签名(`inspect.Signature.bind`、globals-drift 检查、逐参数 cache-key),
cProfile 里 `jit_function.py:_resolve_and_make_cache_key` 居首。**修法:`flyc.compile(fn, *args)` 拿到
`CompiledFunction`,它只刷 data_ptr 不重解签名(全位置参数,含 stream),按标量签名缓存复用 → host 93→7 µs。**
⚠ 标量会被当 constexpr 烤进 artifact,**cache key 必须含每一个标量参数**,否则换 shape 会静默用错内核。

## ★★ 背靠背孤立计时是**偏慢**的一侧,不是"干净"的一侧;而跨张量比每字节 TB/s 会造出整条假杠杆

2026-08-17 gpt-oss-20b down-projection padN campaign,两条口径错误各让一条杠杆凭空存在了两轮。

### ① 同一个 launch,换掉它前面那个核,wall 摆动 3%
in-situ trace 读到 fwd NT 比对照臂慢 36 µs(4.2%),但孤立读数反而更快 ⇒ 曾据此写下"孤立没问题、
in-situ 有问题"。用**同一个 fwd launch、只换前置核**定价(`_probe_r10_ctx.py`,events 只圈 fwd,min-of-8×4):

| 前置 | fwd wall | vs 背靠背 |
|---|---|---|
| 背靠背(同一个核连发,每个孤立探针都这么测) | 854.7 µs | — |
| 它的 B 操作数的 quant cast(277 MB store)紧挨在前 | **828.5 µs** | **−3.07%** |
| 两个 quant cast(663 MB)都在前 = 真实链序 | 840.4 µs | −1.68% |
| 一个 512 MB 无关 fill(纯驱逐,无共享数据) | 843.0 µs | −1.38% |

两个结论:**(a) 生产者相邻值 ~14.5 µs**(`after_bq` 对 `after_flush`:producer 把操作数留在 256 MB MALL 里);
**(b) MFMA 密集核背靠背连发是最慢的排法**(功耗/DVFS + 上一轮自己的输出把 MALL 占满),所以
「孤立比 in-situ 快」**根本不能推出 in-situ 有病**——两者的差里至少有 1.4~3.1% 是排法本身。
⇒ 判 context 效应要**显式换前置核**,别用背靠背当基线;判核本身则两臂都用同一种排法。

### ② 两个核跑在**不同张量**上,它们的 TB/s 不可相减
同一份 trace 里 `padn_row`(权重 808 MB)= 5.70 TB/s、`pad_row`(激活 1141 MB)= 6.38 TB/s,据此
写下"padN 专属 quant 有 10.7% 每字节赤字、约 15 µs 可回收",挂在 goal 计划里两轮。
正确口径是**同一个张量只翻那一个开关**:`w_padn` 200.3 µs/1277 MB/**6.68 TB/s** 对
`w_padk` 195.9 µs/1271 MB/**6.80 TB/s** ⇒ 真实赤字 **1.8%**、专属净额 **4.4 µs**;
而同一探针里**激活路径每字节比它还慢 3.89%**(6.42 TB/s)⇒ 原结论的**排序都是反的**。
⇒ 每字节效率只能在**同形状同读写比**之间比;跨张量要比就先各自对自己的 padK/padN 变体做 A/B。

### ★★★ ③ 减法臂只界定了**和**,不授权把余项记到另一项头上(2026-09-05,一族候选被这样多开了两轮)
`A+B` 读 −0.8%、单独给 B 定价读 +3.2% ⇒ 写下"A = +2.4%"。这一步**在算术上成立、在因果上不成立**:
`A` 与 `B` 的实现互相约束(这里 A=换 MFMA 原子迫使 `n_tiles_b` 变奇数、B=偶数 `n_tiles_b` 才成立的
配对列 store),所以那个 −0.8% 里还含着**为了同时要 A 和 B 而付的第三项**(波格转置 = −15.8%)。
按 +2.4% 立项后实测:配对完好落地的组合是 **−24.4%**,把波格穷举到该原子最优点仍 **−6.3%**,
而"B 在 A 上"只值 **+1.3%**(不是 +3.2%)⇒ **A 这一项从头到尾不存在**。
⇒ 规矩:**要给 A 定价就直接量 A 的机制量**(此例是 `SQ_ACTIVE_INST_ANY/SQ_WAVE_CYCLES` = 19.2%,
一次 20 s 的 PMC 就能证明"减一半指令"无处兑现),不要用"总和减去 B"去反推;两项耦合时
差值的符号可以完全由**没被列进等式的第三项**决定。

---

## ★★★ 逐次 synchronize 计时会把**主机时间**算进 GPU 窗口(2026-08-24,一次把 kernel 低估 2×)

`record(); fn(); record(); synchronize()` 每轮都让 GPU 空转等 Python 入队,CUDA event 把这段**主机时间计进 GPU 时间线**。
踩证:grouped bf16 的 wrapper 在查编译缓存**之前**无条件重建 launch,**主机 2.52 ms/call**(Triton 只要 0.058 ms),
于是 fwd 读成 **650 TF/s,真值 ~1150**。据此还推出了一整套错误的 kernel 侧根因("比 wgrad 慢 2×""每 tile-iter 贵 5.13×"),
排查了 barrier/LDS/占用/ILP 一整天,**量的全是 Python**。

**必须**:一对 event 夹住 **R 次连续调用**,再单独用 `perf_counter` 量主机入队,断言 `t_host < t_gpu * 0.9`:
```python
e0.record()
for _ in range(reps): fn()
e1.record(); torch.cuda.synchronize()
gpu = e0.elapsed_time(e1) / reps
t0 = time.perf_counter()
for _ in range(reps): fn()
host = (time.perf_counter() - t0) / reps * 1e3
assert host < gpu * 0.9, f"host {host:.3f}ms did not lead gpu {gpu:.3f}ms"
```
⚠ **只测自己不会发现**——Triton 的 wrapper 快 45×,所以它不暴露这个坑;**和一个快 wrapper 对照才看得出来**。

## ★★ 噪声地板的量级是**每把尺子自己的属性**,别套用本卡上方的 4~5%

本卡上方说 run-to-run ~5%。**实测(gptoss bf16 grouped,安静的 GPU,连续入队计时)**:
计分 bench 三读 **1409.9 / 1410.4 / 1409.7 = 0.05%**;隔离 per-shape 尺子的同二进制对照臂 **0.007%**。
差两个数量级。一轮若按 4~5% 判,当天所有真实效应(0.58% 的 band、0.90% 的 GROUP_M)都会被当噪声扔掉。

⇒ 本卡的**方法**是对的且正是产出这些数字的原因(别用两次读数估地板、必带同二进制对照臂、回文次序);
   **数值不是常数**。每场 campaign 开工先用同二进制对照臂把自己这把尺子的地板量出来,再拿它判增益。

### ★★ 同一个 bench 里,**每个 bucket 的地板不一样** —— 拿最紧的那一档判决(2026-09-11 cfm_ep4)

同一次 `_bench_cfm_ep4.py` 吐 4 个 bucket。同一棵树重复读 6 次,各档的带宽差 2-5 倍:

| bucket | 权重 | 7 读的带宽 | 相对带宽 |
|---|---|---|---|
| b64 | 62% | 88.519-88.979 | **0.5%** |
| b128 | 31% | 93.257-95.243 | 2.1% |
| fused 总 | — | 21555-21715 | 0.74% |

真实收益 0.7% 时:b64 **零重叠**(判得死死的),b128 完全重叠(判不了),
加权总和只在极值处擦边。**先量出各档自己的地板,再决定用哪一档下判决**;
把注意力放在权重最大**且**带宽最小的那一档上,别去追一个 2.1% 带宽的档里 0.5% 的差。

同族陷阱:某一档的代码路径在 A/B 两版里**逐字相同**时,它两次读数之差就是那一档的地板
—— 这是最便宜的地板标定,不用额外跑对照臂(r11 靠它当场量出 b128 的 ±0.5 µs)。

**r12 更新(同一活动,26 读):b128 那条 2.1% 是被一次离群值撑起来的。**
r12 的四条基线读是 93.60/93.64/93.83/94.14(0.58%),b64 是 88.33/88.41/88.87/89.41(1.2%)。
⇒ 两档的真实地板都在 **±0.5%** 附近,谁宽谁窄**每个 session 会翻面**,
不要把上表当常数用;**每轮先用同 binary 的未改动档现场标一次**,再决定判决口径。
r12 里三次「只动 b128 几何」的实验,b64 就是这样当的免费控制臂。

### ★ 别把 harness 的 `ok` 当正确性门

r12 一读:`rel_l2 = 0.96`(数据全错),但 `"ok": true`、`candidate_failed = 0`,
`speedup` 照常算出 0.588。**判决前必须自己核对每档的 `rel_l2` 是否等于基线值**
—— 一个「变快了很多」的读数,第一件事是看它是不是把正确性丢了。

## ★★★ STREAM/链式尺子里各格**通过 L2 互相耦合**:单格的赢会被邻格还回去

2026-09-01 dense per-tensor fp8 campaign(5 个 GEMM 背靠背成一条链计时,互相冲刷 L2/MALL)。
把 `nt_lin2` 的 whole-loop band 从 `(GROUP_M=4, num_xcd=8)` 钉到 XCD-local 的 `(2,1)`:
**该格自己 −0.73~−1.08%(钉臂回文复测,机制成立:一条 band 的 A 行 15.7 MB ≫ 4 MB 的单 XCD L2 slice,
xcd=8 让 8 个 slice 各拉一遍同样的行)**。但链上**下一格 `nt_mlpdn` +1.1~1.6%**——而探针证明
`nt_mlpdn` 自己选中的 cfg **逐次不变**(始终 `('nt4',192,4,8)` 矩形),它没有任何自身改动。
官方尺子同 session 三读:钉臂 1.0181 / 基线 1.0186 / 基线 1.0220 ⇒ **净持平**。

⇒ 链式尺子上**「单格 −1%」不等于「总分 +0.213×1%」**:改动会改变这一格留给下一格的 L2 驻留,
   加权账必须**在整条链上读**,不能拿单格臂外推。反过来,**孤立单格尺子会高估**这类改动。
⇒ 判据:改动后**每一格都要看**,包括理论上不该动的格;某格动了而它的 cfg 没变 = 耦合项,不是噪声。

## ★★ run-to-run spread 里有一部分是 **autotune race 的双峰**,不是热噪声

同一场 campaign:`nt_lin2` 的 first-call race 在 `[(2,1),(4,8)]` 两条 band 间**8 次里 3 次选 (2,1)、
5 次选 (4,8)**,而两者相差 ~1%。这一格权重 0.213 ⇒ **光是这枚硬币就给总分带来 ~0.23% 的进程间抖动**,
与该尺子记录的 0.2~0.5% "噪声" 同量级。

⇒ 追 <0.5% 的臂之前,先 **dump 每个 shape 选中的 cfg 若干次**,确认选择是确定的;不确定就先钉死再比。
   与 pitfalls/06 §第五个处置方向(干掉竞速直接写死)同源:候选表小的时候,race 的采样谱宽就是尺子的下限。

## ★★ 功耗封顶的卡上,**一批里的第一个臂系统性偏慢 ~1.5%**——烧一个丢弃臂

2026-09-01,gpt-oss D=64 fwd,GPU 钉在 99% TBP(1400 W)。一个进程内顺序跑 N 个 arm(每 arm ≥14 s
持续循环、min-of-inner),**每批的第一个 arm 都读到 ~1.5% 慢、sclk 低 ~4%**:三批的首个基线读
1.8657 / 1.8623 / 1.8533 ms @ 1546/1533/1554 MHz,而**同一棵树**在批尾读 1.8343 / 1.8387 / 1.8363 ms
@ 1594/1602/1617 MHz。稳态本身复现到 **±0.0022 ms**,所以这不是噪声,是 DVFS 的爬坡:卡从半空闲
进入封顶状态要几十秒才把时钟策略收敛到稳态。

⇒ ①**每批先烧一个丢弃臂**(跑什么都行),或把基线在批首批尾各测一次、**只用批尾那个**当对照。
②这条与本卡 §「小-workload 热稳态协议」不同:那条治的是**冷 GPU boost 偏快**,这条是**入场偏慢**,
方向相反,同一场里可以两个都遇到。③判据:同一臂在批首与批尾读数之差 > 尺子地板 ⇒ 你在测爬坡。

### ★★★ 补(2026-09-04):palindromic 排序**不能**消掉上面这个位置项,它把位置项**钉在臂序上**

同一现象在**每个 rep 内部**也成立,于是 `A B … B A` 这种"对称就公平"的直觉是错的:一个 rep 的序是
`0 1 … N-1 N-1 … 1 0`,**臂 0 独占两个最边缘的位置(最慢),末位臂独占两个最居中的位置(最快)**,
而且这个偏差**每个 rep 都同向复现** ⇒ 取中位数不抵消、**换多少个独立进程都不抵消**。

gpt-oss-20b grouped fp8 NT(B=24,1400 W 封顶)实测,`warm=6` 全臂预热 + `iters=9` 取中位:
* 3 臂 `(ship, esp0, aux1)`,`aux1` 在末位 ⇒ **三个独立进程读 +1.33 / +1.45 / +1.29%**,跨进程散布
  只有 0.35pp,看起来是铁证。
* **臂序倒过来** `(aux1, esp0, ship)`,末位换成**逐字节未改动的 `ship`** ⇒ ship 读
  **+2.23 / −1.69 / −0.05 / +3.71%**(均值 +1.05%)⇒ **收益跟着位置走,不跟着代码走**。
* 独立铁证:同一 campaign 早前给一个**源码级 inert 的 kwarg 臂**(`cstore_aux` 那条路径压根没接到
  store 上)读出过 **+0.75%** —— 一个 no-op 不可能有收益,那 +0.75% 只能是位置项。
* PMC 反证机制:该臂的 `TCC_EA0_RDREQ` **+0.94%**、`TCC_HIT` **−0.31%**、`MemUnitStalled` **+2.14%**,
  与它自称的机制**反向**;计分 bench 也纹丝不动。

⇒ ①**任何臂序都要正反各测一遍、取两序均值**(位置项是奇函数,均值把它消掉;本例位置项 ≈ **±0.8~1.0pp**)。
②**< 1.5% 的臂,同进程 palindrome + 多进程复现都不足以定性**,必须再拿到**计数器级机制**或**计分尺**的
支持才能记成赢 —— 这恰好是本库 methodology/03 §「上界≠可达」的测量侧孪生条。
③写探针时**别把候选臂放在末位**(很多人习惯 `(baseline, candidate)`,而候选正好落在最快的那个槽)。

#### ★★★ 处方(2026-09-05 补):给探针加一个 **inert 对照臂**,把位置项**当轮标定**出来
上面的 ①②③ 只教你"怀疑",不给你**基准线**。做法:同一条命令加一个开关,让候选臂编译**与基线逐字节
相同的那个配置**,于是这一对是 `(X, X)`,读数里剩下的全是位置项。实测(syncv3 grouped fp8 NT B=24,
四个 cell):`(bar, bar)` 这对 identical 臂读出 **+1.72 / +1.18 / +1.49 / +0.07%,均值 +1.11%**,
xchk 逐位 0.0。同一批里真候选读 +0.78% ⇒ **真效应 = 0.78 − 1.11 = −0.33%,符号是反的**。
⇒ 没有 inert 对照,这个 +0.78% 会被记成"弱正、可以叠加";有了它当轮就判负。
**规矩:任何期望效应 < 1.5% 的臂,探针必须同时给出 inert 对照臂的读数,gain 一律报"对 inert 的差值"。**

#### ⚠ 跨进程的 `|C|` / checksum **不是**确定性判据(会伪造出 race)
很多探针顺手打印 `float(out.abs().sum())` 当"算得对不对"的哨兵。**如果算子是每进程随机生成的
(`torch.randn` 无固定种子、或 seed 走 `hash()`),那么同一份**未改动**代码在两个进程里的 `|C|` 本来就不同**:
实测 shipped 配置连测三个进程给出 `3.025031e9 / 3.028224e9 / 3.029663e9`(散布 0.15%)。
一次照这个"证据"把一个改动判成 race,后来同进程 xchk 证明它逐位相同。
⇒ **判 race 只认同进程、同算子、逐元素 `max|a-b|`(xchk)**;`|C|` 只能在同一进程内、同一份算子上比。

## ★★★ 孤立单形状循环会让「**填不满卡**的那条臂」回到 boost ⇒ 系统性偏袒不填 CU 的一侧(2026-09-10,差点判反一个 +21.5%)

Llama mxfp4 QKV_wgrad(384 tile / 256 CU = 1.5 波)对比 plain / 均匀裂 / 尾部裂,**同一棵树、同一进程**,
两种预热口径读出**相反的结论**:

| 预热/负载口径 | plain | uniform s2 | tail s2 | 结论 |
|---|---|---|---|---|
| 只循环这一个形状的三条臂,60 s | 375.9 µs @ **1699 MHz / 1165 W** | 407.8 @ 1380/1022 | 390.0 @ 1374/991 | plain 赢 → **裂 K 是负的** |
| 12 形状混合(计分 harness 口径)90 s + 窗口间用整个 mix 压住 | ~456 µs @ 1605/470 | ~408 @ 1604/471 | **~389** @ 1606/469 | tail 赢 **+17.3%** |
| 计分 bench 全表(权威) | 4417 TF/s | 5024 | **5392** | tail **+21.5%** |

根因:**plain 自己就只有 1.5 波**,第二轮 75% 的 CU 空转 ⇒ 瞬时电流低、卡有一半时间半空闲,孤立跑时
di/dt 调速器**放它回 boost(1699 MHz)**;把空转 CU 填满之后同一张卡只给 1605 MHz。于是"填满 CU"这类
杠杆在孤立微基准里要**先自付 ~19% 的时钟税**,而在真实层栈(前后都是满占用 GEMM)里根本不存在这笔税。

⇒ ①**凡是改变 CU 占用/并发宽度的杠杆,预热与窗口间隙都必须用真实负载 mix 压住调速器**,孤立循环的
读数只能用来排序**同占用**的臂。②`sclk` 必须逐臂记录:三条臂**同钟同功耗**才说明你在比代码;本例
第一种口径下 1699 vs 1374 就是"你在比两台机器"。③⚠ 这也是唯一一次**功耗墙回吐(R1)没有兑现**的场景 ——
稳态下三臂都 ~470 W(远低于 ~1400 W 上限),填满空转轮**不花钱**;别把孤立口径的掉频当成功耗墙。

### 附:单向 DVFS 转移下 **min-of-N 会把胜利判给拿到 boost 窗口的那条臂**

上表第一种口径里,`plain` 的**第一个窗口**读 373~376 µs,之后每个窗口都是 445~459 µs(1605 MHz),
即卡是**单向**离开 boost、不再回来。此时 palindrome 也救不了:位置项不再是奇函数(不是"首慢尾快"的
线性爬坡,而是一次**阶跃**),`min` 跨窗口取最小 ⇒ 谁碰上那个 boost 窗口谁赢。
⇒ **丢弃第一轮整轮**(所有臂),只在阶跃之后的稳态里比;或按窗口打上 sclk 标签,只比同档窗口。

### 附:采样式插桩本身会**不等量地**伤害多-launch 的臂

同一进程里每 0.4 s spawn 一次 `rocm-smi` 采样(它要读一堆 sysfs,~200-400 ms CPU),把 tail 臂
(3 kernel + 1 归约 = 4 次 launch/调用)读成 **390 µs**,而无插桩时它是 **306 µs**;plain 臂(1 次 stub
launch)几乎不动。⇒ 需要 W/sclk 时**每个窗口只采一次**(窗口 in-flight 时采),别开采样线程。

## ⚠ `rocm-smi --showgpuclocks` 的正则很容易匹配到**档位号而不是频率**

输出长这样:`sclk clock level: 1 (1546Mhz)`。用 `sclk\D*(\d+)\s*Mhz` 去抓,`\D*` 会跨过 `clock
level: ` 而 `(\d+)` 咬住 **`1`**,于是整轮 sclk 记成 "1 MHz / 2 MHz" 或干脆匹配不上。正确写法是锚在
括号上:`sclk clock level:.*?\((\d+)Mhz\)`;功耗同理用 `Power \(W\):\s*([\d.]+)`。
⇒ 通用教训:**采样脚本第一次跑要把原始行打出来核一眼**,别信自己按记忆写的字段格式——
这类静默失配会让整轮"没有时钟数据",而在功耗封顶的 regime 里(见 methodology/03)
**没有 sclk 就分不清"省了周期"还是"买了时钟",坐标系直接记错**。

### ⚠️ 一个 kernel 名下挂多个形状时,「中位数 × 次数」会乱跳(2026-09-03 gptoss MoE MLP)

- **现象**:`kernel_grouped_nt_persistent` 每步 2 次启动,但那是 **fc1(带融合 epilogue,N=5760)
  和 fc2(普通,N=2880)两个不同形状**。按 per-launch 中位数 × 次数估计,同一改动的 Δ
  跨轮极差 **41-68 us**,而且 nt/nn 之间的分摊会整个翻转(一轮 nt +86/nn −1,下一轮 nt +62/nn +27)。
- **后果**:据此得出过「dGLU 的归约比 GLU 贵 59 倍」的结论,**完全是假象**;改口径后两者是同一量级。
- **修法**:同名多形状一律取 **每步总和**(`sum(dur)/reps`),不要取中位数。改后极差降到 **15-22 us**,
  同一改动的 A/B 才可判。
- **推广**:任何「kernel 名 → 时间」的聚合,先确认这个名字下是不是只有一个形状/一种配置。
  profiler 只给名字,不给 launch 参数。

### ★★★ 用标称峰值定价 = 系统性高估 headroom(2026-09-03,一天内犯三次)

「这个核跑在峰值的 X%」这句话里,**分母必须是这张卡这种形态的实测可达值,不是标称/换算值**。
同一天三次,每次实测都把结论推翻:

| 项 | 我用的分母 | 同卡实测可达 | 结论 |
|---|---|---|---|
| tensorwise fp8 cast | HBM 标称 8 TB/s → 70-78% | **5.95 TB/s**(bf16→e4m3 `copy_`,256M/512M 两点一致) | 是**地板**:95.7-103.7% |
| grouped fp8 GEMM | 5000 TF/s(从别处记录换算)→ 45.7% | **3151 TF/s**(M=N=K=12288 大方阵,两后端都收敛到 3095-3151) | **71-75%**,不是「有 15 倍空间」 |
| 稠密投影后端 | 拿 hipBLASLt 当 TE 的代理 | 开关被 Primus 覆写、根本没开 | 「接不过来」的理由和结论都错 |

★ **参照系要和被测对象同形态**:`copy_ uint8`(1r+1w)只有 4.6-5.2 TB/s,**低于**被测的 cast;
换成同形态的 bf16→1字节 才得到 5.95。分母选错方向都可能反。
★ **代价**:凭标称值我把「GEMM 还有大空间」写进过结论和优先级排序,据此判断「便宜的赢取完了、
下一档在 GEMM 核心」——实测后这个排序不成立。**定价错会一路传染到优先级。**

---

## ⚠ campaign 的卡是**远端**的:本地直跑会静默换掉一颗架构(2026-09-05)
`flydsl_campaigns/<ts>/remote.sh "<cmd>"` 会先 rsync 再在远端容器里执行,所有读数都来自
那颗 gfx950。同一个探针在**本地**直跑不会报"跑错机器",症状长得像代码 bug:
```
hip_global.cpp:109 : Cannot find Symbol with name: _ZN12primus_turbo25compute_group_offs_deviceIlEEv...
timeout: the monitored command dumped core
```
真身是本地是 **MI300X/gfx942**、预编译 .so 里只有 gfx950 的 image。
⇒ ①探针的第一行输出应打印 `torch.cuda.get_device_properties(0).gcnArchName`;
②任何"上一把还好、这一把崩在符号查找/无 kernel image"的场景,先核架构再看代码;
③真正危险的不是崩,是**没崩**——若本地这颗恰好能跑,就会把另一颗卡的数字读进 campaign。

---

## ★ `speedup = ordinary / fused` 会把控制臂的漂移全额记成自己的收益(2026-09-11)

comm-fused MoE 的 bench 每次同时测 `ordinary` 与 `fused`,看起来「自带控制臂、天然抗漂」。
实际相反:一场 25 分钟 / 15 次 bench 里 `ordinary` 从 24509 漂到 **25071(+2.3%)**,
于是**同一份代码**的 speedup 读到 1.12502,也读到 1.14608 —— 差 1.9%,比本轮任何一个
真实增量都大。那次 1.14608 的改动(peer 读用 `nt`)在 `fused` 臂上其实是零收益。

⇒ ①主判据用 **`step_moe_us_fused`**(不含控制臂),`speedup` 只当汇报口径;
②同档噪声要先量:同一 binary 的 b64 读到 88.93 与 90.34(**1.6 µs / 1.8%**)
⇒ 单档 1 µs 以内的增量,一次读数判不了,结论要求两臂分布**零重叠**;
③找一个**本轮没动过代码的 bucket 当同 binary 控制臂**(这场是走 `full_reduce` 的 b8/b16,
以及几何未变的 b128),先确认这一读本身正常,再判被改的那档。完整战报见 pitfalls/14。

---

## ⚠ 影子候选:给「永远不入选的候选」提速,会让**入选者**的读数变慢(2026-09-11)
comm-fused MoE 的计分脚本对每个 bucket 取 `min(atomic, mega)`。mega 四个 bucket 全输,
所以「只动 `MegakernelConfig` 的默认值,分数结构上不可能变差」——这个推理**是错的**。

把 mega b64 从 163.5 µs 调到 94.0 µs(tile_m 16→32 + B 非临时 + BK=256)之后,同 session 配对:

| arm | 读数 | b64 fused | b128 fused | b16 fused |
|---|---|---|---|---|
| 旧默认(mega b64=163.5) | 1.10235 / 1.10284 / 1.10252 | 91.47–92.20 | 98.01–98.35 | 84.81–85.49 |
| 新默认(mega b64=94.0) | 1.08628 / 1.08679 / 1.08898 / 1.08545 / 1.08841 | 92.44–92.95 | 99.25–100.37 | 84.39–84.77 |

b16 确实**变好** 0.6 µs(mega 84.7 真的压过了 atomic 85.0,min 翻面了),但 b64/b128 的
**atomic 自己**慢了 0.75–1.2 µs,而这两个 bucket 的 min 两边都是 atomic。权重 150+75 压过 10 ⇒
score 1.1026 → 1.0872,A-B-A 5 读 vs 3 读**零重叠**,不是漂移(控制臂的 `ordinary` 反而更慢:
24636–24786 vs 24592–24679,机器在控制臂那段更热却打出更好的分)。

机制:候选顺序固定、每候选 iteration 数固定 ⇒ **影子候选跑得越快,轮到计分候选时卡越热、
候选之间的空闲越少**;另一个候选解释是 workspace 布局随 `producer_rows`(156→78)变了、
计分候选的 buffer 落到了不同物理页。两者都在我这侧不可控。

⇒ 三条操作结论:
1. **影子族的提速不能按「族内 µs」入账**,只有当它真的压过计分候选(且余量 > 1.5 µs)才算分;
2. 判这类改动**必须跑全 bench 做同 session 配对**;只跑 `--families <自己那族>` 会读出一个
   真实但**不入账**的漂亮数(94.0),然后误以为分数会涨;
3. 「min 的取胜者不可能变慢」这类**结构性推理不能替代配对实测**——共享的是同一张卡、
   同一个进程、同一段时间,不只是同一个 `min()`。

## ★★★★ FlyDSL 磁盘缓存按 **jit stub** 建键，不按发出的 inline asm ⇒ 同进程多臂 A/B 会**静默跑同一份二进制**(2026-09-11 r39 实测)

改 inline asm 的候选，其"臂"往往只是一个模块级常量（`_MXFP4_ARCH_ACC` 之类）在控制
`_llvm.inline_asm(...)` 的字符串。`RuntimeEnvManager.enable_cache`(默认 **True**)算的
cache key 只含 **jit 函数源码 + globals + 参数签名**，**不含发出的 asm 文本** ⇒ 第 2 臂
命中第 1 臂的产物，**编译都不发生**，两臂跑的是同一份代码。

症状全是"看起来没问题"，所以极难识破：

- A/B 读数 **+0.06% / −0.07%**，像一个诚实的"这条杠杆不值钱"结论；
- 正确性门 **10/10 逐位相同、SNR 55.x dB**，像一个诚实的"swizzle 未变所以位相同"；
- 两条一起看甚至**互相印证**（"位相同 + 零收益" = 完美自洽的死结论）。

⇒ **反例探针（必做，一次 5 s）**：读 jit stub 的 `_cache` 默认参数
（`Fn.__defaults__[-1]`），对每个缓存条目 grep 出被改动的 opcode 计数，确认两臂**不同**；
顺便比对两臂 artifact 对象的 `id()`。r39 第一次跑出 `asm_cache=0`（整轮全部由
`/root/.flydsl/cache_*` 供货，**一次都没 trace**），清掉缓存后第 1 臂 trace 了、第 2 臂
**仍然没有**（`asm_cache` 停在 1）——这才逮到。

⇒ **操作结论**：任何在 asm 轴上做同进程 A/B 的探针，**第一行**就写
`os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"`（在 import 之前）。
另外注意 `FLYDSL_DUMP_DIR` **参与** cache key ⇒ 分臂 dump 的 ISA 探针会碰巧躲过这个坑，
于是"ISA 门显示两臂不同"**不能**用来证明"A/B 的两臂不同"——它们走的是两条不同的键。
campaign 的 `bench.sh` 做 `rm -rf {CACHE}`，`remote.sh` **不做** ⇒ 手写探针默认在坑里。

### 配套：`rocm-smi` **绝不能夹在两个计时批次之间**(同一轮踩到)

想在 A/B 里带上 per-arm 的 sclk/W 列，最自然的写法是每个批次后 `subprocess` 调一次
`rocm-smi`。那一次调用**约 1 s**，而一个批次只排了 ~64 ms 的 GPU 活 ⇒ 卡空转 15 倍于工作
时间，直接掉出 boost。读数后果：绝对值比 `bench.sh` 低 **4~15%**（FC1_fwd 4770 vs 4983、
out_proj_wgrad 4653 vs 5505），**而且两臂都低**，所以比值看着还挺"干净"，只有跟 bench
对绝对值才看得出来。

三处一起改才对得上 bench：(1) 采样搬到**后台守护线程** 1 Hz，样本带时间戳，事后按每臂
的时间窗匹配；(2) `rocm-smi -d 3` —— 不指定卡会读 GPU 0，报出来的是**别人**的功耗；
(3) 批次自动定尺到 **≥300 ms**。改完 FC1_fwd 读 0.9282，bench 读 0.9283。

### 配套:先量一次 bench 的**墙钟**,再决定一轮排几个假设

comm-fused-MoE 那一族的任务描述里写着「一次 bench 约 2-6 分钟」,实测
`tools/_bench_cfm_ep4.py`(torchrun 4 rank)是 **37 s**。按 2-6 min 排计划会把一轮
压缩成 2-3 次读数 —— 而这一族的**单档噪声就有 1-1.8 µs**,2 次读数判不了 1 µs 的增量。
按 37 s 排,同一轮跑了 28 读 / 8 个假设,才够做 A-B-A-B 配对。
⇒ **每轮开工先空跑一次计时**,不要信任务描述里的估值;它决定的不是耐心,是**判优的统计功效**。

### 配套:控制臂往**有利**方向漂时,要主动报出来

r10 的 b128 几何隔离读:候选臂 94.42,对照臂 95.30,而同批**没动过**的 b64/b8/b16
控制臂在对照臂那一批**更快**(89.22 vs 89.54 / 169.65 vs 170.01)。
⇒ 批次偏置和结论**反向**,真实增益比读出来的更大。这种时候照样要写清楚,
理由和"控制臂往不利方向漂就要扣掉"是同一个:**判决靠的是偏置的方向,不是它帮谁**。
### 同会话基线也会漂:一次基线撑不住一轮,要用 A-B-A 夹逼

mxfp8 campaign r8:会话开头 r7 树读 `gm=105.1744 / pt_drift=1.0058`,同一轮结束回滚到
**完全相同的代码**再读,是 `gm=106.1641 / pt_drift=0.9990` —— 同代码同会话相差 **0.94%**,
远超 0.6% 的 keep 阈值。中间那个候选读 105.2075(+0.03%),照"会话开头基线"判是微增,
照回滚对照判是**负的**。

`pt_drift` 是抓手:它和 gm **反向**动(1.0058 → 0.9990),说明是整机变快而不是 kernel 变快。
分母冻结时整机加速会直接抬 gm,所以 `gm × pt_drift` 才近似不变:上面三次是
105.78 / 105.82 / 106.06。

按 run 的时间戳,这两次同代码读数只隔 **~58 min**(基线 03:03:43 收,回滚对照 ~04:01 收),
也就是 **~1%/h**;对 0.6% 的阈值来说,**对照大约 35 min 就过期**。

⇒ (1) "同会话"不是正确的单位,一个对照只能管半小时左右;候选必须
**控制-候选-控制**背靠背夹逼。(2) 每次报 gm 都要带 `pt_drift`,并用
`gm × pt_drift` 做归一化交叉检查。(3) 一次 bench 8.7-11.6 min,所以**一个夹逼 ~30 min**,
那才是"一次判决"的真实成本 —— 排轮次要按夹逼数排,不是按 bench 次数排。
(4) 回过头看,任何"裸的前后对比"结论都要补上这个误差棒。

### 隔离 ruler 的第二个杀手:输出缓冲区的**地址**,同一份二进制摆动 10-15%

同上 campaign。`_probe/r8_addr.py` 固定一份已编译 kernel、固定输入、固定 build 旋钮,
**只**把四个输出平面在各自 allocation 内的字节偏移挪一挪:
16384x21504 读 262.8-301.2 µs(**14.6%**),16384x15360 读 191.6-220.6(**15.1%**),
另外三个 shape 也有 10-11%。256 B 到 256 KB 的 pad 不改 caching allocator 的取整块大小,
所以那是**纯绝对摆放**,不是相对偏移。

后果:每臂各自 allocate 的 ruler 排的是运气。同一对二进制,每臂自带 buffer 时 `ldsc_pad`
读 **+9.4%**,所有臂共用一套 buffer 时读 **+0.3%**。叠加第二个偏置:**第一个被计时的臂
最多慢 12%**(冷)。

⇒ 隔离 ruler 的四条最低要求:(1) 所有臂**共用同一套输出 buffer**;(2) 计时前把**每个臂**
都预热过一遍,再开始计;(3) 正序 + 逆序回文,让时钟漂移对消;(4) 要落盘的东西,每臂在
**6-8 个不同摆放**上各测一次,按均值排序 —— 只在某一个摆放上赢的不算赢。

反过来,这个地址效应**不能拿来当优化**:`_probe/r8_hist.py` 在 5 种 allocator 历史下扫
pad {0,1k,2k,4k,8k},单臂跨历史就摆动 277-306 µs,平面的 2 MB phase 也随历史变,
没有一个 pad 处处为正(21504x3072 全为负)。它是**测量陷阱**,不是杠杆。

---

## ★★ 用"该改动**证明**打不到的行"当**内建零对照**,免费标定噪声地板

踩证(2026-09-11,MXFP4 dense NT campaign,12 行 Llama 训练 shape,geomean 记分)。
问题:campaign 的 `bench.sh` 一次 ~140 s,做"控制-候选-控制"夹逼很贵,而候选的效应只有
零点几个百分点,分不清"赢了"还是"机器飘了"。

办法:这一轮改的是 `_MXFP4_TPW_MAX`(persistent WG 走几个 tile),而 heuristic 里有
`K/256 > 56 → TPW=1` 的短路,**K=32768 的那几行无论旋钮取什么都走同一条 TPW=1 路径**。
用 ISA/哈希确认它们**逐 bit 相同**之后,这几行在**每一次 bench 里**就是一个免费的零对照:
它们的读数变化 100% 是噪声。实测这些不变行在两臂间摆动 **−0.52% ~ +0.63%**,
即单行噪声 ±0.4~1.0%,12 行 geomean 的噪声地板 ≈ **±0.12%**。

结论直接改判:本轮三个候选(`head-defer`、`store bank 2→3`、`TPW 2→4`)读数分别是
+0.08% / −0.005% / +0.07%,**全部落在 ±0.12% 以内**——没有一个是被测量证实的赢。
若没有这个内建零对照,`TPW=4` 的 0.9878 vs 0.9871 会被当成"小赢"写进 lineage。

⇒ 通用做法:选候选时**优先挑那种"记分集里有一部分行按构造不受影响"的旋钮**;
在报数之前先把那部分行的 delta 打出来当误差棒。挑不到天然不变行时,退回
控制-候选-控制夹逼(见上一节)。**这个零对照不额外花一次 bench 的时间。**

### 附:绝对分数的跨小时漂移,比任何候选的效应都大

同一棵树、同一份二进制,当天早些时候读 0.9871,几小时后读 **0.9917 / 0.9923**
(`still_ramping` 由 true 转 false、`pass_drift` 0.083 → 0.013)。**+0.5%**,
量级是候选效应的 4~7 倍。⇒ 任何跨越几十分钟的"前后对比"都不成立;
落盘 lineage 时要同时记 `pass_drift` 和 `still_ramping`,并**只在同一窗口内**比较两臂。

### step 打分器的单次夹逼分辨率 ≈ 0.15% gm(r9 实测)

一轮里同一棵基线树跑了两次,中间夹一个候选,`pt_drift` 基本不动(1.0070 / 1.0077 / 1.0076),
SNR 三次都是 28.27:

| run | 代码 | gm | worst | pt_drift |
|---|---|---|---|---|
| A1 | 基线 | 105.2427 | 101.6098 | 1.0070 |
| B | 候选 | 105.3540 | 101.5440 | 1.0077 |
| A2 | 基线 | 105.3955 | 101.9439 | 1.0076 |

**两次同码对照差 0.153%**,而候选落在两者之间(相对 A 均值 +0.035%)。
⇒ 一次 A-B-A 夹逼能判的最小效应就是 ~0.15%;比这更小的东西**不要用 bench.sh 去判**,
要么用隔离探针(单核拟合)拿机理,要么把效应先做大。
反过来也提供了一个便宜的零对照技巧:**构造一个与生产等价的"假臂"**(例如给一个本就取该值的
参数显式封顶),它的读数差就是当天探针的噪声底 —— 不额外花一次 bench。

### 更正:mxfp4 bench 的**单行**噪声可以到 ±2%,不是 ±0.5%(r43 实测)

之前记的是"行噪声 ~0.5%、geomean 噪声底 ±0.12%"。r43 在同一 session 里对
**逐位相同的两个二进制**各跑了 2~3 次(`_MXFP4_CBANK` 2 vs 3,17 形状 sha 对拍相同):

| 行 | 同码两次读数 | 跨度 |
|---|---|---|
| out_proj_wgrad | 1.0834 / 1.1064 | **2.1%** |
| QKV_fwd | 0.9548 / 0.9634 | 0.90% |
| out_proj_fwd | 0.9736 / 0.9638 | 1.0% |
| FC1_fwd | 0.9486 / 0.9493 | 0.07% |

geomean 侧:2 bank 读 0.9916 / 0.9925,3 bank 读 0.9936 / 0.9929 / 0.9920。
**两臂完全重叠**,均值差 0.078%。⇒ geomean 噪声底 ±0.1% 这个数还站得住,
但**任何单行的 ±2% 都不能当信号**;要看单行,必须同码多跑几次先量出那一行自己的跨度。
还有一条便宜的零对照:K=32768 的四行 wgrad + FC1_dgrad 走 `TPW=1`
(门是 `K//256 > 56`),对任何只走持久化路径的改动是逐位不变的。

同码三连读还出现过**单调下滑**(0.9936 → 0.9929 → 0.9920,`pass_drift` 0.0089/0.0064/0.0042)。
⇒ 只跟"session 开头那一次基线"比会系统性偏向候选;**必须在同一窗口内把基线臂重测一遍**。

### `{"ok": false, "error": "card not idle at start"}` 是显存门,不是利用率门

`_bench_campaign_mxfp4.py` 开头是 `(total - free)/2**30 > 8` 就拒跑。
自己上一次 bench / ISA 探针退出之后显存回收有滞后(实测读到 8.63 GB,
`rocm-smi` 同时显示 GPU use 0%),**几分钟后自己会掉回 299 MB**。
别去 kill 进程(同一容器里还挂着别人 GPU 2 / GPU 7 的 campaign),等就行。

## ★★★ CUDA graph 里的**跨流 fork/join 本身要 24.6 µs** ⇒ 记分尺是 graph 时,多流重叠方案永远亏

踩证(2026-09-11,comm-fused-moe EP4,MI355X / n02-29 容器 kyle_sglang):
想给"把 collective 和 GEMM 重叠"标价,写了 side-stream 别名臂
(`side.wait_stream(main)` → 侧流发 collective → 主流发 GEMM → `main.wait_stream(side)`),
用记分脚本自己的 `_graph_latency_us`(graph capture + replay,max-over-ranks)量。b64:

| 臂 | µs |
|---|---|
| `gemm` 单独 | 79.32 / 79.06 (首尾同 binary 控制,漂 0.2%) |
| `full` = gemm+tail 串行 | 89.15 |
| `par` = tail(96 blocks) 走侧流 | 103.93 |
| `par8` = 换成 8-block 的 collective | 102.97 |
| **`parnoop` = 侧流只放一个 1 元素 `fill_`** | **103.81** |

`parnoop ≈ par ≈ par8` ⇒ **+24.6 µs 与侧流上放什么、放多大完全无关,全是 graph 的
fork/join 脚手架**(两个 event 节点在 replay 时的代价)。

同一探针 `--eager 200`(不 capture,墙钟 / 200 iter,dist.barrier 对齐,max over ranks):
`gemm` 74.10 / `parnoop` 83.63 / `full` 83.72 / `par` 85.59
⇒ eager 下脚手架仍要 **9.5 µs**,但 **`par − parnoop` 只有 +2.0 µs**
⇒ **硬件上"14 µs 的 collective 与 79 µs 的 GEMM 同驻"只收 2 µs**(串行要付 9.6 µs)。

两条判据:
1. **别用跨流 graph 臂给重叠标价** —— 你量到的是 event 节点,不是重叠。先跑 `parnoop` 空对照,
   `par − parnoop` 才是重叠的价。
2. 如果**记分尺本身是 CUDA graph**(本 campaign 是),那么任何双流设计在评分时就带着这 24.6 µs
   ⇒ 重叠红利只能靠**单次 launch 内融合**去取,不能靠 stream。

⚠ 推论:曾经把"co-resident collective 让 GEMM 变慢"归因为**抢 CU**的结论要复查 ——
`par8`(8 个 block,3% CU)和 `par`(96 blocks)收同样的税,说明那不是 CU residency。

## ★★★★ 隔离 ruler 的第三个杀手:**"写放置"类杠杆的信用,在 step 里早被消费者付过了**(2026-09-11 r10 实测,三把尺互证)

第一个杀手是缓存(`nt` 的借方在消费者身上,r6),第二个是输出缓冲区地址(r8)。第三个更隐蔽:
**单算子 ruler 量到的 write-allocate 节省是真的,但它在 step 里不存在**。

mxfp8 dual-cast 的两张 E8M0 scale 平面:一条 128 B line 跨 32 个 tile,一个 tile 只拥有其中 4 B,
所以每个碰到这条 line 的 XCD 都要 fetch + writeback 一次。把**同样的字节**改写到连续地址(只换地址、
不减字节)量出来:**scale 平面占单次 cast 的 6.5–18.8%,其中 58–100% 是地址不是字节**
(row 平面边际带宽 0.34–1.11 TB/s,fp8 数据平面 4.2–9.6 TB/s)。机制清楚、数字很大。

于是按"每个 XCD 拥有一块 gm×Bk tile 方块"搜出每形状最优走位,11 条产线 cast 里 10 条变快
(+1.1/+1.9/+2.4/+0.4/+6.4/+6.5/+4.2/+6.2/+2.8/+1.9%,四张输出平面 bit-identical、回文序、跨三次
独立 sweep 一致)。**放进 step 全部还回去还倒贴**:

| 尺 | 读数 |
|---|---|
| 隔离 cast(`_robust_time`,回文序) | **+1.1 … +6.5%**,10/11 条 |
| `bash bench.sh` 同 session A/B | gm **105.96 vs 106.44 = −0.45%**,QKV −1.42 最差格 |
| 打分器自己的 step(逐 cast 开关,漂移校正后) | −0.39 / −0.61 / −0.57 / +0.26 / +0.02 |

**为什么**:step 里这些输出立刻被 preshuffle 和 GEMM 回读,working set 在 256 MB MALL 射程内,
write-allocate 的 fetch 半程本来就打在 MALL 上、不花 DRAM 能量 ⇒ **贷方早就付过了**;而借方
(方块在 M 方向一长,bf16 读就按 K*2 跨步)是新增的、真金白银的。同一坐标轴上这已经是**第三次**
以同样方式死掉(r8 全局 gm8b64 也是 +1.7% 单算子 / +0.05% 打分器)。

配套的两个定量事实(同一轮 torch profiler,per-launch 拆开):
- **同一个 cast,在 step 里比隔离慢 20–25%**(QKV 三条 cast 隔离 103 µs / step 内 126.5 µs)。
  所以连"把隔离收益按比例折算到 step"都是高估的,不只是符号会翻。
- scale 平面标 `nt`(cm=2)在 step 里是 **−4.9 … −8.1%**。这反过来给消费者对 cast 输出驻留度的
  依赖定了价:**scale 平面的驻留值 5–8% 整步**,比 cast 自己的写放置账大一个数量级。

**★ 反向的一半也要记住:删写请求/删扇区的杠杆在 step 里不打折,甚至放大**(2026-09-11
mxfp8-nt r5 实测)。2d-block cast 半区重排(`TCP_TCC_WRITE_REQ` 1792→1024/WG)孤立测
**−8.3 … −15.4%**,乘上 cast W 在各格 step 里的 2.92–5.52% 占比,朴素折算应得 ≈ **+0.5 pp**;
实测同进程 step **+0.80 pp**、`bench.sh` 中位 **+0.83%** ⇒ **比孤立预测更大**。机制正是上面那条
"同一个 cast 在 step 里比隔离慢 20–25%"的**另一面**:2d cast 孤立跑 4.02–4.26 TB/s、进 step 只
剩 3.04–3.41,同一个百分比对应更大的绝对时间。⇒ **分类规则**:①只换地址/顺序的 → step ruler
定价,孤立数字只做归因;②真删字节或删 64 B 写请求的 → 孤立数字是**下界**,可以放心立项。

**规矩**:凡是只换地址/顺序、不换字节数的杠杆(pid 走位、XCD 绑定、swizzle、store policy),
**一律直接上 step ruler**,单算子 ruler 只用来做机制归因,不用来做取舍。`_probe/r10_instep.py`
是现成的模板(打分器同款 `gemm_fp8(trans_b) + backward`,逐形状开关,末尾复测基线看漂移),
一轮 5 格约 5 分钟,分辨率 ~0.2%/臂 —— 比 `bash bench.sh`(6.5 分钟一次)还便宜。

---

## ★★★ 自写探针里 `current_stream()` 取在 graph 闭包**外面** ⇒ 读数会是「不可能地快」(2026-09-11 r18)

自己搭 CUDA-graph 计时臂时这么写:

```python
stream = torch.cuda.current_stream(device)   # ← 在闭包外,捕获到的是**捕获前**的那条流
def run():
    _run_compiled(kernel, ..., stream)
```

捕获期间当前流被换成了 capture stream,于是 launch 发到了**非捕获流**上,
graph 里**什么都没录进去**。replay 只在量空 graph ⇒ 本轮一个 GEMM 臂报了 **9.6 µs**
(真实 ~79 µs)。

**规则:`torch.cuda.current_stream()` 必须在 run 闭包**内部**调用**,和产品代码
(`_AtomicRunner.__call__`)一致。
**体检:任何比同工作量已知读数快 >2× 的臂,先当作没录进 graph,别当作发现。**
零对照做法:把该臂的输出缓冲区先 `zero_()`,再查 `rel_l2` —— 没录进 graph 的臂
输出会是全零/未更新,`rel_l2` 会异常而不是正常。

## ⚠ campaign 的工作树会被**会话外的进程**改回去(2026-09-11 r18)

`/workspace/code/gpt_oss_docker/sync/inference/aiter` 下的被跟踪文件,
本轮在 10:29:22 被会话外的东西**整批还原到轮次起点**(三个已改文件 mtime 同秒、内容回到基线),
恰好落在两次 `remote.sh` 之间 ⇒ 下一个探针以 `AttributeError: has no attribute '_gemm_args'` 失败。

**规则:改动一旦测出正收益,立刻把改过的文件复制一份到 aiter 树**外面****
(如 campaign 目录下的 `_r18_patch/`),再继续跑。
**体检:探针报「属性/符号不存在」时,先 `grep -c <新符号> <文件>` 查本地树,
再怀疑远端** —— 本轮远端和本地都是 0,说明是本地被还原,不是 rsync 没推到。

## ★★ 自己造的"忠实对拍"探针会造出 5 个假 FAIL —— 只有从**计分 harness 的活张量**上对拍才算数

2026-09-11 GLM-5.2 EP4 r6:换掉 MoE stage1 的 quant+sort 核后要证明"与原 HIP 核逐字节相同"。

第一版探针自己 `torch.randn` + 自己调 `moe_sorting`,10 个形状里 **7 个报 FAIL**
(`out_diff` 上万、`scale_diff` 数千)。两个假阳源头,都跟被测代码无关:
1. **比了 kernel 合约里根本不写的区域**。原 HIP 核按 *sorted row* 遍历,所以
   "没被任何专家选中的 token"那几行 `out` 从来没被写过,是 `torch.empty` 的残留;
   而新核按 token 遍历会写满。两次调用拿到不同的分配 ⇒ 差异全在这些**未定义**行上。
   (同族:pitfalls/02 §NDIFF=0 指纹门的"未初始化内存"签名——**不确定的格每次都不一样**。)
2. **探针自己把 `moe_sorting` 的 `num_experts` / `expert_mask` 配错**(EP 场景下
   全局 258 专家 + 本地 65 的 mask,手搓很容易配成不自洽),扫出来的 `sorted_ids`
   与生产路径根本不是一回事。

正解一步到位:**monkey-patch 被测函数本身**(`Q.fused_dynamic_mx_quant_moe_sort`),
在里面用**同一份入参张量**先跑新路径、再把 dispatch 开关关掉跑旧路径,当场比,
然后返回新路径的结果;外层直接跑计分脚本的 `build_case(bucket)` + `call_moe(case)`。
这样输入分布、`sorted_ids`、mask、`num_valid` 全部来自生产路径,一行都不用自己造。
结果:stage1 四个 bucket **`out_diff=0` / `scale_diff=0`**。

⚠ 比较集合仍要自己收窄到**合约要求写的那部分**:
`rows = [r for r in range(num_valid) if (sorted_ids[r] & 0xFFFFFF) < token_num]`,
scale 只比这些 row × `[0, N/32)` 列(按 `mx_scale_shuffle_idx` 算地址)。
本轮 stage2 走的是同一份 HIP 代码(自带零对照),它照样报 `out_diff≠0` ——
正是因为我没给 stage2 收窄行集合,**这个"FAIL"反过来证明了收窄的必要性**。

⇒ 三条:①对拍**输入必须来自计分 harness**,不要手搓;②只比合约写过的区域,
其余是 allocator 残留;③留一条**两臂逐字节相同的路径**在同一次运行里当对照,
它要是也 FAIL,那就是探针错了不是代码错了。

### 配套:子阶段定价用"把这一段置零"的临时开关,只读时间不读 `rel_l2`

同一轮给新核的三个阶段定价:加一个 `KYLE_PROBE` 环境变量,`=1` 时 `scan_iters=0`、
`=2` 时不发 scatter,直接跑计分脚本。输出的 `rel_l2` 会变 `NaN`(harness 照样
`"ok": true`,见本卡 §别把 `ok` 当正确性门),但 **`us` 是可用的**,一次 12 s 就把
"纯量化 −3.38 / 扫描 +1.72 / scatter +1.92 µs"拆清楚。
比起改代码做三个臂,这是最便宜的归因手段;**用完必须删掉开关**再做判决读数。

### ★★★★ 同进程多臂 asm A/B:关掉 `FLYDSL_RUNTIME_ENABLE_CACHE` **也救不回来**(2026-09-11 r47 实测)

本卡 §838 的处方"import 前写 `FLYDSL_RUNTIME_ENABLE_CACHE=0`"是**必要不充分**。r47 建了一个
同进程 ABBA 交错尺子(两臂各自的 `_MXFP4_LAUNCH_CACHE`/`_MXFP4_AT_CACHE`/`_MXFP4_CFG_CACHE`
按臂整字典换入换出,2016 样本/臂,读数分辨率看着有 **0.02%**),照方关了 runtime cache,
**四个臂仍然跑的是同一份二进制**:

| 臂(减法探针) | 同进程 ABBA 读数 | 分进程 5 臂读数(两端 arm0,跨度 0.28%) |
|---|---|---|
| 删 14/16 条折叠 store 的 WAR `vmcnt(4)` | +0.02% | 判平(与分进程一致,这臂本来就是平的) |
| 删相位屏障的 `vmcnt(10)` 内存等待 | +0.11% | **−0.08%** |
| 删相位 `s_barrier` | +0.03% | **+0.87% min / +0.99% med** |
| 全删 C store(`buffer_store_dwordx2`) | −0.07% 且 `bitsame` | —(不可能逐位相同,这就是破案点) |

⇒ 症状和 §838 记的完全一样:收益 ≈0 + 逐位相同 + 两者互相印证。**"删掉 store 还逐位相同"
是唯一一眼能看破的臂** —— 所以每个同进程多臂探针都必须带一个**物理上不可能逐位相同**
或**物理上必须大亏**的自检臂,而不是只看主臂的读数是否"合理"。

**自检臂的标定(gfx950 mxfp4 whole-loop)**:在 `_ipend` 末尾加 `s_sleep 1`(12 个发射点,
每 K-block 执行 1 次,K=4096/TPW=4 ⇒ 64 次/WG)。理论 64 clk/次 = 2.1%,**分进程实测只有
−0.48% min / −0.46% med** ⇒ `s_sleep 1` 在 gfx950 上的有效代价只有 ~12 clk。
它**够用但很弱**:自检臂必须选一个亏损远大于尺子地板的形式,`s_sleep 1` 已经贴着地板了,
要更狠就用 `s_sleep 4`+。

⇒ **操作结论(覆盖本卡早期"排序只认同进程回文 A/B"的措辞)**:在 **asm/emitter 轴**上,
"同进程"不是优点而是陷阱,**一臂一进程 + 每臂前 `rm -rf {CACHE}` + arm0 两端**才是唯一可信的
排序尺子;同进程交错只适用于**参数轴**(config/shape/派发)那种不重新 trace 的 A/B。
另外 ISA 门**不能**替代 A/B 门:`FLYDSL_DUMP_DIR` 参与 cache key,分臂 dump 的 ISA 会正确地
显示两臂不同(r47 实测 PROBE=0/8/1/2 的 `s_sleep` 0/12/0/0、`vmcnt(63)` 0/0/12/0、
`s_barrier` 27/27/27/15 全部如预期),而计时臂同时在跑同一份二进制。

### 同节点**邻居 campaign** 是尺子地板的一部分(2026-09-11 r47)

r47 的三臂回文里 arm0 两端跨度 0.31%,查出来是同一台 smci355 节点上另一个 campaign
(syncv4/mxfp8)正占着 **GPU 2**,而本 campaign 在 GPU 3。两张卡各自被钉在 1400 W cap,
但 sclk 在 1710~1744 MHz 间漂。⇒ 开跑前 `pgrep -fa campaign_remote` 看一眼有没有邻居;
有邻居时 **<0.4% 的臂靠 3 臂回文定不了性**,要么加臂数,要么等邻居跑完。

---

## guard 零对照的偏置是 per-session 的,摆幅 ±0.9%,而且必须用来决策

同一天、同一台机、同一个 `_bench_attn_bwd2.py`,7 次 bench 的 guard 几何均值(三个 guard 形状
的候选臂与参考臂代码**逐字节相同**,该值本该恒等于 1.000):

`+0.52% / +0.35% / +0.87% / −0.26% / −0.94% / −0.20% / +0.15%` —— 摆幅 **1.8%**。

三个互相独立的形状(D=64 窄带 / D=128 窄带 / D=128 causal)**同向**偏移 ⇒ 这是 ABBA 的
**per-session A/B 槽位偏置**,不是 per-cell 噪声,它按同样的比例污染 TARGET 单元。

★ **决策统计量是校正值 `raw / guard_geomean`,不是 raw。** 编排器复测的期望 E[raw] = 真值,
而真值的最佳估计是校正值。实证:校正值复现了编排器在**不同 session** 对同一棵树的独立复测,
误差 ≤0.26%,而 raw 分数把两个配置**排反了**(详见 pitfalls/13 §r19.7 的对照表)。

★ **但校正值自身带 ±0.5% 噪声**(三个 guard 的 resid 0.4–2.1%,几何均值 ≈ ±0.5%)。
⇒ 凡差异 <0.8% 的候选,必须拿**两次独立 session 的校正值同向**才能采信;单次读数只能用来排序,
不能用来 KEEP。这比 harness 宣称的 0.3% 分辨率保守得多,而 0.3% 那个数只对**同 session 内**的
配对比较成立。

---

## `_bench_moe_ep4.py`(GLM-5.2 EP4)实测噪声是 0.2%,不是任务描述说的 0.035%

r8 在**同一棵 baseline 树**上取了 7 个读数(跨两个 session,GPU 1,无邻居):

`1.07587 / 1.07631 / 1.07705 / 1.07573 / 1.07606 / 1.07698 / 1.07492`
→ 均值 1.07613,sd 0.00070,**极差 0.00213 = 0.20%**。

逐 bucket 看,漂移主要在 b128:同一棵树两个 session 分别给 `217.22/217.21` 和
`216.55/216.84`(**0.7 µs = 0.3%**),而 b64 稳在 `182.3–182.8`(0.25%)。
b8/b16 摆 ±0.5%,但权重只有 15/240,对分数影响小。

★ 后果:r8 中途一度把"stage2 空 block 跳过"判为"b128 −0.72 µs 的真收益",
复测 baseline 后发现那 0.7 µs 就是 b128 的 session 漂移,该改动实测**完全等于 baseline**
(3 读均值 1.07613,与 baseline 均值同到小数点后 5 位)。

⇒ 判据:这个 harness 上**单次读数的差异 <0.25% 一律不可采信**;
要么取 ≥3 读的均值且差异 >0.3%,要么把候选臂和 baseline 在**同一 session 内交替**跑。
0.035% 那个数不要再当作决策门限。

## ⚠ sysfs 读功耗/时钟：本容器只有 `power1_input`，`power1_average` / `freq1_input` **都不存在**

（2026-09-12 mxfp4 campaign 节点实测）测量契约要求每条臂同报 `W` / `sclk`（sysfs 直读，
不许在两批计时之间 fork `rocm-smi`）。但沿用
`/sys/class/drm/card*/device/hwmon/hwmon*/power1_average` + `freq1_input` 的探针在本节点
**静默返回 None**（`except OSError: continue` 把两个缺失文件都吃掉），整条 A/B 就没有功耗证据，
而 json 里只是一个不起眼的 `"W": null`。

- ✅ 可用：`/sys/class/hwmon/hwmon*/` 里 `name == "amdgpu"` 的 8 个目录 —— `power1_input`（µW）、
  `power1_cap`（本卡 1400 W）、`temp2_input`（m°C）。
- GPU 序：`readlink -f <hwmon>/device` 的 PCI 地址 `05/15/65/75/85/95/e5/f5` = GPU0..7
  （本 campaign 用 GPU 3 = `0000:75:00.0`）。取"读数最大的那张卡"也可以——满载时自己一定最高。
- ❌ sclk 没有 sysfs 通路（`freq1_input` 缺失、`pp_dpm_sclk` 为空）⇒ 只能 `rocm-smi -c`（不在计时窗口内）
  或 GRBM 反解。**下结论说"sclk 持平/上升"之前，先确认探针真读到了数。**

## 多核链(kernel A + kernel B)的绝对值跨进程不可比,只有同块 ABBA 可比(2026-09-12,sparse-MLA decode)

同一份未改动代码、同一台卡、同一个 45s 时钟爬坡,`producer+combine` 两核链在三个探针进程里
seq48 分别测到 **17.42 / 17.88 / 18.22us(差 4.6%)**,而同样条件下 producer 单核跨进程复现到 **0.24%**。
差别来自两核之间那段串行间隙(graph 里相邻节点的 barrier + 前核尾排空 + 后核爬坡,实测 0.7~1.2us
= 调用的 4~7%),它对进程级的分配/驻留状态敏感。

⇒ 规矩:
1. 只要指标里含两个以上 kernel,**所有臂必须在同一个进程、同一个 ABBA 块里**测,回文序 + min-of-N。
2. 不同臂需要不同形状的中间 buffer(例如不同 tiling 的 partial 记录数)**不是分开测的理由**:
   给每个臂配自己的中间 buffer、共享输入和输出张量即可。
3. 跨 run 只能比"同一个 run 内的相对差"。拿 A 进程的 ship 去比 B 进程的臂,4.6% 的漂移足以
   把 −4% 的真赢读成 +0.5% 的"不复现"——本族为此误判了一整轮。

## 尺子的总分噪声可能被一个你没在动的 kernel 主导 —— 先看分项 spread(2026-09-12,GLM decode)

轮 8 的尺子 `score` 在**同一份代码**上跨 run 落在 1.643~1.654(±0.4%),而当轮真实增益也就 +0.4~0.7%,
于是"改完跑一次 bench"完全读不出符号。拆开分项就清楚了:一步的公式是
`step = mla_v×79 + mla_d×5 + (uk+uv)×78`,其中 **`uv` 的 run 内 spread 就有 3.5~6.2%**
(`mla_v*` 只有 0.15~0.6%),`uv×78` 又占 c8 的 17%、c14 的 16% ⇒ **总分噪声几乎全是 `uv` 贡献的,
而这一轮根本没碰 `uv`**。

⇒ 规矩:
1. 动手前先把尺子**各分项的 spread 拉出来排序**,认清哪一项在定总分噪声;若它不在你的改动面上,
   **总分就不是这一轮的判据**,分项才是。
2. 用分项做记账:`Δstep = Σ Δkernel × 调用次数`,再折算成分数。本轮据此算出 +0.4~0.7%,
   而六次 `bench.sh` 的总分分布与基线**完全重叠**。
3. 反向的坑更贵:轮 8 一度按"ruler 上 `mla_v14` +0.17%"给已落地的优化加了一道门,
   把两个最重的形状关在门外;后来发现 **`mla_v14` 跨进程自己就漂 0.7%**(把该形状的代码路径
   改回与基线逐字相同,仍读到 +0.23%),门是拿噪声拟合出来的。**别用跨 run 的分项差做门限。**

## 🐞 工作树**不是你独占的**:每次 bench 之后要复核你落的字段还在不在

轮 11(campaign 20260912_134949)踩到:同一份 aiter 树上有**另一个 writer**在改
同一个 tuned 表和同一个 kernel 文件。观察到的三件事:
- `sparse_mla_decode.py` 里刚写进去的编译期参数 + 分支,几秒后 `grep` 已经不见了
  (mtime 还是自己那次写的时刻);
- `BATCHED_GEMM-...-B=16-N=256-K=512.json` 的 `M_LEQ_128` 在 `ns2/wpe4` 与
  `ns3/wpe5` 之间来回翻,mtime 出现在**自己没有编辑的时刻**;
- 自己改的 `M_LEQ_256` 每次都活着,只有那一个桶被别人管。

后果很隐蔽:**探针进程里 `_get_config` 读到的 ship 与你在文件里看到的不是一个东西**,
于是「ship 4.735us / 挑战臂 −0.29%」和「ship 6.767us / 同一个挑战臂 −8.04%」这种
自相矛盾的结果会同时出现在同一轮里,而两次都是真数据 —— 只是 ship 换了人。

⇒ 操作规程,三条:
1. **每个探针都要把 ship 的完整 config 打出来**(不只是名字里那一两个字段),
   并在报告里逐臂给出 —— 这样 ship 漂移能当场看见,而不是事后靠 mtime 考古。
2. **每跑完一次记分尺,立刻复读你落的那几个字段**(几行 `json.load` 就够)。
   本轮最后两跑就是这么发现 `wpe4 → wpe5` 被改掉的。
3. **别和它抢**。先看被改的字段在你自己的实测里是否在平台上
   (本例 `wpe∈{3,4,5,6}` 互差 <0.4%)⇒ 在平台上就随它;只有关键字段
   (本例 `ns3` vs `ns2`,差 5–8%)被改回去才要动手,并在报告里点明,
   否则编排器 re-bench 的是别人的配置,你的分数会莫名其妙复现不了。

⇒ 同源提醒:**同一台机器的「会话状态」也会整体漂移**。本轮同一个探针、同一组臂、
同一个 M,两个会话的绝对值是 4.72–4.74us 与 6.13–6.81us(慢 40%),
而 `ns3 vs ns2` 的比值从 −0.29% 变成 −8.1%。⇒ **只有同进程内的比值能跨会话引用,
绝对微秒数不能**;而且「某臂只在负载态下赢」是常态,结论要写成
「在两种机器状态下分别是 X / Y」,不要只报一个数。

---

## ★★★ `ptr_arg` 只留 `data_ptr()` ⇒ 多臂探针的**静默别名**;`ctrl%` 对照臂是唯一信号

`aiter/ops/flydsl/.../tensor_shim.py` 的 `ptr_arg(t, dtype)` **只取 `t.data_ptr()`,不持有
tensor 引用**。于是探针里任何在构造函数返回后失去引用的 device tensor,会被 caching
allocator 立刻回收**并发给下一个臂**。

**症状(2026-09-13 实测)**:seq60 上「可确定性复现的 NaN + 98% 的 partial 没被写 +
对照臂 +16.7%」,而 kernel 全程正确 —— 三个臂共用了同一块 `q`/`idx`/`po`,后建的臂
把先建的臂的输出当输入。这类现象极易被误读成缓存一致性 / race,我为此追错方向很久。

⇒ 判据两条:
1. **多臂探针必须把每个臂的每个 tensor 显式 root 住**(放进一个 list / 闭包并保持到测完),
   不能只保留 `run` 闭包以外的东西;构造函数返回 `(run, tensors...)` 时,调用方要接住全部。
2. **永远带一个惰性对照臂**(与 ship 逐字相同的第三份,排在末位):`ctrl%` 偏离 1% 以上时
   **先怀疑探针,再怀疑 kernel**。本例 `ctrl% = +16.7%` 是唯一在第一时间就指向探针的信号,
   而 delta% 本身看上去完全"可信"。

同族小坑:`buffer_load(vec_width=1)` 返回**标量**而不是 vector,`fx.Vector(raw, shape=1)`
会抛 `Cannot cast type to VectorType`,要走 `fx.Vector.from_elements([raw], fx.Int32)`。

## ★★★ 单趟计分尺的 +0.5% 与「同块 ABBA 证明为零」是同一个改动(2026-09-13 r15)

同一棵树、同一把计分尺、同一天,三趟读数 **1.76133 / 1.7646 / 1.77086**,展幅 **0.54%**。
其中 1.77086 那趟装的是一个改动,另外两趟是基线 —— 而该改动用同块 ABBA + 惰性对照臂
直接量到 `delta − ctrl = +0.03%`(逐位相同)。**单趟 +0.54% 完全是尺子自己的带宽。**

分项上看得更清楚:那趟"赢"的 +0.54% 里,被改动的 `mla_v14` 只动了 −0.31%(自身 spread
0.3~0.55%),真正的位移来自**没被碰过的** `uv_8`(−1.1%)与 `uv_14`(−2.9%)—— 与本卡
「uv 的展幅主导总分噪声」一节同一个机制。

⇒ 操作判据(比"1.5% 噪声门"更可执行):
1. **改动只影响某一个分项时,不要看总分**,看那个分项,并且和它**自己的 spread** 比。
2. 分项也不够 ⇒ 上**同块 ABBA + 末位惰性对照臂**,报 `delta − ctrl` 而不是 `delta`。
   本例 `delta = +0.536%`、`ctrl = +0.504%`,只报前者会得到完全相反的结论(而且是"变慢"
   的结论,同样是错的)。
3. 臂位置偏置在本机稳定在 **0.5% 量级**,和单轮采纳裕度同阶 ⇒ 对照臂不是可选项。

## ★★★ graph 节点的 1.5 us **没有 kernel 侧旋钮**,而且独立节点也不重叠(轮 13 定价)

本卡与 pitfalls/12 多处引用「空核 ≈1.85-1.93 us」「一步 324 次 launch ≈ 31.5% 的 step」,
但一直没量过**这个数受什么影响**。轮 13 用同一张 HIP graph 里放 N∈{1,2,4,8} 个空核、
取 total-vs-N 的斜率,逐项扫:

| 臂 | 斜率 us/节点 |
|---|---|
| absorb 的完整签名(5 ptr + 13 标量 + 8 constexpr) | 1.5709 |
| 2 个参数 | 1.5311 |
| 2 个参数,`num_warps` 1 / 4 / 8 | 1.5219 / 1.5311 / 1.4987 |
| 2 个参数,grid 1 / 192 / 768 CTA | 1.5387 / 1.5311 / 1.5292 |
| **2 个参数,相邻节点写互不相交的输出** | **1.5142** |

两条可执行结论:
1. **kernarg 大小只值 0.040 us/节点**(24 个参数 vs 2 个),grid 与 workgroup size 完全平
   ⇒ 想靠"瘦身 kernel 签名"回收派发成本是徒劳的,**唯一的杠杆是节点数**。
2. **相邻节点即使写互不相交的 buffer 也不重叠**(1.514 vs 1.531,在展幅内)⇒ 那 1.5 us
   不是探针里的假数据依赖,是派发器的节点间延迟。这同时说明"用空核斜率当派发价、
   用满核斜率减它当核体价"的拆分是站得住的:两者的派发项同价。

配套定价(absorb,B=16,M=48,192 CTA,`delta` 均为斜率):`uk`(N=512,K=192)整次调用
2.841 us = 派发 1.542 + 核体 1.299;`uv`(N=256,K=512)3.206 = 1.475 + 1.731。
**每次调用有 46~54% 是节点本身**,而核体里 per-token-group 前导只占 2.5%(uk)/16%(uv),
**再加一条 dot 链只要 +0.008 / −0.004 us**(MFMA 被完全隐藏)⇒ 这个核的核体在访存/延迟
regime,算力与量化都不是货币。

## ★★★★★ 测量装置本身读错了卡 —— 同一个错连犯三次(2026-09-13 e2e A/B,整夜)

`docker exec` **不继承** `HIP_VISIBLE_DEVICES`。于是容器里 `torch.cuda.mem_get_info(0)` 读的是
**物理 GPU 0**,而本线跑在 2/3/6/7 上。三次现身,每次我只修了发现的那一处:

1. **污染探针**:窗口前后各读一次显存当污染指标。记录到「窗口内 255.9 → 13.8 GiB」,
   以为是自己的服务在释放,**实际是邻居的服务关掉了**。
2. **teardown 的排空循环**同一个 bug。它报 `DRAIN TIMEOUT, still 255 GiB` 时,
   我们自己的四张卡其实已经是 0;而之前每一句 `drained, 0 GiB left` 都是**碰巧**对的。
3. 读对卡了,但**只读四张里的第一张**。rank 退出不同步,GPU 2 先空、3/6/7 还占着,
   检查放行 ⇒ 下一台 server 起来时只剩 20 GB ⇒ `max_total_num_tokens=23680`(正常约 2.5M)
   ⇒ 142k 提示全被 400 拒绝,整点作废。

⇒ **纪律**:(a) 任何读设备状态的探针,先打印它读的是哪张卡,和被测对象比对;
(b) 多卡任务的资源检查取**所有卡的最坏值**,不是 device 0;
(c) **修完一处立刻全仓 grep 同一模式**——「同源的第二处」比「第一处」更容易漏。
★ 指向错卡的污染探针**比没有探针更糟**:它报告无意义的波动,却对被测的卡保持沉默。

## ★★★★★ 两个**你自己的**编排器,会用同一个「归属指纹」互相杀(2026-09-13)

清理孤儿 rank 的标准写法是按 environ 指纹只杀自己的:
```bash
tr '\0' '\n' < /proc/$p/environ | grep -qx "HIP_VISIBLE_DEVICES=2,3,6,7" && kill -TERM $p
```
注释写的是 "Our own ranks only" —— 但 **"own" 被定义成了「这组 GPU」**。
当同一组卡上同时有**你自己的两个编排器**时,这个指纹不再区分归属,
后起的那个会精准杀掉前一个正在测量的 server。

实测代价:一个以为已经杀掉的旧编排器活了 5 小时,期间起了 6 次 server(它自己每点都失败、
零产出),每次启动都执行上面这段清理。**一次 3600s 窗口的 server 在第 11 分钟被杀**,
幸存 rank 的集合通信 10 分钟后 watchdog 超时 ⇒ 日志上看起来是一次干净的
`Watchdog caught collective operation timeout`,**很容易误判成被测代码的 NCCL bug**。

⇒ 指纹必须含**编排器身份**(端口 / run-id / 自己的 PID 树),不能只含资源集合。
⇒ 看到「集合通信超时但没有任何其它错误」,**先查有没有第二个进程在杀 rank**,再怀疑代码。

## ★★★★ `pkill -f <字符串>` 会匹配到**别的任务的命令行**(2026-09-13)

`pkill -f 'E2E_SEQUENCE_DONE'` 本意是清掉一个排队任务,结果把**另一个**编排器的父进程也杀了
——那个词只是因为它结尾要 `echo` 这个标记,才出现在它的命令行里。
父进程死后,当时在跑的子进程照常跑完并写出结果,**之后就没人往下走了**,
两个测量点静默地没有执行。

⇒ 杀进程按 **PID**;非要按模式,先 `pgrep -af` 把命中列出来看一眼。
⇒ 与「清 rank 按 environ 指纹而不是按进程名」是同一条:**别用一个会出现在无关命令行里的串做身份判据**。

## ★★★ 起服务的三个守卫(少一个就是几十分钟)(2026-09-13)

1. **挂死判据不能只看「日志静默」**:权重加载/scheduler 起来那段本来就不打印,
   300s 阈值把一个正常启动的 server 判死(它 60 秒后就打印了下一行)。放到 900s。
2. **失败路径必须 teardown**:原来 `exit 1` 直接跳过排空,把误判的 server 留在卡上占着。
3. **就绪后立刻校验资源**,别等门控超时:`max_total_num_tokens` 小于预期就当场失败。
   两秒的 grep,对比的是 480 秒的空门控 + 一整点作废。
