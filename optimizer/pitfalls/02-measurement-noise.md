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
