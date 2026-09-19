---
name: flydsl-mi355-optimizer
description: >-
  MI355X (gfx950) 上优化 FlyDSL fp8/mxfp4/mxfp8 GEMM/attention kernel 的知识库：远程连接/选卡/同步、
  通用优化与 profiling 方法论、**推理服务端 e2e 对照（sglang/AgentX）**、以及大量踩过的坑（tile 尺寸、occupancy、测量噪声、LDS/寄存器、race、
  autotune、各种死胡同）。methodology/ 与 pitfalls/ 与机器无关、可原样复用；本机环境细节（容器/盘/venv/节点）
  在 connection/。⚠️ 只操作你被指定的容器/盘/venv，共享集群上严禁碰不属于你的容器/盘。章节卡按需打开，别全量加载。
when_to_use: >-
  优化/调试/benchmark FlyDSL 或 Primus-Turbo 的 GEMM/attention kernel；连接远端 gfx950 节点跑测；
  诊断性能回退、掉点、race、SNR 崩；查"某个方向是不是已经试过且失败了"；写 autotune dispatch；
  确认 tile/occupancy/LDS 该怎么选；**起 sglang server 做端到端 A/B、判断两臂能不能比、查 decode 一步的时间去向**。凡是碰 gfx950 FlyDSL kernel 性能/正确性的活都先翻这里。
user-invocable: true
---

# FlyDSL MI355X 内核优化知识库

> 🚨 **环境红线（务必先读，违反=破坏别人环境）** 🚨
> **只操作你被明确指定的容器 / host 盘 / venv。** 多租户或共享集群上，**不属于你的容器或盘严禁做任何 rsync / docker exec / 删改**——会覆盖或污染别人的树。
> 跑 campaign / bench 前，逐一核对 `--container` / `--remote-host-root` / `--venv` 全部指向**你自己的**环境；**尤其名字只差一个字符/一个数字的相邻环境最易误伤**（如 `xxx` vs `xxx2`、`codeN` vs `codeN+1`）。
> **本机的具体边界（哪个容器/盘/venv 是你的、哪些相邻环境严禁碰）见 `connection/`。**

MI355X (gfx950 / CDNA4) 上 FlyDSL fp8/mxfp4/mxfp8 GEMM 内核优化的沉淀。内容来自 myskill、
`.claude/memory`、FlyDSL 官方 `.claude/skills` 与 Primus-Turbo `agent/skills`。

## 怎么用

- **🅰️ 动手前第一站 = [00-decision-index](00-decision-index.md)**：**准备试某个杠杆前,先在决策索引 grep 你要做的动作**(如 `grep -i "占用率\|maxnreg\|direct\|atomic\|DMA\|scale_pack\|s_setprio"`)。命中 = 别重踩,点进详卡看根因。这一步治的就是「读了 skill 还去重踩坑」——把查阅强制到**决策的那一刻**,而不是靠开头读一遍的记忆。
- **章节卡结构**：本文件是导航；每张卡是一个主题章节，卡内用 `## 小节` 分隔子话题。按当前任务打开对应卡（Read），别一次性全读。
- 三个文件夹：`connection/`（怎么连上远端跑）、`methodology/`（怎么优化/debug/看利用率）、`pitfalls/`（踩过的坑——**最重要**，动手前先查）。
- **两条线**：kernel 线（GEMM/attention，绝大多数卡）与**推理服务线**（起 server 跑端到端对照）——后者看 **methodology/21（怎么量）+ 22（推理领域知识：一步由什么构成、有哪些杠杆）+ pitfalls/15（harness 的静默错）**，它们和 17（训练 trace）、19（kernel 级部署打分）不是一层。顶层 `00-decision-index` 是这些卡里所有 ❌ 死路的**动作索引入口**。
- 死胡同用 `❌ 别再试` 标注；卡尾 `来源:` 可回溯；卡间用 `见 <路径>` 或 `[[memory-name]]` 交叉引用。
- ★ **上界≠可达铁律**(methodology/03):subtractive/HALF 探针、roofline 峰值率、纸面 op-count 给的都是**上界**,不是可达值——判负/判正前必须 edit→bench 真实现(踩证:HALF_PV +8.7% 假想 → 真 K32-PV −11%)。
- 硬件默认 **gfx950 (MI355X)**；gfx942 (MI300) 仅作跨代对照。
- **远端环境细节（容器 / host 盘 / venv / 节点 / 跳板 / SSH key）= env-specific，见 `connection/`**：`connection/common/` 是通用连接设施，本机环境差异在 `connection/<env>/`。换机器时只替换 `connection/` 这一层，`methodology/` `pitfalls/` `00-decision-index` 与本导航保持通用、原样复用。
- 动手前最短路径：**① grep [00-decision-index](00-decision-index.md) 确认你要试的动作没被判负** → ② 下方「症状→卡」表按现象分诊 → ③ 翻 `methodology/` 找做法 → ④ 翻 `connection/`（选对环境）把代码弄上卡跑测。

## 症状 → 先查哪张卡

| 你看到的现象 | 先查 |
|---|---|
| 想优化但**不知道 bound 在哪** | methodology/03(profiling/regime 分类)→ methodology/04(occ)|
| **掉速**了(某改动 net-negative) | 00-decision-index(是不是已知死路)→ pitfalls 对应域 |
| **occ 上不去 / 想抬占用率** | 00-decision-index §A → methodology/04 → pitfalls/01 |
| **SNR 崩 / 数值错 / 疑似 race** | pitfalls/04(race/vmcnt/正确性)+ pitfalls/02(SNR 判据)+ methodology/02 |
| **store/LDS-bound** 想减 store | 00-decision-index §B → pitfalls/03,05 |
| **quant/scale** 慢 | 00-decision-index §C → methodology/11,12 → pitfalls/05 |
| **测量数字不可信 / DVFS 漂移** | pitfalls/02 → methodology/01 |
| ★★★**目标是「超过 X」,而 X 是别人给的一组常数**(csv/表格/上一任留下的 REFERENCE) | **★methodology/20 —— 先花一次 run 把 X 在本机量出来再调 kernel**;踩证:为一个 1.1% 的「缺口」连关六条轴全净负,量完分母发现对手自己在本机一行都跑不到那列常数(低 1.0~3.4%),头对头实测其实 12/12 全超 |
| ★★**配对 A/B 里两臂的 autotune 候选数不同**(例:定价「放开候选表」时用删候选当对照) | **pitfalls/02 §第二类偏置** —— 只改调优开销就能造出 −0.14%;对照必须**加一个与赢家重复的候选**而不是删候选 |
| ★★**今天的读数能不能信 / 要不要继续做微调** | **pitfalls/02 §尺子地板会漂** —— 空机 A-vs-A −0.009%,有邻居时 −0.098%(单行 −0.9%)⇒ <0.3% 的效应当天不可分辨;每轮都要当天重标 |
| ★★★**接手推理侧优化，不知道一步 decode 的钱花在哪** | **methodology/22** —— 实测分块（GEMM 21% 可用 / MoE 那行作废 / 其余未验证）、三条固定开销约束、已被定价的杠杆清单（别重新发明） |
| ★★★**decode 里某个算子 M 很小（GEMV），想提速** | **methodology/22 §小 M 的 GEMV** —— 一眼判据是 `us/launch` 恒定；修法是**行循环挪进 kernel + 权重常驻寄存器**（1.62×），**别先抬块大小**（撞寄存器，净收益被吃掉）|
| ★★★**要起 sglang server 做端到端 A/B**（推理侧，不是 kernel 侧） | **methodology/21** —— 先确认口径（榜单 TP8/EP1 c1/c2/c4 ≠ 优化目标 TP4/EP4 c8/c10/c14，一眼判据是 verify 行数=6×并发）；再看改动落在 target 侧还是 draft 侧（决定两臂输出会不会分叉） |
| ★★★**e2e 读数"正常但两臂差不多" / bench rc 非 0 但错误率看着没问题** | **pitfalls/15** —— 七个不报错只给错数的坑与它们的识别指纹；先查 server 进程的 environ 是不是加载了被测树 |
| ★★**agentic replay 的吞吐能不能用来判去留** | **methodology/21 §噪声底** —— 同一份代码三次读出 129/306/307 tok/s（**2.4 倍**）；吞吐不可用，延迟分位数分辨率 ~3% |
| ★★★**要让生产者直接吐出消费者的布局**(quant 直出 packed scale 之类) | **methodology/09 §CPU 重放** —— 先确认是纯置换(pack 函数无算术),再在 CPU 上重放 gather、闭式反解、验双射,**最后**才上 GPU |
| ★★**新加的 store 只写了 1/N 的输出 / 高地址全零** | **pitfalls/07** —— 两条:①`num_records` 用了宿主字面量(不进编译缓存 key,后一个形状复用前一个的 bound;**先大后小的顺序下完全隐身**)②那个「全局行」变量其实是 tile 内相对行(SRD 已 per-tile rebase) |
| ★★★**自写 A/B 探针读出好得离谱的数**(几十 %) | **pitfalls/02 §A-vs-A** —— 先把基线当候选跑一遍,读数必须 ≈0;踩证:两臂清缓存次数不对称造出 +54% 假象,加校准后全部反号 |
| ★★**改一个全局常量,只在目标形状上看到收益** | **pitfalls/02 §全局常量** —— 全局量必须全表定价,无例外(两次踩:mid_sync、TPW_MAX) |
| ★★**想对标对手 ISA 的某个统计量**(barrier 数/指令数/LDS) | **methodology/03 §FlyDSL ISA dump** —— 先量自己的那个量;踩证:我们 barrier 已是 6、对手才 4,原计划「照抄对手密度」方向正好反了 |
| ★**周期口径的杠杆全排完了,kernel 还是慢** | **methodology/03 §Power/clock-bound** —— 换能量坐标系;MFMA 原子选择是能量杠杆(单轮 +9.17%),且 `SQ_VALU_MFMA_BUSY_CYCLES` 看不见它 |
| ★**TF/s 读数低得离谱,尤其只有对手的一半** | **pitfalls/02 §逐次 synchronize** —— 先确认不是把**主机入队时间**算进了 GPU 窗口(踩证:650 vs 真值 1150) |
| **build 失败 / 同步出错 / undefined symbol** | pitfalls/08 → connection/common/04 |
| **跨代移植**(gfx942/RDNA)编译/数值错 | pitfalls/11 |
| **flydsl 前端**(tracer/JIT/if-for)报错或静默错 | pitfalls/07 → methodology/09 |
| **attention(dsv4)** 优化 | pitfalls/12(必读)→ 00-decision-index §D |
| **autotune/dispatch/MoE 变长** | pitfalls/06 → methodology/13 |
| 想把 **activation/SwiGLU 融进 GEMM**(fused epilogue)/ 融合核比 plain GEMM 慢想追 parity | **methodology/18**(融合口径 total-vs-total + gated 的 pitch-vs-real + 部署测量)|
| ★**交付写完了,但说不清每块代码值多少** / 有人问「这么多行都必须吗」 | **methodology/16 §收官第四层** —— 逐组件消融定价;验收标准=**关掉它的代价 < BASE-to-BASE 漂移 ⇒ 删** |
| 要**跑/盯/救一场 campaign**(起不来、卡住、要 resume、数对不上) | **methodology/16(整卡读)** |
| 想在**真实 e2e 训练**里验证某 kernel 有没有真跑(别加 print) | **methodology/17**(开 profiler 看 trace)|
| e2e 跑不起来 / 报 `primus_turbo is not importable` / 卡住不动分不清是编译还是死锁 / 找不到某开关 | **methodology/17 §变体 E** —— 开关可能在**未合入分支**(`git log --all -S`);「不可导入」真因常是 **flydsl 版本**(0.1.1.dev409 无 `expr.typing.Vector`);★★★**编译 vs 死锁判据 = JIT 缓存 2min 新增数 + 8 个 rank 栈顶是否各不相同**,别看 GPU util |
| 要拿**真实推理部署的一步**(sglang decode step)当分数 / 微基准赢了但真机变慢 / 同 commit 读数跨度 20%+ | **★★★methodology/19**(热尺子 30.7us vs 冷 45.1us 且排序翻转;多次**取最小**不取中位;`<rest>` 是减法行) |
| 要在远端跑一个本地分支 / 远端那棵树是 dirty 的、怕覆盖 | **本地是唯一权威,远端全是临时的**:rsync/bundle 覆盖那棵共享树,**绝不在远端另建 `/tmp` 副本**(踩过三次,见 methodology/16 §local-first) |
| 要把 **grouped gemm trace 时长换成达成 TFLOP/s** / 算出的数字比 campaign 低一个数量级 | **methodology/17** §FLOPS 换算(★别把每层指派再除 L)|
| Primus e2e 报 `quantize_fp8_tensorwise expected at most 3 argument(s)` | pitfalls/08 §stale-quant-so(重编 .so)|

---

## 顶层
- [00-decision-index](00-decision-index.md) — **动手前先 grep**：全库 ❌ 死路 / ⚠️条件 / ✅开口 的动作索引入口(占用率/LDS/quant/attention/autotune/正确性/测量/环境/flydsl 九域)

## connection/ — 远程连接 / 选卡 / 同步 / 构建（env-specific 层：换机器时替换/重写本层，KB 其余部分通用）

### connection/common/ — 通用连接设施
- [common/01-remote-access-jump-ssh](connection/common/01-remote-access-jump-ssh.md) — 跳板机拓扑与 SSH key 种钥
- [common/02-pick-free-gpu](connection/common/02-pick-free-gpu.md) — rocm-smi 三件套选卡、共存/OOM 判据（sinfo 不可信）
- [common/03-docker-disk-crash-guard](connection/common/03-docker-disk-crash-guard.md) — docker 根盘清理、docker save 暂存坑、大 spill 崩容器防护
- [common/04-build-git-triton](connection/common/04-build-git-triton.md) — 按需 build/清 flydsl cache、git push、triton 版本
- [common/05-reference-misc](connection/common/05-reference-misc.md) — 权限/硬件表/MFMA 延迟/LDS 规格/ISA dump 等参考
- [common/06-inference-server-env](connection/common/06-inference-server-env.md) — **推理线专用**：四层链路怎么逐层验、被测树 ≠ server 加载的树、僵尸 rank、队列 worker 在宿主机

### connection/gpt_oss/ — 本机环境连接卡（容器 / host 盘 / venv 细节，env-specific；换机器时替换本子目录）
- [gpt_oss/01-container-image-nodes](connection/gpt_oss/01-container-image-nodes.md) — 容器 flag/saved tar/生产节点现状 + 本机边界(哪个容器/盘是你的、相邻环境严禁碰)
- [gpt_oss/02-run-and-venv](connection/gpt_oss/02-run-and-venv.md) — docker exec 跑法 + mxfp4/tensorwise venv 隔离
- [gpt_oss/03-sync-and-remote](connection/gpt_oss/03-sync-and-remote.md) — NFS/rsync 规程/FlyDSL 远端为主

### connection/crusoe/ — Crusoe 集群(spur 调度器)连接卡（2026-07-20 搭建）
- [crusoe/01-access-spur-container](connection/crusoe/01-access-spur-container.md) — 登录(csh坑/host key轮询/node-ssh)+login红线(Guardian杀内存)+spur账号关联+**容器=节点dockerd+docker run**(spur --container-image是死路)+docker-exec引号helper+两venv build(clone pip-shebang污染/flydsl 0.2.2 egg坑/git safe.dir)+存储(/shared_nfs/kyle)+已验证dq 1100TF

### connection/smci355/ — smci355 SLURM+docker 节点连接卡（2026-07-20 迁移，chi 跳板挂掉后）
- [smci355/01-node-setup](connection/smci355/01-node-setup.md) — 直连(.ssh_laptop key/IdentitiesOnly)+sbatch(--partition=Compute-DCPT)+起 kyle_dev(rocm/primus:v26.3)+FlyDSL pip 0.2.2(镜像自带 dev409 缺 expr.math)+meta-attn(FAST=0/FLYDSL_EXTRA_SOURCE_DIRS)+两套 turbo(mxfp4 /opt/venv、tensorwise /opt/venv-tw,CK 从 mxfp4 拷)+**★镜像 tar 与恢复:`/data/kyle_0814/docker_images/` 与 NFS `docker_images/` 各一份 19G,容器被重建时 `docker import` 秒回,别再手搭 5 个 venv** + **★★2026-08-31 `kyle_sglang`(lmsysorg/sglang:v0.5.18-rocm720-mi35x):拉镜像前必 `docker logout`、aiter 装我们 fork 要清旧 `.so`、★fork 只能跑 flydsl 0.3.2(0.2.4 上 `buffer_ops` 的 `soffset=None` 直接崩)**

> ⚠️ connection/ 是 env-specific 层:只放**你自己环境**的连接卡;相邻/别人的环境(名字相近的容器·盘)严禁在此混入或触碰,见上「环境红线」。

---

## methodology/ — 通用优化 / debug / profiling / FlyDSL 编程

- [01-optimize-loop-benchmark](methodology/01-optimize-loop-benchmark.md) — 优化主循环、robust timing、测量噪声、公平对比、性能基线 regime；★★★**造尺子前先 grep 有没有现成的**——自搭探针必重踩已记录的坑(把 bf16 喂公共 op ⇒ 量化占 40% 稀释比值;单核紧循环计时 ⇒ 量热数据,应用 STREAM)
- [14-benchmark-shapes](methodology/14-benchmark-shapes.md) — **canonical 跑分 shape**：dense Llama-2 7B/70B + grouped MoE(gpt_oss-20b/Qwen3-235B/DeepSeek-V3, B=8, M∈{1024,2048,4096})
- [02-correctness-race-cache](methodology/02-correctness-race-cache.md) — SNR/det 门控、cross-wave LDS-barrier race 诊断、长 K 非确定、JIT 缓存失效
- [03-profiling-utilization](methodology/03-profiling-utilization.md) — ISA dump 权威、rocprofv3(kernel-trace/PMC)、regime 分类、ATT stall 根因、hotspot 分析、subtractive探针(★上界≠可达铁律)
- [04-occupancy-and-tile](methodology/04-occupancy-and-tile.md) — 512 寄存器共享池、LDS/SGPR limit、256×256 主 tile、小-M BM128、MFMA 数账
- [05-lds-swizzle-prefetch-sched](methodology/05-lds-swizzle-prefetch-sched.md) — LDS 分配/swizzle、L2 swizzle(group_n/XCD)、双缓冲预取、热循环调度
- [06-register-wholeloop-emit](methodology/06-register-wholeloop-emit.md) — AGPR 累加消 spill、whole-loop LDS-feed bound、mxfp4 生产 emit 杠杆
- [07-epilogue-addressing-transpose](methodology/07-epilogue-addressing-transpose.md) — CShuffle/permlane16_swap epilogue、DS_READ_TR、i64 SRD rebase
- [08-splitk-crosslane-lowprec-wgrad](methodology/08-splitk-crosslane-lowprec-wgrad.md) — Split-K、跨 lane 原语/mxfp4 pack、低精度 dequant 融合、wgrad variable-K
- [09-flydsl-authoring](methodology/09-flydsl-authoring.md) — 骨架选型/layout 代数/tracer 控制流/软件流水/复用范式/debug-OOB/atom 扩展
- [10-code-style](methodology/10-code-style.md) — **★code review 硬要求 checklist(简洁/多复用/注释短·英文/禁调试)** + turbo 命名/复用 helper/五层切片
- [11-mxfp8-dense](methodology/11-mxfp8-dense.md) — dual-cast quant/preshuffle/LDS 转置写/scale_pack 预取/whole-loop 移植/全覆盖
- [12-mxfp8-grouped](methodology/12-mxfp8-grouped.md) — grouped var-K wgrad/quant 融合/公平对标/e2e 计时与节点结果
- [13-autotune-fleet](methodology/13-autotune-fleet.md) — Autotune 五原则/生产四轴 never-regress/N-GPU N-agent 调度
- [16-campaign-harness](methodology/16-campaign-harness.md) — **跑自动优化 campaign 必读**:起(launcher 模板/环境红线)、bench 契约(★必须 mirror 部署配置)、盯盘(★轮内唯一活信号=kernel 文件 mtime)、停/续(工作区必须干净)、★坑清单(pgrep 自杀·孤儿 agent 改文件·effort=max 反而超时零产出·换机必重测 base/best·收官 squash 挑错 base+commit 游离·并发 pool 重叠致分数腰斩)、验收(不打扰在跑 campaign 的隔离复测法;★★★★**收官验收面必须大于计分面**=计分面+部署全表按占比加权+兄弟模型三层,本例计分面只占部署时间 39.2%、14% 的回退只在兄弟模型层暴露)
- [17-e2e-kernel-trace-validation](methodology/17-e2e-kernel-trace-validation.md) — **e2e 训练里验证某 kernel 有没有真跑 = 开 torch profiler 看 trace,别加 print**(print 探针可能在死路);Primus mock 冒烟 + profiler 步骤、trace 落地目录(被 patch 改写)、gzip+json 解析(glob `rank[0]` 字符类坑)、kernel 名→后端对照表、K-pad 铁证(pad-quant kernel 只可能来自 syncv3);MoE 走 ragged 派发不经 PrimusTurboGroupedLinear/公共 grouped_gemm_fp8
- [19-deployment-step-scoring](methodology/19-deployment-step-scoring.md) — **★★★拿真实部署的一步当分数**:热/冷尺子(同 kernel 30.7us vs 45.1us,排序翻转)、部署权重(56 形状均摊 +10% 而加权 −1%)、被抢时**取最小**(2 次取中位曾把基线记高 15%)、GLM-5.2 四个 decode GEMM 形状与调用数、`trace_step` vs `profile_decode` 的区别、四个运行期坑(spawn 孤儿/`/dev/shm` 攒死/别的容器孤儿/自杀式 pkill)；★★★**尺子必须走部署真正走的那条 kernel**——一个没传的 `num_cu` 让 20 轮 campaign 每一轮的「赢」在真路径上都是「输」(bench −5.2% vs 真训练 −0.84%)；拿真训练当 bench 的四个实现坑(步时间只能从 profiler trace 拿/冷缓存 autotune 落进 profile 窗口给出 2423ms 假值/loss 抖 3.2% 不能当紧门/ProfilerStep 每步发两次)
- [20-verify-the-denominator](methodology/20-verify-the-denominator.md) — **★★★★★对标类目标:先验分母,再调 kernel**。分母是别人给的常数时,先用**对手自己的 API + 对手自己的调用约定 + 你这套 harness** 量一遍;四个必须动作(①按对手的预处理契约 ②确认它走的是快路不是 fallback ③同进程 ABBA ④每行带 A-vs-A 校准);何时该怀疑分母(缺口极稳 + 全表等幅 + 所有参数轴推不动)
- [99-misc](methodology/99-misc.md) — 其它零散事实

---

## pitfalls/ — 踩过的坑（最重要，动手前先查）

- [01-tile-occupancy-register](pitfalls/01-tile-occupancy-register.md) — **256×256 唯一可行 tile**、512 合并寄存器池、量子边界、maxnreg/预取 spill 死路
- [02-measurement-noise](pitfalls/02-measurement-noise.md) — 噪声地板/DVFS/掉频/并行口径/host 开销/虚高 TF/SNR 掩盖 race;★★★★**自家孤儿探针空转 18h 把尺子从 0.29% 污染到 >2%、结论直接翻转**(+0.48%→−0.65%)⇒ 关键 A/B 前先扫孤儿;★★本地调用被中断后远端派发的活不会跟着死,必须去容器收尸；★★★**同名多形状必须求和不能取中位数**(nt 名下挂 fc1+fc2 两形状,中位口径让同一改动的 Δ 跨轮极差 41-68us 且归因翻转,求和后降到 15-22us)；★★★**别用标称峰值定价 headroom**——同日三次全被实测推翻(HBM 标称 8→实测 5.95,cast 其实在地板;fp8 GEMM 5000→实测 3151,grouped 其实 71-75% 不是 45.7%);参照系必须与被测同形态
- [03-lds-l2-datapath](pitfalls/03-lds-l2-datapath.md) — LDS 容量/bank/带宽封顶、A/B 直读掉 2.5×、prefetch 何时有害、torch copy_ 假天花板
- [04-race-vmcnt-correctness](pitfalls/04-race-vmcnt-correctness.md) — partial-drain/spill race、SRD 寻址、HW-walled 死路、SNR gate、attention 中性值
- [05-mxfp4-mxfp8-deadends](pitfalls/05-mxfp4-mxfp8-deadends.md) — occ 天花板、epilogue store 暴露、wgrad feed-bound、mxfp8 scale 投递税/WL 死路
- [06-autotune-dispatch](pitfalls/06-autotune-dispatch.md) — 禁 id(tensor) cache、别接不采纳候选、persistent 选择、缓存增益上限;★★★★**dispatch 的 first-call race 不许用静态模型替代**(为省几微秒 host 时间换成模型排序,计分面全绿、计分面外的 llama wgrad 掉 14%;恢复 race 即全好)——静态门控分析只圈嫌疑、定罪必须实测
- [07-flydsl-frontend-tracer](pitfalls/07-flydsl-frontend-tracer.md) — JIT 缓存不失效假象、字面 if/for 坑、前端编写陷阱、ThrVal/atom layout 静默错
- [08-sync-build](pitfalls/08-sync-build.md) — rsync/git 远端同步陷阱、build/环境坑（HK undefined symbol/LLVM OOM）
- [09-isa-profiling](pitfalls/09-isa-profiling.md) — sched_* 计数/s_setprio、ATT vs PMC 分工、code.json AGPR-blind
- [10-code-style-deploy](pitfalls/10-code-style-deploy.md) — 源码禁调试痕迹、放行硬门禁+绿测先证伪、backend 签名/cache key
- [11-porting-encoding](pitfalls/11-porting-encoding.md) — FNUZ/OCP 跨代编码门、CDNA4/RDNA 移植、"是不是自己改坏"诊断捷径
- [12-dsv4-sparse-mla-attn](pitfalls/12-dsv4-sparse-mla-attn.md) — **DSV4 sparse-MLA fwd/bwd 优化必读**(已整理去矛盾)：cr语义/当前实测(fwd mean 509.7 pro cr4 940 / bwd mean 319.5)/已部署赢(fast_path·BLOCK_H128·PV-K32·delta融合·rtr·K32-MFMA)/当前bound(occ统一512池·dQ非tr16-read)/已判负(s_setprio·XPIPE·DMA·QB·K32-PV·dual-KV)/开口(小topk interm·banded-SWA·hybrid scatter)/flydsl专属坑
- [99-misc](pitfalls/99-misc.md) — 其它零散事实
