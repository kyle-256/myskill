# 17 — e2e 训练里验证「某 kernel 到底有没有真跑」= 开 profiler 看 trace,别加 print

> 类别: 方法论 · 主题标签: e2e-validation, torch-profiler, trace, kernel-dispatch, primus-mlperf, K-pad, grouped-gemm, mock-smoke, 别加print

**场景**:你在 turbo/FlyDSL 里做了个 kernel 优化(如 grouped GEMM K-pad),想在**真实 e2e Primus
MLPerf 训练**里确认它真被派发(不是单测、不是理论)。**做法:开 torch profiler 抓 trace,直接看
kernel 名字。别在 python 侧加 print 探针**——派发路径可能跟你以为的不是同一条,print 全哑火白忙。

## ★★ 头号教训:print 探针可能在死路,trace 才是 ground truth

- 真事(2026-08-03,gpt-oss-20b MLPerf,chi2798):想验证 syncv3 K-pad grouped GEMM 是否触发,先在
  `PrimusTurboGroupedLinear.forward_internal` / 公共 `grouped_gemm_fp8` 里加了一圈 `[GL-BRANCH]`
  `[GG-ENTRY]` `[KPAD-FIRE]`,**全为 0**,连无条件一次性打印的 `[GL-INIT]`(类 `__init__`)都 0
  → 误判"K-pad 没触发",耗了好几轮 sync/重跑。
- 开 profiler 一看 trace:`kernel_grouped_nt_persistent` / `quantize_tensorwise_pad_kernel` **明明在跑**。
- 根因:MoE experts 的 grouped GEMM 走的是 **ragged 派发路径**(`use_turbo_ragged_grouped_gemm:true`),
  **不经** `PrimusTurboGroupedLinear`(该类根本没实例化),也**不经**公共 `grouped_gemm_fp8` wrapper。
  我 print 加错了代码路径。**结论:验证 kernel 是否真跑,第一手段是 trace,不是 print。**

## ★★ 第二头号教训:e2e 崩了却看不到 traceback —— 三层遮蔽,真日志在 Primus 自己的 rank 目录

真事(2026-08-13,syncv4/smci355):`primus-cli direct` 跑 mlperf,rank 崩了只给 `exitcode: 1`,
stdout/stderr 干干净净全是 MLLOG,**连跑 6 次都以为是"静默崩溃"**,一路误判到怀疑硬件/通信。
三层遮蔽依次是:

1. **`--local-ranks-filter 0`**:`runner/primus-cli-direct.sh` 按 NODE_RANK 硬算(**无 env 开关**),
   非 rank0 的输出根本不进 console → 崩在 rank2 就只剩一个退出码。
2. **torchrun 重定向**:加 `--redirects=3 --log-dir=<dir>` 能拿到 per-rank 文件,但**崩溃 rank 的
   `stderr.log` 是 0 字节**,`stdout.log` 停在 Gloo 那几行。⚠️ 非 rank0 本来就被 Primus 的 rank0-only
   logging 静音,**几百字节是"正常静默"不是"卡在这里"的证据**——我据此误判过一轮。
3. **addr2line**:`TORCH_SHOW_CPP_STACKTRACES=1` 只打出 `symbolizing C++ stack trace for exception;
   if this hangs, rerun with TORCH_DISABLE_ADDR2LINE=1`,符号化耗 56s 后**异常正文整个丢失**。
   要 C++ 栈必须同时设 **`TORCH_DISABLE_ADDR2LINE=1`**。

**✅ 真 traceback 一直躺在 Primus 自己的 loguru 落盘里**(和 stdout 完全无关):
```
<PRIMUS_PATH>/output/<team>/<user>/<exp>/logs/pre_trainer/rank-<N>/{info,debug,warning}.log
# 实例: output/amd/root/gpt_oss_20b/logs/pre_trainer/rank-0/info.log
```
⚠️ **`error.log` 是 0 字节**——名字最像的那个恰恰是空的;完整 `RuntimeError`+调用栈在 **`info.log`/
`debug.log`**(`warning.log` 有摘要首行)。**e2e 排障第一步就 tail 这几个文件,别在 stdout 里刨。**

**两个提速手段**:
- **不改 Primus 也能给 torchrun 加参数**:`runner/primus-cli-direct.sh` 把 `${LOCAL_RANKS:-}` **不带引号**
  拼进 torchrun argv(注释明说留给 word-splitting)→ `export LOCAL_RANKS="--redirects=3 --log-dir=<dir>"`
  即可加 per-rank 日志,无需 patch 仓库。
- **single-process direct**:绕开 torchrun+primus-cli,自设 `RANK/WORLD_SIZE/LOCAL_RANK/MASTER_*` 后直接
  `python -u -m primus.cli.main train pretrain --config $EXP --backend_path <Megatron-LM>`,输出不经任何
  重定向。**1 卡复现出同一个崩溃 = 立刻排除通信/多卡因素**(本例正是这样把范围缩到 kernel 层),
  且此时唯一的 rank 就是 rank0,不受第 1 层遮蔽。

## 复现步骤(mock 冒烟 + profiler)

1. **run wrapper**(本地 `sync/_run_gptoss_mlperf.sh`,NOT git-add;容器内路径
   `/workspace/code/syncv3/_mlperf_run/_run_gptoss_mlperf.sh`)。mock 段设:
   ```sh
   export PRIMUS_MOCK_DATA=true        # 需 yaml 里 mock_data: ${PRIMUS_MOCK_DATA:false}
   export PRIMUS_TRAIN_ITERS=6         # 必须 > PROFILE_STEP_END
   export PRIMUS_PROFILE=True          # yaml 里 profile/use_pytorch_profiler 已是 ${PRIMUS_PROFILE:False},无需改 yaml
   export PRIMUS_PROFILE_STEP_START=2
   export PRIMUS_PROFILE_STEP_END=4    # schedule: wait=start-1, warmup=1, active=end-start
   ```
   backend 用 FLYDSL:`export PRIMUS_TURBO_GROUPED_GEMM_BACKEND=FLYDSL`;PYTHONPATH 前置
   `syncv3/Primus-Turbo` 让 K-pad turbo 胜出 editable(见 connection/gpt_oss/04)。
2. **跑**(见 connection/gpt_oss/02、03):
   ```sh
   ./.ssh-chi.sh root@chi2798 "docker exec mlperf_gptoss bash -lc '
     rm -rf /root/.flydsl/cache
     find /workspace/code/Primus/primus -name __pycache__ -prune -exec rm -rf {} +
     cd /workspace/code/syncv3/_mlperf_run
     nohup bash ./_run_gptoss_mlperf.sh mock > ./_launch_mock.log 2>&1 &'"
   ```
   ~4–5min 跑完 6 iter(8 卡)。跑之前先确认没别人在用这些卡(共享节点红线,见 SKILL.md 顶部)。
3. **trace 落地**(★不是 yaml 的 tensorboard_dir):`tensorboard_path_patches.py` 会改写到
   `/workspace/code/Primus/output/amd/<team>/<user>/gpt_oss_20b/tensorboard/`,文件名
   `primus-megatron-exp[gpt_oss_20b]-rank[N].<ts>.pt.trace.json.gz`(每 rank 一个,~14MB gz)。
   `profile_ranks` 默认 `[0..7]`;只读 rank0 即可。

## 解析 trace(在容器上 python 跑,别把 14MB 拖回本地)

```python
import gzip, json, glob, collections, re
fs=[f for f in glob.glob(".../tensorboard/*.pt.trace.json.gz") if "rank[0]" in f]  # ★见下坑
d=json.load(gzip.open(fs[0])); ev=d["traceEvents"]
dur=collections.defaultdict(float); cnt=collections.Counter()
for e in ev:
    if e.get("cat") in ("kernel","Kernel") and "dur" in e:
        dur[e["name"]]+=e["dur"]; cnt[e["name"]]+=1
# 按关键字 group|gemm|quant|fp8|flydsl|Cijk|pad 过滤 names.most_common() 看谁在跑 + 累计 us
```
- **坑**:`glob("*rank[0]*")` 里 `[0]` 是 glob **字符类**(匹配字符 '0'),匹配不到文件名里字面的
  `rank[0]`。用 `glob("*.pt.trace.json.gz")` 全取 + python `"rank[0]" in f` 字符串过滤。
- kernel 事件 `cat` 是 `"kernel"`(ROCm 上有时首字母大写,两个都收);`dur` 单位 µs。

## kernel 名字 → 哪个后端(gpt-oss MoE fp8 tensorwise 实测对照)

| trace 里的 kernel 名 | 来自 | 含义 |
|---|---|---|
| `kernel_grouped_nt_persistent_*` | **syncv3 FLYDSL** | grouped GEMM **fwd NT**(K-pad 目标路径) |
| `kernel_grouped_nn_persistent_*` | syncv3 FLYDSL | grouped **dgrad** |
| `kernel_grouped_tn_wgrad_4wave_*` | syncv3 FLYDSL | grouped **wgrad**(4-wave) |
| `primus_turbo::quantize_tensorwise_pad_kernel<...float8_e4m3_t...>` | **syncv3(独有)** | **K-pad pad-align 量化**;容器 editable 1c5a3cc2 turbo 是 3-arg 无 pad,此 kernel 只可能来自 syncv3 → 是 K-pad 真触发的铁证 |
| `compute_group_offs_device` / `_grouped_gemm_output_tail_kernel` / `fused_scaling_group_sum_routing*` | syncv3 turbo | grouped 配套 |
| `Cijk_..._F8BS_...`(如 MT288x256x128) | hipBLASLt | dense/attention 的 fp8 GEMM(**不是 MoE**) |
| `cast_transpose_optimized_kernel` / `rocm_cast_only_kernel`(`te_hip_fp8_e4m3`) | Transformer Engine | dense/attention 的 fp8 量化 |
| `ck_tile::...FmhaFwdKernel...` | CK | attention(`MLPERF_ENABLE_FWD_ATTN_ASM=0` 时的回退) |

**判读**:MoE grouped GEMM 全在 syncv3 FLYDSL(nt/nn/tn-wgrad),K-pad quant 喂在 **fwd NT** 上 →
破了"fwd 默认 NN、K-pad 不触发"的担心;dense/attn 的 fp8 仍走 TE+hipBLASLt——分工符合预期
(只有 experts 用我们的 grouped GEMM)。参考量级(rank0 / 2 active step / 24 层):
nt 96×1270.9µs、nn 96×1155.4µs、tn-wgrad 96×1424.9µs、pad-quant 288×264.7µs。

## 别改 Primus(把改动关进 wrapper env / 一行 yaml 插值)

- 验证只需 mock + profiler,**这些全靠已有的 `${PRIMUS_*}` yaml 插值 + wrapper env 打开**,不用改 Primus
  逻辑。唯一需要的 yaml 改动是 `mock_data: ${PRIMUS_MOCK_DATA:false}`(default 保留生产行为);
  `profile`/`use_pytorch_profiler`/`profile_step_*` 本来就是插值,零改动。
- 别为了 debug 往 Primus 里塞 print(既在死路、又污染别人分支)。要看内部就开 profiler / 调
  `stderr_sink_level`(yaml `${PRIMUS_STDERR_SINK_LEVEL:ERROR}` 可临时设 INFO 看 `[Patch:...]` 决策日志),
  用完回退。

## ★★ regime 判定:配置(EP/G/per-expert M)从 trace Input Dims 实读,别推断(2026-08-09 权威 trace)

算达成 TF 之前,**先钉死 grouped GEMM 每次 call 到底跑多大**——这决定了你该在哪个尺寸上优化。gpt-oss-20b 至少有 **3 个不同 regime,历来被串台**:

| regime | 来源 | EP | G(本地专家) | per-rank Mtot | **每专家 M** |
|---|---|---|---|---|---|
| **权威部署** | 老师 fp8-49457 trace | 1 | 32 | 131072 | **4096** |
| 我的 8卡 e2e | [[project_gptoss_e2e_trace_grouped_flops]] | 8 | 4 | 32768 | 8192 |
| E32 大M campaign | [[project_gptoss_e32_largeM_campaign]] | 1 | 32 | ≈2.36M | 73728 |

**★ 决定性方法 = 用 External id 把 GPU kernel 配回 CPU aten op 读 `Input Dims`(不是从 GBS/S/EP 反推)**:torch chrome trace 里 GPU kernel event 带 `External id`,CPU aten op(如 `FP8GroupedGemmTensorFunc`/grouped gemm wrapper)带 `Input Dims`+`Input type`;按 External id 配对就能读到**精确张量维**——A=[Mtot,K]、B=[G,N,K]。这是**实读不是推断**,一次定死 EP/G/Mtot/per-expert M:
- fwd NT gate_up A=[131072,2880] B=[32,5760,2880] → **Mtot=131072、G=32 → per-expert 4096**(EP1)。
- per-expert M = Mtot / G;**EP 决定 G**(G=E/EP,本地驻留专家数);Mtot = tokens/rank × topk = (S×MBS)×topk。

**★ 权威 fp8-49457 达成(详 [[project_gptoss_teacher_trace_fp8_49457]])**:⚠️**median 被 e2e 并发争用污染 5-9%(单 profiled step 24 call/shape,长尾拖低),要核速看 min**。TF@min:fwd gu3059/dn2677、dgrad gu2910/dn2603、**wgrad gu2350/dn2306**(median gu2839/2168 等偏低)。聚合 2198 TF/s(summed,含争用)=用户口中"2200多T"、median steady 2483。**"2200多T"= 6-shape 聚合被 wgrad 和 down 投影(N=2880 窄方形)拉低 + 争用压 median**。⚠️**wgrad@per-exp 4096 从没专门调过**(短板 2306-2350)——E32 campaign 是 per-exp 73728(18×)、EP8 是 G4;要打短板得在 per-exp 4096 重打 wgrad(+down N=2880)。★复验教训:判"内核是否回退/负优化",**config/旋钮层可扫**(同一份数据比 config 相对快慢:部署 autotune 距全扫最优 1.7%、bm128 更慢 = config 没选错);但**结构性回退别用 zeros/随机字节喂 fp8 比"数据幅度"来证**——那是伪证(见 pitfalls/02 §fp8-data-magnitude-ban)。要判结构性,取旧版内核和当前版**同一份真实数据同进程 A/B**,拿不到旧版就存疑。

**教训**:campaign/单测尺寸必须==部署 regime,否则数字系统性偏高(per-exp 越大摊薄越充分:campaign 73728 的 2810 ≠ 部署 4096 的 2171)。bench 前先解 trace Input Dims 确认 per-expert M,再选尺寸。见 [[feedback_grouped_gemm_bench_pitfalls]]。

## ★ 从 trace 时长换算 grouped GEMM 达成 FLOPS(手算口径 + 头号陷阱)

trace 只给 kernel **时长**(µs),没有 FLOP。要得到「达成 TFLOP/s」= 手算该 kernel 每 iter 的 FLOP ÷ trace 时长。gpt-oss-20b MoE 口径:

**维度**(从训练 args log grep:`grep -aoE "(seq_length|hidden_size|moe_ffn_hidden_size|num_experts|moe_router_topk|num_layers|global_batch_size|expert_model_parallel_size) [.]+ [^ ]+"`):
H=hidden=2880、I=moe_ffn_hidden=2880、E=32、topk=4、L=24、S=4096、GBS=64、EP8/TP1/PP1(world=8 → DP=8),dropless(所有 token 都算,无 capacity drop)。

**FLOP 账**:
- 单 token-expert 指派 fwd = `6HI`(gate_up=2·H·2I=4HI ⊕ down=2·I·H=2HI);dgrad=wgrad=fwd(反向两段各等于 fwd)。
- **★★ 头号陷阱:每层指派 = `GBS×S×topk` = 262144×4 = 1,048,576,这是「每一层」的量,不是全 L 层总和** —— 每个 token **在每一层都重新路由一次**。**千万别再除以 L**。(我第一版栽在这:误算 `assign/EP/L` 把每专家 M 从 32768 缩成 1365,算出假的 114 TF/s,被用户当场抓包。)
- 每层每卡指派 = 1048576 / EP = **131072 → 每专家 M ≈ 32768(大 M!)**。
- fwd 单卡/iter = `131072 × 6HI × L(24)` = **156.55 TFLOP**;三段(fwd+dgrad+wgrad)合计 **469.65 TFLOP/卡/iter**(全 8 卡 3757)。

**★ 3 个 kernel 名 = 6 个逻辑 GEMM(gate_up + down 复用同一 kernel)**:MoE 每层有两个投影——gate_up(N=2I=5760)和 down(N=H=2880),但 fwd 的这两个 GEMM **launch 的是同一个 `kernel_grouped_nt_persistent_0`**(只改 M/N/K + group offsets),trace 按名字聚合就并成一条。dgrad/wgrad 同理。所以 trace 里 3 个名字 = 6 个逻辑 GEMM,每名字每层跑 2 次 → `x96 = 2 step × 24 层 × 2(gate_up+down)`。**别看到 3 个就以为漏了**。要拆成 6 个:按单次 dur 分两簇(大簇=gate_up N 大,小簇=down),实测 nt/nn/tn 各干净分成 48+48。拆出的 6 个达成(fwd gate_up 3043 / down 2453、dgrad gu 2759 / dn 2391、wgrad gu 2864 / dn 2700 TFLOP/s)合并回去恰好==下面 3 个聚合值。注意 **gate_up 比 down 快**(N=5760 复用好,单次时长比只 1.6–1.9× 不是 FLOP 的 2×);**dgrad·down 最低 2391**(dgrad transpose-load 短板 × 窄-N,见 [[project_tw_nn_beats_mxfp8_campaign]])。

**trace 时长 → 达成**(下面按 kernel 名聚合 = gate_up+down 合并;时长是 2 个 profiled step 的和,÷2 得每 iter):

| 核 | 每 iter/GPU 时长 | FLOP/iter/GPU | 达成 |
|---|---|---|---|
| fwd `kernel_grouped_nt_persistent_0` | 55.57ms | 156.55 TFLOP | **2817 TFLOP/s** |
| dgrad `kernel_grouped_nn_persistent_0` | 59.65ms | 156.55 TFLOP | 2624 TFLOP/s |
| wgrad `kernel_grouped_tn_wgrad_4wave_0` | 55.77ms | 156.55 TFLOP | **2807 TFLOP/s** |
| 三段合计 | 170.99ms | 469.65 TFLOP | 2747 TFLOP/s |

**★ 交叉验证 = 数字对得上才算对**:e2e **wgrad 2807 TFLOP/s ≈ kernel-only campaign 的 min_wgrad 2810**(见 [[project_gptoss_e32_largeM_campaign]])→ 坐实真训练里 grouped GEMM 就跑在 largeM regime,内核性能 == 单测 bench。若你算出的达成值比 campaign 记录低一个数量级(如 ~110 而非 ~2800),**先怀疑 FLOP 少乘了 L 或 M 算小了**,不是内核真慢。dgrad(nn)2624 略低是 transpose-load 短板(见 [[project_gptoss_nt_fwd_dgrad_opt]] / [[project_tw_nn_beats_mxfp8_campaign]])。K-pad 另带 e4m3+e5m2 两个 `quantize_tensorwise_pad_kernel` ~27.6ms/iter/GPU(≈三段 GEMM 时长的 16%,非平凡)。

## 变体 B — Crusoe 文件队列 + 非-mlperf `gpt_oss_20B-FP8-pretrain.yaml`(2026-08-08 真跑)

跟上面 chi2798/mlperf 分支那套的**关键差异**(换环境时照这条走):

1. **环境 = Crusoe 177 kyle_box**(见 connection/crusoe/01):没有 login→节点直连,靠**文件队列**——`ssh xianzhao@crs-...spur-013` 把 `.job`(内容=`docker exec gpt-oss-docker bash ...`)写进 `/shared_nfs/kyle/q/`,host q-agent 跑完出 `.out`/`.rc`。容器内 `/workspace/code` == host `/shared_nfs/kyle`。venv = **`/opt/venv-syncv3`**(直接用绝对路径 python,**严禁 `source activate`**)。
   - ⚠️ **q2-agent 变体只写 `.done`(=job 文本回显)+`.rc`,丢弃 stdout** → 分析脚本必须**自己把输出重定向到 NFS 文件**(如 `> /workspace/code/gptoss_trace_kpad/_analyze.txt 2>&1`)再从 login 侧读,别指望 agent 帮你捕获。
   - ⚠️ 本地 Bash tool **没挂 /shared_nfs** → 建远端文件用 `ssh $L 'bash -s' <<'EOSSH'` heredoc 或 `scp`,别 `cat > /shared_nfs/...`(本地无此路径)。
2. **前置:syncv3 turbo `.so` 可能是陈旧 3-arg** → e2e 首个 fwd MoE quant 报 `quantize_fp8_tensorwise() expected at most 3 argument(s) but received 4`。**先重编**(hipify 从 4-arg 真源 `bindings_pytorch.cpp` 重生),见 pitfalls/08 §stale-quant-so 与 [[project_syncv3_so_rebuild_4arg]]。
3. **★ profiler 硬门 = yaml `profile: true`,不是 `use_pytorch_profiler`**:此 yaml 的 profiler 相关键是**硬编码字面量**(不像 mlperf yaml 走 `${PRIMUS_PROFILE}` 插值),且 megatron `training.py` 门在 `args.profile`。所以要 **sed 改 yaml**:`profile: false → true`、`profile_step_start: 6`、`profile_step_end: 8`(active=end−start=2 步)、`train_iters ≥ 10`(> profile 窗口)、`fp8_recipe: tensorwise`(turbo 只吃 current-scaling)、`global_batch_size: 64`、`use_turbo_grouped_mlp: true`。`mock_data: true` 此 yaml 已默认。
4. **★ trace 落 image 层不是 NFS**:此 yaml 的 `tensorboard_dir` 解析到 `/workspace/Primus/output/tas/qyy/gpt_oss_20B-pretrain/tensorboard/`,而 **`/workspace/Primus` 是烘进容器镜像的写层,不是 /shared_nfs 挂载** → login 侧读不到。**必须 `cp` 到 `/workspace/code`(=NFS)** 才能从 login 读。`use_gzip:false` → 文件是 `*.pt.trace.json`(**非 .gz**,rank0 ~310MB),此配置只 emit rank0。
5. 启动 = `torchrun --nproc_per_node=8 ... primus/cli/main.py train pretrain --config $Y`,`setsid` detached(长跑别走同步等待被超时杀)。10 iter ~ 90s,loss 11.8→11.36,~590 TF/s/GPU(整模型 6ND util)。

复现脚本(NOT git-add)= `/shared_nfs/kyle/_smoke2_body.sh` + `_smoke2_launch.sh`;profiler patch(可选,给 fallback dir)= `/workspace/Primus/primus/backends/megatron/patches/torch_profiler_patches.py` 加 `PRIMUS_TRACE_DIR` env fallback(但 profile:true 后 tensorboard_dir 会自动生效,通常用不上 fallback)。

## 变体 C — smci355/syncv4 跑 **官方 mlperf EP1 config**(2026-08-13 跑通)

环境 = smci355 `n01-25` 容器 `kyle_dev`、venv `/opt/venv-syncv4`、repo `/workspace/code/syncv4/{Primus,
Primus-Turbo}`(见 connection/smci355/01、gpt_oss/04)。★**Primus 也要 sync**:本地 canonical `sync/Primus`
→ `sync/syncv4/push_primus_smci.sh`(与 Primus-Turbo 各一份远端拷贝,一个环境改 HEAD 不影响另一个)。
跑通口径:**8 卡 EP1/TP1/PP1、GBS32/MBS4、mock、8 iter、FLYDSL grouped GEMM** → `torchrun finished
successfully (code 0)`,284s,MLLOG `overall_throughput 19.36`;`eval_accuracy: NaN` 是 **mock 数据的必然
结果**(合成 token,loss 无意义),不是 bug。

这份 config(`examples/mlperf/gpt_oss_20b/config_MI355X_1x8x1_tp1pp1ep1_gbs32.sh`)是给**官方容器布局**
写的,搬到自建环境**每条都会直接崩**,照抄这张清单:

1. **`.so` 陈旧 3-arg 会在新环境复发**:syncv4 是 `cp -a` 老容器来的 → 首个 fwd MoE quant 就报
   `expected at most 3 argument(s) but received 4`。修法见 pitfalls/08 §stale-quant-so。
   ⚠️ **smci355(256 核 / MAX_JOBS=96)实测重编 ~18min**,不是 Crusoe 记的 6–7min,**别按 7min 设超时**。
2. **Megatron-LM submodule 必须 init**:`third_party/Megatron-LM` 是空 gitlink,否则 `megatron.core`
   import 不到。`git submodule update --init --depth 1 third_party/Megatron-LM`(37M)。
3. **config 硬编码部署路径**:`PRIMUS_PATH=/workspace/Primus`、`EXP`、`PYTHONPATH` 全在 config 里 export
   → **必须 source 之后再覆盖**,顺序反了全部无效(`run_and_time.sh` 还会 `cd $PRIMUS_PATH/...`)。
4. **config 拼 `${PYTHONPATH}` 无守卫**(其 line 20)→ wrapper 带 `set -u` 会 `unbound variable` 当场退。
   source 前先 `export PYTHONPATH="${PYTHONPATH:-}"`。
5. **LR 三值联动**:`PRIMUS_LR_DECAY_ITERS` 是 source 时按 `TRAIN_ITERS-WARMUP` **算死**的。冒烟只改
   `PRIMUS_TRAIN_ITERS=8` 会留下 decay=1199872 > train_iters → megatron 拒绝。**三个一起改**。
6. **`mock_data` 这版仍是硬编码 `false`**(不像 `profile`/`train_iters` 已插值):节点没有 c4 数据集就得
   加插值 `${PRIMUS_MOCK_DATA:false}`——**与变体 A 相同的、唯一需要的 yaml 改动**(default 保留生产行为)。
7. ★**grouped GEMM backend 默认是 `triton`**(`${PRIMUS_TURBO_GROUPED_GEMM_BACKEND:-triton}`)→ 要测
   FLYDSL **必须显式 export**,否则你以为在测 FLYDSL、其实跑的是 triton(和 04 卡"跑前先验 import 路径"
   同一类自欺)。
8. `MLPERF_RUNTIME_SERIES=v26.3`(容器镜像 rocm/primus:v26.3);默认 v26.5 分支会去调
   `/opt/mlperf-gpt-oss-20b/prewarm_attention.py`,该路径在自建容器不存在。

wrapper(NOT git-add)= `sync/syncv4/_run_mlperf_gptoss.sh`,四模式 `smoke|profile|full|direct`;
`SMOKE_GPUS=1/2/4` 缩世界规模调试(GBS32/MBS4 在各规模都整除,grad-accum 吸收差异,只有通信路径变)。

## 变体 D — smci355/`kyle_train` 跑**真 C4** EP1 down-padk,复现 huangwei 0824 trace(2026-08-25 跑通)

**目标**:在真实 e2e 训练里复现 huangwei 0824 的 down-padk fp8 grouped GEMM trace
(本地参照 `trace/huangwei_0824/gptoss-fp8-down-padk-rank0-step100-gpu-only.pt.trace.json.gz`,
rank0/step100/GPU-only)。跟变体 C 的差异是**用真 C4 数据(非 mock)+ venv-syncv3(近期 down-padk 优化版)**。

**环境 / transport**:节点 `smci355-ccs-aus-n02-29`,容器 **`kyle_train`**(C4 挂 `/data`、tokenizer `/model`),
venv **`/opt/venv-syncv3`**(用户硬令:syncv3 的 Primus-Turbo 才是近期优化的 down-padk 版)。传输
= `kyle_wgrad_work/kt.sh '<cmd>'`(base64 过 login `n01-29` → 节点 `n02-29` → `sudo docker exec -i kyle_train`)。
容器真实脚本在 **`/workspace/code/syncv4/`**(不是本地镜像 `gpt_oss_docker/sync/syncv4/`——两者不同挂载,
Edit 工具改本地这份到不了容器,要么 sync 要么在容器里 `sed`)。

**跑通口径**:8 卡 EP1/TP1/PP1、GBS32/MBS4、**真 C4**、110 iter、profile step 100 →
`torchrun finished successfully (code 0)`、108.45s、samples 3520(=110×32)、零报错。
config 清单照变体 C 第 1–8 条(`.so` 4-arg、Megatron submodule、source 后覆盖、LR 三值联动、
`MLPERF_RUNTIME_SERIES=v26.3`),**再叠加下面 D 专属三条**:

1. ★**`use_turbo_grouped_gemm` 默认 false → 必须开**:mlperf yaml line 187 硬编码 `false`(走 hipBLASLt),
   改成插值 `use_turbo_grouped_gemm: ${USE_TURBO_GROUPED_GEMM:false}`(default 不变),wrapper 里
   `export USE_TURBO_GROUPED_GEMM=true`。**不开的话 grouped GEMM 全走 hipBLASLt(`Cijk_*`),trace 里
   一个 flydsl 核都没有**——第一次就栽这。
2. ★★**必须强制 `export PRIMUS_TURBO_GROUPED_GEMM_BACKEND=FLYDSL`**:只开 `use_turbo_grouped_gemm`
   还不够——**syncv3 全训练路径的默认 dispatch 会静默落到 Triton**
   (`_grouped_fp8_persistent_gemm_kernel`/`_grouped_variable_k_gemm_kernel`,名字全不对、比 flydsl 慢 2–5.5×;
   fallback 警告在 `torch.compiler.is_compiling()` 下被吞)。**这就是变体 C 第 7 条那句"默认 triton"在真
   C4 路径的复发**。强制 FLYDSL 后 grouped GEMM 全出 down-padk 核,`can_handle` 全通过、**不 raise**。
   自欺检查:别信"我开了 turbo",一定 trace census 看核名。
3. **DATA_CACHE_PATH 要可写**:`export DATA_CACHE_PATH=/workspace/code/syncv4/_datacache`
   (Megatron GPTDataset 建索引缓存,默认落只读处会崩)。真 C4 用 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。

**LR/profile 旋钮**(到 step 100):`PRIMUS_TRAIN_ITERS=110`、`LR_WARMUP=10`、`LR_DECAY=100`(三值联动)、
`PRIMUS_PROFILE=True`、`PROFILE_STEP_START=100`/`END=101`(单 step 100)。脚本
`_run_c4_step100_flydsl.sh` = 由短诊断版 `_run_c4_force_flydsl.sh`(20 iter/step15,先验强制生效)`sed`
改 iter/step 派生。init ~4min,110 iter ~108s;**Bash tool 2min 超时,run 在容器里继续**,`nohup ... &`
后另起 poll(`pgrep -f _run_c4_step100_flydsl` + tail log)。

**验收 census(rank0/step100/`cat==kernel`,单 step = 24 层×2 投影 → 各核 n=48)**——我方 vs 参照:

| kernel | huangwei 0824(参照) | syncv3 强制 FLYDSL(我方) |
|---|---|---|
| `kernel_grouped_nt_persistent`(fwd) | n=48, avg **1701.7µs** | n=48, avg **1494.9µs**(更快) |
| `kernel_grouped_nn_persistent`(dgrad) | n=**23**, 1477.2µs | n=**48**, 1326.6µs |
| `kernel_grouped_tn_wgrad_4wave`(wgrad) | n=**23**, 1704.7µs | n=**48**, 1507.3µs |
| `kernel_grouped_tn_wgrad_reduce` | 无 | n=48, 5.8µs(0.28ms 可忽略) |
| `quantize_tensorwise_pad_row_kernel` | n=119, **226.5µs** | n=144, **225.1µs**(<1%) |

**判读(★这是关键,别再误报"完全对上")**:
- **核身份对上、fwd 逐核对齐、quantize 单调用 225.1 vs 226.5µs(<1%)= 同核同形状铁证**;单调用普遍更快
  = syncv3 tip `072e225a` 比 0824 build 多了后续 down-padk 优化。
- ⚠️**参照的反向是「混合」不是全 flydsl**:`nn/tn` 只 n=23(≈24=**只 down 投影**走 flydsl),另一投影
  **gate_up 的 dgrad+wgrad 落 hipBLASLt**(参照反向窗里有 `Cijk_..F8BS..MT256x256x128/MT256x192x128`
  共 ~45 条)。成因=**per-shape `can_handle`**:down 反向(padded variable-K)flydsl 接住 → FLYDSL;
  gate_up 反向在 0824 build 里 flydsl `can_handle`=False → fallback → HBL(hipBLASLt)。fwd 两投影都被
  `nt_persistent` 接住(n=48)。
- 我方**强制 FLYDSL 把 gate_up 反向也拽上 flydsl(48 vs 23)**,比参照更激进/更快,**不是逐核 n 相等的
  忠实混合**,但对"down-padk 核有没有真跑"这个验证目的**完全达标**(用户 08-25 拍板"确认我们的就行")。
  要逐核复现那个混合,得 down 走 FLYDSL、gate_up 单独钉 HBL(单个 env 做不到)——非当前目标。
- 别用我犯过的两个错:(a) 早期不强制那次全落 Triton 却报"对上了"被用户当场否(名字都不对、慢 2–5.5×);
  (b) rank0 trace 用 `ls *rank*0*` 会**被时间戳里的 '0' 匹配到 rank7**,要 `"rank[0]"` 字面过滤
  (同 §解析 trace 的 glob 字符类坑)。

wrapper(NOT git-add,均在容器 `/workspace/code/syncv4/`)= `_run_c4_smoke.sh`(20iter 冒烟)、
`_run_c4_force_flydsl.sh`(强制诊断)、`_run_c4_step100_flydsl.sh`(step100 复现)。见
[[project_gptoss_fp8_wgrad_skew_padk]]、[[project_gptoss_down_padk_native_campaign]]、[[feedback_no_unilateral_campaign_model_change]]。

### ★★★08-27 复跑血泪:「冷缓存第一批 iter 慢」被我反复误判成「RCCL 死锁」,连 kill 五六次
**症状**:8 卡全部 py-spy 冻在 `get_grad_norm_fp32 (clip_grads.py:133)`(该行=`total_norm.item()`,GPU 同步点,
**不是** all_reduce——两个 all_reduce 在 127/130 已返回),GPU 功耗 12 秒连采 **336W 死平**、flydsl/inductor 缓存
60s **零增长**。我据此判「GPU 空转=collective 死锁」→ kill → 重启 → 换容器,循环好几轮全在浪费时间。
**真相**:**根本没死锁**。那个 `.item()` 在等 GPU 排空**整个 backward + grad-norm**;冷缓存下第一批 iter 要现付
flydsl per-shape autotune + torch.compile,单 iter 极慢,`.item()` 长时间阻塞看着像冻住。让它**不动**跑满,
**08:00:47 启动的 run 13.5 分钟后(08:14:17)自己扛到 step 100 写出了全 8 rank trace**。昨天(08-25)之所以
「110 iter ~108s」快,是因为**先跑过 `_run_c4_smoke.sh`/`_run_c4_force_flydsl.sh` 把缓存焐热了**;冷跑第一次
没有捷径,必须付这个 autotune 成本。
**纪律**:(a) 判死锁**别只看功耗死平**——冷 autotune 也可以 100% util 但功耗平;**要看进程是否推进/最终产出**,
不是看瞬时。(b) 冷缓存 e2e 训练**首跑给足 ~15min**,别中途 kill;想快就先跑 smoke 焐缓存。(c) `.item()` 卡住
=在等 stream,**顺着看 stream 上最后是什么**,不要一看到 all_reduce 附近就喊 RCCL。(d) 反复 kill SIGKILL 会漏
`/dev/shm/nccl-*` 残段(实测有一堆),清 shm / 重启容器是廉价保险,但**本例并非 shm 致死,是我自己太急**。
**08-27 强制 FLYDSL 新鲜 census(rank0/step100,median/min/max µs,n=48)**:`nt_persistent` 1578.7/850.7/2950.4、
`nn_persistent` 1509.7/853.6/2699.6、`tn_wgrad_4wave` 1789.5/816.4/16196(max=profiler 假 outlier,`>5ms` 丢)、
`tn_wgrad_reduce` 60.6。量级与上表 avg 一致。见 [[project_gptoss_fwd_nt_slow_investigation]]、
[[feedback_no_cpu_ep_poll_hang_judgement]]。

## ★★ 变体 E — 跑 turbo 的**融合 MLP**(`grouped_mlp_fp8`)做 A/B(2026-09-02, smci355 n02-29 / `kyle_train`)

优化了 `grouped_mlp_fp8`(融合 MoE MLP)想在真训练里 A/B,结果在**接入**上卡了整整一轮。踩到的五件事,
每一件都足以让人误判成"我的 kernel 有问题":

### ★★★ 1. 那个开关不在 main 上 —— `turbo_fused_grouped_gemm`

`grouped_mlp_fp8` 在 **Primus main 和 Primus-Turbo main 里都没有任何调用方**(`PrimusGroupedMLP`
走的是分开的 fc1/fc2 `PrimusTurboColumnParallel/RowParallelGroupedLinear`)。但 huangwei 的 trace 里
`FP8GroupedMLPTensorFunc` 明明在跑 —— 因为开关在**未合入的分支**上:

```
origin/feat/fp8-fused-grouped-mlp-main   # RuibinCheung, 69a53c42 + f32da485
  yaml:  turbo_fused_grouped_gemm: true
  code:  experts.py 加一条分支 -> from primus_turbo.pytorch.ops.grouped_mlp_fp8 import grouped_mlp_fp8
```
⚠ 它落在 **`examples/mlperf/gpt_oss_20b/configs/MI355/`**(mlperf 那套),不是
`examples/megatron/configs/MI355X/`。分支基点比 main 落后 34 个提交,cherry-pick 那 2 笔到最新 main 即可。

★ **教训:代码里 grep 不到调用方 ≠ 部署没用它。先用 `git log --all -S"<符号>"` 扫所有分支。**

### ★★★ 2. `primus_turbo is not importable` 的真因是 flydsl 版本,不是 PYTHONPATH

Primus 报 `PrimusTurbo <feature> was requested, but primus_turbo is not importable` 时,**真异常被
try/except 吞掉了**。手动带 traceback 导一次才看得见:
```
/opt/venv 的 flydsl 是 0.1.1.dev409 -> flydsl.expr.typing 里没有 Vector
-> flash_attn_bwd.py 导入失败 -> 整个 primus_turbo.pytorch 链断掉
```
**必须用装了 flydsl 0.2.2 的 venv(本机 `/opt/venv-syncv4`)**。⚠ C++ 扩展是对着 `/opt/venv` 的 torch
头编的,但 venv-syncv4 的 torch 同版本,可以混用。
★ 排查手法:`PYTHONPATH=<repo> <venv>/bin/python -c "import traceback;
try: import primus_turbo.pytorch
except: traceback.print_exc()"`。

### ★★ 3. 三个 env/yaml 拦路虎(都不用改 Primus 逻辑)

| 报错 | 解法 |
|---|---|
| `Stage 'mlperf_pretrain' requires 'primus_mllog'` | 私有包 pip 装不到;`stage: ${PRIMUS_STAGE:mlperf_pretrain}` 是插值 → `PRIMUS_STAGE=pretrain` |
| tokenizer 401 gated repo | `tokenizer_model: ${MODEL:meta-llama/Llama-3.1-8B}` → `MODEL=/model`(容器已挂) |
| `/data` 只读写不了 index cache | `mock_data` 改成 `${PRIMUS_MOCK_DATA:false}`(唯一需要的 yaml 改动)+ `PRIMUS_MOCK_DATA=true` |

### ★★★ 4. 后端默认是 **triton 不是 FLYDSL**,两个都要改

`config_MI355X_1x8x1_tp1pp1ep1_gbs32.sh` 里:
```sh
export PRIMUS_TURBO_GROUPED_GEMM_BACKEND="${...:-triton}"   # grouped
export PRIMUS_TURBO_GEMM_BACKEND=triton                      # dense
```
不改就**完全走不到 FlyDSL kernel**,A/B 会得出"我的优化毫无影响"的假结论。
★ 事后判据:trace 里 **`_grouped_fp8_persistent_gemm_kernel`(Triton 名)一次都不出现** =
FLYDSL 真的生效了;它和 `kernel_grouped_nt_persistent`(FlyDSL 名)是互斥的一对。

### ★★★ 5. 「卡住了」的三条判据 —— 别看 GPU util

同一个现象(日志几十分钟不动、进程吃着 CPU)出现过两次,一次是编译一次是死锁,**GPU 全 0% 都一样**:

| 判据 | 编译中 | 死锁 |
|---|---|---|
| **JIT 缓存 2 分钟内新增文件数** | >0 | **0** |
| **8 个 rank 的 py-spy 栈顶** | **各不相同**(`_compile`/`deepcopy`/`_broadcast_cu_seqlens`…) | **全堵同一个**(`all_to_all_single`) |
| 最新缓存文件时间戳 | 就在刚才 | 比本次启动还早 |

```sh
find /root/.flydsl -type f -newermt "-2 min" | wc -l     # 编译在推进?
for P in $(pgrep -f "python -u -m primus.cli.main"); do py-spy dump --pid $P | sed -n 5p; done
```
⚠ 死锁那次的根因是**我给 ep1 的配置硬塞 `PRIMUS_EP=8`**(文件名就写着 `...tp1pp1ep1...`,里面还有
`MOE_SKIP_IDENTITY_SORT=1` 明说只在 EP=1/TP=1 成立)。**改并行度必须连配套项一起改,否则 MoE 的
all-to-all 会劈叉死锁。**

### ★★ 6. `LOG_INTERVAL=999999` 会让你以为没跑

mlperf config 把 iteration 日志压掉了。grep 不到 `iteration` **不代表没训练**;判完成看
`Cleanup completed` + trace 文件是否落盘。

### ★★ 7. OOM 说「37 GiB allocated 但 0 free」= 别人占着卡

`rocm-smi --showpids` 一看是 `sglang::schedul` 各占 **248 GiB**。8 卡训练前**必须先核对整机每卡余量**,
不是只看 util:
```sh
rocm-smi --showmeminfo vram | grep Used | awk '{printf "GPU%d: %.1f GiB\n", NR-1, $NF/1073741824}'
```


---
来源: 2026-08-03 gpt-oss-20b MLPerf K-pad e2e 验证(chi2798 mlperf_gptoss)+ 2026-08-08 Crusoe 177 pretrain-yaml 真跑取 grouped-gemm FLOPS + 2026-08-13 smci355/syncv4 官方 mlperf EP1 config 跑通(变体 C + 崩溃排障三层遮蔽)+ 2026-08-25 smci355/`kyle_train` 真 C4 EP1 down-padk 复现 huangwei 0824 trace(变体 D:强制 FLYDSL 破静默 Triton fallback + per-shape can_handle 混合判读);见 [[project_kpad_e2e_trace_validated]]、[[project_gptoss_e2e_trace_grouped_flops]]、[[project_syncv3_so_rebuild_4arg]]、[[project_gptoss_fp8_wgrad_skew_padk]]、connection/gpt_oss/02·03·04、connection/smci355/01、connection/crusoe/01;+ 2026-09-02 smci355/`kyle_train` 融合 MLP A/B 接入(变体 E:开关在未合入分支、flydsl 版本、后端默认 triton、编译 vs 死锁三判据)、pitfalls/07(glob/tracer 坑)、pitfalls/08(重编 .so)
