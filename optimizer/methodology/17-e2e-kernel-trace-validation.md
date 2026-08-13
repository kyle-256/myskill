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

---
来源: 2026-08-03 gpt-oss-20b MLPerf K-pad e2e 验证(chi2798 mlperf_gptoss)+ 2026-08-08 Crusoe 177 pretrain-yaml 真跑取 grouped-gemm FLOPS;见 [[project_kpad_e2e_trace_validated]]、[[project_gptoss_e2e_trace_grouped_flops]]、[[project_syncv3_so_rebuild_4arg]]、connection/gpt_oss/02·03·04、connection/crusoe/01、pitfalls/07(glob/tracer 坑)、pitfalls/08(重编 .so)
