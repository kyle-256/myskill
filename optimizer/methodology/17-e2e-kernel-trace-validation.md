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

---
来源: 2026-08-03 gpt-oss-20b MLPerf K-pad e2e 验证(chi2798 mlperf_gptoss);见 [[project_kpad_e2e_trace_validated]]、connection/gpt_oss/02·03·04、pitfalls/07(glob/tracer 坑)
