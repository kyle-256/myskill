# 14 — preshuffle v2 默认化 + 融合单发射(对齐 mxfp8)+ e2e(含 quant)对比

日期: 2026-07-02(承 13 之后)。文件: `primus_turbo/flydsl/gemm/mxfp4_gemm_kernel.py`。
节点: chi2810 docker `mlperf_gptoss`,`/opt/venv`,`HIP_VISIBLE_DEVICES=6`,MI355X/gfx950。
仓路径(容器内)`/workspace/code/Primus-Turbo`(bind 自 `/mnt/vast/kyle/code2/Primus-Turbo`);
本地镜像 `sync/mxfp4/Primus-Turbo`,`dev/kyle/flydsl_mxfp4_compute`。同步走 `sync/.ssh-chi.sh`(rsync + ssh 跳板)。

## 本次三件事

### 1. preshuffle v2 设为唯一默认(删 v1)
- 旧 v1(一 thread 一 output dword)有 **4× cross-thread 读放大**:4 个 g_byte thread 各自把同样的 4 行源 int32 重读一遍。
- v2:**一 thread 出全部 NG=4 个 g_byte dword**,共享的 4 个源 int32 只读一次;grid 缩到 1/NG,输入 HBM 读流量 ~4×↓,**device 时长减半(~16→8µs)**,bit-exact(已 new-vs-old 校验)。
- 落地:`_build_mxfp4_preshuffle_launch` 直接就是 v2;删掉 `_v2` 后缀与旧实现。`_MXFP4_PRESHUF_NG = 4`。

### 2. 融合单发射(和 mxfp8 backend 完全一致)—— 省两次 host dispatch
**动机**:旧路径每次调用 = `preshuffle_A` + `preshuffle_B` + `gemm` = **3 次 host dispatch**;eager 下每次 `@flyc.jit` dispatch ~38µs(compiled 直调 ~6µs),小 shape 被 host 开销主导。
mxfp8 早已用"融合 stub"解决:GEMM 工厂返回**裸 `@flyc.kernel`**,再由**一个 `@flyc.jit` stub** 在同一 stream 上依次 `.launch()` 发射 preshuffle + gemm → **一次 Python dispatch**(stub 内部的多 kernel enqueue 是廉价 C++ 调用)。

mxfp4 改造(镜像 mxfp8 的 `_compile_mxfp8_fused`/`launch_mxfp8_fused`):
- `_compile_mxfp4_nt` → **`_build_mxfp4_gemm_kernel`**:返回裸 `(kernel_gemm_4w, BLOCK_M, BLOCK_N, ksplit, gemm_value_attrs)`(不再自带 launch)。
- 新增 **`_build_mxfp4_preshuffle_kernel(mode)`** 返回裸 kernel(fused stub 唯一消费者)。~~曾保留独立 jit 包装 `_build_mxfp4_preshuffle_launch`/`preshuffle_mxfp4_scale` 供校验~~ → **07-03 merge-gate 复审删除**(见下 §4:入库代码 0 消费者=死代码)。
- 新增 **`_compile_mxfp4_fused` / `_get_mxfp4_fused_launch`**:`launch_mxfp4_fused(A,B_T,C,A_raw,B_raw,A_scale_ws,B_scale_ws,c_m,c_n,stream)` 内依次发射 **preA → preB → GEMM**。
- autotune(`_autotune_mxfp4_config`)与 wrapper 全改用 fused launch + fused args。preshuffle 是固定常量偏移 → gemm-config 排序不变(mxfp8 同款论证)。
- 删除 `_get_mxfp4_launch` / `_run_mxfp4_preshuffle` / `_MXFP4_PRESHUF_COMPILED`(不再需要单独预编译单发射绕路)。ksplit>1 也走 fused stub(写 `[ksplit*M,N]` workspace),host 只留最后 reduce。
- capture 下 `raw(*args)`(jit 直调可在 capture 内工作),eager 下 `flyc.compile(raw,*args)` 后 compiled 直调 —— 与 mxfp8 一致。

**坑**:pre-commit `ruff-format` 会重排新代码 → 第一次 `--amend` 失败;重新 `git add` 同一文件再 amend 即过(所有 hook pass)。

### 3. 验证
- **SNR 门禁**:`pytest tests/pytorch/ops/test_gemm_fp4.py -k FLYDSL -q` → **44 passed, 372 skipped**(fwd+dgrad+wgrad 全过;非 256 倍数 shape 自动 skip)。
- **eager 全 wrapper(不含 quant)**:small256 = **24µs**(就是一次融合发射的地板);down70B 8192×28672 = **5444 TF**,qkv70B 4944,qkv7B 4320。
- **kernel-only vs aiter 正确峰值**(scale 都提到计时外;aiter 峰值 = min over {`bpre=F`+shufScales, `bpre=T`+shufB+shufScales},**raw-scale 那条 SNR<0 是垃圾,不能当 baseline**):
  down70B 5571/5609=**0.993**,qkv70B 0.989,gateup70B 0.977,qkv7B 0.973,attn7B 0.993;gateup7B/down7B FLY **反超**(aiter 无 tuned config 回退默认)。整体 **0.97–1.0×**。

### 4. merge-gate 复审 + 死代码清理(2026-07-03)
对最终 commit 重跑 `pr-merge-gate`(readonly 子代理独立复审 + 本地 ruff/pre-commit):
- **门禁 A**:`ruff check`/`ruff format`/`pre-commit`(3 文件)全 Passed。
- **BLOCK(已修)——独立 preshuffle 启动链是死代码**:`preshuffle_mxfp4_scale` / `_get_mxfp4_preshuffle_launch` / `_build_mxfp4_preshuffle_launch` / `_MXFP4_PRESHUF_LAUNCH` 这条链**入库代码 0 消费者**(GEMM 主路径走 fused stub 直调裸 `_build_mxfp4_preshuffle_kernel`;唯一外部引用是未入库探针 `_pre_prof.py`,且它引的还是不存在的复数名 `preshuffle_mxfp4_scales`)。mxfp8 参照后端也无对应物。→ 全删(保留仍被 fused stub 用的 `_build_mxfp4_preshuffle_kernel` / `_get_mxfp4_scale_ws` / `_mxfp4_grp_from`)。**教训:"留着作校验 harness 用"若 harness 是未入库探针,对 merge-ready 就是死代码,门禁 §B 必删。**
- **WARN(已修)——注释路线图备忘**:coop 屏障注释里 `(v0: correctness-first; deep-carry recovery is a later step…)` 属 §B 红线过程笔记,删(留"为何 full drain"机制说明)。
- **WARN(未改)**:几处 "measurable win/saves ~64 nops" 软措辞、`scv6` 预留 asm 操作数槽;子代理判非阻断(mxfp8 亦有),保留。
- 改动 **-38/+3**,纯删死代码 + 注释,**融合 GEMM 主路径未动 → 正确性中性(不影响 SNR)**;探针/bench 脚本确认均未入库(§6 OK)。

## e2e(含 quant)对比 —— `bench_llama_mxfp4_flydsl_vs_aiter.py`
计时区间 = `turbo.ops.gemm_fp4(a_bf16,b_bf16)` = **量化(bf16→fp4+E8M0,a&b)+ 融合 preshuffle+GEMM**,fwd+bwd,两边同一套 turbo quant,只切 GEMM 后端(`GlobalBackendManager.set_gemm_backend`)。`profile_gemm_fp4` 里 `fwd_func`/`bwd_func` 都是完整 op。

复现:
```
cd /workspace/code/Primus-Turbo/benchmark/ops/training
rm -rf /root/.flydsl && /opt/venv/bin/python -u bench_llama_mxfp4_flydsl_vs_aiter.py --mbs 1
```

结果(mbs=1,M=4096,MI355X,全部 correctness PASS,SNR out/da/db ≈15–16 dB):

| Model/Layer | N | K | AITER fwd | FLYDSL fwd | fwd× | bwd× |
|---|---|---|---|---|---|---|
| 7B qkv | 12288 | 4096 | 1982 | 2180 | 1.10 | 1.09 |
| 7B attn_out | 4096 | 4096 | 819 | 1370 | **1.67** | **1.83** |
| 7B gate_up | 22016 | 4096 | 1280 | 1866 | 1.46 | 1.06 |
| 7B down | 4096 | 11008 | 1564 | 1698 | 1.09 | 1.36 |
| 70B qkv | 10240 | 8192 | 1659 | 1748 | 1.05 | 1.05 |
| 70B attn_out | 8192 | 8192 | 1773 | 1882 | 1.06 | 1.05 |
| 70B gate_up | 57344 | 8192 | 1685 | 1787 | 1.06 | 1.06 |
| 70B down | 8192 | 28672 | 1846 | 1932 | 1.05 | 1.06 |

**几何平均:fwd 1.174×,bwd 1.171×**(min 1.05× / max 1.83×)。

要点:
- 含 quant 后**绝对 TF 大跌**(fwd ~1700–2200 vs 纯 kernel ~5500):量化 + 小 M(mbs=1 只有 4096)拉低 FLOP 效率,这是 quant 本身 + 尺寸的开销,两边都吃。
- **相对 FlyDSL 全面领先**:大 shape 稳 1.05–1.06×;小方阵/低效 shape(7B attn_out/gate_up)优势更大(1.46–1.83×),因为那里 aiter 无 tuned config 回退默认,而 FlyDSL 融合单发射还省了 host 开销。

## Git
`[feat] add flydsl backend for mxfp4 gemm`:
- 07-02 amend → `4b066c40`,force-push(`5189e5a7...4b066c40`)。
- **07-03 死代码清理 amend → `9f525dba`**,`force-with-lease` 强推(`4b066c40...9f525dba forced update`,ahead1/behind0,author=kyle-256 无 coauthor)。
  - ⚠️ 用 URL(非命名 remote)推时 `--force-with-lease` 无期望值会误报 "stale info";先 `git fetch <url> <branch>` 确认远端仍是旧 hash,再用**显式** `--force-with-lease=<branch>:<old-oid>` 推。
commit 只含 3 文件:`mxfp4_gemm_kernel.py`、`gemm_fp4_impl.py`、`test_gemm_fp4.py`;**未含** `3rdparty/composable_kernel` 子模块改动与所有未跟踪探针。
清理:删了 11 个引用已删符号的坏探针(`_mxfp4_kern/_pre_hostfix/_pre_prof2/_pre_validate/_mxfp4_peak/_mxfp4_fair/_mxjoint/_probe6144/_probe28672` + `benchmark/ops/training/_mxwl/_mxab`),本地+远端都删。

## e2e MXFP4-FlyDSL vs MXFP8-FlyDSL 对比(2026-07-03)—— 回答"mxfp4 e2e 数字是不是差"
脚本 `benchmark/ops/training/bench_llama_mxfp4_vs_mxfp8_flydsl.py`(未入库探针)。**两列都用 FLYDSL 后端**,同 shape 同 harness,只切精度:mxfp4=`gemm_fp4`+mxfp4 cfg,mxfp8=`gemm_fp8`+mxfp8 cfg。FLOP 都按 `2*M*N*K` 逻辑维 → TFLOPS 直接可比。mbs=1,M=4096,MI355X,全 correctness PASS(mxfp4 SNR out~15.4,mxfp8 out~28.3 —— fp8 精度更高,符合预期)。

| Layer(7B/70B) | mxfp4 fwd | mxfp8 fwd | fwd 4/8 | mxfp4 bwd | mxfp8 bwd | bwd 4/8 |
|---|---|---|---|---|---|---|
| 7B qkv | 2122 | 1643 | 1.29 | 2491 | 1847 | 1.35 |
| 7B attn_out | 1372 | 1420 | **0.97** | 1676 | 1691 | 0.99 |
| 7B gate_up | 1885 | 1512 | 1.25 | 2476 | 1881 | 1.32 |
| 7B down | 1663 | 1426 | 1.17 | 2336 | 1686 | 1.39 |
| 70B qkv | 1727 | 1541 | 1.12 | 2773 | 1991 | 1.39 |
| 70B attn_out | 1854 | 1648 | 1.13 | 2676 | 1916 | 1.40 |
| 70B gate_up | 2040 | 1864 | 1.09 | 3151 | 2164 | 1.46 |
| 70B down | 1943 | 1766 | 1.10 | 3187 | 2159 | 1.48 |

**几何平均 mxfp4/mxfp8:fwd 1.135×,bwd 1.337×**;绝对 geomean mxfp4 fwd 1811/bwd 2552,mxfp8 fwd 1596/bwd 1909。

**结论(直接回答"数字差"的疑虑)**:
- mxfp4 e2e 的绝对值(~1700–2200 fwd)看着比纯 kernel(5500)低很多,但**这是 quant + mbs1(M 仅 4096,算术强度低)的通病,mxfp8 一样吃**——同 harness 下 **mxfp8 反而更慢**(fwd 1.14×、bwd 1.34× 落后 mxfp4)。所以 mxfp4 e2e 不差,相对成熟的 mxfp8-flydsl 是**更快**的。
- **bwd 优势(1.34×)大于 fwd(1.14×)**:反向两次 GEMM(dgrad/wgrad)+ dual-cast quant,mxfp4 的 fp4 GEMM 峰值优势与融合单发射在这里体现更足。
- 唯一近平(略输)的是 **7B attn_out(0.97 fwd)**:小方阵 4096³,GEMM 占比低、quant/host 占比高,精度带宽差异被摊平。
- 想看绝对 TF 上量:加 mbs(2/4)提 M → 算术强度上来,两边绝对值都会涨,mxfp4 领先幅度预计收窄但仍 >1×。

### ⚠️ 踩坑:rsync 后 mxfp8 e2e 报 `MXFP8 not support RHT`(colwise_recipe.use_rht==false 断言)
- 现象:`quantize_mxfp8_dual_impl` 断言 `colwise_recipe.use_rht==false` 失败。但当前 csrc `quantize_mxfp8_dual`(quantization.cpp L700-703)**硬编码** use_sr/use_rht=false → 正确 build 绝不该触发。mxfp4 路径同 .so 却正常。
- 根因:**远端 `.so` 与 rsync 过去的 csrc 不匹配**。rsync 更新了 `quantization.cpp`(→ 会 hipify 成 `quantization_hip.cpp`→`.o`),但**增量 `pip install -e .` 没重新 hipify/重编该对象**(rsync `-a` 保留源旧 mtime + hipify 中间产物在 `build/temp/...`,ninja 时间戳判过时机不对),旧 `.o` 用的是上一个 checkout 的 ScalingRecipe 参数序 → colwise use_rht 读到 true。
- 教训:**改了 csrc 后必须 clean 重编**(`rm -rf build primus_turbo/lib/*.so && GPU_ARCHS=gfx950 pip install --no-build-isolation -e .`,~15–25min),别信增量;增量重编(仅 5min "Successfully installed")没真重编变动的 hip 对象。clean 后 mxfp8 立即通过(SNR 28.3)。merge-gate §5 "改 csrc 重编 .so" 要理解成 **clean** 重编。

## determinism / race 复测(2026-07-03,commit 9f525dba)
探针 `benchmark/ops/training/_det_probe.py`(未入库)。**隔离 GEMM kernel**(不含 quant):量化 a/b 到 fp4+raw E8M0 各一次,固定输入连跑 `DETRUNS=100` 次,逐元素 bit-exact 比对 run#0。cross-wave LDS-barrier / SCVGPR race 会表现为 run-to-run 漂移(max|d|≠0)。
- GPU3(空闲 MI355X),8 shape(7B/70B 全 dense + 8192³ 方阵 + 历史最易 race 的 K=28672)× bf16/fp16 = **16 组全部 `det=0`**,`ALL DETERMINISTIC (no race detected)`。
- 复用 cached scale workspace(a_sp/b_sp)= 真实路径;preshuffle→gemm 同流共享 workspace 若有 race 会暴露 → 没有。融合单发射 + preshuffle v2 未引入 race。

## 认知
- **融合单发射是 eager 小 shape 的关键**:kernel 本身早已 0.97–1.0×,之前 e2e/wrapper 的差距主要是每调 2×preshuffle + gemm 的 host dispatch;融进一次 stub 后与 mxfp8 行为一致。
- **aiter kernel-only baseline 必须用 shuffled scales**:`bpreshuffle=False` 只是不做**权重 B** 的离线 preshuffle,**scale 仍必须 shuffle**;raw-scale 直读核 SNR<0 是垃圾,别拿它的"快"当基线。
- **含 quant 的 e2e 才是用户真实数字**:纯 kernel 5500T 只是上限;真实训练每步含激活量化,绝对 TF 会低很多,但 FlyDSL e2e 仍 >1× aiter。
