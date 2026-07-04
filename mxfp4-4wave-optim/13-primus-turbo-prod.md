# Primus-Turbo 生产后端 + rebase/squash/清理里程碑 (2026-07-02)

> ⚠️ **当前实际出货的 mxfp4 dense GEMM = Primus-Turbo 的 `mxfp4_gemm_kernel.py`**，不是本目录 01/03/07/12
> 描述的 FlyDSL standalone `turbo/mxfp4_gemm_4wave.py`+`FP4_*` env 世界。后者是调优 playground(历史
> 记录仍有效)；生产文件把那套旋钮**全部 hardcode 成生产值**，改配置靠 timed autotune，不读任何 env。

## 部署位置(canonical)
- 仓：`sync/mxfp4/Primus-Turbo`(本地 canonical git)，分支 `dev/kyle/flydsl_mxfp4_compute`。
- 文件：`primus_turbo/flydsl/gemm/mxfp4_gemm_kernel.py`(1732 行，纯 asm whole-loop 4-wave)。
- 公开入口：`gemm_mxfp4_flydsl_kernel(a, a_scale, b, b_scale, *, trans_a=False, trans_b=True, out_dtype=bf16, trans_c=False)`。
  - **NT only**(trans_a=F, trans_b=T)；A[M,K] fp4、B[N,K] fp4、C=a@bᵀ bf16。约束 K%256==M%256==N%256==0(odd-KI 即 K%512==256 由 loop 后 MFMA-only tail 处理)。
  - `a_scale`/`b_scale` 收 **canonical E8M0 块 scale**(`[M,K/32]`/`[N,K/32]`，quant 通用直出)；wrapper 内用**独立 FlyDSL preshuffle kernel**(同 stream，GEMM 前跑一次)把它 repack 成 lane-contiguous packed int32(VGPR-direct 读)。**quant 保持通用、不再融合 scale 写**——对齐 mxfp8 GEMM 的 quant/preshuffle 解耦。规则：第一个操作数恒 mode-A、第二个恒 mode-B(fwd/bwd 所有 NT 调用皆然)。
- 接进 Primus-Turbo fp4 GEMM dispatch(`primus_turbo/pytorch/kernels/gemm/gemm_fp4_impl.py` + `ops/gemm_fp4.py`)，走 `ScalingGranularity.MX_BLOCKWISE`。

## Timed autotune(生产选核方式，替代 FP4_* env)
首调每个 (M,N,K) 定时扫候选取 global-min，缓存 per-shape。四条轴，**后三条都是 never-regress**(扫里恒含 baseline + 取全局 min / 带 margin 门槛，只会追平或更快)：
1. **L2 swizzle**(`_MXFP4_AUTOTUNE_CANDIDATES`，字段 `(group_m, group_n, num_xcds)`，12 候选)：`(4,0,8)(8,0,8)(16,0,8)(8,0,4)(16,0,4)(4,4,8)(8,6,8)(4,8,8)(4,16,8)(4,32,8)(8,32,8)(1,8,8)`。纯 WG→tile 双射，bit-identical，只调 L2 residency/尾巴。规律：小-M 要大 group_n(16/32)，NX8 普遍最优。
2. **deep-wl**(phase-barrier vmcnt/lgkmcnt)：默认 `(10,9)`；K≥8192 时额外报 `(16,15)`(深流水藏高-trip-K 的 g2s 延迟)。选择带 `_WL_MARGIN=1.02` 让步(深流水须超噪声带 2% 才选)。
3. **变体轴 (COOP scale-load, TACCW wide-store)**：对 swizzle/wl 赢家再编 {TACCW, COOP, TACCW+COOP} 三孪生，同热窗 round-robin 对比 plain 赢家，0.5% 门槛才换。COOP=4 波协同一次载 4 组 scale 进 LDS(省冗余 HBM，低/中-K、宽-N 赢)；TACCW=acc=Cᵀ + AITER permlane16_swap dwordx4 宽 epilogue(epilogue 暴露的胖/低-K 赢)。
4. **split-K (ksplit)**：`_ksplit_candidates` 只对 few-tile 大-K(一 WG/tile 撑不满 CU)给 2/3/4/6/8/12/16，端到端(split+reduce)定时，恒含 ksplit=1。

CUDA-graph capture 内无法定时 → 回退静态 `_mxfp4_nt_config` 启发式。

## 正确性门禁(现产复现)
```bash
# chi2810 容器 mlperf_gptoss / venv /opt/venv / 空闲 GPU(rocm-smi --showmeminfo vram ~300MB=空)
# csrc 改过 → 先重编 .so：
docker exec mlperf_gptoss bash -lc 'cd /workspace/code/Primus-Turbo && \
  rm -rf /root/.flydsl/cache && /opt/venv/bin/pip install -e . --no-build-isolation'
# 纯 .py 改动 → 只需清 flydsl cache(下面 pytest 里已带)
docker exec -e HIP_VISIBLE_DEVICES=0 mlperf_gptoss bash -lc 'cd /workspace/code/Primus-Turbo && \
  rm -rf /root/.flydsl/cache && /opt/venv/bin/python -m pytest tests/pytorch/ops/test_gemm_fp4.py -q'
```
- 期望：**141 passed / 325 skipped**(含 `test_gemm_fp4_mx_blockwise_flydsl` 的 gfx950 FLYDSL fwd+bwd SNR、AITER preshuffle parity、torch.compile、QuantizedTensor)。SNR 健康。
- 同步见 `remote-sync`；改 csrc 必重编 `.so` 再测(pr-merge-gate 硬门禁 5)。

## 性能成绩(carry-forward + provenance)
> 下列 fly-vs-aiter / peak TF 是 **2026-07-01** 在 FlyDSL standalone / rebase 前 autotune 下测的(见 `12-llama-aiter-baseline.md`、`01-production-config.md`)。本轮 rebase-onto-main + 清理是 **dead-code/注释-only，compute 路径 0 改动**(pytest 141 passed 与清理前逐位一致)，故这些数字对现产仍代表性成立。且生产 wrapper autotune 比当时多了 ksplit/COOP/TACCW/deep-wl 四轴、全 never-regress → 是现产的**保守下界**。
- **Llama 7B/70B 前向 NT fly/aiter geomean ≈ 0.995**(24 点，基本 parity)，全 SNR 55.6 det0。赢：7b-gateup(+6~8%)、7b-oproj@M4096、7b-down(odd-KI 修复后)；输 ~3-6%：大-N 短-K(7b-qkv/70b-gateup/oproj@M8192)。
- standalone 峰(8192×8192)：K=28672 ~5401 med / 5474 min；K=57344 ~5481 med / 5540 min(det0 SNR55.6)。
- ⚠️ 要**权威的现产最新 TF**须重测(现产 public wrapper + 四轴 autotune，非 standalone)。bench 脚本见下(未入库)。

## 2026-07-02 里程碑：rebase main + squash + 代码清理
- **rebase 到最新 origin/main**：main 已经 #390 合入 flydsl mxfp8(与本分支 5 个并行 mxfp8 commit 重复)→ rebase 时丢掉那 5 个，只重放 6 个 mxfp4 commit；main 还迁了 ruff(#392)。
- **squash 6→1**：单 commit `74eaadac` `feat(mxfp4): FlyDSL 4-wave MXFP4 dense GEMM backend (NT)`，author=kyle-256、无 coauthor，17 文件/2341 insertions，1 ahead / 0 behind origin/main(**未 push**)。
- **补回 `preshuffle_layout` 字段**：mxfp4 quant 路径依赖它，原在被丢弃的 mxfp8 commit 里 → 手动加进 main 的 `ScalingRecipe`(C++ `quantization.h` + Python `low_precision.py`)，重编 `.so` 后 pytest 141 passed。
- **pr-merge-gate review → 清理 -233 行**(kernel 1965→1732)，删 6 类死代码/残留(review 两轮 PASS)：
  1. host-side scale preshuffle 链(`preshuffle_mxfp4_scales`/`preshuffle_scale_lane_contig`/`preshuffle_scale_packed`) — 已被 GPU 融合 quant 取代。
  2. `S2RLoaderFp4._load16/.load/zero4`(生产走 `base_addr`) + 冗余构造参数 + `pack_i32x4_i32x8` import。
  3. `ScaleS2RPacked.load/lane`(生产只用 `.rsrc`)。
  4. **全部 `PT_MX_*` 实验 env 旋钮**(GM/GN/XCD pin、NO_AUTOTUNE、PERSIST/NONPERSIST、NO_WLTUNE、NO_TACCW、KSPLIT、COOP_SCALE) → 现产纯 autotune，无 env override。
  5. **从不被采纳的 `persistent` 跨-tile stride-loop 变体**(`_persist_opts=(False,)`，仅 env 可达) — 连 `_compile_mxfp4_nt` 形参/`_PERSIST_GRID`/kernel 分派分支/`launch_gemm` grid 一起删。
  6. 注释里的 benchmark/实验数字(单形状胜负、dB、pp、`+X% 7b-qkv`、具体 MxNxK)。
- 保留(活跃 autotune 轴，勿删)：COOP、TACCW、deep-wl(wlv/elgk)、ksplit。

## 2026-07-02 里程碑：FlyDSL mxfp4 GEMM 作为**普通 sibling backend**接入(对齐 mxfp8，零侵入)
**最终形态**：整个 mxfp4 FlyDSL GEMM 集成 = **仅 3 个文件**，与 main 的 mxfp8 FlyDSL 接法完全一致——quant 出通用 raw E8M0，preshuffle 由 GEMM 侧 FlyDSL kernel 内部 repack。**不动** `gemm_fp4.py` / quant csrc / `low_precision.py` / AITER / HipBLASLt(相对 HEAD 全零差异)。
- **无 ops 层 fast-path**：早期版本在 `gemm_fp4.py` fwd/bwd 加了 `if _flydsl_target and shape%256 ...` 特判块直写 raw scale + `default_backend=FLYDSL` + `ctx.flydsl_mx` 分支——**已彻底删除**(不通用、把 backend 约束泄漏进 ops 层)。FlyDSL 通过**正常 dispatch** 命中：`use_preshuffle=False` 时通用路径本就 quant 出 raw E8M0，pin `set_gemm_backend(FLYDSL)`(user_backend 最高优先级 `backend.py:445`)→ `FLYDSL.can_handle` 认 raw(`[dim,K/32]`,1B)→ `execute`。generic path 的 col-cache(axis0,RHT)与 bwd grad_out 逐一走同一 dispatch。
- **FlyDSL preshuffle kernel**(`mxfp4_gemm_kernel.py`：`_build_mxfp4_preshuffle_launch(mode)` + `_get_mxfp4_scale_ws` + `preshuffle_mxfp4_scale`)：gather 式，一线程一 output int32 dword，gather 4 个 E8M0 字节(16 行一跳)packed 成一 dword；mode 0=A / 1=B 用 `_mxfp4_grp_from` 反演 group map。
  - ⚠️ **坑**:`if mode==...` 分支不能写在 `@flyc.kernel` 体内(AST 变换会转成 traced conditional、分支内赋值不外泄 → `NameError`);须放**普通 Python helper**(`_mxfp4_grp_from`,trace 时求值)——同 mxfp8 `_emit_lds_repack` 的 `if is_a`。
  - ⚠️ 单 dword store 用 **scalar** `buffer_store(packed, rout, gid, mask=...)`(传裸标量值)；包成 1-wide `Vec.from_elements` 会触发 LLVM `Do not know how to scalarize` 崩。
  - **独立 bit-exact 校验**(拿历史融合-quant packed 对拍)：A/B × axis0/1 × RHT 四例 `PACKED_EQ=True`。`dim`/`K` 由 raw scale 形状直接推出。
- **wrapper**(`gemm_mxfp4_flydsl_kernel`)：`a_scale`/`b_scale` 收 raw E8M0;`_get_mxfp4_scale_ws(M,N,K,dev)` 缓存 packed workspace(per-shape,graph-replay 稳);同 stream 先 `launch(a→A)`/`launch(b→B)` 再走 gemm/autotune/split-K(compute 段 0 改动)。
- **backend 注册**(`gemm_fp4_impl.py`)：`GEMMFP4FlyDSLBackend` 是普通 sibling——`execute` 收 raw E8M0 直传 wrapper(内部 preshuffle);`can_handle._scale_ok` 只认 raw(1B,`[dim,K/32]`) + gfx950 + M/N/K%256 + NT。`SUPPORTED_DTYPES = set(_COMMON_SUPPORTED_DTYPES)`(**bf16+fp16 out**,同 HipBLASLt/AITER)。`execute`/`can_handle` 签名与 HipBLASLt/AITER **完全一致**(含 `preshuffled` 形参 + `del preshuffled`——dispatcher `execute(**kwargs)` 结构性要求所有 backend 声明该轴,与 HipBLASLt sibling 一致;`can_handle` 已对 `preshuffled=True` 返 False)。
- **fp16 out 支持**(`mxfp4_gemm_kernel.py`)：`StoreCPlain` 参数化 `out_ty`(bf16/fp16 皆 2B),窄 store 用通用 `.to(out_ty)`;**宽 store `store_tacc_wide` 是 bf16-only**(`cvt_pk_bf16_f32`),故 fp16 在 wrapper 里**强制 `taccw=False`** 走窄路径。`out_fp16` 穿过 `gemm_mxfp4_flydsl_kernel`→`_get_mxfp4_launch`→`_compile_mxfp4_nt`,并进 launch/AT/split 三处 cache key(否则 bf16/fp16 编译产物串号)。autotune 仍按 bf16 选 config(perf 与 dtype 无关),fp16 只覆盖 taccw 轴。
- **测试**(`test_gemm_fp4.py`)：FLYDSL 加进**两个**参数化 test 的 `backend` 轴——`test_gemm_fp4_mx_blockwise`(`[AITER,FLYDSL]`,18 shape×{bf16,fp16}) + `test_gemm_fp4_mx_blockwise_quantized_tensor`(`[None,HIPBLASLT,FLYDSL]`,QT 预量化路径)。约束仅用 M/N/K%256 skip 兜(gfx950 已由 `check_mxfp4_support()` gate,**无需** `is_gfx950` skip;fp16 已支持故**无** dtype skip;`preshuffle=True` 由既有 `backend!=AITER and preshuffle` skip 挡)。**不单开 test**。
- **验证**:纯 `.py` 改动无需重编;`rm -rf /root/.flydsl/cache` 后 pytest **177 passed / 697 skipped**(FLYDSL fwd+bwd SNR bf16+fp16、QuantizedTensor、AITER parity、torch.compile 全绿)。

## bench 脚本(未入库 scratch，勿 git add)
- `benchmark/ops/training/bench_llama_mxfp4_flydsl_vs_aiter.py` — FLYDSL vs AITER，fwd+bwd，走公开 wrapper。**现产权威对标用它**(在工作树，untracked)。
- `benchmark/ops/training/bench_llama_mxfp4_kernel_only.py` — ⚠️ 仍 import 已删的 `preshuffle_mxfp4_scales`，需改成走 wrapper 才能跑。
- 这俩连同 `_*.py` 探针按 pr-merge-gate 第 6 条**不入库**。
