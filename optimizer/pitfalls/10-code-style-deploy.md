# 代码风格与生产化部署：源码禁调试痕迹、ruff --fix 陷阱、backend 签名一致、cache key、比较基线

> 类别: 踩过的坑 · 主题标签: 代码风格, merge-gate, ruff, 死代码, authoring-pattern, autotune-dispatch, cache-key, vendor-reuse

## 代码风格门：源码禁调试痕迹/benchmark 数字、探针脚本不 git add、ruff --fix 查 diff

- **注释精简,严禁 >5 行连续注释/docstring**:要点 1-2 句;推导/实测数字/权衡进 commit/memory/`tuning_results/`。放行前 review 逐块查,`>5 行`即打回。
- **源码注释严禁调试/实验痕迹**：跑数结果(如 `58dB 正确`)、调参备忘、`# debug`/`# tmp`/`# 临时`、带日期的过程笔记——这些进 commit message / memory / PR，**不进源码**。放行前必须 `rg -i 'debug|tmp|临时|verified|实验|print\(|TODO'` 得 **0 命中**才放行。WHY：源码是长期资产，调试痕迹是过程噪声，会误导后续读者。
- **探针脚本绝不 git add**：`_g_*.py` / `_d_*.py` / `opt_*.py` 这类临时探针脚本，永远留本地，不入库。
- **tuning 中间态放 `tuning_results/`(项目内)，不写 memory**：每轮 winner / TFLOPS / cfg 属于易变数据，进 `tuning_results/`。memory 只放**不变的约定**——target 来源、`BK=128` 硬约束等。WHY：memory 是稳定索引，塞入易变调参结果会污染。
- **改动对整除-K 必须是编译期 no-op**：加 K-tail / 新分支等改动，对 `K%128==0` 的 shape 必须编译期 no-op，老 shape SNR 不变。回归=不放行。
- **merge-gate 死代码判定**：`留着作校验 harness 用` 若那个 harness 本身是**未入库探针**，则对 merge-ready 而言就是死代码，**必删**。实例：独立 preshuffle 启动链入库了代码但 0 消费者=死代码。
- **生产文件旋钮全 hardcode**：`FP4_*` 全部 hardcode 成生产值；删掉所有 `PT_MX_*` / `PT_` 实验 env、persistent 变体、以及注释里的 benchmark 数字(单形状胜负 / dB / pp / 具体 MxNxK)。

### ruff check --fix 陷阱
- ruff `--fix` 会删 **F401 未用 import** 和 **F841 简单未用变量**并排序 import。
- ❌ 别再试 无脑接受 `--fix` 结果：它可能删掉那些**本意保留**的变量(编译期决策标记 / 局部可读性变量)。自动修复后**必须查 diff**——若改的是行为而非仅格式/import 卫生，恢复行为逻辑再重跑 formatter。
- ruff **不删**未用函数/类，那些需人工审：autoflake/ruff F401/F841 抓不到未用函数/类/私有 helper，自己 `rg '<name>\b'` 全仓搜，**0 命中即删**(实例 `_get_fp4_dtype`，源自 gpt_oss2 pr-merge-gate/SKILL.md，非本环境)。
- **裸 TODO 处理**：无 issue 号的裸 `# TODO` 要么挂号(关联 issue)要么删，不留悬空。注释掉的旧代码/占位一律删。

### CI 报无关文件失败
- 本地风格检查全过，但 PR CI 仍报**无关文件**失败=PR 分支落后 main，**不是 formatter 的锅**。
- 解法：fetch `origin/main` → merge 进 PR 分支 → 重跑检查 → push merge commit。

## 生产化部署坑：backend 签名一致/setdefault 遮蔽/out_fp16 cache key/比较基线错

- **execute/can_handle 签名必须与 sibling backend 完全一致（含 preshuffled 形参）**：dispatcher 走 `execute(**kwargs)`，结构性要求所有 backend（HipBLASLt/AITER/FlyDSL）都声明同一套轴。FlyDSL 的 execute/can_handle 若少了 `preshuffled` 形参会崩；正确写法是声明该形参 + `del preshuffled`（收下但不用）。WHY：kwargs 广播给每个 backend，谁少一个轴谁就 TypeError。

- **宽 store 是 bf16-only，fp16 out 必须走窄路径**：`store_tacc_wide` 用 `cvt_pk_bf16_f32`，只产 bf16。fp16 输出要在 wrapper 里强制 `taccw=False` 走窄 store 路径。

- **out_fp16 必须进三处 cache key（launch/AT/split）**：否则 bf16/fp16 两种编译产物串号（同一 key 命中错的二进制），fp16 请求拿到 bf16 kernel。三处都要带 `out_fp16` 才不串。

- **mxfp4 生产默认在 4wave.py PROD setdefault，遮蔽 8wave.py**（部署死坑）：`4wave.py` L287-303 `PROD setdefault(ASMMFMA=6 INPLACE=1 DIAG=1 SINNER=1 MMORD=5 ALT=0 GAVOID=1 SC_VGPR=1 PIN...)`，setdefault 在 `8wave.py` 读 environ **之前**生效。所以单改 `8wave.py` 的 environ 默认无效。
  - ❌ 别再试：把 `8wave.py` 默认 `0→9` 无效——已被 4wave PROD setdefault 遮蔽。要改就改 `4wave.py` L290 的 `MMORD '5'→'9'`。

- **比较基线错会造出假收益**（陷阱）：把 base 设成 `FP4_MMORD=0` 去比，得出 mm5/mm9 "+3~5%" 是假象——mm0 从来不是默认，真实默认早就是 mm5。真实收益 = mm9(4×8) 在 mm5 之上仅 **+~2%**（7b-qkv M8192：4117→4210），整体 <1%。
  - ❌ 别再试：任何"跟 mm0 比"得出的提升都要打折，别当真收益上报。

- 移植 4-wave kernel 不复用 gemm_helper.py（vendor helper 原因）：见 pitfalls/08-sync-build.md「build/环境坑」（vendor helper 为何不复用 gemm_helper.py）

- **Primus-Turbo（mxfp4/tensorwise，本环境）放行硬门禁（缺一不可）**：`pre-commit run --files <改动>` 全绿（ruff check/format + clang-format + shellcheck，无 isort/black/autoflake）；SNR 正确（rowwise ~56dB、tensorwise ~40-47dB，odd-K 单独验）；整除-K 零回归；FlyDSL 改动需确认 JIT 缓存确实失效（否则测的是旧核）；性能须是同脚本相对比较+多轮均值下超过 DVFS 噪声带的真增益；改 csrc 须对应 venv 重 `pip install -e . --no-build-isolation`；git 干净、author=`kyle-256 <Kyle.Zhao@amd.com>` 无 coauthor；只在用户明确要求时 commit/push。
  - （源自 gpt_oss2 mxfp8 项目 pr-merge-gate/SKILL.md，非本环境）：该项目门禁另含 `pytest <file> -q -p no:randomly` + `--deterministic-only -q -p no:randomly`，无 `Fatal Python error: Aborted` / HIP 非法访存，det 10 次 repeat bit-exact **0 失败是红线**，改 csrc/triton 用 `setup.py build_ext --inplace` 重 build。

- **绿了先证伪（DVFS 功耗墙 + JIT 缓存陈旧，本环境）**：fp8/mxfp4 GEMM 在真实幅度数据下是 DVFS 功耗受限，同核 zeros→randn 速度差 21-37%，跨脚本绝对 TFLOPS 方差 ~8-12%，极易把噪声/频率差当真增益。证伪手段：同脚本同数据相对比较（只信比值不信绝对值）；多轮均值（单轮噪声 ±0.3-0.5pp）；复用同一输出 buffer 防 overlap 膨胀虚高吞吐；`rocm-smi` 查是否掉频。增幅 < DVFS 噪声带（~2-3%）视为噪声，不放行。
  - （源自 gpt_oss2 mxfp8 项目 pr-merge-gate/SKILL.md 的"绿了也要先证伪"一节，非本环境）：该项目讲的是 `torch.empty` 假绿——低精度 grouped GEMM kernel 靠 allocator 碰巧非 NaN 通过，无关改动改变 allocator 布局会把 latent 越界/未初始化读从隐形变 NaN/Abort。证伪手段：`-e PYTORCH_NO_CUDA_MEMORY_CACHING=1` 复跑（由红变绿=未初始化内存/越界被布局掩盖，非真修）；full suite 连跑 ≥2 次确认稳定；同节点 A/B（`git stash` diff 跑 baseline、pop 跑 diff）锁定触发者。

---
来源: pr-merge-gate/SKILL.md, gpu-fleet-tuning/SKILL.md, 14-fused-preshuffle-e2e.md, format-code/SKILL.md, 13-primus-turbo-prod.md, 12-llama-aiter-baseline.md, remote-sync/SKILL.md
