# 代码风格与生产化部署：源码禁调试痕迹、ruff --fix 陷阱、backend 签名一致、cache key、比较基线

> 类别: 踩过的坑 · 主题标签: 代码风格, merge-gate, ruff, 死代码, authoring-pattern, autotune-dispatch, cache-key, vendor-reuse

## 代码风格门：源码禁调试痕迹/benchmark 数字、探针脚本不 git add、ruff --fix 查 diff

- **注释精简,严禁 >3 行连续注释/docstring**:要点 1-2 句;推导/实测数字/权衡进 commit/memory/`tuning_results/`。放行前 review 逐块查,`>3 行`即打回。

### ★ block/密度 计数器必须数 docstring,copyright 头豁免(2026-08-09 实测)
自建 ship-gate 的 `[block<=3]` 与密度只 grep `^\s*#`,**docstring 是字符串字面量、根本不以 `#` 开头**——于是 8 行、5 行的多行 `"""..."""`(还塞了 `17.0`/`12.75`/`12x12`/`measured flat` 这类实测数字)全程 PASS,连过两轮 workflow review 都没抓到,最后被用户一眼看穿("你不会数数?")。教训:
- **block 连续行计数要把 docstring 计入**(密度沿用 `#` 定义)——用 `ast.get_docstring(node, clean=False).count("\n")+1` 量每个 def/class/module 的 docstring 行数,>3 即 FAIL;禁数字正则也要扫 docstring 文本(`\d+\.\d+|\d+x\d+|dB|TFLOP|measured`)。
- **文件头 copyright/license 块从行数与密度双双剔除**(强制样板,非 WHY,不该浓缩)。
- docstring 改动对 codegen 中性(不进 AST 签名),可放心浓缩;但正因它不进 AST 签名,**行为守门器看不见它 → 必须单独有一条 docstring 计数门**,否则就是这次的洞。
- **源码注释严禁调试/实验痕迹**：跑数结果(如 `58dB 正确`)、调参备忘、`# debug`/`# tmp`/`# 临时`、带日期的过程笔记——这些进 commit message / memory / PR，**不进源码**。放行前必须 `rg -i 'debug|tmp|临时|verified|实验|print\(|TODO'` 得 **0 命中**才放行。WHY：源码是长期资产，调试痕迹是过程噪声，会误导后续读者。
- **探针脚本绝不 git add**：`_g_*.py` / `_d_*.py` / `opt_*.py` 这类临时探针脚本，永远留本地，不入库。
- **tuning 中间态放 `tuning_results/`(项目内)，不写 memory**：每轮 winner / TFLOPS / cfg 属于易变数据，进 `tuning_results/`。memory 只放**不变的约定**——target 来源、`BK=128` 硬约束等。WHY：memory 是稳定索引，塞入易变调参结果会污染。
- **改动对整除-K 必须是编译期 no-op**：加 K-tail / 新分支等改动，对 `K%128==0` 的 shape 必须编译期 no-op，老 shape SNR 不变。回归=不放行。
- **merge-gate 死代码判定**：`留着作校验 harness 用` 若那个 harness 本身是**未入库探针**，则对 merge-ready 而言就是死代码，**必删**。实例：独立 preshuffle 启动链入库了代码但 0 消费者=死代码。
- **生产文件旋钮全 hardcode**：`FP4_*` 全部 hardcode 成生产值；删掉所有 `PT_MX_*` / `PT_` 实验 env、persistent 变体、以及注释里的 benchmark 数字(单形状胜负 / dB / pp / 具体 MxNxK)。

### ruff check --fix 陷阱
- ruff `--fix` 会删 **F401 未用 import** 和 **F841 简单未用变量**并排序 import。
- ❌ 别再试 无脑接受 `--fix` 结果：它可能删掉那些**本意保留**的变量(编译期决策标记 / 局部可读性变量)。自动修复后**必须查 diff**——若改的是行为而非仅格式/import 卫生，恢复行为逻辑再重跑 formatter。
- ruff **不删**未用函数/类，那些需人工审：autoflake/ruff F401/F841 抓不到未用函数/类/私有 helper，自己 `rg '<name>\b'` 全仓搜，**0 命中即删**(实例 `_get_fp4_dtype`)。
- **裸 TODO 处理**：无 issue 号的裸 `# TODO` 要么挂号(关联 issue)要么删，不留悬空。注释掉的旧代码/占位一律删。

### CI 报无关文件失败
- 本地风格检查全过，但 PR CI 仍报**无关文件**失败=PR 分支落后 main，**不是 formatter 的锅**。
- 解法：fetch `origin/main` → merge 进 PR 分支 → 重跑检查 → push merge commit。

## ★★清理探针时误删产品文件：`git diff --name-only` 对未跟踪文件是盲的(2026-07-28 实测)

清掉 campaign 留下的 32 个 `_` 探针时,我用"只保留那两个产品文件"的方式重建分支,
**连带删掉了 `primus_turbo/flydsl/utils/attn_helper.py`(104 KB,自 89b31f29 就在库里,
03357f91 扩充成现在这样)**,而 `flash_attn_fwd.py` 从它 import 27 个符号 ——
分支推上去后任何人 clone 都是 ImportError,fwd 完全跑不起来。commit message 里也没提。

**为什么三次自查都没发现**:删掉后它退化成工作区里的**未跟踪**文件,于是
- `git status` 里混在 `?? _xxx.py` 一堆探针中间,看不出异样
- **`git diff <base> HEAD --name-only` 只报已跟踪文件**,一个从未入库/刚被删的模块永远不出现
- 本地跑测试全绿(工作区有那个文件),CI 之外根本发现不了

**正确的自查方式**(删文件/重建分支后必做):
```bash
git archive HEAD | tar -x -C /tmp/chk && cd /tmp/chk   # 纯 git 内容,不含工作区
# 静态解析每个入口文件的内部 import,逐个确认能在树里找到
```
只信从 `git archive` 解出来的树,别信工作区。另一个便宜的信号:**同目录的兄弟文件**
(`gemm_helper.py` 在库、`attn_helper.py` 不在)——这种不对称几乎总是错误。

**补救时的第二个坑**:我第一次补救直接 `git add` 了工作区那份,结果推上去的是
campaign 改过的版本(比原版多 184 行未验证 knob:LPT_QORDER/PACKED_SOFTMAX/
CLUSTER_NOP/COMPUTE_BARRIER/P_ANCHOR),而那个 campaign 在 r5 就中止、这些从没被 keep、
两个 kernel 也都没引用。**恢复被误删的文件要从原 commit 取(`git show <sha>:<path>`),
不是从工作区拿** —— 工作区那份可能早已被实验污染。

## 生产化部署坑：backend 签名一致/setdefault 遮蔽/out_fp16 cache key/比较基线错

- **execute/can_handle 签名必须与 sibling backend 完全一致（含 preshuffled 形参）**：dispatcher 走 `execute(**kwargs)`，结构性要求所有 backend（HipBLASLt/AITER/FlyDSL）都声明同一套轴。FlyDSL 的 execute/can_handle 若少了 `preshuffled` 形参会崩；正确写法是声明该形参 + `del preshuffled`（收下但不用）。WHY：kwargs 广播给每个 backend，谁少一个轴谁就 TypeError。

- **宽 store 是 bf16-only，fp16 out 必须走窄路径**：`store_tacc_wide` 用 `cvt_pk_bf16_f32`，只产 bf16。fp16 输出要在 wrapper 里强制 `taccw=False` 走窄 store 路径。

- **out_fp16 必须进三处 cache key（launch/AT/split）**：否则 bf16/fp16 两种编译产物串号（同一 key 命中错的二进制），fp16 请求拿到 bf16 kernel。三处都要带 `out_fp16` 才不串。

- **mxfp4 生产默认在 4wave.py PROD setdefault，遮蔽 8wave.py**（部署死坑）：`4wave.py` L287-303 `PROD setdefault(ASMMFMA=6 INPLACE=1 DIAG=1 SINNER=1 MMORD=5 ALT=0 GAVOID=1 SC_VGPR=1 PIN...)`，setdefault 在 `8wave.py` 读 environ **之前**生效。所以单改 `8wave.py` 的 environ 默认无效。
  - ❌ 别再试：把 `8wave.py` 默认 `0→9` 无效——已被 4wave PROD setdefault 遮蔽。要改就改 `4wave.py` L290 的 `MMORD '5'→'9'`。

- **比较基线错会造出假收益**（陷阱）：把 base 设成 `FP4_MMORD=0` 去比，得出 mm5/mm9 "+3~5%" 是假象——mm0 从来不是默认，真实默认早就是 mm5。真实收益 = mm9(4×8) 在 mm5 之上仅 **+~2%**（7b-qkv M8192：4117→4210），整体 <1%。
  - ❌ 别再试：任何"跟 mm0 比"得出的提升都要打折，别当真收益上报。

- 移植 4-wave kernel 不复用 gemm_helper.py（vendor helper 原因）：见 pitfalls/08-sync-build.md「build/环境坑」（vendor helper 为何不复用 gemm_helper.py）

- **Primus-Turbo（mxfp4/tensorwise）放行硬门禁（缺一不可）**：`pre-commit run --files <改动>` 全绿（ruff check/format + clang-format + shellcheck，无 isort/black/autoflake）；SNR 正确（rowwise ~56dB、tensorwise ~40-47dB，odd-K 单独验）；整除-K 零回归；FlyDSL 改动需确认 JIT 缓存确实失效（否则测的是旧核）；性能须是同脚本相对比较+多轮均值下超过 DVFS 噪声带的真增益；改 csrc 须对应 venv 重 `pip install -e . --no-build-isolation`；git 干净、author=`kyle-256 <Kyle.Zhao@amd.com>` 无 coauthor；只在用户明确要求时 commit/push。
  - （另一 MXFP8 grouped 项目的门禁补充，跨项目参考）：该项目门禁另含 `pytest <file> -q -p no:randomly` + `--deterministic-only -q -p no:randomly`，无 `Fatal Python error: Aborted` / HIP 非法访存，det 10 次 repeat bit-exact **0 失败是红线**，改 csrc/triton 用 `setup.py build_ext --inplace` 重 build。

- **绿了先证伪（DVFS 功耗墙 + JIT 缓存陈旧）**：fp8/mxfp4 GEMM 在真实幅度数据下是 DVFS 功耗受限，同核 zeros→randn 速度差 21-37%，跨脚本绝对 TFLOPS 方差 ~8-12%，极易把噪声/频率差当真增益。证伪手段：同脚本同数据相对比较（只信比值不信绝对值）；多轮均值（单轮噪声 ±0.3-0.5pp）；复用同一输出 buffer 防 overlap 膨胀虚高吞吐；`rocm-smi` 查是否掉频。增幅 < DVFS 噪声带（~2-3%）视为噪声，不放行。
  - （另一 MXFP8 grouped 项目"绿了也要先证伪"一节的证伪补充，跨项目参考）：该项目讲的是 `torch.empty` 假绿——低精度 grouped GEMM kernel 靠 allocator 碰巧非 NaN 通过，无关改动改变 allocator 布局会把 latent 越界/未初始化读从隐形变 NaN/Abort。证伪手段：`-e PYTORCH_NO_CUDA_MEMORY_CACHING=1` 复跑（由红变绿=未初始化内存/越界被布局掩盖，非真修）；full suite 连跑 ≥2 次确认稳定；同节点 A/B（`git stash` diff 跑 baseline、pop 跑 diff）锁定触发者。

---
来源: pr-merge-gate/SKILL.md, gpu-fleet-tuning/SKILL.md, 14-fused-preshuffle-e2e.md, format-code/SKILL.md, 13-primus-turbo-prod.md, 12-llama-aiter-baseline.md, remote-sync/SKILL.md

## ★★★★★ 编译缓存 key 漏一维 = 两个 build 撞车,静默算错(2026-09-14)

给 MXFP4 GEMM 加了 `scales_prepacked`(调用方直接给已打包的 scale,kernel 跳过 preshuffle)。
两条路的 launch **语义不同**(一条跑 preshuffle,一条不跑),但 tail-split 的缓存键漏了这一维:

```python
at_key = (M, N, K, _row_b, gm, xcd, gn, _wlv, _elgk, _tw, _coop, out_fp16, accum, scales_prepacked)  # ✓
sk_key = (M, N, K, _row_b, gm, xcd, gn, _wlv, _elgk, ksplit, out_fp16, scales_prepacked)             # ✓
tk_key = (M, N, K, _row_b, gm, xcd, gn, _wlv, _elgk, "tail", ksplit, out_fp16, accum)                # ✗ 漏了
```

后果:调用方在**同一形状**上先用未打包 scale、再用打包 scale,第二次拿回第一次的 build ⇒
**对已经打好包的 scale 再 `pre_ab` 一遍** ⇒ rel_err 0.654(SNR **3.65 dB**)。
代码里上一行明明按 `prepacked=` 取到了正确的闭包,下一行又被撞键的旧 entry 覆盖:

```python
launch = _get_mxfp4_tail_launch(..., prepacked=scales_prepacked)   # 对的
entry = _MXFP4_AT_CACHE.get(tk_key)                                 # 撞键
if entry is None: entry = [launch, None]; _MXFP4_AT_CACHE[tk_key] = entry
raw, compiled = entry                                               # 拿回别人的
```

### 为什么找了两天
症状是「打开 split-K 就错」,于是我一路怀疑 split-K 的 scale 寻址,写了一堆探针,
还因此加了一道「prepacked 强制 ksplit=1」的门(白挡了一条路)。
**相关性被当成因果**:split-K 只是让 tail build **首次被选中**的开关,不是错误来源。

### 定位它的三个正交探针(可复用)
1. 同一份 packed slab 喂给不同 ksplit ⇒ 全对(rel_err ≤0.0023)⇒ **split-K 能正确读 packed**
2. poison workspace 后比对各 mode 写出的 slab ⇒ **逐位相同 0.00%** ⇒ **布局与 mode 无关**
3. 同一 mode 下「打包路径」对(0.0005)而「prepacked 路径」错(0.6544)⇒ **数据对、读法错**

⇒ 数据对 + 布局对 + 写对 + 读错 ⇒ **只可能是取到了另一个 build**。

### ★★★★★ 让故障自己说出形状
把正确性门从 3 个采样格**开到全表**,一眼看出坏的是哪两行:

| 行 | M×N tiles | tiles/256CU | SNR |
|---|---|---|---|
| QKV_wgrad | 24×16=384 | **1.5** | **3.656 坏** |
| FC2_wgrad | 16×56=896 | **3.5** | **3.651 坏** |
| out_proj_wgrad | 16×16=256 | 1.0 | 55.61 好 |
| FC1_wgrad | 112×16=1792 | 7.0 | 55.61 好 |

**非整数倍 ⇒ 走 tail split** —— 一步把范围从整个 kernel 缩到一个函数。
⇒ 定位类 bug,**先花一次 run 把正确性门开到全表,比再扫十个参数值值钱**。

### 推论:哪些维该进 key
| 那一维影响什么 | 判定 |
|---|---|
| **build 语义**(跑不跑某个 pass、参数个数、stub 签名) | **必须进 key**,漏了就是正确性 bug |
| 只影响**调优排名**(swizzle / 流水深度 / launch mode) | **别按「显然该修」推,必须实测** |

同一天的反例:`_MXFP4_CFG_CACHE` / `_MXFP4_KSPLIT_CACHE` 也漏了 `prepacked`,
看着是同一个 bug,补上后**实测 9/12 → 7/12**(prepacked 对着真臂自己调,反而比蹭
重排 build 的 config 差 2.4~5.5%,冷卡调优的解释已被实验排除)⇒ **已撤回**。

### 相关:两个 stub 必须签名不同
同一天另一处:prepacked 与非 prepacked 的 jit stub 如果**签名相同**,
第二个会被发回第一个的编译产物(preshuffle 还在里面,写坏调用方的 scale)。
`launch_mxfp4_fused_prepacked` 必须**少掉 `A_raw`/`B_raw` 两个参数**才算两个 stub。
⇒ 光靠函数体里一个 traced 分支区分不开。

来源: 2026-09-14 `f9d65d32`;见 [[project_mxfp4_llama_gemm_b200_campaign]]
