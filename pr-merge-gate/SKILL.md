---
name: pr-merge-gate
description: 决定 FlyDSL / Primus-Turbo（mxfp4、tensorwise）GEMM 内核改动什么情况下能 commit / push / 交 PR（merge-ready）。当用户说"能不能提/推/合""放行 PR""这改动算过了吗""commit 完要不要 push""SNR 对了能合吗""这个优化能进吗"时使用。核心：①SNR 绿 ≠ 真优化（功耗墙 + JIT 缓存会造假象，先证伪再放行）②不接永不被采纳的候选 ③代码质量门禁（风格一致/复用/零 dead code/零调试残留）。同步/push 机制见 remote-sync（Primus-Turbo）/ flydsl-sync（FlyDSL）；连节点见 claim-mi355x-node。
---

# pr-merge-gate

什么时候 FlyDSL / Primus-Turbo 的 GEMM 内核改动**能放行**（SNR 已对，问题是这算不算真改进、能不能 commit / push / 交 PR）。

> 一句话哲学：**SNR 绿 ≠ 真优化，它可能被 DVFS 功耗墙噪声或 JIT 缓存陈旧掩盖。放行前先证伪（这是真增益吗？这核真编译了吗？），证不伪才放行。** 同步/commit/push 见 `remote-sync`（Primus-Turbo）/ `flydsl-sync`（FlyDSL）；节点/容器见 `claim-mi355x-node`。

## 仓库 / canonical / 验证环境

| 仓 | canonical git | 分支 | venv | 角色 |
|---|---|---|---|---|
| FlyDSL | **远端** `/mnt/vast/kyle/code2/FlyDSL`（本地 `sync/FlyDSL` 是镜像）| `dev/fp8-fused-quant` | `/opt/venv` | `turbo/` fp8/mxfp4 4wave/8wave dense kernel；GPU 编译→远端为主，远端 commit |
| Primus-Turbo (mxfp4) | **本地** `sync/mxfp4/Primus-Turbo` | `dev/kyle/flydsl_mxfp4_compute` | `/opt/venv` | mxfp4 dense GEMM 后端 |
| Primus-Turbo-tensorwise | **本地** `sync/tensorwise/Primus-Turbo` | `dev/kyle/flydsl_grp_gemm` | `/opt/venv-tw` | fp8 grouped + dense GEMM 后端 |

- 验证一律在 **当前节点**（2026-07-03 起 = `chi2811`，chi2810 被别人任务占满、以 claim-mi355x-node 为准）容器 `mlperf_gptoss`（MI355X / gfx950）跑，经 `sync/.ssh-chi.sh` 连。本地（smc300x，无对应 GPU）只能编辑 + 静态查 ISA。
- 选对 venv（remote-sync 布局表）：mxfp4/FlyDSL→`/opt/venv`，tensorwise→`/opt/venv-tw`。

## 放行硬门禁（全过才能 commit/push，缺一不可）

1. **SNR 正确**：改动核在真机跑出预期 SNR。基准：**rowwise ~56dB、tensorwise ~40-47dB（量化误差本身，非 bug）**；odd-K（如 gpt_oss K=2880）也要单独验过。SNR 掉到 30 以下 = 错，不放行。
2. **整除-K / 旧路径零回归**：加 K-tail / 新分支等改动，对原本就整除的 K（K%128==0）必须是**编译期 no-op**，老 shape SNR 不变。回归 = 不放行。
3. **JIT 缓存确实失效（FlyDSL 必查）**：env-gated 探针（`const_expr(env)`）不改 @kernel 源码 hash → 缓存命中旧核 → 改动**根本没生效**（SNR 仍正确=假象）。换 env-gate 必 `rm -rf /root/.flydsl/cache`；金标准 = 故意把核改算错（scale×2）不清缓存跑，出 NaN/错=失效正常，仍出旧正确 SNR=陈旧。详见 `flydsl-sync §JIT 缓存坑`。
4. **性能是真增益，不是噪声**（见下"证伪"）：fp8 GEMM 是 DVFS 功耗受限，跨脚本绝对 TFLOPS 不可信。只认**同脚本同数据相对比较 + 多轮均值**，且增幅须超过 DVFS 噪声带。
5. **改了 csrc/`.cu`/`.cpp`**（Primus-Turbo）→ 对应 venv 重 `pip install -e . --no-build-isolation` 重编 `.so` 再测，否则测的是旧 `.so`。
6. **git 干净**：working tree 只剩预期改动；探针脚本（`_*.py`/`opt_*.py`）**不入库**；commit message 走 Conventional Commits（`type(scope): ...`）；**author=`kyle-256 <Kyle.Zhao@amd.com>`、无任何 Claude/Cursor coauthor**（见 `feedback_commit_style`）。
7. **只在用户明确要求时 commit / push**。没让推就别推。

## 代码质量门禁（diff 自查，红线，和正确性同级）

> 顺序：**先工具全绿，再人工审，最后放行**。SNR 绿但代码脏一样不放行。

### A. lint 必须和仓库 CI 同源

两份 Primus-Turbo 都已迁 **ruff**（PR #392 已合，`requirements.txt` pin 了 ruff/pre-commit/clang-format 版本）。提交前 `pre-commit run --files <改动>` 必须全 Passed：

- `ruff check`（启用 `B/F/I/W`，忽略 `F403/F405/B905/B007/B028`）+ `ruff format --check`（`line-length=110`）
- `clang-format(--style=file)`（`.c/.cc/.cpp/.cu/.h/.hpp/.cuh`）+ `shellcheck`
- 通用 hooks：`trailing-whitespace / end-of-file-fixer / check-yaml / check-toml / check-ast / check-added-large-files / check-merge-conflict / check-case-conflict / debug-statements / detect-private-key`
- 被自动改了就再跑一遍确认干净。确认本地 ruff/pre-commit 版本与 `requirements.txt` 锁的一致。
- FlyDSL 仓无 Primus-Turbo 那套 pre-commit；按其自身规范 + 下面 B/C 人工审。

### B. dead code & 调试残留（红线，工具+人工双查）

| 类别 | 工具 | 处理 |
|---|---|---|
| 未用 import / 变量 | ruff `F401`/`F841` | 删 |
| 未用函数/类/私有 helper | 抓不到 | `rg '<name>\b'` 全仓搜，0 命中即删 |
| 注释掉的旧代码 / 占位 | 抓不到 | 删 |
| `print` / `breakpoint()` / `pdb` | ruff `debug-statements`/`T20` | 删 |
| re-raise 丢异常链 | bugbear `B904` | `raise ... from err` |
| 私钥误入库 | `detect-private-key` | 立即移除 |
| **调试/实验痕迹** | 抓不到 | 见下，严禁入库 |

- **注释红线**：comment 尽量精简，只解释 why / 约束 / 非显然 trade-off，**不复述代码在做什么**。**严禁任何调试/实验痕迹**：跑数结果（"58dB 正确""测了 X 不行"）、调参备忘、`# debug`/`# tmp`/`# 临时`、带日期的过程笔记。这些进 commit message / memory / PR 描述，不进源码。放行前 `rg -i 'debug|tmp|临时|verified|实验|print\(|TODO' <改动文件>` 自查，0 命中。
- **探针脚本不入库**：`_g_*.py`/`_d_*.py`/`opt_*.py` 等是调查/bench 用的 scratch，留在工作树或本地即可，**绝不 `git add`**（git 干净门禁第 6 条）。
- 删完重跑 `pre-commit` + import 冒烟，确认没删错。

### C. 一致性 & 复用

- **照抄既有模式**：4-wave 对着 8-wave、NN 对着 NT/TN、tensorwise 对着 mxfp4 逐字段抄（命名、错误处理、dispatch 候选 grid 结构、`group_n` band 写法）。新加 K-tail/band/vmcnt 等 lever 时镜像姊妹实现（如 `_nn_block_mn` 镜像 `_tn_block_mn`）。
- **最大化复用**：`mask_a_tail`/`_a_tail_mask_vec`、`emit_wholeloop_tile`、`grouped_block_mn`、各 `S2RLoader`/`G2SLoader` 等既有原语优先，别另写一份。
- **禁魔术数**：tile/BLOCK 常量、swizzle 维度从 shape 推导或用既有常量，别散落硬编码。

### D. PR 自查清单（逐条勾）

- [ ] 功能完整，做到 PR/commit 声称的事
- [ ] 难懂处注释解释 why/约束（不复述 what）
- [ ] 行为/API 变了 → docstring/README 同步
- [ ] 无新增 warning
- [ ] 有证明改动的验证（SNR + 多轮性能；必要时 pytest）
- [ ] commit 走 Conventional Commits（type ∈ feat/fix/opt/refactor/test/chore/ci），分支名同前缀
- [ ] 探针脚本未入库、无 coauthor、author=kyle-256

## 绿了也要先证伪（这是本 codebase 的核心教训）

fp8 GEMM 在真实幅度数据（randn，量化后铺满 [-448,448]）下是 **DVFS 功耗受限**：同核 zeros→randn 速度差 21-37%，跨脚本绝对 TFLOPS 方差 ~8-12%（数据分布→频率差）。所以"测出更快"极易是噪声/频率差，不是真增益。

- **可疑信号**：跨脚本比较绝对 TFLOPS；单轮就下结论；增幅在 ±2-3% 内；换了输入数据/分母核后结论翻转。
- **证伪手段（放行前对"性能增益"类改动跑）**：
  - **同脚本同数据相对比较**：A/B 必须在同一脚本、同一输入、同一计时法（warmup + reps + median）里背靠背比，只信比值（如 4w/8w），不信绝对值。
  - **多轮均值**：单轮噪声 ±0.3-0.5pp；达成率/比值取多轮均值，别拿单轮峰值放行。
  - **复用同一输出 buffer 防 overlap 膨胀**：连续 launch 写同一 buffer（WAW 串行），避免新分配 buffer 导致 kernel 重叠虚高吞吐。
  - **JIT 缓存核查**（硬门禁 3）：env-gate 改动先确认缓存失效，否则"增益"可能是根本没跑新核。
- **判增益 vs 噪声**：增幅 < DVFS 噪声带（~2-3%）= 视为噪声，**不作为优化放行**；要进生产 autotune 候选的，须超过既定**采纳门槛**（4-wave vs 8-wave 用 >4% 滞后，正是为压住 DVFS 噪声）。
- **节点抖动**：`rocm-smi` 看是否掉频/被别人抢卡；transient 掉频会把 baseline 拉低制造假胜（如曾把 8w 瞬时掉频测成 4w 赢 1.42×，复测 1.01×）。可疑就复测。

## 不接"永不被采纳"的候选（本会话教训）

新内核变体（如某 wave/persistent/swizzle 组合）要接进生产 autotune dispatch 前，先证明它**在真实全量 8-wave/baseline 候选池里**能按采纳门槛（>4%）被选中：

- 拿**真实 dispatch 的 best**当分母，不是手挑的弱子集（手挑弱子集会高估新候选）。
- 若它能赢的 regime 恰好是它输给已有候选的 regime（两区间不重叠），或对真实 best 过不了门槛 → **不接**，否则只是给每个 shape 白加编译开销、零采纳。
- 例：grouped NT 4w-persistent 对真实 8w-best 仅 ~3% < 4% 门槛、且能用 persistent 的短-K 区它又输 4w-np → 已撤销，净改动为零。

## 不能放行 / 必须上报用户的情况

- **正确性 bug**（SNR 错、odd-K 不对、HIP 非法访存/Abort、整除-K 回归），哪怕某配置看着绿。
- **改动是噪声级"增益"**：拿不出超噪声带的同脚本多轮证据，不要当优化交。
- **怀疑自己的 diff 导致回归**：先 A/B（git stash diff 跑 baseline，pop 再跑）自证清白或定位真因，**别反射性甩锅 diff、也别甩锅环境**——先拿数据。
- ⚠️ **报告口径（用户硬性要求，见 `feedback_no_ceiling_claims`）**：**严禁说"天花板/固有上限/不可达/收益低/到头了"**。即使某轮所有尝试判负，只陈述"这些判负 + 根因 + 下一步还能试什么"，继续干，不得升级成"上限"结论给自己收工。
- lint 没过、build 没过、SNR 没过 —— 任意一条没解决。

## 放行流程（门禁全过后）

**FlyDSL（canonical 在远端 → 远端 commit）：**
```bash
# 0. 本地编辑 sync/FlyDSL → rsync 推远端 → 远端 GPU 验 SNR（见 flydsl-sync）
# 1. 远端 commit（显式覆盖 author=kyle-256，无 coauthor）
./.ssh-chi.sh root@chi2810 "cd /mnt/vast/kyle/code2/FlyDSL && git add <files> && \
  GIT_AUTHOR_NAME=kyle-256 GIT_AUTHOR_EMAIL=Kyle.Zhao@amd.com \
  GIT_COMMITTER_NAME=kyle-256 GIT_COMMITTER_EMAIL=Kyle.Zhao@amd.com \
  git commit -m 'type(scope): ...'"
# 2. git 历史 + 文件 rsync 回本地对齐（见 flydsl-sync §5），校验 HEAD 一致
```

**Primus-Turbo（canonical 在本地 → 本地 commit + SSH-key push）：**
```bash
cd /workspace/code/gpt_oss_docker/sync/<mxfp4|tensorwise>/Primus-Turbo
git add <files> && GIT_AUTHOR_NAME=kyle-256 GIT_AUTHOR_EMAIL=Kyle.Zhao@amd.com \
  GIT_COMMITTER_NAME=kyle-256 GIT_COMMITTER_EMAIL=Kyle.Zhao@amd.com \
  git commit -m "$(cat <<'EOF'
type(scope): summary
EOF
)"
# push 用本机 SSH key + git@ URL 临时覆盖 HTTPS origin（见 remote-sync §Git push）
GIT_SSH_COMMAND="ssh -i /workspace/code/.ssh_docker/id_ed25519 -o StrictHostKeyChecking=no -o IdentitiesOnly=yes" \
  git push git@github.com:AMD-AGI/Primus-Turbo.git <branch>
git rev-list --left-right --count origin/<branch>...HEAD   # 期望 0  0
```

## 已知地雷（institutional knowledge）

- **DVFS 功耗墙**（`project_flydsl_fp8_power_limited`）：fp8 GEMM 真实数据下功耗受限，forward 的指令效率优化（4wave/asm/少 barrier）被掩盖（相同 MFMA→相同功耗→相同频）。所以这些"优化"在 randn 上常打平，别当增益放行；真增益在 4-wave 被采纳的 regime（wgrad/dgrad/大-K/宽-N）+ 同族对比。
- **JIT 缓存陈旧**（`flydsl-sync`）：env-gate 探针不失效 → 跑旧核出假象。换 env 必清缓存。
- **本地 sync 可能比远端旧**：移植/对照前先 `diff` 本地 vs 远端容器那份（bench 实际跑的是远端）。
- **跨脚本绝对 TFLOPS 不可信**：只信同脚本相对比较 + 多轮均值。

## 不要做的事

- SNR 绿就推 —— 没过"证伪"（这是真增益吗/核真编译了吗）。
- 把噪声级波动当优化交；拿单轮峰值放行。
- 接一个真实 dispatch 永不采纳的候选（白加编译开销）。
- 在源码注释留调试/实验/跑数痕迹；把探针脚本 `git add` 进库。
- 说"天花板/不可达/收益低"给自己收工（用户硬性禁止）。
- 在远端跑 `checkout/reset/pull`（FlyDSL 远端是 canonical 可 commit，但别乱 reset）；加 Claude/Cursor coauthor；没让 commit/push 就自作主张。
