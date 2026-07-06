# 代码风格与工程规范

> 类别: 方法论 · 主题标签: code-style, naming, gemm_helper, primus-turbo-layers, git-rebase

## turbo/FlyDSL 代码风格:英文注释、命名规范、复用 helper、五层垂直切片

### 注释与命名
- 注释**一律英文**,禁中/日文。
- 注释**尽量精简**:**严禁单块注释/docstring 超过 5 行**。要点浓缩成 1-2 句;推导过程/实测数字/多方案权衡进 commit message / memory / `tuning_results/`,不进源码(见 pitfalls/10「注释里的 benchmark 数字」)。code review 硬门禁一条:`>5 行`连续注释即打回。
- 函数命名跟 turbo 现有风格,别自造缩写:
  - `_grouped_<noun>`,如 `_grouped_block_mn`。
  - `_wgrad_<verb_or_noun>_<variant>`,如 `_wgrad_wholeloop_asm_3buf`。
  - 反例:`_wl3buf_fused_tail_split` ❌ → 应 `_wholeloop_tail_split_3buf`。
- 新 kernel 变体**照抄既有模式**逐字段抄命名/错误处理/dispatch 候选 grid/group_n band(4-wave 对 8-wave、NN 对 NT/TN、tensorwise 对 mxfp4)。
- **禁魔术数**:tile/BLOCK 从 shape 推导。

### 复用 helper(先 grep 再写)
- 能复用 `gemm_helper.py` 就**必须**复用,现成原语:`ceildiv`、`_readfirstlane_i32`、`xcd_remap_pid`、`make_fp8_buffer_tensor_rebased`、`S2RLoader`/`S2RLoaderTr`、`_robust_time`。
- 最大化复用既有原语:`mask_a_tail`/`emit_wholeloop_tile`/`grouped_block_mn`/`S2RLoader`/`G2SLoader`。先 grep 确认没有再写。

### CI 风格唯一真相源 = `bash scripts/check_python_style.sh`
- Python:black **line-length 120**;ruff 查 **E/W/F/I**;isort 把 `flydsl` 当 first-party。black/ruff 从 repo root 跑自动读 `pyproject.toml`(`[tool.black]`/`[tool.ruff]`,line-length 120)。
- C++:LLVM style,`clang-format-18`,`.clang-format` **ColumnLimit 100**。别传 `--style`,让它读仓库 `.clang-format`。
- **注意 100(.clang-format) vs 120(pyproject) 不一致**。
- 用法:
  - 只检查不改文件:`bash scripts/check_python_style.sh`。
  - 修复:`--fix`(内部按 **black → ruff check --fix → black** 顺序,精确匹配 CI)。
  - 缺工具:`--install`(装 CI 固定版本)。
- **默认只查已提交 `origin/main..HEAD` diff**(匹配 CI 在 pushed branch 所见);`--include-local` 才纳入未提交/staged/untracked;`--fix --include-local` 也格式化本地未提交文件。
- 格式化只对改动文件的 staged+unstaged 并集去重:`(git diff --name-only --cached; git diff --name-only) | sort -u`;无改动则不做事。in-place 编辑会让之前 staged 的文件再次显示 modified → **提醒重新 `git add`**。

### lint 与 CI 同源(Primus-Turbo 已全面迁 ruff,别再套 black 那套)
- **两份 Primus-Turbo(tensorwise、mxfp4)当前签出分支均已完成 ruff 迁移**(PR #392 已合,`bc6c7d2a` 是两分支 HEAD 的祖先):`pyproject.toml` `[tool.ruff]` line-length=110、select **B/F/I/W**、ignore **F403/F405/B905/B007/B028**;CI 跑 `ruff check` + `ruff format --check`,加 check-toml/check-ast/check-case-conflict/debug-statements/detect-private-key;ruff/pre-commit/clang-format 版本 pin 在 `requirements.txt`。`.pre-commit-config.yaml` 也已配 ruff-pre-commit,**没有 black/autoflake/isort 钩子**,不存在仍在用 black 的现行分支。
- 上面 CI 风格一节讲的 FlyDSL 仓库 black(line-length 120)+ruff lint(E/W/F/I) 是另一套体系(FlyDSL 专用),别和 Primus-Turbo 的 ruff-only 体系混为同一议题的两个时间阶段。
- 透传形参为对齐姊妹 API 不算 dead(ruff 的 unused-arg 别误删)。

> （源自 gpt_oss2 mxfp8-grouped-gg-devloop 项目,非本环境）**format 债 vs feature 拆两 commit**:若某文件里有一批 ruff 迁移前就存在的 `ruff format` 不合规行,你的改动一碰这些文件会连既有债一起报红,别把重排版和 feature 混进一个 commit。做法:先 `git stash` feature 改动 → `ruff format` 全文件 → 提一个纯 style commit(`style(...): ruff-format ...`,只有重排版、零逻辑);再 `git stash pop` → `ruff format` → 提干净的 feature commit;若 stash pop 因排版基线变化冲突,直接以格式化后基线为准重放语义 hunk;两个 commit 各自 `ruff format --check` 全绿再 push。(源: gpt_oss2 `myskill/pr-merge-gate/SKILL.md`、`myskill/mxfp8-grouped-gg-devloop/SKILL.md`)

### Primus-Turbo 五层垂直切片
每个 operator 是穿过 5 层的垂直切片,**开发第一步是判断改哪层**:
1. `modules/` — nn.Module 包装。
2. `ops/` — Python API + `torch.autograd.Function`,用户面。
3. `kernels/` — `AutoKernelDispatcher` + `KernelBackend` 选后端。
4. `triton/` — Python kernel,**无 rebuild**。
5. `csrc/` — HIP/CK/hipBLASLt,**需 rebuild**,经 `TORCH_LIBRARY` 绑到 `torch.ops.primus_turbo_cpp_extension.*`。

#### 目录根不一致的坑
- `ops/` = `primus_turbo/pytorch/ops/`
- `kernels/` = `primus_turbo/pytorch/kernels/`
- `triton/` = `primus_turbo/triton/`(**顶层,不在 pytorch/ 下**)
- `csrc/kernels/`、`tests/`、`benchmark/` = repo 根。
- 用户 API 在 `ops/`;`kernels/` 放 dispatcher 或 backend 实现(`*_impl.py`)。

### 三种 op 接线模式
- **Multi-backend**(gemm/gemm_fp8/grouped_gemm):ops → kernels dispatcher → Triton 和/或 csrc。
- **Direct C++**(rmsnorm):ops autograd Function 直接 `torch.ops.primus_turbo_cpp_extension.*`,无 dispatcher。
- **Direct Triton**(swiglu_with_probs):ops Function → kernels/ 里 Triton helper,无 dispatcher。
- 最小可用 op = `ops/` Function + 一个 kernel;dispatcher 和 module 有需要才加。

#### 抄源对照(找最接近的现有 op 抄它的垂直切片改)
| 场景 | 抄源 |
|---|---|
| 简单 2 输入 op | `ops/gemm.py` |
| 多 backend 带 quant 变体 | `ops/gemm_fp8.py` |
| grouped/变形 | `ops/grouped_gemm.py` |
| Direct C++(无 dispatcher) | `ops/normalization.py` |
| Direct Triton | `ops/activation.py` |

#### ops/ 层约定
- 早验证维度;`out_dtype` 为 None 时用 `torch.result_type` 推断。
- 用 `ctx.save_for_backward` 传中间量;每个 forward arg 返一个 grad(非 tensor 返 None)。
- 喂需要连续内存的 kernel 前 `grad_out = grad_out.contiguous()`。
- FP8/FP4 由 `config.granularity` 选跑哪个 `autograd.Function` 变体。

#### kernels/ dispatcher 模式(抄 `kernels/gemm/gemm_impl.py`)
- 每 backend 一个 `KernelBackend` 子类带 `can_handle` + `execute`。
- 一个 `AutoKernelDispatcher` 子类声明 `_backends`(BackendType→BackendEntry)和 `make_key`。
- `*_impl` 入口用 `@torch.library.custom_op(device_types="cuda")` 包装并配 `@*_impl.register_fake` 孪生,内部调 `Dispatcher.dispatch(default_backend, user_backend, **kwargs)`。
- 加新 backend env/setter:扩 `core/backend.py` 和 `common/constants.py`,照 `_gemm_backend`/`ENV_GEMM_BACKEND` 模式。
- Triton kernel 放 `primus_turbo/triton/<area>/`,用 `torch._library.wrap_triton` 在 kernels 层 launcher 里绑定以兼容 torch.compile(参考 `kernels/attention/attention_triton_impl.py`)。

### 暴露 HIP/CK kernel 到 PyTorch — 四处改
1. kernel 放 `csrc/kernels/<area>/` 编入 `.so`;arch 专用用 `gfx942`/`gfx950` 后缀。
2. 在 `csrc/pytorch/extensions.h` 声明 host 函数及其 `_meta` 孪生。
3. 在 `csrc/pytorch/bindings_pytorch.cpp` 注册 schema(`TORCH_LIBRARY`)+ 真实实现(`TORCH_LIBRARY_IMPL CUDA`)+ shape 推断(`TORCH_LIBRARY_IMPL Meta`)。
4. Python 从 `torch.ops.primus_turbo_cpp_extension.<name>` 调。
- 公共头文件在 `csrc/include/primus_turbo/`。C++ 绑定公共 plumbing 所有 csrc op 共用:声明在 `extensions.h`,注册在 `bindings_pytorch.cpp`。
- Attention backend 有 aiter/triton/turbo 三种(`kernels/attention/`);GEMM FP8 有 CK(row/block)和 turbo。

### git rebase(tensorwise 分支 rebase 到 main,单 commit 历史)
```
git fetch origin main
git branch backup                       # 备份
git reset --hard origin/main
git checkout backup -- <分支独有文件>    # 整文件取一方
git commit
# force push
```
- 冲突文件**整文件取一方**(`git checkout <ref> -- file`),别 `-X theirs` 逐 hunk 合(两套并行实现会拼出坏代码)。
- 新旧 main 间 `csrc`/`cmake`/`setup` 零改动 → `.so` 不用重编;editable 装的 Python 改动 rsync 后即时生效。

---
来源: remote-sync/SKILL.md, pr-merge-gate/SKILL.md, CLAUDE.md, format-code/SKILL.md, SKILL.md, develop-feature/SKILL.md
