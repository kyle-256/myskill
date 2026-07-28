# syncv3 三-repo / 三-venv 环境（wgrad skew 工作）★踩坑最深

> 类别: 连接 · 主题标签: syncv3, venv隔离, source-activate坑, PYTHONPATH-override, editable-finder, import路径验证, 血泪教训

容器 `mlperf_gptoss` @ `chi2798`，宿主盘 `/mnt/vast/kyle/code2` ↔ 容器 `/workspace/code`（bind-mount，改宿主即容器可见）。**同一容器里有三份 Primus-Turbo repo，三个 venv，一一对应**。选错就测错代码。

## ★ 三 repo ↔ 三 venv 对应表（`bin/python -c "import primus_turbo; print(__file__)"` 实测）

| venv 解释器（直接调）             | import 的 repo                                   | HEAD       | 用途 |
|----------------------------------|-------------------------------------------------|------------|------|
| `/opt/venv/bin/python`           | `/workspace/code/Primus-Turbo`                  | e361efe3   | main（生产/bwd16384 campaign）|
| `/opt/venv-tw/bin/python`        | `/workspace/code/Primus-Turbo-tensorwise`       | —          | tensorwise 分支 |
| `/opt/venv-syncv3/bin/python`    | `/workspace/code/syncv3/Primus-Turbo`           | fa1ac492   | **wgrad skew 优化（我的 fix 在这）** |

- 宿主对应：`/workspace/code/syncv3/Primus-Turbo` = 宿主 `/mnt/vast/kyle/code2/syncv3/Primus-Turbo`（= push.sh 的 target）。

## ★★ 头号坑：必须直接 `bin/python`，严禁 `source activate`（尤其别叠加）

- **正确**：`/opt/venv-syncv3/bin/python <script>`。直接调解释器 → 稳定 import 对应 repo。
- **错误（我血亏半天的）**：`source /opt/venv-syncv3/bin/activate; source /opt/venv-tw/bin/activate; python ...`。
  - 叠加两个 activate → 后一个覆盖 PATH，`python` 串到别的 venv；即使只 source 一个，`bash -lc` 环境下 `python` 解析也可能不是你以为的那个。
  - **后果**：我一直 `source activate` 跑，实际 import 的是 `/workspace/code/Primus-Turbo`（main，**旧 geomean autotune，skew 崩 0.59**），却以为在测 syncv3 的 fix。把"fix 无效、FLY skew 崩溃"当成结论写进 memory —— **全是串环境的假数据**。改用 `/opt/venv-syncv3/bin/python` 直接跑，同一份代码 flat 0.984。

## ★ 跑任何测量前，先验证 import 路径（不验证 = 可能在骗自己）

```
/opt/venv-syncv3/bin/python -c "import primus_turbo; print(primus_turbo.__file__)"
# 必须打印 /workspace/code/syncv3/Primus-Turbo/... ，否则你测的不是 syncv3 的代码
```

## editable finder 机制 + 容器重建复位坑

- `primus_turbo` 是 PEP660 editable 安装。**三个 venv 共享同一 site-packages** `/opt/venv/lib/python3.12/site-packages`，里面唯一一个 `__editable___primus_turbo_0_0_0_finder.py` 的 `MAPPING` dict 决定 `import primus_turbo` 落到哪个 repo。
- 该 finder 用 `sys.meta_path.append()` 注册 → **排在默认 `PathFinder` 之后**。所以 `sys.path`/`PYTHONPATH` 里的路径会**先**命中，可 override finder。
- **容器重建会把 MAPPING 复位**（[[project_claude_container_persist]]：重建清 /root/环境）。memory 曾记"指针已改指向 syncv3"，但复位后 finder 又指回 main —— 别信旧 memory，**每次开工先跑上面的 import 验证**。
- 三个 venv 能各指各 repo，靠的是各自 `bin/python` 的 `sys.path`/finder 差异；`source activate` 会打乱这层，故一律直接 `bin/python`。

## PYTHONPATH override 兜底（finder 万一又指错时）

```
PYTHONPATH=/workspace/code/syncv3/Primus-Turbo /opt/venv-syncv3/bin/python <script>
# PathFinder 先于 append 的 editable finder 命中 → 强制 import syncv3 repo（已实测 K.__file__=syncv3, HAS_TRACE=True）
```

## 本地设施（`sync/syncv3/`，见 [[project_syncv3_env]]）

- `rexec.sh`：base64 编码远端执行，绕 ssh+docker+bash 嵌套引号。用法 `bash rexec.sh [-g <gpu>] '<cmd>'`；固定 NODE=chi2798 / CONTAINER=mlperf_gptoss。
- `push.sh`：rsync 本地 `sync/syncv3/Primus-Turbo` → 远端 `/mnt/vast/kyle/code2/syncv3/Primus-Turbo`（= 容器 `/workspace/code/syncv3/Primus-Turbo`）。**绝不 --delete**（会删远端 .so）。用法 `bash push.sh [<relpath>]`。
  - ⚠️ push 到 syncv3 repo 后，只有 `/opt/venv-syncv3/bin/python` 能测到；用错 venv 就是白 push。
- harness 脚本（`_probe_flat.py`/`_chk_skew.py`/`_probe_gridmul.py` 等）**只在本地** `sync/syncv3/Primus-Turbo/` 顶层，**push.sh/rexec 都不带它们到远端**。跑法 = 把本地文件内容经 stdin 喂远端 python（比 rsync 可靠，不留残留）：
  ```
  CODE=$(cat _probe_flat.py)
  bash rexec.sh -g 4 "cd /workspace/code/syncv3/Primus-Turbo && /opt/venv-syncv3/bin/python - <<'PYEOF'
  $CODE
  PYEOF"
  ```
- 改 @kernel body 后仍需 `rm -rf /root/.flydsl/cache` + 清 `__pycache__`。

## wgrad skew 优化 = 两个可复用杠杆 + 度量铁律（见 [[project_wgrad_skew_distribution_invariant]]）

- **杠杆① flatness by construction**：分布无关必须靠**构造**(只留 flat 的 xcd=1 候选)，不能靠 autotune 竞速——fp8 GEMM 功耗受限，`_robust_time` 随热相位单调漂移 ~37% > skew 惩罚 ~30%，竞速会按采样热相位选到会崩的配置。autotune operands 要 seeded(fp8 data-dependent)。
- **杠杆② grid over-subscription**：变-K grouped 内核 grid=ncus 时 hot-group 长-K tile 是 HW 调度器回填不了的**尾**；launch grid ×2(`PT_WL_GRIDMUL`)让调度器动态回填。判尾 vs in-kernel：rocprofv3 PMC 看 **MfmaUtil bal vs skew 相同=尾**(idle CU) / 不同=overhead。
- **度量铁律**：flatness(同进程逐 rep 交织 bal/skew 同比)是 **drift-immune** 可跨 run 信；但 **cross-run 绝对 TF 是热噪声**(GPU throttle 砍半，见过同 kernel 683 vs 926)→ 比绝对量/比不同配置**必须单进程交织**(`_probe_gridmul.py` 一个进程里交织 mul×dist)，否则拿的是热相位假数据。

---
来源: 本 session 环境排查 + wgrad skew 收官（2026-07-27）；关联 02-run-and-venv.md, 03-sync-and-remote.md, [[project_syncv3_env]], [[project_wgrad_skew_distribution_invariant]]
