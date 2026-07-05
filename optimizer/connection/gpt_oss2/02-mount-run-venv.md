# gpt_oss2 挂载路径、容器跑法与 venv 隔离/构建

> 类别: 连接 (gpt_oss2) · 主题标签: bind-mount, 路径映射, code3, venv-磁盘, docker-exec, HIP_VISIBLE_DEVICES, flydsl-cache, build, venv-隔离, editable-install, PEP660, primus-turbo, venv-rebuild, csrc-build, build_ext, _C-extension

## gpt_oss2 bind mount 与 host↔容器路径映射

### bind mount 规则
- host bind mount：`/mnt/vast/kyle/code3` → 容器 `/workspace/code`。
- 换算：host 路径去掉前缀 `/mnt/vast/kyle/code3` 换成 `/workspace/code` 即得容器路径（反之亦然）。
- **源/目标一律用 `code3`**：别人的 mlperf_gptoss 挂在 `code2`，勿碰（避免误改他人环境）。

### 两份 turbo host/容器路径
| 分支 | host 路径 | 容器内路径 |
|---|---|---|
| mxfp8 | `/mnt/vast/kyle/code3/mxfp8/Primus-Turbo` | `/workspace/code/mxfp8/Primus-Turbo` |
| mxfp4 | `/mnt/vast/kyle/code3/mxfp4/Primus-Turbo` | `/workspace/code/mxfp4/Primus-Turbo` |

### 磁盘位置（关键：为何别往根盘写）
- `code3` 在 vast 大盘：充裕，含 `.so` build 产物 → 编译产物放这里安全。
- `/opt/venv*` 在容器 overlay 根盘：紧，常年 **97% / 31G** → 别往这里堆东西，会爆盘。

## gpt_oss2 远端容器内跑法:docker exec + HIP_VISIBLE_DEVICES

- **GPU 运行格式**（跳板 → chi2811 → 容器内)：
  ```
  ./.ssh-chi.sh root@chi2811 "docker exec -e HIP_VISIBLE_DEVICES=4 mlperf_gptoss2 bash -lc 'cd /workspace/code/mxfp8/Primus-Turbo && /opt/venv/bin/python <script>'"
  ```
  - `-e HIP_VISIBLE_DEVICES=4`：指定容器内可见 GPU（WHY:多卡机器上锁定单卡跑，避免占用他人卡/结果串扰）。
  - 容器名固定 `mlperf_gptoss2`。
- **两套树 + 两套 venv**（WHY:mxfp8 与 mxfp4 依赖/环境隔离，不可混用）：
  - **mxfp8**：树 `/workspace/code/mxfp8/Primus-Turbo`，解释器 `/opt/venv/bin/python`。
  - **mxfp4**：树 `/workspace/code/mxfp4/Primus-Turbo`（即 mxfp4 树），解释器 `/opt/venv-mxfp4/bin/python`。
- **改 kernel 必清 flydsl 缓存**：改 kernel 后远端必须 `rm -rf /root/.flydsl/cache` 再跑（WHY:否则加载旧 .so，测的是旧内核）。
- **改 csrc/triton 必重 build**：改了 csrc/triton kernel 必须远端重 build（`setup.py build_ext --inplace`）后再跑（WHY:否则测的是旧 .so）。
- 镜像 gpt_oss 的 `09-docker-exec-run-format`（同一套 docker exec + HIP_VISIBLE_DEVICES 跑法）。

## gpt_oss2 两份 Primus-Turbo checkout 与 venv 隔离

- **两份并存的 Primus-Turbo checkout**（都是 git 仓库，本地为 canonical）：
  - **mxfp8** → `sync/mxfp8/Primus-Turbo`，分支 `dev/kyle/flydsl_mxfp8_compute`（2026-06-25 @ `dac31090`）；quant 专题分支为 `dev/kyle/flydsl_mxfp8_quant`。
  - **mxfp4** → `sync/mxfp4/Primus-Turbo`，分支 `dev/kyle/mxfp4_triton_gg`（旧名 `triton_gg` 已废弃）。
- **各自独立 venv**（选错 venv 会 import 错 repo）：
  - mxfp8 用 `/opt/venv`；另建 symlink `/opt/venv-mxfp8` → `/opt/venv`（对称命名）。
  - mxfp4 用 `/opt/venv-mxfp4`。
- **editable(PEP 660) 安装机制**：每个 venv 的 `site-packages/__editable___primus_turbo_0_0_0_finder.py` 里的 `MAPPING` dict 指向某 repo 的 `primus_turbo`，共 3 条映射：`libprimus_turbo_kernels` / `primus_turbo` / `tools`。两 venv 各指各 repo → 实现隔离。WHY：editable finder 用 MAPPING 把 import 路由到源码树，改路径即改所指 repo，无需重装。
- **各 repo 自 build 自己的 .so**：`primus_turbo/lib/libprimus_turbo_kernels.so` 在 vast 上各自编译（rsync 排除 `*.so`，故本地不同步二进制）。
- **验证隔离走对 repo**：
  - `/opt/venv/bin/python -c "import primus_turbo; print(primus_turbo.__file__)"` 应指 `…/mxfp8/…`
  - `/opt/venv-mxfp4/bin/python` 同命令应指 `…/mxfp4/…`

（mirrors gpt_oss 的 10-venv-isolation。）

## gpt_oss2 venv 新建/重建 与 csrc build 规则

### 重编 csrc 铁律
- **必须** 用 `setup.py build_ext --inplace`，**不要** 用 `pip install -e .`。
  - WHY：前者只重 build .so（在 repo 内 vast），不动 finder，零污染；后者会触发坑B。
- 命令带 `-e GPU_ARCHS=gfx950 -e MAX_JOBS=128`（或 `-e MAX_JOBS=64`）。
- 耗时：增量重编 ~3min；冷全量含 `-fgpu-rdc` 设备链接 ~8-12min。

### _C 扩展 build（mxfp4 树）
- `_C` 扩展若未 build（容器重 checkout 会丢），任何 `import primus_turbo.pytorch` 全挂：`ModuleNotFoundError: primus_turbo.pytorch._C`。
- build 命令（后台）：`GPU_ARCHS=gfx950 MAX_JOBS=64 pip install -e . --no-build-isolation`，约 7min。
- **无-_C bit-exact 门**：FLY 模块 `flydsl.gemm.mxfp8_quant_flydsl` + `flydsl.utils.gemm_helper` 只依赖 flydsl+torch，不拉 `pytorch._C`。
  - 故 bit-exact 可先用无-_C 精简脚本（如 `_gg_bquant_bitexact.py`）验证 → 再 build `_C` 跑 HIP 对比/基准/e2e。
  - WHY：先过 bit-exact 门再花 ~7min build，省迭代时间。

### 快速重建 mxfp4 venv（~15s）
- 场景：从 tar 起的新节点、mxfp4 .so 已在 code3(vast) 上。
- **不要** 跑 `pip install -e .`（会触发坑B）。步骤：
  1. `cp -a /opt/venv /opt/venv-mxfp4`（reflink clone）
  2. `sed` 修 `setuptools.pth`（坑A）
  3. `sed` 把 finder 从 mxfp8 改成 mxfp4 repo（坑B）
  4. `docker commit` 持久化
- 仅当 .so 缺才需 `setup.py build_ext --inplace`。

### 新建 turbo venv 完整流程
1. 本地 clone + checkout 分支
2. rsync 推源码（含 `.git`、`--mkpath`）
3. 拷 3rdparty 子模块：`composable_kernel` 只需 headers ~73M；`hipify_torch` 空也能 build
4. `cp -a` 建 venv，修坑A
5. 首次 `pip install -e . --no-build-isolation --no-deps` 注册 editable
6. 修坑B → 验证 → `docker commit`

---
来源: remote-sync/SKILL.md, project_remote_sync_setup.md, flydsl-fp8-gemm-tuning/SKILL.md, pr-merge-gate/SKILL.md, mxfp8-8wave-devloop/SKILL.md, mxfp8-grouped-gg-devloop/SKILL.md
