# gpt_oss2 venv 新建/重建 与 csrc build 规则

> 类别: 连接 (gpt_oss2) · 主题标签: venv-rebuild, csrc-build, build_ext, _C-extension

## 重编 csrc 铁律
- **必须** 用 `setup.py build_ext --inplace`，**不要** 用 `pip install -e .`。
  - WHY：前者只重 build .so（在 repo 内 vast），不动 finder，零污染；后者会触发坑B。
- 命令带 `-e GPU_ARCHS=gfx950 -e MAX_JOBS=128`（或 `-e MAX_JOBS=64`）。
- 耗时：增量重编 ~3min；冷全量含 `-fgpu-rdc` 设备链接 ~8-12min。

## _C 扩展 build（mxfp4 树）
- `_C` 扩展若未 build（容器重 checkout 会丢），任何 `import primus_turbo.pytorch` 全挂：`ModuleNotFoundError: primus_turbo.pytorch._C`。
- build 命令（后台）：`GPU_ARCHS=gfx950 MAX_JOBS=64 pip install -e . --no-build-isolation`，约 7min。
- **无-_C bit-exact 门**：FLY 模块 `flydsl.gemm.mxfp8_quant_flydsl` + `flydsl.utils.gemm_helper` 只依赖 flydsl+torch，不拉 `pytorch._C`。
  - 故 bit-exact 可先用无-_C 精简脚本（如 `_gg_bquant_bitexact.py`）验证 → 再 build `_C` 跑 HIP 对比/基准/e2e。
  - WHY：先过 bit-exact 门再花 ~7min build，省迭代时间。

## 快速重建 mxfp4 venv（~15s）
- 场景：从 tar 起的新节点、mxfp4 .so 已在 code3(vast) 上。
- **不要** 跑 `pip install -e .`（会触发坑B）。步骤：
  1. `cp -a /opt/venv /opt/venv-mxfp4`（reflink clone）
  2. `sed` 修 `setuptools.pth`（坑A）
  3. `sed` 把 finder 从 mxfp8 改成 mxfp4 repo（坑B）
  4. `docker commit` 持久化
- 仅当 .so 缺才需 `setup.py build_ext --inplace`。

## 新建 turbo venv 完整流程
1. 本地 clone + checkout 分支
2. rsync 推源码（含 `.git`、`--mkpath`）
3. 拷 3rdparty 子模块：`composable_kernel` 只需 headers ~73M；`hipify_torch` 空也能 build
4. `cp -a` 建 venv，修坑A
5. 首次 `pip install -e . --no-build-isolation --no-deps` 注册 editable
6. 修坑B → 验证 → `docker commit`

---
来源: remote-sync/SKILL.md, mxfp8-8wave-devloop/SKILL.md, mxfp8-grouped-gg-devloop/SKILL.md
