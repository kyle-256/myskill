# 按需 build:何时重编 .so vs 清 flydsl cache

> 类别: 连接 · 主题标签: build, flydsl-cache, csrc-recompile, hipify-pitfall

## Fresh 容器首次安装(flydsl 没装)
- 现象:fresh `rocm/primus:v26.2` 容器里 flydsl 没装 → 落 stub 分支报 `ImportError: cannot import name '_compile_dense_tn'`(`flydsl_available()` 返回 False)。
- 正解:从源码 editable 安装,**不要 PYTHONPATH hack**。PYTHONPATH 只够 `import flydsl`,但编译 kernel 还要 `_mlir` 的 `LD_LIBRARY_PATH`;editable install 会自动 symlink 解决。
  - `pip install -e FlyDSL`:build-fly/ 已存在时只 symlink(~30s),不重 build MLIR。
  - `GPU_ARCHS=gfx950 pip install --no-build-isolation -e Primus-Turbo`:~15-25min。

## 改动后要重编什么(决策表)
| 改了什么 | 动作 | WHY |
|---|---|---|
| flydsl kernel(.py)/ 纯 .py | `rm -rf /root/.flydsl/cache` 再跑 | 无需重编 .so;不清 cache 会跑到旧编译产物 |
| cpp/csrc/.cu/.cpp | clean 重编 .so | 否则测的是旧 .so,读到错误值 |

- csrc 重编命令:`rm -rf build primus_turbo/lib/*.so && GPU_ARCHS=gfx950 pip install --no-build-isolation -e .`(~15-25min);tensorwise 用对应 venv。重编产物是 `libprimus_turbo_kernels.so`。

## 陷阱:增量 pip install -e . 不真重编 csrc
- 改 csrc 后做增量 `pip install -e .` **不会真重编**变动的 hip 对象:
  - `rsync -a` 保留旧 mtime + hipify 中间产物在 `build/temp`,ninja 按时间戳判"过时"判错。
  - 旧 .o 用上个 checkout 的参数序 → 运行时读到错误值(如 mxfp8 误报 `MXFP8 not support RHT`)。
- 5min 完成的"增量重编"没真重编;**必须 clean 重编**(`rm -rf build primus_turbo/lib/*.so`,~15-25min)。

---
来源: claim-mi355x-node/SKILL.md, remote-mlperf-gptoss/SKILL.md, flydsl-fp8-gemm-tuning/SKILL.md, 13-primus-turbo-prod.md, 14-fused-preshuffle-e2e.md
