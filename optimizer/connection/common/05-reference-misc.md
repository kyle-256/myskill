# 连接：其它零散事实

> 类别: 连接 · 主题标签: build, container, rocprofv3, isa-dump, hardware, env

## Claude Code 权限
- 设置加载顺序：user(`~/.claude/settings.json`) → project(`.claude/settings.json`) → local(`.claude/settings.local.json`)，后者覆盖前者。
- 全局免确认写 `~/.claude/settings.json` 的 `permissions.allow`（Bash/Read/Write/Edit/Skill/Glob/Grep/WebFetch/WebSearch 用 `(*)` 通配）。
- 加了全局还是问 → grep 当前 cwd 下 `.claude/settings*.json` 看有无 `deny`；`Bash(*)` 不 override deny 规则。

## 本地化 / 远端运行
- Primus-Turbo 代码在 `/workspace/code/gpt_oss_docker/sync/Primus-Turbo`，本地直接编辑/运行，无 rsync/ssh/docker。改 flydsl kernel 后必须 `rm -rf /root/.flydsl/cache` 清缓存；复用已编译则跳过。
- wgrad 4-wave kernel canonical 在 `sync/tensorwise/Primus-Turbo`（分支 `feat/kyle/grouped-wgrad-4wave`），远端 chi2811 编译测（remote-sync skill）。kernel 文件 `primus_turbo/flydsl/grouped_gemm/gemm_fp8_grouped_kernel.py`；bench harness `_ow.py`(TF+SNR sweep, untracked)，race 测 `_race_wg.py`，prof 用 `_op3.py`。
- 远端跑 FlyDSL：`ssh -o LogLevel=ERROR hjbog-srdc-39.amd.com 'docker exec hungry_dijkstra bash -c "cd /FlyDSL && python3 my_kernel.py"'`（同法跑 `tests/kernels/test_vec_add.py`、`bash scripts/run_benchmark.sh`）。FlyDSL 装 `/FlyDSL` editable(`pip install -e .`)，Python 3.12，ROCm 7.2。
- 本地跑 FlyDSL 内核：`PYTHONPATH=./ python my_kernel.py`；带 IR dump：`FLYDSL_DUMP_IR=1 PYTHONPATH=./ python my_kernel.py`。
- 长时间远端构建用 `docker exec -d` 后台跑并重定向日志：`docker exec -d <C> bash -c "cd /FlyDSL && bash scripts/build_llvm.sh -j128 > /tmp/build_llvm.log 2>&1"`，再 `docker exec <C> tail -5 /tmp/build_llvm.log` 监控（等 'Creating tarball...'）。

## wgrad harness env
- 生产 2 池 3buf 配置 env：`PT_WL_2BPOOL=1 PT_CS=1024`（commit c5710aa）。
- 1 池 3buf 生产 env：`PT_WL_3BUF=1 PT_WL_3BUF_FUSED=1 PT_WL_VMCNT_MODE=partial`。
- 筛测 shape/M：`PT_ONLY=<shape> PT_MS=2048,4096`。
- `PT_RACE_VM` 只改 2buf 版 `_wgrad_wholeloop_asm`，对 3buf 路径是 no-op。

## FlyDSL 构建
- 两步：Step1 `bash scripts/build_llvm.sh -j128`（克隆 ROCm/llvm-project，checkout `thirdparty/llvm-hash.txt` pinned commit，编译带 Python bindings 的 MLIR 装到 `../llvm-project/mlir_install/`，约 30 分钟 @128 核；`-j64` 也可）；Step2 `bash scripts/build.sh -j128`（编译 Fly dialect C++ 与 Python bindings，约 5 分钟）。
- 已有 MLIR 可跳过 build_llvm.sh：`export MLIR_PATH=/path/to/llvm-project/mlir_install`。磁盘约 50GB；要求 cmake>=3.20、C++17、ninja、Python 3.10+。
- editable 安装：`cd /FlyDSL && pip install -e .`；或只设路径 `export PYTHONPATH=/FlyDSL/build-fly/python_packages:$(pwd):$PYTHONPATH` 且 `export LD_LIBRARY_PATH=/FlyDSL/build-fly/python_packages/flydsl/_mlir/_mlir_libs:$LD_LIBRARY_PATH`。验证 `python3 -c 'import flydsl; print("FlyDSL OK")'`，`bash scripts/run_tests.sh`（GEMM 正确性约 15s）。
- 重建规则：C++ 改动(dialect/MLIR passes)重跑 `bash scripts/build.sh -j128`；纯 Python 改动无需重建（editable 自动生效）。
- 构建产物勿提交 git（已 .gitignore）：`python/flydsl/_mlir/`(平台相关 .so bindings)、`build/`、`build-fly/`。fresh clone 后须重跑构建。
- 加新 atom Op 要改：`.td` 类型声明、`.cpp` 接口实现、`CMakeLists.txt` 加源文件、`FlyROCDLExtension.cpp` 暴露给 Python、`python/flydsl/expr/rocdl/*.py` DSL wrapper、可选 `Atom.td` 扩 `AtomStateField` 枚举、`tests/mlir/Conversion/*.mlir` FileCheck。改 C++/TableGen 后 `bash scripts/build.sh` 重建。

## Primus-Turbo 构建
- editable 安装：先 `pip install -r requirements.txt`（Triton/PyTorch 版本钉死），再 `GPU_ARCHS=gfx942 pip install --no-build-isolation -e . -v`。`--no-build-isolation` 必需（让 build 看到已装 torch/triton）。`pip install .`(非 -e)拷进 site-packages，源码改动不生效。
- `GPU_ARCHS`：gfx942=MI300X/MI325X，gfx950=MI350X/MI355X，native=自动检测，分号 `"gfx942;gfx950"` 同时编。build 自动装钉死版本 amd-aiter 和 origami(setup.py)。
- 是否 rebuild：改 Python/Triton(`primus_turbo/**.py`)不用；改 C++/HIP(`csrc/**`)或 `bindings_pytorch.cpp` op schema 必须重跑 editable install。
- arch 专用源码按后缀过滤：`*_gfx942.{cu,hip}`/`*_gfx950.{cu,hip}` 只在对应 arch 在 GPU_ARCHS 里才编（setup.py `filter_files_by_arch`）。
- 三层产物解耦：`libprimus_turbo_kernels.so`(来自 `csrc/kernels/`，所有 HIP/CK/hipBLASLt/turbo kernel，frontend 无关)、`primus_turbo.pytorch._C`(`csrc/pytorch/`，PyTorch 绑定链接上面 .so)、`primus_turbo.jax._C`(`csrc/jax/`，需 `PRIMUS_TURBO_FRAMEWORK=JAX`)。
- env：`GPU_ARCHS`、`ROCM_HOME`(默认 /opt/rocm)、`MAX_JOBS`(默认 64)、`PRIMUS_TURBO_FRAMEWORK`(PYTORCH/JAX 分号分隔，默认 PYTORCH)、`PRIMUS_TURBO_LOG_LEVEL`(默认 WARNING)。
- 验证 editable 生效：`pip show primus_turbo` 看 `Editable project location`，或 `python -c "import primus_turbo; print(primus_turbo.__file__)"` 应指向源码树而非 site-packages。

## FlyDSL 关键 env / arch 支持
- env（用 `python/flydsl/utils/env.py` 规范名，勿造别名）：`FLYDSL_COMPILE_BACKEND`(默认rocm)、`ARCH`(覆盖编译架构)、`COMPILE_ONLY`、`FLYDSL_RUNTIME_CACHE_DIR`、`FLYDSL_RUNTIME_ENABLE_CACHE`、`FLYDSL_RUNTIME_RUN_ONLY`(1=只加载磁盘缓存跳过 JIT，缓存 miss 报错，与 `FLYDSL_DUMP_IR` 不兼容)、`FLYDSL_DUMP_IR`/`DUMP_DIR`、`HSA_OVERRIDE_GFX_VERSION`、`FLYDSL_DEBUG_PRINT_AFTER_ALL`、`FLYDSL_DEBUG_AST_DIFF`。
- arch 支持：gfx942(MI300X/308X, wave64, MFMA, CDNA3)；gfx950/95*(MI350/355X, wave64, MFMA, CDNA4, FP4/MFMA scale/160KB LDS)；gfx11*(RDNA3, wave32, WMMA, 无 MFMA, 无原生 FP8 fail-fast, v16 WMMA ABI)；gfx120*(RDNA4, wave32, WMMA, 原生 FP8, v8 ABI)；gfx1250(wave32 但 `is_rdna_arch` 返 False 且 `get_warp_size` 返 64，kernel 自己硬编 `WAVE_SIZE=32`，320KB LDS，FP8/FP4/TDM)。
- `scripts/run_tests.sh` 在 `HIP_VISIBLE_DEVICES` 未设时自动选空闲 VRAM 最多 GPU 并设 `FLYDSL_RUN_QUANT=1`。`RUN_TESTS_FULL=1` 加大 shape(CI 用)。单测 `python3 -m pytest tests/kernels/test_pa.py -v`。
- 多 GPU 测试(`test_flydsl_shmem.py`/`test_allreduce.py`)用 `pytest -m multi_gpu`：shmem 回归<2 GPU 跳；allreduce 4-GPU 精度<4 跳、8-GPU<8 跳。跑在 8-GPU runner(linux-flydsl-mi325-8/mi355-8)，不在默认 run_tests.sh。paged-attention 改动从 `tests/kernels/test_pa.py` 起，参考语义 `reference_masked_attention()`/`torch_mha_extend()`。

## rocprofv3 / trace decoder
- 抓 ATT trace（远端 SSH+docker）：`ssh $USER@$HOST "docker exec -e PYTHONPATH=<flydsl>/python:<flydsl>/tests -e FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 $CONTAINER bash -c '<CMD>'"`；本地 `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 PYTHONPATH=./ <CMD>`。
- 部署测试脚本到远端容器：`scp $TEST_SCRIPT $USER@$HOST:/tmp/` 然后 `ssh ... "docker cp /tmp/$TEST_SCRIPT $CONTAINER:/tmp/"`。下载 trace：`ssh ... "docker cp $CONTAINER:$UI_OUTPUT_DIR /tmp/ui_trace_download"` 再 `scp -r $USER@$HOST:/tmp/ui_trace_download/* $LOCAL_DIR/`。
- rocprof-trace-decoder 缺失安装：`wget github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.6/...manylinux-2.28-0.1.6-Linux.sh`，chmod+x 后 `./...sh --skip-license --prefix=/tmp/rtd-install`，把 `*.so*` 拷到 `/opt/rocm/lib/` 再 `ldconfig`。版本须匹配镜像 ROCm 版本（从 `/opt/rocm/.info/version` 用 `sed -E 's/^([0-9]+)\.([0-9]+).*/\1.\2/'` 取主次版本；已知 RTD 0.1.5/0.1.6 可用；安装器名 `rocprof-trace-decoder-manylinux-2.28-${RTD_VERSION}-Linux.sh`）。验证已装：`ls /opt/rocm/lib/librocprof*decoder*`，缺它 rocprofv3 trace 解码会失败。

## 标准 ROCm 开发镜像
- 基于 `rocm/vllm-dev:nightly`（含 rocprofv3 ROCm7.0、PyTorch 2.9），定制：aiter 换 github ROCm/aiter main、FlyDSL 装 ROCm/FlyDSL main、装匹配 ROCm 版本的 rocprof-trace-decoder。
- 已验证环境：容器 `hungry_dijkstra`，镜像 `rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.8.0`，host `hjbog-srdc-39.amd.com`。

## Correctness / performance 跑法（Primus-Turbo）
- correctness：`pytest tests/pytorch/ -n 8`（单 GPU 套件，每 xdist worker 由 conftest.py 钉一个 GPU）；`-k "blockwise and TRITON"` 过滤 op+backend；`--deterministic-only` 跑 bitwise determinism；`--dist-only` 跑多 GPU。conftest.py 定义 deterministic/multigpu marker（普通 run 被 skip）。
- 性能 batch suite：`python benchmark/ops/training/run_suite.py -d output/`（全部 tasks，benchmark_suite.yaml）、`-g gemm_fp8`(一组)、`-n 4`(4 GPU)；`summarize_results.py` 聚合。模块/模型级真实训练 step 用 `benchmark/pretrain/pytorch/`(入口 `pretrain_main.py`，模型 `models/turbo_llama.py`)确认 microbench 收益能迁移到训练。

## ISA dump / 反汇编
- 从 .so 提取 gfx950 code object：`roc-obj-ls` 拿 offset/size → `dd` 切出 `.hsaco` → `/opt/rocm/lib/llvm/bin/llvm-objdump -d --triple=amdgcn-amd-amdhsa --mcpu=gfx950`；寄存器元数据用 `llvm-readelf --notes` 看 `sgpr_count`/`private_segment_fixed_size`/`vgpr_spill_count`。
- 从 results.db 查寄存器信息：`SELECT ks.KernelName,ki.arch_vgpr_count,ki.accum_vgpr_count,ki.lds_size FROM rocpd_kernel_dispatch kd JOIN rocpd_info_kernel_symbol ks ON kd.kernel_symbol_id=ks.id JOIN rocpd_info_kernel ki ON kd.kernel_id=ki.id`。

## MFMA 延迟（cycles = 流水深度）
| 指令 | 延迟 |
|---|---|
| 16x16x16 f16/bf16 | 16 |
| 32x32x8 f16/bf16 | 32 |
| 16x16x32 fp8 | 16 |
| 32x32x16 fp8 | 32 |
| 16x16x128 f8f6f4 (CDNA4) | 16 或 32（A或B为FP8则32）|
| 32x32x64 f8f6f4 (CDNA4) | 32 或 64 |
| 16x16x4 f32 | 32 |
| 32x32x2 f32 | 64 |
| f64 16x16x4 | 64 |

## CDNA4 MFMA 依赖 NOP（Table38）
- XDL写→同 XDL 读 SrcC（累加，完全相同 vDst）= 0-2（有 forwarding）。
- XDL写→VALU/VM/LDS/FLAT/MFMA 读结果（RAW，无 forwarding）= 5/8/12/20 waits（按变体：16x16 4block=5、16x16x16=8、32x32x8/16x16x4 f32=12、32x32x4/32x32x2 f32=20）。
- 非DLops VALU写→MFMA读 = 2；VALU写SGPR→VMEM读SGPR = 5（HW不检查，须自加 wait）；V_CMPX写EXEC→MFMA = 4。

## LDS 硬件 gfx942(CDNA3) vs gfx950(CDNA4)
| 项 | gfx942 | gfx950 |
|---|---|---|
| LDS/CU | 64KB | 160KB (2.5×) |
| banks | 32 × 4字节宽 | 64 × 4字节宽 (640 DWord/bank) |
| bank 索引 | `(addr/4)%32` | `(addr/4)%64` |
| 峰值带宽 | 128 bytes/cycle | 256 bytes/cycle (2×) |
| 延迟 | ~20-40 cycle | 2-64 cycle（取决于冲突）|
| alloc 粒度 | 256 字节 | 1280 字节（按 1280 对齐、不环绕）|
- gfx950 补充：64 线程 wavefront 读分 4 cycle waterfall 派发；硬件并发 32 个 32-bit 读或写（read2/write2 各 64-bit）；32 个整数原子单元。
- bank 冲突 = 同 wave 2+ 线程访不同地址同 bank（串行化）；同地址 = broadcast 无冲突。

## occupancy 模型跨代
- CDNA3→CDNA4 关键差异：LDS 64KB→160KB、LDS alloc block 256B→1280B。VGPR pooling 模型没变（都 512 合并）。VMCNT 6bit(max 63 in-flight)、LGKMCNT 4bit(max 15) 两代相同。

## build target 硬件参数
| | gfx942 (CDNA3, MI300X/MI325X) | gfx950 (CDNA4, MI350X/MI355X) |
|---|---|---|
| offload-arch | `--offload-arch=gfx942` | `--offload-arch=gfx950` |
| active CUs | 304 | 256 |
| LDS/CU | 64 KiB | 160 KiB |
| FP8 格式 | FNUZ | OCP |
- MI355X 液冷 1400W ~2.4 GHz；MI350X 风冷 1000W ~2.2 GHz；两者均 288 GB HBM3E 8 TB/s。

## 回归根因常见模式速查
- 循环边界翻倍=更多工作；加 `s_barrier`=额外 sync stall；tile 变小(BLOCK 64→32)=占用率差/更多迭代；循环内加 load=更多流量；fp16→fp32=2x 带宽和寄存器；删 prefetch/double-buffer=暴露 load 延迟；`waves_per_eu` 2→1=降占用；`num_stages` 1→2=Triton pipeline 变化。

## Campaign / profiling skill 布局
- kernel-optimize campaign dir = `agent/workspace/<op>_<backend>_<gpu>_<date>/`，含 `logs/`(optimize.md + performance_trend.md, append-only 永不删/截断/重写，修正靠追加)、`profiles/`、`rounds/round-N/`(summary.md + kernel_snapshot/ + artifacts/)、`manifest.yaml`。campaign dir 默认不 git-tracked，仅 kernel 代码进 git lineage。当前时间用 shell `date '+%Y-%m-%d %H:%M'` 取，不用 system-prompt date。
- rocprofv3/rocprof-compute mechanics 在 `primus-turbo-develop/run_profile/tool-rocprof/SKILL.md`；项目结构/build/test/benchmark/integration 和 quick-validation 模板在 `primus-turbo-develop/SKILL.md`；历史 tips 在 `agent/historical_experience/<target_gpu>/<target_op>/<target_backend_lower>/tips.md`（backend 目录小写，如 TRITON→triton）。
- poll cap：benchmark 因 timeout 转后台时（此 kernel family 常 30-90 min），用重复 sleep<=900s(15 min) 窗口，绝不一次 multi-hour sleep。窗口间重查 terminal 文件/artifact（grep 最新 `TestID:`、`wc -l` CSV、查 exit_code）。连续两次无进展当 hung 处理（OOM/driver errors），别盲 sleep。

---
来源: global-permissions/SKILL.md, flydsl-fp8-gemm-tuning/SKILL.md, 10-grouped-wgrad-4wave-3buf.md, gfx950-vmcnt-race-debug/SKILL.md, lds-optimization/SKILL.md, flydsl-kernel-authoring/SKILL.md, flydsl-tile-programming/SKILL.md, add-target-atom-op/SKILL.md, capture-kernel-trace/SKILL.md, kernel-trace-analysis/SKILL.md, bisect-perf-regression/SKILL.md, FlyDSL/CLAUDE.md, build-flydsl/SKILL.md, build-rocm-image/SKILL.md, gfx950/overview.md, verify-accuracy/SKILL.md, verify-performance/SKILL.md, optimize-loop.md, SKILL.md
