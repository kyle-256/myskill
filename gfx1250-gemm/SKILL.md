---
name: gfx1250-gemm
description: gfx1250 (MI450) 上用 FlyDSL 写/调试/优化 GEMM kernel 的核心知识：MI450 架构新特性(TDM、WMMA、cluster 多播、MX 微缩放)、TDM 异步搬运、同步原语、descriptor K-loop 更新、常见死锁、FlyDSL 源码构建踩坑、性能/限频定标。当用户要写、调试、优化 gfx1250 GEMM kernel 或从源码构建 FlyDSL 时使用。代码在 /workspace/code/conductor_455/sync/FlyDSL/kernels/gemm_fp8fp4_gfx1250.py 和 gemm_common_gfx1250.py。
---

# gfx1250 GEMM —— FlyDSL 开发手册

代码入口：
- `kernels/gemm_common_gfx1250.py` —— LDS helper、pipeline fence、accumulator store
- `kernels/gemm_fp8fp4_gfx1250.py` —— 完整 FP4/FP8/A8W4 GEMM kernel
- `python/flydsl/expr/rocdl/tdm_ops.py` —— TDM descriptor API

---

## 0. MI450 (gfx1250) 架构新特性总览

gfx1250 (MI450) 是 AMD 把 **RDNA 血统的 WMMA + CDNA 的 HPC 定位**合到一起的新架构，相比上一代 CDNA (gfx94x，MI300/MI325) 有一批全新硬件特性，写/调 GEMM 必须知道：

### 0.1 TDM —— Tensor Data Mover（最大的新特性）
- 专用**硬件异步 DMA 引擎**，一条指令把 Global memory 的整块 2D tile 异步搬进 LDS，**取代**老架构的 `buffer_load + ds_write` 手工搬运 + `s_waitcnt vmcnt`。
- 通过 **descriptor**（2 个向量寄存器组）描述源/目的/形状/stride/tile，硬件自动按 wave 切分 sub-tile。
- 新增独立计数器 **`s_wait_tensorcnt`**（不是 vmcnt/lgkmcnt）—— TDM 的完成等待走这个。
- 支持 **gather 模式**（行不连续，MoE 用）和 **TDM store**（`tensor.store.from.lds`，epilogue 异步写回）。
- 详见第 2 节。**这是 gfx1250 GEMM 的核心，老架构完全没有。**

### 0.2 WMMA 取代 MFMA（RDNA 血统 + Wave32）
- 矩阵指令是 **WMMA**（RDNA 系），**不是** CDNA 的 MFMA。
- **Wave size = 32**（不是 CDNA 的 64）—— 影响所有 lane 级代码、LDS 布局、scale 分配。`get_warp_size()` 会错误返回 64，必须 hardcode 32。
- WMMA tile 形状：**bf16/fp16 = 16×16×32**；**fp8/fp4 = 16×16×128**（低精度 K 维更深）。

### 0.3 原生低精度 + 硬件 Microscaling (MX)
- 硬件原生支持 **fp8 (E4M3/E5M2)、fp4 (E2M1)**，以及混合 **a8w4**（fp8 激活 × fp4 权重）。
- **MX (microscaling) 在硬件内做**：`mxscale` 模式下 **E8M0 block scale（每 32 元素一个）直接在 MMA 内部应用**，不用 epilogue 反量化。另有 `ptpc`（per-token/per-channel，fp32 epilogue 缩放）。
- 这是对标 NVIDIA MXFP 的硬件支持，老 CDNA 没有。

### 0.4 Workgroup Cluster + Multicast（对标 Hopper thread-block cluster）
- 多个 workgroup 可组成 **cluster**（`cluster_m × cluster_n`），cluster 内 WG 可**多播共享 A/B tile**，大幅减少 HBM/L2 重复读取 → 这是大 GEMM 冲高带宽效率的**主力杠杆**。
- 配套 **cluster barrier**（`workgroup_barrier(use_cluster=True)`）。
- ⚠️ **依赖 HIP 运行时支持 cluster 启动**（`hipLaunchAttributeClusterDimension`）。若 HIP 版本不支持，会**回退成普通启动**，而 cluster kernel 的 cluster-barrier 在普通启动下会**在 GPU 上死锁**。用前务必确认 HIP 支持，否则 `cluster_m=cluster_n=1`。

### 0.5 更大 LDS + 新调度/预取
- **LDS = 320KB**（5 × 64KB segment），gfx950 只有 160KB → 能塞更大 tile / 更多 buffer。
- **Expert scheduling mode**（`amdgpu-expert-scheduling-mode` LLVM option）：kernel 默认开（`expert_sched_mode=True`）。
- **指令预取** `s_set_inst_prefetch_distance`（`amdgpu-inst-prefetch-distance`）—— 注意**某些 LLVM 版本不认这个 option**，开 `inst_prefetch=True` 可能报 "Unknown LLVM option"。

### 0.6 性能定标（务必先确认时钟，再下"kernel 慢"的结论）
- bf16 满频理论 ~5 PFLOP/s（按量产卡 ~2100MHz boost 估）。
- **GEMM 性能直接正比于 sclk**。下结论前先 `rocm-smi -s` 看 **Supported sclk 档位** + `--showclocks` 看**满载时**实际 sclk。
- ⚠️ **工程样片（Eng Sample，Card Model 0x75c1）sclk 上限只有 ~1100MHz**（DPM 只有 500/1100 两档，满载 ~1034MHz），约量产 boost 的一半 → 峰值直接腰斩到 ~2.6 PFLOP/s。**实测 bf16 ~1200 TFLOP/s ≈ 限频后峰值的 46%，是这台样片的硬件天花板，不是 kernel 差**。这个限频是**硅本身**的，与节点预约/重启无关（节点稳定 11h 后频率上限依旧）。
- 调优杠杆优先级：① 大 tile（256×256 通常 > 128×x，受 LDS 限制最多 2 buffer）② cluster 多播（需 HIP 支持）③ num_buffers（受 320KB LDS 约束）。小 tile、warp 4×2、tile_k=64 实测都更慢；`wave_specialized_tdm` 快 ~5% 但会出 NaN（当前有 bug）。

---

## 1. 硬件基础（必须记住的坑）

| 项 | 值 | 坑 |
|---|---|---|
| Wave 大小 | **32** | `is_rdna_arch('gfx1250')` 返回 **False**；`get_warp_size()` 返回 **64**（错）→ 所有 gfx1250 kernel 必须 hardcode `WAVE_SIZE = 32` |
| MMA 指令 | **WMMA**（不是 MFMA） | tile = 16×16×128，`SCALE_BLOCK = 32`，`SCALES_PER_WMMA = 4` |
| LDS | **320KB** (5 × 64KB segments) | gfx950 = 160KB |
| 外网 | **有** | pip install / git clone 可用，与旧 gfx950 环境（无外网）不同 |

---

## 2. TDM（Tensor Data Mover）——最核心的新概念

gfx950 用手工 `buffer_load` + `ds_write` 搬数据；gfx1250 有专用硬件 DMA 引擎，一条指令异步把 Global 整块 tile 搬进 LDS。

### 2.1 基本流程

```python
from flydsl.expr import tdm_ops

# 1. 构建 descriptor（编译时确定大部分字段）
desc_a = tdm_ops.make_tensor_descriptor_2d(
    global_ptr=arg_a,
    lds_memref=lds_a,                    # SmemPtr 或 memref
    global_offset=(blk_m, k_base),       # (outer_off, inner_off)，MLIR index
    tensor_shape=(M, K),                 # 整个张量形状（outer, inner）
    strides=(K, 1),                      # (outer_stride, inner_stride)
    tile_shape=(tile_m, tile_k),         # 本次搬运的 tile 大小
    elem_bytes=2,                        # f16/bf16=2, f32=4, fp8=1
    pad_interval=64, pad_amount=8,       # LDS padding（元素数），防 bank conflict
    num_warps=8,                         # workgroup 里 wave 数，自动分配 per-wave sub-tile
)

# 2. 异步发出（不等完成）
tdm_ops.tensor_load_2d(desc_a)

# 3. 做别的（或发下一块 descriptor）
# ... WMMA 计算 / 其他 descriptor ...

# 4. 等 TDM 完成
tdm_ops.tensor_wait(0)   # s_wait_tensorcnt(0)：等所有 in-flight TDM
# tdm_ops.tensor_wait(1)  # 允许 1 个还在飞（double-buffer 场景）
```

### 2.2 Descriptor 内部结构

`TDMDescriptor2D` 由两个向量组成：

| 向量 | 大小 | 内容 |
|---|---|---|
| `dgroup0` | vec<4×i32> | `[pred, lds_addr, global_addr_lo, global_addr_hi]` |
| `dgroup1` | vec<8×i32> | config bitfields + tensor_dim0/1 + tile_dim0/1 + outer_stride + 0 + 0 |

`global_addr_hi[31:30]` = type field（固定为 `0b10`），不能随意覆盖。

### 2.3 Gather 模式（MoE，行不连续）

```python
desc = tdm_ops.make_tensor_gather_descriptor(
    global_ptr=arg_b,
    lds_memref=lds_b,
    row_indices=[idx0, idx1, ...],  # MLIR i32 值列表，最多 8 个（32-bit 模式）
    row_width=tile_k,
    tensor_dim0=K, tensor_dim1=num_tokens,
    stride=K, elem_bytes=2,
)
tdm_ops.tensor_load_gather(desc)
```

---

## 3. 同步原语

### 3.1 简单 fence（单 buffer 或 pipeline 末尾）

```python
from kernels.gemm_common_gfx1250 import pipeline_fence

pipeline_fence(outstanding=0)
# 展开为：s_wait_tensorcnt(0) + gpu.barrier()
```

### 3.2 Split barrier（double-buffer，overlap compute 与 sync）

```python
from kernels.gemm_common_gfx1250 import pipeline_fence_signal, pipeline_fence_wait

# --- TDM 完成后立刻 signal ---
pipeline_fence_signal()
# 展开：s_wait_tensorcnt(N) + s_barrier_signal -1

# --- 在 signal 和 wait 之间可以插 WMMA 计算 ---
do_mma(...)

# --- 读 LDS 前 wait ---
pipeline_fence_wait()
# 展开：s_barrier_wait -1
```

### 3.3 Cluster barrier（多 WG 间同步，advanced）

```python
from kernels.gemm_common_gfx1250 import workgroup_barrier
workgroup_barrier(use_cluster=True)  # cluster_barrier + WG barrier
```

---

## 4. K-loop 里更新 Descriptor（关键！）

K-loop 里 descriptor 只有全局地址在变，其余字段不变 → 只 patch `dgroup0`。

### 4.1 ❌ 危险写法（carry-unsafe）

```python
# 只更新 addr_lo，跨 4GB 边界时 addr_hi 过时 → GPU 死锁
new_desc = tdm_ops.update_tensor_descriptor_2d_addr_lo(base_desc, new_addr_lo)
```

死锁症状：host 卡在 `amdgpu_mes_reg_write_reg_wait`，必须重启机器。
触发条件：大 MoE expert weight buffer（>~3.5GB fp4），per-CTA base + K-tile 累积 delta 超过 4GB。

### 4.2 ✅ 安全写法（carry-safe，推荐）

```python
from flydsl.expr import tdm_ops

# 在 loop 外提取 base 地址（SGPR 缓存）
from flydsl._mlir.dialects import vector as _vd
base_addr_lo = _vd.ExtractOp(base_desc.dgroup0, static_position=[2], dynamic_position=[]).result
base_addr_hi = _vd.ExtractOp(base_desc.dgroup0, static_position=[3], dynamic_position=[]).result

# loop 内：
delta = arith.constant(k_step * elem_bytes, type=T.i32)
new_desc = tdm_ops.update_tensor_descriptor_2d_addr64(
    base_desc, base_addr_lo, base_addr_hi, delta
)
```

### 4.3 Pattern：K-loop descriptor hoist

```python
# loop 外：build base（global_offset inner = 0）
base_desc_a = tdm_ops.make_tensor_descriptor_2d(..., global_offset=(blk_m, 0), ...)
base_lo_a = extract_lane2(base_desc_a.dgroup0)
base_hi_a = extract_lane2(base_desc_a.dgroup0)  # lane 3

for k in range(0, K, tile_k):
    delta = arith.constant(k * elem_bytes, type=T.i32)
    desc_a = tdm_ops.update_tensor_descriptor_2d_addr64(base_desc_a, base_lo_a, base_hi_a, delta)
    tdm_ops.tensor_load_2d(desc_a)
    ...
    tdm_ops.tensor_wait(0)
```

---

## 5. LDS Helper 速查（gemm_common_gfx1250.py）

```python
from kernels.gemm_common_gfx1250 import (
    lds_load_b128,          # 从 memref 加载 128 bit（ds_load_b128）
    lds_store_b128,         # 向 memref 存 128 bit（ds_store_b128）
    lds_load_b128_raw,      # 用预提取的 base_idx + byte_offset 加载（raw LLVM）
    lds_load_b32_raw,       # 4 字节版，alignment 要求低（scale 布局用）
    lds_transpose_load_raw, # ds_load_tr16_b128（转置加载）
    extract_lds_base_idx,   # 从 SmemPtr 提取 LDS byte base 地址
    store_acc_vec8_to_lds,  # 把 vec<8×f32> accumulator 写回 LDS（f16/bf16/f32 output）
    store_acc_vec8_to_buffer, # 把 accumulator 写到 global memory
)
```

---

## 6. compile_fp8fp4_gemm 参数表

```python
from kernels.gemm_fp8fp4_gfx1250 import compile_fp8fp4_gemm

fn = compile_fp8fp4_gemm(
    data_format="fp4"|"fp8"|"a8w4",   # 数据格式
    scale_mode="mxscale"|"ptpc",       # mxscale=E8M0 in-MMA; ptpc=fp32 epilogue
    N=..., K=...,                      # N/K 编译时固定（M 运行时）
    tile_m=128, tile_n=128, tile_k=128,
    m_warp=2, n_warp=2,                # warp 在 M/N 方向的分配
    num_buffers=2,                     # 流水 buffer 数（double=2）
    cluster_m=1, cluster_n=1,          # cluster launch（通常从 1 开始调）
    out_dtype="f32"|"f16"|"bf16",
    ascale_load_path="vgpr"|"shuffled_tdm",  # A scale 加载路径
    split_k=1,                         # split-K（K 维度并行 reduction）
)
# 调用：fn(arg_c, arg_a, arg_b, arg_a_scale, arg_b_scale, M, N, lda, ldc, stream)
```

**数据布局：**
- A：`[M, K_packed]` uint8（fp4: K_packed=K//2，fp8: K_packed=K）
- B：`[N, K_packed]` uint8，**preshuffled**（16×16 byte tiles，不能直接用原始矩阵）
- mxscale_A（vgpr 路径）：`[M, K//32]` uint8 E8M0
- mxscale_B：`[N//32, (K//128)*128]` uint8 E8M0，32×4 packed 布局

---

## 6.5 从源码构建 FlyDSL（gfx1250，踩坑全记录）

当 pip wheel `flydsl` 与系统 ROCm 的 lld 不匹配、或要调 MLIR 内部时，从源码构建。FlyDSL pin 的 LLVM commit 在 `thirdparty/llvm-hash.txt`。

**前置（容器内 pip 装）：** `pip install cmake ninja patchelf`（`build_llvm.sh` 会自己装 nanobind/numpy/pybind11）。

**步骤：**
```bash
# 1. 子模块（rsync 排除 .git 时远程没有，需先在本地 git submodule update --init --recursive 再同步）
#    thirdparty/dlpack, thirdparty/tvm-ffi 必须有内容

# 2. 编 LLVM/MLIR/lld（~6900 目标，含 mlir;clang;lld + MLIR python bindings）
bash scripts/build_llvm.sh -j32      # ⚠️ 别用 -j128：链接阶段会 vfork: Resource temporarily unavailable
# 产物：../llvm-project/mlir_install （ninja 增量，中断可重跑续编）

# 3. 编 FlyDSL（关键 env，少一个就报错）
MLIR_PATH=<repo>/../llvm-project/mlir_install \
HIP_PLATFORM=amd \
hip_DIR=<SDK>/lib/cmake/hip \
CMAKE_PREFIX_PATH=<SDK>/lib/rocm_sysdeps \
  bash scripts/build.sh -j32
# 产物：build-fly/python_packages  → PYTHONPATH 指它
```
其中 `<SDK>` = therock 的 `_rocm_sdk_devel`（`echo $ROCM_PATH`）。

**构建坑（都踩过）：**
1. **`-j128` 链接 vfork 失败** → 全程 `-j32`（编译阶段也行，省得 babysit）。
2. **`hip` 找不到** → 必须 `hip_DIR=<SDK>/lib/cmake/hip`（build.sh 不传 CMAKE_PREFIX_PATH 给 hip）。
3. **`zstd` 找不到 / MLIR_FOUND=FALSE 缺 NVPTX 目标** → 根因是 `CMAKE_PREFIX_PATH` 里放了 `<SDK>` 根目录，导致 `find_package(LLVM)` 命中 **SDK 自带的 LLVM**（无 NVPTX）而非我们编的。**只放 `<SDK>/lib/rocm_sysdeps`**（提供 zstd/zlib，无 llvm cmake），LLVM/MLIR 全走 `MLIR_PATH`。
4. **`patchelf: command not found`** → 最后一步 copy sources 要 patchelf，pip 装。
5. **编译 kernel 极慢（大 kernel 10+ 分钟）** → `build_llvm.sh` 默认 `LLVM_ENABLE_ASSERTIONS=ON`。改 OFF 后 kernel 编译提速 ~10×（16min → ~90s）。做法：`cd build-flydsl && cmake -DLLVM_ENABLE_ASSERTIONS=OFF . && cmake --build . -j32 && cmake --install . --prefix ../mlir_install`，**然后 FlyDSL 必须重编**（assertions 改 ABI）。日常开发强烈建议 OFF。

**MLIR gpu-to-binary 的 lld 调用问题（重要）：**
MLIR 把 `${ROCM_PATH}/llvm/bin/ld.lld` 当 lld 来 exec。两个坑：
- therock SDK 的 `ld.lld` 动态链接 `libLLVM.so`，MLIR 直接 exec（不传 LD_LIBRARY_PATH）会找不到库 → "lld invocation failed"。
- MLIR 不排空 lld 的 stderr 管道，lld 写多了会**管道死锁**（python 0% CPU 卡死）。

**解法**：建一个 shadow toolkit（`ROCM_PATH` 指它），其 `llvm/bin/ld.lld` 换成一个 wrapper（把 lld stderr 重定向到文件 + 用绝对路径转发），`amdgcn` 软链到 SDK 的 gfx1250 bitcode。最干净是让 wrapper 转发**自己源码编的静态 lld**（`mlir_install/bin/ld.lld`，无 libLLVM.so 依赖，与自编 MLIR 同 commit）。参考 `turbo/lld_wrap.sh` + `turbo/shadow_toolkit/`。

---

## 7. 调试 Checklist

1. **改了 kernel .py** → 必须 `rm -rf /root/.flydsl/cache` 再跑，否则用旧 binary
2. **torch import 报错**（命中 `/workspace/pytorch`）→ 加 `PYTHONPATH=/opt/venv/lib/python3.12/site-packages`
3. **GPU 死锁卡住** → 99% 是 descriptor addr_lo carry overflow，改用 `update_tensor_descriptor_2d_addr64`
4. **wave_size 相关 bug** → 检查是否误用了 `get_warp_size()`（返回 64），gfx1250 必须 hardcode `WAVE_SIZE = 32`
5. **数值错误** → 检查 B 是否已 preshuffle，A scale 的布局是否匹配 `ascale_load_path`

---

## 8. 环境速览（⚠️ 易变，仅参考）

> 注：远程节点/账号/容器是易变信息，用前先核实。GEMM/构建知识本身（0–7 节）才是稳定的。
> - **反复重启（容器 Exit 137）** = 节点**无 Conductor reservation** 会被回收；拿到预约后稳定（实测 uptime 11h+）。
> - **sclk 限频 1100MHz** = 工程样片硅本身，**与预约无关**（见 §0.6），预约不会解锁满频。
> - 当前（2026-06 后期）：账号 `xianzhao@heliosp-1b114-c05-3.mnb.dcgpu`（= `10.5.229.70`），挂载 `/home/xianzhao/kyle_code → /workspace/code`，容器 `conductor_455`。源码构建产物在 `/workspace/code/llvm-project/mlir_install` 与 `FlyDSL/build-fly/python_packages`。

### 8.1 历史快照（conductor_455，2026-06-18，zhuang12 借用账号，已失效）

```
节点:      heliosp-1b114-c05-3.mnb.dcgpu
SSH:       /workspace/code/conductor_455/sync/.ssh-helio.sh zhuang12@heliosp-1b114-c05-3.mnb.dcgpu
容器:      conductor_455
GPU:       gfx1250（AMD Eng Sample，单卡）
ROCm:      7.13.0a20260501
PyTorch:   2.10.0+rocm7.13.0a20260501
Triton:    3.6.0+rocm7.13.0a20260501
代码挂载:  /home/zhuang12/kyle_tmp → 容器内 /workspace/code
本地镜像:  /workspace/code/conductor_455/sync/FlyDSL（conductor_455 分支）
调试目录:  /workspace/code/conductor_455/sync/FlyDSL/turbo/
```

执行远程命令：
```bash
SSH=/workspace/code/conductor_455/sync/.ssh-helio.sh
$SSH zhuang12@heliosp-1b114-c05-3.mnb.dcgpu \
  'docker exec conductor_455 bash -lc "cd /workspace/code/FlyDSL && FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 turbo/probe.py"'
```

---

## 9. hipBLASLt 基准（gfx1250 库支持现状，2026-06）

在这台 gfx1250 样片上，**库对 bf16 GEMM 基本没优化**，自研 FlyDSL kernel 远超：

| 实现 | bf16 8192³ | 备注 |
|---|---|---|
| torch `a@b`（默认 dispatch） | **1.8 TFLOP/s** | 走了慢速 fallback |
| hipBLASLt 默认启发式 | **3.0 TFLOP/s** | 启发式给 gfx1250 挑到烂 kernel（比自身最优慢 120×） |
| hipBLASLt 搜全部 solution 取最优 | **371.7 TFLOP/s** | gfx1250 **总共只有 2 个** kernel（成熟架构有几百个）→ 支持是 stub |
| **FlyDSL `compile_wmma_gemm_tdm`** | **~1000 TFLOP/s** | ✅ 比 hipBLASLt 最优还快 **~2.7×** |

- `hipblaslt-bench` 在 `<SDK>/bin/hipblaslt-bench`；直测：`hipblaslt-bench -m 8192 -n 8192 -k 8192 --a_type bf16_r ... --transA N --transB T --algo_method all --requested_solution -1`（`--algo_method all` 搜全部 solution，否则默认启发式会挑到烂的）。
- 结论：gfx1250 是新硅，**别拿 torch/hipBLASLt 默认路径当基准**（会误判"kernel 慢"），要么搜全 solution，要么直接跟自研 kernel 比。

### tile 调优结论（实测 sweep）
- **256×256×128 buf2 ≈ 现成 kernel 的最优**（~990–1200T，波动来自限频）。
- 更大 tile 都撞墙：512/tile_k=256 **超 LDS 320KB**（256×256×256 buf2 要 542KB）；非 2 的幂 tile（320/384）**GPU 内存越界崩溃**（8192 不整除，尾块 bug）；buf1 被拒（最少 2）。
- `tile_k` **不是性能杠杆**：决定算力利用率的是输出 tile `tile_m×tile_n`（已被 LDS 顶死 256×256）；tile_k 只影响 K-loop 开销，且同时撑大 A/B 两块 LDS。
- 真正瓶颈是**样片 1100MHz 限频**（见 §0.6），tile 调优最多提高"限频后峰值"利用率（现 ~46%）。

---

## 10. Primus-Turbo（AMD 训练算子库，用了 FlyDSL）

- 仓库：`https://github.com/AMD-AGI/Primus-Turbo`，gfx1250 分支 **`feat/gfx1250-mi455-build`**。
- 本地：`/workspace/code/conductor_455/sync/Primus-Turbo`；远程：`/workspace/code/Primus-Turbo`（= `kyle_code/Primus-Turbo`）。
- 定位：AMD 大模型训练**高性能算子库**（GEMM / FlashAttention / GroupedGEMM / DeepEP / FP8 / comm-overlap），多后端：**CK（composable_kernel）、hipBLASLt、AITER**，以及 **FlyDSL**。

**结构：**
- `primus_turbo/{pytorch,jax,triton,flydsl}/` —— 各后端。
  - **`primus_turbo/flydsl/gemm/gemm_fp8_kernel.py`** 和 `grouped_gemm/gemm_fp8_grouped_kernel.py` —— 用 FlyDSL 写的 fp8 GEMM（`import flydsl.compiler as flyc`，primitives 从 `primus_turbo.flydsl.utils.gemm_helper` 来）→ **和本 skill 的 gfx1250 GEMM 直接同源**。
- `csrc/{include,kernels,pytorch,jax}` —— C++/HIP，文件名带 `_gfx1250.hip`/`_gfx950.hip` 按 arch 分派（`setup.py` 里 `--offload-arch=gfx1250` 控制）。DeepEP intranode 有 **gfx1250 TDM async-copy 路径**。
- `agent/skills/kernel-optimize/knowledge/backend/flydsl/`（`overview.md`、`optimization-directions.md`）—— 仓库**自带的 FlyDSL 优化知识**，调 kernel 时值得读。
- `3rdparty/`：submodule `composable_kernel`（ROCm CK，分支 pin `78ae383` **不支持 gfx1250,需换 develop,见下方构建坑**）、`hipify_torch`。

**构建（gfx1250）：**
- `setup.py` `SUPPORTED_GPU_ARCHS = ["gfx942","gfx950","gfx1250"]`。
- 需求：ROCm ≥ 7、Python ≥ 3.10、PyTorch ≥ 2.6、**AITER**（部分算子/FP8：`pip install "amd-aiter @ git+https://github.com/ROCm/aiter.git@..."`；注意：**编译 C++ 扩展本身不需要 AITER**，它是部分算子的运行时依赖）。
- 子模块必须有内容（rsync 排除 .git 时先本地 `git submodule update --init --recursive` 再同步）。**别用 `--depth 1`**：只会得到 1 个 commit，没历史、切不了 CK 版本（要补就 `git fetch --unshallow`）。
- 实测可用的构建命令（容器内，源码版/wheel 版 flydsl 都行）：
  ```bash
  cd /workspace/code/Primus-Turbo
  GPU_ARCHS=gfx1250 MAX_JOBS=32 PYTHONPATH=/opt/venv/lib/python3.12/site-packages \
    pip install -e . --no-build-isolation
  ```
  - `GPU_ARCHS=gfx1250` 选 arch（不设则按 `get_gpu_arch()` 自动检测）。
  - `MAX_JOBS=32` 控 CK 编译并行防 OOM。
  - `PYTHONPATH=site-packages` + cwd 在仓库目录（**不能在 `/workspace/pytorch`**，否则 torch 被源码树 shadow）。
  - `--no-build-isolation` 用容器现成的 torch，别让 pip 起隔离环境重装。

**⚠️ 最大的坑：CK 子模块 pin 的版本不支持 gfx1250（2026-07 状态）**
- `feat/gfx1250-mi455-build` 分支把 kernel 移植到了 mi455/gfx1250（`c9f7c46 port mxfp8 to mi455` 改了 `hipblaslt_gemm.cu`/`quantization_mxfp8.cu`），**但忘了升 CK 子模块**——它 pin 的 CK `78ae383`（3月）的 `ck_tile` 架构 static_assert 里**没有 GFX1250**，编译报 `Only one target architecture can be defined` + 缺 `CK_TILE_BUFFER_RESOURCE_3RD_DWORD` / `warp_size` 非常量。
- **修复**：把 CK 子模块切到 **CK develop**（`gfx1250` 支持在 develop 上，首次加入 commit `717f2efe`/2026-05-15；develop tip 如 `604c56bc`）。注意：CK 上游那两个名字带 `gfx1250` 的分支（`lxx/dev/fix_gfx1250_compile`、`users/andriy/ck/1464-pytorch-gfx1250`）**config.hpp 里反而没有 gfx1250**，别被名字骗。`78ae383` 也不在 develop 线上。
  ```bash
  cd 3rdparty/composable_kernel && git fetch origin develop && git checkout <develop-tip>
  # 再 rsync --delete 同步整个 composable_kernel 到远程, 重新 pip install -e .
  ```
- 换 develop CK 后编译通过，`primus_turbo.pytorch._C` 扩展正常加载。有 API 漂移风险（csrc 按 78ae383 时代 CK 写），但实测 tip 能编过。
