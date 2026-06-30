# FlyDSL fp8 GEMM 核心架构

## 三个 kernel 对应的 GEMM 方向

| op | 矩阵乘 | 输入形状 | kernel |
|---|---|---|---|
| **fwd (NT)** | `a[M,K] @ b[G,N,K]^T → [M,N]` | A 行主序，B 转置 | `kernel_grouped_nt_persistent` |
| **dgrad (NN)** | `a[M,K] @ b[G,K,N] → [M,N]` | A/B 都行主序 | `kernel_grouped_nn_persistent` |
| **wgrad (TN)** | `a[M,OUT_M]^T @ b[M,OUT_N] → [G,OUT_M,OUT_N]` | variable-K（contraction=M_total，每组不同） | `kernel_grouped_tn_masked/persist` |

`M` = M_total = 所有 group 的 token 总数；fwd/dgrad 的 K 是固定的（模型权重维度）；wgrad 的 contraction 是 per-group 的 M_g（每组 token 数，variable）。

## 8-wave WG 布局

每个 WG = 8 warp（512 线程），2M × 4N 的 warp tile 布局：

```
wave_m = wave_id // 4   ∈ {0, 1}   ← 2 M 方向 warp
wave_n = wave_id % 4    ∈ {0,1,2,3} ← 4 N 方向 warp
```

每个 warp 负责 64×128 的 output 子区域（4 个 16×16×128 MFMA）。
每个 WG 的 tile 大小 = BLOCK_M × BLOCK_N（默认 256×256 for fwd/dgrad，256×256 for wgrad output）。

gfx950 的 8-wave WG 有 V+A ≤ 256 dword/lane 的上限（2 waves/SIMD，每 SIMD 16384 dword）——这是所有 VGPR/AGPR 设计的硬约束。

## LDS 组织（wgrad）

wgrad LDS = 4-buffer 双缓冲（cur/next 各两路，对应 A/B 的 LDS_BLOCK_M / LDS_BLOCK_N 两半）：

```
A_lds_cur_0, A_lds_cur_1   ← wave_m=0/1 的 A LDS 半区（cur）
A_lds_next_0, A_lds_next_1 ← wave_m=0/1 的 A LDS 半区（next）
B_lds_cur_0, B_lds_cur_1, B_lds_next_0, B_lds_next_1 ← 类似
C_lds_shuffle               ← CShuffle store 暂存（EPL=8 bf16/lane = 128b）
```

MFMAf32_16×16×128_f8f6f4 的操作数：A 和 B 都是 fp8 从 LDS 用 `ds_read_b64_tr_b8`（transpose-load）读出，4 个 MFMA 构成一个 warp tile。

## G2S / S2R / MFMA 流水

核心 pipeline（fwd/dgrad）：
1. **G2S** (`buffer_load_to_lds`)：global → LDS，异步，vmcnt 控制
2. **S2R** (`ds_read_b64_tr_b8`)：LDS → register，lgkmcnt 控制，inline-asm 绕过 backend auto-vmcnt(0)
3. **MFMA** (`v_mfma_f32_16x16x128_f8f6f4`)：register → accumulator

wgrad 的 S2R 走相同的 `ds_read_b64_tr_b8`（因为 TN = 两侧都需要 transpose）。

## 持久 vs 非持久

- **非持久**（nt8w / nn8w）：grid = G × TILES_PER_GROUP，每个 WG 做一个 output tile 然后退出（s_endpgm）。无 scf.for 外层 tile 循环，调度惩罚低。**fwd/dgrad 默认路径**。
- **持久**（scf.for）：grid = min(G×TILES，num_cus)，每个 WG 用 scf.for 跑多个 tile。**wgrad 小-M / comm-overlap 路径**。

关键教训：fwd/dgrad 非持久优于持久（省去 scf.for 调度惩罚 ~11%）；wgrad 小-M（per-group contraction ≤ 1536）持久优于 masked（省去 over-run chunk 的废循环）。

## 文件位置

```
Primus-Turbo/primus_turbo/flydsl/grouped_gemm/gemm_fp8_grouped_kernel.py  ← 所有 kernel
Primus-Turbo/primus_turbo/flydsl/utils/fp8_gemm_helper.py                ← StoreCPerTensor / i64 rebase
Primus-Turbo/primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py ← dispatch
```
