# 占用率与 tile 尺寸选择:512 寄存器共享文件、LDS/SGPR limit、256×256 主 tile、MFMA 数账

> 类别: 方法论 · 主题标签: occupancy, vgpr-agpr-budget, lds-limit, co-saturation, tile-size, mfma, arithmetic-intensity

## 占用率计算:512 寄存器文件 VGPR+AGPR 共享、LDS/SGPR limit、alloc 粒度

### 占用率公式

- **occ(waves/SIMD) = min(vgpr_limit, lds_limit, sgpr_limit, hw_max=8)**,取最小的那个资源决定占用。
- `vgpr_limit = 512 // (arch + accum)` — gfx950 每 SIMD **512 个寄存器,VGPR+AGPR 共用同一 512 文件**。每 wave 用 R 个总寄存器 → `512 // R = waves/SIMD`;`×4 SIMD = waves/CU`;`waves/CU ÷ (waves/wg) = wg/CU`。
- `lds_limit = (LDS_total // lds_per_wg) * waves_per_wg // 4`
- `sgpr_limit = 800 // sgpr_alloc`
- **alloc 粒度:VGPR 8 个一档**。

### Worked example (PA decode gfx942)

- arch 144 + accum 136 = **280 合并** → `512 // 280 = 1 wave/SIMD`,**VGPR-bound**(LDS 允许 5,SGPR 允许 7)。
- 要到 2 waves 需合并 **≤256**,即释放 **~24 VGPR**。

### Worked example (gfx950 MI355X LDS-bound)

- LDS 160KB/CU,某内核 144KB/wg → **1 WG/CU** → 4 waves/CU ÷ 4 SIMD = **1 wave/SIMD(occ=1)**,LDS-bound。

### 权威数据源:只信 CSV,不信 analyzer

- 判 occupancy 用**真实 CSV**(`out_kernel_trace.csv`)的 `LDS_Block_Size` / `VGPR_Count` / `Workgroup_Size`,**不信 analyzer 推断**。
- ISA 里 `v_mfma` 引用 `a[...]` 寄存器会被 analyzer 误当 agpr-form → 误报 'bound by VGPR';**只有 CSV `Accum_VGPR_Count=0` 才权威**(证明在 in-AGPR 累加,非误报)。
- 实测占用判据用 rocprofv3 的 **`MeanOccupancyPerActiveCU`**(waves/SIMD),**不用** rocprofv3 的 `VGPR_Count`(不可信)。

### co-saturation(联合饱和)

- **Joint co-saturation**:联合选 tile shape + MFMA 指令宽度 + waves/SIMD,让 **VGPR+AGPR 预算、LDS footprint、matrix-unit issue rate 同时饱和**,而不是某一资源先饿死其他资源。
- 对编译后 AMDGCN 做 **VGPR liveness pass**(找 dead VGPR 窗口、把 boundary 以上的寄存器 remap 进空洞)可在下一个占用边界抬升 waves/SIMD。
- `next_free_vgpr` 步进边界:**64 / 73 / 85 / 102 / 128 / 170 / 256**。
- ★ **这些边界要再被 CTA 波数向下对齐一次**(2026-07-29 hd64 fwd r25):占用率的分配粒度是 **workgroup**,
  一个 W-wave 的 WG 给每个 SIMD 放 `W/4` 个 wave,所以实际 waves/SIMD 只能是 `W/4` 的整数倍。
  例:8-wave CTA ⇒ 只有 **2/4/6/8** 可达,`alloc ≤102` 算出来的 `floor(512/102)=5` 会**向下取整回 4,一分不涨**;
  该形状的下一档是 **6 waves ⇒ alloc ≤85**。⇒ **定「降到 N 个寄存器」的目标之前,先用 CTA 波数把
  `floor(512/alloc)` 对齐到可达档位**,否则会花整轮去省一批买不到东西的寄存器。

### Occupancy-starved 直接信号:tile 数 < CU 数

- **tile 数 < CU 数** = occupancy-starved 的直接信号。
- 实例(dgrad B=1 grok-up M=512):fwd(N=32768)=**2793TF 满载**;dgrad(N=8192)=**923TF 欠载**。output tile 数 = `G × ceil(M/BM) × ceil(K_fwd/BN)` = **64 tile 只填 64/256 CU**。
- 修法(tile 尺寸选择/BM128 gate 细节):见本卡「GEMM tile 尺寸选择」小节。

## GEMM tile 尺寸选择:256×256 主 tile、小-M BM128、三级 tiling、MFMA 数账

### 主 tile 与 dgrad 小-M 修法
- GEMM 用 **256×256 主 tile**(BLOCK_M=256/BLOCK_N=256)。只有 dgrad **小-M occupancy 欠载**才降 BLOCK_M=128。
- dgrad 小-M 修法 = BLOCK_M=128 M-branch:M-tile 数翻倍(M=512:64→128 tile)填满 CU。gate 通过时 **bm128 永远赢(+5.3~30.6%)**,单 config 直接 return,不需 autotune。
- 约束:BLOCK_M >= 128 且 %128==0(kernel 有 assert)。

### 三级 tiling 与派生量
- 三级 tiling:**block_x→M tiles、block_y→N tiles、K 为 reduction**。
- 线程 256 = 4 wave×64 lane:`wave_id=tid//64` 划 N;`lane_div_16=lane//16` 划 M(4 组×16);`lane_mod_16` 划 MFMA 内 N。
- 派生量:
  - `m_repeat = tile_m//16`
  - `n_per_wave = tile_n//4`
  - `num_acc_n = n_per_wave//16`
  - `k_unroll = tile_k_bytes // a_elem_vec_pack // 64`

### 推荐 tile 配置(tile_m / tile_n / tile_k)
| 场景 | 配置 | WHY |
|---|---|---|
| 小 batch M≤32 | 16 / 64-128 / 256-512 | memory-bound,大 tile_k 换复用 |
| 中 batch | 64 / 256 / 128 | 均衡 |
| 大 batch M≥4096 | 128 / 256 / 128 | compute-dense,需 async copy |
| FP4 (gfx950) | 32-64 / 128-256 / 256 | MFMA_SCALE |

### 每 tile MFMA 数账
- **每 tile MFMA 数 = k_unroll × m_repeat × num_acc_n × 2**(末尾 2 = 每 K64 微步做 2 次 K32 MFMA)。
- 例:tile 64×256×128 FP8 → 2×4×4×2 = **64 MFMA**;tile 64×256×512 → k_unroll=8 → **256 MFMA**。
- MFMA 内循环 K64 微步(FP8/INT8)三重循环:
  - 外 `ku`(k_unroll 个 K64 步):取 b_packs0/b_packs1
  - 中 `mi`(m_repeat 个 16 行块):`lds_load_packs_k64` 取 A(2×i64)
  - 内 `ni`(num_acc_n 个 16 列累加器):`mfma_k64_bytes`;每 K64 做 2 次 K32 MFMA

### MFMA 指令选择表(累加器均 f32×4)
| 数据类型 | 指令 | K |
|---|---|---|
| FP8 | `mfma_f32_16x16x32_fp8_fp8` | 32 |
| INT8 | `mfma_i32_16x16x32i8` | 32 |
| BF16 | `mfma_f32_16x16x16bf16_1k` | 16 |
| FP16 | `mfma_f32_16x16x16f16` | 16 |
| FP4 (gfx950) | `mfma_scale_f32_16x16x128_f8f6f4` | 128 |

### Roofline 判据
- M≤512 通常 memory-bound(关注带宽);M>512 compute-bound(关注 MFMA 利用率)。
- 算术强度 = flops / bytes_moved,与 roofline crossover 比较。

### 方形大 tile 提算术强度(4-wave 长 K 领先 8-wave 核心机制)
- 强度定义 = 每 K-step 的 MAC / (A行 + B列 operand-elem)。
- 具体数据(算术强度对比、LDS insts 实测比值、随 K 的领先幅度):见 methodology/06-register-wholeloop-emit.md（算术强度小节）

---
来源: 08-att-root-cause.md, kernel-trace-analysis/SKILL.md, gfx950/kernel-implementation-notes.md, agpr_phase5_lds.md, diag_4w_vs_8w.md, 03-nn-dgrad-kernel.md, gemm-optimization/SKILL.md

## ★★ 想写「半深 K-tail body」省掉 K-pad 的 MAC 之前:先算 `ceil(K_real/inst_k)` 对 `ceil(K_pad/inst_k)`

2026-08-17 gpt-oss-20b down-projection(fp8 grouped,`K_real=2880`,`K_pad=2944`)实测教训:
计划里挂了两轮的「NN/NT 各 ~20 µs 的半深 K 尾体」,按指令粒度一算是**零**。

- 这两个核唯一的 fp8 MFMA 是 `v_mfma_f32_16x16x128_f8f6f4`(单条覆盖 K=128),
  且 `SQ_INSTS_MFMA == M·N·K/32768` **逐条吻合**(32768 = 16×16×128)⇒ 收缩轴的**指令粒度就是 128**。
- `ceil(2880/128) = 23`,`ceil(2944/128) = 23` ⇒ **两边条数相同**,尾体省不下**一条** MFMA。
  纸面上那 2.2% 的 "pad MAC 浪费" **被向上取整整个吃掉了**,不可寻址。
  能省的只剩尾迭代的 feed 字节(该核 feed 搬运 84 µs 的 2.17% ≈ **2 µs**)。
- 同源踩证:只把**声明**的收缩深度从 2944 改到 2880(不动指令粒度)= **逐位相同但 −1.70%** ——
  那 1.70% 全是丢掉 2944 行 pitch 的 128 B 对齐,不是省下了什么。
- ⇒ **通用配方**:`inst_k` 是硬粒度。判一个 K-pad 值不值得做尾体,只看
  `ceil(K_real/inst_k) < ceil(K_pad/inst_k)` 成不成立;不成立就**只能换 `inst_k` 更小的 MFMA**
  (本例 `32x32x64` 的 K 粒度 64,且 2880 = 45×64 整除 ⇒ 真能吃到 2.17%,顺带每条指令的 operand
  复用翻倍:16 MAC/B 对 8 MAC/B),但那是整套累加器 layout + epilogue 的改写,
  且 pitfalls/01 记着「MFMA nonkdim 32 永远不赢」⇒ 先在孤立单核上比每-MAC 发射效率,别直接改生产核。

### gfx950 fp8 的两条 f8f6f4 MFMA:同宽操作数、2× MAC、4× 累加器（汇编器实证）

`llvm-mc -arch=amdgcn -mcpu=gfx950` 两条都汇编通过（**5 秒判据,别靠猜有没有这条指令**）:

| 指令 | 单条 MAC | 操作数(每侧) | 每字节复用 | 累加器 VGPR |
|---|---|---|---|---|
| `v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[16:23], v[24:31], v[0:3]` | 32768 | 8 VGPR = 32 B/lane | 8 MAC/B | **4** |
| `v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[16:23], v[24:31], v[0:15]` | 65536 | 8 VGPR = 32 B/lane | **16 MAC/B** | **16** |

- 🚨 **但「条数减半」不是吞吐杠杆——这条轴已判负,别立项**:`SQ_VALU_MFMA_BUSY_CYCLES / SQ_INSTS_MFMA`
  实测 **恰好 32.0 对 64.0** ⇒ 32x32x64 单条占管道**正好 2 倍长**,同 FLOP 的管道周期**守恒**。
  实测同核只换主环 MFMA(ISA 除条数 1104→552 外逐项相同):聚合 MFMA_BUSY **逐位相同**,e2e **12/12 慢 0.8–3.0%**。
  根因是 32×32 tile 一个相位只剩 **2 条独立累加链**(16×16 有 8 条)⇒ 遮 SrcC RAW 与 barrier convoy 的独立工作减半。
  **详卡 pitfalls/05 §MFMA 形状轴关闭 · 决策索引第 64 行。**
- ⚠️ **同理「每字节复用翻倍」也是假的**:复用翻倍是**每条指令**的账,而该指令占管道 2 倍时长 ⇒
  **每周期的 operand 需求完全相同**,feed 侧一分钱不省。上表只能用来算 K 粒度与累加器占用,
  **不能用来推吞吐或 feed**。
- ⚠️ 附带代价:256×256 / 8-wave 的累加器 128 AGPR → **256**,512 共享池下 occ 2 waves/SIMD → **1**。
- ⇒ **正确读法**:这张表回答的是「K 粒度能不能整除真实收缩深度」(选 `inst_k` 用)与「累加器要多少寄存器」,
  **不是**「换大形状能不能提速」。后者已有 12/12 的反面实测。
