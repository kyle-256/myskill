# Tile / Occupancy / Register 综合踩坑：512 合并寄存器池模型、量子边界、tile 尺寸约束与预取死路

> 类别: 踩过的坑 · 主题标签: occupancy, register-pressure, VGPR, AGPR, SGPR, LDS, quantum, tile-size, mxfp4, fp8, MFMA, prefetch, double-buffer, race-correctness, vgpr-form

## occupancy 是 register-bound 非 LDS-bound：512 合并 VGPR 池模型

### 核心模型：gfx942/gfx950 是 512 合并池，不是 256/max

- 512 合并寄存器池公式与 co-saturation 方法论：见 methodology/04-occupancy-and-tile.md，本卡只保留该模型下的踩坑细节与反直觉数字。
- gfx942 waves/SIMD 阶梯（`512 // (arch+accum)`）：≤128→4wave、≤170→3wave、≤256→2wave、≤512→1wave、**>512 SPILL 严重回退**。
  - 例：arch=148/accum=148→296→1wave；再加 32 arch→328 仍 1wave；要 2wave 需**合并** ≤256。
- 口径统一：上面阶梯的门槛值均指**合并总量 R=arch+accum**（非单独 arch 或 accum）；≤256 得 ≥2 wave、257-512 得 1 wave（compute-bound 可接受）、**>512 SPILL 严重回退**。
- **AccVGPR 不与 arch 竞争分配**：MFMA 累加器用 accum_vgpr（独立文件），预取缓冲/B tile/A tile 用 arch_vgpr，二者不互相竞争寄存器**分配**——但**共享占用预算**。
- **LDS 地址逻辑也吃占用**：LDS 寻址逻辑增长 arch_vgpr，即便不碰 MFMA 累加器也会吃 occupancy；kernel 靠近 2-wave 边界时要压低 LDS 地址 VGPR 压力。
- **每 buffer_load_dwordx4 = 4 arch_vgpr**；双缓冲净增约一组 'next' 缓冲。
- gfx950 8-wave WG（512 线程，2 waves/SIMD）硬上限 V+A ≤ 256 dword/lane（推导：每 wave-slot 512 dword，8-wave 时每 SIMD 2 wave → 512/2=256 dword/lane；16384 = 每 SIMD 物理寄存器总量 512 dword×32 lane 视角，仅供换算）；occ=1 时 512-VGPR 满载锁死 prefetch 深度。

### ❌ 别再试：砍 LDS 提 8-wave mxfp4 occupancy

- **8-wave mxfp4 GEMM 的 occupancy 是 REGISTER-bound 不是 LDS-bound**（实测推翻）。
- 128 VGPR + 128 AGPR = 256 共享 512 文件 → 硬卡 2 waves/SIMD = 1 wg/CU。
- 把 LDS 从 128KB 砍到 64KB（A 直读）**occupancy 完全不变**（1.98→1.99）。
- 要 2 wg/CU 需总寄存器 ≤128，但**累加器单独就 128 AGPR，不可能**。之前把它当 LDS-bound 是误判。

### 4-wave 真瓶颈：occ=1 单 wave 无延迟隐藏，MFMA idle ~40%

- 4-wave mxfp4 真瓶颈 = **occ=1**（1 wave/SIMD，vgpr432 + agpr256，waves_per_eu=1）→ 单 wave 无跨 wave 延迟隐藏，**MFMA idle ~40%**，不是 shuffle。
- **反例配置 BN128**（用来反驳「occ=1 大 tile 必赢」的对照实验；把 tile 缩到 32 accs=128 AGPR + 2-buffer + LDS≤80K，即降算术强度换 occ=2；`HAS_BR=False` 是关闭某内部分支的 flag）→ occ=2 反而藏 ds_read 延迟更好。
  - PMC 证据：LdsUtil 8%（non-bw-bound），大 SQ_WAIT_INST_LDS 被 1-wave/SIMD 暴露。
- **结论**：occ=1 大 tile 不是无脑赢，是 tile 强度 vs 延迟隐藏的权衡。

## 死路：maxnreg 强制 accum_vgpr=0、AGPR 搬移救不了 VGPR 溢出

- ❌ 别再试：用 `maxnreg` 强制 `accum_vgpr=0` 来给预取/arch-VGPR 腾寄存器。历史环境（flag 曾生效时）实测 **~4.5× GPU kernel 回退**（⚠️旧环境数）：占用率翻倍，但 MFMA 累加器被逼经 `v_accvgpr_read` 溢出到 arch_vgpr（arch-VGPR spills）。MFMA-heavy kernel 绝不能用 `maxnreg`。**AccVGPR 压力只能付在 occupancy 上，无法规避。** ⚠️ 本容器 external codegen 不可用 → `maxnreg` 已完全无效（ISA 逐字节相同，见 methodology/03），此路在本容器连「生效」都做不到。

- **死坑：把 accs 搬 AGPR 救不了溢出**。CDNA occ=2 下 `ArchVGPR + AccVGPR` 共享 256 组合预算（`accum_offset 256`）。把累加器搬到 AGPR **不减少总量**，救不了 VGPR 溢出。
  - fp4 MFMA 有 5 个操作数 `(a, b, sa, sb, c)`；`sa`/`sb`（a、b 各自的 scale 操作数）必须留 VGPR，`acc` 可移 AGPR，但 **V 不下降** → `V + A = 384 > cap`。
  - ❌ 别再试：8-wave BN512 BK128 实测 spill 到 Scratch，1132 → 374 TF（**13× 慢**，每个 MFMA 都读写 scratch = 打 HBM）。

## 占用率量子边界：只有跨 allocation quantum 才买到一个 wave

- **省一个寄存器/字节，只有跨过 allocation quantum 边界才买得到一个 wave**。分配是按量子向上取整的：
  - VGPR 按 **8 Dword** 向上取整
  - SGPR 按 **16 Dword** 向上取整（合法范围 16..102）
  - LDS 按 **512 B (gfx942) / 1280 B (gfx950)** 向上取整
- **85→84 VGPR 可能买到一个 wave**（跨过 8 的倍数边界）；但如果省下的那个寄存器**不跨 8 的倍数**，则白省——占用率一动不动。WHY：硬件按量子块分配，块内多省的部分不会释放给下一个 wave。
- **两个悄悄把你顶进下一量子的隐形推手**（省寄存器时容易被它们抵消）：
  - **64-bit 操作数强制偶对齐（even-pair alignment）**——会占掉本以为空着的相邻寄存器槽。
  - **4-Dword SMEM load 强制目标 SGPR quad 对齐**——为了对齐可能跳过若干 SGPR，直接把你顶进下一个 16-Dword 量子。
  - 结论：trim 后一定要**按取整后的实际预算**核算，别按"名义省了几个"算。

- ❌ **别再试**：盲目 trim 单个 VGPR/SGPR 想抠占用率，却不检查是否跨量子边界。不跨 8(VGPR)/16(SGPR)/512B|1280B(LDS) 的边界 = 零收益，纯浪费调优时间。

- **CDNA3/CDNA4 越界 GPR/LDS 访问不 fault**——sizing/预算 bug 会伪装成数值 bug：
  - 越界 **source 读** → 读到 register 0
  - 越界 **destination 写** → 丢写（multi-dest VMEM/atomic 以 EXEC=0 发射）
  - **LDS 越界**：读返回 0，写被丢弃
  - WHY 危险：没有异常、没有崩溃，结果只是"数值不对"。排查时**必须拿 index 去对取整后的预算交叉核对**，而不是当成算法/精度 bug 去查。

## GEMM tile 别用小/矩形：256×256 唯一可行点，feed-bound 缩 tile 净负

### tile 尺寸硬约束（违反=数据错或编译失败）
- tile_m 必须是 **16 的倍数**（MFMA M 维）。
- tile_n 必须是 **64 的倍数**（4 wave × 16 N）。
- tile_k*elem_bytes 必须被 **64 整除**（K64 字节微步）。
- tile_m*tile_k*elem_bytes 要舒服放进 LDS：**gfx942 64KB / gfx950 160KB**。
- B 预 shuffle 成 `(N/16, K/64, 4, 16, kpack_bytes)`，**tile_k 必须整除 K**。
- default first-pass：HIP/CK 256 threads / 4 waves 拥有一个 (tile_m,tile_n)，合理起点 **tile_m=256, tile_n=128, tile_k=64**。
- 占用边界：越过 **128 arch-VGPR** 掉 2 waves→1 wave/SIMD；越过 **256 VGPR** spill。启动足够 work-group 盖满全部 CU（**304 gfx942 / 256 gfx950**），否则小 grid 灌不满器件。

### ❌ 别再试：feed-bound worst shape 缩小/矩形 tile
- 现场：wgrad worst shape gpt_oss-down **2880²**，256×256 只 **~48% peak（MfmaUtil 54%）= feed-bound**。
- 缩 tile 直接砍算术强度 (M·N)/(M+N)：
  - 256² = **128**（基线）
  - 128×256 = **85（-33%）**
  - 128² = **64（-50%）**
- feed-bound 下缩 tile 净负**远超** padding+grid 能救回来的量。**tile 重写不做。**

### ❌ 别再试：tile-menu 多形状 / 大 tile（AITER 那套）
AITER 有 per-shape 选 128×128 / 192×256 / 256×512… 在本架构（gfx950 mxfp4）不可行：
- **256×512**：acc 超 **256 AGPR** → occ=1。
- **128×512**：standalone 落后。
- **512×256**：超 **144KB LDS**。
- 结论：**256×256 是 gfx950 mxfp4 唯一可行 tile 点。**

### ❌ 别再试：BLOCK_N=512（撞 CShuffle EPL assert）
- BLOCK_N=512 撞 **CShuffle EPL assert（EPL=16 ≠ 8）**。
- bigger tile（256×512、512×256）会报 **INVALID_ISA** 或 CShuffle EPL assert。
- **不要扫 bn=512。**

### ❌ 别再试：对 6144 用 split-K
- 6144³ 残差 fly/ait~**1.05** = tile 量化：256×256 → **576 tiles / 256 CU = 2.25 wave 不均**；AITER 192×256 → **768/256 = 3 整除**。
- 但 split-K 切细**反而更慢**：576 tile compute 已高效、无尾波浪费。**别对 6144 用 split-K。**

## mxfp4 BK128 死路：loop 固定开销碾压 compact-LDS 收益

- ❌ 别再试 mxfp4 4-wave BK128：plain = 4597 med，远 << BK256 5401。根因是 loop 固定开销——BK128 要 224 iter vs BK256 112 iter，iter 数翻倍带来的 loop 固定开销碾压了 compact-LDS（更小 BK 换来的 LDS 占用/复用收益）。
- ❌ 别再试在 fly manual-emit 下压 BK128 loop 开销：fly manual-emit 路径下 BK128 的 loop 固定开销无法消除。aiter 能做是因为靠 hand-asm 手工消掉了这部分开销——对比：BK128 历史顶 4715 vs aiter 5635，差距即来自 hand-asm 的 loop 消除。
- ❌ 别再试 BK384 / BK512：有 `BLOCK_K % 128 == 0` 的 assert 约束；BK384/512 会 Assert 失败或 VGPR 溢出。

## register-bound 墙下预取/double-buffer 全 spill：mxfp4 4w/8w 加 frag 破 occ

### 4-wave (occ=1) 的 VGPR headroom 只有 88
- occ=1 下 V cap=512 / A cap=256 dwords(两个独立堆)。生产 mxfp4 4-wave：V=424 / A=256 / spill=0。headroom = 512-424 = **88 VGPR**。
- ❌ 别再试 register double-buffer / read-once / RING / ROBUF：2 operand set = 2×192V = **384V ≫ 88 headroom** → RAGreedy crash / `couldn't allocate v[256:259]` / LLVM UNREACHABLE。
- BK128 能 fit(192V)但引入 loop 开销，不划算。read-once 在 fly BK256 物理放不下。

### 8-wave (occ=2 天花板) VGPR 预算 256/wave，零余量
- 8-wave = 2 waves/SIMD → VGPR 预算 = 512/2 = **256/wave**。
- ISA 实测 8w whole-loop 已占满 256 VGPR 零余量：32 vec4 accs(128)+ 24 operand frags(96)+ 6 scale + ~26 地址 temp ≈ 256。
- ❌ 别再试 naive register-prefetch(+96 第2套 frag)：352>256 → occ 掉 → 512线程 WG 无法单 CU 驻留，内核起不来。架构性死路。

### A operand 1-deep 预取 (APF=1) — 双重死
- ❌ 别再试 APF=1：每个 i32x4 A frag=4 VGPR，a0p+a1p+a0n+a1n = **128 VGPR 仅 A** → Scratch spill(332)，且有 K 相关 tail race(SNR 15.3 崩)。
- 被 register-bound 墙挡死：1-deep A 预取需 +64 VGPR，raw 的 VGPR 上限只有 256-128(AGPR)=128，加后 **300>256** 破 1 wg/CU。

### MONOHOIST：把 operand ds_read 上提到迭代顶 — 只提 b1 是唯一干净赢
（名词表：a0/a1=A 的两片 operand fragment，b0/b1=B 的两片；数字后缀如 b2/a2=该 frag 下移到第 2 个 barrier 之后的变体。raw=bare-asm whole-loop 路径 / intrinsic=编译器 SSA 路径，二者区别见本卡「raw-AGPR 8-wave」小节。）
- ❌ 别再试 FP4_MONOHOIST=1/both/a：同时持有 a0+a1+b0+b1(~96VGPR)+scales/addr/g2s/readout 超 128 预算 → spill(scratch)，perf 掉到 **3735**(baseline ~4477)。
- ❌ 别再试 FP4_MONOHOIST=a2(a1 下移一 barrier)：单独 SNR29.6 数值错 + spill perf **3385**。b2a2 组合正确但仍 spill perf **3350**。
- ✅ 唯一干净赢点 = 只提 b1(b2，+16VGPR，num_vgpr 128→110 不 spill)：稳超 raw-baseline **+0.7%**，微超 intrinsic **+0.13~0.16%**。

### race 安全红线(真 race，非性能)
- 把 operand ds_read 上提到迭代顶部(首 barrier 之前)会读到 8 波协作填充的 LDS 未同步数据：K28672 SNR46.6 / det262400。
- b1 可安全下移一个 barrier(首用在 quad1 有 slack，且 b_cur1 在更早 barrier 已可见)；再往顶提就 race。
- a1 一个都不能提(无 slack，紧贴其首用 quad2)。

### FP4_LDSR：手动提前发射 ds_read 隐藏延迟 — 反退
- ❌ 别再试 FP4_LDSR(手动把 a1/b1 的 ds_read 提到 iter 顶):反退 ~20%(K2048 **2509 vs 3126**)。
- 机制:破坏编译器已做的分段 lgkmcnt 软流水。FP4_RAWSPLIT=1 每-mfma 一块时，编译器已自动做 operand 前置 + staggered s_waitcnt(vmcnt(5)lgkmcnt(7)→lgkmcnt(6)→lgkmcnt(3)…)。LDS 读延迟隐藏编译器已到位，Python 层加不了价值。

### 单 volatile asm 块把 operand 作 SSA 输入 = 强制全 drain
- ❌ 别再试 单个 volatile asm 块把 operand 作 SSA 输入:LLVM 必在块前插 **lgkmcnt(0) 全 drain**(不透明块读所有输入寄存器 → 等所有 ds_read 落地)，反比 per-mfma split 差。
- 正解:真正隐藏 operand 延迟必须把 ds_read 塞进 asm 块内手控 waitcnt(whole-loop bare-asm 就是这么做，staggered lgkmcnt)。

### raw-AGPR 8-wave 追平但不反超 intrinsic 的结构根因
- 1 wg/CU 要求 VGPR+AGPR≤256:raw = 108V+128A = 236；intrinsic = 256V+0A。
- raw 的 VGPR 上限只有 256-128(AGPR)=128，intrinsic 有 256 VGPR 做在飞 frag/软流水。AGPR 累加器虽不占 VGPR 计数但仍吃 256 共享预算 → raw 能腾的 VGPR 反比 intrinsic 少一半。
- 长 K 残余 0.3~1.5% = volatile-asm 每-mfma 硬编码 AGPR 强序阻止跨 s_barrier 全循环软流水 + VGPR 预算少，非 ds_read 冗余。

## AGPR/VGPR-form 是 occupancy-specific：4-wave 用 AGPR、8-wave 用 VGPR-form

- **AGPR 是 occupancy-specific（实测核心结论）**：`Mfma16x16x128AGPR`（inline-asm `=a,v,v,0` 把累加器钉到 AGPR）对 **4-wave（occ=1）+5~13%**，但对 **8-wave（occ=2）−30~37%**。WHY：8-wave 下 AGPR 预算减半 → 要么掉 occupancy，要么 RF 互倒（register file 打架）。所以 **8-wave 不要 AGPR**，用编译器默认 VGPR-form（SSA）。
- **fp8 下 4w≈8w（±2.5%）**：4w 赢大 N 的 FFN gate、8w 赢小方阵；不像 mxfp4 那样 4w 碾压。选 wave 数要按 shape 分。

- ❌ **别再试 intrinsic+LLVM scheduler 追平手写 asm**：intrinsic+bundled LLVM scheduler **4500** vs 手写 asm **5401**——FlyDSL bundled LLVM 调度精度不足，手写 asm 打败 bundled scheduler。相关旋钮同时坏：`asm-AGPR`/`FP4_AGPR` 在 asm_mma 路径已坏（AccVGPR 仍 0）；`WPEHINT`/`MAXNREG`/`MaxNReg` hints 不在 FlyDSL JIT 传播（VGPR 不变）。

- ❌ **别再试上游 AGPR-pin commit 救 mxfp4**：上游 ROCm/FlyDSL AGPR-pin commit（**#714 aeb5afc**，作用于 fp8 4wave AGPR 原地累加、消 accvgpr-shuffle +5~13%）对 **mxfp4 4-wave 无帮助**。WHY：mxfp4 走 bareasm whole-loop，accs 已用 `=a` tied 原地累加、更彻底，accvgpr-shuffle 问题根本不存在。实测 **agpr1（5407/5322）≈ agpr0（5390/5348）** 噪声内相等。上游针对 SSA-lowered 路径，bareasm 无此路径。

- ❌ **别再试 `amdgpu-mfma-vgpr-form=false` 在生产 4-wave**：对生产 4-wave 内核**零收益**，ISA 逐字节相同（accvgpr_write=1 / read=256 / agpr=256 / vgpr=432 / scratch=0）。WHY：AGPR 累加是 FlyDSL intrinsic 层既成的，**与编译 flag 无关**——本容器 external codegen 下 `maxnreg`/`waves_per_eu`/`amdgpu-mfma-vgpr-form` 全无效（见 methodology/03），累加器本就在 AGPR 里，循环内 **0 条 accvgpr shuffle**（256 条 `v_accvgpr_read` 全在 epilogue、只发生一次）。探针"8164/512 per-MFMA accvgpr shuffle 是头号瓶颈"是**误诊**（来自旧版/8wave/agpr=False 变体）。

---
来源: 01-architecture.md, gemm-optimization/SKILL.md, lds-optimization/SKILL.md, prefetch-data-load/SKILL.md, kernel-trace-analysis/SKILL.md, programming-model.md, agpr_phase5_lds.md, project_mxfp4_vgprform_deadend.md, diag_4w_vs_8w.md, 09-8wave-ceiling.md, gfx942/overview.md, gfx950/overview.md, flydsl-fp8-gemm-results/SKILL.md, flydsl-kernel-authoring/SKILL.md, gemm/overview.md, project_mxfp4_epilogue_store.md, 03-nn-dgrad-kernel.md, 05-dead-ends.md, 04-ceiling-analysis.md, 10-8wave-scvgpr.md, agpr_phase5_ldsr.md, agpr_phase5_mono.md, agpr_rawasm_progress.md, remote-sync/SKILL.md, 11-upstream-agpr-pin-moot.md
