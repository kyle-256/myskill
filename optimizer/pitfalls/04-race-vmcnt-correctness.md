# race / vmcnt / 正确性合章：partial-drain、spill race、SRD 寻址、HW-walled 死路、SNR gate、attention 中性值

> 类别: 踩过的坑 · 主题标签: race-correctness, vmcnt, partial-drain, byte-exact, gfx950, vmcnt-race, spill, LDS-direct, SRD寻址, int64溢出, waterfall, LDS-sync, OOB, LDS-load, SCVGPR-prefetch, flydsl-isa, 精度验证, SNR, fp8, scale-bug, attention, online-softmax, correctness, mfma

## partial-drain race：读写距离决定安全 defer，vmcnt 按所有池写总数算

> 术语：**相位/phase** = 主循环一次迭代的写-读窗口。**drain**（全 drain）= `s_waitcnt vmcnt(0)` 等所有在飞写落地。**partial-drain** = 只等到 `vmcnt(N>0)`，留 N 条写跨 barrier 继续在飞以省等待。**距离** = 一条写与对它的读之间隔的相位数；缓冲越深距离越大，能安全 defer 的在飞写越多。**nsa/nsb** = A/B 两个 3buf 池各自每相位的 `buffer_load` 写条数。

- **读写距离决定安全 defer**：2buf = 距离 1（写完下一相位就读，必须**全 drain**）；3buf = 距离 2（有整整一相位余量，可安全 **partial-drain** 让写跨 barrier）。缓冲深度不足就 defer 写 = 低概率 race。（不是"总能 partial-drain"，取决于 buffer 深度给的距离余量。）

- **多池 partial-drain 的 vmcnt 必须按所有 3buf 池的写总数 `sum = nsa + nsb` 计算**，不能只算第一个池。`_n_outstanding = nsb` 是 bug：2 池都 3buf 时实际延迟 `2×nsb = 8` 条写，但只 drain 4 条 → 留未 accounted 的 in-flight 写 → **1/30000 race**。
  - 修法靠 emit 顺序 + 正确 vmcnt(sum)：把 distance-2 安全的 B 池排**最后** emit，`vmcnt(sum)` 恰好留 B 池在飞、把 distance-1 的 A 池**全 drain**。

- ★★ **距离够 ≠ tail 可以随便放长：在飞 tail 长度本身是安全旋钮**（2026-07 mxfp8 grouped NT 实测）。把 A1 池从距离 1 改成距离 2（每条 g2s 有整整一个 K-iter 落地余量）后，按**发射顺序**推算 `vmcnt(2*(NA+NB))=8`（= 留整整一迭代的 8 条在飞）是充分的 —— 但实测 **race**：N=2944 与 N=5632 上重复跑的输出有 134~389 行/131072 不一致、maxabs 16.0。把 tail 收回**距离-1 流水原本的值** `vmcnt(2*NA+NB)=6`（= 一迭代的发射数减去最早那一组）后，4 个 shape × 4 次重复**全部逐字节一致**，且距离-2 的性能收益仍在。
  - 根因 = 本卡下方已记录的 **gfx950 vmcnt 乱序退役**：`vmcnt(N)` 只保证"在飞 ≤N"，**不保证退役的是最老的 N 条**。所以"按发射顺序算够"是不充分条件；tail 越长，被需要的那条写还在飞的概率越高。
  - **可操作规则**：tail ≤ 一次迭代的 g2s 发射数 **减去最早发射的那一组**；要放宽必须实测重复一致性，别靠推理。
  - ★ **bench 的 det 门抓不到**：窄 shape（本例 g=2/mg=1024/N=256 对齐）上 `det=True` 照常通过、`ok=true` 照常给分，只有**计时用的宽 shape**（N=2944/5632）重复对比才暴露 → 改 vmcnt/barrier 的轮次必须自带宽 shape 重复探针（同 pack=4 的 512 对齐坑，「确定性门 ≠ 正确性门」）。

- **byte-exact 排除法必须组合所有省空间手段再算**：曾判"双池 3buf 放不下"而误否决双池路线。错在没把 **scalar store 省 C_lds（8704B）** + **`_CS=1024` 省 bank-pad（5120B）** 组合起来算。单独算每个都不够、组合才够；单独评估会**假阴性**关掉真路。凡涉及 LDS 容量卡点，先把所有省空间手段叠加后再判可行性。

### ❌ 别再试（HW-walled 死坑，partial-drain 相关）

- dwordx4-lds 直写 LDS 同步死路 + SCVGPR prefetch racy 完整版：见下方「gfx950 HW-walled 死路」章节
- BK128 SCVGPR scale VGPR WAR race 完整机制：见下方「gfx950 HW-walled 死路」章节「BK128 SCVGPR scale VGPR WAR race」

来源: 10-grouped-wgrad-4wave-3buf.md, 05-dead-ends.md

## gfx950 vmcnt 非 FIFO race：scratch_load 与 buffer_load_lds 混排，只在 spill 时暴露

### 根因
- **scratch_load（编译器 SGPR spill 回填）和 buffer_load_dwordx4...offen lds（LDS-direct GMEM→SMEM）都走 vmcnt，但两者之间 vmcnt 不保证 FIFO。**
- 编译器对 spill 发的 `s_waitcnt vmcnt(N>0)` 不安全——scratch_load 可能还在飞，`v_readfirstlane` 读进 stale value → `s_mov_b32 m0` 用 stale m0 → 下一条 buffer_load_lds 写到错 LDS 地址 → SMEM 错位 → MFMA 污染 → bit 不一致。
- **只在 SGPR pressure 触发 spill 时暴露**（无 spill 时不出现）。

### ★★ 手写的 partial `vmcnt(N)` 在 spill≠0 时语义失效(2026-07-30 hd64 fwd 实测)

上面那条是**顺序**问题(两类 load 之间 vmcnt 不保 FIFO);这条是**计数**问题,独立且更容易漏:
scratch 的 load/store 与你的 DMA **共用同一个 vmcnt 计数器**,所以一旦 spill≠0,
`vmcnt(1)` 就不再等于"还有 1 个 DMA 在飞",落后的一组会读到已被覆写的 K/V。

- 踩证:一个改动把 spill 从 2 推到 22 dword,此后手写的 partial drain 全部错位;
  把常数换成 `vmcnt(0)` 反而换出**完全不同**的失效签名,再把 DMA 往后挪则两组都坏 —— 三次都不是相位算错。
- ⇒ **规程:凡是依赖 partial `vmcnt(N)` 的 kernel,`spill != 0` 时一律把 N 视为不可信;
  先把 spill 压到 0,再谈"是不是相位错了"。** 反过来说,调这类常数前必须先看 ISA 的 spill 计数。
- 相关但不同的一条:**以指令条数写死的 `vmcnt` 常量会随 CTA 波数失配** —— `NUM_DMA_K+NUM_DMA_V` 只在
  某个 `dma_wave_reps` 下等于"一个 tile 的指令数",波数翻倍后同一常量会多放一个 tile 在飞。
  签名是"输出全对、det=True、多形状全过",三个常规信号**全部放行**;改 CTA 波数后必须按 **tile** 而不是按指令数重算。

### race harness（如何复现/统计）
- `repro_race_*.py`：`ref=kernel(inputs)` 后循环 `kernel` + `torch.testing.assert_close(out, ref, rtol=0, atol=0)` 严格 bit-exact 统计 races。
- **ITERS 用 50000+ 才有统计意义**：0.005% 量级 race 在 10K iters 看起来像 0/N，要 50K-200K 才稳定可见；多 GPU 跑确认不是单卡 fluke。

### disasm 提取 GPU code（从 .so 抠）
- `LINE=$(roc-obj-ls $SO | awk '/gfx950/{print;exit}')` 取 offset/size
- `dd` 抠出 `/tmp/gpu.hsaco`，`llvm-objdump -d --triple=amdgcn-amd-amdhsa --mcpu=gfx950` 反汇编
- `llvm-readelf --notes` 取 reg notes（`.private_segment_fixed_size` / `.sgpr_count` / `.sgpr_spill_count` / `.vgpr_spill_count`）
- 工具在 `/opt/rocm/bin` 和 `/opt/rocm/lib/llvm/bin`。

### disasm 签名（如何确认是这个坑）
- `private_segment_fixed_size > 0`（有 spill）
- 主循环里 `scratch_load_dword` 夹在 `buffer_load_dwordx4...offen lds` 之间
- 紧跟 `s_waitcnt vmcnt(N>0)` + `v_readfirstlane` + `s_mov_b32 m0` + `buffer_load_lds`
- 若 `private_segment == 0` 但仍 race：看 SGPR 是否 spill 到 VGPR lanes（`v_writelane` / `v_readlane`）——lane ops 不走 vmcnt，不会 race。

### （已废弃的早期尝试，勿用）
> 早期（gpt_oss v1）双管齐下法（A 降 spill flag + B prologue 2-step）已被 reviewer 否决，不修根因，逐条见下方「死胡同」列表；定案修法见修法一/二/三。

reviewer 核心观点：**"硬件 vmcnt 行为是约定，编译器在合理 pressure 下不该 spill。出现 spill 是 kernel 写法逼出来的，去 kernel 源头改。绕通道、加 flag、占 pinned 区都是把症状藏起来。"** 真正的定案修法是下面的修法一/二/三——从源头消除 VGPR spill，reg notes 要求 `private_segment_fixed_size=0` / `vgpr_spill_count=0`。

### 第一类 vmcnt FIFO race 三修法（消 VGPR spill，定案）
**修法一 — SGPR 化地址：**
- SMEM 地址全 SGPR：用 `__builtin_amdgcn_readfirstlane(warp_id*STRIDE)` 强制 SGPR。
- **删 `sts_offsets` 里的 lane_id 部分**：buffer_load_lds 硬件按 data_size 自动 per-lane stride（b128=16B/lane、b32=4B/lane），源码加的 `lane_id*16` 是冗余的。
- scale 路径同理 `readfirstlane(warp_id*64*4)` 删掉 `+lane_id`。
- 实测 grouped fwd `17 vgpr_spill / 72 scratch → 0/0`。

**修法二 — outer 解析基址：**
- outer kernel 解析 per-group 基址，inner `compute_tile` 只接收 resolved 的 **5 个 base ptr**（`a_grp_ptr/b_grp_ptr/a_s_grp_ptr/b_s_grp_ptr/c_grp_ptr`）。
- inner 里 `a_base=a_grp_ptr+(int64_t)pid_m*k` 一步加法、无 i64 mul/split-load → inner SGPR/VGPR live set 接近 dense single-GEMM。
- `compute_tile` 必须 `__forceinline__`（`__noinline__` 会 -25% perf）。

**修法三 — 输出转换避软件分支：**
- float→CType 转换 specialize，bf16 用 raw-bit truncation：`r.data=(uint16_t)(__builtin_bit_cast(uint32_t,f)>>16)`。
- 因为 `hip_bfloat16(float)` 默认 ctor 有 round-to-nearest-even 软件分支 → SCC 结果 lane-spill 到 v111 产生 **7 个 v_writelane/v_readlane**。
- `__half(float)` 是单条硬件 `v_cvt_f16_f32` 但 bf16 无对应；gfx950 有 `v_cvt_pk_bf16_f32`（pack 2 fp32→2 bf16 硬件 round）。
- truncate vs round-to-even 半位精度差、SNR 不变。

> 完整三类 race 速查表见下方「gfx950 HW-walled 死路」章「三类 race 速查表」。本节修法（消 VGPR spill）对应类 1（vmcnt FIFO race，commit abd3833）。

### 死胡同（不能同时拿 0 race + PR HEAD perf，reviewer rejected 列表）
- ❌ 别再试：`__noinline__ compute_tile` → **-25%**（args 通过 scratch 传，反而制造更多 spill）
- ❌ 别再试：全 main loop buffer_load_lds→2-step → **-85%**（破坏 phase_mfma_lds_ldg overlap）
- ❌ 别再试：mid-iter 主循环加 `s_waitcnt vmcnt(0)` → race **不降 0** 且 **-21%**
- ❌ 别再试：FIRST_2STEP（每 phase 第一个 load 2-step）→ race **不降 0** 且 **-30~40%**
- ❌ 别再试：prologue 多个 `vmcnt(0)` drain / mode-switch 藏 spill → 被 reviewer 拒（没修根因）
- ❌ 别再试：`-mllvm -amdgpu-enable-merge-m0=true` → 工程 flag，不修根因
- ❌ 别再试：`-mllvm -sink-insts-to-avoid-spills=true` → 同上，不修根因
- ❌ 别再试：`tile.reserve_pinned_regs()` 多处调用 → 占用 pinned 区的 hack，不解决"为什么编译器要 spill"
- ❌ 别再试：Prologue 2-step（即使只在 prologue）→ 同上根因，counter 通道分工被破坏

来源: gfx950-vmcnt-race-debug/SKILL.md（早期版，类 1 的 A/B 修法已被现行版否决）；gfx950-vmcnt-race-debug/SKILL.md（现行版，类 1 定案修法一/二/三 + 类 2/类 3，commit abd3833/bb48d3f/d7b149a）

## SRD/寻址坑：readfirstlane 必须 pin SGPR、大-G int32 溢出静默错、sched_barrier load-bearing

- **readfirstlane 必须 pin SGPR（waterfall 陷阱）**：任何 per-group scan 派生的值（`m_start`、`group_idx` 等）若直接拿去构建 SRD base，divergence analysis 会判为 VGPR → 触发 waterfall（对每个 lane 值串行化循环）。必须显式 `_readfirstlane_i32(base)` 把它 pin 到 SGPR。WHY：SRD base 要求标量，VGPR base 让编译器插 waterfall loop 逐 lane 展开，串行化整个访存。

- **大-G MoE 累积偏移 int32 静默溢出**：persist wgrad 里 `m_start*OUT_M`，当 `G=256 / M_g=1536 / OUT_M=8192` 时 `m_total*OUT_M = 3.2e9 > 2^31` → 最后几组梯度**静默出错（无报错、无 crash）**。修法：把 `m_start*OUT` 折进 **i64 SRD base**；`num_records` 用 per-group `M_g*OUT`（**不要**用累积的 `m_end`，那才是溢出源）。

- **sched_barrier(0) before-mfma 是 load-bearing 的 LDS sync**：它在小-K 场景承担 LDS 同步职责，**删除会导致小-K 正确性坏（SNR 坏）**。❌ 别再试：把它当作纯调度 hint 删掉/移位来"清理"代码——它是隐式 barrier，删了小-K 出错。

- **OOB 修法优先级：修不变量 > 加 mask**。按根因对症，不要一律糊 mask：
  - 循环 trip count 错 → 减循环次数 / 减向量宽
  - per-lane 所有权变了 → 同步更新 layout + LDS store + reader 三处
  - 边界 partial tile → clamp / predicate
  - descriptor 范围太大或 offset i32 溢出 → chunk buffer resource，或在 truncate 前加宽算术（i64）
  - 修后重跑：失败 shape + 一个相邻边界 shape。

- **create_buffer_resource 的 max_size 坑**：`max_size=True` 会 OOB 读垃圾（把 descriptor 范围拉满，越界读进相邻内存）。必须用 `max_size=False, num_records_bytes=...` 精确给范围。（与「gfx950 HW-walled 死路」章 fp8 ISA 小节同条，此处保留 WHY；fp8 cast 等其余硬约束见该小节。）

来源: 08-deadends.md, 04-tn-wgrad-kernel.md, 02-nt-fwd-kernel.md, oob-detection/SKILL.md, flydsl-sync/SKILL.md

## gfx950 HW-walled 死路：buffer_load_dwordx2_lds 不支持、SCVGPR prefetch racy

### buffer_load...lds 直写 LDS 的同步死路（HW-walled）

- ❌ 别再试 `buffer_load_dwordx2...lds`：gfx950 **不支持** dwordx2 的 lds 变体，只有 `dword` 和 `dwordx4` 两种宽度。
- ❌ 别再试 用 vmcnt / 隔-phase barrier 同步 `dwordx4-lds` 直写 LDS 的完成：**不被可靠同步** → det≠0。
  - 唯一能 det0 的写法是 `vmcnt(0)` + 紧跟 `s_barrier`，但这会**序列化**，实测 **4803 < 5176**（更慢），得不偿失。
  - WHY：LDS 直写完成不保证在 vmcnt/barrier 观测点前落地。

### SCVGPR prefetch racy（HW-walled）

- ❌ 别再试 SCVGPR 的 `SCV2AHEAD` / `SCPF` prefetch：racy。
  - WHY：**vmcnt 乱序退役**，不保证特定 load 在某个时刻落地。

### BK128 SCVGPR scale VGPR WAR race（同一机制）

- ❌ 别再试 BK128 SCVGPR scale VGPR 写法：存在 WAR race。
  - 证据：K=256（0 main iter）SNR **55.6** 正常，但 K=384（1 iter）SNR **-inf** 崩。
  - 机制：phase-B 的 `emit_sc_vgpr(0)` → 写 `v[8:9]`，覆写了 phase-A mfma **还在读**的 scale VGPR。
  - `vmcnt(0)` **不能修**：vmcnt 乱序退役，不保证该特定 load 落地。
  - 与 **BK256 SCVGPR 同一机制**。

### 三类 race 速查表（可修，别当 HW-walled）

区别于上面的 HW-walled 死路，下面三类 race 是同步漏洞，**可修**：

| 类 | 冲突对 | 范围 | 触发 | 修法 | commit |
|---|---|---|---|---|---|
| 1 vmcnt FIFO | `scratch_load` vs `buffer_load_lds` | wave 内 | spill>0 单次即 race | 消 VGPR spill（SGPR 化地址 / outer 解析 ptr） | abd3833 |
| 2 LDS WAR | `ds_read`(lgkmcnt) vs `buffer_load_lds`(vmcnt) | 同 WG 跨 warp | spill=0 单次即 race | `wait_lgkmcnt<0>()` drain | bb48d3f |
| 3 die L2 | 复用 workspace 跨 die 跨调用 | 跨 XCD | 前两类干净、第 2+ 次调用 race | 入口一次 `__threadfence()` + per-tile 退化成 lgkmcnt drain | d7b149a |

#### 第二类：LDS WAR race（commit bb48d3f）
- 现象：spill 全 0 但仍 race（~0.004%/tile，多 config 累积 30-50%/session）。
- 机制：`buffer_load_lds`（vmcnt-tracked LDS write）与 `ds_read_pinned`（lgkmcnt-tracked LDS read）目标同一段 LDS 时，`s_barrier` 只同步执行位置**不 drain lgkmcnt** → warp 的 `ds_read` 还在飞、`buffer_load_lds` 已 commit → 读到 post-write 数据污染 MFMA。
- 修法：两处发 `wait_lgkmcnt<0>()`：(1) `phase_mfma_lds_ldg` 的 mid-phase WAR barrier **之前**（原来只有 `s_barrier`）；(2) `compute_tile` prologue 里 `if(k_iters>2)` 块**之前**（原本 drain 在块之后、顺序错了）。

#### 第三类：跨 die(XCD) L2 coherence race（commit d7b149a）
- 机制：MI355X 一 node=8 个 die(XCD) 每 die 一块 L2、**die 间不自动 coherent**，persistent kernel 256 个 block 散在 8 die 上跑。复用 workspace 时上次写 GMEM 的副本缓在某 die L2，这次 `buffer_load_lds` 命中 stale L2 行 → 污染。**只在复用 workspace 的第 2+ 次调用出**。
- 修法：kernel 入口发一次 `__threadfence()`（gfx950 lower 成 `buffer_wbl2 sc1` + `buffer_inv sc1` = flush+invalidate 本 die L2）。官方 deterministic ×100 全过。
- ❌ 别再试 per-tile L2 flush+inv：MoE 上 **14-40% tax**。拆两半：per-tile 只保留 `wait_lgkmcnt<0>()`（drain in-flight `buffer_load_lds` 的 GMEM→LDS 写，无 L2 流量）；per-block 一次在 kernel 入口 tile loop 之前发 `__threadfence()`（GEMM 输入只读，一次 invalidate 覆盖该 block 所有 tile）。fwd kernel 改后 dgrad 复用自动受益，wgrad 独立要单独改两处。实测拆分后 vs per-tile-fence baseline：Fwd **+10~85%** / Dgrad **+11~60%** / Wgrad **+33~87%**，det60 bit-exact 不回。

#### race 定位法（跨 kernel + kernel 内子结构）
- **pytorch fp32 ref 替换**：python 层加 env-var dispatch，`PRIMUS_TURBO_MXFP8_<STAGE>_REF=1`（`_stage=fwd/dgrad/wgrad`）返回 fp32 dequant+matmul+cast 的 deterministic ref，矩阵化跑 deterministic 测试锁定 race 在哪个 C++ kernel（本 case 锁定 `turbo_grouped_gemm_mxfp8` fwd/dgrad 共享 + `turbo_grouped_gemm_mxfp8_wgrad`）。**commit 前必须删掉 ref 路径+env-var。**
- **补 single-GEMM 覆盖**：single GEMM mxfp8 之前无 deterministic 测试，补最小 deterministic 测试发现 single GEMM 也 race（~30% session fail）→ race 在 shared kernel structure（`phase_mfma_lds_ldg` / 单 GEMM `compute_tile`）而非 grouped 特有的 persistent-loop / per-group 解析；single GEMM 更小迭代更快，后续定位都用它跑。
- **gate 掉嫌疑块**：把 Epi1 LDG 块 gate 掉 `if(k_iters>999999)`，若 `assert_close` 全过（deterministic 恢复）但 SNR 失败（LDG 提供真正参与 MFMA 的数据）→ 确认 race 在该块，下一步是补 `wait_lgkmcnt<0>` 修同步漏洞而非删。
- **每改一处跑 30 runs 统计 session-level fail rate**：无 fix 30 runs ~10 fail(33%)；加 prologue drain ~1-3 fail(7%)；+WAR barrier drain 0 fail；再 50 runs 0 确认。race rate 0.004% 量级 1 run 可能运气过，至少 30 runs、更保险 50。

### FlyDSL / gfx950 fp8 相关 ISA 硬约束

- ❌ 别再试 `cvt_scalef32_pk8_fp8_bf16`：gfx950 **Cannot select**，不可用。
- ❌ 别再试 `Vec.to(Float8E4M3FN)`：走 `arith.truncf`，后端**不 lower**。
  - 正确做法：fp8 cast 用 `fx.rocdl.cvt_pk_fp8_f32`（2 f32 → 2 fp8 / op）。
- ❌ 别再试 `create_buffer_resource(max_size=True)`：会 **OOB 读垃圾**。
  - 正确做法：用 `max_size=False, num_records_bytes=...`。

来源: 05-dead-ends.md, flydsl-sync/SKILL.md, gfx950-vmcnt-race-debug/SKILL.md

## 低精度容差假门：element-wise tolerance 无意义，SNR 是真 gate

- **get_tolerances 的 fp8=1e-1 / fp4=0.5 是故意放松的假门，不能当真实 gate。** 量化后 element-wise 容差本身没意义（低精度 round 误差天然大），拿它判过/不过会漏掉真 bug 或误杀正确结果。
- **低精度主 gate 必须用 `compute_snr(ref, actual)`——参数顺序 reference 在前。** 传反了 SNR 数值会错。`relative_error / mean_squared_error / max_abs_error / cosine_similarity / symmetric_similarity_diff` 只作诊断辅助，不作判定门。
- **FP8 PV MFMA 相对 bf16 参考有 ~0.03 max error，是 FP8 数据通路固有的，不是 bug。** 对应场景容差用 `atol=5e-3`（比 element-wise 假门严得多，但仍是辅助判据；主判据还是 SNR）。
- **per-row Q vs per-tensor Q 量化方式不匹配会有 1-3% 差异。** 参考实现和 kernel 的量化粒度要对齐，否则这 1-3% 会被误当成 kernel bug。
- **常见 scale bug：`v_scale` 被应用两次**——prob scaling 阶段一次、PV 之后又一次。表现为输出整体偏大，查 scale 应用点是否重复。
- **HK fp8 是 `float8_e4m3fnuz`(max=240)，不是标准 E4M3(max=448)。** 手写 `scale_inv=1/112`（按 448 算）会让输出偏 ~5x。必须用 `quantize_fp8_tensorwise_impl` 自动按真实 max 算正确 scale_inv。

- ❌ 别再试：**拿 element-wise tolerance（fp8=1e-1/fp4=0.5）当 gate**——它故意松，通过与否都不说明正确性，低精度只认 SNR。
- ❌ 别再试：**`torch.randn().to(FP8)` 直接造数据**——>240 会 saturate，且 mma 非线性，saturate 后误差不可预测。用 quantize helper（自动 clamp）造数据。

来源: verify-accuracy/SKILL.md, debug-flydsl-kernel/SKILL.md, fp8-gemm-bench/SKILL.md

## attention 正确性坑：online-softmax rescale/NaN 守卫、空 partition 写中性值、全1测局限

### online-softmax NaN / 除零守卫
- **全 mask NaN 根因**：某 partition 所有 token 都被 mask（超出 context）时 `qk_max=-inf`，则 `exp(s-qk_max)=exp(-inf-(-inf))=exp(NaN)=NaN`；`exp2(-inf - -inf)` 同样 NaN，会污染整个循环剩余的 l/O。修法：`safe_diff = (qk_max > NEG_INF).select(diff, ZERO_F)`；并对 corr 和 partition rescale 都做 `select`-guard 防 `m==-inf`。
- **归一化除零**：`exp_sum=0` 时 `1/exp_sum=inf`。修法：`safe_sum = (running_sum > ZERO_F).select(running_sum, fx.Float32(1.0)); inv_sum = 1.0/safe_sum`。
- **rescale 顺序**：每个 block 必须在 PV MFMA **之前** 用 corr 对 `O_acc` 做 rescale，不是之后。重构时极易漏掉，漏了直接结果错。
- ❌ **别再试 错 peer-reduction XOR mask**：wave64 MFMA32 的 lane-reduce mask（如 `lane^32`）绑定 tile 几何，mask 写错会在错误的 lane group 上 reduce，静默地污染 softmax 统计量（不报错，结果错）。

### 空 partition / decode block-split 必写中性值
- decode 按 block 拆 partition 时，空 / 越窗 partition **必须仍写中性值**：`sum=0, max=-inf, out=0`。否则 cross-partition reduce kernel 读到未初始化的槽。
- fused 单 partition 快路径**只有 single-visible-tile 前提成立时才安全**，用编译期开关 gate 住，不能默认走。

### 输出全错的地址/网格根因
- **全 0 输出**：输出地址错（`stride_out_seq`/`stride_out_part` 错，打印 `output.stride()` 核对）；multi-partition 必须写到 `part_z` 槽而非绝对 partition index（reduce kernel 从 `part_z=0..grid_z-1` 读）；主 kernel 没写 `exp_sums`/`max_logits` 则 reduce 出 0（启动前 `fill_(-999.0)` sentinel 验证）。
- **>50% 错**：常因 `grid_z < total_partitions` 且 kernel 每 CTA 只处理一个 partition（无循环），大部分 context 被跳过。验证 `total_parts=ceil(context_len/KV_COMPUTE_BLOCK)`，assert `grid_z==total_parts` 或有 multi-partition 循环。

### wave-collective 读前禁 divergent scf.if(gfx950/flydsl,实测)
- **`ds_read_tr16_b64`(16-lane 协作 transpose 读)前面若有一个按 wave/lane divergent 的运行时
  `if wave==0:` scf.if——即便块内只是纯 side-effect `buffer_store`(不外泄变量)——会破坏随后的
  collective 读**,产出 NaN / 3.4e38 overflow(实测 sparse-MLA bwd dQ:1922 nan/27 inf,localized)。
  - 机制:divergent scf.if 改变了 collective op 所需的 lane 活跃/寄存器状态。
  - **修法:把 divergent 条件写移到 collective 读之后**(如 dS/P 的 wave0-only store 放到 PV loop 之后)。
  - 定位捷径:同 kernel 里不经过该 collective 的输出(dkv/dsink,走普通 MFMA/reduce)全程正确,
    只有经过 ds_read_tr16 的输出(dq)崩 → 精准锁定到 collective 读路径。
  - fwd flash kernel 无此坑,因它无 wave-divergent 分支;bwd 因 dS/P 只需一个 writer 才引入。

### 测试局限与 MFMA 操作数
- **全 1 隔离测试**：`query/key/value.fill_(1.0)` 下 softmax 概率全相等、PV 输出=1.0，任何偏差暴露 layout/寻址 bug。❌ **别只靠全 1 测**：均匀值无论顺序都对，测不出 V/P 操作数错位，需另测非均匀输入。
- **MFMA 操作数顺序**：`mfma(LHS, RHS, acc)` —— LHS→M 维，RHS→N 维。QK 中 K 是 LHS、Q 是 RHS。搞错就是 layout 大错。
- **FP8 容差 / 量化粒度不匹配 / v_scale 双应用**：见本卡「低精度容差假门」章（FP8 PV MFMA ~0.03 max error 固有、atol=5e-3、per-row vs per-tensor 差 1-3%、v_scale 被应用两次）。

来源: debug-flydsl-kernel/SKILL.md, attention/optimization-directions.md

---
来源: 10-grouped-wgrad-4wave-3buf.md, 05-dead-ends.md, gfx950-vmcnt-race-debug/SKILL.md, 08-deadends.md, 04-tn-wgrad-kernel.md, 02-nt-fwd-kernel.md, oob-detection/SKILL.md, flydsl-sync/SKILL.md, verify-accuracy/SKILL.md, debug-flydsl-kernel/SKILL.md, fp8-gemm-bench/SKILL.md, attention/optimization-directions.md
