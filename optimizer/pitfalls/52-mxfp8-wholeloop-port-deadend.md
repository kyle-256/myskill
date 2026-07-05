# MXFP8 whole-loop 移植整体死路：occ=1 结构天花板+vendored 严禁

> 类别: 踩过的坑 · 主题标签: mxfp8, whole-loop, occupancy, E8M0-scale, K-tail

## 硬约束：BK256 LDS 超容 → 只能 BK128
- fp8 operand 2× 字节（vs fp4）→ **BK256 LDS=256KB ≫ 160KB cap**，只能 **BK128（n_sub=1）**。
- ❌ 别再试照搬 mxfp4 BK256 whole-loop：这是 fp8 whole-loop 无法照搬的**硬约束**，不是调参能绕的。

## WL occ=1 结构天花板（dense）
- dense mxfp8 **4-wave WL：64 acc/wave = 256 VGPR → 必须 AGPR acc → occ=1**；per-tensor occ=2。
- WL **非-scaled 也只到 3093 < per-tensor 3277（-6%）**：gap 是**内核结构(occ)差异，非 scaling**。scale 不是根因。
- ⚠️ **8-wave WL 早期诊断"死路"已被撤回**：Stage-0 初期（2026-06-23）曾判 8-wave WL 用 VGPR accs 无法 pin operand 做 fp8 2×b128 → 死路；但同日晚些时候明确撤回此结论——8-wave 仅 32 acc/wave，VGPR acc 放得下 + 留空间 pin fp8 operand，`occ=2 + 手写调度` 被重新评估为冲 98% 的**真正路径**，此后再无来源重新验证/判死 8-wave；WL 最终移出生产是因用户裁定 vendored 严禁（见下），并非 8-wave 被重新证伪。4-wave WL occ=1 是已验证的结构性天花板。
- ❌ **BM=128 死路**：per-tensor 4096² 也用 BM256，BM128 更慢。

## 补充（不同项目/内核，勿与上文 dense WL 混淆）：分组 wgrad var-K whole-loop（源自 gpt_oss2 mxfp8-grouped-gg-devloop 优化6，非本环境 dense WL）
- 以下数据来自**分组（grouped）GEMM 的 wgrad var-K** 4-wave bare-asm whole-loop 实验（`_build_grouped_mxfp8_wgrad_wl_kernel`，代码已于 2026-07-05 删除、从未提交），是与本卡 dense mxfp8 fwd WL **不同的内核/不同的 GEMM pass**，详见 `48-mxfp8-grouped-wgrad-vark.md`。
- WL-unscaled（无 scale 计算地板）**只在 K∈{4096,2048} 3 个 shape 快 5-7%**；**K=7168 / 2880×2880 反慢 51-59%**。
- 反常：**4096×7168（FLOP 更少）WL=1293 比 8192×4096 WL=911 还慢**；baseline 行为正常（857<946）。
- rocprof 定位 4096×7168 vs 8192×4096：每 tile **逐字节相同、无 WG 级失衡**，但 **MfmaUtil 29.9 vs 54.8 / VALUBusy 5.8 vs 10.6 全线砍半、MemStall≈0** → 气泡在 barrier/依赖：per-phase `s_barrier` + `vmcnt(0)/lgkmcnt(0)` drain 在 **tile-grid 16×28（N=28 非 2 次幂）** 触发调度病态。
- 裁决（分组 wgrad var-K WL）：**WL 放弃**，scale 投递税 > 那 5-7% 余量，occ=1 换结构无用。

## WL 计时假象与 store 选择
- ❌ **端到端计时假象**：把每 call `broadcast_to_wl_a`（activation scale → lane-contig 重排）算进去 → **假象 0.83**（比 per-K 还慢）；重排不该计入 kernel 时间，**broadcast 不计时才是 0.97**。
- ❌ **WL 用 buffer_store 掉 7-9%（0.95→0.91）**：WL(occ=1/4-wave/寄存器紧)**只认 copy-atom store**；per-K(occ=2 有余量)两者持平；**SGPR-pin 救不了**（不是 waterfall，是紧预算下 copy-atom 就是更优）。

## WLPAD：只证瓶颈，破坏正确性
- ❌ **WLPAD=16 给 +10.6% 但破坏正确性**（pad 不兼容协作式连续 G2S）。仅用来证实 identity LDS 的 **16-way bank 冲突（row stride 128B = bank period）** 是瓶颈；**正确修法是 swizzle**，不是 pad。

## C++ 后端 raw E8M0 崩 + int32 scale 不可填充
- `test_gemm_fp8_mx_blockwise` 失败根因：quant 吐 **FlyDSL-preshuffled int32 scale**，FlyDSL 处理不了的 case（E5M2/HYBRID、K<256、K%128≠0、N%64≠0、fp16 out）fallback 到 C++ 后端要 **E8M0 → 崩**。
- HIPBLASLT 其实支持 **raw E8M0 MX**；放松 `can_handle` 后 FlyDSL 抢走 unaligned **必须自己全包**，否则 `preshuffle_ab_flydsl` 的 **N%64 崩（回归 180→252）**。
- K-tail：op preshuffle fast-path **必须 gate 到 K/M/N 全对齐（int32 scale 不可填充）** → unaligned 走 **raw 路径**；`execute` 把 **K 零填充到 tile**（padded operand=0 贡献0，**scale pad 127=1.0**）。

## vendored 严禁（用户裁定）
- ❌ **生产环境严禁 vendored dump / `mx_wholeloop` 文件夹存在**。
- WL 的 0.97 精髓 = `call_mxfp4_wholeloop` **2133 行手调 bare-asm 硬件循环（engine.py 914-3046）+ 2500+ 行依赖**，**没有小改能搬进 kernel 的版本** → 最终整个删除，**clean per-K ~0.91**。

---
来源: project_mxfp8_wholeloop_port.md（dense WL，含 8-wave 撤回说明见该文件 line 27/77）, mxfp8-8wave-devloop/SKILL.md, mxfp8-grouped-gg-devloop/SKILL.md 优化6（分组 wgrad var-K WL，均为 gpt_oss2 环境）；分组 wgrad var-K 详见 48-mxfp8-grouped-wgrad-vark.md
