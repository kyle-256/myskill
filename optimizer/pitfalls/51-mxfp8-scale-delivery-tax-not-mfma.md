# MXFP8 残余 gap 是 scale 投递税不是 scaled-MFMA 指令税（≈0）

> 类别: 踩过的坑 · 主题标签: mxfp8, scaled-mfma, scale-delivery, whole-loop

核心诊断卡：优化方向是**隐藏 scale load / 减少分散 scale VGPR**，而不是减 MFMA。

## scaled-MFMA 指令本身税 ≈ 0%（此前 5% 结论是错的）
- 探针 NOSCV / NOSCALE_MMA / NOSCMMA 实测 scaled-MFMA 指令本身税 ≈ 0%：
  - dense Down：scaled 3068 vs 非-scaled 3093 = 0.8%
  - grouped wgrad：807 → 807（去掉 scaled-MFMA 不变）
- ❌ 别再试：把残余 5% 归因为「scaled 税」——结论错。真凶=scale VMEM load 未隐藏（big-K 跳 scale load 后达 0.97-0.99）。
- WHY：scaled-MFMA 只是多吃一个 scale 操作数，指令自身无额外 issue 代价；成本全在把 scale 字节从 VMEM 搬进 VGPR。

## grouped wgrad WL scaled 1.7x 慢的根因（占用率结论作废）
- grouped mxfp8 wgrad WL：scaled 811us（1.7x 慢，另记 817）vs unscaled 445us（反超 baseline 479）。
- ❌ 别再试：把它归因为 occ——4-wave scaled 计算天花板 447 ≈ unscaled，occ 结论作废。真凶=scale 投递机器。
- 两个叠加大头：
  - ① 每 phase **16 条 buffer_load_ubyte**（tiny byte-gather，吞吐/issue-bound；预取隐藏无用——去 load 才 807→448）。WL 全量记为 32 byte-load/iter。
  - ② **16 个分散 scale 目标 VGPR**（SCMIN 砍到 2 就 807→506）。
- 隔离探针：NOSCLOAD=448、SCMIN(16→2 VGPR)=506、全速地板 447（另记 448）。
- scale gap 随 k_iters **线性**（M=1024→1.5x、4096→1.9x、8192→1.94x）→ 是 per-K-iter 开销。

## scale_pack / opsel byte-pack：前提是「裸无预取」，生产不适用
- ❌ 别再试（生产 per-K）：GEMM K-loop 里砍 scale LOAD 数（scale_pack / §9.4 opsel）——探针只加载 k=0 scale 复用测天花板，scale-tax ≈ 0%（4 shape 全 ≤1% 甚至略慢）。
- WHY：现有流水已提前一拍预取 sa/sb（sa0n + s_setprio）把 scale i32 load 藏进 MFMA shadow。§9.4 的 scale_pack=4/opsel 前提是**没预取的裸状态**，本 codebase 不适用。
- WL(4-wave/occ=1) 上 scale_pack 曾被证伪（寄存器压力↑→spill），但那是 WL 的问题；WL 已整体放弃（手调 2500 行汇编不可维护），per-K(occ=2) 才是生产路径。
- pack2（2 个 E8M0 字节合成 1 次 buffer_load_ushort，op_sel:[0,0,0]/[1,1,0] 选低/高字节，载入 32→16/iter）在 WL 吃回 54%（817→620）；但生产 per-K 已被预取藏住，无用。

## scale-128（dwordx4）无提速，WL_SC128 有索引 bug
- dense Down 上 operand:scale 字节比 = 32:1。
- scale-128（LDS-b128 dwordx4 与 VGPR-dwordx4 两条独立路径）perf-neutral：3097 vs 3099。
- 5 探针（无 g2s / 无 ds_read refill / 无两者 / 无 scale / WL_SC128）全 ≈ baseline 3053-3104 → 纯 MFMA-bound，去任何 memory 操作都不提速。
- ❌ WL_SC128 dwordx4 2-K-pack scale 实现完成但 SNR -1.3（确定性索引 bug 未调），速度 3097 vs SC_VGPR-dwordx2 3099 完全中性 → 默认保留 **SC_VGPR dwordx2**（正确同速）。

---
来源: mxfp8-8wave-devloop/SKILL.md, mxfp8-grouped-gg-devloop/SKILL.md, project_mxfp8_wholeloop_port.md, project_mxfp8_grouped_wgrad_wl.md
