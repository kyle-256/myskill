# occupancy 是 register-bound 非 LDS-bound：512 合并 VGPR 池模型

> 类别: 踩过的坑 · 主题标签: occupancy, register-pressure, VGPR, mxfp4

## 核心模型：gfx942/gfx950 是 512 合并池，不是 256/max

- **占用率公式**：`occupancy = min(512 // (arch_vgpr + accum_vgpr), lds_limit, 800 // sgpr, 8)`。VGPR 占用率由 **arch_vgpr + accum_vgpr 的合并 512-entry/SIMD 预算**共同决定（两物理文件共享一个占用预算，每个各自 cap 256）。
- **绝对不是** `256 / max(arch_vgpr, accum_vgpr)`——那只适用于 gfx908/CDNA1。CDNA2/gfx90a 起把 256 arch + 256 accum 合并成单一 512 预算/SIMD，gfx942/gfx950 继承。
- **WHY 误判危险**：把它当两个独立 256 池计算，在 AccVGPR 用量重时会**高估** occupancy（例：128V+128A 按两独立池看似各占一半仍宽裕，实则 256/512 已锁死 2 wave）。
- gfx942 waves/SIMD 阶梯（`512 // (arch+accum)`）：≤128→4wave、≤170→3wave、≤256→2wave、≤512→1wave、**>512 SPILL 严重回退**。
  - 例：arch=148/accum=148→296→1wave；再加 32 arch→328 仍 1wave；要 2wave 需**合并** ≤256。
- gfx942 分级（另一维度视角）：≤128/≤128 得 2 wave（好）；129-256/≤256 得 1 wave（compute-bound 可接受）；>256 SPILL（严重回退）。
- **AccVGPR 不与 arch 竞争分配**：MFMA 累加器用 accum_vgpr（独立文件），预取缓冲/B tile/A tile 用 arch_vgpr，二者不互相竞争寄存器**分配**——但**共享占用预算**。
- **LDS 地址逻辑也吃占用**：LDS 寻址逻辑增长 arch_vgpr，即便不碰 MFMA 累加器也会吃 occupancy；kernel 靠近 2-wave 边界时要压低 LDS 地址 VGPR 压力。
- **每 buffer_load_dwordx4 = 4 arch_vgpr**；双缓冲净增约一组 'next' 缓冲。
- gfx950 8-wave WG（512 线程，2 waves/SIMD）硬上限 V+A ≤ 256 dword/lane（每 SIMD 16384 dword）；occ=1 时 512-VGPR 满载锁死 prefetch 深度。

## ❌ 别再试：砍 LDS 提 8-wave mxfp4 occupancy

- **8-wave mxfp4 GEMM 的 occupancy 是 REGISTER-bound 不是 LDS-bound**（实测推翻）。
- 128 VGPR + 128 AGPR = 256 共享 512 文件 → 硬卡 2 waves/SIMD = 1 wg/CU。
- 把 LDS 从 128KB 砍到 64KB（A 直读）**occupancy 完全不变**（1.98→1.99）。
- 要 2 wg/CU 需总寄存器 ≤128，但**累加器单独就 128 AGPR，不可能**。之前把它当 LDS-bound 是误判。

## 4-wave 真瓶颈：occ=1 单 wave 无延迟隐藏，MFMA idle ~40%

- 4-wave mxfp4 真瓶颈 = **occ=1**（1 wave/SIMD，vgpr432 + agpr256，waves_per_eu=1）→ 单 wave 无跨 wave 延迟隐藏，**MFMA idle ~40%**，不是 shuffle。
- **反证旋钮 BN128**（HAS_BR=False，32 accs=128 AGPR，2-buf，LDS≤80K）→ occ=2 反而藏 ds_read 延迟更好。
  - PMC 证据：LdsUtil 8%（non-bw-bound），大 SQ_WAIT_INST_LDS 被 1-wave/SIMD 暴露。
- **结论**：occ=1 大 tile 不是无脑赢，是 tile 强度 vs 延迟隐藏的权衡。

---
来源: 01-architecture.md, gemm-optimization/SKILL.md, lds-optimization/SKILL.md, prefetch-data-load/SKILL.md, kernel-trace-analysis/SKILL.md, programming-model.md, agpr_phase5_lds.md, project_mxfp4_vgprform_deadend.md, diag_4w_vs_8w.md
