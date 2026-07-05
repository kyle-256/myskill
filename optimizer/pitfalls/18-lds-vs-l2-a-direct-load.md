# 死路：A/B 从 LDS 改 direct global load，长 K 掉 2.5×

> 类别: 踩过的坑 · 主题标签: LDS-vs-L2, occupancy, direct-load, mxfp4-8wave

- ❌ 别再试：把 A operand 从 LDS 改为 direct global `buffer_load`→VGPR（A-direct）。
  - 机制**正确**：LDS 流量砍到 1/3，消除 8-wave 对 A 的重复读。
  - 但把**低延迟 LDS 读换成高延迟 L2 读**，而 1 wg/CU 藏不住 L2 延迟 → 长 K（8192² K28672）**慢 2.5×：1990 vs intrinsic 5033 TF**。
  - 等价复现历史 X 变体失败模式。
  - raw-AGPR 腾 VGPR 也**救不了**：瓶颈是 **occupancy 不是 VGPR 数**。

- ❌ 别再试：B 操作数直载到 VGPR 跳过 LDS → **慢 4×**。
  - 原因：相邻 lane 地址差 **K 步长** → uncoalesced。
  - B 走 LDS 的**唯一目的**就是把 gather 变 coalesce。

- ❌ 别再试（实测推翻旧误判）：把 8-wave mxfp4 GEMM 的 occupancy 当 LDS-bound。真相是 **REGISTER-bound**。
  - 128 VGPR + 128 AGPR = 256 共享 512 寄存器文件 → 硬卡 **2 waves/SIMD = 1 wg/CU**。
  - 把 LDS 从 128KB 砍到 64KB（A 直读）occupancy **完全不变（1.98→1.99）**，证明 LDS 不是限制。
  - 要 2 wg/CU 需总寄存器 ≤128，但**累加器单独就 128 AGPR，不可能**。

---
来源: 08-deadends.md, agpr_phase5_lds.md
