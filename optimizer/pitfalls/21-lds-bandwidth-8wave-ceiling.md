# 8-wave mxfp4 结构封顶 ~4690-4760T：三道墙皆因 2 waves/SIMD

> 类别: 踩过的坑 · 主题标签: 8-wave, LDS-bandwidth, ds_read, register-ceiling

- **结构性封顶**：skill09（2026-06-24）初测给出 **~4900T**（pipe 配置 med/min=4817/4855）；skill10（2026-06-25，PMC+调度实验后）最终裁定修正为 **~4690-4760T**（baseline min/med=4744/4690）。两者是同一课题先后两次迭代的数字，以 skill10 的最终裁定为准。三道墙全部根因 = **2 waves/SIMD**（8-wave = 2 waves/SIMD → 每 wave 硬顶 512/2 = 256 寄存器）。
  - **墙①：LDS 读 A operand 4× 冗余**。8-wave 的 2×4 A frag 被 4 个 N-wave 各读一遍，ds_read/flop = 0.0234 vs 4-wave 0.0156。
  - **墙②：LDS 160KB 装不下大 tile 双缓**。BN512 BK256 = 192KB > 160KB，无法 double-buffer。
  - **墙③：寄存器 256@occ2 装不下 64 accs**。要 128×128 方形 tile 需 256 AGPR 累加器，已占满 256，operand/预取 0 空间。给一个 wave 428 寄存器（172V+256A）的唯一办法 = 降到 1 wave/SIMD = 4 waves/wg = 就是 4-wave kernel 本身。

- **关键修正：瓶颈不是 LDS 带宽 bound**。PMC 推翻带宽假设，LDS 端口 ≥5× 余量。真正瓶颈 = **ds_read 延迟气泡 + MFMA 执行 bound**：occ=2 下第 2 个 wave 用 wave-switching 已尽量盖住 30 读/iter 的深 ds_read 延迟链。
  - ❌ 别再试 SC_VGPR 方向：去掉 20% LDS 访问，对有 5× 余量的端口毫无意义（针对的是不存在的带宽墙）。

- **ds_read 已在理论下限，无冗余可消**（实测推翻"重读冗余"假设）：
  - vraw 8-wave 主循环 A/B frag 已在 Python 层跨 quadrant 完全复用（a0→c00/c01，a1→c10/c11；b0→c00/c10，b1→c01/c11）。
  - ds_read 已在理论下限 **24 b128/iter（A16+B8）**，与 intrinsic 完全相同（均 192@K2048）。
  - 全部 ds_read_b128 已是最大单指令宽度：同 tile s=0/1 隔 64B、tile 间隔 2048B 不连续，无法更宽合并。
  - 所谓"N 子块重读 A 的 1.5× 冗余"**不存在**；phase-5a 看到的 A:B LDS insts = 2:1 是 tile 大小正当读量比，非重复读。
  - phase-4/5 在 8-wave 内追的 0.3~1.5% 残差是"8-wave 局部最优"内部的事，非结构杠杆。

- ❌ **别再试：8-wave 原生 occ=2 藏 store**（换 occ=2 藏 store-bound 形状）。实测全负：28672 −14%、6144³ −20%、8192²×4096 −20%。机制：occ=2 能藏 7-15% store，但 8-wave compute 赤字 14-20%（per-warp tile 减半 → B 复用减半 → ds_read/mfma 翻倍）远大于收益。与 4w+BK128 −13% 同结论。

- **>5200 必须走 4-wave**：occ=1，VGPR 512 能做 register double-buffer，已达 5351。4-wave 是结构性更优工作点（长 K），8-wave 结构上无法达到 4-wave 的长 K 性能。

---
来源: 09-8wave-ceiling.md, 10-8wave-scvgpr.md, agpr_phase5_ldsr.md, diag_4w_vs_8w.md, project_mxfp4_epilogue_store.md
