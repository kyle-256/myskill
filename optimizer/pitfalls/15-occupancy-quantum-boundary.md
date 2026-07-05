# 占用率量子边界：只有跨 allocation quantum 才买到一个 wave

> 类别: 踩过的坑 · 主题标签: occupancy, VGPR/SGPR/LDS 分配, quantum, silent-OOB

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

---
来源: gfx942/overview.md（85→84 VGPR 示例、CDNA3 不 fault 表述，line 47/49）+ gfx950/overview.md（LDS 量子 1280B、CDNA4 不 fault 表述，line 51）
