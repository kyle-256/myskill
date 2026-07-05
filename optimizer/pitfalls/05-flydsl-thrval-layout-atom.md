# FlyDSL ThrVal/atom layout 静默错：#1 静默产错结果 bug 源

> 类别: 踩过的坑 · 主题标签: ThrVal-layout, atom-op, MmaOp/CopyOp, 静默错

- **#1 静默产错结果源**：错误的 ThrVal/ThrBit layout 是 FlyDSL 头号静默错 bug。编译器接受、内核正常运行、输出垃圾，**无任何运行时诊断**。定位极难，因为没有报错。
- **顶层 shape 必须 rank-2 `((thr...),(val...))`**：mode0=线程轴、mode1=值轴。`TiledOpUtils.h` 无条件做 `shape.at(0)`/`shape.at(1)` 切 thr/val。多套一层嵌套或压成 rank-1 会编译通过但静默产出错误 tile 划分。

- **MmaOp ThrVal 乘积不变式**（违反仍编译，结果 UB）：
  - `|thr|` == 协作线程数：AMD wave64 MFMA = **64**，wave32 WMMA = **32**，NVIDIA WGMMA = **128**。
  - A: `|thr|*|val| == M*K`；B: `|thr|*|val| == N*K`；C/D: `|thr|*|val| == M*N`。
  - 各 ThrValLayout 的 `|thr|` 要和 `getThrLayout` 一致；`|val|` 要和 `emitAtomCallSSA` 里线程寄存器向量宽度一致。
  - ⚠️ **线程数不匹配最阴险**：wave64 MFMA 误注册成 FxC(32)，会照常发出指令，但一半线程在旧（陈旧）寄存器上算 → 结果错但无报错。

- **参考坐标系是列主序（非行主序）**：写 ThrValLayout 时必须按列主序算 stride：
  - MmaOp A=(M,K) 基线 stride `(1,M)`；B=(N,K) 基线 `(1,N)`；C/D=(M,N) 基线 `(1,M)`。
  - CopyOp src/dst=(M,N) 基线 `(1,M)`。

- **CopyOp：bit 粒度 vs value 粒度 recast**：
  - CopyOp 发布的是 **bit 粒度** layout（`getThrBitLayout*`，shape 恒为 `(|thr|, bitSize)`）。
  - `CopyAtomType` wrapper 经 `layoutRecast(bitLayout, oldBits=1, newBits=valBits)` 算出 value-layout。
  - 例：32b buffer copy 写 `FxShape(FxC(1), FxC(32))`，valBits=32 recast 成 1 个 f32/线程；**同一 Op 复用**于 16b f16 pair 时自动得 2 个 f16/线程。
  - ❌ 别再试：让 bitSize 不能整除 valBits（如 96b/64b = 1.5）。`layoutRecast` 会**静默产垃圾**，无报错，程序员必须自己发现。

- **verify 必须拒绝非法元组**：新 MmaOp/CopyOp 的 verify（`genVerifyDecl=1`）必须拒绝不支持的 `(m,n,k,elemTy)` 元组并给清晰 `emitError`。否则非法配置会静默命中 `emitAtomCallSSA` 的 `return failure()`，且无诊断信息 → 又一条静默错路径。

---
来源: add-target-atom-op/SKILL.md
