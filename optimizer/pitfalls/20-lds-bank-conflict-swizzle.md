# LDS bank 冲突/swizzle：gfx942 32 banks vs gfx950 64 banks，mask 需重推

> 类别: 踩过的坑 · 主题标签: LDS-swizzle, bank-conflict, gfx950-vs-gfx942, ping-pong

## bank 数按代不同，mask 必须重推
- **gfx942 = 32 banks，512 B 分配块**；**gfx950 = 64 banks，1280 B 分配块**（1280 B 对齐，no wrap）。
- bank 冲突 stride 随 bank 数变：**gfx942(32) stride=128 字节全冲突；gfx950(64) 同样 stride=128 字节只 2-way 冲突**（线程在 2 bank 间交替），要 **256 字节倍数(64*4)** 才全 64-way 冲突。
- WHY：为 32-bank 设计的 XOR swizzle mask 直接搬到 64-bank gfx950 会**留残留 2-way 冲突**，mask 需调宽。
- footprint 陷阱：原本在 512 B 上整除干净的 footprint，在 1280 B 上可能浪费整个块、掉一个 occupancy tier。移植时 swizzle/padding mask 都要重新推导。
- A 型 bank 冲突判据：ds_read/ds_write 自身 stall>100 cycle/hit，read2_b64/write2 的 offset 为 bank 数倍数（gfx942=32 / gfx950=64）。

## swizzle 写读路径必须完全一致
- **每个 LDS write 和每个 LDS read 用完全相同的 XOR swizzle**（XOR 是自逆）。
- ❌ 别再试 不对称 swizzle：写读 mask 不一致会**静默读错行，编译器零信号**——这是最常见的 LDS 数据损坏来源。
- swizzle vs padding 权衡：swizzle 零 LDS 开销但 mask 依赖架构、错了静默冲突；padding(stride+1)简单、无额外寄存器但吃 LDS、超限 kernel fail。**优先 swizzle**；LDS 有余量或 swizzle 难集成时用 padding；gfx950 160KB 给 padding 更多余量。

## LDS 写后读必须同步
- 任何依赖前 ds_write 的 ds_read 之间必须有 **s_waitcnt lgkmcnt(0) 或 s_barrier**。
- 若 **wave A 写、wave B 读，则必须 s_barrier（仅 lgkmcnt 不够）**。
- 写-读距离越长延迟隐藏越好。
- B 型 write-read 延迟暴露判据：ds_write 后紧跟 s_waitcnt lgkmcnt(0) 且 stall>2000，二者间指令太少。
- C 型 跨 wave reduce 串行化：ds_bpermute→lgkmcnt→s_barrier→ds_write→lgkmcnt→s_barrier→ds_read 链，reduce 区 barrier>4。

## ping-pong 每轮必 rotate buffer index
- LDS ping-pong 每次迭代必须旋转 buffer index（以及任何 parity flag / partner register buffer）；**漏一个 swap 下一轮读到 stale LDS，编译期不可见**。
- K loop 要**按对(PAIRS)展开**，使 `write_stage = read_stage ^ 1` 交替对齐；LDS write 之后**恰好保留一个 gpu.barrier()**。

## 无冲突时消冲突无意义（死坑）
- ❌ 别再试 实测本就无冲突时做 padding 消 bank conflict：SQ_LDS_BANK_CONFLICT=0 / ADDR_CONFLICT=0 / UNALIGNED_STALL=0 时消冲突毫无意义。
- 8-wave **SWZ1 已把 bank 冲突清零**：SWZ0=0.75 ratio、MfmaUtil 44% vs **SWZ1 MfmaUtil 60-62%**。
- ❌ 别再试 调度层杠杆：WLDSR 细 staggered lgkmcnt / SS sub-stream / INPLACE 全部 ≤baseline；拆细单一粗同步点只会约束 wave-switching 自由度、暴露更多 stall。

---
来源: 10-8wave-scvgpr.md, lds-optimization/SKILL.md, gfx950/kernel-implementation-notes.md, programming-model.md, optimization-directions.md
