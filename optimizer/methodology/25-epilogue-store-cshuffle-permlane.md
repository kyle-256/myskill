# Epilogue 存策略:CShuffle LDS 重排 vs permlane16_swap 免 LDS 转置宽存

> 类别: 方法论 · 主题标签: epilogue-store, cshuffle, permlane16_swap, preshuffle

## 根因:MFMA 输出布局 row-strided → naive store 非合并
- MFMA-native lane layout 是 row-strided:单 lane 拥 4 连续行同列,相邻列由相邻 lane 持有。naive store 逐列走 → uncoalesced。
- 要宽存(`buffer_store_dwordx4`,8 bf16/lane)必须做 lane→col 真转置。

## 两条 epilogue 路线

| 策略 | 机制 | 开销 | 适用 |
|---|---|---|---|
| Direct Store(默认) | 每线程 MFMA 累加器直写 global | 无额外 LDS;某些 tile 非合并写 | 小 tile_n |
| CShuffle | 累加器 row-major 写入 LDS `[tile_m,tile_n]` tile → barrier → 重映射 threads 到 `(MLane,NLane)`(256 线程=8×32)再读,使一行内 lane 持连续列 → 合并成 128-B 事务(`buffer_store_dwordx2`) | 额外 LDS + 2 barrier | `tile_n≥128` |
| permlane16_swap | 寄存器跨 lane 重排完成转置(无 LDS、无 barrier),免费 VALU;转置后 `buffer_store_dwordx4`(16 store/tile,发射量 1/16) | 仅 VALU | store-bound 胖形状最优 |

## CShuffle 细节
- `e_vec = 4 if tile_n%128==0 else 2`。
- 只在 `tile_n≥128` 值:小 tile_n 时 direct store 胜,两个 barrier 纯属浪费。
- shuffle 数学假设 default MMA row mapping;换 atom 需重写 row iterator + `(MLane,NLane)`。

## permlane16_swap 实战(mxfp4)
- 正确的 permlane16_swap 转置 + dwordx4 是 store-bound 胖形状最优 epilogue:8192²×4096 从 fly/ait `1.039 → 0.998`。
- ⚠️ **转对轴才有效**：`pitfalls/25` 记录过"转错轴"的 permlane 判负(不降 store 事务数)。区别只在转的是不是 MFMA 输出的正确轴——对轴=最优,错轴=白做。
- inline-asm **不需要写死 `s_nop 1`**:该 hazard 只针对后续 VALU 读结果,而消费者是 `buffer_store`(VMEM 读 vgpr)不触发。
- 降 nop 必须两步都做:
  - 去掉写死 s_nop + 两阶段(先全部 cvt+permlane 进独立 VGPR,再 burst 全部 store,把 permlane 与 store 拉开)→ ISA `s_nop 80→0`,bit-exact。
  - 单去 s_nop 只到 19 nop;单两阶段不去 s_nop 仍 64 nop;**两者都做才 0**。

## B 矩阵 preshuffle(配套,减 load 侧 shuffle)
- CPU 上把 `[N,K]` 预转置重排成 `[N/16, K/kpack, 4, 16, kpack_bytes]`。
- kpack:FP8/INT8 = `64//elem_bytes`(=64);BF16/FP16 = 4。
- 维度含义:`4` = 每 lane 4 dword(`buffer_load_dwordx4`),`16` = MFMA 内 16 lane。
- 收益:global load 直映 MFMA 布局免 VALU shuffle、合并访问;一次性 CPU 成本摊薄。

---
来源: gemm-optimization/SKILL.md, gemm/optimization-directions.md, project_mxfp4_epilogue_store.md
