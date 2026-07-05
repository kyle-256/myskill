# Epilogue 存策略与转置/寻址:CShuffle、permlane16_swap、DS_READ_TR、i64 SRD rebase

> 类别: 方法论 · 主题标签: gfx950, transpose-load, LDS, inline-asm, epilogue-store, cshuffle, permlane16_swap, preshuffle, i64-addressing, SRD-rebase, readfirstlane, buffer-load

## gfx950 MFMA transpose load:DS_READ_B64_TR_*、intrinsic 保守 drain vs inline-asm

### DS_READ_B64_TR_* 指令族(gfx950 专属)
边读 LDS 边转置,免掉 ds_write + 置换地址的 ds_read,直接给 MFMA 准备转置布局的 A/B 操作数。

| 指令 | 数据类型 | 写出 VGPR |
|---|---|---|
| DS_READ_B64_TR_B16 | fp16 / bf16 | 2 VGPR |
| DS_READ_B64_TR_B8 | fp8 / bf8 | 2 VGPR |
| DS_READ_B64_TR_B4 | int4 | 2 VGPR |
| DS_READ_B96_TR_B6 | 6-bit | 3 VGPR |

- 硬性要求:EXEC 全 1;地址按数据大小对齐;≥64-bit 的 DS 访问要求偶对齐 VGPR(B96_TR_B6 例外)。

### intrinsic 保守 drain vs inline-asm(big-K 场景)
- 场景:big-K(长 K-loop、square-ish、L2 已经好)。此时瓶颈 = A-side tr8 的 `s_waitcnt vmcnt(0)` 全排空 —— LLVM 对 intrinsic 版 ds_read 插入保守的 vmcnt(0),把所有 in-flight vmem 全排空。
- 根因:SIInsertWaitcnts 看到 intrinsic ds_read 会保守自动 drain。
- WIN = both-path-J inline-asm drain removal:
  - `a_inline_asm=1` + `b_inline_asm=1`,用 `ds_read_b64_tr_b8` 的 inline-asm 版 → 对 SIInsertWaitcnts 是 opaque,不触发自动 drain。
  - 配套:`asm_mma=2` + `agpr_alloc=0` + `vmcnt_hint=3`。

## Epilogue 存策略:CShuffle LDS 重排 vs permlane16_swap 免 LDS 转置宽存

### 根因:MFMA 输出布局 row-strided → naive store 非合并
- MFMA-native lane layout 是 row-strided:单 lane 拥 4 连续行同列,相邻列由相邻 lane 持有。naive store 逐列走 → uncoalesced。
- 要宽存(`buffer_store_dwordx4`,8 bf16/lane)必须做 lane→col 真转置。

### 两条 epilogue 路线

| 策略 | 机制 | 开销 | 适用 |
|---|---|---|---|
| Direct Store(默认) | 每线程 MFMA 累加器直写 global | 无额外 LDS;某些 tile 非合并写 | 小 tile_n |
| CShuffle | 累加器 row-major 写入 LDS `[tile_m,tile_n]` tile → barrier → 重映射 threads 到 `(MLane,NLane)`(256 线程=8×32)再读,使一行内 lane 持连续列 → 合并成 128-B 事务(`buffer_store_dwordx2`) | 额外 LDS + 2 barrier | `tile_n≥128` |
| permlane16_swap | 寄存器跨 lane 重排完成转置(无 LDS、无 barrier),免费 VALU;转置后 `buffer_store_dwordx4`(16 store/tile,发射量 1/16) | 仅 VALU | store-bound 胖形状最优 |

### CShuffle 细节
- `e_vec = 4 if tile_n%128==0 else 2`。
- 只在 `tile_n≥128` 值:小 tile_n 时 direct store 胜,两个 barrier 纯属浪费。
- shuffle 数学假设 default MMA row mapping;换 atom 需重写 row iterator + `(MLane,NLane)`。

### permlane16_swap 实战(mxfp4)
- 正确的 permlane16_swap 转置 + dwordx4 是 store-bound 胖形状最优 epilogue:8192²×4096 从 fly/ait `1.039 → 0.998`。
- ⚠️ **转对轴才有效**：`pitfalls/25` 记录过"转错轴"的 permlane 判负(不降 store 事务数)。区别只在转的是不是 MFMA 输出的正确轴——对轴=最优,错轴=白做。
- inline-asm **不需要写死 `s_nop 1`**:该 hazard 只针对后续 VALU 读结果,而消费者是 `buffer_store`(VMEM 读 vgpr)不触发。
- 降 nop 必须两步都做:
  - 去掉写死 s_nop + 两阶段(先全部 cvt+permlane 进独立 VGPR,再 burst 全部 store,把 permlane 与 store 拉开)→ ISA `s_nop 80→0`,bit-exact。
  - 单去 s_nop 只到 19 nop;单两阶段不去 s_nop 仍 64 nop;**两者都做才 0**。

### B 矩阵 preshuffle(配套,减 load 侧 shuffle)
- CPU 上把 `[N,K]` 预转置重排成 `[N/16, K/kpack, 4, 16, kpack_bytes]`。
- kpack:FP8/INT8 = `64//elem_bytes`(=64);BF16/FP16 = 4。
- 维度含义:`4` = 每 lane 4 dword(`buffer_load_dwordx4`),`16` = MFMA 内 16 lane。
- 收益:global load 直映 MFMA 布局免 VALU shuffle、合并访问;一次性 CPU 成本摊薄。

## >4GB 寻址:per-tile i64 SRD rebase、_readfirstlane pin SGPR 免 waterfall

- **核心技术 = per-tile i64 SRD rebase**：AMD buffer SRD 的 `num_records` 和 per-lane `voffset` 都是 32-bit,上限 4GB。>2^31 元素 / >4GB 寻址时,把每 tile/group 的巨大元素 base(`m_row*K` 等)折进 SRD 的 **i64 base**,in-tile 小偏移留 **i32**。
- **base/num_records 必须 pin SGPR**：i64 rebase 后,SRD base/num_records 必须用 `_readfirstlane_i32` pin 到 SGPR(对 i64 高/低 32-bit 分别 pin)。
  - WHY:不 pin 则 base 来自 `group_scan`,被判为 VGPR → SRD 落 VGPR → 每次 K-loop `buffer_load` 触发 readfirstlane waterfall(16 次/load)。
  - 证据:gateup **-16%** / down **-13%**。
- **i64 entry 传全 rank 张量**：entry 函数必须传全 rank 张量(`a.view(torch.int8)` 保持 2D/3D),**不能 `reshape(-1)`**。
  - WHY:shape-pack `struct 'i'` 在元素数 >2^31 时 crash。
- **i64 K-loop offset 乘法用 `arith.index(k*BLOCK_K)`**：`k` 是 `range_constexpr` 的 python int,必须用 `arith.index(k*BLOCK_K)`,**不能 `arith.index_cast(T.index, python_int)`**。
  - WHY:后者对 python int 会 crash(`_to_raw` 不接受 int,报 `'int' has no _CAPIPtr`)。
- **验证**:检查 SNR 在 `<2^31` 和 `>2^31` threshold 两侧相等(两侧 SNR 相等 = 无溢出断点)。

---
来源: lds-optimization/SKILL.md, flydsl-fp8-gemm-tuning/SKILL.md, gemm-optimization/SKILL.md, gemm/optimization-directions.md, project_mxfp4_epilogue_store.md, 05-int64-addressing.md
