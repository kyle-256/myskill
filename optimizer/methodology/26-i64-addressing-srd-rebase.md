# >4GB 寻址:per-tile i64 SRD rebase、_readfirstlane pin SGPR 免 waterfall

> 类别: 方法论 · 主题标签: i64-addressing, SRD-rebase, readfirstlane, buffer-load

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
来源: 05-int64-addressing.md
