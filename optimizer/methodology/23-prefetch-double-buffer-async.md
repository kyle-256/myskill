# LDS ping-pong 双缓冲与 async copy:2-stage、同步 vs DMA、跨阶段 load 提进 barrier

> 类别: 方法论 · 主题标签: prefetch, lds-double-buffer, async-copy, tile-size

## LDS ping-pong 双缓冲 (lds_stage=2)
- A tile 分两块独立 LDS:两个 SmemAllocator,global_sym_name = smem0 / smem1。一块跑 MFMA 时另一块加载下个 K-tile,隐藏 global→LDS 延迟。
- 每次主循环迭代处理 2 个 K-tile(pong + ping)。
- LDS 预算:`lds_tile_bytes = tile_m × tile_k × elem_bytes`。2-stage 需 `2 × lds_tile_bytes`;CShuffle epilogue 另加 `tile_m × tile_n × 2` bytes。
  - 例:64×128 FP8 = 16KB;128×128 FP8 = 32KB。
  - LDS 容量上限:见 pitfalls/19-lds-capacity-3stage-deadend。

## A 矩阵入 LDS 两条路
| 路径 | 机制 | 特点 |
|---|---|---|
| 同步 (默认) | Global→VGPR→LDS:`prefetch_a_tile`(buffer_load_dwordx4)再 `store_a_tile_to_lds`(ds_write) | 走 VGPR |
| 异步 (use_async_copy=True) | Global→LDS 直 DMA:`raw_ptr_buffer_load_lds` | 绕过 VGPR,降寄存器压力,省 arch_vgpr;gfx942/gfx950 均可 |

- async copy 适用条件:`tile_m ≥ 128`(足够 compute 藏 DMA 延迟)。小 tile_m 低寄存器压力用 sync,大 tile_m 用 async。
- granularity:
  - gfx942:sync 16B(dwordx4)/ async 4B(1 dword/DMA)。
  - gfx950:sync 16B / async 16B(4 dwords/DMA)。

## B 矩阵:preshuffle 后直 Global→VGPR
- B 预 shuffle 后直接 Global→VGPR(buffer_load_dwordx4),布局已匹配 MFMA 寄存器排布,无需 VALU shuffle。
- 每 K64 微步 B 需 `2 × num_acc_n` 个 i64(K32×2)。

## 跨阶段 load 提进 barrier-wait 停顿区
- 若某阶段耗在 s_barrier 等待(如 softmax 跨 wave reduce ~96K cycle),把下阶段需要的 global load(如 V-value ~17K cycle)提前发到 barrier 停顿区。
- WHY:barrier 反正要等,期间发 load 基本免费。

## wgrad 两种流水(按 per-group M 分流)
- masked chunked(大-M,per-group `m_total/G > 1536`):outer runtime `scf.for over ceildiv(k_iters, chunk)` × inner `range_constexpr(chunk)` 的 4-buffer 流水。over-run 由 per-group SRD `num_records` clamp 到 0(无需 host cap)。chunk=8:每 chunk 8 个 K-iter 全展开。
- persist(小-M,per-group `m_total/G <= 1536`):`_wgrad_loop_body_pipe`,2-stage prefetch(prologue prefetch K-tile 0,per-iter prefetch K+1 overlap 当前 MFMA)。短 contraction 下 masked 的 chunk over-run 是废功,persist 精确跑完自己 M_g。
  - 实测 M_g=512:856 → 1369 TF(+60%)。

---
来源: gemm-optimization/SKILL.md, prefetch-data-load/SKILL.md, 04-tn-wgrad-kernel.md
