# grouped MXFP8 quant:融合 meta prologue / batched 权重 quant / HIP 输出契约

> 类别: 方法论 · 主题标签: mxfp8, grouped, quant, meta-prologue, batched-quant, HIP契约

## 融合 meta prologue(优化4,已落地)
- 问题:grouped quant 主 kernel 每 WG 都跑一遍 O(G) 组搜索(读 GO/GR/GC 三个 int32-view offs ~40 copy-atom load,select 链 gate 住整 tile),memory-bound 短 kernel 里延迟全暴露占 ~42us。
- 正解:写融合 prologue kernel `meta`(NBM 线程/256 一 block,每线程对 `base_m` 做一次 O(G) 搜索写 RB/RO/RE[bt]),与主 kernel **背靠背在同一 `@flyc.jit` stub 里 `.launch()`**。搜索从 `512×(NBM×NBK)` 个 WG 各做一遍 → NBM 线程一次。
- 结果:kernel 155、wrapper 181,grouped-FLY **首次快过 grouped-HIP**。`RB=go_orig_g+mrel`、`RO=go_row_g+mrel`、`RE=go_col_g+M_g`。

## batched FLY 权重 quant(compile_qdual_batched,默认 ON)
- 把 dense `compile_qdual` 加 batch 维,一次 launch(`grid=B*NBM*NBK`)量化整块 `[B,N,K]` 权重全部 B 个 expert,替旧 Python 逐 group 循环(逐 group `quant_mxfp8_raw`+`torch.stack`,8× launch,比 HIP 慢 2.7-4.3×)。
- 核心:`batch=pid//NPB`,输入 band 与 4 条输出 band 各按 batch 重定位;**per-batch scale byte base**(`base_row_b/base_col_b`,dword 对齐)保证相邻 batch scale dword-packing 不互踩(`_store_scale` 加 `base_byte` 形参)。等 group 尺寸无需 per-tile 组搜索。
- 性能:快 HIP **1.3-1.54×**(275→207 / 344→239 / 152→116 / 94→61),方向同 dense-FLY ~1.6×。`PT_MXGG_FLYDSL_QUANT=0` 退回 HIP。

## HIP grouped_quantize_mxfp8_dual 输出契约(新 kernel 须 bit-兼容)
- 位置 `quantization.cpp:829`。输入 `x[total_M,N]` + `group_lens/group_offs`(int64 GPU),`ROW_ALIGN=64/COL_ALIGN=128`,`M_pad_row=cdiv(total_M+G*64,64)*64`、`M_pad_col=cdiv(total_M+G*128,128)*128`、`N_pad=cdiv(N,128)*128`。
- 返回 **8 个固定顺序**:0 `rowwise_output[M_pad_row,N_pad]` fp8、1 `rowwise_scale` e8m0、2 `colwise_output[N,M_pad_col]` fp8(已转置)、3 `colwise_scale`、4-7 `group_lens/offs_padded_rowwise(64)/colwise(128)`。
- raw 行/列主 E8M0 **不 preshuffle**(gemm 侧 in-launch preshuffle);组边界 scale 独立不跨组;**padding 行 scale 填 127(=1.0)/ 数据填 0**;padded-layout offs 由 `compute_padded_layout_gpu` 在 GPU 上算(无 D2H)。

---
来源: mxfp8-grouped-gg-devloop/SKILL.md(优化4 / batched FLY 权重 quant / HIP 输出契约)
