# MXFP8 scale preshuffle:动机 / layout-1(A broadcast) / b-comb / 写侧 element-offset / 逐层隔离调试

> 类别: 方法论 · 主题标签: mxfp8, preshuffle, e8m0-scale, layout-1, b-comb, opsel, 隔离调试

## 为什么必须 preshuffle scale
- MXFP8 GEMM 的 `mfma_scale_f32_16x16x128_f8f6f4` 通过 **opsel 字节选择器**读 E8M0 scale,每 wave 64 lane 各需对应 scale 字节。
- 原始布局 `[DIM, K//32]` 是**散点 gather**,实测比 coalesced 慢 ~2.6×。preshuffle 把 scale 重排成 **wave/lane 连续布局**。

## Layout-1(A-scale broadcast)
- 输入 e8m0 `[DIM, K//32]` → 输出 i32 `[DIM//16, K//128, 64, n_tiles]`,`SP[grp,k,lane,s]=broadcast_u8_to_u32(scale[grp*16*n_tiles+s*16+lane%16, 4k+lane//16])`,`n_tiles=BLOCK_M//64`(BLOCK_M=256 时 =4)。
- 索引公式(Python 枚举验证 100%):
  ```
  g=kcol&3; kg=kcol>>2; grp=row//64; sub=(row%64)//16; r=row%16
  lane=g*16+r; dword=((grp*K128+kg)*64+lane)*4+sub
  ```

## B-comb(combined B-scale)
- 用于 fwd B 操作数(dim=N,contract=K)和 bwd `at` 操作数(dim=K,contract=M):
  ```
  block_n=col//256; wn=(col%256)//32
  s_bc=((col>>7)<<1)|(((col&127)&31)>>4); r_bc=col%16
  lane_bc=g_bc*16+r_bc (g_bc=row_block&3); grp_bc=block_n*4+wn
  dword=((grp_bc*K128p+kg_bc)*64+lane_bc)*4+s_bc
  ```

## 写侧用 element-offset(与读侧一致)
- `bo.create_buffer_resource(ASp, max_size=False, num_records_bytes=I32(SP_A_BYTES))` + `bo.buffer_store(bcast, rasp, dword, cache_modifier=1)`,**element offset 默认非字节**,与 GEMM 侧 `ScaleS2R` 读侧完全一致(读写口径不一致会静默错位)。

## 逐层隔离调试 SOP(数值错误找最小复现)
- Step1 Python 公式验证(mydword 枚举 match=2048/2048)
- Step2 GPU 地址路径(把 dword index 当 value 写出,address-correct=1.0)
- Step3 scalar 值(e8 byte vs C++ plain scale)
- Step4 bcast 全程 Int32 路径验证
- Step5 行隔离(只跑 col-half=99.69%)
- Step6 两半并存干扰测试
- Step7 完整 kernel 仍错则**必是 flydsl 缓存或 Python 字节码旧版**(`rm -rf cache` + `rsync --checksum`)。

## FlyDSL 关键 API 速查(preshuffle/quant 写码用)
- `buffer_store(word, rsrc, byte_offset, cache_modifier=1, offset_is_bytes=True)`;`create_buffer_resource(tensor, max_size=False, num_records_bytes=I32(nbytes))`(element-offset 默认);`make_row_band_resource(bo.extract_base_index(T), base_row, c_rows, c_cols, elem_bytes)`(2D int64 rebasing);LDS strided read `make_view(base, make_layout(32,PAD)).load()` 正确生成 32 次 `ds_read_b16`;barrier `rocdl.s_barrier()`。

---
来源: mxfp8-8wave-devloop/SKILL.md
