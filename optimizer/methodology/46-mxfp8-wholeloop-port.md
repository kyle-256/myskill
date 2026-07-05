# MXFP8 whole-loop 从 fp4 移植:MFMA 格式切换 / fp8 operand 读 / swizzle hi-base / store 统一 / WL 裁决

> 类别: 方法论 · 主题标签: mxfp8, whole-loop, MFMA-format, fp8-operand, swizzle, store统一, asm_mma, WL-swizzle

## 合理对标(先定标尺)
- mxfp8 scaled-MFMA 对标口径：见 methodology/50-mxfp8-grouped-fair-compare-occ-ceiling.md「MX vs TW 唯一公平口径」。

## MFMA 格式切换(fp4 → fp8)
- asm 从 fp4 的 `cbsz:4 blgp:4` → **`cbsz:0 blgp:0`**(E4M3)。
- E5M2/HYBRID 时 `cbsz/blgp` 按 operand format(0=E4M3, 1=E5M2),**scale 路径不变**。

## fp8 operand 读格式(NT 布局)
- fp8 **不用 tr_b8**;`S2RLoader`(gemm_helper.py:165)= 2×16B 普通读(swizzle_128, `col=(lane//16)*16+step*64`)→ `pack_i32x4_i32x8` → i32x8;fp4=1×16B+零填充→i32x8。
- 寄存器格式相同,差异 = fp8 每 tile operand 是 **2× `ds_read_b128`**(fp4 1×),**VGPR 不翻倍**。
- swizzle hi-base 技巧:fp8 2×b128 高半区(+64)非线性难点解法 = `hi-base=S2RLoaderFp4.base_addr(s=1)`(s*64 在 s=1 正好给 col `g*16+64` 且已正确 swizzle),两条 b128 都用 `offset=ii*ts`;引擎加 `a/bl/br_base_hi` 输入(仅 `_FP8`)。

## scale 路径 fp4 ↔ fp8 完全复用
- `sc_*` 参数、`ScaleS2RPacked`、`preshuffle_scale_lane_contig`、`grouped_xcd_pid` 原样复用(scale_pack + 大K预取见 45 卡)。

## store 统一(557→164 行,commit 34dc6aee,已被后续 commit 取代)
- `StoreCPerTensor` 加 `A/B_scale=None`(不缩放,mxfp8 scale 已折进 MMA)+ copy_atom 模式,让 per-tensor(scale+buffer_store)/ grouped / per-K mxfp8(scale=None+buffer_store)/ WL(scale=None+copy_atom)**全部共用一个类**。
- `buffer_store(mask=)` 内部 = `select(mask, off, 0x7FFFFFFF)`,和 copy-atom OOB-select 同机制,**mask 本身不是问题**;`make_row_band_resource` 补 SGPR-pin + `0x7FFFFFFF` cap 与内联字节级一致。
- **后续(commit ab7ad885 取代 34dc6aee):WL 连同 mx_wholeloop 文件夹被整体移出生产,copy_atom 分支随之删除**——kernel 恢复成干净 per-K + 共用 `StoreCPerTensor`(scale-optional,去掉 WL 专用的 copy_atom)。当前生产 store 无 WL/copy_atom 模式,只服务 per-tensor/grouped/per-K 三路(均 scale+buffer_store 或 scale=None+buffer_store)。

## WL 裁决(per-shape win 非全面 win,该结论已被后续废弃 WL 取代)
- `asm_mma`(bare-asm scaled MFMA)= +0.3~1.5% 小正收益,`MXFP8_ASM_MMA=1` 可开,低风险默认候选。
- dense WL+swizzle 是 **per-shape win**:`SWIZ=1` 70B Down 3075(1.03× per-K, 0.94× pt)、GateUp 2815(1.00× perK)、O_proj 2800(0.93×);小 shape 0.85-0.98×(prologue/grid 主导)。集成方向 = autotune 在 **WL vs per-K 按 shape 选**。
- 诊断旋钮(dense):`WL_NOSCALE_MMA`(emit 非-scaled v_mfma)、`WL_NOSCV`(跳 scale load)、`FP4_WLNOG2S`、`FP4_WLNODSR`、`WL_SC128`(scale 减半)。
- 诊断旋钮(grouped wgrad,**已从工作树删除,2026-07-05,从未 commit、git 历史里也没有,当前代码不可用**):`PT_MXGG_WGRAD_WL_{UNSCALED/PACK2/NOSCMMA/NOSCLOAD/SCMIN}`。生产 grouped wgrad 现仅 `_build_grouped_mxfp8_wgrad_kernel`(无 WL 路径)。
- **后续裁决(commit ab7ad885):用户最终裁定 `mx_wholeloop` 文件夹严禁存在,WL 从 dense 生产中整体移除**(手调 2133 行 bare-asm 硬件循环无法"小改"搬进 mxfp8_gemm_kernel.py)。**每 shape ≥97% per-tensor 的目标最终靠 WL-free 的纯 per-K + scale_pack(opsel byte-pack)+ 大-K scale 预取达成**,已提交 `dac31090`(分支 `dev/kyle/flydsl_mxfp8_compute`)。WL 代码留档于 gpt_oss2 环境的 `.wl_study/`(仓库外)+ git 历史 fc45f4bb,若将来重做只能走真正 fp8-only clean rewrite。

## C++ quant 配套修 3 处
- `quantize_mxfp8_dual_meta` 的 preshuffle int32 buffer sizing;`preshuffle_n_tiles≥1` 防 `%0`;padded K-block scale byte 恢复 **0**(非 unit/bias)。

---
来源: project_mxfp8_wholeloop_port.md, project_mxfp8_grouped_wgrad_wl.md(均源自 gpt_oss2 环境 `.claude/memory/`,非本 gpt_oss 环境;WL 相关结论已在两文中被后续 commit 标注为废弃/删除,详见正文各节说明)
