# MXFP8 dual-cast quant kernel:架构 / 分段计时 / 集成路径 / 数值验证

> 类别: 方法论 · 主题标签: mxfp8, dual-cast, quant-kernel, e8m0, SK-timing, 集成路径

## dual-cast quant kernel 架构(mxfp8_quant_flydsl.py,唯一生产入口 compile_qdual,即原 v5)
- 输入 `X[M,K] bf16`,一个 kernel 同时吐 4 路输出:
  - `Qr[M,K] fp8`(row cast,fwd A 操作数)
  - `ASp int32`(layout-1 preshuffle A-scale,见 43 卡)
  - `AtQd[K,M] fp8`(col cast **已转置**,bwd `at` 的 B 操作数)
  - `AtSp int32`(b-comb col scale)
- Tile `bm×bk` 约束 `bm*bk==8192`,Block **512 线程**,默认 `bm=64, bk=128`。
- 分工:前 256 线程做 **ROW cast**、后 256 线程做 **COL cast**(两半并发共享 LDS);`barrier` 后**全 512 线程重读 ldsc 合并写转置**(制胜招见 44 卡)。
- scale 元素数:`SP_A_ELEMS=(M//64)*(K//128)*64*4`;`SP_AT_ELEMS=(K//256)*4*(M//128)*64*4`。
- 输出缓冲 `torch.empty` 不需 zero-init(每个 in-range byte 恰写一次)。partial-tile 用 SRD/num_records OOB→0 + 输出 mask;**只需 M/N 是 64 的倍数**即可安全(256 倍数约束已解除)。

## e8m0 exponent 与 fp8 打包(round-even 匹配 C++ compute_tile_scale)
- exp:`ai=I32(amax.bitcast(Int32))+I32(1<<19); ep=((ai>>23)&0x1FF)-135; clamp 到 [-127,128]`。
- fp8 pack:`word=I32(rocdl.cvt_pk_fp8_f32(IRI,a,b,I32(0),0))`(bytes 0,1),再 `cvt_pk_fp8_f32(IRI,c,d,word,1)`(bytes 2,3)。
- `scale=fm.exp2(ep.to(F32))`。⚠️ Int32 无 bitcast,**不能 int→float bitcast**,必须走 `.to(F32)`。

## 分段计时用 SK env(归因用,别扫常数)
- `_SK=int(os.environ.get('SK','0'))`,各阶段 `if _SK==N: return`:SK=0 完整 / SK=1 只 row / SK=2 row+col 计算无写 / SK=3 跳 LDS 写。
- 实测:移除 barrier / 占用 / amax+exp2 **全无效** → 瓶颈是 dual-cast **DRAM 散写的固有代价**,不是计算(带宽定位见 44 卡)。

## 集成路径(算子不直接调 kernel)
- `FP8GemmMXFunction.forward/backward → quantize_fp8_with_trans → quantize_mxfp8_impl → quant_mxfp8_raw(mxfp8_quant_flydsl.py) → compile_qdual(..., **_qdual_tile_cfg(M,K,Mp,Kp))`。
- quant 只发 **raw 行主 E8M0**(`[free, contract//32]`,1 byte/block);preshuffle(layout-1 / b-comb)由各 GEMM backend 在自己 launch 里融合。public 路径若显式要 preshuffled,由 `quantize_mxfp8_impl` 末尾 post-apply 独立 `shuffle_scale` 算子。
- 缓存:`_RAW_QDUAL_CACHE[(M,K,Mp,Kp,in_dtype,out_dtype)]` 存已编译 callable。

## 数值验证 SOP(三级)
1. **self-consistent**:`Qr.view(uint8)==(x/exp2(Sr-127).repeat_interleave(32)).to(e4m3).view(uint8)` 期望 1.0(0.31% 偏差是 cvt round tie-break 正常)。
2. **vs C++**:`quantize_mxfp8_impl(...,ScalingRecipe(preshuffle_layout=1,preshuffle_n_tiles=4),ScalingRecipe(preshuffle_layout=3))`,`cq_ref.shape=[K,M]`,比 AtQd/AtSp。
3. **e2e SNR 验收**:fwd/gradA/gradB SNR > 25 dB(E4M3 阈值)。

---
来源: mxfp8-8wave-devloop/SKILL.md
