# MXFP8 scale_pack(MFMA op_sel)+ 大-K scale group 预取:闭合 per-tensor gap

> 类别: 方法论 · 主题标签: mxfp8, scale_pack, op_sel, 大K预取, per-K-loop, pack歧义

## 制胜双招(缺一不可,per-K occ=2 clean 内核,非 whole-loop)
1. **scale_pack**:`mxfp8_scale_pack(K)=4 if K//128%4==0 else 2 if %2==0 else 1`,与 C++ quant `k_pre_pack` 完全一致。把连续 `pack` 个 K-iter 的 E8M0 byte 打进一个 i32(byte 0..3),每 pack 个 iter 只发 **1 次 scale load**,MFMA 用 `op_sel=k%pack`(编译期立即数)选 byte → 砍 scale VMEM `pack×`,消 pack=1 broadcast 的 V=256 spill;热循环**零额外 ALU/VGPR**。
   - 因 pack 永远整除 K128,**不需 ceil-K128**;`ScaleS2R/ScaleBComb` 维持 `K//(128*pack)`。
2. **大-K scale group 预取**:`_N_SGROUPS=(K_ITERS+pack-1)//pack`,loop 内每 group **首 iter 预取 g+1 的 scale**(sa0n/sa1n/sb_alln 藏进当前 group MFMA shadow)、**末 iter swap**。
- 结果:全 shape ≥0.97× per-tensor,几何均值反超 ~1.01×。

## pack 歧义消除(kernel 分不清 packed / broadcast)
- `execute()` 由 **caller** 按 scale 来源指定 pack:
  - int32 透传(C++ quant 吐 packed)→ `scale_pack=mxfp8_scale_pack(a.shape[1])`;
  - raw 路径(`preshuffle_ab_flydsl` broadcast)→ `scale_pack=1`。
- recipe 开包:`gemm_fp8.py` 把 `_R(1)→_R(2)`、`_R(3)→_R(4)`(A/B broadcast layout 1/3 → packed layout 2/4),C++ quant 据此按 K 自动吐 packed int32,execute 端 `mxfp8_scale_pack(K)` 吻合;non-fast-path(raw E8M0)走 broadcast 不变。

## grouped 版落地(优化7,默认 ON,PACK=4)
- preshuffle 把每 K-iter 一个 broadcast dword 改成 PACK=4 个相邻 K-iter 打进 1 dword → preshuffle 写流量 + scale buffer 占用缩 4×;GEMM 侧 `op_sel=scale_opsel(k)=k%PACK`。
- 落在 **NT(fwd+dgrad)+ dense**(K-iter 编译期已知);**wgrad 保持 pack=1**(变长-K 每组起始 ks0 只 128 对齐非 PACK*128,`k%PACK` 非编译期常量)。
- `_SCALE_PACK=int(os.environ.get('PT_SCALE_PACK',4))`,`PT_SCALE_PACK=1` 逐位复现旧布局。
- 效果:融合 preshuffle kernel **-32%**(14.3→9.7us),GEMM 完全不变,全 shape bit-exact;但 preshuffle 只占 fwd/dgrad ~3.5%,砍 32% ≈ e2e -1%,**是卫生/带宽改进非 e2e 杠杆**。

## dense per-K 达标结果
- pack4+预取生产接通后 kernel-only vs per-tensor 几何均值 ~1.01×(MXFP8 平均反超):7B QKV 1.10 / GateUp 0.97 / Down 1.04-1.06,70B QKV 1.04 / GateUp 0.97 / Down 0.97;FLOP 主导的 GateUp/Down/QKV 全 ≥0.966。

---
来源: mxfp8-8wave-devloop/SKILL.md; project_mxfp8_wholeloop_port.md; mxfp8-grouped-gg-devloop/SKILL.md 优化7
