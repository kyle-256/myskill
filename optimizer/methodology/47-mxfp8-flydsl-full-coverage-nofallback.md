# MXFP8 FlyDSL 必须全覆盖 MX case(禁 fallback)+ shape gate

> 类别: 方法论 · 主题标签: mxfp8, no-fallback, MX全覆盖, shape-gate, dispatcher, K-tail, general-N

## 为什么禁 fallback
- MXFP8 quant 吐的是 **FlyDSL-preshuffled int32 scale**,C++ 后端只认 raw E8M0;**一旦 FlyDSL fallback 到 HIPBLASLT/TURBO 必崩**。FlyDSL 必须自己覆盖所有 MX case。

## 需扩展的 4 块(mx_blockwise 从 275 失败 → 0,1536 passed;源自 gpt_oss2 project_mxfp8_wholeloop_port.md,非本环境)
1. **E5M2/HYBRID**:`MfmaScale` 的 `cbsz/blgp` 按 operand format 设(0=E4M3,1=E5M2)。
2. **fp16 out**:`StoreCPlain` 用 `out_ty`。
3. **K<256 / K%128≠0(K-tail)**:execute 零填充 K → operand=0、scale pad **127(=1.0)**。
4. **任意 N(N%16 / N%64≠0)**:combined-B preshuffle **zero-pad 到 256** + StoreC **mask col≥N**;general-M 用 execute pad M 到 64。
- 关键约束:**int32 scale 不可填充** → preshuffle fast-path **gate 到 K/M/N 全对齐**,unaligned 走 raw(broadcast)路径。

## NT shape gate(`_flydsl_mxfp8_nt_ok`)
```
M%128==0 and N%128==0 and K%128==0
and M>=256 and N>=256 and K>=256
and M*K<2**31 and N*K<2**31
```
- NT-fold 使 M/N/K 都当过 contract 维 → K-loop **无 K-tail** 且预取 2 层。

## Dispatcher 契约
- `granularity==MX_BLOCKWISE` 时 `can_handle()` 返回 **False**(FlyDSL NT 只接 preshuffled,而 dispatcher 喂的是 raw)。

---
来源: project_mxfp8_wholeloop_port.md; mxfp8-8wave-devloop/SKILL.md
