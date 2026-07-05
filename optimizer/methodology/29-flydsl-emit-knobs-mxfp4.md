# mxfp4 4-wave 生产 emit 杠杆:GAVOID/MMORD/INPLACE_ALT/WLBARNOP/ELGK/WLVMCN/SCV_ILV

> 类别: 方法论 · 主题标签: mxfp4, emit-knob, race-correctness, VGPR-scale

## 生产成功 emit 杠杆(按 med 增益)

| env var | 值 | 增益 | 机制 / WHY |
|---|---|---|---|
| `FP4_INPLACE_GAVOID` | 1 | +55T | g2s 避开 refill slot |
| `FP4_MMORD` | 9 | Llama 7b-qkv +2% | blocked-diagonal 4×8:块状对角把同-acc 的 2 K-sub 隔开,消累加器 RAW stall |
| `FP4_INPLACE_ALT` | 0 | +27T | B-side progressive,须配 `MMORD=5` |
| `FP4_WLBARNOP` | 1 | +21T | barrier 后插 1 个 s_nop |
| `FP4_INPLACE_ELGK` | 9 | +27T | barrier 处留 9 个 ds_read 在飞 |
| `FP4_WLVMCN` | 10 | +10T | — |
| `FP4_SCV_ILV` | 1 | +20T (min) | scale load 交织进 mfma 流 |

## race-correctness 边界(稳定 emit)

- `ELGK`:最优 9,`≥15` racy。
- `WLVMCN`:最优 10,`≥20` racy。
- barrier 是 race-critical:减 barrier(`LEANBAR`)必触发 operand race。
- cross-wave race 根因 = **LDS barrier 不足**,不是 vmcnt 乱序:g2s `buffer_load→LDS` 是 wave 协作完成,barrier 确保所有 wave 的 g2s 全部落地后,跨 wave 的 ds_read 才安全。

## authoring:VGPR-direct scale 补 refill-ahead lag

- 去 LDS ds_read、当场消费 scale 时,soffset 必须补 refill-ahead lag:
  `soffset = o_sca - AH*n_sub*256`,`AH=2`(refill-same,默认)/ `AH=1`(refill-OTHER)。
- WHY:LDS 路径经缓冲,延迟 `AH` 个 K-iter 才消费;VGPR-direct 当场消费须手动对齐这个 lag。
- 8w wholeloop:scale gmem 已是 per-lane 布局,consume-lane 天然对齐,**无需改 host 预处理**。

## authoring:PIN 机制(绕过 LLVM RAGreedy 卡死)

- 症状:`=&v` early-clobber 输出太多(如 2-set ping-pong / register double-buffer 的 +6 或 +96 个 `=&v`)触发贪心 RA 病态卡死(compile >130s 不返回)。
- 解法:把内联汇编的 `=&v` 输出换成 ISA 里显式 `v[pb:pb+3]` 物理寄存器字面量,`pb = PINBASE + 组偏移`。
- 对齐约束:PIN 下 scale VGPR 基址须对齐 —— `PINSC=1`(scale 在前)用 `PINBASE`,否则用 `PINBASE + 4*ntmp`;不对齐触发 SNR21 bug。

---
来源: 03-emit-knobs.md, 10-8wave-scvgpr.md
