# cross-wave LDS-barrier race 诊断:逐一排除法定位 g2s→ds_read、barrier 是 race-critical

> 类别: 方法论 · 主题标签: race-diagnosis, cross-wave-lds, barrier, vmcnt-defer

## race 症状识别
- 表现 = 低概率(0.001-1%)bit-non-deterministic:少数 element 偶尔不一致,**不是数值漂移**(不是 NaN/Inf/SNR 下降)。
- 高发场景:grouped GEMM / persistent kernel。dense single-GEMM 通常无 race,只有 SGPR pressure 触发 spill 才暴露。

## det/race 复测协议
- 固定量化后的输入连跑 `DETRUNS=100`(或 50K-150K)次,逐元素 bit-exact(rtol=0 atol=0)比对 run#0。
- cross-wave LDS-barrier / SCVGPR race 表现为 run-to-run 漂移 max|d|≠0。
- 低概率 race 统计量级细节(10K 假阴性/需 50K-150K+ iters/多 GPU 确认)见 pitfalls/28-gfx950-vmcnt-spill-race.md「race harness」。

## 逐一排除法定位(SCVGPR/vmcnt race)
| knob | 设置 | 结果 | 排除的假设 |
|---|---|---|---|
| WLV | =0(vmcnt 全排空) | 仍 racy | 排除 vmcnt 乱序 |
| ELGK | =0(lgkmcnt 全排空) | 仍 racy | 排除边界 drain 不足 |
| CONSTSC | scale 设常量 | 仍 racy | 排除 scale 值 → 定位到 operand cross-wave g2s→read |
| 1BAR | =0(每 phase 都加 s_barrier) | **det 0** | 确认根因 = barrier 不足 |

## 根因:LDS barrier 不足(非 vmcnt 乱序)
- g2s(`buffer_load_lds`/buffer_load→LDS,VMEM 路径)是 wave 协作完成;barrier 确保所有 wave g2s 落地后,跨 wave `ds_read` 才安全。
- operand(ds_read)的 LDS 可见性依赖:g2s vmcnt 落地 **+** s_barrier 跨波同步。读 8 波协作填充的 LDS 必须在某 barrier 之后。
- gfx950 barrier 语义:`wait_barrier(cnt)` = `s_waitcnt vmcnt(cnt)` + `s_barrier`。

## mxfp4 4-wave 稳定 emit 边界
- `ELGK ≥ 15` racy(最优 9);`WLVMCN ≥ 20` racy(最优 10)。
- barrier 是 race-critical:减 barrier(`LEANBAR`)必触发 operand race。

## ISA diff 精确定位 racing gap
- `PT_RACE_VM=0`(safe) vs `=1`(racing) 的 `21_final_isa.s` 只差 **2 行**(两相位 barrier 的前置 drain):
  - safe:`s_waitcnt vmcnt(0) lgkmcnt(0)` 全 drain
  - racing:`vmcnt(16) lgkmcnt(10)`
- 一相位正好 16 条 `buffer_load_dwordx4`(4 pool × 4 step)→ 对上 vmcnt(16)。
- 方法:逐行 ISA diff 精确定位性能/正确性差异来源。

## 安全回收 racing 优势 = 加缓冲深度换 partial-drain
- 1 池 3buf(只给 B1 第 3 缓冲):安全延迟 1/4 写 → 回收 racing 优势约 **51%**。
- 2 池 3buf(B0+B1 都第 3 缓冲):安全延迟 1/2 写 → m2048 打平/超 racing、全线比 1 池 **+2~4%**、m4096 距 racing **0.6%**。

## prefetch:AGPR 累加腾 VGPR → 手工提早 ds_read
- AGPR 累加腾出的 VGPR 余量(如 **128→110**)可用于把 operand ds_read 主动提早进 MFMA 窗口做重叠。
- 编译器因 volatile+barrier 强序做不到;手工把一个 operand 的 ds_read **下移一个 barrier**(在可见性安全范围内)可恢复并反转残差。

---
来源: gfx950-vmcnt-race-debug/SKILL.md, 02-race-diagnosis.md, 03-emit-knobs.md, 10-grouped-wgrad-4wave-3buf.md, agpr_phase5_mono.md
