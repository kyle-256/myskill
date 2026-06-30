# mxfp4 4-wave GEMM 优化 Skill

节点: chi2832 (compute_node_new), GPU7, gfx950/MI355X
最终成果: **5401 med / 5474 min TF, det 0, SNR 55.6** (GPU7, K=28672)
         K=57344: **5481 med / 5540 min TF, det 0**

## 文件列表
- `01-production-config.md` — 最终 production 配置和复现命令
- `02-race-diagnosis.md` — SCVGPR race 根因诊断过程(关键突破)
- `03-emit-knobs.md` — 所有 emit 旋钮实测结果(成功/失败全收录)
- `04-ceiling-analysis.md` — 物理天花板分析(NODSR/CONSTSC/g2s)
- `05-dead-ends.md` — 所有死路(带根因说明,勿重试)
- `06-isa-analysis.md` — 真实 lowered ISA 分析结果
- `07-benchmark.md` — benchmark 方法和工具
- `08-att-root-cause.md` — **ATT trace 根源诊断**：MFMA 89.5% stall + occ=1 LDS-bound(144KB/wg)，唯一突破口 = BN128 whole-loop
- `09-8wave-ceiling.md` — **8-wave 架构天花板 ~4900T**：三道墙(LDS-bw A4×冗余 / LDS 160KB / 寄存器 256@occ2 实测 BN512 spill 374TF)皆因"2 waves/SIMD"定义；5200 需 4-wave(1 wave/SIMD)
- `10-8wave-scvgpr.md` — **8w wholeloop 瓶颈实测穷尽 + 五墙闭合**：(1) PMC 推翻"LDS-bw bound"：LDS_IDX:MFMA_BUSY=1:9.46(端口≥5×余量)，真瓶颈=ds_read 延迟气泡。(2) **ISA 实测 `.vgpr_count=256`(occ=2天花板，rocprof报的128是误报)→零余量，register-prefetch 架构不成立**。(3) PIN 成功移植到 8w(SNR55.6/性能中性)但其目的无 VGPR 余量→无价值；SUBSTREAM/WLDSR/INPLACE 三套读延迟隐藏调度全 ≤baseline(occ=2 粗单同步已最优，拆细约束 wave-switching)。**五墙(VGPR满/LDS余量/bank0/occ锁2/调度最优)全闭合，8w 封顶~4690-4760，>5200 走 4-wave(5351)**
- `11-upstream-agpr-pin-moot.md` — **上游 AGPR-pin commit(#714) 对 mxfp4 4-wave 无帮助**：上游给 fp8 4wave 加 AGPR 原地累加消 accvgpr-shuffle(+5~13%)；但 mxfp4 4-wave 走 bareasm whole-loop，accs `=a` tied 原地累加早已在且更彻底。实测 agpr1(5407/5322)≈agpr0(5390/5348) 噪声内相等→shuffle stall 本就不存在。XCD-swizzle 改动也不适用(mxfp4 用自己的 swizzle)

## 核心认知(必读)
1. **race 根因 = cross-wave LDS barrier 不足**(非 vmcnt 乱序)—— 这是最反直觉的发现
2. **5500 med 在 K=28672 超出 fly BK256 物理顶** — g2s-HBM-BW-bound + 96 ds_read/256mfma
3. **GAVOID 是关键杠杆** — 把 g2s 从 refill-slot 挪开,+55T med
4. **SCVGPR = scale 直读 VGPR** — 去掉 scale LDS ds_read,race 在 barrier 不是 vmcnt
5. **(ATT 实测) 5405T 天花板根因 = occ=1 wave/SIMD，由 LDS 144KB/wg 锁死**(非 VGPR)。
   MFMA stall 89.5% = 等 operand bubble(单 wave 无法填)。occ=2 需 LDS≤80K → 需 BN128
   单 slice → 但 whole-loop bare-asm 是 BN256 专属(BN128 走慢路径 2736T)。详见 08。
