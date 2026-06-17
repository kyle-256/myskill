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

## 核心认知(必读)
1. **race 根因 = cross-wave LDS barrier 不足**(非 vmcnt 乱序)—— 这是最反直觉的发现
2. **5500 med 在 K=28672 超出 fly BK256 物理顶** — g2s-HBM-BW-bound + 96 ds_read/256mfma
3. **GAVOID 是关键杠杆** — 把 g2s 从 refill-slot 挪开,+55T med
4. **SCVGPR = scale 直读 VGPR** — 去掉 scale LDS ds_read,race 在 barrier 不是 vmcnt
