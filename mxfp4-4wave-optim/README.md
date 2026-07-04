# mxfp4 4-wave GEMM 优化 Skill

节点: chi2810/chi2832, gfx950/MI355X
最终成果: **5401 med / 5474 min TF, det 0, SNR 55.6** (GPU7, K=28672)
         K=57344: **5481 med / 5540 min TF, det 0**
Llama 7B/70B fwd fly/aiter **geomean ≈ 0.995 (parity)**, 全 SNR 55.6 det0

> ⚠️ **当前出货后端 = Primus-Turbo `mxfp4_gemm_kernel.py`(env 全 hardcode + timed autotune + 融合单发射)**，
> 部署/成绩基线见 **`13-primus-turbo-prod.md`**，**最新态(preshuffle v2 默认 + 融合 preshuffle+gemm 单发射 + e2e含quant 对比)见 `14-fused-preshuffle-e2e.md`**。
> 本目录 01/03/07/12 描述的 FlyDSL standalone `turbo/mxfp4_gemm_4wave.py` + `FP4_*` env 世界是调优 playground(历史记录)。
> 找"怎么跑现产/最新成绩"看 14→13，找"某旋钮为何这么设/调优史"看 01-12。

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
- `12-llama-aiter-baseline.md` — **Llama 7B/70B 前向 fly vs aiter 基线 + 奇数-KI 修复 + MMORD=9(含自我更正)**。
  奇数-KI(7b-down K=11008)修复(移植 odd-KI tail)。**生产默认早已是 mm5(PROD setdefault),loser 就已 ~.97-.99**;
  本会话真实收益 = mm9(4×8)在 mm5 之上 **+~2%(7b-qkv)/ 整体 <1%**(非最初宣称的 3-5%,那是跟非默认 mm0 比的假象)。
- `11-upstream-agpr-pin-moot.md` — **上游 AGPR-pin commit(#714) 对 mxfp4 4-wave 无帮助**：上游给 fp8 4wave 加 AGPR 原地累加消 accvgpr-shuffle(+5~13%)；但 mxfp4 4-wave 走 bareasm whole-loop，accs `=a` tied 原地累加早已在且更彻底。实测 agpr1(5407/5322)≈agpr0(5390/5348) 噪声内相等→shuffle stall 本就不存在。XCD-swizzle 改动也不适用(mxfp4 用自己的 swizzle)
- `13-primus-turbo-prod.md` — **【生产态基线】Primus-Turbo `mxfp4_gemm_kernel.py` 部署 + timed autotune 四轴(swizzle/deep-wl/COOP+TACCW/ksplit) + 2026-07-02 rebase-onto-main + squash(单 commit `74eaadac`) + pr-merge-gate 清理(-233 行，删 PT_MX_* 旋钮/persistent/死代码)**。复现走 `pytest test_gemm_fp4.py`(141 passed)。成绩 carry-forward 07-01 + provenance。
- `14-fused-preshuffle-e2e.md` — **【当前生产态】preshuffle v2 默认化(删 v1，消 4× 读放大，device 减半)+ 融合单发射(对齐 mxfp8：裸 gemm/preshuffle kernel + 单 `@flyc.jit` stub 发 preA→preB→GEMM，3 次 host dispatch → 1 次)+ e2e(含 quant)对比**。SNR gate `test_gemm_fp4.py -k FLYDSL` 44 passed。kernel-only vs aiter 正确峰值 0.97–1.0×(down70B 0.993)；**e2e(quant+gemm fwd+bwd)fly/aiter geomean fwd 1.174× / bwd 1.171×**。commit `9f525dba`(07-03 merge-gate 复审删死代码 preshuffle 启动链后 amend + force-with-lease；07-02 为 `4b066c40`)。
- `prod_code/mxfp4_gemm_4wave_prod.py` — ⚠️ **FlyDSL standalone 历史快照**(prod 路径精简版，`turbo/*` 仓布局)，仅供调优史参考；当前出货 kernel 见 13，不是这个文件。

## 核心认知(必读)
0a. **⚠️ 部署点在 PROD setdefault(4wave.py L287-303),不是 8wave.py 的 environ 默认**(后者被 setdefault 遮蔽)。
    生产默认早已是 MMORD=5+SINNER=1+ALT=0+GAVOID=1+SC_VGPR=1 等一整套。改默认要改 4wave.py 那个块。
0b. **⚠️ JIT-cache 不含 FP4_MMORD/SINNER/ACC_DIST16/DSRD 等 emit 旋钮** → 扫它们**必须每次 `rm -rf ~/.flydsl/cache`**,
    否则复用 stale kernel 得假阴性。⚠️ 勿把"跟 mm0 比"当提升:mm0 从来不是默认,真实基线是 mm5(见 12 更正)。
1. **race 根因 = cross-wave LDS barrier 不足**(非 vmcnt 乱序)—— 这是最反直觉的发现
2. **5500 med 在 K=28672 超出 fly BK256 物理顶** — g2s-HBM-BW-bound + 96 ds_read/256mfma
3. **GAVOID 是关键杠杆** — 把 g2s 从 refill-slot 挪开,+55T med
4. **SCVGPR = scale 直读 VGPR** — 去掉 scale LDS ds_read,race 在 barrier 不是 vmcnt
5. **(ATT 实测) 5405T 天花板根因 = occ=1 wave/SIMD，由 LDS 144KB/wg 锁死**(非 VGPR)。
   MFMA stall 89.5% = 等 operand bubble(单 wave 无法填)。occ=2 需 LDS≤80K → 需 BN128
   单 slice → 但 whole-loop bare-asm 是 BN256 专属(BN128 走慢路径 2736T)。详见 08。
