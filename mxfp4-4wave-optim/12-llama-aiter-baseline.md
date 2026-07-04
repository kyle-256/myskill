# Llama 7B/70B 前向基线 vs aiter + 奇数-KI 修复 (2026-07-01)

> ⚠️ **部署点已迁移**：下文"部署在 `mxfp4_gemm_4wave.py` PROD L290 setdefault / `FP4_MMORD` 等 env"
> 是 FlyDSL standalone 调优期的说法。**当前出货后端 = Primus-Turbo `mxfp4_gemm_kernel.py`(env 全
> hardcode + timed autotune，2026-07-02 已 rebase-onto-main + squash 成单 commit `74eaadac` 并清理掉
> 所有 PT_MX_*/persistent)，见 `13-primus-turbo-prod.md`**。本文的 fly/aiter 表(geomean 0.995)因
> compute 路径未变仍代表性成立；但"怎么改默认/怎么复现"以 13 为准。

GPU0 chi2810 mlperf_gptoss。harness = `test_mxfp4_4w.py`(fly)+ `bench_aiter.py`(aiter a4w4
bpreshuffle=False),同 session 交替,100-sample med,噪声 ~±1.5%。

## 奇数 KI(K%512==256)正确性修复 ✅
- 症状:standalone 4w 在 KI 为奇数的 shape 算错(K768/1280/11008 SNR 5.2/7.4/16.6;K256 SNR 0.3)。
  偶数 KI 全对。真实影响 **7b-down(K=11008,KI=43)**。
- 根因:whole-loop unroll-2 do-while `cnt=0;do{2 iter};cnt+=2;while cnt<KI` 在奇 KI 多算 1 个
  phantom iter k=KI,其 g2s 偏移=K/2=下一行起点(不 OOB)→ 累加下一行垃圾。
- 修法:移植生产 wrapper 的 odd-KI tail。`call_mxfp4_wholeloop` 加 `ki` kwarg;caller 传
  `nval=KI-(KI&1)`(floor-even)+ `ki=KI`;循环后对 `_INPLACE&_SCVGPR&ki&1` 发 MFMA-only
  phase-A tail(`drain; _scb[0]=0; L+=emit_mm()`,消费最后 phase-B 1-ahead 预取的 k=KI-1)。
  cache key 加 `_oddtail`。改的是模块级方法 → **测前清 `~/.flydsl/cache`**。
- 详细根因/改动/验证见 memory `mxfp4_4w_oddki_llama_aiter.md`。

## Llama fwd(NT)fly vs aiter,med TF,fly/aiter 比(7b-down 为修复后)
```
shape (N×K)            M4096         M8192         M16384
7b-qkv  12288×4096     .953          .937          .985
7b-oproj 4096×4096     1.068         .941          .983
7b-gateup 22016×4096   1.061         1.075         1.058
7b-down  4096×11008    1.054         1.000         1.034   (fixed)
70b-qkv  10240×8192    .974          .995          .975
70b-oproj 8192×8192    .953          .986          .990
70b-gateup 57344×8192  .989          .974          .961
70b-down  8192×28672   .977          .987          .984
```
- **geomean fly/aiter ≈ 0.995(24 点,基本 parity)**。全 SNR 55.6 det0。
- 赢:7b-gateup(+6~8%)、7b-oproj@M4096、7b-down(修复后)。
- 输 ~3-6%:大-N 短-K(7b-qkv、70b-gateup、oproj@M8192)。
- 弱 shape 杠杆实测:group_m/n 最多 +1%,num_xcds NX8 最优,BLOCK_N=128 standalone 算错不可用。

## MMORD emit-order:mm9(4×8)= 现 mm5 默认之上 +~2%(2026-07-01)⚠️含自我更正
### ⚠️ 更正:上表 loser 的比值就是 **mm5 默认**下测的(不是"输 3-6%"的旧态)
- `compile_mxfp4_gemm_4w` 有 **PROD `setdefault` 块**(mxfp4_gemm_4wave.py L287-303,**上会话已提交**):
  `ASMMFMA=6 INPLACE=1 DIAG=1 SINNER=1 MMORD=5 ALT=0 GAVOID=1 SC_VGPR=1 PIN...`。setdefault 在 8wave 读取前生效
  → **生产默认早就是 mm5**。上表(.937 等,memory 老表)其实是更早 mm0-era / 异 session;mm5 默认下 loser 已 ~.97-.99。
- 我把"base"设成 `FP4_MMORD=0`(降级默认)去比,得出"mm5 +3~5%"是**假象**——base 从来不是默认。**本会话没有 3-5% 的提升**。
- **8wave.py 默认 0→9 那次改动被 PROD setdefault "5" 遮蔽、无效**;真正生效点 = **4wave.py PROD L290 "5"→"9"**。

### JIT-cache 陷阱(仍有效教训)
- cache-key tuple(8wave L960-977)**不含 FP4_MMORD/SINNER/ACC_DIST16/DSRD/block-dict** → 扫这些**必须每次
  `rm -rf /root/.flydsl/cache`**,否则复用 stale kernel 得假阴性(SS_UNROLL/WLVMCN/INPLACE 在 key 里,不受影响)。

### ⚠️ 对比没开 autotune → autotune 才是 loser 真解
- standalone `test_mxfp4_4w.py` 是**固定** GM4/GN16/NX8;aiter 库内部按 shape 选 tile → 上表对 fly 不公平。
- per-shape group 调优(GM×GN×NX,运行期参数无 cache 问题)大幅缩小差距:7b-qkv M4096 .89→~.94(GM4/GN32)、
  70b-oproj M4096 .95→~.97(GN32)、7b-oproj M16384 .95→1.01(GM8/GN32)、70b-gateup M8192 .96→~.97(GM1/GN8)、
  7b-oproj M8192 .97→~.98(GN16)。**规律:小-M 要大 GN(16/32);NX8 普遍最优**。
- **生产 kernel 独立文件** `mxfp4_gemm_kernel.py` 有真 autotune(定时扫 `_MXFP4_AUTOTUNE_CANDIDATES` 取 min)。
  已补候选 `(4,16,8)(4,32,8)(8,32,8)(1,8,8)` + emission 2×4→4×8(commit `07b8ee4e`)。该文件 env 全 hardcode、
  不读 FP4_MMORD,故 standalone 的 MMORD 改动不影响它(需单独改 block 常量,已改)。

### 本会话真实收益:mm9(bm4×bn8 blocked-diagonal)> 现 mm5 默认
- 新增 block 码 6-10=(4,4)(2,8)(8,4)(4,8)(8,8);mm9=(4,8)。机理同 mm5(块状对角隔开同-acc 的 2 K-sub → 消累加器 RAW stall)。
- **实测(cache-clear,3 reps,真实 PROD 路径 vs 显式 mm9,med TF)**:7b-qkv M8192 **+2.3%**(4117→4210,6/6 一致)、
  min +0.3%;70b-down M8192 +0.2%(噪声);其余中性。全 SNR55.6 det0(含 odd-KI 7b-down)。
- **净收益 ≈ 7b-qkv +2% / 整体 <1%**。残余 7b-qkv min .94 = aiter 128×512 宽 tile 峰值优势(FlyDSL 结构极限,见 09)。
- **已部署**:4wave.py PROD `MMORD:"9"`(+ 8wave.py fallback 9 覆盖非 PROD 路径)。真正对标仍建议跑生产
  `bench_llama_mxfp4_flydsl_vs_aiter.py`(FLYDSL vs AITER,fwd+bwd)。
