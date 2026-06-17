# Production 配置 (最终交付)

## 结果
- **GPU7 K=28672**: min 5474 / med 5401 TF, det 0, SNR 55.6
- **GPU7 K=57344**: min 5540 / med 5481 TF, det 0, SNR 55.6
- 3 commits 推到 kyle-256/FlyDSL:dev/mxfp4-8wave-band

## Production Env 变量
```bash
FP4_ASMMFMA=6           # bare-asm whole-loop INPLACE 模式
FP4_INPLACE=1           # INPLACE emit (last-use refill)
FP4_INPLACE_DIAG=1      # DIAG traversal (both-side progressive)
FP4_MMORD=5             # 2×4 block order (A-row×B-col, +41T med vs 2×2)
FP4_INPLACE_ALT=0       # B-side progressive traversal (complements MMORD=5)
FP4_SINNER=1            # K-sub innermost (consecutive acc mfma)
FP4_INPLACE_1BAR=0      # 每 phase 都有 s_barrier (修 cross-wave race 关键!)
FP4_INPLACE_ELGK=9      # barrier lgkmcnt=9 (留 9 个 ds_read 在飞,+27T)
FP4_WLVMCN=10           # vmcnt=10 at boundary (g2s 深度)
FP4_INPLACE_GAVOID=1    # g2s 避开 refill slot (+55T med,关键)
FP4_WLBARNOP=1          # 1 s_nop after barrier (+21T med)
FP4_SC_VGPR=1           # scale 直读 VGPR (去 scale LDS ds_read)
FP4_PIN=1               # 固定 operand VGPR 布局
FP4_PINSC=1             # 固定 scale VGPR 在 PINBASE
FP4_PINBASE=8           # scale VGPR 从 v8 开始
FP4_SCV_ILV=1           # scale load 交织进 mfma 流 (+20T min)
BLOCK_K=256             # BK256 (BK128 更慢,loop 开销)
```

## FP4_PROD 机制
4wave.py 中 `_apply_prod_defaults()` 通过 `setdefault` 自动应用上述配置。
显式 env 仍可覆盖;`FP4_PROD=0` 可完全回退到 intrinsic 3583TF 基线。

## 复现命令
```bash
cd /wekafs/kyle/code2/remote_sync

# K=28672 (~5400 med)
env GPU=7 DETRUNS=20 ./rr.sh run ../FlyDSL/turbo/test_mxfp4_4w.py 8192 8192 28672

# K=57344 (~5480 med)
env GPU=7 DETRUNS=20 ./rr.sh run ../FlyDSL/turbo/test_mxfp4_4w.py 8192 8192 57344

# 500-sample 精确测量
env GPU=7 ./rr.sh run ../FlyDSL/turbo/_bench500.py

# 不带任何 env (FP4_PROD 自动生效)
GPU=7 DETRUNS=20 ./rr.sh run ../FlyDSL/turbo/test_mxfp4_4w.py 8192 8192 28672
```

## 进展时间线
| 配置 | med TF | 说明 |
|---|---|---|
| intrinsic (base) | 3583 | FP4_ASMMFMA=0 |
| INPLACE bare-asm | 5173 | FP4_ASMMFMA=6 FP4_INPLACE=1 FP4_INPLACE_DIAG=1 |
| + SCVGPR race-fix | 5305 | +1BAR=0 修 cross-wave race |
| + ILV | 5330 | scale load 交织 |
| + ELGK=7 | 5346 | barrier drain 放宽 |
| + MMORD=5+ALT=0 | 5370 | 2×4 block+B-side progressive |
| + GAVOID | 5391 | g2s 避开 refill slot |
| + BARNOP=1 | **5401** | 1 s_nop after barrier |

## 相关文件
- kernel: `FlyDSL/turbo/mxfp4_gemm_4wave.py`
- emit lib: `FlyDSL/turbo/mxfp4_gemm_8wave.py`
- test: `FlyDSL/turbo/test_mxfp4_4w.py`
- bench: `FlyDSL/turbo/_bench500.py`, `_bench2000.py`
