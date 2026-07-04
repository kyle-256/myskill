# Emit 旋钮全收录(成功+失败)

## ⚠️ 头号陷阱:JIT-cache 不含部分 emit 旋钮 → 假阴性(2026-07-01)
`call_mxfp4_wholeloop` 的 cache-key tuple(mxfp4_gemm_8wave.py ~L960-977)**只含部分旋钮**
(SS_UNROLL/WLVMCN/INPLACE/PIN/PINBASE/INPLACE_*/FEWOP/NSUBFOLD... 这些换值会自动重编译)。
**不含** `FP4_MMORD`、`FP4_SINNER`、`FP4_ACC_DIST16`、`FP4_DSRD`、block-dict 等。
→ 扫这些"不在 key 里"的旋钮时,**不清 cache 会复用上一个已编译 kernel(stale)→ 看似无差异 → 误判**。
**测这些旋钮必须每次 `rm -rf /root/.flydsl/cache`**。本仓早期"MMORD 变体全判负"就是此坑,实为 mm9 +3~5%(见下)。

## 基础配置(必须)
| 旋钮 | 值 | 效果 |
|---|---|---|
| FP4_ASMMFMA | 6 | bare-asm whole-loop(vs 0=intrinsic 3583) |
| FP4_INPLACE | 1 | INPLACE last-use refill emit |
| FP4_INPLACE_DIAG | 1 | DIAG traversal(both-progressive) |
| FP4_SINNER | 1 | K-sub innermost(consecutive acc mfma) |
| BLOCK_K | 256 | BK256(BK128 更慢 4597) |

## 成功杠杆(按 med 增益排序)

### FP4_INPLACE_GAVOID=1 (+55T med)
避开 refill-free slot 放 g2s → g2s 不抢 ds_read 的 issue slot
```python
# emit_inplace 里预计算 refill slot,g2s 放在 非-refill slot
_gavoid = int(environ.get("FP4_INPLACE_GAVOID", "0"))
```

### FP4_MMORD=9 (blocked-diagonal 4×8) — Llama 默认新最优,+3~5% 全局
block dict 已扩展(mxfp4_gemm_8wave.py ~L1396):
```
MMORD 3=2×2  4=4×2  5=2×4  6=4×4  7=2×8  8=8×4  9=4×8(默认)  10=8×8
```
- **mm9=(bm4×bn8)** 相对**现 mm5 默认** +~2%(7b-qkv M8192 med 4117→4210,6/6 一致;其余中性),
  SNR55.6 det0(含 odd-KI 7b-down)。⚠️ 不是"+3~5%全局"——那个数是跟 mm0(从来不是默认)比的假象,见 12 更正。
- **部署点**:`compile_mxfp4_gemm_4w` 的 PROD `setdefault`(4wave.py L290)`MMORD:"9"`。
  8wave.py 的 environ 默认被 PROD setdefault 遮蔽,单改它无效。
- 老 K28672 单-shape 调优里 MMORD=5(2×4)当时最优(+41T vs mm3);6-10 是**本次重定义的新 block 码**,
  与老笔记"6=1×8/7=4×4 racy"是不同定义(旧码已不存在)。**新 6-10 全 det0 不 racy**。

### FP4_INPLACE_ALT=0 (+27T med)
B-side progressive traversal 配合 MMORD=5。默认 ALT=1(A-side)。
DIAG=1+MMORD=5+ALT=0 是最优组合。

### FP4_WLBARNOP=1 (+21T med)
barrier 后插 1 s_nop → 给 cross-wave wavefront 调度 settle 时间。

### FP4_INPLACE_1BAR=0 (修 race → +195T vs racy 5305)
每 phase 都有 s_barrier → 修 cross-wave operand race → det 0。

### FP4_INPLACE_ELGK=9 (+27T vs ELGK=1)
barrier 处 lgkmcnt=9(留 9 个 ds_read 在飞)。
扫描: ELGK=1→7→9 递增,ELGK≥15 racy。最优 ELGK=9。

### FP4_WLVMCN=10 (+10T vs WLV=8)
vmcnt=10 at boundary。扫描: WLV=4/6/8/10/12,10 最优。WLV≥14 racy。

### FP4_SC_VGPR=1 + FP4_PIN=1 + FP4_PINSC=1 + FP4_PINBASE=8
scale 直读 VGPR(去 scale LDS ds_read)→ +128T(const-scale 下)。
PIN 固定寄存器布局避免 RAGreedy crash。
**必须配 FP4_INPLACE_1BAR=0 才能 det 0!**

### FP4_SCV_ILV=1 (+20T min)
scale buffer_load 交织进 mfma 流而非 phase 开头 burst。

## 失败路径(不要重试)

### FP4_EVENSPREAD=1 → -207T
ds_read 均匀分散反而破坏 INPLACE 的 last-use overlap,与 GAVOID 相互干扰。

### FP4_FINELGK=1 → -50T
精细 lgkmcnt drain(per-mfma)序列化开销超收益。

### FP4_MFMANOP=1 → neutral/negative
s_nop pacing 无帮(LDS-bw bound 非 latency)。

### FP4_SCDWX4(lds128 dwordx4-lds) → det≠0 / 慢
dwordx4-lds 写 LDS 完成不被 vmcnt/barrier 可靠同步。det0 需序列化 → 4803 < 5176。

### FP4_LEANBAR=1/2 → racy 或 更慢
减少 barrier 数量会触发 operand race。barrier 是 race-critical 的。

### FP4_CHUNKBAR(细粒度 barrier) → -130T
在单块 BK256 里加细粒度 barrier 纯增成本,aiter 7-barrier 只在 BK128+2K-unroll 里廉价。

### FP4_WLDSR=1 → 对 INPLACE 路径无效
WLDSR 只在非 INPLACE 分支(`else` 里),INPLACE=1 时完全被跳过。

### FP4_PREFETCH=1 → racy
提前发 refill(fraction×last_use)时 nxt_buf g2s 还没落地 → 必然 race。

### FP4_ROBUF(register double-buffer) → RAGreedy crash / 无收益
BK256 下 2 operand set 超 512 VGPR cap → crash。BK128 fit 但 loop 开销。

### FP4_ACCD16=1 → racy + 更慢
cross-bank acc 顺序杀 A-operand 复用,det race。

### FEWOP=1 → 5648 fast 但 SNR garbage
用单 reg 测 operand 多样性天花板,不是真实 kernel。

### (已作废)旧 MMORD=6(1×8)/7(4×4) → racy(-250T)
⚠️ 仅针对**旧 block 定义**。现 dict 已重定义(6=4×4,7=2×8,...,10=8×8),**新 6-10 全 det0 不 racy**。
旧结论"column-first/4×4 打破 DIAG both-progressive → SNR32"很可能也是没清 cache 的假测,新扫全过。

### FP4_WLRING=1(4-buffer ring) → VGPR 溢出
RING 需 4 operand buffer,超出 512 VGPR cap。

### FP4_GLATE + FP4_FINELGK 联合 → 负
两者逻辑冲突,联合时效果互相抵消。

## 旋钮扫描结果对照表
| 旋钮 | 最优值 | 范围 | 说明 |
|---|---|---|---|
| FP4_MMORD | 9(Llama)/5(K28672) | 3-10 | 6-10=新 block 码,全 det0;测前清 cache |
| FP4_INPLACE_ELGK | 9 | 1-15 | ≥15 racy |
| FP4_WLVMCN | 10 | 0-16 | ≥20 racy |
| FP4_WLBARNOP | 1 | 0-8 | 更多 nop 递减收益 |
| FP4_INPLACE_ALT | 0 | 0/1 | 配合 MMORD=5 |
| FP4_INPLACE_GAVOID | 1 | 0/1 | 开 best |
| FP4_SC_VGPR | 1 | 0/1 | 需 1BAR=0 配套 |
| FP4_PINBASE | 8 | int | scale VGPR base |
