# 死路汇总(不要重试,每条都有实测根因)

## BK128 路径

### BK128 SCVGPR → scale VGPR WAR race(HW-walled)
**现象**: K=256(0 main iter)SNR 55.6;K=384(1 main iter)SNR -inf racy
**根因**: phase-B 的 `emit_sc_vgpr(0) → v[8:9]` 覆写 phase-A mfma 还在读的 scale VGPR
**本质**: VGPR WAR hazard,vmcnt(0)不能修(vmcnt 乱序退役不保证特定 load 落地)
**结论**: BK128 SCVGPR HW-walled,与 BK256 SCVGPR 同一机制

### BK128 plain → 4597 med(loop 开销)
**现象**: det 0 SNR 55.6 但 4597 << 5401
**根因**: BK128 loop 224 iter vs BK256 112 iter,loop 固定开销碾压 compact-LDS 收益
**结论**: fly BK128 loop 开销在 fly manual-emit 下无法消除(aiter 靠 hand-asm 消了)

### SUBSTREAM / ASYM(3-buffer 变体) → SNR 1.8 broken
**现象**: 正确性直接坏掉
**根因**: SUBSTREAM 与 INPLACE 路径冲突;ASYM/RING 在 BK256 下 LDS 超 160KB

### 3-buffer BK256 → LDS 超限
```
A3+B3 = 3×32KB + 3×32KB = 192KB > 160KB cap
A3+B2 = 96+64 = 160KB = 恰好满(无 scale)
```
SCVGPR 时省了 scale 16KB → A3+B2 = 144KB 可放,但 A3 = SUBSTREAM broken

## Register 路径

### read-once register double-buffer → 全路径崩或无收益
- full @ BK256: RAGreedy crash(272 inline-asm 输出超压 allocator)
- A-only @ BK256(PIN): "couldn't allocate v[256:259]"
- A-only reduced struct @ BK256: LLVM UNREACHABLE
- full @ BK128: fit(192V) 但 4613 racy + loop 开销

### FP4_BPREF(B VGPR prefetch) → spill 爆
**根因**: B prefetch 需要额外 VGPR,8-wave occ=2 下 V+A cap 256 dwords 被撑破

### asm-AGPR(AGPR 累加器) → V 不降
**根因**: fp4 MFMA 有 5 个操作数(a,b,sa,sb,c),sa/sb 必须 VGPR → acc 移到 AGPR 但 V 不降,V+A=384 > cap

## Scale 路径(全探明)

### dwordx2-lds → INVALID 指令
`buffer_load_dwordx2...lds` gfx950 不支持,只有 dword 和 dwordx4。

### dwordx4-lds 直写 → det≠0 HW-walled
LDS 写完成不被 vmcnt/隔-phase barrier 可靠同步。只有 `vmcnt(0)+s_barrier 紧跟` 能 det0 → 序列化 → 4803 < 5176。

### SCVGPR SCV2AHEAD / SCPF → racy
vmcnt 乱序退役,prefetch 的 scale load 不能保证特定时刻落地。

### TRB8(lane-major scale LDS) → 4539 det0 < 5176
staging(buffer_load→VGPR+ds_write lane-major)的指令开销 > 宽读省的。

## 结构路径

### 8-wave mxfp4 → 4844(LDS-bw 劣势)
每 warp 128×64(小 tile),B operand 复用减半,ds_read/mfma↑
即使 3-stage 也没超过 4-wave BK256。

### instrinsic + LLVM scheduler → 4500(手写 asm 打败 bundled scheduler)
LLVM scheduler 4500 vs 手写 5401,FlyDSL bundled LLVM 调度精度不足。

### BK384/512 → Assert / VGPR 溢出
`BLOCK_K % 128 == 0` assert;或 n_sub×operand > 512 VGPR cap。

### compact-LDS(aiter ~22KB) → fly BK128 loop 开销
aiter 22KB = BK128+read-once;fly 128KB = BK256 ping-pong。
减 LDS 必走 BK128 → fly BK128 历史顶 4715 vs aiter 5635 = hand-asm 精度差距。

## 测量方法死路

### GPU 换挡不算达标
GPU0 = 5536(热) vs GPU7 = 5401(冷)。同一 kernel,不同 GPU clock 状态。
目标必须在同一 GPU 稳定重现。

### FEWOP 高数字 ≠ 真实
FEWOP=1 → 5648(single reg, garbage output)。测 operand 多样性上限,不是真实 kernel。

### SCDWX4/TRB8 "5640" → mirror
SNR < 0 时编译器跳过真实计算 → 时间短 → TF 高。一旦正确,真速度 < 5176。
