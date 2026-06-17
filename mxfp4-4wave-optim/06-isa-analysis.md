# ISA 分析结果

## 真实 lowered ISA(生产 kernel)

### 获取方法
```bash
# 方法1: FLYDSL_DUMP_IR=1 触发 dump
FLYDSL_DUMP_IR=1 DETRUNS=1 GPU=7 ./rr.sh run ../FlyDSL/turbo/test_mxfp4_4w.py 8192 8192 28672
# 文件: /root/.flydsl/debug/kernel_gemm_4w_0/21_final_isa.s

# 方法2: rocprofv3(kernel-only 时间)
rocprofv3 --kernel-trace --output-format csv -d /tmp/rpf -- python ...
```

### Metadata(生产 kernel)
```
.vgpr_count: 424   (V cap 512 → 88 headroom)
.agpr_count: 256   (满:全是 acc)
.vgpr_spill_count: 0
.group_segment_fixed_size: 147456  (144KB LDS)
.wavefront_size: 64
```

### 指令统计(256 mfma / loop body)
| 指令 | 数量 | 说明 |
|---|---|---|
| v_mfma_scale_f32_16x16x128_f8f6f4 | 256 | mfma |
| ds_read_b128 | 96 | operand LDS→VGPR |
| ds_read_b32 | 0 | scale(SCVGPR后消除) |
| buffer_load_dwordx4 | ~16+8 | g2s(operand+scale) |
| s_waitcnt vmcnt | 6 | |
| s_waitcnt lgkmcnt | 4 | |
| s_barrier | 4 | 2 prologue + 2 loop |

### Loop Body 结构
```
label 1:  (line 780)
  vmcnt(10) lgkmcnt(9) + s_barrier   ← phase-A boundary
  [128 mfma + inline ds_read refills]  ← emit_inplace DIAG
  vmcnt(10) lgkmcnt(9) + s_barrier   ← phase-B boundary  
  [128 mfma + inline ds_read refills]
  vmcnt(16) lgkmcnt(1)               ← terminal
  s_cbranch 1b
```

## fly vs aiter 指令对比(K=28672)

| 指标 | fly prod | aiter no-preshuffle |
|---|---|---|
| ds_read/256mfma | **96** (0.375) | 64 (0.25) |
| s_barrier/256mfma | 2 | ~7 |
| s_waitcnt vmcnt(0) | 0 | 0 |
| s_nop | 0 | 17 |
| ds_read_b32(scale) | 0 | 16 |

### 为什么 fly 96 reads 而 aiter 64
- fly BK256: n_sub=2,每 operand 2 个 128b reads,4×ntb×nta×n_sub=96
- aiter BK128: n_sub=1,read-once double-buffer(fresh reg),64 reads
- **read-once 在 fly BK256 物理放不下**(2 operand sets = 384V > 512 cap)

## aiter BK128 结构解码
文件: `/wekafs/kyle/code2/_wf/aiter_noBpreshuffle.s`

```
label_046D (694-1179行): 256 mfma = 2-K-unroll body
.agpr_count: 256  .vgpr_count: ~166  (.vgpr 远小于 fly 424)
- A operand: v76-v107(消费), v44-v75(下 tile prefetch,fresh reg) → read-once
- B operand: v12-v35(消费), v108-v139(fresh)
- scale: v152-v159(ds_read_b32 double-buffer,det-safe)
- barrier: 6/256mfma,每个前 vmcnt(15)(deep-async,跨 barrier g2s 在飞)
- s_nop: 17 pacing nops
```

## ISA 分析工具
```python
# scripts2/isa_regreuse.py - mfma src 复用 + lgkmcnt over-drain 派生
python scripts2/isa_regreuse.py --lgkm /tmp/fly4w_inplace.s
```

## rocprof 测量结果(可靠)
```
n=123 dispatches, best=709124ns, med=721964ns
→ best 5427 / med 5330 TF (kernel-only,无 host overhead)
```

rocprof best(5427) ≈ Event min(5474) 差距 = Event 包含 host/launch overhead。
