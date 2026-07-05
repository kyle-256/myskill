# 寄存器压力:asm-inplace MFMA 把 accum 挪进 AGPR 消 spill、raw-asm 绕 LLVM 拒 AGPR

> 类别: 方法论 · 主题标签: register-pressure, agpr-accum, raw-asm, whole-loop-asm

## 核心手法:把累加器从 VGPR 挪进 AGPR
- **asm-inplace MFMA(`asm_mma=2` mode2)**:D 别名 C in AGPR,`agpr_alloc=128`,把 accum 挪进 AGPR,消掉 accvgpr 拷贝 + spill(**18→0**),big-K **+2.3%** 且 **det0**。  (flydsl-fp8-gemm-tuning)
- **whole-loop bare-asm**:整个 K-loop 写成一个内联汇编 hw-loop,消 per-mfma asm 边界 + per-iter 循环开销;是 mxfp4 4-wave 从 **3583(intrinsic)→ 5401** 的关键 lever(**+16%**)。accs 用 `['=a']` 且 MFMA dst=acc_in(`$q,$a,$b,$q`)实现 AGPR 原地累加,天然消除 accvgpr-shuffle。  (11-upstream-agpr-pin-moot)

## gfx950 scaled/fp4 MFMA:LLVM 拒 `=a` → raw-asm 写死物理寄存器
- **问题**:gfx950 scaled MFMA(`mfma_scale`)LLVM 不肯给 AGPR 分累加器 —— `=a` 约束被拒,AccumVGPR 恒 0。  (agpr_rawasm_progress)
- **绕过**:inline `volatile` asm 文本写死物理 `a[N:N+3]` 做 dst/src,**不用 `=a` 约束**,累加器只放进 clobber `~{aN}`。
  - init:`v_accvgpr_write_b32 aN, 0`
  - readout:`v_accvgpr_read_b32 $k, aN`
- **收益**:32 个 v4f32 累加器从 **256 VGPR 移进 128 AGPR**(`num_vgpr` 256→128);det0 证明跨迭代物理驻留稳定。  (agpr_rawasm_progress)
- **fp4 raw MFMA(`cbsz:4`/`blgp:4`)操作数宽度坑**:要求 A/B operand 是 **4-dword**(`i32x4` / `v[N:N+3]`);官方 intrinsic 用 `i32x8`(高 16B 补零)。用 raw-asm 时 `S2RLoaderFp4` 必须 `pad=False` 返回 `i32x4`,否则汇编器报 `wrong register tuple size for cbsz value 4/blgp value 4`。scale 仍是 `i32`(1 dword)不变。  (agpr_rawasm_progress)

## AGPR 腾出的 VGPR 余量 → 手工预取重叠
- AGPR 累加腾出的 VGPR 余量(如 **128→110**)可用于把 operand `ds_read` 主动提早进 MFMA 窗口做重叠。
- 编译器因 `volatile` + barrier 强序做不到;手工把一个 operand 的 `ds_read` 下移一个 barrier(在可见性安全范围内)能恢复并反转残差。
- **gfx950 barrier 语义**:`wait_barrier(cnt)` = `s_waitcnt vmcnt(cnt)` + `s_barrier`;operand(`ds_read`)的 LDS 可见性依赖 g2s(`buffer_load_lds`,VMEM)vmcnt 落地 + `s_barrier` 跨波同步。读 8 波协作填充的 LDS 必须在某 barrier 之后。  (agpr_phase5_mono)

## wgrad 4-wave whole-loop 结构(AGPR 累加实战)
- 详见 methodology/28-whole-loop-asm-lds-feed-bound.md（4-wave whole-loop 结构小节）。

## GEMM VGPR 估算(算 arch_vgpr,判是否 spill)
| 组成 | 公式 |
|---|---|
| 累加器 | `m_repeat × num_acc_n × 4`(→ accum_vgpr) |
| B tile | `k_unroll × 2 × num_acc_n × 2` |
| A 预取 | `2 × 2` |
| A tile regs | `num_a_loads × 4`(同步 copy) |
| 地址 | ~10-20 |

- 例:64×256×128 FP8 ≈ **148 arch_vgpr**。  (gemm-optimization)

---
来源: flydsl-fp8-gemm-tuning/SKILL.md, 10-grouped-wgrad-4wave-3buf.md, gemm-optimization/SKILL.md, agpr_rawasm_progress.md, agpr_phase5_mono.md, 11-upstream-agpr-pin-moot.md
