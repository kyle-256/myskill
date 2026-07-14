# 寄存器压力/whole-loop 汇编/emit 旋钮:AGPR 累加、LDS-feed bound 与 mxfp4 生产杠杆

> 类别: 方法论 · 主题标签: register-pressure, agpr-accum, raw-asm, whole-loop-asm, lds-feed-bound, bank-conflict, 算术强度, mxfp4, emit-knob, race-correctness, VGPR-scale

## 寄存器压力:asm-inplace MFMA 把 accum 挪进 AGPR 消 spill、raw-asm 绕 LLVM 拒 AGPR

### 核心手法:把累加器从 VGPR 挪进 AGPR
- **asm-inplace MFMA(`asm_mma=2` mode2)**:D 别名 C in AGPR,`agpr_alloc=128`,把 accum 挪进 AGPR,消掉 accvgpr 拷贝 + spill(**18→0**),big-K **+2.3%** 且 **det0**。  (flydsl-fp8-gemm-tuning)
- **whole-loop bare-asm**:整个 K-loop 写成一个内联汇编 hw-loop,消 per-mfma asm 边界 + per-iter 循环开销;是 mxfp4 4-wave 从 **3583(intrinsic)→ 5401** 的关键 lever(**+16%**)。accs 用 `['=a']` 且 MFMA dst=acc_in(`$q,$a,$b,$q`)实现 AGPR 原地累加,天然消除 accvgpr-shuffle。  (11-upstream-agpr-pin-moot)

### gfx950 scaled/fp4 MFMA:LLVM 拒 `=a` → raw-asm 写死物理寄存器
- **问题**:gfx950 scaled MFMA(`mfma_scale`)LLVM 不肯给 AGPR 分累加器 —— `=a` 约束被拒,AccumVGPR 恒 0。  (agpr_rawasm_progress)
- **绕过**:inline `volatile` asm 文本写死物理 `a[N:N+3]` 做 dst/src,**不用 `=a` 约束**,累加器只放进 clobber `~{aN}`。
  - init:`v_accvgpr_write_b32 aN, 0`
  - readout:`v_accvgpr_read_b32 $k, aN`
- **收益**:32 个 v4f32 累加器从 **256 VGPR 移进 128 AGPR**(`num_vgpr` 256→128);det0 证明跨迭代物理驻留稳定。  (agpr_rawasm_progress)
- **fp4 raw MFMA(`cbsz:4`/`blgp:4`)操作数宽度坑**:要求 A/B operand 是 **4-dword**(`i32x4` / `v[N:N+3]`);官方 intrinsic 用 `i32x8`(高 16B 补零)。用 raw-asm 时 `S2RLoaderFp4` 必须 `pad=False` 返回 `i32x4`,否则汇编器报 `wrong register tuple size for cbsz value 4/blgp value 4`。scale 仍是 `i32`(1 dword)不变。  (agpr_rawasm_progress)

### AGPR 腾出的 VGPR 余量 → 手工预取重叠
- AGPR 累加腾出的 VGPR 余量(如 **128→110**)可用于把 operand `ds_read` 主动提早进 MFMA 窗口做重叠。
- 编译器因 `volatile` + barrier 强序做不到;手工把一个 operand 的 `ds_read` 下移一个 barrier(在可见性安全范围内)能恢复并反转残差。
- **gfx950 barrier 语义**:`wait_barrier(cnt)` = `s_waitcnt vmcnt(cnt)` + `s_barrier`;operand(`ds_read`)的 LDS 可见性依赖 g2s(`buffer_load_lds`,VMEM)vmcnt 落地 + `s_barrier` 跨波同步。读 8 波协作填充的 LDS 必须在某 barrier 之后。  (agpr_phase5_mono)

### wgrad 4-wave whole-loop 结构(AGPR 累加实战)
- 详见本文件「whole-loop 4-wave 结构与 LDS-feed bound」章节的「4-wave whole-loop 结构(wgrad)」小节。

### GEMM VGPR 估算(算 arch_vgpr,判是否 spill)
| 组成 | 公式 |
|---|---|
| 累加器 | `m_repeat × num_acc_n × 4`(→ accum_vgpr) |
| B tile | `k_unroll × 2 × num_acc_n × 2` |
| A 预取 | `2 × 2` |
| A tile regs | `num_a_loads × 4`(同步 copy) |
| 地址 | ~10-20 |

- 例:64×256×128 FP8 ≈ **148 arch_vgpr**。  (gemm-optimization)

## whole-loop 4-wave 结构与 LDS-feed bound:byte-exact LDS 账、bank conflict 与算术强度

### 4-wave whole-loop 结构(wgrad)

- occ=1(512 VGPR = 256 操作数 + 256 AGPR 累加器,accum_offset=256),256×256 tile,2×2-wave,两操作数都 transpose-read,whole-loop 裸 asm 主循环,AGPR 累加,CShuffle store。
- 根因瓶颈 = LDS 转置读 feed 带宽(TN 固有税),**非** 占用率/延迟/bank。即使 racing 也只有 fp8 峰值约 44%。
- 主循环 = 两个 ping-pong 相位,相位间一道 `s_waitcnt;s_barrier`。一个 wave 的 transpose-read 要 gather 所有 wave 写的 chunk(`W*chunk_stride`)→ barrier+drain 是跨 wave LDS 一致性硬需求,per-wave 局部 drain 不安全。

### byte-exact LDS 账(2 池 3buf 塞 160KB)

- `_CS=1024`(省 5120B bank-pad) + scalar store(省 8704B C_lds,`PT_WL_2BPOOL` 自动触发)→ 10 buffer × 16384 = 163840 恰好塞满 160KB。
- `_CS` 被双向锁死 = 1024:
  - buffer 需放 16384B → `_CS >= 1024`
  - 10 buffer <= 163840 → `_CS <= 1024`

### LDS bank conflict(_CS=1024 的税)

| 配置 | LDSBankConflict (rocprofv3) | MfmaUtil |
|---|---|---|
| 1 池 @1056 | 0% | — |
| 2 池 @1024 | 14% | 54.3(>50 净胜) |

- `_CS=1024` = bank 周期整数倍 → transpose read 必冲突,但 MfmaUtil 净胜,2 池仍取胜。
- 消 14% 冲突唯一路 = 操作数改 `a_plain`(预转置走 `ds_read_b128`,无 transpose 固定域冲突),但需 upstream 量化产转置副本、跨文件大改。
- `a_plain` 在 kernel 内**零吞吐收益**(gfx950 transpose-read 零 per-op 惩罚),唯一价值是免 pad + 免冲突的 enabler。

### 算术强度:4-wave 长 K 为何领先 8-wave

- 强度定义 = 每 K-step 的 MAC / (A行 + B列 operand-elem)。
  - 4-wave 128×128 方形 = 强度 **64**
  - 8-wave 128×64 长条 = 强度 **42.7**
- 强度高 1.5× → 单位计算的 LDS ds_read 少 1.5×(rocprofv3 实测 8w/4w LDS insts = **1.49×** 精确匹配)→ LDS 延迟更易藏。
- 这是 4-wave 长 K 领先 8-wave 的核心机制,单调随 K 增:**6.4%@K8192 → 10.5%@K28672**。

### fp8 big-K drain removal WIN(both-path-J)

- big-K(K 很大如 8192×8192×28672,长 K-loop,L2 已好)瓶颈 = A-side tr8 的 `s_waitcnt vmcnt(0)` 全排空(LLVM 对 intrinsic `ds_read` 保守插入)。
- **WIN = both-path-J drain removal**:`a_inline_asm=1,b_inline_asm=1`(A、B 都走 inline-asm `ds_read_b64_tr_b8`,opaque 给 SIInsertWaitcnts → 无自动 vmcnt(0) drain)+ `asm_mma=2`(绕过 agpr guard,MMA 仍 intrinsic)+ `agpr_alloc=0` + `vmcnt_hint=3`。配套的 asm-inplace MFMA(`asm_mma=2` wire `_asm_mma_do` mode2,D 别名 C in AGPR,`agpr_alloc=128`,消 accvgpr 拷贝 + spill 18→0):big-K **+2.3%,det0(3×2000 fresh)**,commit **1049c9ee**。
- `vmcnt_hint` 调到 det=0 上限:**vh=3 是 sweet spot**;vh=2 也 det=0 但更慢;big-N 的 det 上限也是 3。(注:此前版本记载的 424fe26b/2580→2756TF/2710-2734TF 数字系与 MXFP8 同名 skill 文件混淆,tensorwise fp8 源文件无此数据,已按 flydsl-fp8-gemm-tuning/SKILL.md 更正)
- **同 FLOPs 不同 shape 对照是金矿定位法**:big-N 慢就拿 big-K(同 FLOPs 同 kernel)对照 profile,差异指标(L2 51 vs 66)直接点出瓶颈。

### wgrad ≠ DVFS 功耗受限(区别于 dense fp8)

- 实测跑核满频 2400MHz ~266W,远低于 dense randn 的 ~1072W 功耗墙。
- 因 MfmaUtil 仅 ~54% 够不到墙 → wgrad 是效率/autotune 优化**真能体现**的路径。
- 反面:dense/fwd/dgrad 的指令效率优化被功耗墙掩盖(相同 MFMA → 相同功耗 → 相同频)。

## mxfp4 4-wave 生产 emit 杠杆:GAVOID/MMORD/INPLACE_ALT/WLBARNOP/ELGK/WLVMCN/SCV_ILV

### 生产成功 emit 杠杆(按 med 增益)

| env var | 值 | 增益 | 机制 / WHY |
|---|---|---|---|
| `FP4_INPLACE_GAVOID` | 1 | +55T | g2s 避开 refill slot |
| `FP4_MMORD` | 9 | Llama 7b-qkv +2% | blocked-diagonal 4×8:块状对角把同-acc 的 2 K-sub 隔开,消累加器 RAW stall |
| `FP4_INPLACE_ALT` | 0 | +27T | B-side progressive,须配 `MMORD=5` |
| `FP4_WLBARNOP` | 1 | +21T | barrier 后插 1 个 s_nop |
| `FP4_INPLACE_ELGK` | 9 | +27T | barrier 处留 9 个 ds_read 在飞 |
| `FP4_WLVMCN` | 10 | +10T | — |
| `FP4_SCV_ILV` | 1 | +20T (min) | scale load 交织进 mfma 流 |

### race-correctness 边界(稳定 emit)

- `ELGK`:最优 9,`≥15` racy。
- `WLVMCN`:最优 10,`≥20` racy。
- barrier 是 race-critical:减 barrier(`LEANBAR`)必触发 operand race。
- cross-wave race 根因 = **LDS barrier 不足**,不是 vmcnt 乱序:g2s `buffer_load→LDS` 是 wave 协作完成,barrier 确保所有 wave 的 g2s 全部落地后,跨 wave 的 ds_read 才安全。

### authoring:VGPR-direct scale 补 refill-ahead lag

- 去 LDS ds_read、当场消费 scale 时,soffset 必须补 refill-ahead lag:
  `soffset = o_sca - AH*n_sub*256`,`AH=2`(refill-same,默认)/ `AH=1`(refill-OTHER)。
- WHY:LDS 路径经缓冲,延迟 `AH` 个 K-iter 才消费;VGPR-direct 当场消费须手动对齐这个 lag。
- 8w wholeloop:scale gmem 已是 per-lane 布局,consume-lane 天然对齐,**无需改 host 预处理**。

### authoring:PIN 机制(绕过 LLVM RAGreedy 卡死)

- 症状:`=&v` early-clobber 输出太多(如 2-set ping-pong / register double-buffer 的 +6 或 +96 个 `=&v`)触发贪心 RA 病态卡死(compile >130s 不返回)。
- 解法:把内联汇编的 `=&v` 输出换成 ISA 里显式 `v[pb:pb+3]` 物理寄存器字面量,`pb = PINBASE + 组偏移`。
- 对齐约束:PIN 下 scale VGPR 基址须对齐 —— `PINSC=1`(scale 在前)用 `PINBASE`,否则用 `PINBASE + 4*ntmp`;不对齐触发 SNR21 bug。

---
来源: flydsl-fp8-gemm-tuning/SKILL.md, 10-grouped-wgrad-4wave-3buf.md, gemm-optimization/SKILL.md, agpr_rawasm_progress.md, agpr_phase5_mono.md, 11-upstream-agpr-pin-moot.md, diag_4w_vs_8w.md, flydsl-fp8-gemm-results/SKILL.md, 03-emit-knobs.md, 10-8wave-scvgpr.md
