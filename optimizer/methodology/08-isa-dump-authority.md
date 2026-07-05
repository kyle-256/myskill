# ISA dump 是寄存器/AGPR/LDS/occupancy 唯一权威:FLYDSL_DUMP_IR + 数指令

> 类别: 方法论 · 主题标签: isa-dump, register-pressure, occupancy, k-loop-stall

## 怎么产出 ISA dump
- env:`FLYDSL_DUMP_IR=1`(必需)、`FLYDSL_DUMP_DIR=/tmp/xx`(可选,不设默认落 `/root/.flydsl/debug/`)、`FLYDSL_RUNTIME_ENABLE_CACHE=0`(避免命中旧缓存)。
- **必须真正 RUN kernel**(`c(*args)`)才触发编译+dump;只 `_compile` 不够。
- 产物路径:`<DUMP_DIR>/kernel_<name>_0/21_final_isa.s`(如 `kernel_dense_tn_0/21_final_isa.s`、`kernel_gemm_0/*_final_isa.s`)。每个 pipeline stage 还产编号 `.mlir`,`21_final_isa.s` 是 lowered 后的最终 ISA。
- `FLYDSL_DEBUG_DUMP_ASM` / `FLYDSL_DUMP_ASM` 不被支持,会报 not-supported —— 只能用 `FLYDSL_DUMP_IR`。
- 相关 env:`FLYDSL_COMPILE_OPT_LEVEL`(默认 2,范围 0-3)、`ARCH`(覆盖架构)、`FLYDSL_DEBUG_ENABLE_DEBUG_INFO`(发 DWARF,验证时查 `final_isa.s` 有无 `.file`/`.loc` directive)。

## 为什么 ISA 是唯一权威(不是 rocprof)
- ISA 元数据是寄存器/LDS/Scratch 占用的**唯一可靠来源**:
  - `.set kernel.num_vgpr` / `.vgpr_count`、`num_agpr` / `.agpr_count`
  - `accum_offset`、`next_free_vgpr`
  - `group_segment_fixed_size`(LDS 字节数)
  - `private_seg_size`(Scratch/spill 字节)、`.vgpr_spill_count`
- rocprof PMC 的 VGPR_Count 误报细节见 methodology/10-rocprofv3-pmc-counters;扫寄存器/spill 前一律先 dump ISA,别信 PMC。

## 指令直方图(快速体检)
```
grep -c v_mfma 21_final_isa.s          # MFMA 数
grep -c scratch_load 21_final_isa.s    # spill 检测(>0 即 spill)
grep -c s_barrier 21_final_isa.s
grep -c buffer_load / ds_read / ds_write / buffer_store
grep -E "num_vgpr|num_agpr|vgpr_spill" 21_final_isa.s
grep -oE "s_waitcnt.*" 21_final_isa.s | sort | uniq -c | sort -rn   # 数 vmcnt(0)/lgkmcnt(0) drain
```
- 对齐参考实现(aiter / hipBLASLt)目标:**MFMA 数匹配参考、barrier 数 ≤ 参考**。

## 稳态 K-loop stall-free 判据
- 定位 hot loop:label `1:` → `s_cbranch 1b`,即两个 `s_barrier` 之间的循环体。
- stall-free 特征:
  - `v_mfma` 背靠背连发;
  - `buffer_load`(g2s 预取)与 `ds_read`(读 operand)精确塞进 MFMA 间隙;
  - 每迭代只 **1 个 `s_waitcnt vmcnt(0) lgkmcnt(0)`** + **1 个 double-buffer barrier**。
- 满足即 compute 已 **MFMA-bound、零浪费,无 asm 可榨** —— 此时优化方向应转到 occupancy/feed,而非循环体指令调度。

---
来源: flydsl-fp8-gemm-tuning/SKILL.md, 10-8wave-scvgpr.md, flydsl-kernel-authoring/SKILL.md, agpr_rawasm_progress.md, project_mxfp4_epilogue_store.md, gemm-optimization/SKILL.md
