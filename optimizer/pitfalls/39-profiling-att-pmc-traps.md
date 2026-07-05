# profiling 坑：ATT 无 cache counter/PMC 多 pass 挂 GPU、code.json AGPR-blind、debug-info 假象

> 类别: 踩过的坑 · 主题标签: rocprofv3, ATT, PMC, debug-info, occupancy

## ATT vs PMC 分工（不能合一个 job）
- ATT (Advanced Thread Trace) 只给**逐指令 stall 时序**，**没有 cache counter**。要问 L2 命中率 / 32B-partial / over-fetch / HBM 效率，必须**单独跑 PMC**——PMC 和 ATT 不能塞进同一个 job。
- PMC 不需要源映射，可以保留 `FLYDSL_RUNTIME_ENABLE_CACHE=1` 提速。

## ❌ 别再试：多 counter 一个 PMC job（gfx942 挂 GPU）
- PMC 每个 job 必须保持**单硬件 pass**（≤ ~4 个 TCC counter）。把多个 counter 塞进一个 job 会强制 **multi-pass 收集**，在 **gfx942 实测触发 GPU Hang (HW Exception)**。
- 正解：拆成多个**单 pass job**（如 L2 组 和 EA 组 分开跑）。

## ATT 常见错误处置
- 空 `ui_output_agent_*` → `kernel_include_regex` 没匹配上，重查 kernel 名。
- `no source mapping` → 确认 `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`。
- trace 截断 → `att_buffer_size` 升到 `0xC000000`。
- `INVALID_SHADER_DATA` → aqlprofile / decoder 版本不匹配，需同时更新。
- `iteration_range` 不匹配 → 试 `"[0,[1-2]]"`。

## ❌ 别再试：靠 code.json 反汇编算占用率（AGPR-blind）
- `code.json` 只含**单 CU、常是 vgpr-form 的反汇编**，无法给出 accum_vgpr / LDS / SGPR / workgroup size。**AGPR-form-blind 的 ISA 扫描会报 `accum=0`**，从而占用率算错。
- 正解：读旁边 staged 的 `out_kernel_trace.csv` 拿权威 `Accum_VGPR_Count` / `LDS_Block_Size` / `SGPR_Count` / `Workgroup_Size`；`arch_vgpr` 取 `max(ISA_scan, CSV)` 防 CSV 低报。

## ❌ 别再试：只加 `-g` flag 想拿 ATT 源码映射
- 光有 `gpu-module-to-binary` 的 `-g` flag 没用：`-g` 只保留 debug info 但**没东西可保留**——`loc()` 元数据在 MLIR→LLVM-IR 翻译时被**静默丢弃**。
- 正解：先跑 `ensure-debug-info-scope-on-llvm-func{emission-kind=LineTablesOnly}` pass（位置在 `reconcile-unrealized-casts` 之后、`gpu-module-to-binary` 之前），把 MLIR `loc()` 转成 LLVM `DISubprogram`/`DICompileUnit`。配好后 PA decode kernel 达 **99.9% 覆盖（1109/1110 指令）**。
- 环境变量要求（`FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`）见 methodology/12-att-trace-mfma-stall.md。

## ❌ 别再试：把 @flyc.kernel 装饰器行当热点（debug-info 假象）
- ATT 中热点若**塌陷到 `@flyc.kernel` 装饰器行**、且 stall 类型是**混合 VMEM-wait + barrier**（Pattern5）——这是 **debug-info 聚合假象**：MLIR/编译器生成指令（地址算术、cndmask、prologue）被映射到最外层 scope 行。
- 正解：忽略此行，只看**有显式用户 op** 的行。

---
来源: capture-kernel-trace/SKILL.md, kernel-trace-analysis/SKILL.md, programming-model.md, flydsl-kernel-authoring/SKILL.md
