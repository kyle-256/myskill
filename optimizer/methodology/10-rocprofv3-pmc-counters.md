# rocprofv3 PMC counter:L2/HBM 效率、LDS 带宽 vs 延迟、哪些计数不可信

> 类别: 方法论 · 主题标签: rocprofv3, PMC, L2/HBM, LDS-bound

## 命令基本盘
- 新版 rocprofv3 **必须加 `--output-format csv`**,否则输出 `.db`。
- 模板:`rocprofv3 --pmc LDSBankConflict MfmaUtil --output-format csv -d /tmp/rp -- python3 script.py`。
- counter collection 输出在 `<output_directory>/pass_1/<output_file>_counter_collection.csv`。
- 可用 counter 名随 ROCm 版本变(拼写会漂),用 `rocprofv3 --list-avail` 查、以本地输出为准。
- FlyDSL 汇总设施:PMC 文件模板 `turbo/mxfp4_prof_pmc.txt`,汇总 `turbo/prof_summary.py`。

## 关键 counter 家族
| 目的 | counter |
|---|---|
| L2 复用/coalescing | `TCC_HIT_sum`, `TCC_MISS_sum`, `TCC_REQ_sum`(+ TCP/TCC 面板) |
| HBM 读效率 | `TCC_EA0_RDREQ_sum`, `TCC_EA0_RDREQ_32B_sum`, `TCC_EA0_RDREQ_DRAM_sum`, `TCP_TCC_READ_REQ_sum` |
| MFMA 使用/效率 | `SQ_INSTS_MFMA`, `SQ_INSTS_VALU_MFMA_MOPS_*`, `MfmaUtil`(MFMA busy%) |
| VMEM/LDS 带宽浪费 | `SQ_INSTS_VMEM_*`, `SQ_INSTS_LDS`, `SQ_LDS_BANK_CONFLICT` / `LDSBankConflict` |
| Occupancy/资源 | occupancy report, scratch, `hipcc --resource-usage` |

## L2/HBM 三指标三决策(scripts/pmc_l2_analyzer.py)
输入 `pmc_l2` + `pmc_ea` 两个 counter CSV,参数 `--kernel --ideal-gb <每 dispatch GB> --ea-channels 2`。
- **L2 命中率** = TCC_HIT/(TCC_HIT+TCC_MISS) → 有无**时间复用**可挖。
- **32B fraction** = `TCC_EA0_RDREQ_32B / TCC_EA0_RDREQ` → 空间局部性/cache 线浪费。
  - ≈0% = 满 64B cache line、无空间浪费;**高 32B%** 才指向 scatter/misaligned,值得重构。
- **over-fetch** = 实取字节 vs `--ideal-gb` → 有无冗余取数。

## LDS 带宽 bound vs 延迟暴露(必区分)
- 量 `SQ_LDS_IDX_ACTIVE`(LDS 端口忙周期)**:** `SQ_VALU_MFMA_BUSY_CYCLES` 比值。
- 实测 8w wholeloop = **1:9.46**(端口只在 MFMA 忙的 ~10% 活动,≥5× 余量)→ **不是端口带宽 bound**。
- `SQ_WAIT_INST_LDS ≈ SQ_LDS_IDX_ACTIVE` 且占 `SQ_WAIT_ANY` 的 **59%** → 是 **ds_read 延迟暴露**而非带宽打满。
- 长 K 冒烟枪:`SQ_WAIT_INST_LDS/GUI`(LDS-wait 归一化,越低越好)、`MfmaUtil`(MFMA busy%)、`SQ_INSTS_LDS`(=算术强度)、`SQ_INSTS_VMEM`、bank_conflict、`GRBM_GUI_ACTIVE`(cyc/dispatch,时钟无关效率)。

## 不可信的计数(权威判据在别处)
- CSV `Accum_VGPR_Count` **恒报 0**。
- `VGPR_Count` 对 256-VGPR 内核也报 **128 等错值**(如 8-wave wholeloop 内核,真实 num_vgpr=256)。
  - 权威 VGPR/AGPR 必须用 `FLYDSL_DUMP_IR` 的 ISA `num_vgpr/num_agpr`。
- prof_summary 的 "MFMA busy %"(除以 GUI*4)对聚合计数**不成比例(>100%)** → 改用权威派生指标 `MfmaUtil` / `MeanOccupancyPerActiveCU`。
- 查 VGPR 分配的 rocprofv3 SQL(仅供参考,同样不足信):
  ```sql
  SELECT ks.KernelName, ki.arch_vgpr_count, ki.accum_vgpr_count
  FROM rocpd_kernel_dispatch kd
  JOIN rocpd_info_kernel_symbol ks ON kd.kernel_symbol_id=ks.id
  JOIN rocpd_info_kernel ki ON kd.kernel_id=ki.id
  WHERE ks.KernelName LIKE '%target%';
  ```

## PMC 环境彻底不可用的 fallback
- 某些环境 PMC **signal-6 崩**:崩在 torch GPU kernel(randint/contiguous-clone),CPU 生成数据也崩。
- 此时带宽只能**解析估算**:输出字节 / 暴露时间 ÷ HBM 峰值。

---
来源: 10-grouped-wgrad-4wave-3buf.md, 10-8wave-scvgpr.md, capture-kernel-trace/SKILL.md, kernel-trace-analysis/SKILL.md, tool-rocprof/SKILL.md, diag_4w_vs_8w.md, agpr_rawasm_progress.md, project_mxfp4_epilogue_store.md, prefetch-data-load/SKILL.md
