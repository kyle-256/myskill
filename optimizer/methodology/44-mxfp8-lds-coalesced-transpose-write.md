# MXFP8 dual-cast 带宽 roofline + LDS-合并转置写(制胜招)+ per-shape tile 选配

> 类别: 方法论 · 主题标签: mxfp8, dual-cast, 带宽roofline, LDS-coalesced, 转置写, grid_swz, tile选配

## 带宽 / 瓶颈定位
- dual-cast quant 流量 = `4.0625*M*K` bytes(读 2MK bf16 + 写 2MK fp8 + scale)。
- 分相探针:读+行写 ~7.5 TB/s;转置**列写**散写崩到 ~1.5 TB/s → 唯一瓶颈是 `AtQd[K,M]` 转置写的**合并度**(每 K-行的 M-run 太短)。
- 判断带宽受限的判据:移除 barrier / 占用 / 计算(SK env)均无效 ⇒ 计算非瓶颈。
- 节点 chi2810 = gfx950 ×8,HBM3e ~8 TB/s 峰值,**实测 1R:1W copy 上限 ~6.3 TB/s**;dual-cast 达 3.2-4.6 TB/s = 真上限的 55-73%。

## LDS-合并转置写(制胜招)
- 根因:列写 run 长度 = `bm`(每 K-行 tile 内连续 M 字节)。旧 `BM=32` 列写 run 仅 32B。
- 做法:`bm×bk` 大 tile + LDS 暂存 fp8 **数据**(不暂存 scale),`barrier` 后**全 512 线程重读 ldsc**,每线程 1 次 `vec4`(16B)合并写 `AtQd`(合并宽度 = bm 字节、指令比 scalar 砍 4×)。
- `pad_extra=4` 错开 32-way LDS bank 冲突(多 shape +3-7%)。
- `grid_swz` 列-major pid 抗 DRAM channel camping。
- 实测 vs 旧 BM=32:每 shape **1.10-1.48×**,geomean ~1.25×,峰值 **5.68 TB/s**,无退化。

## per-shape tile 选配(`_qdual_tile_cfg(M,K,Mp,Kp)`)
- `Kp>=8192 且 Mp<=4096` → `(bm=128, grid_swz)`;否则默认 `bm=64, bk=128, pad_extra=4`(大 M / K=11008 全胜从不退化)。
- `bm=128, bk=64, pad_extra=4` 用于 12288×4096(5.68);再 `grid_swz=True` 用于 4096×11008(5.36, 1.48×)、4096×4096。
- 列/行写宽度权衡:`bm=128,bk=64` 列写=整 128B cache line、行写=64B;`bm=64,bk=128` 列写=64B、行写=128B(**默认最稳**)。

## partial-tile tail(非 256 倍数支持)
- 数据 G2S 用 buffer resource(SRD)OOB→0;scale 读用 `create_buffer_resource(num_records)` OOB→0;输出写用 `make_row_band_resource` + `col<c_n` mask 钳越界。
- 只需 M/N 是 **64 的倍数**即可安全,**256 倍数约束已解除**。

## 节点 GPU 占用检查
- gpt_oss2 用 GPU 4 或 5(`mem_get_info` 现查占用)。
- 检查占用:`docker exec mlperf_gptoss2 bash -c "rocm-smi --showuse | grep 'GPU\[4\]\|GPU\[5\]'"`。

---
来源: mxfp8-8wave-devloop/SKILL.md
