# 10 — 8-wave wholeloop scale-VGPR 直读（SC_VGPR8）探索

> 目标：把 8-wave wholeloop 的 scale 从 LDS 移到 VGPR-direct，砍掉 LDS 端口访问，
> 冲破 ds_read 带宽墙（见 09）。**结论：VGPR-direct scale 正确性已攻克，但要成为
> 性能净赢必须先给 `call_mxfp4_wholeloop_8w` 接 PIN（否则 2-set 预取触发 LLVM RA 卡死）。**
> 日期 2026-06-25。Kyle。

## 动机（接 09 的三道墙）

09 证明 8-wave 被 **ds_read 带宽**锁死（NODSR=6145 vs 实测 4710，暴露 ~1438T）。
`call_mxfp4_wholeloop_8w` 每 K-iter 每 wave 的 LDS 访问：
- operand：24 × `ds_read_b128`（A 2 region×4 tile×2 sub=16，B 2 half×2 tile×2 sub=8）
- scale：**6 × `ds_read_b32`**（3 stream：A-r0/A-r1/B-comb，各 n_sub=2）
- 合计 30 次 LDS 访问/K-iter；scale 占 **20% 访问数**、但只占 ~6% 字节（6×4B vs 24×16B）。

若 LDS 端口是**按访问率**饱和 → 去掉 6 个 scale 访问可回收 ~20%×1438 ≈ 288T → ~5000T；
若按**字节带宽**饱和 → 只回收 ~6% ≈ 85T → ~4795T。这是值得一试的杠杆，且**生产快速
wholeloop 从未用过 SC_VGPR**（SC_VGPR 只活在另一条慢/racy 的 INPLACE builder，
`FP4_SC_VGPR` 对 `mode=wholeloop` 完全无效——实测数字+SNR 与基线逐位相同）。

## 关键发现：基础设施已半就位 + per-lane 映射天然对齐

`call_mxfp4_wholeloop_8w` 签名**已经 plumb 了** `sc_voff/sc_rsa/sc_rsb`
（builder 内 `i_scvoff/i_scrsa/i_scrsb`），但 builder 从不引用——只走 scale-LDS。

**per-lane 映射无需新 host 预处理**：scale 的 gmem 布局已是 per-lane（`i_scvoff`
= `lane_id*4` 是 per-lane voffset）。当前 LDS 路径里 lane L 把 `gmem[i_scvoff[L]]`
经 g2s 写到 `LDS[..+L*4]`，再 `ds_read` lane L 读回**自己**的 dword。VGPR-direct
让 lane L 直接 `buffer_load gmem[i_scvoff[L]]` 进 VGPR——**同一 lane 消费同一 dword，
映射逐位一致**。比 4-wave 的 SC_VGPR（需要 `preshuffle_scale_lane_contig`）简单得多。

## 正确性攻克：读取位置 LAG 修正（这是唯一的坑）

第一次实现 SNR=-1.3dB（scale 全错）。根因：`o_sca` 追踪的是 g2s **写入(refill)位置**，
调用方（mxfp4_gemm_8wave.py ~3534）把 soffset 初值**前移 `AH*N_SUB`**
（`AH = 2` refill-same 默认 / `1` refill-OTHER），因为 prologue 已预填 buf0(k0)+buf1(k1)。
LDS 路径里 scale 经缓冲延迟 AH 个 K-iter 才消费，所以对；VGPR-direct **当场消费**，
必须 `soffset = o_sca - AH*(n_sub*256)` 才命中当前 k。

修正后 **SNR 55.6dB / det 0**（与基线一致）→ VGPR-direct scale 正确性确认。
```python
_AH8 = 1 if FP4_WLOTHER else 2
buffer_load_dword $t_sc[slot], $i_scvoff, $rsrc, ($o_sca[grp] - _AH8*n_sub*256) offen offset:{s*256}
```

## 为什么还不是性能赢：两条路都被堵

### 单 set + `vmcnt(0)` → 过度 drain（4362 < 4710 基线）
scale 当场消费需在 MFMA 前 `s_waitcnt vmcnt(0)` 确保落地。但 scale buffer_load 是当前
phase 最新的 VMEM，要等它必须 `vmcnt(0)`，这会把 operand g2s 的 2-ahead 流水也一起
drain 掉 → 损失 ~350T，盖过 scale-LDS 的收益。**单 set 物理上必然过度 drain**
（要消费的 scale 永远是最新 VMEM，无法选择性等待）。

### 2-set ping-pong 预取 → LLVM RA 卡死（compile-only 实测：PRE-COMPILE 打印、
POST-COMPILE 不打印，`flyc.compile` >130s 不返回）
正确做法是 phase t-1 预取 phase t 的 scale 到 ping-pong 的另一个 set，让 phase 边界的
relaxed vmcnt 自然 drain。实现了（2 set、`emit_mm(sc_set)`、prefetch ahead=1、
可调 `FP4_SCV8_VM`），但 **+6 个 `=&v` early-clobber 输出把贪心 RA 推入病态**
（与 skill 05 记录的 "SCVGPR 无 PIN → RAGreedy crash/VGPR 溢出" 同源）。
单 set 能编译（~7s），2-set 卡死。

## 下一步（若继续攻 SC_VGPR8）

**唯一可行路径 = 给 `call_mxfp4_wholeloop_8w` 接 PIN**（physical VGPR 编号，绕过 RA），
照搬 `call_mxfp4_wholeloop`(line 799) 的 `FP4_PIN/PINSC/PINBASE` 机制。这是大改、高风险
（所有 operand/scale temp 改成 `v[N:M]` 物理寄存器），但能同时解锁 2-set 预取。
预期上限仍只到 ~4795–5000（取决于 LDS 端口是字节带宽还是访问率限制——本次未能测定，
因为 ping-pong 没跑起来）。

## ⚠️ PMC 实测推翻"LDS 带宽 bound"（修正 skill 09）

用 rocprofv3 PMC 直接量了生产 8w wholeloop（8192³ K28672, GPU7）的 LDS 端口占用：

| 计数器 | 值 | 含义 |
|--------|-----|------|
| `SQ_VALU_MFMA_BUSY_CYCLES` | 9.40e8 | MFMA 执行周期（=5.87e7 条 MFMA × 16cyc，核对 ✓）|
| `SQ_LDS_IDX_ACTIVE` | 9.93e7 | LDS 端口忙周期 |
| `SQ_INSTS_LDS_LOAD` | 2.75e7 | LDS 读指令数（=1024WG×8wave×30读×112iter ✓）|
| `SQ_WAIT_INST_LDS` | 9.43e7 | 等 ds_read（lgkmcnt）的周期 |
| `SQ_WAIT_ANY` | 1.59e8 | 全部 wait 周期 |
| VGPR_Count / Accum / Scratch | **128 / 0 / 0** | rocprof 权威，LDS 152KB/wg, 8 wave |

**关键比值**：
- `LDS_IDX_ACTIVE : MFMA_BUSY = 1 : 9.46` → LDS 端口只在 MFMA 忙的 ~10% 时间活动。
  即使按 b128 数据返回最悲观折算（6.8 cyc/load → 1.87e8），也只 **1:5，端口有 ≥5× 余量**。
- `SQ_WAIT_INST_LDS / SQ_WAIT_ANY = 59%` → 等待主要花在 ds_read 上，但那是**延迟**，不是端口饱和
  （`WAIT_INST_LDS ≈ LDS_IDX_ACTIVE`，等的时间 ≈ 端口活动时间，典型的延迟暴露而非带宽打满）。

**结论：8-wave wholeloop 不是 LDS 端口带宽 bound，而是 ds_read 延迟气泡 + MFMA 执行 bound。**
NODSR 的 1438T gap = **operand 就绪延迟气泡**（occ=2 下 2 waves/SIMD 盖不住 30 读/iter 的深
ds_read 延迟链），同 4-wave skill 08 的 "MFMA operand bubble"。这**直接否定了 SC_VGPR 方向**
（去掉 20% LDS 访问对 5× 余量的端口无意义）和 skill 09 的"LDS-bw bound"叙事。

## ⚠️⚠️ ISA 实测推翻"VGPR 有余量"——register-prefetch 也是死路

`FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=...` dump 生产 8w wholeloop 的 `21_final_isa.s` 元数据（**权威**）：

```
.amdhsa_next_free_vgpr 256     .vgpr_count: 256     .agpr_count: 0     .vgpr_spill_count: 0
```

- **rocprof PMC 报的 `VGPR_Count=128` 是误报**；真实 = **256 VGPR**（= occ=2 天花板 512/2）。
  核对：32 vec4 accs(128) + 24 operand frags(96) + 6 scale + ~26 地址 temp ≈ 256，完全吻合。
- **8w wholeloop 已占满 256 VGPR，零余量。** register-prefetch(+96 第 2 套 frag)→352>256→occ
  掉 1→512 线程 WG 无法单 CU 驻留→**起不来**。**naive register-prefetch 架构性死路。**
- **AGPR 救不了**：CDNA occ=2 下 ArchVGPR+AccVGPR 共享 256 组合预算（`accum_offset 256`），
  accs 搬 AGPR 不增加总量。这正是 8w docstring "can't hold AGPR for 32 accs under occ=2" 的根因。

## 结论：8-wave 所有杠杆被硬证据穷尽判死，PIN 不必做

| 杠杆 | 判死证据 |
|------|---------|
| SC_VGPR（减 LDS scale 访问） | PMC：LDS_IDX:MFMA_BUSY=1:9.46，端口 ≥5× 余量，减访问无意义 |
| register-prefetch（藏 ds_read 延迟） | ISA：VGPR 256/256 占满，occ=2 组合预算锁死，加寄存器即掉 occ |
| AGPR offload accs 腾 VGPR | CDNA occ=2 下 VGPR+AGPR 共享 256 预算，搬 AGPR 不增总量 |
| LDS padding 消 bank conflict | PMC：`SQ_LDS_BANK_CONFLICT=0`/`ADDR_CONFLICT=0`/`UNALIGNED_STALL=0`，本就无冲突 |
| 调度（WLDSR/staggered lgkmcnt/barrier） | skill 09：全部扫描在 5400±10 噪声内 |
| occ=3 | 8-wave WG=8 waves 必驻单 CU=2 waves/SIMD，结构上 occ 锁死=2 |

→ **PIN 移植的唯一目的（解锁这两条）架构上不成立，不做。** 8w 在 occ=2 / 256VGPR / LDS-非瓶颈
下，ds_read 延迟气泡已被第 2 个 wave 尽量盖住，4710 接近其物理极限。**确认 skill 09 的三墙天花板
~4900**。>5200 的答案是 **4-wave（occ=1，VGPR 预算 512，故能 register double-buffer/_RING →
已实测 5351）**。

## 调度层杠杆也被实测穷尽判死（PIN/SS/INPLACE/WLDSR 全做了）

应 user 要求继续死磕，移植了 PIN 并实现了 sub-stream，实测三种独立的读延迟隐藏调度，
全部 ≤ baseline，**交叉印证 occ=2 的粗单同步已是最优**：

| 方案（K28672 8192³ GPU7）| min/med TF | 正确性 |
|------|------|------|
| **baseline**（emit_ds 全读→lgkmcnt0→64 mfma，粗单同步）| **4744/4690** | SNR55.6/det0 ✓ |
| FP4_8W_PIN=1（accs v0-127 + frags/scale 钉死，绕 RA）| 4744/4690 | ✓ 性能中性 |
| FP4_WLDSR=1（emit_phase_dsr 细 staggered lgkmcnt）| 4575/4466~4642 | ✓ 更慢 |
| FP4_8W_SS=1（sub 分组：发 sub0+sub1 读→lgkmcnt(15)→mm sub0 盖 sub1 读→barrier→mm sub1）| 4645/4583 | SNR55.6/det0 ✓ 更慢 |
| FP4_INPLACE=1（next-K in-place 预取，relaxed vmcnt16）| 4789/4761 | SNR42.8/**det716464 竞态** ✗ |
| FP4_INPLACE=1 WLVMCN=0（全 drain）| 4728/4701 | ✓ 但加速没了 |

**根因（决定性）**：occ=2 下第 2 个 wave 用 wave-switching 已最优盖住 ds_read 延迟。把单一粗同步
点拆细（WLDSR/SS）只会**约束 wave-switching 自由度→暴露更多 stall**。唯一更快的 INPLACE-relaxed
读的是正被 g2s 写的 other-buffer（2-buffer slack 不够）→**必然竞态**；要正确就全 drain→加速消失。
SS 之所以不竞态（读同 buffer 稳定 LDS 的下一 sub）但也不快——同步点变多压过了延迟隐藏收益。

**PIN 移植成功（机制可用、SNR55.6/det0/性能中性），但其目的(register-prefetch)无 VGPR 余量、
SS 不需要 PIN 也不快 → PIN 对 8-wave 无实际价值。实验代码已全部回退，保持 kernel 生产纯净。**

## 最终裁定：8-wave 架构封顶 ~4690-4760，5200 必须走 4-wave(5351)
寄存器(256/256满)、LDS(5×余量)、bank conflict(0)、occ(锁2)、调度(粗单同步已最优) —— 五道墙全部
实测闭合。8w 的 ds_read 延迟气泡是 occ=2+零空闲寄存器下不可约的。

## （历史）曾以为的真正杠杆 + 共同前置 = PIN —— 已被上面 ISA 证伪

- 既然瓶颈是 ds_read **延迟**且 LDS 端口有余量、VGPR 有 ~128 余量（128/256，occ 已被 152KB
  LDS 锁到 1WG/CU，加 VGPR 到 256 不损 occ）→ **register-prefetch（读超前 1 个 K-iter 进
  第 2 套 frag 寄存器）藏延迟**是指向性最强的杠杆，预期能吃掉部分 1438T bubble。
- **但 register-prefetch（+96 `=&v`）和 scale-2set（+6 `=&v`）都会触发同一个 LLVM RA 卡死。**
  → **PIN（physical VGPR 编号绕过 RA）是这两条杠杆的共同、唯一前置**。下一步啃 PIN。

## PIN 移植规划（关键：4-wave builder 已有完整可参照实现）

`mxfp4_gemm_8wave.py` 里有**两个** wholeloop builder：
- `call_mxfp4_wholeloop` (L799, `_NWc=4`) = **4-wave**，**已实现** `FP4_PIN`/`FP4_PINSC`/
  `FP4_TRB8`/`FP4_SC_VGPR`/`FP4_PINBASE`，且 L1648-1650 记录了 PIN 下 scale 字面量对齐的
  SNR21 bug 修复（scale VGPR 基址须 = PINBASE + 4*ntmp 当 frags 在前；PINSC=1 则 scale 在前用
  PINBASE）。PIN 机制 = 把 `=&v` 输出换成 ISA 里显式 `v[pb:pb+3]` 物理寄存器字面量
  （`pb = PINBASE + 组偏移`），绕过 LLVM RAGreedy。
- `call_mxfp4_wholeloop_8w` (L2257, `_NW=8`) = **8-wave**，dispatch 在 L3528，**缺 PIN**——
  这是 SC_VGPR8/register-prefetch 两条杠杆都卡在 RA 的根因。

**移植要点**（4-wave→8-wave，搬已验证机制而非从零造）：
1. 操作数 frag、scale temp 全部从 `=&v` 编号改为 PINBASE 起的显式 `v[...]` 物理寄存器。
2. 8-wave acc 占 128 VGPR（NT=32 vec4），PINBASE 须避开 acc 区；frag+scale 排在 acc 之后。
3. scale 字面量对齐照搬 L1648-1650 的修复（否则 SNR21）。
4. 移植后即可启用 register-prefetch（第 2 套 frag，+96 VGPR，128→224<256，occ 不变）来藏
   ds_read 延迟气泡——这才是 PMC 指向的真杠杆；SC_VGPR 仅顺带（端口本就不缺）。

## 沉淀的可复用结论

1. **`mode=wholeloop` 的 scale 永远走 LDS**；`FP4_SC_VGPR`（老 INPLACE builder 的旋钮）
   对它无效。生产快速路径 = `call_mxfp4_wholeloop_8w`（line ~2257）。
2. **8w wholeloop 的 scale gmem 已是 per-lane 布局**，VGPR-direct 无需改 host 预处理，
   consume-lane 天然对齐——但必须减掉 `AH*n_sub*256` 的 refill-ahead lag。
3. **scale-VGPR 在 8w wholeloop 已验证 det0/SNR55.6**（单 set），机制本身正确。
4. **瓶颈是 RA**：任何给该 builder 增加 `=&v` 输出的改动（2-set、register double-buffer）
   都会触发 LLVM 贪心 RA 病态卡死，除非先上 PIN。这解释了 09 里 "藏延迟/双缓冲全失败"。
5. 干净基线 wholeloop = **4719 min / 4663 med**（再次确认，单 GPU cache-off）。
