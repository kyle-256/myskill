# 09 — 8-wave mxfp4 架构天花板（K28672, M8192N8192, gfx950）

> 目标：8-wave 冲 5200T。结论：**架构性封顶 ~4900T**，被三道独立的硬墙锁死，
> 全部实测验证。5200T 需要 1-wave/SIMD（即 4-wave，已达 5351）。日期 2026-06-24。
>
> ⚠️ **修正（2026-06-25，见 skill 10 PMC 段）**：本文若把瓶颈归为"LDS 带宽 bound"，已被
> rocprofv3 PMC 推翻。实测 `LDS_IDX_ACTIVE:MFMA_BUSY=1:9.46`（端口 ≥5× 余量），真瓶颈是
> **ds_read 延迟气泡**（occ=2 盖不住深 ds_read 延迟链，`WAIT_INST_LDS` 占 59% wait），
> 不是 LDS 端口带宽。封顶值与三墙结论不变，但根因表述以 skill 10 为准。

## 干净基线（单 GPU 顺序测，排除并行热降频）

| 配置 | med / min TF | 备注 |
|------|------|------|
| 8w pipe BN256 BK256 SWZ1 ASMM1 PRB1 | **4817 / 4855** | 当前最优 |
| 8w pipe BN256 BK128（同优化） | 4449 / 4510 | BK128 loop 开销 7.3% |
| 8w wholeloop BN256 BK256（INPLACE 死路） | ≤4842 racy | 不超 pipe |
| 4-wave BN256 BK256（对照） | 5351–5401 | occ=1 |

- profile：MfmaUtil 60–62%，LDS 128KB（占 1 WG/CU），VGPR 120，Scratch 0。
- **SWZ1 已把 LDS bank 冲突清零**（SQ_LDS_BANK_CONFLICT=0；SWZ0=0.75 ratio, MfmaUtil 44%）。
- 天花板探针（wholeloop BK256，base CONSTSC 4707）：**NODSR=6145**（ds_read 主瓶颈，
  暴露 ~1438T）、NOG2S=5400（g2s ~693T）。
- 所有 knob 在噪声内：MMORD/grouping(GM/GN/NX)/scale 变体/INPLACE/stage3/b_vgpr/asm_iter
  全部 ≤4900 或更差。早先看到的 4896/4910 是冷 GPU 假象。

## 根因：8-wave = "2 waves/SIMD" 把三道资源墙全部收紧一倍

8 waves/WG ÷ 4 SIMD = **强制 2 waves/SIMD**（这是 "8-wave" 的定义）。相比 4-wave 的
1 wave/SIMD，每 wave 的 LDS/寄存器预算砍半。

### 墙 ① LDS 读带宽 — A operand 4× 冗余（proximate cause）
- LDS-read/flop = `wn/BN + wm/BM`（wm·wn=8 wave grid）。
- 8-wave 2×4：A 行被 4 个 N-wave 各读一次（**4× 冗余**），B 列被 2 个 M-wave 读（2×）。
  → ds_read/flop = 4/256 + 2/256 = **0.0234**。
- **4-wave 2×2**：A 2× / B 2× → 0.0156。这就是 4-wave 快的本质。
- 改 tile 形状（2×4↔4×2）总读量不变（对称 shape）。8-wave ds_read-bw 饱和 → 卡 ~4820。

### 墙 ② LDS 容量 160KB — 装不下"大 tile 双缓"
- 要把 8-wave 降到 0.0156，必须加大 tile 到 **BN512 或 BM512**（LDS-read/flop=0.0156）。
- 但 BN512 BK256 双缓 = A 64KB + B 128KB = **192KB > 160KB**，放不下。
- 退路：BK128（96KB，但 loop 开销 7.3%，建模净零）或单缓 B（暴露 g2s）→ 均吃掉收益。

### 墙 ③ 寄存器 256@occ2 — 装不下"大 tile 的 64 accs"（DEAD，实测）
- 大 tile（BN512/BM512）= N_ACCUMS 翻倍 = **64 accs = 256 VGPR**。
- 8-wave 2 waves/SIMD → VGPR 预算 256/wave。64 accs(256) + operands(~64) + scale = **>256 → spill**。
- **实测 BN512 BK128（已实现 + 回退）**：VGPR 128 / **Scratch 1132（spill）** / **374 TF**
  （13× 慢，每 MFMA 读写 scratch=HBM）。
- 逃逸全部失败：`FP4_AGPR=1` 无效（AccVGPR 仍 0，asm_mma 路径此旋钮已坏，见 05）；
  `WPEHINT=2 MAXNREG=512` 无效（VGPR 仍 128，hints 不在 FlyDSL JIT 传播）。
- **4-wave 为何能放 64 accs**：1 wave/SIMD → 512 VGPR + 256 AGPR 预算，accs 进 AGPR、
  operands 进 VGPR，且 BN256 BK256 在 occ=1 下 144KB LDS 放得下。

## 为什么 read-once-register / wholeloop 也救不了
- read-once 不改变 accs 数（BN256 仍 32，ds_read/flop 仍 0.0234），且 8-wave 是
  **带宽-bound（非延迟）**，藏延迟无效。skill 05 记录 fly 反复没攻下（RAGreedy crash/VGPR 溢出）。
- wholeloop INPLACE 即便 racy 也 ≤4842 < pipe 4817~4855，不超 pipe。

## BN512 实现备忘（已回退，验证过的设计，留作参考）
若将来要再试 BN512：保持 2-half 象限不变，N_TILES_B=4（S2RLoaderFp4/StoreCPlain/
call_subs_asm 全按 ntb 参数化，ob=0 j∈0-3 即用）。唯一新代码 = B-comb scale 拆 2 dword
（ntb=4 每 half 4 byte=1 dword，2 dword/grp，loader vec_width=2，_sb 返回 (sp[s][0],0)/(sp[s][1],0)）。
**但墙③ 使其 spill-dead（374TF），不值得做。** 实现时 SNR=24.6 还有 scale layout bug 未查。

## 结论
- **8-wave 在 K28672 架构性封顶 ~4900T**；三道墙皆因 "2 waves/SIMD" 定义而起，不可绕。
- 要 5200：用 **4-wave（1 wave/SIMD，已达 5351）**，或换 1-WG-per-CU 的拓扑（即非 8-wave）。
- aiter 的高数（~5440）若是 8-wave，必走 compact-LDS(22KB) read-once 手写汇编 = fly 历史死路；
  无该参考则维持本结论。
