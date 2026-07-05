# whole-loop 4-wave 结构与 LDS-feed bound:byte-exact LDS 账、bank conflict 与算术强度

> 类别: 方法论 · 主题标签: whole-loop-asm, lds-feed-bound, bank-conflict, 算术强度

## 4-wave whole-loop 结构(wgrad)

- occ=1(512 VGPR = 256 操作数 + 256 AGPR 累加器,accum_offset=256),256×256 tile,2×2-wave,两操作数都 transpose-read,whole-loop 裸 asm 主循环,AGPR 累加,CShuffle store。
- 根因瓶颈 = LDS 转置读 feed 带宽(TN 固有税),**非** 占用率/延迟/bank。即使 racing 也只有 fp8 峰值约 44%。
- 主循环 = 两个 ping-pong 相位,相位间一道 `s_waitcnt;s_barrier`。一个 wave 的 transpose-read 要 gather 所有 wave 写的 chunk(`W*chunk_stride`)→ barrier+drain 是跨 wave LDS 一致性硬需求,per-wave 局部 drain 不安全。

## byte-exact LDS 账(2 池 3buf 塞 160KB)

- `_CS=1024`(省 5120B bank-pad) + scalar store(省 8704B C_lds,`PT_WL_2BPOOL` 自动触发)→ 10 buffer × 16384 = 163840 恰好塞满 160KB。
- `_CS` 被双向锁死 = 1024:
  - buffer 需放 16384B → `_CS >= 1024`
  - 10 buffer <= 163840 → `_CS <= 1024`

## LDS bank conflict(_CS=1024 的税)

| 配置 | LDSBankConflict (rocprofv3) | MfmaUtil |
|---|---|---|
| 1 池 @1056 | 0% | — |
| 2 池 @1024 | 14% | 54.3(>50 净胜) |

- `_CS=1024` = bank 周期整数倍 → transpose read 必冲突,但 MfmaUtil 净胜,2 池仍取胜。
- 消 14% 冲突唯一路 = 操作数改 `a_plain`(预转置走 `ds_read_b128`,无 transpose 固定域冲突),但需 upstream 量化产转置副本、跨文件大改。
- `a_plain` 在 kernel 内**零吞吐收益**(gfx950 transpose-read 零 per-op 惩罚),唯一价值是免 pad + 免冲突的 enabler。

## 算术强度:4-wave 长 K 为何领先 8-wave

- 强度定义 = 每 K-step 的 MAC / (A行 + B列 operand-elem)。
  - 4-wave 128×128 方形 = 强度 **64**
  - 8-wave 128×64 长条 = 强度 **42.7**
- 强度高 1.5× → 单位计算的 LDS ds_read 少 1.5×(rocprofv3 实测 8w/4w LDS insts = **1.49×** 精确匹配)→ LDS 延迟更易藏。
- 这是 4-wave 长 K 领先 8-wave 的核心机制,单调随 K 增:**6.4%@K8192 → 10.5%@K28672**。

## wgrad ≠ DVFS 功耗受限(区别于 dense fp8)

- 实测跑核满频 2400MHz ~266W,远低于 dense randn 的 ~1072W 功耗墙。
- 因 MfmaUtil 仅 ~54% 够不到墙 → wgrad 是效率/autotune 优化**真能体现**的路径。
- 反面:dense/fwd/dgrad 的指令效率优化被功耗墙掩盖(相同 MFMA → 相同功耗 → 相同频)。

---
来源: 10-grouped-wgrad-4wave-3buf.md, diag_4w_vs_8w.md, flydsl-fp8-gemm-results/SKILL.md
