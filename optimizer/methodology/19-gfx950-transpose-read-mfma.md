# gfx950 MFMA transpose load:DS_READ_B64_TR_*、intrinsic 保守 drain vs inline-asm

> 类别: 方法论 · 主题标签: gfx950, transpose-load, LDS, inline-asm

## DS_READ_B64_TR_* 指令族(gfx950 专属)
边读 LDS 边转置,免掉 ds_write + 置换地址的 ds_read,直接给 MFMA 准备转置布局的 A/B 操作数。

| 指令 | 数据类型 | 写出 VGPR |
|---|---|---|
| DS_READ_B64_TR_B16 | fp16 / bf16 | 2 VGPR |
| DS_READ_B64_TR_B8 | fp8 / bf8 | 2 VGPR |
| DS_READ_B64_TR_B4 | int4 | 2 VGPR |
| DS_READ_B96_TR_B6 | 6-bit | 3 VGPR |

- 硬性要求:EXEC 全 1;地址按数据大小对齐;≥64-bit 的 DS 访问要求偶对齐 VGPR(B96_TR_B6 例外)。

## intrinsic 保守 drain vs inline-asm(big-K 场景)
- 场景:big-K(长 K-loop、square-ish、L2 已经好)。此时瓶颈 = A-side tr8 的 `s_waitcnt vmcnt(0)` 全排空 —— LLVM 对 intrinsic 版 ds_read 插入保守的 vmcnt(0),把所有 in-flight vmem 全排空。
- 根因:SIInsertWaitcnts 看到 intrinsic ds_read 会保守自动 drain。
- WIN = both-path-J inline-asm drain removal:
  - `a_inline_asm=1` + `b_inline_asm=1`,用 `ds_read_b64_tr_b8` 的 inline-asm 版 → 对 SIInsertWaitcnts 是 opaque,不触发自动 drain。
  - 配套:`asm_mma=2` + `agpr_alloc=0` + `vmcnt_hint=3`。

---
来源: lds-optimization/SKILL.md, flydsl-fp8-gemm-tuning/SKILL.md
