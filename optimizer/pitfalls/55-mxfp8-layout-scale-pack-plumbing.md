# MXFP8 B-comb sizing / col 已转置 [K,M] / scale_pack 来源歧义穿线

> 类别: 踩过的坑 · 主题标签: mxfp8, authoring-layout, scale-pack, dual-cast

MXFP8 authoring 布局 / 穿线卡。以下每条都带根因（WHY），改动涉及 C++/kernel/autotune 三层，改一处漏一处后半读 0 或读错转置。

- **[B-comb buffer sizing]** B-comb buffer 大小必须用 `cdiv(dim,256)*4` 组（grp 是块跨步，不是 `dim//64`）。
  - WHY：旧公式 `(M/64)*K128p*64*4` 漏掉 partial 256-block 的 grp → 后半读 0。
  - 正确 C++：`dwords = cdiv(M,256)*4*K128p*64*4`，断言从 `%256` 放宽到 `%64`。
  - **两文件都要改**：`quantization.cpp` 与 `quantization_meta.cpp`（meta 漏改 → shape 推断与实际 alloc 不一致）。
  - kernel 端：`nbytes = ((dim+255)//256)*4*K128*64*4*4`。

- **[col 操作数是已转置 [K,M]]** MXFP8 C++ dual-cast 的 col 输出是 `[K,M]`（已转置，**不是** `[M,K]`）。
  - 所以 `AtQd` 必须是 `[K,M]`。
  - WHY 验证要用 `M≠K` 的 shape 才看得出（M==K 时转置错误被掩盖）。
  - bwd grad_b：`NT(go_col[N,M], at[K,M]) → [N,K]`；`trans_b=True` 时 kernel 把 `at` 当 `[K,M]` 读、输出 `[N,K]`。

- **[scale_pack pack 来源歧义]** kernel 自己分不清 scale 是 packed 还是 broadcast → 由 **caller 按来源指定** `scale_pack`。
  - int32 透传（C++ quant 吐的 packed）→ `mxfp8_scale_pack(K)`。
  - raw 路径（preshuffle 是 broadcast）→ `1`。
  - 穿线链：`gemm_mxfp8_flydsl_kernel(..., scale_pack=1)` → `autotune/_get_mxfp8_launch`（cache key 含 scale_pack）；**无 env 开关**。
  - scale 原语（`MfmaScale16x16x128` / `ScaleS2R` / `ScaleBComb` / `mxfp8_scale_pack`）放共用 `gemm_helper.py`，勿在 kernel 重复。

- **[raw E8M0 K-packing 冲突]** ❌ 别再试：在 raw E8M0 上做 K-packing 免 preshuffle。
  - raw E8M0 布局 `[dim, m//32]` 里同一 microblock 的连续 K-block 是 stride-4（列 = `k*4+mb`）。
  - failing mechanism：dword/dwordx 无法把多 K-block 打进一个 load，跨 tile 又 row-separated → K-packing 必须 preshuffle。
  - 这与 WL 刻意选的 raw E8M0 免 preshuffle 设计**直接冲突**（要 K-packing 就得放弃免 preshuffle）。

- **[dead code review]** Bugbot 按无 dead code 标准要删：
  - `peek_mxfp8_cfg`（从未调用）。
  - `gemm_helper.as_i8_flat`（1D flat 版无调用者）。
  - ⚠️ `as_i8_flat` ≠ tensorwise 的 `_as_i8_flat`（后者保持 2D contiguous 且**在用**，不可互换）。

---
来源: mxfp8-8wave-devloop/SKILL.md, project_mxfp8_grouped_wgrad_wl.md, project_mxfp8_wholeloop_port.md
