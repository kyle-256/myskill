# RDNA/WMMA 移植坑：别导 CDNA 调度旋钮，WMMA 代际差异限于 4 处

> 类别: 踩过的坑 · 主题标签: rdna-porting, wmma, autotune-dispatch, stochastic-rounding

- ❌ 别再试：把 CDNA 调度旋钮导入 RDNA。`sched_mfma`/`s_setprio` 的比例是 MFMA/wave64 专用的；WMMA 是 wave32，正确的 interleave 完全不同，照搬 CDNA 比例只会打乱调度。
- **WMMA 代际差异只集中在 4 处**（其余代码路径 CDNA/RDNA 共用，别到处改）：
  1. **`_wmma_op` call**：gfx11 是 v16 operands，lanes 16-31 镜像 lanes 0-15；gfx120x 是 v8 operands。
  2. **LDS-read shape**：读法随代际变。
  3. **accumulator store-back row 公式**：用错公式会**静默地把输出行转置**（不报错，结果错），最难查。
  4. **barrier asm**：gfx11 = `s_waitcnt lgkmcnt(0)` + `s_barrier`；gfx12+ = 拆成 `s_barrier_signal` / `s_barrier_wait` / `s_wait_dscnt` 三条。
- **内层 WMMA 循环保持 'load all B, then 1 A -> reg_n WMMAs'**。❌ 别再试反转成先 load A：反转会膨胀寄存器压力并 spill。
- **CDNA vs RDNA 判定的单一真相是 `is_rdna_arch()`**（`python/flydsl/runtime/device.py`）。❌ 别再试硬编码 `gfx*` 条件。
  - `wave32-true` 只匹配 `gfx10*`/`gfx11*`/`gfx120*` 前缀，**不匹配 `gfx1250`**。
  - wave size 共享逻辑：`get_warp_size(arch) = 32 if is_rdna_arch else 64`，因此 **gfx1250 返回 64**——gfx1250 kernel 须自己显式设 wave32，否则 wave size 错。
- 硬件 stochastic-round FP8 转换（`V_CVT_SR_*`）须自己驱动 PRNG：见 37-cdna4-no-fallback-porting.md #硬件 stochastic-round FP8 转换必须自己推进 PRNG

---
来源: optimization-directions.md, FlyDSL/CLAUDE.md, gfx942/kernel-implementation-notes.md
