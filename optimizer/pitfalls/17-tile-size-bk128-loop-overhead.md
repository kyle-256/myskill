# mxfp4 BK128 死路：loop 固定开销碾压 compact-LDS 收益

> 类别: 踩过的坑 · 主题标签: tile-size, mxfp4, BK128, loop-overhead

- ❌ 别再试 mxfp4 4-wave BK128：plain = 4597 med，远 << BK256 5401。根因是 loop 固定开销——BK128 要 224 iter vs BK256 112 iter，iter 数翻倍带来的 loop 固定开销碾压了 compact-LDS（更小 BK 换来的 LDS 占用/复用收益）。
- ❌ 别再试在 fly manual-emit 下压 BK128 loop 开销：fly manual-emit 路径下 BK128 的 loop 固定开销无法消除。aiter 能做是因为靠 hand-asm 手工消掉了这部分开销——对比：BK128 历史顶 4715 vs aiter 5635，差距即来自 hand-asm 的 loop 消除。
- ❌ 别再试 BK384 / BK512：有 `BLOCK_K % 128 == 0` 的 assert 约束；BK384/512 会 Assert 失败或 VGPR 溢出。

---
来源: 05-dead-ends.md
