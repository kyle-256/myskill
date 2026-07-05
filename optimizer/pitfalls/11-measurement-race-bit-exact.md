# SNR 掩盖低概率 race：只有 bit-exact 30000+ 次多跑能测出

> 类别: 踩过的坑 · 主题标签: race, correctness, measurement-noise, wgrad

- **SNR 会掩盖低概率 race**：0.17% ~ 1/30000 级别的 bit-flip 在 SNR 里几乎看不出来，SNR 数字正常不等于输出干净。唯一可靠的检测是 **bit-exact 多次跑**（`_race_wg.py`，需 **30000+ 次**）。低于这个量级根本采不到那个 flip。
- **任何改缓冲布局 / vmcnt 都必须重跑 race 测**：这两类改动直接影响跨 barrier 的写窗口，SNR 过了也可能已经在腐蚀输出。改完不重跑 `_race_wg.py` = 没验证。
- **racing 优势本质是不安全的跨 barrier 写**：`PT_WL_2BPOOL` 2 池 3buf 时开 racing（`PT_RACE_VM=1`），在 m4096 上 SNR 掉到 **53-54**，就是输出正在被腐蚀的信号。其机制是让 `vmcnt(16)` 把 4 个 pool 的全部 G2S 写放到跨 barrier 之外（约 **0.17% bit-flip**），这是不安全的加速。
- **安全 3buf 才是正解**：不要为 racing 的速度收益牺牲正确性；racing 的"优势"是拿正确性换来的假象。

❌ 别再试：靠 SNR 判断 race 是否存在。SNR=53-54 才暴露、正常 SNR 完全掩盖 1/30000 级 bit-flip，采样量不到 30000+ 次时假阴性。
❌ 别再试：`PT_RACE_VM=1` + 2 池 3buf 的跨 barrier G2S 写。m4096 实测 SNR 掉到 53-54，约 0.17% bit-flip，速度收益是以腐蚀输出为代价的。

---
来源: 10-grouped-wgrad-4wave-3buf.md
