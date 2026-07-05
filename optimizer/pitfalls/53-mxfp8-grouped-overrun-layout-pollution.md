# grouped MX gemm over-run 污染 + BLOCK_M=128 少启动块假象

> 类别: 踩过的坑 · 主题标签: mxfp8-grouped, over-run, SRD-layout, config-sweep

- **deterministic 测试查不出正确性**: deterministic 测试只验 run-to-run 一致性,consistently-wrong 也会"过"。必须跑对拍参考(SNR / bit-exact vs Triton/HIP)才能暴露污染。新增 pipeline 一定要测 `balance=False`(非均衡分布):`balanced=True` 常全过,`unbalanced` 才崩(如 wgrad over-run `b_grad_snr` 崩到 ~6dB)。WHY: balanced 时每组 M 对齐 tile 边界,over-run 不越界;unbalanced 才让越界 tile 读到相邻组。

- **MX wgrad 布局 over-run 污染**: 根因是算子布局差异。
  - tensorwise wgrad 算子 = `[M_total, OUT]`(token 外维),per-group 扁平 SRD `num_records = mg*OUT` 能把 over-read 硬件钳 0。
  - 但 MX wgrad 算子 = `[OUT, M_total]`(token 内维),SRD 用整张量边界 `OUT*mt`,扁平 `num_records` 无法按 `token < m_end` 逐行钳 0 → over-run tile 读到下一组真实数据被 MMA 累加污染。
  - **修法**: body 调用外包运行时守卫 `if k_abs < k_iters:`(`k_iters` 全 WG 一致,分支/barrier 均匀安全),`swap` 留守卫外保持 ping-pong 身份。

- **❌ 别再试 BLOCK_M=128 —— 是少启动块假象**: grouped MX gemm config sweep 里 `BLOCK_M=128` 显示 +9~35% / 反超 TW 是坑。
  - 机制: sweep 时 host `grid_upper` 写死 `/256`,`bm=128` 需 2× 块 → 只启动一半只算一半 tile → 假性变快。
  - 把 `grid_upper` 泛化成 `ceildiv(M_pad, bm) + G` 后,gemm-only `bm=128` 真实 **1.55x 慢**(1666 vs TW 2576)。
  - WHY: 256×256 tile 的 MFMA operand 复用在 compute-bound 形状更优。
  - **教训重申**: sweep/探针任何变快必须先确认 grid/tile 覆盖完整,或跑数值对拍。

- **❌ bm=128 需 nt_a=2 preshuffle 布局(确认死路已回退)**: `bm=128` 还需 A-scale preshuffle 发 `nt_a=2` 布局(32-row group、每 record 2 子块),否则 `ScaleS2R(n_tiles=2)` 只读一半 → 数值全错。已把 `_emit_lds_repack(nt=...)` / preshuffle / workspace / grid 全参数化打通验证,确认 `bm=128` 死路后全回退,dense/wgrad 保持 `nt=4`。

- **输入 band 收紧中性(依赖搬家没净减)**: grouped qa 主 kern 输入 buffer band 从 `[in_rebase, total_M)` 收成 `[in_rebase, RIE)` 靠 `num_records` HW-drop 删 `(grow < real_end)` select,实测中性。WHY: 依赖只是从 RE-load 换成 RIE-load,没净减少;grouped M-remap 的 SRD 本就要 runtime 组信息 gate 住 load,是结构性的。

---
来源: mxfp8-grouped-gg-devloop/SKILL.md
