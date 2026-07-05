# SNR 掩盖低概率 race：只有 bit-exact 30000+ 次多跑能测出

> 类别: 踩过的坑 · 主题标签: race, correctness, measurement-noise, wgrad

- **SNR 会掩盖低概率 race**：0.17% ~ 1/30000 级别的 bit-flip 在 SNR 里几乎看不出来，SNR 数字正常不等于输出干净。唯一可靠的检测是 **bit-exact 多次跑**（`_race_wg.py`，需 **30000+ 次**）。低于这个量级根本采不到那个 flip。
- **任何改缓冲布局 / vmcnt 都必须重跑 race 测**：这两类改动直接影响跨 barrier 的写窗口，SNR 过了也可能已经在腐蚀输出。改完不重跑 `_race_wg.py` = 没验证。
- **racing 优势本质是不安全的跨 barrier 写**：`PT_WL_2BPOOL` 2 池 3buf 时开 racing（`PT_RACE_VM=1`），在 m4096 上 SNR 掉到 **53-54**，就是输出正在被腐蚀的信号。其机制是让 `vmcnt(16)` 把 4 个 pool 的全部 G2S 写放到跨 barrier 之外（约 **0.17% bit-flip**），这是不安全的加速。
- **安全 3buf 才是正解**：不要为 racing 的速度收益牺牲正确性；racing 的"优势"是拿正确性换来的假象。
- （源自 gpt_oss2 mxfp8 grouped GEMM 项目 `dev/kyle_mxfp8_gg_pr`，非本环境 wgrad 4wave 内核，仅作跨项目参考）**官方 deterministic pytest 是该项目的 race 验收口径**：以 `test_grouped_gemm_fp8_mx_blockwise_deterministic` 为准（`rtol=0`/`atol=0` + `empty_cache` churn、`repeats=10`）；d7b149a ×100 全过。全量 `pytest tests/pytorch/ops/test_grouped_gemm_fp8.py -k mx_blockwise` 期望 **3840/3840**。
- （同上，源自 gpt_oss2 mxfp8 项目，与本卡 wgrad 4wave 无关）**mxfp8 真实性能数字（安全实现，供参考）**：gpt_oss-20B Expert shape(B=4 M=2048 N=5760 K=2880) reg notes 修后 grouped fwd/wgrad 达 **106/0/0/0**；Perf(B=16 M=2048 N=4096 K=7168) **Fwd 1435 / Dgrad 1416 / Wgrad 2013 TFLOPS**，SNR **28.23 / 28.23 / 28.08 dB**。grouped MoE bench 第二类修前后基本持平：Fwd 941.02→934.67(**−0.67%**)、Bwd 1109.91→1099.66(**−0.92%**)。

❌ 别再试：靠 SNR 判断 race 是否存在。SNR=53-54 才暴露、正常 SNR 完全掩盖 1/30000 级 bit-flip，采样量不到 30000+ 次时假阴性。
❌ 别再试：`PT_RACE_VM=1` + 2 池 3buf 的跨 barrier G2S 写。m4096 实测 SNR 掉到 53-54，约 0.17% bit-flip，速度收益是以腐蚀输出为代价的。

---
来源: 10-grouped-wgrad-4wave-3buf.md, gfx950-vmcnt-race-debug/SKILL.md
