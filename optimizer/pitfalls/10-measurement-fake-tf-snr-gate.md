# 虚高 TFLOPS 假象：SNR<0 跳过计算 TF 虚高、超 peak 数字、do_bench 不可靠

> 类别: 踩过的坑 · 主题标签: measurement-noise, snr-gate, do-bench, shape-alignment

**核心铁律：高 TF 数字必须先过 SNR/det gate 才算数。** 任何超 peak 或异常高的 TFLOPS 在过 gate 前一律当 bogus。

## SNR<0 编译器跳过真算 → 时间短 → TF 虚高（mirror 死坑）
- SNR<0 时编译器把真实计算优化掉 → kernel 时间短 → TF 虚高。
- **FEWOP=1**（用单 reg 测 operand 多样性天花板）得 **5648 TFLOPS**，但 SNR garbage。
- **SCDWX4 / TRB8 的 '5640'** 同样是 SNR<0 假象；一旦计算正确，实际 **<5176**。
- ❌ 别再试：把这些 5648/5640 当天花板参考——它们是跳过计算的产物，不是可达性能。

## '跳过整条指令测天花板'类探针不可信
- **PT_TR_HALF**（跳过读）之类探针不可信：跳过读 ≠ 换成更少的等效读。
- 真实替换后（`ds_read_b128` 换 2×`tr-b8`）因带宽受限，收益归零。
- ❌ 别再试：靠删指令测'去掉 X 的天花板'。测'去掉 X'必须用**真实替代指令**，不能靠删指令。

## proxy 测量的赢点常是 artifact
- **STORE_PLW / BPERM** proxy 用 contiguous 地址（数据故意错，footprint 变小）显得快 **+8%**；正确 row-strided 数据拿不到——coalescing 受 tile 列宽限，最大 ~64-128B。
- **COALADDR** 把数据写飞进别的行才显快（地址错 → artifact）。
- `s_setprio` 对 mxfp4 neutral（`waves_per_eu=1` 无跨 wave 仲裁对象 + 稳态已 stall-free）。
- **wl-depth** 轴对 Llama shape 全在 ±0.5% 噪声内（"best" 随机跳）。❌ 别再试：扩 wl autotune。

## do_bench 不可靠 → 用 cuda.Event
- ❌ 别再试：`triton.testing.do_bench`。某些 shape 测出**超 peak（4363 TFLOPS = 87% MI355X peak，甚至 4500-22000）**但 kernel 跑 garbage；内部 cudagraph capture / cache eviction 行为不可控。
- 正解：`torch.cuda.Event(enable_timing=True)` + `record()` / `elapsed_time()` 直读硬件 timestamp。

## 非对齐 shape silently early-exit → bogus 超 peak
- HK dense kernel 在非对齐 shape 上 **silently early-exit**（不算 partial tile 直接返回），测出**超 peak（4500-22000 TFLOPS）**的 bogus 数字。
- 必须 **M%256==0 N%256==0 K%128==0**。
  - 安全 N：2048 / 4096 / 8192。
  - 安全 K：128 / 256 / 512 / 1024 / 2048 / 4096 / 7168 / 8192。
- ❌ 别再试：shape 不对齐时比 dense vs grouped——dense baseline 是 bogus。

## SNR gate 正确用法
- `get_tolerances` 返回的 **fp8=1e-1 / fp4=0.5** 容差故意很松，**不能当真实 gate**——量化后 element-wise 容差没意义。
- 低精度主 gate 必须用 **SNR 门**：`compute_snr(ref, actual)`，参数 **reference 在前**。
- 只用于诊断（非 gate）：`relative_error` / `mean_squared_error` / `max_abs_error` / `cosine_similarity` / `symmetric_similarity_diff`。

## fp8 TN big-shape 典型瓶颈画像（profile 解读）
- **VMEM Utilization ~3%** → 不是 memory/store/带宽 bound。
- **Dependency Wait 高（source 未给具体数字）+ MFMA Util ~34-40% + Occupancy ~1 WG/CU** → latency-bound（8 waves 喂不饱 MFMA+tr8 延迟链）。
- **L2 Cache Hit**：square/big-K ~66%，big-N ~51%（L2 复用差是大 N 的主瓶颈）。

## （另一内核）wgrad 4-wave 3buf 的 bank conflict / chunk_stride
- 这是**不同内核**（grouped wgrad 4-wave 3-buffer transpose-read 内核，非上面的 dense-TN autotune 内核）：`chunk_stride=1056` padding 消掉了转置读的 bank conflict，`1 池@1056 = 0%，2 池@1024 = 14%`（`_CS` 越界会导致 LDS 超限编译报错）。
- 来源：10-grouped-wgrad-4wave-3buf.md, project_wgrad_occ_feed_bound.md（不属于 fp8 TN big-shape 画像）。

---
来源: fp8-gemm-bench/SKILL.md, 05-dead-ends.md, 10-grouped-wgrad-4wave-3buf.md, project_mxfp4_epilogue_store.md, verify-accuracy/SKILL.md, flydsl-fp8-gemm-tuning/SKILL.md, project_wgrad_occ_feed_bound.md
