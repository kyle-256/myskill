# 08 — ATT Trace 根源诊断（K=28672, M8192N8192, GPU7, prod defaults）

> 用 rocprofv3 ATT trace 实测 stall 分布，**推翻**了之前基于减法探针（NOGIS/NODSR）的
> "g2s 769T + ds_read 288T" 推理叙事，从硬件层定位了 5405T 天花板的真正根因。
> 日期：2026-06-17。

## 如何复现 trace

容器 `mlperf_gptoss`（chi2811，bind mount `/mnt/vast/kyle/code2`→`/workspace/code`）：

```bash
# 1. decoder 库（已装在 /opt/rocm/lib/librocprof-trace-decoder.so）
# 2. trace 驱动 + input.yaml 已在 turbo/：_trace_kernel.py（8 次 dispatch）、_trace_input.yaml
cd /workspace/code/FlyDSL/turbo
HIP_VISIBLE_DEVICES=7 FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 \
  rocprofv3 -i _trace_input.yaml -- python _trace_kernel.py
# kernel 名 regex = "kernel_gemm_4w"；输出 kernel_trace_output/ui_output_agent_*_dispatch_*/
python _hotspot_analyzer.py kernel_trace_output/ui_output_agent_*_dispatch_538 --topk 20 --mode both
```

## Stall 分布（dispatch_538，total stall 7.76M = 45.2% of 17.18M cycles）

| Stall 类型 | 占比 | 说明 |
|-----------|------|------|
| **MFMA/FMA** | **89.5%** | `v_mfma_scale_f32_16x16x128_f8f6f4` 自身 stall——绝对主导 |
| VMEM-load | 2.7% | buffer_load |
| barrier | 2.5% | s_barrier 跨 wave 同步 |
| VMEM-wait | 1.8% | s_waitcnt vmcnt |
| VMEM-store | 1.1% | 尾部 store |
| LDS + LDS-wait | 1.0% | ds_read |

指令 mix：MFMA 256, buffer_load 70, ds_read 96, ds_write 0（g2s 走 buffer_load_lds，不算 ds_write）。

## 关键洞察

### 1. MFMA stall 89.5% = operand-not-ready bubble，不是浪费

- kernel 实际 5405T / NOGIS 6462T = **83.6% MFMA 效率** → 已是 compute-bound。
- MFMA 的"stall"包含等 operand 就绪：g2s 搬来的 LDS 数据 + ds_read 出来的 a_frag/b_frag
  没及时 ready 时，**MFMA 阻塞等待，stall 记在 MFMA 头上**，而不是记在 s_waitcnt 上。
- 这解释了为什么 VMEM-wait 只有 1.8%：内存延迟不是显式 waitcnt 暴露，而是隐性表现为
  MFMA operand bubble。之前的减法探针（g2s 暴露 769T、ds_read 暴露 288T）实际就是这些
  bubble，ATT 把它们统一归到 MFMA stall——两种视角一致。

### 2. occupancy = 1 wave/SIMD，**LDS-bound（决定性）**

真实 CSV（`out_kernel_trace.csv`）：

| 字段 | 值 |
|------|-----|
| LDS_Block_Size | **147456 字节 = 144 KB/workgroup** |
| VGPR_Count | 216 |
| Workgroup_Size | 256（4 waves） |

- LDS 限制：160KB/CU ÷ 144KB/wg = **1 workgroup/CU** → 4 waves/CU ÷ 4 SIMD = **1 wave/SIMD**。
- 2 个 workgroup 需 288KB > 160KB → 物理不可能。
- **即使 VGPR 允许 2 waves，LDS 也只允许 1。occ=1 由 LDS 锁死。**
  （注：analyzer 误报 "bound by VGPR / accum 256"，因为 ISA 里 v_mfma 引用 a[...] 寄存器被
  当成 agpr-form；CSV 才是权威——Accum_VGPR_Count=0，vgpr-form，combined≈216。真正的锁是 LDS。）

### 3. 根源链条（闭合）

```
5405T whole-loop  需要  BN256 + 深 prefetch(3xA+2xBL+2xBR) + 满累加器(64 accs)
                  →  LDS 144KB/wg  →  occ=1 (LDS-bound)
                  →  单 wave 无法用第二个 wave 的 MFMA 填 operand bubble
                  →  g2s 暴露 769T + ds_read 暴露 288T  →  MFMA 89.5% stall
                  →  6462T(NOGIS) → 5405T，损失 1057T (16%)
```

## occ=2 是唯一的结构突破口——但被 whole-loop 实现绑死

- occ=2 需要 LDS ≤ 80KB/wg。唯一达成方式：**BN128 单 slice**（无 BR，2xA+2xBL=80K，
  32 accs=128 AGPR）。4wave 文件开头注释明说："BN128 → occ=2 hides the ds_read LATENCY
  that 1-wave/SIMD exposes"。
- **但实测 BLOCK_N=128 K28672 = 2736T（比 BN256 5405T 慢一半）**。原因：whole-loop
  bare-asm（`call_mxfp4_wholeloop`，line 422，产生 5405T 的 +16% 关键 lever）是 **BN256
  双 slice 专属**（要 br_base/accR）。BN128 fallback 到 per-K FlyDSL 循环（_ASMM=2/3/4/5），
  缺整个 whole-loop hw-loop 优化 → 慢。

### 理论上限估算（已被下方 gate 实测推翻）

~~occ=2 理论上限 ≈ 6462T~~ —— 这个推断**错了**：它假设 occ=2 不带额外成本，但
occ=2 的唯一达成途径 BK128 会让 g2s 翻倍。见下方阶段 0 实测：occ=2 实际净亏 -13%。

## occ=2 实测验证（阶段 0 gate）= NO-GO

实现 BN128 whole-loop 前，先用 BN256+BK128 廉价探测 occ=2 的真实收益（whole-loop
hw-loop 编译快；BK128 把 LDS 减半到 72KB → occ=2 可达，无需改单 slice 代码）。

| 配置 | 速度 | occ | MFMA stall | VMEM-load stall | 总 stall |
|------|------|-----|-----------|-----------------|---------|
| BN256 BK256 (prod) | **5405T** | 1 (LDS 144K) | 89.5% | 2.7% | 45.2% |
| BN256 BK128 | **4686T** | **2 (LDS 72K, VGPR 176)** | 58.7% ↓ | **30.0% ↑↑** | 53.6% |

**结论：occ=2 在此 shape 是净亏（-719T / -13%）。**
- occ=2 **确实**降低了 MFMA operand bubble（89.5% → 58.7%）——证明 occ 理论成立。
- 但达成 occ=2 的**唯一途径是 BK128**（BK256 LDS 144K/112K 都 > 80K 无法 occ=2），
  而 BK128 让 g2s 频率翻倍 → **VMEM-load stall 从 2.7% 暴增到 30%**，完全压倒 MFMA
  bubble 的改善。BK128 的 KI=224 还导致 per-K 慢路径编译爆炸（>25min 超时）。
- BN128 单 slice（更小 tile、grid 翻倍、per-wave 工作更少）只会让 g2s **更**暴露，
  不可能逆转，故**不实现** BN128 whole-loop。
- BK128 实测同时验证了 BARNOP 之外的另一个 ceiling：g2s-HBM-bound 在细分块下提前触顶。

## 结论与方向

1. **5405T 是 BN256 whole-loop occ=1 的接近物理极限**（83.6% MFMA 效率，LDS+操作数双约束）。
   所有参数扫描（EVENSPREAD/PREFETCH/MMORD/ACC_DIST16/PINBASE/WLVMCN/BARNOP）都在
   5400±10 噪声内——ATT 现在解释了为什么：它们都不改变 occ=1 这个根约束。
2. **唯一真实突破口 = 给 BN128 单 slice 实现 whole-loop bare-asm**，拿 occ=2 隐藏 operand
   bubble。这是较大的代码工程（改造 `call_mxfp4_wholeloop` 支持单 slice / 32 accs /
   2xA+2xBL buffer），收益不确定（grid 翻倍 + 2 wave 竞争同一 XDL unit 可能部分抵消）。
3. 次要：BN256 减 LDS 到 ≤80K（减 buffer 数）也能 occ=2，但会削 prefetch depth，
   大概率得不偿失。
