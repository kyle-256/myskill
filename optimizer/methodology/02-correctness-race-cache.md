# 正确性、race 诊断与缓存失效:SNR/det 门控、cross-wave LDS-barrier race、长K非确定、JIT缓存

> 类别: 方法论 · 主题标签: SNR-gate, determinism, correctness, race-debug, race-diagnosis, cross-wave-lds, barrier, vmcnt-defer, measurement-noise, long-K, race-detection, jit-cache, isa-dump, cache-invalidation

## 正确性门:SNR 阈值、det=0 bit-exact ≥500-1000 run(强测 2×2000)、outdiff=0 证同源

### 三层证据(强弱递进)
- **outdiff=0(bit-identical)= 证同源**:同进程同 device 同输入,原版 vs 移植版逐元素比对,同源 kernel 必须 bit-identical(outdiff=0),再比 TF(应在 ±1.5% 内噪声)。**SNR 只能证'能跑对',证不了'和上游同一版本'**。
- **SNR = 证能跑对**(数值语义正确,不证同版本)。
- **det=0(bitwise reproducible)= 证无 race**;determinism rtol=atol=0 bitwise。

### fp8/fp4 SNR 阈值(硬编码在 test/bench,不在 get_tolerances)
| 类型 | 阈值 | 位置 |
|---|---|---|
| FP8 E4M3 | SNR ≥ 25 dB | SNR_THRESHOLD / check_gemm_correctness_by_snr |
| FP8 E5M2 | SNR ≥ 20 dB | 同上 |
| FP4 (E2M1) | SNR ≥ 10 dB | 同上 |
| 高精度 allclose | bf16/fp16 rtol=atol=1e-2、fp32 1e-4 | get_tolerances in tests/pytorch/test_utils.py |

### fp8 实践基线与判据(FlyDSL 生产)
- ⚠️ **两条线别混**：上表的 `25 dB` 是 test/bench **硬编码的最低通过线**(过了才算"能跑对")；下面的 `50+ dB` 是 FlyDSL grouped fp8 **生产健康实践基线**——低于它说明数值已腐蚀(哪怕过了 25dB 硬线也要查),不是可接受状态。
- fp8 下 **50+ dB 视为通过**;健康基线:rowwise ~56 dB、tensorwise ~40-47 dB。
- **SNR 掉到 30 以下 = 错**;SNR 崩(如 15/29/46 dB)= 数值错/race,必须修或弃该改动。
- wgrad:SNR56 为通过,**SNR<54 视为输出腐蚀**。
- correctness reference 用 default candidate((256,8,4,0) for fwd)对齐;autotune 只在候选间选,**不改数值语义**。

### 上游量化精度 vs kernel bug 的判别法
- 症状:grouped wgrad SNR 偶尔跌到 **51-64**,不一定是 kernel bug,可能是上游量化/参考路径精度。
- **判别法**:同 shape 比两个后端(FLYDSL vs 默认 TRITON)的 SNR,若一样(到小数点后 9 位)→ 上游问题,不是你的 kernel。
- WHY:同源参考路径产生同样量化误差,数值完全一致说明差异来自 quant 而非 kernel 计算。

### det=0(bit-exact)验证:run 数与 fresh 随机
- **必须 ≥500-1000 run**(200-run 会假阳性);SNR 不掉(55/79/87 dB 看 scale)**不代表 det=0**。（源自 gpt_oss2 mxfp8 项目实例，非本环境:`vmcnt_hint=4` 在 200 run 看 det=0,500 run 才暴露 **1.5e-5** race,见 gpt_oss2_docker/myskill/flydsl-fp8-gemm-tuning/SKILL.md）
- 强测 = cache_clear + **≥2×2000 run**,每 pass 换 **fresh mk() 随机 a,b**。WHY:race 是 data-dependent + intermittent,固定随机会漏。
- 小 K(如 K512)门:SNR≥55 dB 且 det=0,DETRUNS≥4(严验用 ≥6 甚至 30)。

### 低精度验证策略(profiling 不够,顺序不能跳)
1. 单独验 conversion 和 packing;
2. block-scaled 路径单独验 scale 处理;
3. kernel 数学对比高精度/dequant 参考;
4. 只有前三步都过才信性能数字。

### 测试写法(抄现有结构)
- dense fwd+bwd 参数化 → test_gemm.py;低精度(SNR gate)→ test_gemm_fp8.py;determinism → test_gemm.py::test_gemm_deterministic;attention/MoE/grouped → 对应 test 文件。
- 参考实现放 tests/pytorch/ref/,用 float()/fp32 算再比低精度输出:gemm_ref.py(含 grouped_gemm_ref)、attention_ref.py(用 sdpa)、quantization_ref.py。
- 规则:torch.manual_seed(42);克隆 _ref 输入并各自 requires_grad_() 让梯度独立;归一化敏感输入 a=a/a.abs().max();复用间 reset a.grad=None;不支持组合用 pytest.skip 而非 fail。
- FP8/FP4 前用 check_fp8_support()/check_mxfp8_support()/check_mxfp4_support() 门控 arch;config 用 Float8QuantConfig/Float4QuantConfig。

### 跨平台数值精度(判 AMD 是否匹配 CPU/NVIDIA,非 pass/fail)
- benchmark/accuracy/eval_gemm_accuracy.py、eval_sf_accuracy.py(GPU-vs-CPU 写 .xlsx)。
- GPU-vs-GPU 是 dump/transfer/compare 三步流程,见 benchmark/accuracy/README.md。

## cross-wave LDS-barrier race 诊断:逐一排除法定位 g2s→ds_read、barrier 是 race-critical

### race 症状识别
- 表现 = 低概率(0.001-1%)bit-non-deterministic:少数 element 偶尔不一致,**不是数值漂移**(不是 NaN/Inf/SNR 下降)。
- 高发场景:grouped GEMM / persistent kernel。dense single-GEMM 通常无 race,只有 SGPR pressure 触发 spill 才暴露。

### det/race 复测协议
- 固定量化后的输入连跑 `DETRUNS=100`(或 50K-150K)次,逐元素 bit-exact(rtol=0 atol=0)比对 run#0。
- cross-wave LDS-barrier / SCVGPR race 表现为 run-to-run 漂移 max|d|≠0。
- 低概率 race 统计量级细节(10K 假阴性/需 50K-150K+ iters/多 GPU 确认)见 pitfalls/04-race-vmcnt-correctness.md「race harness」。

### 逐一排除法定位(SCVGPR/vmcnt race)
| knob | 设置 | 结果 | 排除的假设 |
|---|---|---|---|
| WLV | =0(vmcnt 全排空) | 仍 racy | 排除 vmcnt 乱序 |
| ELGK | =0(lgkmcnt 全排空) | 仍 racy | 排除边界 drain 不足 |
| CONSTSC | scale 设常量 | 仍 racy | 排除 scale 值 → 定位到 operand cross-wave g2s→read |
| 1BAR | =0(每 phase 都加 s_barrier) | **det 0** | 确认根因 = barrier 不足 |

### 根因:LDS barrier 不足(非 vmcnt 乱序)
- g2s(`buffer_load_lds`/buffer_load→LDS,VMEM 路径)是 wave 协作完成;barrier 确保所有 wave g2s 落地后,跨 wave `ds_read` 才安全。
- operand(ds_read)的 LDS 可见性依赖:g2s vmcnt 落地 **+** s_barrier 跨波同步。读 8 波协作填充的 LDS 必须在某 barrier 之后。
- gfx950 barrier 语义:`wait_barrier(cnt)` = `s_waitcnt vmcnt(cnt)` + `s_barrier`。

### mxfp4 4-wave 稳定 emit 边界
- `ELGK ≥ 15` racy(最优 9);`WLVMCN ≥ 20` racy(最优 10)。
- barrier 是 race-critical:减 barrier(`LEANBAR`)必触发 operand race。

### ISA diff 精确定位 racing gap
- `PT_RACE_VM=0`(safe) vs `=1`(racing) 的 `21_final_isa.s` 只差 **2 行**(两相位 barrier 的前置 drain):
  - safe:`s_waitcnt vmcnt(0) lgkmcnt(0)` 全 drain
  - racing:`vmcnt(16) lgkmcnt(10)`
- 一相位正好 16 条 `buffer_load_dwordx4`(4 pool × 4 step)→ 对上 vmcnt(16)。
- 方法:逐行 ISA diff 精确定位性能/正确性差异来源。

### 安全回收 racing 优势 = 加缓冲深度换 partial-drain
- 1 池 3buf(只给 B1 第 3 缓冲):安全延迟 1/4 写 → 回收 racing 优势约 **51%**。
- 2 池 3buf(B0+B1 都第 3 缓冲):安全延迟 1/2 写 → m2048 打平/超 racing、全线比 1 池 **+2~4%**、m4096 距 racing **0.6%**。

### prefetch:AGPR 累加腾 VGPR → 手工提早 ds_read
- AGPR 累加腾出的 VGPR 余量(如 **128→110**)可用于把 operand ds_read 主动提早进 MFMA 窗口做重叠。
- 编译器因 volatile+barrier 强序做不到;手工把一个 operand 的 ds_read **下移一个 barrier**(在可见性安全范围内)可恢复并反转残差。

## 长 K(K28672)HW 级间歇非确定:判 race 必须用短 K 做干净 det0

- **现象**:长 K(K28672)在高负载机上有 HW 级间歇非确定(det)。官方 intrinsic K28672 即使 `DETRUNS=15` 也出 det(15233/152750);raw-baseline 出 det 39926 等。这些 det 幅度**比自研改动本身还大**,足以淹没 <1% 的真信号。
- **对照**:K8192 / K16384 三者均 det0 干净。→ 非确定**阈值在 K16384~K28672 之间**。
- **归因**:纯 HW / 长-kernel 效应,与 kernel 逻辑无关(官方 intrinsic 也复现,证明不是自研 race)。WHY:长 kernel 在高负载机上运行时间更长,更易踩到硬件级的间歇性非确定来源。
- **教训 / 判 race 规程**:分辨 <1% 真信号,必须用**短 K(K8192/K16384)做干净 det0 + 交错对标**。**不能**用 K28672 的 det 指标判 race——K28672 出 det 不等于 kernel 有 race。

## FlyDSL JIT 缓存失效:改算错验金标准、清缓存时机、lru_cache.cache_clear

### 金标准:验缓存是否真失效
- 金标准验证法完整版见 pitfalls/07-flydsl-frontend-tracer.md「金标准验证」

### disk cache 何时自动失效、何时必须手动清
- FlyDSL JIT disk cache 在 `~/.flydsl/cache`,`FLYDSL_RUNTIME_ENABLE_CACHE` 默认 true。
- **kernel 源码或闭包值变化 → 自动失效**;in-memory cache 始终生效。
- 只有以下两种情况需要 `FLYDSL_RUNTIME_ENABLE_CACHE=0` 或 `rm -rf ~/.flydsl/cache`:
  1. 改了 C++ passes;
  2. 改了**非闭包** helper 函数(改动不进 hash,disk cache 不失效)。

### 单进程多 env-variant 对比:必须 cache_clear
- `_compile_dense_tn` 是 `@functools.lru_cache(128)`。
- 多 env-variant 单进程对比时,若不清 lru_cache,函数内读的 env 被**冻在首次调用值**,后续 variant 全用第一次的 env。
- 正确姿势:每换 env 前 `G._compile_dense_tn.cache_clear()` 再 compile。

### dump ISA 前的完整清缓存 + 触发流程
- 扫寄存器/spill 前 dump ISA,设:
  - `FLYDSL_DUMP_IR=1`
  - `FLYDSL_DUMP_DIR=/tmp/dscan`
  - `FLYDSL_RUNTIME_ENABLE_CACHE=0`
- **必须清 comgr 缓存** `~/.cache/comgr`,否则 comgr 缓存 codegen,不重编不 dump。
- **必须真正 RUN kernel**(`c(*args)`)才触发编译+dump;只 `_compile` 不够。
- 产物落在 `/tmp/dscan/kernel_dense_tn_0/21_final_isa.s`。

---
来源: remote-sync/SKILL.md, flydsl-fp8-gemm-tuning/SKILL.md, flydsl-fp8-gemm-results/SKILL.md, 02-nt-fwd-kernel.md, verify-accuracy/SKILL.md, agpr_rawasm_progress.md, agpr_phase5_mono.md, tool-rocprof/SKILL.md, gfx950-vmcnt-race-debug/SKILL.md, 02-race-diagnosis.md, 03-emit-knobs.md, 10-grouped-wgrad-4wave-3buf.md, flydsl-sync/SKILL.md, flydsl-kernel-authoring/SKILL.md, project_mxfp4_epilogue_store.md
