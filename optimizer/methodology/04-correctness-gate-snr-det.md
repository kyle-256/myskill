# 正确性门:SNR 阈值、det=0 bit-exact ≥500-2000 run、outdiff=0 证同源

> 类别: 方法论 · 主题标签: SNR-gate, determinism, correctness, race-debug

## 三层证据(强弱递进)
- **outdiff=0(bit-identical)= 证同源**:同进程同 device 同输入,原版 vs 移植版逐元素比对,同源 kernel 必须 bit-identical(outdiff=0),再比 TF(应在 ±1.5% 内噪声)。**SNR 只能证'能跑对',证不了'和上游同一版本'**。
- **SNR = 证能跑对**(数值语义正确,不证同版本)。
- **det=0(bitwise reproducible)= 证无 race**;determinism rtol=atol=0 bitwise。

## fp8/fp4 SNR 阈值(硬编码在 test/bench,不在 get_tolerances)
| 类型 | 阈值 | 位置 |
|---|---|---|
| FP8 E4M3 | SNR ≥ 25 dB | SNR_THRESHOLD / check_gemm_correctness_by_snr |
| FP8 E5M2 | SNR ≥ 20 dB | 同上 |
| FP4 (E2M1) | SNR ≥ 10 dB | 同上 |
| 高精度 allclose | bf16/fp16 rtol=atol=1e-2、fp32 1e-4 | get_tolerances in tests/pytorch/test_utils.py |

## fp8 实践基线与判据(FlyDSL 生产)
- ⚠️ **两条线别混**：上表的 `25 dB` 是 test/bench **硬编码的最低通过线**(过了才算"能跑对")；下面的 `50+ dB` 是 FlyDSL grouped fp8 **生产健康实践基线**——低于它说明数值已腐蚀(哪怕过了 25dB 硬线也要查),不是可接受状态。
- fp8 下 **50+ dB 视为通过**;健康基线:rowwise ~56 dB、tensorwise ~40-47 dB。
- **SNR 掉到 30 以下 = 错**;SNR 崩(如 15/29/46 dB)= 数值错/race,必须修或弃该改动。
- wgrad:SNR56 为通过,**SNR<54 视为输出腐蚀**。
- correctness reference 用 default candidate((256,8,4,0) for fwd)对齐;autotune 只在候选间选,**不改数值语义**。

## 上游量化精度 vs kernel bug 的判别法
- 症状:grouped wgrad SNR 偶尔跌到 **51-64**,不一定是 kernel bug,可能是上游量化/参考路径精度。
- **判别法**:同 shape 比两个后端(FLYDSL vs 默认 TRITON)的 SNR,若一样(到小数点后 9 位)→ 上游问题,不是你的 kernel。
- WHY:同源参考路径产生同样量化误差,数值完全一致说明差异来自 quant 而非 kernel 计算。

## det=0(bit-exact)验证:run 数与 fresh 随机
- **必须 ≥500-1000 run**(200-run 会假阳性);SNR 不掉(55/79/87 dB 看 scale)**不代表 det=0**。
- 强测 = cache_clear + **≥2×2000 run**,每 pass 换 **fresh mk() 随机 a,b**。WHY:race 是 data-dependent + intermittent,固定随机会漏。
- 小 K(如 K512)门:SNR≥55 dB 且 det=0,DETRUNS≥4(严验用 ≥6 甚至 30)。

## 低精度验证策略(profiling 不够,顺序不能跳)
1. 单独验 conversion 和 packing;
2. block-scaled 路径单独验 scale 处理;
3. kernel 数学对比高精度/dequant 参考;
4. 只有前三步都过才信性能数字。

## 测试写法(抄现有结构)
- dense fwd+bwd 参数化 → test_gemm.py;低精度(SNR gate)→ test_gemm_fp8.py;determinism → test_gemm.py::test_gemm_deterministic;attention/MoE/grouped → 对应 test 文件。
- 参考实现放 tests/pytorch/ref/,用 float()/fp32 算再比低精度输出:gemm_ref.py(含 grouped_gemm_ref)、attention_ref.py(用 sdpa)、quantization_ref.py。
- 规则:torch.manual_seed(42);克隆 _ref 输入并各自 requires_grad_() 让梯度独立;归一化敏感输入 a=a/a.abs().max();复用间 reset a.grad=None;不支持组合用 pytest.skip 而非 fail。
- FP8/FP4 前用 check_fp8_support()/check_mxfp8_support()/check_mxfp4_support() 门控 arch;config 用 Float8QuantConfig/Float4QuantConfig。

## 跨平台数值精度(判 AMD 是否匹配 CPU/NVIDIA,非 pass/fail)
- benchmark/accuracy/eval_gemm_accuracy.py、eval_sf_accuracy.py(GPU-vs-CPU 写 .xlsx)。
- GPU-vs-GPU 是 dump/transfer/compare 三步流程,见 benchmark/accuracy/README.md。

---
来源: remote-sync/SKILL.md, flydsl-fp8-gemm-tuning/SKILL.md, flydsl-fp8-gemm-results/SKILL.md, 02-nt-fwd-kernel.md, verify-accuracy/SKILL.md, agpr_rawasm_progress.md, agpr_phase5_mono.md, tool-rocprof/SKILL.md
