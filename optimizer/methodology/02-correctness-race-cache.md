# 正确性、race 诊断与缓存失效:SNR/det 门控、cross-wave LDS-barrier race、长K非确定、JIT缓存

> 类别: 方法论 · 主题标签: SNR-gate, determinism, correctness, race-debug, race-diagnosis, cross-wave-lds, barrier, vmcnt-defer, measurement-noise, long-K, race-detection, jit-cache, isa-dump, cache-invalidation

## 正确性门:SNR 阈值、det=0 bit-exact ≥500-1000 run(强测 2×2000)、outdiff=0 证同源

### 三层证据(强弱递进)
- **outdiff=0(bit-identical)= 证同源**:同进程同 device 同输入,原版 vs 移植版逐元素比对,同源 kernel 必须 bit-identical(outdiff=0),再比 TF(应在 ±1.5% 内噪声)。**SNR 只能证'能跑对',证不了'和上游同一版本'**。
- **SNR = 证能跑对**(数值语义正确,不证同版本)。
- **det=0(bitwise reproducible)= 证无 race**;determinism rtol=atol=0 bitwise。

### 全 NaN 先查"exp2 模式 vs lse 格式匹配",别当 kernel bug（2026-07 attention bwd 踩）
- attention bwd 有多 exp2 模式,**每模式要不同的预缩放 lse 格式**:Schraudolph fast → `lse_s23 = lse*(-log2e)*2^23 + (127*2^23−486411)`;poly/exact → 平 `lse*(-log2e)`。**喂错格式 → exp2 溢出 → 输出全 NaN**。
- 坑:dq 与 dkdv 模块的**默认 exp2 模式可能不同**(dq 默认 `fast_exp2=True`、dkdv 默认 poly)。测试脚本若统一喂 poly 格式 lse,dq 会 NaN 而 dk/dv 正常——**这是测试脚本 bug,不是交付 bug**。查法:用真 wrapper(格式自匹配)跑 `_test_final` 两模式,若那里 dq 正常(fast 35dB/poly 52dB、det bit-identical),则探针脚本的 NaN 是格式不匹配。修:探针里 build 模块的 exp2 flag 与所喂 lse 格式对齐。
- 一般化:**看到某个输出全 NaN 而同 kernel 其它输出正常时,优先怀疑该输出的输入预处理(格式/缩放)不匹配,而非 kernel 逻辑**。

### ★★ block-scaled(mxfp4/mxfp8)的 SNR 探针**必须喂随机 E8M0 指数**,否则对"拿错 scale 套"整类 bug 全盲(2026-08-08 实测)

- 踩证:mxfp4 grouped 的折叠尾相 `emit_peel_fold` 从 round-9 起**从未为 trailing 半 k-block 重发 scale**,
  它乘的是**两个相位之前**的那套 E8M0。这条错了 19 轮没人发现,12 个计分配置一直在给错数打分。
- 为什么所有门都放行:
  * bench 的 SNR 形状 `K%256==0` ⇒ `_K128=0` ⇒ **折叠路径根本不执行**(形状盲,见本卡"门的形状必须覆盖分支");
  * det 门看的是 run-to-run 一致,拿错的是**固定**的那一套 ⇒ 每次都错得一模一样,`det=True`;
  * 而 SNR 探针当时用 `randn` 量化出 scale —— **N(0,1) 每 32 元素块的 amax 几乎总落在同一个 2 的幂档里**,
    per-block E8M0 指数近似恒定 ⇒ 拿错那套和对的那套**数值相同**,SNR 照样 49.60 dB。
- 换成 `torch.randint(122, 132)` 的随机指数后,同一份码立刻掉到 **10.98~14.00 dB**;补上重发 = 全部 49.60 dB。
- ⇒ **判据**:凡是"scale 的寄存器套/缓冲槽由发射器管理生命周期"的核(whole-loop、ping-pong scale set、
  peel/tail 变体),SNR 探针的 scale **必须独立随机**,不能由 `randn` 数据量化得到;
  数据本身可以随机,**scale 的随机性是另一个自由度,而它才是这类 bug 的唯一探针**。
- ⇒ 配套:融合两个相位(把 phase A+B 合成一个 peel)时,**逐项清点被合并掉的 prefetch** ——
  未融合体在 phase A 里做的 scale/operand 预取,融合体不会自动继承。

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
- **必须 ≥500-1000 run**(200-run 会假阳性);SNR 不掉(55/79/87 dB 看 scale)**不代表 det=0**。（MXFP8 内核实例:`vmcnt_hint=4` 在 200 run 看 det=0,500 run 才暴露 **1.5e-5** race,见 flydsl-fp8-gemm-tuning/SKILL.md）
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
- ⚠️ **更正:`15` 不是调参边界,是 ISA 编码上限** —— gfx950 `lgkmcnt` 是 **4 bit 字段**,
  写 `s_waitcnt lgkmcnt(16)` **汇编器直接报错** `too large value for lgkmcnt`(失败签名
  是一大片 MLIR/lld 噪声 + `target emulation unknown`,真因在很上面,**grep `error:`** 别读 tail)。
  所以那条应读成"9 最优,15 是存在的最大值"。**racy 阈值是 per-kernel 的**:2026-07 tw fp8 TN
  var-K wgrad whole-loop 上 **12 最快**,15 仍在 1500 reps 逐字节一致。

### ★★ partial `vmcnt(N)` 必须留"一整个发射组"的余量,恰好卡边界 = race(2026-07 实测)

- **规则**:一条 g2s 写、到读它的那条 `ds_read` 之间,若中间发射了 `S` 条 load,则该读之前的
  drain 必须取 `N ≤ S − (一个发射组)`。**取 `N = S`(按发射顺序"刚好够")会 race** ——
  gfx950 vmcnt 乱序退役,`in-flight ≤ S` 完全可以是"要的那条还在飞 + S−1 条新的"。
- **踩证(tw fp8 NN dgrad,8-wave)**:去掉 transpose-read 自带的 `vmcnt(2)` 后,只剩每
  K-iter 一条 rendezvous drain。按发射顺序算 distance-1 的 A1 池恰好是 `S=2*NSA+NSB=6`:
  取 `N=6`(margin 0)→ 宽 shape 300 reps **2 次不一致**;取 `N=4`(margin 0,卡的是 epilog
  里另一条 `S=4` 的读)→ 仍 **2 次不一致**;取 `N=NSA+NSB=4` 并把 epilog 的读留着自带
  drain → **500 reps × 4 shape = 2000 run 全部逐字节一致**。
- **别用单调性外推安全性**:同一 kernel 上 `N=8` 只错 3 次、`N=6` 错 13 次、`N=-1`(无 drain)
  错 7 次 —— **非单调**,因为它是代码布局/时序驱动的概率事件,不是阈值。**只能逐个实测**。
- **算 margin 前先把 prelude / epilog 单独算一遍**:稳态主环的 `S` 通常有一整个迭代,
  prelude 第二批 load 往往比稳态迭代**少一个发射组**、epilog 的读离它的 g2s 只有半个迭代,
  这两处才是 margin 最小的地方,也是本次两次 race 的真正落点。

### ★ LLVM 不给 intrinsic LDS 读插 vmcnt —— LDS-DMA 可见性全靠手写 drain

- `buffer_load ... lds`(g2s)写 LDS 走 vmcnt,`ds_read` 读 LDS 走 lgkmcnt;**两者之间
  没有任何自动同步**。实测 gfx950 上把 NN dgrad 主环所有手写 `vmcnt` 去掉后,`21_final_isa.s`
  里**一条 `s_waitcnt vmcnt` 都不剩**(包括 A 侧走 intrinsic `ds_read_b128` 的那条路径)。
- ⇒ **别指望"A 侧是 intrinsic,编译器会保守处理"**。手写的那几条 `vmcnt` 是**整个 kernel
  唯一的 LDS-DMA 可见性装置**,删一条就等于把 A、B 两侧的保护一起删掉。
- 反过来:`s_barrier` 之前 gfx950 **不会**自动插 `vmcnt(0)`(FeatureBackOffBarrier),
  所以 barrier 只同步执行位置、不 drain —— 这正是这套 kernel 能做深流水的前提。

### ISA diff 精确定位 racing gap
- `PT_RACE_VM=0`(safe) vs `=1`(racing) 的 `21_final_isa.s` 只差 **2 行**(两相位 barrier 的前置 drain):
  - safe:`s_waitcnt vmcnt(0) lgkmcnt(0)` 全 drain
  - racing:`vmcnt(16) lgkmcnt(10)`
- 一相位正好 16 条 `buffer_load_dwordx4`(4 pool × 4 step)→ 对上 vmcnt(16)。
- 方法:逐行 ISA diff 精确定位性能/正确性差异来源。

### 安全回收 racing 优势 = 加缓冲深度换 partial-drain
- 1 池 3buf(只给 B1 第 3 缓冲):安全延迟 1/4 写 → 回收 racing 优势约 **51%**。
- 2 池 3buf(B0+B1 都第 3 缓冲):安全延迟 1/2 写 → m2048 打平/超 racing、全线比 1 池 **+2~4%**、m4096 距 racing **0.6%**。
- **2 池 3buf 已经吃满 racing 优势(tw fp8 wgrad 实测,2026-07)**:在它之上再放宽到
  完全 racing 的 `vmcnt(16)` 买到 **0.00%**(0.98033 vs 0.98027);反向收紧到 `vmcnt(0)`
  全 drain 则 **−7.5%**(每个 wgrad 配置 −6.8~11.5%)。⇒ 这条流水正好停在膝点,
  "再加缓冲深度 / 再多 defer 写"在该 kernel 上是**已耗尽的杠杆**,剩下的停顿在 lgkm 侧。

### prefetch:AGPR 累加腾 VGPR → 手工提早 ds_read
- AGPR 累加腾出的 VGPR 余量(如 **128→110**)可用于把 operand ds_read 主动提早进 MFMA 窗口做重叠。
- 编译器因 volatile+barrier 强序做不到;手工把一个 operand 的 ds_read **下移一个 barrier**(在可见性安全范围内)可恢复并反转残差。

## ★ 分级 drain(graded per-consumer vmcnt):把单个 rendezvous 拆到各消费者前

单个 "覆盖下一整轮 LDS 读" 的 `s_waitcnt vmcnt(N)` 必然被**最紧的那个 pool** 钉死:
N 只能取 `min_pool(该 pool 填充之后发出的 load 数) - 一个 issue group`。
**拆开放**——每个 phase 收尾 barrier 各带一条 vmcnt,只覆盖紧随其后的那次 LDS 读——
每条就只需回溯到"上一轮发出的填充",N 直接翻倍。零新增指令(vmcnt 挂在已有 barrier 上)、
零 LDS、byte-exact。2026-07-31 tw NN grouped dgrad 实测:单条 `vmcnt(NSA+NSB)=4`
→ 三条 `vmcnt(2*(NSA+NSB))=8`,dgrad 组 geomean **0.9810 → 0.9878(+0.7pp)**,四个配置全涨,
总 gm 0.98505 → 0.98710。

### ★★ 适用前提:先数每个 pool 的 distance,只有"存在唯一一个 distance-1 pool"才赢
- 2-buffer ping-pong 的 pool,只要在**读之后**回填,拿到的就是 k+2(distance-2);
  若回填的是**另一个** slot,则只能是 k+1(distance-1)。一轮里最后被读的那个 operand
  常常被迫走后者(它的 slot 要到下一轮开头才空出来)= 唯一的 distance-1 pool = 全局 drain 的下界。
- **全 pool 已 distance-2 时,分级/挪动 drain 是中性甚至负的**(同轮三处实测):
  ① NN dgrad 给 A1 加**第 3 块 LDS buffer** 变 distance-2、并撤掉它那条 drain →
     dgrad 0.98800 vs 0.98824,**中性**(白花 16KB LDS + 三槽轮转)。
  ② NT fwd(`nt_dist2=True`,四个 pool 本就 uniform distance-2、每轮一条 drain)照搬分级 →
     fwd geomean 1.0116/1.0127 → **1.0094(−0.3pp)**,`fwd down balanced` 1.036 → 1.024。**判负**。
  ⇒ 结论:**杠杆不是"drain 条数"或"in-flight 上限",是"单条 drain 被 distance-1 pool 钉死"这个结构缺陷**。
  没有那个缺陷就别动 drain 排布。

### ⚠ prelude/首轮 rendezvous 不适用循环内推出来的 margin
同一份 margin 规则(binding - 一个 issue group)在主循环里 1600 rep 干净,搬到**进循环前的
那条 rendezvous** 上 → **wide shape 每 rep 都不一致(400/400)**,其中一个 shape SNR 掉到 34.3dB,
而计分 bench 自带的 `N=1024/K=256` det 门**照样全绿**。原因未根因(疑似上一个 tile 的
`buffer_store` 仍占 vmcnt 槽,使 issue-order 计数在 tile 缝失效)。
⇒ **prelude 的 drain 保持紧;要放松必须单独 400+ rep wide-shape 验证,不能靠推导。**
（同源提醒见 pitfalls/04 §"det gate on narrow shapes passes while wide shapes diverge"。）

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

## ★ graded 逐消费者 drain 的两条边界(tw NN dgrad,2026-08-01 复测)

1. **计数不是越大越好**:把三处 `_dbar()` 的 `_nd` 从 `2*(NSA+NSB)`=8 放到 `3*(NSA+NSB)`=12,
   dgrad 组 geomean 0.98897 → 0.98748。8 是**性能最优**,不只是"安全上限之下的一个值" ——
   放宽 vmcnt 之后 g2s 跑得更远,反而把 LDS 写回压到消费者头上。
2. **能不能用 graded,取决于该体每迭代发几组 g2s**:满体 4 组 ⇒ 8 安全且最快;丢掉 b1 载入的
   半 N 体只发 3 组,同样推导给出的 4/2 两个值都 racy(见 pitfalls/05)。正确解不是"给它另找一个
   计数",而是**别让那个体少发 g2s**:保留 b1 的 g2s、只删它的 MFMA/store,该体就直接继承满体的
   graded 表,dgrad 组 +0.5pp。

## ★★★★★ 把 GPU `Memory access fault` 归因到 kernel：用 host sync，别用 `AMD_SERIALIZE_KERNEL`

2026-09-12 GLM-5.2 部署崩溃，十几轮 server 换来的方法。

### 第一步：先判断"肇事 kernel ≠ 有 bug 的 kernel"

指纹（看到这一组就别再找 faulting kernel）：

- 故障地址**页对齐**、每 rank 不同
- **同参数成功几千次、偶发崩一次**
- 隔离环境换遍形状/并发/内存压力**都复现不出来**
- 给可疑缓冲区加保护带、把可疑的表初始化掉 —— **都无效**

⇒ 是某个 kernel 往**映射内的别人家缓冲区**越界写（自己不崩），后面读到脏数据的算子才崩。
**改找越界写，别找崩溃点。**

### 第二步：归因只认 host sync

`AMD_SERIALIZE_KERNEL=3` **实测在 MI355X + flydsl 这套里没生效**（加了它崩溃点反而更靠后，
墙钟也没变长）。据它得出的"最后一个 launch 就是肇事者"两次把我带偏。

```python
_run_compiled(fn, *args)
if not torch.cuda.is_current_stream_capturing():   # capture 期间 sync 非法
    torch.cuda.synchronize()
    print(f"[probe] {label} ok", flush=True)
```

**sync 返回 = 该 kernel 干净退出，是事实不是排序推断。**
崩溃前最后一条「有发射行、缺配对 `ok`」的才是嫌疑人。一轮就推翻了 SERIALIZE 的结论。
★ `is_current_stream_capturing()` 判断不能漏：capture 期 sync 会让 server 在 warmup 直接退出。

### 第三步：找越界写用金丝雀，但要知道它抓不到什么

```python
buf = torch.empty(n + GUARD, ...); buf[n:].fill_(POISON)
... run ...; torch.cuda.synchronize()
(buf[n:] != POISON).sum()   # >0 = 这个缓冲区被越界写
```

- **必须同时打印"检查了几个缓冲区"**。我第一版只挂上 2 个（另外 3 个的分配语句在两条
  路径里**文本完全相同**，`str.replace(...,1)` 改到了不走的那条），不打计数的话
  "0 越界"就是假阴性。
- **只能抓近距离越界**。野值索引（几 MB~GB 量级）会整个跳过尾巴，必须在邻域内散布多块毒值。
- `PYTORCH_NO_HIP_MEMORY_CACHING=1` 单卡**没用**：hipMalloc 按页对齐，小越界仍在同页内。

### ★★★★★ 「探针没输出」有三种原因，下结论前必须自检

| 表象 | 真因 | 自检 |
|---|---|---|
| 插桩 0 条 | 代码路径没走（模型走 `op_*` 分解路径，不是 `forward`） | 在**无条件**位置打安装横幅 |
| 整块没执行 | env 名撞了框架命名空间（`SGL_` 被 sglang env 注册表当废弃别名吞掉） | 直接查 `/proc/<pid>/environ` |
| 只有横幅没有 op 行 | 包装了模型**根本不实例化**的类（GLM-5.2 DSA 跑的是 `DeepseekV2*` 不是 `Glm4Moe*`） | 横幅里打包装了几个方法 + 运行时计数 |

**判据写成"计数远大于 0"，不是"没看到报错"。**

### 两条取证纪律

- **取崩溃前尾部一律全量输出**（`awk 'NR>=n-20 && NR<n'`），**不加 grep 过滤** ——
  我 grep 掉了 `[flydsl-launch]` 行，于是"最后一条是量化 kernel"只是"最后一条没被我滤掉的"。
- **调试插桩会污染性能测量**：清理前后同树差 **2.8%**，四个桶均匀慢 2-3% 就是 host 开销的指纹
  （保护带分配、每次 launch 建闭包、env 查询）。**计分前必须全清。**
