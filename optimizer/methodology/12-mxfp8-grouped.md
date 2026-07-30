# grouped MXFP8:var-K wgrad 瓶颈 / quant 融合 / 公平对标口径 / e2e 计时与节点结果

> 类别: 方法论 · 主题标签: mxfp8, grouped, wgrad, variable-K, scale-prefetch, chunk-SSA, AST-rewrite, occ2, quant, meta-prologue, batched-quant, HIP契约, 公平对标, autotune, occ=1, PMC, scaled-MFMA对标, e2e-timing, compile-once, GB200对标, 节点检查, gfx950

## grouped MXFP8 var-K wgrad:瓶颈定位 / scale-prefetch 真因 / 软流水 / AST-rewrite 绕坑 / occ=2 ROI

### bwd 追 dense 的主战场 = var-K wgrad
- 逐组件对标(total_M=32768,dense→grouped):**dgrad(grad_a)已追平** dense 1.05-1.18×(同一 NT kernel);**wgrad(grad_b)最大结构性缺口 1.16-1.35×**(绝对 +200~270us),根因 dense wgrad 是普通 NT gemm,grouped wgrad 是 **variable-K kernel**(按组收缩 M);grad_out 量化 grouped 慢 1.2-1.6× 但绝对占比 <10% ROI 低。

### PMC 根因(非访存、非占用)
- var-K vs dense NT wgrad(4096×7168)`rocprofv3 --pmc`:两者 **MemStall≈0**、**occ 相同**(~20-23%,LDS=128KB→1WG/CU occ=1 结构上限)、VGPR/LDS 几乎相同;缺口纯在 **MFMA 利用率 dense 61% vs var-K 45.5%(-16pp)**。var-K grid 恰好 8×(每 group 一套 blocks 448×8),per-group prologue/epilogue + 4-buf 软流水在组边界重启占比大 → MMA 间空隙(组边界依赖/barrier bubble,不是等访存)。

### 提速真因 = scale 载入时机(不是 memref 累加器)
- baseline body 顶部**当拍**载入 sa0/sa1/sb → scale ds_read 延迟压在 MMA 关键路径;dense NT **预取下一拍** scale(sa0n)用上拍已到的 scale 做 MMA。
- 只加 scale 软流水吃到全部 ~9%(单独 SSA 累加=0 提速,单独 scale-prefetch ≈ 全部收益)。
- 落地 = **chunk-local 结构**(通用零判断):`for _c in range(k_iters//chunk)` 调 module-level `_wgrad_ssa_chunk`(chunk 内累加器进 SSA、scale 逐拍预取,只在 chunk 边界读写一次 memref),尾部 `range_constexpr(chunk)+k_abs<k_iters` guard 收余数。成绩 930/1047/530/307 → 851/936/478/274,方阵追平 grad_a NT 兄弟核。

### 4-buffer 软流水移植(优化1,前置)
- MX wgrad 单缓冲串行 → chunked 4-buffer distance-2 软流水(移植 tensorwise `_wgrad_body_4buf`)。LDS 2→4 缓冲(cur0/1+next0/1,A/B 各一套);外层运行时 `scf.for over ceildiv(k_iters,chunk)`,内层 `range_constexpr(chunk=8)` 展开 distance-2,trace 期 swap cur↔next;**偶数 chunk 在边界重置 ping-pong 身份**(scf.for 不能 loop-carry Python 变量);accum 用 rmem in-place(`_wgrad_mx_accum`)。结果:M=2048 全反超快 4-26%。

### 绕开 AST-rewrite state-variable 报错(动态 scf.for)
- 主循环体不能有 `obj.method()` 调用、不能有 list 状态变量。做法:body 是**单个 module-level fn 调用** + 只对 buffer 引用做 tuple-unpack 重赋值,动态 `scf.for` 只 loop-carry buffer 引用。

### 用户硬约束:禁 balance 判断
- 严禁任何地方判断是否 balance(host per-call 判 uniform、once-cache、核内 per-WG 分派全禁);不能特化 uniform-K,只能让**单一变长-K kernel** 对所有分布都用上 scale 预取。host wrapper 里彻底没有 D2H/offs 读取,eager/capture 同一条 kernel 无 sync 税。

### 生产内核 vs WL 对象
- 生产 = `_build_grouped_mxfp8_wgrad_kernel`(8-wave / occ=2 / packed-preshuffle scale / chunk-SSA 软流水)。
- 对象 WL = `_build_grouped_mxfp8_wgrad_wl_kernel`(occ=1 / 4-wave 单块 bare-asm hw-loop,NT 布局 `[OUT,m_total]`,per-1x32 E8M0 折进 `v_mfma_scale_f32_16x16x128_f8f6f4`)。
- 基准(gfx950 B=8 M=4096 N×K=4096×4096 清缓存):baseline 479us / WL-scaled 811-813us(1.7× 慢)/ WL-unscaled 445us(0.93× 反超);SNR 28.1 全对。**WL scale 折进 MMA 是 1.7× 慢的真凶**。
- WL scale-prefetch 修法(收益仅 ~4%,845→811):删 phase-top emit_scale + 阻塞 vmcnt(0),改成在 emit_inplace 里每个 scale VGPR 最后消费 MFMA 之后重载下一 phase scale(藏进 MFMA shadow),prologue 只留一次 emit_scale。瓶颈是吞吐非延迟。真正修复(未做)= packed/preshuffle scale(4 个 E8M0 打进 1 i32 op_sel 选),预期边际 ~5%。

### 下一刀 ROI
- 生产 var-K wgrad 上 **occ=2**(LDS 128KB→≤80KB:**削流水缓冲 4→2 而非缩 tile**,让第二 wave 填 barrier/依赖气泡,唯一能真正抬 MfmaUtil 45→61% 的 lever)。不是 WL、不是 cross-group persistent(均已证伪)。风险:fwd/dgrad occ=2 曾证伪(bm=128 坏),wgrad 削缓冲是新尝试。dense NT 自己也只 61%@occ=1,现实目标是逼近 61% 吃掉 16pp。

## grouped MXFP8 quant:融合 meta prologue / batched 权重 quant / HIP 输出契约

### 融合 meta prologue(优化4,已落地)
- 问题:grouped quant 主 kernel 每 WG 都跑一遍 O(G) 组搜索(读 GO/GR/GC 三个 int32-view offs ~40 copy-atom load,select 链 gate 住整 tile),memory-bound 短 kernel 里延迟全暴露占 ~42us。
- 正解:写融合 prologue kernel `meta`(NBM 线程/256 一 block,每线程对 `base_m` 做一次 O(G) 搜索写 RB/RO/RE[bt]),与主 kernel **背靠背在同一 `@flyc.jit` stub 里 `.launch()`**。搜索从 `512×(NBM×NBK)` 个 WG 各做一遍 → NBM 线程一次。
- 结果:kernel 155、wrapper 181,grouped-FLY **首次快过 grouped-HIP**。`RB=go_orig_g+mrel`、`RO=go_row_g+mrel`、`RE=go_col_g+M_g`。

### batched FLY 权重 quant(compile_qdual_batched,默认 ON)
- 把 dense `compile_qdual` 加 batch 维,一次 launch(`grid=B*NBM*NBK`)量化整块 `[B,N,K]` 权重全部 B 个 expert,替旧 Python 逐 group 循环(逐 group `quant_mxfp8_raw`+`torch.stack`,8× launch,比 HIP 慢 2.7-4.3×)。
- 核心:`batch=pid//NPB`,输入 band 与 4 条输出 band 各按 batch 重定位;**per-batch scale byte base**(`base_row_b/base_col_b`,dword 对齐)保证相邻 batch scale dword-packing 不互踩(`_store_scale` 加 `base_byte` 形参)。等 group 尺寸无需 per-tile 组搜索。
- 性能:快 HIP **1.3-1.54×**(275→207 / 344→239 / 152→116 / 94→61),方向同 dense-FLY ~1.6×。`PT_MXGG_FLYDSL_QUANT=0` 退回 HIP。

### HIP grouped_quantize_mxfp8_dual 输出契约(新 kernel 须 bit-兼容)
- 位置 `quantization.cpp:829`。输入 `x[total_M,N]` + `group_lens/group_offs`(int64 GPU),`ROW_ALIGN=64/COL_ALIGN=128`,`M_pad_row=cdiv(total_M+G*64,64)*64`、`M_pad_col=cdiv(total_M+G*128,128)*128`、`N_pad=cdiv(N,128)*128`。
- 返回 **8 个固定顺序**:0 `rowwise_output[M_pad_row,N_pad]` fp8、1 `rowwise_scale` e8m0、2 `colwise_output[N,M_pad_col]` fp8(已转置)、3 `colwise_scale`、4-7 `group_lens/offs_padded_rowwise(64)/colwise(128)`。
- raw 行/列主 E8M0 **不 preshuffle**(gemm 侧 in-launch preshuffle);组边界 scale 独立不跨组;**padding 行 scale 填 127(=1.0)/ 数据填 0**;padded-layout offs 由 `compute_padded_layout_gpu` 在 GPU 上算(无 D2H)。

## grouped MXFP8:公平对标口径 / dense 目标水平 / fwd-dgrad autotune / occ=1 结构上限 PMC

### MX vs TW 唯一公平口径(否则灌水)
1. TW 和 MX 都用**同一 gemm backend FLYDSL**(`GlobalBackendManager.set_gemm_backend/set_grouped_gemm_backend(FLYDSL,FP8)`),别拿 TW=HIPBLASLT/Triton 对 MX=FLYDSL。
2. **force-nt OFF**(monkeypatch `_deter_use_nt_layout_gemm_in_bwd→False`),否则 TW fwd 白背 `a.t()+b.t()` 转置税。
3. 带量化(fresh `turbo.ops.*gemm_fp8` 每步重量化)+ `torch.utils.benchmark.Timer` + `retain_graph`。
- mxfp8 是 scaled-MFMA,合理对标是 **scaled 的 aiter mxfp8**(mxfp4 达 98%),不是非-scaled per-tensor。

### dense 目标水平(both FLYDSL,force-nt off,含量化)
- fwd MX/TW 打平 ~1.0×(0.98-1.01×);**bwd MX 真反超 TW 0.67-0.82×**(4096×4096×4096 bwd 0.67× 最好)。这是 FlyDSL quant+FlyDSL gemm 真实力(TW 也放 FLYDSL 后 dense MX bwd 依旧 0.67-0.82×)。grouped 目标就是追这个。

### fwd/dgrad 按-shape autotune(优化3,已落地)
- 参考 **pertensor grouped gemm** 的 `_autotune_np_dispatch`(不是 dense mxfp8 gemm)。
- 要点:① 均衡分布计时(`_balanced_mx_targs`,group_offs 换成 M_total/G 均衡切分),选出 config 只依赖静态 shape 与运行时分布无关;② 数值护栏 rel-RMSE<2e-2 且 finite 才采纳;③ 扁平候选+base+1.5% 迟滞(`cand[0]=base(256,4,4,0)`,≥1.5% 才切);④ 计时用 `_robust_time`(一次 sync 内背靠背 launch iters 次再除)——早期每次 launch event.record+sync 把 per-call ~20us 气泡计入导致选错回退。
- 候选:`(256,4,4,0)base`、`(256,8,4,0)`、`(256,1,4,0)`、`(256,8,8,0)`、`(256,4,8,0)` + 2D band `(256,8,4,{8,16})`。缓存键 `(M_pad,N,K,G,cbsz,blgp,out_fp16,persistent)`。`PT_MXGG_AUTOTUNE=0` 退回固定 base。收益 0-4% 无回退。

### fwd/dgrad occ=1 结构上限(PMC)
- M2048 4096×7168:MX `kernel_grouped_mxfp8_nt` MfmaUtil 60.2%/Occ 21.8%/MemStall 0.1%/VGPR128 LDS128KB vs TW `kernel_grouped_nt_persistent` 65.2%/21.2%/0.1%/同。
- MemStall≈0 非访存瓶颈;MfmaUtil 只 60-65% 是 **occ=1(LDS=128KB→1WG/CU)** 下 barrier/依赖 stall 没第二 wave 填;MX 60 vs TW 65 的 5pp 是喂 scale 给 scaled-MMA 的操作数开销(**scaled-MMA 本身税≈0**)。persistent 假设证伪(vs 非 persistent 无差别)。
- 剩余 40% MMA 空闲要动只能上 **occ=2**(LDS 128KB→≤80KB:削流水缓冲 4→2 或缩 tile),TW 同卡 occ=1 是共有结构上限。

### ★ E8M0 scale pack 的硬上限 = 4 / occ 与 LDS 的实测常数(gfx950 实测)
- **`pack ≤ 4` 是 ISA 硬上限,不是调参空间**:`scale_opsel(k,pack)=k%pack` 喂的是 `v_mfma_scale_f32_16x16x128_f8f6f4` 的 **op_sel 字段,只有 2 bit**(一个 dword 4 个字节)→ `pack=8` 在 ISA 层不可表达。❌ 别再试 `pack=8`。
- pack=1→4 的机制(wgrad,per-WG PMC):`SQ_INSTS_VMEM` 3947→3387(**−14.2%**)、MFMA/VALU 不变、SALU +19%(op_sel 字节选择走**独立标量端口,不计价**)→ 教科书式"窄标量 load 换 SALU 解码",实测 +5.4~8.2%。
- **mx NT `group_segment_size = 131,336 B = 128 KiB + 264 B`**(tw 131,072 B)。那 264 B 是 group-find 的两张 `[G+1]` int32 前缀表;160 KB/CU 对 128 KB 的 tile 本来就只够 **1 WG/CU(=2 waves/SIMD)**,所以**这 264 B 不花占用率**,别去抠。要动占用率只能削流水缓冲或缩 tile(见「fwd/dgrad occ=1 结构上限」)。
  ★ 2026-07 更新:改用 lane-resident 前缀表后 **LDS 表整张消失,`group_segment` 回到 131,072 B**(见下节),那 264 B 现在是**给"加波数不加每波 tile"的 5 池方案预留的额度**。

### ✅ 已交付杠杆:边界半 tile 不搬 padding 半区(2026-07 gpt-oss campaign,NT 核)
- 背景:`OUT_N % BLOCK_N == BLOCK_N/2`(gpt-oss down N=2944=11.5×256、gate_up 5760=22.5×256)时最后一个 N-block 只有半个 tile 真实。先做的**象限跳过**(编译期 `_HALF_N` 门 + `_readfirstlane_i32(block_n)` 标量分支选变体)只删 `ds_read` + MFMA,**g2s/barrier/vmcnt 逐条保留** → 半 tile 仍把整个 256 宽 B tile 搬进 LDS,一半是废的。
- **补刀(本轮)**:半-N 变体里把 **b1 半区的 g2s 一并删掉**(prologue 2 条 + 主环每 K-iter 1 组),drain 从 `2*NA+NB`(=6)收到 `2*NA`(=4)—— 即"一次迭代的发射数减去最早那一组",保持与全 tile 相同的 2 条安全余量。
- 实测(kernel-only,tw/mx):`fwd down balanced` 0.985→**1.005**、`dgrad down balanced` 0.988→**1.006**、`fwd/dgrad down heavy` +1.2%;`gate_up`(半 tile 只占 1/23)持平不回退;wgrad 不动。gm +0.6%、**min 0.985→1.003(14/14 全 >1 过验收线)**。3 次清 cache 复跑 gm 1.0708/1.0708/1.0720、min 1.0007~1.0029。
- ★ 纸面上界 ~2.1% traffic、按本 campaign 三次 50~60% 兑现率估 +1.0~1.3%,**实测 down 拿到 +2.0%,高于预测带** —— 与"边界 tile 是 feed-bound"一致(删的是 DMA 不是 MFMA)。
- ⚠️ **同一手术搬到 wgrad 核是净负(−0.09% gm,wgrad 六配置持平~−1.6%),已回滚**:wgrad 的 A1 池是 stage `k+1` 的**距离-1**池且**最先发射**,删掉它 drain 必须同步从 6 收到 4,省下的 DMA 被"更早的等待"吃回去。**两个核的流水不对称,别照抄**(NT 四个池全是距离 2,删 B1 不改变其余池的等待时机)。

### ✅ 已交付杠杆:group-find 前缀表从「串行 carry + LDS + barrier」换成 **lane-resident DPP wave-scan**(2026-07 gpt-oss campaign,NT 核)
- **适用症状**:grouped kernel 的 per-tile `group→tile` 解码。旧实现(本卡 pitfalls/05 记的 "monotonic group-carry commit" 范式)= 入口一次 O(G) 串行前缀 + `_readfirstlane_i32` 钉 SGPR + 两张 `[G+1]` 表停 LDS + 一个 publish `s_barrier`,per-tile 再做 G 宽比较树 + 2 次 `ds_read`。**G=32 时这一整条链值 5.3% 的周期**。
- **新做法(三条原语,已进 `gemm_helper.py`,dense/grouped 通用)**:
  1. `_wave_prefix_add_i32(v)` —— wave64 inclusive add-scan,**6 步 `rocdl.update_dpp`**(`ROW_SHR|1/2/4/8` + `ROW_BCAST15`(row_mask=0xA) + `ROW_BCAST31`(row_mask=0xC)),`bound_ctrl=True` 让移入的 lane 贡献 0,不用 bank_mask 修边。要求**满 EXEC**(放 kernel 入口,任何分叉之前)。
  2. `_lane_tbl_load/_lane_tbl_scan/_lane_tbl_get` —— 表**驻 VGPR lane**(entry i 在 chunk i//64 的 lane i%64),查表 = 一条 `v_readlane`(结果直接落 SGPR,后续地址算术自动标量化)。**既绕开 SGPR 文件溢出(~102/wave,2×(G+1)=66 就炸),又绕开 LDS 的 publish barrier 和每次查表的 `ds_read`。**
  3. `_wave_count_le_i32(v,bound)` = `rocdl.ballot` + `llvm.intr.ctpop` —— 单调表上「≤bound 的条目数」**就是** group index,**O(1) 取代 G 宽比较树**。空组、尾部越界 lane(读 0 → 扫描值 = total_tiles)都天然正确,和比较树逐位等价。
- **实测(gpt-oss 14 配置,kernel-only,tw/mx)**:gm **1.08715 → 1.10632 / 1.10531**(两样本,+1.72%)、min **1.01324 → 1.0439 / 1.0409**(+2.88%)。**8 个 NT 配置全涨 +2.8~5.0%,6 个 wgrad 配置逐项不动** —— 单变量信号干净(wgrad 的 group_idx 是除法不是扫描,本来就没这条链)。
- **ISA 佐证**:prologue 窄 `buffer_load_dword` 69 → 4;`group_segment` 131,336 → **131,072 B**;`.sgpr_count` 61 → 60;spill 仍 0。
- **K-sweep 仿射复核(单进程交织)**:mx per-tile 固定成本对 tw 的超额从 **+0.776 → +0.430 µs**(砍掉 45%);mx 主环斜率 1.2463 µs/iter vs tw 1.3761(**mx 快 9.4%**);模型预测 23-iter 1.0679 / 45-iter 1.0832,实测 1.065 / 1.090 —— **两端都吻合**。
- ⚠️ **flydsl API 坑**:该 build 的 `llvm.TruncOp.__init__(res, arg, overflowFlags)` 是**三个位置参数**,少一个直接 TypeError;用 `arith.trunci(T.i32, v)`(`flydsl.expr.arith`,签名稳定)。`rocdl.update_dpp / ballot / readlane / llvm.intr_ctpop` 在 flydsl 里都可直接调,`readlane` 的 lane 允许传 Python int。

### ★ 功耗墙上「省周期」只能兑现约一半:DVFS 回吞 55%(gfx950 grouped mxfp8 实测)
- 同一 shape(`fwd down balanced`)、同一探针、优化前后两次三栏对照(回文序 mx/tw/tw/mx,每臂持续驱动 14 s):

  | | wall tw/mx | mx sclk | tw sclk | 周期口径 tw/mx |
  |---|---|---|---|---|
  | 优化前(round 10) | 1.0172 | 1957.8 MHz | 1985.3 MHz | 1.0315 |
  | 优化后(本轮) | **1.0392** | **1909.5 MHz** | 1995.3 MHz | **1.0858** |

- **周期上赢了 5.3%,wall 只拿到 2.2%** —— 因为 mx 把同样的工作压进更短时间 ⇒ 瞬时功耗更高(1397.7 vs tw 1382.6 W,均在 1400 W TBP 的 99.8%)⇒ DVFS 把 mx 的时钟又压低 2.5%。**mx 与 tw 的 sclk 差从 1.39% 拉大到 4.3%。**
- ⇒ **在 ≥99% TBP 的 kernel 上,任何纯「省周期」的杠杆按 ~45% 兑现率估收益**(本轮 5.3%→2.2%,兑现 42%);要拿满额必须**同时省能量**(字节/L2 miss/指令发射),否则省下的周期一半被时钟吃掉。
- pJ/FLOP 反而是 mx 更优(0.5601 vs tw 0.5758,好 2.7%):**功耗高 ≠ 效率低,只是把同样的活干得更快**;判 regime 时别把「功耗高」当成「浪费」。

### benchmark shape 表(源 benchmark/ops/training/config.py)
- 3 模型 ×2 GEMM(GateUP/Down),B=8(experts),M∈{2048,4096},trans_b=True。GateUP=(N=2*moe_int,K=hidden),Down=(N=hidden,K=moe_int)。
- deepseekv3(2048/7168):GateUP 4096×7168、Down 7168×2048;qwen3-235b(4096/4096):8192×4096、4096×4096;gpt-oss-20b(2880/2880):5760×2880、2880×2880。

## MXFP8 e2e 计时口径 / 节点占用检查 / vs GB200 结果

### 测量真实 GPU 时间(compile-once)
- FlyDSL `flyc.jit` launch 每次调用有 **~40us Python 派发开销**,cuda-event 会量成派发延迟。正确:先 `comp=flyc.compile(launch, ...)` 编译一次,再 `comp(...)` 直进 GPU stream 用 cuda-event 计时(warmup 30、iter 300)。

### 官方 e2e benchmark 口径(与 TE GB200 一致)
- `torch.utils.benchmark.Timer(stmt="fn()").timeit(100).mean*1e3`;`tflops=2*M*N*K/(ms*1e-3)/1e12`。
- **禁止手搓 cuda-event 计 bwd**(把 autograd dispatch 算进去,bwd 低估 ~10-18%)。

### 节点 / GPU 占用检查
- 选空闲 gfx950 卡:
  ```
  docker exec <容器> bash -c "rocm-smi --showuse --showmeminfo vram"
  ```
  取利用率/显存最低的卡;完整选卡三件套见 `connection/common/02-pick-free-gpu.md`。
- 节点 = gfx950 ×8,HBM3e ~8 TB/s 峰值,实测 1R:1W copy 上限 ~6.3 TB/s。

### LDS-合并转置写 vs GB200 结果
- fwd geomean ~0.99×(≈对齐)、bwd ~1.10×(反超)。
- e2e(Timer 口径,9 Llama shape,SNR 全 28dB)新(LDS-合并转置写)vs 旧 BM=32:**fwd 1710→1824 TFLOPS(+6.7%)、bwd 1839→1878(+2%)**。提升集中在 quant-heavy K=11008 fwd(4096×4096×11008 +21%、8192 +13%、16384 +10%)。
- fwd 差距根因 = B200 硬件 MX cast **近免费** vs MI355X **软件 dual-cast**;fwd 稳过 1.0× 仍需 in-gemm fusion(**用户否决**)。

---
来源: mxfp8-grouped-gg-devloop/SKILL.md(优化1/优化6/var-K PMC/真因&已修复/优化4/batched FLY 权重 quant/HIP 输出契约/总目标/优化3/rocprof PMC/shape 表); project_mxfp8_grouped_wgrad_wl.md; project_mxfp8_wholeloop_port.md; mxfp8-8wave-devloop/SKILL.md
