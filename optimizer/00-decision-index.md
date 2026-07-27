# 00 — 决策索引:动手前先 grep 这里(死路 / 条件 / 开口)

> 类别: 导航 · 主题标签: decision-index, dead-ends, before-you-try, action-lookup, 死路速查, 症状分诊
>
> **这是「动手前」的第一站,不是读物。** agent/人在**准备试某个杠杆前**,先在本表 grep 你要做的动作
> (如 `grep -i "占用率\|maxnreg\|direct\|atomic\|DMA\|scale_pack"`)。命中 = 别重踩,点进详卡看根因。
> 覆盖全库高频被重试的死路(全量 ❌ 散在各详卡,本表是入口)。
>
> **判定图例**:❌**实测DEAD**=真实现+bench 过,别重跑 · ⚠️**分regime/条件**=看形状/占用才定 ·
> 🔬**分析预测**=只探针/纸面判过,可挑战但先读根因 · ✅**开口**=未验证或值得做。
> ★ 铁律(methodology/03):**subtractive/HALF 探针、roofline 峰值率、纸面 op-count 给的都是「上界」不是「可达」——
> 判负/判正前必须 edit→bench 真实现**(反复踩:HALF_PV +8.7% 假想 → 真 K32-PV −11%;SKIPST +30% → DMA/QB 全负)。

## A. 占用率(occupancy)—— 抬 occ 几乎全负,先确认是不是 occ-bound
| 你想试的动作 | 判定 | 一句根因 | 详卡 |
|---|---|---|---|
| `maxnreg`/`waves_per_eu`/`--amdgpu-num-vgpr` 强抬 occ | ❌实测DEAD | 必 spill 到 scratch(灾难);且本容器 external codegen 不可用,flag 对 VGPR 纹丝不动 | pitfalls/01,05 · methodology/03 · [[project_mxfp4_vgprform_deadend]] |
| `amdgpu-mfma-vgpr-form=false` / inline-asm `=a` 逼 AGPR 抬 occ | ❌实测DEAD | 生产 4-wave ISA 逐字节相同;AGPR 与 VGPR **共用同一 512 池**,搬家不抬 occ | pitfalls/01,03 · methodology/04 |
| 8-wave 原生 occ=2 藏 store / 长 K 追平 4-wave | ❌实测DEAD | 28672 −14%/6144³ −20%;8-wave per-warp tile 减半→B 复用减半→mfma 翻倍 > occ 收益 | pitfalls/03,05,06 |
| BK128 换 occ=2(mxfp4 K28672) | ❌实测DEAD | occ=2 唯一途径 BK128→g2s 频率翻倍 VMEM-stall 2.7→30%,总 5405→4686 | pitfalls/05 |
| 缩 tile / 矩形 tile 为抬 occ | ❌实测DEAD | feed-bound worst shape:占用率非杠杆,根因 LDS-feed 带宽 | pitfalls/01,05 |
| attention 强制 occ(WPEATTR/o_acc 进 AGPR/bf16 o_acc) | ❌实测DEAD | 全 latency-bound,occ-1/2 都填不满;强制必 spill/掉速 | pitfalls/12 |
| bwd(occ-2)dual-wave / 8-wave / warp-spec / bare-asm cross-head 显式并行 GEMM↔softmax | ❌实测DEAD | occ-2 baseline 已靠两独立 WG 共驻**免费享无屏障跨-WG overlap**;8-wave 单-WG 换成带屏障税组内 overlap,天花板<baseline(dkdv 1029<1116)。净胜改走冷-load 寄存器预取(+1.81%) | methodology/15 · pitfalls/13 |
| **先做**:判是不是 occ-bound(而非 register/LDS/feed/latency-bound) | — | occ=`512//(arch+agpr)`;`Accum_VGPR_Count=0`+ISA 才权威;多数 kernel 是 feed/latency-bound | methodology/04 · methodology/03 |

## B. LDS / 数据通路 / store
| 你想试的动作 | 判定 | 一句根因 | 详卡 |
|---|---|---|---|
| A/B operand 直载 VGPR 跳过 LDS(a_direct/b_vgpr) | ❌实测DEAD | mxfp4-8wave 慢 ~22%(bit-exact);LDS 复用是必须 | pitfalls/03 |
| 3-stage LDS 双缓冲 / 更深 prefetch distance | ❌实测DEAD | 每 stage×3>160KB 净负;L2 是 capacity-thrash,prefetch 治不了 | pitfalls/03,05 |
| store 与 compute 重叠(STORE_ILV/persistent/CShuffle 宽存) | ❌实测DEAD | gfx950 统一 vmcnt,在途 store 占 vmcnt→串行化;宽存 LDS round-trip 吃掉收益 | pitfalls/05 |
| DMA/`buffer_load_lds` 免 reg→store(attention & GEMM) | ❌实测DEAD | register-prefetch 对 scattered-gather 最优,DMA s_waitcnt 暴露 HBM;attention 版被 flydsl 编译器 bug 挡 | pitfalls/03,04,12 · [[project_dsv4_fwd_dma_experiment]] |
| triple-buffer / prestore 隐藏 exposed store | ❌实测DEAD | store 是 work-bound(工作量不减)/ 占 occ;重排无用 | pitfalls/12 |
| 免一个操作数转置(`ds_read_b64` 换 `_tr`) | ❌实测DEAD | ISA 等价,转置读不比 plain 贵,0 收益 | pitfalls/05 |
| padding 消 bank 冲突(实测本无冲突时) | ❌实测DEAD | BANK_CONFLICT=0 时 pad 白吃 LDS;不对称 swizzle 会静默读错行 | pitfalls/03 |
| register double-buffer/read-once/RING/monohoist(mxfp4) | ❌实测DEAD | 2 operand set 超寄存器预算→RAGreedy crash/spill | pitfalls/01 |

## C. quant / scale(mxfp8/mxfp4)
| 你想试的动作 | 判定 | 一句根因 | 详卡 |
|---|---|---|---|
| 把残余 gap 归因「scaled-MFMA 指令税」 | ❌结论错 | 指令税≈0;真凶=scale VMEM load 未隐藏 | pitfalls/05 |
| `scale_pack`/opsel byte-pack(生产 per-K) | ❌实测DEAD | 现有流水已预取藏 scale;前提是「裸无预取」,生产不适用 | pitfalls/05 |
| scale 也暂存 LDS / scale via LDS 双缓冲 | ❌实测DEAD | 慢 6×/0.42-0.61×;只暂存 fp8 数据不暂存 scale | pitfalls/05 |
| 大 BM 但不跨-lane 合并的 quant 写 | ❌实测DEAD | 0.58-0.98×;死的是「不合并」,大 BM+LDS 合并才是制胜招 | pitfalls/05 |
| whole-loop(WL)移植进生产 | ❌用户裁定 | 2500 行 bare-asm 不可维护;occ=1 结构上限,per-K 才是生产路径 | pitfalls/05 |
| 优化 quant kernel 撬 e2e(grouped) | ❌实测DEAD | quant 绝对占 35-46% 但 e2e 中性;缺口在 gemm(occ=1 上限) | pitfalls/05 |
| e8m0 scale 广播前 cast Uint8 | ❌静默错 | 高位损坏 match~9% 垃圾;位运算全程 Int32 | pitfalls/05 |
| atomic 融合 reduce(dense split-K) | ❌实测DEAD | 同地址 HBM atomic 争用串行;split+reduce 是答案 | pitfalls/05 |

## D. attention(fwd + bwd)—— 优化顺序 playbook 见 methodology/15;实测 win/dead 见 pitfalls/12(dsv4)+13(hd64 dense)
> 优化前先读 **methodology/15-attention-optimize**(定 bound→fwd:消/藏 store·_FMAX0·dual-wave / bwd:减 MFMA·融串行核→藏 MFMA 延迟→exp2·occ·确定性)。

### D. attention(dsv4 sparse-MLA)—— 详见 pitfalls/12
| 你想试的动作 | 判定 | 一句根因 | 详卡 |
|---|---|---|---|
| s_setprio(fwd / bwd dQ) | ❌实测DEAD | latency-bound,优先 MFMA 饿死 VALU/read | pitfalls/12,09 |
| cross-tile 软件流水(XPIPE) | ❌实测DEAD | cr4 LDS-port 饱和/few-tile prologue > 重叠 | pitfalls/12 |
| K=32-PV(2-tile batch) | ❌实测DEAD | 批打断 QK→softmax→PV 交织(occ-1 藏延迟唯一机制) | pitfalls/12 |
| dual-layout-KV 消 QK bank 冲突 | ❌实测DEAD | 多写一份 KV 压在 occ-1 关键路径 −7~17%;冲突本 off-critical | pitfalls/12 |
| register-transpose PV(fwd) | ❌实测DEAD | 每 tile 重做转置不摊薄 −77%(bwd rtr 赢是摊在 rank-tile) | pitfalls/12 |
| query-blocking(cr128 减 store) | ❌实测DEAD | 2×o_acc>256 arch-VGPR→强制 occ-1,占用损失>store 省 | pitfalls/12 |
| ✅ pro cr0/cr128 interm(慢 triton ~20%) | ✅开口 | 研究 triton 小 topk tiling/少 launch | pitfalls/12 |
| ✅ banded-SWA dKV 融合 / hybrid scatter | ✅开口 | 减 interm+gather 的 2× tensor HBM | pitfalls/12 |

## D2. attention(Meta/gpt_oss hd64 DENSE flash bwd,确定性)—— 详见 pitfalls/13
| 你想试的动作 | 判定 | 一句根因 | 详卡 |
|---|---|---|---|
| ★✅ **XCD-major block_id 解码**(`xcd=bid%8`,每 XCD 一整块 (batch,kv_head)) | ✅dkdv+2.81% / dq+0.54% | per-XCD 私有 L2,让共读同份 K/V 的 GQA WG 相邻(dq L2 hit 86.5→94.8%)。**内层轴 dq 要 kv-head 相邻、dkdv 要 q 相邻,选反=−2.5% vs +2.4%**。⚠推翻旧记的"XCD remap GPU-fault 判负"(那是实现炸了) | pitfalls/13 · methodology/05 |
| ★✅ **dq 派发降序 q_tile(LPT)** | ✅+2.50% | 因果工作量单调递增且**派发序=list-schedule 序**→长任务优先。⚠**别假设 in-order 最优**:dkdv 恰好已是 LPT 但 dq 是反的 | pitfalls/13 · methodology/05 |
| ★✅ **冒险锚点减量 per-slot→per-v4** | ✅dq+1.76% / dkdv+0.77% | clamp 是 MFMA→trans 冒险载体(非数值保护),但只需每 v4 一个;其余 slot 靠 inline-asm dead operand 钉序。`v_min` 72→3 / 256→64 | pitfalls/13 |
| ★✅ **causal-aligned q-tile origin** | ✅+0.96% | tile 数×BLOCK_M 的 padding 别落在因果范围最长的末 tile,下移到 tile 0 → 每 tile 少一个 BLOCK_KV 步(7481→7396 访问) | pitfalls/13 |
| ★✅ **`llvm.amdgcn.exp2.f32` intrinsic 替手写 asm+锚点**(dq) | ✅−0.79% | intrinsic 本身即编译器可见的累加器读→编译器自插等待周期,锚点粒度问题消失。⚠**别推广 dkdv**(那边 +1.06% 更慢) | pitfalls/13 |
| ★❌❌ **锚点放大到"组级"**(per-GEMM1a-block / per-mt-pair / min3-group) | ❌**更快但算错** | 快 2.53%/1.79%/0.31%,但 dk/dv **16.9~19.7dB 且 det=FALSE**,**只在 q_split=4·BLOCK_KV=128 或 BLOCK_KV=64 暴露**。★规则:锚点必须读**它保护的那个 v4**,"读遍组内每个 v4"不充分 | pitfalls/13 |
| ✅ drop 冗余 B-GEMM / rho-R 全局修正(dq) | ✅+9.8% | 真结构性减 MFMA,先审计每核有无可丢项 | pitfalls/13 |
| ✅ odo delta 融合(消独立 delta 核) | ✅+5.3% | 全串行辅助核随 Sq 放大;须 out/dout→q.dtype cast | pitfalls/13,02 |
| ✅ exp2 软流水(GQA head 轴) | ✅+1.4% | 藏 head h+1 exp2 进 head h GEMM2 shadow | pitfalls/13 |
| ✅ q_split 铺满 CU grid(dkdv) | ✅关键 | 旧上限卡 2 CU 空转;qsp≥4 铺满→hw 验收 1/4→2/4(4096 594→702) | pitfalls/13 |
| ✅ 冷 load 寄存器预取(dkdv 下一head lse/delta) | ✅+1.81% | occ-2 latency-bound 上唯一 register-neutral 净胜(dt==DT-1 发射) | pitfalls/13 |
| swizzle/pad/prefetch/bank(消 tr16 冲突) | ❌实测DEAD | LDS 4-8× port 富余=latency-bound,非 port-bound | pitfalls/13,05 |
| P5 shared-GEMM1 融合(dq+dkdv) | ❌实测DEAD | q-outer=det 陷阱→KV-outer BLOCK_KV=64 但融合核 1.95-2.55× 慢 | pitfalls/13 |
| fp8 GEMM1 operands(FA-3 plain) | ❌实测DEAD | hd64 收缩维短,SNR 28.2<34 破门 | pitfalls/13 |
| dkdv 拆 dV/dK-only 核(冲 occ3) | ❌实测DEAD | 双份 Q/dO 读+拆分开销>占用率收益(-36%) | pitfalls/13 |
| LDS 双缓冲 | ⚠regime | 仅长 Skv 边际 +1.25%,短 shape -3.6% | pitfalls/13 |
| ~~长 Sq(8192/16384)full-causal 达标~~ | ★**已达标(2026-07-27)** | 旧记"确定性杠杆全 measure-closed、只剩放宽 det 或 research 级 exp-overlap"**已被推翻**——真正的开口不在 kernel 体内,在 **grid/派发映射层**(XCD+LPT+padding 对齐 = +7%)与**锚点粒度**(+3.6%)。**没有放宽任何确定性** | pitfalls/13 |
| ~~收官:perf B4 hw 16/20·fast 18/20~~ | ★**19/20 + 4-square 4/4**(turbo 侧) | 4-square hw 565/765/832/**877**(1.44-2.08×)、fast 4/4;20-config full **9/10**(既有 MI350 CK-det 是 0/10)、SWA 10/10(2.12-3.41×)。唯一未过 `newshape full 2048`=1.37×(98%)。**成果在 `sync/mxfp4/Primus-Turbo`,meta-attn 未移植** | pitfalls/13 |
| ★**先查 grid/派发层再抠 kernel 体** | ✅方法 | kernel 内部(tile/双缓冲/遍历序)调几周只值 ~1%;XCD-L2 亲和+LPT+padding 对齐 = +7%。清单:①co-resident WG 是否共享同份数据 ②派发序是否=list-schedule 序且工作量单调 ③padding 落在最贵还是最便宜的 tile | pitfalls/13 · methodology/05 |
| ★**bench 必须镜像部署配置** | ⚠**曾优化错配置** | `_bench_*` 直调 builder 用默认参数,而 `_get_bwd` 部署传另一组(dq block_kv/wpe、dkdv fold_lse)→**差 1.4%**。修法:让 **builder 默认值本身=部署值**,别靠每个 bench 记得传参 | pitfalls/13 · methodology/01 |

## E. autotune / dispatch / grouped-MoE
| 你想试的动作 | 判定 | 一句根因 | 详卡 |
|---|---|---|---|
| `id(tensor)` 作 cache key | ❌实测DEAD | 真实训练命中率≈0 | pitfalls/06 |
| 把 `M_per_group` 烘成编译期常量 / `assert M%BLOCK==0` | ❌实测DEAD | MoE 变长分布,静态划分崩 | pitfalls/06 |
| `BLOCK_M=128`(grouped/dense config sweep) | ❌实测DEAD | grid 写死 /256 → 少启动块假象,真实 1.55× 慢 | pitfalls/05,06 |
| per-shape `num_xcd` / 旧 `m_total<=2048` gate | ❌实测DEAD | overfitting 噪声 / 误判高-G MoE 走 masked | pitfalls/06 |
| `set_*_backend(BackendType.FLYDSL)` | ❌DEAD | FLYDSL 未注册进 BackendType | pitfalls/06 |

## F. 正确性 / 数据构造(先读,否则静默错)
| 你想试的动作 | 判定 | 一句根因 | 详卡 |
|---|---|---|---|
| `torch.randn().to(FP8)` 直接造数据 | ❌DEAD | >240 saturate + mma 非线性,误差不可预测;用 quantize helper | pitfalls/04,11 |
| 拿 element-wise tolerance 当 gate | ❌DEAD | 故意松,通过与否不说明正确性;用 SNR/bit-exact | pitfalls/04 |
| 靠 SNR 判断 race 是否存在 | ❌DEAD | 正常 SNR 完全掩盖 1/30000 bit-flip;用 det/多采样 | pitfalls/02,04 |
| deterministic 测试当正确性门 | ❌DEAD | 只验 run-to-run 一致,consistently-wrong 也过;要对拍 + unbalanced | pitfalls/05 |
| `create_buffer_resource(max_size=True)` | ❌DEAD | OOB 读垃圾;用 `max_size=False,num_records_bytes` | pitfalls/04,11 |
| `Vec.to(Float8E4M3FN)` / `cvt_scalef32_pk8_fp8_bf16` | ❌DEAD | 后端不 lower / gfx950 Cannot select;用 `cvt_pk_fp8_f32` | pitfalls/04,11 |

## G. 测量 / bench(别用假数下结论)
| 你想试的动作 | 判定 | 一句根因 | 详卡 |
|---|---|---|---|
| 用「删指令/跳读」测「去掉 X 的天花板」 | ❌方法错 | 跳过≠等效替换;真替换后带宽受限收益归零 | pitfalls/02,05 · methodology/03 |
| 拿并行 bench 的绝对 TFLOPS 下结论 | ❌DEAD | 被压低~10%,只有相对值可信;跨并行度对比是假象 | pitfalls/02 |
| `triton.testing.do_bench` / `cuda-event` 隔离小 kernel | ❌DEAD | 测出超 peak 假值;用 `rocprof --kernel-trace` | pitfalls/02 |
| 信裸 mean 差(DVFS 漂移内) | ⚠️ | 用配对窗口 A/B + 逐组数字;清 flydsl+comgr cache | pitfalls/02 · methodology/01 |

## H. 环境 / 同步 / 构建(违反=破坏别人环境)
| 你想试的动作 | 判定 | 一句根因 | 详卡 |
|---|---|---|---|
| rsync 加 `--delete` | ❌DEAD | 删远端不能删的东西(.so 等) | pitfalls/08 · [[feedback_ssh_key]] |
| 碰其他项目/租户的容器或盘 | 🚨红线 | 共享集群上别人的容器/盘严禁 rsync/docker exec(具体环境边界见 SKILL.md · connection/) | SKILL.md · connection/ |
| 清别人容器/停别人进程/删别人镜像 | 🚨DEAD | 即使 GPU 空也禁;srun/salloc 抢 mi355x 也不行(走 docker) | pitfalls/08 |
| `docker exec '... > /tmp/x'` 外层重定向 | ❌DEAD | 重定向落 host 不落容器 | pitfalls/08 |

## I. flydsl 前端 / tracer / 移植
| 你想试的动作 | 判定 | 一句根因 | 详卡 |
|---|---|---|---|
| 合并 NN/NT 编译函数写 `if trans_b` | ❌DEAD | NameError / 变量只在某分支绑定 | pitfalls/07 |
| `const_expr(lane==0)` / `const_expr(thread_id)` | ❌DEAD | 运行时值不能进 const_expr | pitfalls/07 |
| `scf.for` 的 iter_args carry ping-pong buffer | ❌不支持 | 多值 carry 不支持;分支放外层 python | pitfalls/07,12 · [[reference_flydsl_loop_pitfalls]] |
| 2 字节 buffer_store / bf16 vec1 load | ❌不 lower/NaN | store 须 ≥4B;gather 用 i16 vec1 装配 | pitfalls/12 |
| kernel 与 torch 同模块 | ❌RecursionError | JIT 依赖收集爆栈;kernel 放独立纯模块 | pitfalls/05,07 |
| CDNA 旋钮导 RDNA / TF32 导 gfx950 / transpose-load 导 gfx942 / 硬编码 `gfx*` | ❌DEAD | 编码/调度跨代不通;判据用 `is_rdna_arch()` | pitfalls/11 |

---
维护:新判负的杠杆先进对应详卡(带根因+实测数字),再在本表加一行入口。本表只放**高频被重试**的,长尾留详卡。
来源:全库 pitfalls/01-12 的 ❌ 汇总 + memory dsv4/mxfp4 系列。
