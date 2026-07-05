# grouped MXFP8:公平对标口径 / dense 目标水平 / fwd-dgrad autotune / occ=1 结构上限 PMC

> 类别: 方法论 · 主题标签: mxfp8, grouped, 公平对标, autotune, occ=1, PMC, scaled-MFMA对标

## MX vs TW 唯一公平口径(否则灌水)
1. TW 和 MX 都用**同一 gemm backend FLYDSL**(`GlobalBackendManager.set_gemm_backend/set_grouped_gemm_backend(FLYDSL,FP8)`),别拿 TW=HIPBLASLT/Triton 对 MX=FLYDSL。
2. **force-nt OFF**(monkeypatch `_deter_use_nt_layout_gemm_in_bwd→False`),否则 TW fwd 白背 `a.t()+b.t()` 转置税。
3. 带量化(fresh `turbo.ops.*gemm_fp8` 每步重量化)+ `torch.utils.benchmark.Timer` + `retain_graph`。
- mxfp8 是 scaled-MFMA,合理对标是 **scaled 的 aiter mxfp8**(mxfp4 达 98%),不是非-scaled per-tensor。

## dense 目标水平(both FLYDSL,force-nt off,含量化)
- fwd MX/TW 打平 ~1.0×(0.98-1.01×);**bwd MX 真反超 TW 0.67-0.82×**(4096×4096×4096 bwd 0.67× 最好)。这是 FlyDSL quant+FlyDSL gemm 真实力(TW 也放 FLYDSL 后 dense MX bwd 依旧 0.67-0.82×)。grouped 目标就是追这个。

## fwd/dgrad 按-shape autotune(优化3,已落地)
- 参考 **pertensor grouped gemm** 的 `_autotune_np_dispatch`(不是 dense mxfp8 gemm)。
- 要点:① 均衡分布计时(`_balanced_mx_targs`,group_offs 换成 M_total/G 均衡切分),选出 config 只依赖静态 shape 与运行时分布无关;② 数值护栏 rel-RMSE<2e-2 且 finite 才采纳;③ 扁平候选+base+1.5% 迟滞(`cand[0]=base(256,4,4,0)`,≥1.5% 才切);④ 计时用 `_robust_time`(一次 sync 内背靠背 launch iters 次再除)——早期每次 launch event.record+sync 把 per-call ~20us 气泡计入导致选错回退。
- 候选:`(256,4,4,0)base`、`(256,8,4,0)`、`(256,1,4,0)`、`(256,8,8,0)`、`(256,4,8,0)` + 2D band `(256,8,4,{8,16})`。缓存键 `(M_pad,N,K,G,cbsz,blgp,out_fp16,persistent)`。`PT_MXGG_AUTOTUNE=0` 退回固定 base。收益 0-4% 无回退。

## fwd/dgrad occ=1 结构上限(PMC)
- M2048 4096×7168:MX `kernel_grouped_mxfp8_nt` MfmaUtil 60.2%/Occ 21.8%/MemStall 0.1%/VGPR128 LDS128KB vs TW `kernel_grouped_nt_persistent` 65.2%/21.2%/0.1%/同。
- MemStall≈0 非访存瓶颈;MfmaUtil 只 60-65% 是 **occ=1(LDS=128KB→1WG/CU)** 下 barrier/依赖 stall 没第二 wave 填;MX 60 vs TW 65 的 5pp 是喂 scale 给 scaled-MMA 的操作数开销(**scaled-MMA 本身税≈0**)。persistent 假设证伪(vs 非 persistent 无差别)。
- 剩余 40% MMA 空闲要动只能上 **occ=2**(LDS 128KB→≤80KB:削流水缓冲 4→2 或缩 tile),TW 同卡 occ=1 是共有结构上限。

## benchmark shape 表(源 benchmark/ops/training/config.py)
- 3 模型 ×2 GEMM(GateUP/Down),B=8(experts),M∈{2048,4096},trans_b=True。GateUP=(N=2*moe_int,K=hidden),Down=(N=hidden,K=moe_int)。
- deepseekv3(2048/7168):GateUP 4096×7168、Down 7168×2048;qwen3-235b(4096/4096):8192×4096、4096×4096;gpt-oss-20b(2880/2880):5760×2880、2880×2880。

---
来源: mxfp8-grouped-gg-devloop/SKILL.md(总目标 / 优化3 / rocprof PMC / shape 表); project_mxfp8_wholeloop_port.md
