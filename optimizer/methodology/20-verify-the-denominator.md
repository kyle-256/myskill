# 20 — 对标类目标:先验分母,再调 kernel

> 类别: 方法论 · 主题标签: 对标, baseline, 分母, reference-column, 冻结常数, head-to-head, A-vs-A 校准, competitor-bench
>
> **适用场景**:目标写成「超过 X」「打平 X」「达到 X 的 N%」,而 X 是**别人给的一组数字**
> (csv / 表格 / 论文 / 上一任 agent 留下的 `REFERENCE = {...}`)。
> **动手调 kernel 之前,先花一次 run 把 X 在你这台机器上量出来。**

## ★★★★★ 铁律:分母没验过,缺口就不算缺口

2026-09-14 实证。MXFP4 dense GEMM 对标 AITER,计分脚本里是一列冻结常数:

```python
# (tag, M, N, K, B200 frozen, AITER reference, ...)
("FC1_wgrad", 28672, 4096, 32768, 5484, 5607, 5541),
#                                        ^^^^ 目标就是超过它
```

我们稳定读到 `FC1_wgrad = 5545 TF/s = 0.989 × 5607`,**差 1.1%**。
为这 1.1% 我连关六条轴,全部净负或噪声:

| 试过的杠杆 | 实测 |
|---|---|
| split-K / tail-split 强制各 mode | −2.6% ~ −45.9% |
| 同步密度 n=3 / n=4(对齐对手反汇编的 barrier 数) | 全表 −0.050% / −0.191% |
| vmcnt 水位 16→22 / 28(候选表上界) | 全表 −0.357% / −0.099% |
| autotune 缓存键补 `prepacked` 维 | 9/12 → **7/12** |
| barrier 按 MFMA 均衡而非按行均衡 | +0.012% / −0.322% / +0.050% |

然后我花 40 分钟,用**对手自己的公开 API + 对手自己的调用约定 + 我这套 harness**
量了一遍分母:

| | 冻结列 | 本机实测 | 比值 |
|---|---|---|---|
| FC1_wgrad | 5607 | 5477.5 | **0.977** |
| out_proj_wgrad | 5693 | 5565.0 | 0.978 |
| …12 行全部 | | | **0.966 ~ 0.992** |

**对手自己在这台机器上,一行都跑不到那列常数**(低 1.0~3.4%,两跑复现)。
那 1.1% 的「缺口」是分母造出来的,不是 kernel 慢。改用头对头实测分母:**12/12 全超,
最小余量 +2.148%**。

⇒ **六条轴白关。第一件该做的事是量分母,不是调 kernel。**

## 怎么把分母量对(四个必须做的动作)

### ① 用对手自己的调用约定,不是你方便的那个
反汇编告诉我 AITER 的 kernel 叫 `..._BpreShuffle_...` ⇒ 它的 B 数据和 scale 是
**调用方预先摆好的**。翻它自己的 `op_tests/` 拿到确切写法:

```python
quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
x,  x_scales_shuffle = quant_func(x, shuffle=True)   # 注意 shuffle=True
w,  w_scales_shuffle = quant_func(w, shuffle=True)
wshuffle = shuffle_weight(w, layout=(16, 16))        # B 数据也要 shuffle
aiter.gemm_a4w4(x, wshuffle, x_scales_shuffle, w_scales_shuffle, bpreshuffle=True)
```

这三步全在**计时区外**。如果你的 kernel 每次 launch 都自己重排一遍 scale,
而对手的发表数字是「摆好了直接吃」,**你们比的根本不是一件事**。
⇒ 先把自己这边也做出等价的「预打包」入口(见 [[project_mxfp4_llama_gemm_b200_campaign]]),
再比。本例这一项就值 1%。

### ② 确认对手走的是它的快路,不是 fallback
第一次量完我差点直接下结论,但日志里有:

```
[aiter] shape is M:28672, N:4096, K:32768, not found tuned config ... will use default config!
```

`not found tuned config` ⇒ 它 fallback 到 CK 的 `gemm_a4w4_blockscale`,
**不是反汇编里那个快的 asm kernel**。而且那 7 个 fallback 的 shape 恰好包含我要赢的三行。
⇒ **显式点名扫一遍对手的所有 asm kernel,逐 shape 取最好**:

```python
names = [f"_ZN5aiter{len(stem)}{stem}E"                       # 手工 mangle
         for stem in (os.path.basename(p)[:-3]
                      for p in glob.glob("<hsa>/gfx950/f4gemm/*.co"))]
aiter.gemm_a4w4_asm(A, Bsh, As, Bs, out, name, None, 1.0, 0.0, True, 0)
```

扫完 14 个 kernel:最好成绩仍是 0.975~0.990 × 冻结列 ⇒ 疑虑排除,fallback 并没吃亏
(FC1_wgrad 的 default 路 5477.5 甚至比最好的 asm 5464.8 还快)。
**这一步不做,结论会被一句「你量的是它的慢路」推翻。**

### ③ 头对头必须同进程 ABBA 交错
分开跑时两臂各自暴露在本机跨 run 漂移下(实测 0.6~1.4%),
而我要宣称的余量只有 1.2% —— **同一量级,不够格**。
同进程交错后漂移在配对里抵消:

```python
for _ in range(QUARTETS):
    o1, b1, b2, o2 = t_one(ours), t_one(theirs), t_one(theirs), t_one(ours)
    ratios.append(((b1 + b2) / 2) / ((o1 + o2) / 2))
```

### ④ 每行都带一条 A-vs-A 校准臂
把 `ours` 当成对手再跑一遍同样的四元组,读数必须 ≈0。
本例 12 条校准全部落在 **−0.564% ~ +0.186%** ⇒ 尺子可信,
最小余量 +2.148% 显著高于它。
**没有校准臂的 A/B 不能拿来判生死** —— 见 pitfalls/02 §A-vs-A(同一天我靠它
抓出一个 +54% 的假象)。

## 什么时候该怀疑分母

| 信号 | 说明 |
|---|---|
| 缺口**极其稳定**(跨跑纹丝不动)且**所有参数轴都推不动** | 稳定的缺口更像常数偏移,不像可调的 kernel 行为 |
| 缺口在**全表分布均匀**(本例 12 行全部 −1.0~−3.4%,均值 −2.2%) | 真实的 kernel 短板通常挑形状,不会整表等幅 |
| 分母是**别人在别的机器/别的 harness** 上取的 | 时钟、预热、计时口径、是否含 host 开销,每一项都能值 1~3% |
| 分母来源**没人能复述** | 没有复现脚本的常数 = 传说 |

## 结论怎么写(别混口径)

两个分母都报,并且说清哪个可辩护:

| 口径 | 结果 |
|---|---|
| 对 bench 里的冻结常数 | 10/12 |
| 对本机实测的真对手(头对头 + 校准) | **12/12**,余量 2.15% ~ 16.31% |

⇒ 并给出可执行的收尾建议:**把 bench 的参考列从冻结常数换成运行时实测对手**。

来源: 2026-09-14 MXFP4 Llama GEMM 对标 AITER;详见 [[project_mxfp4_llama_gemm_b200_campaign]]。
相关: pitfalls/02(A-vs-A 校准 · 全局常量全表定价) · methodology/01(造尺子前先 grep) ·
methodology/03(ISA dump 权威)
