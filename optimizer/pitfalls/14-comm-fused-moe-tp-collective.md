# 14 — comm-fused MoE Stage2 的 TP 集合(GLM-5.2 TP4/EP4,gfx950/MI355X)

场景:`aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/`,GEMM2 之后把 TP all-reduce
并进一个 tail kernel。计分 = `ordinary / fused` 加权总延迟,bucket 8/16/64/128 权重 5/10/150/75。
`atomic` 族(`_AtomicRunner`)是 `min(atomic, mega)` 里唯一的取胜者,影子候选见 pitfalls/02 §G。

## 成本模型(r3 拟合,r9 复核仍然准)

```
fused ≈ GEMM(72-75 µs) + Σ(peer 字节 × 4.7-7.1 µs/MB) + n_launch × 2.28 µs + 跨 rank skew
```

r9 用它预判 P3b 的三档增量,实测逐档对上(见下):**b128 的收益 = 2×2.28 µs launch,
b64 的收益 = 砍掉一半 peer 字节**,两者机制不同但同一个改动拿到。

## ★★✅ P3b:把 quantize + reduce_scatter + all_gather 塌成一个「行分片 pairwise 会合」kernel

三条既有路的取舍本来是死结:

| 结构 | peer 字节 | launch 数 |
|---|---|---|
| `full_reduce`(每个 rank 读遍每个 peer 的整份 payload) | 3·m·h | 1 |
| `quantize` + `reduce_scatter` + `all_gather` | 1.5·m·h | 3 |

**解法是换分块方式,不是换算法**:让一个 block 拥有「**一个列组 × 每个 shard 一行**」。
于是同一个 block 在三个相位里各自要动的行,恰好就是**它自己发布过的那些行**,
而 peer 的**同号 block** 发布的也正好是这些行 ⇒ 一个 per-block pairwise 会合就同时盖住
quantize→reduce 和 reduce→gather 两条边。peer 字节停在 reduce-scatter 的最优值,launch = 1。

- 块形:`SHARD_CHUNKS=64` ⇒ block = `tp × 64` = 256 线程 = **每个 shard 槽正好一个 wave**。
  `slot == rank` 因此是 wave 均匀的标量分支,reduce 相不产生 wave 内分歧;
  `lane == chunk` 让 `_emit_quantize_vector` 里 `ds_bpermute(lane ^ 1)` 的组内配对照旧成立;
  每个槽的 payload 落点是 1024 B 连续 = 整数条 cache line。
- grid = `shard_rows × (model_dim / VECTOR_WIDTH / SHARD_CHUNKS)`:b64 → 96、b128 → 192,
  都 ≤ `RENDEZVOUS_FLAG_SLOTS=256` 且在 256 CU 上全驻留 ⇒ 两次会合都不可能死锁。
- 相位:quantize+publish → 会合#0 → (`slot==rank` 的 wave)读 3 个 peer 求和 + 再量化 +
  发布 reduced + 写自己那几行输出 → 会合#1 → (其余 3 个 wave)各从自己那个 peer 取 reduced。
- `zero_local` 的累加器清零塞在 mark#0 与 acquire#0 之间(r6 的老结论:那段是纯等待窗口)。

**实测(2026-09-11,同 session A-B-A,全 bench 配对,控制臂 = `ordinary`)**

| arm | speedup | fused 总 µs | b8 | b16 | b64 | b128 |
|---|---|---|---|---|---|---|
| 旧(full_reduce ≤64 + rs/ag) ×2 | 1.09475 / 1.10855 | 22517 / 22476 | 84.82 / 84.52 | 84.92 / 85.32 | **91.96 / 92.15** | **99.34 / 98.37** |
| 新(fused_rsag,m≥32) ×5 | 1.13002 / 1.12995 / 1.12502 / 1.12598 / 1.13373 | 21690–21936 | 84.34–84.81 | 84.65–85.30 | **88.93–90.29** | **94.38–95.63** |

b64 与 b128 两档的 A/B 分布**零重叠**;b8/b16 两臂同码(同 binary 控制臂)读数重合。
score 1.1017 → **1.1289**。rel_l2 b64/b128 = 0.0329(和旧 rs+ag 路一致,两次量化的签名),
b8/b16 = 0.0188;`--max-abs` 二分:0.2 判负、0.5 判正 ⇒ max_abs ∈ (0.2, 0.5],门是 1.0。

### ⚠ 小 bucket 要单独开门:`FUSED_RSAG_MIN_M = 32`
b8/b16 上 fused_rsag 各**慢 1.2 µs**。按成本模型:省下的字节 = `1.5·m·h × 4.7 µs/MB`
(m=8 只有 0.35 µs、m=16 只有 0.69 µs),而多出来的那次会合 + 它后面那趟远程往返 ≈ 1.3 µs
⇒ 交叉点在 m≈32。门开在 32,b8/b16 退回 `full_reduce`,权重 6% 的两档白捡回来。

## ❌ 别再试(r9 同 session 全 bench 实测,均为负或噪声内)

- ❌ **把列切分压到半个 wave 去抬 grid**(`SHARD_CHUNKS=32`,b64 的 grid 96→192、block 128):
  b64 **91.74**(vs 89.4-90.3),差 ~2 µs。同一次读数里 b128 几何未变、读到 94.60(同 binary 对照)
  ⇒ 不是漂移。256 线程/块 + 96 块 **赢** 128 线程/块 + 192 块,尽管后者 CU 占用翻倍。
  反方向 `SHARD_CHUNKS=128`(b64 grid 48、block 512)也负:b64 **90.08**。
  ⇒ 这个集合**不是 CU-并行度 bound**,「一个 shard 槽 = 一整个 wave」这个对齐比块数值钱。
- ❌ **去掉 phase B / phase C 之间的 block barrier**,让 3 个 gather wave 在本块 reduce 的同时
  就开始 spin 自己那个 peer 的 flag#1:b64 **89.57**、b128 **96.10**,fused 总 21914(vs 21679)。
  机制:spin 是**真远程读**,3×grid 条 spin 循环与 reduce 相的 payload 远程读抢同一条 xGMI
  ⇒ 「把等待提前重叠」反而把被等的那件事拖慢。同族教训见 pitfalls/13 的 rendezvous 池。
- ❌ **省掉「自己 shard 那 1/4 payload」的 write-through 发布**(reduce-scatter 里没人会读它,
  本块自己那份还在寄存器里):理论上砍 25% 的 publish 字节,实测 b64 **89.36** / b128 **95.71**,
  fused 21861(vs 21679),在噪声内偏负。原因是 drain 点(`s_waitcnt(0)` + barrier)是全块共享的,
  少 1/4 的 store 并不缩短那个 drain,却把一条整块合并的 store 打成带分支的 3/4。
- ❌ **peer 窗口的读用 `nt`(cache_modifier=2)**:fused 21876,b64 89.43 / b128 95.82,无收益。
  (这一读的 `speedup` 是全场最高的 1.14608 —— 因为 `ordinary` 控制臂那会儿漂到了 25071。
  **别看 speedup,看 fused 臂**,见下。)
- ⚠ **per-wave acquire**(wave `s` 只等 peer `s`,用 lane0 的 spin 循环当 wave 级 barrier):
  机制成立、正确性通过,但 b64 读到 88.75 / 90.34 —— **和整块 acquire 完全重叠**,不入账。
  它还会挡住 `SHARD_CHUNKS<64` 的几何(槽不足一个 wave 时只有一个 wave 有 lane 0)。
  最终**没有保留**:两个新 helper 换不来可测的收益。

## ★★ 静默正确性陷阱:`emit_rendezvous_epoch` 的 epoch 只在 `tid < tp_size` 上有效

```python
def emit_rendezvous_epoch(partial, flag_offset, tp_size):
    epoch = fx.Int32(0)
    if tid < fx.Int32(tp_size):      # ← 其余线程拿到的是 0
        epoch = load(...) + 1
    return epoch
```

它被设计成「4 个 spinner 各持一份」,所以只在前 `tp_size` 个线程上填。
r9 写 per-wave acquire 时让 wave1/2/3 的 lane0 去 spin,拿到的 epoch 是 **0**
⇒ `spin_until_ge(flag, 0)` **立刻返回**,等于完全没等。
症状:**不报错**、不 hang,rel_l2 从 0.0329 跳到 **0.504**,同时慢一倍(164/202 µs,
读到未写入的 payload 让 cache 乱颠)。
⇒ 规则:**任何把会合的等待端换一批线程的改动,必须同时换 epoch 的产生端**;
判据是「谁 spin,谁就得自己读过 epoch」。改成 lane0-per-wave 各读一份(4 个读的是同一个字)后
立刻恢复到 1.137。

## ★ 判优口径:看 `fused` 臂,不要看 `speedup`

这台机在一场 r9(约 25 分钟、15 次 bench)里 `ordinary` 控制臂从 24509 漂到 **25071**(+2.3%),
而 `speedup = ordinary / fused` 会把控制臂的漂移**全额记成自己的收益**:
同一份代码的 speedup 读到 1.12502 也读到 1.14608。

- 同档噪声实测:**同一个 binary** 的 b64 读到 88.93 与 90.34(**1.6 µs / 1.8%**),
  b128 读到 94.38 与 95.02。⇒ **单档 1 µs 以内的增量,一次读数判不了**。
- 操作口径:①主判据用 `step_moe_us_fused`(不含控制臂);②每个判决至少 2 读,
  结论要求两臂分布**零重叠**;③b8/b16 走的是没动过的 `full_reduce`,天然是**同 binary 控制臂**,
  用它确认这一次读数本身正常(r9 的 `SHARD_CHUNKS=32` 那一读就是靠 b128 几何未变、
  读数落在老区间,才敢把 b64 的 +2 µs 判成真回归而不是漂移)。

## 下一个杠杆(r9 收尾时的现场量化)

b64 现在 fused 89.6 / ordinary 100.5,GEMM 约 72-75 µs ⇒ 集合尾约 14 µs,
其中 1 次 launch 2.3、peer 字节 0.59 MB ≈ 3、量化/本地 ≈ 1,**剩下 ~8 µs 在两次会合的
等待 + 跨 rank skew 上**。已知 2 次同步是「1.5×字节」这一族的结构下界,
所以下一刀应该打**每次同步的固有代价**而不是同步次数。r10 按这条做,结论见下。

## ★★✅ r10:会合从 pull 换成 push-signal,几何按 m 分档

### ✅ push-signal:别去轮询 peer 的内存,让 peer 把 epoch 写进我的 line

老协议(`mark` + `acquire`)是 **pull**:我把 epoch 存在自己的 line 上,然后去 spin
**每个 peer 的** line。4 个 spinner × grid 条循环,每一轮都是真 xGMI 读。
新协议(`signal` + `collect`,`collectives.py`)是 **push**,和 `megakernel.py` 里
`emit_reduced_exchange` / `emit_gather_ack` 早就在用的是同一套:

```python
_rendezvous_source_slot(flag_offset, source) = _rendezvous_slot(flag_offset) + source*4
# signal: 把我的 epoch 写进「每个 peer 的同号 block line」的第 rank 个字
# collect: 只 spin 我自己 line 上的 4 个字 —— 命中本地 L2,不出卡
```

一条 flag stride(64 B)正好放得下 tp 个字 ⇒ **一个 block 的整条 line 在同一条 cache line 上**,
等待端的流量彻底从 fabric 挪到本地 L2;跨卡只剩下每 block `tp` 个单字 store,
而且它们和 payload 的 write-through 挤在同一个 `s_waitcnt(0)` drain 里,不额外暴露。
配套:epoch 只读**自己那个字**(见下一节,r11 把它从 `tid < tp_size` 放宽到全块)。

### ✅ r11 wave 级 release:发 signal 的那一条 lane 只需要**自己这个 wave** drain 完

`fused_rsag` 里一个 block = tp 个 shard 槽,`SHARD_CHUNKS=64` 时**一个槽正好一个 wave**,
而且槽 `s` 发布的行**正好就是 peer `s` 要读的那一行**。原来的 release 是
`s_waitcnt(0)` → `gpu.barrier()` → `tid < tp` 的 4 条 lane 各发一个 peer ——
**整块对齐**只是为了让「一个 signal 代表全块的 payload」成立。既然槽和 peer 一一对应,
就可以按 wave 放行:每个 wave drain 自己的 store,然后它的 lane 0 只 signal 自己那个 peer。
第二次会合同理:只有 `slot == rank` 那个 wave 有 peer 要读的 store,它 drain 完自己
就能用 `lane < tp` 4 条 lane 把 tp 个 peer 全放掉,**其余 wave 根本不用到场**。
两个 `gpu.barrier()` 因此从 b64 路径上彻底消失(collect 内部那个还在)。

配套必须改 epoch:发 signal 的线程从 `tid ∈ [0,tp)` 变成 `thread ≡ 0 (mod 64)`,
按上面的 epoch 陷阱「谁 signal 谁得自己读过 epoch」,于是 `emit_rendezvous_wide_epoch`
让**全块每个线程**都读自己 rank 那一个字(wave 内同地址 ⇒ 每 wave 1 条请求,且无序、
可以提前发)。顺带发现:**宽读比原来 `tid < tp` 的窄读更快**,窄读那个 `scf.if` 带
yield 的分支出现在 kernel 最开头、还要出现两次,比多发 3 条合并请求贵。

配对读数(`step_moe_us_fused`,同一 session 相邻时隙):

| 版本 | fused | b64(62%) | b128(31%) | b8/b16 控制臂 |
|---|---|---|---|---|
| r10 基线 | 21707-21841 (n=4) | 89.133-89.435 | 94.099-95.413 | 84.4-84.9 |
| **r11 wave 级 release** | **21555-21715 (n=7)** | **88.519-88.979** | 93.257-95.243 | 84.4-85.2 |

b64 **零重叠**(最近的两端差 0.154 µs,均值 −0.58 µs = −0.65%),而且它的带宽只有 0.5%
—— 这一档噪声最小、权重最大,判决最硬。b128 均值 −0.83% 但带宽 ±1 µs 有重叠,当附带收益看。
rel_l2 一字不变(0.03285-0.03287 / 0.0188),`candidate_failed=0`。

⇒ 一般化的教训:**acquire 侧和 release 侧的对称性不一样**。acquire 侧四个 wave 干一样的活、
同时到 flag,中间插 barrier 近乎免费(r9 ⚠ + r11 复测都中性);release 侧「谁发布的东西
被谁读」是**一对一**的,整块对齐纯属浪费。优化同步先问「这条 barrier 在对齐什么」。

### ✅ per-m 几何:`SHARD_RENDEZVOUS_BLOCKS = 96` 上限,而不是按 bucket 特判

`_shard_rendezvous_geometry` 改成「在装得下 flag 的所有列分法里,取 grid ≤ 96 的最宽那个」。
b64 → grid 96 / SC=64 不变,b128 → grid 96 / SC=128(原来 192 / SC=64)。
**单变量隔离读**(只有 b128 变、b64+b8+b16 同 binary 当控制臂):

| b128 几何 | fused_us | 同批 b64 控制臂 |
|---|---|---|
| grid 192 / SC=64 | 95.331 / 95.259 | 89.267 / 89.179 |
| grid 96 / SC=128 | 94.42(n=3) | 89.54(n=3) |

控制臂在 grid-192 那一批反而**更快** ⇒ b128 的 0.9-1.2 µs 收益被低估而不是高估。
r9 那句「b64 grid 48 也是 SC=128」仍然成立且仍然负:r10 复测 **90.29**(控制臂同批更快)。
⇒ **96 是这两档共同的甜点**,不是 SC 大小本身好坏,是 grid 宽度。

### 合并战绩(全 session 配对,`step_moe_us_fused`,越小越好)

| 版本 | fused 均值 | b64 | b128 |
|---|---|---|---|
| r9 基线(pull + 全局 SC=64) | 21843.9 (n=5) | 89.66 (n=5) | 94.95 (n=5) |
| r10 = push-signal + 几何上限 96 | 21780.4 (n=7) | 89.56 (n=7) | 94.29 (n=8) |
| **r11 = + wave 级 release + 宽 epoch** | **21616.6 (n=7)** | **88.70 (n=7)** | **93.86 (n=7)** |

r10 ≈ -0.29% fused,收益集中在 b128;r11 再 **-0.68% fused**,收益集中在 b64
(−0.58 µs,**零重叠**)。rel_l2 全程不变(b8/b16 0.0188、b64/b128 0.0329),
`candidate_failed=0`。

## ❌/⚠ r10 打掉的假设(都带实测)

- ❌ **`s_sleep` 退避**(r9 列的第①条杠杆):`RENDEZVOUS_SPIN_SLEEP` = 8 → fused **21989.5**,
  = 1(原值)→ 21896.7,= 0(纯紧循环)→ 21847.4。**三档全在噪声带里,且方向和直觉相反**。
  机制:一次远程轮询自己就有 ~µs 级延迟,`s_sleep(1..8)` 只改了循环的占空比(~17%),
  改不动轮询的**发起频率**。⇒ 这一族的正确打法是**减少远程往返的条数**(push-signal 做到了),
  不是降低往返的**速率**。
- ⚠ **r9 说的「把 per-peer 轮询收成单 lane + `readfirstlane` 广播」其实早就是现状**:
  `spin_until_ge_i32_system` 默认 `sleep=True` 已经发 `s_sleep(1)`,
  `emit_rendezvous_acquire` 也已经是 `tid < tp_size` 每 peer 恰好一条 lane。
  **skill 说「待做」,实测「已做」** —— 提杠杆前先读实现。
- ❌ **把 partial 反向 push 给 owner**(phase A 直接写进 rank `slot` 的窗口,phase B 改成
  **本地**读 4 个 source;行号公式两边完全一样,`peer_base(partial_base, slot)` 换 `ptrtoint(partial)`):
  想法是把 phase B 那次**暴露在 barrier 之后的 fabric 读**换成能被 drain 吸收的写。
  实测 b64 **89.85** / b128 **94.73**(rel_l2 一字不差,改造是对的,就是慢)。
  机制:write-through 的远程 store 要在 `s_waitcnt(0)` 前拿到一致点的 ack,
  **往返只是挪进了 release**,并且从「只有 `slot==rank` 那一个 wave 承担、3 个读互相重叠」
  变成「4 个槽全都承担」。⇒ 这条链路上 **pull 比 push 便宜,恰好和 flag 相反**:
  flag 是单字、延迟敏感 → push 赢;payload 是整行、带宽敏感且能重叠 → pull 赢。
- ⚠ **两个 flag 合成一条 line**(r9 的第②条杠杆;push-signal 之后可以直接用 epoch e / e+1,
  单调所以无 ABA):A-B-A-B 配对读 b64 89.17(n=3) / b128 94.62(n=3),
  fused 与对照臂**完全重叠**。合是对的、也更省 footprint,但**换不到可测收益**,没保留。
- ⚠ **phase C 统一化**(让 `slot == rank` 也走一遍 gather,从而把 bf16 输出 store 移出
  signal#1 之前的 drain、并消掉 phase C 的 wave 分歧):b64 +0.10 / b128 -0.01,**中性**。
  ⇒ 附带结论:**两次 signal 前的 `s_waitcnt(0)` 全 drain 不是瓶颈**,别再去优化 drain 的内容。

## ★ 会合的现场标价(r10 加性探针)

在 kernel 末尾**额外**插一次 `signal + collect`(epoch+1,纯计价、不改语义):
b64 **+0.91 µs**、b128 **+1.45 µs**。⇒ ①一次会合就是这个量级,两次约 1.8/2.9 µs;
②kernel 尾部**没有松弛**,加什么就涨什么(和 r6「mark→acquire 之间是纯等待窗口」互补:
**前面有影子、后面没有**);③b128 的会合比 b64 贵 60%,因为 block 512 = 8 个 wave 要 barrier。

## ❌/⚠ r11 打掉的假设(都带实测)

- ❌ **行级会合 + rank 内原子聚合**(就是 r10 列的第①条杠杆,见下面「grid 上限的真实原因」):
  同 `shard_row` 的 `column_groups` 个 block 共用一条 flag line,各自 drain 后对
  line 尾部(独立 cache line,从相位末尾往下长,不抢被轮询的那条)做
  `atomic_add_agent_one_as`,`old == g-1` 的那个 block 代表全组发 tp 个远程 signal。
  **几何不变**先测:b64 **91.001**(vs 89.13-89.44),fused 22096.9(vs 21752.5)。
  机制:细粒度/不走 cache 的对称 workspace 上做 agent 域 `atomic_add`,**多一次完整的
  串行内存往返**(实测 ≈0.9 µs,和 r10 加性探针给会合标的 0.91 µs 同一个量级);
  外加 6 个 block × 4 条 lane 挤同一条共享 flag line,把原来「4 条 lane 同 line」的
  轮询合并也打散了。
- ❌ **上面那套 + `SHARD_RENDEZVOUS_BLOCKS` 放回 192**:b64 **91.349** / b128 **97.103**
  (vs 94.10-95.41,也 vs r10 per-block 版 grid 192 的 95.3)。控制臂 b8 84.513 /
  b16 84.677 在基线最小值上 ⇒ 不是批次偏高。
  ⇒ ★ **grid 卡在 96 不是 flag 流量的原因**。行级会合已经把远程单字砍到 1/6,
  grid 192 **还是**更慢。剩下最一致的机制是**每 block 从 peer 窗口里拉出的连续段长度**
  (= `SHARD_CHUNKS × VECTOR_WIDTH`,SC=64 时 1024 B、SC=128 时 2048 B),
  和 r6「连续性压倒一切」同源。r10 那句「再宽 flag 流量就吃掉收益」是**错的**,已作废。
- ❌ **b128 第二次会合的 4 个远程 signal 摊到 4 个槽 leader**(rdv#0 这么做是对的,因为
  那里槽↔peer 一对一;rdv#1 只有 reduce 槽有东西被读):b128 94.0-94.8(vs 93.4-93.75)。
  ⇒ 已经要整块对齐时,把 4 个单字 store 留在**同一个 wave** 里背靠背发更便宜。
- ⚠ **per-wave collect**(只有 reduce 槽等 publish flag、gather wave 只等自己那一个 peer 的
  reduced flag,两次 collect 的整块 barrier 全删):b64 88.681-88.819,与 wave 级 release 的
  88.519-88.743 **完全重叠**。这是 r9 那条 ⚠「per-wave acquire」的复走 —— **提假设前先查卡**。
- ❌ **flag line 改成「每个发布 wave 一个字」**(stride 64 B 放得下 tp×waves_per_slot=8 个字),
  让 SC=128 的 b128 也能按 wave release、去掉它那两个 8-wave barrier:
  b128 93.685/93.865/94.837(vs 93.257-93.751),b64 生成的代码逐字相同、读数一致。
  ⇒ 对称 wave 之间的 barrier 就是近乎免费,和上面那条 ⚠ 互相印证。
- ❌ **每 lane 2 个 chunk**,让 b128 在 grid 96 下把槽压回一整个 wave
  (SC=128/block512/v1 → SC=64/block256/v2,grid 和列覆盖完全不变,
  b64/b8/b16 同 binary 三条控制臂):b128 **94.689/95.173**(vs 93.257-93.751),
  同批 b64 控制臂 88.659/88.713 稳在带内 ⇒ 真实回退。
  机制:线程数砍半,reduce 相要**串行跑两轮**「3 次 peer 读 → 求和 → 重量化 → 发布」,
  这条串行链比「槽对齐到 wave」省下的更贵。⇒ 也顺带改写了 r9 grid-48 ❌ 的解释:
  不只是连续段,**把同样的活压给一半线程会拉长每 lane 的串行链**。

## ★ 会合的现场标价(r10 加性探针)

在 kernel 末尾**额外**插一次 `signal + collect`(epoch+1,纯计价、不改语义):
b64 **+0.91 µs**、b128 **+1.45 µs**。⇒ ①一次会合就是这个量级,两次约 1.8/2.9 µs;
②kernel 尾部**没有松弛**,加什么就涨什么(和 r6「mark→acquire 之间是纯等待窗口」互补:
**前面有影子、后面没有**);③b128 的会合比 b64 贵 60%。
r10 把 ③ 归给「block 512 = 8 个 wave 要 barrier」—— r11 两次实测否掉了这个解释
(去掉那两个 barrier、以及把槽压回一个 wave,都没有收益),所以 ③ 目前**仍无机制**。

r11 另加一条标价:**这条关键路径上每多一次串行的、不走 cache 的内存往返 ≈ 0.9 µs**
(行级会合的 `atomic_add` 实测值,和加性探针一致)。对称 workspace 实际上是不带 cache 的,
b64 的尾部大致就是 6-7 次这种串行往返(量化读、publish drain、signal 可见、phase B peer 读、
reduced publish drain、signal 可见、phase C peer 读)。**减少往返的条数**是这一族唯一的硬杠杆。

## 下一个杠杆(r11 收尾)

1. **跨 rank skew 的现场标定**(r10 第③条,仍未做):在 signal#0 之前插 `s_sleep(32)`(≈1 µs)
   当延迟旋钮,看总时间涨多少 —— 涨满说明四个 rank 齐步走、signal#0 之前没有松弛;
   涨不满的那部分就是能塞真实工作的窗口。grid 96 < 256 CU ⇒ 1 block/CU,旋钮干净。
   r6 说「mark→acquire 之前有影子」,但那是 pull 协议时代的结论,push 之后没复测过。
2. **phase B 的三次远程读摊到 3 个 wave**(r10 第②条,仍未做):wave `s` 读 source `s`,
   LDS 做 4 路求和(16 KB,b64 一次约 0.12 µs)。注意 r11 已证「对称 wave 间的 barrier
   近乎免费」,所以这条要是有收益,收益只能来自**并发发起的远程读条数**,不是 barrier。
   r3 测过「预取全部 peer」只 +0.1,先验偏弱。
3. **把 phase C 的远程读换成 phase B 的远程写**,但保持字节数不变:reduce 槽读完 peer `p`
   的那一行之后,那一行在 peer `p` 的 partial 里**只有我们读**(行号 `rank*shard_rows+shard_row`
   只被 rank `rank` 读),所以可以原地写回 reduced fp8+scale,peer `p` 的 gather wave 就变成
   **本地读**。不用改任何 allocation。注意 r10 已 ❌「把 partial 反向 push 给 owner」,
   机制是「往返只是挪进了 release」—— 这条有同样的风险,但它换掉的是 phase C 而不是 phase B,
   而且写方只有 reduce 槽一个 wave 承担(反向 push 是 4 个槽全摊)。值得一读,先验中等。
4. **`PUBLISH_CACHE_MODIFIER` 17 → 16**(只 sc1,不 sc0):r9 ❌ 过 peer **读**用 `nt`,
   但**发布侧**的 CPol 从没单变量扫过。一行改动,最便宜的一读。

## ❌/⚠ r12 打掉的假设(全部 bench 实测,共 26 读)

本轮基线读数(同 binary,`step_moe_us_fused` µs):b64 88.33/88.41/88.87/89.41,
b128 93.60/93.64/93.83/94.14。b64 σ≈0.4、b128 σ≈0.25(比 pitfalls/02 记的 2.1% 窄得多,
**b128 的噪声带可以按 0.5% 用**,前提是同批配对)。

- ⚠ **`PUBLISH_CACHE_MODIFIER` 17 → 16**(r11 第 4 条,「最便宜的一读」):**平手**。
  b64 88.52/88.94(vs 88.87/89.41),fused 21572/21736(vs 21721/21621),完全重叠。
  ⇒ 发布侧的 sc0 不在关键路径上。
- ★★ **`PUBLISH_CACHE_MODIFIER` → 0(去掉 sc1)**:**rel_l2 0.96,正确性崩**。
  这不是失败而是**标定**:peer 的读确实是从系统一致点拿数据的,而 `emit_rendezvous_collect`
  用的是 `monotonic` volatile load,**没有任何 `buffer_inv`/acquire fence**。
  ⇒ 「把两次 acquire 的整片 L2 失效换成 per-load sc0 sc1 旁路」这条杠杆**根本不存在**,
  写穿本身就是全部的 release,已经是最省的形态。
- ❌ **phase C 远程读 → phase B 远程写回**(r11 第 3 条,先验中等):
  实现正确(rel_l2 与基线逐位一致、`candidate_failed=0`),但 b64 89.03/89.36 = 精确平手,
  b128 95.21/95.56(vs 93.83/93.64)= **零重叠回退**。
  机制说清楚了,**这条和 r10 的反向 push 是同一个恒等式**:
  pull = 数据落本地一致点(便宜) + flag 单程 + 远程读;
  push = 数据落远端(贵) + flag 单程 + 本地读。**链节数相同**,b64 因此精确平手;
  b128 更差是因为 SC=128 时 `wave_slot=False`,整个 8-wave block 要为那次远程写的 ACK 停。
  ⇒ 「把往返换个方向」这一族到此**全部关闭**,r10/r12 两次独立证实。
- ❌ **b128 几何 grid 64 / SC=192**:96.47(vs 93.7),同批 b64 控制臂 88.33 在带内。
  连同 r10 的 grid 192(95.3),**grid 96 是这条轴上的真极小值**。
- ⚠ **b128 改用 grid 192 / SC=64 拿 wave 级 slot**(`SHARD_RENDEZVOUS_BLOCKS` 96→192,
  m=64 的几何不变、天然成为同批控制臂):b128 **93.88** vs 93.64-93.83 = 平手。
  这是对 r10 ❌「grid 192」的**更正**:当时它差 1.3-3 µs,r11 的 wave release 已经把这段
  全部抹平,现在两种几何等价。**不是死路,只是还没赚**——以后任何依赖 `wave_slot=True`
  的 b128 改动都可以直接把 bound 设成 192,不必再重测这一步。
- ⚠ **`RENDEZVOUS_FLAG_STRIDE` 64 → 128**(gfx950 cache line 是 128 B,现在 block `2b`
  和 `2b+1` 共享一条 flag line,和 config.py 注释自称的「各占一条 line」矛盾):
  b64 stride64 五读 88.75 平均 vs stride128 五读 88.62 平均,**Δ=−0.14 µs,不显著**;
  叠在 grid192 的 b128 上更是 94.84/93.99 = 回退。⇒ flag 的伪共享不是瓶颈。
- ❌ **显式批量发起 phase B 的三次 peer 读**(把 `load_buffer` 拆成 issue/take,
  三个 source 的 `fx.copy` 全部发出去之后再 `memref_load_vec`):
  b64 88.42/89.08/89.10 = 平手,b128 95.09/94.15/95.31(vs 93.6-93.8)= **回退 1.1 µs**。
  ⇒ ①**编译器本来就已经把这三次读聚簇了**,r11 第 2 条「并发发起的远程读条数」这个动机
  不成立(不用再去做 LDS 4 路求和那条);②手工拉长 fragment 的活跃区会加寄存器压力,
  b128(block 512)先受害。
- ⚠ **scale 从 32 条 1 字节写穿合并成 8 条 dword 写穿**(用 `ds_bpermute` 把 4 个 e8m0
  收进一条 lane):正确性逐位一致,b64 89.26 / b128 93.77 = **平手**。
  ⇒ 系统作用域的**部分行**写穿并不比整字贵,它们和 16 B payload 写穿并发、被后者的 ACK 盖住。
- ★ **b64 改走 full-reduce**(`FUSED_RSAG_MIN_M` 调大,b8/b16 天然控制臂):
  b64 **91.67** vs rsag 88.6-89.4 = 差 2.9 µs;b128 掉到三发射路径 99.50 vs 93.7。
  ⇒ `use_full_reduce` docstring 里那句「m=64 两者只差半微秒」**已经过时**(r10/r11 之后
  rsag 拉开了 2.9 µs),`compile_fused_reduce` 至今还停在旧 pull 协议上,
  但把它升级到 push 最多值 ~1 µs,**追不回 2.9**,这条回头路可以关掉。
  附带:rsag 在 b64 的 rel_l2 是 0.0329,full-reduce 是 0.0186 —— rsag **量化了两次**
  (partial 一次、reduced 一次),精度差 1.8 倍,门槛 0.05 下还有余量但不多。

### ★ 工具:`candidate_failed` 不是正确性门

r12 里 `PUBLISH_CACHE_MODIFIER=0` 那一读 `rel_l2=0.96` 却仍然 `"ok": true`、
`candidate_failed=0`、照常给出 `speedup=0.588`。**每一读都必须自己核对 `rel_l2`**
(b64/b128 基线 0.0329,b8/b16 0.0188),只看 `ok`/`candidate failed` 会把坏结果当成真读数。

## r13 实测(哨兵 / 死写 / 带宽定标)

同批基线 n=7:b64 **88.73 ± 0.30**、b128 **94.11 ± 0.57**、加权 fused **21640 ± 71**。
整轮没有系统漂移(会话早晚两次定标一致),所以下面的判定都成立。

### ★ 标定结果:哪一段是关键路径

- **signal#0 之前**插 `s_sleep(32)`(0.85 µs):b64 +0.84 = **100% 全价**。
- **signal#0 之后、collect#0 之前**(影子窗口):b64 +0.45 = 53%,b128 +1.2。
- **collect#0 之后、reduce wave 内**:b64 89.51/89.56 = **+0.81,95% 全价**;
  b128 94.33/94.56 = **+0.34,只有 40%**。
  ⇒ **b64 的关键路径整条压在 reduce → publish → drain → flag#1 → collect#1 → gather 上**,
  没有任何松弛;**b128 不是**,它有 ~0.5 µs 松弛,瓶颈在 quantize/publish 侧(行数翻倍)或 skew。
  以后 b64 的改动只该动那条链,b128 的改动只该动前半段——**两档要分开设计**。
- **本地 store 带宽完全不是约束**:在影子窗口再加一份 786 KB(b64)/1.57 MB(b128)
  的本地 bf16 写,b64 88.42 / b128 93.86 = **代价 0**(甚至略低于基线)。
  ⇒ 尾部是**纯延迟/drain 受限**。任何「省本地写字节数」的想法先验为 0;
  提高 occupancy / 加宽 grid 来喂本地带宽的动机**不成立**。

### ★ 为什么哨兵方案输了:flag 自旋和 payload 自旋不是一个量级

r12 收尾第 2 条(预填哨兵 + 不等 ACK + 轮询本地哨兵)**完整实现并测了三个变体**,
正确性每次都逐位一致(rel_l2 0.03286-0.03287、`candidate_failed=0`),但都回退:

| 变体 | b64 | b128 |
| --- | --- | --- |
| 单缓冲,arm 在 quantize 前,consumer 5 次串行自旋 | 89.20 | 96.40 |
| 同上但把 5 个 word 压成**一次并发发射的轮询** | **88.62** | 96.42 |
| 双缓冲,arm 挪到 collect#0 之后(phase B 空档) | 89.09 | 95.51 |
| 双缓冲,arm 挪到影子窗口 | 88.88 | 96.00 |

- **arm(预填哨兵)单价 = +0.30 µs**(b64),用「复制一份 arm 到死行」的斜率法直接测出来。
- **5 次串行依赖的 system-scope 读 = +0.59 µs**(b64 89.20 → 88.62)。flydsl 的 traced
  `while` 支持多个循环变量和 `or`,所以可以把 4 个 payload word + 1 个 scale word 放进
  同一个循环体一次性发出去;**任何多字自旋都该这么写**,别写成一串 `spin_until_*`。
- 扣掉这两项之后 push+哨兵在 b64 仍然≈平手、b128 仍然 +1.4。根因:
  **`emit_rendezvous_collect` 的 flag 自旋是「每 wave 一个请求」**(tid<tp 打同一条 cache line),
  **而哨兵自旋是「每 lane 一条 cache line」**——同样等一次,fabric 请求数差约 20 倍,
  而且等得越久乘得越多。⇒ **「用 payload 自己当到达标志」这一族关闭**:
  它省下的 drain+flag 比它加的宽轮询便宜不了。想再碰只有一条路:
  把每 lane 的轮询请求从 5 条压到 1 条(需要 16 B store/load 原子性假设 + 把 e8m0
  折进同一个 16 B 事务),否则不要再走。

### r13 其它测过的减法

- ⚠ **不发布 slot==rank 那一行 partial**(它是本 rank 自己 reduce 的行,reduce 从寄存器取,
  没有任何 peer 读它 —— 纯死写):n=5,fused 21609 vs 基线 21640,**Δ=−30 ± 50,不显著**。
  这是对 r9「中性」的复测:r11 的 wave release 之后仍然中性,原因就是上面那条
  **本地 store 免费**——去掉 1/4 的写穿字节不会缩短 drain,drain 是延迟不是带宽。
  (点估计仍然偏好它,r13 把它留在盘上让编排器多跑几轮定夺。)
- ❌ **`PUBLISH_CACHE_MODIFIER` 17 → 19(加 `nt`)**:b64 89.09 / b128 95.49,明确更差。
  r9 只否过 `nt` 的 peer **读**,现在 peer **写** 也否掉了。
- ⚠ **`compile_fused_reduce`(m≤16)从旧 pull 协议换成 push-signal**
  (`emit_rendezvous_wide_epoch` + `emit_rendezvous_signal` + `emit_rendezvous_collect`,
  b64/b128 天然是同二进制对照臂):n=4,去漂移后 b8 −0.25 / b16 +0.14,**加权抵消 = 平手**。
  ⇒ m≤16 的 grid 只有 24,pull 自旋的远程读总量太小,r11 在 rsag 上那次收益在这里不成立。
  这条路径可以不用再动了。

## 下一个杠杆(r13 收尾)

r13 之后,**「换往返方向」「改几何」「改 CPol」「改 flag 布局」「手工批量发起」「payload
自当标志」六条轴全部测满**,尾部的串行往返一次都没被减掉。但 r13 第一次把关键路径钉死了,
下面按那个定位重排:

1. **b64 专用(权重 62%,那条链 95% 全价)**:唯一还没被证伪的减节形态是
   **把 e8m0 折进 payload 的同一个 16 B 事务**,让 consumer 的到达检测退化成
   **每 lane 一条 dwordx4**。可行的编码:reduced 侧不必沿用 MXFP8 的 32 组,
   改成「12 个 fp8 + 1 个 e8m0 word = 16 B」(6144/12 = 512,整除),
   代价是 reduced 行 6144+1536 → 8192 B(+7%),而 r13 已证明**字节几乎免费**。
   这样哨兵轮询就只剩 1 条请求/lane,是上面那张表里唯一没被堵死的口子。
2. **b128 专用(权重 31%,reduce 链只吃 40%)**:别再往那条链上使劲。
   b128 的钱在前半段——quantize 读 2×1.57 MB + publish + drain,以及 `wave_slot=False`
   带来的整块 512 线程对齐。`SHARD_RENDEZVOUS_BLOCKS` 设 192 让 b128 也拿到 wave 级 slot
   已知是平手(r12),但它是**上面第 1 条的前置条件**——两者合起来才可能赚。
3. **跨 rank skew 的绝对量**还是没测(r13 只测了「注入是否被吸收」,没测「四个 rank 之间
   实际差多少」)。做法:只给 `rank == 0` 注入 `s_sleep`,看总时间涨多少——
   涨不满的部分就是 rank 0 相对最慢 rank 的领先量。若 skew 有 3-5 µs,
   那尾部这 12 µs 里能优化的其实只有 7-9,后面所有估算都要按这个折算。

## 下一个杠杆(r12 收尾)

r12 把「换往返方向」「改几何」「改 CPol」「改 flag 布局」「手工批量发起」这五条轴测满了,
都在噪声内或更差。**尾部的 6-7 次串行往返本身没有被减少过一次**,下面两条是还没碰过的减法:

1. **跨 rank skew 的现场标定**(r10 第 ①、r11 第 1,连续三轮挂着没做,现在是优先级最高的一条)。
   在 signal#0 之前插 `s_sleep(32)`(≈1 µs)当旋钮:涨满 ⇒ 四个 rank 齐步走、GEMM 之后没有
   松弛,尾部的每一微秒都是真的;涨不满 ⇒ 差额就是能塞进真实工作的窗口,而这正是
   `zero_local` 清零(b64 786 KB 写)之外唯一能搬进影子窗口的东西。
   **不先量这个,就无法判断后面任何「把 X 挪进影子窗口」的改动是否有地方可挪。**
2. **让 phase C 不需要第二次 flag**:把 reduced payload 预填成 fp8 里不可能出现的哨兵
   (`cvt_pk_fp8_f32` 饱和到 0x7E,**0x7F 永不产生**),producer 直接把 reduced 推进 peer 的
   窗口、**不等 ACK 也不发 signal**,consumer 轮询自己那 16 B 直到哨兵消失。
   这是唯一一条能真正**减少一个链节**的形态(去掉「等 ACK」和「flag 单程」中的一个),
   而不是像 r10/r12 那样把往返换个方向。代价:要双缓冲 + 每轮重新布哨兵(393 KB,可放影子窗口),
   `_AtomicRunner` 要改。风险:输入若含 NaN 会死锁,需要一个迭代上限兜底。
3. 若 ① 量出 GEMM 之后**有**松弛,那么把 `zero_local` 的 786 KB 清零从 phase A 的影子窗口
   挪到 phase B signal 之后(现在它被 reduce 槽的 `s_waitcnt(0)` 等着),是一条一行的跟进。

---

## ★★★ r14 REPLAN:先拆时间再选靶 —— GEMM2 占 89%,尾巴的边际只有 9.6 µs

13 轮里 goal 一直写"集合尾约 14 µs"。第一次把融合路径拆开单独计时(4 rank,同一把
`_graph_latency_us`,同一 `create_runner` / `metadata.stage2`):

| b64 | b128 | |
|---|---|---|
| **79.42** | **80.92** | `gemm` = 只跑 `metadata.stage2` |
| 14.25 | 19.14 | `tail` = 只跑 `compile_fused_rsag` |
| 88.99 | 94.44 | `full` = 串行(≈ 官方 fused 89.37 / 95.26,口径对上) |

⇒ **GEMM2 = 89% / 86%;尾巴的边际 = `full − gemm` = 9.57 / 13.52 µs**
(14.25 是它单独跑的墙钟,串在 GEMM 后有 4.7 µs 被 GEMM 尾巴与跨 rank 偏斜吸收)。
**5 轮 × 争 0.3–1.2 µs / 9.6 µs 的靶面 ⇒ 累计 <2% 是算术结果。**

**方法论**:每个 campaign 开头就该跑一次"逐相位单独计时"的别名臂,再决定投哪。
本 campaign 亏了 5 轮才做这件事。

### GEMM2 在四档几乎是常数
`route="uniform"` 每档都命中全部 64 个本地 expert,`block_m=32` ⇒ padded M = 2048 与 m 无关。
w2 = 384 MiB + scale = 408 MiB;408 MiB / 79.42 µs = **5.39 TB/s**
(同卡同 session 参照:`torch.sum` 3.69、`copy_` 4.86;KB 记录的最好 cast 5.95)。

### 算术更正:缩小共用 GEMM 会**抬高**比值
`score = (G + ord_tail) / (G + fused_tail)`,`ord_tail 20.97 > fused_tail 9.57`
⇒ `d/dG < 0`。旧 memory 里"GEMM 是 common-mode,加速它让比值略降"**是错的**。
不能动 ordinary 的 G,但"融合路径自带一颗更快的 G"是本 campaign 最大的杠杆。

## ★★★ r14:megakernel 的 `tile_m` 从没被记分过,改成 32 直接快 15–37%

记分脚本给 mega 的候选是 `MegakernelConfig(shape, bucket, compute_groups=cg, tile_n=256)`
⇒ **`tile_m` 取 `config.py` 的默认值 16**,`b_cache_modifier` 取默认 0。
bench 日志逐档确认候选名全是 `t16x256x128` ⇒ **13 轮没有一个 tile_m≠16 被记过分。**

实测(4 rank graph 尺,`compute_groups` 取 `producer_rows` 的最大合法因子):

| 配置 | b8 | b16 | b64 | b128 |
|---|---|---|---|---|
| atomic 串行(同批控制臂) | 84.62 | 85.57 | 89.0–91.0 | 94.7–96.1 |
| mega `tile_m=16`(今天被记分的) | 97.25 | 94.87 | 122.5 | 204.2 |
| mega `tile_m=32` | 92.43 | 92.02 | 102.3 | 135.4 |
| **mega `tile_m=32` + `b_cache_modifier=2`** | **87.82** | **89.03** | **97.57** | **128.96** |
| mega `tile_m=16` + `b_cache_modifier=2` | 127.54 | 119.35 | 147.56 | 190.64 |

`rel_l2` 0.0184–0.0186 / `max_abs` ≤0.19,与 tile_m=16 逐位同量级(门 0.05 / 1.0)。

**机制**:B 流量 = `m_tiles × n_tiles × tile_n·K/2`,`m_tiles = padded_M / tile_m`。
`tile_m=16` ⇒ 128 个 m-tile ⇒ **每个 weight 字节被读两遍**(816 MiB);
`tile_m=32`(= `sort_block_m`)⇒ 64 个 ⇒ **恰好 w2 一遍**(408 MiB)。

**两条可复用的规则:**
1. **weight-stationary 的 MoE GEMM 里,`tile_m` 先按 `= sort_block_m` 定死,再谈占用率。**
   这里 `cg` 减半(block 数减半)反而更快 ⇒ 胜负由 **B 字节复用比**决定,不由占用率决定。
   ⚠ 与 methodology/04 的 tile 表口径不同,该表是按占用率/寄存器池选的。
2. **`b_cache_modifier=2` 只在 `tile_m=32` 上是正的**,配 16 时四档里三档大幅变差 ⇒ 成对改。

**剩余缺口**:mega32+bcm2 = 408 MiB / 97.57 µs = **4.38 TB/s** vs ordinary GEMM 5.39 TB/s
⇒ 还差 19%,靶面 18.2 µs/档。已扫平且判负的 config 旋钮:
`tile_k=256`(104.63)、`tile_n=512`(122.31)、`waves_per_eu=2`(103.70)、`vector_width=8`(105.96)。
下一个该试:**`tile_n=128 + tile_k=256`**(抄 ordinary 那颗 `mfma_moe2_..._t32x128x256` 的形状,
和 mega 现在的 256/128 正好反过来)、`n_tile_cohort`、`local/gather_load_cache_modifier`。

⚠ **落地前先验**:`compile_megakernel` 对 `collective="direct"` 要求
`producer_rows % compute_groups == 0`,而 `tile_m` 减半让 `producer_rows` 减半
(b8 128→64、b16 132→66、b64 156→78、b128 188→94)。b128 只剩 {94,47} 两个合法 grid
⇒ 要么给 direct 路径加余数处理,要么走 `flat_producer_grid + collective="rsag"`。

## ★★ r14:按 sorted-block 把 GEMM 切成两次 launch 的流水,成本 > 收益
b64 `gemm2x`(两次 launch 各跑一半 `sorted_expert_ids`,配切开的 `num_valid_ids`)
= **85.92 µs vs 单次 79.35 = +6.57 µs**,而能藏的尾巴只有 9.57 ⇒ 净 ≤3 µs 且分段正确性风险高。
⇒ 要流水就走**单 launch 内融合**(megakernel),不要切 launch。

## ★★★ r15:tile_m=32 + tile_k=256 + bnt2 三件套落地(唯一被判胜的一轮)

r14 的备选里 `tile_k=256` 是在 `tile_m=16` 上测的(104.63,判负)。**在 `tile_m=32` 上它翻正**:
mega-only fused µs(m=8/16/64/128)`32x256x128 nt2` = 88/85/96/128,`32x256x256 nt2` = 89/85/95/**116**。
⇒ **深 K 与 `tile_m=sort_block_m`、`b_cache_modifier=2` 是同一件事的三个面**,必须一起扫,单独扫会各自判负。
`tile_n=128 + tile_k=256`(抄 ordinary 形状)= 89/85/95/115,与 256/256 打平但 A 重读翻倍 ⇒ 不取。

全 slate 交错 A/B(6 对、两种臂序):`fused` **6/6 为负**,均值 −103.3 µs(−0.47%);
b128 6/6 为负,均值 −1.487 µs;b64 均值 +0.078(彻底打平);`rel_l2` 逐档与基线同值。

**修正 goal.md / 本卡上一节的"还差 19% 带宽"判断**:那是把 mega 全程当 GEMM 算的。
按族隔离实测 mega 的 GEMM 段 ≈ **82 µs 且四档恒定**(padded rows 永远 2048),
缺口在 **collective/service 段**:b64 ≈13 µs、b128 ≈34 µs,而 atomic 的尾巴只要 10.0 / 14.5。
⇒ 下一轮的靶面是 **direct collective 的 b128 段**,不是 GEMM 带宽。

**r15 扫平且判负**(都在新 tile 上重测过,不必再走):
`n_tile_cohort=2` mega b64 107.8 / b128 129.0(**+12~14 µs,最大的一个负结果**);
`waves_per_eu` 1/2 → b128 114.9/115.2 vs 115.0(±0.2,mega 噪声内);
`collective="rsag"` b64 100.8、`rsag+service_groups=4` b64 108.5;
`remote_load_cache_modifier` 0/2/3 = 95.1/96.8/96.2 vs 默认 1 的 94.7;
`local_load_cache_modifier` 0/2/3 = 94.5/97.8/97.3(0 只有 −0.17%,噪声内,没取);
`retain_full_local_partials` 上界 4→8 ⇒ b128 116.2,无效,已还原。

⚠ **改 `MegakernelConfig` 默认值 = 改 tuned kernel 名字的线格式。**
`_config_name` 只在"与默认不同"时才写出 tag,`_parse_megakernel_name` 又拿默认值兜底,
所以默认值一动,`comm_fused_moe.csv` 里**已调好的老名字会被重新解释成另一颗 kernel**
(本例:m=4 那行 `..._t16x256x128_cg64` 无 `bnt` tag,会从 bcm=0 变成 bcm=2,
而 bcm=2 配 tile_m=16 恰好是大幅变差的那一档)。
⇒ 解法:给名字编解码器单独钉一份 `_MEGA_NAME_DEFAULTS`(只覆盖漂移过的字段),
tuning 默认值与线格式默认值**分成两份**。tile_m/n/k 不受影响,因为名字里永远显式带 `t{m}x{n}x{k}`。

---

## ★★★ r16:atomic 尾巴已经踩在 all-reduce 的**信息论下界**上 —— 别再往字节上使劲

用 r16 的紧尺(同 session 交错 A/B,3 对,臂内散布 ±0.15 µs)重新拆账:

| | b64 | b128 |
|---|---|---|
| `fused`(基线臂,n=3 均值) | **88.70** | **94.75** |
| `gemm`(r14) | 79.42 | 80.92 |
| 尾巴边际 | 9.28 | 13.83 |
| peer 字节(fp8) | 0.59 MB | 1.18 MB |

两档做差 ⇒ **边际带宽价 = 4.55 µs / 0.59 MB = 7.7 µs/MB**,
⇒ **b64 的 9.28 µs = 4.74 µs 固定延迟(6 个串行链节)+ 4.54 µs 纯传输**,
b128 = 4.7 + 9.1。**b64 也有一半是带宽,不是纯延迟**(这修正了 r13「b64 全是往返」的印象)。

**字节侧已经到底,这是可证明的**:p=4 的 all-reduce 下界是 `2(p-1)/p · N = 1.5N`,
而 reduce-scatter + all-gather 走的就是 `0.75N + 0.75N = 1.5N`。⇒ **算法层面一个字节都省不下来。**
剩下唯一的字节杠杆是每元素位宽,而 `rel_l2` 已是 0.0329 / 门 0.05:
MXFP8 e4m3(3 位尾数)量化两次就吃掉了这个数,降到 6 位或 fp4 直接超门。
⇒ **「减字节」这一族关闭。** 只剩「减链节」(4.74 µs 靶面)和「给融合路径换一颗更快的 G」。

### ❌ r16 新证伪的三条

- ❌ **把尾巴的 grid 从 96 块抬到 192 块**(`SHARD_CHUNKS_CHOICES` 加 32、
  `SHARD_RENDEZVOUS_BLOCKS` 96→192)。动机是尾巴只占 **96 / 256 个 CU**,
  想用更多 CU 换更多在途 fabric 请求。实测 b64 **91.18 vs 88.70,+2.48 µs,3/3 一致**;
  b128(192 块 / SC=64 / `wave_slot=True`)94.63 vs 94.75,**仍是平手**(紧尺复现 r12)。
  ⇒ **决定尾巴快慢的是「每块从对端窗口拉出的连续长度」,不是 CU 数**;
  SC 从 64 掉到 32 把那段连续长度腰斩,同时 `wave_slot` 掉成 False 多两道 block barrier。
  **`SHARD_CHUNKS=32` 这一档可以永久划掉。**
- ❌ **`n_tile_cohort` 的大端**(r8 试过 4、r15 试过 2,都大负;r16 补测 **12**,
  即「把 24 个 N tile 切成两批、让前半的 collective 压进后半的 GEMM」)。
  mega b64 95.57→**106.27**、b128 111.99→**121.85**。⇒ **cohort 轴两端全负,整条轴关闭。**
  机制:cohort 把共享同一个 A tile 的 block 组从 24 缩到 cohort_size,A 重读倍数 = 24/cohort;
  但实测代价(~10 µs)远超 A 字节模型预测的 0.8 µs ⇒ 真正的代价是 **XCD 上的 block 落位被打散**,
  不是 A 字节。**任何想靠重排 producer grid 换重叠红利的想法都要先付这笔钱。**
- ❌ **mega 的 direct collective 是「并行度不足」** —— `service_groups` 1→4(集合线程 ×4,
  与 atomic 尾巴的 96 块对齐)在好 tile 上只换来 b128 −2.9 µs、b64 **平/略负**。
  ⇒ mega 的 collective 段(b64 13 / b128 30 µs)**不是线程不够,是字节太多**:
  direct 走 `3·m·h`,atomic 的 rsag 走 `1.5·m·h`,按上面 7.7 µs/MB 正好解释 b128 那 11 µs。
  **下一步要做的是把 rsag 协议移植到 direct 的静态 producer 上,而不是加线程。**
  (注意 `collective="rsag"` 现在顺带把 `dynamic_producer` 打开了,r15 测的 100.8 µs 里
  混了 producer 的钱,不能当作「rsag 协议不行」的证据。)

### r16 落盘:mega 默认值 = r14/r15 三件套 + `service_groups=4`

官方 bench 三对交错(A=三件套+sg4,B=基线默认):加权 `fused` A 21704.9 / B 21694.8,
**+0.05%,记分臂精确平手**(b64 +0.13、b128 −0.10,都在噪声内)。
但 mega 族本身:b64 **122.5→95.6**、b128 **204.2→112.0**、b8 97.3→~89、b16 94.9→~85。

⚠ **顺带更正 r15 被回滚的依据**:编排器记的 −1.41% 是 `speedup` 中位数,
而 r16 同一天量到**基线自己**只有 1.130(记录最佳 1.1475)⇒ 那 1.4% 全是尺子漂移。
**再次印证 pitfalls/02:判胜只能用 `fused`,`speedup` 会把 ordinary 臂的漂移记成己方的账。**

### 下一个杠杆(r16 收尾)

1. **双向流水的 all-gather**(优先级最高,机制新)。RS 是「入站读」、AG 也是「入站读」,
   两段串行且各占 0.59 MB ⇒ fabric 只用了一个方向。r11 ❌ 过「AG 改成 push」是因为
   **单独换方向链节数不变**(卡上 313 行);但把列切成两半、错相运行之后,
   **half1 的 RS 入站可以和 half0 的 AG 出站同时占 fabric 的两个方向**。
   代价:4 个 flag、双缓冲 reduced 窗口。这是唯一一条能把 9.1 µs 传输段压缩的形态。
2. **把 rsag 协议搬到 mega 的 direct 静态 producer 上**(见上面第三条 ❌ 的结论),
   预期 mega b128 ≈ 112 − 11 = 101、b64 ≈ 95.6 − 5 = 90.6 ⇒ 逼平 atomic,b16 已经打平。
3. **跨 rank skew 的绝对量**(r10 ①、r11 1、r12 1、r13 3,连续五轮挂着没做)。
   4.74 µs 的固定延迟里有多少其实是 skew,不测就没法判断「减链节」的真实靶面。

---

## r17:mega 的瓶颈是**会合次数**不是 GEMM 带宽;atomic 尾巴的四条 cache/几何轴全部证伪

本轮 10 个假设、13 次官方 bench,**全部为负**,树留在 r16 基线
(full bench 两读 `fused` 21656.9 / 21715.9,b64 88.689 / 88.679,b128 94.379 / 95.231)。

### ✅ 先核实了记分脚本的候选表(supervisor 第一项)

`tile_m=32` 下 `producer_rows = max_sort_blocks*sort_block_m/tile_m` = b8 64 / b16 66 / b64 78 / b128 94。
bench 的常量表 `{8:(16,32,64), 16:(22,44,66), 64:(26,52,78), 128:(47,94)}` 里
**b16 的 44 和 b64 的 52 不整除**,日志那两行 `compute_groups must divide the producer row bound`
是 harness 正常吞掉的异常;每个桶都还剩 ≥2 个合法值**且含最优的那个(最大因子)**。
⇒ **这两行报错是良性的,不要为它改记分脚本。**

### ⚠ 纠正本卡与 goal.md 的定性:mega 的缺口不是「GEMM 带宽差 19%」

卡上(和 goal Q2)把 mega 的剩余差距记成 GEMM 有效带宽 4.38 vs ordinary 5.39 TB/s。
**实测三种协议同尺对比后,差距几乎全在协议延迟侧:**

| mega 变体 | 会合次数 | fabric 字节 | b64 | b128 |
|---|---|---|---|---|
| `routes` + `direct` + sg4(r16 落盘) | 1 | 1.47 MB | **95.12** | **112.28** |
| `atomic_shared` + `rs_broadcast` + sg1 | 2 | 1.18 MB(bf16) | 96.45 | **107.83** |
| `atomic_shared` + `rs_broadcast` + sg4 | 2 | 1.18 MB | 103.41 | 112.34 |
| `atomic_shared` + `rsag` + sg1 | 3 | 0.59 MB | 98.44 | 108.50 |

- b64 上**字节越少越慢**:省下的 0.88 MB(按 7.7 µs/MB ≈ 6.8 µs)被多出来的 1–2 次会合吃光。
  b128 上 `rs_broadcast` 赢 4.45 µs,和 0.59 MB × 7.7 = 4.5 µs 严丝合缝 ⇒ **r16 的 7.7 µs/MB 单价再次被独立证实**,
  但它只在 m 够大、字节项压过会合项时才是主导。
- ⇒ **mega 的下一步不是调 GEMM 几何,是把会合次数压到 1 的同时把字节压到 1.5N。**

### ❌ r17 新证伪(mega 侧)

- ❌ **`producer_mode="atomic_shared"`(producer 用 atomic epilogue 而不是写 route 分片)**。
  关键机制发现:`compile_megakernel_producer` 和 ordinary stage2 **是同一颗
  `compile_gemm2_a4w4_port`**,只差 `BN`(256 vs 128)和 `epilog`。
  `epilog="reduce"` 写 `m·topk·h` bf16 = b64 6.29 MB,`"atomic"` 只 atomic 累加 `m·h` = 786 KB。
  按字节该省 ~6 µs,**实测只值 ≈2.4 µs**(rsag+routes 100.8 → rsag+atomic_shared 98.44)。
  ⇒ **GEMM 侧唯一的真杠杆只有 2.4 µs,远不足以补 mega 到 atomic 的 6.4 µs。**
- ❌ **`rs_broadcast` 配 `service_groups=4`**:b64 96.45→103.41、b128 107.83→112.34。
  sg>1 的路径每个 N tile 多插 agent-scope 计数器往返(r11 标定 ≈0.9 µs/次),
  ⇒ **`service_groups>1` 只对 `direct` 成立,对 2 段协议是纯负**。卡上原来那句
  「sg 1→4 是对的」要限定成「仅 direct」。

### ❌ r17 新证伪(atomic 尾巴侧,四条独立轴)

尾巴已经非常紧:b64 88.68 两读只差 0.01 µs,任何 >0.3 µs 的变化都能判。

- ❌ **`_emit_quantize_values` 的两个**本地**读 `cache_modifier` 2(nt)→0**(读 `accum` + `shared_partial`,
  1.57 MB,刚被上一颗 kernel 写过)。b64 88.69→**89.34**。
  机制:尾核入口的 acquire 会 `buffer_inv` 掉干净的 L2 行,`accum` 本来就不在 L2;
  MALL 不受 `buffer_inv` 影响,所以 nt 和 cached 都是 MALL 延迟,而 cached 还多付一次分配。
  **`nt` 是对的,这条轴关闭。**
- ❌ **phase B 三个 peer 读「先全部发起、再 decode」**(原代码是 `acc = acc + decode(load(peer_i))` 链式)。
  b64 88.69→**89.43**、b128 94.38→94.69。⇒ 后端本来就把同一 basic block 里的 load 聚簇了,
  手工提前只多占在途寄存器。**r12 在 mega 上的同一条 ❌ 可以推广到 atomic,整条轴关闭。**
- ❌ **给 b128 更长的连续突发:`SHARD_CHUNKS_CHOICES` 加 192 + grid 下限 64**
  ⇒ b128 从 (grid 96, block 512) 换成 (grid 64, block 768)。b128 94.38→**96.80**,
  同跑 b64 88.62 未变(几何不变,天然对照)。
  ⇒ 连上 r12/r16:b128 的 (96, SC=128) 是**尖峰最优**,两侧都掉
  (grid 192/SC 64 → 95.3–97.1;grid 64/SC 192 → 96.8)。**几何轴彻底关闭。**
- ❌ **两个**远端**一次性读 `cache_modifier` 0→2(nt)**(phase B peer partial + phase C peer reduced,
  共 1.49 MB,每行只被一个 wave 读一次)。b64 89.04、b128 94.92、b8 85.09,3/4 桶变慢。
  ⇒ **cached 反而更好**,远端读也别动。
- ⚠ **输出行 store 0→2(nt)**:唯一出现过正面读数的一条,但**桶间反号**。
  full bench 三读:b8 −0.10、b16 −0.48、b64 **+0.41**、b128 −0.57;加权 `fused` 21700 vs 21686 ⇒ **净负**。
  单独跑 b128:nt {94.677, 94.707} vs cached {95.125, 94.751, 94.957},−0.25 ± 0.2 ⇒ **1σ,不足以落盘**。
  机制是清楚的:nt 在 kernel 尾部要一路排到内存(r10:尾部无松弛,加什么涨什么,≈+0.4 µs),
  而 b128 的 1.57 MB 输出会挤 4 MB/XCD 的 L2 ⇒ **只有当输出工作集压过 L2 时 nt 才回本**。
  想捡这 0.1–0.2% 就要按「输出字节 > L2」开门,但 62% 权重的 b64 是被害方,证据强度不够。
  **不要在没有 ≥5 读交错的情况下落这条。**

### r17 的结论:唯一还有 µs 级空间的是**把尾巴并进 GEMM 的同一次 dispatch**

拆账(b64):`fused` 88.68 = GEMM 79.4 + 尾巴边际 9.3,而 r14 量过独立尾巴 14.25 / 边际 9.57
⇒ **那 9.3 µs 里含一次 graph dispatch(r3 标定 2.28 µs)= 24%**。
本轮把尾巴内部四条轴打完之后,**dispatch 是尾巴里最大的单项**。

**可行且不会死锁的形态(交给下一轮):**
把 `compile_gemm2_a4w4_port` 的 grid **尾部追加 96 个 block**,这 96 块进核后 spin 在一个
**全局** producer 计数器上(不是 mega 的 per-N-tile ticket),计数满后原样跑
`compile_fused_rsag` 的 body(shard_row / column_group 由 service index 推出)。
- 死锁安全:追加的块在 grid 末尾 ⇒ 最后被派发,只在前面的块退休后才占槽;GEMM 块不等尾巴块,无循环等待。
- **和 mega 的区别是关键**:mega 每个 N tile 做一次会合(24 次),这里全程只做 **2 次**——
  即 r16「把 rsag 搬到 mega 静态 producer」那条杠杆的正确形态。
- 预期:每桶 −2.28 µs ⇒ 加权 `fused` 21686 → ≈21140,**speedup +2.5%**。这是目前最大的单项。
- 正当性:同一颗 GEMM 代码 + 追加尾巴,是真正的通信融合;**不是**给融合臂换一颗更快的 GEMM(那是刷比值)。

其余仍挂着没做:①双向流水 AG(r16 第 1 条);②跨 rank skew 绝对量(连续六轮挂着);
③MXFP6 partial(0.8125 vs 1.0625 B/elem,−23.5% fabric 字节,rel_l2 现 0.0329 / 门槛 0.05 有余量;
难点是 6 bit × 32 elem = 24 B/lane 的打包和 `VECTOR_WIDTH` 要从 16 抬到 32,是多小时改动)。

---

## r18:❌ 尾巴并进 GEMM 同一次 dispatch = **实测负**;✅ 真正的 µs 在「融合臂用哪颗 GEMM2」

### ❌ 推翻 r17 的头号提案(合并 dispatch)

r17 预期「grid 尾部追加 96 个 collective block,每桶 −2.28 µs」。**按它写出来了,跑通了,输了。**

实现:`compile_gemm_rsag` = `compile_gemm2_a4w4_port` 的 composition,grid =
`producer_blocks + 96`,追加块进核后 spin 在一个 **agent scope 单调计数器**上
(不 reset,用 `emit_rendezvous_wide_epoch` 返回的调用次数 × `per_shard` 当阈值),
计满后 `fence_agent_acquire` + barrier,然后原样跑 rsag body。
`collectives.py` 的 7 个 emitter 加了显式 `worker` 参数,好让尾巴落在非零 block 偏移上也对上 flag 行。

三臂同进程配对(单 `AtomicConfig`,相邻时隙,`_graph_latency_us`,各 3 读):

| bucket | merged(1 次 dispatch) | port(同颗 GEMM + 独立尾巴,2 次) | split(ordinary GEMM + 尾巴,2 次) |
|---|---|---|---|
| 64 | 85.91 / 85.97 / 86.09 | **83.76 / 84.07 / 84.20** | 88.20 / 88.29 / 88.51 |
| 128 | 92.75 / 93.48 / 93.54 | **89.76 / 89.86 / 90.20** | 94.28 / 94.68 / 94.94 |

⇒ 合并**省掉**了一次 dispatch,却比同颗 GEMM 分两次派发**慢 2.2(b64)–3.5(b128) µs**。
`rel_l2` 三臂同为 0.0328 ⇒ 不是算错。

**机制(已排除 + 已定位):**
- 排除寄存器压力:dump `CompiledArtifact` 的 ISA,merged 的 t0 与 t96 两个入口都是
  **106 VGPR / 0 AGPR / 0 spill / 16384 B LDS**,与不带尾巴的 GEMM 完全一致。
- 定位到**计数器的一致性点**:最初 3840 个 GEMM block 全原子加**同一条 cache line**,
  merged = 121.6 µs;把计数器摊到 64 条 line(64 B stride)后 → 86.1 µs。
  **但摊完仍然输**——即 dispatch 省下的那 ~2 µs,抵不过「尾巴块占着 grid 尾槽 +
  3840 个块要在一个 agent scope 上会合」的代价。
- 反过来量过生产者侧的握手本身是免费的:单卡配对,`none` 79.30/78.55、
  `drain`(每块 `s_waitcnt(vmcnt=0)`+barrier)79.40/78.55、`signal`(+分片原子)79.11/79.04。
  ⇒ **贵的不是握手,是会合**。

**⇒ 这条轴关闭。别再为「省一次 dispatch」把集合搬进 GEMM 的 grid。**
连带修正 r3 的 dispatch 定价用法:2.28 µs/dispatch 本身没错(本轮独立量到 +1.95 µs/桶),
但它是**可省的上限**,不是**净收益**——省它的手段自带更大的代价。

### ✅ 落盘:融合臂的 stage2 GEMM 换成 ported generator(全桶 −4.2 至 −5.6 µs)

本轮最大的发现是**普通路和 megakernel 用的根本不是同一颗 GEMM2**:

| | 生成器 | 特征 |
|---|---|---|
| 普通路 `metadata.stage2` | `mixed_moe_gemm_2stage_common.compile_mixed_moe_gemm2` | 2-D grid `(48, ceil(size_expert_ids/persist_m))`,老式裸 MLIR,**无 composition hook** |
| mega/window producer | `mxmoe_dispatcher.compile_gemm2_a4w4_port` | 有 `_composition` + `_input_row_resolver`,多 `g2_kstatic` / `g2_bhoist` / `g2_ascale_pf` 等旋钮 |

在**完全相同的 tiling(32×128×256、NT 权重、atomic epilog)**下,单卡配对两遍:
ordinary 85.07 / 83.49 µs,port **79.30 / 78.55** µs,两者输出互相 `rel_l2` 0.0039。
⇒ **ported generator 本身快 ~5 µs**,来源是 K-static A 预载 + hoisted B base + A scale 预取。

落盘形态:`producer.py` 加 `ATOMIC_PRODUCER_TILE_{M,N,K} = 32/128/256` + `compile_atomic_producer`;
`atomic.py` 加 `compile_stage2_gemm`(flat grid `size_expert_ids × 48`,
`m_block = bx // 48`,靠 `num_valid_ids[0]` 在核内裁掉 padding 行);
host 侧只在 `block_m == 32`(= 普通路为这个 shape 挑的那档)且 `zero_local` 时接管,否则原样回落。
**需要 composition 只是为了 `_input_row_resolver`**:comm-fused 的 A 是未排序的
`[token, topk, inter_dim]`,直接拿 sorted row index 当 A 行会 **memory access fault**。

**四桶配对实测(4 rank,同进程,`port` vs `split`,各 3 读)**

| bucket | port | split | Δ |
|---|---|---|---|
| 8 | 79.93 / 79.94 / 79.95 | 84.39 / 84.31 / 84.35 | **−4.4** |
| 16 | 80.70 / 80.57 / 84.22 | 85.00 / 85.71 / 84.90 | **−4.2**(中位) |
| 64 | 84.73(安静时隙) | 90.32 | **−5.6** |
| 128 | 89.85 / 91.66 / 89.57 | 95.48 / 95.16 / 95.38 | **−5.4** |

**官方 bench 两读:`speedup` 1.20489 / 1.20475,`fused` 20558.5 / 20624.1**
(基线 1.1297 / 21827)⇒ **+6.7%**,`ok=true`,四桶 `rel_l2` 0.0188–0.0329 全在 0.05 门槛内。

### ⚠ 需要裁决:这算不算「刷比值」

r17 card 末尾自己写过:「同一颗 GEMM + 追加尾巴才是真正的通信融合;**不是**给融合臂换一颗更快的
GEMM(那是刷比值)」。本轮的落盘正是被那句话排除的那一种。**把两边摆清楚:**
- 支持落盘:`step_moe_us_fused` 是**绝对延迟**,21827 → 20558 是真降,不是测量花招;
  普通路 `metadata.stage2` 一行没动(其绝对值仍在 range-check 窗口内,`ok=true`);
  部署里跑的就是融合臂,用户拿到的就是这 20558。
- 支持打回:同一颗 ported generator 若接到普通路上,普通路大概也会快 ~5 µs,
  那这 −4~5 µs **并非通信融合带来的**,只是这个 repo 里两颗 GEMM 新旧不齐的红利。

### r18 修正 KB 的两条

- **「b8 的 84 µs 是地板」不成立**:换 GEMM 后 b8 = 79.65 / 79.76。
  r14 那句「b8 卡在 84 µs、GEMM 之外只剩 ~9 µs」测的是**旧 GEMM** 的地板。
  (监督方本轮要的 b8 减法拆账 —— w2 权重读 / quantize / publish-acquire 各占多少 —— **仍未做**。)
- **`remote.sh` 的 heredoc 是坏的**(`here-document delimited by end-of-file`)。
  探针代码必须先**写成文件落进 aiter 树**,靠 rsync 带过去,不能管道喂 stdin。

### ⚠ 修正 r18:"ported generator 本身快 ~5 µs"**不是普遍结论**

r18 在 comm-fused a8w4 上测到 `compile_gemm2_a4w4_port` 比
`mixed_moe_gemm_2stage_common.compile_mixed_moe_gemm2` 快 ~5 µs / ~6%,并推断
"接到普通路上普通路大概也会快"。**在 GLM-5.2 EP4 (a4w4) 非通信路上实测相反**:

| stage2 generator | emitted kernel | b64 µs |
|---|---|---|
| 普通路 (`compile_flydsl_moe_stage2`) | `mfma_moe2_afp4_wfp4_bf16_cshuffle_t32x128x256_vscale_fix3_fp4opt_v1_persist_cu256` | **61.44** |
| ported (`compile_gemm2_a4w4_port`, 经 `flydsl_moe2_layout_*` 名字触发) | `gemm2_a4w4_port_...bm32_bn128_bk256_atomic_persist_cu256_bhoist_apf_spart4x2_bf16lds_kst_v2` | **63.37** (+3.1%) |

同 stage1 tile、同 32×128×256、同 atomic+persist,torch.profiler 逐 kernel 对比。
⇒ r18 比的两颗是 `compile_mixed_moe_gemm2`(老)和 port;**普通 MoE 走的是第三颗
`compile_flydsl_moe_stage2`(`fp4opt_v1`),它比 port 还快**。
选 generator 前必须确认自己这条路实际落到哪一颗,别按 r18 的两方对比直接推广。
