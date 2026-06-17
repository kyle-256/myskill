# NN dgrad kernel 优化

## 问题根因

dgrad 的输出 shape = [M_total, K_fwd]（K_fwd = 模型隐藏维度）。
fwd 的 N（= 权重输出维）是 K_fwd；dgrad 的 N（= 输出维）是 K_fwd（通常 4096~8192）。

当 M_total 小（B=1 small-M MoE GateUP，M=512）时：
- output tiles = G × ceil(M/BLOCK_M) × ceil(K_fwd/BLOCK_N)
- 例：G=1, M=512, K_fwd=8192, BLOCK_M=256, BLOCK_N=256 → 2×32 = **64 tiles**
- 256 CU 上只有 64 tile → **严重 occupancy 不足**，923 TF

rocprof 验证（B=1 grok-up M=512）：
- fwd（N=32768）: 2793 TF（已满载）
- dgrad（N=8192）: **923 TF**（欠载）

## 修法：M-branch autotune（BLOCK_M=128）

判断 128-row tiling 是否还在 CU wave 之内：

```python
bm128_tiles = G * ceil(pm/128) * ceil(N/256)
if not trans_b and bm128_tiles <= _num_cus():  # NN only, 小-M 路径
    return mk(128, 1, 0, 0)   # 单 config，无需 autotune
```

BLOCK_M=128 把 M-tile 数翻倍（M=512: 64→128），填满 CU。

## 边界 sweep 结论（N×G×pm 24 case，0 mismatch）

| gate=Y（bm128 wins） | gate=N（bm256 wins） |
|---|---|
| **+5.3~30.6%**，无例外 | **+21.8~47.5%**，无例外 |

→ gate=Y 时 bm128 永远赢，**单 config 直接 return，不需 autotune**（不存在 bm256 赢的情况）。

## gate 公式

```python
G * ((pm + 127) // 128) * ((N + 255) // 256) <= _num_cus()   # _num_cus() = 256 on MI355X
```

对应物理含义：128-row M-tiling 总 tile 数不超过 CU 数。M≥1280（N=8192）时 tile 数=320>256，触发 tail cliff，bm128 反输。

## BLOCK_M=256 大-M 保持 3 个 bm256 候选

```python
cands = [(256, 8, 4, 0), (256, 1, 0, 0), (256, 8, 8, 0)]  # 同 fwd/NT
```

NT fwd 不受影响（NT 的 N 已经很大，不存在 occupancy 不足的问题）。

## 实测数据（persistent=False / nn8w 路径，GPU0-3 并行）

| M | bm128 | bm256 | gain |
|---|---|---|---|
| 512 | **1056** | 886 | +19% |
| 1024 | **2038** | 1764 | +16% |
| 4096 | 1928 | **3040** | bm256 +58% |

## 陷阱

BLOCK_M=128 要求 `BLOCK_M >= 128 and BLOCK_M % 128 == 0`（kernel 有 assert）。
BLOCK_N=512 会撞 CShuffle EPL assert（EPL=16 ≠ 8），不要扫 bn=512。
