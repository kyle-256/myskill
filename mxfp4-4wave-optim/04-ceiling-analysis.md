# 物理天花板分析(K=28672, GPU7, 干净 GPU)

## Ceiling 探针(500-sample Event 测量)

| 探针 | TF med | 含义 |
|---|---|---|
| prod real | **5401** | 当前最优 |
| FP4_CONSTSC=1 | **5401** | scale-load 已被 GAVOID 完全藏住! |
| FP4_WLNODSR=1 | **5693** | skip ds_read(broken),ds_read 成本=292T |
| FP4_WLNOG2S=1 | **6462** | skip g2s(broken),g2s 成本=1061T |
| FEWOP=1 | **5648** | 单 operand reg(garbage),operand 多样性成本 |

## 关键洞察

### 1. Scale-load 成本已为 0
CONSTSC(常量 scale)= prod real(5401) → **scale-load 成本已被 GAVOID 完全藏住**。
SCVGPR+ILV 把 scale buffer_load 交织进 mfma,GAVOID 确保不竞争 issue slot。

### 2. g2s 是主瓶颈(1061T)
g2s = buffer_load_dword×4...lds(全局→LDS 的 DMA),是 HBM-BW-bound。
减少 g2s 需更大 tile(更多 B 复用),但 4-wave 256 AGPR 锁死 256×256 tile。

### 3. ds_read 成本(292T)= 唯一可部分优化的
96 ds_read_b128 / 256mfma。减少到 64(aiter)需 BK128 compact-LDS,
但 fly BK128 loop 开销使总 TF 更低(4597 vs 5401)。

### 4. 测量方法对比
| 方法 | 说明 | 数字 |
|---|---|---|
| test_mxfp4_4w DETRUNS=20 min/med | Event 100-sample,20次 det | 5474/5401 |
| _bench500.py 500-sample | 更精确 | 5460/5401 |
| _bench2000.py 2000-sample | 分布分析 | 5453/5397 |
| rocprof(--kernel-trace) | 真实 kernel 时间 | best 5427 / med 5330 |

**rocprof 是唯一 kernel-only 数字**(Event 包含 host overhead)。

### 5. 不同 K 的 TF 对比(prod 配置)
| K | med TF | 说明 |
|---|---|---|
| 8192 | 4603 | 短 K-loop,固定开销大 |
| 14336 | 5110 | |
| 28672 | 5401 | 标准测试 shape |
| 57344 | **5481** | 2× K,hiding 更充分 |

## VGPR/AGPR 预算(4-wave occ=1)
- V cap = 512 dwords/lane(独立寄存器堆)
- A cap = 256 dwords/lane(独立)
- 生产 kernel: V=424, A=256, spill=0
- 可用 headroom: 512-424 = **88 VGPR**
- 但 2 operand set = 2×192V → 总 384V → 超 512 cap → RAGreedy crash

## LDS 预算(BK256)
- 总 LDS: 160KB/CU
- 生产用量: 144KB(A 2×32KB + B 2×32KB + scale 16KB)
- scale LDS(SC_lds)即使 SCVGPR 时也被分配(浪费 16KB)
- A3+B2 = 96+64+0 = 160KB = 满(无法 3-buffer)

## 为什么 5500 med 在 K=28672 不可达
```
prod 5401 = g2s-HBM-BW-bound
g2s 1061T 只能减不能消,但减 g2s 需更大 tile = AGPR 溢出
ds_read 292T 只能减不能消,减 ds_read 需 BK128 = loop 开销更大
SCVGPR 已消 scale-load = const 天花板 = prod 天花板
∴ K=28672 下 5500 med 超出 4-wave BK256 物理顶
```
