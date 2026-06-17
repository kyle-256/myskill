# SCVGPR Race 根因诊断(最关键突破)

## 背景
SCVGPR(scale 直读 VGPR,去掉 scale LDS ds_read)路径快(~5370),但 racy(det≠0)。
历史上一直被误判为"vmcnt 乱序 race",实际根因完全不同。

## 诊断过程

### Step 1: 排除 vmcnt
测试 WLV=0(vmcnt 全排空)→ **仍然 racy(det 98K)**
结论: race **不在 vmcnt**

### Step 2: 排除 lgkmcnt 边界
测试 ELGK=0(lgkmcnt 全排空)→ **仍然 racy**
结论: race **不在边界 drain**

### Step 3: 决定性实验 — CONSTSC
```bash
FP4_CONSTSC=1  # scale 全设 127 常量
```
结果: **仍然 racy(det 612K)**
结论: race **根本不在 scale 值**,而在 **operand 的 cross-wave g2s→read**

### Step 4: 根因定位
SCVGPR 去掉 scale 的 LDS ds_read → stream 变短 → 
FP4_INPLACE_1BAR=1(只 phase-B 有 barrier)的 phase-A 缺少 cross-wave 同步 →
**operand cross-wave g2s→read race**

### Step 5: 修法
```bash
FP4_INPLACE_1BAR=0  # 每 phase 都有 s_barrier
```
结果: **SNR 55.6 det 0,5358/5282 TF** — 一发即中!

## 结论
| 修法 | 结果 |
|---|---|
| vmcnt(0) | 仍 racy |
| lgkmcnt(0) | 仍 racy |
| CONSTSC | 仍 racy |
| **1BAR=0** | **det 0!** |

**Race 根因 = cross-wave LDS barrier 不足(非 vmcnt)**

## 为什么会误判
- SCVGPR 缩短了 vmem stream
- 旧 1BAR=1 配置:只在 phase-B 放 barrier,相当于 2-buffer 里 A side 无同步
- 原来有 scale LDS ds_read 时 stream 更长,phase-A 的 cross-wave race 被掩盖了

## 关键代码位置
```python
# mxfp4_gemm_8wave.py line ~1909
_elgkb = 0 if _ROBUF else _elgk
_ipend = f"s_waitcnt vmcnt({_WLV}) lgkmcnt({_elgkb})" + _bar
_ipenda = f"s_waitcnt vmcnt({_WLV}) lgkmcnt({_elgkb})" + ("" if (_1bar or _nobar) else "\ns_barrier")
```
`_1bar` 控制是否在 phase-A 也加 s_barrier。FP4_INPLACE_1BAR=0 → `_1bar=False` → phase-A 也有 barrier。

## 深层原理
gfx950 的 4-wave WG 中,wave 0/1 共享 A operand,wave 0/1/2/3 共享 B operand。
g2s(buffer_load→LDS)是 wave 协作完成的。barrier 确保所有 wave 的 g2s 都落地,
跨 wave 的 ds_read 才安全。SCVGPR 减少了 stream 长度,使 phase-A 的 g2s 在没有
barrier 保护时可能被其他 wave 的 phase-B ds_read 读到 stale 数据。
