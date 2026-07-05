# wgrad feed-bound 死路全清单：占用率非杠杆，根因 LDS 转置读 feed 带宽

> 类别: 踩过的坑 · 主题标签: wgrad, occupancy, LDS-feed, dead-ends

**根因定性**：wgrad 这条 kernel 的瓶颈是 **LDS 转置读 feed 带宽**，不是占用率。occ=1 的 512-VGPR 满载锁死了 prefetch 深度；任何"升 occ"的改动都会引入 feed 争用，净负。真正剩余杠杆 = 在 256 tile 下提 feed 效率（LDS bank / transpose-read / phase-barrier），需 rocprof 定位，别再盲扫参数。

## 占用率不是杠杆（升 occ 全负）

- ❌ 别再试：**8-wave 升 occ(occ≥2)**。反而更慢——occ=1 的 512-VGPR 满载锁死 prefetch，升 occ 引入 feed 争用。
- ❌ 别再试：**4-wave whole-loop occ=2**。实测否决：8-wave 2148 < safe-3buf occ=1 2214，feed 恶化盖过 latency-hiding。
- ❌ 别再试：**缩 tile**（为了升 occ）。净负。

## 结构杠杆本轮全判负（勿重试）

- ❌ 别再试：**store-defer**。LDS 160KB 悬崖：x2 中性，x3 **-18.6%**。
- ❌ 别再试：**免一个操作数转置**（`ds_read_b64_tr_b8` 换 plain `ds_read_b64`）。ISA 等价，**0 收益**——转置读不比 plain 读贵，省不下来。
- ❌ 别再试：**epilogue write128 / tr16**。**-2~7.7%**。
- ❌ 别再试：**跨 tile 重叠**。**-5%**。
- ❌ 别再试：**epilogue 双缓冲去 drain**。**-1~3.5%**。
- ❌ 别再试：**barrier 参数扫**。全平。
- ❌ 别再试：**增大 reg tile**。根因是 occ=1 满载锁死 prefetch，加 reg 只会更挤。

## 旋钮全平（<2%，别再扫）

- ❌ 别再试：worst shape 的 **swizzle{5} × xcd{1,2,4,8} × vmcnt{1..8}** 全平（<2%）。瓶颈是 feed 带宽，不是这些旋钮。

## 测量方法论坑

- ❌ 别再信：**"跳过整条指令测天花板"类探针**（如 `PT_TR_HALF` 跳过读）。跳过读 ≠ 换成更少的等效读。真实替换后（`ds_read_b128` 换 2×`tr-b8`）因带宽受限**收益归零**。测"去掉 X 的天花板"必须用真实替代指令，不能靠删指令——否则天花板虚高、误导方向。

---
来源: flydsl-fp8-gemm-results/SKILL.md, 10-grouped-wgrad-4wave-3buf.md
