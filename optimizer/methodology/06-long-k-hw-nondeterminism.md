# 长 K(K28672)HW 级间歇非确定:判 race 必须用短 K 做干净 det0

> 类别: 方法论 · 主题标签: measurement-noise, long-K, determinism, race-detection

- **现象**:长 K(K28672)在高负载机上有 HW 级间歇非确定(det)。官方 intrinsic K28672 即使 `DETRUNS=15` 也出 det(15233/152750);raw-baseline 出 det 39926 等。这些 det 幅度**比自研改动本身还大**,足以淹没 <1% 的真信号。
- **对照**:K8192 / K16384 三者均 det0 干净。→ 非确定**阈值在 K16384~K28672 之间**。
- **归因**:纯 HW / 长-kernel 效应,与 kernel 逻辑无关(官方 intrinsic 也复现,证明不是自研 race)。WHY:长 kernel 在高负载机上运行时间更长,更易踩到硬件级的间歇性非确定来源。
- **教训 / 判 race 规程**:分辨 <1% 真信号,必须用**短 K(K8192/K16384)做干净 det0 + 交错对标**。**不能**用 K28672 的 det 指标判 race——K28672 出 det 不等于 kernel 有 race。

---
来源: agpr_phase5_mono.md
