# MXFP8 e8m0 scale 广播必须全程 Int32（提前 cast Uint8 高位损坏静默错）

> 类别: 踩过的坑 · 主题标签: mxfp8, e8m0-scale, flydsl-bitops, silent-numeric-bug

- **最关键的 MXFP8 authoring bug**：e8m0 scale 做 OR/SHIFT 广播时，若提前 cast 到 `Uint8`，高位会损坏——广播出的 4 个字节各不相同，match 仅 **~9%**，结果是数值垃圾（静默错，不报错）。
- **正确做法：全程保持 Int32**，位运算前绝不 cast：
  - `e8_i32 = ep + I32(127)`（范围 0..255，**不 cast**）
  - `bcast = e8_i32 | (e8_i32<<8) | (e8_i32<<16) | (e8_i32<<24)`
- **通用规律**：flydsl 里凡做位运算（OR/SHIFT 广播等）不要提前 cast 到 `Uint8`；Uint8 只有 8 位，左移 8/16/24 会把有效位移出、高字节全 0 或截断，导致每字节不一致。
- ❌ 别再试：先 `Uint8(e8m0)` 再做 `| <<` 广播——high-byte 损坏、match ~9%、纯垃圾。失败机制=8 位容器承不住 <<8/<<16/<<24 的位移。

---
来源: mxfp8-8wave-devloop/SKILL.md
