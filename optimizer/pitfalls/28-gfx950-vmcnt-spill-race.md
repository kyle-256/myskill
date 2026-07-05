# gfx950 vmcnt 非 FIFO race：scratch_load 与 buffer_load_lds 混排，只在 spill 时暴露

> 类别: 踩过的坑 · 主题标签: gfx950, vmcnt-race, spill, LDS-direct

## 根因
- **scratch_load（编译器 SGPR spill 回填）和 buffer_load_dwordx4...offen lds（LDS-direct GMEM→SMEM）都走 vmcnt，但两者之间 vmcnt 不保证 FIFO。**
- 编译器对 spill 发的 `s_waitcnt vmcnt(N>0)` 不安全——scratch_load 可能还在飞，`v_readfirstlane` 读进 stale value → `s_mov_b32 m0` 用 stale m0 → 下一条 buffer_load_lds 写到错 LDS 地址 → SMEM 错位 → MFMA 污染 → bit 不一致。
- **只在 SGPR pressure 触发 spill 时暴露**（无 spill 时不出现）。

## race harness（如何复现/统计）
- `repro_race_*.py`：`ref=kernel(inputs)` 后循环 `kernel` + `torch.testing.assert_close(out, ref, rtol=0, atol=0)` 严格 bit-exact 统计 races。
- **ITERS 用 50000+ 才有统计意义**：0.005% 量级 race 在 10K iters 看起来像 0/N，要 50K-200K 才稳定可见；多 GPU 跑确认不是单卡 fluke。

## disasm 提取 GPU code（从 .so 抠）
- `LINE=$(roc-obj-ls $SO | awk '/gfx950/{print;exit}')` 取 offset/size
- `dd` 抠出 `/tmp/gpu.hsaco`，`llvm-objdump -d --triple=amdgcn-amd-amdhsa --mcpu=gfx950` 反汇编
- `llvm-readelf --notes` 取 reg notes（`.private_segment_fixed_size` / `.sgpr_count` / `.sgpr_spill_count` / `.vgpr_spill_count`）
- 工具在 `/opt/rocm/bin` 和 `/opt/rocm/lib/llvm/bin`。

## disasm 签名（如何确认是这个坑）
- `private_segment_fixed_size > 0`（有 spill）
- 主循环里 `scratch_load_dword` 夹在 `buffer_load_dwordx4...offen lds` 之间
- 紧跟 `s_waitcnt vmcnt(N>0)` + `v_readfirstlane` + `s_mov_b32 m0` + `buffer_load_lds`
- 若 `private_segment == 0` 但仍 race：看 SGPR 是否 spill 到 VGPR lanes（`v_writelane` / `v_readlane`）——lane ops 不走 vmcnt，不会 race。

## （已废弃的早期尝试，勿用）双管齐下法
早期（gpt_oss v1）方案 A/B 已被 reviewer 明确否决（见下方"死胡同"），不是真正修法：
- A) 降 spill：`-mllvm -sink-insts-to-avoid-spills=true` + 保留 `tile.reserve_pinned_regs()`
- B) prologue 2-step：`buffer_load_lds` 拆成 `buffer_load → VGPR → ds_write`，配 `-mllvm -amdgpu-enable-merge-m0=true`

reviewer 核心观点：**"硬件 vmcnt 行为是约定，编译器在合理 pressure 下不该 spill。出现 spill 是 kernel 写法逼出来的，去 kernel 源头改。绕通道、加 flag、占 pinned 区都是把症状藏起来。"** 真正的定案修法是下面的修法一/二/三——从源头消除 VGPR spill，reg notes 要求 `private_segment_fixed_size=0` / `vgpr_spill_count=0`。

## 第一类 vmcnt FIFO race 三修法（消 VGPR spill，定案）
**修法一 — SGPR 化地址：**
- SMEM 地址全 SGPR：用 `__builtin_amdgcn_readfirstlane(warp_id*STRIDE)` 强制 SGPR。
- **删 `sts_offsets` 里的 lane_id 部分**：buffer_load_lds 硬件按 data_size 自动 per-lane stride（b128=16B/lane、b32=4B/lane），源码加的 `lane_id*16` 是冗余的。
- scale 路径同理 `readfirstlane(warp_id*64*4)` 删掉 `+lane_id`。
- 实测 grouped fwd `17 vgpr_spill / 72 scratch → 0/0`。

**修法二 — outer 解析基址：**
- outer kernel 解析 per-group 基址，inner `compute_tile` 只接收 resolved 的 **5 个 base ptr**（`a_grp_ptr/b_grp_ptr/a_s_grp_ptr/b_s_grp_ptr/c_grp_ptr`）。
- inner 里 `a_base=a_grp_ptr+(int64_t)pid_m*k` 一步加法、无 i64 mul/split-load → inner SGPR/VGPR live set 接近 dense single-GEMM。
- `compute_tile` 必须 `__forceinline__`（`__noinline__` 会 -25% perf）。

**修法三 — 输出转换避软件分支：**
- float→CType 转换 specialize，bf16 用 raw-bit truncation：`r.data=(uint16_t)(__builtin_bit_cast(uint32_t,f)>>16)`。
- 因为 `hip_bfloat16(float)` 默认 ctor 有 round-to-nearest-even 软件分支 → SCC 结果 lane-spill 到 v111 产生 **7 个 v_writelane/v_readlane**。
- `__half(float)` 是单条硬件 `v_cvt_f16_f32` 但 bf16 无对应；gfx950 有 `v_cvt_pk_bf16_f32`（pack 2 fp32→2 bf16 硬件 round）。
- truncate vs round-to-even 半位精度差、SNR 不变。

## 三类 race 速查表（本卡是类 1；类 2/3 详情见 pitfalls/45-gfx950-hw-walled-races.md「三类 race 速查表」）
| 类 | 冲突指令对 | 触发 | 修法 | commit |
|---|---|---|---|---|
| (1) vmcnt FIFO | scratch_load vs buffer_load_lds（wave 内） | spill>0 单次即 race | 消 VGPR spill（SGPR 化地址 / outer 解析 ptr） | abd3833 |

## 死胡同（不能同时拿 0 race + PR HEAD perf，reviewer rejected 列表）
- ❌ 别再试：`__noinline__ compute_tile` → **-25%**（args 通过 scratch 传，反而制造更多 spill）
- ❌ 别再试：全 main loop buffer_load_lds→2-step → **-85%**（破坏 phase_mfma_lds_ldg overlap）
- ❌ 别再试：mid-iter 主循环加 `s_waitcnt vmcnt(0)` → race **不降 0** 且 **-21%**
- ❌ 别再试：FIRST_2STEP（每 phase 第一个 load 2-step）→ race **不降 0** 且 **-30~40%**
- ❌ 别再试：prologue 多个 `vmcnt(0)` drain / mode-switch 藏 spill → 被 reviewer 拒（没修根因）
- ❌ 别再试：`-mllvm -amdgpu-enable-merge-m0=true` → 工程 flag，不修根因
- ❌ 别再试：`-mllvm -sink-insts-to-avoid-spills=true` → 同上，不修根因
- ❌ 别再试：`tile.reserve_pinned_regs()` 多处调用 → 占用 pinned 区的 hack，不解决"为什么编译器要 spill"
- ❌ 别再试：Prologue 2-step（即使只在 prologue）→ 同上根因，counter 通道分工被破坏

---
来源: gpt_oss_myskill/gfx950-vmcnt-race-debug/SKILL.md（早期版，类 1 的 A/B 修法已被下方 gpt_oss2 版否决）；gpt_oss2/myskill/gfx950-vmcnt-race-debug/SKILL.md（现行版，类 1 定案修法一/二/三 + 类 2/类 3，commit abd3833/bb48d3f/d7b149a）
