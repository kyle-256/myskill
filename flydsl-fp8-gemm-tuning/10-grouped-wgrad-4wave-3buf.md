# 4-wave whole-loop grouped TN wgrad：3buf / racing-gap / 2-pool 突破

> 这是**另一条** wgrad kernel（区别于 04 的 masked/persistent）：`_compile_grouped_tn_wgrad_4wave`，occ=1 / 256×256 / 2×2-wave / 两操作数都 transpose-read / whole-loop 裸 asm / AGPR 累加 / CShuffle store。文件 `primus_turbo/flydsl/grouped_gemm/gemm_fp8_grouped_kernel.py`。完整推演在 memory `project_wgrad_occ_feed_bound`（続1-66）。代码 canonical 在 `sync/tensorwise/Primus-Turbo`（分支 feat/kyle/grouped-wgrad-4wave），远端 chi2811 编译测（见 remote-sync skill）。

## 结构与瓶颈

- occ=1（512 VGPR = 256 操作数 + 256 AGPR 累加器，满配；`accum_offset=256`）。LDS 上限 **163840B=160KB/CU**（gfx950 实测 group_segment）。
- 主循环 = 两个 ping-pong 相位，相位间一道 `s_waitcnt ... ; s_barrier`。跨 wave 共享：一个 wave 的 transpose-read gather 所有 wave 写的 chunk（`S2RLoaderTr._ptr_off` 的 `W*chunk_stride`）→ barrier + drain 是跨 wave LDS 一致性的**硬需求**，per-wave 局部 drain 不安全。
- **根因瓶颈 = LDS 转置读 feed 带宽**（不是占用率/延迟/bank）。即使 racing 也只有 fp8 峰值 ~44%。

## racing gap 的精确来源（asm diff 铁证）

`PT_RACE_VM=0`(safe) vs `=1`(racing) 的 `21_final_isa.s` **只差 2 行**（两个相位 barrier 的前置 drain）：
```
safe:   s_waitcnt vmcnt(0)  lgkmcnt(0)   # 全 drain
racing: s_waitcnt vmcnt(16) lgkmcnt(10)  # 让整整一相位的 G2S 写飞过 barrier
```
一相位正好 16 条 `buffer_load_dwordx4`（4 pool × 4 step）→ `vmcnt(16)` = **放松全部 4 个 pool 的写跨 barrier**（不安全，~0.17% bit-flip race）。racing 的全部优势 = 这一件事。（注：`PT_RACE_VM` 只改 2buf 版 `_wgrad_wholeloop_asm`，对 3buf 路径是 no-op。）

## 安全回收：3buf → 1 池 → 2 池

写读距离决定安全 defer：2buf=距离1（写完下相位就读，必须 drain）；3buf=距离2（有整整一相位余量，可安全 partial-drain）。

- **1 池 3buf**（生产 `PT_WL_3BUF=1 PT_WL_3BUF_FUSED=1 PT_WL_VMCNT_MODE=partial`）：只给 pool3(B1) 第 3 缓冲，安全延迟 **1/4** 的写 → 回收 racing 86TF 优势的 ~51%（safe-2buf 2170 → safe-3buf 2214 @m2048 deepseek-up）。
- **2 池 3buf ★突破★**（`PT_WL_2BPOOL=1 PT_CS=1024`，commit c5710aa）：B0+B1 都上第 3 缓冲，安全延迟 **1/2** 的写。塞进 160KB 的关键：`_CS=1024`（省 5120B bank-pad）**+** scalar store（省 8704B C_lds，PT_WL_2BPOOL 自动触发）→ 10 buffer × 16384 = 163840 恰好。
  - 实测（同 GPU，SNR56，30000×7shape race-free）：**m2048 打平/超过 racing**（deepseek-up 2273 vs racing 2268；7 shape 中 6 个超），**全线比生产 1 池 +2~4%**；m4096 距 racing 0.6%（且 racing 在 m4096 SNR 掉到 53-54=正在腐蚀输出）。

### 多池 partial-drain 的 vmcnt 修复（关键 bug）

`_wgrad_wholeloop_asm_3buf`：初版 `_3buf_pool = _3buf_pools[0]`（只取第一个 3buf 池）、`_n_outstanding = nsb`（只算 1 池的写）。2 池都 3buf 时延迟了 2×nsb=8 条写但 drain 只按 4 算 → 留下未 accounted 的 in-flight 写 → **1/30000 race**。修复：`_n_outstanding = sum(nsa|nsb for p in _3buf_pools)`。emit 顺序(0,1,2,3)把 3buf 的 B 池排在最后，`vmcnt(sum)` 恰好留下这些（都 distance-2 安全）的写在飞、把 A 池（distance-1 不可延迟）全 drain。修复后 0/30000。

## 剩余头寸 + 已锁死项

- **m4096 剩 0.6% = _CS=1024 的 14% LDS bank 冲突**（rocprofv3 `LDSBankConflict`：1 池@1056 = 0%，2 池@1024 = 14%，但 MfmaUtil 54.3 > 50 净胜）。`_CS` **锁死 =1024**：buffer 需放 16384B 数据→_CS≥1024；10 buffer ≤163840→_CS≤1024。1024=bank 周期整数倍→transpose read 必冲突，**无 padding 空间**。
- 消 14% 冲突的**唯一**路 = 操作数 **a_plain**（预转置走 `ds_read_b128`，无 transpose 固定域冲突）——但需 upstream 量化时顺带产转置副本（MX_BLOCKWISE 路径有先例：`grouped_quantize_fp8_with_trans`），跨文件大改。a_plain 本身**在 kernel 内零吞吐收益**（gfx950 transpose-read 零 per-op 惩罚，実測），它唯一价值是 enabler（免 pad + 免冲突）。
- **cheap polish**：C_lds aliasing（把 epilogue-dead 的 B buffer 区借给 CShuffle）可恢复 scalar-store 的 ~0.3-1%，纯 kernel、不占额外 LDS，但对 m4096（epilogue 占比小）帮助有限。

## 方法论教训（血泪）

1. **byte-exact 排除法必须组合所有省空间手段**：続57/60 曾判"双池 3buf 放不下"而否决——错在没把 scalar store 省 C_lds(8704) + `_CS=1024` 省 pad(5120) 组合起来算（合计恰好腾出第 2 池）。**单独算每个都不够、组合才够**；单独评估会假阴性关掉真路。
2. **det/race 验证**：SNR 掩盖低概率 race，唯一可靠是 bit-exact 多次跑（`_race_wg.py N`，需 **30000+**）。改缓冲布局/vmcnt 必重跑。
3. **PT_TR_HALF 类"跳过整条指令测天花板"探针不可信**：跳过读 ≠ 换成更少的等效读；真实替换后（ds_read_b128 换 2×tr-b8）因**带宽受限**收益归零。测"去掉 X 的天花板"必须用真实替代指令。
4. occ=2 实测否决（8-wave 2148 < safe-3buf 2214，feed 恶化盖过 latency-hiding）；缩 tile 净负（続57）；纯 swizzle 去冲突被 transpose 固定域挡住。

## 复现命令（远端 chi2811，见 remote-sync）

```bash
# TF+SNR 全 sweep（_ow.py harness，untracked）
docker exec -e HIP_VISIBLE_DEVICES=5 -e PT_CS=1024 -e PT_WL_2BPOOL=1 mlperf_gptoss bash -lc \
 'cd .../Primus-Turbo-tensorwise/primus_turbo && rm -rf /root/.flydsl/cache && \
  PT_WL_3BUF=1 PT_WL_3BUF_FUSED=1 PT_WL_VMCNT_MODE=partial PT_ONLY=deepseek-up PT_MS=2048,4096 python3 _ow.py'
# 30K race（_race_wg.py，同 env）：... python3 _race_wg.py 30000
# rocprofv3（新版必须 --output-format csv，否则出 .db）：
#   rocprofv3 --pmc LDSBankConflict MfmaUtil --output-format csv -d /tmp/rp -- python3 _op3.py
```
**计时坑**：A/B 对照分开 GPU 或同 GPU 串行，**绝不同 GPU 并行两个计时任务**（"m4096 低于 racing"曾是同 GPU 并行假象）；跨 GPU ~1% 方差 → 关键结论同 GPU 交替多 trial 取中位数。`docker exec "... > /tmp/x"` 的重定向落在 **host** 不在容器。
