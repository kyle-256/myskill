---
name: mxfp8-8wave-devloop
description: 迭代 PT turbo MXFP8 8-wave grouped GEMM kernel（dev/kyle_mxfp8_gg_pr）时的一条龙 sync+build+ISA+correctness+perf 基建。当在 /wekafs/kyle 下 build/profile/调试 8-wave (turbo_grouped_gemm_mxfp8_*8wave*) 或问"怎么重编/怎么看 ISA/怎么测 MfmaUtil"时用。封装了 chi2811 远程 build 的所有坑（前台 build、ISA 从 gfx950 code object dd 切、KSYM 必须排除 wgrad）。配合 remote-sync / remote-mlperf-gptoss / claim-mi355x-node。
---

# mxfp8-8wave-devloop

迭代 8-wave MXFP8 grouped GEMM 的固定工作流。**别再手搓 sync/touch/.o/pip/sleep 轮询/dd ISA**——用 `scripts/dev.sh`。

工作树：本地 `/wekafs/kyle/remote_sync/mxfp8/Primus-Turbo`（= 容器 `/workspace/code/mxfp8/Primus-Turbo`，分支 `dev/kyle_mxfp8_gg_pr`）。节点见 [claim-mi355x-node]（当前 chi2811）。

## 一条命令搞定

```bash
cd /wekafs/kyle/remote_sync/mxfp8/Primus-Turbo
GPU=2 scripts/dev.sh build    # sync(csrc+scripts) + 前台重编(~5min) + 报 OK/错误
GPU=2 scripts/dev.sh isa      # 切 gfx950 ISA，报 fwd kernel 的 mfma/setprio/accvgpr/scratch/acc寄存器/vgpr/agpr（可靠）
GPU=2 scripts/dev.sh corr     # valid_corr SNR（8wave，要 ~25-28dB）
GPU=2 scripts/dev.sh det      # det60 bit-exact（atol=0）
GPU=2 CFG=K7168 scripts/dev.sh perf   # kernel-trace 时长 + MfmaUtil + L2（CFG=K7168 大K / Down 小K）
GPU=2 scripts/dev.sh all      # build + isa + corr + perf(K7168) + perf(Down)
```

- 脚本：`scripts/dev.sh`（本地驱动，sync 后 ssh 调远程）+ `scripts/_dev_remote.sh`（容器内执行）。helper `_prof_min.py`/`parse_kt.py` 也在 `scripts/`（随 sync 进容器，持久）。
- **build 是前台跑（~5min）**——调用时给 Bash 工具**长 timeout（≥360000ms）**。**别再 `nohup ... & sleep 轮询**`（之前一直 exit 143 超时）。
- 改动只 push `csrc` + `scripts`（`sync.sh push mxfp8/Primus-Turbo csrc --force`），不碰 build/。

## ☠️ 头号大坑：ISA 一直读成 wgrad（烧过很久）

fwd kernel 符号 `..._mxfp8_256x256x128_16x16x128_8wave_persistent_kernelINS_13float8_e4m3_tES2_12hip_bfloat16`，wgrad 是 `..._mxfp8_wgrad_256x256x128_..._8wave_persistent_kernelINS_13float8_e4m3_tES2...`。**松的 KSYM（不含 `mxfp8_256`）会同时匹配两者，且 wgrad 在反汇编排前 → awk 抓到 wgrad，你看到的全是 wgrad 的寄存器/scratch，完全误导。**
- KSYM 必须含 `mxfp8_256`（fwd）以排除 `mxfp8_wgrad_256`。`_dev_remote.sh` 里已固定。
- span 取到**下一个函数标签**（`/<_Z.*>:$/`），**不是首个 `s_endpgm`**（早退 `return` 会 emit 中段 s_endpgm 截断）。
- `verify_8wave_wf.js`（多级验证 workflow）的 **ISA agent 同坑 + dd 提取偶发解析错 → 不可信**（它报的 VGPR256/setprio0 其实是 wgrad）。**ISA 只信 `dev.sh isa`**；correctness/perf agent 可信。

## ISA 切法（dev.sh isa 内部，手动 debug 时备用）

```bash
SO=build/lib/libprimus_turbo_kernels.so
line=$(roc-obj-ls $SO | awk '/gfx950/{print $NF}')   # 取 gfx950 code object 的 URI
off=$(sed -E 's/.*offset=([0-9]+).*/\1/' <<<"$line"); sz=$(sed -E 's/.*size=([0-9]+).*/\1/' <<<"$line")
dd if=$SO of=/tmp/g.co bs=1 skip=$off count=$sz status=none
/opt/rocm/llvm/bin/llvm-objdump -d /tmp/g.co | awk -v k="$KSYM" 'index($0,k){p=1;print;next} p&&/<_Z.*>:$/{exit} p{print}' > /tmp/k.s
/opt/rocm/llvm/bin/llvm-readelf --notes /tmp/g.co | grep -A60 "$KSYM" | grep -E '\.(vgpr|agpr)_count|private_segment_fixed_size'
```
- acc 在 AGPR 看 mfma 操作数：`v_mfma_scale ... a[0:3], v[..], v[..], a[0:3]`（`a[]`=AGPR 好；`v[]`=acc 在 VGPR）。
- `private_segment_fixed_size` = scratch（要 0）。`vgpr_count`>128 → 1 wave/SIMD（8-wave 2-wave 要 ≤128）。`accvgpr` 多 = AGPR↔VGPR 洗牌（run_agpr SSA acc 的病）。

## 测试方法论铁律

- 8-wave 走 env gate `TURBO_MX_8WAVE=1`（C++ `static` 进程内只读一次）→ **必须 shell env + 新进程**，绝不进程内 `os.environ` 切。
- 参考真值 = 同进程纯 torch GEMM。`scripts/`：valid_corr.py(SNR)、det60.py(bit-exact)、_prof_min.py(CFG=K7168/Down,ITERS,纯 fwd kernel)、bench8w.py。
- 重编内部（_dev_remote.sh do_build）：`printf '\n//rb\n' >> .../turbo_grouped_gemm.cu`（touch）+ `rm build/temp/.../turbo_grouped_gemm.o` + `pip install --no-build-isolation -e . -v`，grep `Successfully installed` 判成功。改 `.h` 必须 touch `.cu` 否则不重编。

## 关键基线/目标（GateUp K7168 纯 fwd kernel）

4-wave(fence) 990us / MfmaUtil 46% ; committed 8-wave 1214us / 35% ; 目标:无 race + scratch=0 + L2>60% + 对齐 no-fence-4wave。小K Down:8-wave 已能赢(868 vs 954)。详见 memory `mxfp8_8wave_design`。
