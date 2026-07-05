# gpt_oss2 快速开发 checklist + bench/工具脚本

> 类别: 连接 (gpt_oss2) · 主题标签: devloop, bench, mxfp8, rocm-smi

## 快速开发 checklist（改一次 kernel 走一遍）
1. 改 kernel `.py`（**本地**改，别直接在 docker 里改）
2. `rsync -av --checksum` 同步到远端 —— 用 `--checksum` 因为改 py 内容变但 mtime 可能骗过默认比对
3. `docker exec mlperf_gptoss2 rm -rf /root/.flydsl/cache` —— **必清 FlyDSL 缓存**，否则 loader 跑旧编译版
4. `docker exec -e HIP_VISIBLE_DEVICES=4 <测试>` —— 先跑数值
5. 数值通过后 `gms()` 测速，对比 C++ 基线
6. `rocm-smi --showuse` 确认 GPU 干净后再 bench（脏 GPU 测速无意义）
7. `bench` 正式测
8. `git commit` —— **pre-commit 自动格式化可能需两次 `add`+`commit`**（第一次被格式化改文件、第二次才过）

## 干净 GPU 判定（bench / race 前必查）
- `rocm-smi --showuse` → **GPU% = 0**
- `rocm-smi --showmemuse` → **VRAM% < 50**
- 理由：noisy GPU 上 race rate 会被 OOM 干扰，测速也失真

## 并行测试
- `PT_GPUS=4,5,6,7 pytest -n 4` —— conftest 已加 `PT_GPUS` 支持，指定 GPU 集并行

## bench / 工具脚本
| 脚本 | 位置 | 用途 |
|---|---|---|
| `bench_mxfp8_pk.py` | `sync/` | per-tensor vs pk1/2/4 **kernel-only** bench；已重写为生产 API：quant layout 2/4 + scale_pack |
| `correctness_mxfp8_pk.py` | `sync/` | mxfp8 pk **SNR 验证** |
| `probe_gn.py` | `sync/` | gn/gm/xcd 探针 |
| `_wl_perf.py` | `sync/` | grouped wgrad **WL bench** |
| `bench_mxfp8_ratio.py` | `sync/`（chi2774 GPU4） | dense mxfp8 vs 非-scaled per-tensor 的 kernel-only bench |
| `spot_bench_kernel.py` | `scripts/` | kernel-only **TFLOPS**（不含 quant pre-process）|

---
来源: mxfp8-8wave-devloop/SKILL.md, project_mxfp8_wholeloop_port.md, project_mxfp8_grouped_wgrad_wl.md, gfx950-vmcnt-race-debug/SKILL.md
