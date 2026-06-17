# Benchmark 方法论

## 可靠的 benchmark 协议

### rocprofv3 冷测（kernel-level TFLOPS）

最可靠的 kernel TFLOPS，免受 autotune/overhead 污染：

```bash
rocprofv3 --kernel-trace -d /tmp/rp_out -o tr -- python bench_script.py
# 结果在 /tmp/rp_out/tr_results.db (SQLite)

python3 - <<PY
import sqlite3
c = sqlite3.connect("/tmp/rp_out/tr_results.db")
for nm, calls, dur, pct in c.execute(
    "SELECT name,total_calls,total_duration,percentage FROM top_kernels ORDER BY total_duration DESC"
).fetchall()[:10]:
    print(f"{pct:5.1f}%  calls={calls}  {nm[:80]}")
PY
```

min duration = 稳态单次（排除 autotune 冷启动噪声）：

```python
for kn in ["kernel_grouped_nn_persistent", "kernel_grouped_tn_persist"]:
    rows = c.execute(f"SELECT duration FROM kernels WHERE name LIKE '{kn}%'").fetchall()
    ds = sorted(r[0] for r in rows)
    print(f"{kn}: min={ds[0]}ns  {FLOP/(ds[0]*1e-9)/1e12:.0f} TF  median={ds[len(ds)//2]}ns")
```

### graph-replay min-of-8（op-level，含 quant）

```python
results = []
for _ in range(8):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.cuda.graph(g):
        out = op(a, b, ...)
    g.replay()
    torch.cuda.synchronize()
    results.append(time.perf_counter() - t0)
return min(results)   # min-of-8，免 CPU 抖动
```

### 避免 timeit.Timer 均值

```python
# ❌ 不可靠：CPU scheduler jitter + GPU boost clock 未稳定
t = torch.utils.benchmark.Timer("fn()", globals={"fn": fn}).timeit(50).mean

# ✓ 可靠：CUDA Event，5-median，250 warmup
t = _robust_time(launch, args, warmup=250, reps=5, iters=50)
```

## 僵尸进程（必查！）

**测速前必须查杀僵尸 GPU 进程**，否则带宽/时钟被压 ~10%，结果严重偏低（曾把 per-iter SRD 前进的 ~2% 回退掩盖了 3 次）。

```bash
# 查 GPU 上的真实 KFD PID
rocm-smi --showpidgpus | grep "using.*DRM"

# 容器内找对应进程
ps -eo pid,pcpu,etime,comm | grep python | grep -v defunct

# 杀（确认是自己的进程后）
kill -9 <pid>

# 确认
rocm-smi --showuse | grep "GPU use"  # 全 0% 才是干净
```

注意：`<defunct>` 状态的 python 进程已经死了（不占 GPU），真正占 GPU 的是有 KFD entry 的。

## 并行 benchmark 的污染

8 卡并行跑时，每张卡的内存带宽被共享 L3/HBM 和电源分摊 → 所有 kernel 约慢 10%。

**用并行 bench 做相对比较（同跑）可以，不能用来读绝对 TFLOPS，也不能用来做 before/after 对比（两次用的并行度不同）。**

before/after 对比必须用同口径（都单卡，或都 N-way 同时）。

## rocprof 诊断：确认 kernel 归属

B=1 等疑似路由到其他 backend 的情况，直接用 rocprof 看：

```bash
HIP_VISIBLE_DEVICES=6 rocprofv3 --kernel-trace -d /tmp/rpout -o tr -- python my_op.py
# 看 top_kernels 里是 kernel_grouped_* 还是 Cijk_* (hipBLASLt) 还是其他
```

## 标准 MoE bench shapes

文件：`scripts2/_gg_fwd_wgrad.py`（GG_BACKEND=FLYDSL/TRITON/HIPKITTEN 切换）

```python
MODELS = {
    "deepseekv3": (7168, 2048),   # hidden=7168, inter=2048
    "qwen235b":   (4096, 1536),
    "gpt_oss":    (2880, 2880),
}
# up: K=hidden, N=2*inter; down: K=inter, N=hidden; M∈{2048,4096}; B=8 balanced
```

unbalanced 测试（GG_UNBAL=1）：

```bash
GG_UNBAL=1 GG_BACKEND=FLYDSL python scripts2/_gg_fwd_wgrad.py
# 30:1 skew：sizes=[4918,3932,2949,1966,1310,819,327,163] sum=B*M
```

## GB200 全量对比

```bash
# 8 片并行（每片 ~36 case，避免单进程跑 289 case 的累积 lld hang）
for s in 0 1 2 3 4 5 6 7; do
    OUT=/tmp/fly_sh$s.csv NSH=8 SHARD=$s HIP_VISIBLE_DEVICES=$s \
    PYTHONUNBUFFERED=1 nohup python scripts2/_gb200_fight.py >/tmp/gb200_sh$s.log 2>&1 &
done
wait
# 合并
head -1 /tmp/fly_sh0.csv > /tmp/fly_all.csv
for s in 0 1 2 3 4 5 6 7; do tail -n +2 /tmp/fly_sh$s.csv >> /tmp/fly_all.csv; done
# 分析
python3 _cmp_gb200.py /tmp/fly_all.csv grouped_gemm_te_fp8_tensorwise_*.csv
```

注意：8 片并行数值会比单卡低 ~10%，只用来看相对比值（fly/GB200），不看绝对值。
