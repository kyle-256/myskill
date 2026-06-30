# Autotune 设计

## 核心原则

1. **balanced 计时**：autotune 在 `_balanced_targs`（M_total 均分到 G 组）上计时，不被第一次 call 的真实（可能倾斜）分布带偏
2. **M-branch**：不同 M size 用完全不同的 candidate set（per-group M_g 决定）
3. **hysteresis 1.5%**：只在候选比当前快 ≥1.5% 时才切换，防噪声 mis-pick
4. **cache key 是纯静态维度**：`(op, N, K, G, M_total, out_fp16, cbsz, blgp)`，永远不含 tensor id/version
5. **warmup 要长**：短-K shape 冷测 mis-pick 严重（短-K 冷/热差异 >20%）；`warmup=250 reps/50 iters` 是合适值

## fwd/dgrad autotune（_autotune_np_dispatch）

```python
# NN small-M: gate 触发时 bm128 永远赢，直接 return 单 config（不 autotune）
bm128_tiles = G * ceil(pm/128) * ceil(N/256)
if not trans_b and bm128_tiles <= _num_cus():
    return mk(128, 1, 0, 0)

# 大-M / NT: 3 bm256 swizzle 候选
cands = [(256,8,4,0), (256,1,0,0), (256,8,8,0)]
# cands[0] = correctness reference；后面只有 ≥1.5% 更快才切换
```

## wgrad autotune（_autotune_wgrad_dispatch）

```python
if m_total // G <= 1536:
    # 小-M：2 persistent 候选
    cands = [
        _wgrad_compile_cfg(OUT_M, OUT_N, G, ..., num_xcd=8, group_m=4, group_n=0, unroll_n=4),
        _wgrad_compile_cfg(OUT_M, OUT_N, G, ..., num_xcd=8, group_m=4, group_n=8, unroll_n=4),
    ]
else:
    # 大-M：3 masked 候选
    cands = [
        _wgrad_masked_cfg(OUT_M, OUT_N, G, ..., chunk=8, group_m=4, num_xcd=8),
        _wgrad_masked_cfg(OUT_M, OUT_N, G, ..., chunk=8, group_m=0, num_xcd=8),
        _wgrad_masked_cfg(OUT_M, OUT_N, G, ..., chunk=4, group_m=4, num_xcd=8),
    ]
```

## 计时实现（_robust_time）

```python
def _robust_time(launch, targs, warmup=250, reps=5, iters=50):
    for _ in range(warmup): launch(*targs)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(iters): launch(*targs)
        e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) / iters)
    ts.sort()
    return ts[len(ts)//2]   # median
```

**不能用 `timeit.Timer`**：`Timer.timeit(50)` 会被 CPU 抖动污染；`CUDA Event` 更干净。

## cache 设计

```python
_GROUPED_AT_CACHE: dict = {}   # key=(op,N,K,G,M_total,cbsz,blgp,...), value=[launch, compiled]
_GROUPED_WGRAD_AT_CACHE: dict = {}  # key=(OUT_M,OUT_N,G,cbsz,blgp,M_total), value=launch
```

**禁止**：result cache（quantize/transpose/group_offs 结果），任何 `id(tensor)`-keyed cache。

compiled object（`flyc.compile`）和 launch closure 可以 cache（不含 data）。

## mode-split launch（eager vs graph capture）

```python
entry = [raw_launch, compiled_or_None]
if torch.cuda.is_current_stream_capturing():
    raw_launch(*args)   # graph capture 用 raw @flyc.jit（graph-friendly）
else:
    if compiled is None:
        compiled = flyc.compile(raw_launch, *args)  # 一次性编译
        entry[1] = compiled
    compiled(*args)     # eager 用 compiled（跳过 per-call drift-check overhead）
```

小 shape eager per-call：18.4→17.2us（~7% 提升）。

## balanced group_offs 构造

```python
def _balanced_group_offs(m_total, G, device):
    base = m_total // G
    sizes = torch.full((G,), base, dtype=torch.int64, device=device)
    rem = m_total - base * G
    if rem: sizes[:rem] += 1
    offs = torch.zeros(G+1, dtype=torch.int64, device=device)
    offs[1:] = sizes.cumsum(0)
    return offs.view(torch.int32)   # kernel 读 int64 low-words via int32-view
```

args 里只替换 group_offs slot（index 5）：

```python
targs = args[:5] + (balanced_offs,) + args[6:]
```
