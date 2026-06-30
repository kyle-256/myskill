# Benchmark 方法和工具

## 可信测量方法排序
1. **rocprofv3 kernel-trace** — 最可靠,kernel-only 时间(排除 host overhead)
2. **Event 500-sample** — 实用,min/p5/p10/med 分布完整
3. **Event 100-sample (test_mxfp4_4w)** — 日常快速对比
4. **do_bench** — 偏差大,不同场景数字不可比

## 工具

### test_mxfp4_4w.py (日常使用)
```bash
GPU=7 DETRUNS=20 ./rr.sh run ../FlyDSL/turbo/test_mxfp4_4w.py 8192 8192 28672
# 输出: SNR dB, det, min/med TF
# DETRUNS=20: det 测 20 次重跑
```

### _bench500.py (精确分布)
```bash
GPU=7 ./rr.sh run ../FlyDSL/turbo/_bench500.py
# 500-sample: min / p5(最好25/500) / p10 / med TF
# 支持 BM= BN= BK= env override
```

### _bench2000.py (捕捉极值)
```bash
GPU=7 ./rr.sh run ../FlyDSL/turbo/_bench2000.py
# 2000-sample: 同上 + 打印 n>=5500 的次数
```

### rocprofv3 (kernel-only)
```bash
rm -rf /tmp/rpf
CUDA_VISIBLE_DEVICES=7 HIP_VISIBLE_DEVICES=7 \
  rocprofv3 --kernel-trace --output-format csv -d /tmp/rpf \
  -- python test_mxfp4_4w.py 8192 8192 28672

# 分析
python3 -c "
import csv; rows=list(csv.DictReader(open('/tmp/rpf/chi2811/*_kernel_trace.csv')))
durs=[int(r['End_Timestamp'])-int(r['Start_Timestamp']) for r in rows if 'gemm' in r.get('Kernel_Name','').lower()]
durs.sort(); M=N=8192; K=28672
print(f'n={len(durs)} best={durs[0]}ns → best {2*M*N*K/durs[0]/1e3:.1f} / med {2*M*N*K/durs[len(durs)//2]/1e3:.1f} TF')
"
```

## GPU 状态检查(测试前必做)

```bash
# 查看 GPU 占用率
GPU=7 ./rr.sh sh 'rocm-smi --showuse 2>/dev/null | grep -E "GPU\[[0-7]\]" | grep "use"'

# 杀僵尸进程
GPU=7 ./rr.sh sh 'ps -eo pid,etime,pcpu,cmd | grep -E "python|torch" | grep -v grep | head'
```

**GPU 必须 0% 才能信任 perf 数字!** 被抢占时 perf 会波动 20-30%。

## 多 GPU 对比注意
**不同 GPU 的 clock boost 状态不同** → 同一 kernel 可能差 10-30%。
- GPU0 (22GB): 测到 5536 med (GPU boost 高)
- GPU7 (25GB): 5401 med (正常工作温度)

**必须在同一 GPU 对比**,否则数字无意义。

## 测量最佳实践
1. 跑 test 前检查 GPU 占用 = 0%
2. 用 DETRUNS=20 确认 det=0
3. 用 500-sample 做决策级比较(20-sample 噪声±20T)
4. 只信 med,不信 min(min 是瞬时最优 dispatch)
5. rocprof best = Event min × 0.98(Event 含 launch overhead)

## 典型噪声水平
- 20-sample Event: ±25T med (±0.5%)
- 500-sample: ±10T med (±0.2%)
- rocprof: ±3T (最稳定)

## aiter 对标
```bash
# aiter no-preshuffle 对标(同 Event harness)
GPU=7 ./rr.sh run ../FlyDSL/turbo/bench_aiter.py 8192 8192 28672
# → aiter_a4w4(pre=0) min 5495/med 5440 TF (GPU7, idle)
```
fly prod / aiter = 5401/5440 = **98%**(Event 同方法)
