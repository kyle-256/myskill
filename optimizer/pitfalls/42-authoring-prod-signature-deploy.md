# 生产化部署坑：backend 签名一致/setdefault 遮蔽/out_fp16 cache key/比较基线错

> 类别: 踩过的坑 · 主题标签: authoring-pattern, autotune-dispatch, cache-key, vendor-reuse

- **execute/can_handle 签名必须与 sibling backend 完全一致（含 preshuffled 形参）**：dispatcher 走 `execute(**kwargs)`，结构性要求所有 backend（HipBLASLt/AITER/FlyDSL）都声明同一套轴。FlyDSL 的 execute/can_handle 若少了 `preshuffled` 形参会崩；正确写法是声明该形参 + `del preshuffled`（收下但不用）。WHY：kwargs 广播给每个 backend，谁少一个轴谁就 TypeError。

- **宽 store 是 bf16-only，fp16 out 必须走窄路径**：`store_tacc_wide` 用 `cvt_pk_bf16_f32`，只产 bf16。fp16 输出要在 wrapper 里强制 `taccw=False` 走窄 store 路径。

- **out_fp16 必须进三处 cache key（launch/AT/split）**：否则 bf16/fp16 两种编译产物串号（同一 key 命中错的二进制），fp16 请求拿到 bf16 kernel。三处都要带 `out_fp16` 才不串。

- **mxfp4 生产默认在 4wave.py PROD setdefault，遮蔽 8wave.py**（部署死坑）：`4wave.py` L287-303 `PROD setdefault(ASMMFMA=6 INPLACE=1 DIAG=1 SINNER=1 MMORD=5 ALT=0 GAVOID=1 SC_VGPR=1 PIN...)`，setdefault 在 `8wave.py` 读 environ **之前**生效。所以单改 `8wave.py` 的 environ 默认无效。
  - ❌ 别再试：把 `8wave.py` 默认 `0→9` 无效——已被 4wave PROD setdefault 遮蔽。要改就改 `4wave.py` L290 的 `MMORD '5'→'9'`。

- **比较基线错会造出假收益**（陷阱）：把 base 设成 `FP4_MMORD=0` 去比，得出 mm5/mm9 "+3~5%" 是假象——mm0 从来不是默认，真实默认早就是 mm5。真实收益 = mm9(4×8) 在 mm5 之上仅 **+~2%**（7b-qkv M8192：4117→4210），整体 <1%。
  - ❌ 别再试：任何"跟 mm0 比"得出的提升都要打折，别当真收益上报。

- **移植 4-wave kernel 要单独 vendor helper，不复用 gemm_helper.py**：turbo 产品化的 8-wave 把原语分叉了——`Mfma16x16x128` 去掉 `call_one`、`G2S/S2RLoader` 去掉 `load_one`、`StoreC→StoreCPerTensor`（per-tensor 标量 scale，非 row/col-wise）。4-wave 直接 import 会崩。vendor 一份保证 bench-identical 且不碰 8-wave。

---
来源: 13-primus-turbo-prod.md, 12-llama-aiter-baseline.md, remote-sync/SKILL.md
