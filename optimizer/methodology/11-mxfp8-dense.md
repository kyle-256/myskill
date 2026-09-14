# MXFP8 dense:dual-cast quant kernel / scale preshuffle / LDS转置写 / scale_pack预取 / whole-loop移植 / FlyDSL全覆盖

> 类别: 方法论 · 主题标签: mxfp8, dual-cast, quant-kernel, e8m0, SK-timing, 集成路径, preshuffle, layout-1, b-comb, opsel, 隔离调试, 带宽roofline, LDS-coalesced, 转置写, grid_swz, tile选配, scale_pack, op_sel, 大K预取, per-K-loop, pack歧义, whole-loop, MFMA-format, fp8-operand, swizzle, store统一, asm_mma, WL-swizzle, no-fallback, MX全覆盖, shape-gate, dispatcher, K-tail, general-N

## MXFP8 dual-cast quant kernel:架构 / 分段计时 / 集成路径 / 数值验证

### dual-cast quant kernel 架构(mxfp8_quant_flydsl.py,唯一生产入口 compile_qdual,即原 v5)
- 输入 `X[M,K] bf16`,一个 kernel 同时吐 4 路输出:
  - `Qr[M,K] fp8`(row cast,fwd A 操作数)
  - `ASp int32`(layout-1 preshuffle A-scale,见下方「MXFP8 scale preshuffle」节)
  - `AtQd[K,M] fp8`(col cast **已转置**,bwd `at` 的 B 操作数)
  - `AtSp int32`(b-comb col scale)
- Tile `bm×bk` 约束 `bm*bk==8192`,Block **512 线程**,默认 `bm=64, bk=128`。
- 分工:前 256 线程做 **ROW cast**、后 256 线程做 **COL cast**(两半并发共享 LDS);`barrier` 后**全 512 线程重读 ldsc 合并写转置**(制胜招见下方「LDS-合并转置写」节)。
- scale 元素数:`SP_A_ELEMS=(M//64)*(K//128)*64*4`;`SP_AT_ELEMS=(K//256)*4*(M//128)*64*4`。
- 输出缓冲 `torch.empty` 不需 zero-init(每个 in-range byte 恰写一次)。partial-tile 用 SRD/num_records OOB→0 + 输出 mask;**只需 M/N 是 64 的倍数**即可安全(256 倍数约束已解除)。

### e8m0 exponent 与 fp8 打包(round-even 匹配 C++ compute_tile_scale)
- exp:`ai=I32(amax.bitcast(Int32))+I32(1<<19); ep=((ai>>23)&0x1FF)-135; clamp 到 [-127,128]`。
- fp8 pack:`word=I32(rocdl.cvt_pk_fp8_f32(IRI,a,b,I32(0),0))`(bytes 0,1),再 `cvt_pk_fp8_f32(IRI,c,d,word,1)`(bytes 2,3)。
- `scale=fm.exp2(ep.to(F32))`。⚠️ Int32 无 bitcast,**不能 int→float bitcast**,必须走 `.to(F32)`。

### 分段计时用 SK env(归因用,别扫常数)
- `_SK=int(os.environ.get('SK','0'))`,各阶段 `if _SK==N: return`:SK=0 完整 / SK=1 只 row / SK=2 row+col 计算无写 / SK=3 跳 LDS 写。
- 实测:移除 barrier / 占用 / amax+exp2 **全无效** → 瓶颈是 dual-cast **DRAM 散写的固有代价**,不是计算(带宽定位见下方「带宽 / 瓶颈定位」节)。

### 集成路径(算子不直接调 kernel)
- `FP8GemmMXFunction.forward/backward → quantize_fp8_with_trans → quantize_mxfp8_impl → quant_mxfp8_raw(mxfp8_quant_flydsl.py) → compile_qdual(..., **_qdual_tile_cfg(M,K,Mp,Kp))`。
- quant 只发 **raw 行主 E8M0**(`[free, contract//32]`,1 byte/block);preshuffle(layout-1 / b-comb)由各 GEMM backend 在自己 launch 里融合。public 路径若显式要 preshuffled,由 `quantize_mxfp8_impl` 末尾 post-apply 独立 `shuffle_scale` 算子。
- 缓存:`_RAW_QDUAL_CACHE[(M,K,Mp,Kp,in_dtype,out_dtype)]` 存已编译 callable。

### 数值验证 SOP(三级)
1. **self-consistent**:`Qr.view(uint8)==(x/exp2(Sr-127).repeat_interleave(32)).to(e4m3).view(uint8)` 期望 1.0(0.31% 偏差是 cvt round tie-break 正常)。
2. **vs C++**:`quantize_mxfp8_impl(...,ScalingRecipe(preshuffle_layout=1,preshuffle_n_tiles=4),ScalingRecipe(preshuffle_layout=3))`,`cq_ref.shape=[K,M]`,比 AtQd/AtSp。
3. **e2e SNR 验收**:fwd/gradA/gradB SNR > 25 dB(E4M3 阈值)。

## MXFP8 scale preshuffle:动机 / layout-1(A broadcast) / b-comb / 写侧 element-offset / 逐层隔离调试

### 为什么必须 preshuffle scale
- MXFP8 GEMM 的 `mfma_scale_f32_16x16x128_f8f6f4` 通过 **opsel 字节选择器**读 E8M0 scale,每 wave 64 lane 各需对应 scale 字节。
- 原始布局 `[DIM, K//32]` 是**散点 gather**,实测比 coalesced 慢 ~2.6×。preshuffle 把 scale 重排成 **wave/lane 连续布局**。

### Layout-1(A-scale broadcast)
- 输入 e8m0 `[DIM, K//32]` → 输出 i32 `[DIM//16, K//128, 64, n_tiles]`,`SP[grp,k,lane,s]=broadcast_u8_to_u32(scale[grp*16*n_tiles+s*16+lane%16, 4k+lane//16])`,`n_tiles=BLOCK_M//64`(BLOCK_M=256 时 =4)。
- 索引公式(Python 枚举验证 100%):
  ```
  g=kcol&3; kg=kcol>>2; grp=row//64; sub=(row%64)//16; r=row%16
  lane=g*16+r; dword=((grp*K128+kg)*64+lane)*4+sub
  ```

### B-comb(combined B-scale)
- 用于 fwd B 操作数(dim=N,contract=K)和 bwd `at` 操作数(dim=K,contract=M):
  ```
  block_n=col//256; wn=(col%256)//32
  s_bc=((col>>7)<<1)|(((col&127)&31)>>4); r_bc=col%16
  lane_bc=g_bc*16+r_bc (g_bc=row_block&3); grp_bc=block_n*4+wn
  dword=((grp_bc*K128p+kg_bc)*64+lane_bc)*4+s_bc
  ```

### 写侧用 element-offset(与读侧一致)
- `bo.create_buffer_resource(ASp, max_size=False, num_records_bytes=I32(SP_A_BYTES))` + `bo.buffer_store(bcast, rasp, dword, cache_modifier=1)`,**element offset 默认非字节**,与 GEMM 侧 `ScaleS2R` 读侧完全一致(读写口径不一致会静默错位)。

### 逐层隔离调试 SOP(数值错误找最小复现)
- Step1 Python 公式验证(mydword 枚举 match=2048/2048)
- Step2 GPU 地址路径(把 dword index 当 value 写出,address-correct=1.0)
- Step3 scalar 值(e8 byte vs C++ plain scale)
- Step4 bcast 全程 Int32 路径验证
- Step5 行隔离(只跑 col-half=99.69%)
- Step6 两半并存干扰测试
- Step7 完整 kernel 仍错则**必是 flydsl 缓存或 Python 字节码旧版**(`rm -rf cache` + `rsync --checksum`)。

### FlyDSL 关键 API 速查(preshuffle/quant 写码用)
- `buffer_store(word, rsrc, byte_offset, cache_modifier=1, offset_is_bytes=True)`;`create_buffer_resource(tensor, max_size=False, num_records_bytes=I32(nbytes))`(element-offset 默认);`make_row_band_resource(bo.extract_base_index(T), base_row, c_rows, c_cols, elem_bytes)`(2D int64 rebasing);LDS strided read `make_view(base, make_layout(32,PAD)).load()` 正确生成 32 次 `ds_read_b16`;barrier `rocdl.s_barrier()`。

## MXFP8 dual-cast 带宽 roofline + LDS-合并转置写(制胜招)+ per-shape tile 选配

### 带宽 / 瓶颈定位
- dual-cast quant 流量 = `4.0625*M*K` bytes(读 2MK bf16 + 写 2MK fp8 + scale)。
- 分相探针:读+行写 ~7.5 TB/s;转置**列写**散写崩到 ~1.5 TB/s → 唯一瓶颈是 `AtQd[K,M]` 转置写的**合并度**(每 K-行的 M-run 太短)。
- 判断带宽受限的判据:移除 barrier / 占用 / 计算(SK env)均无效 ⇒ 计算非瓶颈。
- 节点/HBM 带宽上限数据：见 methodology/12-mxfp8-grouped.md「e2e 计时/节点占用检查」;dual-cast 达 3.2-4.6 TB/s = 真上限的 55-73%。
- ★★★ **⚠ 上面"唯一瓶颈 = 列写合并度 / 3.2-4.6 TB/s"只对 1d(per-32)半区成立,别套到 `use_2d_block` 半区**(2026-09-11 mxfp8-nt r5 实测,MI355X)。1d 体在 step 内已达 **3.97-5.54 TB/s**、孤立 **5.76**(越过本卡上界);而 **2d-block(权重 operand,`ScalingRecipe(use_2d_block=True)`)那一路在五格里都是最慢的搬字节者,仅 3.04-3.41 TB/s**。同形状同 buffer 的三臂 A/B(nat/惰性 ctl/alt)测 15 条 cast:**2d 比 1d 慢 +9.6~+20.1%,15/15 同号**,ctl −5.1~+0.5% ⇒ **慢在 emission,不在 2d 语义**。根因(PMC 每 WG,l1.w 21504×3072):旧 2d 半区"一个 warp 领一块 32×32"(`riw=lane>>3`/`tir=lane&7`,4 pass)使**数据写退化成单 dword 4 B/lane** ⇒ 64 条 store / 512 请求 / **2.00× 扇区放大**,两张 scale 平面又各 512 请求,再加 `_WR64` 两轮 **12 条 `ds_bpermute`**:`SQ_INSTS_VMEM_WR 208`、`TCP_TCC_WRITE_REQ 1792`、`VGPR 28`。**修法 = 让一个 wave 整整领两块 32×32**(ROW:`rq=lane>>2`/`qc=lane&3`,拿 `rb*32+rq` 与 `+16` 两行 × 16 列,4 lane 拼满 64 B;蝶形掩码 `[1,4,8,16,32]` 保住 lane bit 1;scale 字节 `(qc&1)==0` 发;COL:`c=ckb*32+(lane&31)`、`mblk=mp*2+(lane>>5)`,逐字复用 1d 体 + 掩码 `[1,2,4,8,16]`)⇒ **56 / 1024 / VGPR 24**,`GRBM_GUI_ACTIVE` −10.2% 且与 1d 参照只差 0.4%(缺口整条抹平);四张输出平面在 11 形状 × 3 种 2d 模式上**逐位相同**(max 与顺序无关)。定价:孤立 −8.3~−15.4%(ctl ±1.1%)、同进程 step gm **+0.80%**、`bench.sh` 3 rep 中位 **106.496→107.383 = +0.83%**。⇒ 树上旋钮 `_QD_2D_FLAT`(默认 True)。**残余**:ROW/COL scale 平面各 256 请求,1d 只要 128(一行 4 个 scale 字节被两个 wave 各写一半),要合到 128 得让一个 wave 领满 128 列 = 每 lane 64 个 f32 活寄存器。

### LDS-合并转置写(制胜招)
- 根因:列写 run 长度 = `bm`(每 K-行 tile 内连续 M 字节)。旧 `BM=32` 列写 run 仅 32B。
- 做法:`bm×bk` 大 tile + LDS 暂存 fp8 **数据**(不暂存 scale),`barrier` 后**全 512 线程重读 ldsc**,每线程 1 次 `vec4`(16B)合并写 `AtQd`(合并宽度 = bm 字节、指令比 scalar 砍 4×)。
- `pad_extra=4` 错开 32-way LDS bank 冲突(多 shape +3-7%)。
- `grid_swz` 列-major pid 抗 DRAM channel camping。
- 实测 vs 旧 BM=32:每 shape **1.10-1.48×**,geomean ~1.25×,峰值 **5.68 TB/s**,无退化。

### per-shape tile 选配(`_qdual_tile_cfg(M,K,Mp,Kp)`)
- `Kp>=8192 且 Mp<=4096` → `(bm=128, grid_swz)`;否则默认 `bm=64, bk=128, pad_extra=4`(大 M / K=11008 全胜从不退化)。
- `bm=128, bk=64, pad_extra=4` 用于 12288×4096(5.68);再 `grid_swz=True` 用于 4096×11008(5.36, 1.48×)、4096×4096。
- 列/行写宽度权衡:`bm=128,bk=64` 列写=整 128B cache line、行写=64B;`bm=64,bk=128` 列写=64B、行写=128B(**默认最稳**)。

### partial-tile tail(非 256 倍数支持)
- 数据 G2S 用 buffer resource(SRD)OOB→0;scale 读用 `create_buffer_resource(num_records)` OOB→0;输出写用 `make_row_band_resource` + `col<c_n` mask 钳越界。
- 只需 M/N 是 **64 的倍数**即可安全,**256 倍数约束已解除**。

### 节点 GPU 占用检查
- 见 methodology/12-mxfp8-grouped.md「e2e 计时/节点占用检查」

## MXFP8 scale_pack(MFMA op_sel)+ 大-K scale group 预取:闭合 per-tensor gap

### 制胜双招(缺一不可,per-K occ=2 clean 内核,非 whole-loop)
1. **scale_pack**:`mxfp8_scale_pack(K)=4 if K//128%4==0 else 2 if %2==0 else 1`,与 C++ quant `k_pre_pack` 完全一致。把连续 `pack` 个 K-iter 的 E8M0 byte 打进一个 i32(byte 0..3),每 pack 个 iter 只发 **1 次 scale load**,MFMA 用 `op_sel=k%pack`(编译期立即数)选 byte → 砍 scale VMEM `pack×`,消 pack=1 broadcast 的 V=256 spill;热循环**零额外 ALU/VGPR**。
   - 因 pack 永远整除 K128,**不需 ceil-K128**;`ScaleS2R/ScaleBComb` 维持 `K//(128*pack)`。
2. **大-K scale group 预取**:`_N_SGROUPS=(K_ITERS+pack-1)//pack`,loop 内每 group **首 iter 预取 g+1 的 scale**(sa0n/sa1n/sb_alln 藏进当前 group MFMA shadow)、**末 iter swap**。
- 结果:全 shape ≥0.97× per-tensor,几何均值反超 ~1.01×。

### pack 歧义消除(kernel 分不清 packed / broadcast)
- `execute()` 由 **caller** 按 scale 来源指定 pack:
  - int32 透传(C++ quant 吐 packed)→ `scale_pack=mxfp8_scale_pack(a.shape[1])`;
  - raw 路径(`preshuffle_ab_flydsl` broadcast)→ `scale_pack=1`。
- recipe 开包:`gemm_fp8.py` 把 `_R(1)→_R(2)`、`_R(3)→_R(4)`(A/B broadcast layout 1/3 → packed layout 2/4),C++ quant 据此按 K 自动吐 packed int32,execute 端 `mxfp8_scale_pack(K)` 吻合;non-fast-path(raw E8M0)走 broadcast 不变。

### grouped 版落地(优化7,默认 ON,PACK=4)
- preshuffle 把每 K-iter 一个 broadcast dword 改成 PACK=4 个相邻 K-iter 打进 1 dword → preshuffle 写流量 + scale buffer 占用缩 4×;GEMM 侧 `op_sel=scale_opsel(k)=k%PACK`。
- 落在 **NT(fwd+dgrad)+ dense**(K-iter 编译期已知);**wgrad 保持 pack=1**(变长-K 每组起始 ks0 只 128 对齐非 PACK*128,`k%PACK` 非编译期常量)。
- `_SCALE_PACK=int(os.environ.get('PT_SCALE_PACK',4))`,`PT_SCALE_PACK=1` 逐位复现旧布局。
- 效果:融合 preshuffle kernel **-32%**(14.3→9.7us),GEMM 完全不变,全 shape bit-exact;但 preshuffle 只占 fwd/dgrad ~3.5%,砍 32% ≈ e2e -1%,**是卫生/带宽改进非 e2e 杠杆**。

### dense per-K 达标结果
- pack4+预取生产接通后 kernel-only vs per-tensor 几何均值 ~1.01×(MXFP8 平均反超):7B QKV 1.10 / GateUp 0.97 / Down 1.04-1.06,70B QKV 1.04 / GateUp 0.97 / Down 0.97;FLOP 主导的 GateUp/Down/QKV 全 ≥0.966。

## MXFP8 whole-loop 从 fp4 移植:MFMA 格式切换 / fp8 operand 读 / swizzle hi-base / store 统一 / WL 裁决

### 合理对标(先定标尺)
- mxfp8 scaled-MFMA 对标口径：见 methodology/12-mxfp8-grouped.md「MX vs TW 唯一公平口径」。

### MFMA 格式切换(fp4 → fp8)
- asm 从 fp4 的 `cbsz:4 blgp:4` → **`cbsz:0 blgp:0`**(E4M3)。
- E5M2/HYBRID 时 `cbsz/blgp` 按 operand format(0=E4M3, 1=E5M2),**scale 路径不变**。

### fp8 operand 读格式(NT 布局)
- fp8 **不用 tr_b8**;`S2RLoader`(gemm_helper.py:165)= 2×16B 普通读(swizzle_128, `col=(lane//16)*16+step*64`)→ `pack_i32x4_i32x8` → i32x8;fp4=1×16B+零填充→i32x8。
- 寄存器格式相同,差异 = fp8 每 tile operand 是 **2× `ds_read_b128`**(fp4 1×),**VGPR 不翻倍**。
- swizzle hi-base 技巧:fp8 2×b128 高半区(+64)非线性难点解法 = `hi-base=S2RLoaderFp4.base_addr(s=1)`(s*64 在 s=1 正好给 col `g*16+64` 且已正确 swizzle),两条 b128 都用 `offset=ii*ts`;引擎加 `a/bl/br_base_hi` 输入(仅 `_FP8`)。

### scale 路径 fp4 ↔ fp8 完全复用
- `sc_*` 参数、`ScaleS2RPacked`、`preshuffle_scale_lane_contig`、`grouped_xcd_pid` 原样复用(scale_pack + 大K预取见上方「MXFP8 scale_pack」节)。

### store 统一(557→164 行,commit 34dc6aee,已被后续 commit 取代)
- `StoreCPerTensor` 加 `A/B_scale=None`(不缩放,mxfp8 scale 已折进 MMA)+ copy_atom 模式,让 per-tensor(scale+buffer_store)/ grouped / per-K mxfp8(scale=None+buffer_store)/ WL(scale=None+copy_atom)**全部共用一个类**。
- `buffer_store(mask=)` 内部 = `select(mask, off, 0x7FFFFFFF)`,和 copy-atom OOB-select 同机制,**mask 本身不是问题**;`make_row_band_resource` 补 SGPR-pin + `0x7FFFFFFF` cap 与内联字节级一致。
- **后续(commit ab7ad885 取代 34dc6aee):WL 连同 mx_wholeloop 文件夹被整体移出生产,copy_atom 分支随之删除**——kernel 恢复成干净 per-K + 共用 `StoreCPerTensor`(scale-optional,去掉 WL 专用的 copy_atom)。当前生产 store 无 WL/copy_atom 模式,只服务 per-tensor/grouped/per-K 三路(均 scale+buffer_store 或 scale=None+buffer_store)。

### WL 裁决(per-shape win 非全面 win,该结论已被后续废弃 WL 取代)
- `asm_mma`(bare-asm scaled MFMA)= +0.3~1.5% 小正收益,`MXFP8_ASM_MMA=1` 可开,低风险默认候选。
- dense WL+swizzle 是 **per-shape win**:`SWIZ=1` 70B Down 3075(1.03× per-K, 0.94× pt)、GateUp 2815(1.00× perK)、O_proj 2800(0.93×);小 shape 0.85-0.98×(prologue/grid 主导)。集成方向 = autotune 在 **WL vs per-K 按 shape 选**。
- 诊断旋钮(dense):`WL_NOSCALE_MMA`(emit 非-scaled v_mfma)、`WL_NOSCV`(跳 scale load)、`FP4_WLNOG2S`、`FP4_WLNODSR`、`WL_SC128`(scale 减半)。
- 诊断旋钮(grouped wgrad,**已从工作树删除,2026-07-05,从未 commit、git 历史里也没有,当前代码不可用**):`PT_MXGG_WGRAD_WL_{UNSCALED/PACK2/NOSCMMA/NOSCLOAD/SCMIN}`。生产 grouped wgrad 现仅 `_build_grouped_mxfp8_wgrad_kernel`(无 WL 路径)。
- **后续裁决(commit ab7ad885):用户最终裁定 `mx_wholeloop` 文件夹严禁存在,WL 从 dense 生产中整体移除**(手调 2133 行 bare-asm 硬件循环无法"小改"搬进 mxfp8_gemm_kernel.py)。**每 shape ≥97% per-tensor 的目标最终靠 WL-free 的纯 per-K + scale_pack(opsel byte-pack)+ 大-K scale 预取达成**,已提交 `dac31090`(分支 `dev/kyle/flydsl_mxfp8_compute`)。WL 代码留档于 `.wl_study/`(仓库外)+ git 历史 fc45f4bb,若将来重做只能走真正 fp8-only clean rewrite。

### C++ quant 配套修 3 处
- `quantize_mxfp8_dual_meta` 的 preshuffle int32 buffer sizing;`preshuffle_n_tiles≥1` 防 `%0`;padded K-block scale byte 恢复 **0**(非 unit/bias)。

## MXFP8 FlyDSL 必须全覆盖 MX case(禁 fallback)+ shape gate

### 为什么禁 fallback
- MXFP8 quant 吐的是 **FlyDSL-preshuffled int32 scale**,C++ 后端只认 raw E8M0;**一旦 FlyDSL fallback 到 HIPBLASLT/TURBO 必崩**。FlyDSL 必须自己覆盖所有 MX case。

### 需扩展的 4 块(mx_blockwise 从 275 失败 → 0,1536 passed)
1. **E5M2/HYBRID**:`MfmaScale` 的 `cbsz/blgp` 按 operand format 设(0=E4M3,1=E5M2)。
2. **fp16 out**:`StoreCPlain` 用 `out_ty`。
3. **K<256 / K%128≠0(K-tail)**:execute 零填充 K → operand=0、scale pad **127(=1.0)**。
4. **任意 N(N%16 / N%64≠0)**:combined-B preshuffle **zero-pad 到 256** + StoreC **mask col≥N**;general-M 用 execute pad M 到 64。
- 关键约束:**int32 scale 不可填充** → preshuffle fast-path **gate 到 K/M/N 全对齐**,unaligned 走 raw(broadcast)路径。

### NT shape gate(`_flydsl_mxfp8_nt_ok`)
```
M%128==0 and N%128==0 and K%128==0
and M>=256 and N>=256 and K>=256
and M*K<2**31 and N*K<2**31
```
- NT-fold 使 M/N/K 都当过 contract 维 → K-loop **无 K-tail** 且预取 2 层。

### Dispatcher 契约
- `granularity==MX_BLOCKWISE` 时 `can_handle()` 返回 **False**(FlyDSL NT 只接 preshuffled,而 dispatcher 喂的是 raw)。

---
来源: mxfp8-8wave-devloop/SKILL.md; project_mxfp8_wholeloop_port.md; mxfp8-grouped-gg-devloop/SKILL.md 优化7; project_mxfp8_grouped_wgrad_wl.md(WL 相关结论已在两文中被后续 commit 标注为废弃/删除,详见正文各节说明)

## dual cast:已被证伪的四个方向(r8,全部在地址鲁棒 ruler 上)

先看 pitfalls/02 关于"输出地址摆动 10-15%"那条 —— 下面每个结论都是在共用 buffer、
全臂预热、回文、跨 4-8 个摆放取均值的 ruler 上得到的,四个输出平面逐位 `torch.equal` 校验。

1. **`mrun`(M 轴长跑,krun 的镜像):11/11 cast 全负**,−1.1% 到 −25.9%,包括
   21504x3072 / 16384x3072 / 3072x15360 这些 col 平面散射的窄 cast。
   机理:`krun` 拉长的是**输入读**和 row 平面写的连续段;`mrun` 两个 M-tile 是**分开的
   store 阶段**,col 平面仍然是两次 128 B 而不是一次 256 B,col E8M0 仍是两次 4 B。
   换来的只有 workgroup 寿命翻倍和 grid 减半 —— 那是 pitfalls/03 已经标价的成本。
   ⇒ K 轴和 M 轴**不对称**,不要因为 krun 成功就推 mrun。
2. **非对称 tile(256x64 / 64x256)** —— 真正能把那两次 store 并成一次 256 B 的做法,
   `bm*bk==16*nth` 且 `nth<=1024` 时只有这两个可达,且不费 LDS 不掉 occupancy。
   实测 −6.3% 到 **−43.1%**:两边都拿满 128 B burst 远胜"一边 256 B 一边 64 B",
   而且 krun 越大越差。128x128 是真最优点。
3. **`ldsc_pad`(col stage 行 padding,消 LDS bank 冲突)**:11/11 cast 上 +0.0~+0.6%,
   pad 4/8/16/32 都一样。LDS 在这个 kernel 里**不是成本**(和 pitfalls/03 的"LDS 端口
   ≥5× 余量"一致)。之前读到的 +8~9% 全是地址假象。
4. **`col_map=1`(让每个 lane 拿不同 K-col,LDS 读无冲突)**:−0.4% 到 −21.9%。
   生产 map 下同 K-col、相邻 mblk 的四个 lane 写的是**连续四个 col-E8M0 字节(一个 dword)**;
   col_map=1 把 64 个字节 store 散到 64 条不同的 line。scale 平面的成本是 **line 数**,不是 store 数。

## r7 的 krun=3 + gm0.b8 在新 ruler 上复核通过

跨 6 个摆放取均值:krun=1 比生产慢 4.6-9.4%,r6 的 k1g1 慢 3.0-14.9%。
**符号和排序是真的**,但 r7 归给它的**幅度**(lin1.go 单靠 band +11.3%)有相当一部分是地址运气。

## 每 step 的 preshuffle 到底多少

`kern_0` 这个名字同时盖住 dual cast 和 fused scale preshuffle,历轮没分开过。
单独按生产 launch 维度计时:每次 launch 有 ~5.1 µs 的**与尺寸无关的地板**(3.3 MB 和 26 MB
都是 ~5.3 µs),地板之上的边际速率 ~11 TB/s;整 step 15 次 launch 合 15.9-20.8 µs/cell。
但那个地板很可能是**探针自己的 per-call Python dispatch**:生产里 preshuffle 和 GEMM 共用
同一个 `@flyc.jit` stub,不付第二次 dispatch。
⇒ audit 表里的 `cast+preshuffle` 基本**就是 cast**,不要把它当成 preshuffle 的优化空间。

## NT GEMM 的 per-tile 固定成本 F:量级、构成、以及它值多少

把 `wall/tiles` 拟合成 `F + R·K_ITERS`(M=N=8192,1024 tile,任何 K 都恰好 4.000 轮,
load balance 不变;K 一直扫到 `K_ITERS=3`,让 F 在它占主导的区间被**读出**而不是外推)。
再把同一拟合跑两遍:一遍四个 C-store quadrant 全活,一遍每个 quadrant 的 row-band SRD 用
`span_rows=0` 建(整条指令流原封不动 —— 同样的 cvt / permlane / 地址 VALU、同样的
`buffer_store` 发射 —— 只掉流量)。`written=1.000` vs `0.000` 就是"这一臂确实减掉了东西"的证据。

```
keep=4/4   F = 31.84 ns/tile   R = 5.249 ns/tile-Kiter
keep=0/4   F = 19.42 ns/tile   R = 5.438 ns/tile-Kiter
```

- **C-store 流量 = 12.4 ns/tile(F 的 39%)**。131072 B/tile / 12.4 ns ≈ 10.6 TB/s,就是 MALL
  写速率。C 是输出,这些字节不可谈判。
- **launch + prologue + epilogue 发射 = 19.4 ns/tile(61%)**。折成每 workgroup 是
  `256 × 19.4 ns = 4.97 µs`,按这个核实际跑的 ~1.52 GHz 算约 7,600 cycle。

**R 已经贴在硬件地板上**:一个 K-iter 每 workgroup 是 256 条
`v_mfma_scale_f32_16x16x128_f8f6f4`,每条 32 cycle,8 wave 分 4 个 SIMD = 每个 MFMA 单元 2 个
wave ⇒ 2048 cycle = 1347 ns @1.52 GHz;实测 `5.249 × 256 = 1344 ns`。⇒ **稳态 K 循环没有任何
暴露的访存**,g2s 双缓冲已经做满,K body 的下一个杠杆是 MFMA 调度本身,不是预取深度。

**F 值多少**:`F × tiles_per_step` —— linear1 7152 tile → 228 µs / 2733(8.3%);linear2 5328 →
170 / 1989(8.5%);QKV 2096 → 67 / 708(9.4%);MLP-up 2816 → 90 / 903(9.9%);MLP-down 同 9.8%。
这也正是对 Tensile 剩余赤字的形状:PT 在还赢 GEMM 的那几格用 MT320x256x128 / MT384x192x128
(MLP-up −14.4 µs / MLP-down −40.3 / linear1 −89.0)—— **更大的 tile = 更少的 workgroup = 更少的 F**。

### 已从 F 里抠出来的一刀:scale 预取提到算子 prologue 之前

scale plane 是**第三条** DRAM 流(A、B_T、两张 E8M0),而 k=0 的 scale load 原本发在两条
`wait_barrier` **之后**,离第一条吃 `sa0/sb0` 的 MFMA 只隔 ~2 个 ds_read 和一个 barrier ⇒
一次冷访存延迟**串行**叠在 stage fill 上,每 workgroup 一次,且在 occ=1(LDS 128 KB/WG)下
是纯暴露时间 —— CU 没有第二个 workgroup 可跑。提到 prologue 之前在统一 in-order vmcnt 上是安全的:
它们先于 `cur0` 退休,所以每个 `wait_barrier` 字面量含义不变,**barrier 数量也不变**
(顶部的半波 skew barrier 要求 wave 0-3 / 4-7 严格配对,增删 barrier 会死锁)。
实测 **F 31.84 → 30.85(−3.1%)**,R 不变;`K_ITERS=24`(所有 fwd pass 的 K)那档 157.94 → 154.95。

**顺带把"再多发一点"证伪了**:把 k=1 的算子 stage 也提上来凑成一发 7 条的 burst,F = 31.14,
不比只提 scale 好。机制:prologue 卡在**第一段的争用**上,不是发射率上 —— 256 个共驻 workgroup
一次性问内存要 16.8 MB 而不是 8.4 MB,并不会让 `cur0` 更早到。**只有那条被串行化的第三条流值得动。**

## r10:两张 scale 平面的完整放置账 + step 内 kernel 分解

### 放置账(单 cast ruler,只换地址不换字节;`_probe/r10_qbw.py`)

拆法:每个平面各一条 DROP 臂(`span_rows=0` / `num_records_bytes=0`,指令流不变、只掉流量)和一条
STREAM 臂(字节数不变、地址改成连续)。STREAM ≈ DROP 就说明**这个平面的成本是地址不是字节**。

| cast | prod | d.scal | s.scal | s.read | s.rowd | s.cold | **s.all4** |
|---|---|---|---|---|---|---|---|
| lin1.go 16384×21504 | 291.9 µs / 4.90 TB/s | +6.29% | **+6.59%** | +2.54% | +1.75% | +0.64% | **+13.10%** |
| lin1.w 21504×3072 | 51.1 µs | — | **+9.57%** | +3.00% | +0.80% | +1.66% | **+14.52%** |
| lin1.x 16384×3072 | 35.7 µs | — | **+7.74%** | −0.20% | +1.12% | +0.11% | **+9.24%** |

- **col-major fp8 store 已经基本完美**(s.cold 只有 +0.11…+2.59%)—— 定向怀疑它是错的。
- row E8M0 边际带宽 0.34–1.11 TB/s vs fp8 数据平面 4.2–9.6 TB/s,差 10 倍,全在 line 共享上:
  row line 的 32 个共享者是**连续 pid** ⇒ 8 个 XCD 全碰(≈16× 放大);col line 的共享者相距 NBK,
  `NBK % 8 == 0` 时落在同一个 XCD(lin1.x:同样 1.6 MB,row 3.2 µs vs col 0.4 µs)。
- **分母要用同形参照**:同进程 1431 MB 下 bf16→bf16(2r2w)只有 **5.30 TB/s**、u8→u8 5.33
  (201 MB 时是 7.06/6.74)。拿 6.5 TB/s 只读峰值定价会把 headroom 系统性放大 —— lin1.go 的
  4.90 TB/s 其实已经是同形连续拷贝的 **92.5%**。

### "让两条 line 都只归一个 XCD"的方块模型:**被证伪**

按 `units = ceil(rl/Bk) + ceil(32/gm)` 枚举 (gm, Bk) 方块,**u2 臂在宽 cast 上恰恰是最差的一档**
(lin1.go g32b896/u2 −2.62%、g64b896/u2 −6.04%),而赢家是 u5–u9 的中等方块。原因:M 方向一长,
bf16 读就按 K*2 跨步。**scale line 的归属和读的连续性是对抗的,没有共同最优,只有每形状的折中。**
`krun` 能把"placement left"压到 0.00%(机制确认),但 grid 随之缩小(NBM·NBKG/krun),总账 −1.3…−48%。

### 走位表本身在 step 里是负的 ⇒ 别再走这条路

见 `pitfalls/02` 的第三个杀手。**顺带在 step ruler 上复核了 r7 当年用单算子尺定下的生产配置,
全部站得住**(每一项消融都是负的,漂移之外):`plain(gm1/b0/krun1)` −0.47…−1.61、`krun1` −1.03…−2.47、
`band0` −0.33…−0.81、`gm1` −0.56…−1.04;`pad_extra` 0/8、`cm_row/col=0`、`cm_data=0/2` 也都不如现状。
**这一套 knob 已经在 step 尺上到位了,下一刀不在 knob 里。**

### step 内 kernel 分解(`_probe/r10_pre.py`,per-launch 拆开)

`kern_0 ×6` 不是 6 次 cast:FlyDSL 发出的核都叫 `kern_0`,实际是 **3 次 dual cast + 3 次 scale
preshuffle**(后者由 fused gemm stub 在同一条流上先发)。拆开后:

| 格 | 3×cast | 3×preshuffle | preshuffle 占整步 |
|---|---|---|---|
| QKV | 126.5 µs | **14.5 µs** | 2.1% |
| MLP-up | 167.5 | 15.0 | 1.7% |
| MLP-down | 158.5 | **20.5** | 2.3% |
| linear1 | 417.5 | 26.0 | 1.0% |
| linear2 | 295.5 | 22.0 | 1.1% |

preshuffle 只搬 15–56 MiB(~3–11 µs 的带宽),却花 14.5–26 µs ⇒ **是发射/延迟档,不是带宽档**。
**下一个杠杆:让 cast 直接写出 GEMM 要的 preshuffle 布局,把这 3 次 launch 整个删掉**,几何平均
约 +1.5%(阈值 0.6% 的 2.5 倍)。可行性:`_MX_SCALE_PACK=4` 意味着一个 dword 装 4 个 K-group 的
scale,而 `bk=128` 的 tile 每行正好产 4 个 scale ⇒ **每行恰好一个 dword**,`_store_scale` 的
`pack==1` 分支(整 dword store)本来就在;难点是 A-scale 的 preshuffle 布局跟 GEMM 的
`_BLOCK_M=256` 绑定(见 `gemm_mxfp8_kernel.py` 注释),cast 的 bm=128 要按 256 行块的次序落盘。
这同时也会把 row-scale 的 4 B/line 写放置一并解决 —— 但按本轮的教训,**收益要在 step 尺上认**。
