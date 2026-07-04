---
name: flydsl-sync
description: FlyDSL 仓库(standalone,含 turbo/ fp8 4wave/8wave dense GEMM、kernels/ helper)的远端编译/测试 + 本地 git 同步规程。FlyDSL kernel 需要 GPU 编译/跑 → 必须在远端(当前 chi2810,以 claim-mi355x-node 为准)做;canonical git 在远端 /mnt/vast/kyle/code2/FlyDSL,本地 sync/FlyDSL 是镜像。当用户要"改/测/跑 FlyDSL kernel(fp8_gemm_4wave/8wave、turbo、fused-quant)""同步 FlyDSL 本地↔远端""commit FlyDSL"时用此 skill。区别于 remote-sync(那个只管 Primus-Turbo)。
---

# flydsl-sync

> ⚠️ **2026-07-03：当前节点 = chi2811（不是 chi2810）**，chi2810 被别人任务占满 GPU，见 claim-mi355x-node 的换节点记录。下文命令示例里的 `chi2810` 用前先替换成 `chi2811`（或查 claim-mi355x-node 确认最新节点）。
>
> FlyDSL kernel 是 GPU kernel,**编译(flyc.compile)就要 GPU** → 不能像 Primus-Turbo 那样纯本地跑。所以 FlyDSL 是 **远端为主**:在远端 chi2810 编译/测/commit,再把 git+文件 rsync 回本地对齐。

## 拓扑 / 位置

```
本地 (smc300x, JuiceFS /workspace/code)
  └─ /workspace/code/gpt_oss_docker/sync/FlyDSL/   ← 本地镜像(git 仓库, branch dev/fp8-fused-quant)
        └─ .ssh-chi.sh                              ← 在 sync/ 上层: sync/.ssh-chi.sh
经跳板机 root@149.28.124.225 →
远端 chi2810 (MI355X, 8 卡当前全空;别人 idle 容器 pdval-vllm/dlrmv3,长任务前 rocm-smi 复查)
  └─ /mnt/vast/kyle/code2/FlyDSL/                   ← canonical git(/mnt/vast=aac 共享 NFS,持久)
        = 容器 mlperf_gptoss 内 /workspace/code/FlyDSL
```
- 连接见 [[project_chi2811_sync]];节点/容器见 [[claim-mi355x-node]]。当前节点 chi2810、镜像 mlperf_gptoss:saved-20260625b。
- 关键文件:`turbo/fp8_gemm_4wave.py`(fp8 4wave + fused-quant,`fuse_q` flag)、`turbo/fp8_gemm_8wave.py`、`kernels/fp8_gemm_utils.py`(共享 helper)。

## 标准工作流(改 FlyDSL kernel)

```bash
cd /workspace/code/gpt_oss_docker/sync
RS="rsync -rlptDzh --no-o --no-g -e $PWD/.ssh-chi.sh"   # --no-o --no-g 避开 JuiceFS chown/chmod 报错(cosmetic)

# 1. 本地编辑 sync/FlyDSL/...  →  2. 推到远端
$RS FlyDSL/turbo/fp8_gemm_4wave.py root@chi2810:/mnt/vast/kyle/code2/FlyDSL/turbo/fp8_gemm_4wave.py

# 3. 远端 GPU 测试(GPU4-7;GPU6 常用)。带 FLYDSL_EXTRA_SOURCE_DIRS 见下方缓存坑
./.ssh-chi.sh root@chi2810 "docker exec mlperf_gptoss bash -lc 'cd /workspace/code/FlyDSL && HIP_VISIBLE_DEVICES=6 python opt_fq_test.py'"

# 4. 满意后在远端 commit(canonical)。署名见 [[feedback_commit_style]](kyle-256,无 Claude)
./.ssh-chi.sh root@chi2810 "cd /mnt/vast/kyle/code2/FlyDSL && git add <files> && GIT_AUTHOR_NAME=kyle-256 GIT_AUTHOR_EMAIL=Kyle.Zhao@amd.com GIT_COMMITTER_NAME=kyle-256 GIT_COMMITTER_EMAIL=Kyle.Zhao@amd.com git commit -m '...'"

# 5. 把 git 历史 + 文件 rsync 回本地对齐(关键:不然本地 git 落后)
$RS root@chi2810:/mnt/vast/kyle/code2/FlyDSL/turbo/fp8_gemm_4wave.py FlyDSL/turbo/fp8_gemm_4wave.py
rsync -rlptDzh --no-o --no-g --delete -e "$PWD/.ssh-chi.sh" root@chi2810:/mnt/vast/kyle/code2/FlyDSL/.git/ FlyDSL/.git/
# 校验:本地 == 远端 HEAD
cd FlyDSL && git rev-parse HEAD   # 应等于远端 git rev-parse HEAD
```

- 本地 git 设 `git config core.fileMode false`(JuiceFS rsync 翻动权限位会假报一堆 M,关掉即净)。
- **能直连**:本地 sync/ 已种公钥,`./.ssh-chi.sh root@chi2810 "..."` 直接用;rsync 用同一 transport。
- 换节点:/mnt/vast 全集群同源,rsync 到任一节点 /mnt/vast/kyle/code2/FlyDSL 后所有节点可见(见 [[claim-mi355x-node]])。

## ⚠️⚠️ JIT 缓存坑(必读,踩过大坑)

flydsl 编译缓存(`~/.flydsl/cache`,jit_function.py `_jit_function_cache_key`)的 key **只 hash**:① flydsl 自身包(compiler/expr/runtime/utils)② @kernel 函数及其**嵌套 def 的源码** ③ FLYDSL_EXTRA_SOURCE_DIRS 指定目录的 .py。

**不 hash**:模块级**辅助类的方法源码**(`_collect_dependency_sources` 不跟进实例化类的方法)。

→ 后果:改一个**模块级 class 的方法**(如曾经的 module-level `FusedQuantG2SLoader`),缓存**不失效**,跑出**旧版**(实测假象:改坏 kernel 仍出旧正确 SNR)。flydsl 安装在 site-packages(saved-image 烤的副本),kernel 在独立的 /workspace/code/FlyDSL → 更易踩。

**两个解法**:
1. **把可编辑的 kernel 逻辑放进 @kernel 函数体 / 其嵌套 def / 嵌套 class**(已对 FusedQuantG2SLoader 这么做,commit f80c52f2)→ 改它自动失效,**首选,零踩坑**。
2. 测试加 `FLYDSL_EXTRA_SOURCE_DIRS=/workspace/code/FlyDSL/turbo:/workspace/code/FlyDSL/kernels` → hash 整个目录。
- 实在拿不准:`rm -rf /root/.flydsl/cache` 强清(最后手段)。
- **验证缓存是否失效的金标准**:故意把 kernel 改算错(如 scale×2),不清缓存跑 —— 出 nan/错=失效正常;仍出旧正确 SNR=陈旧(踩坑)。

## 其他坑(踩过)
- `Vec.to(Float8E4M3FN)` 走 arith.truncf,**后端不 lower**;fp8 cast 用 `fx.rocdl.cvt_pk_fp8_f32`(2 f32→2 fp8/op)。`cvt_scalef32_pk8_fp8_bf16` **gfx950 Cannot select,不可用**。
- `create_buffer_resource(max_size=True)` 会 OOB 读垃圾;用 `max_size=False, num_records_bytes=...`。
- fused-quant 现状/结论见 [[project-flydsl-fused-quant]];4wave asm 主线见 [[project-flydsl-nt-rewrite-design]]。
