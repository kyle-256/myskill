---
name: remote-sync
description: 管理两份 Primus-Turbo（"turbo"）的本地镜像 + 远程 docker 布局（当前节点 chi2810，以 claim-mi355x-node 为准）。mxfp4 与 tensorwise 各一份，各自独立 venv。当用户要"修改/编辑/运行 turbo 代码""同步到远程""在 GPU 上跑 turbo"时使用。注意：turbo = Primus-Turbo，和 FlyDSL 无关。
---

> ⚠️ **现在有两份 turbo（Primus-Turbo）并存**，各自独立 venv：
> - `sync/mxfp4/Primus-Turbo`（分支 `dev/kyle/flydsl_mxfp4_compute`）→ 远程 venv `/opt/venv`
> - `sync/tensorwise/Primus-Turbo`（分支 `dev/kyle/flydsl_grp_gemm`）→ 远程 venv `/opt/venv-tw`
>
> **turbo = Primus-Turbo**（origin `https://github.com/AMD-AGI/Primus-Turbo.git`），**与 FlyDSL 无关**。

# remote-sync

工作模型：**本地编辑（无 GPU）→ rsync 推到 chi2810 容器 → 用对应 venv 在 GPU 上跑**。每份 turbo 有独立的远程目录 + 独立 venv，互不污染。

## 布局对照表

| | **mxfp4** | **tensorwise** |
|---|---|---|
| 本地镜像 | `/workspace/code/gpt_oss_docker/sync/mxfp4/Primus-Turbo` | `/workspace/code/gpt_oss_docker/sync/tensorwise/Primus-Turbo` |
| git 分支 | `dev/kyle/flydsl_mxfp4_compute` | `dev/kyle/flydsl_grp_gemm` |
| 远程 host 路径 | `/mnt/vast/kyle/code2/Primus-Turbo` | `/mnt/vast/kyle/code2/Primus-Turbo-tensorwise` |
| 容器路径 | `/workspace/code/Primus-Turbo` | `/workspace/code/Primus-Turbo-tensorwise` |
| 独立 venv | `/opt/venv` | `/opt/venv-tw` |
| editable 指向 | `…/Primus-Turbo/primus_turbo` | `…/Primus-Turbo-tensorwise/primus_turbo` |

- host `/mnt/vast/kyle/code2` 整目录 bind-mount 到容器 `/workspace/code`（所以 host 路径去掉 `/mnt/vast/kyle/code2` 换成 `/workspace/code` 即容器路径）。
- 远程容器名：`mlperf_gptoss`。chi2810 经跳板机连（见 `.ssh-chi.sh`）。
- 本地 `.git` 完整保留（canonical git 在本地）；远程**不需要** `.git` 即可 import/build（rsync 排除了 `.git`，远程 build_info 的 commit 会显示 `unknown`，纯装饰）。

## 在 GPU 上运行（选对 venv！）

```bash
cd /workspace/code/gpt_oss_docker/sync
# mxfp4
./.ssh-chi.sh root@chi2810 "docker exec -e HIP_VISIBLE_DEVICES=7 mlperf_gptoss bash -lc \
  'cd /workspace/code/Primus-Turbo && /opt/venv/bin/python <script>'"
# tensorwise
./.ssh-chi.sh root@chi2810 "docker exec -e HIP_VISIBLE_DEVICES=7 mlperf_gptoss bash -lc \
  'cd /workspace/code/Primus-Turbo-tensorwise && /opt/venv-tw/bin/python <script>'"
```

验证 import 走对了 repo：

```bash
/opt/venv/bin/python    -c "import primus_turbo; print(primus_turbo.__file__)"  # → …/Primus-Turbo/…
/opt/venv-tw/bin/python -c "import primus_turbo; print(primus_turbo.__file__)"  # → …/Primus-Turbo-tensorwise/…
```

## rsync 同步（本地 ↔ chi2810）

`-e "$PWD/.ssh-chi.sh"` 用 rsync transport 包装（连 chi2810），`--exclude-from=.rsync-exclude` 排除 `.git`/`*.so`/build/venv 等。**必须从 `sync/` 目录跑**（`$PWD` 解析正确）。注意 trailing `/`（同步目录内容）。

```bash
cd /workspace/code/gpt_oss_docker/sync

# 推（本地 → 远程，跑测前同步代码）—— mxfp4
rsync -azh --exclude-from=.rsync-exclude -e "$PWD/.ssh-chi.sh" \
  "$PWD/mxfp4/Primus-Turbo/" root@chi2810:/mnt/vast/kyle/code2/Primus-Turbo/
# 推 —— tensorwise
rsync -azh --exclude-from=.rsync-exclude -e "$PWD/.ssh-chi.sh" \
  "$PWD/tensorwise/Primus-Turbo/" root@chi2810:/mnt/vast/kyle/code2/Primus-Turbo-tensorwise/

# 拉（远程 → 本地，从 chi2810 取最新）—— 把上面 src/dst 对调即可
```

校验对齐：`md5sum <file>` 本地 vs 远程相等即同步（`.so`/build 产物只在远程，属正常差异）。

⚠️ **不要加 `--delete`（2026-06-25 踩过）**：本地是全新 checkout，会让 rsync `--delete` 删掉远程一堆**不能删的东西**——
- `primus_turbo/_build_info.py`（build 生成、**运行时被 import**，删了 import 直接挂）；
- hipify 生成的中间件（`*_hip.cpp/.h/.cuh/.hip`，远程编译产物）；
- `3rdparty/` 子模块源码（本地是空 gitlink，远程才有，build 必需）。

用**增量 rsync（不 delete）**即可：覆盖更新所有 Python 源码、补齐新文件、保留上述产物。代价仅是 benchmark 旧布局副本残留（纯 cosmetic）。真要 delete 必须 `--exclude=/3rdparty/` 并保护 `_build_info.py`，且先 `-n` dry-run 审查。

## 两份 venv 独立的关键机制（务必理解，否则会互相污染）

`primus_turbo` 是 **editable 安装**（PEP 660）：每个 venv 的 `site-packages/__editable___primus_turbo_0_0_0_finder.py` 里有一个 `MAPPING` dict 指向某个 repo 的 `primus_turbo`。两 venv 各指各的 repo → 独立。

**⚠️ 致命坑（2026-06-25 踩过）：clone 出来的 venv 的 `setuptools.pth` 会把源 venv 的 site-packages 追加到 `sys.path`**，导致：
- 新 venv fallback 到老 venv（不是真隔离）；
- 在新 venv 里 `pip install -e` 时，pip 把 editable **写进了老 venv**（污染另一份！），表现为两 env 的 import 路径互换。

**正确建第二份 venv 的步骤**：

```bash
docker exec mlperf_gptoss bash -lc '
  cp -a /opt/venv /opt/venv-tw                                   # 1) clone（overlay reflink，~10s，7.9G）
  # 2) 先消除泄漏：把 setuptools.pth 里的 /opt/venv 改成 /opt/venv-tw
  sed -i "s#/opt/venv/lib#/opt/venv-tw/lib#" \
    /opt/venv-tw/lib/python3.12/site-packages/setuptools.pth
  # 验证隔离：sys.path 不应含 /opt/venv
  /opt/venv-tw/bin/python -c "import sys; print(\"LEAK\" if \"/opt/venv/lib/python3.12/site-packages\" in sys.path else \"ISOLATED_OK\")"
  # 3) 再装 editable（这步会 CMake build C++ → primus_turbo/lib/libprimus_turbo_kernels.so，~4min）
  cd /workspace/code/Primus-Turbo-tensorwise && /opt/venv-tw/bin/pip install -e . --no-build-isolation
'
```

若忘了第 2 步就装了，**修复**：直接改两个 venv 的 finder `MAPPING` 路径（无需重 build）：

```bash
F=__editable___primus_turbo_0_0_0_finder.py
sed -i "s#/workspace/code/Primus-Turbo-tensorwise/#/workspace/code/Primus-Turbo/#g" /opt/venv/lib/python3.12/site-packages/$F
sed -i "s#/workspace/code/Primus-Turbo/#/workspace/code/Primus-Turbo-tensorwise/#g" /opt/venv-tw/lib/python3.12/site-packages/$F
```

## 新建 / 重建一份 turbo 的完整流程

1. **本地 clone + checkout 分支**（chi2810 容器无 github 凭证，分支必须从**本机**用 SSH key fetch）：
   ```bash
   cd /workspace/code/gpt_oss_docker/sync/<folder>
   git clone --no-recurse-submodules /…/sync/mxfp4/Primus-Turbo Primus-Turbo   # 复用本地对象，快
   cd Primus-Turbo
   GIT_SSH_COMMAND="ssh -i /workspace/code/.ssh_docker/id_ed25519 -o StrictHostKeyChecking=no -o IdentitiesOnly=yes" \
     git fetch git@github.com:AMD-AGI/Primus-Turbo.git <branch>:<branch>
   git checkout <branch>
   git remote set-url origin https://github.com/AMD-AGI/Primus-Turbo.git
   ```
2. **rsync 推源码到新远程目录**（见上）。
3. **拷子模块**（CK/hipify_torch，build 必需；若分支间 submodule 指针相同可直接 host 级 cp）：
   ```bash
   cp -a /mnt/vast/kyle/code2/Primus-Turbo/3rdparty/. /mnt/vast/kyle/code2/<NewRemote>/3rdparty/
   ```
4. **建独立 venv + build**（见上"两份 venv 独立"）。
5. **验证**：两 venv import 路径分离 + 完整 import 加载 `.so` 无报错。

## 排除规则（`.rsync-exclude`）

排除：`/build/`、`/build-fly/`、`/dist/`（**根级锚定**，不误伤 Python 模块同名目录）、`*.so`/`*.so.*`/`*.o`/`*.a`、`*.pyc`、`__pycache__/`、`*.egg-info/`、`.pytest_cache/`、`pytest-of-*/`、`.rocprofv3/`、大二进制（`*.rpd/parquet/npy/npz`）、`venv/`/`.venv/`/`env/`/`.env/`、`.idea/`/`*.swp`、`*.log`。

**保留**：`.git/`（本地 canonical）、`*.csv`、`*.md`、`auto_optimize_logs/`（用户要看，~230M）。

注意 `*.so` 被排除 → 远程的 `primus_turbo/lib/libprimus_turbo_kernels.so` 是远程 build 产物，**不会被本地空目录覆盖**；本地镜像没有它属正常。

要改 → 编辑 `/workspace/code/gpt_oss_docker/sync/.rsync-exclude`。

## Git push（用本机 SSH key，chi2810 容器无凭证）

origin 是 HTTPS（`AMD-AGI/Primus-Turbo`），本机/容器都无 GitHub HTTPS 凭证。push 用本机的 jump-host SSH key（`/workspace/code/.ssh_docker/id_ed25519`，认证 kyle-256）：

```bash
cd /workspace/code/gpt_oss_docker/sync/<folder>/Primus-Turbo
GIT_SSH_COMMAND="ssh -i /workspace/code/.ssh_docker/id_ed25519 -o StrictHostKeyChecking=no -o IdentitiesOnly=yes" \
  git push git@github.com:AMD-AGI/Primus-Turbo.git <branch>
```

（`git@…` SSH URL 临时覆盖 HTTPS origin；不改 remote 配置。）

## 远程 docker 镜像管理（commit / save 备份）

容器 `mlperf_gptoss`（`sleep infinity` 开发容器）持久数据 = bind mount `/workspace/code`（host 上）+ 容器内 `/opt/venv*`。**容器内改动（如新建 venv-tw）必须 `docker commit` 才持久化**（bind mount 部分本来就在 host）。

**commit 命名**：`mlperf_gptoss:saved-YYYYMMDD`
```bash
docker commit mlperf_gptoss mlperf_gptoss:saved-20260625
```

**删旧镜像前必须重建容器**：运行中的容器占用旧镜像 → `docker rmi` 被拒（dangling 不算真删）。env 已 baked 进 commit 的新镜像，重建无需重传 env：
```bash
docker rm -f mlperf_gptoss
docker run -d --name mlperf_gptoss --network host --ipc host \
  --shm-size 68719476736 --device /dev/dri --device /dev/kfd \
  --group-add video --cap-add CAP_SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt label=disable \
  -v /mnt/vast/kyle/code2:/workspace/code \
  mlperf_gptoss:saved-YYYYMMDD sleep infinity
docker rmi mlperf_gptoss:saved-<old>     # 重建后旧镜像才删得掉
```

**save 成 tar 备份** → `/mnt/vast/kyle/code2/docker_images/mlperf_gptoss-YYYYMMDD.tar.zst`（zstd，~20G）。
⚠️ **关键坑（2026-06-25 踩过）**：`docker save` 会把整个 ~82G 镜像先暂存到 `/var/lib/docker/tmp`（在只剩 ~30G 的根盘 sdb2）→ `no space left`。**必须先把暂存目录 bind 到 /mnt/vast（144T）**：
```bash
mkdir -p /mnt/vast/kyle/code2/docker_tmp
mount --bind /mnt/vast/kyle/code2/docker_tmp /var/lib/docker/tmp
docker save mlperf_gptoss:saved-YYYYMMDD | zstd -T0 -q \
  -o /mnt/vast/kyle/code2/docker_images/mlperf_gptoss-YYYYMMDD.tar.zst
zstd -t /mnt/vast/kyle/code2/docker_images/mlperf_gptoss-YYYYMMDD.tar.zst   # 校验：解压字节数≈镜像 SIZE
umount /var/lib/docker/tmp && rm -rf /mnt/vast/kyle/code2/docker_tmp        # 还原+清暂存
```
注意：`docker save | zstd` 管道退出码取的是 zstd 的，docker save 报错走 stderr 易被误判成功 → 用 `${PIPESTATUS[0]}_${PIPESTATUS[1]}` 或 `zstd -t` 核实。`mlperf_gptoss2` 是另一独立容器/镜像，别动。

## 测试 / 验证两个环境（2026-06-25 实测）

各环境用**自己的 venv** 跑 pytest，挑空闲 GPU（`rocm-smi --showmeminfo vram`，找 Used ~300MB 的卡），MI355X = gfx950。

```bash
# mxfp4（代表性：fp4 dense gemm）
./.ssh-chi.sh root@chi2810 "docker exec -e HIP_VISIBLE_DEVICES=6 mlperf_gptoss bash -lc \
  'cd /workspace/code/Primus-Turbo && /opt/venv/bin/python -u -m pytest tests/pytorch/ops/test_gemm_fp4.py -q -p no:cacheprovider'"
# → 实测 133 passed, 0 failed（~34s）

# tensorwise（FlyDSL grouped gemm 后端）
./.ssh-chi.sh root@chi2810 "docker exec -e HIP_VISIBLE_DEVICES=5 mlperf_gptoss bash -lc \
  'cd /workspace/code/Primus-Turbo-tensorwise && /opt/venv-tw/bin/python -u -m pytest \
   tests/pytorch/ops/test_grouped_gemm_fp8.py -k FLYDSL -x --tb=long -q -p no:cacheprovider'"
```

**关键坑（都 2026-06-25 踩过）：**
- **`-u` 必须加**：否则 pytest stdout 块缓冲，看不到进度；尤其别 `| tail -N`（tail 等 EOF 才输出 → 全程 0 输出像卡死）。
- **ssh 中断 ≠ 杀远程**：本地 Ctrl-C/中断 `./.ssh-chi.sh` 只断 ssh，**远程 `docker exec` 里的 pytest 仍在跑**。多次中断重跑 → 一堆僵尸 pytest **抢同一块 GPU**，越来越慢。重跑前先 `docker exec mlperf_gptoss pkill -9 -f "pytest <file>"`。
- **FlyDSL 测试参数爆炸**：`test_grouped_gemm_fp8.py -k FLYDSL` 选中 **~2.3 万** case（大多运行时 skip，但跑到的每个做 per-shape autotune 编译，极慢，全量几百分钟）。**只跑有界子集**（`-x` 停首个失败，或定死 shape：`-k "FLYDSL and Format.E4M3-ori_dtype0-NK3-2048-1"`）。`deterministic` 版对 flydsl 全 skip，没用。
- **E4M3 tensorwise grouped gemm 的 SNR 失败是预存的、与后端无关**：同一 shape 下 `backend=None`(默认 **TRITON**，非 CK！op 层 `default_backend=BackendType.TRITON.value`)与 `FLYDSL` 的 `Out-SNR` **一致到小数点后 9 位（11.976079940795898）** → 误差在二者共用的 **量化/参考路径**，不是 kernel/后端问题，也不是 rebase 引入。flydsl 大多数 shape 通过（114 passed）。判断"是不是自己改坏"的捷径：**同 shape 跑两个后端比 SNR，一样就是上游量化精度问题**。

## tensorwise 分支 rebase 到 main（单 commit 历史）

需求："基于 main，只保留最新 commit 的有用改动"。分支与 main 都独立实现了 flydsl grouped gemm（main 已合 `#384`），13 个 commit 里只有 flydsl autotune 微调是分支独有。做法（结果 = 最新 main + 1 个 squash commit）：

```bash
cd .../tensorwise/Primus-Turbo
git fetch origin main
git branch backup/pre-rebase <old-tip>                 # 安全备份
git reset --hard origin/main                           # 分支重置到最新 main
git checkout backup/pre-rebase -- <分支独有/要覆盖的文件>  # 只覆盖分支版本的那几个 flydsl 文件
git commit -m "<最新 commit 的信息>"                     # 单 commit
GIT_SSH_COMMAND="ssh -i /workspace/code/.ssh_docker/id_ed25519 ..." \
  git push --force git@github.com:AMD-AGI/Primus-Turbo.git <branch>
```
冲突文件**整文件取一方**（用 `git checkout <ref> -- file`），别 `-X theirs` 逐 hunk 合（两套并行实现会拼出坏代码）。新旧 main 间 csrc/cmake/setup **零改动** → `.so` 不用重编，editable 装的 Python 改动 rsync 后即时生效。

## FlyDSL playground kernel 移植进 turbo（fp8 4wave 实例，2026-06-25）

把 FlyDSL playground 的 fp8 **4-wave** kernel 移植进 tensorwise turbo：
- 落两文件（**独立、不接产品 dispatch**）：`primus_turbo/flydsl/gemm/gemm_fp8_4wave_kernel.py`（入口 `compile_fp8_gemm_4w`，含 `Mfma16x16x128AGPR`）+ `primus_turbo/flydsl/utils/gemm_fp8_4wave_helper.py`（vendored 自 playground `kernels/fp8_gemm_utils.py`）。
- **为什么单独 vendor helper，不复用 `gemm_helper.py`**：turbo 产品化的 8-wave 把原语分叉了——`Mfma16x16x128` 去掉 `call_one`、`G2S/S2RLoader` 去掉 `load_one`、`StoreC`→`StoreCPerTensor`（per-tensor 标量 scale，非 row/col-wise）。4-wave 直接 import 会崩。vendor 一份保证 bench-identical 且不碰 8-wave。
- **验证移植对不对的金标准**：同进程同 device 同输入，把 playground 原版和移植版的输出**逐元素比对**，应 `outdiff=0`（同源 kernel bit-identical）；再比 TF（应 ±1.5% 内噪声）。SNR 只能证"能跑对"，证不了"和上游同一版本"。
- **4wave 现有"接近 autotune"入口**（2026-06-25 加）：`gemm_fp8_4wave(a,b_t,c,a_scale,b_scale,*,b_preshuffled,autotune)`——首调对 `_4WAVE_CANDIDATES`（tile + xcd_remap，256×256 几乎总最优）微基准、按 `(M,N,K,b_preshuffled)` 缓存最快 launch（缓存命中 ~0.1ms）；`compile_fp8_gemm_4w` 仍是裸单配置编译。结构：裸 compile → `_compile_4wave_cached`(lru_cache) → `_get_compiled` → `_autotune_dispatch` → 公开 `gemm_fp8_4wave`。
- **AGPR 是 occupancy-specific（实测结论）**：`Mfma16x16x128AGPR`（inline-asm `=a,v,v,0` 钉累加器到 AGPR）对 **4-wave（occ=1）+5~13%**，但对 **8-wave（occ=2）−30~37%**（AGPR 预算减半→掉 occ 或 RF 互倒）。所以 **8-wave 不要 AGPR**，用编译器默认 VGPR-form(SSA)。fp8 下 4w≈8w（±2.5%，4w 赢大 N 的 FFN gate、8w 赢小方阵），不像 mxfp4 那样 4w 碾压。

**致命坑（2026-06-25 踩过）：本地 `sync/FlyDSL` 比远程容器 `/workspace/code/FlyDSL` 旧** → 照本地 vendor 会搬到**补丁前**版本。具体：fp8 4wave 的 AGPR 提速（ROCm/FlyDSL **PR #714**，`Mfma16x16x128AGPR` inline-asm `=a,v,v,0` 把 f32x4 累加器钉 AGPR、消 `v_accvgpr_mov`+`s_nop`，+5~13%）只在远程的 `kernels/fp8_gemm_4wave.py`，本地 fork 分支 `dev/fp8-fused-quant` 的 `kernels/` 没有。表现为移植版慢 ~15%。
- **教训**：移植 playground kernel 前，先 `docker exec ... diff <(cat 本地) <(远程 cat)`，或直接从**远程容器**（bench 实际跑的那份）取源，别只信本地。
- 取远程文件:host 上没有 FlyDSL（在容器内），`rsync root@host:/workspace/code/FlyDSL/...` 会 No such file。用 `./.ssh-chi.sh root@chi2810 "docker exec mlperf_gptoss cat <容器内路径>" > 本地文件` 落盘。

## 常见坑

- **dubious ownership（JuiceFS）**：这套挂载 uid 错配，git 到处报 dubious ownership。已全局 `git config --global --add safe.directory '*'` 一劳永逸。
- **`git clone` 本地路径报 SSH "access rights"**：其实是源 repo `.git` 的 dubious-ownership（不是真要 SSH）。加 `safe.directory '*'` 即解。
- **rsync 排除默认匹配任意深度同名目录** → `/build/`、`/dist/` 用 `/` 开头锚定到 repo 根。
- **submodule 指针**：换分支前先 `git diff <b1> <b2> -- .gitmodules 3rdparty` 看是否变；变了要重新拉/拷子模块再 build。
- **磁盘**：节点 `/`(docker 盘) 常年高占用（chi2810 清理后 ~89% / ~94G free；换节点时常需 `docker image prune -a -f` 腾 >90G 给 87.4G 镜像）。每份 venv 7.9G、每份编译 lib ~60M。clone venv 前先 `df -h /`。

## 故障排查

- import 走错 repo → 看对应 venv 的 `__editable___primus_turbo_0_0_0_finder.py` 的 `MAPPING`，sed 改回正确路径（无需重 build）。
- 怀疑 venv 没隔离 → `/opt/venv-tw/bin/python -c "import sys;[print(p) for p in sys.path]"`，不应出现 `/opt/venv/lib/...`。
- 改了 csrc/`.cu`/`.cpp` → 在对应 venv 重跑 `pip install -e . --no-build-isolation` 重编 `libprimus_turbo_kernels.so`。
- 看远程变更 → 在对应本地镜像跑 `git status`/`git diff`。
