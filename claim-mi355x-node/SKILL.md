---
name: claim-mi355x-node
description: 当原本固定使用的 mi355x 计算节点不可用、容器被删、或被别的 slurm job 占满时，从 login_node2 找一台 GPU 实际空闲的 compute 节点，把本机公钥种到节点上，再起 mlperf_gptoss 容器。首选从共享盘保存的镜像 mlperf_gptoss:saved-*（含 triton 3.7 + flydsl/primus_turbo，开箱即用）恢复；没有才从 fresh rocm/primus:v26.2 起再升级安装。挂 /mnt/vast/kyle/code2 → /workspace/code。当用户说"找台空机器/换节点/容器没了/重建容器/这节点被占了/保存容器/重建保存镜像"等场景时使用。配合 remote-mlperf-gptoss 与 remote-sync skill。
---

# claim-mi355x-node

> ⚠️ **本机不适用**：此 skill 用于在远程集群（经 login_node2）claim 一台 GPU compute 节点并起 docker 容器，是纯远程基建操作。当前本地机器（smc300x docker 容器）无 docker CLI、不能 claim 远程节点，**此流程在本机跑不了**，仅作参考。
> 本机的 Primus-Turbo 代码已同步到 `/workspace/code/gpt_oss_docker/sync/Primus-Turbo`（从 chi2811:/mnt/vast/kyle/code2/Primus-Turbo 拉取）。

集群里 mi355x 分区大半 down，剩下的节点 slurm 都标 `ALLOCATED`，但**实际 GPU 利用率经常是 0%**（别人的容器只占着显存等请求）。这个 skill 的工作流：经登录节点找一台真闲的，授权自己 ssh，起容器，升 triton。

当前用哪台节点看 `~/.ssh/config` 里 `compute_node_new` 的 `HostName`（这是 single source of truth）；不要在本 SKILL 里硬编节点名。

## 拓扑回顾

```
本地 (/wekafs/kyle)
  └─ ssh login_node2          149.28.124.225, root, key=/wekafs/kyle/.ssh/id_ed25519
       └─ ssh chiXXXX         登录节点上的 root 可以无密码 ssh 任意 chi 节点（hostbased/共享 known_hosts）
            └─ docker run mlperf_gptoss   # 挂 /mnt/vast/kyle/code2:/workspace/code
```

要点：
- **本机 → 任意 compute 节点的直连**只有目标节点 `~/.ssh/authorized_keys` 里有我们公钥才行。chi2811 之前已经种过；新节点要自己种。
- **从 login_node2 → 任意 chi 节点**不需要密码也不需要我们的公钥（root 在登录节点上有别的 trust）。所以"种公钥"这一步必须**经 login_node2 跳过去**做。
- bind-mount 路径 `/mnt/vast/kyle/code2` 在 **`/mnt/vast`（10.2.123.177:/aac-8634674/aac/shared/data，152T 集群共享 NFS）** 上，**每个节点同源**，写一次全集群可见 —— 换节点不丢数据。
  **注意**：旧文档用的 `/mnt/shared`（login_node2:/mnt/nvmeraid）在 2026-06-09 实测**空闲节点上是坏/空的 NFS（root 都 Permission denied）**，已弃用，改走 `/mnt/vast`。`sync.sh` 的 `REMOTE_BASE` 也要同步指向 `/mnt/vast/kyle/code2`。

## 公钥（粘到 authorized_keys 的内容）

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDNk0/FA7DjtcE50WfO1Xf+qGUARLegsXwliz4TU0sPk kyle-20260509
```

源文件：`/wekafs/kyle/.ssh/id_ed25519.pub`。指纹片段 `DNk0/FA7DjtcE50WfO1Xf+qGUARLegsXwliz4TU0sPk`（grep 去重用）。

## 步骤

### 1. 看集群整体状态

```bash
ssh login_node2 'sinfo'
ssh login_node2 'squeue'
```

读法：
- `sinfo` 只看 `mix` / `alloc` / `idle` 的节点，down* 是大批维护中的，跳过。idle 几乎不会有。
- `squeue` 找 `mi355x` 分区的 R 状态 job 拿到对应 `NODELIST`。这些是"slurm 视角占着的"候选 —— 不代表 GPU 真的在跑。

### 2. 经 login_node2 批量调查候选节点的真实 GPU 占用

```bash
ssh login_node2 'for n in chiAAAA chiBBBB chiCCCC; do
  echo "===== $n ====="
  ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 $n "
    echo \"[GPU%]\"; rocm-smi --showuse 2>/dev/null | awk \"/GPU\\[.*GPU use/\" | head -8
    echo \"[docker]\"; docker ps --format \"{{.Names}}\\t{{.Image}}\\t{{.Status}}\" 2>/dev/null
    echo \"[GPU pids]\"; rocm-smi --showpids 2>/dev/null | grep -E \"^[0-9]+\" | head -10
  " 2>&1 | grep -v "^Warning:"
done'
```

挑节点的优先级：

1. **首选**：`GPU use=0%` 且节点上**已经有 `rocm/primus:v26.2` 镜像**（看 `docker ps` 或 `docker images`）—— 远程没外网，省掉拉镜像。
2. **次选**：GPU 0% 但镜像没有 —— 得手动想办法把镜像搞过去（其他节点 `docker save | ssh ... docker load`）。
3. **避开**：GPU 真在 ~99% 跑（`affectionate_galileo` jax MAD CI、`dflash-sgl` 这种），即使 slurm 显示一样的 ALLOCATED，别动。
4. **避开**：节点上别人的容器在用同一组 GPU（`rocm-smi --showpids` 有进程）—— 起容器抢 GPU 会冲突。

GPU%=0 但 `rocm-smi --showpids` 有 vllm/sglang worker 进程占着大额显存（200GB+），意味着是 idle waiting，**显存被锁**：你能起容器但跑大 batch 会 OOM，小任务可以共存。

#### 2026-06-01 血泪教训（节点争抢，反复踩）

- **slurm `sinfo idle` / `squeue` 完全不可信**：大家都在 slurm 之外直接 `docker run` 抢 GPU。一台 slurm 标 `idle` 的节点可能有 9 个 GPU 进程 + 3 个容器。**唯一可信的是直接 ssh 进去 `rocm-smi --showpids` 数进程 + `docker ps`。**
- **真·可用判据**（鲁棒批量扫，挑出来再复核）：`rocm-smi --showpids 2>/dev/null | grep -cE '^[0-9]'` == 0 **且** `docker ps` 无业务容器。`vram0=0%` 也要,但 pid 数是金标准。
- **`primus-training` 是多节点训练 job**，会同时铺到一排节点（chi2832/2835/2816/2800...）**每节点吃满 8 卡 ~96% VRAM 真算** → 落在这种节点上，容器会反复 OOM 崩(exit 137)。看到 `primus-training` 容器或某 PID `GPU(s)=8` + 各卡 VRAM 96% → **立刻换**，不可共存。
- **能共存 vs 不能**：用 `rocm-smi --showmemuse`（per-GPU VRAM%）+ `--showpids`（CU OCCUPANCY 列）。VRAM 28% 且 CU OCCUPANCY=0（idle-waiting worker）→ 小 GEMM 可共存、timing 干净；VRAM 96% 或 CU 在真算 → 不行。
- **本机 ssh 解析不了 chiXXXX**，必须经 login_node2 跳：`ssh login_node2 "ssh chiXXXX '...'"`。批量扫用 `for n in ...; do ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new $n '...'; done`。
- **污染环境的 bench 数不可信**：被 primus-training 抢卡时测的 TFLOPS 偏低（结构性的 V/spill/det 来自编译期+正确性，仍可信；但绝对 TFLOPS 必须在干净节点重测）。
- 镜像 `rocm/primus:v26.2` 在大多数节点都有；user 说不强求该镜像时，任何能跑的镜像都行,但 v26.2 最省事(rr.sh/skill 都按它)。

### 3. 把公钥种到选中的节点

```bash
PUBKEY='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDNk0/FA7DjtcE50WfO1Xf+qGUARLegsXwliz4TU0sPk kyle-20260509'
TARGET=chiXXXX
ssh login_node2 "ssh -o StrictHostKeyChecking=accept-new $TARGET 'mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && grep -qF \"DNk0/FA7DjtcE50WfO1Xf+qGUARLegsXwliz4TU0sPk\" ~/.ssh/authorized_keys || echo \"$PUBKEY\" >> ~/.ssh/authorized_keys; echo done; tail -1 ~/.ssh/authorized_keys'"
```

`grep -qF || echo >>` 这个组合**带去重** —— 重复跑不会写两遍。

立即测直连：

```bash
# 注意：本机直连 `ssh chiXXXX` 通常 **解析不了 hostname**（chi 节点名只在 login_node2 内网可见）。
# 要么经跳板机测：
ssh login_node2 'ssh -o ConnectTimeout=15 chiXXXX "hostname && whoami"'
# 要么先做下面的 config 更新，再用别名测：ssh compute_node_new 'hostname'
```

> 节点的 single source of truth：本机走 `sync/.ssh-chi.sh root@chiXXXX`（脚本通用，host 作参数；只硬编跳板机 149.28.124.225），没有 `~/.ssh/config` 别名。**当前 = `chi2811`（2026-07-03，从 chi2810 换出）**。历史：chi2810（2026-06-30，从 chi2774 迁出）→ chi2774（2026-06-22 从 chi2811）→ chi2811（2026-06-11 从 chi2832）。**永远避开 yanyuqin 的 hold**。
>
> **2026-06-30 换节点踩坑**：集群多数 mi355x `down*`(维护);`alloc` 节点(chi2761/2762/2811)被别人 `yambda` 容器占 ~200G VRAM(idle-waiting,GPU-pid=0,但显存锁住) 且 **docker 盘满**(13-27G free,装不下 87.4G 镜像,满盘全是别人镜像不能删);**chi2761 根本路由不到 aac NFS 服务器(10.2.123.177 No route to host)**→ 没代码没 tar,弃。当时可用的是 `down` 但实际空的 chi2810/chi2879(都挂 aac、GPU 全空)。**清 docker 盘只清"未被任何容器引用"的镜像**:`docker image prune -a -f` + `docker builder prune -f`(不碰运行/停止容器及其镜像) → chi2810 由此腾出 ~107G。
>
> **2026-07-03 chi2810→chi2811 换节点**：chi2810 被别人两个 SGLang 服务(`rail-probe`/`xb_sglang`)8 卡吃满(100% use)，`sinfo`/`squeue` 对这类 slurm 外直接 `docker run` 抢卡完全看不出来——**唯一可信的还是直接 ssh 进节点跑 `rocm-smi --showuse` + `--showpids`**。批量扫描 `chi2761/2762/2811/2866/2879` 后，`chi2811`(sinfo 显示 `resv`)实测 **8 卡 0% use、每卡仅 ~298MB 底噪、0 pid**——比 sinfo 标 `idle` 的 chi2879 更可信（chi2879 实际 VRAM 分配 95%，被 maxtext-slurm 多节点训练任务锁着，是假空闲）。chi2811 已挂 `/mnt/vast/kyle/code2`（同源可见，含所有历史代码/tar），root 盘 131G 空闲。**教训**：sinfo 的 `idle`/`resv`/`down` 标签和真实 GPU 占用完全对不上号，必须每次都用 `rocm-smi` 三件套（use%/pid 数/VRAM%）实测，不能信任一个指标。

### 4. 起 mlperf_gptoss 容器

#### ★ 首选：从保存的镜像恢复（跳过 §5/§5.5，开箱即用）

已把配好环境的容器（triton 3.7 + flydsl/primus_turbo editable + 全部依赖）commit 并导出到集群共享盘。
**任何节点直接 load + run 即可，不用再升 triton、不用重装 flydsl/primus_turbo。**

- **最新保存镜像 tar（2026-07-03，用这个）**：`/mnt/vast/kyle/code2/docker_images/mlperf_gptoss-20260703.tar.zst`（20.3G 压缩 / **93G 解压**，triton3.7 + flydsl/primus_turbo(含 tensorwise venv `/opt/venv-tw`，含 wgrad 3buf 融合收尾实验改动)，`/mnt/vast`=aac 共享盘每节点同源可见）
- 镜像 tag：`mlperf_gptoss:saved-20260703`（旧 `saved-20260625b` 已从 chi2810 删除，别的节点上若还有别用）
- ⚠️ **解压 93G**：load 前确认 `df -h /var/lib/docker` 有 **>100G 空闲**。不够就 `docker image prune -a -f`(只删未被任何容器引用的镜像,不碰别人在跑的容器)。
- ⚠️ **2026-07-03 实测：`docker commit`+`save` 对根盘的实际压力比想象中大**（chi2810 上从 7GB 空闲一路挤到 0），**在空闲节点上 load 一个已有 tar 通常没这个风险**（load 不需要 commit 那步的 diff 层），但 `df -h /` 提前查一下总是好的。

```bash
TARGET=chiXXXX
# (a) 该节点若还没有这个镜像，先 load（~93GB 解压，几分钟；docker 盘需 >100G 空闲，先 df -h /var/lib/docker）
ssh $TARGET 'docker images | grep -q "mlperf_gptoss.*saved-20260703" \
  || zstd -dc /mnt/vast/kyle/code2/docker_images/mlperf_gptoss-20260703.tar.zst | docker load'
# (b) 用保存的镜像起容器（flag 与下方固定版本一致，只把 image 换成保存的 tag）
ssh $TARGET 'docker run -d \
  --name=mlperf_gptoss \
  --network=host --ipc=host \
  --device /dev/dri --device /dev/kfd --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size=64G \
  -v /mnt/vast/kyle/code2:/workspace/code \
  mlperf_gptoss:saved-20260703 sleep infinity'
```

- editable 安装（flydsl/primus_turbo）的 python 指针在镜像 rootfs 里，源码 + build 产物（`build-fly/`、`_C*.so`）在 bind mount `/mnt/vast/kyle/code2` 上，两者一拼就直接 `import flydsl` / `import primus_turbo.pytorch` 可用 → **§5、§5.5 全跳过**，直接去 §6 验证。
- 验证里若 `import primus_turbo.pytorch` 仍报缺符号/版本不匹配（极少见，通常是 bind mount 里的 `_C*.so` 被别的分支覆盖），才回退去跑 §5.5 (b) 重 build。
- 镜像更新了（装了新依赖/升级了包）→ 重新 commit + save 覆盖这个 tar，并把本文件里的日期/文件名同步更新。重 commit 命令见本文件末尾「重建保存镜像」。

#### fallback：从 fresh `rocm/primus:v26.2` 起（没有保存镜像时才用）

固定命令（用户审定的版本，别擅自改 flag）：

```bash
ssh chiXXXX 'docker run -d \
  --name=mlperf_gptoss \
  --network=host \
  --ipc=host \
  --device /dev/dri \
  --device /dev/kfd \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size=64G \
  -v /mnt/vast/kyle/code2:/workspace/code \
  rocm/primus:v26.2 sleep infinity'
```

要点：
- `--name=mlperf_gptoss` 沿用原名 —— `remote-mlperf-gptoss` skill 里所有 `docker exec mlperf_gptoss ...` 不用改。
- `-v /mnt/vast/kyle/code2:/workspace/code` 是 bind mount —— `/mnt/vast` 是集群共享 NFS，所有节点同源，换节点数据自动可见。
- `--network=host` —— 容器直接用 host 网络栈（远程没外网这点不变，但同节点其他服务的端口能直连）。
- `sleep infinity` —— 容器常驻，靠 `docker exec` 进去干活，不要用 `docker run -it` 跑业务。
- 走 fallback 路径才需要继续做 §5（升 triton）+ §5.5（装 flydsl/primus_turbo）。

### 5. 升级 triton 到 3.7（走 fresh fallback 才需要；从保存镜像恢复可跳过）

`rocm/primus:v26.2` 镜像里 triton 默认 3.6.0；user 要求 fp8 grouped GEMM bench 走 triton 3.7（autotune cfg 更全）。每次新起容器都要做一次：

```bash
ssh chiXXXX 'docker exec mlperf_gptoss bash -lc "/opt/venv/bin/pip install --upgrade triton 2>&1 | tail -2 && /opt/venv/bin/python -c \"import triton; print(triton.__version__)\""'
```

预期最后一行输出 `3.7.0`。装完后任何之前用 triton 3.6 跑的 bench 数字（`v2/Triton` ratio 等）都作废，必须重 bench 立新 baseline。

### 5.5 从源码安装 FlyDSL + 重装 primus_turbo（走 fresh fallback 才需要；从保存镜像恢复可跳过）

> 注：下方命令里的 `/workspace/code/Primus-Turbo` 是**远程节点容器内**的路径（bind mount 自 `/mnt/vast/kyle/code2/Primus-Turbo`）。本机查看 Primus-Turbo 源码请用本地副本 `/workspace/code/gpt_oss_docker/sync/Primus-Turbo`（远程命令路径保持原样，勿改）。

fresh 容器里 `flydsl` **没装** → `from primus_turbo.flydsl.gemm.gemm_fp8_kernel import _compile_dense_tn`
落 stub 分支报 `ImportError: cannot import name '_compile_dense_tn'`（`flydsl_available()` False）。
**正解是从源码 editable 安装，不要用 PYTHONPATH hack**（PYTHONPATH 只够 `import flydsl`，编译 kernel 还要 `_mlir` 的 LD_LIBRARY_PATH，editable install 会自动 symlink 解决）。

```bash
# (a) 装 FlyDSL —— build-fly/ 已存在则只 symlink，~30s，不会重 build MLIR
ssh compute_node_new 'docker exec mlperf_gptoss bash -lc "cd /workspace/code/FlyDSL && /opt/venv/bin/pip install -e . 2>&1 | tail -3"'
# (b) 重装 primus_turbo（rebuild cpp ext 匹配当前源，~15-25min）
ssh compute_node_new 'docker exec mlperf_gptoss bash -lc "cd /workspace/code/Primus-Turbo && GPU_ARCHS=gfx950 /opt/venv/bin/pip install --no-build-isolation -e . 2>&1 | tail -5"'
# 验证（都应成功，big-N 走纯 flydsl）
ssh compute_node_new 'docker exec mlperf_gptoss bash -lc "cd /tmp && python -c \"import flydsl, torch; import primus_turbo.pytorch as t; from primus_turbo.flydsl.gemm.gemm_fp8_kernel import flydsl_available; print(flydsl_available())\""'
```

装完 `import flydsl` / `import primus_turbo.pytorch` 都不再需要 PYTHONPATH。改 flydsl kernel 后 `rm -rf /root/.flydsl/cache` 再跑（改 cpp 才要重跑 (b)）。

**⚠️ 若 (b) 后 `import primus_turbo.pytorch` 报 `undefined symbol: ...hk_gemm_bf16...`**：
这是 untracked 的 HipKittens dense binding 引用了一个**不存在**的 kernel `.cu`
(`csrc/kernels/gemm/HipKittens/hk_gemm_gfx950.cu`)。HK 后端已于 2026-06-01 **整体移除**
（dense+grouped 全是 untracked WIP，无 python 引用）。若残留导致 build 失败，删掉
`csrc/pytorch/gemm/hk_gemm_hip.cpp`、`csrc/pytorch/grouped_gemm/hk_grouped_gemm_hip*.cpp`、
`csrc/kernels/grouped_gemm/HipKittens/`，并从 `csrc/pytorch/bindings_pytorch_hip.cpp` +
`extensions_hip.h` 去掉所有 `hk_gemm*`/`hk_grouped*` 的 def/impl/声明，nuke `build/` + `_C*.so` 重跑 (b)。

### 6. 验证

```bash
ssh chiXXXX 'docker ps --filter name=mlperf_gptoss && \
  docker exec mlperf_gptoss bash -lc "hostname && pwd && ls /workspace/code | head && python --version && python -c \"import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())\""'
```

预期输出：
- `Up X seconds`
- hostname = chiXXXX（容器内外一致，host network）
- `/workspace/code` 下能看到 `Primus-Turbo / HipKittens / code2 / triton`
- `python --version` → `Python 3.12.3`
- `torch ... True 8` —— 8 张 MI355X 全可见

任一项失败 → **不要**自己改 docker flag，先停下来报告给用户。

## 收尾

成功后做这几件事：

1. **更新 `~/.ssh/config`**：把 `compute_node_new` 的 `HostName` 改成新节点（这就是 canonical 节点的 single source of truth，**别在 SKILL.md 里再硬编节点名**）。让旧的 `ssh compute_node_new ...` 命令继续生效。
2. **更新 `../remote-mlperf-gptoss/SKILL.md`**：第一段拓扑图的旧节点名和"远程环境"表的"主机名"行换成新节点。日期也更新到当天。
3. **如果旧节点的容器是我们起的**：`ssh oldNODE 'docker rm -f mlperf_gptoss'` 清掉，别留垃圾。如果是别人的容器（如 `vllm-gptoss`），**不要碰**。
4. **`pip install -e .` primus_turbo**：新容器没装，bench 之前要 `cd /workspace/code/Primus-Turbo && GPU_ARCHS=gfx950 pip install --no-build-isolation -e .`，长（~15-30 min），go-grab-coffee 级别。
5. **告诉用户**：新节点 hostname、镜像、GPU 数、bind mount 路径都正常 + triton 版本 3.7。提示一句"这节点是某 slurm job 的，他可能随时上来跑，长任务前再 `rocm-smi` 复查一下"。

## 不要做的事

- **不要**用 `docker run` 跑用户没审过的 flag 组合（特别是 GPU isolation flag —— 默认 `--device /dev/dri --device /dev/kfd` 是全 8 卡可见）。
- **不要**在 idle 节点上起容器 *并立刻* 跑满 8 卡的工作负载 —— 节点主人(slurm job 拥有者)是优先权，自己先小任务跑通再慢慢上量。
- **不要**清理别人的容器、停别人的进程、删别人的镜像。即使 GPU 是空的，节点不是我们的。
- **不要**用 `srun` / `salloc` 抢 mi355x —— 全节点已 ALLOCATED，slurm 不会让我们进；正确路径就是 docker。
- **不要**修改/删除 chi2811（旧节点）上的别人遗留容器(`vllm-gptoss` 等)。

## 重建保存镜像（环境变了就刷新这个 tar）

当容器里装了新依赖、升级了包、或想把一段时间的环境固化，重新 commit + 导出覆盖共享盘上的 tar。
源码/build 产物在 bind mount 上不进镜像，**只固化容器 rootfs**（pip 包、apt 包、editable 指针、triton 版本）。

```bash
DATE=$(date +%Y%m%d)   # 本机算好日期填进去，远程别依赖
# (a) commit 当前容器（--pause=false 不打断容器里在跑的活）
ssh compute_node_new "docker commit --pause=false mlperf_gptoss mlperf_gptoss:saved-$DATE"
# (b) 导出 + zstd 并行压缩到共享盘（~70GB，128 线程，几分钟；后台跑）
ssh compute_node_new "bash -c 'set -o pipefail; cd /mnt/vast/kyle/code2/docker_images && \
  docker save mlperf_gptoss:saved-$DATE | zstd -T0 -3 -q -o mlperf_gptoss-$DATE.tar.zst && \
  ls -lh mlperf_gptoss-$DATE.tar.zst'"
```

刷新后：把上面 §4「首选」里的 tar 文件名 + tag 日期同步改掉；旧 tar 确认新的能 load 后再删，省共享盘空间。
注意 commit 不保存 bind mount（`/workspace/code` 即 `/mnt/vast/kyle/code2`）—— 那本来就持久在共享盘。

## 故障排查

| 症状 | 排查 |
|---|---|
| `ssh chiXXXX` Permission denied | 公钥没种成，回到步骤 3。检查 `tail -1 ~/.ssh/authorized_keys` 的输出是不是我们的 key。 |
| 从保存镜像恢复后 `import flydsl` 失败 | bind mount 没挂上或 `/workspace/code/FlyDSL` 不在（editable 指针指向它）。先 §6 验证 `ls /workspace/code`；真缺才回退 §5.5。 |
| `docker load` 报 no space | 节点 `/` 盘满（镜像解压占 ~70GB）。`docker system df` + `docker image prune`，或换盘大的节点。 |
| 步骤 3 经 login_node2 也卡在 password prompt | login_node2 → 该节点的免密信任不存在（罕见）。`ssh login_node2 'ssh -v chiXXXX hostname' 2>&1 | grep -E "Authentications|Accepted"` 看用了哪种方法。 |
| `docker run` 报 name conflict | 已经有同名容器残留：`ssh chiXXXX 'docker rm -f mlperf_gptoss'`，再重跑。 |
| 容器起来但 `torch.cuda.device_count()=0` | `--device /dev/kfd` 或 `--device /dev/dri` 没挂上。`docker inspect mlperf_gptoss | grep -A2 Devices` 检查。 |
| `/workspace/code` 是空的 | bind mount 路径错了或该节点 `/mnt/vast` 没挂。本应是 host `/mnt/vast/kyle/code2`（`/mnt/shared` 已弃用，常是坏 NFS）。 |
| `docker images` 里没有 `rocm/primus:v26.2` | 这台节点没拉过镜像。从一个有镜像的节点 `docker save rocm/primus:v26.2 | ssh root@target 'docker load'`，比走外网（没有）靠谱。 |
