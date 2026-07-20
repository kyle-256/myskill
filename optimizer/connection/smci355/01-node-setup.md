# smci355 SLURM 节点：登录 / 起容器 / 两套 turbo 环境 / 跑 meta-attn

> 类别: 连接 · 主题标签: smci355, slurm, sbatch, docker, kyle_dev, flydsl-0.2.2, venv-tw, mxfp4, tensorwise, meta-attn, rocm-primus-v26.3
>
> 本卡是 gpt_oss_docker 项目迁到 **smci355 SLURM 集群** 后的从零环境搭建规程（2026-07-20 实测）。
> 换节点时只替换 `connection/` 这一层；`methodology/`、`pitfalls/` 通用。

## 🚨 环境边界
- **你的**：容器 **`kyle_dev`**、host 盘 **`/home/xianzhao/code_from_juicefs_20260623_0233`**（→ 容器 `/workspace/code`）、venv **`/opt/venv`**(mxfp4) / **`/opt/venv-tw`**(tensorwise)。
- **严禁碰**：`vidgoyal` 的容器 `primus_v26.5rc_rocm7.14`、盘 `/home/vidgoyal`、`/apps/gpuperf/vidgoyal`（其他用户，无写权限，别 exec/删改）。

## 0. 为什么迁到这里
- 旧节点 **chi2774（经跳板机 149.28.124.225）整条链路挂掉**（jump host `connect ... port 22: Connection timed out`）。基础设施故障，非代码问题。
- 用户提供新入口 `xianzhao@smci355-ccs-aus-n01-29.prov.aus.ccs.cpe.ice.amd.com`，直连（无跳板），MI355X / gfx950。

## 1. 登录（关键：用 .ssh_laptop key，不是 .ssh_docker）
```bash
SSHK="ssh -i /workspace/code/.ssh_laptop/id_ed25519 -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=no -o ConnectTimeout=25 \
  xianzhao@smci355-ccs-aus-n01-29.prov.aus.ccs.cpe.ice.amd.com"
$SSHK "hostname"
```
- ⚠️ **key 是 `/workspace/code/.ssh_laptop/id_ed25519`**（不是 gpt_oss 惯用的 `.ssh_docker`——那个是 chi 跳板 key，此节点认不了 → `Permission denied`）。
- ⚠️ 必须 `-o IdentitiesOnly=yes`，否则 agent 里多把 key 轮流试 → `Too many authentication failures` 被断开。

## 2. SLURM 申请节点（要选对 partition）
```bash
$SSHK "sbatch --partition=Compute-DCPT --time=24:00:00 --gpus=8 --nodes=1 \
  --job-name=primus-turbo --wrap='tail -f /dev/null'"
$SSHK "squeue --me"   # 看分到哪个节点（本例 job 落回 login 节点 smci355-ccs-aus-n01-29 本身）
```
- ⚠️ **partition 必须是 `Compute-DCPT`**（我在 `dcpt_users` 组）。默认/`Compute-Group01` 会 PD 挂起：`uid ... not in group permitted ... groups allowed: dlc_..._mori_users`。用 `groups` 看自己在哪些组，`sinfo -o '%P %a %l %D %t'` 看 partition 状态（找 `idle`）。
- job 起来后容器就在该节点上，直接 `docker` 干活（此集群用 docker，不是 crusoe 的 spur）。

## 3. 起自己的容器 kyle_dev（固定 flag）
```bash
$SSHK "docker run -d --name=kyle_dev --network=host --ipc=host \
  --device /dev/dri --device /dev/kfd --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size=64G \
  -v /home/xianzhao/code_from_juicefs_20260623_0233:/workspace/code \
  rocm/primus:v26.3 sleep infinity"
```
- 镜像 **`rocm/primus:v26.3`**（本地已有；`docker images | grep primus:v26`）。
- 验证：`docker exec kyle_dev bash -lc 'python -c "import torch;print(torch.cuda.device_count())"'` → **8**。
- 代码目录 `/home/xianzhao/code_from_juicefs_20260623_0233/gpt_oss_docker`（含 `sync/{FlyDSL,meta-attn,aiter,Primus,mxfp4,tensorwise}`）。

## 4. FlyDSL：pip 装 0.2.2（⭐ 关键，别 build MLIR）
```bash
$SSHK "docker exec kyle_dev bash -lc 'export PATH=/opt/venv/bin:\$PATH; pip install flydsl==0.2.2'"
# 验证有 expr.math（meta-attn 依赖）：
$SSHK "docker exec kyle_dev bash -lc 'python -c \"from flydsl.expr import math; print(\\\"OK\\\", math.__file__)\"'"
```
- ⚠️ **rocm/primus:v26.3 自带的 flydsl 是 `0.1.1.dev409` egg，缺 `flydsl.expr.math`** → meta-attn `from flydsl.expr import math as fmath` 直接 `ImportError`。**必须 `pip install flydsl==0.2.2`**（有 wheel，含 expr.math；dq dual-wave 也用 0.2.2）。
- `amd-aiter ... requires flydsl==0.1.1.dev409` 的警告**无害**（meta-attn 直接用 flydsl，不走 aiter）。

## 5. meta-attn（dkdv/attention）—— 只需 flydsl 0.2.2，不用 turbo build
```bash
MA=/workspace/code/gpt_oss_docker/sync/meta-attn
$SSHK "docker exec -e HIP_VISIBLE_DEVICES=0 -e FLYDSL_EXTRA_SOURCE_DIRS=$MA/meta_aiter_attn/flydsl kyle_dev \
  bash -lc 'rm -rf /root/.flydsl/cache; cd $MA && export PATH=/opt/venv/bin:\$PATH && \
  FAST=0 python meta_aiter_attn/flydsl/_bench_dkdv.py'"
```
- ⚠️ 测 bare-asm / hw-exp 正确性**必须 `FAST=0`**（fast_exp2=True 把 lse 预缩放成 ~10⁹ → hw-exp2 得 inf → nan；见 [[project_dkdv_hwexp_overlap]]）。
- 改 kernel 后命令里必 `rm -rf /root/.flydsl/cache`（JIT 缓存）。
- meta-attn **不需要** Primus-Turbo 的 csrc build；flydsl 0.2.2 单独就跑。

## 6. 两套 turbo 环境（mxfp4 + tensorwise，venv 隔离）
两份 checkout（都在 `sync/` 下），各自的 branch + venv：

| 环境 | repo | branch(本节点当前) | venv | 备注 |
|---|---|---|---|---|
| mxfp4 | `sync/mxfp4/Primus-Turbo` | `dev/kyle/dsv4-bwd-opt` | `/opt/venv` | CK submodule 已填充 |
| tensorwise | `sync/tensorwise/Primus-Turbo` | `dev/kyle/flydsl_mxfp4_grouped` | `/opt/venv-tw` | CK 曾空，从 mxfp4 拷 |

### 6a. mxfp4（用 /opt/venv）
```bash
$SSHK "docker exec kyle_dev bash -lc 'export PATH=/opt/venv/bin:\$PATH; git config --global --add safe.directory \"*\"; \
  cd /workspace/code/gpt_oss_docker/sync/mxfp4/Primus-Turbo && \
  GPU_ARCHS=gfx950 pip install --no-build-isolation -e . > /tmp/mxfp4_build.log 2>&1'"
```
- csrc build ~15-25min，产物 `primus_turbo/lib/libprimus_turbo_kernels.so`。
- 长 build 用 `nohup bash -c '...' >/tmp/x.log 2>&1 &` 后台跑，`until grep -qE "Successfully installed|ERROR|Traceback" /tmp/x.log` 轮询等完成。

### 6b. venv-tw（cp clone /opt/venv，修 finder）
```bash
$SSHK "docker exec kyle_dev bash -lc '
  [ -d /opt/venv-tw ] || cp -a /opt/venv /opt/venv-tw
  # 消 setuptools.pth 里对源 venv 的引用（否则 fallback 污染，非真隔离）
  grep -rl \"/opt/venv/\" /opt/venv-tw/lib/python3.12/site-packages/*.pth | while read f; do sed -i \"s#/opt/venv/#/opt/venv-tw/#g\" \"\$f\"; done
  /opt/venv-tw/bin/python -c \"import sys; print([p for p in sys.path if p.startswith(\\\"/opt/venv/\\\")])\"  # 应为 []
'"
```

### 6c. tensorwise（用 /opt/venv-tw）
```bash
# CK submodule 若空：从 mxfp4 拷（同 CK；免走网络 submodule）
$SSHK "docker exec kyle_dev bash -lc 'cd /workspace/code/gpt_oss_docker/sync/tensorwise/Primus-Turbo; \
  ls 3rdparty/composable_kernel/include >/dev/null 2>&1 || cp -a ../../mxfp4/Primus-Turbo/3rdparty/composable_kernel/. 3rdparty/composable_kernel/'"
# build（等 mxfp4 build 完再起，避免双 ninja 抢 CPU / LLVM OOM）
$SSHK "docker exec kyle_dev bash -lc 'export PATH=/opt/venv-tw/bin:\$PATH; \
  cd /workspace/code/gpt_oss_docker/sync/tensorwise/Primus-Turbo && \
  GPU_ARCHS=gfx950 pip install --no-build-isolation -e . > /tmp/tw_build.log 2>&1'"
```
- ⚠️ **两个 build 别并行**（ninja CPU 满 + LLVM 易 OOM）→ 串行。
- 改 csrc 后必 **clean 重编**（`rm -rf build primus_turbo/lib/*.so`），增量 `pip install -e .` 不真重编（旧 mtime/hipify 中间件骗过 ninja）。

## 7. 从 agent sandbox 同步代码（rsync，用 .ssh_laptop key）
```bash
cd /workspace/code/gpt_oss_docker/sync/meta-attn/meta_aiter_attn/flydsl
rsync -azh --no-o --no-g \
  -e 'ssh -i /workspace/code/.ssh_laptop/id_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no' \
  flash_attn_bwd_rect16_kernel.py \
  xianzhao@smci355-ccs-aus-n01-29.prov.aus.ccs.cpe.ice.amd.com:/home/xianzhao/code_from_juicefs_20260623_0233/gpt_oss_docker/sync/meta-attn/meta_aiter_attn/flydsl/
```
- ❌ 别加 `--delete`（会删远端 build 产物/submodule 实体）；排除规则用 `/build/` 锚定 repo 根。

## 8. 常见坑速查
| 症状 | 原因 | 修 |
|---|---|---|
| 跳板 `port 22 timed out` | chi 老链路挂了 | 用本卡的 smci355 直连（.ssh_laptop key） |
| `Permission denied` 登录 | 用错 key（.ssh_docker） | 换 `.ssh_laptop/id_ed25519` |
| `Too many authentication failures` | agent 多 key 轮试 | 加 `-o IdentitiesOnly=yes` |
| sbatch PD `not in group permitted` | partition 权限 | `--partition=Compute-DCPT`（`groups` 核对） |
| `ImportError: cannot import name 'math' from flydsl.expr` | 镜像自带 flydsl dev409 缺 expr.math | `pip install flydsl==0.2.2` |
| meta-attn 跑出 nan | 用了 fast_exp2 harness 测 hw-exp | `FAST=0` |
| `docker exec ... > file` 写不进容器 | 重定向落 host | 在 `bash -lc '... > file'` 内部重定向 |

---
来源: 本 session (2026-07-20) 迁移到 smci355 实测；沿用 crusoe/01 与 gpt_oss/01-02 的 flydsl-0.2.2 / venv 隔离 / 两套 turbo 规程。
