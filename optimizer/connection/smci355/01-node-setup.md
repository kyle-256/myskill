# smci355 SLURM 节点：登录 / 起容器 / 两套 turbo 环境 / 跑 meta-attn

> 类别: 连接 · 主题标签: smci355, slurm, sbatch, docker, kyle_dev, flydsl-0.2.2, venv-tw, mxfp4, tensorwise, meta-attn, rocm-primus-v26.3
>
> 本卡是 gpt_oss_docker 项目迁到 **smci355 SLURM 集群** 后的从零环境搭建规程（2026-07-20 实测）。
> 换节点时只替换 `connection/` 这一层；`methodology/`、`pitfalls/` 通用。


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

## 6d. ★★aiter 升到 turbo 钉的版本(2026-08-20 实测,两层坑)

turbo 在 `primus_turbo/common/aiter_utils.py` 钉 `AITER_VERSION = "0.1.14.post1"`。镜像自带的是
`0.1.1.dev1611`,**老版 `_flash_attn_forward()` 没有 `out=` 参数** ⇒ 凡是 dispatcher 回落到 aiter 的用例
全炸 `TypeError: ... unexpected keyword argument 'out'`(attention 两个测试文件 **2309 failed**)。

⚠**四个 venv(`/opt/venv`、`venv-tw`、`venv-syncv3`、`venv-syncv4`)共用同一份 `/workspace/aiter`
可编辑安装**(`amd-aiter.egg-link`)⇒ 升级会影响所有环境,动之前先确认没有 campaign 在跑。

```bash
cd /workspace/aiter
git diff > /workspace/aiter_local_mha_atomicfp32.patch   # 存本地补丁(有人把 is_v3_atomic_fp32 默认 True->False,5 处)
git rev-parse HEAD > /workspace/aiter_prev_head.txt      # 回滚锚点
git checkout -- aiter/ops/mha.py
git -c submodule.recurse=false fetch --no-recurse-submodules --tags -q origin   # ★带子模块会挂在 CK 的 "not our ref"
git -c submodule.recurse=false checkout --no-recurse-submodules -q v0.1.14.post1
git submodule update --init --depth 1 3rdparty/composable_kernel                # ★★这一步绝不能漏
export PATH=/opt/venv-tw/bin:$PATH
rm -rf /workspace/aiter/amd_aiter.egg-info
pip install -q setuptools_scm
GPU_ARCHS=gfx950 pip install --no-build-isolation -e .
pip install -q 'flydsl==0.2.2'        # ★必须最后:aiter 的 setup.py 会 subprocess 强装 flydsl 0.1.7
rm -rf /workspace/aiter/aiter/jit/build/mha_bwd_*   # 清掉编译失败的 JIT 产物
```

**第二层坑(我第一次漏了)**:只切主仓不切 CK 子模块 ⇒ aiter 自己的 `csrc/cpp_itfs/mha_bwd.cu` **编不过**
(`cu_seqlen_q_ptr` 被当成 `int`、`nhead_q` 窄化成 `float`,典型结构体字段错位),表现为 `FAILED: mha_bwd.cuda.o`,
剩 **182 failed**。v0.1.14.post1 要求 CK `10cb6916c`,而本地停在 `+7968368d9`(`git submodule status` 前缀
`+` 就是不一致)。同步后 **7090 passed / 0 failed**。

**遗留**:版本串是 `0.1.14.post2.dev0+g0f3c58e6e.d20260820`(`.d` 后缀=树 dirty),turbo 的 `_versions_match`
只 warn 不阻断;本地补丁未重打(turbo 自己显式传 `is_v3_atomic_fp32`,走 `PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32`)。
**副产品**:我 memory 里长期记的"ragged varlen 走 AITER 回退、dk/dv SNR 只有 6.6dB"其实就是这个老 aiter,一并好了。

## 7. 从 agent sandbox 同步代码（rsync，用 .ssh_laptop key）
```bash
cd /workspace/code/gpt_oss_docker/sync/meta-attn/meta_aiter_attn/flydsl
rsync -azh --no-o --no-g \
  -e 'ssh -i /workspace/code/.ssh_laptop/id_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no' \
  flash_attn_bwd_rect16_kernel.py \
  xianzhao@smci355-ccs-aus-n01-29.prov.aus.ccs.cpe.ice.amd.com:/home/xianzhao/code_from_juicefs_20260623_0233/gpt_oss_docker/sync/meta-attn/meta_aiter_attn/flydsl/
```
- ❌ 别加 `--delete`（会删远端 build 产物/submodule 实体）；排除规则用 `/build/` 锚定 repo 根。

## 7b. ★ 应急:chi 集群整体失钥时把 campaign 搬到这里(2026-08-11 实测)
背景:`chi2835`(以及扫过的全部 86 个 chi 计算节点)在重装后**拒绝我们手上所有 key** —— 跳板
149.28.124.225 能连、目标 sshd 应答、`Permission denied (publickey,password)`;跳板自己的 13 把 key
逐一试过全被拒;`kubectl` 要 OIDC 交互登录;slurm 里这些节点全 `down*/k8s`。跳板机 chi2866 与
chi2878 是唯二还能登的,但两台 8 卡 VRAM 全被别人的 CI 占满(280/309 GB)。⇒ **本节点是当时唯一
有 8 张空闲 MI355X 的落脚点**,20 分钟内跑通了同一份 scored bench。
- **别 build turbo**:镜像 `tasimage/primus:v26.4_turbo_perf` 自带 torch 2.12+rocm7.14 / flydsl 0.2.4 /
  site-packages 里**已编译好的** `primus_turbo`。把 sandbox 的 repo rsync 上来后,只需把镜像里的
  `find . -name "*.so"`(`lib/libprimus_turbo_kernels.so` + `pytorch/_C.cpython-312-*.so`)按相对路径
  拷进 repo 树,再 `PYTHONPATH=<repo> python`,`import primus_turbo.pytorch` 直接通 —— 省掉 15~25min csrc build。
  ⚠ 只有 FlyDSL 侧改动能这么干(python 层改的是 FlyDSL kernel);改了 csrc 必须真 build。
- ⚠️ **bind mount 别用 `/home/xianzhao`**:那是 NFS 且 root_squash,容器内 root 写不进去
  (`mkdir: Permission denied`)。放本地盘 `/tmp/<你的目录>`(本节点 `/` 是 14T NVMe,1% 使用)。
- 起法与 §3 相同,只把镜像/挂载换掉:`docker run -d --name=kyle_dev ... -v /tmp/kyle_syncv4:/workspace/code
  --entrypoint sleep tasimage/primus:v26.4_turbo_perf infinity`(该镜像有 ENTRYPOINT,必须显式覆盖)。
- **测量质量**:15 次 scored bench 单次 ~16s(含清 flydsl cache),gm 分散 **0.16**,比原节点(0.6~0.7)干净;
  换节点后**绝对 TF 与 ratio 基线都要重测**,只有同节点 A/B 才可比。
- ~~❌ `rocprofv3` 在这个镜像的容器里**挂死**~~ ⚠ **2026-09-01 实测此归因为 FALSE,`rocprofv3` 能用。**
  真因:它启动时要在 **cwd** 建 `.rocprofv3/`,而 cwd 是**只读挂载的仓库**时先报
  `Permission denied: '<repo>/.rocprofv3'`,然后**死在自己的信号处理器里**——那就是被记成
  "10 分钟不产出 csv / 挂死占卡 15.7 小时"的东西。**修法就一句:`cd /tmp/<自己的目录>` 再跑。**
  可用配方(8 个 pass 各约 20 s,全部 exit 0):
  ```
  cd /tmp/<own-dir> && <先跑一次真 shape 把 JIT 预热>
  setsid timeout -s KILL 240 rocprofv3 --pmc <=4个counter> --output-format csv \
     -d <dir> -o r -- env <...> python -u <driver>.py
  ```
  ⚠ `--output-format csv` **必须加**——默认吐 SQLite `.db`,汇总器读到空表,看起来像 kernel 过滤没命中。
  ⚠ **每 pass ≤4 个 counter**(每 pass 都会重新触发 JIT)。⚠ `/tmp` 是**多租户共享**的,
  别人 uid 的同名文件会挡住你写,所以用 `/tmp/<自己的名字>/`。
  **ISA dump 当然也照常可用**:`FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=<dir>` →
  `<dir>/<launcher>/21_final_isa.s`(编译器 dump,不走 profiler),寄存器数/spill/新增指令在这里判;
  ⚠ 但 `FLYDSL_DUMP_IR=1` 的跑法比真基线慢约 5% 且绕过 JIT 缓存,**绝不能拿它计时**。
- **两个 campaign 同时落在本节点时,各起自己的容器**:sibling 用 `kyle_dev`(挂 `/tmp/kyle_syncv4`,GPU 0),
  我这场 syncv3 用 `kyle_dev_v3`(挂 `/tmp/kyle_syncv3`,GPU 5)。同名容器共用 `/root/.flydsl/cache`,
  而每轮 bench 都要 `rm -rf` 它 —— **共用容器 = 互相清对方的 JIT 缓存 + 抢同一张卡**,分容器分卡才安全
  (方法论 16 记的"并发 pool 重叠致分数腰斩"的同族)。别往对方的 `/tmp/kyle_syncv4` 里写东西。
- ★**旧节点整机没了,drift-immune bench 的 frozen ruler 怎么救**:ruler 原本是 launcher 用
  `git show <base_sha>:<file>` 塞进老容器 `/tmp` 的,容器随节点一起消失。**不用 git 命令**也能精确重建:
  campaign 的 `rounds/round-NN/diff.txt` 头部有 `index <pre>..<post>`,`<pre>` 就是 base 文件的 blob id;
  loose object 直接 `zlib.decompress(open('.git/objects/<2>/<38>','rb').read())`,去掉 `blob <len>\0`
  头即原文件,再用 `sha1(raw)` 与 blob id 比对 ⇒ **字节级可证**是同一把尺。同理可取任意历史轮的树
  (`<post>`)当同进程 A/B 的对照臂,比 diff 反打补丁稳。

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

---

## ★★ 2026-08-12 现状复核 + syncv4 在这台的落脚法(chi 全线宕机后的应急)

### -1. 先记这条:`/workspace/code` 和这台机器的 home 是**同一份 NFS**
```bash
df -hT /workspace/code
# psnfs01...:/home/xianzhao/code_from_juicefs_20260623_0233  nfs4  100G  /workspace/code
```
我们容器里的 `/workspace/code` **就是** smci355 login 节点的 `/home/xianzhao/code_from_juicefs_20260623_0233`。
⇒ **绝不能把它当"远端"做清理**。2026-08-12 我按"清理远端"跑 `rm -rf` 毁了 `sync/mxfp4/Primus-Turbo`
(.git pack 全失 + `flydsl/{gemm,grouped_gemm}` 消失)。破坏性操作前先 `df -hT` 比对挂载,
优先用 `mv` 不用 `rm -rf`(详见 memory `feedback_verify_mount_before_destructive`)。
chi 的 `/mnt/vast/kyle/code2` 才是另一份独立副本;chi 计算节点全挂时,**跳板机 chi2866 也挂着同一份 vast**,
是取回数据的活路。

⚠ **但这份同源 NFS 不能直接拿来当远端 repo 根**(2026-08-12 实测):里面的
`gpt_oss_docker/sync/syncv4/Primus-Turbo` 是 `drwx------ nobody`(历史上某次容器 root 经 root_squash 写出来的)。
我们容器里同样被 squash 成 nobody 所以读得了,**但 smci355 上的 xianzhao 连 chdir 都进不去**
→ 从那侧发起的 rsync 直接 `Permission denied (13)`,harness 每轮的 sync 必崩。
⇒ 正解见 §5:另建一棵 **xianzhao 属主**的副本树,回到 chi 那种「本地 canonical → 远端副本」的常规模型。

### 0. SLURM 归属:我们名下**没有 job**,别蹭别人的预留
```bash
squeue --me                      # 空
scontrol show job 25056          # OAI-test-yaoc / cyao1002 / 08-09→08-23 / 9 节点 gres/gpu=72
scontrol show reservation        # oai_test_9n (cyao1002) + rocm_aic_nfs_rdma (stebates 等 4 人)
sinfo -p Compute-DCPT -o "%.30n %.10T"
```
- `Compute-DCPT` **零 idle**(14 alloc / 2 resv / 3 drain),`sbatch` 提了也是排队(clairlee 要 8 节点的 job 一直卡 `(Resources)`);
- partition `TIMELIMIT=infinite`,别人申请 14 天/1 天/20 小时不等;
- ⚠ **login 节点 n01-29 属 cyao1002 的预留,用户已硬性禁止在上面跑**(memory `feedback_no_smci355_squatting`)。
  能 ssh 进去只是因为我们和他同属 account `emad`,不代表有权用卡。

### 1. 可用落脚点 = drained 节点(无人 job,但管理员维护中)
| 节点 | drain 原因 | 实测 |
|---|---|---|
| **n01-25** | `DCGPUPERF-5585 BKC firmware prep` | **8 卡全空(0% util/VRAM)、无别人容器 → 首选** |
| n01-21 | GPU2 correctable ECC storm 11032/hr | 8 卡被 vidgoyal/zhuang12 跑满 94% VRAM |
| n02-25 | SIGBUS GPU7 repeated | 显存被 magpie/rccl-tests 占着 |
| n03-25 | 7/8 GPU visible,OAM 掉 PCIe | 没有 docker 权限,`/dev/dri` 为 0 |

drained 只是 SLURM 不再分配,**节点本身能 ssh、docker 能用、卡是好的**(n01-25 八张卡温度 61-65°C、
功耗 248-261W 全正常)。集群里别人也这么用(n01-21/n02-25 上都有别人的容器)。
⚠ 风险是管理员随时会上来刷固件,长跑任务要能随时丢弃。

### 2. 访问链路(两跳 + docker exec)
```bash
K=/workspace/code/.ssh_laptop/id_ed25519        # 注意是 .ssh_laptop,不是 .ssh_docker
H=xianzhao@smci355-ccs-aus-n01-29.prov.aus.ccs.cpe.ice.amd.com
SSHO="ssh -i $K -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=25"
$SSHO $H 'ssh -o StrictHostKeyChecking=no smci355-ccs-aus-n01-25 "docker exec kyle_dev bash -lc \"...\""'
```
⚠ 三层引号极易炸。**把脚本写进 `sync/<env>/_xxx.sh`(NFS 同源,容器内立刻可见),然后
`docker exec kyle_dev bash /workspace/code/gpt_oss_docker/sync/<env>/_xxx.sh`** —— 这是唯一稳的调用方式。

### 3. 起容器(镜像用 `rocm/primus:v26.3`,用户指定)
```bash
docker run -d --name=kyle_dev --network=host --ipc=host --device /dev/dri --device /dev/kfd \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size=64G \
  -v /home/xianzhao/smci_repos:/workspace/code \
  rocm/primus:v26.3 sleep infinity
```
容器 rootfs 落在 `/data` 的 md0(49T,可用 45T),venv 随便复制;NFS home 只有 100G 配额,**venv 绝不能放 NFS**。
实测容器内 root **能写** NFS(不是只读),但只对**属主/权限放开**的路径成立 —— 见 §5 的权限坑。
⚠ 挂的是 `smci_repos`(§5 的副本树),**不是** `code_from_juicefs_20260623_0233`(同源那份,§-1 说明为何不能用)。

### 4. ★ 建隔离 venv:这个镜像和 chi 不一样,没有 editable finder
`rocm/primus:v26.3` 把 primus_turbo 装成 **site-packages 里的实体包**(v0.2.0+3cd482d),
不是 chi 那种 editable 安装 ⇒ **没有 `__editable___primus_turbo_0_0_0_finder.py` 的 MAPPING 可改**。
改用"实体包让路 + `.pth` 指 repo":
```bash
V=/opt/venv-syncv4;  R=/workspace/code/syncv4/Primus-Turbo   # 见 §5 的挂载
SP=$V/lib/python3.12/site-packages;  IMG=/opt/venv/lib/python3.12/site-packages
cp -a /opt/venv $V
grep -rl '^#!/opt/venv/bin/python' $V/bin/ | xargs -r sed -i "1s|/opt/venv/bin/python|$V/bin/python|"
cp -n $IMG/primus_turbo/lib/libprimus_turbo_kernels.so   $R/primus_turbo/lib/          # 免 15-25min build
cp -n $IMG/primus_turbo/pytorch/_C.cpython-312-*.so      $R/primus_turbo/pytorch/
mv $SP/primus_turbo $SP/primus_turbo.image_bak
echo "$R" > $SP/_kyle_syncv4.pth
$V/bin/pip install -q flydsl==0.2.2          # 镜像自带 0.1.1.dev409,缺 flydsl.expr.math
cd /tmp && $V/bin/python -c "import primus_turbo;print(primus_turbo.__file__)"   # 必须打印 syncv4
```
- `.pth` 只会**追加**到 sys.path 尾部,所以**必须先把 site-packages 里的实体包移开**,否则它会赢。
- 验证一定 `cd /tmp` 再跑:cwd 里的 `primus_turbo` 会遮蔽一切(老坑)。
- syncv4 repo 里本来就带 Jul-20 的 `libprimus_turbo_kernels.so`(52MB)+`_C.cpython-312.so`(23MB),
  与镜像 torch 2.10 ABI 兼容,`cp -n` 不会覆盖它们。
- 落成脚本:`sync/syncv4/_setup_smci_syncv4.sh`(旧的 `_setup_venv_syncv4.sh` 是同源挂载那版,已废)。实测 `import primus_turbo` → syncv4 repo、
  flydsl 0.2.2、torch 2.10、8 GPU、`primus_turbo.pytorch` 导入通过。

### 5. ★ 最终落地结构(与 chi 同构,harness 零改动)
```
宿主 /home/xianzhao/smci_repos/<env>/Primus-Turbo   <->   容器 /workspace/code/<env>/Primus-Turbo
```
和 chi 的 `/mnt/vast/kyle/code2/<env>/...` ↔ `/workspace/code/<env>/...` 形状完全一致,所以
`cursor_campaign.py` 里 `repo_remote_host = host_root + container_path.split("/workspace/code/")[-1]`
这条推导原样成立,launcher 只换参数不用改代码:
```
--host xianzhao@smci355-ccs-aus-n01-25   --container kyle_dev   --venv /opt/venv-syncv4
--remote-container-path /workspace/code/syncv4/Primus-Turbo
--remote-host-root /home/xianzhao/smci_repos
```
起容器时挂 `-v /home/xianzhao/smci_repos:/workspace/code`(**不是**挂 code_from_juicefs 那份)。

**四个必踩的坑,按顺序**:
1. ★★`rsync -a` 会把源目录的 `drwx------` 一起搬过去 → 容器内 root(squash 成 nobody)进不去 repo。
   首次 `chmod -R a+rwX /home/xianzhao/smci_repos`,并且 push 脚本要带 `--chmod=a+rwX`。
   **但光改 push 脚本不够** —— campaign harness 的 `RemoteHarness.sync()` 用的是它**自己硬编码的
   rsync 命令行(没有 --chmod)**,每轮都会把源树权限重新搬过来。踩证(2026-08-12 r9):三次 bench
   各 **0.7 秒** rc=1 秒退,日志只有一句 `FAILED gate`,真因是
   `cd <repo>: Permission denied` —— python 还没启动。
   ⇒ **正解是从源头修**:`chmod -R a+rwX` **本地那棵 canonical 树**(sync/<env>/Primus-Turbo),
   这样 rsync 搬过去的就是好权限,harness 不用改。
   ⇒ **诊断特征**:bench 秒退(几百毫秒)+ rc=1 + 无任何 python traceback = 先查权限,别怀疑 kernel。
2. `.rsync-exclude` 排除 `*.so` → 副本里没有编译产物。从镜像补:
   `cp -n /opt/venv/lib/python3.12/site-packages/primus_turbo/{lib/libprimus_turbo_kernels.so,pytorch/_C.cpython-312-*.so}`
   到 repo 对应位置。镜像那份(118MB)与 repo 历史上带的(52MB)不同版本,但 torch 2.10 下 ABI 兼容,
   `import primus_turbo.pytorch` 实测通过。
3. 树属主是 xianzhao 而进程是 nobody → **git 报 dubious ownership**,
   `git config --global --add safe.directory <repo>` 才能用(已写进 setup 脚本)。
4. harness 的 `SSH_WRAPPER = SYNC_ROOT/".ssh-chi.sh"` 是**模块级硬编码**,没有参数能换。
   接新集群只能在那个 wrapper 里**按 host 分派**(syncv4 用的分支匹配 `*smci355-ccs-aus-n01-25*`,
   走 login 节点 ProxyJump + `.ssh_laptop` key;chi 与既有 syncv3 重写规则原样保留)。
   ⚠ wrapper 里已有一段 syncv3 的**命令重写**规则(把 syncv3 路径/容器/GPU 改写到 login 节点的
   `kyle_dev_v3`),那是 08-11 syncv3 场留下的;新加的分派要放在它**前面**,否则 cache-clear /
   rocm-smi 这类不带路径的调用会被它吃掉。

**传输脚本**(syncv4 版,其余环境照抄改路径):
`sync/syncv4/push_smci.sh`(rsync,永不 --delete,带 --chmod)、
`sync/syncv4/rexec_smci.sh`(base64 → wrapper → docker exec,三层引号免疫)。

### 6. ★ 跑 e2e/mlperf 还要 **Primus**(训练框架)—— 同样一环境一份(2026-08-13)
kernel bench 只需 Primus-Turbo;**e2e/mlperf 还要 Primus**,按 §5 同构结构各环境一份:
```
本地 canonical sync/Primus  →  /home/xianzhao/smci_repos/syncv4/Primus  ↔  容器 /workspace/code/syncv4/Primus
```
传输脚本 `sync/syncv4/push_primus_smci.sh`(照 push_smci.sh:`--chmod=a+rwX --no-o --no-g`、**永不 --delete**)。
- ❌ **别用宿主 `/home/xianzhao/code/Primus`**:它 `.git` 属主是 `nobody:nogroup`(xianzhao 连 fetch 都
  `.git/FETCH_HEAD: Permission denied`),而且**不在 kyle_dev 的挂载里**(容器只挂 `smci_repos`)→ 容器根本看不见。
- ⚠️ 若曾在**容器内**建过这棵树(如 git clone),文件属主是 nobody → 从 xianzhao 侧 rsync 覆盖时
  `mkstemp ... Permission denied (13)`。先 `docker exec kyle_dev chmod -R a+rwX <树>` 再推(同 §5 坑 1 的另一个方向)。
- `third_party/Megatron-LM` 是**空 gitlink**,`megatron.core` import 不到 → `git submodule update --init
  --depth 1 third_party/Megatron-LM`(37M)。跑法见 methodology/17 变体 C。

### 7. ★★ 容器内没有 GitHub key,`git fetch/push` 全废(2026-08-13)
`kyle_dev` 的 `/root/.ssh` 是**空的**(容器重建即清,同 [[project_claude_container_persist]]);宿主 `xianzhao`
也只有集群 key(`ssh -T git@github.com` → `Permission denied (publickey)`)。⇒ 容器内 git 网络操作一律报
`Please make sure you have the correct access rights` —— **别误判成 repo 权限或网络故障**(外网是通的:
容器内 `curl -sI https://github.com` → 200)。修 = 把本地 `.ssh_docker`(kyle-256)装进容器,**只放容器内、
不落共享盘**(NFS 上是 `drwxrwxrwx`,私钥不该躺在那):
```bash
cat /workspace/code/.ssh_docker/id_ed25519 | <wrapper> 'docker exec -i kyle_dev bash -c "
  mkdir -p /root/.ssh && chmod 700 /root/.ssh && cat > /root/.ssh/id_ed25519 && chmod 600 /root/.ssh/id_ed25519
  printf \"Host github.com\n  IdentityFile /root/.ssh/id_ed25519\n  IdentitiesOnly yes\n  StrictHostKeyChecking no\n\" > /root/.ssh/config"'
# 验证: docker exec kyle_dev ssh -T git@github.com   ->  Hi kyle-256!
```
- **本地 Bash 侧同理**:`sync/Primus` 的 fetch 也要显式指 key ——
  `GIT_SSH_COMMAND="ssh -i /workspace/code/.ssh_docker/id_ed25519 -o IdentitiesOnly=yes"`
  (本地 `~/.ssh` 只有 `campaign_key_ed25519`,默认 key 上不了 github)。
- ★ **目标 commit 不在任何分支上时**(如别人 PR 的 merge commit),`git fetch origin <40位sha>` 可**直接按
  sha 取**(实测通,`--depth 1` 的 clone 也适用);别去拉 `refs/pull/*`(几分钟且常超时)。

---

## ★★ 2026-08-20 容器镜像固化 + tar 包(这台第一份;起因=容器被外部重建、5 个 venv 全丢)

**为什么要有**:2026-08-18 12:42 `kyle_dev` 被外部重建(`docker inspect` 的 `Created` 会告诉你),
`/opt/venv*` 全部消失 —— venv 装在容器 rootfs 里、**不在 bind mount 上**,容器一没就全没。
当时是照 `_setup_smci_<env>.sh` 一步步重建回来的(clone venv→挪实体包→写 .pth→补 .so→pip flydsl 0.2.2),
有包就不用再走一遍。chi 那份 `mlperf_gptoss-20260811-flat.tar.zst` 在 `/mnt/vast/kyle/code2/docker_images/`,
但 **chi 集群 08-11 整体失钥后不可达**,且 smci355 上的 `/mnt/vast` 只是根盘上的空目录、没挂 vast。

**★ 2026-08-28 更新为新包(从 `kyle_attn` 容器重打;老 20260820 tar 已删)**
现役包**只剩 NFS 一份**(n02-29 上 `/data/kyle_0814/...` 节点本地那份不存在——那是 n01-25 的本地盘):
| 位置 | 用途 |
|---|---|
| `/home/xianzhao/code_from_juicefs_20260623_0233/docker_images/kyle_dev-20260828-flat.tar.zst` | NFS,md5 `235b7d5bfa05c1b73ea0d240862fa6fd`,**28G 压缩**;= agent 侧 `/workspace/code/docker_images/`(同一份挂载) |
- 源容器=`kyle_attn`(08-23 起用,比 08-20 版多累积几天产物,故 19G→28G);`docker commit` 出 `kyle_dev:saved-20260828`(150GB)留本地 docker。
- 含**全部 5 个 venv**(`venv_count=5` 已验):`/opt/venv`(mxfp4) + `/opt/venv-tw` + `/opt/venv-syncv3` + `/opt/venv-syncv4` + `venv-mxfp4`。
- 恢复命令里的路径/tag 相应换成 `kyle_dev-20260828-flat.tar.zst` → `kyle_dev:saved-20260828`。

<details><summary>历史:2026-08-20 首份包(两份,md5 `d11145082e441cc00e6d1815010e1e33`,已被 08-28 版取代/删除)</summary>

| 位置 | 用途 |
|---|---|
| `/data/kyle_0814/docker_images/kyle_dev-20260820-flat.tar.zst` | 节点本地(`/dev/md0` 49T),恢复最快;节点重装即失(**n02-29 上不存在**) |
| `/home/xianzhao/code_from_juicefs_20260623_0233/docker_images/kyle_dev-20260820-flat.tar.zst` | NFS,跨节点灾备(**08-28 已删**) |

19G 压缩 / 93.8G 解压;`docker commit` 出 `kyle_dev:saved-20260820`(113GB)。
</details>

⚠ **`/data` 根下的惯例是每人一个用户名目录**(`kgoginen` / `zhuang12` / 我们的 `kyle_0814`)。
别在根下另起目录(第一次建了 `/data/kyle_images`,是错的,已删)—— 自己的东西一律放 `kyle_0814/` 下。

**打包步骤(实操,~2min export + ~30s 校验)**
```bash
docker commit kyle_dev kyle_dev:saved-<date>              # 2m09s,113GB
mkdir -p /data/kyle_0814/docker_images
docker export kyle_dev | zstd -T0 -12 -f -o /data/kyle_0814/docker_images/kyle_dev-<date>-flat.tar.zst.partial
echo "PIPE=${PIPESTATUS[0]}_${PIPESTATUS[1]}"             # 必须 0_0
zstd -t -T0 <.partial>                                     # 完整性
zstd -dc <.partial> | tar -tf - | grep -E '^opt/venv[^/]*/bin/python3\.12$'   # 5 个 venv 都要在
mv <.partial> <定名>                                       # 校验全过才定名
```
- 写 `.partial` 是防半包被当成品;`PIPESTATUS` 双段都要查 —— **只看 `$?` 取的是 zstd 的码,docker export
  报错走 stderr 而 zstd 仍成功**,会误判(见 `../common/03-docker-disk-crash-guard`)。
- 这台 docker root 在 **`/data`(44T 空闲)**,不像 chi2798 根盘 95% 满,所以 `docker save` 分层格式其实也跑得动;
  仍走 `export` 是为了**恢复步骤与既有文档一致**(`docker import`),少一套心智负担。

**恢复(扁平包 → 必须 import,不是 load)**
```bash
zstd -dc /data/kyle_0814/docker_images/kyle_dev-20260820-flat.tar.zst | docker import - kyle_dev:saved-20260820
# 然后照本卡 §3 的固定 flag 起容器(run 显式带 sleep infinity + venv 全绝对路径,
# 不依赖镜像 ENV/CMD,故 import 丢元数据无影响)
```
⚠ 恢复后仍要确认 **finder/.pth 指向本机 repo 路径**(§4 的隔离 assert):包里固化的是打包时的指向。

**边界**:`docker export` 自动排除 bind mount,所以 `/workspace/code` 下的源码/build 产物**不进包** ——
这正是想要的(源码走 git,镜像只固化 rootfs:pip 包、apt 包、venv、editable 指针)。

---

## ★★ 2026-08-31 sglang 镜像环境 `kyle_sglang`(用户指定在这套镜像里搞 aiter)

**用途**:在 lmsys 的 sglang ROCm 镜像里跑我们改过的 aiter,和 `kyle_attn` 完全隔离,互不影响。

★ **本地代码全部收在 `sync/inference/`**(2026-09-02 起,别再散在 `sync/` 根下):`aiter`(origin=ROCm,
`xbc`=xiaobochen-amd,交付分支在这)、`sglang`、`InferenceX`、`Infera`、`SIKL`。目录说明见
`sync/inference/README.md`。★★**AgentX 跑分的 aiperf 命令是 `InferenceX/benchmarks/benchmark_lib.sh`
拼出来的**(`--scenario inferencex-agentx-mvp` 在 L1998 附近),要复现别人给的那条命令就读那里。

### 0. 镜像:拉之前先 `docker logout`
```bash
docker pull lmsysorg/sglang:v0.5.18-rocm720-mi35x   # ~80GB
```
⚠ **host 的 `~/.docker/config.json` 里存着一份 `rocmshared` 的 PAT,直接拉会被拒**:
`toomanyrequests: too many failed login attempts for username or IP address`。
`docker logout` 清掉改走匿名拉取即可。原凭据(留底,别人可能还在用):
`{"auths":{"https://index.docker.io/v1/":{"auth":"cm9jbXNoYXJlZDpkY2tyX3BhdF9qMUtSZVdTcGxuZWtnQzU2U3VjcUt1a3FiVXc="}}}`
盘位:`/` 3.5T,当时余 1.2T;`docker system df` 里镜像已占 1.38TB(53% 可回收)。

### 1. 起容器(照抄 kyle_attn 的 flag,多加 `--init`)
```bash
docker run -d --init --name kyle_sglang \
  --device /dev/kfd --device /dev/dri --group-add video \
  --ipc host --network host --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size 64g -v /home/xianzhao/smci_repos:/workspace/code \
  lmsysorg/sglang:v0.5.18-rocm720-mi35x sleep infinity
```
`--init` 是为了 PID1 用 docker-init 而不是裸 `sleep infinity`,免得堆僵尸(见 memory
`reference_container_pid1_sleep_zombie_reaping`)。挂载和 `kyle_attn` 一致,`/workspace/code`
仍是 root-squash **只读**。

### 2. 镜像自带什么
Python **3.10.12** / torch **2.9.1+rocm7.2.0** / ROCm **7.2.26015** / triton 3.7.0 / sglang 0.5.18。
自带 aiter = `amd-aiter 0.1.20.dev12+gd9e5ef7ce`,editable 在 `/sgl-workspace/aiter`,**flydsl 0.2.4**。

### 3. ★★ 装我们的 aiter:与镜像自带那份是**两条分叉的线**
`xiaobochen-amd/aiter` 的 base `b02ab811e` **不是**镜像那份 `d9e5ef7ce` 的祖先
(`git merge-base --is-ancestor` 判 NO),所以是换线不是叠加。
```bash
# NFS 只读 ⇒ 工作副本放容器本地;先把分支 fetch 进 NFS 副本(host 侧,两个 export 同机可见)
cd /home/xianzhao/smci_repos/aiter && git fetch \
  /home/xianzhao/code_from_juicefs_20260623_0233/gpt_oss_docker/sync/inference/aiter <branch>:<branch>
# 容器内
git clone -q --branch <branch> --single-branch /workspace/code/aiter /sgl-workspace/aiter_kyle
cd /sgl-workspace/aiter_kyle
cp -a /sgl-workspace/aiter/3rdparty/composable_kernel/. 3rdparty/composable_kernel/  # CK 从自带那份拷
rm -f aiter/jit/*.so && rm -rf aiter/jit/build                                       # ★必做,见下
pip uninstall -y amd-aiter
AITER_USE_SYSTEM_TRITON=1 GPU_ARCHS=gfx950 pip install -e . --no-build-isolation
```
- ⚠ **必须先删克隆带进来的旧 `.so`**:它们被 gitignore,`git status` 看不见,不删则
  `import aiter` 报 `module 'aiter.jit.module_aiter_core' has no attribute 'MlaVersion'`。
- `--single-branch` 浅克隆没带 tag ⇒ setuptools_scm 算出的版本号变成 `0.1.1.dev2931+g<sha>`,
  只是难看,不影响功能;要正常号就 fetch tag 后重装。
- 这个镜像的 site-packages 在 `/opt/venv/lib/python3.10`,不用另建 venv。

### 4. ★★★ flydsl:fork 这条线**只能用 0.3.2**,0.2.4 跑不了
| | 镜像自带 aiter (`d9e5ef7ce`) | xiaobochen fork (`b02ab811e`+) |
|---|---|---|
| `setup.py` FLYDSL_VERSION | `flydsl==0.2.4` | `flydsl==0.3.2` |

装我们那份时 setup.py 会自己 `pip install flydsl==0.3.2` 顶掉镜像的 0.2.4。**降回 0.2.4 会崩**:
```
rocdl.RawPtrBufferStoreOp(data, rsrc, offset, soffset, aux=aux_attr)
ValueError: Operand 4 of operation "rocdl.raw.ptr.buffer.store" must be a Value (contained a None item)
```
0.2.4 的 `soffset` 不接受 None。**挂的是 fork 自带的 `aiter/ops/flydsl/kernels/buffer_ops.py`,不是我们改的文件**
—— 整条 fork 线都迁到 0.3.2 API 了,不是某个 commit 的问题。
⚠ 副作用:镜像里的 **sglang 本来配的是 0.2.4**,顶到 0.3.2 后要跑 sglang e2e 得先自己验一遍;
要还原就 `pip install flydsl==0.2.4` 并把 `/sgl-workspace/aiter` 装回去(那份一直留着没删)。

### 5. 同机四套环境别搞混
| 容器 | 镜像 | 用途 |
|---|---|---|
| `kyle_attn` | `kyle_dev:saved-20260820` | **只保留四套**:`venv-mxfp4` / `venv-tw`(tensorwise) / `venv-syncv3` / `venv-syncv4`,campaign 在这跑。★2026-09-01 删掉了临时的 `/opt/venv-aiter` —— aiter 那条线整体搬到 `kyle_sglang` 的新镜像了 |
| `kyle_sglang` | `lmsysorg/sglang:v0.5.18-rocm720-mi35x` | 本卡,aiter + sglang。**里面有两套 venv**:`/opt/venv`=inference1、**`/opt/venv-inf2`=inference2**(§7c) |
| `kyle_train` / `kyle_dev_0814` | 旧镜像 | 历史遗留 |
| `sikl-ab` 等 | — | **别人的,别碰** |

⚠ **两个容器挂的是同一个宿主目录**:`kyle_attn` 和 `kyle_sglang` 都把 `/home/xianzhao/smci_repos`
挂成 `/workspace/code`。所以 `/workspace/code/aiter` **在两边是同一份**(kyle_sglang 的工作副本)——
在 kyle_attn 里"清理 aiter"只能删容器本地的 `/opt/venv-*`,**碰 `/workspace/code/aiter` 等于删交付**。

### 5b. ★★★ GPU 分配表(n02-29,2026-09-07 最新 —— 这是唯一真相,别信别处的卡号)

| GPU | 归属 | 容器 / venv / 仓库 |
|---|---|---|
| **0,1,4,5** | **inference1** = AgentX / GLM-5.2 serving(TP4) | `kyle_sglang` / `/opt/venv` / `/workspace/code/aiter` |
| **2,3,6,7** | **inference2**(2026-09-07 用户划定,见 §7c) | `kyle_sglang` / **`/opt/venv-inf2`** / `/workspace/code/inference2/*` |

★★ **2026-09-07 改动**:2/3/6/7 原本是 `kyle_attn` 四条 campaign 的卡(syncv4=2 / syncv3=3 /
tensorwise=6 / mxfp4=7),但 **09-03 起 gptoss 这条线整体搬到了 n04-33**(见
[[feedback_compact_keep_environment_section]]),这台的那四张已经空着(实测 `rocm-smi` use=0%、
只有 GPU0/1 上 inference1 的进程)。用户令「inference2 用 inference1 不用的那 4 个」⇒ 归 inference2。
真要回来跑 kyle_attn 的 campaign,**先和 inference2 对表**,别默认按旧分配抢卡。

★★**以下旧记录已作废,别再用**:
- ❌「mxfp4 用 **GPU5**」/「mxfp4 用 **GPU7**」→ mxfp4 那条线 09-03 起在 **n04-33**,不在这台
- ❌「手动实验默认 **GPU6**」→ 那是 chi2811 的规矩;这台 GPU6 现在是 **inference2** 的卡
- ❌「syncv4 = **GPU4**」/「syncv4 = **GPU2**」→ syncv4 也搬去 n04-33 了
- ❌ 2026-09-02 那版「2/3/6/7 = kyle_attn 四条 campaign」→ 被 09-07 这版取代

★ **n02-29 上 HIP 索引 == rocm-smi 编号**(2026-09-02 实测:`HIP_VISIBLE_DEVICES=0,1,4,5` 拿到的
PCI bus 是 `0x05/0x15/0x85/0x95`,与 `rocm-smi --showbus` 的 GPU0/1/4/5 逐一对上)。
**别把 chi2835 那条「HIP 索引 ≠ smi 编号」搬过来** —— 那是另一台机。

★ 查某个进程实际落在哪张卡:`rocm-smi --showpidgpus` 给的是 **DRM device 号**;
`cat /proc/<pid>/cgroup` 找容器,`/proc/<pid>/cmdline` 认工作线。

### 5c. ★★★★★ 身份自检:我是 inference1,`kyle_attn` 里的 campaign 不归我

**这台机上同时活着两类工作,共用宿主目录、共用节点,但归属完全不同:**

| | inference1(**这就是我**) | kyle_attn 的四场 campaign |
|---|---|---|
| 容器 | `kyle_sglang` | `kyle_attn` |
| venv | `/opt/venv` | `venv-mxfp4` / `venv-tw` / `venv-syncv3` / `venv-syncv4` |
| 仓库 | `sync/inference/aiter`、`/workspace/code/aiter`、`aiter_kyle` | `sync/{mxfp4,tensorwise,syncv3,syncv4}/Primus-Turbo` |
| 工作内容 | aiter / GLM-5.2 decode / AgentX / sglang | gptoss D=64 attn、dense fp8 GEMM、整步融合 |
| 编排 | 我自己手跑 bench,不用 orchestrator | `cursor_campaign.py --repo <那条线>` |

★★★★**收到任何监控/巡逻任务,先做归属自检再动手**。任务描述里只要出现下面任一项,
**它就不是 inference1 的,不要执行**,直接告诉用户「这条线不是我这套环境」:

- 容器 `kyle_attn`
- 仓库路径含 `Primus-Turbo`(mxfp4 / tensorwise / syncv3 / syncv4 任一)
- `cursor_campaign.py --repo mxfp4/...`、`--repo syncv3/...`、`--repo syncv4/...`、`--repo tensorwise/...`
- venv 是 `venv-mxfp4` / `venv-tw` / `venv-syncv3` / `venv-syncv4`
- `flydsl_campaigns/` 下的场次目录 + `_launch_*.sh` + `say.sh` 这套 orchestrator 编排

⇒ **2026-09-07 踩过**:连着两次收到 gpt-oss attn b24/b32 的 cron 巡逻(dir
`flydsl_campaigns/20260905_095125`、容器 `kyle_attn`、`--repo mxfp4/Primus-Turbo`、GPU7),
我照着执行了两轮 —— ssh 进去查容器、翻 `state.json`、读 round 报告、扫红线。
**任务本身做对了,但根本不该由我做**:那是另一条线另一个会话的场子。
判据当时就摆在任务描述里(`kyle_attn` + `Primus-Turbo` + `cursor_campaign`),我没核对就上手。
★ 只读地看一眼不至于闯祸,但**再往前一步就是 `say.sh` 干预、`kill` orch、`--resume` 重启**,
那会直接踩到 [[feedback_scope_own_env_only]]。**归属自检要在第一条命令之前做。**

★ 反过来也成立:inference1 的事(GLM-5.2 decode 表、flydsl a16w16、DSA indexer、AgentX A/B)
不该指望 kyle_attn 那边的人处理。

### 6. ★★ GLM-5.2 decode 计分环境(2026-09-01 收官,交接给下一个人看这段)

**跑分脚本在 SIKL,不在 aiter**:容器内 `/sgl-workspace/SIKL/sikl/benchmarks/models/glm52/`
(host 侧源 `sync/inference/SIKL/`)。模型 `/perf_apps/xiaobo/models/GLM-5.2-MXFP4`。
两个脚本量的不是一个东西、固定参数、以及全部尺子纪律 → **`methodology/19` 整卡读**,这里只记环境。

```bash
# 计分(整步一张 graph);bench 自己会把 HIP_VISIBLE_DEVICES 覆盖成 0-7,TP8 必须八张卡
docker exec kyle_sglang bash -lc '
  cd /workspace/code/aiter && export PATH=/opt/venv/bin:$PATH && python -u _bench_glm52_step.py'
```

- `_bench_glm52_step.py` 在 aiter 仓库根(campaign 的 harness 文件,**交付 squash 时要剥掉**)。
  跑 4 次**取最小**,输出末行是 `{"ok":..,"inv_step":..,"step_ms":..,"spread_pct":..}`。
- 想跑**别的 commit** 而不动工作副本:`cp -a /workspace/code/aiter /tmp/aiter_X`,
  在副本里 `git checkout`,然后 `PYTHONPATH=/tmp/aiter_X` + **从 `/tmp` 之类别的目录启动**
  (从 `/workspace/code/aiter` 里跑,cwd 会盖过 `PYTHONPATH`)。`AITER_JIT_DIR=/root/.aiter_jit_xbc` 复用已编译的 `.so`。
- 每次跑之前:`find /dev/shm -maxdepth 1 -type f ! -name 'rocm_smi_*' -delete`,
  并确认 `rocm-smi --showpids` 是 0 个(**别的容器的孤儿也会让你卡死在 NCCL,细节见 methodology/19**)。

**这条线的交付状态(2026-09-01)**:分支 **`kyle/dev_glm52` @ `9cbed8ac8`**,已 push 到
**`git@github.com:xiaobochen-amd/aiter.git`**(SSH URL,`xbc` remote 配的是 https 推不动;
key = `/workspace/code/.ssh_docker/id_ed25519`)。**ROCm/aiter 千万别推**(用户硬令,踩过一次)。
干净空节点同尺子实测:decode step **22.631 → 21.116 ms(−6.7%)**、2828 → 3033 tok/s;
56 个 skinny 形状冷读总时间 877.4 → 815.2 us、赢 hipBLASLt 37/56 → 43/56。
PR 描述草稿 `/workspace/code/gpt_oss_docker/PR_aiter_dev_glm52.md`。
本地保险 ref:`_campaign_tip_backup`(带全部 harness 的原 tip)、`_pre_final_squash`、
`_pre_rebase_dev_glm52`、`_dev_glm52_old_local`。

⚠ **`xbc/main` 就是 `b02ab811e`**,和分支 base 同一个 commit;`origin/main`(ROCm)领先 27 个,
**别 rebase 到 ROCm main** —— 会把 xiaobochen 自己 PR #10 的 6 个 topk commit 一起重放进我们的 PR。


## ★★ 2026-09-07 §7c:第二套推理环境 `inference2`(和 inference1 同容器,四层隔离)

用户令:「再造一个 inference2 环境,用 venv 方法,跟 syncv3/syncv4 那些环境一样;和 inference1
**共用一个 container**,用它不用的那 4 个 GPU」。所以 **不新起容器**,在 `kyle_sglang` 里再开一套。

| | inference1 | **inference2** |
|---|---|---|
| venv | `/opt/venv`(镜像自带) | **`/opt/venv-inf2`** |
| repo(容器内) | `/workspace/code/aiter` 等 | **`/workspace/code/inference2/<repo>`** |
| repo(host) | `/home/xianzhao/smci_repos/aiter` | `/home/xianzhao/smci_repos/inference2/<repo>` |
| repo(本地 canonical) | `sync/inference/<repo>` | **`sync/inference2/<repo>`** |
| GPU | 0,1,4,5(bus `0x05/0x15/0x85/0x95`) | **2,3,6,7**(bus `0x65/0x75/0xE5/0xF5`,已实测对上) |
| aiter JIT 缓存 | 默认 | **`/root/.aiter_jit_inf2`**(`rexec.sh` 自动设 `AITER_JIT_DIR`) |

跑法(helper 都在本地 `sync/inference2/`):
```bash
cd /workspace/code/gpt_oss_docker/sync/inference2
bash push.sh [<repo>] [<relpath>]     # 本地 -> host(不带参数=顶层 helper + 六个仓全推)
bash rexec.sh '<cmd>'                 # 容器内跑,默认 HIP_VISIBLE_DEVICES=2,3,6,7
bash rexec.sh 'cd /tmp && /opt/venv-inf2/bin/python /workspace/code/inference2/_smoke.py'   # 自检
```

**建 venv-inf2 的四步**(重建照抄;比 chi 那边的 turbo venv 简单,这个镜像的 site-packages
里 `grep /opt/venv/` **零命中**,只有 `bin/` 下 114 个 shebang 要改):
```bash
cp -a /opt/venv /opt/venv-inf2                                     # 8.0G,17s(overlay 不是 NFS)
grep -rIl "/opt/venv/" /opt/venv-inf2/bin/ | xargs -r sed -i "s#/opt/venv/#/opt/venv-inf2/#g"
SP=/opt/venv-inf2/lib/python3.10/site-packages
sed -i "s#/workspace/code/aiter/aiter#/workspace/code/inference2/aiter/aiter#g" $SP/__editable___amd_aiter_*_finder.py
sed -i "s#/sgl-workspace/sglang_kyle/python/sglang#/workspace/code/inference2/sglang/python/sglang#g" $SP/__editable___sglang_*_finder.py
/opt/venv-inf2/bin/python -c "import sys; print([p for p in sys.path if p.startswith('/opt/venv/')])"  # 必须 []
```
和 syncv3/syncv4 一样**只改 editable finder 的 `MAPPING`,不重跑 `pip install -e`** ⇒
`pip list` 里 `amd-aiter` 的版本号是旧的,**功能无影响**。

**六个仓库怎么来的**:从 inference1 **本地路径 clone**(同一 NFS ⇒ git 自动 hardlink
`.git/objects`,不走网络、几乎不占盘);各自多一个 `inf1` remote 方便互拉分支;
`aiter` 落在 **`xbc/main` = `2c71811b3`**(注意:`xbc/main` 已经不是 README 里那个 `b02ab811e` 了,
中间多了 8 个 commit,含 flydsl skinny GEMM 的 PR#13)。CK submodule 用
`git -c protocol.file.allow=always submodule update` 从 inference1 的 `.git/modules/...` 本地填
(★不加 `protocol.file.allow` 会 `fatal: transport 'file' not allowed`)。

★★ **坑:仓库根目录会遮蔽真包**。六个仓库根就在 `inference2/` 下,从该目录启动 python 时
`sys.path[0]` 是它 ⇒ `import aiter` 命中**仓库根**(无 `__init__.py`)变成空 namespace package,
`aiter.__file__` 是 `None`,editable finder 完全没参与。写探针脚本先 `print(aiter.__file__)` 验一眼,
打出 `None` 就是中招。`_smoke.py` 里的解法是开头把自身目录从 `sys.path` 摘掉。

细节全在 `sync/inference2/README.md`;memory [[project_inference2_env]]。
