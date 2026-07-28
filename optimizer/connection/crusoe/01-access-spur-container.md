# Crusoe 集群接入 · spur 调度 · 节点 dockerd 容器 dev box

> 类别: 连接 · 主题标签: crusoe, spur, slurm, docker, dockerd, sbatch, srun, account, rocm-primus, flydsl, venv, ssh, gpt-oss, bench
> 状态: 2026-07-20 首次搭建，端到端跑通(容器起 + torch 2.10/8GPU + flydsl 0.2.2 + dq bench 1100TF + turbo csrc build)。2026-07-22 复用保存的 tar `docker load` 直接起(§4b 一条龙,3-4min,免重装 flydsl)再次实测通。✅=已实测。
> **★怎么找到自己当前的活节点**(节点名每次分配都变,别记死值):
> ```bash
> RSH 'squeue -u xianzhao'          # 拿 JOBID + NODELIST;ST 必须是 R
> RSH 'squeue -j <JOBID>'           # 返回空 = job 已终止(别信上一条的 R,它滞后~2min,见 §4.0)
> RSH 'srun --overlap --jobid=<JOBID> --pty true'   # ❌ 别用这个探活,见 §4.-1
> RSH 'srun -A amd-primus -p amd-spur -t 2 bash /shared_nfs/kyle/probe_node.sh'  # ✅ 探活/探 GPU
> ```
> 没有活 job 就按 §0 第 3 步重新 sbatch 一个。**只认自己的 job** —— `squeue -u xianzhao` 里别人的节点不要碰。

> **🚨 2026-07-28 更正:跳板机挂了,chi 全线不可达 —— 下面这条回退路线暂时无效。**
> `149.28.124.225` **本身** SSH 超时,而 `sync/.ssh-chi.sh` 是靠 ProxyCommand 经它中转的,
> 所以 **chi2798 / chi2774 / chi2811 全部连不上**。症状是 `Connection timed out during banner exchange`,
> 看着像节点故障、实为跳板故障 —— 我据此误判"chi2798 节点挂了",让一个 campaign 空等 3.5 小时。
> ★**诊断顺序:先单独试 `ssh root@149.28.124.225`,再怀疑节点。**
> ~~⚠️ 2026-07-23 回退提示:crusoe spur 常排队,急用 GPU 时优先回 chi2774~~(跳板恢复后才重新适用):
> chi2774 经跳板 + `sync/.ssh-chi.sh`,容器 `mlperf_gptoss` 长期 Up,GPU4-7 干净,
> meta-attn 分支 Primus-Turbo 在容器 `/workspace/code/Primus-Turbo`。
> 详见 [[../../../../.claude/memory/project_crusoe_env]] + [[../../../../.claude/memory/project_chi2811_sync]]。
> chi 侧文件传输走 base64→容器 `/root`(`docker cp /dev/stdin`/scp-to-NFS/`docker exec -i` 都挂)。

## 0. TL;DR（一条龙）
Crusoe = AMD 内部集群，调度器 `spur`(slurm 兼容)。**容器不用 spur 的 --container-image(那条死路)，用节点自带的 dockerd**。流程：
1. login: `ssh -i .ssh_laptop/id_ed25519 xianzhao@crs-m2m-cpu-spur-login.crusoe.amd.com`（csh！命令包 `bash -lc`）
2. 账号关联一次：`spur accounts -i add user name=xianzhao account=amd-primus`
3. ★**sbatch 脚本体里直接跑 setup + `sleep infinity`**(**不带** container flag),一步到位 —— **别** `--wrap="sleep infinity"` 占完节点再想办法进去,那条路会撞上 §4.-1 的 `srun --overlap` 陷阱、而 node-ssh 又不通。
   ```bash
   #SBATCH -A amd-primus -p amd-spur --qos=amd-primus-qos --exclusive -t 7-00:00:00
   #SBATCH -o /shared_nfs/kyle/kyle.out
   bash /shared_nfs/kyle/node_setup.sh
   sleep infinity
   ```
4. `node_setup.sh` 里 **从保存的 tar 直接起**(flydsl 0.2.2/egg-fix/双 venv 全烤进镜像,免重装):`docker load -i /shared_nfs/kyle/images/gpt-oss-docker_kyle-20260720.tar`(25G,~3min)→ `docker run -d --name=gpt-oss-docker --network=host --ipc=host --device=/dev/kfd --device=/dev/dri --group-add video -v /shared_nfs/kyle:/workspace/code gpt-oss-docker:kyle-20260720 sleep infinity`。进度看 sbatch 的 `-o` 输出文件。**兜底**(无 tar):`docker pull rocm/primus:v26.3` + §6 重装 flydsl。
5. 干活:短命令用独立 `srun ... bash /shared_nfs/kyle/xx.sh`(见 §4.-1),容器内用 `docker exec gpt-oss-docker bash -lc '...'`。coding base = `/shared_nfs/kyle`(=容器内 `/workspace/code`)。⚠ 那里目前**只有 `meta-attn`,没有 `Primus-Turbo`** —— 要跑 turbo 得先把整个仓(**含 `.git`**)传过去。⚡ 一条龙脚本见 §4b。

## 1. 登录 ✅
- host `xianzhao@crs-m2m-cpu-spur-login.crusoe.amd.com`；key **`/workspace/code/.ssh_laptop/id_ed25519`**(pub=`xianzhao@amd.com`)。首次把该 pub 加进 login `~/.ssh/authorized_keys`(共享 NFS home，一次全集群生效)：`echo 'ssh-ed25519 AAAA... xianzhao@amd.com' >> ~/.ssh/authorized_keys`。
- ⚠️ login 多台轮询(-005/-012/-013…)，host key 变 → 用 `-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null`，别 pin。
- ⚠️ **login 与 compute 节点默认 shell 都是 csh/tcsh** → 远程命令必须包 `bash -lc '...'`，否则 `2>`/`2>&1` 报 "Ambiguous output redirect"。
- 连接函数：
  ```bash
  RSH(){ ssh -i /workspace/code/.ssh_laptop/id_ed25519 -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o ConnectTimeout=25 \
    xianzhao@crs-m2m-cpu-spur-login.crusoe.amd.com "bash -lc '$1'" 2>&1 \
    | grep -vE 'Warning: Permanently|known_hosts|@@@|nasty|eavesdrop|fingerprint|host key'; }
  ```
- **node-to-node ssh**(login→compute 必需)：在 login 上 `ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N '' && cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys`(共享 home，全节点即通)。

## 2. login 红线 ✅(踩过)
**Login Node Guardian 杀超内存进程**(实测 `spur image import` 吃 35G 被 kill+告警邮件)。→ 重负载(build/跑测/镜像) **一律在 compute 节点**；login 只做 sbatch/squeue/scancel/scp/rsync 轻活。

## 3. spur / slurm + 账号 ✅
- `srun`/`sbatch`/`squeue`/`scancel`/`sinfo` 都在；`salloc`→`spur alloc`；`sacctmgr`→`spur accounts`。
- 账号 `-A amd-primus`、分区 `-p amd-spur`(全集群唯一，256 台共享)、`--qos=amd-primus-qos`(**只 sbatch 收；srun 不收 --qos/--pty**)。
- **必须先账号关联**否则全报 `not associated with account`：`spur accounts -i add user name=xianzhao account=amd-primus`(~10s 生效)。验证 `srun -t1 -A amd-primus -p amd-spur hostname`。查成员 `spur sshare -a | grep <user>`。
- 负载：`sinfo -p amd-spur`(常 100+ idle)。

## ⭐4. 容器 dev box = 节点 dockerd + docker run ✅端到端验证

### 4.-1 ★★★怎么在节点上执行命令(2026-07-28 实测,比 §4.0 更早会踩)

**结论先行:进不去已分配的节点。三条交互式通道全不通,必须用 §4.-2 的文件队列 agent。**

| 方式 | 结果 | 说明 |
|---|---|---|
| node-to-node `ssh <node>` | ❌ | `Permission denied (publickey)`,即便 key 已在共享 home 的 `authorized_keys`、且在 compute 节点上 `ls ~/.ssh/authorized_keys` 可见(home 确为共享 NFS)。原因未查明 —— §1 那套配置法**不管用**。 |
| `srun --overlap --jobid=<已有job>` | ❌ | **静默失败 rc=128,连错误信息都没有**(spur 对 slurm overlap 支持不完整)。更坑的是它偶尔会返回**残缺环境**下的输出:在**确实有 GPU** 的节点上报 `/dev/kfd` 不存在、`docker: command not found` —— 我因此误判"申请到了 CPU 节点",还去找根本不存在的"GPU 分区"。 |
| `srun -w <已被占节点>` | ❌ | 节点被 `--exclusive` 独占,新 job 只能 `Pending`,永远等不到。`--oversubscribe` 参数 spur 不认。 |
| 独立 `srun -A amd-primus -p amd-spur -t 5 bash /shared_nfs/kyle/xx.sh` | ✅ | **只适合一次性探测** —— 它会**新分配一台节点**,不是你已有 job 的那台。别拿它去操作已有容器。 |
| **文件队列 agent(§4.-2)** | ✅✅ | **唯一能持续驱动已有节点/容器的方式。** |

⚠ 别在这三条死路上反复试 —— 已经有两个 session 分别在 node-ssh 和 srun --overlap 上耗掉大量轮次,
其中一个还得出了"srun --overlap 死路 → node-ssh 是唯一正道"的**错误结论**(两条都不通)。

### 4.-2 ★★文件队列 agent = 唯一可用的持续通道(2026-07-28 端到端验证)
思路:节点上常驻一个轮询循环,从 NFS 读 `.job` 脚本执行、结果写回。login 侧只需 scp 文件。

**`/shared_nfs/kyle/node_agent.sh`**(sbatch 脚本体里直接 `bash` 它,不要 `sleep infinity`):
```bash
Q=/shared_nfs/kyle/q; mkdir -p "$Q"
echo "[agent] $(hostname) starting $(date)" >> "$Q/agent.log"
# 1) 起容器(已存在则复用 —— 换 job 重跑时不会重复 docker load)
if ! docker ps --format '{{.Names}}' | grep -qx gpt-oss-docker; then
  docker images | grep -q gpt-oss-docker || \
    docker load -i /shared_nfs/kyle/images/gpt-oss-docker_kyle-20260720.tar
  docker rm -f gpt-oss-docker 2>/dev/null
  docker run -d --name=gpt-oss-docker --network=host --ipc=host \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -v /shared_nfs/kyle:/workspace/code \
    gpt-oss-docker:kyle-20260720 sleep infinity
  sleep 5
fi
echo "[agent] container up: $(docker ps --format '{{.Names}}' | tr '\n' ' ')" >> "$Q/agent.log"
# 2) 命令队列:*.job -> 执行 -> *.out + *.rc,原文件改名 *.done
while true; do
  for f in "$Q"/*.job; do
    [ -e "$f" ] || continue
    b="${f%.job}"; bash "$f" > "$b.out" 2>&1; echo $? > "$b.rc"; mv "$f" "$b.done"
  done
  sleep 3
done
```
**用法**(从本地/login 侧):
```bash
scp ... mytask.job xianzhao@<login>:/shared_nfs/kyle/q/mytask.job   # 投递
sleep 5 && RSH 'cat /shared_nfs/kyle/q/mytask.out; cat /shared_nfs/kyle/q/mytask.rc'  # 取结果
```
`.job` 内容就是普通 shell,里面爱怎么 `docker exec gpt-oss-docker bash -lc '...'` 都行。
✅ 实测:投递后 3 秒内执行,返回 `crsuse2-m2m-328 / MI355X / torch 2.10.0 gpus 8 / rc=0`。
★ 好处:换节点/重启 job 只需重投 sbatch,队列目录在 NFS 上跨节点存活;容器复用不重复 load。
- ★★**多层引号地狱**:本地 bash → ssh → csh → `bash -lc` → srun → bash,内联命令里的 `"` `$` `|` 被层层吃掉。
  实测:`head -8` 被当成选项报错、`grep -ciE "kfd|dri"` 被拆成两条命令、`awk "{print \$1}"` 直接语法错、
  `sinfo -o "%20P %6a"` 格式串整个消失。**一律把命令写成脚本 scp 到 `/shared_nfs/kyle/`,远端只执行文件名。**
- **分区认知**:全集群**只有 `amd-spur` 一个分区**(254 节点),GPU 节点就在里面。login 主机名里的 `cpu`
  (`crs-m2m-cpu-spur-login`)只表示 login 机自身是 CPU 机,**不代表分区没 GPU** —— 别被它误导去找"GPU 分区"。
- 探测脚本模板(放 `/shared_nfs/kyle/probe_node.sh`):
  ```bash
  echo "HOST=$(hostname)"; echo "KFD=$(ls /dev/kfd 2>/dev/null && echo yes || echo no)"
  echo "DRI=$(ls /dev/dri 2>/dev/null | wc -l)"; echo "DOCKER=$(command -v docker || echo none)"
  ```
  健康 GPU 节点应返回 `KFD=/dev/kfd yes` / `DRI=129` / `DOCKER=/usr/bin/docker`。

### 4.0 跑前先确认节点活死(2026-07-22 血泪)⚠️
`squeue` 的 `R` 状态**会滞后 ~2min** —— 节点 NODE_FAIL 后 squeue 快照仍可能显 `R`(301 实测:06:35:06 NODE_FAIL,而 06:33 快照还是 `R 4:56:34`,信了它会拿死节点跑/或以为数据有效)。**别信 squeue 的 R**:
- 确认死活:`squeue -j <JOBID>`(返回空=job 已终止)+ `spur accounts`/账务库看 `NODE_FAIL`;或直接 `bash ~/dx.sh <NODE> <脚本里第一行 hostname>` 看是否连得上并落在预期节点。
- 换节点后:`/shared_nfs` 是 NFS **跨节点存活**(harness/kernel/tar 无需重传),只需在新节点重 `docker load+run`(§4b)。
- **别碰别人 sbatch 的节点**(如队友的 mxfp4 bench 节点);`squeue -u xianzhao -t RUNNING` 看清哪台是自己该用的。
- perf 数据只认**确认活着的健康节点**:临挂节点热态劣化会给偏悲观数(301 vs 289 同 kernel ratio 差最多 ~1.8×,小 shape 尤甚)。
compute 节点自带 **dockerd + containerd**(本地存储 `/mnt/m2m_nobackup/docker`=node-local 快盘)，xianzhao 直接能用 docker。别人(xiaompen primus-dev、chaojhou ML jobs)都这么跑。
1. **占裸节点**(不带 container flag)：本地写脚本 scp 提交(避免多行 `\` 被 ssh 单引号拆断)：
   ```bash
   #!/bin/bash
   #SBATCH -A amd-primus
   #SBATCH -p amd-spur
   #SBATCH --qos=amd-primus-qos
   #SBATCH -N1
   #SBATCH --exclusive
   #SBATCH -t 24:00:00
   #SBATCH -J kyle_box
   #SBATCH --output=/home/xianzhao/logs/box.%j.out
   sleep infinity
   ```
   `squeue -u xianzhao -o "%i %j %T %N"` 拿节点名(如 crsuse2-m2m-171)。
2. **node-ssh 进节点** → **首选 = 从保存的 tar `docker load`**(§4c 存的 `gpt-oss-docker:kyle-20260720`,flydsl 0.2.2/egg-fix/双 venv 全烤好,免 §6 重装):`docker load -i /shared_nfs/kyle/images/gpt-oss-docker_kyle-20260720.tar`(24G,~3-4min)。**兜底** = `docker pull rocm/primus:v26.3`(节点常已缓存 v26.2/3/4;harbor 代理 `harbor.crusoe.primus-safe.amd.com/proxy/...`),但要再走 §6 装 flydsl。
3. **docker run 起持久容器**(镜像用 tar 的 `gpt-oss-docker:kyle-20260720`;兜底用 `rocm/primus:v26.3`)：
   ```
   docker run -d --name=gpt-oss-docker --network=host --ipc=host --device /dev/dri --device /dev/kfd \
     --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size=64G \
     -v /shared_nfs/kyle:/workspace/code gpt-oss-docker:kyle-20260720 sleep infinity
   ```
   实测容器内 torch 2.10.0+8 GPU + /workspace/code 挂载 OK；ROCm 7.2.1、arch gfx950。
4. 干活 `docker exec gpt-oss-docker bash -lc '...'`。**换节点**：容器在 node-local，节点变了要重 docker load+run(tar/代码在 /shared_nfs,几分钟)。

## 4b. 一条龙起容器脚本(tar 优先,已实测 2026-07-22)✅
引号地狱解法:脚本落 `/shared_nfs/kyle`(login 端 base64 写,node-host 后台跑),全程不嵌引号。
```bash
# ①login 端: 提交占节点 → 拿 NODE
sbatch -A amd-primus -p amd-spur --qos=amd-primus-qos -N1 --exclusive -t 12:00:00 -J kyle_box \
  --output=/home/xianzhao/logs/box.%j.out --wrap="sleep infinity"
squeue -u xianzhao -o "%i %T %N"            # 拿 NODE(如 crsuse2-m2m-301)
# ②login 端: 把下面脚本写到 /shared_nfs/kyle/_load_run.sh(用 base64 -d 落地,避免引号)
#   内容:
#     IMG=gpt-oss-docker:kyle-20260720
#     docker image inspect $IMG >/dev/null 2>&1 || docker load -i /shared_nfs/kyle/images/gpt-oss-docker_kyle-20260720.tar
#     docker rm -f gpt-oss-docker 2>/dev/null
#     docker run -d --name=gpt-oss-docker --network=host --ipc=host --device /dev/dri --device /dev/kfd \
#       --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size=64G \
#       -v /shared_nfs/kyle:/workspace/code $IMG sleep infinity
# ③node-host 后台跑(24G load 会超 ssh 2min → 必须后台 + 轮询日志):
bash ~/nbg.sh <NODE> /shared_nfs/kyle/_load_run.sh /shared_nfs/kyle/_load_run.log
# 轮询: tail /shared_nfs/kyle/_load_run.log  直到 "Loaded image" + docker ps 见 Up
```
⚠️ **`docker ps --format '{{.Names}}'` 里的 `{{}}` 会被 login/csh 层吃掉** → 查状态也用脚本文件里的 `docker ps -a | grep gpt-oss`,别在命令行直接传 `--format`。
⚠️ node-ssh 里的**内层命令用双引号**(`ssh NODE "docker ..."`),别用单引号 —— 会和 RSH 的 `bash -lc '...'` 外层单引号碰撞,命令截断跑到 login 上("docker: command not found" 就是这坑)。

## 4c. 持久化容器(docker commit + tar)✅
容器是 **node-local**，节点变/容器删就没了。venv-tw/flydsl 0.2.2/egg 修复/finder/git config 等**改动在容器可写层**(代码在 bind-mount /shared_nfs/kyle 不在镜像；`.so` 也在 repo=共享盘)。**commit/save 在节点 host 跑(不是 docker exec)**：
```bash
# 在节点上(node-ssh 进去,或 nbg 脚本后台):
docker commit gpt-oss-docker gpt-oss-docker:kyle-20260720
mkdir -p /shared_nfs/kyle/images
docker save gpt-oss-docker:kyle-20260720 -o /shared_nfs/kyle/images/gpt-oss-docker_kyle-20260720.tar   # ~22G,慢(NFS),后台跑
```
换节点恢复：
```bash
docker load -i /shared_nfs/kyle/images/gpt-oss-docker_kyle-20260720.tar
docker run -d --name=gpt-oss-docker --network=host --ipc=host --device /dev/dri --device /dev/kfd \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size=64G \
  -v /shared_nfs/kyle:/workspace/code gpt-oss-docker:kyle-20260720 sleep infinity
```
tar 在共享盘 → 任何节点 `docker load` 即用，省去重装 flydsl/重 build。跑法 helper：`~/nbg.sh <node> <script> <log>`(node-host 后台) 对应 `~/dxbg.sh`(容器内后台)。

## 5. 在容器里跑东西（引号地狱的解法）✅
三层嵌套(login bash -lc → node ssh → docker exec bash -lc → python -c)引号必崩。**解法：脚本落 `/shared_nfs/kyle`(=容器内 `/workspace/code`)，用 login 端 helper 调 docker exec 跑脚本**，全程不嵌引号：
```bash
# ~/dx.sh   (login上): 容器内前台跑脚本
#   ssh -o Strict.. -o UserKnownHostsFile=/dev/null "$1" "docker exec gpt-oss-docker bash /workspace/code/$2"
# ~/dxbg.sh (login上): 容器内后台跑 + 日志到 /workspace/code
#   ssh .. "$1" "docker exec gpt-oss-docker bash -c 'nohup bash /workspace/code/$2 >/workspace/code/$3 2>&1 </dev/null & echo LAUNCHED pid=\$!'"
# 用: RSH 'bash ~/dx.sh crsuse2-m2m-171 _mytest.sh'   (节点名传字面量，别用没展开的 $VAR)
```
- 长任务(build/bench)用 dxbg 后台 + 轮询日志(`RSH 'tail -n20 /shared_nfs/kyle/xxx.log'`)。
- 跑 GPU：脚本里 `HIP_VISIBLE_DEVICES=N /opt/venv-tw/bin/python -u bench.py`；改 flydsl kernel 后先 `rm -rf /root/.flydsl/cache`(JIT 缓存坑)。
- ⚠️ 别用 `find`/`du` 扫 NFS 大目录(十几万文件会超时)。

## 6. 两套 turbo 环境 build(mxfp4=/opt/venv，tensorwise=/opt/venv-tw)✅
镜像自带 `/opt/venv`(torch+primus_turbo+flydsl)。两套隔离(见 gpt_oss/02-run-and-venv)：mxfp4→`/opt/venv`，tensorwise→`/opt/venv-tw`。代码在 `/shared_nfs/kyle/{mxfp4,tensorwise}/Primus-Turbo`(从本 agent sandbox `rsync -azh --mkpath --exclude-from=.rsync-exclude` 上来)。
- **建 /opt/venv-tw**：`cp -a /opt/venv /opt/venv-tw` 后必须修两处，否则污染 /opt/venv：
  1. `setuptools.pth`/`pyvenv.cfg` 里 `/opt/venv`→`/opt/venv-tw`(否则 sys.path fallback 老 venv)。
  2. ⚠️**clone 的 `bin/pip` shebang 仍指 `/opt/venv/bin/python`** → `venv-tw/bin/pip install` 会装进 **/opt/venv**！**一律用 `/opt/venv-tw/bin/python -m pip ...`**(强制用 venv-tw 的 python)，别用 `venv-tw/bin/pip`。
- **flydsl 0.2.2**：`/opt/venv-tw/bin/python -m pip install --force-reinstall --no-deps flydsl==0.2.2`。⚠️**镜像自带 flydsl 0.1.1.dev409 是 egg**(`site-packages/flydsl-0.1.1.dev409-*.egg` + `easy-install.pth` 里一行)，**egg 在 sys.path 前面会 shadow pip 装的 0.2.2** → import 仍 0.1.1。修：`rm -rf 该egg目录 && sed -i '/flydsl-0.1.1.dev409/d' easy-install.pth`，再 import 就是 0.2.2。(meta-attn dq 用 0.2.2。)
- **build turbo csrc**：先 `git config --global --add safe.directory '*'`(否则 setup.py 的 `git submodule sync` 报 dubious-ownership 挂 build)+ CK submodule 就位(`git submodule update --init 3rdparty/composable_kernel`，仓库已钉正确 commit)。然后：
  ```
  cd /workspace/code/tensorwise/Primus-Turbo && rm -rf build primus_turbo/lib/*.so
  GPU_ARCHS=gfx950 MAX_JOBS=128 /opt/venv-tw/bin/python -m pip install -e . --no-build-isolation --no-deps
  ```
  ~20-25min(CK 实例化 + 尾部 `-fgpu-rdc` 设备链接最慢)；ninja -j128，`.o` 编进 `build/`，产物 `primus_turbo/lib/libprimus_turbo_kernels.so`。`.so` 在 repo(共享盘)，哪个 venv 的 editable 指过来都能用。
- **finder 归属**：`pip install -e .` 会把 editable finder(`__editable___primus_turbo_..._finder.py` 的 MAPPING)写进"当前 python 的 venv"。用错 python(venv-tw/bin/pip=老python)会写进 /opt/venv 污染 mxfp4。修法：直接 sed 两 venv finder 的 MAPPING 路径改回各指各 repo(无需重 build)。

## 7. 存储 ✅
- `/shared_nfs`(50T 集群共享)→ coding base `/shared_nfs/kyle`(和 /shared_nfs/<user> 放一起)；容器里挂到 `/workspace/code`。**跨节点存活**,换节点只需重 `docker load + run`,代码/harness/镜像 tar 都不用重传。
  ⚠**2026-07-28 实测那里只有 `meta-attn` 和 `images/`,没有 `Primus-Turbo`** —— 要跑 turbo 相关(如 fwd campaign)必须先把整个仓传过去,**含 `.git`**(campaign 要做 git 操作)。
- `/home/xianzhao`(5T NFS)→ ssh key/脚本/日志，别放大模型。
- `/mnt/m2m_nobackup`(compute 本地)→ 大模型 + docker 存储；`/mnt/m2m_nobackup/huggingface/` 有预置。

## ❌8. 死路(2026-07-20 穷举踩过，别再试)
- **spur `--container-image` squashfs 机制**：任何 `sbatch/srun --container-image=...` 对 xianzhao 一律 `JobHoldMaxRequeue`(container-launch 失败)，与镜像大小/来源(docker://、拷 .sqsh、正规 import)/注册/flag 形式全无关；个别 sbatch 静默忽略把命令跑裸 host。**本集群容器只走节点 dockerd(§4)，spur 只用来占节点。**
- `spur image import` 走 NFS home `~/.spur/images`(`/var/spool/spur/images` 不可写、spur 0.5.0 无 config 改 image dir)，解包十几万小文件 ~45min 且无 --exclusive 会 cgroup OOM。**别用**，直接 `docker pull` 到节点本地盘。

## 9. 坑速查
| 症状 | 原因 | 修 |
|---|---|---|
| `Ambiguous output redirect` | 节点是 csh | 命令包 `bash -lc '...'` |
| `not associated with account` | 没账号关联 | `spur accounts -i add user name=.. account=amd-primus` |
| 容器 job `JobHoldMaxRequeue` | 用了 spur --container-image(死路) | 改节点 dockerd + docker run(§4) |
| docker exec 三层引号崩 | 嵌套引号 | 脚本落 /shared_nfs/kyle + dx.sh/dxbg.sh helper(§5) |
| `pip install` 装错 venv | clone venv 的 bin/pip shebang 指老 python | 用 `venv-tw/bin/python -m pip` |
| flydsl import 版本不对 | 0.1.1 egg 在 easy-install.pth shadow | rm egg + 删 easy-install.pth 那行 |
| build 报 dubious ownership | git safe.directory 没设 | `git config --global --add safe.directory '*'` |
| login 进程被杀 | Guardian 限内存 | 重负载挪 compute 节点 |
| host key changed | login 多台轮询 | `StrictHostKeyChecking=no UserKnownHostsFile=/dev/null` |
| squeue 显 R 但节点其实挂了 | R 状态滞后 ~2min(NODE_FAIL) | `squeue -j <id>` 查空 + `hostname` 探活;换活节点重 docker load+run(§4.0) |
| 改 flydsl kernel 后行为没变 | JIT 缓存 | `rm -rf /root/.flydsl/cache` |

## 10. 已验证里程碑(2026-07-20)
- 容器 gpt-oss-docker(rocm/primus:v26.3) on crsuse2-m2m-171：torch 2.10.0 / 8 GPU / gfx950 / ROCm7.2.1 ✅
- meta-attn dq bench(flydsl 0.2.2)：**dq ~1096–1164 TF**(Sq 2048/4096/8192/16384，hw-exp 16x16x32，B4 Hq128 Hkv16 D64 Skv16384)，dkdv ~1100–1147 TF ✅
- tensorwise csrc build(gfx950)进行中/可复现 ✅

## 11. PMC / rocprofv3 工具链(crsuse2-m2m-328,gfx950,2026-07-28 实测)
- `rocprofv3` **1.1.0** / rocm-7.2.1;`llvm-mc` = `/opt/rocm/lib/llvm/bin/llvm-mc`(AMD LLVM 22.0.0git,`-mcpu=gfx950 -show-encoding` 验指令是否 lower)。
- 用法(文件队列 job 内):`rocprofv3 --pmc <派生指标...> -d <NFS路径> --output-format csv -- /opt/venv-syncv3/bin/python -u <script>`。`-d` **必须落 NFS `/workspace/code/...`**(容器 /tmp 会清)。
- **派生指标可按名请求**:`MfmaUtil VALUBusy MemUnitStalled LDSBankConflict MeanOccupancyPerActiveCU`(自动多趟回放绕过 ≤3 raw-counter/组、err38)。CSV = `<out>/<host>/<pid>_counter_collection.csv`,每 dispatch 一行(含 `Kernel_Id/VGPR_Count/Accum_VGPR_Count/SGPR_Count/LDS_Block_Size/Scratch_Size/Counter_Name/Counter_Value`);量大(几 MB)→ 节点侧 python 聚合按 Kernel_Id 取均值,别整传回。
- ⚠ **`MemUnitBusy` 在 gfx950 rocprofv3 1.1.0 不存在**(报 "Unable to find counter"),只有 `MemUnitStalled`。判 feed/latency-bound 用 MfmaUtil(空闲%)+ MemUnitStalled(≈0=非带宽限)+ LDSBankConflict 组合。
- 判据速记:MfmaUtil 低 + MemUnitStalled≈0 + LDSBankConflict≈0 = **latency/dependency-bound**(等依赖链,非带宽非计算);此时 swizzle/加宽/prefetch 多为红鲱鱼(铁律),真杠杆是缩依赖链或跨-tile overlap。

---
来源: 本 session (2026-07-20) 迁移 Crusoe 全程实测 + crusoe_user_guide 摘录;§11 = 2026-07-28 Shape A PMC 诊断实测。
