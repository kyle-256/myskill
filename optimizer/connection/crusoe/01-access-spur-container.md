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

## 0. TL;DR（一条龙）
Crusoe = AMD 内部集群，调度器 `spur`(slurm 兼容)。**容器不用 spur 的 --container-image(那条死路)，用节点自带的 dockerd**。流程：
1. login: `ssh -i .ssh_laptop/id_ed25519 xianzhao@crs-m2m-cpu-spur-login.crusoe.amd.com`（csh！命令包 `bash -lc`）
2. 账号关联一次：`spur accounts -i add user name=xianzhao account=amd-primus`
3. ★**sbatch 脚本体里直接跑 setup + `sleep infinity`**(**不带** container flag),一步到位 —— **别** `--wrap="sleep infinity"` 占完节点再想办法进去,那条路会撞上 §4.-1 的 `srun --overlap` 陷阱、而 node-ssh 又不通。
   ```bash
   #SBATCH -A amd-primus -p amd-spur --qos=amd-burst-qos --exclusive -t 7-00:00:00   # ★burst 池,别用 amd-primus-qos(16 节点上限,见 §3)
   #SBATCH -o /shared_nfs/kyle/kyle.out
   bash /shared_nfs/kyle/node_setup.sh
   sleep infinity
   ```
4. `node_setup.sh` 里 **从保存的 tar 直接起**(flydsl 0.2.2/egg-fix/4 套 venv(含 venv-syncv4)全烤进镜像,免重装):`docker load -i /shared_nfs/kyle/images/gpt-oss-docker_kyle-20260807.tar`(25G,~3min)→ `docker run -d --name=gpt-oss-docker --network=host --ipc=host --device=/dev/kfd --device=/dev/dri --group-add video -v /shared_nfs/kyle:/workspace/code gpt-oss-docker:kyle-20260807 sleep infinity`。进度看 sbatch 的 `-o` 输出文件。**兜底**(无 tar):`docker pull rocm/primus:v26.3` + §6 重装 flydsl。
5. 干活:短命令用独立 `srun ... bash /shared_nfs/kyle/xx.sh`(见 §4.-1),容器内用 `docker exec gpt-oss-docker bash -lc '...'`。coding base = `/shared_nfs/kyle`(=容器内 `/workspace/code`)。⚠ 新仓要跑 turbo 先把整个仓(**含 `.git`**)传过去(rsync 到 login `/shared_nfs/kyle/<name>/Primus-Turbo`,.so 排除后从现成仓补,见 [[../../../../.claude/memory/project_syncv4_env]] 的 syncv4 standup 流程)。⚡ 一条龙脚本见 §4b。

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
- `srun`/`sbatch`/`squeue`/`scancel`/`sinfo` 都在(都是 `spur <子命令>` 的符号链接);`salloc`→`spur alloc`;`sacctmgr`→`spur accounts`(登录机上没有独立 sacctmgr,用 `spur accounts show qos/account/...`)。
- 账号 `-A amd-primus`、分区 `-p amd-spur`(全集群唯一,254 台共享)。
- **★★★ QOS 选 `amd-burst-qos`(弹性池),别用 `amd-primus-qos`(2026-07-28 实测定案)**:
  - `amd-primus-qos` GrpTRES **node=16** —— 这是 amd-primus 组共享的硬上限,组里别人占满了你就 **Pending(Reason=QOSGrpNodeLimit)**,哪怕集群里几十台 idle 也拿不到。之前被卡数小时的根因就是它。
  - `amd-burst-qos` GrpTRES **node=128** / 每用户 **node=32** —— 弹性 burst 池,能吃到其它 pool 的空闲机。`spur accounts show qos format=name,grptres,maxtrespu` 可复核所有 QOS 上限。
  - **实测**:`sbatch -A amd-primus -p amd-spur --qos=amd-burst-qos -N1 -t ... <脚本>` 直接被接收,job 秒落 idle 节点(crsuse2-m2m-153),`Reason=None` 零排队。参考来源=Primus `run_deepseek_v4_flash.sh`(dev/tas/deepseek-v4-phase2)里 `export SLURM_QOS=amd-burst-qos`。
  - 用法:sbatch 脚本头写 `#SBATCH --qos=amd-burst-qos`(**只 sbatch 收 --qos;srun 不收 --qos/--pty**)。其它 QOS 上限见 `spur accounts show qos`(amd-general=8 / amd-spur-qos=8 等)。
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

### 4.-1b ★★login 连接时好时坏的真因 + 动态选健康后端(2026-08-07 定案,已推翻旧 pin 规程)
**症状**:同一把 key(`.ssh_laptop/id_ed25519`)连 `crs-m2m-cpu-spur-login` 有时成功、有时失败;campaign 远端 bench 偶发 rsync 255 崩 round / 卡住。
**真因(修订版)**:`crs-m2m-cpu-spur-login.crusoe.amd.com` 只解析到**单个 VIP** `10.180.168.227`,轮询发生在 **VIP 后端**(-005/-009/-012…),而**后端各机会各自 flap**:接受 TCP + **key auth 成功**("Server accepts key")后,后端立刻 **post-auth 掐断 session**("Connection closed by … port 22")。表现为**成片抖动**(一阵 12/12 通,一阵 0/8 断),不是 fail2ban、不是 key 问题,是后端 session-setup 侧的故障。命中健康后端就通,命中正在 flap 的就断。
- ⚠ **旧结论已作废**:早前说"后端直连主机名 firewall 挡、直连不通、只能走 VIP"是**错的**。实测**后端直连 FQDN 可达**:`crs-m2m-cpu-spur-005/012.crusoe.amd.com` 直接 `ssh` 通;哪台健康是随时间 flap 的(如 08-07 -009 DOWN、-012 UP)。因此客户端**可以按 FQDN 直选后端**,绕开 VIP 轮盘。
- ⚠ 所有后端共享同一 `/shared_nfs`,但**只有部分后端挂了我们的队列目录**(`/shared_nfs/kyle/q*`);选后端必须同时满足**可达 + 挂了对应队列**。
**修复 = 动态解析健康后端(取代 pin -009)**:见 memory `project_crusoe_dynamic_backend_resolve`。
- **`sync/crusoe/resolve_login.sh <need_dir>`**:探测后端直连 FQDN(默认序 `012 009 005 006 007 008 010 011 013 014`,可 `CRUSOE_BACKEND_IDS=` 覆盖),打印第一个可达且 `test -d <need>` 通过的 `user@host`,按 `<need>` md5 缓存 `/tmp/crusoe_login_*`;`CRUSOE_RESOLVE_FORCE=1` 跳缓存换后端。
- **`sync/crusoe/lib.sh`**:`CRUSOE_LOGIN` 为空即经解析器动态选(4 套 push/rexec 共用);`crusoe_exec` 轮询中途后端死会 `crusoe_relogin`;仍可 `export CRUSOE_LOGIN` 覆盖。
- **`cursor_campaign.py` `CrusoeRemoteHarness`**:删 `-009` 常量,`_resolve_login(force=)`;`sync()`/`_docker()` 推送与轮询失败都 `force=True` 跳健康后端重试 → 后端 flap 不再崩 round。
- **在飞老代码 campaign 保活**:已在跑的 campaign 不热加载 .py,仍用固定 `-009` 名 → 靠 `/root/.ssh/config` 里 `-009` 别名的 `HostName` override 转到健康后端;**`sync/crusoe/refresh_alias.sh`**(cron)在别名后端掉线时自动重指,健康时静默。
- ★换机/pool 全变时**只改 `resolve_login.sh` 的 `IDS` 顺序**即可,不用动 campaign。
- ⚠ **`pin_009.sh` 已弃用**(ControlMaster 硬钉单台 -009 的老法):-009 会 flap、钉死反而崩;保留文件仅作历史。容器重建清 `/root/.ssh/config` 后,新 campaign 靠解析器自愈,无需手动补 config;老代码在飞进程才需 refresh_alias 维持别名。

### 4.-2 ★★文件队列 agent = 唯一可用的持续通道(2026-07-28 端到端验证)
思路:节点上常驻一个轮询循环,从 NFS 读 `.job` 脚本执行、结果写回。login 侧只需 scp 文件。

**`/shared_nfs/kyle/node_agent.sh`**(sbatch 脚本体里直接 `bash` 它,不要 `sleep infinity`):
```bash
Q=/shared_nfs/kyle/q; mkdir -p "$Q"
echo "[agent] $(hostname) starting $(date)" >> "$Q/agent.log"
# 1) 起容器(已存在则复用 —— 换 job 重跑时不会重复 docker load)
if ! docker ps --format '{{.Names}}' | grep -qx gpt-oss-docker; then
  docker images | grep -q gpt-oss-docker || \
    docker load -i /shared_nfs/kyle/images/gpt-oss-docker_kyle-20260807.tar
  docker rm -f gpt-oss-docker 2>/dev/null
  docker run -d --name=gpt-oss-docker --network=host --ipc=host \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -v /shared_nfs/kyle:/workspace/code \
    gpt-oss-docker:kyle-20260807 sleep infinity
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

### 4.-2b ★★同节点多并行队列 = 主队列被队友长任务串行堵住时的正解(2026-07-28 实测)
**症状**:文件队列 agent 是 `while true; for f in $Q/*.job; do bash "$f"; done` —— **单目录 = 严格串行**。
队友(或另一个 agent-team teammate)投了个长 bench(如 `_bench_gptoss_20b.py`,跑几分钟),
你后投的 job 全部排在它后面干等,`.job` 迟迟不变 `.done`。**别误判成"agent 死了"**:
`ls -lat $Q/*.done | head` 看最近完成时间、`tail $Q/<前面那个job>.out` 看它是否还在出数——
还在写 = 正常串行,不是卡死。

**❌ 别为了躲队列去 `sbatch` 一台新节点** —— 白占一台 `--exclusive` GPU 节点 + 多花 ~3min `docker load`,
纯浪费(踩证:2026-07-28 我因此误申请 job 4601,被当场喊停)。节点已经在跑(job 3198 / 328),
容器 `gpt-oss-docker` 已 Up —— 要的是**在这台已起的节点上再挂一个并行队列**,不是换机器。

**✅ 做法:同一节点上再起一个只盯 `q2` 的后台 agent 循环**(容器复用,**不 docker load、不新节点**):
1. `node_agent2_loop.sh`(**只有循环体**,去掉 §4.-2 里的 docker load/run —— 容器已在):
   ```bash
   Q=/shared_nfs/kyle/q2; mkdir -p "$Q"
   while true; do
     for f in "$Q"/*.job; do
       [ -e "$f" ] || continue
       b="${f%.job}"; bash "$f" > "$b.out" 2>&1; echo $? > "$b.rc"; mv "$f" "$b.done"
     done
     sleep 3
   done
   ```
2. **bootstrap job 投进主队列 `q`**(node-ssh/srun --overlap 都是死路,主队列是上节点的唯一通道 →
   bootstrap 这一次仍要过主队列、等它空出一个 slot;此后 q2 就独立并行了):
   ```bash
   # a0_boot_agent2.job —— 名字用 a0 前缀让它在 glob 里排最前,主队列一空立刻先跑它
   mkdir -p /shared_nfs/kyle/q2
   pgrep -f node_agent2_loop.sh >/dev/null && { echo "agent2 already running"; exit 0; }
   nohup bash /shared_nfs/kyle/node_agent2_loop.sh >/shared_nfs/kyle/q2/agent2.boot.log 2>&1 &
   sleep 1; echo "agent2 launched pid=$!"
   ```
   `nohup ... &` 让循环在**节点 host 后台常驻**,bootstrap job 立刻返回、主 agent 继续;q2 循环独立活着。
3. 之后自己的活全投 `q2`,和队友的主 `q` **两个 agent 并行**(各自内部仍串行,跨队列互不阻塞)。

**关键点/坑**:
- **GPU 隔离**:两队列的 job 各自 `docker exec -e HIP_VISIBLE_DEVICES=...` 指**不同卡**,别撞
  (如主队列队友用 GPU2,你 q2 用 GPU4,5)。同容器多 exec 并行没问题,GPU 各占各的。
- bootstrap **必须过主队列一次**(唯一上节点通道);若主队列正卡在长 bench 上,这一次仍得等它跑完。
  想彻底免等只能等那个 slot —— 但一旦 agent2 起来,后续**永久并行**,一劳永逸。
- `pgrep -f node_agent2_loop.sh` 幂等守卫,重复投 bootstrap 不会起第二份。
- 想要 N 条并行队列就 N 个 `qN` + N 个 loop(各自 `Q=` 改掉)。队列目录都在 NFS,跨节点存活。
- ⚠ **区分 §4b 的 sbatch agent(带 docker load,换节点用)vs 本节的 loop-only agent(容器已在,纯加并行度)**:
  换了节点(容器没了)用前者;同节点加队列用后者。

### 4.-2c ★4 套环境的成品传输系统(2026-08-07,已端到端验证)
本仓已把上面的多队列模式落成 **4 套 Crusoe-native 传输系统**,一套一环境,别再手搓 job:
- 核心 `sync/crusoe/lib.sh`(`crusoe_rsh/push/exec`)+ 通用 worker `sync/crusoe/qloop.sh`(部署为 NFS `crusoe_qloop.sh`)。
- 每套薄封装 `sync/<env>/{push.sh,rexec.sh}`,固定该 env 的 venv/独立队列/repo:
  mxfp4→`/opt/venv`+`q_mxfp4`;tensorwise→`/opt/venv-tw`+`q_tw`;syncv3→`/opt/venv-syncv3`+`q_syncv3`;syncv4→`/opt/venv-syncv4`+`q_syncv4`(各自 `<env>/Primus-Turbo`)。
- 用法:`sync/<env>/push.sh [relpath]`(rsync→login NFS,永不 --delete)/ `sync/<env>/rexec.sh [-g GPU] 'cmd'`(命令里 `"$VENV/bin/python"`)。
- `node_agent.sh` 已加 `CRUSOE_ENV_LOOPS` 段,换节点自动重起 4 条队列 loop。
- 完整说明见 `sync/crusoe/README.md`;设计与验证见 [[../../../../.claude/memory/project_syncv4_env]]。
- ★坑:薄封装用 `set -euo pipefail`,`crusoe_rsh` 必须整体 `... || true`(否则 grep 空匹配 / login ssh 抖动的非零会静默 abort);一次性 `qrun.sh` 无 `set -e` 故免疫。

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
   #SBATCH --qos=amd-burst-qos
   #SBATCH -N1
   #SBATCH --exclusive
   #SBATCH -t 24:00:00
   #SBATCH -J kyle_box
   #SBATCH --output=/home/xianzhao/logs/box.%j.out
   sleep infinity
   ```
   `squeue -u xianzhao -o "%i %j %T %N"` 拿节点名(如 crsuse2-m2m-171)。
2. **node-ssh 进节点** → **首选 = 从保存的 tar `docker load`**(§4c 存的 `gpt-oss-docker:kyle-20260807`,flydsl 0.2.2/egg-fix/4 套 venv(含 venv-syncv4)全烤好,免 §6 重装):`docker load -i /shared_nfs/kyle/images/gpt-oss-docker_kyle-20260807.tar`(24G,~3-4min)。**兜底** = `docker pull rocm/primus:v26.3`(节点常已缓存 v26.2/3/4;harbor 代理 `harbor.crusoe.primus-safe.amd.com/proxy/...`),但要再走 §6 装 flydsl。
3. **docker run 起持久容器**(镜像用 tar 的 `gpt-oss-docker:kyle-20260807`;兜底用 `rocm/primus:v26.3`)：
   ```
   docker run -d --name=gpt-oss-docker --network=host --ipc=host --device /dev/dri --device /dev/kfd \
     --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size=64G \
     -v /shared_nfs/kyle:/workspace/code gpt-oss-docker:kyle-20260807 sleep infinity
   ```
   实测容器内 torch 2.10.0+8 GPU + /workspace/code 挂载 OK；ROCm 7.2.1、arch gfx950。
4. 干活 `docker exec gpt-oss-docker bash -lc '...'`。**换节点**：容器在 node-local，节点变了要重 docker load+run(tar/代码在 /shared_nfs,几分钟)。

## 4b. 一条龙起容器脚本(tar 优先,已实测 2026-07-22)✅
引号地狱解法:脚本落 `/shared_nfs/kyle`(login 端 base64 写,node-host 后台跑),全程不嵌引号。
```bash
# ①login 端: 提交占节点 → 拿 NODE
sbatch -A amd-primus -p amd-spur --qos=amd-burst-qos -N1 --exclusive -t 12:00:00 -J kyle_box \
  --output=/home/xianzhao/logs/box.%j.out --wrap="sleep infinity"
squeue -u xianzhao -o "%i %T %N"            # 拿 NODE(如 crsuse2-m2m-301)
# ②login 端: 把下面脚本写到 /shared_nfs/kyle/_load_run.sh(用 base64 -d 落地,避免引号)
#   内容:
#     IMG=gpt-oss-docker:kyle-20260807
#     docker image inspect $IMG >/dev/null 2>&1 || docker load -i /shared_nfs/kyle/images/gpt-oss-docker_kyle-20260807.tar
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
容器是 **node-local**，节点变/容器删就没了。venv-tw/venv-syncv3/venv-syncv4/flydsl 0.2.2/egg 修复/finder/git config 等**改动在容器可写层**(代码在 bind-mount /shared_nfs/kyle 不在镜像；`.so` 也在 repo=共享盘)。**commit/save 在节点 host 跑(不是 docker exec)**：
> **★ 2026-08-07 最新 tar = `gpt-oss-docker:kyle-20260807`**:在 20260720 基础上多烤进 **`/opt/venv-syncv4`**(承载 mxfp4_syncv4 campaign,见 [[../../../../.claude/memory/project_syncv4_env]])。现镜像内共 **4 套 venv**:`/opt/venv`(mxfp4)、`/opt/venv-tw`(tensorwise)、`/opt/venv-syncv3`、`/opt/venv-syncv4`。save 因 docker load 会超 ssh 2min → **必须后台 nohup + 轮询 `.tar` 大小/`.done` 标记**(本 session 用 `sync/crusoe/qrun_host.sh` + `_commit_save.sh` 跑通)。
```bash
# 在节点上(node-ssh 进去,或 nbg 脚本后台):
docker commit gpt-oss-docker gpt-oss-docker:kyle-20260807
mkdir -p /shared_nfs/kyle/images
# ~28G,慢(NFS),后台跑:
nohup bash -c "docker save gpt-oss-docker:kyle-20260807 -o /shared_nfs/kyle/images/gpt-oss-docker_kyle-20260807.tar && echo OK > /shared_nfs/kyle/images/gpt-oss-docker_kyle-20260807.tar.done" >/shared_nfs/kyle/images/save.log 2>&1 &
```
换节点恢复：
```bash
docker load -i /shared_nfs/kyle/images/gpt-oss-docker_kyle-20260807.tar
docker run -d --name=gpt-oss-docker --network=host --ipc=host --device /dev/dri --device /dev/kfd \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size=64G \
  -v /shared_nfs/kyle:/workspace/code gpt-oss-docker:kyle-20260807 sleep infinity
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
  ⚠ 新仓要跑 turbo 相关必须先把整个仓传过去,**含 `.git`**(campaign 要做 git 操作)。2026-08-07 起 NFS 上已有 `syncv3/`、`syncv4/`(mxfp4_syncv4 campaign)等多套 Primus-Turbo。
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
| login/rsync 时好时坏、bench 偶发 rsync 255 崩 round | VIP 后端各自 **flap**(auth 成功后 post-auth 掐断);硬钉单台会随其 flap 崩 | 动态选健康后端:`sync/crusoe/resolve_login.sh`(lib.sh+cursor_campaign 已接入,失败跳后端);在飞老代码进程靠 `refresh_alias.sh`。**别再用 pin_009**。详见 §4.-1b + memory `project_crusoe_dynamic_backend_resolve` |
| squeue 显 R 但节点其实挂了 | R 状态滞后 ~2min(NODE_FAIL) | `squeue -j <id>` 查空 + `hostname` 探活;换活节点重 docker load+run(§4.0) |
| 改 flydsl kernel 后行为没变 | JIT 缓存 | `rm -rf /root/.flydsl/cache` |
| campaign 的整树 `rsync` 挂死几十分钟、0% CPU | **login 节点的 NFS 客户端间歇性卡在 >32 KB 的写**(170 用户 / load avg >200 时高发;小写进 page cache 照样返回,所以 `echo x > f` 骗你说没事)。`dd bs=64k count=4 conv=fsync` 才能测出来。GPU 节点自己那份挂载是好的(实测 125 MB/s) | ①别再整树 rsync,**只推改动**;②文件 >32 KB 就改推 `git diff` 的 gzip+base64(几 KB),塞进 job 文件里,让健康的 GPU 侧 `git apply`(见 `flydsl_campaigns/*/\_fast_remote.py` 的 `patched_run`);③ssh 传输用**短超时+多次重试**(45 s × 25),别用一次 180 s —— 卡住的窗口是间歇的,长超时只会白烧 deadline |
| 杀掉本地 rsync 后远端仍写不进那个目录 | 远端 `rsync --server` 变孤儿,卡着 `.<name>.XXXXXX` 临时文件的 inode;该目录连 `rm` 都会挂 | 在 login 上 `kill -9` 那些 `rsync --server`;临时文件即使删不掉,换一个全新文件名照样能写 |

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
