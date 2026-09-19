# 15 — 自建 e2e harness 的七个「不报错、只给错数」

> 一夜实测：14 小时里前 9 小时**零有效数据**。七个 bug 没有一个抛异常，
> 每个都让 bench 跑满全程、写出结果文件、读数看着完全正常。
> 这张卡按**识别指纹**组织——因为发现它们靠的不是报错，是对不上的小细节。

## ★★★★★★ 总纲：铺矩阵前先做一次单点端到端干跑

我先铺了 12 个点才验证单点。**120 秒的冒烟恰好绕过了全部七个**——它没有并发臂、
没有 flag 臂、没有长轮询、没有重启循环。

⇒ **冒烟要覆盖的是"矩阵里最复杂的那个点"，不是最简单的那个**：
至少一个带 flag 的臂 + 一次臂间切换 + 一次完整轮询，哪怕窗口只有 120 秒。

## ① 注释插进命令的行接续链 ⇒ 环境变量被丢掉

**最贵的一个。**

```bash
docker exec -d $C bash -lc "
  ... AITER_MOE_FORCE_UNSHUFFLE=1 \
  # 这行注释被上一行的 \ 接了上来
  PYTHONPATH=$TREE/python \
  ... python -m sglang.launch_server ..."
```

反斜杠把注释拼成同一条语句 ⇒ **命令在注释处终止**。前半条只剩一串赋值（无命令），
`PYTHONPATH` 落在被丢弃的那半 ⇒ **server 安静地 import 了镜像自带的 sglang**。

一行复现：
```bash
bash -lc 'A=1 \
B=2 \
# 注释
C=3 \
env | grep -E "^(A|B|C)="'      # 只打印 C=3
```

**指纹**：三个"成功"的臂读数极其接近（因为跑的是同一份代码）；server 启动比平时快。
**修法**：环境变量走 `docker exec -e`（由 docker 解析，没有 shell 参与）；
命令串里**一条注释都不要**，说明写在 `docker exec` 那行之上。
**守门量**：读 `/proc/<server_pid>/environ`，`PYTHONPATH` 必须含被测树。

## ② 被取消的臂脚本睡满轮询后醒来，杀掉下一轮的 server

臂脚本的等待循环是 `for i in $(seq 1 90); do sleep 40` = 整 3600 秒。
取消矩阵时漏掉一个实例，它睡满后执行收尾的 `stop_server`，而那时容器里跑的
已经是**下一轮的 server**——`stop_server` 的实现是"杀容器里所有 launch_server"，不分敌我。

**指纹**：`bench rc=1`，但**错误率只有 1.96%（门槛 10%）看着正常**；
真正的 fatal 是 AIPerf 的 `ProfileMetricCoverageError`（TTFT/ITL 覆盖率 91.6% < 95%）；
错误摘要里 27 次 `ConnectionRefusedError` 打到服务端口；
server 日志里 `SIGTERM received` 出现在 bench 结束前 6 分钟。

**修法**：按**所有权**杀，不按 pattern 杀。锁文件写 `tag-conc-PID`，
`stop_server` 不持锁就 return；开跑前锁被**活着的**进程持有就退出（`kill -0 $holder_pid` 判死锁，
否则崩溃留下的锁会永久挡路）。
★ 详见 `[[reference_arm_script_kills_live_server]]`。

## ③ `rx "grep -c 'X' file"` —— 嵌套单引号让轮询永不成立

远程执行封装是 `ssh ... "bash -lc '<命令>'"`，**内层再用单引号会把外层提前闭合**。
grep 收到坏参数，轮询判据永远为假 ⇒ 每个点耗满 `200 × 40s ≈ 2.2 小时`才肯往下走。

**指纹**：**不报错，只是慢**。靠"臂在 18:16 就 DONE 了，驱动到 18:30 还没记 finished"
的时间差才发现。12 个点会变成 26 小时。

**修法**：内层用双引号（`bash -lc '...'` 里的 `"` 是安全的），
或把命令写成 NFS 上的 `.sh` 文件调用。**改完要实测一次判据本身**（我现在会单独跑
`grep -c "ARM base c8 DONE" <log>` 看它回 1，而不是改完就信）。

## ④ `grep -c ... || echo 0` 产出两行，整数比较全线崩

`grep -c` 在无匹配时**既打印 `0` 又 exit 1** ⇒ `|| echo 0` 再补一行 ⇒ 变量成了 `"0\n0"`。

**指纹**：`[: 0\n0: integer expression expected` 混在几百行输出里；
依赖它的**守门量静默失效**（我那道 flag 硬门因此从没触发过）。

**修法**：`grep -c ... 2>/dev/null | head -1`，再 `VAR=${VAR:-0}`。`pgrep -c` 同病。

## ⑤ readiness 只看端口 ⇒ 连上了上一轮正在排空的 server

**指纹**：**TP4 server 106 秒就 "ready"**，而正常冷启动要 8~10 分钟。

**修法**：readiness 要两个条件——**本轮自己的** server 日志出现 `Load weight begin`
**且**端口应答。日志文件名带时间戳，天然只属于本轮。

## ⑥ 上一轮的 `DONE` 标记被当成本轮完成

驱动轮询臂日志里的 `ARM <tag> c<N> DONE`，而臂日志没被清 ⇒ 点在 30 秒内"完成"。

**修法**：驱动**投递前**先删臂日志。
★ 同族：任何"靠日志里某个字符串判完成"的机制，都要保证那个字符串**只可能来自本轮**。

## ⑦ 往别人的 harness 加参数前没 grep

给 `REPLAY_CMD` 加 `--random-seed`，而上游 9 行之上已经有 `--random-seed 42`
⇒ `Parameter --random-seed specified multiple times`，两个点各白跑 5 分钟。
**顺带**：我据此错误地下了"默认用系统熵、每轮重放不同"的结论并写进了报告，
实际上种子一直是固定的，负载分叉另有原因（见 methodology/21 §闭环）。

**修法**：`grep -n <参数名> <脚本>` 再动手。

## ★★★★★ 另外两个同类（这轮也踩了）

* **容器里的 root 建的日志，宿主脚本截断不了**（Permission denied）⇒ 残留日志喂给守门量。
  修法：每轮用带时间戳的新文件名，不要截断。
* **`pgrep -f <脚本名>` 做驱动级互斥恒真**：调用方自己的命令行（`setsid bash xxx.sh`、
  agent 的 bash 包装）也含那个字符串。**该上锁的是被争抢的资源（server），不是驱动进程。**

## ★★★★★★ 三个 09-19 新踩的（都不报错，或报得完全不像根因）

### ⑧ 在脚本**跑着**的时候编辑它 ⇒ 它崩在收尾，而正事已经做完了

bash 是**边读边执行**的：改文件会让正在执行的 shell 的字节偏移错位，它接着把乱码当代码跑。
我的臂因此死在 `line 103: unexpected EOF`——**`bench rc=0`、结果已写盘**，但
`stop_server` 和 `rm $LOCK` 没跑，于是 server 占着 8 张卡、锁还挂着，下一臂起不来。

**修法**：跑之前 `cp` 成一次性冻结副本，跑副本。源文件随便改。

```bash
R=/shared_nfs/kyle/_run_${TAG}.sh; cp "$SRC" "$R"
setsid bash "$R" ... </dev/null >/dev/null 2>&1 & disown
```

### ⑨ NFS 文件队列：`rm` 掉 worker 已持有的 `.out` ⇒ 新投的 job 被改名成 `.done`，一次没跑

时间线（worker 的循环是 `bash $f > $b.out; echo $? > $b.rc; mv $f $b.done`）：

1. 坏 job 被 worker 捡走，`.out` 已打开
2. 我 `rm` 了 `.job/.out/.rc` —— **unlink 掉 worker 持有的 fd，输出进了已删除的 inode**
3. 我 `mv` 出新 job（同名）
4. 坏 job 跑完，worker 写 rc，然后 **`mv` 的是那个还没跑的新 job** ⇒ 直接变 `.done`

**指纹**：`.rc` 存在但 **`.out` 是空的或根本不存在**，而 `.done` 的内容是你的新版本。
**修法**：重投**永远用新文件名**，不要覆盖 worker 可能已持有的名字。

### ⑩ 登录节点写不了容器 root 写过的 `.git/objects`

容器内以 root 跑过 `git checkout` 之后，登录节点的普通用户再做 git 写操作会报
`error: insufficient permission for adding an object to repository database .git/objects`。
注意它**报错前一行是 `Auto-merging ...`**，很容易被当成合并冲突。
**修法**：对这类共享仓库，git 写操作统一在容器里做。

## 取证顺序（下次直接照这个走）

1. `bench rc` 非 0 ⇒ 先找 **harness 自己的 fatal 原因**（覆盖率 / 校验错），别只看错误率
2. 有 `ConnectionRefused` ⇒ 去 **server 日志搜 `SIGTERM received`**，定位精确时刻
3. 看 SIGTERM 前一两行是谁在访问（`/server_info` = 有别的 bench 在做前置检查）
4. 读数"正常但都一样" ⇒ **查 server 进程的 environ**，确认它加载的是被测树
5. server 起得异常快 ⇒ 八成连的是上一轮的进程
6. 守门量恒为 0 ⇒ 先验**守门量本身**（单独跑那条 grep），再怀疑被测功能

来源：2026-09-18/19 GLM-5.2 TP4/EP4 AgentX 四臂对照。
配套做法见 methodology/21。
