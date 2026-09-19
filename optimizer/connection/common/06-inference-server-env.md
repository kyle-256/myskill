# 06 — 起推理 server 做端到端对照：环境层怎么搭（与 kernel 线不同的那些）

> 类别: 连接 · 主题标签: sglang, serving, e2e-ab, container, pythonpath, zombie-rank, queue-worker
>
> kernel 线只要一张卡 + 一个 venv；**推理线要一整台机器 + 一个能跑 TP 的容器 + 一套
> 客户端 harness**，多出来的每一层都出过事。做法见 methodology/21，坑见 pitfalls/15。

## 四层，每层都要能独立验证

```
本地容器（agent）──ssh──> 登录节点 ──srun --jobid --overlap──> 计算节点 ──docker exec──> 容器
                                                                              ├─ sglang server (TP4)
                                                                              └─ bench 客户端
```

| 层 | 验证方式 | 出过的事 |
|---|---|---|
| 登录节点 | `squeue -u <user>` | — |
| 计算节点 | `srun --jobid=<J> --overlap hostname` | **登录节点上跑不了 docker**（permission denied on docker.sock），必须 srun 进去 |
| 容器 | `docker ps` | 有的节点 **dockerd 是死的**（socket 是旧的残留）。★ 探针要用 `docker ps`，**不要用 `pgrep dockerd`**——作业命名空间看不到宿主进程，好节点上也读 0 |
| server | `/proc/<pid>/environ` 里的 `PYTHONPATH` | **最凶**：环境丢了也能跑，只是加载了镜像自带的 sglang |

## ★★★★★★ 被测的树 ≠ server 加载的树

镜像自带 editable 安装的 sglang（`/sgl-workspace/sglang`）。要测自己的树必须靠
`PYTHONPATH` 压过去，而**这件事静默失败的方式有好几种**：

```bash
# 验证一（够不够）：解释器能不能解析到被测树
docker exec $C bash -lc "PYTHONPATH=$TREE/python /opt/venv/bin/python \
  -c 'import sglang,os;print(os.path.dirname(sglang.__file__))'"

# 验证二（真正要的）：跑着的 server 进程自己有没有带上
docker exec $C bash -c "tr '\0' '\n' < /proc/$(server_pid)/environ" | grep ^PYTHONPATH=
```

★ **验证一通过 ≠ 验证二通过**。我就是只做了验证一，而 server 的启动命令因为一条注释被截断，
`PYTHONPATH` 从没到过 server 进程——三个"成功"的对照点全在测镜像的代码。

## 起 server：环境变量走 `docker exec -e`

```bash
docker exec -d \
  -e HIP_VISIBLE_DEVICES=0,1,4,5 -e PYTHONPATH="$TREE/python" -e PYTHONNOUSERSITE=1 \
  -w / "$C" bash -c "exec /opt/venv/bin/python -m sglang.launch_server ... > $SRV 2>&1"
```

不要写成 `bash -lc "VAR=... VAR=... python ..."` 那种引号串：那串要同时活过**本脚本的解析器**
和**容器里的解析器**，出错的样子是"server 安静地少了几个环境变量"。`-e` 由 docker 解析。

★ **server 日志文件名带时间戳**：日志由容器里的 root 创建，宿主脚本**截断不了**
（`Permission denied`），残留日志会喂给守门量。

## 僵尸 rank：TP server 起不来的头号原因

反复 kill server 会留下僵尸 rank 占着显存和端口，新 server 卡在进程组初始化：

```
torch.distributed.DistStoreError: Timed out after 601 seconds waiting for clients. 1/4 clients joined.
```

★ **`docker restart` 无效**（容器 PID 1 是裸 `sleep`，不 reap）⇒ 只能 `docker rm -f` + `docker run`
重建容器。重建脚本要能一条命令跑完，并打印 `zombies: N` 自证。

## 文件队列 worker：跑在**宿主机**，不在容器里

job 体里带 `docker exec`，所以 worker 必须在宿主机上跑。

★ 我查 worker 存活时用 `docker exec ... pgrep -c cam_worker` 读到 0，差点判它死了——
**查错了命名空间**。正确是 `srun --jobid=<J> --overlap ps -eo cmd | grep -c <worker>`。

★ worker 要 `setsid` 起，别 `nohup`：`nohup` 挡不住进程组连坐。

## 长命令走队列，不要 `srun ... &`

`srun --jobid=<J> --overlap setsid bash <script> &` 里的后台 step **会随 ssh 会话退出被杀**。
要跑几十分钟的东西，写成 `.sh` 丢进文件队列让 worker 执行。

★ `srun` 在作业繁忙时会排队，命令超时被杀表现为 **Exit 143** ——别把它当成脚本失败。

## 共享节点的边界

* 同一节点上可能有**别人的容器**（名字只差几个字符）。动手前 `docker ps` 看一眼，
  确认自己只碰被指定的那个；判断它有没有占卡看 `rocm-smi` 的显存实数，不是看容器在不在。
* 清理**只按 PID kill**。`pkill -f <pattern>` 会匹配到**调用方自己那条命令行**
  （我用 `pkill -f _moe_knob_sweep.sh` 把自己那条 ssh 会话杀了）。

## 相关

* methodology/21（怎么量：口径、可比性、七道守门量）
* pitfalls/15（harness 的七个静默错与识别指纹）
* `connection/crusoe/`、`connection/smci355/`（具体节点怎么拿）

来源：2026-09-18/19 GLM-5.2 TP4/EP4 AgentX 对照的环境搭建与事故。
