---
name: remote-conductor455
description: 连接远程计算节点 heliosp-1b114-c05-3，进入 conductor_455 容器在 /workspace/code 下执行命令。含 Docker 环境快照/恢复、节点稳定性坑。当用户提到"远程/容器/workspace/code/ROCm/HIP/GPU/恢复容器"等远程操作时使用。配合 remote-sync skill：代码在本地编辑，rsync 推到远程再执行。
---

# remote-conductor455

直连远程节点，`docker exec` 进入 `conductor_455` 容器执行命令。容器默认工作目录 `/workspace/code`。

## ⚡ 当前环境（2026-07，用户自己的账号）

之前用的是借用的 `zhuang12` 账号（已弃用，见下方历史）。**现在用用户自己的 `xianzhao` 账号**：

| 项 | 值 |
|---|---|
| 账号 | `xianzhao@heliosp-1b114-c05-3.mnb.dcgpu`（= IP `10.5.229.70`，hostname 有时 DNS 解析不了，**优先用 IP**） |
| 挂载 | host `/home/xianzhao/kyle_code` → 容器 `/workspace/code` |
| 容器 | `conductor_455`（镜像同 `therock-npi@sha256:feba897e...`） |
| 源码构建产物（在挂载卷里，持久） | `/workspace/code/llvm-project/mlir_install`（LLVM 23 @ 7f77ca0d，assertions-OFF）、`/workspace/code/FlyDSL/build-fly/python_packages`（源码版 flydsl）、`/workspace/code/FlyDSL/turbo/shadow_toolkit`（lld wrapper toolkit，跑 flydsl kernel 时 `ROCM_PATH` 指它） |

SSH wrapper 同一个：`/workspace/code/conductor_455/sync/.ssh-helio.sh`（key = `/workspace/code/.ssh_docker/id_ed25519`，这个 key **也是 kyle-256 的 GitHub key**，push fork 用 `GIT_SSH_COMMAND="ssh -i /workspace/code/.ssh_docker/id_ed25519 -o IdentitiesOnly=yes"`）。

跑命令示例（当前账号）：
```bash
cd /workspace/code/conductor_455/sync
timeout -k 3 30 ./.ssh-helio.sh xianzhao@10.5.229.70 "docker exec conductor_455 bash -lc 'cd /workspace/code/FlyDSL && ...'"
```

## 连接链路

```
本地 (/workspace/code/conductor_455/)
  └─ .ssh-helio.sh → zhuang12@heliosp-1b114-c05-3.mnb.dcgpu  （直连，无跳板机）
       └─ docker exec conductor_455
            └─ /workspace/code        ← 默认 cwd
                 ↑
                 └── host bind mount: /home/zhuang12/kyle_tmp
```

- SSH wrapper：`/workspace/code/conductor_455/sync/.ssh-helio.sh`（封装了 key + StrictHostKeyChecking=no）
- 容器常驻（`sleep infinity`），写入的文件下次仍在。
- Host 路径 `/home/zhuang12/kyle_tmp` 即容器内 `/workspace/code` —— rsync 直接推 host 路径，容器立即可见。

## 远程环境（硬件/软件参数，稳定）

> 账号/挂载见上方「当前环境」；下表是硬件+软件栈参数（不随账号变）。

| 项 | 值 |
|---|---|
| 主机名 | `heliosp-1b114-c05-3.mnb.dcgpu`（IP `10.5.229.70`，优先用 IP） |
| GPU | gfx1250（AMD Eng Sample，256 CU / 4 SIMD / wave32；sclk 上限 ~1100MHz 且波动，**性能以 eff 为准**） |
| Python | 3.12.3 @ `/opt/venv/bin/python3`（PATH 已含 venv，无须手动 activate） |
| PyTorch | 2.10.0+rocm7.13.0a20260501 |
| ROCm | 7.13.0a20260501 |
| Triton | 3.6.0+rocm7.13.0a20260501 |
| 镜像 | `therock-npi@sha256:feba897e...` |
| 外网 | **有**（可 pip install / git clone） |
| flydsl | 未装（按需 `pip install -e /workspace/code/FlyDSL`） |

⚠️ **torch import 已知问题**：容器内 `/workspace/pytorch` 是残缺源码树，`import torch` 可能命中它而报错。规避方法：
```bash
PYTHONPATH=/opt/venv/lib/python3.12/site-packages python3 -c "import torch; print(torch.__version__)"
```

## 在容器内执行命令

**SSH wrapper 变量**（简化后续命令）：
```bash
SSH="/workspace/code/conductor_455/sync/.ssh-helio.sh"
HOST="zhuang12@heliosp-1b114-c05-3.mnb.dcgpu"
```

**单条命令**：
```bash
$SSH $HOST 'docker exec conductor_455 bash -lc "cd /workspace/code && <CMD>"'
```

**多行脚本**（heredoc）：
```bash
$SSH $HOST 'docker exec conductor_455 bash -lc "$(cat)"' <<'EOF'
cd /workspace/code
<commands>
EOF
```

**指定 repo**：
```bash
$SSH $HOST 'docker exec conductor_455 bash -lc "cd /workspace/code/FlyDSL && python3 tests/foo.py"'
```

**指定 GPU**（环境变量放 bash -lc 内）：
```bash
$SSH $HOST 'docker exec conductor_455 bash -lc "HIP_VISIBLE_DEVICES=0 python3 script.py"'
```

**长任务**（Bash 工具用 `run_in_background: true`，**不要**在命令里加 `&`）：
```bash
$SSH $HOST 'docker exec conductor_455 bash -lc "cd /workspace/code/FlyDSL && python3 train.py 2>&1 | tee /tmp/run.log"'
```

## /workspace/code 下的子目录

```
/workspace/code/
└── FlyDSL/    # kyle-256/FlyDSL fork，本地镜像在 /workspace/code/conductor_455/sync/FlyDSL/
```

## 约定

- 命令一律用 `bash -lc "..."` —— 确保 venv PATH / ROCm env 加载。
- **不要在远程跑 git 命令** —— git 在本地 `/workspace/code/conductor_455/sync/<repo>/` 操作。
- **不要在远程编辑文件** —— 本地改，rsync 推上去（见 remote-sync skill）。
- **不要动 `/home/zhuang12/` 以外的路径** —— 这是借用账号，其他目录是别人的。
- `docker exec` 报 `No such container` → 先查 `docker ps -a --filter name=conductor_455`，**不要擅自重新 docker run，先问用户**。

## 故障排查

| 症状 | 排查 |
|---|---|
| ssh 卡住 | `.ssh-helio.sh zhuang12@heliosp-1b114-c05-3.mnb.dcgpu 'echo ok'` 验连通性 |
| `No such container` | `docker ps -a --filter name=conductor_455` 查状态，停了就问用户 |
| `python: command not found` | 没用 `bash -lc`，改成 `bash -lc "..."` |
| `import torch` 报错 | 加 `PYTHONPATH=/opt/venv/lib/python3.12/site-packages` 绕过 `/workspace/pytorch` 残缺树 |
| GPU 不可见 | `docker inspect conductor_455 | grep -i kfd` 确认设备挂载 |

---

## 🐳 Docker 环境快照与恢复（重要）

容器**会被删/宿主会重启**（见下方稳定性），所以把配好的容器 commit 成镜像存 tar,丢了能秒恢复。

**当前快照**:`conductor_455:setup`(32GB),tar 在 `/home/xianzhao/kyle_code/conductor_455_setup.tar`(= 容器内 `/workspace/code/conductor_455_setup.tar`)。里面含容器层的全部设置:`pytest`、`cmake ninja patchelf py-spy`、**rocprofv3 的重复参数 bug 修复**等。

**这些是容器层要重装的东西**(镜像本身没有,新建容器要补):
```bash
pip install pytest cmake ninja patchelf py-spy
# rocprofv3 打包 bug: --selected-regions-ref-count 重复注册, 连 --version 都挂。删重复块:
sed -i "760,764d" /opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/bin/rocprofv3
# (flydsl 用源码版 build-fly, 不必装 wheel; 要 wheel 则 pip install flydsl==0.2.2)
```

**重新生成快照**:
```bash
docker commit conductor_455 conductor_455:setup
nohup bash -c 'docker save conductor_455:setup -o /home/xianzhao/kyle_code/conductor_455_setup.tar' &
```

**从 tar 恢复**(容器没了时):
```bash
docker load -i /home/xianzhao/kyle_code/conductor_455_setup.tar
docker run -d --name=conductor_455 --network=host --ipc=host \
  --device /dev/dri --device /dev/kfd --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size=64G \
  -v /home/xianzhao/kyle_code:/workspace/code \
  conductor_455:setup sleep infinity
```
注:tar 是**镜像层**(环境),**不含挂载卷** `kyle_code`(FlyDSL 源码/构建/探针)——那些本就在 host 的 `kyle_code` 持久,`-v` 挂回即可。**tar + kyle_code 目录 = 完整可复现**。

## ⚠️ 节点稳定性(踩过的坑)

- **无 Conductor reservation → 节点被回收/反复重启**:容器 `Exit 137`(SIGKILL,非 OOM),所有用户容器一起挂;宿主 uptime 会重置。**症状**:ssh 卡死超时 或 hostname DNS 解析不了。**解法**:去 conductor.amd.com 确认/申请 reservation(找管理员)。有预约后稳(实测 uptime 10h+)。
- **容器可能被整个删掉**(`docker ps` 报 `No such container`,不是 stopped)——可能是清理。→ 用上面的 tar 恢复。
- **宿主重启但容器只是 stopped**:`docker start conductor_455` 即可(挂载卷数据还在)。
- 探测连通用带超时的:`timeout -k 3 25 ./.ssh-helio.sh xianzhao@10.5.229.70 "echo ALIVE; uptime"`。

## 历史(2026-06,借用 zhuang12 账号,已弃用)

早期用借用账号 `zhuang12@heliosp-1b114-c05-3.mnb.dcgpu`,挂载 `/home/zhuang12/kyle_tmp`。该账号目录已在借用结束时清理删除。当时的铁律:**绝不动 /home/zhuang12/ 以外目录**。现已换成用户自己的 xianzhao 账号,约束放宽。
