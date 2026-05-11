---
name: claim-mi355x-node
description: 当原本固定使用的 mi355x 计算节点（如 chi2811）不可用、容器被删、或被别的 slurm job 占满时，从 login_node2 找一台 GPU 实际空闲的 compute 节点，把本机公钥种到节点上，再起一个新的 mlperf_gptoss 容器（rocm/primus:v26.2，挂 /mnt/shared/kyle/code2 → /workspace/code）。当用户说"找台空机器/换节点/容器没了/重建容器/这节点被占了"等场景时使用。配合 remote-mlperf-gptoss 与 remote-sync skill。
---

# claim-mi355x-node

集群里 mi355x 分区大半 down，剩下的节点 slurm 都标 `ALLOCATED`，但**实际 GPU 利用率经常是 0%**（别人的容器只占着显存等请求）。这个 skill 的工作流：经登录节点找一台真闲的，授权自己 ssh，起容器。

## 拓扑回顾

```
本地 (/wekafs/kyle)
  └─ ssh login_node2          149.28.124.225, root, key=/wekafs/kyle/.ssh/id_ed25519
       └─ ssh chiXXXX         登录节点上的 root 可以无密码 ssh 任意 chi 节点（hostbased/共享 known_hosts）
            └─ docker run mlperf_gptoss   # 挂 /mnt/shared/kyle/code2:/workspace/code
```

要点：
- **本机 → 任意 compute 节点的直连**只有目标节点 `~/.ssh/authorized_keys` 里有我们公钥才行。chi2811 之前已经种过；新节点要自己种。
- **从 login_node2 → 任意 chi 节点**不需要密码也不需要我们的公钥（root 在登录节点上有别的 trust）。所以"种公钥"这一步必须**经 login_node2 跳过去**做。
- bind-mount 路径 `/mnt/shared/kyle/code2` 是 **wekafs 上的共享路径**，每个 mi355x 节点都看得到同一份 —— 换节点不丢数据。

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

### 3. 把公钥种到选中的节点

```bash
PUBKEY='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDNk0/FA7DjtcE50WfO1Xf+qGUARLegsXwliz4TU0sPk kyle-20260509'
TARGET=chiXXXX
ssh login_node2 "ssh -o StrictHostKeyChecking=accept-new $TARGET 'mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && grep -qF \"DNk0/FA7DjtcE50WfO1Xf+qGUARLegsXwliz4TU0sPk\" ~/.ssh/authorized_keys || echo \"$PUBKEY\" >> ~/.ssh/authorized_keys; echo done; tail -1 ~/.ssh/authorized_keys'"
```

`grep -qF || echo >>` 这个组合**带去重** —— 重复跑不会写两遍。

立即测直连：

```bash
ssh -o ConnectTimeout=15 chiXXXX 'hostname && whoami'
```

> 如果想后续 `ssh compute_node_new` 还能用旧别名，记得改 `~/.ssh/config` 里 `compute_node_new` 的 `HostName` 指向新节点（默认是 chi2811）。

### 4. 起 mlperf_gptoss 容器

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
  -v /mnt/shared/kyle/code2:/workspace/code \
  rocm/primus:v26.2 sleep infinity'
```

要点：
- `--name=mlperf_gptoss` 沿用原名 —— `remote-mlperf-gptoss` skill 里所有 `docker exec mlperf_gptoss ...` 不用改。
- `-v /mnt/shared/kyle/code2:/workspace/code` 是 bind mount —— 等于把同一份 wekafs 共享数据挂进容器，跟原来 chi2811 上的 `/workspace/code` 完全一样。
- `--network=host` —— 容器直接用 host 网络栈（远程没外网这点不变，但同节点其他服务的端口能直连）。
- `sleep infinity` —— 容器常驻，靠 `docker exec` 进去干活，不要用 `docker run -it` 跑业务。

### 5. 验证

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

1. **更新 `~/.ssh/config`**：把 `compute_node_new` 的 `HostName` 改成新节点，让旧的 `ssh compute_node_new ...` 命令继续生效。
2. **更新 `../remote-mlperf-gptoss/SKILL.md`**：第一段拓扑图的"chi2811"和"远程环境"表的"主机名"行换成新节点。日期也更新到当天。
3. **告诉用户**：新节点 hostname、镜像、GPU 数、bind mount 路径都正常。提示一句"这节点是某 slurm job 的，他可能随时上来跑，长任务前再 `rocm-smi` 复查一下"。

## 不要做的事

- **不要**用 `docker run` 跑用户没审过的 flag 组合（特别是 GPU isolation flag —— 默认 `--device /dev/dri --device /dev/kfd` 是全 8 卡可见）。
- **不要**在 idle 节点上起容器 *并立刻* 跑满 8 卡的工作负载 —— 节点主人(slurm job 拥有者)是优先权，自己先小任务跑通再慢慢上量。
- **不要**清理别人的容器、停别人的进程、删别人的镜像。即使 GPU 是空的，节点不是我们的。
- **不要**用 `srun` / `salloc` 抢 mi355x —— 全节点已 ALLOCATED，slurm 不会让我们进；正确路径就是 docker。
- **不要**修改/删除 chi2811（旧节点）上的别人遗留容器(`vllm-gptoss` 等)。

## 故障排查

| 症状 | 排查 |
|---|---|
| `ssh chiXXXX` Permission denied | 公钥没种成，回到步骤 3。检查 `tail -1 ~/.ssh/authorized_keys` 的输出是不是我们的 key。 |
| 步骤 3 经 login_node2 也卡在 password prompt | login_node2 → 该节点的免密信任不存在（罕见）。`ssh login_node2 'ssh -v chiXXXX hostname' 2>&1 | grep -E "Authentications|Accepted"` 看用了哪种方法。 |
| `docker run` 报 name conflict | 已经有同名容器残留：`ssh chiXXXX 'docker rm -f mlperf_gptoss'`，再重跑。 |
| 容器起来但 `torch.cuda.device_count()=0` | `--device /dev/kfd` 或 `--device /dev/dri` 没挂上。`docker inspect mlperf_gptoss | grep -A2 Devices` 检查。 |
| `/workspace/code` 是空的 | bind mount 路径错了。本应是 host `/mnt/shared/kyle/code2`（不是 `/mnt/shared/kyle/code`）。 |
| `docker images` 里没有 `rocm/primus:v26.2` | 这台节点没拉过镜像。从一个有镜像的节点 `docker save rocm/primus:v26.2 | ssh root@target 'docker load'`，比走外网（没有）靠谱。 |
