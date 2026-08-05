# 容器镜像与生产节点

> 类别: 连接 · 主题标签: docker, mlperf_gptoss2, ROCm, 验证, container, docker-image, bind-mount, prod-node, 换节点, 根盘容量, saved-tar

## 🚨 本机环境边界（哪个是你的、哪些严禁碰）
> 这是本机(gpt_oss2)环境的**具体边界**——SKILL 的通用「环境红线」在此落地。
- **你的**:容器 **`mlperf_gptoss2`**(带 "2")、host 盘 **`/mnt/vast/kyle/code3`**(→ 容器 `/workspace/code`)、venv **`/opt/venv`**(mxfp8,别名 `/opt/venv-mxfp8`)/**`/opt/venv-mxfp4`**。
- **严禁碰**:容器 **`mlperf_gptoss`**(无 "2") / 盘 **`/mnt/vast/kyle/code2`** —— 那是另一个项目的地盘,对其做任何 rsync / docker exec / 删改都会污染别人的树。同理,节点上别人的容器(`primus_train` / `mtt_pd` / `dbgbuild` / `repro-etcd` / `robust-*` / `yambda_smoke` / `xiaoming-*` / `dsv4-bench` / `dlrmv3-*` / `dazzling_khayyam`)与镜像一律不动。
- **认边界**:容器名有无 "2"、盘 `code2` vs `code3`。跑 campaign/bench 前逐一核对 `--container`/`--remote-host-root`/`--venv` 全部指向 code3 侧。

## 起 mlperf_gptoss2 容器固定 flag 与验证

### 固定 docker run flag(用户审定,别擅改)

```
docker run -d --name=mlperf_gptoss2 \
  --network=host --ipc=host \
  --device /dev/dri --device /dev/kfd \
  --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size=64G \
  -v /mnt/vast/kyle/code3:/workspace/code \
  mlperf_gptoss2:flydsl sleep infinity
```

- **`sleep infinity` 常驻**:后台 `-d` 起来后用 `docker exec` 干活。
- **`--name=mlperf_gptoss2` 沿用固定名**:所有 `docker exec` 命令不用改。
- **`-v /mnt/vast/kyle/code3:/workspace/code`**:⚠️ 别误写 `code2`(别人的)。
- 其余 flag(kfd/dri/video/shm/ipc/ptrace/seccomp)语义同 common 层。

### 起容器后必做验证(全部通过才算好)

| 验证项 | 预期值 | 失败根因 |
|---|---|---|
| `torch.cuda.device_count()` | `==8`(8 张 MI355X / gfx950) | `--device /dev/kfd` 或 `/dev/dri` 没挂上 |
| `/workspace/code` 非空,有 `mxfp8/` `mxfp4/` `flydsl/` | 有代码 | bind mount 路径错,或该节点 `/mnt/vast` 没挂 |
| 两个 venv 都在 | `/opt/venv` + `/opt/venv-mxfp4` | 用了旧 tar(见下) |
| 各 venv import 指对 repo | 见 02-run-and-venv | finder MAPPING 被污染 |

- **任一验证项失败:不要自己改 docker flag,先停下报告用户。**

## saved 镜像与共享盘 tar

- 镜像 **`mlperf_gptoss2:flydsl`**,基于 `rocm/primus:v26.3`(triton 3.7.1 + flydsl 0.2.2 + primus_turbo gfx950)。
- 跨节点持久备份 = 共享盘 tar **`/mnt/vast/kyle/mlperf_gptoss2_flydsl.tar`**(2026-07-07 刷新后 ~80G,**已含 `/opt/venv` + `/opt/venv-mxfp4` 两个 venv**,开箱即用)。fresh 节点首选:
  ```bash
  ./.ssh-chi.sh root@chiXXXX 'docker load -i /mnt/vast/kyle/mlperf_gptoss2_flydsl.tar'
  # 预期最后一行: Loaded image: mlperf_gptoss2:flydsl
  ```
  实测 load 耗时 75s ~ 7min,视层缓存/盘压。
- 容器内改动(`/opt/venv*`、finder 修改、symlink)**必须 `docker commit mlperf_gptoss2 mlperf_gptoss2:saved-YYYYMMDD` 才持久**;bind-mount 的 code3/repo/`.so` 本就在 host 持久。

## 换节点

当前节点写在命令里(本容器无 `~/.ssh/config`,不存在别名 single-source-of-truth),换节点就把 `.ssh-chi.sh root@chiXXXX` 的参数换掉。流程:①登录节点看 `sinfo`/`squeue` → ②经跳板机批量探候选节点(GPU% + docker ps + **根盘容量**) → ③种公钥 → ④`docker load` 起容器。①③见 common/01,②的选卡判据见 common/02。

### ⚠️ 根盘容量是硬约束,和 GPU 空闲同等重要

tar 80G、镜像展开 ~93G ⇒ 候选节点根盘**至少要 ~100G 可用**。探节点时和 GPU 占用一起采集:

```bash
df -h --output=avail / | tail -1          # 期望 >= 100G
docker info | grep -i "root dir"          # 常见 /var/lib/docker,也见过 /mnt/nvme0/docker
df -h <root dir>                          # ⚠️ /mnt/nvme0/docker 往往只是 / 上的普通目录,不是独立盘;
                                          #    节点上 8 块 7T nvme 可能全部未挂载,别被路径名骗了
docker system df                          # reclaimable 大头多是别队未用镜像,不要删
```

**踩证(2026-07-30)**:chi2798 八卡全空、看着完美,但 `/` 只剩 30G(772G/838G 被别人的镜像占满),`docker load` 根本装不下。唯一出路是删别人的镜像 —— 违反环境红线,只能换节点。**先看盘再种公钥,能省一轮无用功。**

## 节点沿革(踩过的坑)

- 每次换/弃节点都清掉**旧节点上我们自己的**容器+镜像;别人的一律不碰。
- mi355x 分区长期基本全 down/drain,可用机多在 **k8s 分区**——`sinfo` 显示 `alloc` 但 GPU 实际常 0%(别人的容器只占显存等请求),所以**别信 sinfo,信 `rocm-smi`**(见 common/02)。
- 历史:chi2810(~07-03 弃) → chi2811(07-03~07-09) → chi2762(07-09~07-15,后不可达,本容器与跳板机均 connection timed out) → chi2798(07-15~07-17) → chi2835(07-17~07-30) → **chi2879(2026-07-30 起,317G 根盘 + 8 卡全空)**。
- 其间 07-07 一度想换 chi2774,但该节点很快被别人 docker 接管(起了 `mlperf_gptoss`),遂换回 chi2811。
- **节点会被别人清空/重置**:2026-07-30 复查发现 chi2835 与 chi2798 上我们的容器+镜像**均已不存在**。所以"上次那台还在"不能假设,每轮先探。
- chi2762 下线后其上残留无法清理(saved tar 在共享盘,无备份丢失)——这正是**把 tar 放共享盘**的价值。

---
来源: claim-mi355x-node/SKILL.md, remote-sync/SKILL.md, 2026-07-30 换节点实测
