# 容器镜像与生产节点

> 类别: 连接 · 主题标签: docker, mlperf_gptoss, ROCm, 验证, container, docker-image, bind-mount, editable-install, prod-node, rocm-version

## 🚨 本机环境边界（哪个是你的、哪些严禁碰）
> 这是本机(gpt_oss)环境的**具体边界**——SKILL 的通用「环境红线」在此落地。
- **你的**:容器 **`mlperf_gptoss`**(无 "2")、host 盘 **`/mnt/vast/kyle/code2`**(→ 容器 `/workspace/code`)、venv **`/opt/venv`**(mxfp4)/`/opt/venv-tw`(tensorwise)。
- **严禁碰**:容器 **`mlperf_gptoss2`** / 盘 **`/mnt/vast/kyle/code3`** —— 那是另一个项目 `gpt_oss2_docker` 的地盘,对其做任何 rsync / docker exec / 删改都会污染别人的树。
- **认边界**:容器名有无 "2"、盘 `code2` vs `code3`。跑 campaign/bench 前逐一核对 `--container`/`--remote-host-root`/`--venv` 全部指向 code2 侧。

## 起 mlperf_gptoss 容器固定 flag 与验证

### 固定 docker run flag(用户审定,别擅改)

```
docker run -d --name=mlperf_gptoss \
  --network=host --ipc=host \
  --device /dev/dri --device /dev/kfd \
  --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size=64G \
  -v /mnt/vast/kyle/code2:/workspace/code \
  <image> sleep infinity
```

- **`sleep infinity` 常驻**:容器后台 `-d` 起来后用 `docker exec` 干活,不用 `docker run -it` 跑业务。
- **`--name=mlperf_gptoss` 沿用固定名**:让所有 `docker exec` 命令不用改。
- **`--device /dev/kfd` + `/dev/dri`**:GPU 可见性必需;缺任一 → `device_count=0`。
- **`--group-add video`**:GPU 访问权限。
- **`--shm-size=64G`**:PyTorch 多进程/大 tensor 必要(小了会 OOM/共享内存崩)。
- **`--network=host`**:host 网络;docker build 拉 git 仓库时同样用它确保 `git clone` 走得通。
- **`--ipc=host` / `--cap-add=SYS_PTRACE` / `--security-opt seccomp=unconfined`**:进程间通信 + profiling/调试。
- **`-v /mnt/vast/kyle/code2:/workspace/code`**:bind mount 代码目录。

### 起容器后必做验证(全部通过才算好)

| 验证项 | 预期值 | 失败根因 |
|---|---|---|
| `torch.cuda.device_count()` | `==8`(8 张 MI355X / gfx950) | `--device /dev/kfd` 或 `/dev/dri` 没挂上 |
| `/workspace/code` 非空 | 有代码 | bind mount 路径错,或该节点 `/mnt/vast` 没挂 |
| Python | `3.12.3` | 镜像不对 |
| PyTorch | `2.10.0` | 镜像不对 |
| triton | `3.7.0` | 镜像不对 |

- **任一验证项失败:不要自己改 docker flag,先停下报告用户。**

### 其他连接注意

- 通用带 GPU 的 ROCm 容器交互式起法:`docker run -it --device=/dev/kfd --device=/dev/dri --group-add video --shm-size=64g <IMAGE> bash`(`--shm-size=64g` 同样对多进程/大 tensor 必要)。
- `docker build` 看完整构建日志:加 `--progress=plain`。
- SSH 连接防挂死:`ssh -o ConnectTimeout=30`。

## 从共享盘 tar 恢复容器(首选)

- **首选方式**:从共享盘保存镜像恢复容器,开箱即用 —— 跳过升 triton / 装 flydsl 的一整套步骤。
- **★最新 tar(2026-08-11,含全部 4 个 venv)**:`/mnt/vast/kyle/code2/docker_images/mlperf_gptoss-20260811-flat.tar.zst`
  - 17.8G 压缩 / 103G(export)解压 → 110.5G tar(zstd -t -T0 已验 VERIFY_OK;PIPE=0_0 export/zstd 双段退出 0)
  - 在 20260728 基础上**新增 `/opt/venv-syncv4`**(第4套隔离环境);现含全部 4 venv:`/opt/venv`(mxfp4)+`/opt/venv-tw`(tensorwise)+`/opt/venv-syncv3`+`/opt/venv-syncv4`(均已 verify `bin/python3.12` 在包内)+triton3.7+flydsl/primus_turbo
  - **⚠️ 这是 `docker export` 扁平单层包(不是 save/load 分层包)**:恢复必须用 `docker import`,不是 `docker load`:
    ```
    zstd -dc /mnt/vast/kyle/code2/docker_images/mlperf_gptoss-20260811-flat.tar.zst | docker import - mlperf_gptoss:saved-20260811
    ```
    然后照本卡上方固定 flag 起容器(run 命令显式带 `sleep infinity`+venv 全绝对路径,不依赖镜像 ENV/CMD,故 import 丢元数据无影响)。
  - **为什么 export 不 save**:chi2798 docker 根盘 `/dev/sdb2` 95% 满(仅 44G 空闲),`docker save` 需先把整镜像 ~111G 暂存到 `<data-root>/tmp` 再打包→盘不够;`docker export mlperf_gptoss | zstd -T0 -12` 流式导出 rootfs 直写共享盘 `code2/docker_images`,不占根盘暂存、不碰他人=安全路径(~2.5min)。换到有 ~111G 空闲的干净节点可仍走 `docker save`/`docker load` 分层格式。
  - **★2026-08-11 打包步骤(实操)**:① `docker export mlperf_gptoss | zstd -T0 -12 -f -o <dir>/mlperf_gptoss-20260811-flat.tar.zst.partial`(写 `.partial` 防半包被当成品);② 校 `PIPE=${PIPESTATUS[0]}_${PIPESTATUS[1]}` 须 `0_0`;③ `zstd -t -T0 <.partial>` 须 VERIFY_OK + `tar -tf` 抽检 4 个 venv 都在;④ **校验全过才** `mv .partial → 定名` 并删旧包。export 自动排除 bind-mount `/workspace/code`(code2 源码不进包)。
  - **旧包已删(2026-08-11)**:20260625b / 20260703 / 20260728-flat 三个及 `_save_20260728.log` 均已 rm,目录只留 20260811。⚠️`/mnt/vast/kyle/mlperf_gptoss2_flydsl.tar` 是 **mlperf_gptoss2**(别人的 code3 项目),严禁删/碰。
- **load/import 前**:`df -h /var/lib/docker` 确认 >100G 空闲(解压 + 余量),否则失败。
- **为什么一拼即 import**:editable 安装(flydsl / primus_turbo)的 python 指针在镜像 rootfs 里;源码 + build 产物(`build-fly/`、`_C*.so`)在 bind mount `/mnt/vast/kyle/code2` 上。两者一拼即可 import。
- **docker commit 固化范围**:只固化容器 rootfs(pip 包、apt 包、editable 指针、triton 版本);bind mount 上的源码 / build **不进镜像** —— 所以镜像轻、源码始终跟 bind mount 走最新。

## 🚨 2026-08-11: chi 集群整体失钥(campaign 被打断的实际原因)
- 症状:`chi2835`(当时的生产节点)从超时变成 **`Permission denied (publickey,password)`** —— 节点在线、
  sshd 应答、我们的 key 不再被接受。跳板 149.28.124.225(=chi2866)可登;从跳板用它自己 `/root/.ssh` 下
  **13 把 key 逐一试** chi2834/2835/2836 全被拒;并行扫全部 86 个 chi 节点,**只有 chi2866 与 chi2878 还能登**
  (两台 8 卡 VRAM 都被别人的 CI 占满,且 chi2866 根盘只剩 35G,装不下 103G 的容器包)。
- 判据:`getent hosts chi2835` 正常、ping 0.05ms、jumpbox 的 known_hosts 里**主机 key 是新的**(首次 add)
  ⇒ 重装/换机,`/root/.ssh/authorized_keys` 里种过的公钥没了。种钥要走 login→chi 的信任,而这条信任也没了。
- `kubectl`(这些节点在 k8s partition)需要 **OIDC 交互登录**,agent 环境里过不去。slurm 视图里全 `down*`。
- ⇒ **应急落脚点 = smci355(直连,`.ssh_laptop` key),见 `../smci355/01-node-setup` §7b**:那边有 8 张空闲
  MI355X,且用自带已编译 turbo 的镜像可在 ~20min 内跑通同一份 bench(不用 build csrc)。
  换节点后 **base/best 必须重测**(methodology/16 已有此条),只有同节点 A/B 可比。

## 生产节点/容器/rocm 版本现状

- **当前生产节点**: gfx950/MI355X 当前 = **chi2762**（2026-07-08 从 chi2811 迁移——chi2811 被 atom-bench 等占满 8 卡 100%，落节点选卡见 `../common/02-pick-free-gpu`；更早 chi2811/chi2810/chi2832）。落节点前务必按 02-pick-free-gpu 核实占用与 docker 盘容量：历史上 chi2810 曾 docker 盘只剩 37G/0G 无法 load 93G 镜像、8 卡被别人占满导致容器 OOM 杀（Exit 137）等坑。镜像来自 code2 的 tar（**当前 = `docker_images/mlperf_gptoss-20260811-flat.tar.zst`，17.8G 压缩/103G 解压，扁平包用 `zstd -dc <tar> | docker import - mlperf_gptoss:saved-20260811`**；盘不够先 `docker image prune -af` 清未用镜像）。均为单卡稳态测。
- **迁移到新节点的完整流程**（2026-07-08 chi2762 实操验证）：① `.ssh-chi.sh root@<node>` 确认 code2 已挂 + `rocm-smi --showpids` 8 卡 0 pid；② 盘不够 `docker image prune -af`（chi2762 一次回收 245G）；③ **扁平包 import**:`zstd -dc /mnt/vast/kyle/code2/docker_images/mlperf_gptoss-20260811-flat.tar.zst | docker import - mlperf_gptoss:saved-20260811`（不是 `docker load`）；④ 固定 flag `docker run -d --name=mlperf_gptoss ...`（见本卡上方）；⑤ 验证容器 `device_count==8`/code2/venv-tw；⑥ campaign 改 `--host root@<node>`（REPO_REMOTE_HOST 在 code2 共享 NFS 上，路径不变）后 resume。
- **容器**: 名 `mlperf_gptoss`（镜像 `mlperf_gptoss:saved-20260703`，旧 `saved-20260625b` 是 chi2810 时代镜像已删除）；rocm 版本未在 chi2811/saved-20260703 语境下重新确认（旧源标注 rocm 7.2 是 chi2810/saved-20260625b 时代的数字，不保证仍适用）；venv 现有 4 套:`/opt/venv`(mxfp4) + `/opt/venv-tw`(tensorwise) + `/opt/venv-syncv3` + `/opt/venv-syncv4`(均在 20260811-flat 包内)。
- **仓库挂载**: bind mount 宿主 `/mnt/vast/kyle/code2` → 容器内 `/workspace/code`（chi2762/chi2811/chi2810 各节点一致，code2 为集群共享 NFS；换节点无需重新 sync 远端 repo）。
- **连接**: 经 `sync/.ssh-chi.sh` 跳板，如 `sync/.ssh-chi.sh root@chi2762`（当前节点；跳板 149.28.124.225 内网可解析任意 chiXXXX）。
- **指定 GPU**: 用 `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` 环境变量。
- **运行示例**: `docker exec mlperf_gptoss bash -lc "cd /workspace/code/FlyDSL && HIP_VISIBLE_DEVICES=7 python turbo/test_vmono.py M N K"`。
- **WHY/证据**: 旧绝对 TFLOPS 多在被争用节点测，**仅 ratio 可信**，干净节点须重测。

---
来源: claim-mi355x-node/SKILL.md；build-rocm-image/SKILL.md(仅借用其中 `--progress=plain` 与交互式 `docker run -it` 两条通用 docker 用法提示，该 skill 其余内容是构建 rocm-dev-custom:main 通用镜像的完全不同工作流，与本卡片的 mlperf_gptoss 容器无关)；README.md, 09-perf-numbers.md, 13-primus-turbo-prod.md, agpr_phase5_mono.md, project_chi2811_sync.md
