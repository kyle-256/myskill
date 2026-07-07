# 容器镜像与生产节点

> 类别: 连接 · 主题标签: docker, mlperf_gptoss, ROCm, 验证, container, docker-image, bind-mount, editable-install, prod-node, rocm-version

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
- **最新 tar**:`/mnt/vast/kyle/code2/docker_images/mlperf_gptoss-20260703.tar.zst`
  - 20.3G 压缩 / 93G 解压
  - 含 triton3.7 + flydsl / primus_turbo,含 tensorwise venv `/opt/venv-tw`
  - tag:`mlperf_gptoss:saved-20260703`
- **load 前**:`df -h /var/lib/docker` 确认 >100G 空闲(93G 解压 + 余量),否则 load 失败。
- **为什么一拼即 import**:editable 安装(flydsl / primus_turbo)的 python 指针在镜像 rootfs 里;源码 + build 产物(`build-fly/`、`_C*.so`)在 bind mount `/mnt/vast/kyle/code2` 上。两者一拼即可 import。
- **docker commit 固化范围**:只固化容器 rootfs(pip 包、apt 包、editable 指针、triton 版本);bind mount 上的源码 / build **不进镜像** —— 所以镜像轻、源码始终跟 bind mount 走最新。

## 生产节点/容器/rocm 版本现状

- **当前生产节点**: gfx950/MI355X 当前 = **chi2774**（2026-07-07 起；历史主用 chi2811，更早 chi2810/chi2832）。切换史：chi2811 长期主用 → **2026-07-07 切到 chi2774**，原因：chi2811 8 卡被别人占满（100% / ~290GB 每卡）且我们的容器被 OOM 杀（Exit 137）；chi2810 虽 8 卡空但 docker 盘只剩 37G（无法 load 93G 镜像，占盘的全是别人 tagged 镜像不可清）；chi2774 则 8 卡真空闲 + docker 盘 289G，遂在 chi2774 从 code2 的 tar（`docker_images/mlperf_gptoss-20260703.tar.zst`）`docker load` 起容器。均为单卡稳态测。
- **容器**: 名 `mlperf_gptoss`（镜像 `mlperf_gptoss:saved-20260703`，旧 `saved-20260625b` 是 chi2810 时代镜像已删除）；rocm 版本未在 chi2774/saved-20260703 语境下重新确认（旧源标注 rocm 7.2 是 chi2810/saved-20260625b 时代的数字，不保证仍适用）；venv 在 `/opt/venv`（chi2774 亦有 /opt/venv）。
- **仓库挂载**: bind mount 宿主 `/mnt/vast/kyle/code2` → 容器内 `/workspace/code`（chi2774/chi2810/chi2811 各节点一致，code2 为集群共享 NFS）。
- **连接**: 经 `sync/.ssh-chi.sh` 跳板，如 `sync/.ssh-chi.sh root@chi2774`。
- **指定 GPU**: 用 `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` 环境变量。
- **运行示例**: `docker exec mlperf_gptoss bash -lc "cd /workspace/code/FlyDSL && HIP_VISIBLE_DEVICES=7 python turbo/test_vmono.py M N K"`。
- **WHY/证据**: 旧绝对 TFLOPS 多在被争用节点测，**仅 ratio 可信**，干净节点须重测。

---
来源: claim-mi355x-node/SKILL.md；build-rocm-image/SKILL.md(仅借用其中 `--progress=plain` 与交互式 `docker run -it` 两条通用 docker 用法提示，该 skill 其余内容是构建 rocm-dev-custom:main 通用镜像的完全不同工作流，与本卡片的 mlperf_gptoss 容器无关)；README.md, 09-perf-numbers.md, 13-primus-turbo-prod.md, agpr_phase5_mono.md, project_chi2774_sync.md
