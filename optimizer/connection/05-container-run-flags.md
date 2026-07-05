# 起 mlperf_gptoss 容器固定 flag 与验证

> 类别: 连接 · 主题标签: docker, mlperf_gptoss, ROCm, 验证

## 固定 docker run flag(用户审定,别擅改)

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

## 起容器后必做验证(全部通过才算好)

| 验证项 | 预期值 | 失败根因 |
|---|---|---|
| `torch.cuda.device_count()` | `==8`(8 张 MI355X / gfx950) | `--device /dev/kfd` 或 `/dev/dri` 没挂上 |
| `/workspace/code` 非空 | 有代码 | bind mount 路径错,或该节点 `/mnt/vast` 没挂 |
| Python | `3.12.3` | 镜像不对 |
| PyTorch | `2.10.0` | 镜像不对 |
| triton | `3.7.0` | 镜像不对 |

- **任一验证项失败:不要自己改 docker flag,先停下报告用户。**

## 其他连接注意

- 通用带 GPU 的 ROCm 容器交互式起法:`docker run -it --device=/dev/kfd --device=/dev/dri --group-add video --shm-size=64g <IMAGE> bash`(`--shm-size=64g` 同样对多进程/大 tensor 必要)。
- `docker build` 看完整构建日志:加 `--progress=plain`。
- SSH 连接防挂死:`ssh -o ConnectTimeout=30`。

---
来源: claim-mi355x-node/SKILL.md, build-rocm-image/SKILL.md
