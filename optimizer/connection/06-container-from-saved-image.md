# 从共享盘 tar 恢复容器(首选)

> 类别: 连接 · 主题标签: container, docker-image, bind-mount, editable-install

- **首选方式**:从共享盘保存镜像恢复容器,开箱即用 —— 跳过升 triton / 装 flydsl 的一整套步骤。
- **最新 tar**:`/mnt/vast/kyle/code2/docker_images/mlperf_gptoss-20260703.tar.zst`
  - 20.3G 压缩 / 93G 解压
  - 含 triton3.7 + flydsl / primus_turbo,含 tensorwise venv `/opt/venv-tw`
  - tag:`mlperf_gptoss:saved-20260703`
- **load 前**:`df -h /var/lib/docker` 确认 >100G 空闲(93G 解压 + 余量),否则 load 失败。
- **为什么一拼即 import**:editable 安装(flydsl / primus_turbo)的 python 指针在镜像 rootfs 里;源码 + build 产物(`build-fly/`、`_C*.so`)在 bind mount `/mnt/vast/kyle/code2` 上。两者一拼即可 import。
- **docker commit 固化范围**:只固化容器 rootfs(pip 包、apt 包、editable 指针、triton 版本);bind mount 上的源码 / build **不进镜像** —— 所以镜像轻、源码始终跟 bind mount 走最新。

---
来源: claim-mi355x-node/SKILL.md
