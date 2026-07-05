# docker 根盘清理与 docker save 暂存坑

> 类别: 连接 · 主题标签: docker-disk, image-prune, docker-save, PIPESTATUS

**清根盘(docker 盘)——只清未被引用的镜像**
- 节点根盘(docker 盘)常年高占用,可能只剩个位数 GB。
- 只清"未被任何容器引用"的镜像:`docker image prune -a -f` + `docker builder prune -f`。这两条不碰运行/停止的容器及其镜像。WHY:安全,不会误删别人在用的东西。chi2810 由此腾出 ~107G(来源见 claim-mi355x-node/SKILL.md)。
- 别人的具名(named)镜像不能删。
- `docker commit` 一步可吃掉 ~5GB。
- 删旧镜像前必须先重建容器:运行中容器占用旧镜像 → `docker rmi` 会被拒。

**docker save 暂存撑爆根盘**
- `docker save` 会把整个镜像(~82G)先暂存到 `/var/lib/docker/tmp`。该目录在根盘上,而根盘可能只剩 ~30G → no space。
- 解法:先 `mount --bind /mnt/vast/kyle/code2/docker_tmp /var/lib/docker/tmp`(bind 到共享盘),save 完再 `umount` + `rm`。

**管道退出码核实**
- `docker save | zstd` 管道的 `$?` 取的是 zstd 的退出码;docker save 报错走 stderr,zstd 仍成功 → 易误判成功。
- 用 `${PIPESTATUS[0]}_${PIPESTATUS[1]}` 检查两段退出码,或 `zstd -t` 核实产物完整性。

---
来源: remote-sync/SKILL.md, claim-mi355x-node/SKILL.md
