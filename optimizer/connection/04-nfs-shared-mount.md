# 集群共享 NFS 挂载 /mnt/vast/kyle/code2

> 类别: 连接 · 主题标签: NFS, 集群共享, 挂载, 节点弃用

- **主用挂载**: bind-mount `/mnt/vast/kyle/code2` = `10.2.123.177:/aac-8634674/aac/shared/data`,152T 集群共享 NFS。每个节点同源,**写一次全集群可见**,换节点不丢数据/代码/tar。
- **已弃用 /mnt/shared**: 旧路径 `/mnt/shared`(= `login_node2:/mnt/nvmeraid`)在空闲节点上是坏/空 NFS,root 都 `Permission denied`。弃用。
- **节点级弃用坑**: 某些节点(如 **chi2761**)根本路由不到 aac NFS 服务器 → `10.2.123.177 No route to host`。这类节点上没代码、没 tar,直接弃用换节点。WHY: 路由不通,共享挂载失效,无法拉取工作目录。

---
来源: claim-mi355x-node/SKILL.md
