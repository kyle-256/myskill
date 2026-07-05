# 远程访问:跳板机与 SSH

> 类别: 连接 · 主题标签: jump-host, ssh-key, cluster-topology, hostname-resolution

## 跳板机拓扑与 SSH key 种钥

### 三级拓扑

- 链路:本地 → `ssh login_node2`(`149.28.124.225`, root, 跳板机)→ `ssh chiXXXX`(登录节点 root 可无密码 ssh 任意 chi 节点)→ `docker exec mlperf_gptoss`。
- login_node2→chi 节点免密 = hostbased/共享 known_hosts trust。
- **本机直连 `ssh chiXXXX` 通常解析不了 hostname**:chi 节点名只在 login_node2 内网可见。WHY→ 批量扫必须经跳板机:
  ```
  ssh login_node2 "ssh chiXXXX '...'"
  ```

### SSH key 种钥

- 本机→任意 compute 节点直连,需目标节点 `authorized_keys` 里有公钥。
- **种公钥必须经 login_node2 跳过去做**(login_node2→chi 节点有别的免密 trust,本机无法直达 chi)。
- 去重写法(自带去重,重复跑不写两遍):
  ```
  grep -qF <指纹> || echo "$PUBKEY" >>
  ```
- 默认跳板 SSH key = `/workspace/code/.ssh_docker/id_ed25519`(认证 kyle-256)。
- 集群公钥源文件 = `/wekafs/kyle/.ssh/id_ed25519.pub`(kyle-20260509)。

---
来源: SKILL.md
