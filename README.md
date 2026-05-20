# myskill

`/wekafs/kyle` 下的 skill 集合。每个 skill 一个子目录，里面放一个 `SKILL.md`。

## 目录结构

```
/wekafs/kyle/myskill/
├── README.md                          # 本文件
└── <skill-name>/
    └── SKILL.md                        # 必需：带 frontmatter 的说明
    └── ...                             # 可选：辅助脚本、模板等
```

## SKILL.md 格式

```markdown
---
name: skill-name              # 短横线小写
description: 一句话说清楚什么时候用、做什么。Claude 靠这行决定要不要打开 SKILL.md。
---

# skill-name

正文：连接方式、约定、命令示例、故障排查……
```

## 现有 skill

- [`remote-mlperf-gptoss/`](remote-mlperf-gptoss/SKILL.md) —— 连接远程计算节点（当前 chi2761），进入 mlperf_gptoss 容器在 /workspace/code 下操作。
- [`remote-sync/`](remote-sync/SKILL.md) —— 在本地 `/wekafs/kyle/code2/remote_sync/{Primus-Turbo,HipKittens}` 编辑代码，rsync 推到远程 host 路径再执行（远程无外网用的工作流）。
- [`gpu-fleet-tuning/`](gpu-fleet-tuning/SKILL.md) —— N 个 GPU + N 个 sub-agent 做 kernel/config 调优的事件驱动调度模式（GPU 永不空闲，状态文件持久化，broad → refine → diversify → done）。
- [`mi300-blockwise-gg-tuning/`](mi300-blockwise-gg-tuning/SKILL.md) —— MI300X 上 Triton blockwise FP8 grouped GEMM 调优的硬约束、已验证 priors、和 tensorwise 公平对比方法、常见坑（fwd persistent + bwd variable-K 两条路径都覆盖）。
- [`global-permissions/`](global-permissions/SKILL.md) —— 把 permissions allow 规则合并写入 `~/.claude/settings.json`，所有项目共享免确认（Bash/Read/Write/Edit/Skill/Glob/Grep/WebFetch/WebSearch）。
- [`claim-mi355x-node/`](claim-mi355x-node/SKILL.md) —— 原 mi355x 节点不可用时，经 login_node2 找一台 GPU 真闲的 compute 节点（sinfo/squeue + rocm-smi 巡检）、种公钥、起 `mlperf_gptoss` 容器（rocm/primus:v26.2，挂 /mnt/shared/kyle/code2）。

## 加新 skill

```bash
mkdir /wekafs/kyle/myskill/<新名字>
$EDITOR /wekafs/kyle/myskill/<新名字>/SKILL.md
```

写完之后告诉 Claude 一声，让它在 memory 里加一条指向新 skill 的索引。

## 为什么不放在 ~/.claude/skills/

放这里是为了跟项目走、能进 git、好分享。代价是 Claude Code 的 `/skill` 菜单不会自动列出来 —— 不过 `/wekafs/kyle` 的 memory 里有索引，进入这个目录的会话会自动加载，效果一样。
