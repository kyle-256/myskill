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

- [`remote-mlperf-gptoss/`](remote-mlperf-gptoss/SKILL.md) —— 连接远程计算节点 chi2811，进入 mlperf_gptoss 容器在 /workspace/code 下操作。

## 加新 skill

```bash
mkdir /wekafs/kyle/myskill/<新名字>
$EDITOR /wekafs/kyle/myskill/<新名字>/SKILL.md
```

写完之后告诉 Claude 一声，让它在 memory 里加一条指向新 skill 的索引。

## 为什么不放在 ~/.claude/skills/

放这里是为了跟项目走、能进 git、好分享。代价是 Claude Code 的 `/skill` 菜单不会自动列出来 —— 不过 `/wekafs/kyle` 的 memory 里有索引，进入这个目录的会话会自动加载，效果一样。
