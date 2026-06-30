---
name: global-permissions
description: 把 Claude Code 的 permissions allow 规则写入全局 ~/.claude/settings.json，让所有项目目录都免确认执行常用工具（Bash、Read、Write、Edit、Skill、Glob、Grep、WebFetch、WebSearch）。当用户说"加全局权限"、"给你 Bash/Read/Write 权限"、"免确认"或要把已有项目级 permissions 提升为全局时使用。
---

# global-permissions

把 permissions allow 规则写到 `~/.claude/settings.json`（全局用户设置），所有项目共享。

## 适用判断

- **写全局 (`~/.claude/settings.json`)**：用户希望所有项目都免确认 → 用本 skill。
- **写项目 (`<project>/.claude/settings.json`)**：仅当前项目生效 → 不要用本 skill，直接编辑对应文件。
- 用户原话提到"在 X 目录下"但意图是"我以后都不想再点确认"，确认一下是不是要全局；通常答案是"是"。

## 标准 allow 集合

```
Bash(*)
Read(*)
Write(*)
Edit(*)
Skill(*)
Glob(*)
Grep(*)
WebFetch(*)
WebSearch(*)
```

通配 `(*)` 等同于"该工具的所有调用都免确认"。范围很宽 —— 用户既然要全局,通常已经接受这个权衡。如果用户只点名某几个,只加那几个。

## 流程

1. **读现有文件** —— 必须先 `Read /root/.claude/settings.json`（或 `~/.claude/settings.json`）。文件可能已有 `theme`、`model`、其他 permissions 等，**合并不替换**。如果文件不存在，新建。
2. **合并 allow 数组** —— 已有的 allow 条目保留，新条目追加在后面，去重。其他顶层字段（theme 等）原样保留。
3. **Write 整个文件** —— Write 工具会整体覆盖，所以新内容必须包含所有保留字段。
4. **校验** —— `jq -e '.permissions.allow' ~/.claude/settings.json`，退出码 0 且打印数组即成功。退出码非 0 说明 JSON 坏了或路径不对，立即修。
5. **清理误写** —— 如果之前误写到了项目级路径（典型：`/.claude/settings.json`），删掉那个文件和空目录，避免两边不一致。

## 模板

合并后大致长这样（保留原有 `theme` 为例）：

```json
{
  "theme": "dark",
  "permissions": {
    "allow": [
      "Bash(*)",
      "Read(*)",
      "Write(*)",
      "Edit(*)",
      "Skill(*)",
      "Glob(*)",
      "Grep(*)",
      "WebFetch(*)",
      "WebSearch(*)"
    ]
  }
}
```

## 设置加载顺序（避免被覆盖）

`user (~/.claude/settings.json)` → `project (.claude/settings.json)` → `local (.claude/settings.local.json)`，**后者覆盖前者**。如果某个项目目录里也有 permissions 配置且把这些 allow 显式 deny 掉了，全局设置救不了你。出现"加了全局还是问"的情况，先 grep 当前 cwd 下的 `.claude/settings*.json`。

## 常见坑

- **不要用 `mkdir /.claude` 然后写 `/.claude/settings.json`** —— 那是把权限绑到从根目录 `/` 启动 Claude Code 的项目级配置，几乎没人这么用。要全局就写 `~/.claude/`。
- **Write 之前必须先 Read** —— Write 工具要求过；更重要的是不读就写会丢掉文件里其他字段。
- **`Bash(*)` 不会 override `deny`** —— 如果有 `deny` 规则匹配到具体命令，仍然会拦截。
