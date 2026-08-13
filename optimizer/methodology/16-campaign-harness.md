# Campaign 编排器 — 怎么跑一场自动优化 campaign(起 / 盯 / 救 / 验收)

> 类别: 方法论 · 主题标签: campaign, flydsl_campaign, cursor_campaign, orchestrator, resume, supervisor, bench 契约, 监控, 验收
> 状态: 2026-07-29 从 bwd16384(30 轮 +12.89%)、fwd16384(进行中 +14.3%)、gptoss-sbhd-bwd 三场实跑沉淀。✅=实测过。
> 适用:`sync/flydsl_optimizer/{flydsl_campaign,cursor_campaign}.py`。**动手前整卡读一遍**,坑几乎全在 §6。

---

## 0. 两个版本,选哪个

| | `flydsl_campaign.py` | `cursor_campaign.py` |
|---|---|---|
| agent 层 | `claude -p`(Claude CLI) | `cursor-agent -p`(headless) |
| 相位/记忆/harness/supervisor/squash | ——— **完全相同** ——— | |
| 计费 | 每轮 USD 预算硬上限 | Cursor credits,`--*-budget` **只是建议值**,日志里的 $ 是估算 |
| effort | `--*-effort` 直接传 | ★**Cursor 把 effort 烤进模型名**:`claude-opus-5-thinking` + `-xhigh` → `claude-opus-5-thinking-xhigh`。合法组合用 `cursor-agent models` 查 |

Claude 预算紧就用 cursor 版,行为一致。**两者都从不 push**,所有产物留在本地 `flydsl_campaigns/<ts>/`。

---

## 1. 一场 campaign 长什么样

三相位(规模随 `--rounds` 缩放,30 轮是默认形状):

- **ANALYZE**(第 1..A 轮,A∈[`--analyze-min`, `--analyze-max`]):只采数据、读 KB、写 `goal.md`,**不留 kernel 改动**。
- **OPTIMIZE**(中间大部分轮):每轮**只做一个改动** → orchestrator 用 `--bench-cmd` 打分 → supervisor 裁决 → 给下一轮 directive。
- **REVIEW**(最后 `--review-rounds` 轮):按 code-review 方法论修,然后把**所有 kept commit squash 成一个**(作者 kyle-256,无 AI 署名)。

每轮的判定链:
- 候选跑 `--repeat` 次(默认 3),**取 MAX**(理由:选最少被节流的一次)→ 与 `best_score` 比。
- 涨幅 ≥ `--keep-threshold`(默认 0.5%)才 **KEPT**(commit 成新 best);否则 "no new best" 但**保留工作副本继续迭代**。
- 连续 `--revert-patience` 轮没新 best → 工作副本回滚到 best(给"先降后升"的重写留空间)。
- 连续 `--stall-window`(5)轮累计涨幅 < `--stall-threshold`(2%)→ 插一轮 **REPLAN** 重写 goal。
- 连续 `--max-round-crashes`(3)轮抛异常 → 停止(进度与 kept commit 保住)。
- **supervisor 每轮都跑,且被硬性禁止给否定/天花板结论**("不行/不可能/到上限" 会被拒绝并重问)。

---

## 2. 起一场:launcher 脚本模板

**别在命令行手敲**,写成 `_launch_<tag>.sh` 存 `flydsl_optimizer/`(可复现、resume 时复用同一份参数)。

```bash
cd /workspace/code/gpt_oss_docker/sync/flydsl_optimizer
exec python3 cursor_campaign.py \
  --repo mxfp4/Primus-Turbo \                        # 相对 sync/ 的本地 checkout
  --target primus_turbo/flydsl/attention/flash_attn_fwd.py \   # 主攻文件(非白名单,可改别的实现文件)
  --goal "$(cat goal_fwd16384.txt)" \
  --tag fwd16384 \                                   # 决定 CURRENT_<tag> 指针,并发 campaign 不撞
  --bench-cmd _bench_campaign_fwd16384.py --score-key tflops \
  --rounds 30 --analyze-min 2 --analyze-max 3 --review-rounds 2 \
  --min-round-sec 1200 --revert-patience 15 --repeat 3 \
  --deep-model claude-opus-5-thinking --deep-effort xhigh --deep-budget 120 --deep-timeout 7200 \
  --optimize-model claude-opus-5-thinking --optimize-effort xhigh --optimize-budget 120 \
  --supervisor-model claude-opus-5 --supervisor-effort high \
  --review-model claude-sonnet-5-thinking --review-effort xhigh \
  --remote-backend chi --host root@chi2798 \
  --remote-container-path /workspace/code/Primus-Turbo \
  --remote-host-root /mnt/vast/kyle/code2 \
  --venv /opt/venv --container mlperf_gptoss \
  --gpu 6 --gpu-pool 6,7 --gpu-free-vram-mb 200000 \
  --bench-timeout 2400 --profile-timeout 2400 \
  "$@"                                               # ← 必须有,才能 --resume
```

起法(detach,断连不死):
```bash
cd .../flydsl_optimizer && setsid env IS_SANDBOX=1 ./_launch_fwd16384.sh \
  > /tmp/fwd_launch.log 2>&1 < /dev/null & disown
```
`IS_SANDBOX=1` 是 claude-under-root 的要求;cursor 版不需要但带着无害。

🚨 **环境红线**:`--container` / `--remote-host-root` / `--venv` 三者必须全指向**你自己的**环境。脚本里有一道硬 guard 会拒绝 `mlperf_gptoss2` / `code3` / `venv-mxfp4`(那是 gpt_oss2 的),但**只挡这三个名字**,别指望它兜住所有误配。见 SKILL.md 顶部红线 + `connection/`。

`--remote-backend`:`chi` = ssh 经跳板 + `docker exec`;`crusoe` = scp 一个 job 到 login 的 NFS 队列让节点 agent 跑(见 `connection/crusoe/01` §4.-2)。**优先 chi** —— crusoe 的 `--exclusive` 分配到期后重排会卡 `QOSGrpNodeLimit`(实测排 9 小时没排到,campaign 全程停摆)。

---

## 2b. ★goal 怎么写(死路清单决定你会不会白烧一轮)

goal 里最贵的一节不是目标定义,是**死路清单**。agent 只知道你写进去的东西 —— 它会认真读 KB,
但不会自动去翻你这条分支的祖先都判负过什么。

- ★★**死路清单必须覆盖 base 分支自己的历史判负,不只是当前 kernel 的**。踩证:mxfp4gg campaign
  的 r8 花 **130 分钟做 NT persistent tile-loop,做完自己回滚,整轮零产出**(`agent.txt` 0 字节,
  判负经验连 memory.md 都没进)。而 base 分支(mxfp8 grouped)对 persistent 是**两次独立判负**
  (NT 0.70–0.81×),白纸黑字记在 pitfalls/05 —— 只是我写 goal 时没抄进去。**这一轮是 goal 的锅,
  不是 agent 的。** 起 campaign 前把 base 分支对应的 memory 卡通读一遍,把「已 measure-closed」
  整段抄进 goal 第三节。
- **把"我不知道"和"已判负"分开写**。前者是让 agent 去做的,后者是让它别做的。含糊其辞的
  「这条可能没什么用」会被当成前者。
- **给两三条 campaign 特有的具体线索**,比泛泛的"提升性能"有用得多。踩证:同一场 campaign 里
  我写了「wgrad 1.6× 而 NT 只有 1.1×,同一份计算体,差异在哪」和「down 窄 N 恒为 min」两条,
  r1 一轮就把它们**结清成同一个根因**(NT prologue 的 O(G) 串行扫描),直接兑现 +9.4%。
- **写清楚"真验收"和"score-key"不是一回事**。score 是给 orchestrator 排序用的(通常 geomean),
  验收往往是 min/全过。不点破的话,agent 会一路优化 geomean,而 geomean 会被容易的那几格拉高。
- ★★**对标类 campaign(「打赢某个具体 kernel」):在 goal 里直接命令「第一轮先 dump 标尺的 ISA 与
  dispatch 元数据」。** 踩证(mx8tw):十七轮里前十五轮只测自己,r16 才第一次拆标尺,一拆就发现
  「我们在用 8-wave 几何打一个 4-wave/1-wave-per-SIMD/256-AGPR 的核」,整个打击面当场换掉 ——
  而标尺的结构其实早就写在 KB 里(methodology/06 §4-wave whole-loop),**只是没人被要求去读**。
  agent 会默认「优化 = 改自己的核」,不会主动去读对手。**这一句话值十五轮。**
- ★**短板会在 campaign 中途换位置,goal 里要写「先跑一遍全表自己确认 min 在哪」**。同一场里 min
  从 `wgrad gate_up heavy`(0.776)迁到了 `fwd down balanced`(0.906),而 goal 第二节还写着
  「主攻 wgrad」—— 沿用旧结论会让整轮打在已经不是短板的地方。
- ★**别被 ragged/skew 格的高 ratio 误导**:那三格 ratio 好常常是**标尺在 skew 上更弱**(标尺
  down balanced 2997 TF → heavy 2795,而我们自己是平的),不是我们在 skew 上更强。
  **攻势要瞄准标尺最快的那一列。**

---

## 3. ★bench 契约(整条链的地基,写错了后面全白干)

`--bench-cmd` 指向的脚本必须:

1. **最后一行 stdout 是 JSON**,含 `ok`(bool)与 `--score-key`(数值)。其余输出随便打。
2. **正确性门写在 bench 里**:SNR ≥ 阈值 **且** 逐字节确定性 → `ok`。`ok=false` 的候选无论多快都不计分。这是唯一防"改坏换性能"的闸。
3. **热稳态计时**:连续满载预热(**不要 sleep**),再取中位数。
4. ★★★**只计分一个 shape,campaign 就会拿别的 shape 去换分,而且日志里看不见**。踩证:fwd campaign 只计分
   full-causal S=16384,某一轮的 GQA-merge 买到 full-causal +1.8%、却让同形状 **SWA −12.2%**,而出货比例是
   3 SWA : 1 full-causal —— 按混合口径那一轮是净亏。campaign 全程 `ok=true`、分数一路涨,完全没有信号。
   ⇒ **收官必须跑一遍真实出货口径的验收表**(全部形状 × 全部 mask 模式,按出货比例混合),
   回退/门控掉那些"只对计分 shape 有利"的改动;能按 trait 门控就门控(如 `window_left < 0`),不必整轮回退。
5. ★★**必须 mirror 部署配置**。踩证:bwd campaign 的 bench 直接调 builder 默认值,和生产 `_get_bwd` 的 `block_kv`/`waves_per_eu` 不同 —— 于是它自报 789.6→891.4(+12.89%),而按部署配置测的方阵表在同一份代码上是 880,**同一份 kernel 两把尺差 1.3%,且基线差得更多**。bench 不 mirror 部署,campaign 就是在优化一个不出货的配置。

6. ★★**分数是「对某个标尺的比值」时,标尺必须冻结 —— 而且要在 goal 里写死禁止改哪些文件**。
   把标尺(另一个 kernel)merge 进同一棵树、同进程交错计时,是消除 DVFS 漂移的正确做法
   (mxfp4gg 和 mx8tw 两场都靠它),但它带来一个新风险:**agent 能改到标尺**。
   实际发生过两次,性质不同,都要防:
   - **良性但仍需盯**:把 `_wgrad_split_policy` 里硬编码的 `6` 改成 `slice_floor=6` 默认参数,
     标尺侧调用点不传新值 ⇒ 行为逐字节不变。这种是合法扩展,**验收时要逐调用点确认默认值路径没变**。
   - **会毁掉可比性**:动 `_grouped_block_mn` 这种**两边共用的 helper**(把 `ceildiv` 换成
     `ceildiv_pow2`)。数值等价、agent 也没有作弊动机(标尺变快反而压低比值),但**基线在 campaign
     中途漂了**,前后轮次的分数不再可比。
   ⇒ goal 里直接列出**标尺文件与共享 helper 的白名单/黑名单**;验收时除了跑 bench,还要
   `git diff <base> <head> -- <标尺文件>` 逐行看,并确认复测里**标尺的绝对 TF 仍落在基线区间**
   (mx8tw 收官时 tw 稳在 2839–3097 TF,与开跑时一致,才敢认那个比值)。

harness 会在每次候选 bench 前 **`rm -rf /root/.flydsl/cache`**(FlyDSL 按源码 hash 缓存,不清会测到旧二进制)。

轮内 agent 手上有两个现成工具(orchestrator 自动生成在 campaign 目录):
- `bash bench.sh` —— 同步当前工作树 → 清缓存 → 跑 bench,**和 orchestrator 同一把尺**
- `bash remote.sh '<cmd>'` —— 在容器里跑任意命令(rocprofv3 / ISA dump)

---

## 4. 日常盯盘

**目录结构** `flydsl_campaigns/<ts>/`:

| 文件 | 内容 |
|---|---|
| `state.json` | ★**唯一真值**:`base_score` / `best_score` / `rnum` / `kept_commits` / `phase` / `directive` |
| `run.log` | 人读主日志(轮次、每 rep 分数、KEPT/no-new-best、manager 裁决) |
| `raw_scores.jsonl` | 每次 bench 的原始读数(round/label/rep/ok/score) |
| `goal.md` / `memory.md` | agent 的目标与跨轮记忆 |
| `manager_status.md` | ⚠ **展示用快照**,manager 收轮时才刷新;和 state.json 不一致时**信 state.json** |
| `rounds/round-NN/` | `agent.txt`(轮报告)、`diff.txt`(本轮改动)、`deep_raw.jsonl`(整轮 agent 流)、`manager_raw.jsonl` |
| `say.sh` | **人←→管理者通道**:`bash <dir>/say.sh "一轮只做一个改动"` |
| `CURRENT_<tag>` | 工具目录下的稳定软链,指向本次 campaign 目录 |

**判活跃度(★误杀过一次)**:agent 大部分时间在等 API,**CPU≈0 / GPU 0% 都正常**,不能作为死活判据。三个信号:
1. `rounds/round-NN/deep_raw.jsonl` 字节增长 —— ⚠ **它只在收轮时落盘**,轮内根本不存在,**没法用来判轮内活跃**;
2. campaign 目录 mtime;
3. ★**轮内唯一的活信号 = 被改 kernel 文件的 mtime**。

三者**都**超 30 分钟没变才算卡死。单轮 25–60 分钟都正常。补充证据:`ps -eo pid,lstart,args | grep cursor-agent`,看 agent 进程的启动时刻是否和 run.log 的开轮时刻对得上。

---

## 5. 停 / 续

```bash
# 1) 先救在制品(kill 会丢掉本轮未提交的改动)
cd <repo> && git diff -- <pkg> > <campaign_dir>/salvaged_rN.patch
# 2) 杀:campaign 进程 + ★它的子 cursor-agent(见 §6.2)
# 3) 工作区必须还原干净,否则 resume 会 refuse to start
cd <repo> && git checkout -- <pkg> && git status --porcelain -- <pkg>   # 必须为空
# 4) resume(★必须先 cd 到 flydsl_optimizer,脚本里是相对路径)
cd .../flydsl_optimizer && setsid env IS_SANDBOX=1 ./_launch_<tag>.sh \
  --resume .../flydsl_campaigns/<ts> > /tmp/launch.log 2>&1 < /dev/null & disown
```
resume 成功的标志(日志):`RESUMED: N rounds done, next round N+1, phase=..., best=..., kept=K (skipping baseline + completed rounds)`。
**`campaign done` 出现 = 正常收官,不要再 resume。**

resume 只读 `state.json` 的进度,**参数取自 launcher 脚本** —— 所以改模型/GPU/后端就改脚本再 resume,改完立刻生效。

### 5b. ★预算数字是「每轮」,别拿 `total_cost` 当停机理由

`--deep-budget / --optimize-budget` 原样传给 `claude -p --max-budget-usd`,那是 **单轮上限**;
`state.json` 的 `total_cost` **只是记账**,harness 不会因为累计花费停机 —— **这是设计,不是缺陷**。

★2026-08-09 我把用户说的 "budget=$120" 当成整场总额,还外挂 watcher 去"守" ⇒
**NN 场在 r10/20 被白停 2.5 小时**,同一个误解也白杀过 TN 场的 R9。用户澄清:
"total 应该是每一轮的预算,不是总预算"。

* **预算数字吃不准就问一句** —— 补错语义的代价是杀掉在跑的工作,问的成本接近零;
* **`total_cost` 只报不判**。监控提示词里要明写"不据此停机",否则每一轮监控都会重犯;
* 真要限总额,**用 `--rounds` 卡轮数**。实测单价(opus xhigh、GEMM 场):
  优化轮 **$13–20/轮**、replan **$9.4**、manager **$0.3–0.5**;
* 万一确实要中途停:**停在轮末入账后**,轮中杀 = 这一轮的钱花了却拿不到分。

---

## 6. ★★坑清单(全部实际踩过)

1. **`pgrep -f "cursor_campaign.py --repo mxfp4"` 会匹配到你自己这条 bash**(命令行里含同样字符串)→ `kill $(pgrep ...)` **把自己一起杀掉**,后半段命令(`git checkout`)不执行,现场留一半。**先把 PID 打出来肉眼确认,再按显式 PID kill。**
2. ★★**杀了 campaign 不等于杀了 agent**。子 `cursor-agent` 会变孤儿**继续改 kernel 文件**。踩证:一个超时后没死透的 agent 在另一轮里覆盖了三次编辑,其中一次正好落在 bench 前后之间,产生不可归因读数。**kill 时必须连 `pgrep -P <campaign_pid>` 的子进程一起杀,并确认它们变成 `Z <defunct>`。**
   ⚠ **孤儿的父不再是 campaign,`pgrep -P` 抓不到它**。轮次 timeout 也会产生孤儿(不只是你手动 kill):mx8tw 那场的 REVIEW 轮 19:16 撞满 2400s 超时,那个 agent 一直活到次日 01:23 被我发现,**游离了近 7 小时**,`ps -o ppid=` 显示 PPID=1。⇒ 停机时除了 `pgrep -P`,还要 **`ps -eo pid,ppid,args | grep cursor-agent | grep <campaign_dir>`** 按 `--add-dir` 里的 campaign 目录名兜一遍全局。注意这个 grep 也会匹配到你自己的 bash(同 §6.1),先打印 PPID 肉眼分辨再动手。
3. **撞满 `--deep-timeout` = 整轮报废**:`agent.txt` 0 字节(没来得及写报告),两小时零产出。在制品还在 `rounds/round-NN/diff.txt` 里,工作区会被 harness 自动回滚。
4. ★**effort=max 反而更差**。实测同一 campaign:`-max` 三轮(一轮撞满 2h 超时报废、两轮各 60–120 分钟)**净收益 0**;换 `-xhigh` 后**两轮各约 25 分钟、+0.79% 并刷新 best**。Opus 5 在 xhigh 与 max 都保 1M 上下文,降档不掉上下文。**默认用 xhigh。**
5. ★★★**换机器必须重测 base/best 并写回 `state.json`**。分数是**原始 TF/s 绝对值**,跨机不可比(同代码 Crusoe 1085.8 vs chi2798 1123.9,差 3.5%)。不重测:换到慢机 → 所有候选被当回归 revert;换到快机 → 垃圾改动被当 win。做法:在新机上把 `base_commit` 与当前 HEAD 各测一遍(中位数),写回 `base_score`/`best_score`,旧值备份成 `state.json.<oldmachine>.bak`。
   - ★★**score 是「同进程交错」的比值也一样要重测,别以为比值就跨机免疫**。mx8tw 的 score 是
     `t_标尺 / t_我们`、两边在同一进程里交错计时(对 DVFS 漂移确实免疫),换机后仍**整体上抬 0.45%**:
     同一 commit `d652a51c` 在 chi 0.95908 / 在 Crusoe **0.96341**(机内三读 ±0.07%,极稳)。
     不改 state 的后果很具体:下一轮真实 +0.40% 会被算成 +0.86%,**直接跨过 0.5% 的 keep 阈值被误 keep**,
     而且此后每一轮的 gain 都带着这个偏移。
   - 顺手记下**标尺在新机上的绝对值区间**(如 tw wgrad 2740–3034 TF),§7 验收时要复核它没漂。
   - ⚠ 恢复工作区时 `git checkout <base> -- <files>` 会把内容**留在 index**,要 `git reset -q`
     才能让 `git diff` 回到「HEAD vs 工作区」的正常口径。
6. ★★**收官 squash 会挑错 base**。踩证:一场 30 轮 campaign 的最终 squash commit **把 `attn_helper.py` 整个删了(−2691 行)**、还夹带了 csrc/pyproject/tools 的无关改动。**收官后必须 `git diff --stat <base> <squash>` 逐文件核对**,别直接 merge。
7. ★★**campaign 的 worktree 一删,收官 commit 就变游离**。分支头停在中途某轮(实测停在 r17=875.3),而真正的收官版(891.4)是个没有任何 ref 指着的 dangling commit,随时可能被 gc。**收官后立刻 `git branch <name> <squash_sha>`。** 判断"哪个是最终版"要看 `state.json` 的 `best_score` + run.log 的 `campaign done` 行,**不要看分支头**。
8. **并发 campaign 的 `--gpu-pool` 不能重叠**。两场都含 GPU 6 时,任一场自动切卡就会两个 bench 抢同一张卡 —— 症状是分数**腰斩到约一半**(同一份 job 被并发执行)。
9. `manager_status.md` 与 `state.json` 不一致时信后者(§4)。
10. **`--repeat` 取 MAX,系统性偏高**(实测约 +0.2%,机器抖动大时更多)。验收要自己另跑中位数,见 §7。
11. crusoe 后端的文件队列**单目录严格串行**:`.job` 迟迟不变 `.done` 先看 `ls -lat $Q/*.done`,可能只是排在队友的长 bench 后面,不是 agent 死了(`connection/crusoe/01` §4.-2b)。
12. ★★★**一轮 CRASH 之后,工作区可能被回滚到最近一次 *commit*,把没提交的"KEPT_WIP"轮全部丢掉**。
    踩证(2026-08-01):r10 在收尾的 `rocm-smi` 上 TimeoutExpired 崩了;harness 回滚工作区,而
    r8/r9 的改动因为增益小(+0.23% / +0.30%)只被记为 `KEPT_WIP`、**从未 commit**(git log 最后一条
    是 r7),于是这两轮的成果一起消失。下一轮 agent 拿到的 memory.md 仍然把 r8/r9 写成"已保留",
    baseline 却低了 0.15%,读数落在噪声带里、看不出来。
    ⇒ **每轮开工第一件事:拿 memory.md 里上一轮 `scheme:` 描述的标志物 grep 目标文件**
    (本例:r8 的 `_dbar`、r9 的 `cands = [(256, 2, 8, 0)...]`),确认前几轮的改动真的在树里,
    再去测 baseline。缺了就先从 `rounds/round-NN/crash_worktree.patch` / `diff.txt` 里
    把那几个 hunk 抠回来 —— 那是**零风险、已验证**的收益,比当轮任何新想法都划算。
    (本轮就是这么找回 dgrad 组 0.9822→0.9847 与 min_speedup +0.6pp 的。)
13. **`bench.sh` / `remote.sh` 没有重试,跳板机 banner 超时会直接打死一轮**。症状:
    `Connection timed out during banner exchange` + `rsync ... exit status 255`,但同一时刻
    **裸 `ssh` 是通的** —— 是 `ConnectTimeout=20` 在负载下不够,不是机器挂了。
    解法:复制一份 ssh wrapper 把 `ConnectTimeout` 抬到 90 并加 `ServerAliveInterval=30`,
    再套一层对 rsync/docker-exec 各自重试的 driver(本场记在 campaign 目录的 `rr.py`)。
    ⚠ wrapper 必须放在**可执行**的路径下(campaign 目录挂载可能是 `nobody:nogroup` 且不可 chmod,
    rsync 会报 `Failed to exec ...: Permission denied (13)`)—— 放 `/tmp/` 下 chmod +x。
14. ★★★**agent 侧 API 故障会被 harness 记成「no change produced」而不是 crash,于是 `--max-round-crashes` 永远不触发,campaign 以约 5 分钟一轮的速度把剩余轮次静默烧光**。
    踩证(mx8tw,2026-08-04):`cursor-agent failed rc=1 ... RetriableError: [unavailable] PING timed out`
    从 17:04 起连续发作,**r9–r18 十轮全废**(累计 14 次),其中包括 harness 因停滞自动插的
    **REPLAN #1** —— `goal.md` 的 mtime 至今停在开跑那天上午,**那次重规划从未真正发生**,
    而 agent 后续几轮还在拿旧 goal 干活。20 轮的 campaign 实际只有 r1–r8 在工作。
    - **判据 = `grep -c "cursor-agent failed" run.log` 的增量**,不是看 best_score 有没有涨
      (烧轮时分数纹丝不动,和"改动没效果"长得一模一样)。30 分钟内新增 ≥3 次就该停机等故障过去。
    - ★**别用短请求测 API 来判断"已恢复"**:我用 `cursor-agent -p "reply OK"` 测了两次都秒回 OK,
      据此判定恢复、放它继续跑,结果又烧了五轮 —— 故障只在 **agent 的长会话**上暴露
      (实测跑到 200–484 秒才被打断),短请求根本触不到那个超时。
    - 停机后别急着 resume:先看 `goal.md` mtime,若 REPLAN 那轮被打断,得手工把重规划补回去
      (或在 `state.json` 的 `directive` 里写清新方向),否则 agent 会继续按已榨干的旧方向走。
15. ★★**resume 一个已经 `campaign done` 的 campaign,有三处状态会让它秒退**。想加轮次接着跑时:
    - `phase` 停在 `REVIEW` → optimize 主循环的条件是 `while rnum < opt_budget_end and phase != "REVIEW"`,
      整段直接跳过;
    - `action` 停在 `GO_REVIEW` → 即使改了 phase,manager 一读就打印 `manager -> GO_REVIEW: ending
      optimize early` 再次秒退;
    - `rnum` 已经被那次秒退 +1,得退回 `len(state["rounds"])`。
    三个都要改,顺带清空 `opt_window`(否则一进来就满足停滞条件、立刻触发 REPLAN)。改前备份 `state.json`。
    - ★**还必须加 `--no-squash`**:`finish()` 会拿 `state["kept_commits"]` 去 squash,而那些 per-round
      commit 在上一段收官时已经被 squash/amend 掉、**不在当前分支历史上了** —— 直接踩 §6.6 的
      "挑错 base"。让它把新轮次的 commit 留在分支上,收官时你手工 `git reset --soft <上段收官 commit>`
      再提交一个干净的。
16. ★★★**基础设施宕机会把 campaign march 到假「campaign done +0.00% kept 0」,长得和「没东西可优化」一模一样**。
    踩证(gptoss_e32,2026-08-06 首跑):chi 跳板机整段死 → r4/r5/r6 **连三轮 `round CRASHED: RuntimeError:
    command failed (255): rsync ...`** → 到 rnum 后 phase 转 REVIEW,r7/r8 各 3 分钟空跑 → `campaign done.
    best 2404.1 -> 2404.1 (+0.00%). kept 0 commit(s). est spend ~$29.93`。**`--max-round-crashes`(默认 3)没
    救成** —— REVIEW 阶段的 rnum 先到,尾部两轮把 campaign 「正常」收官了。
    - **判据 = 崩因是传输层(`rsync`/`ssh` 退 255 / banner timeout)且 `best` 纹丝停在 baseline**。分数没涨 +
      崩因是 255,几乎必是 backend host 宕机(不是「改动没效果」也不是 §6.14 的 agent-API PING 超时,那个是
      「no change produced」不抛异常)。§6.13 的 banner-timeout 是**单机在但慢**、加重试能救;这里是 **host 整段
      不在**,重试无用。
    - **解法 = 迁 backend + resume 重测 base/best(§6.5)**,别原地干等跳板机。本场迁 Crusoe 文件队列 + 从 r3
      resume(base 重测 2405.9)后,r3/r4/r5 **连三赢** +9.64%/+4.11%/+2.36%。一句话:`+0.00% kept 0` 先看
      `grep CRASHED run.log`,是 255 就是基建的锅,不是你的 kernel 没搞头。
17. ★★**campaign 每轮 commit 会自动 `git add` agent 落在 repo 根的 `_an_*`/`_*.py` 探针,卷进每个 KEPT commit**。
    踩证(gptoss_e32):r3/r4/r5 三个 win commit 各自夹带 `_an_r4_*`/`_an_r5_*`/`_an_r9_*`/`_an_wg_cfg.py`,
    `git diff --stat <base>..HEAD` 里 **1182 行只有 246 行是目标核,其余 936 行全是探针**。「探针绝不 git add」
    这条规矩是**对人的,harness 不守** —— 收官不能靠「我没加过」,必须主动剥:squash 前 `git diff --stat
    <base>..HEAD` 必须**只剩目标核那一个文件**,多出来的 `_*` 一律从历史剔除(见 §6.6 逐文件核对、code-style
    卡「探针不 git add」)。
18. ★★**轮次死在 bench 阶段(不是 agent 阶段)时,那一轮的交付件从没被计过分,别当它判负了**。
    踩证(mx8tw):r14 的 agent 跑满 127 分钟产出了完整交付件,harness 刚要 bench 就撞上传输层 255 崩掉;
    r17 同样。**harness 会把在制品存进 `rounds/round-NN/{diff.txt, crash_worktree.patch}` 然后
    revert 工作区**,所以下一轮的 agent 看不到它,memory 里也只有一行 `CRASHED`。
    - **恢复做法**:`git apply` 那份 diff 回工作区 → resume 时带 **`--allow-dirty`**(否则
      `ensure_clean_tree` 会拒绝启动)→ 让它在 round N 重新计分。**同时把交付件内容手工补进
      `memory.md`**,否则重跑的 agent 会从头再做一遍那 127 分钟。
    - ⚠ **`rounds/round-NN/diff.txt` 可能被截断**(harness 写它有长度上限,实测 682 行版本尾行缺换行、
      `git apply` 报 `corrupt patch`)。**从 `crash_worktree.patch` 里按 `diff --git` 段抽目标文件那一段**
      才是完整的(它同时含 untracked 探针,别整份 apply)。
    - r14 重跑后拿到 **+1.04%,是那一场最大的一笔** —— 崩在 bench 阶段的轮次值得原样重跑。
19. ★**未过 keep 阈值的 WIP 会一直留在工作区,resume 时会挡住启动**。KEPT_WIP 的语义是「保留工作副本
    继续迭代」,`revert_worktree()` 只在 streak 到 `--revert-patience` 时才触发。所以停机后 resume
    **几乎总是需要 `--allow-dirty`**。⚠ 另注:`no_improve_streak` **不在 state.json 里**,resume 后归零 ——
    连续未达标的轮次计数会重新开始,比不重启更宽容。

---

## 7. 验收:越过目标≠达标

campaign 自报的数**不能直接当结论**。逐条过:

1. `ok=true`、SNR ≥ 门限、`det=True`(从 `raw_scores.jsonl` / bench 输出确认,不是听 agent 说);
2. **远端 md5 == 本地 HEAD** 的对应文件(防止测的是别的版本);
3. **单独复跑取中位数**(≥5 次),抵消 `--repeat` MAX 的偏高;注明冷/热;
4. 顺带跑一遍多形状正确性扫描(varlen / SWA / 非整 tile / 小序列),别只信打分那一个 shape。
5. ★★**把目标文件 dispatch 的每一条分支都调一次真入口** —— 不是只跑计分的那几格。
   计分口径就是 campaign 的全部视野:**没被计分的兄弟路径可以被改断而全程不报警**。
   2026-08-09 dense fp8 NN 场:bench 只跑 NN+TN,某轮顺手把 NT 的
   `cands.append([launch, (bm, gm, gn, xcd, ag), c])` 也改成 5-tuple,而 NT 的循环是
   `for bm, gm, xcd, ag in _NT_CANDIDATES` ⇒ `gn` 未定义 → NameError → 被
   `except Exception: continue` 吞掉 → 每个候选静默丢弃 → `NT autotune found no working cfg`。
   **整条前向在交付 commit 上是断的,10 轮 bench 全绿。**
   低成本挡法两条,都做:
   - **AST 查 use-before-assign**(比读 diff 可靠):
     `函数内 Load 的 Name 集合 − Store − 全局`,一把抓出 `uses gn: True | assigns gn: False`;
   - **每条 layout 各调一次真入口的 3 行 smoke**。
6. ★**验收前先确认硬规矩里的路径真的存在**:`git diff <base> HEAD -- <不存在的路径> | wc -l` **恒为 0**。
   那一场我全程 grep `flydsl/gemm/gemm_helper.py`,真文件在 `flydsl/utils/gemm_helper.py` ——
   "零 diff"这条检查空转了一整场。加检查时先 `test -f`。
7. ★**写 probe 先核对真入口签名,并且 base / HEAD 同 probe 对照**。
   同一场我连吃两次 false negative:先漏了 `c_m` 直调内部 autotune,再把
   `fly(a, a_scale, b, b_scale)` 当成 `fly(a, b, s, s)` —— **单臂失败什么都不能证明**,
   差点把自己 probe 的 bug 当内核回归报出去。

★**不打扰在跑 campaign 的复测法**(工作区里是 agent 的在制品,不能直接测 HEAD):
```bash
git show <HEAD>:<file> > /tmp/x.py            # 从 git 取干净版本,不碰工作区
# 远端容器里:拷一份包(带编译好的 .so)→ 覆盖那几个 .py → 用 PYTHONPATH/cwd 隔离运行
V=/workspace/code/PT_verify; rm -rf $V; mkdir -p $V
rsync -a --exclude __pycache__ <live_repo>/primus_turbo $V/
cp /path/x.py $V/primus_turbo/.../file.py
cp <live_repo>/<bench>.py $V/ && cd $V && python3 <bench>.py    # 脚本目录进 sys.path[0],不会串到 live 树
```
用一张**没被任何 campaign 占用**的卡跑(`--gpu-pool` 之外),既不撞车也不污染对方读数。
要跑另一个分支的版本,用**临时 worktree**(`git worktree add`)而不是在共享 checkout 上 `git checkout` —— 后者会把在跑的 campaign 的工作树掀了。

---

来源: bwd16384(20260726_152547,30 轮 789.6→891.4)、fwd16384(20260728_082734,chi2798 983.2→1124+)、gptoss-sbhd-bwd(20260728_123346)三场实跑;
交叉引用: `connection/crusoe/01-access-spur-container.md`(文件队列/节点恢复)、`methodology/01-optimize-loop-benchmark.md`(计时口径)、
`pitfalls/02-measurement-noise.md`(噪声地板/DVFS)、`pitfalls/08-sync-build.md`(rsync/git 同步坑)、`methodology/13-autotune-fleet.md`(多 GPU 调度)。

20. ★★★**小增量连环不入库 = `--revert-patience` 到点一次性全丢**(2026-08-12 mxfp4 m4096 场,最贵的一次)。
    r13–r19 **七轮**每轮都比 best 高(+0.12 / +0.25 / +0.22 / +0.42 / +0.27 / r19 读到 **101.0–101.1**),
    但每轮都够不到 keep 门槛 ⇒ 全部记 `KEEPING working copy to iterate`,streak 一路涨到 8 ⇒
    `revert_worktree` 把七轮累积的 **73 insertions / 55 deletions 一次清光**。
    - 病根不在 harness,在 agent 的行为模式:**每轮换一个新点子,把上一轮的增量稀释掉**,于是每轮都是
      "略高于 best 但不够",永远攒不出一次能过门槛的候选。
    - ⇒ **stall 到 5/8 就必须 say.sh 干预**,让它停止开新方向、把工作树里已验证的部分收敛成一次候选去撞门槛。
      我等到 7/8 才发,inbox 在 `16:08:52` 被消费,而 revert 就发生在**同一秒** —— 晚一步就全没。
    - ⇒ 另一个更稳的做法:**发现某轮 A/B 确实为正就自己动手 commit**(流程见下面第 26 条),
      别指望 harness 的 keep 阈值替你保管收益。
    - **抢救窗口**:本地 `rounds/round-NN/diff.txt` 是截断摘要(78 行 vs 工作树 73+/55−,对不上);
      **完整改动在远端 repo 里** —— harness 下一轮 sync 才会覆盖它。revert 之后**立刻**
      `rexec '...git diff'` 抓回来,晚了就没了。本场靠这个把 r19 完整捞回并最终入库(`c49b8ef4`)。

21. ★★**pre-commit 的 `ruff-format` 会重排代码 ⇒ 提交的字节 ≠ 你实测的字节**。
    r9 手工 commit 时 hook 报 `files were modified by this hook`,diff 从 19+/11− 变成 **43+/18−**,
    第一次 commit 直接失败要重来。ruff 理论上只动排版,但**别拿"理论上"当证据**:
    ⇒ **在提交后的版本上重测一遍**再写 `best_score`。实测格式化前 100.61 / 格式化后 100.54,
    差 0.07%(噪声内)⇒ 确认无行为改变,`best_score` 取**提交后**那个数。
    (反例:r19 那次 hook 一次通过、diff 规模未变,就不必重测。)

22. ★★**ratio 类 score 必须 ×100 再交给 harness**。日志打分数用 `round(s, 1)`,
    对 TF(4139.5)刚好,对 0.97 这种比值会被压成 `"1.0"` —— **整场日志全是 1.0,读不出任何增益**,
    监控和纠偏都瞎。第一次起场就踩了(dir 20260811_042510 作废重起)。
    ⇒ bench 里同时输出 `gm_ratio`(原值,给人看)和 `gm_ratio_pct = ratio*100`(给 `--score-key`)。
    ⚠ goal 里的目标值要跟着改成百分比,否则 agent 会拿 1.10 去对 100.5 的分数。

23. ★★**收官核对 `git diff --stat` 的基准选错 = 漏检探针**。我核 `<上一场 kept>..<本场 squash>` 得出
    "零探针夹带",实际 `_probe_wgcfg.py` 是**上一场那个 kept commit 自己带进来的**(§6.17),
    把基准换成真正的 ship 起点(已 push 的那个 commit)才暴露。
    ⇒ **核对基准 = 你打算 PR 的那个 base,不是本场的 base_commit**;`git rm --cached` 剥掉探针时
    文件留在磁盘上(还要接着用),只从版本库摘出去。

24. **`--resume` 的 `rnum` 是 `+1` 语义**。想让它从 round N 重跑,`state.json` 里要写 `rnum = N-1`。
    我写 8 得到的是 `next round 9`。真想重跑某轮,先确认 `len(state["rounds"])` 与 `rnum` 的关系再改。

25. ★★**bench "秒退"(几百毫秒)+ `rc=1` + 没有任何 python traceback = 环境/权限问题,不是 kernel 问题**。
    症状在日志里只有一行 `FAILED gate`,极易被当成"这轮方案判负"。本场 r9 就是这么被误判的 ——
    真因是 `cd <repo>: Permission denied`(NFS root_squash + harness 自己那条无 `--chmod` 的 rsync 把
    源树 700 权限搬到远端),而那一轮其实是**全场最大的一笔 +2.02%**。
    ⇒ 判据:正常 bench 至少几秒;**低于 1 秒的失败先查环境**(权限 / venv / .so / cache),
    修完必须 `say.sh` 告诉 agent **那轮不算判负**,否则它会把方向记成死路,后面所有轮都被带偏。

26. ★★**手工把"被环境吞掉/没过门槛但确实为正"的一轮补测入库**(本场用了两次,一次 +2.02% 一次 +0.5%,
    都是全场前二的收益 —— 不做这一步它们就全没了)。完整流程:
    1. **取在制品**:优先 `rounds/round-NN/crash_worktree.patch`(崩溃轮)或**远端 repo 的 `git diff`**
       (revert 轮,趁下一轮 sync 覆盖前抓)。`diff.txt` 只是截断摘要,别用。
    2. **只抽目标核那一段**:`crash_worktree.patch` 含 untracked 探针,整份 apply 会把垃圾一起带进来。
       ⚠ 抽出来的 patch **可能缺末尾换行** ⇒ `git apply` 报 `corrupt patch at line N`,`printf '\n' >>` 补上即可。
    3. **打不上就手工合并**:review 轮常把同一段的注释精简过,上下文对不上但**代码本身逐字相同**;
       `git apply -3` 虽能打但会留冲突标记,不如按 4 处改动手工落。合并时**保留 review 后的注释风格**,
       别把 patch 里的旧长注释搬回来(那是 review 刚清掉的)。
    4. **交错配对 A/B 定性**:同一时段 W/O 轮换各 3 次、每次清 JIT cache,**别拿跨轮次的读数比**
       (跨轮含时钟漂移;本场基线自身三次就散 1.06,大于臂间差异)。
    5. **多改动要拆开归因**:一笔 patch 里常绑着两个独立改动,构造 O/G/S/W 四臂交错测,
       否则不知道收益出自哪一半(本场就测出配对门单独用反而压低 min)。
    6. 过了就 `commit`(只 add 目标核、`kyle-256` 署名),然后按第 21 条**在提交后的版本上重测**,
       再把 `best_score` / `kept_commits` / `rnum` 写回 `state.json`(旧值备份)。
