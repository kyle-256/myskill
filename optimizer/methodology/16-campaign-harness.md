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

---

## 6. ★★坑清单(全部实际踩过)

1. **`pgrep -f "cursor_campaign.py --repo mxfp4"` 会匹配到你自己这条 bash**(命令行里含同样字符串)→ `kill $(pgrep ...)` **把自己一起杀掉**,后半段命令(`git checkout`)不执行,现场留一半。**先把 PID 打出来肉眼确认,再按显式 PID kill。**
2. ★★**杀了 campaign 不等于杀了 agent**。子 `cursor-agent` 会变孤儿**继续改 kernel 文件**。踩证:一个超时后没死透的 agent 在另一轮里覆盖了三次编辑,其中一次正好落在 bench 前后之间,产生不可归因读数。**kill 时必须连 `pgrep -P <campaign_pid>` 的子进程一起杀,并确认它们变成 `Z <defunct>`。**
3. **撞满 `--deep-timeout` = 整轮报废**:`agent.txt` 0 字节(没来得及写报告),两小时零产出。在制品还在 `rounds/round-NN/diff.txt` 里,工作区会被 harness 自动回滚。
4. ★**effort=max 反而更差**。实测同一 campaign:`-max` 三轮(一轮撞满 2h 超时报废、两轮各 60–120 分钟)**净收益 0**;换 `-xhigh` 后**两轮各约 25 分钟、+0.79% 并刷新 best**。Opus 5 在 xhigh 与 max 都保 1M 上下文,降档不掉上下文。**默认用 xhigh。**
5. ★★★**换机器必须重测 base/best 并写回 `state.json`**。分数是**原始 TF/s 绝对值**,跨机不可比(同代码 Crusoe 1085.8 vs chi2798 1123.9,差 3.5%)。不重测:换到慢机 → 所有候选被当回归 revert;换到快机 → 垃圾改动被当 win。做法:在新机上把 `base_commit` 与当前 HEAD 各测一遍(中位数),写回 `base_score`/`best_score`,旧值备份成 `state.json.<oldmachine>.bak`。
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

---

## 7. 验收:越过目标≠达标

campaign 自报的数**不能直接当结论**。逐条过:

1. `ok=true`、SNR ≥ 门限、`det=True`(从 `raw_scores.jsonl` / bench 输出确认,不是听 agent 说);
2. **远端 md5 == 本地 HEAD** 的对应文件(防止测的是别的版本);
3. **单独复跑取中位数**(≥5 次),抵消 `--repeat` MAX 的偏高;注明冷/热;
4. 顺带跑一遍多形状正确性扫描(varlen / SWA / 非整 tile / 小序列),别只信打分那一个 shape。

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
