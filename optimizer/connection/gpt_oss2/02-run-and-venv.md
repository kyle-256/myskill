# 远端运行方式与 venv 隔离

> 类别: 连接 · 主题标签: docker-exec, HIP_VISIBLE_DEVICES, flydsl-cache, venv, venv隔离, PEP660-editable, editable-finder, import路径验证, build_ext

## 远端容器内跑法:docker exec + HIP_VISIBLE_DEVICES

- **通用调用格式**:`./.ssh-chi.sh root@chiXXXX "docker exec -e HIP_VISIBLE_DEVICES=$G mlperf_gptoss2 bash -lc '...'"`。跳板脚本 `.ssh-chi.sh` 进节点 → `docker exec` 进容器 `mlperf_gptoss2` → `bash -lc` 跑命令。
- **选卡**:`HIP_VISIBLE_DEVICES=$G` 作为 `docker exec -e` 传入。选哪张见 common/02;**用一张已核实空闲的卡**——节点被别人占了一半时惯例取 GPU `1/2/3`,8 卡全空时听用户指定(2026-07-30 chi2879 用的 GPU4)。跑 bench 前复查,邻居大任务会共享功耗/散热给相邻卡降频。
- **改 kernel 必清缓存**:改完 kernel 命令里加 `rm -rf /root/.flydsl/cache`。**判据与更深的缓存坑见 methodology/02**——FlyDSL 的 cache key 只折入**标量**闭包值,类型对象(如 `fx.Float16`)被静默丢弃,能导致"用 A 配置编的内核喂给 B 配置的调用"。

```bash
cd /workspace/code/gpt_oss2_docker/sync
# mxfp8
./.ssh-chi.sh root@chi2879 "docker exec -e HIP_VISIBLE_DEVICES=2 mlperf_gptoss2 bash -lc \
  'cd /workspace/code/mxfp8/Primus-Turbo && /opt/venv/bin/python <script>'"
# mxfp4
./.ssh-chi.sh root@chi2879 "docker exec -e HIP_VISIBLE_DEVICES=2 mlperf_gptoss2 bash -lc \
  'cd /workspace/code/mxfp4/Primus-Turbo && /opt/venv-mxfp4/bin/python <script>'"
```

- **ssh 包装脚本路径写绝对值**:`.ssh-chi.sh` 是相对 `sync/` 的,session 里 `cd` 过别的目录后再用 `./.ssh-chi.sh` 会 `No such file or directory`(尤其后台任务)。稳妥写法:`/workspace/code/gpt_oss2_docker/sync/.ssh-chi.sh`。

## ★★ 直接 `bin/python`,严禁 `source activate`

- **正确**:`/opt/venv-mxfp4/bin/python <script>`。直接调解释器 → 稳定 import 对应 repo。
- **错误**:`source /opt/venv-mxfp4/bin/activate; python ...`,尤其叠加两个 activate —— 后一个覆盖 PATH,`python` 串到别的 venv;`bash -lc` 下 `python` 解析也可能不是你以为的那个。**后果是拿另一份 repo 的代码当成你的 fix 在测,得到的全是假数据。**
- **跑任何测量前先验证 import 路径**:

```bash
/opt/venv/bin/python       -c "import primus_turbo; print(primus_turbo.__file__)"  # → …/mxfp8/…
/opt/venv-mxfp4/bin/python -c "import primus_turbo; print(primus_turbo.__file__)"  # → …/mxfp4/…
```

## 两份 turbo ↔ 两个 venv 对应表

| venv 解释器(直接调) | import 的 repo | 容器内路径 | 用途 |
|---|---|---|---|
| `/opt/venv/bin/python`(别名 `/opt/venv-mxfp8`) | `…/mxfp8/Primus-Turbo/primus_turbo` | `/workspace/code/mxfp8/Primus-Turbo` | mxfp8 树 |
| `/opt/venv-mxfp4/bin/python` | `…/mxfp4/Primus-Turbo/primus_turbo` | `/workspace/code/mxfp4/Primus-Turbo` | mxfp4 树 |

> ⚠️ **树名 ≠ 内容**:flydsl 的 MXFP8 量化/GEMM 开发一律在 **mxfp4 树 + `/opt/venv-mxfp4`** 里做,不是 mxfp8 树。用户明确要求过;曾误在 mxfp8 树做 rescope+rebase 被纠正。

host 路径 = 容器路径把 `/workspace/code` 换成 `/mnt/vast/kyle/code3`。

## 为什么两个 venv 真隔离 + 两个污染坑

`primus_turbo` 是 editable 安装(PEP 660):每个 venv 的 `…/site-packages/__editable___primus_turbo_0_0_0_finder.py` 里的 `MAPPING` dict 指向某个 repo 的 `primus_turbo`,两 venv 各指各的 → 独立。每份 repo 各自 build 自己的 `primus_turbo/lib/libprimus_turbo_kernels.so`(在 vast 上;rsync 排除 `*.so`,所以远端 build 产物不会被本地空目录覆盖)。

- **坑 A — clone venv 的 `setuptools.pth` 泄漏**:`cp -a /opt/venv …` 后,新 venv 的 `setuptools.pth` 仍是绝对自引用 `/opt/venv/lib/...`,新 venv 会 fallback 到老 venv。建后必须 `sed` 改成新路径,并验证 `sys.path` 不含 `/opt/venv`。
- **坑 B — `pip install -e .` 跨 venv 污染 finder**(实测必现):即使修好坑 A、`sys.path` 已隔离,在 `/opt/venv-mxfp4` 里跑 `pip install -e .` 仍会把 mxfp4 的 MAPPING **写进 `/opt/venv`**,两个 finder 的 repo 路径**互换**。

### → 两条铁律

1. **重编 csrc 用 `setup.py build_ext --inplace`,不要用 `pip install -e .`**。前者只重 build `.so`(在 repo 内),**不动 finder**,零污染:
   ```bash
   ./.ssh-chi.sh root@chi2879 "docker exec -e GPU_ARCHS=gfx950 -e MAX_JOBS=128 mlperf_gptoss2 bash -lc \
     'cd /workspace/code/mxfp4/Primus-Turbo && /opt/venv-mxfp4/bin/python setup.py build_ext --inplace'"
   ```
   增量 ~3min;`rm -rf build` 后的冷全量(含 `-fgpu-rdc` 设备链接)~7–12min。
2. **凡是跑过 `pip install -e .`(只有首次建 venv 才需要),立即校验+修两个 finder 的 MAPPING**(无需重 build):
   ```bash
   F=__editable___primus_turbo_0_0_0_finder.py
   grep -oE "/workspace/code/[a-z0-9_]+/Primus-Turbo" /opt/venv/lib/python3.12/site-packages/$F | sort -u
   grep -oE "/workspace/code/[a-z0-9_]+/Primus-Turbo" /opt/venv-mxfp4/lib/python3.12/site-packages/$F | sort -u
   # 错了就 sed 改回(每个 finder 有 3 条映射,整体替换 repo 路径段)
   sed -i "s#/workspace/code/mxfp4/Primus-Turbo#/workspace/code/mxfp8/Primus-Turbo#g" /opt/venv/lib/python3.12/site-packages/$F
   sed -i "s#/workspace/code/mxfp8/Primus-Turbo#/workspace/code/mxfp4/Primus-Turbo#g" /opt/venv-mxfp4/lib/python3.12/site-packages/$F
   ```

### ⚡ 快速重建 mxfp4 venv(仅当 finder 指错或 venv 缺失)

2026-07-07 刷新后的 tar 已自带 mxfp4 venv,正常无需此步。若 `/opt/venv-mxfp4` 缺失但 `.so` 已在 code3(vast) 上 build 好,**不要**跑 `pip install -e .`(会触发坑 B 且没必要),只需 clone venv + 修 pth + sed finder(~15s):

```bash
docker exec mlperf_gptoss2 bash -lc '
  git config --global --add safe.directory "*"
  cp -a /opt/venv /opt/venv-mxfp4                                                        # reflink clone
  sed -i s#/opt/venv/lib#/opt/venv-mxfp4/lib#g /opt/venv-mxfp4/lib/python3.12/site-packages/setuptools.pth   # 坑A
  F=__editable___primus_turbo_0_0_0_finder.py
  sed -i s#/workspace/code/mxfp8/Primus-Turbo#/workspace/code/mxfp4/Primus-Turbo#g /opt/venv-mxfp4/lib/python3.12/site-packages/$F
'
```

之后 `docker commit` 持久化(见 01)。若 `.so` 也缺(fresh 无 build 产物),才需 `setup.py build_ext --inplace`。

---
来源: remote-sync/SKILL.md, 2026-07-30 chi2879 实测
