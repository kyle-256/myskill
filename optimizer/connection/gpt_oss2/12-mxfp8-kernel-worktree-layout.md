# gpt_oss2 mxfp8 内核工作树位置(住在 mxfp4 树里)与关键文件

> 类别: 连接 (gpt_oss2) · 主题标签: worktree-layout, mxfp8-grouped-gemm, WL-未提交, 关键文件

## 反直觉:mxfp8 内核住在 mxfp4 树里
- gpt_oss2 环境的 mxfp8 grouped GEMM 内核实际住在 **mxfp4 树**里,工作树路径 **`sync/mxfp4/Primus-Turbo`**(不是 mxfp8 repo)。mxfp8 内核代码放在 mxfp4 checkout 中——找错 repo 会白费时间。

## 关键文件(均在 sync/mxfp4/Primus-Turbo 下)
- `primus_turbo/flydsl/grouped_gemm/mxfp8_grouped_kernel.py`
- `primus_turbo/flydsl/gemm/mxfp8_quant_flydsl.py`
- `mxfp8_gemm_kernel.py`
- `flydsl.utils.gemm_helper`(即 `gemm_helper.py`)
- `quantization.py`
- `grouped_gemm_fp8.py`

## WL(whole-loop)实验代码留档状态
| 场景 | 是否 commit | 留档位置 | WHY / 陷阱 |
|---|---|---|---|
| dense mxfp8 WL | 有 | 仓库外 `/workspace/code/gpt_oss2_docker/.wl_study/` + git 历史 `fc45f4bb` | 可从留档恢复 |
| grouped wgrad WL | **从未 commit**,已 revert 回 HEAD(git 历史里没有) | 唯一残余 = 删除前的 `/tmp/sp_backup` | `/tmp` ephemeral,容器重启即失,**勿依赖**;重跑得照结论重写 |

- grouped wgrad WL 全套实验代码(`_build_grouped_mxfp8_wgrad_wl_kernel` / `_compile_grouped_mxfp8_wgrad_wl` / `PT_MXGG_WGRAD_WL*` 探针 env / `S2RLoader.base_addr` / `PT_MXGG_*_CFG` autotune 覆盖钩子)全部 revert,不在 git 里。

---
来源: mxfp8-grouped-gg-devloop/SKILL.md, project_mxfp8_grouped_wgrad_wl.md, project_mxfp8_wholeloop_port.md
