# 旧系统 L7 报告

日期：2026-11-08。分支 `feature/legacy-system-maintenance`。本包完成命令完成语义、Scheme C 原保存聚合核对，以及 2026-11-08 论文的实际编译。五项研究消耗都是 0：环境步 0，低层推断 0，评分前向 0，训练更新 0，反向传播 0。没有重跑 L6 的 48 步，也没有重跑 R1、L3–L5 或历史全套。

## 命令完成语义

`wt.legacy_system` 现在把目录准备放进异常处理。已存在的空目录可以准备；非空目录仍拒绝覆盖，退出码 2，并且不写入该目录。`matched` 同时要求数值、字段，以及 `payload_matches_served` 和 `step_served_matches_episode`。成功才返回 0。运行错误、未完成、回归不一致、模型身份不符或官方策略变化返回 1。保存的 JSON 带同一个 `exit_code`。

最小检查写在 `results/legacy_delivery_l7/completion_check.json`。空目录 `results/legacy_delivery_l7/empty_dir_probe` 被接受并保持为空。`results/legacy_system_l6` 被拒绝，其 episode 哈希前后都是 `3a980bc09eb4b61dcc94a5b274cc8d4d6cf0832e7cfd76a299f7f35774f734f4`。L6 相对 R1 的 322000 Scheme C 记录重算为 matched，退出码 0。322001 Scheme C 相对同一参照是 different，退出码 1。把 L6 记录在内存中的 `payload_matches_served` 改为 0 后，状态变为 different。L6 原件没有覆写。检查没有导入 torch。

## 证据

Scheme C 原文件 `results/legacy_formal_repro_r1/evidence_archive/results/phase6if_ranking_loss/runs/c/result.json` 的 best epoch 是 1，样本单位是 48 个验证决策状态。原保存聚合是 payload `19240091.24202323` bit，即 19.240091242 Mbit，Top-1 `0.6666666666666666`，Top-3 和 Top-5 `0.7083333333333334`，normalized regret `0.035473433715573995`。按论文已有显示位数，对应 19.240、0.6667、0.7083 和 0.0355。claim 类型是“原保存聚合已核对”，不是重新计算。

该文件的 Git blob 是 `e7be8da9c24f007ee5c30eef25a446d716fe3971`，原字节 SHA-256 是 `7ff7f6f13a42afb7e7d860063776375e2211df8c28504bc1da49c48f08bfeff4`。原 loss 是非数值，派生索引用 null，没有把原文件改成 0。`validation_loss_backfill.json` 的 `c_epoch_001.validation_loss` 是 `7.448080003261566`，这是另一个字段，没有接进 result。顶层 `results/phase6if_ranking_loss/summary.json`、基线逐状态和 best 逐状态仍是 MISSING。论文里从 18.892 Mbit 开始的基线侧仍是 reported-only。新索引是 `docs/paper/evidence_20261108.json`，没有 NaN 或 Infinity。AB/AC 沿用 11 月 4 日索引，没有重做整轮审计。

## 编译和页面

11 月 8 日正文是 `docs/paper/Execution_Grounded_Hierarchical_Graph_Learning_MultiUAV_Revised_20261108.md`，LaTeX 已同步。GitHub 上的官方 zip 返回 504，没有拿到该文件。实际编译器是 conda-forge 的 Tectonic 0.17.0 预编译包，包 sha256 `0e50ad47a61b38b7b7545b7b2357aff73c581be3a18b22390189fb820855a4c8`。`tectonic -X compile --keep-logs --keep-intermediates` 退出码 0。PDF 有 7 页，另存为 `docs/paper/Execution_Grounded_Hierarchical_Graph_Learning_MultiUAV_Revised_20261108.pdf`。7 页都已查看。日志没有 Overfull，引用已解析。来源、命令和哈希见 `docs/paper/latex/TECTONIC_L7_CN.md` 与 `docs/paper/latex/source_manifest_20261108.json`。

## 尚未完成

- ranking-loss 的 summary.json、基线侧格值和逐状态序列仍缺失。
- `lambda_H` 和 `delta_payload_max` 仍未设置。
- 官方 GitHub zip 没有下载成功，发布页登记的 zip sha256 尚未用本机文件复核。
- 新方案仍留在 `research/ocean-observation-isolated`，没有合并。
- 没有开始下一轮算法研究。
