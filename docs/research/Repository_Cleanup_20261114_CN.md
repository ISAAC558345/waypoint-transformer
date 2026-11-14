# 研究分支清理记录

清理前提交是 `e44e5227caa71c2bb5150fe7a1ab62c6fd7fbd05`。工作区没有未提交的跟踪改动。早先只存在于本地的 M1.6 结果提交 `111500ff9e4b0b827bc2889d6bec495f2188ab88`（分支 `local/m16-unpushed-111500f`）没有并入本轮。M1.6 调度修复仍是待办。

## 清理前参照

入口是 `wt.active_model_identification.run_m13._play`，配置是 `results/active_model_identification_m13/m13_20261114/frozen_config.json`，场景来自同一目录的 `scenario_manifest.json`。

- 场景 `b1_tight`，方法 `lookahead_calibration_off`，种子 `1101`，能耗系数 `[1.0, 1.0]`
- 按时交付 `40000` bit
- 耗电 `42749.350821545166` J
- 效用 `3.8812518032734857`
- 两机都返航：`returned=2`，`not_returned=0`，`not_landed=0`，`below_reserve=0`，`energy_exhausted=0`，`return_failures=0`
- 120 条请求动作与实际执行动作

## 删除

研究入口只导入 `src/wt/active_model_identification/`、`src/wt/ocean_observation_m0/`、`src/wt/ocean_observation_m0a/` 和 `src/wt/ocean_observation_m0b/`。下面这些路径没有被该入口、其配置、测试或字符串加载使用。默认分支 `feature/maritime-wt-stage1a`（`e58e5d190109e5a0e860cb5c6a412321c0d74e31`）和清理前提交都能按原路径取回。提交差异是完整文件清单，共 3028 个路径。

- 旧实现：`src/wt/lowlevel_v2/`、`src/wt/benchmark_v2/`、`src/wt/htw_phase*`、`src/wt/htwc_phase1/`、`src/wt/maritime/`、`src/wt/round4/` 至 `src/wt/round36/`、`src/wt/l2a/`、`src/wt/l3a/`、`src/wt/l3b/`、`src/wt/final/`，以及 `src/wt` 下的训练与可视化单文件
- 上游包目录：`src/algorithms/`、`src/envs/`、`src/models/`、`src/policies/`、`src/utils/`
- 根目录 `round_01/` 至 `round_36/` 及其锁文件，`CURRENT_ROUND.txt`，`ROUND_VERSIONING.md`
- 上游与逐轮启动脚本：`experiments/`、`scripts/`
- 旧结果与可再生缓存：`results/phase5*`、`results/phase6*`、`paper_assets/`（含 Matplotlib 字体缓存）、`final_artifacts/`
- 旧文档与测试：`docs/research/HTW_*`、`docs/research/NEXT_PHASE_LOWLEVEL_CLOSED_LOOP_OPTIMIZATION_CN.md`，以及不覆盖海洋观测和主动辨识的测试
- 旧配置：`configs/environment/l3b.json`、`configs/high_level/scheme_c.json`、`configs/low_level/mappo_v2.json`、`configs/experiments/closed_loop_splits.json`

没有发现内容相同的海洋实验配置，所以没有合并配置。`closed_loop_splits.json` 没有任何引用，并指向已停用的 322047 以后场景，因此删除。

## 保留

多机服务与回传、能耗与返航、业务遥测、候选评分仍在 `src/wt/active_model_identification/`。风场航段预算在 `flight.leg_budget`，浮标漂移、时间戳和有效期在 `ocean_observation_m0/information.py` 与 `accounting.drift_report`。M1 到 M1.5 的配置、结果和报告，以及 M1.6 指令，都保留。`archive/`、`docs/paper/`、`LICENSE`、`requirements.txt`、`environment.yml` 保留。`round_manifest.json` 和 `BRANCH_ISOLATION_MANIFEST.json` 的原文没有改。

## 验收

清理后用同一冻结配置、同一场景和种子重放。120 条请求与执行动作、整数 bit、耗电和返航状态与清理前逐项相同，浮点也相同。文档中的命令 `py -3 -m wt.active_model_identification.run_m13 --scenario b1_tight --method lookahead_calibration_off` 给出同一结果，并且没有写入结果目录。

`tests.test_active_model_identification_m1` 到 `m15` 共 40 项通过，用时 0.877 秒。`tests.test_ocean_observation_m0`、`m0a`、`m0b` 共 18 项通过，用时 0.033 秒。

## 仍保留的疑点

- `FORMAL_CHECKPOINT` 仍写着 `src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt`。海洋运行只把这个字符串记入来源，不打开文件。权重在默认分支和清理前提交。
- `round_manifest.json` 与 `BRANCH_ISOLATION_MANIFEST.json` 仍提到已经移出工作区的路径。两份历史 JSON 没有改写。
- `python -m wt.active_model_identification.run` 在重建 M1 敏感电量时停止。这是 M1.1 修正空中能耗之后就有的行为，本轮没有改预算公式。

风浪感知和能耗预测没有开始。
