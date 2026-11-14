# 海洋通信仿真研究分支

本分支是 `research/ocean-observation-isolated`。它运行两架无人机、四个静止浮标的服务、回传和返航，并用每架无人机自己的能耗系数做业务遥测。当前主试验的风是零、浮标不漂移。时空环境、跨机观测和候选航段能耗预测是后续接入点，还没有接到这条闭环上。

上游训练代码来自 [StanfordAI4HI/waypoint-transformer](https://github.com/StanfordAI4HI/waypoint-transformer)，许可见 `LICENSE`。旧通信系统对照在默认分支 [`feature/maritime-wt-stage1a`](https://github.com/ISAAC558345/waypoint-transformer/tree/feature/maritime-wt-stage1a)。

## 运行一个场景

在仓库根目录执行。场景使用已冻结的 M1.3 配置，不改写结果目录。

```powershell
$env:PYTHONPATH = "src"
py -3 -m wt.active_model_identification.run_m13 --scenario b1_tight --method lookahead_calibration_off
```

```bash
PYTHONPATH=src python -m wt.active_model_identification.run_m13 --scenario b1_tight --method lookahead_calibration_off
```

输入：

- 配置 `results/active_model_identification_m13/m13_20261114/frozen_config.json`
- 场景表 `results/active_model_identification_m13/m13_20261114/scenario_manifest.json`
- 场景 `b1_tight`，种子 `1101`，两机能耗系数 `[1.0, 1.0]`
- 方法 `lookahead_calibration_off`：正常业务遥测保留，专用校准不进入候选
- 风 `[0, 0]`，浮标坐标固定，备用电量 4000 J

输出是一行场景的 JSON：每个决策时刻两架无人机的请求动作和实际执行动作、按时交付 bit、总耗电，以及返航、未落地、低于备用电量和电量耗尽。命令只打印，不覆盖 `results/`。

同一入口还可以换 `--scenario`。冻结表里的场景是 `b1` 到 `b4` 与 `tight`、`window`、`loose` 的组合，例如 `b3_loose`。`--method lookahead_calibration_on` 是带专用校准的对照。

## 必要测试

```powershell
$env:PYTHONPATH = "src"
py -3 -m unittest tests.test_active_model_identification_m13 tests.test_active_model_identification_m15 tests.test_ocean_observation_m0 tests.test_ocean_observation_m0a tests.test_ocean_observation_m0b
```

`requirements.txt` 和 `environment.yml` 是上游训练环境的版本钉，跑上面的场景不需要安装它们。

## 代码入口

| 职责 | 文件 |
| --- | --- |
| 场景 | `configs/experiments/active_model_identification_m13.json`，`src/wt/active_model_identification/scenarios.py`，`src/wt/active_model_identification/spec.py` |
| 通信与回传 | `src/wt/active_model_identification/world.py`，`src/wt/ocean_observation_m0/radio.py`，`src/wt/ocean_observation_m0a/links.py` |
| 能耗与返航 | `src/wt/active_model_identification/physics.py`，`src/wt/active_model_identification/world.py`，`src/wt/ocean_observation_m0/flight.py` |
| 遥测与观测 | `src/wt/active_model_identification/belief.py`，`src/wt/active_model_identification/scenarios.py`，`src/wt/ocean_observation_m0/information.py` |
| 候选决策 | `src/wt/active_model_identification/policies.py`，`src/wt/active_model_identification/segment.py` |
| 评估 | `src/wt/active_model_identification/run_m13.py` |
| 风场航段预算 | `src/wt/ocean_observation_m0/flight.py` 的 `leg_budget` |
| 浮标漂移、时间戳与有效期 | `src/wt/ocean_observation_m0/information.py`，`src/wt/ocean_observation_m0/accounting.py` 的 `drift_report` |

M1.3 与 M1.5 的冻结结果分别在 `results/active_model_identification_m13/m13_20261114/` 和 `results/active_model_identification_m15/m15_20261114/`。M1.5 的 `b3_loose` 少交付 20000 bit，记录见 [M1.5 报告](docs/research/Active_Model_Identification_M15_Report_CN.md)。显式候选 `initial_guard_consistent` 在 [M1.6 报告](docs/research/Active_Model_Identification_M16_Report_CN.md) 中收回这 20000 bit；默认选择模式仍是 legacy。

`python -m wt.active_model_identification.run` 是 M1 的历史全量入口。M1.1 修正空中能耗之后，它在重建敏感电量时会停止，不能当作现在的快速开始。

分支范围和清理记录见 [README_BRANCH_SCOPE.md](README_BRANCH_SCOPE.md) 与 [清理记录](docs/research/Repository_Cleanup_20261114_CN.md)。
