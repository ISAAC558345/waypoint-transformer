# FINAL-HIER 复现说明

这不是训练入口。Final untouched 已消费，不能再当调参或验证集。

## 1. 项目目标

在 Level-3-B（SINR、竞争服务）上，用冻结的分层控制器完成海上浮标数据采集：高层做合法全局 assignment，低层用规则飞向目标并在服务区内零位移驻留。

## 2. 最终方法架构

FINAL-HIER =

DeepSets-util
→ Permutation-Equivariant Order-Free Decoder
→ assignment / Task Option（hold=15）
→ fly-toward
→ InRangeZeroMotionGate
→ L3-B environment

HGNN、低层 PPO、Transformer、旧 Task-Waypoint **不是**最终主模型。

## 3. 目录结构

- `src/wt/final/`：最终 facade 与 demo 入口
- `final_artifacts/`：最终 checkpoint、配置、结果表、图
- `src/wt/htw_phase*`、`src/wt/round*`：历史实验（保留）
- `scripts/plot_final_results.py`：只读已保存表的绘图

## 4. 安装依赖

Python 与包版本见 `final_artifacts/requirements.txt`（由当前解释器记录：Python 3.10.20，PyTorch 2.7.1+cu128，NumPy 1.26.4，matplotlib 3.10.9）。仓库根目录 `requirements.txt` 是更早的 Decision Transformer 实验依赖，**不要盲目升级**。

建议：已有的 `.venv-dt-gym`，并设置 `PYTHONPATH=src` 或从仓库根目录运行 `python -m src.wt.final.run_final_hier`。

## 5. 如何运行最终模型

```bash
python -m src.wt.final.run_final_hier --seed 8
```

或：

```bash
set PYTHONPATH=src
python -m wt.final.run_final_hier --seed 8
```

## 6. Demo 场景

默认 `--seed 8`（design/demo）。禁止 `106000–106063`（Phase-10.6 untouched）。

## 7. 如何重新生成论文图

```bash
python scripts/plot_final_results.py
```

只读取 `src/wt/htw_phase106/outputs/untouched/` 中已保存的 CSV/JSON，**不**再跑环境和模型。

## 8. 主要 benchmark

- Phase-10.5：Pre-Final Independent Benchmark（已消费设计用途）
- Phase-10.6：Final Untouched Test（一次性，已消费）

Primary 对照：M0 合法 Myopic，M1 sequential Exclusive，M2 DeepSets-old，M3 FINAL-HIER。

## 9. 最终实验结果摘要（Phase-10.6 磁盘）

M0/M1/M2/M3 payload ≈ 8.984e6 / 1.719e6 / 6.703e6 / 1.022e7。FINAL-A，`final_system_confirmed=true`。飞行距离 M3 更高。详见 `final_artifacts/results/`。

## 10. 历史模块

HGNN、低层 PPO、Transformer、旧 TW / Hungarian / 固定顺序 AR 均保留在历史 phase 目录中，仅作研发与消融证据。

## 11. 重要限制

无严格能耗模型；飞行距离 ≠ 能耗。M1 是冻结的 UAV 下标 sequential exclusive，弱但不改定义。单高层 seed0。

## 12. 复现注意事项

- 不得重新训练或改 hold/gate/reward。
- 不得再次访问 untouched 做调参。
- checkpoint SHA 必须为 `dbd6196b23c111a88a4d86a9054ca1709e459e5b035c1d9210639bedd2dda109`。
