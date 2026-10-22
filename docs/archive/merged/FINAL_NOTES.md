# FINAL_NOTES

以下内容由 `docs/archive/reports/` 中的过程文档按原文件全文合并。合并后的正文是这些过程记录的归档副本。

## 源文件 `FINAL_ABLATION_SUMMARY_CN.md`

# 最终消融证据汇总（只整理历史结果，不新算）

区分：**当前 L3-B 最终证据** vs **历史机制证据**。

## A. Order-Free Decoder — Phase-9.4.3

当前 L3-B 架构证据。ORDERFIX-A：置换等变全局 assignment decoder 消除固定 UAV 顺序 AR 的顺序依赖。最终接口保留该 decoder。

## B. Utility-Aligned Objective — Phase-9.4.5

当前 L3-B 最终证据。UOBJ-A：相对旧 single-reference 监督，独立 confirmation 上 payload 稳定提升。最终高层使用 utility-aligned DeepSets。

## C. DeepSets vs HGNN — Phase-9.4.5

当前 L3-B 最终证据。GRAPH-U-C：HGNN-util 未稳定优于 DeepSets-util。`HGNN_selected=false`。

## D. Task Option / hold — Phase-6.4

**历史 L2-A 机制证据**，不是当前 L3-B 正式主表。hold=5/10/15 的 payload 趋势支持时间扩展任务接口。最终系统冻结 hold=15，本阶段未在 L3-B 重做 hold 网格。

## E. 低层 PPO vs rule — Phase-10.4

当前 L3-B 独立验证。LLEXEC-C：gated PPO 未稳定优于 gated fly-toward。`low_level_PPO_selected=false`。

## F. InRangeZeroMotionGate — Phase-10.3 / 10.4

当前 L3-B 证据。10.3 design 闭环改善完成量；10.4 中 B2−B1 对 5/5 低层 seed 为正。最终把它当作**确定性执行规则**，不是 PPO 贡献。

## G. Final system — Phase-10.6

当前 L3-B **final untouched** 证据。FINAL-A，`final_system_confirmed=true`。M3 相对 M0/M2 的 payload CI 下限 > 0。代价是更大飞行距离。该集已消费，不得再跑。

## 源文件 `FINAL_PROJECT_MANIFEST.md`

# 最终工程目录清单

## A. 最终模型

- `src/wt/final/pipeline.py`
- `src/wt/final/run_final_hier.py`
- 复用：`src/wt/htw_phase100/high_level.py`、`src/wt/htw_phase943/order_free_decoder.py`、`src/wt/htw_phase103/gate.py`、`src/wt/l3b/`

## B. 最终配置

- `final_artifacts/configs/environment.json`
- `final_artifacts/configs/model.json`
- `final_artifacts/configs/decoder.json`
- `final_artifacts/configs/executor.json`
- `final_artifacts/configs/evaluation.json`

## C. 最终结果

- `final_artifacts/results/final_results_table.csv`
- `final_artifacts/results/final_pairwise_statistics.csv`
- 源：`src/wt/htw_phase106/outputs/untouched/`（只读）

## D. 论文图

- `scripts/plot_final_results.py` → `final_artifacts/figures/`
- Phase-10.6 原始图仍在 `src/wt/htw_phase106/outputs/untouched/`

## E. 历史实验

- `src/wt/htw_phase*`、`src/wt/round*`、`src/wt/htw_phase61`–`64`、HGNN/PPO/TW 代码与报告
- 根目录 `HTW_Phase*.md`

## F. Checkpoint

- `final_artifacts/checkpoints/deepsets_util_seed0.pt`
- 源归档：`src/wt/htw_phase100/outputs/handoff/frozen_deepsets_util/deepsets_util_seed0.pt`

## G. 测试

- `tests/test_htw_phase11.py`（工程测试，禁止 untouched）
- 历史：`tests/test_htw_phase10*.py` 等

## H. 复现入口

- `python -m src.wt.final.run_final_hier`
- `python -m wt.final.pack_artifacts`
- `python scripts/plot_final_results.py`
- `FINAL_README_CN.md`

## 源文件 `FINAL_README_CN.md`

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

## 源文件 `FINAL_RESEARCH_SUMMARY_CN.md`

# 最终研究摘要（汇报 / 论文 / 答辩）

## 研究问题

多架无人机如何在共享频谱、动态队列与空间移动约束下，为海上浮标集合做长期合法任务分配，并完成数据采集。需要同时处理 UAV–浮标匹配、UAV–UAV 竞争和 SINR 耦合，而不是只做单步就近服务。

## 最终方法

FINAL-HIER。高层：Utility-Aligned Global Assignment Policy，即 DeepSets Utility-Aware Encoder + Permutation-Equivariant Order-Free Global Assignment Decoder + Utility-Aligned Supervision。接口：Task Option（目标浮标或 idle），hold=15。低层：Deterministic Task Execution Rule，即 fly-toward + InRangeZeroMotionGate。HGNN、低层 PPO、Transformer 不是主模型。

## Benchmark

Level-3-B：M=6，K=12，T=48，hold=15，服务半径 16 m，独占 assignment，idle，动态 queue，SINR。主指标累计 payload。正式主表只使用 Phase-10.6 final untouched（64 场景，已消费）。不要与 Level-1/2 或 Phase-10.5 pre-final 混表。

## 主要实验（均已冻结）

- Phase-9.4.3：Order-Free decoder，ORDERFIX-A。
- Phase-9.4.5：Utility-Aligned 目标与 DeepSets vs HGNN；UOBJ-A，GRAPH-U-C。
- Phase-10.3/10.4：门控驻留与低层 PPO 多 seed；LLEXEC-C。
- Phase-10.5：架构冻结与 pre-final 独立集。
- Phase-10.6：一次性 final untouched，FINAL-A。

## 最终结果（Phase-10.6 磁盘）

Payload：Myopic 8984375，Exclusive 1718750，DeepSets-old 6703125，FINAL-HIER 10218750。  
FINAL-HIER vs Myopic：mean Δ 1234375，CI [531250, 1969141]，相对增益 13.7%。  
vs DeepSets-old：mean Δ 3515625，相对增益 52.4%。  
Completed blocks：10.219 vs 8.984 / 6.703。合法性均为 1。  
飞行距离：456.5 vs Myopic 80.4。残差队列未优于 Myopic。

一句话：FINAL-HIER 提高了累计数据采集量与任务完成量，但伴随着更大的无人机运动距离。

## 消融

Utility-Aligned 显著提高高层 confirmation payload。Order-Free 消除顺序伪影。HGNN 无稳定优势。门控驻留有效且属于规则。PPO 低层无稳定优势。hold=5/10/15 仅 L2-A 机制证据。

## 失败路线

HGNN 未成为最终编码器。低层 PPO 未成为最终执行器。Transformer 未进入最终栈。旧空间 waypoint 预测器已退役。弱 Exclusive 对照不能用来夸大贡献。

## 最终结论

在冻结的 L3-B final untouched 上，效用对齐的全局 assignment 加上时间扩展 Task Option 与确定性服务区驻留，能够提高合法数据采集，但以更多飞行为代价。方法选择由负结果约束：图编码器与学习低层都没有被数据支持。

## 限制

单高层 checkpoint；仿真；无真实能耗；未优化功率/波束；hold 固定；untouched 不可再用。

## 代码复现入口

`python -m src.wt.final.run_final_hier --seed 8`（demo，禁止 106000–106063）。  
论文图：`python scripts/plot_paper_assets.py`（只读已存表）。  
checkpoint SHA：`dbd6196b23c111a88a4d86a9054ca1709e459e5b035c1d9210639bedd2dda109`。  
说明：`FINAL_README_CN.md`。
