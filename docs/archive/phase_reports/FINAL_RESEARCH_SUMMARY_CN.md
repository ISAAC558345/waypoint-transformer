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
