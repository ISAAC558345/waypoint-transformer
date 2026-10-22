# PAPER_NOTES

以下内容由 `docs/archive/reports/` 中的过程文档按原文件全文合并。合并后的正文是这些过程记录的归档副本。

## 源文件 `PAPER_ARCHITECTURE_FIGURE_SPEC_CN.md`

# 论文架构图设计说明（不改科研代码）

建议作为 Figure 1。只画 FINAL-HIER 主栈。图注写明：HGNN / 低层 PPO / Transformer 不属于最终主架构。

## 建议从左到右

1. **左：Environment State**  
   UAV set、Buoy set、queue、位置、service progress。标注 M=6、K=12。

2. **中左：DeepSets Set Encoder**  
   标题可用 “DeepSets Utility-Aware Encoder”。不要画 HGNN 主框。

3. **中：Utility-Aligned Order-Free Assignment Decoder**  
   输出 Assignment（合法、互斥、含 idle）。

4. **中右：Task Option, hold=15**  
   每 UAV：目标浮标或 idle。标注这是 Task Option / Task-Waypoint interface，不是空间 waypoint predictor。

5. **右：fly-toward + InRangeZeroMotionGate**  
   标题 “Deterministic Task Execution Rule”。服务区内 movement=0。

6. **最右/底部：UAV Movement + Communication Service**  
   箭头回到环境；底部标注 Payload。

## 图中必须出现的对象

UAV set，Buoy set，Assignment，Task Option，Movement，Payload。

## 不要出现在主路径上的模块

HGNN 编码器、低层 PPO、Transformer、旧空间 waypoint 预测器。如需对比，放到附录灰色框并标注 “not selected”。

当前仓库已有 post-processing 示意：`paper_assets/figures/fig01_architecture.png`（及 PDF）。论文排版可按本说明重绘矢量图，但不得改数据或算法。

## 源文件 `PAPER_BENCHMARK_DESCRIPTION_CN.md`

# 论文 Benchmark 说明

## 最终评测环境：Level-3-B

冻结配置来自 `L3BConfig` 与 Phase-10.5/10.6 清单，不是口头回忆。

| 量 | 值 |
| --- | --- |
| UAV 数 M | 6 |
| 浮标数 K | 12 |
| 视野 T | 48 |
| Task Option hold | 15 |
| 服务半径 | 16 m |
| 最大步长 | 22 m（max_speed×Δt） |
| 队列容量 | 4 blocks |
| block payload | 1e6 bits |
| 带宽 | 1e6 Hz |
| 发射功率 / 噪声 | 1.0 / 1e-3 |
| 路损指数 | 2.2 |
| 同簇/跨簇干扰系数 | 1.0 / 0.08 |
| 独占 assignment | 多 UAV 不得争同一浮标 |
| idle | 环境码 −1；decoder token 12 |

动态 queue、UAV–浮标关系、UAV–UAV 竞争、SINR 耦合、服务进度与累计 payload 都在同一闭环里结算。主指标：episode 累计 payload。辅指标：completed blocks、residual queue、drop、flight distance、legality。

## 主对照（Phase-10.6）

- M0 合法 Myopic
- M1 sequential Exclusive（冻结弱对照，不改定义）
- M2 DeepSets-old
- M3 FINAL-HIER

Final untouched：64 场景，与历史场景交集为 0，已一次性消费。

## 相对早期 Level-1 / Level-2

更早期环境缺少完整的长期队列、空间移动、任务竞争与 SINR 同时在线耦合，更适合接口/机制调试，不适合作为最终长期多 UAV 任务规划的正式性能表。

**不要**把不同 Level 上的 payload 放进同一张正式主表直接横向比较。Phase-6.4 的 hold 网格属于 L2-A 机制证据。Phase-10.5 是 pre-final 独立集，可写系统设计过程，主结论以 Phase-10.6 为准。

## 源文件 `PAPER_CONTRIBUTIONS_CN.md`

# 论文方法创新点草稿

不要把 HGNN 或低层 PPO 写成最终贡献。

## Contribution 1. 长期多 UAV–多浮标共享频谱任务规划问题与 Level-3-B benchmark

面向多 UAV、多浮标共享频谱的数据采集，显式包含动态队列、空间移动、独占任务竞争与 SINR 耦合。规模冻结 M=6、K=12、T=48、hold=15。该建模把“分给谁服务”和“几何上能否形成稳定链路”放在同一结算里，比早期无 SINR/弱竞争的 Level 更接近长期任务规划检验。

## Contribution 2. Utility-Aligned、Order-Free 的全局 assignment

提出 Utility-Aligned Global Assignment Policy：DeepSets 集合编码 + 置换等变 Order-Free 全局分配解码 + 以真实 hold 窗口 payload increment 对齐的集合监督。避免单一 reference assignment 的监督歧义，并消除固定 UAV 解码顺序依赖。HGNN 只是对照编码器，最终未入选。

## Contribution 3. Hierarchical Task Option 执行

将高层长期任务分配与低层运动执行分离。高层输出 Task Option（目标浮标或 idle，hold=15）；低层使用 Deterministic Task Execution Rule（fly-toward + InRangeZeroMotionGate），在服务阶段把位移置零以提高持续服务稳定性。学习型低层 PPO 没有稳定优势，不作为创新点。

## 源文件 `PAPER_EXPERIMENT_SECTION_CN.md`

# 论文实验章节骨架

建议编号可按全文调整。图表文件在 `paper_assets/`。

## 4.1 Experimental Setup

写清冻结栈 FINAL-HIER、checkpoint SHA、hold=15、不再训练。  
图：Figure 1 架构。  
数据：`FINAL_CODE_FREEZE.json`，Phase-10.0/10.5 清单。

## 4.2 Benchmark and Baselines

Level-3-B：M=6 K=12 T=48，SINR、队列、独占、idle。Baselines M0–M2。说明 10.6 为 final untouched（64，已消费）；10.5 仅为 pre-final。  
表：可引用配置表（`final_artifacts/configs/environment.json`）。  
文：`PAPER_BENCHMARK_DESCRIPTION_CN.md`。

## 4.3 Main Results

Figure 2 payload，Figure 3 paired delta，Figure 4 completed blocks，Figure 5 queue/drop，Figure 6 flight。  
表：`PAPER_TABLE_MAIN_RESULTS.csv`，`PAPER_TABLE_PAIRWISE_STATS.csv`。  
来源：仅 Phase-10.6。结论句用“提高采集与完成量，代价是更大飞行距离”。

## 4.4 Utility-Aligned Assignment Ablation

Figure 7。表：消融第 1 节。来源 Phase-9.4.5 UOBJ-A。不要把 confirmation 绝对值与 10.6 混表。

## 4.5 Encoder and Decoder Analysis

Figure 8a DeepSets vs HGNN；Figure 8b Order-Free PPV。  
来源 Phase-9.4.5 GRAPH-U-C、Phase-9.4.3 ORDERFIX-A。明确 HGNN 未入选。

## 4.6 Hierarchical Task Option Analysis

描述 hold=15 接口。可把 Phase-6.4 hold=5/10/15 放在此处并标注 L2-A 历史机制，禁止与 10.6 比绝对值。

## 4.7 Low-Level Execution Analysis

InRangeZeroMotionGate：Phase-10.3 design 数字 + Phase-10.4 B2−B1。  
PPO vs 规则：Phase-10.4 LLEXEC-C。可选用 10.4 历史图，须标注 independent validation，不是 untouched。

## 4.8 Limitations and Discussion

`PAPER_RESULTS_DISCUSSION_CN.md` + `PAPER_LIMITATIONS_CN.md`。不建议下一轮训练。

## 源文件 `PAPER_LIMITATIONS_CN.md`

# 论文限制

不隐藏负结果。

- 最终高层只用一枚冻结 DeepSets-util checkpoint（seed 0，SHA `dbd6196b23c111a88a4d86a9054ca1709e459e5b035c1d9210639bedd2dda109`），没有在最终系统上报告多高层 seed 区间。
- 没有证明 HGNN 优于 DeepSets（GRAPH-U-C）。
- 学习型低层没有稳定优势（LLEXEC-C）；最终执行器是规则。
- 当前 movement distance 未转换为真实能耗或推进功率。
- Level-3-B 仍是仿真 benchmark，不是海上实测。
- 未联合优化发射功率、波束、频谱调度等资源。
- hold=15 是当前冻结设置；L3-B 上没有正式 hold 网格。Phase-6.4 的 5/10/15 只属于 L2-A。
- final untouched 只能使用一次，不能再当调参或二次确认集。
- 结论局限于当前环境分布、到达过程与 64 个一次性场景。
- Exclusive baseline 很弱，夸大相对 M1 的增益会误导。
- FINAL-HIER 不是全指标最优：飞行距离明显更大，残差队列未优于 Myopic。
- Transformer 路径未进入最终系统，也没有显著贡献的最终证据。

## 源文件 `PAPER_MAIN_FINDINGS_CN.md`

# 论文可写结论分层

数据来源：Phase-10.6 主表；Phase-9.4.5 / 9.4.3 / 10.4 / 10.3 / 6.4 为消融或历史机制。名词：浮标；Task Option；Utility-Aligned Global Assignment Policy；Deterministic Task Execution Rule。

## A. 可以正式写进论文

1. FINAL-HIER 在 final untouched 上相对合法 Myopic 提高累计 payload（mean Δ=1234375，95% CI [531250, 1969140.625]，相对增益 0.1374）。来源 Phase-10.6。
2. FINAL-HIER 相对 DeepSets-old 有明显 payload 提升（mean Δ=3515625，CI [2703125, 4281250]，相对增益 0.5245）。来源 Phase-10.6。
3. Utility-Aligned 训练流程对高层任务分配有效（Phase-9.4.5 UOBJ-A；DeepSets confirmation payload 3200000 → 6950000）。最终系统沿用该高层。
4. FINAL-HIER 提高 completed blocks（10.21875 vs Myopic 8.984375 vs DeepSets-old 6.703125）。来源 Phase-10.6。
5. Order-Free decoder 作为最终组合任务分配接口保留（Phase-9.4.3 ORDERFIX-A，UAV/buoy/joint 等变=1，PPV=0）。
6. Task Option hold=15 作为最终时间扩展任务接口（系统冻结；L3-B 主实验未重做 hold 网格）。
7. InRangeZeroMotionGate 作为稳定执行规则（Phase-10.3/10.4；不是 PPO 学会驻留）。

同时应写：该 payload/完成量提升伴随更大飞行距离（M3 456.5 vs M0 80.4375）。

## B. 只能作为历史机制证据

- Phase-6.4：hold=5/10/15 在 **Level-2-A** 上 payload 6.00e6 / 8.50e6 / 1.10e7。支持时间扩展 option，不是 L3-B 主表。
- Phase-10.4：低层 PPO 多 seed 独立验证，LLEXEC-C，stable_low_level_advantage=false。
- Phase-9.4.5：HGNN-util vs DeepSets-util，GRAPH-U-C，均值差 −825000，无稳定图优势。

Phase-10.5 是 Pre-Final Independent Benchmark，可作系统设计过程，不要与 10.6 主表混成一张正式性能表。

## C. 不能写

- 不能写 HGNN 优于 DeepSets。
- 不能写低层 PPO 优于规则执行器。
- 不能写 Transformer 具有显著贡献。
- 不能写 FINAL-HIER 在所有指标全面最优。
- 不能把 flight distance 直接写成 energy consumption。
- 不能把 residual queue 写成全面改善：M3 残差队列 30.96875，略高于 Myopic 的 30.015625；drop 略降（75.8125 vs 78.0）。没有磁盘证据支持更细的机制故事。

## 源文件 `PAPER_METHOD_PIPELINE_CN.md`

# 论文方法流程（FINAL-HIER）

统一名称：FINAL-HIER。最终高层称为 Utility-Aligned Global Assignment Policy；最终低层称为 Deterministic Task Execution Rule。浮标一律称浮标，不称漂标。Task-Waypoint 最终语义是 Task Option / Task-Waypoint interface，不是未来空间 waypoint 预测器。

HGNN、低层 PPO、Transformer 不属于主模型。

## 职责分离

- **高层做什么**：在合法、互斥的全局 assignment 空间上，为 UAV 集合选择当前 Task Option（每机一个目标浮标或 idle），并在 hold=15 内保持该决策。高层不输出连续位移。
- **低层怎么执行**：不改 association。对非 idle 目标用 fly-toward 飞向浮标；一旦进入有效服务区且仍需服务，InRangeZeroMotionGate 将 movement 置零，以便持续占用服务几何。

## Step 1. 环境状态

输入当前多 UAV、多浮标状态：位置、队列、服务进度、到达过程、以及由环境给出的 SINR/竞争相关观测量。规模冻结为 M=6、K=12、T=48。

## Step 2. DeepSets 集合编码

DeepSets Utility-Aware Encoder 分别编码 UAV 集合与浮标集合，得到置换等变的集合表示。这是最终选用的编码器，不是 HGNN。

## Step 3. Order-Free Global Assignment Decoder

Permutation-Equivariant Order-Free Decoder 在剩余 UAV 与合法浮标∪idle 上联合打分，逐步组成一份合法全局 assignment。分配顺序由模型选择，而不是固定 UAV 下标循环。Decoder idle token=12，环境 idle=-1。

## Step 4. Utility-Aligned Supervision

训练时不以单一 reference assignment 为唯一模仿目标。对合法候选池估计冻结环境、hold=15、fly-toward 下的实际 payload increment，使高层概率质量偏向真实高 payload 组合。这是监督目标，不是低层强化学习。

## Step 5. Assignment → Task Option

将 assignment 写成每架 UAV 的 Task Option：目标浮标或 idle。低层不得改写该 association。

## Step 6. hold=15

Task Option 保持 15 步。到期后高层再根据新状态重选。hold=15 是当前冻结设置。

## Step 7. fly-toward

Deterministic Task Execution Rule 对非 idle 目标产生指向浮标的位移（步长受环境最大速度约束）。

## Step 8. InRangeZeroMotionGate

若 UAV 非 idle、目标合法、queue>0 且已在服务半径内，则把该机 movement 设为 (0,0)。这是确定性执行规则，不是学到的驻留策略。

## Step 9. 环境结算 payload

L3-B 根据 SINR 耦合、队列、服务进度与独占 assignment 语义计算实际累计 payload（block_payload_bits）。论文主指标是 episode 累计 payload，不是飞行距离。

## 源文件 `PAPER_RESULTS_DISCUSSION_CN.md`

# 论文结果讨论草稿

主数字一律来自 Phase-10.6 冻结表。飞行距离不是能耗。

## 1. FINAL-HIER vs Myopic

M3 payload 10218750 vs M0 8984375，配对 mean Δ=1234375，CI 下限 531250>0，场景正向比例 0.546875。完成块 10.219 vs 8.984。说明在合法贪心之上，全局 utility-aligned assignment 仍能提高数据采集。不是每一场景都赢（正向比例约 55%），论文应报告分布而不只报告均值。

## 2. FINAL-HIER vs DeepSets-old

相对旧目标 DeepSets，Δ=3515625，正向比例 0.8125，相对增益 0.5245。这是同一编码器家族、不同监督与最终执行栈下的系统差。主文可把它与 Phase-9.4.5 的 UOBJ-A 一起讨论：监督对齐是高层有效的关键证据之一。

## 3. 为什么 Exclusive 较弱，但不能过度强调

M1 payload 仅 1718750，Δ 很大。该对照是冻结的 UAV 下标 sequential exclusive，能力上限低。可以说明“合法但非组合优化的独占规则不够”，不要把它当成强学习 baseline 来夸大 FINAL-HIER。

## 4. Utility-Aligned supervision 的效果

Phase-9.4.5：DeepSets confirmation payload 3.20e6 → 6.95e6，regret 下降，Spearman 由负转正。最终 10.6 的 M3 vs M2 与该方向一致。不要把 9.4.5 confirmation 绝对值与 10.6 untouched 绝对值当成同一评测。

## 5. HGNN 为何最终未采用

GRAPH-U-C：HGNN-util 均值低于 DeepSets-util（−825000），跨 seed 不稳定。没有证据支持“图编码器是必要创新”。

## 6. PPO 低层为何最终未采用

Phase-10.4：5 seed 中仅 3 个 B2>B3，总体均值差为负，CI 跨 0，失败 seed 3/4 明显。不能写 PPO 优于规则。

## 7. 为什么规则执行器反而更稳定

冻结高层 assignment 后，fly-toward + 门控是确定、可复现的执行。PPO 在相同 gate 下没有稳定超过该规则，且部分 seed 飞行更长、换目标丢失更多（10.4 报告 B2 switch-loss / flight 高于 B3）。当前数据支持“执行层用确定性规则更稳”，不支持“低层必须学习”。

## 8. 为什么 flight distance 增加

M3 飞行距离 456.5，Myopic 仅 80.4。更高 payload 伴随更多跨目标移动与服务切换，而不是原地贪心。应写成代价，而不是效率全面胜利。没有能耗模型，到此为止。

## 9. residual queue 没有全面改善

M3 残差队列 30.96875，略高于 M0 的 30.015625；drop 75.8125 vs 78.0，略降。磁盘没有给出“队列结构被系统清空”的证据。只能写：完成块与 payload 上升的同时，终端残差队列并未相对 Myopic 全面下降。不要编造 SINR/热点/到达过程的额外故事。

## 10. 方法限制

单冻结高层 checkpoint；仿真 L3-B；hold 固定 15；未联合优化功率/波束；untouched 已消费；结论限于当前场景分布。详见 `PAPER_LIMITATIONS_CN.md`。

## 源文件 `PAPER_TABLE_ABLATION_EVIDENCE_CN.md`

# 论文消融证据表

只整理已有磁盘结果。L3-B 最终证据与 L2-A 历史机制必须分开引用。

## 1. Old objective vs Utility-Aligned objective

| 量 | DeepSets-old | DeepSets-util | HGNN-old | HGNN-util |
| --- | ---: | ---: | ---: | ---: |
| Confirmation payload | 3200000 | 6950000 | 3393750 | 6125000 |
| Candidate regret | 643750 | 368750 | 637500 | 368750 |
| Spearman(utility, logP) | −0.0260 | 0.2852 | −0.0047 | 0.3125 |

DeepSets util−old mean Δ=3750000，95% CI [2478125, 4878203]。Gate：UOBJ-A。来源 Phase-9.4.5 `summary.json`。

结论：Utility-Aligned 使高层分配更接近真实高 payload 候选，而不是单一 reference imitation。

## 2. DeepSets vs HGNN

HGNN-util − DeepSets-util = −825000。GRAPH-U-C，`stable_graph_value=false`。R4 mean=−700000，CI 跨 0。来源 Phase-9.4.5。

结论：HGNN 没有稳定优势，最终不采用。

## 3. Fixed-order vs Order-Free decoder

旧固定 UAV 顺序 AR：ORDER-C（报告 OSR≈0.39/0.55，优劣反转≈0.50）。Order-Free：等变审计 1.0000，PPV DeepSets/HGNN=0.0000，ORDERFIX-A。来源 Phase-9.4.3。

结论：消除 decoder 顺序伪影；最终接口保留 Order-Free。该阶段 **不能** 写成 HGNN 普遍优于 DeepSets（报告第 28 条：否）。

## 4. hold=5/10/15

| 设置 | hold | payload（L2-A） |
| --- | ---: | ---: |
| A_hold5 | 5 | 6000000 |
| A_hold10 | 10 | 8500000 |
| A_hold15 | 15 | 11000000 |

来源 Phase-6.4。**注明：Level-2-A 历史机制证据**，不可与 L3-B 主表横向比 payload 绝对值。

## 5. Low-level PPO vs gated fly-toward

独立 64 场景，5 个低层 seed。B3（gated fly-toward）mean payload=10671875。B2−B3 每 seed Δ：[1968750, 2734375, 218750, −1625000, −6218750]，正向 3/5，overall mean Δ=−584375，hierarchical CI 跨 0。LLEXEC-C。来源 Phase-10.4。

结论：PPO 无稳定优势。最终低层是规则，不是 PPO。

## 6. InRangeZeroMotionGate

Phase-10.3 design（非 untouched）：B0/B1/B2/B3 payload = 8.50e6 / 7.25e6 / 12.875e6 / 10.75e6；完成块 8.5 / 7.25 / 12.875 / 10.75。Phase-10.4：B2−B1 对 5/5 低层 seed 为正。

结论：进入服务区后驻留是稳定有效的**确定性执行机制**；提升不能记成 PPO 贡献。

## 源文件 `PAPER_TABLE_MAIN_RESULTS_CN.md`

# 论文主结果表（Phase-10.6 Final Untouched）

来源：`src/wt/htw_phase106/outputs/untouched/phase106_final_results.csv`。64 个一次性 final untouched 场景。未重跑环境。机器可读副本：`PAPER_TABLE_MAIN_RESULTS.csv`。

不要把 Level-1 / Level-2 或 Phase-10.5 pre-final 的 payload 与本表混排。

| method_id | 方法 | Payload | Completed Blocks | Residual Queue | Drop | Flight Distance | Legality |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M0 | Myopic | 8984375.0 | 8.984375 | 30.015625 | 78.0 | 80.4375 | 1.0 |
| M1 | Exclusive | 1718750.0 | 1.71875 | 37.328125 | 77.953125 | 129.9375 | 1.0 |
| M2 | DeepSets-old | 6703125.0 | 6.703125 | 33.609375 | 76.6875 | 249.90625 | 1.0 |
| M3 | FINAL-HIER | 10218750.0 | 10.21875 | 30.96875 | 75.8125 | 456.5 | 1.0 |

正式表述：FINAL-HIER improves data collection and service completion, at the cost of increased UAV travel distance.

中文：FINAL-HIER 提高了累计数据采集量与任务完成量，但伴随着更大的无人机运动距离。

Flight distance 不是能耗。不要写“全面优于所有 baseline”。
