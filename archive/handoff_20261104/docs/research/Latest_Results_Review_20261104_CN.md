# R1 与 M0A 最新结果审阅

更新日期：2026-11-04。旧系统 R1 已在执行机器上完成正式路径复现；新方向 M0A 完成解析比较，但移动收益的归因还缺少同资源固定驻点对照。下一步优先 R1A 修复结果交付，独立开展 M0B 归因修订。沿用 [必要测试规则](Research_Priority_and_Necessary_Testing_CN.md)。

## 1 原系统 R1

固定来源：`1abcf1b264b6df1763cbcf2cfe410923cdc57dae`；运行代码 `ed2c87c83dcdaccee1420ee3e14a8dfa19982e92`。

| TRAIN场景 | Scheme C Mbit | G1_RATE k=6 Mbit | 配对差 Mbit |
| --- | --- | --- | --- |
| 322000 | 63.785945 | 64.731142 | -0.945196 |
| 322001 | 68.952074 | 63.445122 | 5.506952 |

四臂均完成48步，192步预算恰好用完，决策为0/15/30/45；实际served累计与payload一致。四项必要检查通过；正式策略文件和模型身份保持。这个结果支持当前入口可运行，不能据两个训练场景判定总体优劣。

已恢复40份原始文件，包括AB、AC汇总和相应episode记录；恢复不是重跑holdout。ranking-loss的原summary仍缺失。206历史失败继续按已有日志分类，本次没有重跑全历史套件。

## 2 R1 当前交付缺口

`runner.py` 把执行机器绝对路径存入 `policy_path`，已存结果检查直接读取该路径。把结果移到另一检出目录后，这个检查不能依靠新根目录工作。

`run.py` 默认发现已有episode会执行 `finalize_saved()`，调用 `_write_manifest` 用当前机器的依赖、路径和模型信息重写清单。该流程混合了原运行来源和后续整理环境。已有192步数值不需要重跑；应修复读取与重新运行的边界。

[R1A指令](Legacy_Formal_Repro_R1A_Instructions_CN.md) 增加只读检查入口、可迁移路径和独立派生视图。新研究环境步、训练、模型加载均为零。

## 3 M0A 的有效范围

固定来源：`12ff4ae7fdf3863ac56c3a81bd2168983453043b`。地平线口径修正为显式k=4/3；原6个上下文及18条方法结果已提交，漂移没有进入任何一臂。

- 原2km域：直传完成8包，系统约3960J，中继约18164J。
- 单一远区：固定中继和当前C交付、时延及能耗相同。
- 分离区域：当前单驻点B未覆盖；C用两架无人机分别在东、西两个接收点完成8包。

分离区域C的系统合计约147.3kJ（静风）和169.6kJ（有风）；单机约71.7/75.3kJ与82.4/86.9kJ，均低于161.7kJ单机上限。系统总量和单机限制不能混用。

## 4 需要收窄的归因

B的实现是一个固定接收点，另一架无人机闲置。C为每架无人机分配一个点，在两个产品窗口重复使用相同点；并没有在多个服务区之间动态切换。因此现有结果支持双驻点相对单驻点的覆盖差，尚不能证明动态移动调度有额外收益。

应增加B2：允许两架无人机各一个固定接收点，并使用与C相同的进出和发包时机。驻点固定不要求在两次产品生成之间持续悬停。若B2与C的任务和时序相同，应报告策略等价，不能把重合结果当作多个独立证据。

另外，`_fly` 固定初始化UAV编号0，使部分C输出身份重复；能量按列表位置计量，暂未发现因此重复收费，但记录应修正。现有回传时刻还存在两架机并行完成的记录，没有统一岸站串行化。绝对时延结论需明确通信资源合同，不能将独立链路计时直接解释为共享单信道结果。

[M0B指令](Ocean_Observation_M0B_Instructions_CN.md) 仅补同资源固定驻点对照及直接影响该比较的记录/时序问题，不增加几何、训练或新环境回合。

## 5 后续顺序

先把原系统结果变成可迁移、只读、来源准确的交付。无冲突时并行处理M0B。两包都完成后，再用恢复的原始证据同步论文，并讨论旧系统到达感知方法的有限A/B；新方向依据B2对照决定采用驻点部署还是确有必要的动态调度。当前不启动训练或海况估计闭环。

执行入口见 [下一步交接](Next_Execution_20261104_CN.md)。

## 6 证据

- [R1报告](https://github.com/ISAAC558345/waypoint-transformer/blob/1abcf1b264b6df1763cbcf2cfe410923cdc57dae/docs/research/Legacy_Formal_Repro_R1_Report_CN.md)
- [R1清单](https://github.com/ISAAC558345/waypoint-transformer/blob/1abcf1b264b6df1763cbcf2cfe410923cdc57dae/results/legacy_formal_repro_r1/run_manifest.json)
- [R1运行与整理入口](https://github.com/ISAAC558345/waypoint-transformer/blob/1abcf1b264b6df1763cbcf2cfe410923cdc57dae/src/wt/legacy_formal_repro_r1/run.py)
- [M0A报告](https://github.com/ISAAC558345/waypoint-transformer/blob/12ff4ae7fdf3863ac56c3a81bd2168983453043b/docs/research/Ocean_Observation_M0A_Report_CN.md)
- [M0A时序实现](https://github.com/ISAAC558345/waypoint-transformer/blob/12ff4ae7fdf3863ac56c3a81bd2168983453043b/src/wt/ocean_observation_m0a/schedule.py)
