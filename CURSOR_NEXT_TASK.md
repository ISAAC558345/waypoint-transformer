# Cursor 当前执行任务

当前分支：`research/ocean-observation-isolated`。

2026-11-14 **M1.3 已完成**。三档闭环试跑、8 条 window 消融和再评分已经写入结果，不再继续加航次或改预算。

- [M1.3 结果报告](docs/research/Active_Model_Identification_M13_Report_CN.md)
- 结果目录：`results/active_model_identification_m13/m13_20261114/`
- 数值状态 PASS。主试使用分段评分，外层 15 点、内层 7 点。7 点外层的 |ΔQ| 为 0.010327，没有保留。
- 核心航次 24，消融续跑 8，再评分快照 12 且排名无变化。
- window 预算按事前公式算出，两架都是 58319.37060582891 J。tight 与 loose 保持原登记值。
- 三档里允许校准的净效用都更低，按时比特相同，首次服务目标没有变化。消融净效用差为 0。

没有新的执行指令。不要把本轮负的效用差当成待修错误去重跑，不要搜索新电池，不要把 L8/L9 或未登记场景接进规划器。

保留历史：[M1.3 指令](docs/research/Active_Model_Identification_M13_Instructions_CN.md)、[M1.2 报告](docs/research/Active_Model_Identification_M12_Report_CN.md)、[M1.1 报告](docs/research/Active_Model_Identification_M11_Report_CN.md)、[M1 报告](docs/research/Active_Model_Identification_M1_Report_CN.md) 及各自结果。

代码和结果只在当前研究分支。不训练、不扩风浪/规模、不覆盖历史产物，也不合并到主分支、默认分支或 `feature/legacy-system-maintenance`。
