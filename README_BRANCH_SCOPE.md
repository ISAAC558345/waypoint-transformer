# 海洋观测独立研究分支

分支：`research/ocean-observation-isolated`。

## 当前执行任务

2026-11-13，用户指定的主动功耗辨识 M1 已在本分支完成并记录。入口说明见 [CURSOR_NEXT_TASK.md](CURSOR_NEXT_TASK.md)，结果见 [M1 报告](docs/research/Active_Model_Identification_M1_Report_CN.md)。不要重复已完成的 32 条核心任务。

已有功能的 18 项针对性测试、6 项结果复算和论文生成检查此前已通过。M1 新增 8 项必要测试，记录在 `results/active_model_identification_m1/m1_20261113/necessary_tests.log`。

本轮没有启动神经网络训练。未经用户明确允许，不得合并到主分支、默认分支或原系统维护分支。

## 历史隔离保存记录

以下保留 2026-11-04 的隔离背景，其“暂不执行”状态已被上面的本轮 M1 指令在指定范围内更新。历史 M0、M0A、M0B 结果和原始来源清单仍作为各自证据保留。

分支：`research/ocean-observation-isolated`。依据用户2026-11-04当时的指令，新方案暂不继续修改或执行，未来是否合并原系统由用户另行决定。

本分支保存M0、M0A、M0B全部代码、结果、测试和方案文档，以及最新文档分支中的方案指令。61个新方案专用文件直接复用原Git blob；本次没有改变其内容。混合阶段的交接文档按原blob保存在 `archive/handoff_20261104/`，仅作历史记录，不是当前执行指令。

最新结果来源为 `01490662dea1d79d91a478df912eeee684e90072`。M0B记录B2与C在六个上下文中策略等价；隔离保存不代表采用或并入原系统。

旧系统在 [独立维护分支](https://github.com/ISAAC558345/waypoint-transformer/tree/feature/legacy-system-maintenance) 继续完善。原正式分支和已有研究分支均保留。当前不要根据历史交接自动推进新方案，不创建自动合并流程。

文件与来源清单见 `BRANCH_ISOLATION_MANIFEST.json`。2026-11-04 的分支整理本身未运行新方案测试、训练或仿真。
