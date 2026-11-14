# 海洋观测独立研究分支

分支：`research/ocean-observation-isolated`。

## 当前执行任务

2026-11-14 当前阶段为 **M1.2：完整任务成本统一与校准价值验证**。入口见 [CURSOR_NEXT_TASK.md](CURSOR_NEXT_TASK.md)，详细步骤见 [M1.2 执行指令](docs/research/Active_Model_Identification_M12_Instructions_CN.md)。本次提交发布指令，M1.2 尚待执行。

先统一 service 准入、终端价值、场景登记与真实完整业务链的成本和时间，复核原候选预算；修正空中 wait 的正常返航核查及积分选择状态。核查通过后冻结配置，运行同一规划器下仅切换校准权限的 16 条核心航次，最多 8 条信息消融续跑，分别计数。

新配置为 `configs/experiments/active_model_identification_m12.json`，结果为 `results/active_model_identification_m12/<run_id>/`，报告为 `docs/research/Active_Model_Identification_M12_Report_CN.md`。不覆盖此前产物、不训练、不扩风浪或节点规模、不自动合并。

## M1.1 已记录的进展与待复核判断

M1.1 实现提交为 `427a360fcc9ac423dc82a687f31b8721410ff135`，见 [M1.1 报告](docs/research/Active_Model_Identification_M11_Report_CN.md) 与 `results/active_model_identification_m11/m11_20261114/`。远端回传/等待改为空中悬停，机坪降落和交付时刻得到修正，16 项必要测试通过；登记场景未通过，主试和信息消融均未运行。

后续审查发现 service 准入仍漏算完整回传成本，且参数积分候选没有全部达到登记容差却回退选用。因此本轮先纠正成本与数值判定，再复核场景；不把 M1.1 的 `INFEASIBLE` 解释成已证明没有可行电量窗口，也不把未运行对照解释成零收益。原始 M1.1 文件保留。

## 已完成的 M1 与既有核查

M1 已于 2026-11-13 完成，历史入口为 [M1 报告](docs/research/Active_Model_Identification_M1_Report_CN.md)，结果保存在 `results/active_model_identification_m1/m1_20261113/`。保留原始 32 条核心任务和 8 项必要测试记录，不覆盖或重复当作新结果。M1 已知的回传推进功耗问题由 M1.1 修正，旧性能输出不作为修正后物理模型的证据。

此前已有功能的 18 项针对性测试、6 项结果复算及论文生成检查保留为各自历史记录。本轮的物理、数值核查与新试验须单独记录。

当前使用固定执行器，不启动神经网络训练或风浪扩展。代码、测试与结果继续推送本研究分支；主分支、默认分支和原系统维护分支的合并由用户另行决定，不创建自动合并流程。

## 历史隔离保存记录

以下保留 2026-11-04 的隔离背景，其“暂不执行”状态已被上面的 M1、M1.1 与 M1.2 指令在指定范围内更新。历史 M0、M0A、M0B 结果和原始来源清单仍作为各自证据保留。

分支：`research/ocean-observation-isolated`。依据用户2026-11-04当时的指令，新方案暂不继续修改或执行，未来是否合并原系统由用户另行决定。

本分支保存M0、M0A、M0B全部代码、结果、测试和方案文档，以及最新文档分支中的方案指令。61个新方案专用文件直接复用原Git blob；本次没有改变其内容。混合阶段的交接文档按原blob保存在 `archive/handoff_20261104/`，仅作历史记录，不是当前执行指令。

最新结果来源为 `01490662dea1d79d91a478df912eeee684e90072`。M0B记录B2与C在六个上下文中策略等价；隔离保存不代表采用或并入原系统。

旧系统在 [独立维护分支](https://github.com/ISAAC558345/waypoint-transformer/tree/feature/legacy-system-maintenance) 继续完善。原正式分支和已有研究分支均保留。当前不要根据历史交接自动推进新方案，不创建自动合并流程。

文件与来源清单见 `BRANCH_ISOLATION_MANIFEST.json`。2026-11-04 的分支整理本身未运行新方案测试、训练或仿真。


