# 海洋观测独立研究分支

分支：`research/ocean-observation-isolated`。

## 当前执行任务

2026-11-14 **M1.6 可执行候选与浮标分配一致性修正**。入口见 [CURSOR_NEXT_TASK.md](CURSOR_NEXT_TASK.md)，按 [M1.6 完整指令](docs/research/Active_Model_Identification_M16_Instructions_CN.md) 执行。

先复核 b3_loose 在 180 s 的期初准入，将被原守卫即时改成 RETURN 的 service 标签规范化，避免继续占用浮标。继续使用 M1.5 尾估值，保留原冲突强弱规则与精度。本轮以正常业务遥测的 OFF 为主要候选、ON 为对照，先行两条中主要 OFF 满足保留标准后补齐固定 24 条新航次，直接复用旧基线，逐场景确认交付与净效用。完成后推送本研究分支并停止，未经用户明确允许不合并主分支，修改其他分支前先询问用户。

## M1.5 已完成的结果

2026-11-14 **M1.5 已完成**：无正收益业务分支的返航尾成本修正。报告见 [M1.5 结果](docs/research/Active_Model_Identification_M15_Report_CN.md)，结果在 `results/active_model_identification_m15/m15_20261114/`。指令见 [M1.5 指令](docs/research/Active_Model_Identification_M15_Instructions_CN.md)。

候选 `return_tail_when_no_business` 只在显式打开时替换旧 none 分支的零终端。六个快照的 legacy 评分与 M1.4 保存值一致，三条 single_score 快照确认负尾进入正式 Q。24 条新航次对照 M1.3 核心基线。唯一回退是 b3_loose，同开关都少 20000 bit；其余 11 个场景 bit 与 J 不回退。该负结果保留，不改默认模式，不重跑旧基线。

## M1.4 已完成的结果

2026-11-14 **M1.4 已完成**：校准请求来源和一步价值分解。报告见 [M1.4 结果](docs/research/Active_Model_Identification_M14_Report_CN.md)，结果在 `results/active_model_identification_m14/m14_20261114/`。入口状态见 [CURSOR_NEXT_TASK.md](CURSOR_NEXT_TASK.md)。

76 条请求中 9 条来自冲突修复，67 条是单机首选。六个固定快照上，正差距来自无更新的运动或状态项；信息消融的续跑效用差为 0。这些结果保持原样，诊断阶段已结束。M1.5 仅按上述独立候选指令推进，不重跑旧归因或旧三档基线。

## M1.3 已完成的结果

2026-11-14 **M1.3 已完成**：三档资源下的闭环试跑。报告见 [M1.3 结果](docs/research/Active_Model_Identification_M13_Report_CN.md)，结果在 `results/active_model_identification_m13/m13_20261114/`。入口状态见 [CURSOR_NEXT_TASK.md](CURSOR_NEXT_TASK.md)。

24 条核心航次和 8 条 window 消融已经跑完。分段积分通过，主试用 15 点外层和 7 点内层。三档允许校准的净效用都低于关闭校准，按时比特相同，消融差为 0。这些负结果保持原样，不据此重跑或改预算。

配置与结果分别保存在 `configs/experiments/active_model_identification_m13.json` 和 `results/active_model_identification_m13/<run_id>/`。不覆盖历史，不训练、不扩风浪/规模、不自动合并。

M1.4 复核补充：76 条校准请求中 9 条由冲突修复产生，67 条为单机评分首选。已有 window 消融只测试首次校准观测和关闭后续校准的续策，不能概括全部校准信息。备用电量标准不满足与物理未返航分开报告；M1.3 原始结果保留。

## M1.2 已完成的诊断

实现提交 `94f19e8a67df1bac968b3d90a92449b4ad6fd71a`，见 [M1.2 报告](docs/research/Active_Model_Identification_M12_Report_CN.md) 与 `results/active_model_identification_m12/m12_20261114/`。

24 项必要测试通过，完整任务报价与实际执行一致，后验矩积分通过；原紧预算下零噪声校准后的保守分位仍不放行远任务，临界 Q 的离散积分差未满足原标准，所以主试和消融均未运行。该结果不等同于校准收益为零。

M1.3 已把代表观测窗口诊断与场景可运行性分开，并保留原紧预算作为可能有负收益的有效任务；中间预算按事前公式登记，实际风险规则没有改。M1.2 代码、报告与原始结果作为历史证据保留。

## M1.1 已记录的进展与待复核判断

M1.1 实现提交为 `427a360fcc9ac423dc82a687f31b8721410ff135`，见 [M1.1 报告](docs/research/Active_Model_Identification_M11_Report_CN.md) 与 `results/active_model_identification_m11/m11_20261114/`。远端回传/等待改为空中悬停，机坪降落和交付时刻得到修正，16 项必要测试通过；登记场景未通过，主试和信息消融均未运行。

后续审查发现 service 准入仍漏算完整回传成本，且参数积分候选没有全部达到登记容差却回退选用。因此本轮先纠正成本与数值判定，再复核场景；不把 M1.1 的 `INFEASIBLE` 解释成已证明没有可行电量窗口，也不把未运行对照解释成零收益。原始 M1.1 文件保留。

## 已完成的 M1 与既有核查

M1 已于 2026-11-13 完成，历史入口为 [M1 报告](docs/research/Active_Model_Identification_M1_Report_CN.md)，结果保存在 `results/active_model_identification_m1/m1_20261113/`。保留原始 32 条核心任务和 8 项必要测试记录，不覆盖或重复当作新结果。M1 已知的回传推进功耗问题由 M1.1 修正，旧性能输出不作为修正后物理模型的证据。

此前已有功能的 18 项针对性测试、6 项结果复算及论文生成检查保留为各自历史记录。本轮的物理、数值核查与新试验须单独记录。

当前使用固定执行器，不启动神经网络训练或风浪扩展。代码、测试与结果继续推送本研究分支；主分支、默认分支和原系统维护分支的合并由用户另行决定，不创建自动合并流程。

## 历史隔离保存记录

以下保留 2026-11-04 的隔离背景，其“暂不执行”状态已被上面的 M1、M1.1、M1.2、M1.3、M1.4、M1.5 与 M1.6 指令在指定范围内更新。历史 M0、M0A、M0B 结果和原始来源清单仍作为各自证据保留。

分支：`research/ocean-observation-isolated`。依据用户2026-11-04当时的指令，新方案暂不继续修改或执行，未来是否合并原系统由用户另行决定。

本分支保存M0、M0A、M0B全部代码、结果、测试和方案文档，以及最新文档分支中的方案指令。61个新方案专用文件直接复用原Git blob；本次没有改变其内容。混合阶段的交接文档按原blob保存在 `archive/handoff_20261104/`，仅作历史记录，不是当前执行指令。

最新结果来源为 `01490662dea1d79d91a478df912eeee684e90072`。M0B记录B2与C在六个上下文中策略等价；隔离保存不代表采用或并入原系统。

旧系统在 [独立维护分支](https://github.com/ISAAC558345/waypoint-transformer/tree/feature/legacy-system-maintenance) 继续完善。原正式分支和已有研究分支均保留。当前不要根据历史交接自动推进新方案，不创建自动合并流程。

文件与来源清单见 `BRANCH_ISOLATION_MANIFEST.json`。2026-11-04 的分支整理本身未运行新方案测试、训练或仿真。



