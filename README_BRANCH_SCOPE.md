# 海洋观测独立研究分支

分支：`research/ocean-observation-isolated`。

## 当前执行任务

[W6.1 本机测风与跨机风报文的增量验证](CURSOR_NEXT_TASK.md) 已完成。W6 已归档。

## W6.1 已完成的结果

2026-11-15 **W6.1 已完成**：接收机自己测风之后，再加入 0 号机的早发报文。报告见 [W6.1 结果](docs/research/Wind_Observation_W61_Report_CN.md)，结果在 `results/wind_observation_w61/w61_20261115/`。

飞行期 10 组里有 9 组，加上远端报文后的 RMSE 更低。局部场 11034 的 30 秒高 0.075 m/s。等到达时刻下，位置更好的飞机随种子变化。到 240 秒多用 4 条报文、128 字节。默认调度不变。

## W6 已完成的结果

2026-11-15 **W6 已完成**：用第一拍的 30 条最新样本辨认全局均值和局部扰动。报告见 [W6 结果](docs/research/Wind_Observation_W6_Report_CN.md)，结果在 `results/wind_observation_w6/w6_20261115/`。

240 秒、4 条报文时，五个场的真实模型权重都高于 0.999。对照场没有偏向局部模型。飞行期加权预报相对真实类别的固定模型，对照场高 0.053 m/s 和 0.028 m/s，局部场的配对差正负都有。默认调度不变。

## W5.1 已完成的结果

2026-11-15 **W5.1 已完成**：每两拍只发一条最新风样本，比较固定早发、固定晚发和在线选择。报告见 [W5.1 结果](docs/research/Wind_Observation_W51_Report_CN.md)，结果在 `results/wind_observation_w51/w51_20261115/`。

在线规则在对照场和局部场都是 30 次早发、0 次晚发，误差与固定早发相同。对照场早发比晚发低 0.323 m/s 和 0.396 m/s。局部场有一个种子晚发更低，另外三个早发更低。默认调度不变。

## W5 已完成的结果

2026-11-15 **W5 已完成**：在冻结的 W4 逆风轨迹上比较最新风样本和按预测航段挑选的样本。报告见 [W5 结果](docs/research/Wind_Observation_W5_Report_CN.md)，结果在 `results/wind_observation_w5/w5_20261115/`。

59 个可以换样本的发送机会里，相关规则每一次都选了最新样本。30 秒和 60 秒的风矢量 RMSE 配对差都是 0。对照场无报文是 6.000 m/s，有报文后约为 0.786 m/s 和 0.792 m/s；这只说明全局均值能被观测，不能说明选样策略有效。局部风场四个实现同样没有差距。不支持继续用这套选样替代发送最新样本。默认调度不变。

## W4.2 已完成的结果

2026-11-15 **W4.2 已完成**：分开共享链路和后续重选对 3 号浮标到岸时刻的影响。报告见 [W4.2 结果](docs/research/Wind_Observation_W42_Report_CN.md)，结果在 `results/wind_observation_w42/w42_20261115/`。

首包晚 82.613 秒，其中共享链路 22.613 秒，两次返航 60.000 秒，残差为 0。两机都按回传脚本继续时，末端电量降到 −143.945 J，回传不再准入。按这个新选择闭环多耗 212.468 J。不推荐扩大验证。

## W4.1 已完成的结果

2026-11-15 **W4.1 已完成**：归因逆风及时组在 240 秒选择回传的原因。报告见 [W4.1 结果](docs/research/Wind_Observation_W41_Report_CN.md)，结果在 `results/wind_observation_w41/w41_20261115/`。

没有报文时回传报价的末端电量是 3911.725 J，低于 4000 J，不能进入排序。报文把它抬到 5227.488 J，70000 bit 的分数因此被选中。5202.704 J 可以逐段对上；在同一报文信息下把首动作改回返航并不能省下这笔电。默认调度不变。

## W4 已完成的结果

2026-11-15 **W4 已完成**：从起飞起用真风重建 `b3_loose`，并检查提前到达的风观测能否改变交付和备用。报告见 [W4 结果](docs/research/Wind_Observation_W4_Report_CN.md)，结果在 `results/wind_observation_w4/w4_20261115/`。

原调度已经交付 140000 bit，两机末端电量 16221.762 J 和 13178.946 J。报文赶不上 30 秒的团队承诺，60 秒才进入 1 号机的决策，动作却要到 240 秒才分开。及时和迟到都把返航改成回传，按时 bit 和备用没有变化，全队多耗 5202.704 J。不推荐扩大验证。

## W3 已完成的结果

2026-11-15 **W3 已完成**：对账逆风及时组 270–300 秒的返航耗电，并用控制器积分旁路估计任务末电量。报告见 [W3 结果](docs/research/Wind_Observation_W3_Report_CN.md)，结果在 `results/wind_observation_w3/w3_20261115/`。

658.483 J 里，飞行段多出 671.809 J，降落报价高出 24.343 J，漏计的待机和无线电是 11.017 J。真风代入旧公式只增加 55.739 J。及时报文在手时，0.95 分位从 180 秒起就低于 4000 J。180 秒之后已检验的动作不能同时保住按时缓存和备用。默认策略不变。

## W2.1 已完成的结果

2026-11-15 **W2.1 已完成**：核对逆风及时组 300 秒的落地回传和任务末备用。报告见 [W2.1 结果](docs/research/Wind_Observation_W21_Report_CN.md)，结果在 `results/wind_observation_w21/w21_20261115/`。

1 号机当时已经完成降落，电量 4909.496 J。旧准入的 31.137 J 只覆盖缓存送完之前。1 W 地面待机 1500 秒后的任务末上界是 3409.496 J。显式报价不再把已完成的降落重算一遍，也没有因此放行回传。六臂相对 W2 的差是 0。默认守卫不变。

## W2 已完成的结果

2026-11-15 **W2 已完成**：同一条风观测在 180 秒之后每 30 秒重选。报告见 [W2 结果](docs/research/Wind_Observation_W2_Report_CN.md)，结果在 `results/wind_observation_w2/w2_20261115/`。

顺风三组的截止交付都是 140000 bit。及时组比迟到组早 30 秒把两包送到岸上，比不发送早 60 秒；净效用只高出约 0.0003，等于少用约 120 J。逆风三组仍是 90000 bit，与 W1 相同。1 号机 300 秒已经落地，返航守卫把回传改写成返航，结束电量 3408.658 J，低于 4000 J。该结果只属于这条显式实验，默认调度不变。

## W1 已完成的结果

2026-11-14 **W1 已完成**：在 `b3_loose`、被动 OFF、180 秒这个已记录决策上，比较不发送、及时到达和迟到到达的同一条风观测。报告见 [W1 结果](docs/research/Wind_Observation_W1_Report_CN.md)，结果在 `results/wind_observation_w1/w1_20261114/`。

零风回放与 M1.6 的动作、140000 bit、115684.957 J 和两机返航一致。两条及时报文都把决策时的风速均方根误差从 4.243 m/s 降到 0.3 m/s 左右。逆风三组都直接返航，任务结果只差 0.010 J 的报文开销。顺风的及时报文把 1 号机从直接返航改成先回传缓存：按时交付多 50000 bit，净效用多 4.985，两机返航和 4000 J 备用都保持；同一条报文迟到则没有这个变化。`service:0` 的评分更高，但逆风报价加备用电量放不进当前电池。该结果只属于这条显式实验。

## M1.6 已完成的结果

2026-11-14 **M1.6 已完成**：期初不可执行的服务标签不再占用浮标。报告见 [M1.6 结果](docs/research/Active_Model_Identification_M16_Report_CN.md)，结果在 `results/active_model_identification_m16/m16_20261114/`。指令见 [M1.6 指令](docs/research/Active_Model_Identification_M16_Instructions_CN.md)。

`initial_guard_consistent` 与 M1.5 返航尾只在显式打开时生效，默认选择和默认终端仍是 legacy。被动 OFF 在 12 个场景对 M1.3 OFF 和 M1.5 OFF 都没有少交付或降低净效用，`b3_loose` 收回 20000 bit。该候选保持显式身份，不改默认模式。风浪感知与能耗预测另立任务，本轮没有开始。

清理记录见 [Repository_Cleanup_20261114_CN.md](docs/research/Repository_Cleanup_20261114_CN.md)。执行与交接遵循[分支协作约定](AGENTS.md)。

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



