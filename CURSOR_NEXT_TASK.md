# Cursor 当前执行任务

当前分支：`research/ocean-observation-isolated`。

2026-11-14 下一阶段为 **M1.3：三档资源下的闭环试跑**。

- [M1.3 完整执行指令](docs/research/Active_Model_Identification_M13_Instructions_CN.md)

完整成本和后验矩已核对通过。本轮直接推进实际对照：保留原 tight/loose，按事前后验成本公式登记 window，最多运行 `4基础场景×3预算×2方法=24` 条核心航次，window 的信息消融最多 8 条续跑。window 无效时仍运行原两档 16 条合法任务，不另搜预算。

先用物理区间缓存和噪声阈值分段积分改善临界 Q。计算预算、并列和后备评分规则在主试前冻结；数值未全部达标时可按指令进行有明确状态标注的近似策略 pilot，不能伪装 PASS，也不再因某个代表观测不放行远任务而让全部核心航次为 0。q95=0.95 与备用电量不变。

请直接按本指令完成实现、必要测试、试跑与提交，无需再次请求启动确认。历史 M1.2 的停止状态不阻止本轮已限定的执行。

新结果为 `results/active_model_identification_m13/<run_id>/`，新报告为 `docs/research/Active_Model_Identification_M13_Report_CN.md`。本次交接只发布指令，M1.3 尚未执行。

保留历史：[M1.2 报告](docs/research/Active_Model_Identification_M12_Report_CN.md)、[M1.1 报告](docs/research/Active_Model_Identification_M11_Report_CN.md)、[M1 报告](docs/research/Active_Model_Identification_M1_Report_CN.md) 及各自结果。M1.2 的 24 项必要测试、成本一致性和后验矩已通过，16 主试与消融均未运行，不能记为零收益。

代码和结果只推送当前研究分支，不训练、不扩风浪/规模、不覆盖历史产物。完成本轮后停止，其他分支合并由用户另行决定。
