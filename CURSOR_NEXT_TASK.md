# Cursor 当前执行任务

当前分支：`research/ocean-observation-isolated`。

请读取并执行 [主动功耗辨识 M1 完整指令](docs/research/Active_Model_Identification_M1_Instructions_CN.md)。这是用户 2026-11-13 指定的当前任务。

本轮连续完成最小闭环实现、必要测试和已登记的小试，随后把代码、配置、测试、全部结果和中文报告正常提交并推送到本研究分支。核心任务为 8 个场景 × 4 个非学习方法，共 32 条；固定接收点诊断最多另加 8 条，总实际任务最多 40 条，规划副本另计。具体口径以完整指令为准。

现有功能和论文检查已通过，见 [测试记录](results/active_model_identification_checks/20261113/summary.json)。不要重复执行历史阶段或只重建论文来替代 M1。

只做新增机制及受影响模块的必要测试，不运行无关全库测试或安全专项，不启动神经网络训练。历史“隔离保存、暂不执行”的说明不阻止本次已明确指定的 M1；不要续开旧 round_36 或创建 round_37。

未经用户明确允许，不得合并主分支、默认分支或原系统维护分支。完成 M1 并推送后停止，汇报提交 SHA 和结果。

- 实现目录：`src/wt/active_model_identification/`
- 配置：`configs/experiments/active_model_identification_m1.json`
- 结果：`results/active_model_identification_m1/<run_id>/`
- 报告：`docs/research/Active_Model_Identification_M1_Report_CN.md`
