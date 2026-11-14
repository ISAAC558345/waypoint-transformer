# Cursor 当前执行任务

当前分支：`research/ocean-observation-isolated`。

请读取并执行 [M1.4 校准动作来源与价值分解](docs/research/Active_Model_Identification_M14_Instructions_CN.md)。这是 2026-11-14 基于最新推送 `57d3da36969e017d044025e90f90e4e43f0a38e8` 的下一步指令。

M1.3 已完成：24 条核心航次和 8 条信息消融已保存。三档预算的按时交付无差、允许校准的净效用均更低，保留这些负结果，不重跑或改电池。报告见 [M1.3 结果](docs/research/Active_Model_Identification_M13_Report_CN.md)。

本次复核发现 76 条校准请求中，9 条来自联合冲突修复，67 条为单机评分首选。M1.4 先用现有日志分类，再用固定六个已有快照分解运动/状态成本与信息溢价；必要时才做最多 24 条受控续跑。完整范围和计算上限以 M1.4 指令为准。

只做必要测试，不修改现有策略追求正收益，不训练，不扩展场景或风浪。代码、测试和结果完成后推送当前研究分支，报告保存为 `docs/research/Active_Model_Identification_M14_Report_CN.md`，结果放在 `results/active_model_identification_m14/<run_id>/`。

未经用户明确允许，不合并主分支、默认分支或 `feature/legacy-system-maintenance`。完成本轮后停止。
