# Cursor 当前执行任务

当前分支：`research/ocean-observation-isolated`。

2026-11-14 用户已授权继续验证。当前任务是 **主动功耗辨识 M1.1：物理核算修正与同规划器配对验证**，完整指令：

- [M1.1 执行指令](docs/research/Active_Model_Identification_M11_Instructions_CN.md)

请直接按指令完成实现、测试、小试和提交，不必再次请求启动确认。顺序为：

1. 修正远端回传/等待的悬停功耗、合法降落、任务末返航检查和完整包交付时刻。
2. 核查窄后验与未来观测噪声的规划积分，复核场景后冻结配置。
3. 使用同一规划器，仅开关校准权限，在 8 个登记场景运行 16 条核心航次；信息消融最多 8 条续跑分支，分别计数。
4. 保存新结果、中文报告和复现入口，推送同一研究分支，完成后停止。

本次交接提交只发布指令；M1.1 尚未标记实现完成或验证通过。新结果应保存到 `results/active_model_identification_m11/<run_id>/`，报告为 `docs/research/Active_Model_Identification_M11_Report_CN.md`。

M1 已完成的历史记录继续保留：

- `results/active_model_identification_m1/m1_20261113/`
- `docs/research/Active_Model_Identification_M1_Report_CN.md`

不要重复或覆盖 M1 的 32 条核心任务。M1.1 是修正后的独立实验，不是重新生成旧结果。继续使用固定执行器，不启动神经网络训练、风浪扩展或自动合并。

代码、测试与结果只推送至当前研究分支；主分支、默认分支和原系统维护分支的合并由用户另行决定。
