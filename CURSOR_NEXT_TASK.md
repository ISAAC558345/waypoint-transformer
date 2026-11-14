# Cursor 当前执行任务

当前分支：`research/ocean-observation-isolated`。

2026-11-14 当前任务更新为 **主动功耗辨识 M1.2：完整任务成本统一与校准价值验证**。

- [M1.2 完整执行指令](docs/research/Active_Model_Identification_M12_Instructions_CN.md)

请直接按指令实现、验证和提交，无须再次请求启动确认。首先统一 service 准入、终端价值、场景检查与真实完整业务链的成本口径，复核原 `52,156.04 J` 候选电量；不要仅通过调低电池去迎合漏算回传的旧准入判据。

随后修正空中 wait 的正常返航检查、积分未达标却回退最后候选的状态问题，并在临界开发状态核对 Q。通过核查后冻结配置，运行同规划器校准开关的 16 条核心航次，信息消融最多 8 条续跑，独立计数。

本次交接只发布 M1.2 指令，尚未产生新实现或主试结果。新结果放在 `results/active_model_identification_m12/<run_id>/`，报告为 `docs/research/Active_Model_Identification_M12_Report_CN.md`。

历史记录保留：

- [M1.1 报告](docs/research/Active_Model_Identification_M11_Report_CN.md) 与 `results/active_model_identification_m11/m11_20261114/`；16 项必要测试通过，但主试/信息消融均未运行。上轮场景判定须按新指令的完整成本口径复核。
- [M1 报告](docs/research/Active_Model_Identification_M1_Report_CN.md) 与 `results/active_model_identification_m1/m1_20261113/`；原 32 条结果保留，不覆盖重写。

M1.2 已给出新的成本和场景复核安排，不因历史入口中“没有新预算规则前暂停”的说明重复请求确认。不得把未运行对照写成零收益，不为制造校准收益改几何、包量或噪声。

继续使用固定执行器，不训练、不扩风浪或节点规模。代码和结果只推送当前研究分支，完成本轮后停止；其他分支的合并由用户另行决定。
