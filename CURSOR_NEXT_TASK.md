# Cursor 当前执行任务

当前分支：`research/ocean-observation-isolated`。

2026-11-14 **M1.2 已执行并停止**。完整任务报价已与低层控制器对齐，原 `52,156.04 J` 没有替换。理想脚本窗口成立，但校准后的后验 0.95 分位放不下剩余航程；临界状态下 Q 也没有达到 `0.01`。16 条核心航次和信息消融都没有运行，未测差值不是 0。

- [M1.2 报告](docs/research/Active_Model_Identification_M12_Report_CN.md)
- 结果：`results/active_model_identification_m12/m12_20261114/`
- [M1.2 指令](docs/research/Active_Model_Identification_M12_Instructions_CN.md)

没有新的预注册指令前，不要下调电量、不要搜索新场景、不要补跑这 16 条航次，也不要把未运行对照写成校准收益为零。

历史记录保留：

- [M1.1 报告](docs/research/Active_Model_Identification_M11_Report_CN.md) 与 `results/active_model_identification_m11/m11_20261114/`；16 项必要测试通过，但主试/信息消融均未运行。上轮场景判定须按新指令的完整成本口径复核。
- [M1 报告](docs/research/Active_Model_Identification_M1_Report_CN.md) 与 `results/active_model_identification_m1/m1_20261113/`；原 32 条结果保留，不覆盖重写。

M1.2 的成本、后验窗口和数值判定已经记在上面的报告里。不得把未运行对照写成零收益，不为制造校准收益改几何、包量或噪声。

继续使用固定执行器，不训练、不扩风浪或节点规模。代码和结果只推送当前研究分支，完成本轮后停止；其他分支的合并由用户另行决定。
