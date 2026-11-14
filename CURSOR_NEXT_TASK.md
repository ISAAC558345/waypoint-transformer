# Cursor 当前执行任务

当前分支：`research/ocean-observation-isolated`。

M1.5 已完成，结论是「本轮未满足性能保留标准」。报告见 [M1.5 结果](docs/research/Active_Model_Identification_M15_Report_CN.md)，原始结果在 `results/active_model_identification_m15/m15_20261114/`。

候选 `return_tail_when_no_business` 保持显式打开才生效。默认终端模式仍是 `legacy`。不要把本轮改成默认，不要为通过标准而改权重、电池、场景或积分精度，也不要追加信息消融。

唯一回退是 b3_loose：同开关两条比较都少 20000 bit。其余 11 个场景没有 bit 或 J 回退。物理完成计数和备用电量计数没有变差。这些负结果保持原样。

本轮到此停止。没有新的明确指令时，不开始下一阶段，不重跑 M1.3 或 M1.4。未经用户明确允许，不合并主分支、默认分支或维护分支；需修改其他分支时先询问用户。
