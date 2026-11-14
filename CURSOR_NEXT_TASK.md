# 当前状态：M1.6 候选分配一致性验证已完成

`research/ocean-observation-isolated` 的 M1.6 已经做完。不要从本文件把 `initial_guard_consistent` 改成默认，也不要开始风浪感知或能耗预测。两项都等待新的明确指令。

- 报告：[M1.6 结果](docs/research/Active_Model_Identification_M16_Report_CN.md)
- 结果：`results/active_model_identification_m16/m16_20261114/`
- 判定：被动候选在本组登记场景得到性能支持。`b3_loose` 的 OFF 和 ON 都收回浮标 0 的 20000 bit。其余 11 个场景相对 M1.5 的按时 bit、净效用和能耗差为 0。
- 默认选择模式和默认终端模式仍是 legacy。本轮候选只在配置显式打开时使用。
- 风浪感知与能耗预测是下一步开发，本轮没有开始。
