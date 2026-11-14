# Cursor 当前执行任务

当前分支：`research/ocean-observation-isolated`。

执行 [M1.6 完整指令](docs/research/Active_Model_Identification_M16_Instructions_CN.md)，修正期初已被原准入器改成 RETURN 的 service 标签仍占用浮标的问题。

M1.5 在 11 个场景保持交付并节能，但 b3_loose 两种开关均少交付浮标 0 的 20000 bit。180 s 日志显示 UAV0 由于同目标冲突退让，UAV1 获分配的 service:0 却实际执行 RETURN。先恢复 ON/OFF 两份同一前置状态，复核期初准入，再把有效动作映射与去重接入正常评分入口，继续使用原冲突规则、M1.5 尾估值及积分精度。

本轮事前以保留普通业务遥测的 OFF 为主要候选，ON 为对照。先跑 b3_loose 两条，主要 OFF 保持两组旧 OFF 的交付量与净效用后补其余 22 条；旧 M1.3/M1.5 基线直接复用。逐场景检查性能，不追加信息消融或风浪 ISAC 扩展。

先自检测试必要性，完成后将代码、必要测试、原始结果和中文报告推送本分支并更新入口。本轮保持显式候选。未经用户明确允许不合并主分支、默认分支或维护分支，修改其他分支前先询问用户。
