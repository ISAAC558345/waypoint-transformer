# Task Option 最小接口

高层输出：`targets`（每机 buoy 或 idle=-1）、`decision_time`、`remaining_hold`、`uav_mask`、`buoy_mask`、`association`（实际传入 `env.step`）。
执行器输入：当前可观测 L3-B 状态 + 当前 TaskOption。
执行器输出：`(M,2)` 单位方向 movement；环境按 `step_length_m` 限幅。
idle 编码：assignment `-1`；decoder token `12`。
高层负责 association；低层只负责 movement。adapter 若发现 association≠targets 则报错，不重做 matching。
Hold：T=48，每 15 步在 pending<=0 时重决策（0/15/30/45），最后一段被 episode 截断。无完成即换目标、无队列清空换目标、无新 termination/安全投影/奖励。
