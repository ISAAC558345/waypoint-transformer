# 裁剪分组研究线结案

- direction_status: 用户选择 A 已执行完毕
- E15 scientific_gate: `CLIP_GROUP_BENCHMARK_INCONCLUSIVE`
- 本阶段 scientific_gate: `CLIP_GROUP_LINE_CLOSED_UNADOPTED`
- release_eligible: false
- merged: false
- PPO 更新: 1800
- Adam 步: 28800

这次结案只关闭裁剪分组的本轮采用尝试。payload 主目标加上小权重紧急服务的整条研究没有因此结束。E15 的 INCONCLUSIVE 保持原样，没有被改写成 FAIL，也没有被改写成通过。

三个种子的 tagged 配对均值都为正，六个对应的 95% 区间都跨过零。这不能证明稳定改善，也不能声称已经证明无效或显著变差。相对 FULL_J 的 payload 均值有两个种子为负，三个 payload 区间下界都低于预登记的 -0.5% 容忍线，相对 L0 的平均干扰差三个种子都为正。开发评估的安全和约束配对差之和为零。通过完整预登记标准的种子数是 0。

E14 仍然只说明裁剪分组改变了实际 Adam 更新。这个机制结论不能升级成“分组裁剪提高了任务性能”，也不能升级成“联合裁剪已经被证明是前期失败的原因”。L0 继续是当前正式采用的基线。这不表示 L0 是全局最优模型。

## 逐项对照

- seed 0 相对 FULL_J: tagged 均值 0.40447076059376397 Mbit，区间 [-0.19818168366362054, 1.0032846904632111]；payload 均值 0.3751243881885421 Mbit，容忍线下界 -0.28810899008937396 Mbit；原因：failed: tagged_ci_low_positive,payload_ci_within_floor。
- seed 0 相对 L0: tagged 均值 0.356633722911988 Mbit，区间 [-0.24741719062969167, 0.9564963990429587]；payload 均值 1.9334692984757271 Mbit，容忍线下界 -0.28031726553793807 Mbit；原因：failed: tagged_ci_low_positive,interference_mean_nonpositive。
- seed 1 相对 FULL_J: tagged 均值 0.03181670509008109 Mbit，区间 [-0.5746843648334865, 0.6356701383778249]；payload 均值 -0.46174847723803225 Mbit，容忍线下界 -0.28850294984710795 Mbit；原因：failed: tagged_ci_low_positive,payload_mean_nonnegative,payload_ci_within_floor,interference_mean_nonpositive,leave_one_tagged,leave_two_tagged。
- seed 1 相对 L0: tagged 均值 0.15331069450097162 Mbit，区间 [-0.5560690363933112, 0.8362789036844965]；payload 均值 1.1753883845959447 Mbit，容忍线下界 -0.28031726553793807 Mbit；原因：failed: tagged_ci_low_positive,payload_ci_within_floor,interference_mean_nonpositive,leave_two_tagged。
- seed 2 相对 FULL_J: tagged 均值 0.5315656037894 Mbit，区间 [-0.2852147342348255, 1.435222346138045]；payload 均值 -0.1618732578715831 Mbit，容忍线下界 -0.28524152563587873 Mbit；原因：failed: tagged_ci_low_positive,payload_mean_nonnegative,payload_ci_within_floor,sinr_mean_nonnegative。
- seed 2 相对 L0: tagged 均值 0.2667589913887702 Mbit，区间 [-0.609215283658819, 1.2243067458307015]；payload 均值 0.8229787617165456 Mbit，容忍线下界 -0.28031726553793807 Mbit；原因：failed: tagged_ci_low_positive,payload_ci_within_floor,interference_mean_nonpositive,leave_one_tagged,leave_two_tagged。

## 安全记录

训练事件共 41 条，其中安全违规 40 次，约束失败 1 次。
FULL_J 的训练安全违规是 29 次，FULL_S 是 11 次。这是训练池计数，不是开发场景上的安全改进证明。
40 次违规的执行后余量范围是 -3.503584961039863e-07 到 -1.0006615536184427e-09 米。
执行计数器用距离加 1e-9 米仍小于 1 米来记违规。投影器用全局最小预测距离加 1e-6 米仍不小于 1 米来接受间隔。同一个落在两者之间的距离会一边记违规、一边被投影接受。边界分类不减少原来的 40 次计数，也不放宽正式安全指标，更不表示运动风险已经排除。
predicted_distance_m 是投影器的全局最小预测距离。保存的事件没有全体 UAV 位置，不能把它改写成当前记录 UAV 对的预测距离。
唯一的约束失败位于 FULL_J、seed 1、update 119、env 2。投影候选的全局最小预测距离是 0.9999192009856496 米，状态是 failure，实际执行动作是全零。执行后的最近距是 8.75938528785752 米，且没有安全违规。被拒绝的投影没有进入轨迹，不能把这两个距离的差叫作预测误差。环境时间和场景编号在旧记录里没有保存。
E14 的 4 条事件仍然没有位置和距离，记为未知。没有用 E15 回填。

## 研究线

- E9：归因重放完成。至少两个 L2 种子的年龄服务配对均值为正，并且大于标签增量。合紧急度变化主要由年龄承担。这是目标错位证据，不是性能通过。
- E10：年龄和直接标签分开的奖励对照没有通过。它不支持把该奖励接入正式系统。
- E11：标签暴露、动作机会和数值边界审计完成。短时域差异只是诊断，不是新策略已经更好。
- E12：actor 前瞻观测的配对对照没有通过。
- E13：奖励、优势和梯度审计完成，没有定位到单一可改组件。
- E14：裁剪分组对同一源状态的实际 Adam 更新有机制效应。它不是性能结论。
- E15：完整预算对照保持 INCONCLUSIVE，当前不采用 FULL_J 或 FULL_S。

FULL_J、FULL_S 和 32 次试点都是研究参考。定点快照不含环境、账本和 option，不能无缝续训。正式加载路径和正式 checkpoint 没有替换。

## 尚未回答、且与本次结案不同的问题

1. 高层任务分配本身是否会改变实际 tagged 服务。这会改变高层职责，需要用户另行选择，并单独预登记对照和证据。它不是把同一条裁剪规则换一个名字再训练。
2. 是否先把安全计数器的 1e-9 米判断和投影器的 1e-6 米接受写成一个明确的测量定义。这只应在已经保存的轨迹上核对，不训练、不放宽正式指标，也要单独预登记。

本阶段没有估计新的收益，也没有承诺新的算力。

## 测试

- 干净基线测试数: 1795
- 加上本阶段后的测试数: 1799
- 基线失败签名数: 206
- 对照失败签名数: 206
- new_failure_count: 0
- 既有失败仍然是发布阻塞项。完整测试没有写成全通过。

## 停止

E15 保持 `CLIP_GROUP_BENCHMARK_INCONCLUSIVE`。本阶段 scientific_gate 是 `CLIP_GROUP_LINE_CLOSED_UNADOPTED`。
结案且不采用只在核对一致时成立。没有合并，也没有追加实验。正式系统继续使用现有 L0。

