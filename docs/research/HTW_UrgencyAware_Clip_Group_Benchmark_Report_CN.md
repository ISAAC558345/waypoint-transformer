# 裁剪分组全预算对照

- direction_status: `CHOSEN_A`
- PPO 更新: 1800
- Adam 步: 28800
- 评估回合: 224
- scientific_gate: `CLIP_GROUP_BENCHMARK_INCONCLUSIVE`
- release_eligible: false
- 性能问题: 现有开发场景证据不足以判定分组裁剪提高了实际 tagged 服务并守住 payload。

用户已选择 A。本次从共享 LEGACY 初始化全新训练 FULL_J 与 FULL_S，各 3 个种子、每组 300 次 PPO 更新。没有从 32 次试点 checkpoint 续跑。奖励、种子、总梯度上界和判定门槛都保持预登记值。

E12 仍是 `QUEUE_LOOKAHEAD_BENCHMARK_FAIL`。E13 仍是 `TAG_CREDIT_AUDIT_COMPLETE`。E14 仍是 `OPTIMIZER_INTERVENTION_COMPLETE` / `MEASURABLE_OPTIMIZER_EFFECT`。`merged` 为 false。

## 同一起点的开发场景对照

FULL_S 对同种子 FULL_J 是裁剪分组对照，对 L0 是正式系统对照。区间来自按场景配对的 bootstrap，10000 次、种子 0、分位 2.5 和 97.5。开发池已经被多轮使用，这不是未见场景泛化。

- seed 0: 相对 FULL_J 的 tagged 均值 404470.76059376396，CI 下界 -198181.68366362056；相对 L0 的 tagged 均值 356633.72291198804，CI 下界 -247417.19062969167；payload 均值相对 FULL_J 375124.3881885421，相对 L0 1933469.2984757272；安全差 FULL_J/L0 0.0/0.0；约束差 FULL_J/L0 0.0/0.0；本种子标准 不满足。
- seed 1: 相对 FULL_J 的 tagged 均值 31816.705090081086，CI 下界 -574684.3648334865；相对 L0 的 tagged 均值 153310.69450097161，CI 下界 -556069.0363933112；payload 均值相对 FULL_J -461748.47723803227，相对 L0 1175388.3845959448；安全差 FULL_J/L0 0.0/0.0；约束差 FULL_J/L0 0.0/0.0；本种子标准 不满足。
- seed 2: 相对 FULL_J 的 tagged 均值 531565.6037894，CI 下界 -285214.73423482553；相对 L0 的 tagged 均值 266758.99138877017，CI 下界 -609215.283658819；payload 均值相对 FULL_J -161873.2578715831，相对 L0 822978.7617165456；安全差 FULL_J/L0 0.0/0.0；约束差 FULL_J/L0 0.0/0.0；本种子标准 不满足。

通过种子数: 0
原因记录: registered full-budget comparison finished

## 训练安全事件

训练安全计数合计 40，约束失败合计 1。
E14 的 4 条事件没有位置和距离，记为未知，原计数保留，没有重训补造。E15 的新事件记录执行前后位置、距离、预测距离、原始动作、投影动作和数值余量。

## 测试

- 干净基线测试数: 1784
- 加上本阶段后的测试数: 1795
- 基线失败签名数: 206
- 对照失败签名数: 206
- new_failure_count: 0
- 相关测试通过: True
- 既有失败仍然是发布阻塞项。完整测试没有写成全通过。

## 停止

本轮交付后按原停止规则结束。FAIL 或 INCONCLUSIVE 不改阈值、不换种子、不追加预算，也不把结果回写成奖励系数搜索依据。

