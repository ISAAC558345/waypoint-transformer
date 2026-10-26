# Stage E3 执行指令：基于 E2 冻结轨迹的有界紧急奖励形式比较

- 仓库：ISAAC558345/waypoint-transformer
- 分支：feature/maritime-wt-stage1a
- 前置报告：docs/research/HTW_UrgencyAware_Reward_Scale_Replication_Report_CN.md
- 前置 Gate：URGENCY_REWARD_SCALE_INCONCLUSIVE
- 本阶段 Gate：URGENCY_REWARD_FORM_READY / URGENCY_REWARD_FORM_INCONCLUSIVE / URGENCY_REWARD_FORM_BLOCKED
- 性质：只使用 E2 已有 step/episode 轨迹做离线 reward-form 比较；不重跑环境、不训练、不修改正式系统，不选择最终 lambda_u。

## 1. E2 结论

E2 已在 8 个合法训练场景、48 步完整 episode 上完成冻结策略评估，计账和账本检查通过，正式 checkpoint 未变。Gate 为 URGENCY_REWARD_SCALE_INCONCLUSIVE。

关键结论：
- 最小正候选 lambda_u=0.0005 的 bonus/payload 均值约 0.000130（约 0.013%），但 bonus/penalty 的 P95 为 11.61，超出预注册上限 1。
- lambda_u=0.001 和 0.002 在冻结轨迹上分别造成 1、2 个场景的实质 payload-step 排序变化；0.0005 没有实质重排，但惩罚比例仍未过门。
- 冻结策略的 urgency/served 均值约 0.2979，与 E1 的 scripted smoke 0.1241 相差约 1.40，说明 E1 不能作为正式策略的尺度代理。
- 因此不直接选择 0.0005，也不训练 L1/L2。需要先比较“纯加法小权重”与“明确受 payload 项比例限制的有界 bonus”两种奖励形式。

## 2. 本阶段目标与限制

仅从已提交的：
- results/phase6i_priority_tag_reward_scale_e2/step_rows.csv
- results/phase6i_priority_tag_reward_scale_e2/episode_rows.csv
- results/phase6i_priority_tag_reward_scale_e2/summary.json
- results/phase6i_priority_tag_reward_scale_e2/schema.json

读取现有记录。先核对这些文件的 Git commit、SHA-256、schema 和记录数量。不得重新运行环境、重新抽样 synthetic tag、修改历史数据或读取 validation/reserved test。

本阶段不替负责人决定最终 reward form、lambda_u 或 cap。所有候选只用于敏感性分析。

## 3. 比较的奖励形式

保持原 base reward 不变：
r_base = payload_bits/1e6 - 0.05*c_project - 0.01*c_residual

定义 E2 中每步的 tag-aware urgency term：
U_t = urgency_served_bits_t/1e6

### Form A：未封顶的加法奖励
r_A = r_base + lambda_u * U_t
使用 E2 已算的 lambda_u ∈ {0, 0.0005, 0.001, 0.002}。0.05 只作高风险参照，不作为候选。

### Form B：相对 payload 的有界奖励
r_B = r_base + min(lambda_u * U_t, rho * payload_bits_t/1e6)
其中 rho 是每步紧急度 bonus 占 payload 项的最大比例。仅分析预注册候选 rho ∈ {0.001, 0.0025, 0.005}，即最多 0.1%、0.25%、0.5% 的本步 payload 项。lambda_u 仍取 {0.0005, 0.001, 0.002}，不挑选最终组合。

若 payload_bits_t=0，则本步 payload-relative cap 为 0，bonus 也为 0；必须在报告中说明这一设计可能使“没有成功服务的步骤”没有紧急度奖励，但不会给未成功交付的数据发奖励。

注意：Form B 只限制 bonus 相对 payload 的大小，不保证 bonus 小于投影/残差惩罚，也不保证训练后的 payload 不下降。必须报告这一限制，不能把 cap 描述成性能保证。

## 4. 必须计算的离线指标

对每个 lambda/rho 组合及每个场景：
- bonus 均值、P50/P95/P99、最大值；
- bonus/payload 的均值与 P95；
- bonus/(projection penalty + residual penalty) 的 P50/P95/P99，并单独列出惩罚接近零的步骤数量；不要用极小分母产生的比例替代绝对 bonus；
- 原 base reward、Form A reward、Form B reward 的逐步排序改变；
- payload 差超过 1 bit 的步骤中，奖励排序是否改变；
- episode 累计 payload、urgent-service bits、累计 bonus 和惩罚。相同轨迹上的累计 payload 必须保持完全一致，因为这里没有重新执行策略；
- 分场景结果、总体分布与最差场景。

对于奖励排序分析，要清楚区分“在固定轨迹上重算 reward 的排序”与“不同奖励训练后策略表现”。本阶段只做前者，不允许声称某种形式能提升闭环性能。

## 5. 决策规则

报告需回答：
1. Form A 是否有任何正权重在所有 8 个场景都满足 bonus/payload 均值 ≤1%、无超过 1 bit 的 payload-step 实质重排，以及 bonus/penalty P95 ≤1？
2. Form B 在每个 rho 下能否满足 payload-relative cap，实际 bonus 是否仍经常大于运动惩罚？
3. cap 是否过度压低了有用的 urgency signal，尤其是 payload 小或为零的步骤？
4. 如果当前标准仍导致没有候选可用，应明确报告“没有证据支持进入训练”，而不是事后放宽标准来选出一个权重。

不得从数据中挑一个最好看的组合并称其最优。最终 lambda_u 与 rho 都保持未选，除非报告只给出有条件的推荐并明确等待负责人批准。

## 6. 交付物

- docs/research/HTW_UrgencyAware_Reward_Form_Comparison_Report_CN.md
- results/phase6i_priority_tag_reward_form_e3/form_rows.csv
- results/phase6i_priority_tag_reward_form_e3/summary.json
- results/phase6i_priority_tag_reward_form_e3/schema.json

报告必须记录源文件 hashes、每种形式的公式、全部 lambda/rho 组合、分场景统计、奖励排序分析、字段缺口和 Gate。

## 7. Gate 与停止条件

URGENCY_REWARD_FORM_READY：
- 离线分析可复现；
- 明确比较 Form A/Form B 的奖励尺度和限制；
- 给出候选而非声称最优，并明确需要负责人批准的参数。

URGENCY_REWARD_FORM_INCONCLUSIVE：
- 所有形式都不能同时满足预注册的保守标准，或 cap 导致紧急度信号过弱。

URGENCY_REWARD_FORM_BLOCKED：
- E2 源文件缺失、hash/schema 不匹配、奖励字段无法复核或需要新 rollout 才能完成。

完成后提交报告和结果并停止。不要运行 rollout，不训练 L1/L2，不修改正式奖励/Planner/checkpoint，不进入 E4。
