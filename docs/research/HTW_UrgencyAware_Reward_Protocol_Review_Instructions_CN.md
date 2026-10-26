# Stage E4 执行指令：紧急奖励准入标准与训练前协议审查

- 仓库：ISAAC558345/waypoint-transformer
- 分支：feature/maritime-wt-stage1a
- 前置报告：docs/research/HTW_UrgencyAware_Reward_Form_Comparison_Report_CN.md
- 前置 Gate：URGENCY_REWARD_FORM_INCONCLUSIVE
- 本阶段 Gate：URGENCY_REWARD_PROTOCOL_READY / URGENCY_REWARD_PROTOCOL_WAITING_FOR_OWNER / URGENCY_REWARD_PROTOCOL_BLOCKED
- 性质：只审查 E2/E3 已有证据和奖励目标的数学含义；不运行 rollout、不训练、不修改正式系统、不更改既有预注册结果。

## 1. E3 的结论

E3 已复核 E2 的 384 条 step rows 和 8 条 episode rows，源文件哈希、schema 和计账检查通过。Gate 为 URGENCY_REWARD_FORM_INCONCLUSIVE：
- 所有正权重 Form A 以及全部 Form B 的 lambda/rho 组合都没有满足原先预注册的全部标准；
- 最小正权重 lambda=0.0005 的平均 bonus/payload 约为 0.013%，但 bonus/penalty 的 P95 为 11.61；
- Form B 能按 rho 限制 bonus 不超过本步 payload 项的 0.1%、0.25% 或 0.5%，但不保证 bonus 小于投影/残差惩罚；
- 固定轨迹重算不能说明训练后策略的 payload、轨迹或服务量会如何变化；
- lambda_u 和 rho 仍未选择，未训练任何 L1/L2 模型。

本阶段不能把 E3 的失败阈值事后放宽来挑选组合。应先审查这些阈值是否对应研究负责人实际目标，尤其要区分：
1. bonus 相对于主要任务收益（payload）的比例；
2. bonus 相对于动作正则项/约束代价（projection/residual penalty）的比例；
3. bonus 是否改变固定轨迹的奖励排序；
4. 训练后的闭环 payload 是否下降（只能由训练后独立 benchmark 回答）。

## 2. 本阶段目标

只基于已提交的 E0–E3 报告、summary、schema 和现有正式 reward 代码，输出一份训练前协议审查：
- 明确 E3 的原预注册门槛及其结论，不能改写为“通过”；
- 解释为什么 bonus/penalty P95 > 1 本身不等同于系统目标失败：运动惩罚是正则项，payload 是主要任务项；但若正则项原本用于控制动作变化，bonus 在大量步骤超过它仍可能改变策略，需要作为风险保留；
- 解释为什么 payload-relative cap 只限制单步奖励比例，不保证完整 episode payload 不下降；
- 比较两种可检验的后续设计：
  A. 纯 additive urgency bonus；
  B. payload-relative capped urgency bonus；
- 建议优先用 B 做小规模、独立的低层训练候选，因为它能将紧急度 bonus 限制在本步成功服务 payload 项的固定小比例；但必须明确这是设计建议，不是性能结论。
- 不从 E2/E3 固定轨迹中宣称某个 lambda/rho 最优，不自动选定参数。

## 3. 参数与可执行规则

依据负责人已明确的目标“payload 是主目标、紧急信息可获额外奖励、权重不要太大、标签和等待年龄共同决定紧急度”，后续候选设计应满足：
- urgency 使用 E1 的 synthetic_priority_bucket_v1，合成 tag 必须明确标记 synthetic；
- 高层 Scheme C 暂时冻结，继续使用 payload-first；不要把低层奖励 bonus 同时加到高层；
- 低层候选使用 r = r_base + min(lambda_u * U_t, rho * payload_term)；
- payload_term = served_bits/1e6，U_t = urgency_served_bits/1e6，均使用本步真实成功服务，不使用累计 payload；
- 当本步 payload_term=0 时 bonus=0；
- 初始候选范围仅限 E3 已测试的 lambda_u ∈ {0.0005, 0.001, 0.002} 与 rho ∈ {0.001, 0.0025, 0.005}，不得扩大网格；
- 不允许同时改变 observation、reward、动作、SafetyProjector 或任务分布；
- 不得宣称 capped reward 保证 payload 不下降。

E3 中 lambda=0.0005, rho=0.001 可作为首个有界候选进行后续受控试验的候选之一，因为它的 cap 为 payload 项的 0.1%，且在 E3 固定轨迹上未造成 payload 项相差超过 1 bit 的实质重排；但它未通过 bonus/penalty P95 门槛，因此必须标记为 CONDITIONAL_CANDIDATE_NOT_APPROVED，不能当作已选参数或直接启动训练。

## 4. 后续训练前必须明确的决定

生成一个短决策章节，列出需要负责人确认：
- 是否接受在训练后 benchmark 中若出现 payload 下降就停止/回滚的准则；
- 是否批准将 lambda=0.0005, rho=0.001 作为唯一首个受控低层训练候选（仅训练新模型，绝不覆盖 L0）；
- 如果未批准，维持不训练状态。

不需要负责人另行指定高层 delta_payload_max，因为本方案不允许高层牺牲实质 payload；lambda_H 仍保持未选。

## 5. 本阶段交付物

- docs/research/HTW_UrgencyAware_Reward_Protocol_Review_CN.md
- results/phase6i_priority_tag_reward_protocol_e4/summary.json
- results/phase6i_priority_tag_reward_protocol_e4/schema.json

报告必须引用 E2/E3 的精确 commit 和文件 SHA-256，列出原始 Gate、公式、候选参数、设计建议、未解决风险和负责人需要确认的事项。不得重新计算/覆盖 E2/E3 结果，不得读取 validation/reserved test。

## 6. 禁止事项

- 不重跑环境、不训练 L1/L2、不修改正式 reward/Planner/action/checkpoint；
- 不重新抽样 synthetic tags；
- 不事后放宽 E3 的门槛或把 E3 Gate 改为 READY；
- 不从 E3 的固定轨迹奖励排序推断训练后性能；
- 不开始 E5 或自动执行受控训练。

## 7. Gate 与停止条件

URGENCY_REWARD_PROTOCOL_READY：源数据和 Gate 复核通过；明确区分固定轨迹奖励尺度与训练后策略性能；给出有界奖励候选与风险；待负责人确认的训练准入事项写清楚。

URGENCY_REWARD_PROTOCOL_WAITING_FOR_OWNER：审查完成，但未获准使用唯一首个训练候选；这是正常停止状态。

URGENCY_REWARD_PROTOCOL_BLOCKED：E2/E3 源报告、哈希或 reward 字段无法核验。

完成后提交报告和 summary/schema 到当前分支并停止。只有负责人明确批准后，才可进入独立 observation/reward builder 的 smoke 阶段；未经批准不得训练。
