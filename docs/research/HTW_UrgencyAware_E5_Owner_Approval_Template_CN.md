# E5 待确认：有界紧急奖励 builder 尚未获准

- 分支：`feature/maritime-wt-stage1a`
- 指令提交：`2a95691dfd59244a504ee6af3ff0a569d3b9c795`
- 前置报告：`docs/research/HTW_UrgencyAware_Reward_Protocol_Review_CN.md`
- 前置提交：`3efd71b8cc83aad099791a5f98ceba9cb53af70a`
- 前置 Gate：`URGENCY_REWARD_PROTOCOL_WAITING_FOR_OWNER`
- 本阶段 Gate：`URGENCY_REWARD_BUILDER_WAITING_FOR_OWNER`
- 本文件是待确认模板，不是批准记录。

## 1. 核查结果

已核查当前分支上的 E4 报告、`results/phase6i_priority_tag_reward_protocol_e4/summary.json`、`schema.json`，以及研究文档中对 `lambda_u=0.0005`、`rho=0.001` 的记录。没有找到研究负责人对下面两项的明确批准：

1. 把 `lambda_u=0.0005`、`rho=0.001` 定为首个有界奖励候选。
2. 如果新策略在独立 benchmark 中降低总 payload，则停止并回滚，不替换正式低层。

E4 的记录仍是：

- `candidate_status = CONDITIONAL_CANDIDATE_NOT_APPROVED`
- `approved = false`
- `training_authorized = false`
- `lambda_selected = null`
- `rho_selected = null`
- `accept_stop_or_rollback_if_benchmark_payload_drops = null`
- `approve_lambda_0_0005_rho_0_001_as_only_first_lowlevel_candidate = null`

E5 指令本身只要求先做这项核查。指令里写出参数，不等于批准这些参数。E4 里“优先考虑 Form B”是设计建议，也不是批准。一般性要求“权重不要太大”不等于批准 `0.0005` 和 `0.001`。

因此没有实现 observation/reward builder，没有运行 smoke，没有训练，也没有修改正式 Planner、奖励、observation、SafetyProjector、动作、checkpoint 或历史结果。

## 2. 仍待负责人填写

两项都必须单独写成明确的“批准”或“不批准”。只回复其中一项，或者只同意“权重不要太大”，都不能启动 builder。

| 事项 | 当前记录 | 负责人决定 |
|---|---|---|
| 首个有界候选为 `lambda_u=0.0005`、`rho=0.001` | 未批准 | 未填写 |
| 独立 benchmark 若降低总 payload，则停止并回滚，不替换正式低层 | 未填写 | 未填写 |

负责人若批准，需要另有一份明确的决策记录，至少写明：

- 批准人；
- 两项都是“批准”，不能留空；
- 批准范围只包括独立 builder 和最多 2 个登记 `TRAIN_POOL` 场景、每场景 16 步的 smoke；
- 不批准训练 MAPPO，不覆盖 L0，不修改正式系统；
- 该记录的提交必须晚于本模板，不能把本文件本身当作批准。

在那份记录进入仓库之前，Gate 保持 `URGENCY_REWARD_BUILDER_WAITING_FOR_OWNER`。

## 3. 停止

本阶段不进入 builder 实现，不运行 smoke，不自动训练。
