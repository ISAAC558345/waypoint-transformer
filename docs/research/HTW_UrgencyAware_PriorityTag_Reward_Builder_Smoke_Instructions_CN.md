# Stage E5：有界紧急奖励 builder 的准入与 smoke 计划

仓库：ISAAC558345/waypoint-transformer
分支：feature/maritime-wt-stage1a
前置报告：docs/research/HTW_UrgencyAware_Reward_Protocol_Review_CN.md
前置 Gate：URGENCY_REWARD_PROTOCOL_WAITING_FOR_OWNER

## 目标
仅规划独立的 observation/reward builder 和短程 smoke。不得直接训练 MAPPO，不得修改正式 Planner、正式 reward、正式 observation、SafetyProjector、动作空间、checkpoint 或历史数据集。

## 准入条件
先核查是否有研究负责人明确批准以下两项的决策记录：
1. 首个有界奖励候选为 lambda_u=0.0005、rho=0.001；
2. 如果新策略在独立 benchmark 中降低总 payload，则停止并回滚，不替换正式低层。

如果没有明确批准，只提交一份 owner approval template，Gate 为 URGENCY_REWARD_BUILDER_WAITING_FOR_OWNER；不得实现 builder 或运行 smoke。一般性要求“权重不要太大”不等于批准该具体参数。

## 获批后才执行
在独立目录新增 builder，复用 E1 priority-tag ledger。紧急度使用 age/48 与 synthetic binary tag：
u=clip(clip(age/48,0,1)+0.25*tag,0,1)。
tag 必须标记为 synthetic，不得描述为真实海事业务标签。

低层候选奖励：
r = r_base + min(0.0005 * urgency_served_bits/1e6, 0.001 * served_bits/1e6)
r_base = served_bits/1e6 - 0.05*c_project - 0.01*c_residual

bonus 只按本步实际成功服务的 FIFO 桶计算；服务前读取 urgency；served_bits=0 时 bonus=0。不得改变 FIFO 或正式环境转移。旧 observation 和旧 checkpoint 默认行为必须保持不变。

## Smoke 范围
- 复用 E1 的 tag 可复现、FIFO、served/drop/snap、capture/restore 和 q_bits 守恒测试。
- 验证无未来信息泄漏、lambda=0 时 reward 与正式函数一致、有界 bonus 不超过 rho*payload_term、零服务无 bonus。
- 最多 2 个登记 TRAIN_POOL 场景，每场景 16 步；先运行一个场景估时。不得读取 DEV/validation/reserved/test。
- 检查日志开启/关闭时环境转移一致，checkpoint 维度和 metadata 明确。

## 交付物
获批且 smoke 执行后：
- docs/research/HTW_UrgencyAware_PriorityTag_Reward_Builder_Smoke_Report_CN.md
- 独立 builder 和测试
- results/phase6i_priority_tag_reward_builder_smoke/summary.json
- results/phase6i_priority_tag_reward_builder_smoke/schema.json

未获批时只提交：
- docs/research/HTW_UrgencyAware_E5_Owner_Approval_Template_CN.md

## Gate
URGENCY_REWARD_BUILDER_SMOKE_PASS：所有接口、守恒、未来信息和环境不变性测试通过。
URGENCY_REWARD_BUILDER_SMOKE_BLOCKED：任一账本、维度、reward 或环境一致性测试失败。
URGENCY_REWARD_BUILDER_WAITING_FOR_OWNER：没有明确批准。

完成后停止。无论结果如何，不自动训练，不修改正式系统。
