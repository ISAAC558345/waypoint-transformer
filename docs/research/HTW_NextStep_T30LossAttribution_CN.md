# P5：t=30 反事实实验的收益分解与归因

## 前置结果
依据 `docs/research/HTW_NextStep_T30ActiveAssignmentPreserved_CN.md` 与 `results/phase6i_t30_active_preserve/summary.json`。
Gate：`T30_ACTIVE_ASSIGNMENT_PRESERVED_COMPLETE`。

16 个验证场景中，受约束距离重分配相对 Scheme C：
- t=30–45 payload 平均差：-409,993.669 bit；95% CI [-782,864.113, -99,672.570]。
- 完整 48 步 payload 平均差：-466,677.647 bit；95% CI [-908,245.693, -67,179.517]。
- 10 个场景改变分配，6 个不变；3 正、7 负、6 持平。
- 可达 UAV 数没有增加；目标距离仅小幅下降。
- t=45 两臂会基于各自状态重新调用 Scheme C，因此完整 episode 的差值不能全部归因于 t=30–45。
- 计账误差约 1.82e-8 bit，安全违规率为 0。保留测试未使用。

## 本阶段目标
不再试新的手工距离规则。利用 P4 现有数据，回答：
1. 负收益发生在 t=30–45 还是 t=45–48？
2. 负收益来自被重新分配的 UAV，还是未改变关联的 UAV？
3. t=45 两臂重新选择的 assignment 是否扩大或抵消了前 15 步的差异？
4. 现有证据能否支持下一步优先研究低层运动、通信/干扰，或高层排序？

## 数据与冻结边界
- 仅使用验证场景 322048–322063 和 P4 已生成的结果。
- 不读取 reserved/independent holdout 322064–322079。
- 不训练、不改正式 Planner、HGNN、MAPPO、奖励、动作或 SafetyProjector。
- 不覆盖 P1/P2/P3/P4 原始结果。
- 先检查是否有逐步 per-UAV served bits、SINR、干扰、位置和动作日志。缺失字段要明确报告，不得用窗口平均值伪造逐步数据。
- 若必须补充回放，只能使用相同验证池，在新目录 `results/phase6i_t30_loss_attribution/` 保存；两臂都要重跑，且核对状态哈希、模型和配置 SHA-256。

## 必须完成的分析
### A. 时间分解
对每个场景分别报告：
- t=30–45 的 payload 差；
- t=45–48 的尾窗 payload 差；
- 完整 episode payload 差。

验证：
`delta_episode = delta_window_30_45 + delta_tail_45_48`
并检查 payload 计账误差。

### B. UAV 级分解
对 10 个 changed 场景逐架 UAV 比较两臂：
- t=30 assignment、是否被重分配；
- t=30–45 served bits、有效服务步数、路径长度；
- t=45 位置、队列、目标和距离；
- t=45 新 assignment；
- t=45–48 served bits；
- SINR、干扰和 SafetyProjector 触发。

分别汇总被重分配 UAV 与固定关联 UAV 的交付差异。相关性不得直接表述为因果；如果缺少逐步链路日志，结论必须停留在窗口级。

### C. Scheme C 分数
仅比较 P4 已记录的原 assignment 和 preserve assignment：
- 分数差方向是否与 payload 差一致；
- 分数更高的候选与实际 payload 更高的候选是否一致。
不重训排序器，不扩展候选搜索，不根据未来 payload 选择 assignment。

### D. 数据一致性
检查分支前状态哈希、每步/窗口 payload 与 served bits 计账、完整 episode 差值分解。所有异常和失败场景都必须保留并报告。

## 交付物
- `docs/research/HTW_NextStep_T30LossAttribution_CN.md`
- `results/phase6i_t30_loss_attribution/scenario_decomposition.csv`
- `results/phase6i_t30_loss_attribution/uav_decomposition.csv`
- `results/phase6i_t30_loss_attribution/summary.json`
- 数据充分时生成差值图；数据不足时报告缺失字段。

summary 必须记录 split、场景 ID、P4 输入文件哈希、代码 commit、模型/配置 SHA-256、缺失字段、计账误差和 Gate。

## Gate
- `T30_LOSS_ATTRIBUTION_COMPLETE`：16 场景时间分解完成；10 个 changed 场景逐 UAV 分解完成或明确说明数据缺失；计账检查通过。
- `T30_LOSS_ATTRIBUTION_INCONCLUSIVE`：缺少关键逐步字段，不能区分窗口损失与尾窗损失。
- `T30_LOSS_ATTRIBUTION_BLOCKED`：split、状态哈希、计账或模型/配置一致性检查失败。

## 完成后的决策
- 若差值主要来自 t=30–45 的重分配 UAV，停止距离规则，提出一个可验证的低层执行假设。
- 若差值主要来自固定 UAV 的通信/干扰变化，先审计信道、干扰和速率记录，不直接改 MAPPO 奖励。
- 若差值主要在 t=45–48，研究两臂不同状态下的高层重新分配效应，但不能据此宣称 Scheme C 整体有错。
- 若机制无法区分，停止在这 16 个场景上继续堆叠规则，改为设计更广泛、预注册的跨任务低层评估。

本阶段只做归因，不训练、不修改正式系统。完成后汇报 Gate、证据、限制和唯一建议的下一步。
