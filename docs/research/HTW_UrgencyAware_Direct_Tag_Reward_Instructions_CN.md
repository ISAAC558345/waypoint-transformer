# Stage E10：直接 tagged-service 有界奖励实验

- 仓库：ISAAC558345/waypoint-transformer
- 正式基线分支：feature/maritime-wt-stage1a
- 来源实验分支：experiment/priority-tag-urgency-reward
- 归因来源分支：experiment/priority-tag-attribution-full-replay
- 当前实验分支：experiment/direct-tag-service-reward
- 来源归因提交：88c0e0296d6f64998a856af47f9c55dc6667c04b
- 前置 Gate：PRIORITY_TAG_ATTRIBUTION_COMPLETE
- 本阶段 Gate：DIRECT_TAG_REWARD_PASS / DIRECT_TAG_REWARD_INCONCLUSIVE / DIRECT_TAG_REWARD_BLOCKED
- 目标：直接优化实际 tagged bits 的交付量，同时保持总 payload 为主指标；不再用 age-weighted urgency 作为 L3 的额外 reward 项。
- 工作方式：无需单独审批；独立分支完成实现、训练和配对评估。只有达到预注册标准才开 PR/合并，结果差或不确定则不合并。

## 1. E9 证据与设计动机

E9 在已验证的历史运行时上下文下完成 224 条重放，所有记录复现、计账和 age/tag 分解检查通过。Gate 为 PRIORITY_TAG_ATTRIBUTION_COMPLETE。

主要发现：
- 当前 combined urgency 的提升主要由 age-only service 贡献。
- L2 三个 seed 的年龄加权服务差均为正，约 +0.845 到 +0.940 Mbit/场景平均值。
- L2 的 tag-increment service 差均为负，约 -0.057 到 -0.088 Mbit。
- L2 实际 tagged served bits 差均为负，约 -0.154 到 -0.346 Mbit；配对置信区间仍跨零。
- 标签增益完全被 clip 饱和的比例为 0%，所以主要问题不是 clipping，而是 reward 的优化目标与实际 tagged service 指标不一致。
- 标签是 synthetic，不代表真实海事应用类别。

因此本阶段只测试一个新假设：保留 L1 的 age/tag/validity observation，但奖励直接依据实际成功服务的 tagged bits，而不是用 age-weighted urgency 代理 tagged service。

## 2. 新变体 L3

基线奖励保持不变：
r_base = served_bits/1e6 - 0.05*c_project - 0.01*c_residual

L3 奖励：
r_L3 = r_base + min(lambda_tag * served_tagged_bits/1e6, rho_tag * served_bits/1e6)

固定候选：
- lambda_tag = 0.002
- rho_tag = 0.001
- synthetic tag protocol 与 E1/E2/E3 相同：tag_rate=0.25、beta_tag=0.25、SeedSequence([606, scenario_id, admission_step, buoy])。
- 仅按当前 step 实际从 tag=1 FIFO 桶服务的 bits 计算 served_tagged_bits；不得使用生成量、接纳量、队列长度或未来服务量。
- served_bits=0 时 bonus=0。
- 不使用 age_service_bits 或 combined_urgency_served_bits 作为 L3 的 bonus 输入。年龄特征仍可保留在 observation 中，作为策略状态信息。
- 不改变 FIFO、队列更新、动作空间、SafetyProjector、其他 reward 项或任务分布。
- 该配置是单一预注册实验候选，不是已证明最优。不要扩展参数网格，不因结果不理想而事后调参。

由于实际 tagged bits 不超过 served bits，该奖励的 bonus 满足：
0 <= bonus <= 0.001 * served_bits/1e6
并且只奖励成功交付的合成 tagged 数据。报告必须写明它仍不能保证训练后总 payload 不下降。

## 3. 隔离实现与 smoke

- 复用 E1 priority ledger 和 E8/E9 已验证的运行时上下文。
- 在当前分支独立新增 L3 reward builder/config；不修改正式 reward 或正式 observation 默认行为。
- L0：原始冻结 checkpoint phase6g_seed0.pt。
- L1：已有的 observation-only 变体，作为输入特征对照。
- L2：已有的 age-weighted bounded reward 变体，作为旧奖励对照。
- L3：L1 observation + direct tagged-service bounded reward。
- 保持原有 observation schema 兼容规则；新实验 checkpoint metadata 记录 reward schema、tag protocol、lambda_tag/rho_tag、source commit 和 hash。
- 先运行单元测试，再最多 2 个注册 TRAIN_POOL 场景、每场景 16 步 smoke。
- 必须测试：tag ledger 守恒、服务按 FIFO 桶精确分解、served_tagged_bits <= served_bits、零服务 bonus=0、bonus cap、lambda_tag/rho_tag config 校验、同一状态动作可复现、日志/ledger 不改变环境状态转移。
- 如果 smoke 失败，只修复隔离实现；若无法通过，标记 BLOCKED，不训练。

## 4. 训练协议

只有 smoke 全通过才训练 L3：
- 3 个固定模型种子：0、1、2。
- 与 L1/L2 使用相同 MAPPO 训练入口、优化器、更新次数（300 updates）、rollout 长度、任务分布、tag seed 协议和训练预算。
- 每个种子独立训练，不从 L1/L2 权重热启动；初始化方式遵循原实验统一协议。
- L0 使用冻结原 checkpoint，不重训。
- 不读取 DEV_POOL/validation/reserved/test 训练数据，不用评估结果调参数。
- 非有限 loss/gradient、账本错误、安全失败或训练中断时停止并记录，不擅自延长训练预算。
- 每个 L3 checkpoint、config、日志和源文件记录 SHA-256。

## 5. 配对评估

完成 L3 三个种子后，按原有协议对 L0、L1、L2、L3 在相同 DEV_POOL 场景 321096–321127、相同 reset/assignment seeds、相同显式历史运行时上下文下进行 deterministic paired evaluation。运行前通过 registry/split guard；禁止访问 reserved/test。

每个 model seed/场景报告：
- total delivered payload bits（首要指标）；
- actual tagged served bits（L3 直接目标）；
- age-only weighted service 与 tag-increment weighted service（诊断指标，不作为 L3 奖励）；
- ordinary/tagged dropped bits、final queue、tagged queue；
- mean/max age at service by tag；
- SINR、interference、path length、projection rate、safety violations、constraint failures；
- episode return、训练稳定性、运行耗时。
按场景配对计算相对 L0 和 L1/L2 的差值、95% confidence intervals 和所有退化场景。

## 6. 预注册合并标准

仅当以下条件全部满足才允许创建 PR 并合并至 feature/maritime-wt-stage1a：
1. L3 的 actual tagged served bits 相对 L0 的配对均值为正，95% CI 下界高于 0；
2. 总 payload 不劣于 L0：配对平均差非负，且 95% CI 不显示有实质负退化；同时公开所有逐场景退化；
3. 不增加安全违规或约束失败，不造成 SINR/interference、丢弃量或队列的实质恶化；
4. tagged-service 改善至少在 3 个训练 seed 中的 2 个方向一致，且不是单一场景驱动；
5. 相对 L1，L3 的 tagged service 有改善，能证明直接奖励项带来额外价值，而不只是新增 observation 的效果；
6. 所有测试通过，正式基线 checkpoint/code hashes 未变，PR diff 只含预期独立实验代码、测试和文档。

如果 L3 tagged service 改善但 payload 明显下降、安全恶化或统计不确定，则不合并。若结果不如 L1/L2，记录失败原因，不自动换权重重跑。训练 reward 上升不构成合并依据。

## 7. 交付物

- docs/research/HTW_UrgencyAware_Direct_Tag_Reward_Report_CN.md
- 独立 L3 reward builder、配置和测试
- results/phase6i_direct_tag_service_reward/summary.json
- results/phase6i_direct_tag_service_reward/scenario_rows.csv
- results/phase6i_direct_tag_service_reward/seed_summary.csv
- results/phase6i_direct_tag_service_reward/checkpoint_manifest.json
- 训练日志与环境 manifest
- 达到全部合并标准时创建 PR；否则只保留实验分支和报告。

## 8. 停止与合并

- 不改正式系统，不重写历史结果，不合并之前未达标的 priority-tag-urgency-reward 分支。
- 只有 L3 自身达到第 6 节全部标准才合并。
- 若结果 INCONCLUSIVE 或 BLOCKED，保持正式分支不变并报告最有证据支持的下一步。
