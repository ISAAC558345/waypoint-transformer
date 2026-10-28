# Stage E12：队列前瞻观测的受控实验指令

- 仓库：ISAAC558345/waypoint-transformer
- 正式基线：feature/maritime-wt-stage1a
- 来源分支：experiment/priority-tag-signal-audit
- 固定来源提交：bd9ffe66a4ecff5562ccb9d3f1629d58891687a9
- 本阶段分支：experiment/priority-tag-queue-lookahead
- E11 Gate：DIRECT_TAG_SIGNAL_AUDIT_COMPLETE，仅表示冻结审计完成
- E10 Gate：DIRECT_TAG_REWARD_BENCHMARK_FAIL，继续保持，不回改
- 本阶段：先完成前置核验与无训练可行性检查，通过后直接执行两组受控训练及配对评估，无额外 owner-approval gate。
- 正式 Planner、默认观测、奖励、SafetyProjector、FIFO、标签协议及 L0–L3 checkpoint 均冻结。

## 1. E11 已有证据与下一步假设

E11 证明标签 bonus 进入了采样奖励；诊断中非零标签奖励在 float32 中仍能分辨。冻结轨迹上的归一化优势变化约为 1e-4 量级，但它不是历史训练优势，不能据此断言奖励太小或直接增加系数。

8 条诊断轨迹中，队首标签的可见率按全部 UAV-step 约为 4.2%–7.3%。这个分母包含没有有效任务关联的步，不能与约 25% 的生成标签率直接比较。后续必须同时报告全部 UAV-step、有效关联 UAV-step 和含有标签积压的有效关联 UAV-step 三种分母。

64 个可复现快照、每个 5 候选的四步筛选中，有 18 个候选行满足局部 tagged 服务增加、payload 和安全不退化的诊断阈值。这是局部机会证据，不能当成 18 个独立场景或长期性能改善。

两处新增安全事件的执行后距离分别为 0.9999999988795294 m 和 0.9999999989652907 m；计数器容差 1e-9 m，投影器接受容差 1e-6 m。预测距离与执行后距离在报告精度内相同。不能把“采用不同容差的判断不一致”直接解释成“预测与真实运动不一致”，也不能因此删除 E10 的违规计数。

下一步只检验一个新假设：
> 在奖励、任务分配、FIFO 和训练算法不变时，让 actor 在 tagged 数据成为队首前看到已入队的标签积压与 FIFO 阻挡信息，能否改善实际 tagged 服务，并守住 payload 和安全标准？

低可见率不是已证明的唯一失败原因。本实验可能失败，失败时不得顺势加大奖励或改高层。

## 2. 阶段 A：训练前必须完成的核验

### 2.1 测试基线

- 在固定来源提交的干净、可运行副本中，用相同环境执行仓库现有完整测试入口，保存基线日志。
- E11 原始完整日志为 Ran 1890 tests，FAILED (failures=1, errors=15)，实际是 16 个失败/错误测试；summary 的 17 条字符串包含了汇总尾行。新报告按真实测试 ID 统计，不能把尾行当成测试项，不覆盖 E11 原记录。
- 新增代码后再次执行相关测试及完整测试，按完整测试 ID、异常类型和缺失路径比较基线与新结果。未证明在基线上存在的失败，不能直接称为历史问题。
- 所有本阶段相关测试必须通过，不能新增失败。只有与本实验无关且在干净基线中复现的缺失历史产物问题，才可登记为既有问题继续独立研究；发布条件仍未满足。
- 不伪造冻结锁文件、历史报告或数据，不删除测试，不修改阈值使其变绿。

### 2.2 安全计量核验

- 保留原始安全计数、安全距离和两套现有容差。
- 用原上下文复现 E11 已登记的两个 E10 事件及对应 L0，对齐保存行。
- 针对事件的同一 UAV 对，分别报告上一时刻距离、执行后距离、预测距离、预测/执行差、距离相对 1 m 的有符号余量，以及两套容差下的判断。
- 不用上一时刻“全局最近 UAV 对”的距离冒充事件 UAV 对的上一时刻距离。
- 将数值边界分类作为补充诊断，不作为合并标准豁免，不修改正式 SafetyProjector。
- 正式 checkpoint 哈希必须仍为 92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e。

### 2.3 仅用已入队数据构造两列前瞻特征

保留 L3 的三列队首特征（age urgency、head synthetic tag、validity）。只为 actor 追加以下两列；critic 仍使用原 L3 输入与结构，两组完全相同。

对 UAV 当前已分配的浮标，在服务前从现有 priority-tag ledger 获取：
- Q：所有有效已入队桶的剩余 bits；
- T：其中 tag=1 桶的剩余 bits；
- B：按原 FIFO 顺序，第一只 tag=1 桶之前的剩余 bits 之和。

新增列：
1. assigned_tagged_backlog_fraction = T / Q；
2. assigned_fifo_bits_before_next_tag_fraction = B / Q。

当无有效分配、Q 为零或队列为空时，两列为零；无 tagged 桶时两列也为零。tagged 桶已在队首时，第一列大于零、第二列为零。使用原有 dust 规则，非零分母不另造任意 epsilon。

所有桶必须满足已经接纳且 admission_step 不晚于当前决策步；不能访问未来到达、未来服务、标签 RNG 的下一次抽样或未接纳数据。FIFO 仍按原 admission_step / insert_index 排序，不能按 tag 重排。

这是实验中假设 synthetic tag 在入队时已知的观测；不宣称真实业务具备该数据，也不给整个浮标贴单一标签。

### 2.4 前瞻观测可行性与接口测试

- 固定 TRAIN 场景 321000、321001，沿用 E11 的冻结 L0 与三个 L3 各一条完整诊断轨迹，总计 8 条。
- 只在旁路计算两列新特征，不把新列塞进不兼容的冻结旧 actor，不训练。
- 报告有效关联时新列为非零、head tag 仍为零但已有 tagged 积压，以及首次可见时间提前的数量与分布。首次时间比较只用实际已入队桶，保留未可见/未服务的删失计数。
- 必须至少观察到一个有效关联、队首为普通数据但存在 tagged 积压的自然训练步；否则将前瞻可行性记为 INCONCLUSIVE，停止，不扩大场景池挑样本。
- 测试空队列、无标签、头部标签、尾部标签、部分服务、丢弃、capture/restore、终止/reset、无未来信息泄漏、特征范围和 dtype。
- 所有新特征关闭时，在同一快照上保持旧输入列、旧 actor 行为和基础环境转移不变。相关验证不过则不训练。

## 3. 阶段 B：只改变 actor 观测的成对训练

两组使用相同的五列 actor 特征接口、相同网络宽度、相同初始化和相同奖励，组名为：
- L4_C：保留原三列，新增两列在训练和评估时始终置零；
- L4_Q：保留原三列，新增两列使用真实的服务前队列前瞻信息。

critic 在两组都保持原 L3 输入与结构，不加入前瞻信息。这样本轮差异限制在 actor 能否看到两列额外信息，不能声称已经检验了 critic 信息扩展。

网络仅为接纳新增 actor 输入列而扩展相应输入层，其他隐藏层、输出层、动作范围和 MAPPO/PPO 设置不变。新增输入权重显式置零；旧输入对应权重、偏置及其他参数从同一个初始化副本复制。每个 seed 两组训练开始前的 actor/critic 完整参数必须逐项相等、初始 optimizer 状态相同，并记录哈希。

沿用旧协议的 LEGACY warm start 及头部特征初始化方式，不从训练后的 L3 热启动；保留并记录该 warm-start checkpoint 的真实哈希。L0–L3 是冻结参考，绝不覆盖或续训。

奖励原样使用 E10 的 L3 公式：
- P = served_bits / 1e6；
- A = pre-service age_service_bits / 1e6；
- T = actual served_tagged_bits / 1e6；
- r_base = P - 0.05*c_project - 0.01*c_residual；
- bonus_age = min(0.0005*A, 0.00025*P)；
- bonus_tag = min(0.0010*T, 0.00075*P)；
- r = r_base + bonus_age + bonus_tag。

系数、0.1% payload 总 bonus 上界及零服务零 bonus 规则不变；不增加辅助损失，不做 reward/advantage 重加权，不替换 GAE 归一化。

训练预算：
- 两组各模型种子 0、1、2，每个 300 updates，共 6 次训练 / 1800 updates；
- 这是新阶段成对对照的预声明总预算，不是给某组追加搜索预算；
- 相同 optimizer、rollout_length、环境数、训练池、任务分布、synthetic tag 协议与训练随机数约定；
- 每次训练重新显式设置对应 model seed。评估的历史进程 seed 2 不能冒充训练种子；
- 相同训练场景/外生事件随机数计划，记录因动作和轨迹导致的正常队列差异，不以强行同步内部状态改变动力学。

阶段 A 全部必要检查通过后直接训练，不增加审批环节。非有限损失/梯度、账本错误、特征泄漏或源码漂移出现时停止；不得改超参数后继续冒充同一注册实验。

## 4. 训练源码与数据追溯必须先于训练

E10 的训练时实现字节缺失是无法用当前源码补造的历史证据缺口。本轮必须解决将来的可追溯性，不要求恢复不存在的历史证据。

- 实现与协议测试完成后，先提交真实训练实现，训练再开始。
- 记录 instruction_commit、implementation_commit、相关源文件 SHA-256、完整配置、依赖版本、运行环境、随机数角色与初始化哈希。
- 核查训练开始时所有执行源码与 implementation_commit 一致；有未提交源码变化则不启动训练。
- checkpoint 与 runtime manifest 都保存上述信息，不能只保存“指令提交”作为运行代码来源。
- 每个 update 记录训练稳定性及标签暴露、实际 tagged 服务、年龄/标签 bonus、运动惩罚、GAE/优势统计；保存足以核对奖励进入采样器的有界逐步样本，给出采样规则和覆盖范围。
- 冻结核心源码哈希在训练前后核对；新增日志与 checkpoint 使用本轮独立输出目录。

## 5. 开发集配对评估与统计

保持注册开发池 321096–321127，不读取保留集或测试集。显式使用已验证的评估上下文，并分别记录 model seed、env reset seed、assignment RNG seed 与进程 seed 角色。

评估：
- L0：32 条；
- 冻结 L3 的三个模型种子：96 条，作为描述性历史参考；
- L4_C 与 L4_Q：每组 3 × 32 = 96 条；
- 共 320 条，新增两组训练模型对应 192 条。

L4_Q 对同 seed 的 L4_C 是观测改动的主要成对对照；L0 是正式性能对照；L3 只作历史参考，不将其不完整的训练来源用于严格训练归因。两个新组均报告相对 L0、L3 的结果。

报告实际 tagged/ordinary 服务、总 payload、年龄加权服务、等待与队列年龄、按标签丢弃及最终队列、SINR、干扰、投影、原始安全计数、约束失败和分项奖励。年龄加权服务上升不等于等待下降。

继续使用按场景配对的 bootstrap：10000 次、seed 0、2.5/97.5 分位。按模型种子分别给区间；不能把同场景跨种子的重复当独立样本。完整列出逐场景退化、留一敏感性和去掉两条最大 tagged 增益后的均值。

该开发池已用于前期研究，结论仅限这轮注册开发实验，不能声称未见数据泛化。

## 6. 预先锁定的科学门槛与发布门槛

scientific_gate：
- QUEUE_LOOKAHEAD_BENCHMARK_PASS；
- QUEUE_LOOKAHEAD_BENCHMARK_FAIL；
- QUEUE_LOOKAHEAD_BENCHMARK_INCONCLUSIVE；
- QUEUE_LOOKAHEAD_BLOCKED。

至少同两个 model seeds 同时满足以下各项，才可给科学 PASS：
1. L4_Q 相对同 seed L4_C 的实际 tagged 服务增量 95% CI 下界 > 0。
2. L4_Q 相对 L0 的实际 tagged 服务增量 95% CI 下界 > 0。
3. L4_Q 相对 L4_C 和 L0 的总 payload 平均增量均 >= 0，且各自 95% CI 下界分别 >= 对应参考平均 payload 的 -0.5%。
4. 对两个参考分别检查：每次去掉一个场景后 tagged 增益均值 > 0；去掉两个最大 tagged 增益后均值仍 > 0。
5. 同一配对集合上平均 SINR 不降、平均干扰不升，两个参考均检查。

任何一个 L4_Q 种子的原始安全违规或约束失败相对 L0 或同 seed L4_C 的配对总数增加，都给科学 FAIL。数值边界说明不能豁免该规则。其他情形下，若至少两个种子的对 L0 tagged 或 payload 平均增量为负，给 FAIL；必要完整性/相关测试/复现失败给 BLOCKED；证据未满足 PASS 且无明确 FAIL 条件时给 INCONCLUSIVE。

独立记录 release_eligible 与 release_blockers：
- 科学 PASS 仍不等于允许整条继承分支合并；
- 本轮完整测试存在任何未解决失败，release_eligible 为 false；
- 相对正式分支的变更尚未整理为本实验及其必要依赖的最小差异，release_eligible 为 false；
- 正式文件/旧 checkpoint 哈希变化、来源缺失或历史产物混入发布，也不得发布。

保持旧 L1/L2/L3 不合并的结论。不要为了得到 PASS 放宽阈值、删除场景、换种子、延长训练或更换奖励。本轮交付完整科学结果和发布阻塞项后停止，不自动开展下一组实验；正式采用仍须满足既有完整 benchmark、测试和最小差异要求。

## 7. 交付物

- docs/research/HTW_UrgencyAware_Queue_Lookahead_Experiment_Report_CN.md
- src/wt/benchmark_v2/phase6i_priority_tag_queue_lookahead/ 下的独立实现、注册协议与配置；
- 新增特征、成对初始化、奖励链路、FIFO、无泄漏、评估对齐和 Gate 测试；
- results/phase6i_priority_tag_queue_lookahead/ 下的：
  - stage_a_summary.json、feature_exposure_rows.csv、安全口径明细；
  - baseline_tests.log、related_tests.log、repository_tests.log、test_comparison.json；
  - scenario_rows.csv、summary.json、schema.json；
  - 分组训练日志、独立 checkpoint 与 checkpoint_manifest.json；
  - runtime_manifest.json、source_manifest.json、初始化与正式文件哈希清单；
  - evidence_gaps.json 与 release_blockers.json。

报告开头直接给出科学 Gate、发布资格、研究假设是否得到支持以及适用范围。即使未观察到改善，也完整保存这次受控对照，不把失败结果回写成奖励系数选择依据。
