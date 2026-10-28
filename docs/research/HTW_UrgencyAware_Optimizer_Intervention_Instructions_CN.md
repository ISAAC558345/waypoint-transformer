# Stage E14：相同总梯度上界下的裁剪分组配对干预试验

- 仓库：ISAAC558345/waypoint-transformer
- 正式基线：feature/maritime-wt-stage1a
- 来源分支：experiment/priority-tag-credit-audit
- 固定来源提交：2bce391e226b0b22657dd93b4201fb630bbaf4bb
- E13 实现提交：0787d8ddcbb900f659051d04162f4039b3ccfb91
- 本阶段分支：experiment/priority-tag-optimizer-intervention
- E13 Gate：TAG_CREDIT_AUDIT_COMPLETE；不是性能验收
- E10 / E12 FAIL 继续保持，不回改
- 本阶段只做小预算优化器干预试验，不做正式性能 benchmark，不合并
- 常规实现、测试和已登记试验直接推进，不另设 owner-approval gate

## 1. 研究问题与证据边界

E13 的 recommendation 为 no_single_component：没有把此前失败隔离到单一原因。标签奖励和前瞻观测进入过真实训练；6 个初始批次与已保存的 env=0 切片吻合，但不等于完整历史批次被证实复现。

在 12 个冻结批次上：
- 全奖励策略梯度范数约为 1.17–2.10；
- 分别重新归一化后的 tag 反事实梯度差约为 1.29e-4–3.69e-4；
- critic 占联合平方梯度范数约为 0.993–0.999999；
- 联合裁剪系数约为 3.0e-4–3.3e-2。

这些事实没有证明联合裁剪是主要瓶颈。Adam 的一阶/二阶矩会改变缩放的影响，静态裁剪系数不等于实际参数更新系数。

因此本阶段将“描述梯度尺度”推进为一个有界的因果干预：
> 在总梯度范数上界相同、奖励和模型输入相同的条件下，只改变 actor/critic 的裁剪分组，是否会持续改变实际 Adam 的 actor 更新？

这是新注册的试验假设，不是对 E13 结论的反转。即使观察到优化器更新差异，也不能据此认定此前 tagged 服务失败已被解释，更不能认定策略变好。

## 2. 范围与唯一干预因素

冻结正式 Planner、默认观测、SafetyProjector、FIFO、环境动力学、任务分配、动作空间、synthetic tag 协议、旧源码及 L0–L4 checkpoint。所有新增代码放在 src/wt/benchmark_v2/phase6i_priority_tag_optimizer_intervention/。

两组均采用 E12 L4_Q 的模型与输入：
- actor 接收原三列队首特征和两列真实队列前瞻特征；
- critic 仍是原三列队首信息对应的输入与结构；
- 相同 LEGACY warm start，不从已训练 L3/L4 热启动；
- 新输入权重置零，每个模型种子两组初始 actor、critic 和 optimizer 状态逐项相同；
- 一个 Adam 优化器覆盖各自组内的 actor 与 critic 参数，所有 Adam 超参数和参数顺序相同；不能另外改成两个优化器。

奖励固定为 E10/E12：
P = served_bits/1e6；A = pre-service age_service_bits/1e6；T = actual served_tagged_bits/1e6。
r = P - 0.05*c_project - 0.01*c_residual
    + min(0.0005*A, 0.00025*P)
    + min(0.0010*T, 0.00075*P)。

保持原 GAE、优势归一化、策略/价值/熵损失、团队广播及 masks，不增加损失，不提高奖励系数。

唯一干预是裁剪规则：
- PILOT_J：沿用原联合裁剪，对 actor ∪ critic 使用 max_grad_norm = G；
- PILOT_S：分别裁剪 actor 和 critic，每组上界均为 G/sqrt(2)。

G 从原注册 PPO 配置读取，当前应为 0.5；发现不符先核查并停止，不能默默换值。分组阈值由 G 推导，不进行阈值搜索。

actor/critic 参数集合必须不重叠。分组后联合范数上界仍不超过 G（允许预先登记的 1e-6 数值核对容差）。不能各给 0.5 后把联合上界增加到 sqrt(2)*0.5，再声称只比较同一总预算下的分组。

这个干预改变了 actor/critic 的梯度预算分配，不能将结果进一步归因为其中一个子项的独立作用。

## 3. 实现、来源与前置测试

1. 在独立模块中实现裁剪分组，不修改共享的正式/历史 MAPPO 文件。除裁剪位置及可验证的日志采集外，两组的 update 流程保持等价。
2. PILOT_J 在同一固定批次、相同初始化、相同 minibatch 顺序上，须与原 ExperimentMAPPO 的更新数值对齐；不能把代码重构差异当成处理效应。
3. 新增测试覆盖：参数集合不重叠、总范数上界、零梯度、非有限梯度拒绝、J 路径等价、两组初始状态相同、奖励恒等式、标签正服务路径、影子副本无共享存储、Adam 状态完整复制及源模型不变。
4. 使用已有 E13 批次或构造 fixture 做前置测试，不为挑选效果新增采样。
5. 先提交真实实现和协议，通过必要测试再开始登记试验。保存 instruction_commit、implementation_commit、配置、共享 MAPPO/GAE/Adam/worker/ledger/projector 源码摘要及初始化哈希。
6. 训练启动时执行源码必须与实现提交一致。依赖和运行时沿用已验证环境，记录版本，不静默降级旧 requirements 或切换设备。
7. 正式 checkpoint 哈希必须保持 92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e；所有旧模型和历史结果前后哈希一致。

## 4. 严格的小预算配对试验

- 两组各 model seeds 0、1、2；
- 每组每个种子恰好 32 个 PPO updates，共 6 次 / 192 PPO updates；
- PPO.num_envs、rollout_length、update_epochs、minibatch 数、训练池、场景/分配随机数协议和优化器其他设置沿用 E12；
- 一个 PPO update 内包含多个 Adam steps，日志必须区分二者，不能混计预算；
- 每组开跑前显式重设对应 model seed，同 seed J/S 使用共同初始化和随机数计划；轨迹因更新不同而正常分叉，不强行同步内部队列；
- 使用注册 TRAIN_POOL，不读取保留集、测试集，也不新增 32 场景 DEV 性能评估；
- torch 单线程、确定性算法开启，记录完整随机数角色；不能把历史评估进程 seed 2 当作所有训练种子。

这个 32-update 试验用于优化器机制识别，不能用于判断 300-update 策略优劣。不得根据结果追加 updates、换种子、换阈值或挑 checkpoint。

非有限损失/梯度、账本失衡、源码漂移或动力学/投影执行异常时停止并标记 BLOCKED。安全违规及约束失败沿用原计数完整记录，不因数值边界改变容差或删去事件，也不以重启种子掩盖问题。

本阶段用独立 pilot checkpoint，不命名为已验证 L5，不覆盖任何历史模型。

## 5. 直接记录实际 Adam 参数更新

每个真实 optimizer step，至少记录：
- variant、model_seed、PPO update、epoch、minibatch、optimizer step 编号；
- actor 的策略/熵合成梯度范数、加权 critic 梯度范数及联合范数；
- 裁剪规则、阈值、实际缩放系数与裁剪后分项/联合范数；
- Adam 学习率、betas、epsilon、step 计数及一阶/二阶矩范数摘要；
- actor/critic 的实际参数增量范数，以及新增输入列的参数增量；
- KL、PPO ratio 裁剪比例、价值损失、熵及解释方差。

同时记录每个 PPO update 的实际 payload、tagged/ordinary 服务、年龄/标签 bonus、运动惩罚、暴露、投影及原始安全/约束计数。训练曲线仅作描述，不作为公平 benchmark 或独立场景样本。

不能用 loss.backward 后的梯度范数替代 optimizer.step 前后实测参数差，也不能把梯度缩放系数直接称作 Adam 更新衰减比例。

## 6. 同源状态、同一梯度的影子干预

真实 J/S 训练一旦分叉，其梯度、轨迹和 Adam 状态都会不同。直接比较两个运行的参数增量，不能隔离裁剪规则本身。因此必须增加以下有界影子干预。

在每次运行的 PPO updates 1、16、32 的第一个 epoch / 第一个 minibatch，真实裁剪和 step 之前：
1. 保存同一源 actor/critic 参数、原始未裁剪梯度、完整 Adam state_dict（含一阶/二阶矩和参数映射）、同一 minibatch 以及 RNG 状态。
2. 创建三个完全独立副本：影子 J、影子 S、影子 J-repeat。三个副本使用完全相同的源参数、原始梯度与 Adam 状态。
3. 分别应用本阶段 J、S、J 规则，再各执行一个真实的影子 Adam step。
4. 核对 J 与 J-repeat，证明重复复制和更新可复现；固定状态不得被影子操作污染。
5. 比较 J/S 的 actor 参数增量大小、增量方向、每层差异、新输入列差异，以及在同一观测上的策略 KL/输出变化。零范数单独计数，不加任意小分母。
6. 影子更新后的参数与 optimizer 状态全部丢弃，绝不接回真实训练；恢复 RNG，证明影子操作不改变后续真实运行。

共 6 次运行 × 3 个登记点 = 18 个源状态；每个源状态 3 个单步副本，共最多 54 个影子 Adam steps，无额外环境 rollout。这些单步参数更新明确属于离线优化器干预，不称为“零参数更新”；它们也不是额外的 PPO 训练 updates。

update 1 的 Adam moments 为空，而 updates 16/32 的源状态含本轮真实形成的 moments。它们不是 E12 的历史 moments，不能回填成 E12 的更新证据。

## 7. 预先固定的机制判定

分别记录 trial_gate 和 effect_assessment，不设置性能 PASS。

trial_gate：
- OPTIMIZER_INTERVENTION_COMPLETE：六次登记运行及 18 组影子干预完成，必要测试、来源、复制/恢复和总梯度上界核对通过；
- OPTIMIZER_INTERVENTION_BLOCKED：必要核对失败或预算未能完成，列出具体原因与已消耗预算，不追加补跑。

机制指标：
R = ||delta_actor_S - delta_actor_J||_2 / ||delta_actor_J||_2，
只在 J 更新范数非零且 J-repeat 通过时定义。另报告绝对差和方向余弦。

实际差异筛选阈值预先固定为 R >= 0.01。1% 是本阶段选择后续研究方向的描述性阈值，不是统计显著性、性能改善或安全界限。

- MEASURABLE_OPTIMIZER_EFFECT：至少 2/3 model seeds，在 updates 16 和 32、来自两个真实运行的 4 个同源影子比较中，至少 3 个达到 R >= 0.01；
- NO_PRACTICAL_EFFECT_IN_PILOT：全部有效的 16/32 同源比较都低于该阈值，且三个种子覆盖完整；
- MIXED_OR_INSUFFICIENT：其他情况，包括零更新或有效覆盖不足。

不能把 18 个源状态或大量 minibatches 当作独立模型种子来获得显著性。对于少于阈值的结果，结论仅限当前预算和已观测状态，不证明整个训练生命周期都没有影响。

trial_gate 为 COMPLETE 仅表示机制试验完成。MEASURABLE_OPTIMIZER_EFFECT 也只说明裁剪分组在相同源状态上改变了 Adam 的实际更新，不能证明它是 E12 FAIL 的主因或已提高 tagged 服务。

## 8. 测试、发布与停止

- 在固定来源的干净副本与加上本阶段代码后的同一环境中执行完整测试，逐测试 ID/异常签名对照；新增相关测试必须通过，不新增失败。
- 已有历史文件缺失继续登记，不伪造锁文件、不删除测试、不以工作区残留代替干净基线。
- release_eligible 始终为 false，merged 始终为 false；不调整 E10/E12 FAIL、正式指标口径或原合并门槛。
- 任何组的安全/约束计数增加、KL 异常或 payload 下降都须列出，不能因影子更新差异达到阈值而隐藏风险。
- 若效果可分辨且无实现/安全疑点，可建议另行注册一个完整预算的性能对照；否则说明为什么不值得扩大训练。不得在本阶段自动延长到 300 updates 或开始 benchmark。
- 交付后停止，最多建议一个下一步方向。

## 9. 交付物

- docs/research/HTW_UrgencyAware_Optimizer_Intervention_Report_CN.md；
- 独立实现、注册配置及测试；
- results/phase6i_priority_tag_optimizer_intervention/ 下：
  - summary.json、schema.json、runtime_manifest.json、source_manifest.json；
  - training_update_rows.csv、optimizer_step_rows.csv；
  - shadow_update_rows.csv、repeat_checks.json、完整登记源状态与 Adam 状态清单；
  - 六个独立 pilot checkpoint、初始化/最终哈希和 checkpoint_manifest.json；
  - 原始安全/约束事件、训练数据覆盖及删失说明；
  - 相关/完整测试日志、干净基线对照、evidence_gaps.json 和 release_blockers.json。

报告开头给出 trial_gate、effect_assessment、真实 PPO updates/Adam steps/影子 steps 的实际数量，以及“尚无性能改善结论”。明确本轮填补的是实际 Adam 干预证据，不是再次发布 E13 的静态梯度比例。
