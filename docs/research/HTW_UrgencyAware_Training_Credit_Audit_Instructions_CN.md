# Stage E13：冻结训练信号与团队信用分配审计指令

- 仓库：ISAAC558345/waypoint-transformer
- 正式基线：feature/maritime-wt-stage1a
- 来源分支：experiment/priority-tag-queue-lookahead
- 固定来源提交：fabe0406fe8127da1f2b9fb59a6da347adffe21a
- E12 实现提交：7067295c7d2d545eed22a3bab66bb48f3becd35b
- 本阶段分支：experiment/priority-tag-credit-audit
- 上阶段科学 Gate：QUEUE_LOOKAHEAD_BENCHMARK_FAIL，保持不变
- 本阶段 Gate：TAG_CREDIT_AUDIT_COMPLETE / TAG_CREDIT_AUDIT_BLOCKED
- 工作方式：直接实现独立审计、运行必要测试和有界冻结探针；常规技术选择无需再次审批。
- 本阶段不训练、不新增策略、不改变奖励、观测或优化器，不合并。

## 1. 为什么进入本阶段

E12 的 actor 前瞻观测在训练前可行性检查中提前暴露了已入队的 tagged 积压，但没有通过完整的受控开发评估。L4_Q 相对同 seed L4_C 的实际 tagged 服务均值分别为：
- seed 0：-11,545.370 bit；
- seed 1：-446,241.877 bit；
- seed 2：+102,219.122 bit。

三组 95% 区间都跨零。相对 L0 的 tagged 均值在两个种子为负，触发预注册 FAIL，0/3 通过全部正向标准。这里的 FAIL 是注册规则的结论，不能改写成已经统计证明显著退化。

新增安全/约束失败相对 L0 的配对差之和为零；L4_Q 相对 L0 的平均干扰差在三个种子都为正，也不满足原发布要求。前瞻观测没有得到支持，不等于证明所有额外观测都无效。

6 次训练共 1800 个 update 的日志记录：
- 正 tag bonus 步共 110323；
- float32 仍可分辨的步共 110320；
- L4_C 的前瞻输入非零步数为零；
- L4_Q 的前瞻输入非零步数与真实 tagged backlog 暴露步数相符。

这些事实不支持把“奖励没有进入训练”或“大面积 float32 消失”直接认定为主要解释，但也不能证明梯度信用已有效。

E11 缺少历史逐步 tag bonus；E12 新增了分项日志、三个固定更新点的奖励样本及训练前实现提交。本阶段应利用这些新证据审计学习链路，避免再做一轮没有机制依据的 1800-update 实验。

## 2. 冻结与隔离

- 只在 src/wt/benchmark_v2/phase6i_priority_tag_credit_audit/ 新增审计代码，新增独立测试。
- 冻结 E0–E12 报告、CSV、日志、源码和所有旧 checkpoint，不覆盖、不改写。
- 正式 Planner、默认观测、FIFO、SafetyProjector、动作空间、奖励系数和 MAPPO/PPO 参数均保持不变。
- 不创建 L5、不续训或重训 L0–L4，不调用任何 optimizer.step、训练 update 入口或参数更新 API；禁止通过“重放历史”重新跑 300 updates。
- 允许在独立的内存模型副本上运行前向、GAE 和 autograd.grad，计算静态梯度；不更新模型参数、持久化 buffers 或优化器状态。
- 前后检查参数、buffers、checkpoint、正式文件及历史结果哈希，并恢复探针使用的全局 RNG 状态。
- 新日志全部进入 results/phase6i_priority_tag_credit_audit/。自然历史数据、初始批次重建和终态诊断代理必须明确分开。
- 不读取保留集或测试集；开发集只用于重算已有 CSV，除原有 L0 / 321101 预检外不新增性能评估。

## 3. 证据完整性与真实历史日志审计

1. 独立核对 E12 的 320 行 CSV、唯一键、配对顺序和注册开发范围，重算均值、置信区间及科学 Gate，保持 FAIL 不变。
2. 检查 6 组 checkpoint、初始化哈希、implementation_commit 和 source manifest；同时记录执行所依赖的共享 MAPPO、GAE、Adam、worker、账本及 projector 源码摘要，不能只哈希 E13 自己的文件。
3. 校验每组恰好 300 个 update，检查缺行、重复行、非有限值和配置漂移。逐组重算暴露计数、正奖励计数和 float32 可分辨计数。
4. 按固定更新区间 1–100、101–200、201–300，以及全程，分别报告：
   - 全部 UAV-step、有效关联 UAV-step、有效关联且有 tagged backlog 的 UAV-step；
   - 队首普通但有 tagged 积压的步、实际输入非零步；
   - 实际 tagged 服务、年龄/标签 bonus、payload 与运动惩罚；
   - advantage mean/std、KL、PPO clip fraction、价值损失、explained variance、梯度范数、动作标准差、投影与安全频率。
5. 比值明确说明是“两个均值之比”还是“逐步比值统计”；不能把两者混用。逐步比值只用真实保存样本，零分母和覆盖范围另报。
6. 检查全部 6 组 training_samples：注册 update 为 1、150、300，仅 env_index=0，每组 144 条。重算奖励恒等式、上界与 float32 写入，不能把这个有界样本称为所有环境/所有训练步的逐步记录。
7. 同场景、同 update 和同一 rollout 中的 UAV 不独立；日志分布为描述性证据，不把数十万相关步数包装成数十万独立实验。
8. 共享初始 actor 在不同种子间相同可能来自相同 legacy warm start，不据此断言“种子没有生效”。分开检查 critic 初始化、动作采样和训练分配 RNG。

历史样本没有保存完整 obs、values、dones、old log-probs、minibatch 顺序和 Adam moments。因此不能精确还原 update 150/300 的真实梯度或参数更新。所有缺项写入 evidence_gaps，不用最终 checkpoint 或新采样补造历史数据。

## 4. 有界探针：最多 12 个 rollout 批次，零参数更新

### 4.1 六个初始批次重建

对 L4_C/L4_Q × seeds 0、1、2，严格按已提交的 E12 build_pair 和 train_group 在第一次 update 之前的流程重建一次采样批次。使用真实训练 model seed、原始训练分配方案、PPO.num_envs 和 PPO.rollout_length，不调用 agent.update。

逐项对齐保存的 update=1 / env=0 样本，检查奖励、服务量、前瞻输入及 float32 值。精确项、bit 容差和浮点容差在探针运行前写入协议并固定，不因结果调整。

即使 env=0 样本全部匹配，也只能说已保存部分得到核对；其余环境缺少历史逐步证据，不能称为完整历史批次已被证明复现。样本不匹配时停止历史归因，报告具体差异；仍可执行清楚标记为新诊断的终态探针，但不得以它替代初始匹配。

### 4.2 六个终态冻结批次

每个已保存 L4_C/L4_Q checkpoint 各采样一批，保持冻结权重、原 PPO 批次尺寸和随机动作采样方式：
- env_index 为偶数使用 TRAIN 场景 321000，奇数使用 321001；
- 环境 reset seed 为对应场景编号；
- 诊断进程 seed = 9100000 + model_seed；
- 诊断分配 RNG seed = 9200000 + model_seed*100 + env_index；
- 同一 seed 的 C/Q 使用同样的随机数计划和外生规则，记录正常的轨迹分叉；
- torch 线程 1、确定性算法开启；新诊断 seed 与历史评估进程 seed 2 分开记录。

不得根据收益更换场景或扩大预算。每批保存足够的 obs、pre-tanh、old log-probs、values、dones、last_values、关联、真实 ledger 分解、分项奖励与 masks，以便独立核算静态梯度。

终态数据是 diagnostic_proxy，不是最后一次真实训练 rollout。对可重复性做状态/RNG capture-restore 核对；不得通过挑选轨迹使信号变强。

## 5. GAE、归一化与静态策略梯度分解

保持同批状态、采样动作、old log-probs、critic values、terminal flags、bootstrap 和全部 PPO 参数不变，比较：
- base reward；
- base + age bonus；
- base + age bonus + tag bonus。

### 5.1 奖励到优势

- 先检查 stored_reward = base + age + tag，与 ledger 的实际服务分解一致。
- 固定 critic 时，用线性 GAE 扰动核对 A_full = A_base + delta_A_age + delta_A_tag；bonus 扰动的 value/bootstrap 扰动为零。
- 同时报告共享 full-advantage 归一化尺度下的分解，以及分别按原训练规则重新归一化得到的 full 与 no-tag 差异。
- 两种统计不能混称。分别归一化具有非线性，不能声称其分项天然可加。
- 保留原 float32 路径；若增加 float64 交叉核算，只作数值诊断，不声称历史训练使用 float64。
- 报告平均绝对差、分位数、正标签子集、零奖励子集，以及可分辨下限；不能只用含大量零值的全体中位数下结论。

### 5.2 策略、熵与 critic 梯度

使用原 tanh log-prob、mask、PPO surrogate 和系数，在相同批次上通过 autograd.grad 分别计算：
- full reward 的策略梯度；
- 移除 tag bonus、保留 age 后的策略梯度；
- 移除两个 bonus 后的策略梯度；
- 熵项对 actor 的梯度；
- 按 value_coef 加权的价值损失对 critic 的梯度。

将 g_policy_full - g_policy_no_tag 称为“固定 critic/批次上的 tag 反事实梯度差”，不能把它当成已证明的真实长期标签因果贡献。记录范数、零范数计数、与 base/full 梯度的夹角以及 actor 新增输入列上的梯度，不用任意小分母制造比值。

所有模型都是采样时的冻结模型，old/new log-prob 在起点应一致、ratio 约为 1。该探针不复现多 epoch 参数变化后的 PPO clipping；历史 clipping 频率从真实训练日志获取，不能人为移动参数制造结论。

## 6. 团队信用与联合梯度裁剪

### 6.1 团队优势的分配

核对源码与保存批次：是否将同一个团队标量 advantage 广播给各 UAV，policy_active_mask 是否仍为全 1。

按当前实际任务关联与服务的 ledger 事实，描述标签暴露/实际服务 UAV 与其他 UAV 的 policy-term 分布，并给出共享参数梯度分项求和核对。对未来 bonus 的 GAE，不能把“当前未直接服务”自动认定为没有未来影响。

共享团队优势是原 MAPPO 设定，本身不证明有错误。任何“无关 UAV 导致信用稀释”的结论都需要标明它是诊断假设，不能只凭广播或相关性确认因果。

### 6.2 联合裁剪和 Adam

实际实现把 actor 与 critic 参数合并，使用同一 max_grad_norm，优化器是 Adam。静态记录：
- actor 的策略/熵合成梯度范数；
- 加权 critic 梯度范数；
- 联合范数、critic 所占平方范数比例；
- 按原 clip_grad_norm_ 规则计算的缩放系数；
- 仅作反事实展示的 actor/critic 分别裁剪系数，不执行更新，也不把它接进训练。

必须区分“梯度被缩放”与“Adam 的参数更新被同样缩放”。Adam 的一阶/二阶状态会影响更新；E12 checkpoint 没有保存历史 optimizer moments，不能从裁剪系数推断历史更新同比衰减，也不能据大梯度直接推荐拆分裁剪。

不运行真实或影子 optimizer.step。若没有足够证据说明裁剪是主要瓶颈，结论写为未确认。

## 7. 测试、来源与判定

- 先提交实际审计实现与注册协议，再运行探针，记录实现提交、配置、共享依赖哈希和运行时信息。
- 新增测试覆盖：奖励与 GAE 分解、归一化语义、固定 critic 的梯度差、mask 广播、梯度求和、裁剪系数、零范数、float32、同一数据重复探针与参数不变。
- 通过守卫让误调用 optimizer.step 直接失败；前后核对模型 state_dict、buffers 和历史文件哈希。
- 新相关测试必须通过；执行完整测试，并与同一固定来源/相同环境的干净基线按完整测试 ID 和异常签名比较。不只看失败数量，不把新的错误归为历史问题。
- 既有完整测试失败保留为发布阻塞；本阶段不伪造历史产物、不删除测试、不宣称完整测试全绿。
- 源数据、必要相关测试、参数冻结或重算 Gate 失败时，TAG_CREDIT_AUDIT_BLOCKED。
- 初始历史匹配存在超差且无法定位时，不发布“历史梯度已复现”的结论；若它阻止必需的初始分析，整体标记 BLOCKED，并仍交付已完成的独立终态诊断。
- 必要审计完成且所有历史/新诊断来源明确，即使机制尚无法确定，也可标记 TAG_CREDIT_AUDIT_COMPLETE。该 Gate 不是性能验收或合并依据。

## 8. 交付、结论与停止规则

交付：
- docs/research/HTW_UrgencyAware_Training_Credit_Audit_Report_CN.md；
- 独立审计代码、协议、配置及测试；
- results/phase6i_priority_tag_credit_audit/ 下的：
  - summary.json、schema.json、runtime_manifest.json、source_manifest.json；
  - historical_log_summary.csv、historical_sample_checks.csv；
  - initial_batch_match.json、probe_manifest.json、可复核的有界批次数据；
  - advantage_rows.csv、gradient_rows.csv、team_credit_rows.csv、clipping_rows.csv；
  - 测试日志、基线对照、checkpoint/冻结文件哈希和 evidence_gaps.json。

报告开头明确 E12 FAIL 保留、零训练、零合并，分别列出：
1. 历史日志证实了什么，缺失了什么；
2. 前瞻暴露是否贯穿真实训练；
3. 标签增量在奖励、优势和策略梯度各阶段的规模；
4. 团队广播与联合裁剪有哪些证据、哪些仍是推测；
5. 是否有证据支持下一轮只改变一个环节，以及对应必要对照。

已有 phase3d/phase3h 和 temporal credit 审计研究的是不同候选/轨迹协议，不把其 oracle 增益当作本轮 synthetic-tag MAPPO 的信用证据，也不覆盖那些目录。

若确有依据，最多推荐一个新的受控实验方向，说明证据与不确定性，不选择新奖励系数、不实施新 learner、不追加训练。本阶段完成后停止；保持正式系统与 L0–L4 冻结、E10/E12 的失败结论和原合并标准不变。
