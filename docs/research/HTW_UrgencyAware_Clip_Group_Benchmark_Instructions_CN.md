# Stage E15：裁剪分组的全预算性能对照执行指令

- 仓库：ISAAC558345/waypoint-transformer
- 正式基线：feature/maritime-wt-stage1a
- 来源分支：experiment/priority-tag-optimizer-intervention
- 固定来源提交：38ed6b91fc1df002f9eeb347c927f83a0791d00e
- E14 实现提交：9c971f7cbaf86da7187e0bd90ba93727bc1a990d
- 本阶段分支：experiment/priority-tag-clip-benchmark
- 研究方向选择：CHOSEN_A；用户于 2026-10-29（Asia/Shanghai）明确选择 A，完整对照已获授权
- 已选方向：A 做一次全预算公平对照；常规代码、种子、参数和测试无需再次确认
- E10/E12 FAIL、E13 审计结论、E14 机制结论全部保留
- 当前不改变正式系统、不合并实验历史

## 1. E14 结果：已经知道什么

E14 Gate 为 OPTIMIZER_INTERVENTION_COMPLETE，效应为 MEASURABLE_OPTIMIZER_EFFECT。共完成：
- 6 次运行，各 32 PPO updates，共 192 PPO updates；
- 3072 个真实 Adam steps；
- 18 个同源状态、54 个影子 Adam steps。

两组同奖励、同 actor/critic 输入、同一次 legacy 初始化。J 联合裁剪 G=0.5；S 对 actor/critic 分别使用 G/sqrt(2)，总梯度范数上界没有放宽。每组仍使用一个 Adam。

更新 16/32 的 12 个有效同源比较全部超过预登记的 R>=0.01 筛选线，3/3 种子满足效应规则。R 是 ||delta_actor_S-delta_actor_J||/||delta_actor_J||，范围约为 0.316–4.563。J-repeat 的最大绝对差为零。

这证明在已采样状态上改变裁剪分组会改变实际 Adam 更新，不只改变静态梯度尺度。R 不是 payload、tagged 服务或性能提升率，也不证明联合裁剪是 E12 FAIL 的主因。

## 2. E14 结果：仍然没有证明更好

训练池中每环境步服务均值如下。Mbit 为 bits/1e6，百分比为 S 相对 J 的变化：

| seed | J tagged（Mbit/步） | S tagged（Mbit/步） | tagged 变化 | 总 payload 变化 | 安全 J/S |
|---|---:|---:|---:|---:|---:|
| 0 | 0.230286 | 0.226269 | -1.74% | -0.94% | 1 / 1 |
| 1 | 0.229559 | 0.223973 | -2.43% | -0.84% | 1 / 1 |
| 2 | 0.227664 | 0.227288 | -0.16% | +0.18% | 0 / 0 |

这些数值按 training_update_rows.csv 重算，属于相关的训练 rollout 统计，不是独立开发场景的公平 benchmark，没有性能显著性结论。32 次仅相当于 300 次训练预算的约 10.7%；更长训练可能不同，也可能仍然失败。

- 原始安全事件共 4 条，J/S 各 2；约束失败为零。相同计数不等于相同事件，也不能证明对 L0 安全不劣。
- 现有 safety_events.json 主要记录组、种子、update 和 env index。没有完整距离证据时，不把这 4 条一概归为此前发现的数值边界事件。
- 相关测试通过；干净基线 1770 项、加入本阶段 1784 项，两边已有失败签名数均为 206，新增失败为零。不能写成完整测试全通过。
- 既有文件哈希核对通过，正式 checkpoint 未改变；release_eligible=false，merged=false。

## 3. 已记录的研究方向选择

用户本轮要求明确方向性选择，并已回复“A：做一次完整公平对照（推荐）”。本节保留原先两项选择的含义以便追溯，执行状态以 CHOSEN_A 和 direction_decision.json 为准。

### 已选 A：只做一次全预算公平对照

- 6 次全新训练，每次 300 PPO updates，共 1800 PPO updates；
- 按当前配置每 PPO update 有 16 个 Adam steps，预计真实训练 Adam steps 为 28800；实际数量从配置与日志核算；
- 训练量约为 E14 试点的 9.375 倍；这是计算量比例，不是运行时间承诺；
- 最终只用注册 DEV_POOL 做一次成对评估，回答是否改善实际 tagged 服务并守住 payload、安全与通信指标；
- 若 FAIL 或 INCONCLUSIVE，结束本轮裁剪改动的采用尝试，不继续换阈值或追加预算追逐 PASS。

选择 A 的理由：现有干预已经确认“确实改变优化过程”，完整公平评估能直接回答该改变是否有用。它不意味着预计会成功。

### 未选 B：暂停裁剪研究线

保留机制结论，继续使用现有正式 L0，不进行本方案的 1800 updates。整理这条研究线的得失；其他方向另行规划，不自动改奖励或转改高层。

### 激活规则（用户选择已满足）

- 已于 2026-10-29 收到用户的明确选择，结果记录在 results/phase6i_priority_tag_clip_benchmark/direction_decision.json。
- 依次完成前置检查、真实实现提交、六次 300-update 训练和 224 条配对评估；不要再请求相同方向选择或常规参数审批。
- 保持原预登记的奖励、预算、数据范围、门槛和停止规则，不因已获授权放宽标准。
- 只有用户随后明确改变或撤回研究方向，才更新该方向记录；不能把未选择的 B 当成当前状态。

## 4. 可立即完成的前置工作

1. 独立核对 E14 的 192 行训练 CSV、18 行影子 CSV、12 个正式效应比较、重复核对、安全事件及旧文件哈希，保存对账。
2. 核查 4 条训练安全事件已有证据。缺少位置/距离的项记为未知，不重训 E14 补造，也不减免原计数。为 E15 加入完整在线安全事件记录。
3. 核查试点 checkpoint 是否含完整环境、worker 与 RNG 状态。即使已有 Adam 状态，也不凭其恢复不完整的运行状态，把 32+268 拼成无缝 300。本方案统一从公共初始化全新训练，保持起点、预算和采样协议清楚。
4. 复用 E14 已验证的 J/S 裁剪与数值等价路径，只新增独立的全预算 runner、配置和评估汇总，不修改 E14 的 32-update 协议或历史结果。
5. 在 src/wt/benchmark_v2/phase6i_priority_tag_clip_benchmark/ 隔离实现；所有输出进入 results/phase6i_priority_tag_clip_benchmark/。
6. 准备两组逐 seed 完全相同的初始化与优化器、特征/奖励链路、J 等价、S 总范数上界、无未来信息泄漏、训练/eval seed 区分、评估配对和 Gate 测试。
7. 对同一固定来源、相同环境执行干净基线与新代码的完整测试，比较完整测试 ID 和异常签名。相关测试必须通过，不得新增失败。既有历史缺失继续登记，不伪造产物、不删除测试。
8. 先提交真实实现，再生成训练前 source manifest。记录实现提交、共享源码、完整配置、依赖与运行环境哈希。
9. 可做既有 L0 / 321101 的冻结上下文预检；不能在方向选择前训练或评估新候选。

## 5. 选择 A 后的完整训练：只改变裁剪分组

组名：
- FULL_J：actor 与 critic 联合裁剪，G=0.5；
- FULL_S：actor 与 critic 分组裁剪，各 G/sqrt(2)，总上界仍为 G。

两组都使用 E14 的五列 actor 输入与三列 critic 输入；真实队列前瞻在两组均开启。相同 hidden layers、动作空间、邻居编码器、一个 Adam、参数顺序、学习率、betas、epsilon、GAE、优势归一化和损失系数。

初始化与训练：
- 相同 LEGACY warm start，新输入列权重从零开始，不从已训练 L3/L4 或 pilot checkpoint 热启动；
- seeds 0、1、2，每个 seed 的 J/S actor、critic 与 Adam 初始状态逐项相同；
- 每组每 seed 300 PPO updates；不选最佳中间 checkpoint，只评估预定最终 checkpoint；
- 保持原 PPO.num_envs、rollout_length、update_epochs、num_minibatches、TRAIN_POOL、任务分布及标签协议；
- 开跑前显式重设 model seed；同 seed 两组使用相同的外生随机数约定，记录正常轨迹分叉；
- 单线程、确定性算法与已验证环境显式固定，不从父进程继承未记录的设置。

奖励完全沿用 E14：
P = actual served_bits/1e6，A = pre-service age_service_bits/1e6，T = actual served_tagged_bits/1e6；
r = P - 0.05*c_project - 0.01*c_residual
    + min(0.0005*A, 0.00025*P)
    + min(0.0010*T, 0.00075*P)。

不加大奖励、不给未传输数据服务奖励、不更改 FIFO、任务分配或 SafetyProjector。保持原始安全计数及两套既有容差，不能靠调整计量让候选通过。

所有 optimizer step 保留实际 actor/critic 参数增量、原始与裁剪后梯度范数、Adam 矩摘要和稳定性指标。定点保存完整模型、优化器、RNG 和采样状态的有界快照，给出恢复范围与哈希，不把缺少 worker 状态的 checkpoint 说成可无缝续跑。

新增安全事件在线记录同一 UAV 对执行前后位置与距离、预测距离、原始/投影动作、投影状态和数值余量。非有限量、账本失衡、执行异常或源码漂移时停止并报告 BLOCKED，不静默调参重跑。

## 6. 注册开发场景上的公平评估

只使用 DEV_POOL 321096–321127，不访问保留集或测试集：
- L0：1 × 32 = 32 条；
- FULL_J：3 × 32 = 96 条；
- FULL_S：3 × 32 = 96 条；
- 共 224 条。旧 L3/L4/pilot 只作历史描述，不代替本轮同起点全预算对照。

环境 reset seed 为场景编号；assignment RNG 为 7000000 + scenario_id；每回合显式应用已验证的历史进程上下文（线程 1、确定性算法、进程 seed 2），并保留其历史角色说明。模型种子、环境种子、分配种子、进程 seed 分开记录。

FULL_S 对同 seed FULL_J 是裁剪分组的主要对照；对 L0 是正式系统性能对照。所有 episode 使用确定性动作、相同配置、相同初始环境与外生规则。

CSV 报告：
- 总 payload（首要目标）、实际 tagged/ordinary 服务；
- 年龄加权服务、分项奖励、队列年龄与等待；
- 按标签丢弃与最终队列；
- SINR、干扰、轨迹与投影代价；
- 原始安全违规和约束失败；
- checkpoint/source/runtime/seed 元数据。

按场景配对，bootstrap 10000 次、seed 0、2.5/97.5 分位，逐模型种子给出相对 FULL_J 与 L0 的 95% CI。全量列出退化场景及留一/删除两条最大 tagged 增益后的敏感性。禁止将跨种子重复场景、UAV 步或训练 updates 当独立样本。

开发池已被多轮使用，本次是该固定开发协议的证据，不宣称未见场景泛化。

## 7. 预登记的性能判定与停止

状态字段分开：
- direction_status：PENDING / CHOSEN_A / DIRECTION_CLOSED；
- scientific_gate：NOT_STARTED / CLIP_GROUP_BENCHMARK_PASS / CLIP_GROUP_BENCHMARK_FAIL / CLIP_GROUP_BENCHMARK_INCONCLUSIVE / CLIP_GROUP_BENCHMARK_BLOCKED；
- release_eligible：默认 false，独立判断。

至少同两个模型种子同时满足下列全部要求，才可给科学 PASS：
1. FULL_S 实际 tagged 服务相对同 seed FULL_J 的配对 95% CI 下界 > 0；
2. 相对 L0 的实际 tagged 服务配对 95% CI 下界 > 0；
3. 对两个参考，总 payload 平均增量均 >= 0，95% CI 下界均不低于对应参考平均 payload 的 -0.5%；
4. 对两个参考，平均 SINR 不下降、平均干扰不增加；
5. 对两个参考，每个留一子集的 tagged 增益均值 > 0，去掉两条最大 tagged 增益后均值仍 > 0。

任何一个 FULL_S seed 的原始安全/约束计数相对同 seed FULL_J 或 L0 增加，科学 Gate 为 FAIL。数值边界诊断不构成豁免。否则，若至少两个种子对 L0 的 tagged 或 payload 均值为负，给 FAIL；必要测试/完整性/复现失败给 BLOCKED；其他没有满足 PASS 的情形给 INCONCLUSIVE。

payload 的 -0.5% 是区间不确定性容忍，不是允许主动降低主要目标；均值仍必须非负。通过该标准不等于证明总 payload 显著高于 L0，报告措辞必须准确。

即使科学 PASS，完整测试仍有失败、正式文件/旧模型哈希变化、源码不可追溯或差异尚含历史实验，release_eligible 仍为 false。不得把整个继承分支合并。只有完整验收、测试及最小差异要求全部满足时，才可能整理正式采用；本方案不提前授权忽略发布阻塞。

若 FAIL 或 INCONCLUSIVE，保留结果并停止该轮采用尝试；不改阈值、不换 seed、不增加预算、不挑场景，也不把试点或本轮失败回写为奖励系数搜索依据。

## 8. 交付物

- docs/research/HTW_UrgencyAware_Clip_Group_Benchmark_Report_CN.md；
- 独立实现、注册配置与测试；
- results/phase6i_priority_tag_clip_benchmark/ 下的：
  - direction_decision.json（已记录 CHOSEN_A、用户原话、日期和固定预算）；
  - preflight_summary.json、相关/完整测试日志、干净基线对照；
  - training_update_rows.csv、optimizer_step_rows.csv、安全事件明细；
  - 六个独立最终模型、checkpoint_manifest.json、初始化与冻结哈希；
  - scenario_rows.csv、summary.json、schema.json；
  - source_manifest.json、runtime_manifest.json、完整状态快照清单；
  - evidence_gaps.json、release_blockers.json。

报告开头明确：当前方向状态、实际投入、性能 Gate、发布资格，以及“分组裁剪是否提高了实际 tagged 服务并守住 payload”。用户已经选择 A；本轮应完成已授权的完整对照，交付后按原停止规则结束，不追加后续实验。
