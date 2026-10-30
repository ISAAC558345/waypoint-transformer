# Stage E19：固定小标签规则的前瞻配对诊断

- 仓库：ISAAC558345/waypoint-transformer
- 固定来源及 E18 结果提交：c6aac5b71f23a33653e21db1f2d558f212167be3
- E18 实现提交：f79ad10139eacbb298b4a306a2b917057b32ef6b
- E18 回放前选择清单提交：83fe1476f0dd109e49425db67dfa50ead5638445
- 本阶段分支：experiment/priority-tag-assignment-rule
- 沿用用户已选 A：冻结低层，研究任务分配能否改善标签服务
- 本阶段只诊断一条固定规则；零训练、零梯度、零参数更新
- 独立环境 step 上限：1728；旧阶段预算不能转入本轮
- 正式 Planner / Scheme C、L0、环境、FIFO、奖励、SafetyProjector 和历史输出继续冻结

## 1. 已收到的 E18 结论

E18 已完整完成，不能重发 E17/E18 执行交接：
- scientific_gate=ASSIGNMENT_COVERAGE_DIAG_COMPLETE；
- opportunity_evidence=REPLICATED_LOCAL_OPPORTUNITY；
- 24/24 状态有同掩码替代，2696 个合格唯一替代中固定抽样 96 个；
- 120 行候选评估，S0=11，S1=11，S2=5；
- S0 出现在 5 个场景，S2 出现在 321002、321006 两个场景；
- 所有 S0 均在 t=15；已知 tagged 服务增加约 1e6–2e6 bit，payload 增加 0–9e6 bit；
- 2544 steps，基准复现和重复回放通过；模型哈希保持；
- 相关测试通过，干净基线/本轮代码均有 206 个历史失败，新增失败 0；历史套件仍不是全绿。

E17 的旧前缀截断零机会和 E18 的同掩码局部机会可以同时成立。不要覆盖旧 Gate，不将 E18 的 hindsight best 作为执行策略，不声称正式 Scheme C 已经改进。

下一项待解决的问题是：只看决策前状态的一条固定小标签规则，能否事前选择出守住 payload 的标签服务差；与屏蔽标签的同一基础规则相比，标签项是否实际改变选择并产生收益。

## 2. 固定场景与比较单位

- TRAIN 场景固定为 321008–321023，共 16 个；不访问 DEV、reserved 或 test。
- 它们未用于 E17/E18 这两轮候选诊断，但可能被早期低层训练使用过，不得称为全项目未见的独立留出集。
- 各场景沿原自然 G1_RATE k=5 sampler 运行完整 48 步参考轨迹，只在决策前 t=15 保存一个完整 snapshot。
- 主比较窗口仅为 [15,30)，H=15；不检验 t=0、t=30、t=45 的新干预，也不检验一整条新规则闭环轨迹。
- 聚焦 t=15 的依据是 E18 的全部 S0 都出现在此处；报告中明示这项开发选择及其限制。
- 每场景一个配对比较单位，不能把 3 个策略或 15 个时间步当作独立样本。

不挑掉没有已知标签、没有同掩码替代或规则不改变选择的场景。这些场景保留在 16 场景的零差异/覆盖统计中。

## 3. 三个事前确定的分配

对同一个完整 snapshot，恢复后按旧 sampler 恰好抽样一次得到 C0，自然基准。保持该次抽样之后的 assignment RNG 状态一致。

从原 G1_RATE k=5 联合池按生成顺序去重，保留原索引；只保留 active_mask 与 C0 逐 UAV 完全相同的关联，包含 C0。这里 k=5 仍是每 UAV 的 top-k 目标数，不是五个联合分配。

三个分配为：
- N：C0，原自然基准；
- M：只按下面 P 最大的关联，标签屏蔽对照；
- T：按下面 P + 0.001 * K 最大的关联，小标签规则。

M 与 T 使用完全相同的候选集合、数值运算、稳定排序和 tie-break。分数严格相等时取原池最小索引；不加随机扰动，不使用观察到的服务结果打破平局。

选择不再使用 E18 的四层抽样；可以遍历同掩码联合池计算静态分数，但每状态最多只回放 N/M/T 三个分配。保持原生成器和 k，不引入其他候选族。

## 4. 规则公式：payload 代理为主，标签项相对有界

只读取该 snapshot 的当前几何、原 env.assignment、q_bits、原配置以及已接纳 ledger 桶。

复用 phase6e.generators.feature_matrices 的当前 rate（bit/s）与 eta（步）定义，不修改其历史源码或信道公式。设 h=15，dt=env.config.delta_t。对 UAV u 与目标 b：

c[u,b] = min(max(q_bits[b],0), max(0,h-eta[u,b]) * dt * max(rate[u,b],0))。

这是“当前速率保持、扣除几何 ETA”的有限窗口服务量代理，不是实际传输量、联合干扰预测或未来服务保证。不能把代理的数学边界当成真实 payload 的边界。

在当前浮标 b 的已接纳存活桶中，按 admission_step、insert_index 的 FIFO 顺序，仅用 c[u,b] 的前缀容量逐桶取量；其中 priority_tag=1 的取量和记为 k[u,b]。不读取未来到达、不读取未来 tag、不跳过普通桶，不以整浮标 tagged 比例替代 FIFO 前缀。

对合法关联 a：
- P(a) = sum_u c[u,a[u]]，空闲 UAV 贡献 0；
- K(a) = sum_u k[u,a[u]]，空闲 UAV 贡献 0；
- score_M(a)=P(a)；
- score_T(a)=P(a)+0.001*K(a)。

合法关联不重复目标，因此 0<=K<=P；标签加分不超过 P 的 0.1%。0.001 是本次固定的启发式选择权重，不是训练奖励、不代表已证明最优，也不是把旧低层奖励实验参数原样迁移为结论。

选择器只接收上述决策前特征和当前 FIFO 桶，不接收 scenario_id、reference_window、候选实际服务、E18 候选最佳索引或未来信息。scenario_id 仅供外层重现环境/tag 流，不能进入排序规则。

M 的排序路径屏蔽 tag，计量时仍使用真实 synthetic tags；不要把整个 M 环境/账本的 tag 改为零。若 T 与 M 相同，诚实记录零决策差异，不调大权重找信号。

实现必须验证：
- 选择器不调用 env.step，不读取 reference_window 或历史结果表；
- 不修改环境/账本/option，也不消耗任何 RNG；
- 每个选中关联合法、属于原池且同掩码；
- FIFO 普通前缀阻挡、部分 tagged 桶、空队列等边界处理正确；
- 标签全部为零时 T 与 M 必须相同；
- P 和 K 有限、非负且满足代理边界；ledger 与 q_bits 不一致时 BLOCKED；
- 固定相同输入重复调用产生相同选择；
- 动态使用 num_uavs、num_buoys，不硬编码六架或十二个目标。

## 5. 先注册实现，再固定输入，再回放

1. 从固定来源建立独立实现目录，提交公式、唯一权重、场景、窗口、选择代码和配置；不得先跑这 16 场景的候选对照再补登记。
2. 收集 16 条自然参考轨迹和 t=15 完整 capture。沿用 E18 codec 保存可恢复的环境、env RNG、ledger、option、pending、last_movement、assignment RNG 和进程 RNG，并保存参考窗口。
3. 为全部 16 状态计算 N/M/T，输出原池哈希、基准索引/掩码、合格池大小、每条关联的 P/K/两项分数、选中索引、T/M 是否不同及选择清单哈希。
4. 将完整 snapshot 索引与选择清单提交后，才允许开始任何新 M/T 回放。正式回放只读已提交清单，并重新计算验证一致；禁止看到一部分结果后调整规则或重选。
5. 每个分支恢复同一完整 snapshot，按旧 sampler 抽样一次使状态与基准一致，再安装 N/M/T。额外重复 N 一次，验证参考窗口、首次 N 和重复 N 一致。

若三条分配重合，可在相同快照内按关联哈希复用完全相同的回放，保留各策略逻辑行与零差异；不能把重复策略当成多个独立观测。是否复用及实际预算逐项记录。额外 N 重复必须实际执行。参考 reset 时核对已知 tagged 为零；这是既有初始化约定的检查，不额外运行 t=0 候选。

不从旧 E18 事后优胜候选中挑策略，不在新增场景中先搜索最优回放，再评价所提规则。

## 6. 独立预算和运行时

- 自然参考：16*48=768 steps；
- N/M/T hold 回放：最多 16*3*15=720 steps；
- 额外 N 重复：16*15=240 steps；
- 总计最多 1728 steps；无额外仿真 fixture、训练或 shadow 更新。
- 若实际回放少于上限，节省步数不转成更多场景/权重/候选/窗口。本阶段新增规则测试使用静态或 mock 输入；本轮研究入口的参考、候选、重复及调试/fixture 中，任何真实 env.step 均计入上限。完整历史回归套件的既有测试调用单独记录，不充当本轮研究样本；不能以测试名义额外采集候选证据。
- 不因某个场景结果不佳、没有标签或 T=M 而换 seed、补场景、改窗口或重新回放。

环境 reset seed=scenario_id；assignment RNG=7000000+scenario_id；模型加载后、每条独立参考 reset 前显式重置历史进程上下文 seed 2，单线程，确定性算法打开，模型确定性动作。分别记录这些种子的作用；seed 2 不是通用模型/评估 seed。选择器本身不用 RNG。

正式 L0 checkpoint：
src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt
SHA256=92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e。

完整恢复与共同外生流核对沿用 E18：bit 容差 1 bit、路径/动作核对沿既有 1e-9 标准、桶级 1e-6 bit；不提高容差掩盖失败。原始安全/约束事件及原回退行为照实保留。

## 7. 指标：先比较标签项，再检查自然基准

主要配对差是 T-M；T-N 是必要的自然基准检查；M-N 用于说明基础排序自身的影响。

对每个 hold，沿用 E18 稳定桶 ID，分别报告：
- t=15 时已经接纳的 tagged / ordinary 实际服务量；
- 此后才接纳的 tagged / ordinary 实际服务量（描述项，不混入已知 tagged 主指标）；
- 总 payload、丢弃、末队列、年龄等价服务；
- 原始安全计数、约束失败、最小距离、路径；
- 平均 SINR、平均干扰；
- P/K 预测与实际指标的差别、T-M 的原始 P 损失及加分占比。

对 T-M、T-N、M-N 分别计算原 S0/S1/S2：
- S0：已知 tagged 增量 >1 bit，payload 增量 >=-1 bit；
- S1：S0 且安全/约束计数都不增加；
- S2：S1 且均值 SINR 不下降、均值干扰不增加。

报告每个场景的差值和方向、符合条件场景数；均值/中位数及配对场景 bootstrap 95% 区间为描述证据。固定 10000 次、default_rng(9400019)，以场景整体重抽，禁止把策略行或时间步当独立样本。报告区间跨零或负值，不能仅凭均值为正宣布性能通过。

单窗口实验没有测量新策略的完整闭环累计效应，不能将 16 个局部 hold 的和称为新策略整集 episode 收益。

## 8. 预登记的结案判定

scientific_gate：
- FIXED_TAG_RULE_DIAG_COMPLETE：16/16 状态及必要完整性核对有效；
- FIXED_TAG_RULE_DIAG_BLOCKED：来源/恢复/选择冻结/预算/共同外生流等必要完整性失败。

在 COMPLETE 内给出 mechanism_evidence，按顺序：
1. NO_TAG_DECISION_EFFECT：所有 16 状态 T 与 M 都相同。只能说明这个固定权重和代理没有改变本样本选择，不是标签无效。
2. REPLICATED_GUARDED_TAG_SIGNAL：至少两个不同场景中，T 同时相对 M 与 N 达到 S2；全部 16 场景中，T 的原始安全/约束计数均不高于 M 与 N，且 T-M、T-N 的场景平均 payload 差均 >=-1 bit。
3. LOCAL_OR_UNGUARDED_TAG_EFFECT：T/M 至少一个状态不同，至少一个场景 T-M 达 S0，但未满足上一类；明确是哪项覆盖、安全、通信或总体 payload 检查未满足。
4. NO_GUARDED_TAG_SIGNAL：其余完整执行情况。保留负差值和未通过原因，不转成成功。

该分类是小规模机制证据，不是采用标准。即使出现 REPLICATED_GUARDED_TAG_SIGNAL，若区间跨零，必须注明总体收益仍不确定；本阶段始终 performance_pass=false、adopted=false、release_eligible=false、merged=false。

BLOCKED 时不得用不完整样本发表正向分类。若无效场景出现，不能悄悄剔除后按较小分母结案。

## 9. 交付与停止条件

新增：
- docs/research/HTW_UrgencyAware_Fixed_Tag_Rule_Diagnosis_Report_CN.md；
- src/wt/benchmark_v2/phase6i_priority_tag_assignment_rule/ 独立实现和必要测试；
- results/phase6i_priority_tag_assignment_rule/ 中的 protocol/schema、runtime/source/seed manifest、完整 snapshots 及索引、回放前 selection_manifest、静态分数清单、baseline_reconciliation、各策略原始行、配对差、桶级归因、安全事件、预算实际用量、统计输出、evidence_gaps、测试日志与基线对照。

相关测试通过后，按仓库规则运行完整套件，与相同来源干净基线逐项比较；既有失败仍是发布阻塞，不以既有 206 项失败为新增失败豁免标准。Git 内容与 Windows CRLF 差异分开核对，正式源代码和模型不可语义变化。

只新增本阶段文件，不修改 E17/E18 代码、结果或正式文件。报告明确：固定规则、屏蔽标签对照、16 个 TRAIN 场景、仅 t=15、最多 1728 steps、零训练、非正式 Scheme C、非完整新策略闭环。

本阶段完成后停止。不得搜索其他权重、扩大 k、接触 DEV/test、启动 HGNN/RL 训练、改正式加载入口或合并研究分支。最多建议一个有证据的下一阶段，并另行登记；不能自行启动。
