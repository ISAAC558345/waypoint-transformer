# Stage D3 执行指令：Payload-first + urgency tie-break 闭环可行性试验

- 仓库：ISAAC558345/waypoint-transformer
- 分支：feature/maritime-wt-stage1a
- 前置报告：docs/research/HTW_UrgencyAware_HighLevel_Utility_Rule_Report_CN.md
- 前置 Gate：HIGHLEVEL_UTILITY_RULE_READY
- 本阶段 Gate：URGENCY_TIEBREAK_CLOSEDLOOP_PILOT_COMPLETE / URGENCY_TIEBREAK_CLOSEDLOOP_PILOT_INCONCLUSIVE / URGENCY_TIEBREAK_CLOSEDLOOP_PILOT_BLOCKED
- 性质：小规模、oracle-label 驱动的策略可行性试验；不训练 HGNN、不替换正式 Scheme C、不改低层 MAPPO/奖励/checkpoint，不读取 validation/reserved test。

## 1. D2 结论与本阶段问题

D2 对 D0/D1 已有的 24 个训练状态、每状态 128 个候选进行了 Pareto 和 lambda 敏感性分析。Gate 为 HIGHLEVEL_UTILITY_RULE_READY，但：
- 21/24 状态的紧急服务最优候选已在 payload 的 1-bit 并列组内；
- 只有 3 个状态存在实质 payload 损失，且临界 lambda_H 分别约为 1.213、8.653、10.545；
- 没有确定全局 lambda_H，也没有确定 payload 损失预算 delta_payload_max；
- 当前结果只覆盖每状态抽样的 128/13,327 候选，不能代表全池前沿。

下一步不训练 weighted-sum HGNN，而是用小规模闭环 pilot 检验一个保守规则：

**先最大化候选的 15-step payload 标签；只在 payload 相差不超过 1 bit 的候选之间，选择 urgent-service 标签最高的候选。**

该规则不允许牺牲实质 payload。它目前只是一项 oracle-label 诊断规则，因为使用候选 rollout 标签，不是现有 HGNN 可直接执行的在线策略。报告必须明确这一限制。

## 2. 数据范围与预算

- 仅使用 highlevel_v2_train registry 的前 4 个训练场景 322000–322003；必须通过 registry 和 `assert_allowed(split="train")` 验证。
- 不使用 D0/D1 的 validation 322048–322063 或 reserved test 322064–322079。
- 在每个场景的 t=0、15、30、45 四个决策点建立候选子集。每个状态最多 128 个候选，按既定方法：32 个稳定 payload top + 96 个按原 utility_rank 分层抽样。使用新预注册种子 603，候选列表必须在 urgency rollout 前保存。
- 对 t=0、15、30 使用 15-step label horizon；t=45 仅剩 3 步，必须使用真实剩余 horizon=3，并在分析中与前面 15-step 标签分开标记。不得把 3-step label 与 15-step label 的 payload 直接横向比较。
- 总计最多 4 场景 × 4 决策点 × 128 候选 = 2,048 个候选 rollout；先跑一个状态计时。若单状态超过 900 秒或总预计时间超过 30 分钟，停止并报告，不扩容。
- 每个场景完整闭环只运行两臂：sampled-pool payload-first 与 sampled-pool payload-first + urgency tie-break。使用相同 reset seed、相同冻结低层与同一候选子集协议。

## 3. 冻结对象

- 冻结低层：`phase6g_seed0.pt`，确定性 actor mean。
- 冻结 Scheme C checkpoint、matching decoder、adapter config 和正式候选合法性规则。
- 环境、队列转移、SafetyProjector、TaskOption hold=15 保持不变。
- 不训练、不修改正式 Planner、不覆盖 D0/D1/D2 标签、旧数据集或 checkpoint。
- 新输出写入 `results/phase6i_urgency_tiebreak_closedloop_pilot/`。

## 4. 标签与两种选择规则

每个决策状态都从同一状态快照与同一 ledger 快照评估同一组 128 个合法候选。候选 label rollout 遵循既有 J15 约定：只在 scratch candidate environment 中将 arrivals/arrivals_at 置零；当前状态重放仍保留真实 arrivals。urgency 使用本步服务前的 Stage A ledger view。

### Rule A：sampled-pool payload-first
- 从这 128 个候选中取 payload label 最大者。
- payload 在 1 bit 容差内并列时，按稳定候选 ID 最小者选择，确保规则确定且不看 urgent-service 标签。

### Rule B：payload-first + urgency tie-break
- 先筛选 payload 距最大值不超过 1 bit 的候选。
- 仅在这个并列组内，选择 urgent-service label 最大者。
- urgent-service 在 1 bit 容差内并列时，按稳定候选 ID 最小者选择。
- 绝不允许选择 payload 低于最大值超过 1 bit 的候选。
- 两条规则的候选子集、payload labels 和状态快照必须完全相同；Rule B 不能通过查看非并列候选的 urgency 改变 payload 主目标。

由于标签来自冻结低层的候选 rollout，Rule B 是 oracle-label upper-bound style pilot，而不是可部署的 HGNN 策略。不得把它描述为 learned policy，也不得声称结果能直接迁移到正式 Scheme C。

## 5. 闭环执行

- 在每个决策点按对应规则选出 assignment，保持 15-step TaskOption；t=45 使用剩余 3 步。
- 每臂从相同场景 reset 开始，后续状态根据各自真实轨迹自然分化；在 t=15/30/45 重新生成当前状态的候选子集并按规则选择，不复用另一臂未来状态或标签。
- 对每臂记录四个决策点的候选池 hash、候选子集 hash、payload-first 候选、tie-break 候选、payload regret、urgent-service gain、实际 assignment 和选中理由。
- 记录完整 48-step episode payload、逐步/逐浮标 served bits、urgency、age ledger、队列与 dropped bits、首次服务时间、SINR/干扰、路径长度、SafetyProjector 与安全指标。
- 检查两臂每个 episode 的 payload 计账、ledger 守恒与候选合法性。不得静默丢弃失败场景。

## 6. 指标与限制

主要指标：
- 完整 48-step episode 的实际 payload 配对差（tie-break 减 payload-first）。

次要指标：
- 紧急服务量、urgent-service bits；
- 发生 tie-break 的决策次数及占比；
- 选择的 assignment 是否实际不同；
- 每次 tie-break 的 payload 差（必须在 1 bit 容差内）；
- 最老/平均等待年龄、剩余队列、dropped bits；
- 路径、SINR/干扰、投影率、安全违规。

统计单位只有 4 个场景，样本很小。本阶段只报告每场景配对差、均值/中位数和逐场景明细，不宣称统计显著或泛化成功。即使 Rule B 提升紧急服务，也只能说明在 sampled-pool oracle label 条件下，payload 并列 tie-break 具有可行性；它不证明现有 HGNN 可以实现同样效果。

## 7. 交付物

- `docs/research/HTW_UrgencyAware_Tiebreak_ClosedLoop_Pilot_Report_CN.md`
- `results/phase6i_urgency_tiebreak_closedloop_pilot/selected_candidates.csv`
- `results/phase6i_urgency_tiebreak_closedloop_pilot/decision_rows.csv`
- `results/phase6i_urgency_tiebreak_closedloop_pilot/episode_rows.csv`
- `results/phase6i_urgency_tiebreak_closedloop_pilot/summary.json`
- `results/phase6i_urgency_tiebreak_closedloop_pilot/schema.json`

报告必须列出场景/候选覆盖、各决策点的 horizon、运行耗时、候选选择变化、逐场景闭环差值、计账误差、模型/config hashes、缺失字段与 Gate。

## 8. Gate 与停止条件

URGENCY_TIEBREAK_CLOSEDLOOP_PILOT_COMPLETE：
- 4 个场景两臂均完成 48 步；
- 所有选择都符合 1-bit payload tie 约束；
- ledger/payload accounting、状态/候选哈希和合法性检查通过；
- 全部逐场景结果与局限已报告。

URGENCY_TIEBREAK_CLOSEDLOOP_PILOT_INCONCLUSIVE：
- 大多数决策没有发生实际 tie-break、assignment 不变，或紧急服务变化没有稳定方向。

URGENCY_TIEBREAK_CLOSEDLOOP_PILOT_BLOCKED：
- split、标签复现、状态恢复、候选合法性、账本/计账或 checkpoint/config 检查失败，或预算超限。

完成后停止，不训练 HGNN、不生成完整数据集、不改正式系统。报告该规则是否值得进入更大规模开发集评估，并等待确认。
