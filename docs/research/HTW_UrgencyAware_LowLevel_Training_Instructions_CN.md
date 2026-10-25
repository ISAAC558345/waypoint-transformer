# Stage C 执行指令：在满编任务分布下训练紧急度感知低层 MAPPO

- 仓库：ISAAC558345/waypoint-transformer
- 分支：feature/maritime-wt-stage1a
- 依据报告：docs/research/HTW_CrossTask_LowLevel_Baseline_CN.md
- 前置 Gate：CROSSTASK_BASELINE_COMPLETE
- 本阶段 Gate：URGENCY_LOWLEVEL_INPUT_SMOKE_PASS / URGENCY_LOWLEVEL_TRAINING_COMPLETE / URGENCY_LOWLEVEL_EVAL_COMPLETE / URGENCY_LOWLEVEL_INCONCLUSIVE / URGENCY_LOWLEVEL_BLOCKED
- 目标：训练低层 urgency-aware MAPPO 候选，并把“增加 urgency 输入”与“奖励中显式强调紧急服务”作为不同实验因素。
- 保留正式 Scheme C、phase6g_seed0.pt、原始 observation/reward 和全部历史结果；新模型/日志写入独立目录。

## 1. Stage B 结果与关键决策

Stage B Gate 为 CROSSTASK_BASELINE_COMPLETE。32 个低层开发场景 321096–321127 在四种任务分布下完成 48 步评估，256 个 episode 无失败，账本与环境计账检查通过。

主要 payload 均值：
- A Scheme C：69,190,346 bit；
- B G1_RATE k=6：65,553,117 bit；
- C G1_RATE k=5 均匀随机候选：59,782,318 bit；
- D G1_RATE k=6 合法候选池均匀采样：62,026,998 bit。

配对结果：
- B−A：-3,637,229 bit，95% CI [-4,978,450, -2,177,066]；
- C−A：-9,408,029 bit，95% CI [-11,267,198, -7,555,574]；
- D−A：-7,163,348 bit，95% CI [-8,484,172, -5,862,965]；
- D−C：+2,244,681 bit，95% CI [+369,332, +4,201,026]。

随机候选分布平均每个决策有约 1.77–2.10 个 idle UAV，而 Scheme C 在这批轨迹中没有 idle UAV。随机分布的有效服务步和 payload 均更低。因此不能把 C/D 的下降直接解释成低层控制能力差，也不能在这个混杂分布上直接训练 urgency reward 并把变化归因于 urgency。

另一个重要发现：B 相对 A 的 payload 更低，但 urgency-weighted service bits 略高。这说明“紧急加权服务指标变好”和“总 payload 变好”可能冲突，必须同时报告，不能只优化单一指标。

## 2. 本阶段设计原则

1. 先在以 Scheme C 产生的满编任务为主的训练分布上训练，避免随机候选大量 idle 槽位混淆紧急度因素。
2. 先训练输入增强、原奖励不变的低层候选；再独立训练 urgency-weighted reward 候选。不得同时改变观测、奖励、动作和任务分布。
3. 所有候选都使用同一份冻结的高层 Scheme C 任务生成协议、相同环境配置和相同的训练场景池。
4. 低层仍只执行高层 assignment，不自行重新分配浮标。动作结构保持 `fly_toward + residual -> clip -> SafetyProjector`。
5. Stage A 的 urgency ledger/逐步日志继续作为唯一年龄数据源；不得在 observation 里重新独立估算一套年龄。
6. 不使用已反复分析的 validation 322048–322063 选择模型，不使用 reserved test 322064–322079 调参。

## 3. 冻结基线与训练数据

冻结：
- Scheme C 当前正式 checkpoint、matching decoder、G1_RATE k=6 候选池配置；
- 原低层 `phase6g_seed0.pt`；
- 原始奖励系数、SafetyProjector、环境队列转移；
- Stage A 的 `episode_admission_age_v1`，`age_scale_steps=48`。

训练场景：
- 从仓库 registry 读取现有低层训练池 `321000–321095`，按现有 `phase6g` 规则执行 split guard。
- 不使用 DEV_POOL 321096–321127 做训练；该池已经用于 Stage B 开发诊断。
- 不读取 reserved pools、historical pools 或高层 reserved test。
- 新训练任务在每个高层决策点 t=0、15、30、45，由冻结 Scheme C 对当前状态重新选择正式合法的 G1_RATE k=6 候选 assignment。记录候选数、pool hash、选中 index、分数和 idle slot 数。
- 如果训练场景无法安全调用正式 Scheme C scorer，必须报告 BLOCKED；不得关闭 split guard 或以 G1_RATE 随机采样静默替代 Scheme C。

需要先确认冻结 Scheme C 在 321000–321095 上的场景来源是否符合仓库既有 split 契约。如果不符合，暂停并提出经 registry 允许的替代方案，不要自行扩大 split。

## 4. 先做输入接口实现与 smoke test

新建版本化实验模块，不修改旧的 `build_actor_observation`、`build_critic_state` 或旧训练入口的默认行为。优先用新的 observation builder/显式 feature flag。

### Actor
- 对当前 TaskOption 的目标浮标加入 `oldest_wait_age / 48` 与 `urgency_valid`。
- idle task 的 urgency 数值为 0，validity 为 0；非空但年龄为 0 的任务 validity 为 1。
- 年龄来自本步服务发生前的 Stage A ledger view。
- 其他观测字段保持原样，明确记录新的 TASK_DIM 与 Actor 第一层输入维度。

### Critic
- 增加每架 UAV 当前目标的 urgency/validity，并按固定顺序拼接。
- 如增加全局 age-weighted backlog 摘要，必须在配置中声明并固定；第一版优先只加每架 UAV 目标 urgency/validity，避免同时增加多个新因素。
- 显式记录新 Critic 输入维度和 checkpoint metadata。

### Smoke 验收
- urgency 在当前目标改变时映射到正确浮标；
- 不读取未来 arrivals、未来 served bits 或未来轨迹；
- observation、critic state、action、loss/gradient 全部 finite；
- checkpoint 可保存、重载，维度不匹配会明确报错；
- 原始模型仍按原维度加载，正式 Planner 未改变；
- 短程环境 smoke 通过账本、队列、payload 和安全检查。

新 checkpoint 独立保存。不得静默部分加载旧权重；必须输出逐层加载清单，明确复用层、新建层和初始化方式。

## 5. 训练实验分组

### L0：原始低层基线
- 原始 MAPPO-V2，原 observation、原 reward。
- 只评估/复用已有结果，不覆盖权重。

### L1：Urgency-input only
- 新 Actor/Critic 观测加入 urgency/validity。
- 奖励完全保持原定义：
  `r = served_bits / 1e6 - 0.05*c_project - 0.01*c_residual`
- 目的：检验紧急度状态输入本身是否有助于给定任务的运动执行。不能宣称该组显式优化了紧急服务优先级，因为奖励尚未改变。

### L2：Urgency-input + urgency-weighted reward
- 以 L1 为起点，单独增加 urgency-weighted served bits 奖励：
  `r = served_bits / 1e6 + lambda_urgency * sum_k(urgency_k[t] * served_bits_k[t]) / 1e6 - 0.05*c_project - 0.01*c_residual`
- urgency 必须取服务发生前的 ledger view，served bits 按实际 buoy index 聚合。
- 先在训练/开发数据上审查各项奖励尺度，再在训练前写死 `lambda_urgency` 与归一化配置。至少保留 `lambda_urgency=0` 的 L1 对照；不能根据 reserved test 调权重。
- 若奖励尺度审计不能支持一个合理固定权重，先报告并停止，不得随意选择大权重。

L1 与 L2 必须使用同样的任务生成协议、场景、模型 seed 组和更新预算。不能用 L2 更多训练步数来制造优势。

## 6. 训练要求

- 优先 warm-start 原 Actor 的兼容层；新增输入权重使用明确的初始化策略。若 Critic/Actor 结构不能合理迁移，则使用 scratch 并在报告中说明。
- 至少 3 个预注册 model seeds；具体 seeds 在启动训练前写入 protocol，不得看到结果后挑选。
- 更新数、rollout 长度、PPO 参数沿用当前正式 MAPPO-V2，除非有单独的配置审计理由；不同时调学习率、奖励系数、动作残差系数或 SafetyProjector。
- 每次更新记录 policy loss、value loss、entropy、梯度范数、平均 payload reward、urgency reward、投影代价、非有限值和 checkpoint hash。
- 先跑 smoke test，再启动训练。任何账本/计账/接口错误立即停止。

## 7. 评估与闭环 Benchmark

### 固定任务执行诊断
在相同状态快照和相同 Scheme C assignment 下，比较冻结 L0、L1、L2 的低层执行；使用同一场景/reset seed 和同一 15 步 TaskOption。报告局部 15 步 served bits、首次进入服务半径、有效服务步数、路径长度、SINR/干扰、投影率与安全指标。

### 完整闭环
每个低层候选都搭配冻结 Scheme C，运行完整 48 步 episode。主指标为实际累计 payload。紧急度指标包括：
- urgency-weighted served bits；
- 高 urgency 分位组 served bits；
- oldest/mean wait age；
- 高紧急度队列首次服务时间；
- age-weighted backlog、剩余队列与 dropped bits。

所有评估在登记的开发池执行；模型选择与超参数只在训练/开发数据内完成。最终 reserved test 只在全部设计和 checkpoint 冻结后一次性确认。

报告每场景配对差、95% bootstrap CI、正/负/持平场景数、seed 间方差、最差场景及完整逐步日志。总 payload 与紧急服务指标必须并列报告。

## 8. Gate

- `URGENCY_LOWLEVEL_INPUT_SMOKE_PASS`：L1 新输入接口和短程 smoke 通过。
- `URGENCY_LOWLEVEL_TRAINING_COMPLETE`：L1/L2 所有预注册 seeds 完成，无静默失败。
- `URGENCY_LOWLEVEL_EVAL_COMPLETE`：L0/L1/L2 固定任务与完整闭环对照齐全。
- `URGENCY_LOWLEVEL_INCONCLUSIVE`：收益区间跨 0、seed 不稳定或紧急服务与 payload 权衡不清。
- `URGENCY_LOWLEVEL_BLOCKED`：split、checkpoint、年龄账本、计账、维度或安全检查失败。

不允许仅凭训练 reward 上升宣布成功。L2 若提升 urgency-weighted service 但降低 payload，必须报告该 trade-off；只有在预注册标准下综合结果支持时才保留 L2。

## 9. 交付物

- docs/research/HTW_UrgencyAware_LowLevel_Training_Report_CN.md
- 新的 urgency-aware low-level 代码与测试，放入独立 phase 目录；
- 新的配置、checkpoint、训练日志和哈希；
- results/phase6i_urgency_lowlevel/ 的 episode/step 结果与 summary；
- L0/L1/L2 的配对 benchmark 图表和 Gate。

完成后提交全部变更到当前分支，汇报实际代码改动、训练 seeds/预算、测试结果、Gate 和下一步建议。不要在本阶段训练高层 HGNN；高层紧急度改造留到低层实验完成后独立进行。
