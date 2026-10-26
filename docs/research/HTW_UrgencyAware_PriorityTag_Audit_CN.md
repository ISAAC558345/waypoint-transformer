# E0 审计：优先级标签、桶级账本与小权重紧急奖励

- 仓库：`ISAAC558345/waypoint-transformer`
- 分支：`feature/maritime-wt-stage1a`
- 指令提交：`3923691d248f9cc20aa73eac60ae3523117f7b31`
- 依据：`docs/research/HTW_UrgencyAware_PriorityTag_Reward_Plan_CN.md`、`docs/research/HTW_UrgencyAware_Objective_Decision_CN.md`、`docs/research/HTW_UrgencyAware_LowLevel_Training_Report_CN.md`
- 本阶段 Gate：`URGENCY_PRIORITY_SCHEMA_BLOCKED`
- 性质：只读审计。没有训练，没有 rollout，没有修改正式 Planner、正式奖励、动作、SafetyProjector、checkpoint 或旧数据集。

## 1. 直接结论

当前环境不能区分普通信息和紧急标签信息。`arrivals_at` 只返回每个浮标的块数，`served_bits` 只按浮标聚合。仓库里没有 `priority_tag`。不能从已有的 buoy-level served bits 反推哪一类信息被服务。

桶级等待年龄可以由现有 `AdmissionAgeLedger` 的 FIFO 记账方式延伸，但必须新增带 tag 的桶，并且 tag 在数据进入队列时生成。tag 不改变正式队列的出队顺序。

`lambda_u=0.05` 相对 payload 项有上界：urgency 被夹到 `[0,1]` 时，bonus 不超过同期 payload 项的 5%。相对已记录的投影/残差惩罚，它不是小影响。已保存的一次 L1 更新探针里，惩罚约 `0.000170`，而 5% 的 payload 项约 `0.0735`。因此本阶段不把小权重说成已经成立，不改这个数，也不启动 L1/L2 训练。

高层仍是 payload-first。紧急度只允许在 payload 相差不超过 1 bit 时打破平局。`delta_payload_max` 继续是 `OWNER_DECISION_REQUIRED`，`lambda_H` 继续是 `NOT_SELECTED`。D0–D7 的 urgent 列是年龄账本，不是 tag-aware served bits，不能拿来做新的 tie-break。

## 2. 现有字段

| 位置 | 已有内容 | 没有的内容 |
|---|---|---|
| `L3BEnv.arrivals_at` | `t` 为 4–36 的偶数时，浮标 6–11 各到 1 个块；其余为 0。与场景无关 | priority tag、到达时刻以外的块身份 |
| `L3BBitQueueEnvV2` | `q_bits`、`served_bits`、`generated_arrival_bits`、`admitted_arrival_bits`、`dropped_bits`、`float_snap_bits`，都是长度 12 的浮标向量 | 按 tag 拆开的服务、丢弃或 snap |
| 队列容量 | `queue_capacity_blocks=4`，`block_payload_bits=1e6`，容量 `4e6` bit。容差 `tol_bits=1` bit | 分类容量 |
| reset 存量 | 浮标 0–2 为 4 块，3–5 为 1 块，6–11 为 0 | 这些存量没有到达事件，也没有 tag |
| `AdmissionAgeLedger` | 每浮标按 `admission_step` 分桶；`age_origin` 为 `reset_convention` 或 `environment_arrival`；服务前 `view` 不改桶；`urgency=clip(oldest_age/48,0,1)`，`urgency_valid` 在 `q_bits>0` 时为 1 | `priority_tag`、桶级 urgency、按 tag 的 served bits |
| 低层 Actor | 正式 task 8 维，含目标队列块数。Stage C 另有独立 36 维 Actor，在 task 后追加 buoy 级 `urgency` 和 `urgency_valid` | 队首 tag、tagged 比例 |
| 低层 Critic | 正式 188 维。Stage C 为 200 维，末尾按 UAV 追加 12 维 | tag 特征 |
| 正式奖励 | `served_bits/1e6 - 0.05*c_project - 0.01*c_residual` | urgency bonus |
| Stage C L2 | `lambda_urgency=1.0`，bonus 用服务前的 buoy 级 oldest urgency 乘该浮标全部 `served_bits` | 桶级、tag 级服务统计 |
| D0–D7 标签 | 零到达或真实到达下的 payload 与年龄加权 urgent-service | tag-aware served bits |

服务顺序已经是“先服务本步之前的队列，再接纳本步到达”。溢出只丢新到达中放不进剩余容量的部分，不丢旧数据。正的 float snap 在年龄账本里从最老的桶扣，负 snap 记到最新的桶。这些规则可以沿用。正式环境本身不暴露桶。

## 3. priority tag 的来源

没有真实海事业务标签可以读取。下一阶段若实现，只能在独立实验配置里生成合成 tag，并且在 schema、日志和报告里写成 `synthetic_priority_tag`。不得写成真实业务标签，也不得在服务发生之后按聚合 served bits 补标签。

预注册候选，尚未实现，也没有用 validation 或 reserved test 调过：

- `beta_tag=0.25`。刚入队的 tag=1 桶，年龄紧急度为 0，合成紧急度为 0.25。它不是最优值。
- `tag_rate=0.25`。每个新到达块单独抽一次，整块 `1e6` bit 共用一个二元 tag。不把一块拆成混合 tag。
- 种子 `SeedSequence([606, scenario_id, admission_step, buoy])`。
- reset 存量全部 `priority_tag=0`，`age_origin=reset_convention`，`admission_step=0`。
- 年龄：`age=max(0, t-admission_step)`，`u_age=clip(age/48, 0, 1)`，`u=clip(u_age+beta_tag*tag, 0, 1)`。`48` 沿用 Stage A 的 `AGE_SCALE_STEPS`。
- 空队列用 `urgency_valid=0`。`urgency=0` 且 `tag=0` 不能单独表示空队列。

年龄达到 48 步时，tag=0 和 tag=1 都会被夹到 1，标签加成不再增加。年龄 12 步的普通信息与刚入队的 tag=1 信息，合成紧急度都是 0.25。这是该候选定义的直接结果。

负责人可以在第一次新 rollout 之前改 `tag_rate` 或种子。实现开始之后这三项冻结。

## 4. 桶级守恒与服务统计

新账本必须是正式队列旁边的只读跟随器，放在新的代码目录。不修改 `bit_queue_v2.py`、`AdmissionAgeLedger` 和正式 `step`。

每个桶保存 `admission_step`、`age_origin`、`priority_tag`、`bits_remaining`。出队键只有 `admission_step` 和插入顺序，与 Stage A 的 `_drain` 相同。`priority_tag` 不进入排序。tag 改变奖励统计，不改变谁先被服务。若以后要做优先级队列调度，必须另立实验。

对每个浮标、每个环境步，使用该步已经返回的向量：

- 服务：从最老的桶扣 `served_bits`。扣到 tag=0 的记入 `served_ordinary_bits`，扣到 tag=1 的记入 `served_tagged_bits`。二者之和等于该浮标的 `served_bits`。
- 丢弃：`dropped_bits` 只从本步新到达桶扣除，因为环境的溢出发生在服务之后、只作用于新到达。丢弃量不能超过本步 `generated`。
- 接纳：新到达减去丢弃后，剩余 bit 带本步抽到的 tag 进入队尾。
- snap：正 snap 继续从最老的桶扣，负 snap 加到最新的桶。snap 不计入服务奖励。
- 守恒：桶内 bit 之和与 `q_bits` 的差不超过 `tol_bits`。`generated = admitted + dropped`。`served_ordinary + served_tagged = served`。`tag0_bits + tag1_bits = q_bits`。

奖励里的 `urgency_served_bits` 只对实际扣掉的服务 bit，乘该桶服务前的 `u`。generated、admitted、dropped 和 snap 都不乘进服务奖励。urgency 在 `apply_transition` 之前计算。

Stage A 的 `_drain` 在剩余超过 `1e-6` bit 时失败，桶尘埃阈值是 `1e-8` bit，环境容差是 `1` bit。新账本沿用这三套阈值，不放宽。

现有 D0–D7 表没有这些列。在新账本的守恒测试通过之前，不生成 tag-aware 标签，也不回填旧表。

## 5. 低层观测接口

不改 `lowlevel_v2/observation.py`、`critic_state.py` 和正式 Actor。新观测放在独立模块，正式向量保持原样，新列追加在 task 块之后。

队首桶才是 FIFO 下一次会服务的数据。同一浮标里可以既有普通旧数据，也有靠后的 tag=1 数据。Actor 若只看到浮标级最大 tag，会把队首的普通数据误当成紧急数据。第一版目标浮标特征因此是：

- `head_u`：队首桶服务前的合成紧急度；
- `head_tag`：队首桶的 `priority_tag`；
- `urgency_valid`：目标队列非空时为 1；
- `tagged_fraction`：该浮标 tag=1 的 bit 占 `q_bits` 的比例。

空闲目标四项都为 0，并且 `urgency_valid=0`。这与“队列非空但队首是 tag=0、年龄为 0”分开。

维度：正式 Actor 为 `5+8+4+16+1=34`。Stage C 追加 2 列后是 36，且那两列是 buoy 级 oldest urgency。本实验不加载 Stage C checkpoint。从冻结 MAPPO-V2 热启动，四列新权重置零。新 Actor 宽 38。Critic 保留正式 188 维，按 UAV 顺序追加上述 4 项（24 维），再追加全局 `ordinary_bits/1e6` 和 `tagged_bits/1e6`（2 维），宽 214。checkpoint metadata 写入这两项宽度、schema 版本、`beta_tag`、`tag_rate`、种子和 `lambda_u`。宽度不符就拒绝加载，不静默丢列。

新文件独立保存。`phase6g_seed0.pt` 的 SHA-256 继续是 `92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e`。

观测只用当前队列和当前 assignment。不读 `remaining_arrivals`，不用未来 served bits 回填当前 urgency。

## 6. `lambda_u=0.05` 的尺度

正式奖励是：

`r = served_bits/1e6 - 0.05*c_project - 0.01*c_residual`

其中 `LAMBDA_PROJECT=0.05`，`LAMBDA_RESIDUAL=0.01`，`PAYLOAD_SCALE=1e6`，`ACTION_SCALE=1`。计划中的公式就是这项正式奖励，再加 `lambda_u * urgency_served_bits/1e6`。惩罚系数不改。

代数上界：每个桶的 `u` 在 `[0,1]`，所以

`0 <= urgency_served_bits <= served_bits`

`lambda_u=0.05` 时，bonus 不超过同期 payload 项的 5%。这是相对 payload 的小权重。它不是相对惩罚项的小权重。

已提交探针 `results/phase6i_urgency_lowlevel/smoke_summary.json` 的 `one_update_timing`：L1、model seed 0、1 次更新，`communication_reward=1.4707129520117694`，`mean_reward=1.4705432653427124`。二者之差是该批的惩罚，为 `0.000169686669057`。同一批 payload 项的 5% 是 `0.073535647601`，约为惩罚的 433 倍。

`results/phase6i_urgency_lowlevel/reward_scale.json` 在训练场景 321000 上记录年龄账本的 `urgent_service_bits/payload_bits=0.2682825626921449`。那是 buoy 级 oldest urgency，不是 tag 桶。若服务构成仍接近这个比例，`lambda_u=0.05` 的 bonus 约为 payload 项的 1.34%，在上述探针的均值上约 `0.01973`，仍约为惩罚的 116 倍。tag 桶的真实比例要等新账本才会有，不能用这个 0.268 代替。

Stage C 的 `lambda_urgency=1.0` 已经在开发池上形成 payload 与紧急服务的 tradeoff，且没有替换正式 MAPPO。`0.05` 比 1.0 小 20 倍，但上面的惩罚比较说明它仍可能压过投影和残差惩罚。本审计不把它改成 0.02 或其他值。`lambda_u` 记为 `OWNER_CONFIRMATION_REQUIRED`。确认之前不训练。`0.1` 不是默认值。`{0, 0.02, 0.05}` 的开发集敏感性只属于确认之后的限定检查。

## 7. 高层边界

高层第一版继续 payload-first：先最大化真实到达 payload，只在 1 bit 以内用 urgent-service 打破平局，再用更小候选 ID。不允许用 urgency 选择 payload 明显更低的 assignment。不采用 D7 的比例网格，不设置 `delta_payload_max`。

这个 tie-break 需要新的真实到达、桶级 `urgency_served_bits`。D6 的 `real_urgent_service_bits` 来自服务前的 buoy 级 oldest urgency，不能拆成 ordinary 与 tagged。高层 HGNN、正式 Planner 和旧标签保持冻结。低层实验和新标签接口分别通过之前，不训练高层。

时刻继续分开。t=0、15、30 的标签 horizon 为 15。t=45 的 horizon 为 3，不与 15 步标签混合。低层轨迹变化仍可能改变整集 payload。冻结 Scheme C 下的配对闭环必须同时报告 payload 和紧急服务，不能把离线 tie-break 写成 payload 不下降的保证。

## 8. L0、L1、L2 消融

三组使用同一环境、同一 SafetyProjector、同一动作结构 `fly_toward + residual -> clip -> SafetyProjector`、同一低层训练池 321000–321095 和开发池 321096–321127。不读取高层 validation 322048–322063、reserved 322064–322079，也不读取低层 reserved 320100–320131。种子在训练前写入协议。

| 组 | 观测 | 奖励 | 训练 |
|---|---|---|---|
| L0 | 正式 34 维 Actor、188 维 Critic | 正式奖励，`lambda_u=0` | 不重训，使用冻结 `phase6g_seed0.pt` |
| L1 | 新 38 维 Actor、214 维 Critic | 与 L0 同一正式奖励 | 新目录热启动，新列置零 |
| L2 | 与 L1 相同 | 正式奖励加已确认的 `lambda_u` 桶级 bonus | 与 L1 相同的更新次数、PPO 设置和场景顺序 |

L0 对 L1 只比较新输入。L1 对 L2 只比较 bonus。L2 不得比 L1 多看特征或少受惩罚。成功标准不是训练 reward 上升。配对场景上报告 payload、ordinary/tagged served bits、等待年龄、tagged service rate、队列、丢弃、SINR、干扰、投影率和安全违规。高层在这三组里都保持冻结的 payload-first；低层变化引起的 payload 差照实记录。

这一节是确认 `lambda_u` 并且桶级守恒测试通过之后的实验设计。本阶段不运行它。

## 9. 需要新增的内容与测试

新增独立 schema，建议版本名 `synthetic_priority_bucket_v1`。至少包括：`priority_tag`、`age_origin`、`admission_step`、`bits_remaining`、`u_age`、`u`、`beta_tag`、`tag_rate`、`tag_seed`、`served_ordinary_bits`、`served_tagged_bits`、`dropped_ordinary_bits`、`dropped_tagged_bits`、`urgency_valid`、`head_u`、`head_tag`、`tagged_fraction`。旧列不改名、不覆盖。

实现前的测试：

- reset 存量全是 tag=0，且与 `q_bits` 一致；
- 新到达块的 tag 之和等于 `generated`，同一块只有一个 tag；
- 更老的普通桶先于更年轻的 tag=1 桶被服务，总 `served_bits` 与环境一致；
- 丢弃只来自本步新到达，不从旧桶扣除；
- snap 不进入服务奖励；
- 空队列 `urgency_valid=0`，与“非空但紧急度为 0”可区分；
- `u=clip(age/48+0.25*tag,0,1)` 在服务前计算；
- 桶和、分类服务、分类丢弃都在 `tol_bits` 内；
- 观测不含未来到达；
- `lambda_u=0` 时奖励与 `low_level_reward` 逐项一致；
- 正式 checkpoint、Planner 和 D0–D7 文件的哈希不变。

## 10. Gate

`URGENCY_PRIORITY_SCHEMA_BLOCKED`。

tag 的合成规则、年龄公式、FIFO 跟随方式和消融分组已经写明。阻塞点是奖励尺度：`lambda_u=0.05` 相对 payload 项不超过 5%，相对已记录惩罚约大两个数量级。同时，当前环境的 served bits 还不能按 tag 分解；上面的账本是最小接口，尚未实现，也不能用旧标签冒充。

负责人确认 `lambda_u` 之前，不实现训练、不跑完整 benchmark、不进入下一阶段的长训练。确认之后的下一步才是独立账本、观测和奖励构造，以及第 9 节的单元测试和 smoke test。高层 HGNN 继续冻结。
