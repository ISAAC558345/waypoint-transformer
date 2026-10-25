# P0：紧急度语义与逐步诊断日志审计

计划文本在提交 `dd8e378c3168deba2d1127e95fd51b682b56ca30`。本文件是该计划的 P0 结果。P5 报告在 `docs/research/HTW_NextStep_T30LossAttribution_CN.md`。本次只读源码和已有报告，没有训练，没有改 FormalPlanner、HGNN、MAPPO、奖励、动作、SafetyProjector 或队列转移，没有覆盖 checkpoint 和 P1–P5 结果，没有读取 reserved test `322064–322079`。

## Gate

`URGENCY_STATE_AUDIT_COMPLETE`

正式环境 `L3BBitQueueEnvV2` 的服务量只取决于服务前的聚合 `q_bits` 和本步链路容量。到达时刻由 `arrivals_at(t)` 决定，与动作无关。因此可以在环境转移之外维护按入院步分桶的 bit 账本，使桶内比特之和与 `q_bits` 一致，并用它定义本 episode 的等待年龄。reset 时已经在队列里的比特没有历史时间戳。它们的年龄从 `t=0` 起算，来源标记为 `reset_convention`。这个量叫 `episode_admission_age`。它是本 episode 内可复核的等待步数，不是 episode 开始前的真实时延，也不是 deadline。

现有浮标特征里已经有一列名叫 `urgency`，数值是当前队列除以该状态的最大队列。那一列继续表示队列长度的相对值。新年龄特征使用新字段名和新 schema 版本，不改写这一列。

## 1. 到达、服务和初始队列

正式闭环用的是 `src/wt/benchmark_v2/bit_queue_v2.py` 的 `L3BBitQueueEnvV2`。父类 `src/wt/l3b/env.py` 的 `L3BEnv.step` 仍按整块进度扣队列，正式 bit 账本不走那条转移。

配置来自 `src/wt/l3b/config.py`：`M=6`，`K=12`，`T=48`，`delta_t=1`，`block_payload_bits=1e6`，`queue_capacity_blocks=4`，`num_hot=3`。计账容差是 `max(1e-6 * 1e6, 1e-9)=1` bit。

`L3BEnv._reset_state` 把队列设为块数，再由 V2 乘上 `1e6`：

| 浮标 | reset 时的队列 | 本 episode 内的到达 |
| --- | --- | --- |
| 0、1、2 | 4 block，已满 | 无。到达只写给下标 `>=6` 的浮标 |
| 3、4、5 | 1 block | 无 |
| 6–11 | 0 | `t` 为 4 到 36 的偶数时，每个这样的步各到 1 block |

`arrivals_at(t)` 只看时间下标。`arrivals()` 读取的是 `step` 自增前的 `env.t`。4 到 36 的偶数步共 17 次，每次 6 个浮标。

`L3BBitQueueEnvV2.step` 的顺序是：

1. 用本步动作更新位置，并写入本步 assignment。
2. 对在服务半径内的关联链路计算 SINR 和速率。同一浮标上的链路容量相加，`raw = rate * delta_t`。合法候选不允许两架 UAV 共用一个浮标；环境本身仍会把重复关联的容量加总。
3. 用 `apply_service_then_arrival` 先服务本步开始时已经在队列里的比特：`served = min(q_before, raw)`。
4. 再把本步 `arrivals_at(t)` 放进剩余容量。放不下的新到达记为 `dropped`，不替换队列里的旧比特。
5. `progress` 只作遗留诊断，函数在更新后检查它没有改动 `q_bits` 和 payload。
6. 最后 `t += 1`。

守恒式是 `q_bits = initial_q_bits + generated - served - dropped - snap`。`service_remainder_bits` 只统计整块完成数，不从 `q_bits` 再扣一次。

聚合队列没有逐比特顺序。服务量不取决于桶的先后。按入院步从老到新扣掉恰好 `served` 比特，再把本步新入院比特放进新桶，得到的 `q_after` 与现在的转移相同。同一入院步内的比特互相可替换。丢弃只发生在新到达放不进剩余容量时，年龄账本必须从本步新桶扣除，才能和 `dropped_bits` 对齐。`float_snap_bits` 会把容差内的残差清零，Stage A 的账本要使用同一次 snap，否则桶之和会偏离 `q_bits`。

reset 存量没有入院记录。Stage A 把这些比特放进 `admission_step=0`、`age_origin=reset_convention` 的桶。在 `t=0` 服务前，它们的年龄是 0。随后每个未服务的步，年龄加 1。这样 0–2 号浮标上未服务的满队列会随步数变老；6–11 号浮标的新到达在入院那一步不能被服务，下一步开始时年龄为 1。空队列的 `urgency_valid_k=0`。非空且年龄为 0 时 `urgency_valid_k=1`，两者分开。

建议在实验配置里于测量任何回报之前写死：

- `age_scale_steps = 48`，等于 `episode_horizon`。
- `urgency_k = clip(oldest_wait_age_k / 48, 0, 1)`。
- 同时保存 `mean_wait_age_k`（按比特加权）和 `age_weighted_backlog_k = sum(age * bits)`。
- 不在 `322048–322063` 或 reserved test 上改这个尺度。

`src/wt/benchmark_v2/phase6d/shadow.py` 的 `capture_state` 保存 `q_bits` 和累计服务、到达、丢弃，不保存年龄桶，也不保存 rng。分支之后各臂的服务历史会不同，桶的组成不能从当时的一个 `q_bits` 标量恢复。Stage A 若使用 wrapper，就要在 wrapper 自己的 capture/restore 里复制桶。正式 `capture_state` 这次不改。

## 2. 高层特征和 J15 标签

`hgnn_tensors` 从当前状态组装 Scheme C 的图。`state_arrays` 只放当前几何、`q_bits`、关联、距离、增益、假设速率、当前 SINR、当前速率和 `time_index`。

| 张量 | 宽度 | 通道 |
| --- | --- | --- |
| UAV 节点 | 8 | 归一化位置、速度、progress、是否有目标、目标队列占容量比、末位常数 0 |
| 浮标节点 | 8 | 归一化位置、`queue/capacity`、上一拍 `arrivals_at(t-1)/capacity`、目标上的 progress、是否被占用、`K/12`、`queue/max(queue)` |
| UAV–浮标边 | 模型输入 4 | 距离、归一化假设速率、ETA、再由 `append_queue_edge` 把浮标第 3 列 `queue/capacity` 复制到每条边 |
| UAV–UAV 边 | 2 | 距离、由增益和簇系数得到的干扰 |
| 全局 | 3 | `t/48`、总队列占满容量的比例、`M/6` |

常量 `BUOY_DIM` 在 `src/wt/htw_phase9/const.py` 里是 7，`EDGE_UB_DIM` 是 3。实际 `hgnn_tensors` 写出 8 维浮标特征，图在进入模型前把边从 3 维补成 4 维。`QUEUE_INDEX=2`。正式 `graph_from_env` 按这个契约取图。Stage A 以后若增加年龄，读取 checkpoint 的 `encoder.buoy_enc.0.weight` 和 `encoder.edge_enc.0.weight` 作为旧宽度，不拿 `BUOY_DIM=7` 代替实测宽度。

浮标第 8 列的计算在 `hgnn_revision.py` 里变量名叫 `urgency`，公式是 `clip(queue / (max(queue)+1e-8), 0, 1)`。它随本状态的队列长度单调，不随等待步数单调：两个浮标队列相同、入院步不同时，这一列相同。上一拍到达列只在 `t>0` 时等于 `arrivals_at(t-1)`，也不是年龄。

J15 标签由 `batch_j15_mappo` 生成，名字是 `J15_MAPPO_CURRENT`。对每个合法候选，它恢复决策时的状态，把 `arrivals` 和 `arrivals_at` 置为 0，再用冻结 MAPPO-V2 的确定性均值走完 `HOLD=15` 步，标签等于这 15 步 `payload_increment_bits` 之和。数据集时间只有 `PRIMARY_TIMES=(0, 15, 30)`。剩余步数不足 15 时，`jobs.py` 直接拒绝，所以 `t=45` 的 3 步窗口没有 J15 标签。`utility_rank` 是这些比特标签的 1 bit 稠密名次，`EPS_BITS=1.0`。标签文件里的字段是 `j15_bits` 和 `utility_rank`。排序损失读的是这组 payload 标签。图里不允许带 `j15_bits`。

因此当前高层监督信号是 15 步、到达被置零的 payload。完整闭环评估仍会在偶数步接收真实到达。新的 urgency utility 必须写明它沿用置零到达，还是沿用真实 `arrivals_at`。两套标签不能共用同一个 `j15_bits` 数组。

## 3. 低层观测和奖励

Actor 观测在 `build_actor_observation`。宽度是 `SELF_DIM=5`、`TASK_DIM=8`、`COMM_DIM=4`，邻居边 `NEIGHBOR_EDGE_DIM=11`，邻居编码器默认输出 16，再加 `d_nearest`。`ResidualActor` 第一层输入是 `5+8+4+16+1=34`。

任务向量：有目标时为有效标记、相对位置、距离、目标 `queue` 的块数、`progress`、剩余 hold 比例、剩余 episode 比例。空闲目标时有效标记为 0，队列和距离槽保持 0，只保留 hold 和剩余时间。空闲和“年龄为 0 的真实目标”要靠有效标记区分。通信向量是该 UAV 的 SINR、速率、总干扰和干扰噪声比，来自当前 `communication_features`。观测函数拒绝 `remaining_arrivals`、`arrivals_future`、`future_trajectory`。

Critic 把全体 UAV 的 self、task、association、全部 `queue`、progress、SINR、速率、两两距离、干扰矩阵、hold 和剩余时间接成一个向量。`expected_critic_dim(6, 12)=188`。队列项是块数 `q_bits/1e6`，没有年龄。

`low_level_reward` 与 `unchanged_step_reward` 同一结构：

`r = sum(served_bits) / 1e6 - 0.05 * c_project - 0.01 * c_residual`

`c_project` 是执行动作相对投影前期望动作的均方误差，`c_residual` 是 `0.25 * tanh(pre)` 的均方。`PAYLOAD_SCALE=1e6`，`ACTION_SCALE=1`。主项是全队 served bits 之和，没有按浮标加权。动作仍是 `fly_toward + residual -> clip_unit_step -> SafetyProjector`。投影失败时执行零动作。

训练采样在 `TrainWorker`：低层训练池 `321000–321095`，每 15 步从 `G1_RATE k=5` 的合法池里均匀抽一个 assignment。这不是 Scheme C，也不是 `G1_RATE k=6`。`worker.step` 的记录只有全队平均 SINR、平均干扰和本步 served 总和，done 时立刻 reset，不留下逐步轨迹。

## 4. P5 缺的逐步字段记在哪里

P5 已确认 UAV 表只有窗口合计。逐步量在当时的循环里出现过，写盘前被平均掉了。

| 字段 | 现在的来源 | 记录时机 |
| --- | --- | --- |
| UAV 绝对位置 | `env.uav_positions` | `step` 返回后立刻复制。服务用的是这一步移动后的位置 |
| assignment、目标浮标、hold | `TaskOption.association`、`remaining_hold` | 调用 `step` 之前复制 |
| `a_des`、投影动作、执行动作、投影状态和范数 | `clip_unit_step` 与 `SafetyProjector.project` 的返回值 | `env.step` 之前复制。P4 只累计了范数超过 `1e-3` 的次数 |
| 每浮标 served bits、生成、入院、丢弃 | `info["served_bits"]`、`generated_arrival_bits`、`admitted_arrival_bits`、`dropped_bits_step` | `step` 返回后复制。按浮标存。合法分配下再按唯一目标抄到 UAV |
| SINR、速率 | `info["sinr"]`、`info["rate_bps"]` | 用返回值。它们是本步服务容量所用的链路量 |
| 干扰 | `communication_features(env)` | 在本步 `step` 返回后、下一步动作之前调用。此时位置还是本步服务位置 |
| 队列、年龄、`urgency_valid` | wrapper 账本和 `env.q_bits` | 本步服务前记年龄，服务后记 `q_bits`、入院和丢弃 |
| 高层决策、候选池哈希、候选下标、Scheme C 分数 | `FormalPlanner.select_assignment` 的 `Selection` | 只在 `t=0,15,30,45` 记录。分数是当时已经算出的分数 |
| payload 与守恒误差 | `payload_increment_bits`、`conservation_error_bits` | 每步记录 |

这些量都来自已经发生的状态或本步返回值。日志函数只复制数组，不写回 `env`，不改 `arrivals_at`，不把未来到达或未来 served bits 放进 observation。

记录点放在新的实验 runner 里，包住现有 `env.step`。不把写盘放进 `L3BBitQueueEnvV2.step`，也不改 P1–P5 的 `run_attribution.py`、`run_counterfactual.py`、`run_preserve.py`。`phase6g` 的 `evaluate.py` 在内存里有过 `series`，内容是全队均值，函数返回后不再作为 P5 的逐步表。新日志写入独立目录。启用和关闭日志时，同一动作序列的最终 payload 和 `q_bits` 必须一致。

## 5. 以后实现紧急度时要动的接口

本次不改这些接口。Stage A 及以后若进入实现，改动和风险如下。

| 位置 | 改动 | 风险 |
| --- | --- | --- |
| 新 wrapper，而不是 `bit_queue_v2.py` 的正式 `step` | 服务前读 `q_bits` 和容量，服务后按 `served`、`admitted`、`dropped`、snap 更新桶 | 改正式 `step` 会改变已冻结闭环的队列语义 |
| wrapper 的 capture/restore | 分支时复制桶和 `age_origin` | 只恢复 `q_bits` 会把混合年龄压成一个标量 |
| 新 schema 模块和新数据集目录 | 浮标节点追加 `oldest_wait_age`、`urgency_valid`；边若追加，只允许复制同一状态的浮标年龄，所有候选共用 | 改 `hgnn_tensors` 或浮标第 3、8 列会让旧 Scheme C 读到另一套输入。`state_hash` 会因新数组变化 |
| 新标签文件 | 保留 `j15_bits`。另存 urgency utility，并写明到达是置零还是真实 `arrivals_at` | 覆盖 shard 里的 `j15_bits` 会改变 Scheme C 的监督标签 |
| `ResidualActor`、`CentralizedCritic` | 任务向量追加目标年龄和有效标记；Critic 追加每架 UAV 的目标年龄和约定好的全局摘要。新 checkpoint 单独存放 | 第一层宽度从 34 和 188 变化。旧 `phase6g_seed0.pt` 不能静默加载。权重迁移必须显式列出哪些层复用 |
| 奖励的实验副本 | 第一组保持现有三项。第二组才加 `lambda_urgency * sum(urgency_k * served_bits_k) / 1e6`，`lambda_urgency=0` 作为对照 | `urgency_k` 取本步服务前的年龄，下标与 `served_bits` 的浮标一致。大权重会压过 `1e6` 归一化后的 payload |
| 动作和 SafetyProjector | 保持现有结构 | 这次审计没有动作结构失效的证据 |
| split | 从 `phase6ic_hgnn_smoke/protocol.py` 读取高层 `322000–322047`、`322048–322063`、`322064–322079`；低层训练池仍是 `321000–321095` | `322048–322063` 已用于 P1–P5，只作开发诊断。reserved test 不参与本次和后续调参 |

`age_scale_steps=48` 在 Stage A 写进实验配置后即固定。

## 测试要求

本次没有新增可执行测试，也没有跑全量测试。Stage A 开始后，至少要有这些检查，并且原有队列回归仍通过：

- 每步桶内比特之和与 `q_bits` 的差不超过 1 bit，覆盖服务、到达、容量丢弃、snap、reset 和 `t=48`。
- 本步新到达不进入本步 `served`。丢弃只减少本步新桶。
- 空队列 `urgency_valid=0`。`t=0` 的 reset 存量年龄为 0，且 `age_origin=reset_convention`。
- 同一动作序列在日志开启和关闭时，最终 payload 与 `q_bits` 一致。
- 旧 Scheme C 与 `phase6g_seed0.pt` 仍按原宽度加载。新模块的输入宽度写在 checkpoint metadata。
- 特征函数不读取 `remaining_arrivals` 或未来 served bits。
- 测试场景来自 registry 的 train 或单独的单元夹具，不打开 `322064–322079`。

## 下一步

进入 Stage A 时，只在新目录实现 wrapper 账本和逐步日志，先完成上表的守恒和“日志不改变转移”测试。通过前不改奖励、不改 HGNN 输入、不训练。Stage A 未开始。
