# P0 时间尺度、GAE 与回报语义审计

- 依据：`docs/research/NEXT_PHASE_LOWLEVEL_CLOSED_LOOP_OPTIMIZATION_CN.md` 第 3 节
- 范围：只审计现有调用链，不训练，不改正式权重，不改 FormalPlanner
- 复现测试：`tests/test_nextstep_temporal_credit_audit.py`，4 项通过

## Gate

| 对象 | Gate | 含义 |
| --- | --- | --- |
| 本阶段 | `TEMPORAL_CREDIT_AUDIT_COMPLETE` | 调用链、终止条件和回报公式已经核对 |
| `phase6g` 正式低层训练 | `TEMPORAL_CREDIT_IMPLEMENTATION_VALID` | 15/30/45 不是环境终止，GAE 跨过这些边界 |
| `phase6ir` 记录窗口训练 | 窗口级目标与代码一致 | 窗口结束被写成终止，下一步是另一条恢复记录，回报不跨窗口 |

不使用 `TEMPORAL_CREDIT_BUG_CONFIRMED`。`phase6ir` 的 `done=True` 不是把同一条 episode 的下一步误截断；窗口结束后代码立刻 `reset` 并恢复另一条记录。若不把该步标成终止，GAE 会把这条无关记录的价值接进当前回报。

不使用 `TEMPORAL_CREDIT_AUDIT_BLOCKED`。

若后续目标改成同一条 48 步 episode 内跨高层边界的长期回报，不能沿用 `phase6ir` 的终止条件。正式训练路径 `phase6g` 已经是这种语义。

## 实际代码位置

| 环节 | 位置 |
| --- | --- |
| 正式低层一步 | `src/wt/benchmark_v2/phase6g_lowlevel_mappo/worker.py` `TrainWorker.step` |
| 正式 rollout 与 bootstrap | `src/wt/benchmark_v2/phase6g_lowlevel_mappo/train.py` `_collect`、`build_last_value` |
| 方案 C 记录窗口一步 | `src/wt/benchmark_v2/phase6ir_scheme_c_align/worker.py` `SchemeCTrainWorker.step` |
| 方案 C rollout 与 bootstrap | `src/wt/benchmark_v2/phase6ir_scheme_c_align/train.py` `_collect` |
| 状态恢复 | `src/wt/benchmark_v2/phase6d/shadow.py` `capture_state`、`restore_state`；`phase6ir` 的 `_restore` 另外写回 `initial_q_bits` |
| 奖励 | `worker.py` `low_level_reward` |
| GAE | `src/wt/round30/ppo.py` `compute_gae` |
| MAPPO 更新 | `src/wt/lowlevel_v2/phase1b/mappo.py` `ICARMAPPOLocked.update` |
| PPO 常数 | `src/wt/htw_phase101/protocol.py`：`rollout_length=48`，`gamma=0.99`，`gae_lambda=0.95` |

两条训练路径都把 `dones` 交给同一个 `compute_gae`。`nonterminal = 1 - done`。`done=1` 的那一步保留自己的奖励，不把下一步的价值或奖励加进来。critic target 是 `advantage + value`。

## 终止语义

环境自身只在 `t >= episode_horizon`（48）时把 `info["terminated"]` 设为真。训练器另外定义了自己的 `done`。

| 边界 | `phase6g` `TrainWorker` | `phase6ir` `SchemeCTrainWorker` |
| --- | --- | --- |
| 环境走到 48 | `done=True`，随后 `reset()`，bootstrap 为 0 | `done=True`，随后加载下一条记录，收集器 bootstrap 为 0 |
| SafetyProjector 失败或投影非有限 | `done=True` 并 `reset()`。这比环境的 horizon 终止更早 | 同样 `done=True`，并离开当前窗口 |
| 高层边界 t=15、30、45 | `done` 仍为 False。同一环境继续，下一步才安装新 TaskOption | 窗口步数耗尽时 `done=True`。t=15/30 也会终止，尽管 episode 还没结束 |
| rollout 在中途截断 | 仅当最后一步 `done=False` 且 option 还在时，用当前 critic 做 bootstrap | 仅当最后一步 `done=False` 时调用 `bootstrap_value()` |
| 窗口步数已经耗尽但还没加载下一条 | 不适用 | `bootstrap_value()` 在 `steps_left<=0` 时返回 0 |

`phase6g` 在 `pending` 减到 0 时把 `remaining_hold` 写成 0，但这一步不终止。下一次 `step` 开头才采样新 assignment，并把 `pending` 设回 15。因此边界上的回报使用的是边界之后、同一环境上的价值。

`phase6ir` 每步把 `remaining_hold` 减 1。窗口长度来自记录里的 `window_steps`：前三个窗口 15 步，最后一窗 3 步。最后一窗的 hold 计数仍从 15 往下减，不改成 3。这只影响观测里的剩余 hold，不决定 `done`。`done` 由 `steps_left`、失败和 horizon 决定。

标准 `phase6g` 调度里，rollout 长度和 episode 长度都是 48。从 t=0 对齐时，最后一步是环境终止，`last_values` 为 0。hold 结束恰好落在 rollout 最后一格、从而用“已经过期的 option”做 bootstrap 的情况，不是这条对齐调度上的常规路径。

## 恢复、到达和奖励

`phase6g` 不从快照恢复。失败或走到 48 时直接 `reset(seed=scenario)`，队列、位置和 payload 回到该场景的初始状态。

`phase6ir` 每次窗口先 `reset(seed=scenario_id)`，再 `restore_state`。恢复字段包括位置、队列、`q_bits`、服务余量、累计服务、累计丢弃、累计到达、SINR、速率、`t` 和 `payload_bits`。`initial_q_bits` 单独写回。TaskOption 用记录中的 assignment 重建，`association` 与 `targets` 相同，`remaining_hold` 从 15 开始。`last_movement` 用决策前保存的动作，不用窗口跑完后的动作。

`restore_state` 不替换 `arrivals_at`。到达仍是 `L3BEnv.arrivals_at(t)`：t 为 4 到 36 的偶数时，6 号及以后的浮标各到达 1 个 block。恢复 `t` 之后，窗口内到达与原环境在同一时刻的到达一致。`phase6d.shadow._zero_arrivals` 只服务于诊断用的 macro return，不在这条训练恢复路径上。

奖励是当前步的 `sum(info["served_bits"]) / 1e6`，再减去 `0.05 * c_project` 和 `0.01 * c_residual`。它不读取累计 `payload_bits`，也不把 `bits15` 或下一窗口的服务当成这一步的奖励。`phase6ir` 另外检查 `payload_bits` 的增量等于这一步的 `served_bits`，误差超过环境容差就停止。数据集里的 `bits15` 是采集时的窗口合计，训练循环使用的是逐步 `reward`。

48 步复现里，逐步 `served_bits` 之和等于该 episode 结束时记录的 `payload`。边界步 t=14、29、44 的 `done` 都是 False；第 48 步 `done` 为 True。

## 复现测试

`py -3 -m unittest tests.test_nextstep_temporal_credit_audit`

| 测试 | 结果 |
| --- | --- |
| `done=1` 时 GAE 不接收下一步奖励；`done=0` 时接收 | 通过 |
| `r_comm * 1e6` 等于当前步 served bits | 通过 |
| `phase6g` 跑满 48 步，15/30/45 不终止，服务之和等于 payload | 通过 |
| `phase6ir` 两步窗口的第二步终止，并装入下一条记录；收集器 bootstrap 为 0 | 通过 |

测试用固定全零动作和固定 assignment `0..5`，不读取保留测试，不写 checkpoint。

## 结论

正式低层训练把高层决策边界当成任务切换，不把它当成 episode 结束。回报可以穿过 15、30、45。真正截断回报的是 horizon 和 SafetyProjector 失败。

方案 C 对齐训练是另一套目标：一条记录窗口结束就终止，下一条样本来自独立快照。这个截断与它自己的数据组织一致，不能解释成正式 `phase6g` 训练已经切断了跨窗口回报。

P0 之后不启动长时间训练。P1 仍只做执行归因。
