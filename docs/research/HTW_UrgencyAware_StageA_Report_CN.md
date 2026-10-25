# Stage A：episode admission-age 账本与逐步诊断日志

执行指令是 `docs/research/HTW_UrgencyAware_StageA_Implementation_Instructions_CN.md`，提交 `cd383b0b7e42bd24e5d18968a475489d18953a70`。前置审计是 `docs/research/HTW_UrgencyAware_Training_Audit_CN.md`，Gate `URGENCY_STATE_AUDIT_COMPLETE`。本次只新增独立 wrapper、独立 runner、单元测试和一份新的结果目录。没有训练，没有改正式队列转移、FormalPlanner、HGNN、MAPPO、奖励、动作或 SafetyProjector，没有覆盖 checkpoint、旧数据集和 P1–P5 结果，没有读取 reserved test `322064–322079`。

## Gate

`URGENCY_LEDGER_TESTS_PASS`

`URGENCY_LOGGING_INVARIANT_PASS`

年龄桶在 reset、服务、到达、丢弃、snap、episode 结束和 restore 之后与 `q_bits` 的差都在 1 bit 容差内。短程 smoke 的最大账本差是 `1.862645149230957e-09` bit，最大环境守恒误差是 `3.725290298461914e-09` bit。同一场景在记录日志和不记录日志时，最终 payload、`q_bits`、assignment、位置和终止时刻一致；同一动作序列再跑一次，urgency 轨迹也一致。

## 1. 实际修改

新增，不改既有正式文件：

| 路径 | 作用 |
| --- | --- |
| `src/wt/benchmark_v2/phase6i_urgency_stage_a/ledger.py` | episode admission-age bit ledger |
| `src/wt/benchmark_v2/phase6i_urgency_stage_a/runner.py` | 独立逐步日志 runner 和短程 smoke |
| `src/wt/benchmark_v2/phase6i_urgency_stage_a/protocol.py` | schema 版本、年龄尺度 48、输出目录、smoke 场景 |
| `tests/test_urgency_stage_a.py` | 账本、restore、确定性和日志开关测试 |
| `results/phase6i_urgency_stage_a/` | 本次 smoke 的日志、schema 和 summary |
| 本报告 | Stage A 结果 |

Schema 版本是 `episode_admission_age_v1`。`age_scale_steps` 固定为 48，等于 `episode_horizon`，没有在 `322048–322063` 或 reserved test 上调整。

每个浮标的桶记录 `admission_step`、`age_origin` 和 `bits_remaining`。来源只有两种：`reset_convention` 和 `environment_arrival`。

Reset 时已经在队列里的比特全部放入 `admission_step=0`、`age_origin=reset_convention`。在 `t=0` 服务前，这些比特的年龄是 0。这是本 episode 内的记账约定，不是 episode 开始前的真实等待时间，也不是 deadline。

每步在 `env.step` 之前，年龄是 `max(0, env.t - admission_step)`。导出服务前的 `oldest_wait_age`、按比特加权的 `mean_wait_age`、`age_weighted_backlog = sum(age * bits_remaining)`、`urgency_valid` 和 `urgency`。`q_bits > 0` 时 `urgency_valid=1`，否则为 0。`urgency = clip(oldest_wait_age / 48, 0, 1)`；空队列的 urgency 数值是 0，靠 validity 和“非空但年龄为 0”区分。读取 view 不改桶。

`env.step` 返回后的更新顺序是：按最老入院桶扣掉本步实际 `served_bits`；本步新到达在服务之后入桶，不能被同一步服务；`dropped_bits_step` 只从本步新桶扣除；再用本步实际 snap 对齐。`info` 没有 `float_snap_bits`。snap 取的是同一次 step 已经写入的 `cumulative_snap_bits` 增量，不第二次调用 `arrivals()`，也不重算队列转移。每步和 episode 结束都检查 `|ledger_sum - q_bits| <= tol_bits`。容差是环境自己的 `env.tol_bits`，等于 1 bit。

正式 `shadow.capture_state` 没有改。账本自己的 `capture` / `restore` 保存 schema 版本、年龄尺度和每个桶的三个字段。分支后的两份账本可以各自扣减，不能从一个 `q_bits` 标量重建。

逐步日志包住既有 step，只复制数组。`record=False` 仍走同一条动作和账本更新，只是不保留行，用来对照日志开关。日志行按场景、时间步、UAV 或浮标索引展开，served bits 记在浮标行上，不把全队服务量平均到某一架 UAV。

每步 UAV 行记录服务前绝对位置、服务后位置、assignment、剩余 hold、`a_des`、SafetyProjector 输出、最终执行动作、`constraint_status`、投影范数和投影前后差。浮标行记录服务前 `q_bits`、年龄、urgency、validity，以及服务后的 served、generated、admitted、dropped 和 `q_bits_after`。UAV 行另记本步 `info["sinr"]`、`info["rate_bps"]` 和干扰矩阵。干扰用服务发生时的位置、关联和服务前队列重算：`active_links` 会跳过已经空掉的浮标，若在服务后直接重算，本步曾经承载服务的链路会消失。重算前后会把 `env.queue` 原样放回。Smoke 里 SINR 和速率的复算差都是 0。`t=0,15,30,45` 记录 `candidate_pool_hash`、`candidate_index` 和 Scheme C score；其他步这三列为空。分数来自这一次 `planner.scorer.infer` 的返回值，FormalPlanner 源码没有改。

冻结 checkpoint 的输入宽度没有变。Smoke 加载后 Actor 第一层仍是 34，`TASK_DIM` 仍是 8。单元测试检查 `expected_critic_dim(6, 12)=188`，并确认 `bit_queue_v2.py`、MAPPO worker、SafetyProjector 和 `hgnn_revision.py` 没有引用本阶段模块。旧浮标特征里名叫 `urgency` 的 `queue/max(queue)` 列没有改写。

## 2. 测试

`py -3 -m unittest tests.test_urgency_stage_a`：14 项，0 失败，约 0.185 秒。覆盖指令里的 12 类检查：

1. Reset 后桶之和等于 `q_bits`，存量年龄为 0，来源是 `reset_convention`。
2. 本步新到达不进入本步 served。真实环境在 `t=4` 给 6 号浮标到达 `1e6` bit，该步 served 为 0，新桶的 `admission_step` 是 4。
3. 合成桶按最老入院步扣减，扣减量等于给定 served。
4. 容量丢弃只减少本步新桶。
5. 正 snap 清掉与环境相同的残差，账本与 `q_bits` 对齐。
6. 空队列 `urgency_valid=0`；非空且年龄为 0 时 `urgency_valid=1`。
7. 未服务比特的年龄随 `env.t` 增加，view 不改 `bits_remaining`。
8. capture/restore 后两臂可以独立扣减。真实环境分支通过既有 `capture_state` / `restore_state` 复制队列，账本用自己的快照。
9. 日志开和关的最终 payload、`q_bits`、assignment、位置和终止时刻一致。
10. 同一场景和同一飞向动作重复运行，urgency 轨迹一致。
11. 新模块源码没有 `remaining_arrivals`，也没有把未来 served bits 写进特征。
12. Actor 34 维、任务向量 8 维、Critic 188 维保持不变。另有一项检查：服务前队列下重算的 SINR、速率与 `info` 一致，且 `env.queue` 被放回。

这些测试使用 registry 的 `TRAIN_POOL` 前两个场景中的第一个，动作为 `fly_toward` 加固定合法关联，不加载 Scheme C。Reserved 场景在 runner 入口被拒绝。

短程 smoke 在单元测试之后运行：`py -3 -m wt.benchmark_v2.phase6i_urgency_stage_a.runner`。场景 ID 来自 `smoke_scenario_ids()`，即 `TRAIN_POOL[:2]`，实际是 `322000` 和 `322001`，都属于 `train_322000_322047`。每个场景走满 48 步：第一次记录逐步日志，随后用不记录模式重复两次，分别检查日志开关和重复轨迹。高层是冻结 Scheme C，低层是冻结 MAPPO-V2 的确定性均值。没有启动长时间训练。

| 场景 | 终止 `t` | payload (bit) | 最大账本差 (bit) | 最大守恒误差 (bit) | 日志开关一致 | 重复一致 |
| --- | --- | --- | --- | --- | --- | --- |
| 322000 | 48 | 63785945.28984571 | 1.862645149230957e-09 | 3.259629011154175e-09 | 是 | 是 |
| 322001 | 48 | 68952073.58225587 | 1.3969838619232178e-09 | 3.725290298461914e-09 | 是 | 是 |

两次场景合计的最大 SINR 复算差是 0，最大速率复算差是 0。Actor 参数 token 在 smoke 前后不变。

日志文件 `results/phase6i_urgency_stage_a/step_rows.csv` 有 1728 行：2 个场景、48 步、每步 6 条 UAV 行和 12 条浮标行。`t=0` 的 24 条浮标行年龄都是 0；空队列 validity 为 0，非空队列 validity 为 1。8 条决策步的 UAV 0 行都有 candidate pool hash 和 Scheme C score，非决策步这些列为空。CSV 中最大 `|ledger_mismatch_bits|` 与 summary 相同。

Checkpoint 与配置哈希和冻结值一致：

- Scheme C `f77c0bcef3f359e01de35b75753f3776a9e8332bf65fd191f112236df074f864`
- MAPPO `92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e`
- candidate config `3535008ed197f28ed3620a25493a9cc17fbb1fca9742d49a3d58957883319a63`

Smoke 开始时的 `git rev-parse HEAD` 是 `052792c14b0a1ff65964bd277a369515f623b113`。结果提交在指令提交 `cd383b0` 之后，summary 里的这个字段保持 smoke 当时的 HEAD，没有事后改写。

仓库既有的完整测试套件这次没有整库重跑。本阶段只运行了上面的 14 项单元测试和这次 smoke。

## 3. 数据限制

- Reset 存量的年龄从 `t=0` 起算，来源标记为 `reset_convention`。它不能被读成 episode 开始前的真实等待。
- `info` 不提供逐浮标 `float_snap_bits`。账本使用 `cumulative_snap_bits` 的步增量。这个计数器由正式 step 在同一次转移里更新。
- 现有 HGNN 浮标特征里的 `urgency` 仍是 `queue/max(queue)`。本阶段的年龄写在新日志和新账本里，没有进入模型输入或奖励。
- Smoke 只有训练池开头的两个场景。它不代替 `322048–322063` 的验证，也没有使用 reserved test。
- 日志里的 Scheme C score 是该次选择已经算出的标量，不是用未来 payload 重新挑选的结果。
- 干扰矩阵对应服务发生时的位置、关联和服务前队列。服务后的空队列不能再拿来重建本步链路。
- 没有把 urgency 接入 HGNN 或 MAPPO，没有改奖励，没有训练。Stage B 没有开始。
