# Stage D3 报告：Payload-first + urgency tie-break 闭环试验

- 分支：`feature/maritime-wt-stage1a`
- 指令提交：`cb264320323bbc15b3763f53b0f664bc8421d994`
- 前置 Gate：`HIGHLEVEL_UTILITY_RULE_READY`，本阶段没有修改
- 本阶段 Gate：`URGENCY_TIEBREAK_CLOSEDLOOP_PILOT_INCONCLUSIVE`
- 性质：oracle-label 诊断。不是 learned policy，不替换 Scheme C
- 更大开发集评估：当前不值得进入

## 1. 直接结论

这个规则不值得进入更大规模开发集评估。它在每个决策上守住了 1 bit 的标签 payload 约束，但四条完整 48 步轨迹里有三条的实际 payload 明显下降，紧急服务也没有稳定方向。

规则是 oracle-label 上界式诊断，不是可部署策略。两个臂都读取冻结 MAPPO-V2 对候选做 rollout 之后的 payload 标签和 urgent-service 标签，再选分配。现有 HGNN 不能直接执行这个选择，结果也不能迁移到正式 Scheme C。`lambda_H` 和 `delta_payload_max` 仍然未定。本阶段没有训练 HGNN，没有生成正式标签或完整数据集，也没有修改 Planner、低层、奖励、动作、旧数据集或 checkpoint。

## 2. 规则与预算

场景是 registry 里 `highlevel_v2_train` 的前四个，并经过 `assert_allowed(split="train")`：322000–322003。没有读取 validation 或 reserved test。

每个决策状态只评估 128 个候选：32 个稳定高分候选，加 96 个按 rank 等频分层。种子 603，`SeedSequence([603, scenario_id, time_step])`。候选列表在该状态的 urgent rollout 之前写入 `selected_candidates.csv`。

两种规则使用同一份 128 个标签：

- Payload-first：标签 payload 最大；1 bit 内并列时取最小候选 ID，不看紧急服务。
- Payload-first + urgency tie-break：只在距最大标签 payload 不超过 1 bit 的候选里取紧急服务最大者；紧急服务再并列时取最小候选 ID。

实测选择时的标签 payload 差最大只有 7.45e-9 bit。没有任何一次选择离开 1 bit 并列组。

标签 horizon 与闭环 hold 分开：t=0、15、30 是 15 步，t=45 只剩 3 步。下面不把 3 步标签和 15 步标签放在同一个平均值里比较。闭环指标用的是完整 48 步的实际 payload，两臂可以比较。

t=0 的两臂状态相同，标签只计算一次再复用。t=15、30、45 两臂已经分叉，各自重建候选池和标签，不复用另一臂的未来状态。因此实际标签状态是 28 个，而不是把两臂后期状态也算进同一个 4×4×128。每个状态仍然只有 128 个候选。第一状态 9.801 秒，28 个状态的估计是 274.4 秒，实际标签时间 213.6 秒，低于单状态 900 秒和总估计 30 分钟的停止线。

## 3. 候选子集的来源

t=0 的状态哈希与冻结 shard 一致，8 个决策（两臂各 4 个场景）使用已有 J15 和 `utility_rank` 抽样。t=0 的 tie-break 紧急增益与 D2 的并列内增益一致：322000 为 188,875.10 bit，322001 为 240,421.39 bit，322002 为 460,694.72 bit，322003 为 145,768.18 bit，标签 payload 差都在 1 bit 内。

t=15、30、45 的闭环状态不再等于冻结 shard，也没有 t=45 的旧 J15。这些状态的 128 个名额改用当前观测上的 G1 加性速率做稳定排序和 dense rank 分层。规则本身仍然只看这 128 个候选的 MAPPO rollout 标签，不看速率分数。24 个决策属于这个来源。抽样前沿不是 13,327 全池前沿。

## 4. 闭环差值

主指标是 tie-break 臂的 48 步 payload 减去 payload-first 臂。四场景均值 -2,252,512.21 bit，中位数 -1,660,694.16 bit。紧急服务均值 -18,282.36 bit，中位数 +135,467.48 bit。样本只有 4 个场景，这里没有显著性或泛化结论。

| 场景 | payload 差 (bit) | 紧急服务差 (bit) | dropped 差 (bit) | 终态队列差 (bit) | 最老等待差 | 安全违规步差 | 投影触发差 | 平均 SINR 差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 322000 | -2,341,140.80 | +201,835.99 | +2,461,771.12 | -120,630.32 | 0 | +2 | +12 | -0.0648 |
| 322001 | -6,994,271.46 | -1,133,954.62 | +3,081,891.41 | +3,912,380.05 | +3 | -4 | -5 | -0.0565 |
| 322002 | -980,247.52 | +69,098.97 | -1,007,143.78 | +1,987,391.31 | -6 | -1 | -5 | -0.3472 |
| 322003 | +1,305,610.96 | +789,890.21 | -79,918.42 | -1,225,692.55 | +2 | +3 | -1 | -0.0104 |

tie-break 臂的 16 个决策里，8 个改变了分配，比例 0.5。改变全部发生在 t=0 和 t=30；四个场景的 t=15 和 t=45 都只有一个 payload 最大候选，两规则选出同一分配。t=30 的标签紧急增益分别是 212,796.07、773,274.18、61.58 和 340,032.66 bit，标签 payload 差仍不超过 1 bit。

标签并列没有保住完整 episode 的 payload。零到达的 15 步标签可以几乎相同，但真实到达和后续决策会把轨迹分开。322001 同时损失约 6.99e6 bit payload 和 1.13e6 bit 紧急服务。四个场景的 tie-break 臂平均 SINR 都更低。安全违规和投影触发没有稳定改善。

## 5. 计账、哈希与耗时

八条 48 步轨迹的状态都是 `ok`。episode 账本最大差 1.40e-9 bit，episode 守恒最大差 5.59e-9 bit。标签账本最大差 2.33e-9 bit，标签守恒最大差 8.06e-9 bit。逐步日志在 `step_rows.csv`，共 384 行。决策 32 行，候选子集 4,096 行。

MAPPO-V2 seed 0：`92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e`。Scheme C：`f77c0bcef3f359e01de35b75753f3776a9e8332bf65fd191f112236df074f864`。epoch 6：`7859f9c94940d9259ab1bece1f8dddf9655a4dcb1cd45d537ad127347526bcde`。adapter config：`3535008ed197f28ed3620a25493a9cc17fbb1fca9742d49a3d58957883319a63`。D0、D1、D2 结果文件的哈希在运行前后一致。选择过程没有调用 Scheme C。

必需字段没有缺失。t=45 没有旧 J15，这个缺口用当前 G1 加性速率只决定子集，并在 `sampling_source` 里标明。

## 6. Gate

机械检查、1 bit 约束、48 步完成和计账都通过，所以不是 `BLOCKED`。紧急服务的闭环差值一正一负，没有稳定方向，Gate 为 `URGENCY_TIEBREAK_CLOSEDLOOP_PILOT_INCONCLUSIVE`。

因此停止。不把该规则送进更大开发集，不训练 HGNN，不构建完整高层数据集，不修改正式系统。
