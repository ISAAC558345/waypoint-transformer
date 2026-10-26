# Stage D0 报告：高层紧急度标签可行性试验

- 分支：`feature/maritime-wt-stage1a`
- 指令提交：`a83869d5155c485c8cc752a43d41d246130e784c`
- Gate：`HIGHLEVEL_URGENCY_LABEL_PILOT_INCONCLUSIVE`
- `lambda_H`：未定，本阶段没有生成正式训练标签
- 建议：先不构建完整高层紧急度数据集，也不训练 HGNN

## 1. 结论

4 个训练场景、3 个决策时刻、每状态 128 个候选全部完成。状态重放、候选合法性、payload 复现、年龄账本守恒、重复种子和 checkpoint/config 哈希都通过。

排序信号按运行前写定的规则判断：12 个状态的平均 Spearman 为 0.9507，平均成对逆序比例为 0.0431。有用信号要求平均 Spearman 低于 0.95，或平均逆序比例高于 0.05。两项都没有达到，因此 Gate 为 `HIGHLEVEL_URGENCY_LABEL_PILOT_INCONCLUSIVE`。

紧急服务第一名在 10/12 个状态上与 payload 第一名的 payload 相同；另外两个状态少 67,008.16 bit 和 99,002.36 bit。紧急项相对 payload 项的平均比值在 t=0 约为 0.082，在 t=15 和 t=30 约为 0.33 和 0.32。这个尺度随决策时刻变化，不足以锁定一个固定的 `lambda_H`。

## 2. 实际规模

场景来自 `phase6ib_data_registry.json` 的 `highlevel_v2_train` 前 4 项，并与 `TRAIN_POOL` 前缀核对：322000、322001、322002、322003。时间点为 t=0、15、30。未读取 validation 322048–322063 和 reserved test 322064–322079。

每个状态的冻结候选池有 13,327 个 G1_RATE k=6 候选。评估子集固定为 32 个现有 J15 稳定 top，加其余候选按 `utility_rank` 等频分位抽出的 96 个。抽样种子为 601，使用 `SeedSequence([601, scenario_id, time_step])`，在任何紧急服务 rollout 之前确定。12 个状态的补齐数都是 0，短分层未出现。

评估 rollout 为 4 × 3 × 128 = 1,536 个 15-step 冻结 MAPPO-V2 seed 0 rollout。确定性复查另重复了第一状态的 128 个候选，两次标签差为 0 bit。复查使用同一批候选，没有增加场景或候选上限。

执行器是冻结的 `phase6g_seed0.pt`。标签沿用现有 J15 约定：只在候选副本上把 `arrivals` 和 `arrivals_at` 置零。重放状态本身保留真实到达。年龄账本是 Stage A 的 `episode_admission_age_v1`，`age_scale_steps=48`，reset 存量为 `reset_convention`。

工作区里的 `dataset/train/scenario_32200x` 只保留 manifest。state/candidate npz 从删除这些 dump 之前的提交 `aadbb3c5c5954669dd335dce9afb3462d7de114c` 以 git blob 只读取出，并与 manifest 的 SHA-256 核对。运行结束后工作区 dataset 文件哈希未变，npz 没有写回旧数据集。

## 3. 校验

| 检查 | 结果 |
| --- | --- |
| 12 个状态的 state hash、state arrays、q_bits | 对齐，q_bits 最大绝对差 0 |
| 候选池顺序 | 与现场重建的 G1_RATE k=6 排序键一致，非法候选 0 |
| payload 相对已有 `j15_bits` | 1,536 行最大绝对差 0 bit |
| 标签 rollout 账本与 q_bits | 最大差 1.630e-9 bit |
| 标签 rollout 环境守恒 | 最大差 5.588e-9 bit |
| 重放账本 | 四个场景最大差 9.313e-10 bit |
| ledger capture/restore | 12 个决策点均可独立恢复，且不改变环境快照 |
| 同一 seed 的样本与第一状态标签 | 样本序列一致，标签差 0 bit |
| MAPPO / Scheme C / epoch 6 / adapter config | 与冻结常量一致，运行前后不变 |

容差为环境计账容差 1 bit。MAPPO-V2 seed 0 的 SHA-256 为 `92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e`。Scheme C 为 `f77c0bcef3f359e01de35b75753f3776a9e8332bf65fd191f112236df074f864`。epoch 6 为 `7859f9c94940d9259ab1bece1f8dddf9655a4dcb1cd45d537ad127347526bcde`。adapter config 为 `3535008ed197f28ed3620a25493a9cc17fbb1fca9742d49a3d58957883319a63`。

公式仍是 `payload_term = payload_increment_bits / 1e6`，`urgent_service_term = sum(urgency_before_service * served_bits) / 1e6`。`utility(lambda_H)` 只写在 schema 中，没有赋具体权重。

t=0 时 12 个浮标的最老等待年龄都是 0。t=15 时浮标 0–5 的最老年龄为 0，浮标 6–11 为 11。t=30 时浮标 0–5 仍为 0，浮标 6–11 的最老年龄在 2 到 24 之间，场景之间不同。年龄 0 表示该浮标在决策时刻已经没有剩余比特；reset 存量在进入 t=15 之前已经被服务完。

## 4. 尺度与排序

下面的分布描述的是每个状态抽中的 128 个候选，不是全部 13,327 个候选的总体均值。`payload_term` 和 `urgent_service_term` 的单位都是 1e6 bit。

| 时刻 | 平均 payload_term | 平均 urgent_service_term | 平均比值 | 平均 Spearman | 平均逆序比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| t=0 | 11.57 | 0.955 | 0.0825 | 0.9714 | 0.0052 |
| t=15 | 17.87 | 5.962 | 0.3336 | 0.9529 | 0.0289 |
| t=30 | 15.74 | 5.079 | 0.3222 | 0.9277 | 0.0953 |
| 12 状态平均 |  |  |  | 0.9507 | 0.0431 |

逐状态的逆序比例在 t=30 为 0.0687、0.1075、0.1172、0.0879，Spearman 最低为 0.9091。t=0 的逆序比例最高只有 0.0153。成对比较只统计 payload 相差超过 1 bit 的候选对；这些对里没有紧急服务标签也落在 1 bit 以内的平局。

11/12 个状态的紧急服务第一名与 payload 第一名不是同一个候选 ID。对应的 payload 损失只出现在 322000 t=15（99,002.36 bit）和 322002 t=15（67,008.16 bit）。其余 10 个状态的紧急服务第一名同时取得该 128 人子集里的最高 payload。

因此，t=30 的中段排序确实会变化，但子集顶部的 payload 选择基本保留。整体相关性仍高于预注册的有用信号线。

## 5. 耗时

第一状态（322000，t=0，128 个候选）用时 9.505 秒。按 12 个状态估计为 114.061 秒，低于预注册的单状态 900 秒上限，所以继续完成其余状态。12 个状态的标签计算共 124.629 秒。四个场景的状态重放合计约 3.44 秒。包含候选池核对和 128 次重复 rollout 的总墙钟时间为 143.149 秒。

## 6. 缺失字段

1,536 行标签都有 payload、紧急服务、逐浮标 served bits、决策时刻的最老年龄、payload 复现误差和账本误差。没有因为缺 shard 或中途失败而留下空标签。`lambda_H` 是故意留空，不是漏算。完整候选池的紧急服务标签没有计算；本阶段的上限就是每状态 128 个候选。

## 7. 建议

当前不值得继续构建完整高层紧急度数据集。通过的是重放和计账，整体排序仍然紧贴现有 J15。t=30 的逆序更高，但也只覆盖 4 个场景的分层子集，而且不能单独改写预注册的平均阈值。下一步若要重新考察 t=30，需要新的指令；本阶段到此停止，不训练高层，不改低层 MAPPO、正式 Planner、奖励、旧数据集或 checkpoint。

单元测试 `tests/test_urgency_highlevel_pilot.py` 为 6 项通过。仓库既有的完整测试套件本次没有重跑。
