# Stage D1 报告：t=30 紧急度标签信号试验

- 分支：`feature/maritime-wt-stage1a`
- 指令提交：`fa32cbbb928add8ae3b4e1263325b22c780a0aae`
- 前置 Gate：`HIGHLEVEL_URGENCY_LABEL_PILOT_INCONCLUSIVE`
- 本阶段 Gate：`T30_URGENCY_SIGNAL_SUPPORTED`
- `lambda_H`：未定
- 建议：排序信号达到本次探索阈值，同时平均 top-1 payload regret 为负。下一步只设计 utility 权衡和 `lambda_H` 规则，不训练 HGNN，也不构建完整数据集。

## 1. 结论

registry 按 scenario ID 排序并排除 D0 的 322000–322003 之后，前 12 个训练场景是 322004–322015，与预期一致。每个场景只评估 t=30 的 128 个候选，共 1,536 个 15-step rollout。状态重放、候选合法性、payload 复现、账本守恒、重复种子和 checkpoint/config 哈希全部通过。

12 个状态的平均 Spearman 为 0.9447，低于 0.95。平均成对逆序比例为 0.0673，高于 0.05。其中 10/12 个状态的逆序比例高于 0.05。三项探索性条件同时成立，Gate 为 `T30_URGENCY_SIGNAL_SUPPORTED`。

这仍是 t=30 的探索性结果。它不把 D0 的总体 Gate 改写成确认性显著，也不表示完整高层策略已经改进。

平均 top-1 payload regret 为 -133,418.33 bit，低于预注册的 -100,000 bit。因此只能说明存在 payload 权衡，不能宣称紧急度标签的第一名带来更高的高层效用。这个平均值几乎全部来自 322015：该状态的 regret 为 -1,601,019.99 bit。其余 11 个状态的 regret 绝对值都小于 1e-8 bit，紧急服务第一名仍落在 payload 最高的计账平局里。

## 2. 实际规模

场景由 `phase6ib_data_registry.json` 的 `highlevel_v2_train` 经 `assert_allowed(split="train")` 得到，并排除 `pilot_scenarios()` 返回的 D0 四个场景。未读取 validation 322048–322063 和 reserved test 322064–322079。

每个状态的冻结池有 13,327 个 G1_RATE k=6 候选。子集是 32 个现有 J15 稳定 top，加 96 个按 `utility_rank` 等频分层的候选。种子 602，`SeedSequence([602, scenario_id, 30])`。12 个状态的补齐数都是 0。候选列表在任何紧急服务 rollout 之前写入 `selected_candidates.csv`。

评估 rollout 为 12 × 128 = 1,536。第一状态又按同一批候选完整重复了 128 次，标签差为 0 bit。执行器是冻结 MAPPO-V2 seed 0。候选副本继续把 `arrivals` / `arrivals_at` 置零。重放路径保留真实到达，并沿用 Stage A 的 `episode_admission_age_v1`。

state/candidate npz 仍从 `aadbb3c5c5954669dd335dce9afb3462d7de114c` 的 git blob 只读读取，没有写回旧 dataset。D0 结果目录的文件哈希在运行前后一致。

## 3. 校验与耗时

| 检查 | 结果 |
| --- | --- |
| t=0、15、30 的 state hash 与 q_bits | 12 个场景全部对齐，q_bits 最大绝对差 0 |
| 候选池顺序与合法性 | 与 G1_RATE k=6 排序键一致，非法候选 0 |
| payload 相对已有 `j15_bits` | 1,536 行最大绝对差 0 bit |
| 标签账本 | 最大差 2.328e-9 bit |
| 标签环境守恒 | 最大差 5.588e-9 bit |
| 重放账本 | 最大差 1.397e-9 bit |
| 第一状态重复 | 128 个候选，payload 与紧急服务差 0 bit |
| checkpoint / adapter config | 与冻结常量一致，运行前后不变 |

MAPPO-V2 seed 0：`92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e`。Scheme C：`f77c0bcef3f359e01de35b75753f3776a9e8332bf65fd191f112236df074f864`。epoch 6：`7859f9c94940d9259ab1bece1f8dddf9655a4dcb1cd45d537ad127347526bcde`。adapter config：`3535008ed197f28ed3620a25493a9cc17fbb1fca9742d49a3d58957883319a63`。

第一状态 322004 用时 8.352 秒，12 个状态估计 100.225 秒，低于单状态 900 秒和总估计 30 分钟的停止线。标签计算共 115.382 秒，重放合计 10.181 秒，总墙钟时间 139.765 秒。

t=30 时浮标 0–5 的最老等待年龄都是 0，浮标 6–11 的年龄在 0 到 26 之间，场景之间不同。年龄 0 表示该浮标在决策时刻没有剩余比特。

## 4. 尺度与排序

分布描述的是每个状态抽中的 128 个候选。`payload_term` 和 `urgent_service_term` 的单位是 1e6 bit。regret 的单位是 bit，符号是紧急服务第一名的 payload 减去 payload 第一名的 payload。

| 场景 | Spearman | 逆序比例 | urgent/payload | top-1 regret (bit) |
| --- | ---: | ---: | ---: | ---: |
| 322004 | 0.9585 | 0.0519 | 0.3415 | 0 |
| 322005 | 0.9439 | 0.0713 | 0.3452 | 0 |
| 322006 | 0.9325 | 0.0748 | 0.3500 | 0 |
| 322007 | 0.9483 | 0.0656 | 0.3170 | 0 |
| 322008 | 0.9343 | 0.0901 | 0.3486 | 0 |
| 322009 | 0.9444 | 0.0671 | 0.3136 | 0 |
| 322010 | 0.9306 | 0.0862 | 0.3135 | 0 |
| 322011 | 0.9453 | 0.0697 | 0.3661 | 0 |
| 322012 | 0.9442 | 0.0707 | 0.3347 | 0 |
| 322013 | 0.9557 | 0.0456 | 0.3496 | 0 |
| 322014 | 0.9570 | 0.0455 | 0.3400 | 0 |
| 322015 | 0.9414 | 0.0690 | 0.3118 | -1,601,019.99 |
| 平均 | 0.9447 | 0.0673 | 0.3360 | -133,418.33 |

12/12 个状态的紧急服务第一名与 payload 第一名不是同一个候选 ID。除 322015 外，两者的 payload 差都在 1 bit 以内。322013 和 322014 的逆序比例为 0.0456 和 0.0455，没有超过 0.05。最低 Spearman 为 0.9306。

紧急项与 payload 项的比值集中在 0.31 到 0.37。这个 t=30 子集的尺度比 D0 的 t=0（约 0.082）更稳定，但仍不足以在本阶段指定 `lambda_H`。正式训练标签没有生成。

## 5. 缺失字段与限制

1,536 行都有 payload、紧急服务、逐浮标 served bits、决策时刻最老年龄和计账误差。`lambda_H` 是故意留空。完整 13,327 候选池没有计算紧急服务标签。样本只覆盖 12 个训练场景的 t=30，不能外推到 t=0、t=15 或完整系统。

## 6. 建议

当前停止高层紧急度标签扩展，不构建完整数据集，不训练 HGNN，正式 Planner 和低层 MAPPO 保持冻结。

排序信号在本次预注册规则下成立，但 322015 的 top-1 少了约 1.60 Mbit payload，把 12 个状态的平均 regret 拉到 -133,418 bit。下一步只适合先写 utility 权衡和 `lambda_H` 的选择规则。那条规则写定之前，不提出完整高层数据集或 HGNN 训练计划。

`tests/test_urgency_highlevel_pilot.py` 与 `tests/test_urgency_highlevel_t30.py` 共 10 项通过。仓库既有的完整测试套件本次没有重跑。
