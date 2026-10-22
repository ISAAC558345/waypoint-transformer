# HTW Phase-6I-D：JointUtilityHGNN 受控训练

## 结论

Gate 为 **JOINT_UTILITY_TRAINING_INCONCLUSIVE**。

10 个 epoch 都已完成，验证对比也已写完。按预先规定的选择规则，最佳 checkpoint 是 epoch 6。相对冻结的 Phase-5L 初值，验证集 Spearman 从 0.3507 升到 0.9465，Kendall 从 0.2538 升到 0.8323，normalized regret 从 0.2411 降到 0.0507，预测最优候选的平均 J15 从 14,372,008 bit 升到 18,892,120 bit。Top-1 从 0.6042 降到 0.5625，Top-3 从 0.6458 降到 0.6250。48 个验证状态里只有 22 个的预测 J15 高于初值。因此不能记成明确改善。

训练在 CPU 上完成。没有 NaN 或 Inf，score collapse 比例全程为 0。原始 `joint_utility_encoder.pt` 与 `matching_decoder.pt` 的 SHA-256 在训练后保持不变。322064–322079 没有读取。没有接入 planner，没有修改 MAPPO，`planner_ready = False`，`deployment_verified = False`。

## 训练前确认的配置

入口是 `src/wt/benchmark_v2/phase6ic_hgnn_smoke/` 的 Dataset。新训练代码在 `src/wt/benchmark_v2/phase6id_joint_training/`。输出目录是 `phase6id_joint_utility_training/`。

| 项目 | 取值 |
| --- | --- |
| 初始化 | `phase5l_joint_encoder/joint_utility_encoder.pt` |
| 打分头 | `phase5k_joint_decoder/matching_decoder.pt`，33,025 个参数，`requires_grad = False` |
| 可训练参数 | HGNN 696,320 个 |
| 优化器 | Adam，学习率 `3e-4`，权重衰减 `1e-4` |
| 梯度 | 裁剪阈值 1.0；记录的是裁剪前的总范数 |
| loss | `mean(-logsigmoid(s_i - s_j))`，只在 `j15_i > j15_j` 时计算 |
| 设备 | CPU |

`j15_bits` 是监督收益，与 `assignment` 逐行对应。`utility_rank` 只用于 Top-k 命中和审计，不进入 graph batch。完全相等的 `j15_bits` 不产生 loss 配对。Top-1、Top-3、Top-5 把 `utility_rank == 1` 的 1 bit 最优组视为命中。normalized regret 是 `(max j15 - 所选 j15) / (max j15 + 1)`。score collapse 指一个状态的分数标准差或极差小于 `1e-6`。

最佳 checkpoint 的排序键是：验证 Spearman，然后 Top-1，然后更低的 normalized regret，最后更低的验证 loss。明确改善还要求 Spearman 上升、Top-1 不低于初值、regret 不升高，并且平均 J15 上升。

第一个完整 epoch 用了 642 s，其中训练 543 s、验证 99 s。这低于 smoke 外推的约 807 s 训练步，因此继续跑完其余 epoch。

## 每个 epoch

验证集是 322048–322063 的 48 个状态。collapse 比例都是 0。

| epoch | train loss | val loss | Spearman | Kendall | Top-1 | Top-3 | Top-5 | regret | 平均 J15 bit | 梯度均值 / 最大 | 秒 | 工作集 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.2138 | 0.1468 | 0.9363 | 0.8043 | 0.6042 | 0.6458 | 0.6875 | 0.0645 | 18,544,498 | 0.117 / 5.541 | 642 | 5.50 GiB |
| 2 | 0.1387 | 0.1351 | 0.9351 | 0.8013 | 0.6250 | 0.6667 | 0.6875 | 0.0594 | 18,669,022 | 0.097 / 0.875 | 719 | 5.51 GiB |
| 3 | 0.1307 | 0.1250 | 0.9405 | 0.8145 | 0.6458 | 0.6875 | 0.7083 | 0.0562 | 18,746,079 | 0.123 / 0.860 | 761 | 5.51 GiB |
| 4 | 0.1255 | 0.1229 | 0.9425 | 0.8206 | 0.6250 | 0.6667 | 0.6875 | 0.0569 | 18,724,802 | 0.144 / 1.341 | 747 | 5.51 GiB |
| 5 | 0.1235 | 0.1218 | 0.9435 | 0.8237 | 0.6042 | 0.6250 | 0.6667 | 0.0522 | 18,840,858 | 0.189 / 1.562 | 738 | 5.50 GiB |
| 6 | 0.1220 | 0.1166 | 0.9465 | 0.8323 | 0.5625 | 0.6250 | 0.6667 | 0.0507 | 18,892,120 | 0.159 / 2.517 | 738 | 5.51 GiB |
| 7 | 0.1199 | 0.1174 | 0.9460 | 0.8313 | 0.5833 | 0.6042 | 0.6042 | 0.0394 | 19,154,296 | 0.221 / 2.455 | 682 | 5.51 GiB |
| 8 | 0.1194 | 0.1168 | 0.9463 | 0.8324 | 0.5625 | 0.6250 | 0.6458 | 0.0522 | 18,850,999 | 0.255 / 2.206 | 755 | 5.51 GiB |
| 9 | 0.1187 | 0.1178 | 0.9445 | 0.8268 | 0.5833 | 0.6667 | 0.6875 | 0.0470 | 18,966,043 | 0.208 / 1.924 | 740 | 5.52 GiB |
| 10 | 0.1184 | 0.1162 | 0.9453 | 0.8289 | 0.5625 | 0.6250 | 0.6667 | 0.0480 | 18,948,107 | 0.191 / 2.245 | 734 | 5.52 GiB |

10 轮训练加验证合计 7,257 s。epoch 7 到 epoch 10 都没有超过 epoch 6 的选择键，patience 为 4，因此停在第 10 轮，没有使用最后一个 epoch。每个 epoch 都有独立文件 `epoch_001.pt` 至 `epoch_010.pt`。`best.pt` 是 epoch 6 的权重。

## 同一验证集上的对比

三个对象使用同一批候选和同一套 `j15_bits`。oracle 是每个状态里 J15 最大的候选，只作为上界。

| 指标 | 冻结初值 | epoch 6 | oracle |
| --- | ---: | ---: | ---: |
| Spearman | 0.3507 | 0.9465 | 标签自身 |
| Kendall | 0.2538 | 0.8323 | 标签自身 |
| Top-1 | 0.6042 | 0.5625 | 1 |
| Top-3 | 0.6458 | 0.6250 | 1 |
| Top-5 | 0.6458 | 0.6667 | 1 |
| normalized regret | 0.2411 | 0.0507 | 0 |
| 预测最优的平均 J15 | 14,372,008 bit | 18,892,120 bit | 20,071,099 bit |

epoch 6 相对冻结初值的平均 J15 差是 +4,520,112 bit。48 个状态中 22 个为正。oracle 比 epoch 6 仍高 1,178,979 bit。

epoch 3 的 Top-1 是 0.6458，高于冻结初值，Spearman 是 0.9405。选择规则把 Spearman 放在 Top-1 之前，所以最佳文件仍是 epoch 6。

## 尚未解决的问题

相关性、regret 和平均 J15 同时改善，Top-1 与 Top-3 没有改善，状态层面的 J15 提升也不是多数。这个结果还不能当作可部署的高层策略。reserved test、planner 和 MAPPO 都保持原状。下一步需要单独指令。
