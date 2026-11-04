# 到达感知高层方法的公平 A/B 协议

本协议登记后续对照的数据和预算。本包的训练更新、环境步和模型前向都是 0。协议写完之后仍然不开始训练，不重开旧 holdout，也不扫描标签权重。

## 已经知道和仍然未知

D6 已经回答标签先知问题：在 t=15 与 t=30，零到达标签和真实到达标签的冠军排序并不总是相同。这是离线标签重排，不是模型结果。当前信息模型能不能从现有观测里学到这一改进，仍然未知。本协议只准备后一个问题。

## 唯一变量

这个对照只改变纯 payload 监督来源：同一候选、同一快照上的 `zero_payload_bits`，或 `real_payload_bits`。候选集合、图、样本顺序、初始化、结构、优化器和 `top_regret_loss` 都相同。

`lambda_H` 与 `delta_payload_max` 属于另一项紧迫性权衡。二者继续空着，不阻挡这次纯 payload 准备，也不在本协议里取值。E19 的 0.001 标签项不进入对照，也不增大。

## 当前信息输入

选择时只使用决策时刻已经存在的状态。不读取未来到达、未来 payload、标签名次、scenario_id 特征或事后最优索引。真实到达只进入离线标签。身份字段放在配对表，不放进图张量。两臂共用同一张图和同一候选顺序。

## 已有数据状态

D5 的场景是 322000–322003，只检查 t=30。结论是 `ARRIVAL_AWARE_URGENCY_LABEL_COMPLETE`：零到达 J15 可以复现，真实到达改变 payload 排序。`lambda_H` 与 `delta_payload_max` 仍未定。

D6 的场景是 322016–322023，时刻是 t=0、15、30、45。结论是 `ARRIVAL_AWARE_LABEL_EXPANSION_INCONCLUSIVE`。t=0 与 t=45 的两臂标签相同；t=15 与 t=30 达到预注册的 material 标准。四个时刻没有收成一个训练目标。本准备只使用 t=15 与 t=30。

两组都沿 `MAX_RATE_INDEPENDENT` 重放和冻结 MAPPO-V2 seed 0，不是 Scheme C 轨迹。两组都没有读取 validation 或 reserved test。

L2 配对把上述 16 个状态固定分成两组。322016–322021 的两个时刻一起进入 train，共 12 个状态、1536 个候选。322022–322023 的两个时刻一起进入 explore-eval，共 4 个状态、512 个候选。这些场景都在既有训练池内，并且已经被到达标签诊断打开。explore-eval 不是新的独立验证，也不是正式 holdout。

## 以后若获准的训练草稿

草稿使用 `src/wt/benchmark_v2/phase6if_ranking_loss/train.py` 里已经存在的 Scheme C 契约，不更换损失。

- 初始化文件是 `results/phase6id_joint_utility_training/epoch_006.pt`。本包只计算文件哈希，不加载权重。该文件存在，SHA-256 是 `7859f9c94940d9259ab1bece1f8dddf9655a4dcb1cd45d537ad127347526bcde`。`run_scheme` 要求其中 `epoch == 6`，然后把 `payload["model"]` 载入编码器。`_build` 会先读取 `results/phase5l_joint_encoder/joint_utility_encoder.pt` 的 `state_dict`，再读取 `results/phase5k_joint_decoder/matching_decoder.pt`。解码器文件存在，SHA-256 是 `de1809fab8290b5be3e89cd0d31e22cd04638fd9a43207957aa20b1de255eca5`，并保持 `eval` 且 `requires_grad` 为假。编码器基础文件在当前工作区不存在，所以既有 `_build` 现在不能启动。真正替换编码器的是 epoch 6 的 `model`。本包不补造缺失的编码器文件。
- 优化器是 `torch.optim.Adam(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4)`。学习率和权重衰减来自 `phase6id_joint_training/protocol.py` 的 `LEARNING_RATE` 与 `WEIGHT_DECAY`，`phase6if` 原样导入。梯度裁剪是 `clip_grad_norm_(model.parameters(), 1.0)`。
- 损失是 `LOSSES["c"]`，即 `top_regret_loss`。标签先脱离计算图，按 `(label - min) / max(max - min, 1e-6)` 缩放到 `[0, 1]`，目标分布是 `softmax(30 * (quality - max(quality)))`，`LISTNET_BETA = 30`。损失是该目标与 `log_softmax(scores)` 的交叉熵。不改用 `gap_weighted_loss` 或 `tolerance_margin_loss`。
- 既有随机种子常量是 `SEED = 0`，在建模型前调用 `torch.manual_seed` 和 `np.random.seed`。数据顺序是 `DataLoader(..., batch_size=1, shuffle=False)`，每个状态一次更新。
- 既有命令是 `py -3 -m wt.benchmark_v2.phase6if_ranking_loss.train --scheme c --epochs 1`。它不能直接完成本草稿，差异如下。

既有接口的具体差异：

1. `DecisionDataset("train")` 读取完整修订训练集和 `j15_bits`，不会读取 L2 的 16 个状态及 `zero_payload_bits` / `real_payload_bits`。
2. 每个 epoch 后 `_evaluate` 按 payload、top1、regret 选择 `best_epoch`。草稿固定采用第 10 个 epoch，不在 explore-eval 上挑 checkpoint。
3. 文件里的种子只有 0。草稿预先固定三个配对种子，顺序为 0、1、2。每一对的两臂共享同一次初始化、同一候选顺序和同一个种子；只有标签列不同。
4. 输出目录是 `results/phase6if_ranking_loss/runs`。后续运行必须改到新目录，避免覆盖既有 ranking-loss 产物。
5. 评估时不能对真实到达标签做 argmax 来挑选动作。主读数是模型所选候选在真实到达离线标签上的 payload 和 regret。选择是否改变只是辅助读数。
6. `results/phase5l_joint_encoder/joint_utility_encoder.pt` 不在当前工作区。后续运行必须先找到这个既有文件，不能用另一份权重代替。epoch 6 和解码器的哈希以上面的登记为准。

预算是 10 个 epoch，每个状态一次更新，冻结最后一轮，不做超参数扫描。一个配对种子是 2 臂 × 12 状态 × 10 = 240 次更新。三个配对种子是 720 次更新。这个预算尚未执行。离线标签差不能写成 48 步正式策略收益，也不能写成新的独立泛化证据。

`results/phase6if_ranking_loss/summary.json` 仍然缺失。E19 的正式重放是 1488 步，另有 30 步作废计数，合计 1518；1488 不是全部消耗。本协议不新增科学结论。
