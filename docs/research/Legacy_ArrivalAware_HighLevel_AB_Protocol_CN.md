# 到达感知高层方法的公平 A/B 协议

本协议把此前 720 更新的待定草案收敛为 L3 的一次单配对。配置在运行前固定：seed 0、两臂各 120 次更新、合计 240 次。不重开旧 holdout，也不扫描标签权重。实测结果写入 L3 报告，不在这里预写。

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

## L3 固定单配对

不运行 `phase6if_ranking_loss.train`，也不读取缺失的 `joint_utility_encoder.pt`。入口是 `wt.legacy_arrival_aware_ab.runner`。默认只打印配置。

```text
py -3 -m wt.legacy_arrival_aware_ab.runner --check-only --data results/legacy_arrival_aware_ab_l2 --output results/legacy_arrival_aware_ab_l3
py -3 -m wt.legacy_arrival_aware_ab.runner --run-pair --data results/legacy_arrival_aware_ab_l2 --output results/legacy_arrival_aware_ab_l3
```

- 初始化文件是 `results/phase6id_joint_utility_training/epoch_006.pt`，SHA-256 为 `7859f9c94940d9259ab1bece1f8dddf9655a4dcb1cd45d537ad127347526bcde`。要求 `epoch == 6`，并严格加载完整 `model`。
- 解码器文件是 `results/phase5k_joint_decoder/matching_decoder.pt`，SHA-256 为 `de1809fab8290b5be3e89cd0d31e22cd04638fd9a43207957aa20b1de255eca5`。它保持 eval，参数不更新，训练时梯度仍从解码器传到编码器。
- 每臂从同一份 E6 状态开始，使用空状态 Adam，学习率 `3.0e-4`，权重衰减 `1.0e-4`，梯度裁剪 `1.0`。
- 损失是 `LOSSES["c"]` / `top_regret_loss`。A 用 `zero_payload_bits`，B 用 `real_payload_bits`。
- 固定 CPU、单线程、确定性算法和 seed 0。顺序是训练场景升序，每个场景先 t15 后 t30。只做这一对，不重复 seed 1 或 seed 2。
- 训练 12 个状态、10 轮，每臂 120 次更新，合计 240。只评估最后一轮的 4 个探索状态。分数并列时取较小的候选编号，不用 payload 打破并列。

这 4 个探索状态不是新的独立验证或正式 holdout。离线标签差不能写成 48 步正式策略收益。

`results/phase6if_ranking_loss/summary.json` 仍然缺失。E19 的正式重放是 1488 步，另有 30 步作废计数，合计 1518；1488 不是全部消耗。本协议不新增科学结论。
