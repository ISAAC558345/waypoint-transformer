# 实验报告：紧急度奖励的年龄与标签归因

- 诊断分支：`experiment/priority-tag-reward-attribution`
- 来源实验分支：`experiment/priority-tag-urgency-reward`
- 来源结果提交：`f4ec753d7611d72dc25ace3aeaa8e877d8ef92bb`
- 指令提交：`2d7572ac9a5a134c763d7780a853e858a297dd25`
- Gate：`URGENCY_REWARD_ATTRIBUTION_BLOCKED`
- 是否合并：否
- 标签性质：synthetic priority tag，不是真实海事业务标签。

## 1. 本阶段不做的事

没有重新训练，没有改 lambda_u、rho、beta_tag、tag_rate、队列顺序、动作空间、观测、任务分布或训练预算。
正式奖励、正式 Planner、正式观测、SafetyProjector 和 L0 checkpoint 保持原文件。
来源实验分支不合并。诊断分支也不合并。

## 2. 来源对账

保存的 Gate 是 `PRIORITY_TAG_EXPERIMENT_NOT_MERGED`。
CSV 行数 224，期望 224。
按 CSV 重算后，区间和计数的最大绝对差是 0.000000。
对账问题：无。

## 3. 定义

服务前对每个 FIFO 桶计算 `u_age = clip(age/48, 0, 1)`，`u = clip(u_age + 0.25*synthetic_tag, 0, 1)`。
年龄服务是 `u_age * served_bits`，标签增量是 `(u - u_age) * served_bits`，合紧急度是两者之和。
tagged 服务只统计 `synthetic_tag = 1` 的实际出队 bit。空队列和 tag=0 不计入标签增量。
FIFO 键仍是 admission step 和 insert index。标签不改变出队顺序。

## 4. 重放

首个场景用时 0.445 秒。
按这个速度估计全部 224 段需要 99.8 秒。
时间上限是 600.0 秒。
实际完成 6 段，用时 2.9 秒。
相对已保存评估的最大差 1.553167 bit。
年龄加标签增量相对账本合紧急度的最大差 0.000000 bit。

停止于 L0 seed 0 场景 321101。
仪器重放和已保存结果的字段差：
- payload_bits：1.553167
- urgency_served_bits：0.135255
- served_tagged_bits：0.262044
- served_ordinary_bits：1.291123
- dropped_tagged_bits：0.058690
- dropped_ordinary_bits：1.494478
- path_length_m：0.000006
该场景的分解差是 0.000000 bit。
同一检查点、同一确定性动作、不经过归因包装的原评估路径，字段差为：
- payload_bits：1.553167
- urgency_served_bits：0.135255
- served_tagged_bits：0.262044
- served_ordinary_bits：1.291123
- dropped_tagged_bits：0.058690
- dropped_ordinary_bits：1.494478
- path_length_m：0.000006
原评估路径和仪器路径的差相同，说明差不来自归因包装。
这些差大于 1 bit 的记账容差。重放在这里停止，不用未完成的场景解释年龄或标签。

## 5. 归因

完整重放没有执行，因此没有年龄/标签对照表。

## 6. 下一步

字母是 C。已保存 CSV 和 summary 的区间差是 0，年龄加标签增量也回到账本里的合紧急度。
不能进入容差的是冻结评估的重新运行：payload 差 1.553167 bit，普通服务差 1.291123 bit，普通丢弃差 1.494478 bit，路径差 0.000006 米。
归因包装和原评估路径给出同一组差，所以这不是包装改了动作。
容差保持 1 bit，不因为重放差超过 1 bit 而放宽。
在重放能够复现已保存 bit 之前，不用部分场景解释年龄或标签，不改奖励，也不训练。

固定轨迹上的奖励上升不能代替 tagged 服务。正式分支和原始 checkpoint 保持不变。
