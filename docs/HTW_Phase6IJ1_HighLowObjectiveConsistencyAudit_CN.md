# Phase-6I-J1：高低层目标与执行奖励一致性审计

本阶段只阅读正式入口和已保存结果。没有训练，没有修改算法，没有覆盖 checkpoint，也没有重新运行验证集或保留测试。

正式链路是 `run_active_episode` → `FormalPlanner` → `run_closed_episode`。当前策略文件仍是方案 C、G1_RATE、k=6、hold=15。

## 1. 高层

### 1.1 推理时真正进入 HGNN 的状态

`graph_from_env` 把当前环境编成方案 C 的图。节点和边来自 `hgnn_tensors`。

- UAV 节点 8 维：归一化位置、上一步位移、`progress`、是否已有目标、目标队列占比，以及一个恒为 0 的占位。
- 浮标节点 8 维：归一化位置、队列占比、上一拍到达、写到浮标上的 `progress`、是否被占用、常数 `12/12`、按当前最大队列归一化的紧急度。
- UAV–浮标边：距离、当前假设速率、按步长折算的时间、队列占比。队列通道由 `append_queue_edge` 接到第 4 维。
- UAV–UAV 边：距离，以及当前已经激活链路之间的干扰。
- 全局 3 维：时间占比、总队列占比、无人机数量占比。

`j15_bits`、`utility_rank`、未来 SINR、MAPPO value 和 `served_bits` 不在图里。假设速率用的是当前几何和当前队列，不读取未来到达。

这里的 `progress` 是 `L3BEnv.progress`。在 bit 队列环境里，它只是停留计时器，`_update_legacy_progress` 明确禁止它改动 `q_bits` 或 payload。

### 1.2 候选、排序和合法性

G1_RATE 对每架无人机计算当前假设速率，除以全矩阵最大值，再用 `mergesort` 取前 6 个浮标。相等速率保留较小的浮标编号。

`enumerate_legal` 先放空闲 `-1`，再按该无人机的 top-6 顺序放浮标。空闲不占用浮标，重复的非负浮标被跳过。正式池因此有 13327 个合法 assignment。

方案 C 用冻结的 matching decoder 对每个候选求和打分，`np.argsort(-scores, kind="stable")` 取第一名。空闲不加分。分数非有限或候选非法时，这一次决策中止，不改用别的生成器。

G1_RATE 基线不是神经网络分数。它把同一套归一化速率按 assignment 相加，再用稳定顺序上的 `argmax` 选择。

### 1.3 `j15_bits` 的定义

方案 C 的训练标签名是 `J15_MAPPO_CURRENT`。生成函数是 `batch_j15_mappo`，不是 `phase6e/value.py` 里用 `fly_toward` 的 `batch_j15`。

对一个候选，标签的计算是：

1. 从当前状态复制环境。
2. 把后续到达改成全 0。干扰系数保持原值。
3. 用 `make_option` 固定这个 assignment。
4. 冻结 MAPPO-V2 每步输出确定性动作，执行 `min(15, 剩余步数)` 步。
5. 把每步 `payload_increment_bits` 相加。

所以标签是冻结 MAPPO 在零未来到达下、保持同一 assignment 的 15 步交付量。数据集主时间是 t=0、15、30。t=45 不在这组标签里。

方案 C 的 ListNet 损失只读分数和这份标签。推理时不再计算 `j15_bits`。

### 1.4 标签、推理分数和 48 步 payload

三者不是同一个量。

- 标签只覆盖一个 hold，并且关掉了未来到达。
- 推理分数是 HGNN 对当前图的排序，不输出 bit。
- 完整 episode 有真实到达，并在 t=0、15、30、45 重新决策。t=45 的决策只剩下 3 步，而训练标签没有覆盖这个决策时刻。

因此，一个在 15 步零到达标签上更好的 assignment，不必在 48 步真实到达的完整 episode 上更好。已保存的 Phase-6I-Z 短窗口对照区间跨过 0，而 Phase-6I-AB/AC 的完整 episode 平均配对为正，这两类结果本来就不能互相代替。

标签和完整收益还会分叉，因为：

- 15 步标签看不到后三次决策的后果。
- 标签期间远簇本应在偶数步到达的业务被置零，真实 episode 不会置零。
- 标签执行器是 MAPPO，但 MAPPO 训练时抽的是 G1_RATE k=5 的随机 assignment，不是方案 C 的 k=6 选择。
- 高层看到的 `progress` 不推动队列，不能代表剩余服务量。

## 2. 低层 MAPPO-V2

### 2.1 动作

推理函数 `_actor_move` 的动作是：

```text
executed = clip(fly_toward(assignment) + 0.25 * tanh(actor_mean))
```

`fly_toward` 是指向当前目标浮标的单位向量。空闲目标对应零向量。Actor 的均值头输出 2 维，推理不采样。残差先乘 `ALPHA_RESIDUAL = 0.25`，再与基础方向相加，并截断到单位步长。

这个残差只能改变运动方向和步长比例，不能输出浮标编号。

### 2.2 SafetyProjector

投影器把单位动作变成下一步位置，然后做三件事：把过近的无人机沿连线推开，把已经进入服务包络的无人机拉回目标周围的内半径，再把超出 22 m 的位移缩回。约束失败或结果非有限时，正式执行把该步动作改成 0，并记一次干预。投影量大于 `1e-3` 也记干预。

所以低层想飞的方向可以被投影改掉，甚至变成原地不动。投影不改 assignment。

### 2.3 直接输入网络的量

Actor 每个无人机的输入是：

| 组 | 宽度 | 内容 |
| --- | --- | --- |
| self | 5 | 位置、上一步实际动作、常数 1 |
| task | 8 | 是否有目标、相对目标的位移和距离、目标队列的 block 当量、`progress`、剩余 hold 占比、剩余 episode 占比 |
| communication | 4 | 当前链路的 SINR、速率、总干扰、干扰噪声比 |
| neighbor | 11×5，编码成 16 | 其他无人机的相对位置、相对动作、距离、是否在服务、是否空闲、对方目标相对位置、对方 `progress`、对方是否在服务半径内 |
| nearest | 1 | 到最近其他无人机的距离 |

没有目标时，task 只保留 hold 和剩余时间，communication 为 0。通信特征来自 `active_links`：必须已经分配、没有冲突、进入 16 m 且队列为正。飞向目标但还没进半径时，网络看得到相对位置，看不到这条未来链路的 SINR。

Critic 额外看到全体 assignment、全体队列、全体 `progress`、全体 SINR 和速率、无人机两两距离、干扰矩阵、剩余 hold 和剩余时间。Critic 不参与推理动作。

没有单独的“接近目标”特征超出 task 里的相对位移。假设速率矩阵只在高层边上，不进 Actor。

### 2.4 奖励

`low_level_reward` 是：

```text
reward = served_bits / 1e6 - 0.05 * c_project - 0.01 * c_residual
```

`served_bits` 是该步所有浮标实际入账 bit 之和。`c_project` 是执行动作和投影前动作的均方误差，`c_residual` 是残差向量的均方。两者都按 `ACTION_SCALE = 1` 归一化。

奖励里没有服务进度项，没有靠近目标的奖励，也没有单独的 SINR 项。进度和 SINR 只能通过以后的 `served_bits` 或观测影响策略。

单位动作每架无人机的平方和最多为 2。两项惩罚的上限大约是每步 `0.05×2 + 0.01×2 = 0.12` 个 block。因此，一步交付少于约 `1.2×10^5` bit 时，惩罚可以让这一步的奖励低于“没有交付、也没有惩罚”的一步。这就是奖励上升而交付下降的单步路径。

### 2.5 训练尺度

Phase-6G 使用 `PPOConfig`：`gamma = 0.99`，`gae_lambda = 0.95`，`rollout_length = 48`。更新时对这 48 步奖励做 GAE，优化的是折扣回报，不是 48 步 payload 的无折扣和。

训练环境会跑满 48 步，并每 15 步重新抽样。抽样来自 G1_RATE k=5 的均匀随机合法 assignment，不是方案 C，也不是 k=6。推理却执行方案 C 的 k=6 目标。低层没有在自己的训练分布里见过当前高层的选择规则。

## 3. 高低层接口

正式循环在 `run_closed_episode`。

1. `pending` 初始为 15，每步减 1。只有减到 0 才重新调用 `select_assignment`。窗口内 `option.association` 不变。若步中 association 被改写，该步仍按改写前的目标入账，然后停止。
2. 新目标写入后，`env.step` 发现某架无人机的目标编号变了，就把该机的 `progress` 清零。位置不重置，队列不重置，已经累计的 payload 不重置。
3. MAPPO 的输出只进入 `_actor_move` 的位移。`env.step(movement, association)` 的 association 是高层复制出来的向量。低层可以改变飞行方向，不能改分配。
4. 下一步高层决策重新调用 `graph_from_env`。新的位置、队列、假设速率和当前激活链路的干扰会进入图。SINR 本身不是图上的一个独立输入，它通过速率和干扰边间接出现。`progress` 会进入图，但它不控制 bit 账。
5. `payload_bits` 定义为 `cumulative_served_bits` 之和。闭环把每步 `served_bits` 相加，并要求与 `payload_bits` 的差不超过 1 bit，同时检查 bit 守恒。已保存的 AB/AC 完整 episode 没有计账异常。
6. 存在奖励和交付不一致的路径。单步奖励包含最多约 0.12 block 的动作惩罚；折扣回报也不是 payload 本身。高层标签又是零到达的 15 步和，完整指标是带到达的 48 步和。所以低层奖励变好、15 步标签变好、48 步交付变差，可以同时发生。AB 的 5 个负场景和 AC 的 2 个负场景已经表明，方案 C 的选择不是每个场景都提高完整交付。

## 结论

闭环可以执行，计账也把 payload 和 `served_bits` 绑在一起。协同缺口在目标定义：

- 高层用零到达的 15 步 MAPPO 交付量训练，用当前图的排序做 48 步决策。
- 低层用带惩罚的单步 bit 奖励和 0.99 折扣训练，执行时只跟踪高层给定的目标。
- 两边共享的 `progress` 不推动真实队列。
- 低层训练候选是随机 k=5，正式高层候选是方案 C 的 k=6。

这些差异都来自当前代码路径，不需要新实验就能成立。
