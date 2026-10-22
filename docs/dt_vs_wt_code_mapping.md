# Decision Transformer 与 Waypoint Transformer 文件级映射

## 1. 对比边界

本文件比较两个独立仓库：

```text
project_parent/
├── decision-transformer-master/    # 当前 DT Gym/Atari 基线，只读保留
└── waypoint_transformer_official/  # 官方 WT，阶段 0.5 开发分支
```

DT 基线不是 WT 的旧版本，也不包含隐藏的 Waypoint Network。两套实现的训练目标、token 语义、优化循环和 checkpoint 格式均不同，必须作为两个独立实验基线。

符号：`B` batch，`T` context，`S` state dimension，`G` global condition dimension，`W` waypoint dimension，`A` action dimension，`H` Transformer hidden dimension。

## 2. 总体映射

| 功能 | 当前 DT 基线 | 官方 WT | 海上迁移时采用哪个 |
|---|---|---|---|
| 数据集读取 | `gym/experiment.py:68-164` 读取本地 `data/{env}-{dataset}-v2.pkl` 轨迹列表 | `src/wt/dataset.py` 通过 D4RL `env.get_dataset()`，切 episode 后采窗口 | 新任务自建中立 episode schema；WT/DT 各写 adapter |
| 轨迹采样 | 按长度加权，可按 return 保留 top `pct_traj` | 按长度加权；同轨迹采两个索引得到 start/global goal | 可复用按有效 timestep 加权思想，不共享环境特例 |
| RTG 构造 | `experiment.py:18-23,141-152` 累计 RTG 除 `scale`，作为独立 token | `dataset.py:238,295-302` 平均 RTG + 粗归一累计 RTG，作为拼接 condition | DT RTG 保持为直接目标基线；WT 重新定义任务 condition |
| Goal 输入 | Gym locomotion 无显式空间 goal，主要条件为目标 return | AntMaze/Kitchen 使用未来 achieved state 作为 global goal | 海上 WT 使用显式空间 goal；DT goal-conditioned 扩展须单独命名 |
| Waypoint 标签 | 不存在 | Goal：真实 30 步后状态前 `W` 维；Reward：逐时刻真实二维 RTG | WT 主线采用；第一版优先空间 Goal Waypoint |
| Waypoint Network | 不存在 | `src/wt/goal_net.py::KForwardGoalNetwork` | 从官方 WT 迁移，DT 仓库不添加 |
| Transformer 策略 | `decision_transformer.py` + `trajectory_gpt2.py` | `transformer.py` + `policies.py` | 分别保留，不能互相覆盖 |
| token 结构 | 每步 `(RTG_t,state_t,action_t)` 三个 token，`[B,3T,H]` | 默认每步一个 `concat(state,global condition,waypoint)` token，`[B,T,H]` | WT 用复合 waypoint token；DT 保持三元 token作对照 |
| timestep/position | episode timestep embedding 加到三种 token | 无 episode timestep；只有窗口内 learned position | 不直接复制；若增加全局时间必须做显式消融 |
| attention mask | `[B,T]` 扩为 `[B,3T]` 并传给 GPT2 | 固定 causal mask；padding mask 不进 attention，只过滤 loss | 新 WT 应显式处理 padding；DT 行为保持不变 |
| action head | `Linear(H,A)`，通常带 `Tanh` | GPT 输出 512 latent，再经 SB3 policy MLP 和动作分布 | 保留各自原头 |
| action loss | mask 后 action MSE | mask 后 `-log_prob(target_action)`；Box 为 Gaussian NLL | 两种 loss 作为独立基线，不能静默互换 |
| 训练器 | 自定义 `Trainer/SequenceTrainer`、AdamW、warmup、grad clip | PyTorch Lightning；Waypoint Adam，policy Adam | 不混用训练状态与优化循环 |
| 评估流程 | `evaluation/evaluate_episodes.py`，按 target return 自回归更新 RTG | `step.py` + `analyze_d4rl.py`，Manager 保存最近20步 | 新环境提供中立 evaluator，由两个 adapter 调用 |
| checkpoint | Gym 原始主线无完整统一保存/恢复流程 | Lightning ModelCheckpoint；`GOAL_NETWORK_CKPT` 加载并冻结 | 采用 WT 两阶段语义并补 metadata；DT 权重不可互载 |
| normalized score | 以环境 return 为主 | 只输出 raw return，未调用 `get_normalized_score()` | 海上任务定义单位明确的新指标 |

## 3. DT 文件级调用链

```text
gym/experiment.py
├── 读取本地 pickle 轨迹
├── discount_cumsum() 构造 RTG
├── get_batch() 生成 state/action/RTG/timestep/mask
├── decision_transformer/models/decision_transformer.py
│   └── decision_transformer/models/trajectory_gpt2.py
├── decision_transformer/training/seq_trainer.py
│   └── masked action MSE
└── decision_transformer/evaluation/evaluate_episodes.py
```

DT shape：

```text
state embeddings          [B,T,H]
return embeddings         [B,T,H]
action embeddings         [B,T,H]
stacked DT tokens         [B,3T,H]
stacked attention mask    [B,3T]
action prediction         [B,T,A]
masked target/prediction  [N_valid,A]
```

`decision_transformer.py:73-83` 的顺序：

```text
R_1, s_1, a_1, R_2, s_2, a_2, ...
```

动作由 state token 位置的 hidden state 预测；`seq_trainer.py:17-29` 对有效 timestep 做 action MSE。

## 4. WT 文件级调用链

### 4.1 Waypoint Network

```text
experiments/launch_waypoint_nets.sh
→ src/wt/train.py --train_goal_net
→ src/wt/dataset.py
→ D4RLIterableDataset
→ src/wt/goal_net.py::KForwardGoalNetwork
→ MSE
→ Lightning checkpoint
```

### 4.2 WT 策略

```text
experiments/{antmaze,gym,kitchen}/*.sh
→ src/wt/train.py
→ src/wt/policies.py::RvS
→ ExtendedActorCriticPolicy
→ src/wt/transformer.py::TransformerExtractor.goal_extract()
→ frozen KForwardGoalNetwork checkpoint
→ concat(state/global condition, waypoint)
→ GPT
→ SB3 policy MLP + action distribution
→ masked negative log-likelihood
```

WT shape：

```text
raw state/condition       [B,T,S+G]
waypoint                  [B,T,W]
combined feature          [B,T,S+G+W]
WT token                  [B,T,H]
causal attention mask     [1,1,T,T]
GPT latent                [B*T,512]
action parameters         [B*T,A]
loss mask                 [B*T]
```

## 5. 核心语义差异

### 5.1 目标不同

DT：

```text
target return → RTG token → 每步按实际 reward 更新
```

WT：

```text
global spatial/reward condition
→ Waypoint Network
→ local intermediate condition
→ WT policy
```

不能把 DT 的 RTG token 直接改名为 waypoint。

### 5.2 Transformer 不同

DT：

- HuggingFace 风格 GPT2；
- 三种 modality embedding；
- episode timestep embedding；
- `[B,3T]` padding attention mask；
- action MSE。

WT：

- 仓库内自定义 GPT；
- 默认一个复合 observation token；
- 只有窗口位置 embedding；
- padding mask 不进入 attention；
- SB3 action distribution NLL。

所以 DT 的 `trajectory_gpt2.py` 不能替换 WT 的 `transformer.py`，反向也不成立。

### 5.3 checkpoint 不可互载

DT 参数名、token embedding、action head 和 optimizer 状态与 WT 不同。WT 还包含：

```text
Waypoint checkpoint（冻结）
→ policy checkpoint
```

即使 state/action 维相同，两边权重也不能互载。

## 6. DT 中可保留的直接目标条件基线

| DT 文件 | 对照价值 |
|---|---|
| `gym/experiment.py` | 本地轨迹读取、累计 RTG、按回报筛选、batch/mask |
| `gym/decision_transformer/models/decision_transformer.py` | 标准 `(RTG,state,action)` token |
| `gym/decision_transformer/models/trajectory_gpt2.py` | DT 专用 causal Transformer |
| `gym/decision_transformer/training/seq_trainer.py` | mask 后 action MSE |
| `gym/decision_transformer/evaluation/evaluate_episodes.py` | desired-return rollout |

后续接入海上任务时，只应增加薄的数据/环境 adapter，保持 DT 核心语义，才能形成可信对照。

## 7. WT 中用于海上迁移的模块

本阶段只选择，不实现：

| WT 文件 | 后续用途 | 注意 |
|---|---|---|
| `src/wt/goal_net.py` | Spatial Waypoint Network 起点 | 将维度、horizon、metadata 配置化 |
| `src/wt/dataset.py` | episode/window/label pipeline | 替换 D4RL；validation/padding 修复需测试 |
| `src/wt/transformer.py` | waypoint 注入和 causal policy | 显式配置维度，不靠 checkpoint 文件名 |
| `src/wt/policies.py` | 动作分布和 NLL | 固定兼容 SB3，或受控去耦 |
| `src/wt/train.py` | 两阶段训练/checkpoint 编排 | 不用环境名字符串决定 condition 类型 |
| `src/wt/step.py` | rollout/evaluation 接口参考 | 必须替换环境专用分支 |
| `src/wt/util.py` | checkpoint/恢复概念 | 将真实配置写入 metadata |

## 8. 绝对不能混用

1. 不把 DT 三元 token 和 WT 复合 token 混成一个仍称“原 DT/WT”的模型。
2. 不给 DT MSE trainer 直接接 WT 的 SB3 action distribution。
3. 不将 WT Gaussian NLL 静默替换为 DT MSE；比较必须单独命名。
4. 不共用 DT/WT checkpoint。
5. 不将 DT return scale 直接用于 Reward Waypoint normalization。
6. 不将 AntMaze `(x,y)`、Kitchen 60 维 goal、MuJoCo normalizer 当作通用海上接口。
7. 不让两个仓库共享同一个依赖环境。
8. 不复制 `trajectory_gpt2.py` 覆盖 WT `transformer.py`。

## 9. 是否复制 DT 代码到 WT 仓库

不需要，也不推荐。

理由：

- 保持两种算法的版本历史和实验归属；
- 避免同名 Transformer/trainer/evaluation 被误接；
- 对比需要统一的数据协议和指标，不需要复制模型；
- 大规模复制会破坏基线可辨识性。

若以后需要统一运行入口，应在父目录增加独立 orchestration 层，或为两个仓库分别写薄 adapter，不复制核心模型。

## 10. 推荐组织

```text
project_parent/
├── decision-transformer-master/
│   ├── gym/
│   └── atari/                       # DT 独立基线
├── waypoint_transformer_official/
│   ├── src/wt/
│   ├── experiments/
│   └── docs/                        # WT 主实现
└── experiment_outputs/              # 可选，模型外部输出
    ├── dt/
    └── wt/
```

若共享海上离线数据，使用版本化只读数据目录，通过两个独立 adapter 消费，不通过复制源码实现共享。

## 11. 后续实验边界（未实现）

```text
共同海上 episode schema / 指标
├── DT adapter → 原 DT 三元 token + action MSE
└── WT adapter
    ├── Spatial Waypoint Network
    └── frozen waypoint + WT policy NLL
```

第一版只建议：

- 单架无人机；
- 4 个静止浮标；
- 二维连续移动；
- 简化通信速率；
- 空间 Goal Waypoint。

这只是阶段 1 建议，本阶段没有编写相关代码。
