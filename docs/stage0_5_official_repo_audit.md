# 阶段 0.5：Waypoint Transformer 官方实现审计

## 1. 范围和结论

本阶段只确认、审计并做启动层最小修正；没有创建海上无人机环境，没有生成通信数据，没有训练模型，也没有修改 Waypoint Network、Transformer、标签或损失函数的数学逻辑。

结论：

1. 当前同级 `decision-transformer-master` 是 DT Gym/Atari 基线，不含 Waypoint Network。
2. `waypoint_transformer_official` 是 StanfordAI4HI 官方 WT 仓库。
3. 官方 Waypoint Network 是残差 MLP；Goal/Reward Waypoint 共用网络类，但标签定义不同。
4. 官方 WT 默认把 `state + global condition + predicted waypoint` 拼成一个 token，并非 DT 的 `(RTG,state,action)` 三元 token。
5. 官方 HEAD 的全部 launch script 指向不存在的 `src/rvs/train.py`，Waypoint launcher 还使用 parser 不接受的 AntMaze v0；这些启动问题已最小修正。
6. 当前 DT 环境不满足 WT 完整依赖，不能据此宣称 WT 已完整运行。
7. 验证集、padding mask、checkpoint 配置推断等语义缺陷只记录，阶段 0.5 未修改。

## 2. 仓库身份

| 项目 | 已确认值 |
|---|---|
| 路径 | `E:\srdp\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\waypoint_transformer_official` |
| remote | `https://github.com/StanfordAI4HI/waypoint-transformer.git` |
| 审计基准 commit | `716f514b942dfe413dd23fd51cf1c523d8211c0f` |
| commit subject | `added requirements` |
| 开发分支 | `feature/maritime-wt-stage0-5` |
| README 标题 | `Waypoint Transformer: Reinforcement Learning via Supervised Learning with Intermediate Targets` |

关键结构已确认：

```text
README.md
requirements.txt
experiments/
├── launch_waypoint_nets.sh
├── antmaze/
├── gym/
├── kitchen/
└── config/
src/wt/
├── dataset.py
├── goal_net.py
├── policies.py
├── transformer.py
├── train.py
├── step.py
├── analyze_d4rl.py
├── visualize.py
└── util.py
```

当前 DT 基线的 `.git` 元数据不可用；未执行 `git init`，也未移动、覆盖或重命名其中的用户文件。

## 3. 官方代码结构

| 功能 | 文件 | 关键对象 |
|---|---|---|
| 总训练入口 | `src/wt/train.py` | `run_training()` |
| Waypoint Network | `src/wt/goal_net.py` | `KForwardGoalNetwork` |
| 数据和标签 | `src/wt/dataset.py` | `D4RLIterableDataset`、RvS-G/R DataModule |
| Transformer | `src/wt/transformer.py` | `GPT`、`TransformerExtractor` |
| 策略和动作 loss | `src/wt/policies.py` | `RvS`、`ExtendedActorCriticPolicy` |
| 环境与 rollout | `src/wt/step.py` | `create_env()`、各任务评估函数 |
| checkpoint 查找 | `src/wt/util.py` | `get_best_val_checkpoint()` |
| D4RL 分析 | `src/wt/analyze_d4rl.py` | `analyze_performance()` |
| Waypoint 启动 | `experiments/launch_waypoint_nets.sh` | `--train_goal_net` |
| WT 策略启动 | `experiments/{antmaze,gym,kitchen}/*.sh` | 环境、配置、seed |

## 4. Waypoint Network

主类在 `src/wt/goal_net.py:17-81`：

```python
KForwardGoalNetwork(
    obs_dim, goal_dim, hidden_dim, max_T,
    recurrent=False, learning_rate=1e-3,
    batch_size=1024, reward=False
)
```

`recurrent` 只保存为属性，没有进入 forward；配置中的 `depth` 也不控制该网络。实际结构固定为：

```text
Linear(S+W,Hw) → ReLU
Linear(Hw,Hw)  → ReLU
Linear(Hw,Hw)  → ReLU
Linear(Hw,Hw)  → ReLU
Linear(Hw,W)
```

最后一层无激活；没有为离散/连续状态设置不同 waypoint 输出头。

- Goal：`output = current_state[..., :W] + MLP(state, global_goal)`
- Reward：`output = global_reward_condition + MLP(state, global_reward_condition)`

Reward 分支强制 `W=2`。Goal 和 Reward 共用结构，仅残差锚点和监督标签不同。

官方 `Hw=1024`：

| 任务 | `S` | `W` | 输入 | 输出 | 参数量 |
|---|---:|---:|---:|---:|---:|
| AntMaze | 29 | 2 | 31 | 2 | 3,183,618 |
| Kitchen | 60 | 60 | 120 | 60 | 3,334,204 |
| Hopper | 11 | 2 | 13 | 2 | 3,165,186 |
| HalfCheetah/Walker2d | 17 | 2 | 19 | 2 | 3,171,330 |

另有 `ManualGoalNetwork`，它从硬编码 AntMaze 坐标中选点，不是训练得到的 Waypoint Network。

## 5. Waypoint 标签和数据流

### 5.1 调用链

```text
D4RL env.get_dataset()
→ D4RLRvSGDataModule / D4RLRvSRDataModule
→ terminal/timeout 切分 episode
→ D4RLIterableDataset
→ 按轨迹长度加权采样
→ 当前窗口和 global condition
→ KForwardGoalNetwork
→ MSE
→ Adam
```

代码位置：数据分流 `dataset.py:31-87`；索引采样 `dataset.py:197-222`；batch 构造 `dataset.py:224-271`；iterable yield `dataset.py:280-289`；loss/optimizer `goal_net.py:47-81`。

### 5.2 Goal Waypoint

同一 episode 内均匀采两个索引并排序：

```text
si = min(t1,t2)
gi = max(t1,t2)
```

- 当前窗口：`observations[si:si+40]`。
- global goal：`observations[gi, goal_columns]`，沿窗口重复。
- AntMaze `goal_columns=(0,1)`，goal/waypoint 是二维绝对坐标。
- Kitchen `goal_columns=None`，goal/waypoint 是完整 60 维 observation。
- Waypoint 数据窗口 `L=40`，监督 horizon `max_T=30`。
- 对 `i=0,...,9`：

  ```text
  input_i  = concat(state_i, global_goal)
  target_i = state_{i+30}[:W]
  ```

标签是真实 30 步后绝对状态，不是状态差值；网络内部通过 current state 加残差输出绝对 waypoint。

`gi` 与固定 30 步标签独立；源码只保证 `gi>=si`，不保证 `gi>=si+30`，所以监督 waypoint 不一定严格位于当前点与随机 global goal 之间。

### 5.3 Reward Waypoint

默认 `discount_cumsum()` 产生两通道：

```text
rtg[...,0] = remaining cumulative reward / remaining steps
rtg[...,1] = remaining cumulative reward / normalization_const
normalization_const = round(max episode return, -3)
```

随后：

```text
local_rtg  = 每个窗口时刻的真实二维 RTG
global_rtg = 窗口第一个 RTG 在 40 步上重复
input      = concat(state, global_rtg)
target     = local_rtg
```

Reward loss 对 `[B,40,2]` 做整体 MSE，完全不使用 `max_T=30`。它不是“K 步后奖励”，而是给定初始回报目标时预测每个当前状态应使用的更新回报条件。

`AVG_REWARD`/`CM_REWARD` 可令函数返回一维，但调用方仍 reshape 为最后一维 2，checkpoint 配置也固定 `goal_dim=2`，存在 shape 风险。

### 5.4 Episode 尾部和归一化

不足 40 步时进行左填充：

| 张量 | padding |
|---|---|
| state | `0` |
| action | `-10` |
| goal/RTG | `0` |
| mask | padding `0`、有效数据 `1` |

`KForwardGoalNetwork.training_step()` 解包了 mask，却从未使用。因此 Goal/Reward MSE 都会包含 padding；Reward 分支还可能把左侧零 padding 当作 initial RTG。

state/goal normalization 已被注释，实际未执行。Reward 第二通道使用上述粗略缩放；若最大 episode return 很小，round 后还可能为 0。

## 6. 训练和 checkpoint

Waypoint 配置 `experiments/config/d4rl/goal_network.cfg`：

| 参数 | 值 |
|---|---:|
| learning rate | `1e-3` |
| hidden size | `1024` |
| batch size | `1024` |
| max steps | `40000` |
| checkpoint interval | `100` steps |
| val fraction | `0.1` |

无固定 epoch 数，以 40,000 gradient steps 停止。optimizer 为 Adam，loss 为 MSE。

checkpoint：

- 目录：W&B run 下的 `checkpoints/`；
- 周期保存：每 100 steps、全部保留；
- 最优保存：最低 `val_loss`，另存 `last.ckpt`；
- 模板：`gcsl-{env_name}-{epoch:03d}-{val_loss:.4e}.ckpt`。

验证集存在严重问题：Reward 和 Goal DataModule 都使用 `paths[train_indices]` 创建 validation dataset（`dataset.py:523,619`），不是 `paths[val_indices]`。因此官方 `val_loss` 不是严格 held-out 轨迹损失；本阶段未改。

WT 策略通过 `GOAL_NETWORK_CKPT` 手动接入 checkpoint：

```text
GOAL_NETWORK_CKPT
→ ExtendedActorCriticPolicy
→ TransformerExtractor
→ 首次 forward 时 load_from_checkpoint()
→ eval()
→ requires_grad=False
→ torch.no_grad() + detach()
→ 与原 feature 拼接
```

冻结代码在 `transformer.py:326-343`，所以策略训练时 Waypoint Network 确实冻结。

checkpoint 架构不从 metadata 通用恢复，而根据路径是否包含 `antmaze/kitchen/hopper/cheetah/walker` 推断；重命名可能导致断言或 shape 错误。

## 7. Waypoint Transformer 策略

主类为 `transformer.py::GPT/TransformerExtractor`、`policies.py::RvS/ExtendedActorCriticPolicy`。

默认参数：

```text
context T=20
Transformer hidden H=128
heads=16
layers=2
dropout=0.15
GPT decoder width=512
policy MLP width=1024
```

### 7.1 token

默认 `MODEL_ACTIONS` 关闭：

```text
raw feature      = concat(state, global goal/reward)
waypoint         = frozen waypoint network(raw feature)
combined feature = concat(raw feature, waypoint)
token            = Tanh(Linear(combined feature,H))
```

Goal、Reward/RTG 和 Waypoint 都不是独立 token。

开启 `MODEL_ACTIONS` 后为 `obs_0, action_0, obs_1, action_1, ...`，长度 `2T`，obs 位置输出用于动作预测。该模式默认关闭，且 `act_dim` 硬编码为 8，只与 AntMaze 偶然匹配。

### 7.2 position、attention、动作 loss

- 没有 episode 全局 timestep 输入，只有 learned local position embedding；
- attention 只有固定 causal triangular mask；
- 数据 padding mask 不进入 attention，只在 log probability 后过滤 loss；
- `GPT.head=Linear(H,512)+Tanh` 是 latent decoder，不是最终动作头；
- 最终动作分布由 Stable-Baselines3 创建：Box 为 diagonal Gaussian，Discrete 为 categorical；
- 训练 loss 为 `-mean(log_prob(target_action))`。

连续动作是 Gaussian NLL，离散动作是 categorical NLL。动作 MSE 只记录日志，不作为 loss。

## 8. 关键 shape

定义：`B` batch；`L=40` Waypoint 窗口；`T<=20` WT context；`S` state；`G` global condition；`W` waypoint；`A` action；`Hw=1024` Waypoint hidden；`H=128` Transformer hidden。

Goal Waypoint：

```text
state                [B,L,S]
global goal          [B,L,G]
obs_goal             [B,L,S+G]
单个 offset input     [B,S+G]
30-step target       [B,W]
waypoint output      [B,W]
padding mask         [B,L]       # loss 未使用
```

Reward Waypoint：

```text
state                [B,L,S]
global initial RTG   [B,L,2]
input                [B,L,S+2]
local RTG target     [B,L,2]
reward waypoint      [B,L,2]
padding mask         [B,L]       # loss 未使用
```

WT 策略：

```text
raw state/condition  [B,T,S+G]
waypoint             [B,T,W]
combined feature     [B,T,S+G+W]
token embedding      [B,T,H]
causal mask          [1,1,T,T]
padding mask         [B,T]       # 不进入 attention
transformer hidden   [B,T,H]
GPT latent           [B,T,512]
flattened latent     [B*T,512]
policy latent        [B*T,1024]
action parameters    [B*T,A]
target action        [B*T,A]
loss mask            [B*T]
```

| 任务 | `S` | `G` | `W` | `A` | combined |
|---|---:|---:|---:|---:|---:|
| AntMaze | 29 | 2 | 2 | 8 | 33 |
| Kitchen | 60 | 60 | 60 | 9 | 180 |
| Hopper | 11 | 2 | 2 | 3 | 15 |
| HalfCheetah/Walker2d | 17 | 2 | 2 | 6 | 21 |

## 9. 实验脚本

共 20 个：Waypoint 1、AntMaze 9、Gym/MuJoCo 7、Kitchen 3。修正后统一调用：

```bash
PYTHONPATH=src python -m wt.train ...
```

Waypoint launcher 使用 `goal_network.cfg`，默认 6 个 AntMaze v2、seed 42、`--train_goal_net --use_gpu`。6 个任务用 `&` 并行，可能一次耗尽显存；Kitchen/MuJoCo 数组仍是注释，需要人工选择。

策略配置：

- AntMaze：batch 1024、30k steps、每 3k checkpoint；
- Kitchen：相同参数，分析 `kitchen_subtasks`；
- Gym/MuJoCo：Reward conditioning，相同训练规模。

策略脚本不会自动寻找 Waypoint checkpoint。未设置 `GOAL_NETWORK_CKPT` 时，`goal_extract()` 原样返回 feature，实际没有 waypoint。

## 10. 环境接口

- D4RL：`gym.make(env_name)` + `env.get_dataset()`；
- 旧 Gym API：`reset()->obs`、`step()->(obs,reward,done,info)`、`env.seed()`；
- AntMaze：goal 来自 `env.target_goal[2]`，检查 `observation[:2]==env.get_xy()`；
- Kitchen：60 维 task goal，可按剩余子任务动态更新；
- MuJoCo Reward：Manager 保存最近 20 个 observation、19 个 action 并补零 action，按累计 reward 更新 remaining condition；
- 终止依赖环境 `done`/TimeLimit；
- 评估输出 raw return，没有调用 `env.get_normalized_score()`。

可复用概念：episode→window→mask、空间 Goal Waypoint 两阶段监督、frozen checkpoint→policy feature、Gym 风格 reset/step、实验生命周期。

必须重写：D4RL 数据源、环境特例、reward/terminal/timeout/success、海上任务指标、checkpoint 维度与 metadata。

## 11. 依赖风险

官方主要 pin：

```text
gym==0.18.3
d4rl==1.1
numpy==1.21.6
numba==0.56.4
torch==1.13.0
pytorch-lightning==1.5.10
torchmetrics==0.10.2
mujoco-py==2.1.2.14
mujoco==2.3.0
wandb==0.13.5
CUDA 11.7 packages
```

源码使用但 requirements 未显式声明/固定：`configargparse`、`stable-baselines3`、`seaborn`、`tqdm`、`dm-control`、`pybullet`。

README 要 MuJoCo 2.1.0（mujoco-py）和 2.1.1（dm-control），requirements 却固定 Python `mujoco==2.3.0` 且未列 dm-control，声明相互矛盾。

当前 `.venv-dt-gym`：

```text
Python 3.10.20
pip 26.1.2
torch 2.7.1+cu128
gym 0.23.1
numpy 1.26.4
d4rl 1.1
mujoco-py 2.1.2.14
wandb 0.19.11
```

关键缺失包括 PL、TorchMetrics、SB3、configargparse、GCSL、mujoco、dm-control、pybullet、pandas、scipy、numba、seaborn、scikit-learn。`pip check` 明确报告 D4RL 缺 dm-control 和 pybullet。

主要风险：

1. 当前 Torch/Gym/NumPy/W&B 与官方 pin 不同；
2. PL 1.5 的旧 Trainer API 与 PL 2.x 不兼容；
3. Torch 1.13/CUDA 11.7 不支持 RTX 5060 Blackwell；
4. Numba 0.56.4 与 NumPy 1.26 不兼容；
5. `nvidia-*-cu11` 是 Linux 导向，不能在 Windows 原样安装；
6. SB3 未锁版本，源码又依赖私有 `_get_latent()`；
7. 当前 DT 的 `run_dt.ps1` 和 MuJoCo 注入不会自动作用于 WT。

建议保留 DT 环境，后续在 WSL2/Linux 建独立 Python 3.9 环境。官方原样环境与 RTX 5060 现代兼容环境应分开定义；本阶段没有安装。

## 12. 已发现问题

已做启动层修正：

1. 20 个 launch script 改为模块入口；
2. Kitchen Partial 移除 `~/rvs`；
3. Waypoint 默认 AntMaze v0 改 v2；
4. README Kitchen 路径补 `experiments/kitchen/`；
5. 恢复 `gcsl.algo.variants` import；
6. 20 个 `.sh` 统一 LF。

只审计、未修改：

1. validation dataset 误用 train trajectories；
2. Waypoint loss 忽略 padding mask；
3. Reward Waypoint 不使用 30-step horizon；
4. state/goal normalization 被注释；
5. Reward normalization constant 可能为 0；
6. AVG/CM Reward shape 冲突；
7. checkpoint 配置靠路径关键词；
8. MODEL_ACTIONS 布尔解析和固定动作维 8；
9. padding mask 不进入 attention；
10. 无全局 timestep embedding；
11. `TRACK_GOALS` 在 CUDA tensor 上直接 `.numpy()`；
12. requirements 缺关键依赖；
13. 评估不输出 normalized score；
14. `Manager` 类重复定义；
15. 两处 `np.ndarary` 拼写错误因 postponed annotations 暂不阻塞。

## 13. 测试记录

已通过：

| 检查 | 摘要 |
|---|---|
| Git 身份 | remote、commit、branch、README 和结构正确 |
| `python -m compileall -q src` | 通过 |
| 20 个脚本 `bash -n` | 修正前、修正后均通过；最终输出 `bash -n: OK (20 scripts)` |
| `PYTHONPATH=src import wt` | 通过；namespace package，`wt.__file__ is None` |
| 合成 batch | Goal `(4,40,29)/(4,40,8)/(4,40,2)`；Reward `(4,40,11)/(4,40,2)` |
| 隔离 torch forward | 输出均无 NaN |

隔离 forward 使用轻量 stub 绕开当前缺失依赖，只执行官方文件中的自包含 PyTorch 网络，不等于完整包导入成功：

| 模块 | 输入 | 输出 | 参数量 |
|---|---|---|---:|
| AntMaze Goal Waypoint | `(2,40,31)` | `(2,40,2)` | 3,183,618 |
| Hopper Reward Waypoint | `(2,40,13)` | `(2,40,2)` | 3,165,186 |
| GPT (`feature_dim=33`) | `(2,20,33)` | `(40,512)` | 535,424 |

实际阻塞：

| import/运行 | 原因 |
|---|---|
| `wt.goal_net` | 缺 `pytorch_lightning` |
| `wt.transformer` | 缺 `stable_baselines3` |
| `wt.train` | 缺 `configargparse`，后续还需 GCSL/D4RL |
| `wt.dataset/step` | D4RL 数据目录及完整 MuJoCo/Kitchen 依赖 |
| 完整 WT forward | PL/SB3/GCSL/MuJoCo 链不完整 |
| 训练 | 阶段 0.5 禁止，未运行 |

## 14. 本阶段修改

| 文件 | 内容 |
|---|---|
| `README.md` | 修正 Kitchen launch 路径 |
| `experiments/launch_waypoint_nets.sh` | 修正入口和 AntMaze 版本 |
| `experiments/antmaze/*.sh`（9） | 修正入口 |
| `experiments/gym/*.sh`（7） | 修正入口 |
| `experiments/kitchen/*.sh`（3） | 修正入口/移除主目录硬编码 |
| `src/wt/step.py` | 恢复 GCSL variants import |
| `docs/stage0_5_official_repo_audit.md` | 本报告 |
| `docs/dt_vs_wt_code_mapping.md` | 文件级映射 |

没有修改 `goal_net.py`、`dataset.py`、`transformer.py`、`policies.py` 或任何数学逻辑。

## 15. 阶段 1 建议（仅计划）

下一阶段只建议先形成最小闭环：

1. 单架无人机；
2. 4 个静止浮标；
3. 二维连续移动；
4. 可单元测试的简化通信速率；
5. 空间 Goal Waypoint；
6. 明确定义 observation/action/reward/terminal/timeout/success 和离线 episode schema；
7. 分别保留直接目标条件基线与 frozen Spatial Waypoint + WT。

阶段 0.5 到此停止；上述内容尚未编码。
