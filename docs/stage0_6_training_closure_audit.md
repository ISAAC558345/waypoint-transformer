# 阶段 0.6：Waypoint Transformer 训练闭环修复审计

## 1. 结论

阶段 0.6 的代码级目标已经完成：

- validation 不再复用 training trajectories；
- reward normalization 只由 training trajectories 统计并由 validation 复用；
- Goal/Reward Waypoint 标签统一由 dataset 按可配置 horizon 构造；
- 所有 Waypoint loss 和 policy loss 都在 reduce 前排除 padding；
- padding mask 已进入 GPT/Transformer attention；
- `MODEL_ACTIONS` 下 observation/action token 使用同步扩展的 mask；
- action、goal、waypoint 等维度不再由常数或 checkpoint 文件名猜测；
- 新 Waypoint checkpoint 会保存架构、reward representation、训练集
  normalization 与 episode horizon metadata；旧 checkpoint 仅允许显式
  config 补齐缺失字段；
- learned Waypoint 子模块会在 policy 构造期注册并冻结，可参与严格 state-dict
  重载，且不会被 SB3 初始化覆盖；
- reward representation 与训练 split 的 normalization metadata 会随 policy
  checkpoint 保存；训练与评估都以滑动 context 的起点 RTG 作为窗口内固定命令；
- average RTG 使用 checkpoint 记录的名义 episode horizon，提前终止后的剩余步
  视为零奖励，避免在线阶段使用未知实际终止长度；
- 外层 RvS checkpoint 已嵌入并可独立恢复 frozen Waypoint 权重，不再依赖原始
  Waypoint 文件仍位于旧路径；
- 已建立并通过 CPU 合成数据的 dataset → forward → loss → backward →
  optimizer step 最小闭环。

本阶段没有实现 UAV、buoy、channel、SINR、power allocation 或任何海上通信
环境。代码接口与单 batch 训练闭环已经具备进入阶段 1 的条件；但当前 Windows
环境还不能完成真实 Lightning Trainer/CLI 启动，因此不能据此宣称完整运行验收
或 D4RL 正式基准复现已经完成。完整 D4RL/MuJoCo 集成仍需在 Linux/WSL2 的
官方兼容环境中验证。

审计基线为分支 `feature/maritime-wt-stage0-5`、提交
`716f514b942dfe413dd23fd51cf1c523d8211c0f`。阶段 0.5 的未提交修复与文档
仍保留，本阶段没有回滚或覆盖它们。

## 2. 阶段 0.6 修改文件

### 训练与模型

- `src/wt/train.py`
- `src/wt/dataset.py`
- `src/wt/goal_net.py`
- `src/wt/policies.py`
- `src/wt/transformer.py`
- `src/wt/step.py`
- `src/wt/analyze_d4rl.py`

### 配置与依赖

- `requirements.txt`
- `environment.yml`
- `experiments/config/d4rl/antmaze_rvs_g.cfg`
- `experiments/config/d4rl/goal_network.cfg`
- `experiments/config/d4rl/gym_rvs_r.cfg`
- `experiments/config/d4rl/gym_rvs_r_medium.cfg`
- `experiments/config/d4rl/kitchen_rvs_g.cfg`

### 测试与审计

- `tests/test_stage0_6_training_closure.py`
- `docs/stage0_6_training_closure_audit.md`

## 3. 问题、根因与修复

| 问题 | 根因 | 修复 |
| --- | --- | --- |
| validation 使用训练轨迹 | RvS-R 和 RvS-G 的 `val_dataset` 都错误地读取 `paths[train_indices]` | 使用独立的 `paths[val_indices]`；保存 train/val trajectory indices 供审计 |
| split 不完全可复现 | split 使用全局 `np.random` 状态 | `d4rl_trajectory_split(..., seed)` 使用局部 `np.random.default_rng(seed)` |
| flat BC split 会打散 episode | 原逻辑把 timestep 数组当作 trajectory 集合直接随机划分 | 新增 `d4rl_timestep_split`，先按 done marker 划分完整 trajectory，再映射回 timestep |
| train/val/eval RTG 尺度不一致 | 独立 split 后两个 dataset 分别计算 normalization constant，评估又按环境名硬编码常量 | 只从 training trajectories 计算一次并显式传给 validation；该值写入 RvS checkpoint，评估优先读取；旧 checkpoint 才从环境 dataset 计算兼容值 |
| 最后一条 truncated trajectory 被丢弃 | `_get_paths()` 仅在 terminal/timeout 时 append | 循环结束后 flush 非空 trajectory；缺少 timeouts 时使用 terminal 并保留末尾片段 |
| 极小数据集产生空 split | `val_frac>0` 但只有一条 trajectory 时仍构造 iterable dataset | 独立 validation 要求至少两条 trajectory，`val_frac=1` 明确拒绝 |
| Waypoint loss 计入 padding | MSE 先对完整 tensor reduce，外层 mask 无法撤销 padding 对均值的影响 | `_masked_mse` 先得到逐 timestep loss，再用 `[B,K]` mask 选择，最后 reduce；全 padding 返回可反传的零 |
| Reward Waypoint 没有真正使用 30-step horizon | reward dataset 返回当前 RTG 序列，loss 直接拟合该序列 | dataset 构造 `RTG[min(t + H, episode_last)]`，`H=waypoint_horizon`，默认 30 |
| Goal 与 Reward 的 horizon 逻辑分散 | Goal 网络自行在窗口内移位，Reward 路径在 dataset 中使用另一套逻辑 | 两类 Waypoint 都由 dataset 输出显式 horizon-aligned target；网络只负责 masked prediction loss |
| 短 trajectory 可能越界 | 直接使用 `t + horizon` | target index 对本 trajectory 的末端 clamp，不会访问下一条 trajectory |
| Transformer 会 attend 到 padding | GPT 只有 causal mask，没有 key padding mask | attention 使用 `causal_mask ∩ valid_key_mask`；padding query 使用有限的对角 fallback 防止全 `-inf`，随后输出清零 |
| `MODEL_ACTIONS` mask shape 不匹配 | observation/action 交错后 token 数翻倍，原 timestep mask 未扩展 | 使用 `mask.repeat_interleave(2, dim=1)` 生成 `[B,2K]` token mask |
| `context_length=1` 时动作历史不截断 | Python 切片 `[-K + 1:]` 在 `K=1` 时变成 `[0:]`，保留全部历史动作并使 action token 宽度错位 | 两个 Manager 路径都显式保留 `max(K-1,0)` 个历史动作；`K=1` 时清空历史，并用 `MODEL_ACTIONS` 前向回归测试覆盖 |
| policy action loss 仍受 padding 影响 | log-prob 与 prediction error 在完整 `[B*K]` 上 reduce | log-prob 和 MSE 均先用 flatten 后的有效 timestep mask 选择 |
| action dim 固定为 8 | GPT 构造函数内写死 `act_dim=8` | 从 environment `action_space` 推导；显式 config 与 environment 冲突时立即报错 |
| checkpoint 文件名决定模型维度 | 代码根据路径中 `antmaze/kitchen/hopper/...` 关键词写死维度 | checkpoint 仅提供权重；维度来自 environment space、dataset metadata 和显式 config |
| checkpoint 缺少架构/标签 metadata | KForward 没有调用 `save_hyperparameters()`；一维 average/cumulative 仅凭 shape 无法区分，normalization 也可能跨 split 漂移 | 新 checkpoint 保存 `obs/goal/hidden/waypoint_horizon/reward/reward_representation/reward_normalization_const/reward_episode_horizon`；旧 checkpoint 只补真正缺失的字段，已存在字段冲突立即报错 |
| policy checkpoint 依赖进程外 waypoint 路径 | Lightning 在 outer state-dict 写入前先构造模型，原实现会提前重新打开旧 Waypoint 路径 | 新增显式 `goal_network_ckpt` CLI；正常训练仍校验源文件，outer RvS 重载检测内嵌 `goal_network.*` keys 后仅重建拓扑，再由严格 outer state-dict 恢复权重；`GOAL_NETWORK_CKPT` 只在 CLI 解析时作为默认值 |
| learned Waypoint strict reload 出现 unexpected keys | learned 子模块在首次 forward 才注册，Lightning 加载 policy state dict 时模型拓扑尚不完整 | learned/manual Waypoint 都在 `TransformerExtractor.__init__` 注册并冻结；strict state-dict round-trip 验证无 missing/unexpected keys |
| eager Waypoint 权重可能被 SB3 重置 | SB3 1.2 默认对整个 policy 递归做 orthogonal initialization | `ExtendedActorCriticPolicy.init_weights` 跳过被冻结的 pretrained Waypoint 参数，其他 policy 模块继续使用官方初始化 |
| Goal network 维度写死 | `train.py` 根据环境名写死 2/60 等维度，reward 模型还断言 `goal_dim == 2` | data module 暴露 `state_dim/action_dim/goal_dim/waypoint_dim`；训练入口读取这些属性；删除模型内部断言 |
| horizon/context 隐藏在代码常数中 | `max_T=30`、policy `K=20`、Waypoint `K=40` 分散在实现中 | 新增 `waypoint_horizon`、`context_length`、`waypoint_context_length` 参数并写入 D4RL 配置 |
| reward 表示依赖隐式环境变量 | average/cumulative/both 通道由 `AVG_REWARD/CM_REWARD` 决定，CLI cumulative flag 没有进入 iterable dataset，评估仍可能选择另一种通道 | 新增 `reward_representation={average,cumulative,both}` 并贯穿 CLI、dataset metadata、Waypoint/RvS checkpoint、condition space 和评估；环境无关的转换函数稳定输出 1/1/2 通道；`0/false/no/off` 不再被误判为开启 |
| reward 训练/评估窗口语义不同 | dataset 对每个随机起点 `s` 把 `RTG[s]` 固定重复在该窗口中，在线评估却整局保留 `RTG[0]` | 在线维护 reward prefix；对 `s=max(0,t-K+1)` 重新计算窗口起点命令，并在当前 context 内重复，使 `t>=K` 后仍与训练窗口一致 |
| 提前终止轨迹的 average 分母不一致 | dataset 原来用实际 suffix 长度，评估把 scalar average 乘环境最大步数；在线终止前又不可能知道未来真实长度 | 引入并保存 `reward_episode_horizon`；统一定义 `average_s=C_s/(H_episode-s)`，提前终止后的未执行步视为零奖励；checkpoint 与评估环境 horizon 冲突时报错 |
| legacy reward flags 与显式配置冲突被静默吞掉 | dataset、policy、step 对 `AVG_REWARD/CM_REWARD` 的优先级不同；cumulative flag 会覆盖显式枚举 | 集中使用同一 resolver；两个 legacy flag 同时出现时报错；`cumulative_reward_to_go` 与非 cumulative 显式表示冲突时报错，不再静默覆盖 |
| 调用者的 conditioning 被环境名覆盖 | `train.py` 无条件按名称改写 `reward_conditioning`，导致 unconditional/GCSL 等入口失真 | 删除覆盖，CLI/配置成为唯一权威来源 |
| 训练依赖不完整 | 源码直接 import ConfigArgParse、SB3、seaborn、dm-control 等，但 requirements 未声明 | 补齐兼容版本并增加 Python 3.8 的 `environment.yml` |

## 4. 数据流与 shape

令：

- `B`：batch size；
- `K`：context length；
- `H_w`：waypoint horizon，默认 30；
- `H_e`：reward episode horizon，来自配置/环境 TimeLimit 并写入 checkpoint；
- `S`：state dimension；
- `A`：action dimension；
- `G`：global goal/return condition dimension；
- `W`：waypoint dimension，官方 KForward 网络中 `W = G`；
- `E`：Transformer embedding dimension；
- `V`：Transformer output/MLP input width，默认 512。

### Dataset structured batch

`D4RLIterableDataset.sample_components()` 返回：

| 字段 | shape | dtype/含义 |
| --- | --- | --- |
| `states` | `[B,K,S]` | `float32`，左侧补零 |
| `actions` | `[B,K,A]` | `float32`，padding 值保持官方 `-10` |
| `rewards` | `[B,K,1]` | `float32`，原始即时 reward |
| `goals` | `[B,K,G]` | `float32`，重复的 global goal 或初始 RTG |
| `waypoints` | `[B,K,W]` | `float32`，horizon-aligned 目标 |
| `mask` | `[B,K]` | `bool`，`True` 表示有效 timestep |

其中：

```text
Goal Waypoint_t   = achieved_goal[min(t + H_w, episode_last)]
Reward Waypoint_t = RTG[min(t + H_w, episode_last)]
```

`reward_representation` 决定 `G/W`：

```text
average    -> 1 channel
cumulative -> 1 channel
both       -> 2 channels（官方历史默认）
```

令 `C_s=sum(reward[s:episode_last+1])`，则 reward 通道统一为：

```text
average_s    = C_s / (H_e - s)
cumulative_s = C_s / train_normalization_const
both_s       = concat(average_s, cumulative_s)
```

`normalization_const` 仅由 training trajectories 统计。提前终止后到 `H_e`
之间等价于零奖励，因此 average 不依赖在线阶段尚不可知的实际终止长度。
policy 输入的 `goals[:, t]` 是当前窗口起点 `s` 的命令在 K 个 timestep 上重复；
在线评估在 context 滑动后使用 `s=max(0,t-K+1)` 和 reward prefix 更新该命令。
`waypoints[:, t]` 才是未来 `t + H_w` 的中间 RTG 目标。

legacy DataLoader API 保持三元组：

```text
policy training:
  obs_goal [B,K,S+G], actions [B,K,A], mask [B,K]

waypoint training:
  obs_goal [B,K,S+G], waypoints [B,K,W], mask [B,K]
```

### Waypoint network

```text
[B,K,S+G] -> KForwardGoalNetwork -> [B,K,W]
                                     |
                                     +-- mask [B,K] before mean reduction
```

### Policy Transformer

没有 waypoint checkpoint 时：

```text
Transformer input = obs_goal [B,K,S+G]
```

启用 learned/manual waypoint 时：

```text
Waypoint output   = [B,K,W]
Transformer input = concat(obs_goal, waypoint) = [B,K,S+G+W]
```

普通 token 模式：

```text
tokens [B,K,E], attention mask [B,K]
```

`MODEL_ACTIONS` 模式：

```text
tokens [B,2K,E] = (obs_0, action_0, ..., obs_{K-1}, action_{K-1})
attention mask [B,2K] = repeat_interleave(mask, 2)
```

当前兼容路径的 GPT 输出 reshape 为 `[B*K,V]`，SB3 产生
`log_prob [B*K]`，随后由 `mask.reshape(B*K)` 在 reduce 前筛选。

## 5. 维度与 checkpoint 管理

维度来源现在为：

| 字段 | 来源 |
| --- | --- |
| `state_dim` | `environment.observation_space` / dataset metadata |
| `action_dim` | `environment.action_space` |
| `goal_dim` | dataset condition metadata |
| `waypoint_dim` | dataset metadata；官方网络要求等于 `goal_dim` |
| policy context | `context_length` 配置，默认 20 |
| waypoint context | `waypoint_context_length` 配置，默认 40 |
| waypoint horizon | `waypoint_horizon` 配置，默认 30 |
| waypoint hidden width | `waypoint_hidden_size` 配置，默认 1024 |
| reward channels | `reward_representation` 与 dataset metadata |
| reward normalization | training split metadata，并保存到 RvS checkpoint |
| reward episode horizon | `reward_episode_horizon`；默认读取环境 TimeLimit，并保存到 Waypoint/RvS checkpoint |

`goal_network_ckpt` 现在是显式 CLI/API 参数，并随 `args.json` 与 RvS checkpoint
hparams 保存。`GOAL_NETWORK_CKPT` 只在 CLI 解析时提供默认值，不会在模型构造
或 checkpoint 重载时意外启用 Waypoint；checkpoint 文件名也不再用于猜测维度。
阶段 0.6 以后保存的 Waypoint checkpoint 自带 architecture 与 reward semantic
metadata，并校验
`obs_dim/goal_dim/hidden_dim/max_T/reward/reward_representation/
reward_normalization_const/reward_episode_horizon`。对旧 checkpoint，加载器先校验
其中已经存在的字段，只用显式 config 补真正缺失的字段，禁止覆盖冲突。

learned Waypoint 会在 policy 构造期 eager 注册并冻结，因此 policy state dict 在
加载前后拥有相同 key 拓扑；冻结参数也会绕过 SB3 的递归 orthogonal
initialization。外层 RvS state dict 已包含 `goal_network.*` 权重；Lightning 1.5
加载 outer checkpoint 时先按已保存的 resolved config 重建拓扑，再严格写入内嵌
权重，所以移动 outer checkpoint 后不再要求原 Waypoint 文件仍位于旧路径。正常
新建 policy 时仍必须提供有效源 checkpoint，路径错误不会静默回退为随机网络。
正式 Lightning 1.5 文件级 round-trip 仍需在完整目标环境中补测。

## 6. 依赖审计

### 推荐基线

```text
Linux/WSL2 Ubuntu
Python 3.8.x
Gym 0.18.3
Stable-Baselines3 1.2.0
Torch 1.13.0
PyTorch Lightning 1.5.10
```

环境文件：

```bash
conda env create -f environment.yml
conda activate waypoint-transformer
```

关键约束：

- 代码调用 SB3 `ActorCriticPolicy._get_latent` 私有 API，因此固定
  `stable-baselines3==1.2.0`，不能直接升级到 SB3 2.x；
- 官方代码使用旧 Gym API，不切换到 Gymnasium；
- `pytorch-lightning==1.5.10` 对 setuptools 有历史兼容要求；
- `mujoco-py==2.1.2.14` 需要旧 Cython 和 MuJoCo 2.1 原生运行库；
- Kitchen 还需要 `dm-control==1.0.3` 与 `mujoco==2.3.0`；
- D4RL 的 `pybullet/click/termcolor` 已显式固定；
- GCSL 继续使用 requirements 中锁定的 Git commit。

任务依赖分层：

| 层级 | 主要依赖 |
| --- | --- |
| WT 核心张量 | torch、numpy、Lightning、SB3、Gym |
| 启动/记录 | ConfigArgParse、wandb、pandas |
| 可视化/分析 | matplotlib、seaborn、tqdm、scipy、scikit-learn |
| D4RL 数据 | d4rl、h5py、click、termcolor |
| AntMaze/MuJoCo locomotion | mujoco-py、MuJoCo 2.1、pybullet、系统 OpenGL/GLFW |
| Kitchen | dm-control、mujoco 2.3 及其传递依赖 |
| GCSL | 固定 GCSL commit 及其 rlutil 依赖 |

## 7. 测试命令与结果

### CPU 合成数据单元/闭环测试

实际执行：

```powershell
$env:PYTHONPYCACHEPREFIX='..\decision-transformer-master\.stage06_pycache'
& '..\decision-transformer-master\.venv-dt-gym\python.exe' `
  -m unittest discover -s tests -v
```

结果：

```text
Ran 33 tests
OK
```

覆盖：

- seeded train/val trajectory split 可复现、互斥、并集完整；
- RvS-R/RvS-G 的 validation dataset 确实来自 validation paths；
- train/val 共用由训练轨迹计算的 reward normalization；
- flat dataset 不会把同一 episode 拆到 train/val 两侧；
- 尾部无 done marker 的 truncated trajectory 不丢失；
- 极小数据集独立 validation 约束；
- structured dataset 字段、shape、dtype；
- Goal/Reward Waypoint horizon 与 episode 末端 clamp；
- average/cumulative/both reward channel shape 与 policy condition space；
- 环境无关 reward target 转换在三种表示下稳定输出 1/1/2 通道；
- 名义 episode horizon 下的 early-terminal average RTG、滑动窗口命令与训练
  窗口完全一致；
- checkpoint/environment reward horizon 冲突与过短 horizon 会报错；
- legacy reward flags 冲突会报错，`false/0` 不会误启用，显式枚举/conditioning
  metadata 优先；
- train-split reward normalization 会写入 data-module/policy metadata；
- policy checkpoint 记录 reward representation、normalization、episode horizon、
  resolved Transformer config 与 Waypoint 来源；
- Waypoint checkpoint semantic metadata 与已有字段冲突保护；
- learned Waypoint 在首次 forward 前已注册、冻结，并可 strict state-dict
  round-trip；
- outer checkpoint 的 embedded restore 不会重新读取已丢失的 Waypoint 文件；
- SB3 初始化不会改写冻结的 pretrained Waypoint 权重；
- Waypoint loss 和 policy loss 的 padding 不变性；
- 全 padding Waypoint loss 可 backward 且无 NaN；
- GPT 普通 token 与 `MODEL_ACTIONS` token 的 padding attention；
- `context_length=1`、`MODEL_ACTIONS=true` 时动作历史清空且 GPT 前向 shape 正确；
- action dim 和 context length 来自配置；
- CLI 的 `reward_episode_horizon` 确实转发到 `run_training`；
- 真实 `step.py` 的三步 early-terminal rollout 验证在线 context 滑动后命令为
  `[2.0,1.0] -> [2.0,1.0] -> [2.25,0.9]`；
- dataset → Waypoint forward/loss/backward/optimizer step；
- GPT → masked loss/backward/optimizer step。

测试文件对 D4RL、GCSL、MuJoCo、Lightning、SB3 的“导入边界”使用最小 stub；
dataset、Waypoint 网络、policy loss、GPT 与 reward rollout 函数均来自本仓库
真实实现，rollout 使用三步 fake environment。这个测试是 CPU 逻辑/张量闭环，
不等价于完整 D4RL 环境集成测试。

此外，阶段 0.6 先前的隔离依赖检查以官方 Stable-Baselines3 1.2.0 源码执行了
真实
`RvS -> ExtendedActorCriticPolicy -> TransformerExtractor -> GPT ->
DiagGaussian` 定向检查：Box action、`B=2,T=5`、含左 padding，
`MODEL_ACTIONS` 关闭/开启和 AntMaze manual waypoint 三条路径均完成
forward、masked NLL 与 backward，action dimension 为 2/8 时都正确。该检查
仍未包含真实 Lightning Trainer 或 checkpoint 反序列化；当前 Python 3.10
环境缺少 SB3，不能在同一环境重复执行该项检查。

### Python 编译检查

```powershell
& '..\decision-transformer-master\.venv-dt-gym\python.exe' `
  -m compileall -q src tests
```

结果：`PASS`。测试时把 pycache 重定向到可写临时目录，避免污染仓库。

### Bash 启动脚本语法

```powershell
Get-ChildItem -Recurse -File experiments -Filter *.sh |
  ForEach-Object { & 'E:\Git\bin\bash.exe' -n $_.FullName }
```

结果：`20/20 PASS`。

### diff whitespace

```powershell
git diff --check
```

结果：`PASS`。

## 8. 当前仍未解决/未验证

1. **完整 D4RL 训练尚未在本机跑通。** 当前可用环境是 Python 3.10.20；
   `mujoco_py` 已存在，但缺少 ConfigArgParse、Lightning、SB3、dm-control、
   `mujoco` Python 包、GCSL 和 pybullet。`pip check` 还报告当前 D4RL 缺少
   dm-control 与 pybullet。执行
   `PYTHONPATH=src python -m wt.train --help` 会在导入阶段报
   `ModuleNotFoundError: configargparse`。requirements 已修复，但没有把
   不兼容的旧栈强行安装进现有 Python 3.10 环境。
2. **原生 Windows 不是正式 D4RL 复现目标。** `mujoco-py` 2.x 不支持现代
   原生 Windows 安装；正式 AntMaze/Kitchen/MuJoCo 验证应使用 WSL2/Linux。
3. **尚未用真实 Lightning 1.5 完成文件级 checkpoint round-trip 和 benchmark。**
   目前已验证 eager 子模块注册、strict state-dict round-trip、checkpoint
   metadata 与真实 SB3 前向路径，但本机缺少正式 Lightning 1.5/D4RL 栈，因此
   尚未执行 Trainer 保存 → 清空进程状态 → 文件重载；也没有产生可用于论文对比
   的 checkpoint、return 或成功率。
4. **源码仍有 eager import 耦合。** `train.py`、`step.py` 会在启动时导入
   GCSL、D4RL、Kitchen 和分析模块，所以完整 CLI 启动需要大部分任务依赖。
   本阶段没有进行超出范围的模块重构。
5. **SB3 私有 API 仍是历史兼容点。** 当前实现保留官方 RvS 的
   `_get_latent` 路径，因此 requirements 必须保持 SB3 1.2.0。

在 WSL2 环境和真实 D4RL 数据就绪后，建议补做：

```bash
python -m pip check
PYTHONPATH=src python -m wt.train \
  --configs experiments/config/d4rl/antmaze_rvs_g.cfg \
  --env_name antmaze-umaze-v2 \
  --seed 0 \
  --max_steps 1 \
  --checkpoint_every_n_steps 1
```

建议同时设置 `WANDB_MODE=offline`，并检查 `env.get_dataset()`、一个 validation
batch、一个 checkpoint 保存与重新加载。

## 9. 阶段 1 的接口边界

从代码接口与 CPU 单 batch 闭环看，可以进入阶段 1；从真实 Trainer/CLI 与
D4RL 基准运行验收看，仍需先在目标 WSL2/Linux 环境安装固定依赖并补测。
后续海上环境应适配现有接口，而不是重新把任务维度写入 Transformer：

- 环境通过 `observation_space` 和 `action_space` 暴露 `S/A`；
- dataset adapter 输出
  `states/actions/rewards/goals/waypoints/mask`；
- 明确提供 `goal_dim`、`waypoint_dim` 和 achieved-goal 映射；
- episode terminal/truncation 必须在构造 horizon target 前分段；
- Waypoint horizon 继续由配置传入；
- 新动作空间不得修改 GPT 内部常数，只能通过 environment/config 传递；
- 海上 reward/goal 的语义变化应发生在 dataset/environment adapter，
  不应破坏当前 policy 与 attention mask 合约。

因此，阶段 1 的“代码接口开发”已具备前提；“论文官方 D4RL 结果已复现”仍不是
当前结论。
