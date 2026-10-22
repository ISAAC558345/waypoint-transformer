# 旧低层 PPO 与 L3-B 兼容性

**结论：结构/输入需要独立适配后才能训练**（`needs_independent_adapter`）。
- Phase-6.2 checkpoint `E:\srdp\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\waypoint_transformer_official\src\wt\htw_phase63\outputs\checkpoints\low_level_c05_d005.pt` 存在：True
- 问题清单：
  - {'item': 'num_uavs', 'phase62': 5, 'l3b': 6, 'ok': False}
  - {'item': 'num_buoys', 'phase62': 12, 'l3b': 12, 'ok': True}
  - {'item': 'option_uav_dim', 'phase62': 21, 'l3b_needed': 21, 'ok': True}
  - {'item': 'actor_max_uavs', 'phase62': 5, 'l3b': 6, 'ok': False}
  - {'item': 'sinr_in_low_level_reward', 'phase62': False, 'l3b': True, 'ok': False}
  - {'item': 'checkpoint_on_disk', 'path': 'E:\\srdp\\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\\waypoint_transformer_official\\src\\wt\\htw_phase63\\outputs\\checkpoints\\low_level_c05_d005.pt', 'ok': True}
  - {'item': 'hardcoded_MAX_UAVS_in_phase61_features', 'value': 5, 'ok': False}
  - {'item': 'trained_env', 'phase62': 'L2AEnv M=5 no SINR', 'l3b': 'L3BEnv M=6 SINR', 'ok': False}
  - {'item': 'phase62_association_source', 'phase62': 'oracle_option_assignment inside GoalOptionVector', 'l3b_required': 'frozen high-level association; low-level movement only', 'ok': False}
  - {'item': 'goal_reward', 'phase62': 'r_comm + lambda_c * in_range_completion - lambda_d * dist_slots on L2-A', 'l3b': 'SINR-scaled service; D=3 not a fixed three-step finish', 'ok': False}
- 可复用：Round-30 PPO core (GAE/clip/Adam) as an algorithm, not as loaded weights；goal-conditioned idea: concatenate option one-hot onto UAV features；completion + distance auxiliary reward template from Phase-6.2
- 必须适配：MAX_UAVS 5 -> L3-B M=6 (hard-coded pads and actor sharing)；option feature attachment currently rejects non-(5,21) UAV stacks；SINR-aware service: D=3 is not a guaranteed three-step completion；Phase-6.2 checkpoint exists on disk but must not be loaded into an L3-B M=6 actor；refuse silent fly-toward fallback labeled as PPO；utility labels in 9.4.5 were generated under fly-toward, not a learned executor
- 本阶段未训练、未随机初始化评估、未在加载失败时静默退回 fly-toward 并标成 PPO。

## 后续低层实验计划（不执行）
- B0：冻结 DeepSets-util 五种子 + fly-toward。
- B1：同一冻结高层 + L3-B 目标条件低层 PPO（需先完成 M=6 特征适配后再训）。
- 同环境、同 hold、同评估初始状态与到达过程；各自闭环允许高层按各自状态重决策，不强行对齐 assignment。
- 先用 1 个 design seed、极少 updates 做 smoke 测墙钟，再估全量；禁止预设「PPO 必须超过规则」。规则足够则可保留 B0。
- 指标必须区分：进入服务区、持续有效服务、实际完成块、累计 payload；SINR 下 D=3 不是固定三步完成。
