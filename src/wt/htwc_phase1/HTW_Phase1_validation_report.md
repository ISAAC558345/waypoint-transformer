# HTW Phase-1 Validation Report

本报告对应 **mini-budget 验证**，不是 Round-30 规模正式训练。未访问 fresh/untouched/protected JSONL，未修改 HGNN、环境、reward、PPO 核心公式。

## 1. 三个 baseline 结构

| Baseline | 名称 | UAV 输入维 | TW 来源 |
|---|---|---:|---|
| 1 | State-only PPO | 8 | 无 |
| 2 | Original Task-Waypoint PPO | 19 | 冻结 Round-21 DeepSets `phi=[service,progress,return]`，不重训预测器 |
| 3 | New HTW Task-Waypoint PPO | 16 | warmup 后的 TW_new 预测器 `goal,waypoint,horizon` |

## 2. 输入差异

- Baseline 1：`feature_dict` 原始 8 维 UAV 特征。
- Baseline 2：拼接 11 维 TW_old（每机 service5 + progress5 + return1）。
- Baseline 3：拼接 8 维 TW_new（goal5 + waypoint2 + horizon1）。
- 环境、reward、PPO clip/GAE/optimizer、seed、episode 评估集相同。

## 3. 训练配置

- PPO：`rollout_length=16`, `num_envs=2`, `update_epochs=1`, `clip_eps=0.2`。
- Mini-formal 更新次数：8 / baseline（环境步约 256）。
- Seed：0。训练场景：`training_scenarios()[:16]`；评估：随后 4 个训练场景（非 fresh、非 untouched）。
- TW_new predictor warmup loss 末值：1.098182。

## 4. Payload 结果

- State-only：34310913.765 bits
- Old TW：34204515.575 bits
- New TW：34310913.765 bits

## 5. Reward 结果

- State-only return：3.673736
- Old TW return：3.625093
- New TW return：3.650085

## 6. Queue 结果

- State-only residual queue：2925000.000
- Old TW residual queue：3031398.190
- New TW residual queue：2925000.000

## 7. Action relevance

- Old TW mean rho：0.04222458577833025
- New TW mean rho：-0.45274101296916225
- rho_new > rho_old：false

Fairness / energy：

- Jain fairness New/Old/State：0.9951 / 0.9948 / 0.9928
- Energy efficiency New/Old/State：12106046.0099 / 11691269.7827 / 12988782.8858

## 8. Waypoint 有效性

- New TW mean ||W-p||（归一化坐标）：0.5788689889947521
- New TW 动作与 waypoint 方向夹角均值（度）：119.6494392149114

## 9. 结论

**New Task-Waypoint 在本次 mini-budget 下是否被证明优于 Old Task-Waypoint：否。**

补充观测：

- New TW 与 State-only 的 cumulative payload、residual queue **完全相同**（4 个评估 episode），说明在该预算下匹配/队列动力学主导收益，运动条件几乎没有改变服务结果。
- New TW 的 mean rho 为 **-0.45**，动作与预测 waypoint 平均夹角约 **120°**，即低层没有跟踪 subgoal，反而大致反向。这更支持“low-level 无法利用 TW / warmup 预测器尚未给出可执行方向”，而不是“表示已经有效但网络不够深”。
- Old TW 的 mean rho 接近 0（0.04），同样几乎无方向引导，但至少不是系统性反向。

说明：

- mini-budget PPO 下 New TW 的 cumulative payload 未超过 State-only，也未相对 Old TW 形成可解释的控制优势。
- mean rho_new 未大于 rho_old。
- TW_new 预测器仅做了 planner 窗口上的短 warmup，不是正式监督训练。
- planner 为 independent_queue_rate，不是最优轨迹。
- PPO 更新次数远小于 Round-30 正式预算，结果只作接口级验证。

不要据此去改 HGNN 或加深网络。若继续：先用 **planner 教师 TW_new**（非学习预测器）做上界，检验低层能否跟踪；再决定是否进入 Hierarchical Controller。当前不进入 HGNN。

原始 JSON：`E:/srdp/NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference/waypoint_transformer_official/src/wt/htwc_phase1/outputs/validation_mini/summary.json`
