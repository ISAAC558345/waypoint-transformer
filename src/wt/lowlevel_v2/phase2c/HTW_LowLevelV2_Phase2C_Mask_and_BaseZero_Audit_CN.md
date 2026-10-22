# HTW Low-Level V2 Phase-2C：Base-Zero Freeze 与 Active-Policy-Mask 审计（LL-V2.2C）

本阶段只做冻结代码/数据/checkpoint 审计。没有 `optimizer.step`，没有重新训练 seed0/1，没有训练 seed2/3/4，没有改 beta / epsilon / reward / lr / PPO clip / entropy / Actor / Critic，没有 relation 模块，没有访问 Phase-2 validation 222000–222063。

恢复对象：LL-V2.2B，门控 **P2B-B**。磁盘协议：`beta_residual=0.25`，`epsilon_base_fraction=0.05`，`a_max=1.0` 故 `epsilon_base=0.05`。design u100：seed0 freeze `0.9247`，seed1 freeze `0.9052`；全部 design 评估点 `cap_activation_rate=0`。

---

## 1. 训练期间 cap 是否真的触发？

**不能从训练 log 证明逐 update 触发次数。**  
`training_cap_activation_resolved=false`：Phase-2B training history 没有 `cap_activation_count/fraction` 字段，也没有保存训练 rollout。

冻结证据（design 全部 `u0/u20/u40/u60/u80/u100`，以及 train16 的 u100）：**cap_activation_count=0，fraction=0**。  
Phase-2B design eval 曲线上每个记录点的 `cap_activation_rate` 也全是 0。

归因选择：**A。cap 在已观测的开发/冻结评估中基本从未生效。**  
不能声称“训练时曾经生效、只是 final evaluation 没生效”，因为训练 rollout 未保存。

## 2. 最终 cap_activation=0 说明什么？

在 `policy_active_mask=1` 的状态下，`||r_raw|| / (||a_base|| + eps)` 的 p99 为：

- seed0 design：0.123 < 0.25
- seed1 design：0.091 < 0.25
- frac_gt_beta = 0

因此 **beta=0.25 的 Euclidean norm cap 在可学习子集上是冗余的**。最终 `cap_activation=0` 不是统计漏记；实现把 cap 标在 `r_norm > beta * ||a_base||` 且非 freeze 上，这些状态根本没越界。

## 3. 90% 以上 base-zero 主要由哪些状态组成？

对 **pre-action** 几何（恢复位置与 queue）的互斥分解（design u100，占全部 base-zero）：

| 原因 | seed0 | seed1 |
|---|---|---|
| IDLE_OPTION | 59.8% | 61.1% |
| SERVICE_LOCK | 40.2% | 38.9% |
| EMPTY_QUEUE / INVALID / ALREADY_REACHED / OTHER | 0 | 0 |
| MULTI_REASON | 0 | 0 |

train16 同样是 **idle ≈ 59–60% + service ≈ 40–41%**，没有第三类。

结论：约 90% freeze = **高层 idle（约 60%）+ 在服务区内的 service lock（约 40%）**。不是 empty queue，也不是神秘的 BASE_EXECUTOR_ZERO_OTHER。

## 4. Service lock 与 base-zero 重叠多少？

design u100：

**seed0**：P(bz)=0.925，P(sl)=0.371，P(bz ∧ sl)=0.371，P(bz ∧ ¬sl)=0.553，P(sl ∧ ¬bz)=0。  
P(bz|sl)=1，P(sl|bz)=0.402。

**seed1**：P(bz)=0.905，P(sl)=0.352，P(bz ∧ sl)=0.352，P(bz ∧ ¬sl)=0.553，P(sl ∧ ¬bz)=0。  
P(sl|bz)=0.389。

**L0 design**（无 trust-region）：P(sl)=0.363，P(bz)=0。

解释：service lock **被 base-zero 完全覆盖**（在服务时 `a_base` 也 ≈0，所以 freeze 与 lock 重复）。但 **base-zero 不是 service-lock 的重复**：一半以上 freeze 来自 idle。新机制的增量主要是把 idle（以及其余 `||a_base||≤0.05`）从 Actor 更新中拿掉；服务区内本来已被 ServiceResidualLock 置零。

## 5. Actor 真正 active fraction 是多少？

`policy_active_mask = NOT(service_lock OR base_zero_lock)`。

训练 log（100 update，真实训练 mask 均值）：

- seed0：mean 7.21%，min 2.86%，max 13.11%，p10 4.34%，p50 6.99%，p90 10.43%
- seed1：mean 7.38%，min 2.60%，max 12.15%，p10 4.75%，p50 7.20%，p90 10.34%

冻结 design u100：seed0 7.53%（347/4608），seed1 9.48%（437/4608）。  
冻结 train16 u100：6.23% / 7.36%。

**Actor 大约只在 7–10% 的 UAV-step 上学习。**

## 6. policy loss 当前除数是 total 还是 active samples？

**ACTIVE_SAMPLES（形式 B）。**

`policy_loss = sum(mask * per_uav) / max(sum(mask), 1)`  
`actor_mask_denominator = ACTIVE_SAMPLES`  
不是 `mean(mask * loss_i)`。

## 7. entropy 除数是什么？

**同样是 ACTIVE_SAMPLES。**  
`entropy_term = sum(mask * entropy) / max(sum(mask), 1)`。

## 8. 是否存在 zero-active minibatch？

冻结 replay（与训练相同的 `rollout_length=48, num_envs=4, num_minibatches=4`，UAV-step 每批 288）：**zero-active minibatch count = 0**。

但 active 很少且不均：

- 合并 20 个诊断 minibatch：`<5%` 共 4 个（全部 seed1）；`<10%` 共 17 个。
- seed1 最小 active samples = 7（2.4%）。
- 没有空批，但大量 batch 只有几十个有效 UAV-step。

原训练 minibatch 未保存；以上是冻结 forward/data pipeline 重放，**不是训练**。

## 9. Advantage normalization 是否包含 locked samples？

**是，形式 A。**  
GAE 在全部 env-step 上 `(A - mean) / std`，再 broadcast 到 6 个 UAV；mask 只进 policy/entropy。

冻结 replay 上，在“至少一架 UAV active”的 env-step 子集中：all-sample 归一化后的 advantage 均值 seed0 为 −0.605（相对 active-only 的 0），seed1 为 +0.088。locked 多数 env-step **会平移真正可控样本的 advantage 尺度**，尤其 seed0。

## 10. 当前 Actor 梯度被 mask 缩小多少？

相对 **正确的 active 归一化**：`G_current / G_active_normalized = 1`（10+10 个冻结 minibatch）。**没有**被 total-sample 除数缩小。

若错误地用 `mean(mask * loss)`（形式 A），梯度大约会再乘 `active_fraction`，诊断比 `G_current / G_totalmean` 约 **9–41×**（随该批 active 比例变化）。那不是当前代码。

## 11. seed0/seed1 active fraction 是否明显不同？

**训练均值几乎相同（7.21% vs 7.38%）。** 不能把 seed 差异归因于长期 active 比例不同。

有效更新代理 `active_fraction * grad_norm` 均值：seed0 1.47 vs seed1 0.64。seed0 均值被个别极大 `gradient_norm_preclip`（例如 update 55 ≈ 487，update 92 ≈ 384）拉高。mean KL 0.019 vs 0.015，clip fraction 0.170 vs 0.157，接近。seed 性能差更可能来自轨迹/优势噪声，而不是稳定的 active 比例差。

## 12. base-zero 状态里 Actor raw residual 多大？

冻结 design，`r_raw = α tanh(pre)`，entropy 由全局 `log_std` 决定（同类状态几乎常数）：

- seed0：SERVICE 与 IDLE 的 mean `||r_raw||` 都约 **0.059**。
- seed1：SERVICE 约 **0.0048**，IDLE 约 **0.077**。

Actor **会在不可控状态输出非零 residual**；seed0 在 lock/idle 上几乎一样大。这些状态已 mask Actor loss，长期随机/共享骨干泄漏属于预期，不能当策略错误。seed1 在 service 上 raw 更小，说明骨干仍能把服务几何和 idle 分开，尽管 service 不进 policy loss。

## 13. base-zero freeze 是否误杀有价值动作？

目标集合：`base_zero AND NOT service_lock AND NOT idle`。  
design u100 上 **种群为空（0 步）**。所有 `¬service_lock` 的 freeze 都是 **IDLE_OPTION**。

一步 C0/C1 反事实因此没有非 idle 样本可评。局部诊断结论：freeze 没有额外锁住“有任务、不在服务、但 base 很小”的飞向目标步；那些步要么 `||a_base||>ε`（active），要么 idle。**未观察到误杀有价值动作的样本。** 这是局部几何事实，不是正式性能结论。

## 14. 为什么 beta=0.25 从未触发？

Active 子集上 `||r_raw|| / ||a_base||` 的 p99 < 0.13 < 0.25，max < 0.13。raw residual 相对仍在运动的 base 太小。cap 条件 `||r|| > 0.25 ||a_base||` 从未成立。不是实现把 cap 算错：若大量 ratio>0.25 而 cap 仍为 0 才会报警；这里 **frac_gt_beta=0**。

## 15. Phase-2B 真正起作用的机制是什么？

不要再统称 “Trust Region 成功”。

- **动作约束：A. BaseZeroResidualFreeze。** Norm cap 未绑定（非 B、非 C）。
- **优化侧：policy/entropy 已按 active 归一化（非 MASK-A），但只有 ~7% UAV-step 进 Actor，构成 MASK-B 噪声。**
- 不能把 payload 改善归因于 `beta_residual=0.25`。
- seed1 vs L0 的 +1.8125e6 与旧 seed1 的 +3.75e6，更合理的机制叙述是：冻结 idle/近零 base 上的 residual，并把 Actor 更新限制在真正飞向目标的少数步。

## 16. MASK Gate？

**MASK-B。**

不是 MASK-A（除数已经是 active）。  
不是 MASK-C：active 过少（~7%）、大量 minibatch <10%、seed1 诊断批可低至 2.4%，足以造成 seed-dependent update noise。

## 17. BASEZERO Gate？

**BASEZERO-A。**

freeze 主体是 idle（无 option）与已在服务区（与 ServiceResidualLock 重复）的状态；非 idle、非 service 的 freeze 种群为空，没有证据显示误杀可获益飞行动作。

## 18. 下一步唯一 minimal change？

**按 MASK-B：只允许 active-sample-aware minibatching（或等价的单一优化机制）。**  
不改 beta，不改 epsilon，不改 reward，不重新设计 base-zero 条件（那是 BASEZERO-B 才允许的）。本阶段不执行该修改。

## 19. 本阶段是否训练？

**否。**

## 20. 是否进入 relation 增强？

**否。REL-1 保持。** 禁止 HGNN / attention / relation loss / collision reward / interference reward。问题仍是 Residual 训练稳定性与有效样本构造。

---

## 产物

目录：`src/wt/lowlevel_v2/phase2c/` 与 `outputs/`、仓库根目录副本。

- `LLV2_PHASE2C_PROTOCOL.json`
- `LLV2_PHASE2C_CAP_ACTIVATION.csv`
- `LLV2_PHASE2C_BASE_ZERO_DECOMPOSITION.csv`
- `LLV2_PHASE2C_LOCK_OVERLAP.json`
- `LLV2_PHASE2C_ACTIVE_SAMPLE_STATS.csv`
- `LLV2_PHASE2C_MASK_REDUCTION_AUDIT.md`
- `LLV2_PHASE2C_MINIBATCH_ACTIVE_STATS.csv`
- `LLV2_PHASE2C_ADVANTAGE_NORMALIZATION.md`
- `LLV2_PHASE2C_GRADIENT_SCALING.csv`
- `LLV2_PHASE2C_BASEZERO_COUNTERFACTUAL.csv`
- `LLV2_PHASE2C_RESIDUAL_BASE_RATIO.csv`
- `LLV2_PHASE2C_MECHANISM_GATE.json`

辅助：`LLV2_PHASE2C_UPDATE_INTENSITY.csv`，`LLV2_PHASE2C_BASEZERO_RAW_RESIDUAL.csv`，`LLV2_PHASE2C_ADVANTAGE_REPLAY.json`。

Phase-2C 停止。
