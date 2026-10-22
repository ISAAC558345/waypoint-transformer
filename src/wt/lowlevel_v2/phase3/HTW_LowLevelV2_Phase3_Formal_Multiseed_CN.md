# HTW Low-Level V2 Phase-3：正式五种子独立验证（LL-V2.3）

本阶段只验证、不修改方法。冻结对象为 Phase-2D 正式方法（磁盘 gate **P2D-A**，`active_sample_optimization_supported=true`）。未改 Active-Sample-Aware Actor Optimization、ServiceResidualLock、BaseZeroResidualFreeze、reward、lr、clip、entropy、Actor/Critic、Neighbor Encoder、SafetyProjector、`d_safe`、`alpha_residual`、`epsilon_base`、`beta_residual`、高层、Task Option、hold。未增加 relation 模块、collision reward、interference penalty、训练预算。Primary checkpoint 一律为 **u100 final**，未按 validation 挑选中间 checkpoint。未访问 Phase-2 independent validation `222000–222063`、Phase-10.6 `106000–106063`、以及全部历史 confirmation / final untouched。

正式低层结构：

Frozen DeepSets High-Level → Task Option → fly-toward base → Residual MAPPO Actor → ServiceResidualLock → BaseZeroResidualFreeze → BaseRelativeResidualCap → SafetyProjector → ServiceSafeEnvelope → environment。

Actor 优化冻结为 Active-Sample-Aware：advantage 归一化与 minibatch 仅用 `policy_active_mask=1`；每 PPO epoch 每个 active sample 只用一次；Critic 仍用全部 transition。

`beta_residual=0.25` 保留以维持方法定义。本阶段五 seed 的 cap activation 仍为 0，故 **`residual_norm_cap_empirically_active=false`**，**`residual_norm_cap_redundant=true`**。beta-cap **不是**主要贡献。

超参数与 Phase-2D 正式配置一致：`d_safe=1.0 m`，`alpha_residual=0.25`，`service_margin=0.5 m`，`epsilon_base_fraction=0.05`，`beta_residual=0.25`；reward `r = Δpayload_bits/1e6 − 0.05 C_project − 0.01 C_residual`；MAPPO `γ=0.99`，`λ_GAE=0.95`，clip `0.2`，lr `3e-4`，value `0.5`，entropy `0.01`，grad clip `0.5`，rollout `48`，`num_envs=4`，`update_epochs=4`，`updates=100`；Actor 每 epoch 最多 4 个非空 active-only minibatch。

高层：DeepSets-util seed0，checkpoint SHA `dbd6196b23c111a88a4d86a9054ca1709e459e5b035c1d9210639bedd2dda109`，训练前后一致，`requires_grad=false`。

Independent validation：`LLV2_PHASE3_INDEPENDENT_VALIDATION_V2`，64 个 scenario `223000–223063`。全库 train/design/validation/confirmation/untouched 扫描后无交集（blocked universe 2809 个历史 seed）。`LLV2_PHASE3_VALIDATION_V2_consumed=true`。L0 为 gated fly-toward（无 Residual RL），与五 seed 共享同一 64 scenario 的初始状态与到达实现；高层允许在 hold 边界重新 assignment。

---

## 29 问正式回答

1. **最终 5 个低层 seed？**  
   seed0、seed1、seed2、seed3、seed4。seed0/seed1 复用 Phase-2D `update100`；seed2/3/4 从头初始化 Actor/Critic 新训。全部进入最终统计，无剔除。

2. **seed2/3/4 是否与 seed0/1 完全同预算？**  
   是。每 seed：100 updates × 48 rollout × 4 envs = **19,200** environment transitions = **115,200** agent transitions。未加 update、未改单 seed 配置。

3. **新的 independent validation 有多少 scenario？**  
   **64**。

4. **是否与全部历史 scenario 无交集？**  
   是。与 `LLV2_train`、`LLV2_design`、Phase-2 `222000–222063`、Phase-10.6 `106000–106063` 及扫描到的全部历史 seed 无交集。

5. **L0 mean payload？**  
   **9.9375e6 bits**（9.9375 完成块）。

6. **五 seed 各自 RL mean payload？**  
   - seed0：**10.6875e6**  
   - seed1：**10.578125e6**  
   - seed2：**10.625e6**  
   - seed3：**10.500e6**  
   - seed4：**10.4375e6**

7. **五个 Delta_s？**  
   - seed0：**+0.750e6**  
   - seed1：**+0.640625e6**  
   - seed2：**+0.6875e6**  
   - seed3：**+0.5625e6**  
   - seed4：**+0.500e6**  
   L0 是每 scenario 共享参考，未复制 5 次当独立 baseline。

8. **正向 seed 数量？**  
   **5/5**。

9. **overall mean delta？**  
   **+0.628125e6 bits**。

10. **overall median delta？**  
    **+0.640625e6 bits**（五 seed 均值的中位数）。scenario-pair 中位数为 +1.0e6。

11. **hierarchical 95% CI？**  
    **[+0.4187e6, +0.8313e6]**，下限 > 0。

12. **seed-cluster 95% CI？**  
    **[+0.550e6, +0.703e6]**，下限 > 0。scenario-paired 辅助 CI：[+0.297e6, +0.978e6]。

13. **completed blocks？**  
    L0：9.9375。RL：10.6875 / 10.578 / 10.625 / 10.500 / 10.438。总体不恶化。

14. **arrival time（mean arrival steps）？**  
    L0：5.542。RL：5.466 / 5.500 / 5.766 / 5.883 / 5.776。与 L0 同量级，无系统性变差。

15. **collision count？**  
    五 seed 均为 **0**。

16. **minimum separation（worst）？**  
    五 seed 均为 **≥ 1.0 m**（最差约 `1.0 − 1e-9`，冻结 float 容差内）。

17. **projector failure？**  
    五 seed 均为 **0**。service-lock violation 亦为 **0**。

18. **active fraction？**  
    训练期均值约 **0.070–0.074**（seed0–4：0.0725 / 0.0733 / 0.0703 / 0.0736 / 0.0732）。验证期 `base_zero_freeze_rate ≈ 0.931–0.934`，与 Phase-2D 机制一致：多数步被 lock/freeze，少数 active sample 驱动 Actor。

19. **Actor minibatch 是否正常？**  
    是。五 seed：每 update **4** 个非空 active minibatch；min size 均值约 20；**zero-active minibatch = 0**。active advantage 归一化后 mean≈0、std≈1。

20. **Actor KL / clip / gradient 是否跨 seed 稳定？**  
    是。mean KL ≈ 0.012–0.014，|KL|max ≈ 0.07–0.12；clip fraction ≈ 0.15–0.18；Actor grad mean ≈ 7.8–9.2，max ≈ 12.7–17.0。**未出现数百量级 gradient spike，无严重 KL 异常，无 entropy collapse。**

21. **beta cap 是否仍几乎不触发？**  
    是。五 seed 验证 `cap_activation_rate=0`。正式标记 **`residual_norm_cap_redundant=true`**。后续方法描述不再把 beta-cap 当关键模块。

22. **neighbor 是否真正被利用？**  
    **否（作为动作决定因素几乎不被用）。** neighbor shuffle 的 Actor pre-tanh L2 ≈ 1e-9；neighbor zero 约 0.008，且 near 并不高于 far。REL-1 诊断状态未改。

23. **interference / SINR 是否真正被利用？**  
    **interference：基本不。** high-interference shuffle L2 ≈ 1e-7。  
    **SINR/rate：有输入敏感。** sinr_shuffle 全体 mean L2 ≈ 0.15，sinr_zero 在 high-SINR 约 0.29。这是观测特征敏感，**不能**等同于主动干扰优化。

24. **是否出现主动 separation response？**  
    **没有可靠证据。** near-neighbor 上 `dot(a_des−a_base, v_away)` 均值接近 0，median=0，正方向比例仅约 1–3%。**collision=0 由 SafetyProjector 保证，不能写成 Actor 主动避碰。**

25. **是否出现主动改善 SINR/rate 倾向？**  
    **没有。** 用 **pre-projector** `a_des` vs `a_base` 预测：五 seed 的 high-interference `ΔSINR=Δrate=Δinterference=0`。不能用 projected action 主张主动通信优化。

26. **MS2 Gate = A/B/C？**  
    **LLV2-MS2-A**。mean>0，两条主 CI 下限>0，5/5 seed 正向，median>0，全安全，completed blocks 不恶化，Active-Sample Actor 跨 seed 正常。

27. **stable_residual_advantage？**  
    **true**。

28. **REL2 Gate = 0/1/2？**  
    **REL2-0**。neighbor / interference 多数 seed 基本不影响 Actor。SINR 敏感不改变 REL2（REL2 定义在 neighbor/interference 关系关键状态）。Relation Gate **不能**替代 Performance Gate。正式关系状态仍为 **REL-1**（未改关系网络）。

29. **下一阶段唯一建议？**  
    MS2-A **+** REL2-0：基础 Residual MAPPO 在独立 64 scenario、5 seed 上稳定优于 L0。  
    **允许（非自动）**下一阶段专门增强 collision / interference relation utilization。  
    **本阶段停止。不自动修改 relation 模块。**

---

## Design 学习曲线（仅机制观察，不用于选模）

磁盘 `LLV2_PHASE3_EVAL_CURVE.csv`（design，非 validation）。Primary 仍是 u100。

| seed | u0 | u20 | u40 | u60 | u80 | u100 |
|---|---|---|---|---|---|---|
| 0（P2D） | 10.375e6 | 10.188e6 | 10.812e6 | 10.875e6 | 10.875e6 | 10.875e6 |
| 1（P2D） | 10.375e6 | 10.625e6 | 11.000e6 | 11.438e6 | 11.125e6 | 11.188e6 |
| 2 | 10.375e6 | 11.062e6 | 10.875e6 | 10.750e6 | 10.562e6 | 10.750e6 |
| 3 | 10.375e6 | 11.125e6 | 11.625e6 | 11.125e6 | 10.688e6 | 10.875e6 |
| 4 | 10.375e6 | 10.250e6 | 11.312e6 | 11.375e6 | 10.812e6 | 10.875e6 |

Design L0 参考 10.375e6。中间点仅观察，未用于 Phase-3 模型选择。

---

## 安全与残差摘要

- collision=0，projector_failure=0，service_lock_violation=0，全体 seed。  
- projector intervention（验证）：约 0.55%–1.87%。  
- service_lock_rate ≈ 0.377–0.381；used residual ≪ raw residual。  
- 证据支持的机制：ServiceResidualLock、BaseZeroResidualFreeze、Active-Sample-Aware Actor Optimization、Residual MAPPO、SafetyProjector 硬安全。  
- 无证据支持 beta-cap 性能贡献。

图：`src/wt/lowlevel_v2/phase3/figures/`（payload 对比、per-seed delta、hierarchical CI、learning curves、active fraction、Actor grad/KL、min sep、projector、relation sensitivity）。
