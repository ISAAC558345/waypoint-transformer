# HTW Low-Level V2 Phase-4A：碰撞关系内化（Safety-Projector Distillation）

基础 Residual MAPPO 冻结自 Phase-3（**LLV2-MS2-A**，`stable_residual_advantage=true`，**REL2-0**）。Phase-3 磁盘对照：L0 ≈ 9.9375e6；五 seed RL mean 10.688 / 10.578 / 10.625 / 10.500 / 10.438 ×1e6；Δ +0.750 / +0.641 / +0.688 / +0.562 / +0.500 ×1e6；5/5 正向；hierarchical 95% CI ≈ [+0.419, +0.831]e6；seed-cluster ≈ [+0.550, +0.703]e6；collision=0；min sep≥1.0 m；projector failure=0。

本阶段**只计划**解决 UAV-UAV 碰撞关系内化。通信干扰完全冻结。未改 reward、MAPPO lr/clip/entropy、Critic、高层、Task Option、hold、ServiceResidualLock、BaseZeroResidualFreeze、Active-Sample-Aware Actor Optimization、SafetyProjector 规则、`d_safe`、neighbor encoder、alpha/beta。未加 interference penalty / SINR reward / HGNN / Transformer。未增加训练预算。未访问 Phase-3 consumed validation `223000–223063`。未创建新 independent validation。

训练前冻结：`lambda_collision_distill=0.10`，`payload_drop_tolerance=0.05`，teacher 稀缺阈值 0.5% of active。无 lambda 网格搜索。

在 `LLV2_train` 上，用**未训练**随机初始化 Actor 做一次冻结 rollout（无 `optimizer.step`）。结果：

- agent-steps = 18432
- active samples = 1258
- collision-distill eligible **active** = **3**
- eligible / active = **0.002385 < 0.005**
- pairwise intervened agent-steps = 6
- eligible 分层：near=0，middle=0，far=3

因此协议强制：`collision_teacher_signal_too_sparse=true`，**停止训练**。未改 `d_safe`，未制造碰撞场景。C1 双 seed Pilot **未启动**。C0 按协议不重训。

---

## 20 问正式回答

1. **本轮唯一新增机制？** Safety-Projector Collision Distillation：当 projector 因 PAIRWISE 或 MULTI-含-PAIRWISE 修改动作时，`r_teacher=a_projected−a_base`（detach）作为 supervised auxiliary。因 teacher 过稀，**该损失未被训练启用**。

2. **lambda_collision_distill？** **0.10**（写入 `LLV2_PHASE4A_PREFREEZE.json`）。

3. **teacher 来自哪里？** 实际 **SafetyProjector** 的 `a_projected`。禁止人工“远离最近邻” teacher。

4. **哪些 projector 事件进入 distillation？** 仅 **PAIRWISE**，以及 **MULTI 中包含 PAIRWISE**。禁止纯 SERVICE_SAFE_ENVELOPE、纯 STEP_LIMIT。

5. **teacher signal 比例？** eligible/active = **0.238%**（3/1258）。**too_sparse=true**。

6. **两个 seed C0/C1 payload？** C1 未训练；C0 未重训。无 C1 payload。

7. **PAIRWISE intervention C0→C1？** 无 C1，无法比较。

8. **near-neighbor shuffle sensitivity C0→C1？** 无 C1，无法比较。

9. **separation_response C0→C1？** 无 C1，无法比较。

10. **teacher MSE 是否下降？** 未训练，无 u0→u100。

11. **teacher cosine 是否提高？** 未训练。

12. **distill/PPO gradient ratio？** 未训练。

13. **是否 auxiliary dominates？** 未训练。`auxiliary_dominates` 不适用；未改 lambda。

14. **collision 是否仍为 0？** 未跑 C1 closed-loop。基础 SafetyProjector 规则未改。

15. **min separation？** 无 C1 正式表。

16. **projector failure？** 无 C1 正式表。

17. **service 机制是否退化？** 未训练，ServiceResidualLock 未改，无回退。

18. **COLREL Gate？** **C**。

19. **collision_relation_internalized？** **false**。

20. **是否支持进入 5-seed collision-aware 验证？** **否**。

**下一阶段唯一建议：** 停止本条 collision-distill 线的自动推进；不要改 `d_safe`、不要制造碰撞场景、**不要自动进入 interference 增强**。当前 PAIRWISE teacher 在真实 `LLV2_train` 上过稀，无法迫使 Actor 学习 neighbor/collision relation。

本阶段停止。
