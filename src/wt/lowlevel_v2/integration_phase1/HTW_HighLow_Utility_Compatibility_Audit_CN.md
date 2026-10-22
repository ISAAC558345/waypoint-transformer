# HTW Integration Phase-1：高层效用兼容性审计（冻结 Residual MAPPO 低层）

本阶段 **training=false**。未重新训练高层或低层。未改 reward / Actor / Critic / SafetyProjector / Task Option / hold。未开启 collision 或 interference 增强。未加 HGNN / Transformer。未访问 `106000–106063`、`222000–222063`、`223000–223063` 或 confirmation / final untouched。

恢复：高层 DeepSets-util + Order-Free + Utility-Aligned + Task Option + hold=15。旧低层 L0 gated fly-toward。新低层 Residual MAPPO + ServiceResidualLock + BaseZeroResidualFreeze + Active-Sample-Aware + SafetyProjector + ServiceSafeEnvelope。

确认：LLV2-MS2-A，`stable_residual_advantage=true`，REL2-0，COLREL-C，`collision_relation_internalized=false`，COMM-B，`communication_trajectory_headroom=false`。碰撞继续由 SafetyProjector 硬保证。不能写成 Actor 已学会主动避碰或 interference-aware trajectory。

开发数据：仅 `LLV2_design`。64 个高层决策 state = 16 场景 × hold 边界，预注册，不按结果挑选。Candidate 使用 Phase-9.4.5 冻结 model-independent generator，L=24。U_L0 与 U_RL 都是同一 state、同一 assignment、H=15、不重调高层下的累计 payload increment。Primary RL utility 是五 seed 均值。

---

## 24 问正式回答

1. **用了多少 state？** 64
2. **每 state 多少 candidate？** 24
3. **L0 utility？** 冻结 assignment 后 H=15，L0 gated fly-toward（zero residual + SafetyProjector）累计 payload increment。
4. **RL utility？** 同样 H=15 shadow，冻结 Phase-3 Residual MAPPO。
5. **是否使用 5 低层 seed mean？** 是。Primary=`U_RL_mean`。未挑最好 seed。
6. **median Spearman？** 0.9965（mean=0.9761）
7. **median Kendall tau？** 0.7464
8. **top-1 agreement？** median=1.0000，mean=0.9219
9. **top-3 overlap？** median=1.0000
10. **top-25% overlap？** median=1.0000
11. **near-optimal Jaccard？** median=1.0000
12. **old L0-best 在 RL 下 normalized regret？** median=0.0000，mean=0.0057，p90=0.0000
13. **当前 DeepSets policy 在 RL 下 normalized regret？** median=**0.2000**，p90=0.7500。mean 被个别 `U_RL_best≈0` 的状态拉成极端值，正式门控只用 median。
14. **高层 probability 和 RL utility Spearman？** median=0.4013
15. **executor gain 是否 assignment-dependent？** 是
16. **哪些 assignment 特征与 executor gain 相关？** mean_uav_target_dist=0.047, max_uav_target_dist=0.013, queue_pressure=-0.020, idle_count=0.030, current_mean_sinr=0.041, current_interference=-0.026, uav_pairwise_proximity=-0.087, competition_indicator=0.812。只作描述，不作因果。
17. **五低层 seed ranking agreement？** median pairwise Spearman=0.9913
18. **Frozen high-level + L0 closed-loop payload？** 1.037500e+07
19. **Frozen high-level + 五 RL seed closed-loop payload？** 1.088e+07, 1.119e+07, 1.075e+07, 1.088e+07, 1.088e+07；mean=1.091250e+07
20. **COMPAT Gate？** B
21. **high_level_retraining_needed？** true
22. **下一阶段唯一建议？** Do not train in this phase. Next: regenerate U_new(Gamma|s)=mean of five frozen RL seeds H=15 payload, then retrain DeepSets high-level.
23. **本阶段是否训练？否。**
24. **是否重新开启 collision/interference 增强？否。**

New low-level executor changes part of the high-level utility structure.

本阶段停止。
