# HTW Phase-9.4.4 HGNN Encoder and Training Instability Causal Audit

诊断阶段。未修改 HGNN / decoder / set likelihood / optimizer / lr / L3-B / SINR / Task-Waypoint。未做 PPO。未按 confirmation 选模型。
phase943_confirmation_consumed=**true**；phase944_model_tuning_on_phase943_confirmation=**false**。

1. seed0-4 train loss（日志 last epoch）DeepSets=[0.6012, 0.5916, 0.5904, 0.6297, 0.685]；HGNN=[0.7179, 0.7624, 0.6342, 0.677, 0.657]
2. val set NLL（eval 8–15 轨迹子样本 last ckpt）DeepSets=[1.9410647158219945, 3.0984916147038652, 2.5876999281220683, 2.23303551964594, 2.134089793971119]；HGNN=[2.3956374699094645, 2.7618412039155373, 1.958661015670259, 2.4252224611987003, 1.4424574524773561]
3. free-running per-UAV acc DeepSets=[0.8229166666666666, 0.8229166666666666, 0.8541666666666666, 0.8229166666666666, 0.8159722222222222]；HGNN=[0.8368055555555557, 0.8090277777777777, 0.8159722222222222, 0.826388888888889, 0.8680555555555555]
4. positive/negative HGNN 优化差异：**不明显：负向 payload 的 seed 3/4 监督 NLL/FR-acc 并不更差（train/val NLL 甚至更好）**
5. gradient 异常：**否（last-epoch 诊断 backward 未见爆炸/消失）**
6. oversmoothing：**否（终层 OS_U 均值=0.143）**
7. effective rank 坍缩：**否（终层 erank_uav=[3.2903326970748936, 2.9830603331206493, 3.09248867495905, 3.13041344267163, 3.0659708111861907]）**
8. seed3 UU 使用：IRS=0.0176，shuffle-UU disagreement=0.0417
9. seed3 UB 使用：IRS_UB=28.9388，shuffle-UB disagreement=0.6250
10. seed4 UU/UB：IRS_UU=0.0080 / IRS_UB=53.3543；positive mean IRS_UU=0.0125
11. decoder logits：margin pos=5.7783 seed3=6.3923 seed4=5.8305
12. idle：pos=0.3325 seed3=0.3763 seed4=0.4805（confirmation）
13. assignment accuracy vs payload Pearson：HGNN=-0.0436 DeepSets=0.1049
14. set NLL vs payload Pearson：HGNN=0.0189 DeepSets=-0.0288
15. SINR utility vs payload Pearson：HGNN=0.8608 DeepSets=0.8594
16. near-optimal 不同 assignment（影子搜索 5% 窗）均值数量=**3.12**
17. label ambiguity：**中等偏高**（utility_equivalent_assignment_ambiguity=True）
补充：全部 HGNN seed 的 UU-shuffle IRS 都接近 0，UB-shuffle IRS 都很大；UU 低敏感是普遍现象，不是 seed 3/4 独有。
18. seed3 最可能原因：**OBJ** — acc-payload r=-0.353 multiplicity5=3.12
19. seed4 最可能原因：**DEC|OBJ|GEN** — idle=0.480 margin=5.831; acc-payload r=0.056 multiplicity5=3.12; val assignment acc not low but confirmation payload far below paired DeepSets
20. DeepSets 是否同一问题：**是：DeepSets 同样出现 assignment-acc/NLL 与 payload 弱相关（标签: NONE,OBJ,OBJ,OBJ,OBJ）**
21. 剩余 Graph instability 主要来自：**objective**
22. INST Gate=**INST-B**
23. 下一阶段：**utility_aligned_supervised_objective**

ORDERFIX-A 保持；decoder_order_dependence_fixed=true。HGNN confirmation payload 仍 seed-不稳定：[515625.0, 2515625.0, 2421875.0, -203125.0, -3031250.0]。
设计验证只用 historical eval 8–15 与 collect 状态；943 confirmation 只做 post-hoc 相关。

JSON/CSV：`src/wt/htw_phase944/outputs/hgnn_instability/`
