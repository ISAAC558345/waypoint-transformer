# HTW Phase-6I-B：高层宏效用数据集

## 结论

Gate 为 **DATA-A HIGHLEVEL_MACRO_DATASET_VALID**。监督标签是 J15_MAPPO_CURRENT，低层执行器是 phase6g_seed0.pt，候选规则是 G1_RATE k=6。没有训练。`planner_ready = False`，`deployment_verified = False`。

## Q1. 最终高层监督标签是什么？

J15_MAPPO_CURRENT。

## Q2. 标签对应哪个 low-level executor？

phase6g_seed0.pt。选择规则是 MINIMUM_SEED_ID。

## Q3. 候选规则是什么？

G1_RATE k=6。idle 是单独的合法动作，不占用 buoy 的 top-6 名额。

## Q4. 为什么不是 k=5？

Phase-6I-A 的 Gate 是 BOUND-B K6_REQUIRED。k=6 相对 k=5 在 design 和 confirmation 上都有稳定的归一化增益，k=7 相对 k=6 在 design 上没有增益。

## Q5. 是否使用未来 realized arrival？

NO。

## Q6. 训练、validation 是否 scenario-level 隔离？

YES。训练是 322000–322047，validation 是 322048–322063。

## Q7. label reuse 复用了多少已有结果？

0 条，比例 0。Phase-6H 和 Phase-6I-A 只保存了其他 scenario 的 optimum，没有带 state hash 的完整 k=6 标签，因此节省的 J15 时间为 0。

## Q8. 总共生成多少 scenario、state、candidate labels？

scenario 64，主状态 192，candidate labels 2558784。训练状态 144，validation 状态 48。

## Q9. 每个 state 平均 candidate 数？

13327.00。

## Q10. 标签是否具有足够动态范围？

至少 2 个不同 J15 组的状态有 192 个，至少 5 个不同组的状态有 192 个。平均 top1-top2 margin 是 12753.12 bit。

## Q11. all-candidates-tied state fraction 是多少？

0.000000。DATASET_LABEL_DEGENERACY = False。

## Q12. label recomputation 是否一致？

抽查 119764 个 state-assignment。最大绝对误差 0.000000 bit。阈值是 1 bit。PASS。

## Q13. shadow rollout 是否污染 state？

NO。每个已写 state 的 shadow_state_contamination 都是 0。

## Q14. 数据里是否仍然存在 relational planning signal？

边可加分数选出的 assignment 与 J15 optimum 不一致的状态比例是 1.000000。这些状态上的平均 J15 regret 是 1063791.05 bit。这只说明标签不是瞬时速率的简单加和，不是 HGNN 优于 DeepSets 的证据。

## Q15. rank-6 boundary fraction 是多少？

0.921875。k 没有因此改动。

## Q16. 322064–322079 是否完全 untouched？

YES。

## Q17. 320100–320131 是否完全 untouched？

YES。

## Q18. 最终 Gate？

DATA-A。

## Q19. 如果 DATA-A，下一阶段是否可以开始公平高层训练？

YES。本阶段没有开始训练。

## Q20. 现在是否重新训练 MAPPO？

NO。

串行与并行在 scenario 322000 上的最大绝对误差是 0.000000 bit，argmax 一致 = True。
