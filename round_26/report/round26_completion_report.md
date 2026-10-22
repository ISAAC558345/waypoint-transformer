# Round-26 Completion Report

1. Exact same-state action ambiguity: 0.358650.
2. Versus Task-Waypoint ambiguity: similar (relative ratio 0.974042).
3. Largest behavior-pair action divergence: h10_vs_random.
4. High-quality within action ambiguity: 0.358419.
5. kNN multimodality SSE reduction: 0.767540.
6. Mean-action averaging-risk share: 0.417969.
7. Conflict/non-conflict local ambiguity ratio: 1.079272.
8. Bridge/non-bridge local ambiguity ratio: 1.127765.
9. K=3 versus K=5 local ambiguity: 0.193840 versus 0.279967.
10. Eta-stratified ambiguity is recorded for eta=0, .25, .5, and 1 in action_ambiguity_grouped.csv.
11. True Task-Waypoint ARR: 0.016262.
12. Predicted Task-Waypoint ARR: 0.015650.
13. Head ARR: {'true_full': 0.01626183585290724, 'predicted_full': 0.015649697761233754, 'true_service': 0.015123969843092517, 'true_progress': 0.016102538556597105, 'true_return': 0.013168005588024233, 'true_service_progress': 0.017423943357850566}.
14. Planner same-state variance: 0.000000.
15. H10-to-planner RMSE: 0.000000.
16. High versus low planner RMSE: 0.375725 versus 0.378643.
17. Planner distance/payload Spearman: 0.061362.
18. Frozen policy-family planner alignment: [{'family': 'predicted_task_waypoint_transformer', 'mean_planner_action_rmse': 0.36525664450030493, 'mean_planner_A_action': 0.26573365606873717, 'seed_count': 8}, {'family': 'state_only_transformer', 'mean_planner_action_rmse': 0.36413950596398825, 'mean_planner_A_action': 0.25535692226001405, 'seed_count': 8}, {'family': 'predicted_task_waypoint_mlp', 'mean_planner_action_rmse': 0.3628301202926616, 'mean_planner_A_action': 0.2535587637084861, 'seed_count': 8}].
19. Task-WP versus state-only planner alignment is reported in taskwp_state_planner_alignment.csv.
20. BC-conflict/seed-disagreement Spearman: 0.276471.
21. AS Gate: AS-B.
22. TD Gate: TD-C.
23. PAD Gate: PAD-B.
24. action_supervision_ambiguity_confirmed: false.
25. planner_action_distillation_candidate: false.
26. Round-26 recommendation: R26-D_conditional_planner_action_distillation_design_only.
27. Training recommendation only: true; no training was performed.
28. Fresh new rollout count: 0.
29. Untouched-evaluation access count: 0.
30. Round-27 created: false.
