# Round-27 Completion Report

1. Planner-useful ratio: 0.203125.
2. Neutral ratio: 0.765625.
3. Harmful ratio: 0.031250.
4. Mean planner utility: 79781.488900 bits.
5. Median utility: 0.000000 bits.
6. Conflict utility is recorded in planner_utility_by_group.csv.
7. Duplicate-intent utility is recorded in planner_utility_by_group.csv.
8. Assignment-conflict utility is recorded in planner_utility_by_group.csv.
9. High-queue utility is recorded in planner_utility_by_group.csv.
10. K3/K5 utility is recorded in planner_utility_by_group.csv.
11. Eta=.25/.5 utility is recorded in planner_utility_by_group.csv.
12. Oracle conditional mean payload gain: 126374.000740 bits on validation.
13. Oracle validation CI: [0.000000, 315338.086880].
14. Trigger A validation metrics: {'trigger': 'A_interference_conflict', 'split': 'validation', 'threshold_source': 'none', 'queue_competition_threshold': 0.8812921057205046, 'population_count': 16, 'activation_count': 0, 'activation_rate': 0.0, 'precision_useful': 0.0, 'recall_useful': 0.0, 'specificity': 0.8125, 'average_active_utility_payload': 0.0, 'average_inactive_utility_payload': 88412.37041386776, 'active_utility_ci95_low': 0.0, 'active_utility_ci95_high': 0.0, 'harmful_action_rate_active': 0.0, 'oracle_trigger': False, 'deployable': True}.
15. Trigger B validation metrics: {'trigger': 'B_duplicate_intent', 'split': 'validation', 'threshold_source': 'none', 'queue_competition_threshold': 0.8812921057205046, 'population_count': 16, 'activation_count': 7, 'activation_rate': 0.4375, 'precision_useful': 0.14285714285714285, 'recall_useful': 0.3333333333333333, 'specificity': 0.7777777777777778, 'average_active_utility_payload': -33922.77156386471, 'average_inactive_utility_payload': 183561.92528543746, 'active_utility_ci95_low': -161401.03054185052, 'active_utility_ci95_high': 59632.71585025638, 'harmful_action_rate_active': 0.14285714285714285, 'oracle_trigger': False, 'deployable': True}.
16. Trigger C validation metrics: {'trigger': 'C_queue_interference', 'split': 'validation', 'threshold_source': 'train_p75', 'queue_competition_threshold': 0.8812921057205046, 'population_count': 16, 'activation_count': 0, 'activation_rate': 0.0, 'precision_useful': 0.0, 'recall_useful': 0.0, 'specificity': 0.8125, 'average_active_utility_payload': 0.0, 'average_inactive_utility_payload': 88412.37041386776, 'active_utility_ci95_low': 0.0, 'active_utility_ci95_high': 0.0, 'harmful_action_rate_active': 0.0, 'oracle_trigger': False, 'deployable': True}.
17. Trigger D validation metrics: {'trigger': 'D_assignment_conflict', 'split': 'validation', 'threshold_source': 'none', 'queue_competition_threshold': 0.8812921057205046, 'population_count': 16, 'activation_count': 5, 'activation_rate': 0.3125, 'precision_useful': 0.0, 'recall_useful': 0.0, 'specificity': 0.7272727272727273, 'average_active_utility_payload': -75320.48091953024, 'average_inactive_utility_payload': 162836.3937472305, 'active_utility_ci95_low': -225961.44275859074, 'active_utility_ci95_high': 0.0, 'harmful_action_rate_active': 0.2, 'oracle_trigger': False, 'deployable': True}.
18. Train-selected causal trigger: B_duplicate_intent.
19. Validation positive-effect stability: false.
20. Selected-trigger harmful rate: 0.142857.
21. K/eta stability is recorded by group and gate criteria.
22. Planner utility / Task-WP relation is in planner_utility_taskwp_relation.csv.
23. Planner utility / seed disagreement is in planner_utility_seed_disagreement.csv.
24. CPA Gate: CPA-B.
25. Trigger Gate: TR-B.
26. conditional_planner_action_distillation_ready: false.
27. Round-27 recommendation: R27-B_continue_trigger_representation_study_no_training.
28. Training recommendation only; no training was performed.
29. Fresh new rollout count: 0.
30. Untouched-evaluation access count: 0.
31. Round-28 created: false.
32. Direct utility uses real frozen communication transitions, not action distance.
