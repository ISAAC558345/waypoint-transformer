# Round-25 Completion Report

Round-25 is a frozen-data, frozen-model and frozen-planner diagnostic. It performs no neural training, policy training, checkpoint write, data/split/quality modification, fresh rollout, development-test decode, or untouched evaluation.

## Required answers

1. Same-initial-state Service ambiguity A_service: 0.061182.
2. Progress ambiguity A_progress: 0.132319.
3. Return ambiguity A_return: 0.135424.
4. Behavior-pair divergences: h10_vs_independent=0.165190; h10_vs_h1=0.064547; h10_vs_noisy_h10_noise015=0.040740; h10_vs_noisy_h10_noise035=0.081546; h1_vs_independent=0.166819.
5. Quality-conditioned target differences: 0.088844 mean pair distance.
6. Bridge versus non-bridge local ambiguity: ratio=0.982296.
7. Conflict versus non-conflict local ambiguity: ratio=1.183273.
8. Predictor closer to local behavior mean: 0.044746.
9. Predictor distance to high-quality target: 0.071191.
10. Service entropy: target=0.650745, predicted=0.822063, local-mean=0.772842.
11. Progress comparison: predicted=2.451505, high=2.428816, local=2.410479.
12. Return comparison: predicted=0.280228, high=0.323108, local=0.252519.
13. Causal planner-derived target: frozen Round-20 joint-H5 causal trace, verified by direct current-state H5 first-action replay.
14. Planner target versus H10 behavior: 0.000190.
15. Planner target versus high-quality mean: 0.089923.
16. Planner target versus low-quality mean: 0.082494.
17. Predictor-to-planner-like distance versus local mean: 0.089872 versus 0.044746.
18. Frozen policy seed mean planner-action RMSE: s0=0.336878, s1=0.371409, s2=0.351065, s3=0.376470, s4=0.350438, s5=0.379783, s6=0.386582, s7=0.369427.
19. Ambiguity to policy-seed instability Spearman: -0.161765; descriptive only.
20. K3/K5 local ambiguity: K3=0.068860, K5=0.084174.
21. Eta=.25/.5 local ambiguity: eta=.25=0.064808, eta=.5=0.086898.
22. SA-TW Gate: SA-TW-C.
23. task_waypoint_supervision_ambiguity_confirmed: false.
24. Planner-target Gate: PT-B.
25. planner_derived_task_waypoint_candidate: false.
26. Recommended next step: stop_current_task_waypoint_overall_policy_advantage_claim.
27. Training recommendation only: true; no training was performed.
28. Fresh new rollout count: 0.
29. Untouched-evaluation access count: 0.
30. Round-26 created: false.
