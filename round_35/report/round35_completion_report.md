# Round-35 completion report

## Outcome

Round-35 implements the authorized Level-2 relational shared-spectrum
benchmark. It adds no model, model inference, training, optimizer, backward
pass, planner trigger, physics/reward change, dynamic channel, drift, or new
resource-control action. Its only new controller-owned field is an external
association-preference matrix. The exact decoder retains feasibility,
injectivity, one-service-per-UAV capacity, idle legality, valid indices, and a
deterministic tie rule, but compares no queue/rate/distance/interference
heuristic when selecting a feasible assignment.

The disk-authoritative partial-injective count for M=4/K=8 is **3,393**. The
instruction's 3,993 value is a transcription discrepancy; the same disk
baseline used by Round-34 records 3,393, and exhaustive enumeration confirms
it. This report makes no learned-policy performance claim.

## Evidence

- Legacy M=2/K={3,5}, T={12,20} exact compatibility: PASS across observation,
  transition, reward, matching, queue, rate, interference, safety, and
  termination.
- Engineering-only scenario family: 64 new generated scenarios, covering
  η={0.25,0.5}, medium/high queue pressure, compact/spread geometry; no
  performance-based selection.
- Association causality, rank response, exact constraint stress audit, and
  joint UAV/buoy permutation audit: PASS.
- Graph decision ownership changes from L1 GDOR 0/3 to Level-2 GDOR 1/3:
  association preference/selection is controller-owned, while feasibility and
  shared-spectrum service/interference remain lawful consequences.
- Exact M=4/K=8 decoding evaluates 3,393 candidates per step. Runtime and
  memory results are recorded in `round35_runtime_profile.csv`; the measured
  engineering profile is feasible without approximation.
- The new structural chains are opportunities only: Task-Waypoint re-entry is
  still false and HGNN performance readiness is false.

## Required answers

1. Legacy exact regression PASS: yes.
2. Level-2 M/K: M=4, K=8.
3. Episode horizon: T=20.
4. U-B edges: 32.
5. U-U edges: 12 directed, 6 undirected.
6. Legal association count: 3,393 exact partial injective assignments.
7. External association action: (4,8).
8. Movement action: (4,2).
9. Preference changes assignment: yes.
10. Decoder secretly uses old heuristic score: no.
11. Feasibility constraints preserved: yes.
12. Permutation audit: PASS.
13. Old GDOR: 0/3.
14. New GDOR: 1/3.
15. GDOR > 0: yes.
16. Movement/association independently controllable: yes.
17. Association affects service: yes.
18. Association affects queue: yes.
19. Association affects interference: yes.
20. M4/K8 matching runtime: recorded as exact 3,393-candidate step timing in
    `round35_runtime_profile.csv`.
21. Environment runtime acceptable: yes, for the registered engineering profile.
22. Task-Waypoint service-to-action causal chain shorter: structurally yes.
23. HGNN has a clear edge-action interface: structurally yes, edge logits.
24. This proves HGNN is better: no.
25. This proves Task-Waypoint is better: no.
26. RAO gate: RAO-A.
27. Level-2 gate: L2-A.
28. `level2_relational_benchmark_ready`: true.
29. `hgnn_structural_revalidation_ready`: true; performance readiness false.
30. `task_waypoint_reentry_ready`: false.
31. Next stage: `non_learning_planning_and_action_relevance_validation`.
32. Protected access count: 0.
33. Second protected access: false.
34. Untouched access count: 0.
35. Training performed: false.
36. Round-36 created: false.

## Final verification

The 38 dedicated Round-35 tests and all 1,125 repository tests passed;
`compileall` passed. The original Round-34 lock rehash matches its immutable
digest. Locked-artifact rehash and Round-35 lock results are recorded in the
implementation log and `ROUND_LOCK.json`. The round stops after locking.
