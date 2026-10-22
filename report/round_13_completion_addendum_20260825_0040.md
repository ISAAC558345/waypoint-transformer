# Round-13 completion addendum

`amendment_only=true`; `neural_training_performed=false`; and
`original_round13_results_modified=false`.  This addendum is additive: the
original Round-13 lock, formal output, controllability conclusion, and
SA-C-discrete result remain unchanged.

## Direct answers

1. **Original scientific conclusion preserved?** Yes.  All original-formal
   hashes and the original `ROUND_LOCK.json` hash are unchanged.
2. **C_service:** 0.0037064542.
3. **C_progress:** 0.0096376949.
4. **C_return:** 0.0030245169.
5. **Lowest controllability:** C_return (the H-step-return head).
6. **All three heads low?** Yes.  They are movement-mediated/indirectly
   controllable, not direct service controls.
7. **Continuous priority diagnostic completed?** Yes, on the same ten
   permitted validation scenarios, with zero decoder violations.
8. **Continuous minus current payload:** -0.424 Mbit.
9. **Its 95% scenario-bootstrap CI:** [-0.957,
   0.154] Mbit.
10. **Continuous minus current completion:** -0.002962.
11. **Continuous minus current residual-queue improvement:**
    -0.527 Mbit (negative means a larger residual queue).
12. **Continuous minus current return:** -0.606916.
13. **Continuous minus discrete:** payload 9.387 Mbit, completion
    0.039982, residual-queue improvement 3.987 Mbit,
    return 2.318620; the payload CI is [3.732,
    16.345] Mbit.
14. **Continuous Service-Action Gate:** SA-C-continuous.
15. **service_action_extension_ready:** false.
16. **Current reward formula:**
    `payload_reward + energy_penalty + boundary_penalty +
    invalid_action_penalty + energy_depletion_penalty +
    unfinished_queue_penalty + task_success_bonus`; exact formulas,
    coefficients, normalizations, units, signs, and ranges are documented in
    `reward_semantics_audit.md`.
17. **Why did the original discrete heuristic show worse communication but
    higher return?** Its sidecar was not formula-equivalent to the environment:
    it used allocated-rate rather than theoretical-full-rate normalization for
    replacement payload, and a different terminal residual normalization.
    The immutable original result is still reported, but an exact reward
    reconstruction produces discrete-current return -2.925536
    (95% CI [-4.337321, -1.782891]), not a
    positive reward gain.
18. **Episode return vs cumulative payload:** Pearson 0.998749,
    Spearman 0.996329 (pooled fixed-movement sidecars; grouped
    rows are in the audit CSV).
19. **H-step return vs H-step payload:** Pearson 0.985446,
    Spearman 0.997743.
20. **Return Task-Waypoint versus primary metric:** strong.
21. **Reward Alignment Gate:** RA-A.
22. **Best-supported TW-POL-B factor:** action controllability mismatch.
23. **Predictor-error evidence:** moderate.
24. **Action-mismatch evidence:** strong.
25. **Reward/Return semantic-mismatch evidence:** unsupported.
26. **Representation-to-action-mapping evidence:** moderate.
27. **Amended next recommendation:** `task_waypoint_action_mapping_redesign`.
28. **PLANS/ROUND_VERSIONING corrected?** Yes; their current-status amendment
    states Round-13 completed/locked and no Round-14.  Before/after hashes are
    recorded in the amendment lock.
29. **Old-development-test access count:** 0.
30. **Untouched-holdout access count:** 0.
31. **Original ROUND_LOCK hash unchanged?** Yes.
32. **Round-14 created?** No.

## Formal status

The original discrete gate is retained as `SA-C-discrete`; it applies only to
the tested `argmax(queue * current/full-bandwidth-rate)` heuristic.  The
continuous masked-simplex candidate is separately `SA-C-continuous`, so it
does not justify a service-action extension.  No action-space change, model
training, policy training, checkpoint, optimizer, backward pass, old
development test, holdout access, HGNN, PPO, multi-UAV system, or Round-14
was introduced.

Final verification: 511 repository tests passed and `compileall`
passed.  The formal amendment bundle is `round_13/amendments/round13_completion_amendment_20260825_0040`.
