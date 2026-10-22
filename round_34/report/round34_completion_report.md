# Round-34 completion report

## Outcome

Round-34 completed the frozen **Benchmark Complexity, Relational-Decision
Opportunity, and Method Suitability Audit**. Its formal result is **BCX-B**:
the current M=2, K={3,5} movement-only benchmark underexpresses learned
relational decision-making because deterministic current-state matching owns
association, service priority, and interference allocation. This is a
benchmark-suitability conclusion, not evidence that any HGNN, Task-Waypoint,
or RL method improves performance.

The formal audit used only the hash-locked clean train/validation manifests:
7,040 Task-Waypoint samples and 7,680 transition action records. It performed
no model fitting or inference, environment or planner execution, source
physics/reward modification, fresh rollout, or untouched evaluation access.

## Governance Exception

One accidental protected access occurred before scientific computation:
`development_test` was decoded exactly once during the original preflight,
was detected immediately, and execution stopped. It is recorded in
`round_34/ROUND34_PROTECTED_ACCESS_EXCEPTION.json` and in the successor
amendment to the immutable Round-33 lock.

- Scientific use of that record: false.
- Formal clean-manifest protected records: 0.
- Second protected access: false.
- Untouched access: 0.
- Strict lexical metadata allowlisting is now applied before the sole JSON
  decoder; unknown split metadata fails closed.
- Formal outputs, figures, gates, and this report derive exclusively from the
  clean manifests. The quarantine boundary is explicitly excluded.

## Audit answers

1. **RQ1 — Current relational complexity:** M=2 and K={3,5}; there are 6/10
   potential U-B edges, two directed U-U edges, and 13/31 legal partial
   injective assignments (not K^M).
2. **RQ2 — Decoder ownership:** exhaustive matching consumes current queue,
   geometry-derived rate, feasibility, interference quantities, and a
   candidate score; it emits assignment and service allocation outside the
   learned movement interface.
3. **RQ3 — Movement semantic distance:** movement affects queue only through
   geometry → rate → matching → service → queue, a five-link mediated chain.
4. **RQ4 — Edge information:** most U-B/U-U numeric features are exactly
   derivable from node/global variables; assignment identity and selected
   interference relations remain pair-specific.
5. **RQ5 — Scale projection:** M=4/K=8 has 32 U-B edges and 3,393 legal
   assignments; M=4/K=16 has 64 and 58,625; M=8/K=16 has 128 and
   1,174,226,049.
6. **RQ6 — Method suitability:** HGNN is structurally plausible only after a
   benchmark gives the learned controller ownership of graph relations; the
   current one does not.
7. The direct neural-policy category share is 1/8=0.125: movement only.
8. GDOR is 0/3=0 because none of association, service priority, or
   interference coordination is a neural relational decision.
9. Current matching is injective and permits idle UAVs; it is not an
   unconstrained K^M decoder.
10. Power and bandwidth are fixed configuration/rules, not policy actions.
11. Sensing is not exposed as an environment action.
12. Safety is an environment projection, not a learned action.
13. `current_assignment_relation` is the U-B feature irreducible to edge-free
    pooled nodes.
14. `both_assigned_distinct`, selected receiver interference, and ISR are the
    corresponding pair-specific U-U features.
15. Frozen clean-label correlation audit finds several U-B geometry/rate
    features related to Task-Waypoint service targets, but this does not add a
    learned association interface.
16. Coarse DeepSets pooling has relation-matrix collisions; decision-relevant
    collision rows are separately reported.
17. Task-Waypoint heads describe service/progress/return; the policy output
    remains only two movement coordinates per UAV.
18. The hypothetical association interface is design-only and not
    implemented.
19. The Round-33 frozen H1 mean payload is 44,793,881.7 bits.
20. Frozen H5 is 45,095,770.8 bits, an observed +301,889.1-bit mean over H1.
21. Frozen H10 equals H5 in the audited aggregate and has zero H10-H5
    first-action divergence and same-snapshot utility difference.
22. Hence the observed effective planning value saturates at or below H5 in
    this benchmark; it is not a universal claim.
23. HGNN future re-entry is conditional on a BCX-B redesign increasing
    learned graph-decision ownership; it is not an improvement claim.
24. Task-Waypoint re-entry remains false; a redesigned benchmark must first
    establish planning necessity and does not reuse the current predictor.
25. L1 is the present M=2/K=3,5 benchmark.
26. L2 proposes M=4/K=8 plus a learned masked association preference while
    preserving all current physical variables and fixed resources.
27. L3 would train at M=4/K=8 and test M=6/K=12 or M=8/K=16 only after a
    separately authorized scale choice.
28. Future causal evaluation must hold head, decoder, compute budget, seeds,
    and data protocol fixed.
29. The strategy gate is R34-B: a future environment-design-only round could
    be considered with new authorization; no training is authorized now.
30. No protected record contributed to a metric, threshold, feature choice,
    benchmark design, gate, figure, or report metric.
31. The full regression suite passed 1,087/1,087 and the dedicated Round-34
    suite passed 47/47.
32. Compileall passed; Round-34 is sealed and Round-35 was not created.

## Artifacts and gates

The formal tables and markdown analyses are in
`round_34/outputs/round34_benchmark_complexity_audit_20260901_1540`; 12 PNG
and 12 SVG figures are in the paired figure directory. The gates are
`BCX-B`, `HGNN-FUTURE-B`, `TW-FUTURE-C`, and `R34-B`.
