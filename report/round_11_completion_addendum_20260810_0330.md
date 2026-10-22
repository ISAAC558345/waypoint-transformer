# Round-11 completion addendum: detailed Task-Waypoint evaluation

This append-only addendum corrects the evidence gap in the earlier Round-11
lock without changing it. It evaluates only the existing six best checkpoints
on the locked Round-10 train/validation samples. No optimizer, backward update,
old development test, untouched holdout, policy, Transformer, or closed loop
was used in this addendum.

## Required answers

1. Input is the observable maritime state (variable 30/44 fields, padded to
   44), including the already-present remaining-horizon fraction, plus the
   dynamic buoy mask; no duplicate time feature is appended.
2. Constant goal is completely absent from input, embedding and loss.
3. Yes: service CE is 1.4062 (H10) / 1.4125 (H20), consistently below the
   train-mean CE 1.4346 / 1.4373.
4. Yes: it also beats the current-state queue-times-allocated-rate heuristic
   CE 1.5780 / 1.5910; see baseline metrics.
5. Yes: gross-service-progress is learnable and non-negative.
6. Yes: H10 raw RMSE is 0.888 Mbit versus 1.227 Mbit train mean, 3.460 Mbit
   current-rate, and 4.596 Mbit zero; H20 has the same ordering.
7. Yes: the return head is learnable.
8. Yes: H10 return RMSE is 0.384 versus 1.019 train mean and 1.035
   remaining-time-only; current-reward is unavailable because it is not an
   observable model input and was not fabricated.
9. Return is easiest (R2 0.83--0.89; Pearson 0.92--0.95).
10. Service distribution is hardest: its gain is modest despite valid,
    non-collapsed probabilities.
11. No obvious negative-transfer collapse: all per-head gradients are finite,
    all heads beat train mean, and no head has zero gradient.
12. Gradient scales differ but remain finite; return trunk gradients are
    largest, which is recorded rather than counterweighted.
13. Yes: effective rank, variance and sample-distance statistics in the
    information audit show non-constant predictions.
14. No: no probability, progress or return collapse was found.
15. H10 is more stable overall, with lower service CE, progress RMSE and return
    RMSE across the three seeds.
16. No: Round-10's H20 priority is retained as historical context, but H10 is
    the validation-only Round-11 recommendation.
17. The predictor gate is **TW-A**.
18. Not applicable: no head failed.
19. Recommended heads are service distribution, gross actual service progress,
    and discounted H-step return.
20. The predictor prerequisite for a future frozen-waypoint-conditioned
    Transformer study is met, but such a study is not authorized or run here.
21. Old development test remains zero accesses in the Round-11 loader/audit.
22. Untouched holdout remains zero accesses in the Round-11 loader/audit.

## Evidence

- Per-head metrics: `round_11/outputs/round11_task_waypoint_20260810_0310/`.
- Baselines use train statistics or current observable state only.
- 20 PNG and 20 SVG detailed diagnostics:
  `round_11/figures/round11_task_waypoint_20260810_0320/`.
- The physical-upper-bound rate is marked not identifiable from the normalized
  state alone; no future data was used to make a deployment-time correction.
- Six final best checkpoints remain exactly the original saved checkpoints;
  last checkpoints were not emitted by the earlier basic runner and are
  explicitly listed as an immutable documentation limitation.
