# Phase-5G Closed-loop Validation

Validate the frozen utility HGNN with the frozen Residual MAPPO. Do not train.

- Do not modify B2, tau, Gate-1, Candidate-C, the planner, or the low-level policy.
- Do not call deployment `decide()`.
- Do not use seeds 230000-230127 or the Phase-5F training scenario 310100.
- planner_ready and deployment_verified stay false.
- Do not start Phase-5H.
