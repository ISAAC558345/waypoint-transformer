# Phase-5C High-Level Utility Redesign Audit

Audit the high-level objective and specify a trajectory-aware label. Do not train.

- Do not modify B2, tau, Gate-1, Candidate-C, the planner, or MAPPO.
- Do not call `decide()`.
- Do not use seeds 230000-230127.
- planner_ready and deployment_verified stay false.
- Do not start joint training.
