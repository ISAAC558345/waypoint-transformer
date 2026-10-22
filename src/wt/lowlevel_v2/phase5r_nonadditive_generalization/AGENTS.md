# Phase-5R Frozen Non-Additive Scorer Generalization

Audit the Phase-5Q Model-A and Model-C weights on scenarios 310102-310109.

- Do not train on these scenarios, and do not change the encoder, scorer, loss, or candidate width.
- Do not modify Residual MAPPO, mappo_original.pt, the reward, B2, tau, Gate-1, Candidate-C, or the planner.
- Do not call deployment decide().
- Do not use seeds 230000-230127.
- 310102-310109 are development-generalization scenarios, not a deployment holdout.
- planner_ready and deployment_verified stay false.
