# Phase-5W Residual Mechanism Audit

Explain the remaining one-block error. Do not train.

- Do not add a network, feature, or checkpoint.
- Do not open 310118-310125 or 230000-230127.
- Frozen MAPPO replays on 310100-310117 are diagnostic labels only.
- Do not modify Residual MAPPO, the reward, B2, tau, Gate-1, Candidate-C, or the planner.
- Do not call deployment decide().
- planner_ready and deployment_verified stay false.
