# Phase-5T Scorer Failure Mechanism Audit

Explain the remaining Phase-5S scorer regret. Do not train.

- Do not update the encoder, the scorer, the loss, or the candidate generator.
- Do not open a new scenario. Use only 310100-310117.
- Do not modify Residual MAPPO, the reward, B2, tau, Gate-1, Candidate-C, or the planner.
- Do not call deployment decide().
- Do not use seeds 230000-230127.
- planner_ready and deployment_verified stay false.
