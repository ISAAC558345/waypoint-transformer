# Phase-5K Joint Assignment Decoder

Train only the new high-level decoder heads on J(A).

- Do not modify Residual MAPPO, the low-level policy, the reward, B2, tau, Gate-1, Candidate-C, or the planner.
- Do not call deployment decide().
- Do not use seeds 230000-230127.
- The HGNN encoder stays frozen.
- planner_ready and deployment_verified stay false.
- Do not start deployment verification or a full encoder retrain.
