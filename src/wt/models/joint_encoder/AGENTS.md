# Phase-5L Joint Utility HGNN Training

Train the HGNN encoder on J(A). The Phase-5K matching decoder stays frozen.

- Do not modify Residual MAPPO, the low-level policy, the reward, or the planner.
- Do not call deployment decide().
- Do not use seeds 230000-230127.
- planner_ready and deployment_verified stay false.
- Do not start deployment verification or a hierarchical joint planner.
