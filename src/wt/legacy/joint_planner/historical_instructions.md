# Phase-5M Hierarchical Joint Planner

Use the HGNN as a top-K candidate generator and refine with J(A).

- Do not modify Residual MAPPO, the reward, the low-level policy, or deployment.
- Do not call deployment decide().
- Do not use seeds 230000-230127.
- planner_ready and deployment_verified stay false.
- Do not start deployment verification.
