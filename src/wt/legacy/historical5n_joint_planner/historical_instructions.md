# Phase-5N Learnable Joint Planner

Train only the joint assignment head. The Phase-5L encoder stays frozen.

- Do not modify Residual MAPPO, the reward, the low-level policy, or deployment.
- Do not call deployment decide().
- Do not use seeds 230000-230127.
- planner_ready and deployment_verified stay false.
- Do not start deployment verification.
