# Phase-5P Executor-Conditioned Joint Planner Retraining

Replace J_FLY with the frozen Phase-5O J_MAPPO labels. Do not change the head.

- Do not modify the Phase-5L encoder, Sinkhorn, Hungarian, candidate mask, Residual MAPPO, reward, or planner.
- Do not call deployment decide().
- Do not use seeds 230000-230127.
- Do not regenerate J_MAPPO by rolling out MAPPO.
- Do not use 310101 for gradients, early stopping, or checkpoint selection.
- planner_ready and deployment_verified stay false.
- Do not start deployment verification.
