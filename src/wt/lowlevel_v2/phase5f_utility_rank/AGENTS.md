# Phase-5F HGNN Utility Ranking

Retrain only the existing high-level HGNN on Phase-5E utility ranks.

- Do not modify B2, tau, Gate-1, Candidate-C, the planner, or Residual MAPPO.
- Do not call `decide()`.
- Do not add a new network module.
- Do not use seeds 230000-230127 or generate a new scenario.
- planner_ready and deployment_verified stay false.
- Do not start Phase-5G.
