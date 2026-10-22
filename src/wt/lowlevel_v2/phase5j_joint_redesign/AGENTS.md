# Phase-5J Joint Utility HGNN Redesign Audit

Compare joint-assignment labels and readouts. Do not train a deployment model.

- Do not modify B2, tau, Gate-1, Candidate-C, the planner, deployment decide(), or Residual MAPPO.
- Do not retrain the low-level policy.
- Do not use seeds 230000-230127.
- planner_ready and deployment_verified stay false.
- Do not start deployment verification.
