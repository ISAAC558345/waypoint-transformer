# Phase-5V Candidate-Conditioned Temporal State

Add execution history that changes with the candidate assignment.

- Do not train or edit the Phase-5L encoder.
- Do not change Phase-5U static relational features, the ranking loss, or the candidate width.
- Do not use 310110-310117 for gradients, early stopping, or checkpoint selection.
- Do not invent dwell time, AoI, or a future SINR.
- Do not modify Residual MAPPO, the reward, B2, tau, Gate-1, Candidate-C, or the planner.
- Do not call deployment decide().
- planner_ready and deployment_verified stay false.
