# Phase-5U Assignment Representation Repair

Add current-state relational features to the non-additive scorer.

- Do not train or edit the Phase-5L encoder.
- Do not change the ranking loss, candidate width, or training scenarios.
- Do not use 310110-310117 for gradients, early stopping, or checkpoint selection.
- Do not add episode time, option age, hold remaining, or future rollout features.
- Do not modify Residual MAPPO, the reward, B2, tau, Gate-1, Candidate-C, or the planner.
- Do not call deployment decide().
- planner_ready and deployment_verified stay false.
