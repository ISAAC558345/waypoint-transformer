# Phase-5Q Encoder versus Assignment Scorer

Separate two causes of the Phase-5P failure. Do not change both at once.

- Model-A reuses the Phase-5P J_MAPPO head. Do not reinterpret it.
- Model-B trains the existing Phase-5L encoder with the unchanged additive head.
- Model-C keeps that encoder frozen and scores each legal assignment with a non-additive set scorer.
- Do not modify Residual MAPPO, the reward, the candidate pool, J_MAPPO, B2, tau, Gate-1, Candidate-C, or the planner.
- Do not call deployment decide().
- Do not use seeds 230000-230127 or regenerate scenarios.
- Do not use 310101 for gradients, early stopping, checkpoint selection, or hyperparameter changes.
- planner_ready and deployment_verified stay false.
