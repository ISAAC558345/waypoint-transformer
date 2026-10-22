# Phase-5S Multi-Scenario Non-Additive Scorer

Train the unchanged Phase-5Q scorer on scenarios 310100-310109.

- Do not change the scorer, encoder, loss, learning rate, epoch count, or candidate width.
- Do not open scenarios 310110-310117 until every seed checkpoint has been saved.
- Do not use confirmation results to retrain, stop early, or select a checkpoint.
- Do not modify Residual MAPPO, the reward, B2, tau, Gate-1, Candidate-C, or the planner.
- Do not call deployment decide().
- Do not use seeds 230000-230127.
- planner_ready and deployment_verified stay false.
