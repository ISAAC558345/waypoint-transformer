# Phase-5Y Executor-Conditioned Joint Value Distillation

Learn V_exec(s, A) from frozen MAPPO 15-step payload.

- Train only a new value head on 310100-310109.
- Keep the Phase-5L encoder, Phase-5U features, and Phase-5V features frozen.
- J15 is a label. Do not feed rollout statistics or future SINR into the network.
- Do not use 310110-310117 for gradients, early stopping, or hyperparameter search.
- Do not open 310118-310125 or 230000-230127.
- Do not modify Residual MAPPO, the reward, B2, tau, Gate-1, Candidate-C, or the planner.
- Do not call deployment decide().
- planner_ready and deployment_verified stay false.
