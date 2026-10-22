# Phase-5Z Representation Collision Audit

Locate the Phase-5Y collision. Do not train.

- Do not update the encoder, REP, TEMP, value head, scorer, or MAPPO.
- Do not compute a new J15 rollout. Use the saved Phase-5X labels.
- Do not open 310118-310125 or 230000-230127.
- Do not fit a new scorer or value head.
- Do not modify Residual MAPPO, the reward, B2, tau, Gate-1, Candidate-C, or the planner.
- Do not call deployment decide().
- planner_ready and deployment_verified stay false.
