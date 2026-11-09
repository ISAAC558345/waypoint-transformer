# Phase-5X Frozen Executor Short-Horizon Lookahead

Measure whether a short frozen MAPPO rollout predicts the 15-step payload.

- Do not train a scorer, value network, encoder, or MAPPO.
- Do not add hand-made REP or TEMP features.
- Do not open 310118-310125 or 230000-230127.
- Shadow rollouts must leave the caller environment unchanged.
- H=15 is an oracle reference, not a planner.
- Do not call deployment decide().
- planner_ready and deployment_verified stay false.
