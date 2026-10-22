# Phase-5ZB Cross-Moment Audit

Diagnose whether a second-order UAV-target moment separates assignments.

- Do not train phi, the value head, the encoder, or MAPPO.
- Use 310100-310109 for the structural rates.
- 310114 t=15 is the pre-registered case, not a search set.
- Do not open 310118-310125 or use 310110-310117 to choose an aggregation.
- Do not modify Residual MAPPO, the reward, B2, tau, Gate-1, Candidate-C, or the planner.
- Do not call deployment decide().
- planner_ready and deployment_verified stay false.
