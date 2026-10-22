# Phase-5ZD

Train only the matched ExecutorConditionedJointScorer.
The HGNN encoder, REP, TEMP, candidate pool, and Residual MAPPO stay frozen.
Do not open 310118-310125. Do not call deployment decide().
Read 310110-310117 only after every seed has an epoch-10 checkpoint.
`planner_ready` and `deployment_verified` stay false.
Thresholds in `protocol.py` are locked.
