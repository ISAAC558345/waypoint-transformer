# Phase-5ZC

Audit only. Do not train a probe, scorer, value head, dynamics model, or MAPPO.
Do not open 310118-310125. Do not call deployment decide().
`planner_ready` and `deployment_verified` stay false.
Thresholds in `protocol.py` are locked and must not be moved after the audit.
