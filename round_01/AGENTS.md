# round_01 Scope

Goal: implement and validate the minimal single-UAV, static multi-buoy maritime
communication environment.

Allowed scope was `src/wt/maritime/{config,channel,env,smoke}.py`, its public
exports, tests, and documentation. Learning models, queues, multi-UAV dynamics,
floating buoys, resource optimization, and offline data generation were
forbidden.

Acceptance was satisfied by 45 maritime tests, 78 total repository tests,
compile checks, a deterministic 20-step smoke rollout, and unchanged hashes for
the seven protected Waypoint Transformer core files.

This round is completed. Do not modify its lock record or add new functionality
under this directory.
