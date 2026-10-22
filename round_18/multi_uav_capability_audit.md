# Current multi-UAV capability audit

This audit was completed before the Round-18 extension.  It reads current
source only and does not inspect any protected data.

| Capability | Status | Evidence / consequence |
| --- | --- | --- |
| `num_uavs` configuration | missing | `MaritimeEnvConfig` exposes only single-UAV initial coordinates. |
| UAV position / velocity / last action | partial | `MaritimeEnv` stores one position and velocity, with one planar action. |
| UAV energy | partial | one remaining-energy scalar and movement energy are implemented. |
| joint action | missing | public action shape is `(2,)`, not `(M,2)`. |
| UAV mask | missing | no UAV node mask exists. |
| UAV--UAV distance / collision handling | missing | no pairwise UAV geometry or safety projection exists. |
| UAV--buoy rate | partial | current channel computes one UAV-to-all-buoys rate vector. |
| service association | missing | v1 uses fixed parallel service, not exclusive UAV--buoy assignment. |
| shared buoy queue | partial | one truthful queue per buoy already exists; no multi-UAV joint update exists. |
| existing multi-UAV helper | missing | no current source helper or planner supports multiple UAVs. |

Round-18 therefore adds only the missing multi-UAV task coupling around the
existing static physical primitives.  It does not duplicate queues or alter
the existing single-UAV path.
