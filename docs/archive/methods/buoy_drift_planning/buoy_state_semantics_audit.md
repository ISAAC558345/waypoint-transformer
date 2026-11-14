# Existing buoy-state semantics audit

Audit source: `src/wt/maritime/env.py`, `config.py`, `trajectory.py`, and
`dataset_v1.py`, inspected before any Round-16 environment modification.

| Field | Existing implementation | Observation exposure |
| --- | --- | --- |
| Buoy x/y position | `_buoy_positions_m`, shape `(num_buoys, 2)` | Relative x/y per buoy and raw `info['buoy_positions_m']` |
| Buoy velocity / drift velocity | Not implemented | Not present |
| Predicted future buoy position | Not implemented | Not present |
| Queue | `_queue_bits` | v1 per-buoy normalized queue and raw `raw_queue_bits` |
| Arrival | `_last_arrival_bits` | v1 per-buoy normalized last arrival and raw `raw_arrival_bits` |
| Rate information | channel metrics computed from current positions | v1 normalized allocated rate and raw `rates_bps` |

The pre-Round-16 v1 schema has nine global and seven per-buoy features. It is
static-buoy specific: `dataset_v1.py` intentionally rejects trajectories where
buoy positions change. Round-16 therefore preserves the default static schema
exactly and uses a new dynamic-only observation schema with two additional
per-buoy, physically normalized drift-velocity features. Diagnostic
trajectories are serialized directly under Round-16 and are not supplied to
the locked training-dataset interface.
