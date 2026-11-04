"""Draft interfaces for a later closed loop. Nothing here starts that loop."""

from __future__ import annotations


def interface_draft() -> dict:
    return {
        "started": False,
        "performance_pass": False,
        "adopted": False,
        "release_eligible": False,
        "node_counts": "from the case configuration, not a fixed 6 or 12 inside the checker",
        "isolation": "new package src/wt/ocean_observation_m0; formal planner, reward, observation and checkpoint are not loaded",
        "not_reused": [
            "configs/environment/l3b.json service_radius_m of 16 m is not a link budget",
            "configs/environment/l3b.json block_payload_bits of 1000000 is not the Hs product",
            "formal phase6g checkpoint weights",
            "old maritime episode transitions as the ocean truth model",
        ],
        "pattern_reused": [
            "config-driven node counts",
            "a transport ledger that records bytes, time, energy and duplicates separately",
            "a feasibility guard that can reject a task before launch",
        ],
        "interfaces": [
            {
                "name": "ObservationProduct",
                "fields": [
                    "version",
                    "buoy_id",
                    "seq",
                    "window_start_s",
                    "window_end_s",
                    "mean_x_m",
                    "mean_y_m",
                    "hs_m",
                    "measurement_variance_m2",
                    "quality",
                    "n_samples",
                    "framed_bytes",
                ],
                "event": "encoded at window end; unreadable by the shore until a complete frame arrives",
            },
            {
                "name": "NodeInbox",
                "fields": ["node_id", "received_at_s", "payload_kind", "version", "checksum_ok", "duplicate"],
                "event": "append on complete reception; partial frames stay out of the fused set",
            },
            {
                "name": "TransportLedger",
                "fields": ["link", "bytes", "airtime_s", "tx_J", "rx_J", "wait_s", "closed", "reason"],
                "event": "one row per transmission; retries are separate rows and are not used in M0",
            },
            {
                "name": "ShoreBelief",
                "fields": ["mean", "variance", "fused_keys", "last_update_s"],
                "event": "updates only from a complete, checksum-valid, new, unexpired product",
            },
            {
                "name": "CandidateBuilder",
                "fields": ["visit_buoy", "backhaul", "return", "wait"],
                "event": "every buoy remains a candidate even when its rate is low; M0 rates are high for every registered buoy",
            },
            {
                "name": "JointValuePlanner",
                "fields": ["predicted_error_drop", "arrival_time_s", "energy_J", "redundancy"],
                "event": "not implemented; would score only information the shore already holds",
            },
            {
                "name": "FeasibilityGuard",
                "fields": ["track_ground_speed", "collection_link", "return_energy", "reserve_J", "command_version", "valid_until_s"],
                "event": "reject before launch; geometry does not replace the energy reserve",
            },
            {
                "name": "Evaluator",
                "fields": ["shore_map_rmse", "delivery_time_s", "energy_J", "age_s", "events"],
                "event": "proposed for a later preregistered comparison; not run",
            },
        ],
        "minimum_closed_loop_not_started": {
            "scenario": "one fixed Hs field, one fixed wind, 2 UAVs and 4 buoys, abundant schedule",
            "baselines": ["direct buoy-to-shore", "one fixed tour"],
            "metric": "time-and-space-weighted shore RMSE over the whole square and the whole horizon",
            "threshold": "the 5% paired RMSE proposal stays unregistered and unevaluated",
            "training_updates": 0,
        },
    }
