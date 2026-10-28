"""Stage A writes the locked schema and the pre-training feasibility evidence."""
from __future__ import annotations

import csv
import json
import platform

import numpy as np

from .feasibility import run_feasibility
from .protocol import (
    ACTOR_FEATURE_COUNT,
    DIAGNOSTIC_SCENARIOS,
    DISTANCE_THRESHOLD_M,
    E11_RESULT_COMMIT,
    FORMAL_ACTOR_SHA256,
    HEAD_FEATURE_COUNT,
    INSTRUCTION_COMMIT,
    LOOKAHEAD_NAMES,
    OUTPUT,
    PAYLOAD_CI_FLOOR_FRACTION,
    SAMPLE_ENV_INDEX,
    SAMPLE_UPDATES,
    SCHEMA_VERSION,
    UPDATES,
)
from .safety_audit import localize


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _cell(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    return str(value)


def _csv(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row[key]) for key in fields})


def schema() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "instruction_commit": INSTRUCTION_COMMIT,
        "parent_result_commit": E11_RESULT_COMMIT,
        "actor_feature_count": int(ACTOR_FEATURE_COUNT),
        "head_feature_count": int(HEAD_FEATURE_COUNT),
        "lookahead_names": list(LOOKAHEAD_NAMES),
        "critic_features": "three_head_features_only",
        "updates": int(UPDATES),
        "sample_updates": list(SAMPLE_UPDATES),
        "sample_env_index": int(SAMPLE_ENV_INDEX),
        "diagnostic_scenarios": list(DIAGNOSTIC_SCENARIOS),
        "payload_ci_floor_fraction": float(PAYLOAD_CI_FLOOR_FRACTION),
        "distance_threshold_m": float(DISTANCE_THRESHOLD_M),
        "formal_actor_sha256": FORMAL_ACTOR_SHA256,
        "reward_formula": "r_base + min(0.0005*A, 0.00025*P) + min(0.0010*T, 0.00075*P)",
        "groups": {"L4_C": "lookahead columns always zero", "L4_Q": "pre-service queue fractions"},
    }


def main() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "schema.json").write_text(json.dumps(schema(), indent=2), encoding="utf-8")
    feasibility = run_feasibility()
    safety = localize()
    _csv(OUTPUT / "feature_exposure_rows.csv", feasibility["rows"])
    _csv(OUTPUT / "feature_visibility_events.csv", feasibility["events"])
    (OUTPUT / "safety_pair_replay.json").write_text(json.dumps(_plain(safety), indent=2), encoding="utf-8")
    summary = {
        "stage": "A",
        "schema_version": SCHEMA_VERSION,
        "instruction_commit": INSTRUCTION_COMMIT,
        "feasibility_status": feasibility["status"],
        "trajectories": int(feasibility["trajectories"]),
        "ordinary_head_with_tagged_backlog_uav_steps": int(feasibility["ordinary_head_with_tagged_backlog_uav_steps"]),
        "episodes": feasibility["episodes"],
        "safety_matched_saved_rows": bool(safety["matched"]),
        "formal_actor_sha256": safety["formal_actor_sha256"],
        "formal_actor_hash_ok": bool(safety["formal_actor_hash_ok"]),
        "safety_cases": safety["cases"],
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
    }
    try:
        import torch

        summary["runtime"]["torch"] = torch.__version__
        summary["runtime"]["cuda"] = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - runtime record only
        summary["runtime"]["torch_error"] = str(exc)
    (OUTPUT / "stage_a_summary.json").write_text(json.dumps(_plain(summary), indent=2), encoding="utf-8")
    print(json.dumps(_plain({key: summary[key] for key in ("feasibility_status", "ordinary_head_with_tagged_backlog_uav_steps", "safety_matched_saved_rows", "formal_actor_hash_ok")}), indent=2))
    return summary


if __name__ == "__main__":
    main()
