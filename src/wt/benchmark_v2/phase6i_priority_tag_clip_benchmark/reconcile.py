"""Check the frozen E14 pilot without retraining it or repairing its event log."""
from __future__ import annotations

import csv
import json

import torch

from .protocol import E14_OUTPUT, E14_RESULT_COMMIT


def _rows(path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: str) -> float:
    if value == "":
        raise RuntimeError("E14 reconciliation found an empty numeric cell")
    return float(value)


def reconcile_e14() -> dict:
    updates = _rows(E14_OUTPUT / "training_update_rows.csv")
    shadows = _rows(E14_OUTPUT / "shadow_update_rows.csv")
    events = json.loads((E14_OUTPUT / "safety_events.json").read_text(encoding="utf-8"))
    summary = json.loads((E14_OUTPUT / "summary.json").read_text(encoding="utf-8"))
    if len(updates) != 192 or len(shadows) != 18 or len(events) != 4:
        raise RuntimeError("E14 row counts do not match the registered pilot")
    if summary.get("scientific_gate") != "OPTIMIZER_INTERVENTION_COMPLETE":
        raise RuntimeError("frozen E14 gate changed")
    if summary.get("effect_assessment") != "MEASURABLE_OPTIMIZER_EFFECT":
        raise RuntimeError("frozen E14 effect changed")
    effect = []
    for row in shadows:
        if int(row["ppo_update"]) not in (16, 32):
            continue
        if row["r"] == "" or row["repeat_ok"] not in {"1", "True", "true"}:
            raise RuntimeError("an E14 effect comparison is not usable")
        if _float(row["repeat_max_abs"]) != 0.0:
            raise RuntimeError("E14 J-repeat gap is no longer zero")
        ratio = _float(row["r"])
        if ratio < 0.01:
            raise RuntimeError("E14 effect screen no longer holds")
        effect.append(ratio)
    if len(effect) != 12:
        raise RuntimeError("E14 does not have 12 effect comparisons")
    groups = {}
    for row in updates:
        key = (row["variant"], int(row["model_seed"]))
        groups.setdefault(key, []).append(row)
    if set(groups) != {("PILOT_J", seed) for seed in (0, 1, 2)} | {("PILOT_S", seed) for seed in (0, 1, 2)}:
        raise RuntimeError("E14 training groups drifted")
    if any(len(rows) != 32 for rows in groups.values()):
        raise RuntimeError("E14 did not record 32 updates for every group")
    table = []
    for seed in (0, 1, 2):
        joint = groups[("PILOT_J", seed)]
        separate = groups[("PILOT_S", seed)]
        joint_tagged = sum(_float(row["mean_served_tagged_bits"]) for row in joint) / 32.0
        separate_tagged = sum(_float(row["mean_served_tagged_bits"]) for row in separate) / 32.0
        joint_payload = sum(_float(row["mean_served_bits"]) for row in joint) / 32.0
        separate_payload = sum(_float(row["mean_served_bits"]) for row in separate) / 32.0
        table.append(
            {
                "model_seed": seed,
                "joint_tagged_mbit": joint_tagged / 1.0e6,
                "separate_tagged_mbit": separate_tagged / 1.0e6,
                "tagged_change": (separate_tagged - joint_tagged) / joint_tagged,
                "payload_change": (separate_payload - joint_payload) / joint_payload,
            }
        )
    published = (
        (0.230286, 0.226269, -0.0174, -0.0094),
        (0.229559, 0.223973, -0.0243, -0.0084),
        (0.227664, 0.227288, -0.0016, 0.0018),
    )
    for item, expected in zip(table, published):
        observed = (
            round(item["joint_tagged_mbit"], 6),
            round(item["separate_tagged_mbit"], 6),
            round(item["tagged_change"], 4),
            round(item["payload_change"], 4),
        )
        if observed != expected:
            raise RuntimeError(f"E14 training-pool table drifted for seed {item['model_seed']}: {observed}")
    checkpoints = []
    for path in sorted((E14_OUTPUT / "checkpoints").glob("pilot_*.pt")):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        checkpoints.append(
            {
                "path": str(path.relative_to(E14_OUTPUT.parent.parent)).replace("\\", "/"),
                "keys": sorted(str(key) for key in blob.keys()),
                "has_optimizer": "optimizer" in blob,
                "has_env": "env" in blob,
                "has_worker": "worker" in blob,
                "has_rng": "rng" in blob,
                "seamless_resume": False,
                "reason": "the pilot checkpoint has no environment, ledger, or worker cursor, so 32 plus 268 is not one continuous run",
            }
        )
    if len(checkpoints) != 6 or any(item["has_env"] or item["has_worker"] or item["seamless_resume"] for item in checkpoints):
        raise RuntimeError("E14 checkpoints cannot be treated as resumable 300-update runs")
    geometry_known = any("position_before" in event or "distance_after_m" in event for event in events)
    return {
        "e14_result_commit": E14_RESULT_COMMIT,
        "training_rows": len(updates),
        "shadow_rows": len(shadows),
        "effect_comparisons": len(effect),
        "effect_r_min": min(effect),
        "effect_r_max": max(effect),
        "repeat_max_abs": 0.0,
        "safety_events": len(events),
        "constraint_failures": int(sum(int(event["constraint_failure"]) for event in events)),
        "safety_geometry": "present" if geometry_known else "unknown",
        "training_pool_table": table,
        "table_role": "correlated_training_rollout_not_a_development_benchmark",
        "checkpoints": checkpoints,
        "fresh_start_required": True,
    }
