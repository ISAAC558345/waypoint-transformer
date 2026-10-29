"""Read-only reconciliation of the finished clip-group comparison."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from wt.benchmark_v2.phase6i_priority_tag_clip_benchmark.criteria import choose_gate, score_full
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import WATCHED_PATHS
from wt.benchmark_v2.phase6i_priority_tag_reward.smoke import file_sha256
from wt.lowlevel_v2.phase1.protocol import D_SAFE_M

from .protocol import (
    COUNTER_EPS_M,
    DEV_SCENARIOS,
    E14_OUTPUT,
    E15_GATE,
    E15_OUTPUT,
    E15_RESULT_COMMIT,
    FORMAL_ACTOR_SHA256,
    MBIT,
    PROJECTOR_ACCEPT_EPS_M,
    PUBLISHED_PAYLOAD_MBIT,
    PUBLISHED_TAGGED_MBIT,
    ROOT,
)


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: str) -> float:
    if value == "":
        raise RuntimeError("closure reconciliation found an empty numeric cell")
    return float(value)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def counter_flags(distance_m: float) -> bool:
    return float(distance_m) + float(COUNTER_EPS_M) < float(D_SAFE_M)


def projector_accepts_separation(distance_m: float) -> bool:
    return float(distance_m) + float(PROJECTOR_ACCEPT_EPS_M) >= float(D_SAFE_M)


def pair_distance(left, right) -> float:
    return float(np.linalg.norm(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)))


def _scenario_rows() -> list[dict]:
    numeric = (
        "payload_bits",
        "served_tagged_bits",
        "sinr_mean",
        "interference_mean",
        "safety_violations",
        "constraint_failures",
    )
    rows = []
    for raw in _rows(E15_OUTPUT / "scenario_rows.csv"):
        row = dict(raw)
        row["model_seed"] = int(raw["model_seed"])
        row["scenario_id"] = int(raw["scenario_id"])
        for key in numeric:
            row[key] = _float(raw[key])
        rows.append(row)
    return rows


def _coverage(rows: list[dict]) -> dict:
    problems = []
    keys = [(row["variant"], int(row["model_seed"]), int(row["scenario_id"])) for row in rows]
    if len(rows) != 224:
        problems.append(f"scenario row count is {len(rows)}, not 224")
    if len(keys) != len(set(keys)):
        problems.append("scenario keys are not unique")
    counts = {}
    for variant, seed, _scenario in keys:
        counts[(variant, seed)] = counts.get((variant, seed), 0) + 1
    expected = {("L0", 0): 32}
    expected.update({("FULL_J", seed): 32 for seed in (0, 1, 2)})
    expected.update({("FULL_S", seed): 32 for seed in (0, 1, 2)})
    if counts != expected:
        problems.append(f"group coverage is {counts}")
    for variant, seed in expected:
        scenarios = sorted(row["scenario_id"] for row in rows if row["variant"] == variant and int(row["model_seed"]) == seed)
        if tuple(scenarios) != DEV_SCENARIOS:
            problems.append(f"{variant} seed {seed} does not cover the registered development pool once")
    return {"problems": problems, "l0_rows": counts.get(("L0", 0), 0), "full_j_rows": sum(counts.get(("FULL_J", seed), 0) for seed in (0, 1, 2)), "full_s_rows": sum(counts.get(("FULL_S", seed), 0) for seed in (0, 1, 2))}


def _max_gap(saved, recomputed) -> float:
    gap = 0.0
    if isinstance(saved, dict) and isinstance(recomputed, dict):
        for key, value in saved.items():
            if key not in recomputed:
                continue
            gap = max(gap, _max_gap(value, recomputed[key]))
        return gap
    if isinstance(saved, list) and isinstance(recomputed, list):
        if len(saved) != len(recomputed):
            return max(gap, 1.0e9)
        for left, right in zip(saved, recomputed):
            gap = max(gap, _max_gap(left, right))
        return gap
    if isinstance(saved, (int, float)) and isinstance(recomputed, (int, float)) and not isinstance(saved, bool):
        return max(gap, abs(float(saved) - float(recomputed)))
    return gap


def _criteria_matrix(score: dict) -> list[dict]:
    rows = []
    for seed in score["seeds"]:
        for key, label in (("versus_full_j", "FULL_J"), ("versus_l0", "L0")):
            item = seed[key]
            checks = {
                "tagged_ci_low_positive": float(item["tagged"]["ci_low"]) > 0.0,
                "payload_mean_nonnegative": float(item["payload"]["mean"]) >= 0.0,
                "payload_ci_within_floor": float(item["payload"]["ci_low"]) >= float(item["payload_ci_floor"]),
                "sinr_mean_nonnegative": float(item["sinr_mean"]) >= 0.0,
                "interference_mean_nonpositive": float(item["interference_mean"]) <= 0.0,
                "leave_one_tagged": bool(item["leave_one_tagged"]),
                "leave_two_tagged": bool(item["leave_two_tagged"]),
            }
            failed = [name for name, ok in checks.items() if not ok]
            rows.append(
                {
                    "model_seed": int(seed["model_seed"]),
                    "reference": label,
                    "tagged_mean_mbit": float(item["tagged"]["mean"]) / MBIT,
                    "tagged_ci_low_mbit": float(item["tagged"]["ci_low"]) / MBIT,
                    "tagged_ci_high_mbit": float(item["tagged"]["ci_high"]) / MBIT,
                    "payload_mean_mbit": float(item["payload"]["mean"]) / MBIT,
                    "payload_ci_low_mbit": float(item["payload"]["ci_low"]) / MBIT,
                    "payload_ci_floor_mbit": float(item["payload_ci_floor"]) / MBIT,
                    "sinr_mean_delta": float(item["sinr_mean"]),
                    "interference_mean_delta": float(item["interference_mean"]),
                    "safety_diff_sum": float(item["safety_diff_sum"]),
                    "constraint_diff_sum": float(item["constraint_diff_sum"]),
                    "leave_one_tagged": bool(item["leave_one_tagged"]),
                    "leave_two_tagged": bool(item["leave_two_tagged"]),
                    "reference_checks_met": not failed,
                    "failed_checks": failed,
                    "reason": "all registered reference checks hold" if not failed else "failed: " + ",".join(failed),
                }
            )
    return rows


def _regressions(rows: list[dict]) -> list[dict]:
    index = {(row["variant"], int(row["model_seed"]), int(row["scenario_id"])): row for row in rows}
    found = []
    for seed in (0, 1, 2):
        for scenario in DEV_SCENARIOS:
            current = index[("FULL_S", seed, scenario)]
            for reference, ref_seed in (("FULL_J", seed), ("L0", 0)):
                baseline = index[(reference, ref_seed, scenario)]
                tagged = float(current["served_tagged_bits"]) - float(baseline["served_tagged_bits"])
                payload = float(current["payload_bits"]) - float(baseline["payload_bits"])
                if tagged < 0.0 or payload < 0.0:
                    found.append(
                        {
                            "model_seed": seed,
                            "reference": reference,
                            "scenario_id": int(scenario),
                            "tagged_delta_bits": tagged,
                            "payload_delta_bits": payload,
                            "tagged_regression": tagged < 0.0,
                            "payload_regression": payload < 0.0,
                        }
                    )
    return found


def _budget() -> dict:
    updates = _rows(E15_OUTPUT / "training_update_rows.csv")
    steps = _rows(E15_OUTPUT / "optimizer_step_rows.csv")
    manifest = _load_json(E15_OUTPUT / "checkpoint_manifest.json")
    problems = []
    if len(updates) != 1800:
        problems.append(f"training update rows are {len(updates)}, not 1800")
    if len(steps) != 28800:
        problems.append(f"optimizer step rows are {len(steps)}, not 28800")
    optimizer_sum = sum(int(float(row["optimizer_steps"])) for row in updates)
    if optimizer_sum != 28800:
        problems.append(f"recorded optimizer_steps sum to {optimizer_sum}, not 28800")
    groups = {}
    for row in updates:
        groups.setdefault((row["variant"], int(row["model_seed"])), 0)
        groups[(row["variant"], int(row["model_seed"]))] += 1
    if groups != {("FULL_J", seed): 300 for seed in (0, 1, 2)} | {("FULL_S", seed): 300 for seed in (0, 1, 2)}:
        problems.append("a full-budget group does not have 300 updates")
    checkpoints = []
    for item in manifest:
        path = ROOT / str(item["path"])
        blob = torch.load(path, map_location="cpu", weights_only=False)
        observed = file_sha256(path)
        record = {
            "path": item["path"],
            "manifest_sha256": item.get("sha256"),
            "recomputed_sha256": observed,
            "ppo_updates": int(blob.get("ppo_updates", -1)),
            "seamless_resume": blob.get("seamless_resume"),
            "has_env": "env" in blob,
            "has_worker": "worker" in blob,
            "has_ledger": "ledger" in blob,
        }
        checkpoints.append(record)
        if observed != item.get("sha256") or int(blob.get("ppo_updates", -1)) != 300 or blob.get("seamless_resume") is not False:
            problems.append(f"checkpoint identity drifted for {item['path']}")
        if record["has_env"] or record["has_worker"] or record["has_ledger"]:
            problems.append(f"checkpoint contains environment state: {item['path']}")
        del blob
    snapshots = []
    for path in sorted((E15_OUTPUT / "snapshots").glob("*.pt")):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        snapshots.append(
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "seamless_resume": blob.get("seamless_resume"),
                "omitted": list(blob.get("omitted", [])),
                "has_env": "env" in blob,
                "has_ledger": "ledger" in blob,
                "has_option": "option" in blob,
            }
        )
        if blob.get("seamless_resume") is not False or "env" in blob or "ledger" in blob or "option" in blob:
            problems.append(f"snapshot claims more state than it stores: {path.name}")
        del blob
    if len(snapshots) != 18:
        problems.append(f"snapshot count is {len(snapshots)}, not 18")
    return {"problems": problems, "ppo_updates": len(updates), "adam_steps": len(steps), "checkpoints": checkpoints, "snapshots": snapshots}


def _event_geometry(event: dict, index: int, source_sha: str) -> dict:
    recorded_after = event.get("distance_after_m")
    recorded_before = event.get("distance_before_m")
    recorded_margin = event.get("margin_after_m")
    row = {
        "source_event_index": int(index),
        "source_sha256": source_sha,
        "variant": event.get("variant"),
        "model_seed": event.get("model_seed"),
        "ppo_update": event.get("ppo_update"),
        "env_index": event.get("env_index"),
        "safety_violation": int(event.get("safety_violation", 0)),
        "constraint_failure": int(event.get("constraint_failure", 0)),
        "env_t": "unrecorded",
        "scenario_id": "unrecorded",
        "geometry": event.get("geometry", "unknown"),
        "predicted_distance_scope": "projector_global_minimum_not_this_pair",
        "predicted_distance_m": event.get("predicted_distance_m"),
    }
    if event.get("geometry") != "measured" or "position_after_i" not in event:
        row["classification"] = "geometry_unknown"
        row["distance_recomputed"] = False
        return row
    after = pair_distance(event["position_after_i"], event["position_after_j"])
    before = pair_distance(event["position_before_i"], event["position_before_j"])
    margin = after - float(D_SAFE_M)
    row.update(
        {
            "distance_before_recomputed_m": before,
            "distance_after_recomputed_m": after,
            "margin_after_recomputed_m": margin,
            "distance_before_gap_m": abs(before - float(recorded_before)),
            "distance_after_gap_m": abs(after - float(recorded_after)),
            "margin_gap_m": abs(margin - float(recorded_margin)),
            "counter_flags": counter_flags(after),
            "projector_would_accept_separation": projector_accepts_separation(after),
            "distance_recomputed": True,
        }
    )
    if int(event.get("safety_violation", 0)) == 1 and counter_flags(after) and projector_accepts_separation(after):
        row["classification"] = "counter_violation_inside_projector_acceptance_band"
    elif int(event.get("safety_violation", 0)) == 1:
        row["classification"] = "safety_violation_outside_or_unclassified"
    elif int(event.get("constraint_failure", 0)) == 1:
        row["classification"] = "constraint_failure"
    else:
        row["classification"] = "other"
    return row


def _safety() -> dict:
    path = E15_OUTPUT / "safety_events.json"
    events = _load_json(path)
    source_sha = file_sha256(path)
    reviewed = [_event_geometry(event, index, source_sha) for index, event in enumerate(events)]
    safety = [row for row in reviewed if int(row["safety_violation"]) == 1]
    constraints = [row for row in reviewed if int(row["constraint_failure"]) == 1]
    problems = []
    if len(events) != 41 or len(safety) != 40 or len(constraints) != 1:
        problems.append(f"event split is records={len(events)} safety={len(safety)} constraint={len(constraints)}")
    by_variant = {}
    for row in safety:
        by_variant[row["variant"]] = by_variant.get(row["variant"], 0) + 1
    if by_variant.get("FULL_J") != 29 or by_variant.get("FULL_S") != 11:
        problems.append(f"training safety split is {by_variant}")
    if any(int(row["constraint_failure"]) != 0 for row in safety) or any(int(row["safety_violation"]) != 0 for row in constraints):
        problems.append("a record mixes the safety count with the constraint count")
    gaps = [float(row["distance_after_gap_m"]) for row in safety if row.get("distance_recomputed")]
    margins = [float(row["margin_after_recomputed_m"]) for row in safety if row.get("distance_recomputed")]
    if len(gaps) != 40 or max(gaps) > 1.0e-9 or max(float(row["margin_gap_m"]) for row in safety) > 1.0e-9:
        problems.append("recomputed safety geometry does not match the saved event")
    boundary = [row for row in safety if row.get("classification") == "counter_violation_inside_projector_acceptance_band"]
    if len(boundary) != 40:
        problems.append("not every saved safety violation sits inside the projector acceptance band")
    updates = _rows(E15_OUTPUT / "training_update_rows.csv")
    update_safety = {}
    update_constraint = {}
    for row in updates:
        update_safety[row["variant"]] = update_safety.get(row["variant"], 0) + int(float(row["safety_violation_count"]))
        update_constraint[row["variant"]] = update_constraint.get(row["variant"], 0) + int(float(row["constraint_failure_count"]))
    if update_safety != {"FULL_J": 29, "FULL_S": 11} or update_constraint != {"FULL_J": 1, "FULL_S": 0}:
        problems.append(f"training csv counts are safety {update_safety}, constraint {update_constraint}")
    target = constraints[0] if len(constraints) == 1 else None
    constraint_review = _constraint_review(target, events)
    if constraint_review["problems"]:
        problems.extend(constraint_review["problems"])
    e14_events = _load_json(E14_OUTPUT / "safety_events.json")
    e14_geometry = "unknown" if all("position_before_i" not in event and "distance_after_m" not in event for event in e14_events) else "present"
    if len(e14_events) != 4 or e14_geometry != "unknown":
        problems.append("E14 safety evidence no longer matches the frozen unknown-geometry record")
    return {
        "problems": problems,
        "source_sha256": source_sha,
        "records": len(events),
        "safety_violations": len(safety),
        "constraint_failures": len(constraints),
        "safety_by_variant": by_variant,
        "margin_min_m": None if not margins else min(margins),
        "margin_max_m": None if not margins else max(margins),
        "boundary_count": len(boundary),
        "boundary_does_not_reduce_count": len(safety) == 40,
        "rows": reviewed,
        "constraint_review": constraint_review,
        "e14_safety_events": len(e14_events),
        "e14_geometry": e14_geometry,
        "training_counts_are_not_development_counts": True,
    }


def _constraint_review(row: dict | None, events: list[dict]) -> dict:
    problems = []
    if row is None:
        return {"problems": ["the single constraint failure was not found"], "supported": False}
    event = events[int(row["source_event_index"])]
    executed = np.asarray(event.get("executed_action", []), dtype=np.float64)
    predicted = event.get("predicted_distance_m")
    after = row.get("distance_after_recomputed_m")
    supported = (
        event.get("variant") == "FULL_J"
        and int(event.get("model_seed")) == 1
        and int(event.get("ppo_update")) == 119
        and int(event.get("env_index")) == 2
        and event.get("constraint_status") == "failure"
        and int(event.get("safety_violation", 0)) == 0
        and executed.size > 0
        and float(np.max(np.abs(executed))) == 0.0
        and predicted is not None
        and abs(float(predicted) - 0.9999192009856496) <= 1.0e-12
        and after is not None
        and abs(float(after) - 8.75938528785752) <= 1.0e-9
    )
    if not supported:
        problems.append("the constraint-failure record does not match the saved geometry")
    return {
        "problems": problems,
        "supported": bool(supported),
        "source_event_index": int(row["source_event_index"]),
        "variant": event.get("variant"),
        "model_seed": event.get("model_seed"),
        "ppo_update": event.get("ppo_update"),
        "env_index": event.get("env_index"),
        "env_t": "unrecorded",
        "scenario_id": "unrecorded",
        "constraint_status": event.get("constraint_status"),
        "predicted_distance_m": predicted,
        "predicted_distance_scope": "projector_global_minimum_of_the_rejected_candidate",
        "executed_action_max_abs": float(np.max(np.abs(executed))) if executed.size else None,
        "executed_action": "all_zero_fallback",
        "distance_after_m": after,
        "distance_after_scope": "closest_saved_pair_after_the_zero_action",
        "not_a_prediction_error": True,
        "reason": "the rejected projection was not executed; the zero fallback and its later distance are a different action from the predicted candidate",
    }


def _published(score: dict) -> list[str]:
    problems = []
    for seed, expected_tagged, expected_payload in zip(score["seeds"], PUBLISHED_TAGGED_MBIT, PUBLISHED_PAYLOAD_MBIT):
        joint = seed["versus_full_j"]["tagged"]
        base = seed["versus_l0"]["tagged"]
        observed = (
            round(float(joint["mean"]) / MBIT, 6),
            round(float(joint["ci_low"]) / MBIT, 6),
            round(float(joint["ci_high"]) / MBIT, 6),
            round(float(base["mean"]) / MBIT, 6),
            round(float(base["ci_low"]) / MBIT, 6),
            round(float(base["ci_high"]) / MBIT, 6),
        )
        if observed != expected_tagged:
            problems.append(f"tagged Mbit table drifted for seed {seed['model_seed']}: {observed}")
        payload = round(float(seed["versus_full_j"]["payload"]["mean"]) / MBIT, 6)
        if payload != expected_payload:
            problems.append(f"payload Mbit mean drifted for seed {seed['model_seed']}: {payload}")
        if float(seed["versus_full_j"]["payload"]["ci_low"]) >= float(seed["versus_full_j"]["payload_ci_floor"]):
            problems.append(f"seed {seed['model_seed']} payload interval is not below the registered floor")
        if float(seed["versus_l0"]["interference_mean"]) <= 0.0:
            problems.append(f"seed {seed['model_seed']} interference delta versus L0 is not positive")
    return problems


def _blob(relative: str) -> bytes:
    import subprocess

    proc = subprocess.run(
        ["git", "-c", "core.longpaths=true", "show", f"{E15_RESULT_COMMIT}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"cannot read {relative} at the E15 result commit")
    return proc.stdout


def _same_committed_source(relative: str) -> bool:
    current = (ROOT / relative).read_bytes()
    blob = _blob(relative)
    if current == blob:
        return True
    if relative.endswith(".py"):
        return current.replace(b"\r\n", b"\n") == blob.replace(b"\r\n", b"\n")
    return False


def _formal() -> dict:
    problems = []
    current = {}
    changed = []
    for relative in WATCHED_PATHS:
        digest = file_sha256(ROOT / relative)
        current[relative] = digest
        if not _same_committed_source(relative):
            changed.append(relative)
            problems.append(f"watched path changed after E15: {relative}")
    formal = "src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt"
    if current.get(formal) != FORMAL_ACTOR_SHA256:
        problems.append("formal low-level checkpoint hash is not the registered SHA-256")
    return {
        "problems": problems,
        "watched": current,
        "changed_after_e15": changed,
        "formal_actor_sha256": current.get(formal),
        "matches_e15_manifest": not problems,
        "comparison": "working tree against the E15 result blob; Python sources ignore checkout newline conversion",
    }


def _lineage() -> dict:
    specs = (
        ("E9", ROOT / "results/phase6i_priority_tag_attribution_full_replay/summary.json", "gate", "PRIORITY_TAG_ATTRIBUTION_COMPLETE"),
        ("E10", ROOT / "results/phase6i_priority_tag_direct_service_reward/summary.json", "scientific_gate", "DIRECT_TAG_REWARD_BENCHMARK_FAIL"),
        ("E11", ROOT / "results/phase6i_priority_tag_signal_audit/summary.json", "gate", "DIRECT_TAG_SIGNAL_AUDIT_COMPLETE"),
        ("E12", ROOT / "results/phase6i_priority_tag_queue_lookahead/summary.json", "scientific_gate", "QUEUE_LOOKAHEAD_BENCHMARK_FAIL"),
        ("E13", ROOT / "results/phase6i_priority_tag_credit_audit/summary.json", "scientific_gate", "TAG_CREDIT_AUDIT_COMPLETE"),
        ("E14", E14_OUTPUT / "summary.json", "scientific_gate", "OPTIMIZER_INTERVENTION_COMPLETE"),
        ("E15", E15_OUTPUT / "summary.json", "scientific_gate", E15_GATE),
    )
    problems = []
    found = {}
    for name, path, key, expected in specs:
        body = _load_json(path)
        found[name] = body.get(key)
        if body.get(key) != expected:
            problems.append(f"{name} gate is {body.get(key)!r}, not {expected}")
    e14 = _load_json(E14_OUTPUT / "summary.json")
    e13 = _load_json(ROOT / "results/phase6i_priority_tag_credit_audit/summary.json")
    e15 = _load_json(E15_OUTPUT / "summary.json")
    if e14.get("effect_assessment") != "MEASURABLE_OPTIMIZER_EFFECT":
        problems.append("E14 effect assessment changed")
    if e13.get("recommendation") != "no_single_component":
        problems.append("E13 recommendation changed")
    if int(e15.get("score", {}).get("passing_seeds", -1)) != 0 or e15.get("release_eligible") is not False or e15.get("merged") is not False:
        problems.append("E15 adoption fields changed")
    if int(e15.get("ppo_updates", -1)) != 1800 or int(e15.get("adam_steps", -1)) != 28800:
        problems.append("E15 saved budget fields changed")
    return {"problems": problems, "gates": found}


def audit() -> dict:
    rows = _scenario_rows()
    coverage = _coverage(rows)
    score = score_full(rows)
    recomputed_gate = choose_gate(tests_ok=True, train_ok=True, hashes_ok=True, preflight_ok=True, score=score)
    saved = _load_json(E15_OUTPUT / "summary.json")
    gap = _max_gap(saved.get("score"), score)
    problems = []
    problems.extend(coverage["problems"])
    if recomputed_gate != E15_GATE or saved.get("scientific_gate") != E15_GATE:
        problems.append(f"recomputed gate {recomputed_gate} does not preserve {E15_GATE}")
    if int(score["passing_seeds"]) != 0 or int(saved.get("score", {}).get("passing_seeds", -1)) != 0:
        problems.append("passing_seeds is not 0")
    if gap > 1.0e-6:
        problems.append(f"saved score and recomputed score differ by {gap}")
    problems.extend(_published(score))
    budget = _budget()
    safety = _safety()
    formal = _formal()
    lineage = _lineage()
    for piece in (budget, safety, formal, lineage):
        problems.extend(piece["problems"])
    return {
        "problems": problems,
        "coverage": coverage,
        "score": score,
        "recomputed_e15_gate": recomputed_gate,
        "score_max_abs_gap": gap,
        "criteria_matrix": _criteria_matrix(score),
        "regressions": _regressions(rows),
        "budget": budget,
        "safety": safety,
        "formal": formal,
        "lineage": lineage,
        "consistent": not problems,
    }
