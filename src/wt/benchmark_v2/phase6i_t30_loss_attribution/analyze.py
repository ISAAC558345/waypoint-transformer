"""Attribute the saved t=30 preserving counterfactual. No replay and no training."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
P4_RESULTS = REPO / "results" / "phase6i_t30_active_preserve"
OUTPUT = REPO / "results" / "phase6i_t30_loss_attribution"
VALIDATION_IDS = tuple(range(322048, 322064))
BIT_TIE = 1.0
SCORE_EPS = 1.0e-6
DISTANCE_EPS_M = 1.0e-4
P4_RESULTS_COMMIT = "0ef2d332575400e8b278e63c3c0391568dfeccc2"
PLAN_COMMIT = "5a2075f3d701717dfa7c8bd4f5ae42ba582844b0"
EXPECTED_HASHES = {
    "config_hash": "3535008ed197f28ed3620a25493a9cc17fbb1fca9742d49a3d58957883319a63",
    "scheme_c_sha256": "f77c0bcef3f359e01de35b75753f3776a9e8332bf65fd191f112236df074f864",
    "mappo_sha256": "92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e",
    "decoder_sha256": "de1809fab8290b5be3e89cd0d31e22cd04638fd9a43207957aa20b1de255eca5",
}
MISSING_FIELDS = (
    "per_step_served_bits",
    "per_step_sinr",
    "per_step_interference",
    "per_step_uav_position",
    "per_step_action",
    "absolute_uav_coordinates",
    "per_buoy_queue",
    "per_uav_queue",
    "per_step_safety_projector_events",
)
SCENARIO_FIELDS = (
    "scenario_id",
    "comparison_status",
    "state_hash",
    "branch_hash_match",
    "pool_hash",
    "changed_uavs",
    "fixed_uavs",
    "reassignable_uavs",
    "scheme_assignment",
    "preserve_assignment",
    "scheme_reassignable_reachable",
    "preserve_reassignable_reachable",
    "scheme_reassignable_distance_m",
    "preserve_reassignable_distance_m",
    "delta_reassignable_distance_m",
    "scheme_score",
    "preserve_score",
    "score_delta",
    "score_compared",
    "score_sign_matches_window",
    "score_sign_matches_episode",
    "delta_window_bits",
    "tail_payload_scheme_bits",
    "tail_payload_preserve_bits",
    "delta_tail_bits",
    "delta_episode_bits",
    "delta_branch_bits",
    "identity_error_bits",
    "uav_window_sum_error_bits",
    "uav_tail_sum_error_bits",
    "arm_level_error_bits",
    "accounting_error_scheme_bits",
    "accounting_error_preserve_bits",
    "tail_effect",
    "dominant_interval",
    "reassigned_window_bits",
    "fixed_window_bits",
    "reassigned_tail_bits",
    "fixed_tail_bits",
    "t45_assignment_changed",
    "t45_scheme_assignment",
    "t45_preserve_assignment",
    "safety_violation_rate_scheme",
    "safety_violation_rate_preserve",
    "queue_before_bits",
    "queue_after_window_scheme_bits",
    "queue_after_window_preserve_bits",
    "config_hash",
    "scheme_c_sha256",
    "mappo_sha256",
    "decoder_sha256",
)
UAV_FIELDS = (
    "scenario_id",
    "comparison_status",
    "uav_id",
    "group",
    "role_at_t30",
    "t30_target_scheme",
    "t30_target_preserve",
    "t30_assignment_changed",
    "window_served_scheme_bits",
    "window_served_preserve_bits",
    "delta_window_served_bits",
    "window_service_steps_scheme",
    "window_service_steps_preserve",
    "delta_window_service_steps",
    "window_path_scheme_m",
    "window_path_preserve_m",
    "delta_window_path_m",
    "t30_distance_scheme_m",
    "t30_distance_preserve_m",
    "delta_t30_distance_m",
    "t45_distance_to_t30_target_scheme_m",
    "t45_distance_to_t30_target_preserve_m",
    "window_entered_scheme",
    "window_entered_preserve",
    "window_first_entry_scheme",
    "window_first_entry_preserve",
    "window_projection_scheme",
    "window_projection_preserve",
    "window_mean_sinr_scheme",
    "window_mean_sinr_preserve",
    "delta_window_mean_sinr",
    "window_mean_interference_scheme",
    "window_mean_interference_preserve",
    "delta_window_mean_interference",
    "t45_target_scheme",
    "t45_target_preserve",
    "t45_assignment_changed",
    "t45_distance_to_t45_target_scheme_m",
    "t45_distance_to_t45_target_preserve_m",
    "tail_served_scheme_bits",
    "tail_served_preserve_bits",
    "delta_tail_served_bits",
    "tail_service_steps_scheme",
    "tail_service_steps_preserve",
    "delta_tail_service_steps",
    "tail_path_scheme_m",
    "tail_path_preserve_m",
    "tail_end_distance_scheme_m",
    "tail_end_distance_preserve_m",
    "tail_projection_scheme",
    "tail_projection_preserve",
    "tail_mean_sinr_scheme",
    "tail_mean_sinr_preserve",
    "delta_tail_mean_sinr",
    "tail_mean_interference_scheme",
    "tail_mean_interference_preserve",
    "delta_tail_mean_interference",
)


def classify_tail_effect(window_bits: float, tail_bits: float) -> str:
    """Say whether the t=45-48 gap enlarges or offsets the t=30-45 gap."""
    window_open = abs(window_bits) > BIT_TIE
    tail_open = abs(tail_bits) > BIT_TIE
    if not tail_open:
        return "tie"
    if not window_open:
        return "tail_only"
    if (window_bits > 0) == (tail_bits > 0):
        return "enlarges"
    return "offsets"


def dominant_interval(window_bits: float, tail_bits: float) -> str:
    if abs(window_bits) <= BIT_TIE and abs(tail_bits) <= BIT_TIE:
        return "neither"
    if abs(window_bits) > abs(tail_bits) + BIT_TIE:
        return "window"
    if abs(tail_bits) > abs(window_bits) + BIT_TIE:
        return "tail"
    return "comparable"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict, key: str) -> float:
    return float(row[key])


def _optional_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _sign_agree(left: float, right: float) -> int:
    return int((left > 0) == (right > 0))


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    if float(np.std(left)) <= 1.0e-12 or float(np.std(right)) <= 1.0e-12:
        return None
    return float(np.corrcoef(np.asarray(left), np.asarray(right))[0, 1])


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _index_uavs(rows: list[dict]) -> dict[tuple[str, str, str, str], dict]:
    return {(row["scenario_id"], row["arm"], row["window_t"], row["uav_id"]): row for row in rows}


def _pair_uav(scenario: dict, uav_id: int, changed: set[str], index: dict) -> dict:
    scenario_id = scenario["scenario_id"]
    key = str(uav_id)
    scheme_w = index[(scenario_id, "scheme", "30", key)]
    preserve_w = index[(scenario_id, "preserve", "30", key)]
    scheme_t = index[(scenario_id, "scheme", "45", key)]
    preserve_t = index[(scenario_id, "preserve", "45", key)]
    if scheme_w["role_at_t30"] != preserve_w["role_at_t30"]:
        raise ValueError(f"{scenario_id} UAV {uav_id} role differs across arms")
    scheme_now = [int(item) for item in scenario["scheme_assignment"].split(",")]
    preserve_now = [int(item) for item in scenario["preserve_assignment"].split(",")]
    if int(scheme_w["target_buoy_id"]) != scheme_now[uav_id] or int(preserve_w["target_buoy_id"]) != preserve_now[uav_id]:
        raise ValueError(f"{scenario_id} UAV {uav_id} t=30 target does not match the scenario assignment")
    scheme_targets = scenario["t45_scheme_assignment"].split(",")
    preserve_targets = scenario["t45_preserve_assignment"].split(",")
    distance_scheme = _optional_float(scheme_w["initial_distance_m"])
    distance_preserve = _optional_float(preserve_w["initial_distance_m"])
    distance_delta = None
    if distance_scheme is not None and distance_preserve is not None:
        distance_delta = distance_preserve - distance_scheme
    return {
        "scenario_id": int(scenario_id),
        "comparison_status": scenario["comparison_status"],
        "uav_id": uav_id,
        "group": "reassigned" if key in changed else "fixed",
        "role_at_t30": scheme_w["role_at_t30"],
        "t30_target_scheme": int(scheme_w["target_buoy_id"]),
        "t30_target_preserve": int(preserve_w["target_buoy_id"]),
        "t30_assignment_changed": int(key in changed),
        "window_served_scheme_bits": _float(scheme_w, "served_bits"),
        "window_served_preserve_bits": _float(preserve_w, "served_bits"),
        "delta_window_served_bits": _float(preserve_w, "served_bits") - _float(scheme_w, "served_bits"),
        "window_service_steps_scheme": int(scheme_w["effective_service_steps"]),
        "window_service_steps_preserve": int(preserve_w["effective_service_steps"]),
        "delta_window_service_steps": int(preserve_w["effective_service_steps"]) - int(scheme_w["effective_service_steps"]),
        "window_path_scheme_m": _float(scheme_w, "path_length_m"),
        "window_path_preserve_m": _float(preserve_w, "path_length_m"),
        "delta_window_path_m": _float(preserve_w, "path_length_m") - _float(scheme_w, "path_length_m"),
        "t30_distance_scheme_m": distance_scheme,
        "t30_distance_preserve_m": distance_preserve,
        "delta_t30_distance_m": distance_delta,
        "t45_distance_to_t30_target_scheme_m": _optional_float(scheme_w["end_distance_m"]),
        "t45_distance_to_t30_target_preserve_m": _optional_float(preserve_w["end_distance_m"]),
        "window_entered_scheme": int(scheme_w["entered_service_radius"]),
        "window_entered_preserve": int(preserve_w["entered_service_radius"]),
        "window_first_entry_scheme": _optional_float(scheme_w["first_entry_step"]),
        "window_first_entry_preserve": _optional_float(preserve_w["first_entry_step"]),
        "window_projection_scheme": int(scheme_w["projection_count"]),
        "window_projection_preserve": int(preserve_w["projection_count"]),
        "window_mean_sinr_scheme": _float(scheme_w, "mean_sinr"),
        "window_mean_sinr_preserve": _float(preserve_w, "mean_sinr"),
        "delta_window_mean_sinr": _float(preserve_w, "mean_sinr") - _float(scheme_w, "mean_sinr"),
        "window_mean_interference_scheme": _float(scheme_w, "mean_interference"),
        "window_mean_interference_preserve": _float(preserve_w, "mean_interference"),
        "delta_window_mean_interference": _float(preserve_w, "mean_interference") - _float(scheme_w, "mean_interference"),
        "t45_target_scheme": int(scheme_targets[uav_id]),
        "t45_target_preserve": int(preserve_targets[uav_id]),
        "t45_assignment_changed": int(scheme_targets[uav_id] != preserve_targets[uav_id]),
        "t45_distance_to_t45_target_scheme_m": _optional_float(scheme_t["initial_distance_m"]),
        "t45_distance_to_t45_target_preserve_m": _optional_float(preserve_t["initial_distance_m"]),
        "tail_served_scheme_bits": _float(scheme_t, "served_bits"),
        "tail_served_preserve_bits": _float(preserve_t, "served_bits"),
        "delta_tail_served_bits": _float(preserve_t, "served_bits") - _float(scheme_t, "served_bits"),
        "tail_service_steps_scheme": int(scheme_t["effective_service_steps"]),
        "tail_service_steps_preserve": int(preserve_t["effective_service_steps"]),
        "delta_tail_service_steps": int(preserve_t["effective_service_steps"]) - int(scheme_t["effective_service_steps"]),
        "tail_path_scheme_m": _float(scheme_t, "path_length_m"),
        "tail_path_preserve_m": _float(preserve_t, "path_length_m"),
        "tail_end_distance_scheme_m": _optional_float(scheme_t["end_distance_m"]),
        "tail_end_distance_preserve_m": _optional_float(preserve_t["end_distance_m"]),
        "tail_projection_scheme": int(scheme_t["projection_count"]),
        "tail_projection_preserve": int(preserve_t["projection_count"]),
        "tail_mean_sinr_scheme": _float(scheme_t, "mean_sinr"),
        "tail_mean_sinr_preserve": _float(preserve_t, "mean_sinr"),
        "delta_tail_mean_sinr": _float(preserve_t, "mean_sinr") - _float(scheme_t, "mean_sinr"),
        "tail_mean_interference_scheme": _float(scheme_t, "mean_interference"),
        "tail_mean_interference_preserve": _float(preserve_t, "mean_interference"),
        "delta_tail_mean_interference": _float(preserve_t, "mean_interference") - _float(scheme_t, "mean_interference"),
    }


def _group_descriptor(rows: list[dict]) -> dict:
    served = [float(row["delta_window_served_bits"]) for row in rows]
    sinr = [float(row["delta_window_mean_sinr"]) for row in rows]
    interference = [float(row["delta_window_mean_interference"]) for row in rows]
    paths = [float(row["delta_window_path_m"]) for row in rows]
    steps = [int(row["delta_window_service_steps"]) for row in rows]
    distances = [float(row["delta_t30_distance_m"]) for row in rows if row["delta_t30_distance_m"] is not None]
    step_buckets = {"decreased": 0.0, "unchanged": 0.0, "increased": 0.0}
    step_counts = {"decreased": 0, "unchanged": 0, "increased": 0}
    for step, bits in zip(steps, served):
        name = "unchanged" if step == 0 else "decreased" if step < 0 else "increased"
        step_buckets[name] += bits
        step_counts[name] += 1
    same_sign = 0
    for bits, sinr_delta in zip(served, sinr):
        if abs(bits) <= BIT_TIE or abs(sinr_delta) <= 1.0e-12:
            continue
        same_sign += int((bits > 0) == (sinr_delta > 0))
    comparable = sum(abs(bits) > BIT_TIE and abs(sinr_delta) > 1.0e-12 for bits, sinr_delta in zip(served, sinr))
    return {
        "n": len(rows),
        "window_served_sum_bits": float(np.sum(served)) if served else 0.0,
        "tail_served_sum_bits": float(np.sum([row["delta_tail_served_bits"] for row in rows])) if rows else 0.0,
        "window_negative": sum(bits < -BIT_TIE for bits in served),
        "window_positive": sum(bits > BIT_TIE for bits in served),
        "window_tie": sum(abs(bits) <= BIT_TIE for bits in served),
        "mean_sinr_delta": _mean(sinr),
        "mean_interference_delta": _mean(interference),
        "mean_path_delta_m": _mean(paths),
        "mean_service_step_delta": _mean([float(step) for step in steps]),
        "mean_t30_distance_delta_m": _mean(distances),
        "closer_targets": sum(delta < -DISTANCE_EPS_M for delta in distances),
        "farther_targets": sum(delta > DISTANCE_EPS_M for delta in distances),
        "unchanged_targets": sum(abs(delta) <= DISTANCE_EPS_M for delta in distances),
        "served_sinr_same_sign": same_sign,
        "served_sinr_comparable": comparable,
        "pearson_served_vs_sinr": _pearson(served, sinr),
        "pearson_served_vs_interference": _pearson(served, interference),
        "service_step_bucket_served_sum_bits": step_buckets,
        "service_step_bucket_counts": step_counts,
        "t45_target_changed": sum(int(row["t45_assignment_changed"]) for row in rows),
    }


def analyze_rows(scenarios: list[dict], uav_rows: list[dict]) -> dict:
    """Decompose saved P4 rows. Deltas are preserve minus scheme."""
    problems: list[str] = []
    ids = [int(row["scenario_id"]) for row in scenarios]
    if ids != list(VALIDATION_IDS):
        problems.append("split")
    if any(scenario_id >= 322064 for scenario_id in ids):
        problems.append("reserved_test")
    index = _index_uavs(uav_rows)
    scenario_out = []
    uav_out = []
    identity_errors = []
    uav_window_errors = []
    uav_tail_errors = []
    arm_errors = []
    recorded_errors = []
    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        for name, expected in EXPECTED_HASHES.items():
            if scenario[name] != expected:
                problems.append(f"model_config:{scenario_id}:{name}")
        if int(scenario["branch_hash_match"]) != 1:
            problems.append(f"state_hash:{scenario_id}")
        window = _float(scenario, "delta_window_bits")
        episode = _float(scenario, "delta_episode_bits")
        branch = _float(scenario, "delta_branch_bits")
        tail_scheme = _float(scenario, "branch_payload_scheme_bits") - _float(scenario, "window_payload_scheme_bits")
        tail_preserve = _float(scenario, "branch_payload_preserve_bits") - _float(scenario, "window_payload_preserve_bits")
        tail = tail_preserve - tail_scheme
        identity = max(abs(episode - (window + tail)), abs(episode - branch))
        for arm, tail_payload, window_key, branch_key, episode_key in (
            ("scheme", tail_scheme, "window_payload_scheme_bits", "branch_payload_scheme_bits", "episode_payload_scheme_bits"),
            ("preserve", tail_preserve, "window_payload_preserve_bits", "branch_payload_preserve_bits", "episode_payload_preserve_bits"),
        ):
            arm_error = abs(
                _float(scenario, episode_key)
                - _float(scenario, "payload_at_branch_bits")
                - _float(scenario, window_key)
                - tail_payload
            )
            arm_errors.append(arm_error)
            if arm_error > BIT_TIE:
                problems.append(f"arm_accounting:{scenario_id}:{arm}")
        changed = {item for item in scenario["changed_uavs"].split(",") if item != ""}
        paired = []
        try:
            paired = [_pair_uav(scenario, uav_id, changed, index) for uav_id in range(6)]
        except KeyError as exc:
            problems.append(f"missing_uav_row:{scenario_id}:{exc}")
        reassigned_window = sum(row["delta_window_served_bits"] for row in paired if row["group"] == "reassigned")
        fixed_window = sum(row["delta_window_served_bits"] for row in paired if row["group"] == "fixed")
        reassigned_tail = sum(row["delta_tail_served_bits"] for row in paired if row["group"] == "reassigned")
        fixed_tail = sum(row["delta_tail_served_bits"] for row in paired if row["group"] == "fixed")
        uav_window_error = abs((reassigned_window + fixed_window) - window) if paired else BIT_TIE + 1.0
        uav_tail_error = abs((reassigned_tail + fixed_tail) - tail) if paired else BIT_TIE + 1.0
        if scenario["comparison_status"] == "selected" and not changed:
            problems.append(f"selected_without_changed_uav:{scenario_id}")
        for row in paired:
            if row["group"] == "fixed" and row["t30_target_scheme"] != row["t30_target_preserve"]:
                problems.append(f"fixed_target_changed:{scenario_id}:{row['uav_id']}")
            distance_delta = row["delta_t30_distance_m"]
            if row["group"] == "fixed" and distance_delta is not None and abs(distance_delta) > DISTANCE_EPS_M:
                problems.append(f"fixed_distance_changed:{scenario_id}:{row['uav_id']}")
        score_delta = _float(scenario, "preserve_score") - _float(scenario, "scheme_score")
        compared = int(abs(score_delta) > SCORE_EPS and abs(window) > BIT_TIE)
        identity_errors.append(identity)
        uav_window_errors.append(uav_window_error)
        uav_tail_errors.append(uav_tail_error)
        recorded_errors.append(max(_float(scenario, "accounting_error_scheme_bits"), _float(scenario, "accounting_error_preserve_bits")))
        if identity > BIT_TIE or uav_window_error > BIT_TIE or uav_tail_error > BIT_TIE:
            problems.append(f"accounting:{scenario_id}")
        scenario_out.append(
            {
                "scenario_id": int(scenario_id),
                "comparison_status": scenario["comparison_status"],
                "state_hash": scenario["state_hash"],
                "branch_hash_match": int(scenario["branch_hash_match"]),
                "pool_hash": scenario["pool_hash"],
                "changed_uavs": scenario["changed_uavs"],
                "fixed_uavs": scenario["fixed_uavs"],
                "reassignable_uavs": scenario["reassignable_uavs"],
                "scheme_assignment": scenario["scheme_assignment"],
                "preserve_assignment": scenario["preserve_assignment"],
                "scheme_reassignable_reachable": int(scenario["scheme_reassignable_reachable"]),
                "preserve_reassignable_reachable": int(scenario["preserve_reassignable_reachable"]),
                "scheme_reassignable_distance_m": _float(scenario, "scheme_reassignable_distance_m"),
                "preserve_reassignable_distance_m": _float(scenario, "preserve_reassignable_distance_m"),
                "delta_reassignable_distance_m": _float(scenario, "preserve_reassignable_distance_m") - _float(scenario, "scheme_reassignable_distance_m"),
                "scheme_score": _float(scenario, "scheme_score"),
                "preserve_score": _float(scenario, "preserve_score"),
                "score_delta": score_delta,
                "score_compared": compared,
                "score_sign_matches_window": _sign_agree(score_delta, window) if compared else "",
                "score_sign_matches_episode": _sign_agree(score_delta, episode) if compared else "",
                "delta_window_bits": window,
                "tail_payload_scheme_bits": tail_scheme,
                "tail_payload_preserve_bits": tail_preserve,
                "delta_tail_bits": tail,
                "delta_episode_bits": episode,
                "delta_branch_bits": branch,
                "identity_error_bits": identity,
                "uav_window_sum_error_bits": uav_window_error,
                "uav_tail_sum_error_bits": uav_tail_error,
                "arm_level_error_bits": max(arm_errors[-2:]),
                "accounting_error_scheme_bits": _float(scenario, "accounting_error_scheme_bits"),
                "accounting_error_preserve_bits": _float(scenario, "accounting_error_preserve_bits"),
                "tail_effect": classify_tail_effect(window, tail),
                "dominant_interval": dominant_interval(window, tail),
                "reassigned_window_bits": reassigned_window,
                "fixed_window_bits": fixed_window,
                "reassigned_tail_bits": reassigned_tail,
                "fixed_tail_bits": fixed_tail,
                "t45_assignment_changed": int(scenario["t45_scheme_assignment"] != scenario["t45_preserve_assignment"]),
                "t45_scheme_assignment": scenario["t45_scheme_assignment"],
                "t45_preserve_assignment": scenario["t45_preserve_assignment"],
                "safety_violation_rate_scheme": _float(scenario, "safety_violation_rate_scheme"),
                "safety_violation_rate_preserve": _float(scenario, "safety_violation_rate_preserve"),
                "queue_before_bits": _float(scenario, "queue_before_bits"),
                "queue_after_window_scheme_bits": _float(scenario, "queue_after_window_scheme_bits"),
                "queue_after_window_preserve_bits": _float(scenario, "queue_after_window_preserve_bits"),
                "config_hash": scenario["config_hash"],
                "scheme_c_sha256": scenario["scheme_c_sha256"],
                "mappo_sha256": scenario["mappo_sha256"],
                "decoder_sha256": scenario["decoder_sha256"],
            }
        )
        uav_out.extend(paired)
    changed_rows = [row for row in scenario_out if row["comparison_status"] == "selected"]
    changed_ids = {row["scenario_id"] for row in changed_rows}
    changed_uavs = [row for row in uav_out if row["scenario_id"] in changed_ids]
    reassigned = [row for row in changed_uavs if row["group"] == "reassigned"]
    fixed = [row for row in changed_uavs if row["group"] == "fixed"]
    reassigned_stats = _group_descriptor(reassigned)
    fixed_stats = _group_descriptor(fixed)
    windows = [row["delta_window_bits"] for row in scenario_out]
    tails = [row["delta_tail_bits"] for row in scenario_out]
    episodes = [row["delta_episode_bits"] for row in scenario_out]
    episode_mean = float(np.mean(episodes))
    window_mean = float(np.mean(windows))
    tail_mean = float(np.mean(tails))
    compared = [row for row in changed_rows if row["score_compared"] == 1]
    window_sum = float(np.sum([row["delta_window_bits"] for row in changed_rows]))
    windows_separable = bool(identity_errors) and max(identity_errors) <= BIT_TIE and len(scenario_out) == 16
    uav_complete = len(changed_rows) == 10 and len(reassigned) + len(fixed) == 60
    if problems or not windows_separable:
        gate = "T30_LOSS_ATTRIBUTION_BLOCKED"
    elif not uav_complete:
        gate = "T30_LOSS_ATTRIBUTION_INCONCLUSIVE"
    else:
        gate = "T30_LOSS_ATTRIBUTION_COMPLETE"
    def _ids(predicate) -> list[int]:
        return [int(row["scenario_id"]) for row in changed_rows if predicate(row)]
    return {
        "gate": gate,
        "split": "validation_322048_322063",
        "scenario_ids": ids,
        "reserved_test_used": False,
        "replay_performed": False,
        "training_performed": False,
        "n_scenarios": len(scenario_out),
        "n_changed_scenarios": len(changed_rows),
        "n_unchanged_scenarios": len(scenario_out) - len(changed_rows),
        "changed_scenario_ids": [row["scenario_id"] for row in changed_rows],
        "p4_results_commit": P4_RESULTS_COMMIT,
        "plan_commit": PLAN_COMMIT,
        "analysis_module": "src/wt/benchmark_v2/phase6i_t30_loss_attribution/analyze.py",
        "model_config_sha256": dict(EXPECTED_HASHES),
        "model_config_consistent": not any(item.startswith("model_config") for item in problems),
        "state_hash_check": {
            "branch_hash_match_all": not any(item.startswith("state_hash") for item in problems),
            "n_state_hashes": len({row["state_hash"] for row in scenario_out}),
        },
        "accounting": {
            "bit_tie": BIT_TIE,
            "max_identity_error_bits": float(max(identity_errors) if identity_errors else 0.0),
            "max_uav_window_sum_error_bits": float(max(uav_window_errors) if uav_window_errors else 0.0),
            "max_uav_tail_sum_error_bits": float(max(uav_tail_errors) if uav_tail_errors else 0.0),
            "max_arm_level_error_bits": float(max(arm_errors) if arm_errors else 0.0),
            "max_recorded_p4_accounting_error_bits": float(max(recorded_errors) if recorded_errors else 0.0),
        },
        "time_decomposition_all_scenarios": {
            "window_mean_bits": window_mean,
            "tail_mean_bits": tail_mean,
            "episode_mean_bits": episode_mean,
            "window_sum_bits": float(np.sum(windows)),
            "tail_sum_bits": float(np.sum(tails)),
            "episode_sum_bits": float(np.sum(episodes)),
            "window_share_of_signed_episode_sum": float(np.sum(windows) / np.sum(episodes)),
            "tail_share_of_signed_episode_sum": float(np.sum(tails) / np.sum(episodes)),
            "window_positive_negative_tie": [
                sum(value > BIT_TIE for value in windows),
                sum(value < -BIT_TIE for value in windows),
                sum(abs(value) <= BIT_TIE for value in windows),
            ],
            "tail_positive_negative_tie": [
                sum(value > BIT_TIE for value in tails),
                sum(value < -BIT_TIE for value in tails),
                sum(abs(value) <= BIT_TIE for value in tails),
            ],
            "episode_positive_negative_tie": [
                sum(value > BIT_TIE for value in episodes),
                sum(value < -BIT_TIE for value in episodes),
                sum(abs(value) <= BIT_TIE for value in episodes),
            ],
        },
        "tail_effect_changed_scenarios": {
            "enlarges": _ids(lambda row: row["tail_effect"] == "enlarges"),
            "offsets": _ids(lambda row: row["tail_effect"] == "offsets"),
            "tie": _ids(lambda row: row["tail_effect"] == "tie"),
            "window_dominates_absolute": _ids(lambda row: row["dominant_interval"] == "window"),
            "tail_dominates_absolute": _ids(lambda row: row["dominant_interval"] == "tail"),
            "t45_assignment_changed": _ids(lambda row: row["t45_assignment_changed"] == 1),
            "window_tail_sign_flip": _ids(lambda row: row["tail_effect"] == "offsets"),
            "window_episode_sign_flip": _ids(
                lambda row: abs(row["delta_window_bits"]) > BIT_TIE
                and abs(row["delta_episode_bits"]) > BIT_TIE
                and (row["delta_window_bits"] > 0) != (row["delta_episode_bits"] > 0)
            ),
        },
        "uav_groups_changed_scenarios": {
            "definition": "reassigned means the UAV id is in changed_uavs at t=30; fixed means the t=30 target stayed the same",
            "reassigned": reassigned_stats,
            "fixed": fixed_stats,
            "reassigned_window_share_of_changed_window_sum": float(reassigned_stats["window_served_sum_bits"] / window_sum) if window_sum else None,
            "fixed_window_share_of_changed_window_sum": float(fixed_stats["window_served_sum_bits"] / window_sum) if window_sum else None,
        },
        "scheme_c_score": {
            "compared_scenarios": [row["scenario_id"] for row in compared],
            "preserve_score_lower": [row["scenario_id"] for row in compared if row["score_delta"] < -SCORE_EPS],
            "sign_agrees_with_window": [row["scenario_id"] for row in compared if row["score_sign_matches_window"] == 1],
            "sign_agrees_with_episode": [row["scenario_id"] for row in compared if row["score_sign_matches_episode"] == 1],
            "window_disagreement_ids": [row["scenario_id"] for row in compared if row["score_sign_matches_window"] == 0],
            "episode_disagreement_ids": [row["scenario_id"] for row in compared if row["score_sign_matches_episode"] == 0],
            "future_payload_used_to_select": False,
        },
        "safety_violation_rate_max": float(
            max(max(row["safety_violation_rate_scheme"], row["safety_violation_rate_preserve"]) for row in scenario_out)
        ) if scenario_out else None,
        "missing_fields": list(MISSING_FIELDS),
        "available_aggregates": [
            "window_and_tail_served_bits",
            "effective_service_steps",
            "path_length",
            "distance_to_active_target",
            "window_mean_sinr",
            "window_mean_interference",
            "projection_count",
            "scenario_mean_queue",
            "scenario_safety_violation_rate",
        ],
        "problems": problems,
        "scenarios": scenario_out,
        "uavs": uav_out,
        "next_step": "design_preregistered_cross_task_low_level_evaluation",
        "next_step_started": False,
    }


def analyze_directory(directory: Path = P4_RESULTS) -> dict:
    result = analyze_rows(_read_csv(directory / "scenario_pairs.csv"), _read_csv(directory / "uav_window_rows.csv"))
    result["p4_input_sha256"] = {
        "scenario_pairs.csv": _sha256(directory / "scenario_pairs.csv"),
        "uav_window_rows.csv": _sha256(directory / "uav_window_rows.csv"),
        "summary.json": _sha256(directory / "summary.json"),
    }
    return result


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in fields})


def _figures(scenarios: list[dict], directory: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    saved = []
    labels = [str(row["scenario_id"]) for row in scenarios]
    positions = np.arange(len(scenarios))
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.38
    ax.bar(positions - width / 2, [row["delta_window_bits"] / 1.0e6 for row in scenarios], width, label="t=30-45")
    ax.bar(positions + width / 2, [row["delta_tail_bits"] / 1.0e6 for row in scenarios], width, label="t=45-48")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("preserve minus scheme (Mbit)")
    ax.set_title(f"window and tail payload gap, n_scenarios={len(scenarios)}, unit=scenario")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "window_tail_delta.png", dpi=120)
    plt.close(fig)
    saved.append("window_tail_delta.png")
    changed = [row for row in scenarios if row["comparison_status"] == "selected"]
    fig, ax = plt.subplots(figsize=(8, 4))
    positions = np.arange(len(changed))
    ax.bar(positions - width / 2, [row["reassigned_window_bits"] / 1.0e6 for row in changed], width, label="reassigned UAV")
    ax.bar(positions + width / 2, [row["fixed_window_bits"] / 1.0e6 for row in changed], width, label="fixed association")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(row["scenario_id"]) for row in changed], rotation=45, ha="right")
    ax.set_ylabel("t=30-45 served-bit gap (Mbit)")
    ax.set_title(f"reassigned vs fixed delivery, n_scenarios={len(changed)}, unit=scenario")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "reassigned_vs_fixed_window.png", dpi=120)
    plt.close(fig)
    saved.append("reassigned_vs_fixed_window.png")
    return saved


def write_outputs(result: dict, output: Path = OUTPUT) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "scenario_decomposition.csv", SCENARIO_FIELDS, result["scenarios"])
    _write_csv(output / "uav_decomposition.csv", UAV_FIELDS, result["uavs"])
    figures = _figures(result["scenarios"], output / "figures")
    public = {key: value for key, value in result.items() if key not in {"scenarios", "uavs"}}
    public["figures"] = figures
    public["output_dir"] = "results/phase6i_t30_loss_attribution"
    with (output / "summary.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(public, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    return public


def main() -> None:
    public = write_outputs(analyze_directory())
    print(public["gate"])
    print("max_identity", public["accounting"]["max_identity_error_bits"])
    print("window_share", public["time_decomposition_all_scenarios"]["window_share_of_signed_episode_sum"])
    print("reassigned_share", public["uav_groups_changed_scenarios"]["reassigned_window_share_of_changed_window_sum"])


if __name__ == "__main__":
    main()
