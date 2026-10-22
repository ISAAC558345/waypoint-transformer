"""Offline check: can the t=15 state predict executed SINR? No rollout."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import PRIMARY_SEED
from wt.benchmark_v2.phase6ib_highlevel_dataset.hgnn_revision import REVISION_DATASET
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST, VALIDATION_POOL
from wt.benchmark_v2.phase6ii_candidate_features.features import kinematic_delivered_bits
from wt.benchmark_v2.phase6im_local_swap.audit import FLOORS, HIGHER_IS_BETTER
from wt.l3b.config import L3BConfig


ROOT = Path(__file__).resolve().parents[4]
AUDIT_DIR = ROOT / "results/phase6ip_dynamic_sinr"
SOURCE = ROOT / "results/phase6im_local_swap" / "states.json"
CLOSURE = ROOT / "results/phase6ik_execution_closure" / "states.json"
CASES = ROOT / "results/phase6io_per_uav_response" / "cases.json"
AGREEMENT_BAR = 0.75
DRAWS = 10000
SEED = 0
BIT_TOLERANCE = 1.0
MATERIAL_BITS = 1.0e6
SINR_FLOOR = 0.05
INTERFERENCE_FLOOR = 0.01
MIN_STATES = 12
MAGNITUDE_RATIO = 0.1
HOLD = 15
# Lower distance and lower interference are the directions that would
# predict a higher executed SINR. Rates point the same way as SINR.
SINR_HIGHER_IS_BETTER = HIGHER_IS_BETTER | {"straight_sinr", "executed_sinr"}
DECISION_FEATURES = (
    "distance_m",
    "steps_to_radius",
    "hypothetical_rate_bps",
    "geometric_rate_bps",
    "geometric_interference_w",
    "service_overlap_s",
    "instant_rate_bps",
    "kinematic_bits",
    "hover_interference_w",
)


def assign_gate(coverage_blocked: bool, static_explains: bool, physics_explains: bool, execution_tracks: bool) -> str:
    """Choose one route. A blocked inventory cannot support either positive route."""
    if coverage_blocked:
        return "OBSERVABILITY_INCONCLUSIVE"
    if static_explains or physics_explains:
        return "CURRENT_STATE_FEATURES_SUFFICIENT"
    if execution_tracks:
        return "DYNAMIC_RESPONSE_REQUIRED"
    return "OBSERVABILITY_INCONCLUSIVE"


def straight_line_channel(arrays: dict[str, np.ndarray], assignment: np.ndarray, config) -> dict[str, np.ndarray]:
    """Max-speed straight flight. Same inputs as ``kinematic_delivered_bits``.

    Active-step SINR is returned per UAV. An empty active set stays NaN,
    which callers must not replace with a measured zero.
    """
    uav_position = np.asarray(arrays["uav_position_m"], dtype=np.float64)
    buoy_position = np.asarray(arrays["buoy_position_m"], dtype=np.float64)
    queue_bits = np.asarray(arrays["q_bits"], dtype=np.float64)
    rows = np.asarray(assignment, dtype=np.int64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.shape[1] != uav_position.shape[0]:
        raise ValueError("assignment width does not match the UAV count")
    valid = rows >= 0
    safe = np.where(valid, rows, 0)
    counts = np.zeros((rows.shape[0], int(queue_bits.shape[0])), dtype=np.int16)
    index = np.arange(rows.shape[0], dtype=np.int64)[:, None]
    index = np.broadcast_to(index, rows.shape)
    np.add.at(counts, (index, safe), valid.astype(np.int16))
    unique = counts[index, safe] == 1
    target = buoy_position[safe]
    current = np.broadcast_to(uav_position[None, :, :], target.shape).copy()
    remaining = np.where(valid, queue_bits[safe], 0.0)
    eligible = valid & unique & (remaining > 1.0e-9)
    n_rows, n_uavs = rows.shape
    delivered = np.zeros((n_rows, n_uavs), dtype=np.float64)
    sinr_sum = np.zeros((n_rows, n_uavs), dtype=np.float64)
    interference_sum = np.zeros((n_rows, n_uavs), dtype=np.float64)
    active_steps = np.zeros((n_rows, n_uavs), dtype=np.int64)
    entry = np.full((n_rows, n_uavs), -1, dtype=np.int64)
    cluster = (safe >= 6).astype(np.int64)
    same = cluster[:, :, None] == cluster[:, None, :]
    kappa = np.where(same, float(config.kappa_same_cluster), float(config.kappa_cross_cluster))
    eye = np.eye(n_uavs, dtype=bool)
    step = float(config.step_length_m)
    minimum = float(config.min_link_m)
    exponent = float(config.path_loss_exp)
    radius = float(config.service_radius_m)
    for step_index in range(int(config.option_hold)):
        delta = target - current
        distance = np.linalg.norm(delta, axis=-1)
        scale = np.minimum(step, distance) / np.maximum(distance, 1.0e-12)
        current = current + delta * scale[:, :, None]
        pair_delta = current[:, None, :, :] - target[:, :, None, :]
        pair_distance = np.linalg.norm(pair_delta, axis=-1)
        gain = np.maximum(pair_distance, minimum) ** (-exponent)
        serving = pair_distance[:, np.arange(n_uavs), np.arange(n_uavs)]
        active = eligible & (serving <= radius) & (remaining > 1.0e-9)
        signal = gain[:, np.arange(n_uavs), np.arange(n_uavs)] * float(config.tx_power)
        pair = active[:, :, None] & active[:, None, :] & ~eye
        interference = np.sum(gain * float(config.tx_power) * kappa * pair, axis=-1)
        sinr = np.where(active, signal / np.maximum(float(config.noise_power) + interference, 1.0e-18), 0.0)
        rate = np.where(active, float(config.bandwidth_hz) * np.log2(1.0 + sinr), 0.0)
        taken = np.minimum(remaining, rate * float(config.delta_t))
        remaining = remaining - taken
        delivered += taken
        sinr_sum += np.where(active, sinr, 0.0)
        interference_sum += np.where(active, interference, 0.0)
        active_steps += active.astype(np.int64)
        entry = np.where((entry < 0) & active, step_index, entry)
        eligible = eligible & (remaining > 1.0e-9)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_sinr = np.where(active_steps > 0, sinr_sum / np.maximum(active_steps, 1), np.nan)
        mean_interference = np.where(active_steps > 0, interference_sum / np.maximum(active_steps, 1), np.nan)
    return {
        "delivered_bits": delivered,
        "mean_sinr": mean_sinr,
        "mean_interference_w": mean_interference,
        "service_steps": active_steps,
        "entry": entry,
    }


def _bootstrap_mean(values: list[float]) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(SEED)
    draws = np.empty(DRAWS, dtype=np.float64)
    for index in range(DRAWS):
        draws[index] = float(np.mean(data[rng.integers(0, data.size, data.size)]))
    return float(np.mean(data)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _agreement(per_state: list[float], magnitude_ratio: float | None = None) -> dict:
    mean, low, high = _bootstrap_mean(per_state)
    return {
        "states_with_movement": int(len(per_state)),
        "mean_state_agreement": mean,
        "agreement_ci95": [low, high],
        "ci_above_bar": int(np.isfinite(low) and low > AGREEMENT_BAR),
        "magnitude_ratio": None if magnitude_ratio is None else float(magnitude_ratio),
    }


def _directional_match(delta: float, higher: bool, outcome: float) -> bool:
    predicted = delta > 0.0 if higher else delta < 0.0
    return predicted == (outcome > 0.0)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _fleet_mean(per_uav: list[dict], name: str) -> float | None:
    chosen = [float(item[name]) for item in per_uav if int(item["service_steps"]) > 0]
    if not chosen:
        return None
    return float(np.mean(chosen))


def _load_state(scenario: int) -> dict[str, np.ndarray]:
    if int(scenario) in RESERVED_TEST or int(scenario) not in VALIDATION_POOL:
        raise RuntimeError("state load is limited to the validation pool")
    path = REVISION_DATASET / "validation" / f"scenario_{int(scenario)}" / "t015_state.npz"
    with np.load(path) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _predict_rows(arrays: dict[str, np.ndarray], assignments: np.ndarray, config) -> dict[str, np.ndarray]:
    channel = straight_line_channel(arrays, assignments, config)
    reference = kinematic_delivered_bits(arrays, assignments, config)
    total = np.sum(channel["delivered_bits"], axis=1)
    if float(np.max(np.abs(total - reference))) > 1.0e-6:
        raise RuntimeError("straight-line bits diverged from kinematic_delivered_bits")
    sinr = []
    interference = []
    for row in range(assignments.shape[0]):
        active = channel["service_steps"][row] > 0
        sinr.append(float(np.mean(channel["mean_sinr"][row, active])) if np.any(active) else None)
        interference.append(float(np.mean(channel["mean_interference_w"][row, active])) if np.any(active) else None)
    return {
        "delivered_bits": total,
        "fleet_sinr": sinr,
        "fleet_interference_w": interference,
        "per_uav_sinr": channel["mean_sinr"],
        "per_uav_interference_w": channel["mean_interference_w"],
        "per_uav_entry": channel["entry"],
        "per_uav_steps": channel["service_steps"],
    }


def _pair_agreement(rows: list[dict], feature: str, higher: bool, floor: float, outcome: str, outcome_floor: float) -> dict:
    pooled = {"agree": 0, "disagree": 0, "unresolved": 0, "outcome_missing": 0}
    per_state = []
    ratios = []
    for row in rows:
        agree = disagree = 0
        for swap in row["swaps"]:
            observed = swap.get(outcome)
            if observed is None:
                pooled["outcome_missing"] += 1
                continue
            if abs(float(observed)) <= outcome_floor:
                continue
            delta = swap["decision"].get(feature)
            if delta is None or abs(float(delta)) <= floor:
                pooled["unresolved"] += 1
                continue
            if _directional_match(float(delta), higher, float(observed)):
                agree += 1
                pooled["agree"] += 1
            else:
                disagree += 1
                pooled["disagree"] += 1
            if abs(float(observed)) > 0.0:
                ratios.append(abs(float(delta)) / abs(float(observed)))
        if agree + disagree:
            per_state.append(agree / (agree + disagree))
    report = _agreement(per_state, _median(ratios))
    report["pooled"] = pooled
    return report


def _execution_agreement(rows: list[dict], outcome: str, higher: bool, floor: float) -> dict:
    """Sign match between an executed channel change and the bit change."""
    pooled = {"agree": 0, "disagree": 0, "unresolved": 0, "outcome_missing": 0}
    per_state = []
    for row in rows:
        agree = disagree = 0
        for swap in row["swaps"]:
            if abs(float(swap["bit_delta"])) <= MATERIAL_BITS:
                continue
            observed = swap.get(outcome)
            if observed is None:
                pooled["outcome_missing"] += 1
                continue
            if abs(float(observed)) <= floor:
                pooled["unresolved"] += 1
                continue
            if _directional_match(float(observed), higher, float(swap["bit_delta"])):
                agree += 1
                pooled["agree"] += 1
            else:
                disagree += 1
                pooled["disagree"] += 1
        if agree + disagree:
            per_state.append(agree / (agree + disagree))
    report = _agreement(per_state)
    report["pooled"] = pooled
    return report


def _feature_floor(name: str) -> float:
    if name == "straight_sinr":
        return SINR_FLOOR
    if name == "straight_interference_w":
        return INTERFERENCE_FLOOR
    return float(FLOORS.get(name, 0.0))


def _coverage(local_rows: list[dict], cases: dict) -> dict:
    arms = [arm for row in local_rows for arm in row["arms"]]
    keys = [(int(row["scenario_id"]), tuple(arm["assignment"])) for row in local_rows for arm in row["arms"]]
    per_uav_fields = ("target", "entry", "entry_distance_m", "service_steps", "delivered_bits", "mean_sinr", "mean_interference_w", "end_progress")
    missing_compact = {name: 0 for name in per_uav_fields}
    empty_sinr_placeholder = 0
    uav_rows = 0
    for arm in arms:
        for item in arm["rollout"]["per_uav"]:
            uav_rows += 1
            for name in per_uav_fields:
                if item.get(name) is None:
                    missing_compact[name] += 1
            if int(item["service_steps"]) == 0:
                empty_sinr_placeholder += 1
    step_arms = 0
    position_missing_arms = 0
    step_fields_missing = 0
    required_step = ("uav", "target", "distance_m", "in_range", "sinr", "interference_w", "rate_bps", "progress", "served_bits")
    for case in cases.values():
        for arm in case["arms"].values():
            step_arms += 1
            steps = arm.get("steps") or []
            if not steps or any(item.get("position_m") is None for item in steps[0]["uavs"]):
                position_missing_arms += 1
            for step in steps:
                for item in step["uavs"]:
                    if any(name not in item for name in required_step):
                        step_fields_missing += 1
    closure_steps = 0
    if CLOSURE.exists():
        closure = json.loads(CLOSURE.read_text(encoding="utf-8"))
        closure_steps = sum(1 for row in closure if "steps" in row)
    return {
        "validation_states": int(len(local_rows)),
        "compact_arms": int(len(arms)),
        "unique_compact_keys": int(len(set(keys))),
        "duplicate_compact_keys": int(len(keys) - len(set(keys))),
        "compact_uav_rows": int(uav_rows),
        "compact_missing_fields": missing_compact,
        "compact_uavs_with_no_service_step": int(empty_sinr_placeholder),
        "compact_has_step_position": 0,
        "compact_has_per_uav_projector": 0,
        "full_step_arms_in_phase6io": int(step_arms),
        "full_step_arms_missing_position": int(position_missing_arms),
        "full_step_records_missing_required_field": int(step_fields_missing),
        "phase6ik_states_with_full_steps": int(closure_steps),
        "double_swap_executions_outside_named_case": 0,
        "reserved_states_read": 0,
    }


def _swaps(local_rows: list[dict], predicted: dict[int, dict]) -> tuple[list[dict], list[dict]]:
    rows = []
    examples = []
    for state in local_rows:
        scenario = int(state["scenario_id"])
        arms = state["arms"]
        base = arms[0]
        base_key = tuple(int(item) for item in base["assignment"])
        base_pred = predicted[scenario][base_key]
        base_sinr = _fleet_mean(base["rollout"]["per_uav"], "mean_sinr")
        base_interference = _fleet_mean(base["rollout"]["per_uav"], "mean_interference_w")
        swaps = []
        for arm in arms[1:]:
            key = tuple(int(item) for item in arm["assignment"])
            pred = predicted[scenario][key]
            executed_sinr = _fleet_mean(arm["rollout"]["per_uav"], "mean_sinr")
            executed_interference = _fleet_mean(arm["rollout"]["per_uav"], "mean_interference_w")
            decision = {name: float(arm["features"][name]) - float(base["features"][name]) for name in DECISION_FEATURES}
            decision["straight_sinr"] = None if pred["fleet_sinr"] is None or base_pred["fleet_sinr"] is None else float(pred["fleet_sinr"] - base_pred["fleet_sinr"])
            decision["straight_interference_w"] = (
                None
                if pred["fleet_interference_w"] is None or base_pred["fleet_interference_w"] is None
                else float(pred["fleet_interference_w"] - base_pred["fleet_interference_w"])
            )
            swap = {
                "scenario_id": scenario,
                "uav_i": arm["uav_i"],
                "uav_j": arm["uav_j"],
                "assignment": [int(item) for item in arm["assignment"]],
                "bit_delta": float(arm["delta_bits"]),
                "decision": decision,
                "executed_sinr_delta": None if executed_sinr is None or base_sinr is None else float(executed_sinr - base_sinr),
                "executed_interference_delta": None if executed_interference is None or base_interference is None else float(executed_interference - base_interference),
                "entry_delta_mean": float(np.mean([int(item["entry"]) - int(other["entry"]) for item, other in zip(arm["rollout"]["per_uav"], base["rollout"]["per_uav"])])),
                "service_step_delta": int(arm["rollout"]["service_steps"] - base["rollout"]["service_steps"]),
            }
            close = (
                abs(decision["distance_m"]) <= FLOORS["distance_m"]
                and abs(decision["hypothetical_rate_bps"]) <= FLOORS["hypothetical_rate_bps"]
                and abs(decision["geometric_interference_w"]) <= FLOORS["geometric_interference_w"]
            )
            opened = (
                swap["executed_interference_delta"] is not None
                and abs(swap["executed_interference_delta"]) > INTERFERENCE_FLOOR
                and abs(swap["bit_delta"]) > 1.0e5
            )
            swap["decision_close"] = int(close)
            swap["execution_interference_moved"] = int(opened)
            if close and opened and len(examples) < 6:
                examples.append({"scenario_id": scenario, "uav_i": arm["uav_i"], "uav_j": arm["uav_j"], "bit_delta": swap["bit_delta"], "distance_delta_m": decision["distance_m"], "geometric_interference_delta_w": decision["geometric_interference_w"], "executed_interference_delta_w": swap["executed_interference_delta"], "executed_sinr_delta": swap["executed_sinr_delta"]})
            swaps.append(swap)
        rows.append({"scenario_id": scenario, "baseline": [int(item) for item in base["assignment"]], "swaps": swaps})
    return rows, examples


def _positions_at_entry(arrays: dict[str, np.ndarray], assignment: np.ndarray, config) -> np.ndarray:
    uav_position = np.asarray(arrays["uav_position_m"], dtype=np.float64)
    buoy_position = np.asarray(arrays["buoy_position_m"], dtype=np.float64)
    row = np.asarray(assignment, dtype=np.int64).reshape(-1)
    target = buoy_position[np.where(row >= 0, row, 0)]
    current = uav_position.copy()
    found = np.full(row.shape[0], np.nan, dtype=np.float64)
    for _step in range(int(config.option_hold)):
        delta = target - current
        distance = np.linalg.norm(delta, axis=-1)
        scale = np.minimum(float(config.step_length_m), distance) / np.maximum(distance, 1.0e-12)
        current = current + delta * scale[:, None]
        serving = np.linalg.norm(current - target, axis=-1)
        arrived = (found != found) & (row >= 0) & (serving <= float(config.service_radius_m))
        found = np.where(arrived, serving, found)
    return found


def _entry_distance_gap(local_rows: list[dict], arrays_by_scenario: dict[int, dict], config) -> dict:
    gaps = []
    same = 0
    compared = 0
    for state in local_rows:
        scenario = int(state["scenario_id"])
        arrays = arrays_by_scenario[scenario]
        for arm in state["arms"]:
            assignment = np.asarray(arm["assignment"], dtype=np.int64)
            predicted = _positions_at_entry(arrays, assignment, config)
            channel = straight_line_channel(arrays, assignment, config)
            for uav, item in enumerate(arm["rollout"]["per_uav"]):
                if int(item["entry"]) >= HOLD or int(channel["entry"][0, uav]) < 0:
                    continue
                compared += 1
                same += int(int(item["entry"]) == int(channel["entry"][0, uav]))
                if item.get("entry_distance_m") is not None and predicted[uav] == predicted[uav]:
                    gaps.append(abs(float(item["entry_distance_m"]) - float(predicted[uav])))
    return {
        "uavs_compared": int(compared),
        "same_entry_step": int(same),
        "entry_distance_abs_gap_median_m": _median(gaps),
        "entry_distance_abs_gap_p90_m": None if not gaps else float(np.percentile(np.asarray(gaps), 90)),
    }


def _named_case(cases: dict, arrays_by_scenario: dict[int, dict], config) -> list[dict]:
    report = []
    for name, case in cases.items():
        scenario = int(case["scenario_id"])
        if scenario in RESERVED_TEST:
            raise RuntimeError("named case overlaps the reserved test")
        arrays = arrays_by_scenario[scenario]
        arms = {}
        for arm_name, arm in case["arms"].items():
            assignment = np.asarray(arm["assignment"], dtype=np.int64)
            channel = straight_line_channel(arrays, assignment, config)
            active = channel["service_steps"][0] > 0
            executed = []
            for item in arm["per_uav"]:
                measured_sinr = None if int(item["service_steps"]) == 0 else float(item["mean_sinr"])
                measured_interference = None if int(item["service_steps"]) == 0 else float(item["mean_interference_w"])
                executed.append(
                    {
                        "service_steps": int(item["service_steps"]),
                        "entry": None if int(item["entry"]) >= HOLD else int(item["entry"]),
                        "mean_sinr": measured_sinr,
                        "mean_interference_w": measured_interference,
                        "delivered_bits": float(item["delivered_bits"]),
                    }
                )
            position_missing = int(not arm["steps"] or any(item.get("position_m") is None for item in arm["steps"][0]["uavs"]))
            arms[arm_name] = {
                "assignment": [int(item) for item in assignment],
                "position_missing": position_missing,
                "straight_fleet_sinr": None if not np.any(active) else float(np.mean(channel["mean_sinr"][0, active])),
                "straight_fleet_interference_w": None if not np.any(active) else float(np.mean(channel["mean_interference_w"][0, active])),
                "executed_fleet_sinr": _fleet_mean(executed, "mean_sinr") if all(item["mean_sinr"] is not None or item["service_steps"] == 0 for item in executed) else None,
                "executed": executed,
                "straight_per_uav_sinr": [None if not np.isfinite(item) else float(item) for item in channel["mean_sinr"][0]],
            }
            # _fleet_mean skips missing service. Rebuild it from executed rows that measured SINR.
            measured = [float(item["mean_sinr"]) for item in executed if item["mean_sinr"] is not None]
            arms[arm_name]["executed_fleet_sinr"] = float(np.mean(measured)) if measured else None
            measured_i = [float(item["mean_interference_w"]) for item in executed if item["mean_interference_w"] is not None]
            arms[arm_name]["executed_fleet_interference_w"] = float(np.mean(measured_i)) if measured_i else None
        report.append({"name": name, "scenario_id": scenario, "arms": arms})
    return report


def _explains(report: dict, minimum_ratio: float | None) -> bool:
    if int(report["states_with_movement"]) < MIN_STATES or int(report["ci_above_bar"]) != 1:
        return False
    if minimum_ratio is None:
        return True
    return report["magnitude_ratio"] is not None and float(report["magnitude_ratio"]) > minimum_ratio


def run_audit() -> dict:
    if any(int(scenario) in RESERVED_TEST for scenario in VALIDATION_POOL):
        raise RuntimeError("validation pool overlaps the reserved test")
    local_rows = json.loads(SOURCE.read_text(encoding="utf-8"))
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    config = L3BConfig()
    coverage_blocked = False
    if len(local_rows) != 16 or any(int(row["scenario_id"]) in RESERVED_TEST for row in local_rows):
        coverage_blocked = True
    arrays_by_scenario = {int(row["scenario_id"]): _load_state(int(row["scenario_id"])) for row in local_rows}
    predicted: dict[int, dict] = {}
    current_sinr_max = 0.0
    any_pair_in_range = 0
    candidate_links_in_range = 0
    current_links_in_range = 0
    empty_current_links = 0
    instant_rate_max = 0.0
    switch_abs_max = 0.0
    formula_error = 0.0
    for row in local_rows:
        scenario = int(row["scenario_id"])
        arrays = arrays_by_scenario[scenario]
        current_sinr_max = max(current_sinr_max, float(np.max(np.abs(arrays["current_sinr"]))))
        inside = np.asarray(arrays["uav_buoy_in_range"])
        any_pair_in_range += int(np.sum(inside))
        association = np.asarray(arrays["association"], dtype=np.int64).reshape(-1)
        current_links_in_range += int(sum(int(inside[uav, int(buoy)]) for uav, buoy in enumerate(association) if int(buoy) >= 0))
        queues = np.asarray(arrays["q_bits"], dtype=np.float64)
        for uav, buoy in enumerate(association):
            if int(buoy) >= 0 and int(inside[uav, int(buoy)]) and float(queues[int(buoy)]) <= 1.0e-9:
                empty_current_links += 1
        for arm in row["arms"]:
            assignment = [int(item) for item in arm["assignment"]]
            candidate_links_in_range += int(sum(int(inside[uav, buoy]) for uav, buoy in enumerate(assignment) if buoy >= 0))
            instant_rate_max = max(instant_rate_max, float(arm["features"]["instant_rate_bps"]))
            switch_abs_max = max(switch_abs_max, abs(float(arm["features"]["switch"]) - float(row["arms"][0]["features"]["switch"])))
        assignments = np.asarray([arm["assignment"] for arm in row["arms"]], dtype=np.int64)
        channel = _predict_rows(arrays, assignments, config)
        stored = np.asarray([float(arm["features"]["kinematic_bits"]) for arm in row["arms"]], dtype=np.float64)
        formula_error = max(formula_error, float(np.max(np.abs(channel["delivered_bits"] - stored))))
        predicted[scenario] = {}
        for index, arm in enumerate(row["arms"]):
            predicted[scenario][tuple(int(item) for item in arm["assignment"])] = {
                "fleet_sinr": channel["fleet_sinr"][index],
                "fleet_interference_w": channel["fleet_interference_w"][index],
                "per_uav_entry": None,
                "per_uav_steps": None,
            }
    if formula_error > BIT_TOLERANCE:
        coverage_blocked = True
    rows, examples = _swaps(local_rows, predicted)
    entry = _entry_distance_gap(local_rows, arrays_by_scenario, config)
    population = [swap for row in rows for swap in row["swaps"]]
    material = [swap for swap in population if abs(float(swap["bit_delta"])) > MATERIAL_BITS]
    moved = [swap for swap in population if abs(float(swap["bit_delta"])) > BIT_TOLERANCE]
    channel_overlap = {"both": 0, "sinr_only": 0, "interference_only": 0, "neither": 0}
    sinr_sign_exceptions = []
    for swap in material:
        sinr_moved = swap["executed_sinr_delta"] is not None and abs(float(swap["executed_sinr_delta"])) > SINR_FLOOR
        interference_moved = swap["executed_interference_delta"] is not None and abs(float(swap["executed_interference_delta"])) > INTERFERENCE_FLOOR
        if sinr_moved and interference_moved:
            channel_overlap["both"] += 1
        elif sinr_moved:
            channel_overlap["sinr_only"] += 1
        elif interference_moved:
            channel_overlap["interference_only"] += 1
        else:
            channel_overlap["neither"] += 1
        if sinr_moved and (float(swap["executed_sinr_delta"]) > 0.0) != (float(swap["bit_delta"]) > 0.0) and len(sinr_sign_exceptions) < 4:
            sinr_sign_exceptions.append(
                {
                    "scenario_id": swap["scenario_id"],
                    "uav_i": swap["uav_i"],
                    "uav_j": swap["uav_j"],
                    "bit_delta": float(swap["bit_delta"]),
                    "executed_sinr_delta": float(swap["executed_sinr_delta"]),
                    "executed_interference_delta": None if swap["executed_interference_delta"] is None else float(swap["executed_interference_delta"]),
                }
            )
    sinr_reports = {}
    interference_reports = {}
    for name in DECISION_FEATURES:
        higher = name in SINR_HIGHER_IS_BETTER or name in HIGHER_IS_BETTER
        if name in {"distance_m", "steps_to_radius", "geometric_interference_w", "hover_interference_w", "service_overlap_s"}:
            higher = False
        sinr_reports[name] = _pair_agreement(rows, name, higher, _feature_floor(name), "executed_sinr_delta", SINR_FLOOR)
        interference_higher = name not in {"distance_m", "steps_to_radius", "hypothetical_rate_bps", "geometric_rate_bps", "instant_rate_bps", "kinematic_bits"}
        interference_reports[name] = _pair_agreement(rows, name, interference_higher, _feature_floor(name), "executed_interference_delta", INTERFERENCE_FLOOR)
    sinr_reports["straight_sinr"] = _pair_agreement(rows, "straight_sinr", True, SINR_FLOOR, "executed_sinr_delta", SINR_FLOOR)
    interference_reports["straight_interference_w"] = _pair_agreement(
        rows, "straight_interference_w", True, INTERFERENCE_FLOOR, "executed_interference_delta", INTERFERENCE_FLOOR
    )
    execution_sinr = _execution_agreement(rows, "executed_sinr_delta", True, SINR_FLOOR)
    execution_interference = _execution_agreement(rows, "executed_interference_delta", False, INTERFERENCE_FLOOR)
    static_explains = any(
        _explains(sinr_reports[name], None) or _explains(interference_reports[name], MAGNITUDE_RATIO if "interference" in name else None)
        for name in DECISION_FEATURES
    )
    physics_explains = _explains(sinr_reports["straight_sinr"], MAGNITUDE_RATIO) or _explains(
        interference_reports["straight_interference_w"], MAGNITUDE_RATIO
    )
    execution_tracks = _explains(execution_sinr, None) or _explains(execution_interference, None)
    early_zero = 0
    early_seen = 0
    for state in local_rows:
        for arm in state["arms"]:
            series = arm["rollout"].get("step_mean_sinr") or []
            if len(series) >= 2:
                early_seen += 1
                early_zero += int(abs(float(series[0])) == 0.0 and abs(float(series[1])) == 0.0)
    summary = {
        "gate": assign_gate(coverage_blocked, static_explains, physics_explains, execution_tracks),
        "coverage_blocked": int(coverage_blocked),
        "static_explains": int(static_explains),
        "physics_explains": int(physics_explains),
        "execution_tracks": int(execution_tracks),
        "kinematic_bit_abs_error_max": float(formula_error),
        "current_sinr_abs_max": float(current_sinr_max),
        "any_uav_buoy_in_range_count": int(any_pair_in_range),
        "candidate_links_in_range": int(candidate_links_in_range),
        "current_association_links_in_range": int(current_links_in_range),
        "current_association_in_range_empty_queue": int(empty_current_links),
        "instant_rate_max_bps": float(instant_rate_max),
        "switch_abs_delta_max": float(switch_abs_max),
        "early_step_sinr_zero_arms": int(early_zero),
        "early_step_sinr_arms": int(early_seen),
        "single_swaps": int(len(population)),
        "bit_moved_swaps": int(len(moved)),
        "material_swaps": int(len(material)),
        "material_channel_overlap": channel_overlap,
        "sinr_sign_exceptions": sinr_sign_exceptions,
        "decision_close_and_interference_moved": int(sum(int(swap["decision_close"] and swap["execution_interference_moved"]) for swap in population)),
        "material_and_decision_close": int(sum(int(swap["decision_close"]) for swap in material)),
        "bit_delta_median": _median([abs(float(swap["bit_delta"])) for swap in moved]),
        "executed_sinr_abs_median": _median([abs(float(swap["executed_sinr_delta"])) for swap in moved if swap["executed_sinr_delta"] is not None]),
        "executed_interference_abs_median_w": _median([abs(float(swap["executed_interference_delta"])) for swap in moved if swap["executed_interference_delta"] is not None]),
        "geometric_interference_abs_median_w": _median([abs(float(swap["decision"]["geometric_interference_w"])) for swap in moved]),
        "distance_abs_median_m": _median([abs(float(swap["decision"]["distance_m"])) for swap in moved]),
        "hypothetical_rate_abs_median_bps": _median([abs(float(swap["decision"]["hypothetical_rate_bps"])) for swap in moved]),
        "straight_sinr_abs_median": _median([abs(float(swap["decision"]["straight_sinr"])) for swap in moved if swap["decision"]["straight_sinr"] is not None]),
        "straight_interference_abs_median_w": _median(
            [abs(float(swap["decision"]["straight_interference_w"])) for swap in moved if swap["decision"]["straight_interference_w"] is not None]
        ),
        "coverage": _coverage(local_rows, cases),
        "entry": entry,
        "sinr_agreement": sinr_reports,
        "interference_agreement": interference_reports,
        "execution_sinr_tracks_bits": execution_sinr,
        "execution_interference_tracks_bits": execution_interference,
        "decision_close_examples": examples,
        "named_cases": _named_case(cases, arrays_by_scenario, config),
        "primary_seed": int(PRIMARY_SEED),
        "new_rollouts": 0,
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    path = AUDIT_DIR / "summary.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    print(
        json.dumps(
            {
                "gate": summary["gate"],
                "static": summary["static_explains"],
                "physics": summary["physics_explains"],
                "execution": summary["execution_tracks"],
                "close_open": summary["decision_close_and_interference_moved"],
                "formula_error": summary["kinematic_bit_abs_error_max"],
            }
        ),
        flush=True,
    )
    return summary


if __name__ == "__main__":
    run_audit()
