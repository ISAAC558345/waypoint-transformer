"""Short frozen-MAPPO prefixes on three saved states. No training."""
from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6d.shadow import capture_state, restore_state
from wt.benchmark_v2.phase6e.generators import is_legal
from wt.benchmark_v2.phase6e.value import _zero_arrivals
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import limit_threads, load_primary, parameter_token
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import PRIMARY_SEED
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import make_option
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST, VALIDATION_POOL
from wt.benchmark_v2.phase6ij_executor_audit.audit import _matches_saved, _reach
from wt.benchmark_v2.phase6ik_execution_closure.audit import _rng_token
from wt.benchmark_v2.phase6im_local_swap.audit import _label_table, single_swaps
from wt.benchmark_v2.phase6io_per_uav_response.audit import _move
from wt.l3a.env import conflict_mask
from wt.l3b.env import interference_matrix, path_gain


ROOT = Path(__file__).resolve().parents[4]
AUDIT_DIR = ROOT / "results/phase6iq_short_horizon"
SOURCE = ROOT / "results/phase6im_local_swap" / "states.json"
CASES = ROOT / "results/phase6io_per_uav_response" / "cases.json"
PILOT_SCENARIOS = (322051, 322049, 322062)
HORIZONS = (1, 3, 5)
MAX_STEPS = 5
MAX_NEW_ROLLOUTS = 48
BIT_TOLERANCE = 1.0
MATERIAL_BITS = 1.0e6
METRICS = (
    ("delivered_bits", True, BIT_TOLERANCE),
    ("mean_distance_m", False, 0.5),
    ("in_range_count", True, 0.5),
    ("mean_sinr", True, 0.05),
    ("mean_interference_w", False, 0.01),
)
NAMED = {
    322051: ((1, 2),),
    322049: ((3, 4),),
    322062: ((0, 4), (1, 2)),
}


def assign_gate(blocked: bool, supported_metrics: int) -> str:
    if blocked:
        return "SHORT_HORIZON_RESPONSE_BLOCKED"
    if int(supported_metrics) > 0:
        return "SHORT_HORIZON_RESPONSE_SUPPORTED"
    return "SHORT_HORIZON_RESPONSE_INCONCLUSIVE"


def relation(left: float | None, right: float | None, *, higher: bool, floor: float) -> str:
    """Which side is better. A missing observation is not a numeric zero."""
    if left is None or right is None:
        return "missing"
    delta = float(left) - float(right)
    if abs(delta) <= float(floor):
        return "tie"
    better_left = delta > 0.0 if higher else delta < 0.0
    return "left" if better_left else "right"


def evaluate_metric(states: list[dict], *, higher: bool, floor: float) -> dict:
    stable = 0
    pairs = 0
    material = 0
    consistent = 0
    by_horizon = {str(horizon): {"agree": 0, "disagree": 0, "tie": 0, "missing": 0, "material": 0} for horizon in HORIZONS}
    for state in states:
        items = state["items"]
        for left_index in range(len(items)):
            for right_index in range(left_index + 1, len(items)):
                left = items[left_index]
                right = items[right_index]
                pairs += 1
                relations = [
                    relation(left["values"][str(horizon)], right["values"][str(horizon)], higher=higher, floor=floor)
                    for horizon in HORIZONS
                ]
                if len(set(relations)) == 1:
                    stable += 1
                label = relation(left["j15"], right["j15"], higher=True, floor=BIT_TOLERANCE)
                if label not in {"left", "right"} or abs(float(left["j15"]) - float(right["j15"])) <= MATERIAL_BITS:
                    continue
                material += 1
                aligned = True
                for horizon, observed in zip(HORIZONS, relations):
                    bucket = by_horizon[str(horizon)]
                    bucket["material"] += 1
                    bucket[observed if observed in {"tie", "missing"} else ("agree" if observed == label else "disagree")] += 1
                    if observed != label:
                        aligned = False
                if aligned:
                    consistent += 1
    return {
        "pairs": int(pairs),
        "rank_stable_pairs": int(stable),
        "rank_stable": int(pairs > 0 and stable == pairs),
        "material_pairs": int(material),
        "consistent_pairs": int(consistent),
        "consistent": int(material > 0 and consistent == material),
        "supported": int(pairs > 0 and stable == pairs and material > 0 and consistent == material),
        "by_horizon": by_horizon,
    }


def _horizon_snapshot(steps: list[dict]) -> dict[str, dict]:
    snapshots = {}
    served = np.zeros(len(steps[0]["uavs"]), dtype=np.float64)
    service_steps = np.zeros(len(served), dtype=np.int64)
    entry = np.full(len(served), -1, dtype=np.int64)
    sinr_sum = np.zeros(len(served), dtype=np.float64)
    interference_sum = np.zeros(len(served), dtype=np.float64)
    active_steps = np.zeros(len(served), dtype=np.int64)
    projector = np.zeros(len(served), dtype=np.int64)
    conflicts = np.zeros(len(served), dtype=np.int64)
    for step_index, step in enumerate(steps):
        for item in step["uavs"]:
            uav = int(item["uav"])
            served[uav] += float(item["served_bits"])
            projector[uav] += int(item["projector"])
            conflicts[uav] += int(item["conflict"])
            if int(item["link_active"]):
                service_steps[uav] += 1
                active_steps[uav] += 1
                sinr_sum[uav] += float(item["sinr"])
                interference_sum[uav] += float(item["interference_w"])
                if int(entry[uav]) < 0:
                    entry[uav] = int(step_index)
        horizon = step_index + 1
        if horizon not in HORIZONS:
            continue
        per_uav = []
        for item in step["uavs"]:
            uav = int(item["uav"])
            per_uav.append(
                {
                    "uav": uav,
                    "target": int(item["target"]),
                    "position_m": item["position_m"],
                    "distance_m": item["distance_m"],
                    "in_range": int(item["in_range"]),
                    "link_active": int(active_steps[uav] > 0),
                    "sinr": None if active_steps[uav] == 0 else float(sinr_sum[uav] / active_steps[uav]),
                    "interference_w": None if active_steps[uav] == 0 else float(interference_sum[uav] / active_steps[uav]),
                    "rate_bps": float(item["rate_bps"]) if int(item["link_active"]) else None,
                    "progress": float(item["progress"]),
                    "delivered_bits": float(served[uav]),
                    "service_steps": int(service_steps[uav]),
                    "entry": None if int(entry[uav]) < 0 else int(entry[uav]),
                    "idle": int(item["idle"]),
                    "projector_steps": int(projector[uav]),
                    "conflict_steps": int(conflicts[uav]),
                }
            )
        active = [item["sinr"] for item in per_uav if item["sinr"] is not None]
        active_interference = [item["interference_w"] for item in per_uav if item["interference_w"] is not None]
        distances = [float(item["distance_m"]) for item in per_uav if item["distance_m"] is not None]
        snapshots[str(horizon)] = {
            "delivered_bits": float(sum(item["delivered_bits"] for item in per_uav)),
            "mean_distance_m": float(np.mean(distances)) if distances else None,
            "in_range_count": int(sum(item["in_range"] for item in per_uav)),
            "mean_sinr": None if not active else float(np.mean(active)),
            "mean_interference_w": None if not active_interference else float(np.mean(active_interference)),
            "service_steps": int(sum(item["service_steps"] for item in per_uav)),
            "entered": int(sum(item["entry"] is not None for item in per_uav)),
            "progress_sum": float(sum(item["progress"] for item in per_uav)),
            "projector_steps": int(sum(item["projector_steps"] for item in per_uav)),
            "per_uav": per_uav,
            "seconds": float(step["seconds"]),
        }
    return snapshots


def _short_rollout(env, actor, snap: dict, assignment: np.ndarray) -> dict:
    restore_state(env, snap)
    _zero_arrivals(env)
    chosen = np.asarray(assignment, dtype=np.int64).reshape(-1).copy()
    option = make_option(env, chosen)
    if int(option.remaining_hold) != 15:
        raise RuntimeError("hold was not reset to 15")
    rng_before = _rng_token(env)
    last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    served_before = np.asarray(snap["served_k"], dtype=np.int64).reshape(-1)
    switches = [int(int(buoy) != int(previous)) for buoy, previous in zip(chosen, served_before)]
    started = time.perf_counter()
    steps = []
    for _step_index in range(MAX_STEPS):
        movement, team, touched = _move(env, actor, option, last)
        info = env.step(movement, option.association)
        if not np.array_equal(np.asarray(env.assignment, dtype=np.int64), chosen):
            raise RuntimeError("assignment changed during the short prefix")
        served = np.asarray(info["served_bits"], dtype=np.float64)
        matrix = interference_matrix(env)
        row = []
        for uav, buoy in enumerate(chosen):
            target = int(buoy)
            idle = int(target < 0)
            distance = None if idle else float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[target]))
            inside = int((not idle) and distance <= float(env.config.service_radius_m))
            active = int(float(env.last_rate[uav]) > 0.0)
            row.append(
                {
                    "uav": int(uav),
                    "target": target,
                    "position_m": [float(item) for item in env.uav_positions[uav]],
                    "distance_m": distance,
                    "in_range": inside,
                    "link_active": active,
                    "sinr": float(env.last_sinr[uav]) if active else None,
                    "interference_w": float(np.sum(matrix[uav])) if active else None,
                    "rate_bps": float(env.last_rate[uav]) if active else None,
                    "progress": float(env.progress[uav]),
                    "served_bits": 0.0 if idle else float(served[target]),
                    "idle": idle,
                    "projector": int(touched[uav]),
                    "team_projector": int(team),
                    "conflict": int(bool(conflict_mask(env.assignment, env.config.num_buoys)[uav])),
                    "signal_w": 0.0 if idle else float(path_gain(env.config, env.uav_positions[uav], env.buoy_positions[target])) * float(env.config.tx_power),
                }
            )
        steps.append({"seconds": float(time.perf_counter() - started), "delivered_bits": float(np.sum(served)), "uavs": row})
        option.remaining_hold -= 1
        last = movement
    bit_sum = float(sum(sum(float(item["served_bits"]) for item in step["uavs"]) for step in steps))
    fleet_sum = float(sum(float(step["delivered_bits"]) for step in steps))
    return {
        "steps": steps,
        "horizons": _horizon_snapshot(steps),
        "switches": switches,
        "rng_consumed": int(_rng_token(env) != rng_before),
        "bit_sum_error": abs(bit_sum - fleet_sum),
    }


def _prefix_error(saved_steps: list[dict], new_steps: list[dict]) -> dict:
    gaps = {"served_bits": 0.0, "distance_m": 0.0, "sinr": 0.0, "interference_w": 0.0, "missing": 0}
    for saved, new in zip(saved_steps[:MAX_STEPS], new_steps[:MAX_STEPS]):
        for left, right in zip(saved["uavs"], new["uavs"]):
            for name in ("served_bits", "distance_m", "sinr", "interference_w"):
                if left.get(name) is None or right.get(name) is None:
                    if name == "sinr" or name == "interference_w":
                        inactive = int(right.get("link_active") == 0 and abs(float(left.get(name) or 0.0)) == 0.0)
                        if inactive:
                            continue
                    gaps["missing"] += 1
                    continue
                gaps[name] = max(gaps[name], abs(float(left[name]) - float(right[name])))
    gaps["matched"] = int(
        gaps["missing"] == 0
        and gaps["served_bits"] <= 1.0e-3
        and gaps["distance_m"] <= 1.0e-4
        and gaps["sinr"] <= 1.0e-4
        and gaps["interference_w"] <= 1.0e-4
    )
    return gaps


def _saved_prefixes() -> dict[tuple[int, tuple[int, ...]], list[dict]]:
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    found = {}
    for case in cases.values():
        scenario = int(case["scenario_id"])
        if scenario not in PILOT_SCENARIOS:
            continue
        for arm in case["arms"].values():
            found[(scenario, tuple(int(item) for item in arm["assignment"]))] = arm["steps"]
    return found


def run_audit() -> dict:
    if any(int(scenario) in RESERVED_TEST or int(scenario) not in VALIDATION_POOL for scenario in PILOT_SCENARIOS):
        raise RuntimeError("pilot scenarios leave the validation pool")
    limit_threads()
    tracemalloc.start()
    actor = load_primary()
    token = parameter_token(actor)
    local = {int(row["scenario_id"]): row for row in json.loads(SOURCE.read_text(encoding="utf-8"))}
    saved = _saved_prefixes()
    blocked = False
    reasons = []
    rollouts = []
    metric_states = {name: [] for name, _higher, _floor in METRICS}
    named = []
    new_rollouts = 0
    for scenario in PILOT_SCENARIOS:
        scheme = tuple(int(item) for item in local[int(scenario)]["assignments"]["scheme_c"])
        labels = _label_table(int(scenario))
        catalog = [(None, None, scheme)] + [(int(left), int(right), tuple(int(item) for item in row)) for left, right, row in single_swaps(np.asarray(scheme))]
        if any(not is_legal(np.asarray(row), 6, 12) or row not in labels for _left, _right, row in catalog):
            blocked = True
            reasons.append(f"{scenario}:illegal_or_unlabeled")
        reached = time.perf_counter()
        env = _reach(int(scenario), actor)
        reach_seconds = float(time.perf_counter() - reached)
        match = _matches_saved(env, int(scenario))
        if int(match["matched"]) != 1:
            blocked = True
            reasons.append(f"{scenario}:saved_state_mismatch")
        snap = capture_state(env)
        rng_before = _rng_token(env)
        items = {name: [] for name, _higher, _floor in METRICS}
        executed = []
        for left, right, assignment in catalog:
            if new_rollouts >= MAX_NEW_ROLLOUTS:
                raise RuntimeError("short-horizon rollout budget exceeded")
            trace = _short_rollout(env, actor, snap, np.asarray(assignment))
            new_rollouts += 1
            if int(trace["rng_consumed"]) == 1 or float(trace["bit_sum_error"]) > BIT_TOLERANCE:
                blocked = True
                reasons.append(f"{scenario}:rng_or_bit_sum")
            key = (int(scenario), assignment)
            if key in saved:
                prefix = _prefix_error(saved[key], trace["steps"])
                if int(prefix["matched"]) != 1:
                    blocked = True
                    reasons.append(f"{scenario}:{assignment}:prefix_mismatch")
            else:
                prefix = None
            record = {
                "scenario_id": int(scenario),
                "uav_i": left,
                "uav_j": right,
                "assignment": list(assignment),
                "j15_bits": float(labels[assignment]),
                "switches": trace["switches"],
                "horizons": trace["horizons"],
                "prefix": prefix,
                "reach_seconds": reach_seconds,
            }
            executed.append(record)
            for name, _higher, _floor in METRICS:
                items[name].append(
                    {
                        "j15": float(labels[assignment]),
                        "values": {str(horizon): trace["horizons"][str(horizon)][name] for horizon in HORIZONS},
                    }
                )
            if (left, right) in NAMED[int(scenario)] or (left is None and right is None):
                named.append(record)
        if _rng_token(env) != rng_before:
            blocked = True
            reasons.append(f"{scenario}:rng_consumed")
        for name, _higher, _floor in METRICS:
            metric_states[name].append({"scenario_id": int(scenario), "items": items[name]})
        rollouts.extend(
            {
                "scenario_id": int(item["scenario_id"]),
                "assignment": item["assignment"],
                "uav_i": item["uav_i"],
                "uav_j": item["uav_j"],
                "j15_bits": item["j15_bits"],
                "seconds": {str(horizon): item["horizons"][str(horizon)]["seconds"] for horizon in HORIZONS},
                "fleet": {
                    str(horizon): {key: item["horizons"][str(horizon)][key] for key in ("delivered_bits", "mean_distance_m", "in_range_count", "mean_sinr", "mean_interference_w", "service_steps", "entered", "progress_sum")}
                    for horizon in HORIZONS
                },
            }
            for item in executed
        )
    if parameter_token(actor) != token:
        blocked = True
        reasons.append("actor_changed")
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    reports = {}
    for name, higher, floor in METRICS:
        report = evaluate_metric(metric_states[name], higher=higher, floor=floor)
        report["higher_is_better"] = int(higher)
        report["floor"] = float(floor)
        reports[name] = report
    times = {str(horizon): [float(item["seconds"][str(horizon)]) for item in rollouts] for horizon in HORIZONS}
    summary = {
        "gate": assign_gate(blocked, sum(int(item["supported"]) for item in reports.values())),
        "blocked": int(blocked),
        "blocked_reasons": reasons,
        "new_rollouts": int(new_rollouts),
        "max_new_rollouts": int(MAX_NEW_ROLLOUTS),
        "pilot_scenarios": list(PILOT_SCENARIOS),
        "horizons": list(HORIZONS),
        "metrics": reports,
        "seconds_median": {horizon: float(np.median(values)) for horizon, values in times.items()},
        "seconds_max": {horizon: float(np.max(values)) for horizon, values in times.items()},
        "peak_traced_bytes": int(peak),
        "primary_seed": int(PRIMARY_SEED),
        "actor_unchanged": int(parameter_token(actor) == token),
        "named": [
            {
                "scenario_id": int(item["scenario_id"]),
                "uav_i": item["uav_i"],
                "uav_j": item["uav_j"],
                "assignment": item["assignment"],
                "j15_bits": item["j15_bits"],
                "switches": item["switches"],
                "prefix_matched": None if item["prefix"] is None else int(item["prefix"]["matched"]),
                "horizons": item["horizons"],
            }
            for item in named
        ],
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    for stem, payload in (("summary", summary), ("rollouts", rollouts)):
        path = AUDIT_DIR / f"{stem}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
    print(
        json.dumps(
            {
                "gate": summary["gate"],
                "rollouts": new_rollouts,
                "blocked": reasons,
                "seconds_median": summary["seconds_median"],
                "supported": {name: int(item["supported"]) for name, item in reports.items()},
            }
        ),
        flush=True,
    )
    return summary


if __name__ == "__main__":
    run_audit()
