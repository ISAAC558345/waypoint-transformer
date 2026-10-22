"""Local UAV-buoy swaps inside one service set. No training."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6e.generators import is_legal
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import limit_threads, load_primary, parameter_token
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import HOLD
from wt.benchmark_v2.phase6ib_highlevel_dataset.hgnn_revision import REVISION_DATASET
from wt.benchmark_v2.phase6ib_highlevel_dataset.schema import state_arrays
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST, VALIDATION_POOL
from wt.benchmark_v2.phase6ii_candidate_features.features import candidate_features, feature_sums
from wt.benchmark_v2.phase6ij_executor_audit.audit import _matches_saved, _reach, buoy_set
from wt.benchmark_v2.phase6ik_execution_closure.audit import _rng_token, _trace
from wt.benchmark_v2.phase6d.shadow import capture_state


ROOT = Path(__file__).resolve().parents[4]
AUDIT_DIR = ROOT / "results/phase6im_local_swap"
PRIOR = ROOT / "results/phase6ik_execution_closure" / "states.json"
# One baseline plus fifteen single swaps on each of the sixteen validation states.
MAX_NEW_ROLLOUTS = 256
BIT_TOLERANCE = 1.0
MATERIAL_BITS = 1.0e6
EXPLAIN_STATES = 12
HIGHER_IS_BETTER = {
    "hypothetical_rate_bps",
    "noise_only_rate_bps",
    "geometric_rate_bps",
    "instant_rate_bps",
    "straight_line_bits",
    "kinematic_bits",
}
FLOORS = {
    "distance_m": 1.0,
    "steps_to_radius": 1.0 / 22.0,
    "hypothetical_rate_bps": 1.0e4,
    "noise_only_rate_bps": 1.0e4,
    "geometric_rate_bps": 1.0e4,
    "geometric_interference_w": 1.0e-6,
    "instant_rate_bps": 1.0e4,
    "straight_line_bits": 1.0e5,
    "kinematic_bits": 1.0e5,
    "switch": 0.5,
    "abandoned_progress": 0.05,
    "hover_interference_w": 1.0e-6,
    "service_overlap_s": 1.0,
}


def assign_gate(blocked: bool, material_states: int, explained_states: int) -> str:
    if blocked:
        return "LOCAL_PAIR_AUDIT_BLOCKED"
    if int(material_states) >= EXPLAIN_STATES and int(explained_states) >= EXPLAIN_STATES:
        return "LOCAL_PAIR_EFFECT_SUPPORTED"
    return "LOCAL_PAIR_EFFECT_INCONCLUSIVE"


def single_swaps(assignment: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    """Exchange the targets of two UAVs. Idle-idle pairs are omitted."""
    base = np.asarray(assignment, dtype=np.int64).reshape(-1).copy()
    found = []
    for left in range(base.size):
        for right in range(left + 1, base.size):
            if int(base[left]) == int(base[right]):
                continue
            swapped = base.copy()
            swapped[left], swapped[right] = swapped[right], swapped[left]
            found.append((int(left), int(right), swapped))
    return found


def two_swap_assignments(assignment: np.ndarray) -> list[np.ndarray]:
    """Unique assignments reached by two successive UAV-target exchanges."""
    seen = {tuple(int(item) for item in np.asarray(assignment, dtype=np.int64).reshape(-1))}
    for _left, _right, row in single_swaps(assignment):
        seen.add(tuple(int(item) for item in row))
    found = []
    for _left, _right, row in single_swaps(assignment):
        for _other_left, _other_right, child in single_swaps(row):
            key = tuple(int(item) for item in child)
            if key in seen:
                continue
            seen.add(key)
            found.append(child)
    return found


def transposition_distance(source: np.ndarray, target: np.ndarray) -> int | None:
    """Minimum exchanges that turn ``source`` into ``target`` when the buoy set matches."""
    left = [int(item) for item in np.asarray(source, dtype=np.int64).reshape(-1)]
    right = [int(item) for item in np.asarray(target, dtype=np.int64).reshape(-1)]
    if sorted(left) != sorted(right):
        return None
    index = {buoy: position for position, buoy in enumerate(left)}
    permutation = [index[buoy] for buoy in right]
    seen = [False] * len(permutation)
    cycles = 0
    for start in range(len(permutation)):
        if seen[start]:
            continue
        cycles += 1
        cursor = start
        while not seen[cursor]:
            seen[cursor] = True
            cursor = permutation[cursor]
    return int(len(permutation) - cycles)


def _service_overlap(steps: np.ndarray, hold: float) -> np.ndarray:
    start = np.asarray(steps, dtype=np.float64)
    end = np.full(start.shape, float(hold), dtype=np.float64)
    begin = np.maximum(start[:, :, None], start[:, None, :])
    finish = np.minimum(end[:, :, None], end[:, None, :])
    overlap = np.maximum(finish - begin, 0.0)
    eye = np.eye(start.shape[1], dtype=bool)
    return np.sum(np.where(eye, 0.0, overlap), axis=(1, 2)) * 0.5


def _label_table(scenario: int) -> dict[tuple[int, ...], float]:
    path = REVISION_DATASET / "validation" / f"scenario_{int(scenario)}" / "t015_candidates.npz"
    with np.load(path) as payload:
        rows = np.asarray(payload["assignment"], dtype=np.int64)
        labels = np.asarray(payload["j15_bits"], dtype=np.float64)
    table: dict[tuple[int, ...], float] = {}
    for row, label in zip(rows, labels):
        table[tuple(int(item) for item in row)] = float(label)
    return table


def _compact_trace(trace: dict) -> dict:
    per_uav = []
    for index, item in enumerate(trace["per_uav"]):
        entry = int(item["entry"])
        distance = None
        if entry < int(HOLD):
            distance = float(trace["steps"][entry]["uavs"][index]["distance_m"])
        per_uav.append(
            {
                "target": int(item["target"]),
                "entry": entry,
                "entry_distance_m": distance,
                "service_steps": int(item["service_steps"]),
                "delivered_bits": float(item["delivered_bits"]),
                "mean_sinr": float(item["mean_sinr"]),
                "mean_rate_bps": float(item["mean_rate_bps"]),
                "mean_interference_w": float(item["mean_interference_w"]),
                "end_progress": float(item["end_progress"]),
            }
        )
    blocks = 0.0
    conflict_steps = 0
    step_sinr = []
    step_interference = []
    for step in trace["steps"]:
        blocks += float(sum(float(uav["completed_blocks"]) for uav in step["uavs"]))
        conflict_steps += int(any(int(uav["conflict"]) for uav in step["uavs"]))
        active = [uav for uav in step["uavs"] if float(uav["served_bits"]) > 0.0]
        step_sinr.append(float(np.mean([uav["sinr"] for uav in active])) if active else 0.0)
        step_interference.append(float(np.sum([uav["interference_w"] for uav in active])) if active else 0.0)
    entered = [float(item["entry_distance_m"]) for item in per_uav if item["entry_distance_m"] is not None]
    return {
        "delivered_bits": float(trace["delivered_bits"]),
        "completed_blocks": float(blocks),
        "rng_consumed": int(trace["rng_consumed"]),
        "projector_steps": int(sum(int(step["projector"]) for step in trace["steps"])),
        "conflict_steps": int(conflict_steps),
        "mean_entry": float(np.mean([item["entry"] for item in per_uav])),
        "mean_entry_distance_m": float(np.mean(entered)) if entered else None,
        "service_steps": int(sum(item["service_steps"] for item in per_uav)),
        "mean_sinr": float(np.mean([item["mean_sinr"] for item in per_uav if item["service_steps"] > 0] or [0.0])),
        "mean_interference_w": float(np.mean([item["mean_interference_w"] for item in per_uav if item["service_steps"] > 0] or [0.0])),
        "step_delivered_bits": [float(item) for item in trace["step_delivered_bits"]],
        "step_mean_sinr": step_sinr,
        "step_sum_interference_w": step_interference,
        "per_uav": per_uav,
    }


def _best_index(score: np.ndarray, higher: bool) -> int:
    signed = score if higher else -score
    return int(np.argsort(-np.asarray(signed, dtype=np.float64), kind="mergesort")[0])


def _feature_report(sums: dict[str, np.ndarray], labels: np.ndarray) -> dict:
    winner = int(np.argsort(-np.asarray(labels, dtype=np.float64), kind="mergesort")[0])
    report = {}
    for name, values in sums.items():
        column = np.asarray(values, dtype=np.float64)
        spread = float(np.ptp(column))
        separated = int(spread > float(FLOORS.get(name, 0.0)))
        picked = _best_index(column, name in HIGHER_IS_BETTER)
        report[name] = {
            "spread": spread,
            "separated": separated,
            "picked": picked,
            "matches_j15": int(separated == 1 and picked == winner),
        }
    return report


def run_audit() -> dict:
    if any(int(scenario) in RESERVED_TEST for scenario in VALIDATION_POOL):
        raise RuntimeError("validation pool overlaps the reserved test")
    limit_threads()
    actor = load_primary()
    token = parameter_token(actor)
    prior = {int(row["scenario_id"]): row for row in json.loads(PRIOR.read_text(encoding="utf-8"))}
    rows = []
    new_rollouts = 0
    durations = []
    label_errors = []
    blocked = False
    for scenario in VALIDATION_POOL:
        if int(scenario) in RESERVED_TEST:
            raise RuntimeError("refused a reserved scenario")
        source = prior[int(scenario)]
        scheme = np.asarray(source["assignments"]["scheme_c"], dtype=np.int64)
        oracle = np.asarray(source["assignments"]["oracle"], dtype=np.int64)
        epoch6 = np.asarray(source["assignments"]["epoch6"], dtype=np.int64)
        labels = _label_table(int(scenario))
        singles = single_swaps(scheme)
        doubles = two_swap_assignments(scheme)
        illegal = 0
        set_changes = 0
        kept = []
        for left, right, swapped in singles:
            legal = bool(is_legal(swapped, 6, 12))
            same_set = buoy_set(swapped) == buoy_set(scheme)
            illegal += int(not legal)
            set_changes += int(not same_set)
            if legal and same_set:
                kept.append((left, right, swapped))
        missing = [row for _left, _right, row in kept if tuple(int(item) for item in row) not in labels]
        missing_doubles = [row for row in doubles if tuple(int(item) for item in row) not in labels]
        if illegal or set_changes or missing or not is_legal(scheme, 6, 12):
            blocked = True
        env = _reach(int(scenario), actor)
        saved = _matches_saved(env, int(scenario))
        if int(saved["matched"]) != 1:
            blocked = True
        arrays = state_arrays(env)
        arrays["service_progress"] = np.asarray(env.progress, dtype=np.float64).copy()
        catalog = [(None, None, scheme)] + kept
        if new_rollouts + len(catalog) > MAX_NEW_ROLLOUTS:
            raise RuntimeError("local rollout budget exceeded before another state")
        stacked = np.stack([row for _left, _right, row in catalog])
        raw = candidate_features(arrays, stacked, env.config)
        sums = feature_sums(raw)
        sums["service_overlap_s"] = _service_overlap(raw["steps_to_radius"], float(env.config.option_hold))
        cached = np.asarray([labels[tuple(int(item) for item in row)] for _left, _right, row in catalog], dtype=np.float64)
        snap = capture_state(env)
        rng_before = _rng_token(env)
        executed = []
        for index, (left, right, row) in enumerate(catalog):
            started = time.perf_counter()
            trace = _compact_trace(_trace(env, actor, snap, row))
            durations.append(float(time.perf_counter() - started))
            new_rollouts += 1
            error = abs(float(trace["delivered_bits"]) - float(cached[index]))
            label_errors.append(float(error))
            if error > BIT_TOLERANCE or int(trace["rng_consumed"]) == 1:
                blocked = True
            executed.append(
                {
                    "role": "scheme_c" if index == 0 else "single_swap",
                    "uav_i": None if left is None else int(left),
                    "uav_j": None if right is None else int(right),
                    "assignment": [int(item) for item in row],
                    "j15_bits": float(cached[index]),
                    "delta_bits": float(cached[index] - cached[0]),
                    "label_abs_error": float(error),
                    "features": {name: float(np.asarray(values)[index]) for name, values in sums.items()},
                    "rollout": trace,
                }
            )
        if _rng_token(env) != rng_before:
            blocked = True
        feature_rows = _feature_report(sums, cached)
        double_labels = [labels[tuple(int(item) for item in row)] for row in doubles if tuple(int(item) for item in row) in labels]
        double_delta = np.asarray(double_labels, dtype=np.float64) - float(cached[0]) if double_labels else np.asarray([0.0])
        rows.append(
            {
                "scenario_id": int(scenario),
                "saved_match": saved,
                "assignments": {
                    "scheme_c": [int(item) for item in scheme],
                    "oracle": [int(item) for item in oracle],
                    "epoch6": [int(item) for item in epoch6],
                },
                "legal_single_swaps": int(len(kept)),
                "illegal_single_swaps": int(illegal),
                "buoy_set_changes": int(set_changes),
                "missing_single_labels": int(len(missing)),
                "transposition_distance": transposition_distance(scheme, oracle),
                "oracle_is_single_swap": int(any(np.array_equal(row, oracle) for _left, _right, row in kept)),
                "epoch6_is_single_swap": int(any(np.array_equal(row, epoch6) for _left, _right, row in kept)),
                "scheme_bits": float(cached[0]),
                "oracle_bits": float(labels[tuple(int(item) for item in oracle)]),
                "best_single_bits": float(np.max(cached)),
                "worst_single_bits": float(np.min(cached)),
                "max_up_bits": float(np.max(cached[1:] - cached[0])),
                "max_down_bits": float(np.min(cached[1:] - cached[0])),
                "positive_swaps": int(np.sum(cached[1:] - cached[0] > BIT_TOLERANCE)),
                "negative_swaps": int(np.sum(cached[0] - cached[1:] > BIT_TOLERANCE)),
                "tie_swaps": int(np.sum(np.abs(cached[1:] - cached[0]) <= BIT_TOLERANCE)),
                "best_minus_oracle_bits": float(np.max(cached) - labels[tuple(int(item) for item in oracle)]),
                "double_swaps": int(len(doubles)),
                "missing_double_labels": int(len(missing_doubles)),
                "double_max_up_bits": float(np.max(double_delta)) if double_labels else None,
                "double_max_down_bits": float(np.min(double_delta)) if double_labels else None,
                "oracle_is_double_swap": int(any(np.array_equal(row, oracle) for row in doubles)),
                "features": feature_rows,
                "arms": executed,
            }
        )
        print({"scenario_id": int(scenario), "rollouts": new_rollouts, "max_abs_error": max(label_errors)}, flush=True)
    if parameter_token(actor) != token:
        blocked = True
    material = sum(1 for row in rows if max(abs(float(row["max_up_bits"])), abs(float(row["max_down_bits"]))) > MATERIAL_BITS)
    explained = {}
    for name in FLOORS:
        explained[name] = sum(int(row["features"][name]["matches_j15"]) for row in rows)
    explained_states = max(explained.values()) if explained else 0
    gate = assign_gate(blocked, material, explained_states)
    summary = {
        "gate": gate,
        "states": int(len(rows)),
        "new_rollouts": int(new_rollouts),
        "max_new_rollouts": int(MAX_NEW_ROLLOUTS),
        "rollout_seconds_median": float(np.median(durations)) if durations else None,
        "rollout_seconds_max": float(np.max(durations)) if durations else None,
        "label_abs_error_max": float(np.max(label_errors)) if label_errors else None,
        "rng_consumed_max": int(max(int(arm["rollout"]["rng_consumed"]) for row in rows for arm in row["arms"])),
        "saved_match_count": int(sum(int(row["saved_match"]["matched"]) for row in rows)),
        "actor_unchanged": int(parameter_token(actor) == token),
        "material_states": int(material),
        "explained_states": int(explained_states),
        "explained_by_feature": explained,
        "legal_single_swaps_per_state": [int(row["legal_single_swaps"]) for row in rows],
        "snapshot_omits": ["rng", "initial_q_bits", "_phase6h_last"],
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, payload in (("summary", summary), ("states", rows)):
        path = AUDIT_DIR / f"{name}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
    print(json.dumps({"gate": gate, "new_rollouts": new_rollouts, "label_abs_error_max": summary["label_abs_error_max"]}), flush=True)
    return summary


if __name__ == "__main__":
    run_audit()
