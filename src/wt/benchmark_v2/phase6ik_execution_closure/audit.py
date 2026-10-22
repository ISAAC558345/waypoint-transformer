"""Close the Phase-6I-J execution audit. No training and no label edits."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6e.generators import is_legal
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import limit_threads, load_primary, parameter_token
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import HOLD
from wt.benchmark_v2.phase6ib_highlevel_dataset.hgnn_revision import REVISION_DATASET
from wt.benchmark_v2.phase6ib_highlevel_dataset.schema import state_arrays
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST, VALIDATION_POOL
from wt.benchmark_v2.phase6ii_candidate_features.features import candidate_features
from wt.benchmark_v2.phase6ij_executor_audit.audit import _execute, _matches_saved, _reach, buoy_set
from wt.benchmark_v2.phase6d.shadow import capture_state
from wt.benchmark_v2.phase6e.value import _zero_arrivals
from wt.benchmark_v2.phase6g_lowlevel_mappo.evaluate import _actor_move
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import make_option
from wt.l3a.env import conflict_mask
from wt.l3b.env import interference_matrix, path_gain


ROOT = Path(__file__).resolve().parents[4]
AUDIT_DIR = ROOT / "results/phase6ik_execution_closure"
PRIOR = ROOT / "results/phase6ij_executor_audit" / "states.json"
ARMS = ("oracle", "scheme_c", "epoch6")
TRACE_SCENARIOS = (322048, 322051, 322055)


def vector_hamming(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.sum(np.asarray(left, dtype=np.int64).reshape(-1) != np.asarray(right, dtype=np.int64).reshape(-1)))


def assign_gate(consistent: bool, reproduced: bool, rng_consumed: bool) -> str:
    if not consistent:
        return "EXECUTION_AUDIT_INCONSISTENT"
    if rng_consumed or not reproduced:
        return "EXECUTION_AUDIT_PARTIALLY_REPRODUCIBLE"
    return "EXECUTION_AUDIT_REPRODUCIBLE"


def _rng_token(env) -> str:
    return json.dumps(env.rng.bit_generator.state, default=lambda item: np.asarray(item).tolist(), sort_keys=True)


def _label_of(scenario: int, assignment: np.ndarray) -> float:
    path = REVISION_DATASET / "validation" / f"scenario_{int(scenario)}" / "t015_candidates.npz"
    chosen = np.asarray(assignment, dtype=np.int64).reshape(1, -1)
    with np.load(path) as payload:
        rows = np.asarray(payload["assignment"], dtype=np.int64)
        labels = np.asarray(payload["j15_bits"], dtype=np.float64)
    hits = np.flatnonzero(np.all(rows == chosen, axis=1))
    if hits.size != 1:
        raise RuntimeError("assignment is not a unique candidate row")
    return float(labels[int(hits[0])])


def _trace(scratch, actor, snap: dict, assignment: np.ndarray) -> dict:
    _zero_arrivals(scratch)
    from wt.benchmark_v2.phase6d.shadow import restore_state

    restore_state(scratch, snap)
    _zero_arrivals(scratch)
    chosen = np.asarray(assignment, dtype=np.int64).reshape(-1).copy()
    option = make_option(scratch, chosen)
    if int(option.remaining_hold) != int(HOLD):
        raise RuntimeError("hold was not reset to 15")
    rng_before = _rng_token(scratch)
    last = np.zeros((scratch.config.num_uavs, 2), dtype=np.float64)
    delivered = np.zeros(int(HOLD), dtype=np.float64)
    per_uav = []
    for uav in range(chosen.shape[0]):
        per_uav.append({"target": int(chosen[uav]), "entry": int(HOLD), "service_steps": 0, "delivered_bits": 0.0, "sinr": [], "rate": [], "signal_w": [], "interference_w": [], "progress": []})
    steps = []
    for step_index in range(int(HOLD)):
        movement, intervened, _residual = _actor_move(scratch, actor, option, last)
        info = scratch.step(movement, option.association)
        if not np.array_equal(np.asarray(scratch.assignment, dtype=np.int64), chosen):
            raise RuntimeError("assignment changed during the hold")
        served = np.asarray(info["served_bits"], dtype=np.float64)
        blocks = np.asarray(info["completed_block_equivalent_increment"], dtype=np.float64)
        matrix = interference_matrix(scratch)
        delivered[step_index] = float(np.sum(served))
        row = []
        for uav, buoy in enumerate(chosen):
            target = int(buoy)
            idle = int(target < 0)
            distance = float("nan") if idle else float(np.linalg.norm(scratch.uav_positions[uav] - scratch.buoy_positions[target]))
            inside = int((not idle) and distance <= float(scratch.config.service_radius_m))
            if inside and per_uav[uav]["entry"] == int(HOLD):
                per_uav[uav]["entry"] = int(step_index)
            active = float(scratch.last_rate[uav]) > 0.0
            signal = 0.0 if idle else float(path_gain(scratch.config, scratch.uav_positions[uav], scratch.buoy_positions[target])) * float(scratch.config.tx_power)
            interference = float(np.sum(matrix[uav]))
            bits = 0.0 if idle else float(served[target])
            if active:
                per_uav[uav]["service_steps"] += 1
                per_uav[uav]["sinr"].append(float(scratch.last_sinr[uav]))
                per_uav[uav]["rate"].append(float(scratch.last_rate[uav]))
                per_uav[uav]["signal_w"].append(signal)
                per_uav[uav]["interference_w"].append(interference)
            per_uav[uav]["delivered_bits"] += bits
            per_uav[uav]["progress"].append(float(scratch.progress[uav]))
            row.append(
                {
                    "uav": int(uav),
                    "target": target,
                    "idle": idle,
                    "distance_m": distance,
                    "in_range": inside,
                    "sinr": float(scratch.last_sinr[uav]),
                    "rate_bps": float(scratch.last_rate[uav]),
                    "signal_w": signal,
                    "interference_w": interference,
                    "progress": float(scratch.progress[uav]),
                    "served_bits": bits,
                    "completed_blocks": 0.0 if idle else float(blocks[target]),
                    "conflict": int(bool(conflict_mask(scratch.assignment, scratch.config.num_buoys)[uav])),
                }
            )
        steps.append({"delivered_bits": float(delivered[step_index]), "projector": int(intervened), "uavs": row})
        option.remaining_hold -= 1
        last = movement
    rng_after = _rng_token(scratch)
    summary = []
    for item in per_uav:
        summary.append(
            {
                "target": int(item["target"]),
                "entry": int(item["entry"]),
                "service_steps": int(item["service_steps"]),
                "delivered_bits": float(item["delivered_bits"]),
                "mean_sinr": float(np.mean(item["sinr"])) if item["sinr"] else 0.0,
                "mean_rate_bps": float(np.mean(item["rate"])) if item["rate"] else 0.0,
                "mean_signal_w": float(np.mean(item["signal_w"])) if item["signal_w"] else 0.0,
                "mean_interference_w": float(np.mean(item["interference_w"])) if item["interference_w"] else 0.0,
                "end_progress": float(item["progress"][-1]) if item["progress"] else 0.0,
            }
        )
    return {
        "delivered_bits": float(np.sum(delivered)),
        "step_delivered_bits": [float(item) for item in delivered],
        "per_uav": summary,
        "steps": steps,
        "rng_consumed": int(rng_before != rng_after),
        "remaining_hold_start": int(HOLD),
    }


def _decision_gap(env, oracle: np.ndarray, other: np.ndarray) -> dict:
    arrays = state_arrays(env)
    arrays["service_progress"] = np.asarray(env.progress, dtype=np.float64).copy()
    features = candidate_features(arrays, np.stack((oracle, other)), env.config)
    return {
        "distance_l1_m": float(np.sum(np.abs(features["distance_m"][0] - features["distance_m"][1]))),
        "hypothetical_rate_l1_bps": float(np.sum(np.abs(features["hypothetical_rate_bps"][0] - features["hypothetical_rate_bps"][1]))),
        "geometric_interference_l1_w": float(np.sum(np.abs(features["geometric_interference_w"][0] - features["geometric_interference_w"][1]))),
        "kinematic_bits_gap": float(features["kinematic_bits"][0] - features["kinematic_bits"][1]),
        "abandoned_progress_gap": float(np.sum(features["abandoned_progress"][0] - features["abandoned_progress"][1])),
    }


def run_audit() -> dict:
    if any(int(scenario) in RESERVED_TEST for scenario in VALIDATION_POOL):
        raise RuntimeError("validation pool overlaps the reserved test")
    limit_threads()
    actor = load_primary()
    before = parameter_token(actor)
    prior = {int(row["scenario_id"]): row for row in json.loads(PRIOR.read_text(encoding="utf-8"))}
    rows = []
    rng_consumed = 0
    for scenario in VALIDATION_POOL:
        env = _reach(int(scenario), actor)
        saved = _matches_saved(env, int(scenario))
        again = _reach(int(scenario), actor)
        replay_gap = float(np.max(np.abs(env.uav_positions - again.uav_positions)))
        source = prior[int(scenario)]
        assignments = {name: np.asarray(source["assignments"][name], dtype=np.int64) for name in ARMS}
        snap = capture_state(copy.deepcopy(env))
        scratch = copy.deepcopy(env)
        traces = {}
        repeats = {}
        for name in ARMS:
            traces[name] = _trace(scratch, actor, snap, assignments[name])
            repeats[name] = _execute(scratch, actor, snap, assignments[name], zero_arrivals=True)
            rng_consumed = max(rng_consumed, int(traces[name]["rng_consumed"]))
        label = {name: _label_of(int(scenario), assignments[name]) for name in ARMS}
        changed = [int(uav) for uav, (left, right) in enumerate(zip(assignments["oracle"], assignments["scheme_c"])) if int(left) != int(right)]
        oracle_by_buoy = {int(item["target"]): float(item["delivered_bits"]) for item in traces["oracle"]["per_uav"] if int(item["target"]) >= 0}
        scheme_by_buoy = {int(item["target"]): float(item["delivered_bits"]) for item in traces["scheme_c"]["per_uav"] if int(item["target"]) >= 0}
        buoy_gap = {str(buoy): float(oracle_by_buoy.get(buoy, 0.0) - scheme_by_buoy.get(buoy, 0.0)) for buoy in sorted(set(oracle_by_buoy) | set(scheme_by_buoy))}
        changed_share = float(sum(abs(value) for value in buoy_gap.values()) )
        row = {
            "scenario_id": int(scenario),
            "saved_match": int(saved["matched"]),
            "replay_position_gap_m": replay_gap,
            "assignments": {name: [int(item) for item in assignments[name]] for name in ARMS},
            "vector_hamming": {name: vector_hamming(assignments["oracle"], assignments[name]) for name in ARMS},
            "buoy_set_equal": {name: int(buoy_set(assignments["oracle"]) == buoy_set(assignments[name])) for name in ARMS},
            "legal": {name: int(is_legal(assignments[name], 6, 12)) for name in ARMS},
            "label_j15": label,
            "prior_rollout_bits": {name: float(source["causal"][name]["delivered_bits"]) for name in ARMS},
            "prior_completed_blocks": {name: float(source["causal"][name]["completed_blocks"]) for name in ARMS},
            "replay_bits": {name: float(traces[name]["delivered_bits"]) for name in ARMS},
            "repeat_bits": {name: float(repeats[name]["delivered_bits"]) for name in ARMS},
            "label_abs_error": {name: abs(float(traces[name]["delivered_bits"]) - float(label[name])) for name in ARMS},
            "prior_abs_error": {name: abs(float(traces[name]["delivered_bits"]) - float(source["causal"][name]["delivered_bits"])) for name in ARMS},
            "repeat_abs_error": {name: abs(float(traces[name]["delivered_bits"]) - float(repeats[name]["delivered_bits"])) for name in ARMS},
            "delta_label": float(label["oracle"] - label["scheme_c"]),
            "delta_replay": float(traces["oracle"]["delivered_bits"] - traces["scheme_c"]["delivered_bits"]),
            "delta_epoch6": float(traces["oracle"]["delivered_bits"] - traces["epoch6"]["delivered_bits"]),
            "changed_uavs": changed,
            "buoy_bit_gap": buoy_gap,
            "buoy_abs_gap_sum": changed_share,
            "per_uav": {name: traces[name]["per_uav"] for name in ("oracle", "scheme_c")},
            "step_delivered_bits": {name: traces[name]["step_delivered_bits"] for name in ("oracle", "scheme_c")},
            "decision": _decision_gap(env, assignments["oracle"], assignments["scheme_c"]),
            "rng_consumed": int(traces["oracle"]["rng_consumed"]),
        }
        if int(scenario) in TRACE_SCENARIOS:
            row["steps"] = {name: traces[name]["steps"] for name in ("oracle", "scheme_c")}
        rows.append(row)
        print(f"close {scenario} ham={row['vector_hamming']['scheme_c']} delta={row['delta_replay']:.3e} label_error={row['label_abs_error']['oracle']:.3e}", flush=True)
    if parameter_token(actor) != before:
        raise RuntimeError("MAPPO weights changed")
    consistent = all(
        int(row["saved_match"]) == 1
        and all(int(row["legal"][name]) == 1 for name in ARMS)
        and all(float(row["label_abs_error"][name]) <= 1.0 for name in ARMS)
        and all(float(row["prior_abs_error"][name]) <= 1.0 for name in ARMS)
        and all(int(row["vector_hamming"][name]) > 0 for name in ("scheme_c", "epoch6"))
        and all(int(row["buoy_set_equal"][name]) == 1 for name in ARMS)
        for row in rows
    )
    reproduced = all(float(row["replay_position_gap_m"]) <= 1.0e-6 and all(float(row["repeat_abs_error"][name]) <= 1.0 for name in ARMS) for row in rows)
    summary = {
        "gate": assign_gate(consistent, reproduced, bool(rng_consumed)),
        "states": len(rows),
        "rng_consumed": int(rng_consumed),
        "vector_hamming_scheme_median": float(np.median([row["vector_hamming"]["scheme_c"] for row in rows])),
        "vector_hamming_epoch6_median": float(np.median([row["vector_hamming"]["epoch6"] for row in rows])),
        "self_hamming_max": int(np.max([row["vector_hamming"]["oracle"] for row in rows])),
        "label_abs_error_max": float(np.max([max(row["label_abs_error"].values()) for row in rows])),
        "prior_abs_error_max": float(np.max([max(row["prior_abs_error"].values()) for row in rows])),
        "repeat_abs_error_max": float(np.max([max(row["repeat_abs_error"].values()) for row in rows])),
        "replay_position_gap_max_m": float(np.max([row["replay_position_gap_m"] for row in rows])),
        "delta_replay_mean": float(np.mean([row["delta_replay"] for row in rows])),
        "delta_replay_median": float(np.median([row["delta_replay"] for row in rows])),
        "buoy_abs_gap_sum_median": float(np.median([row["buoy_abs_gap_sum"] for row in rows])),
        "decision_distance_l1_median_m": float(np.median([row["decision"]["distance_l1_m"] for row in rows])),
        "decision_rate_l1_median_bps": float(np.median([row["decision"]["hypothetical_rate_l1_bps"] for row in rows])),
        "decision_kinematic_gap_median": float(np.median([row["decision"]["kinematic_bits_gap"] for row in rows])),
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / "states.json").write_text(json.dumps(rows), encoding="utf-8")
    (AUDIT_DIR / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps({"gate": run_audit()["gate"]}), flush=True)
