"""Paired frozen-MAPPO execution of same-buoy assignments. No training."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6d.shadow import capture_state, restore_state
from wt.benchmark_v2.phase6e.generators import feature_matrices, is_legal
from wt.benchmark_v2.phase6e.value import _zero_arrivals
from wt.benchmark_v2.phase6g_lowlevel_mappo.evaluate import _actor_move
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import make_option
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import limit_threads, load_primary, parameter_token
from wt.benchmark_v2.phase6h_executor_conditioned_macro.jobs import _live_step
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import HOLD
from wt.benchmark_v2.phase6ib_highlevel_dataset.hgnn_revision import REVISION_DATASET, _choose, _open_env
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST, VALIDATION_POOL
from wt.l3a.env import conflict_mask
from wt.l3b.env import interference_matrix


ROOT = Path(__file__).resolve().parents[4]
AUDIT_DIR = ROOT / "results/phase6ij_executor_audit"
PRIOR = ROOT / "results/phase6ii_candidate_features" / "states.json"
ARMS = ("oracle", "scheme_c", "epoch6")
DRAWS = 10000
SEED = 0
BIT_TOLERANCE = 1.0


def buoy_set(assignment: np.ndarray) -> tuple[int, ...]:
    return tuple(sorted(int(item) for item in np.asarray(assignment, dtype=np.int64).reshape(-1) if int(item) >= 0))


def pair_ok(left: np.ndarray, right: np.ndarray) -> bool:
    rows = [np.asarray(item, dtype=np.int64).reshape(-1) for item in (left, right)]
    return bool(
        all(is_legal(row, 6, 12) for row in rows)
        and buoy_set(rows[0]) == buoy_set(rows[1])
        and int(np.sum(rows[0] != rows[1])) > 0
    )


def assign_gate(rows: list[dict], blocked: bool) -> str:
    if blocked or any(int(row["failure"]) for row in rows):
        return "EXECUTION_AUDIT_BLOCKED"
    paired = [row for row in rows if int(row["paired"]) == 1]
    if len(paired) < 12:
        return "EXECUTION_GAIN_INCONCLUSIVE"
    delta = np.asarray([float(row["delta_causal"]) for row in paired], dtype=np.float64)
    low, _high = _bootstrap_ci(delta)
    positive = float(np.mean(delta > BIT_TOLERANCE))
    if low > 0.0 and float(np.median(delta)) > BIT_TOLERANCE and positive >= 0.5:
        return "EXECUTION_GAIN_SUPPORTED"
    return "EXECUTION_GAIN_INCONCLUSIVE"


def _bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    draws = np.empty(DRAWS, dtype=np.float64)
    for index in range(DRAWS):
        draws[index] = float(np.mean(data[rng.integers(0, data.size, data.size)]))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 3 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    rx = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort").astype(np.float64)
    ry = np.argsort(np.argsort(y, kind="mergesort"), kind="mergesort").astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def _reach(scenario: int, actor) -> L3BBitQueueEnvV2:
    env = _open_env(int(scenario))
    option = None
    pending = 0
    while int(env.t) < 15:
        if pending <= 0:
            option = _choose(env)
            pending = int(HOLD)
        _live_step(env, actor, option, use_fly=False)
        pending -= 1
    if int(env.t) != 15 or option is None:
        raise RuntimeError("replay did not stop at t=15")
    return env


def _matches_saved(env: L3BBitQueueEnvV2, scenario: int) -> dict:
    path = REVISION_DATASET / "validation" / f"scenario_{int(scenario)}" / "t015_state.npz"
    with np.load(path) as payload:
        position = np.asarray(payload["uav_position_m"], dtype=np.float64)
        queue = np.asarray(payload["q_bits"], dtype=np.float64)
        association = np.asarray(payload["association"], dtype=np.int64)
        progress = np.asarray(payload["service_progress"], dtype=np.float64)
    gaps = {
        "position_m": float(np.max(np.abs(env.uav_positions - position))),
        "queue_bits": float(np.max(np.abs(env.q_bits - queue))),
        "progress": float(np.max(np.abs(env.progress - progress))),
        "association": int(np.array_equal(env.assignment, association)),
    }
    gaps["matched"] = int(gaps["position_m"] <= 1.0e-6 and gaps["queue_bits"] <= 1.0e-4 and gaps["progress"] <= 1.0e-6 and gaps["association"] == 1)
    return gaps


def _bind_clock(scratch: L3BBitQueueEnvV2) -> None:
    scratch.arrivals_at = lambda time_step, env=scratch: L3BBitQueueEnvV2.arrivals_at(env, time_step)
    scratch.arrivals = lambda env=scratch: L3BBitQueueEnvV2.arrivals(env)


def _execute(scratch: L3BBitQueueEnvV2, actor, snap: dict, assignment: np.ndarray, *, zero_arrivals: bool) -> dict:
    restore_state(scratch, snap)
    if zero_arrivals:
        _zero_arrivals(scratch)
    else:
        _bind_clock(scratch)
    chosen = np.asarray(assignment, dtype=np.int64).reshape(-1).copy()
    option = make_option(scratch, chosen)
    last = np.zeros((scratch.config.num_uavs, 2), dtype=np.float64)
    delivered = 0.0
    blocks = 0.0
    series = []
    first_entry = np.full(chosen.shape[0], int(HOLD), dtype=np.int64)
    service_steps = 0
    sinr_values = []
    interference_values = []
    interventions = 0
    conflicts = 0
    stays = 0
    switches = int(np.sum(chosen != np.asarray(scratch.served_k, dtype=np.int64)))
    worst = 0.0
    for step_index in range(int(HOLD)):
        movement, intervened, _residual = _actor_move(scratch, actor, option, last)
        info = scratch.step(movement, option.association)
        if not np.array_equal(np.asarray(scratch.assignment, dtype=np.int64), chosen):
            raise RuntimeError("assignment changed during the hold")
        distance = np.linalg.norm(scratch.uav_positions - scratch.buoy_positions[np.where(chosen >= 0, chosen, 0)], axis=-1)
        distance = np.where(chosen >= 0, distance, np.nan)
        entered = np.isfinite(distance) & (distance <= float(scratch.config.service_radius_m))
        first_entry = np.where((first_entry == int(HOLD)) & entered, step_index, first_entry)
        active = np.asarray(scratch.last_rate, dtype=np.float64) > 0.0
        service_steps += int(np.sum(active))
        if np.any(active):
            sinr_values.append(float(np.mean(np.asarray(scratch.last_sinr, dtype=np.float64)[active])))
            interference_values.append(float(np.sum(interference_matrix(scratch))))
        increment = float(info["payload_increment_bits"])
        delivered += increment
        blocks += float(np.sum(info["completed_block_equivalent_increment"]))
        interventions += int(intervened)
        conflicts += int(np.sum(conflict_mask(scratch.assignment, scratch.config.num_buoys)))
        stays += int(float(np.sum(np.linalg.norm(movement, axis=-1))) * float(scratch.config.step_length_m) < 1.0)
        worst = max(worst, float(np.max(np.abs(scratch.conservation_error_bits()))))
        series.append(
            {
                "delivered_bits": increment,
                "mean_distance_m": float(np.nanmean(distance)),
                "in_range": int(np.sum(entered)),
                "mean_sinr": float(np.mean(np.asarray(scratch.last_sinr, dtype=np.float64)[active])) if np.any(active) else 0.0,
                "service_progress_sum": float(np.sum(scratch.progress)),
            }
        )
        option.remaining_hold -= 1
        last = movement
    finite = int(np.isfinite(delivered) and np.all(np.isfinite(scratch.q_bits)) and np.all(np.isfinite(scratch.progress)))
    return {
        "delivered_bits": float(delivered),
        "completed_blocks": float(blocks),
        "first_entry_median": float(np.median(first_entry)),
        "service_steps": int(service_steps),
        "mean_sinr": float(np.mean(sinr_values)) if sinr_values else 0.0,
        "mean_interference_w": float(np.mean(interference_values)) if interference_values else 0.0,
        "projector_steps": int(interventions),
        "conflict_steps": int(conflicts),
        "stay_steps": int(stays),
        "switches": int(switches),
        "end_queue_bits": float(np.sum(scratch.q_bits)),
        "end_progress": float(np.sum(scratch.progress)),
        "conservation_error": float(worst),
        "finite": finite,
        "series": series,
    }


def _predicted_rate(env: L3BBitQueueEnvV2, assignment: np.ndarray) -> float:
    rate = feature_matrices(env)["rate"]
    total = 0.0
    for uav, buoy in enumerate(np.asarray(assignment, dtype=np.int64).reshape(-1)):
        if int(buoy) >= 0:
            total += float(rate[uav, int(buoy)])
    return float(total)


def _load_prior() -> list[dict]:
    rows = [row for row in json.loads(PRIOR.read_text(encoding="utf-8")) if int(row["time_step"]) == 15]
    if any(int(row["scenario_id"]) in RESERVED_TEST for row in rows):
        raise RuntimeError("prior rows include the reserved test")
    return rows


def run_audit() -> dict:
    if any(int(scenario) in RESERVED_TEST for scenario in VALIDATION_POOL):
        raise RuntimeError("validation pool overlaps the reserved test")
    limit_threads()
    actor = load_primary()
    before = parameter_token(actor)
    prior = {int(row["scenario_id"]): row for row in _load_prior()}
    rows = []
    for scenario in VALIDATION_POOL:
        env = _reach(int(scenario), actor)
        saved = _matches_saved(env, int(scenario))
        source = prior[int(scenario)]
        labels = {name: float(source["j15"][name]) for name in ARMS}
        assignments = {name: np.asarray(source["assignments"][name], dtype=np.int64) for name in ARMS}
        legal = {name: int(is_legal(assignments[name], 6, 12)) for name in ARMS}
        same = {name: int(buoy_set(assignments["oracle"]) == buoy_set(assignments[name])) for name in ARMS}
        predicted = {name: _predicted_rate(env, assignments[name]) for name in ARMS}
        decision_position = np.asarray(env.uav_positions, dtype=np.float64).copy()
        scratch = copy.deepcopy(env)
        snap = capture_state(scratch)
        causal = {}
        clock = {}
        for name in ARMS:
            causal[name] = _execute(scratch, actor, snap, assignments[name], zero_arrivals=True)
            clock[name] = _execute(scratch, actor, snap, assignments[name], zero_arrivals=False)
        repeat = _execute(scratch, actor, snap, assignments["oracle"], zero_arrivals=True)
        restore_state(scratch, snap)
        parent_gap = float(np.max(np.abs(scratch.uav_positions - decision_position)))
        failure = int(
            saved["matched"] != 1
            or any(value != 1 for value in legal.values())
            or abs(float(repeat["delivered_bits"]) - float(causal["oracle"]["delivered_bits"])) > BIT_TOLERANCE
            or parent_gap > 1.0e-6
            or any(int(item["finite"]) != 1 or float(item["conservation_error"]) > float(env.tol_bits) for item in list(causal.values()) + list(clock.values()))
        )
        paired = int(pair_ok(assignments["oracle"], assignments["scheme_c"]))
        row = {
            "scenario_id": int(scenario),
            "saved": saved,
            "legal": legal,
            "same_buoy_set": same,
            "hamming": {name: int(np.sum(assignments["oracle"] != assignments[name])) for name in ARMS},
            "assignments": {name: [int(item) for item in assignments[name]] for name in ARMS},
            "stored_j15": labels,
            "predicted_rate_bps": predicted,
            "causal": causal,
            "clock": clock,
            "repeat_abs_error": abs(float(repeat["delivered_bits"]) - float(causal["oracle"]["delivered_bits"])),
            "parent_position_gap_m": parent_gap,
            "j15_abs_error": {name: abs(float(causal[name]["delivered_bits"]) - float(labels[name])) for name in ARMS},
            "delta_causal": float(causal["oracle"]["delivered_bits"] - causal["scheme_c"]["delivered_bits"]),
            "delta_clock": float(clock["oracle"]["delivered_bits"] - clock["scheme_c"]["delivered_bits"]),
            "delta_epoch6": float(causal["oracle"]["delivered_bits"] - causal["epoch6"]["delivered_bits"]),
            "paired": paired,
            "failure": failure,
        }
        rows.append(row)
        print(
            f"execute {scenario} paired={paired} delta={row['delta_causal']:.3e} j15_error={row['j15_abs_error']['oracle']:.3e}",
            flush=True,
        )
    if parameter_token(actor) != before:
        raise RuntimeError("MAPPO weights changed during the audit")
    paired_rows = [row for row in rows if int(row["paired"]) == 1]
    delta = np.asarray([row["delta_causal"] for row in paired_rows], dtype=np.float64)
    low, high = _bootstrap_ci(delta) if delta.size else (0.0, 0.0)
    mechanism = _mechanism(paired_rows)
    summary = {
        "gate": assign_gate(rows, False),
        "states": len(rows),
        "paired_states": len(paired_rows),
        "failure_count": int(np.sum([row["failure"] for row in rows])),
        "j15_abs_error_max": float(np.max([max(row["j15_abs_error"].values()) for row in rows])),
        "repeat_abs_error_max": float(np.max([row["repeat_abs_error"] for row in rows])),
        "saved_match_count": int(np.sum([row["saved"]["matched"] for row in rows])),
        "delta_causal": _delta_summary(delta, low, high),
        "delta_clock": _delta_summary(np.asarray([row["delta_clock"] for row in paired_rows], dtype=np.float64), *_bootstrap_ci(np.asarray([row["delta_clock"] for row in paired_rows], dtype=np.float64))),
        "delta_epoch6": _delta_summary(np.asarray([row["delta_epoch6"] for row in paired_rows], dtype=np.float64), *_bootstrap_ci(np.asarray([row["delta_epoch6"] for row in paired_rows], dtype=np.float64))),
        "mechanism": mechanism,
    }
    if int(summary["failure_count"]) > 0:
        summary["gate"] = "EXECUTION_AUDIT_BLOCKED"
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / "states.json").write_text(json.dumps(rows), encoding="utf-8")
    (AUDIT_DIR / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return summary


def _delta_summary(values: np.ndarray, low: float, high: float) -> dict:
    data = np.asarray(values, dtype=np.float64)
    return {
        "n": int(data.size),
        "mean": float(np.mean(data)) if data.size else 0.0,
        "median": float(np.median(data)) if data.size else 0.0,
        "positive_fraction": float(np.mean(data > BIT_TOLERANCE)) if data.size else 0.0,
        "negative_fraction": float(np.mean(data < -BIT_TOLERANCE)) if data.size else 0.0,
        "mean_ci95": [float(low), float(high)],
    }


def _mechanism(rows: list[dict]) -> dict:
    if not rows:
        return {}
    delta = np.asarray([row["delta_causal"] for row in rows], dtype=np.float64)
    fields = {
        "first_entry_median": lambda row: float(row["causal"]["oracle"]["first_entry_median"] - row["causal"]["scheme_c"]["first_entry_median"]),
        "service_steps": lambda row: float(row["causal"]["oracle"]["service_steps"] - row["causal"]["scheme_c"]["service_steps"]),
        "mean_sinr": lambda row: float(row["causal"]["oracle"]["mean_sinr"] - row["causal"]["scheme_c"]["mean_sinr"]),
        "mean_interference_w": lambda row: float(row["causal"]["oracle"]["mean_interference_w"] - row["causal"]["scheme_c"]["mean_interference_w"]),
        "end_progress": lambda row: float(row["causal"]["oracle"]["end_progress"] - row["causal"]["scheme_c"]["end_progress"]),
    }
    scored = {name: {"spearman": _spearman(np.asarray([fn(row) for row in rows]), delta), "median_gap": float(np.median([fn(row) for row in rows]))} for name, fn in fields.items()}
    return scored


if __name__ == "__main__":
    print(json.dumps({"gate": run_audit()["gate"]}), flush=True)
