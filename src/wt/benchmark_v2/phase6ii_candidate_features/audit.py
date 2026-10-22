"""Offline observability check for candidate-conditioned features. No training."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6c.channel import link_rates
from wt.benchmark_v2.phase6ib_highlevel_dataset.hgnn_revision import REVISION_DATASET
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST, VALIDATION_POOL
from wt.benchmark_v2.phase6ii_candidate_features.features import candidate_features, feature_sums


ROOT = Path(__file__).resolve().parents[4]
AUDIT_DIR = ROOT / "results/phase6ii_candidate_features"
PRIOR_TOP1 = ROOT / "results/phase6ig_candidate_audit" / "states.json"
T15_TOP1 = ROOT / "results/phase6ih_t15_audit" / "states.json"
TIMES = (0, 15, 30)
HIGHER_IS_BETTER = {
    "in_range": True,
    "hypothetical_rate_bps": True,
    "noise_only_rate_bps": True,
    "instant_rate_bps": True,
    "geometric_rate_bps": True,
    "straight_line_bits": True,
    "kinematic_bits": True,
    "distance_m": False,
    "steps_to_radius": False,
    "switch": False,
    "abandoned_progress": False,
    "instant_interference_w": False,
    "geometric_interference_w": False,
    "hover_interference_w": False,
}
FLOORS = {
    "distance_m": 1.0,
    "steps_to_radius": 1.0 / 22.0,
    "in_range": 0.5,
    "hypothetical_rate_bps": 1.0e4,
    "noise_only_rate_bps": 1.0e4,
    "switch": 0.5,
    "abandoned_progress": 0.05,
    "instant_rate_bps": 1.0e4,
    "instant_interference_w": 1.0e-6,
    "geometric_rate_bps": 1.0e4,
    "geometric_interference_w": 1.0e-6,
    "straight_line_bits": 1.0e5,
    "kinematic_bits": 1.0e5,
    "hover_interference_w": 1.0e-6,
}
PROPOSED = (
    "noise_only_rate_bps",
    "instant_rate_bps",
    "instant_interference_w",
    "geometric_rate_bps",
    "geometric_interference_w",
    "straight_line_bits",
    "kinematic_bits",
    "abandoned_progress",
    "hover_interference_w",
)
RELATIVE = 0.05


def separated(left: float, right: float, name: str) -> bool:
    gap = abs(float(left) - float(right))
    scale = max(abs(float(left)), abs(float(right)))
    return bool(gap > max(float(FLOORS[name]), RELATIVE * scale))


def favors_oracle(oracle: float, other: float, name: str) -> bool:
    if HIGHER_IS_BETTER[name]:
        return float(oracle) > float(other)
    return float(oracle) < float(other)


def assign_gate(t15: list[dict], controls: dict[int, list[dict]]) -> str:
    """One pre-registered rule. J15 is used only to name the oracle row."""
    comparable = [row for row in t15 if int(row["hamming"]["scheme_c"]) > 0 and float(row["j15_gap"]["scheme_c"]) > 1.0]
    if len(comparable) < 12:
        return "CANDIDATE_FEATURE_DESIGN_INCONCLUSIVE"
    for name in PROPOSED:
        separated_rows = [row for row in comparable if int(row["separated"][name]) == 1]
        if len(separated_rows) < 12:
            continue
        agreement = float(np.mean([int(row["favors_oracle"][name]) for row in separated_rows]))
        t15_range = float(np.median([float(row["same_set_relative_range"][name]) for row in t15]))
        control_ok = False
        for rows in controls.values():
            differing = [row for row in rows if int(row["hamming"]["scheme_c"]) > 0 and float(row["j15_gap"]["scheme_c"]) > 1.0]
            control_range = float(np.median([float(row["same_set_relative_range"][name]) for row in rows])) if rows else 0.0
            if len(differing) >= 4:
                rate = float(np.mean([int(row["separated"][name]) for row in differing]))
                control_ok = control_ok or (rate >= 0.5 and control_range > RELATIVE)
            elif control_range > RELATIVE:
                control_ok = True
        if agreement >= 0.75 and t15_range > RELATIVE and control_ok:
            return "CANDIDATE_FEATURE_DESIGN_READY"
    any_signal = any(int(row["separated"][name]) == 1 for row in comparable for name in PROPOSED)
    if not any_signal:
        return "CANDIDATE_FEATURE_DESIGN_BLOCKED"
    return "CANDIDATE_FEATURE_DESIGN_INCONCLUSIVE"


def _load_prior_top1() -> dict[tuple[int, int], dict[str, int]]:
    rows = json.loads(PRIOR_TOP1.read_text(encoding="utf-8"))
    found = {}
    for row in rows:
        scenario = int(row["scenario_id"])
        if scenario in RESERVED_TEST:
            raise RuntimeError("prior audit index points at the reserved test")
        found[(scenario, int(row["time_step"]))] = {
            "frozen": int(row["frozen"]["chosen"]),
            "epoch6": int(row["epoch6"]["chosen"]),
            "scheme_c": int(row["scheme_c"]["chosen"]),
        }
    return found


def _check_t15_indices(prior: dict[tuple[int, int], dict[str, int]]) -> None:
    rows = json.loads(T15_TOP1.read_text(encoding="utf-8"))
    for row in rows:
        key = (int(row["scenario_id"]), 15)
        for name in ("frozen", "epoch6", "scheme_c"):
            if int(prior[key][name]) != int(row["models"][name]["top1"]):
                raise RuntimeError(f"top-1 index mismatch at {key} {name}")


def _buoy_key(row: np.ndarray) -> tuple[int, ...]:
    return tuple(sorted(int(item) for item in row if int(item) >= 0))


def _same_set_range(sums: dict[str, np.ndarray], assignment: np.ndarray, oracle_index: int) -> dict[str, float]:
    target = _buoy_key(assignment[oracle_index])
    mask = np.asarray([_buoy_key(row) == target for row in assignment], dtype=bool)
    ranges = {}
    for name, values in sums.items():
        chosen = np.asarray(values, dtype=np.float64)[mask]
        span = float(np.max(chosen) - np.min(chosen)) if chosen.size else 0.0
        scale = max(float(np.max(np.abs(chosen))) if chosen.size else 0.0, 1.0e-12)
        ranges[name] = span / scale
    ranges["count"] = float(np.sum(mask))
    return ranges


def _pair_record(features: dict[str, np.ndarray], sums: dict[str, np.ndarray], oracle: int, other: int) -> dict:
    per_uav = {}
    separated_flags = {}
    favors = {}
    for name in HIGHER_IS_BETTER:
        array = np.asarray(features[name])
        gap = float(np.sum(np.abs(array[oracle] - array[other]))) if array.ndim == 2 else abs(float(sums[name][oracle]) - float(sums[name][other]))
        per_uav[name] = gap
        scale = max(abs(float(sums[name][oracle])), abs(float(sums[name][other])), gap)
        separated_flags[name] = int(gap > max(float(FLOORS[name]), RELATIVE * scale))
        favors[name] = int(favors_oracle(float(sums[name][oracle]), float(sums[name][other]), name))
    return {"sum_gap_l1": per_uav, "separated": separated_flags, "favors_oracle": favors, "oracle_sum": {name: float(sums[name][oracle]) for name in HIGHER_IS_BETTER}, "other_sum": {name: float(sums[name][other]) for name in HIGHER_IS_BETTER}}


def _state_path(scenario: int, time_step: int) -> tuple[Path, Path]:
    folder = REVISION_DATASET / "validation" / f"scenario_{int(scenario)}"
    stem = f"t{int(time_step):03d}"
    return folder / f"{stem}_state.npz", folder / f"{stem}_candidates.npz"


def _load_state(scenario: int, time_step: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    state_path, candidate_path = _state_path(scenario, time_step)
    with np.load(state_path) as state_file:
        arrays = {key: np.asarray(state_file[key]) for key in state_file.files}
    with np.load(candidate_path) as candidate_file:
        assignment = np.asarray(candidate_file["assignment"], dtype=np.int64)
        labels = np.asarray(candidate_file["j15_bits"], dtype=np.float64)
    return arrays, assignment, labels


def _matches_link_rates(arrays: dict[str, np.ndarray], assignment: np.ndarray, features: dict[str, np.ndarray], config) -> bool:
    _sinr, rate, _interference = link_rates(
        np.asarray(arrays["uav_buoy_gain"], dtype=np.float64),
        assignment,
        np.asarray(arrays["q_bits"], dtype=np.float64),
        np.asarray(arrays["uav_buoy_in_range"], dtype=np.uint8) > 0,
        bandwidth_hz=float(config.bandwidth_hz),
        tx_power=float(config.tx_power),
        noise_power=float(config.noise_power),
        kappa_same=float(config.kappa_same_cluster),
        kappa_cross=float(config.kappa_cross_cluster),
    )
    return bool(np.allclose(rate, features["instant_rate_bps"][0], rtol=1.0e-8, atol=1.0e-6))


def run_audit() -> dict:
    if any(int(scenario) in RESERVED_TEST for scenario in VALIDATION_POOL):
        raise RuntimeError("validation pool overlaps the reserved test")
    prior = _load_prior_top1()
    _check_t15_indices(prior)
    config = L3BBitQueueEnvV2().config
    rows = []
    timing = []
    for time_step in TIMES:
        for scenario in VALIDATION_POOL:
            arrays, assignment, labels = _load_state(int(scenario), int(time_step))
            oracle = int(np.flatnonzero(labels == float(np.max(labels)))[0])
            chosen = prior[(int(scenario), int(time_step))]
            indices = {"oracle": oracle, "frozen": int(chosen["frozen"]), "epoch6": int(chosen["epoch6"]), "scheme_c": int(chosen["scheme_c"])}
            if any(index < 0 or index >= assignment.shape[0] for index in indices.values()):
                raise RuntimeError("a top-1 index is outside the candidate file")
            started = time.perf_counter()
            features = candidate_features(arrays, assignment, config)
            elapsed = time.perf_counter() - started
            if not _matches_link_rates(arrays, assignment[oracle], {key: value[oracle: oracle + 1] for key, value in features.items()}, config):
                raise RuntimeError(f"instant rate disagrees with link_rates at {scenario} t={time_step}")
            sums = feature_sums(features)
            order = np.arange(assignment.shape[0])
            reversed_features = candidate_features(arrays, assignment[::-1], config)
            reversed_sums = feature_sums(reversed_features)
            if not np.allclose(reversed_sums["straight_line_bits"], sums["straight_line_bits"][::-1]) or not np.allclose(reversed_sums["kinematic_bits"], sums["kinematic_bits"][::-1]):
                raise RuntimeError("candidate order changed a feature value")
            same_set = _same_set_range(sums, assignment, oracle)
            scheme = _pair_record(features, sums, oracle, indices["scheme_c"])
            record = {
                "scenario_id": int(scenario),
                "time_step": int(time_step),
                "oracle_index": oracle,
                "indices": indices,
                "assignments": {name: [int(item) for item in assignment[index]] for name, index in indices.items()},
                "j15": {name: float(labels[index]) for name, index in indices.items()},
                "j15_gap": {name: float(labels[oracle] - labels[indices[name]]) for name in ("frozen", "epoch6", "scheme_c")},
                "hamming": {name: int(np.sum(assignment[oracle] != assignment[indices[name]])) for name in ("frozen", "epoch6", "scheme_c")},
                "same_buoy_set": {name: int(_buoy_key(assignment[oracle]) == _buoy_key(assignment[indices[name]])) for name in ("frozen", "epoch6", "scheme_c")},
                "same_set_count": int(same_set["count"]),
                "same_set_relative_range": {name: float(same_set[name]) for name in HIGHER_IS_BETTER},
                "separated": scheme["separated"],
                "favors_oracle": scheme["favors_oracle"],
                "sum_gap_l1": scheme["sum_gap_l1"],
                "oracle_sum": scheme["oracle_sum"],
                "scheme_sum": scheme["other_sum"],
                "elapsed_sec": elapsed,
                "feature_bytes": int(sum(np.asarray(value).nbytes for value in features.values())),
                "candidate_count": int(assignment.shape[0]),
                "unchanged_order_probe": int(order[0] == 0),
            }
            for name in ("frozen", "epoch6"):
                other = _pair_record(features, sums, oracle, indices[name])
                record[f"separated_{name}"] = other["separated"]
                record[f"sum_{name}"] = other["other_sum"]
                record[f"gap_{name}"] = other["sum_gap_l1"]
            rows.append(record)
            timing.append(elapsed)
            print(f"features {scenario} t={time_step} hamming={record['hamming']['scheme_c']} straight={record['sum_gap_l1']['straight_line_bits']:.3e}", flush=True)
    by_time = {int(time_step): [row for row in rows if int(row["time_step"]) == int(time_step)] for time_step in TIMES}
    gate = assign_gate(by_time[15], {0: by_time[0], 30: by_time[30]})
    summary = {
        "gate": gate,
        "states": len(rows),
        "elapsed_sec_median": float(np.median(timing)),
        "elapsed_sec_max": float(np.max(timing)),
        "feature_bytes": int(rows[0]["feature_bytes"]),
        "candidate_count": int(rows[0]["candidate_count"]),
        "pairwise_matrix": 0,
        "t15": _aggregate(by_time[15]),
        "t0": _aggregate(by_time[0]),
        "t30": _aggregate(by_time[30]),
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / "states.json").write_text(json.dumps(rows), encoding="utf-8")
    (AUDIT_DIR / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return summary


def _aggregate(rows: list[dict]) -> dict:
    comparable = [row for row in rows if int(row["hamming"]["scheme_c"]) > 0 and float(row["j15_gap"]["scheme_c"]) > 1.0]
    payload = {"states": len(rows), "comparable_scheme": len(comparable), "same_buoy_set_scheme": int(np.sum([row["same_buoy_set"]["scheme_c"] for row in rows]))}
    for name in HIGHER_IS_BETTER:
        payload[name] = {
            "median_l1": float(np.median([row["sum_gap_l1"][name] for row in rows])),
            "median_oracle_sum": float(np.median([row["oracle_sum"][name] for row in rows])),
            "median_scheme_sum": float(np.median([row["scheme_sum"][name] for row in rows])),
            "separated_comparable": int(np.sum([row["separated"][name] for row in comparable])),
            "agreement_when_separated": float(np.mean([row["favors_oracle"][name] for row in comparable if int(row["separated"][name]) == 1])) if any(int(row["separated"][name]) == 1 for row in comparable) else 0.0,
            "same_set_relative_range_median": float(np.median([row["same_set_relative_range"][name] for row in rows])),
        }
    return payload


if __name__ == "__main__":
    print(json.dumps({"gate": run_audit()["gate"]}), flush=True)
