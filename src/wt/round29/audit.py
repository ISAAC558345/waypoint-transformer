"""Diagnostic-only Round-29 cross-scale deferred-service pressure audit.

No model is fitted here.  The only new environment interactions are the one
pre-registered, independent confirmation diagnostic and its same-snapshot H5
versus H1 counterfactual branches.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from wt.maritime.config import MaritimeEnvConfig
from wt.round19.multi_env import SharedSpectrumMultiUAVEnv, SharedSpectrumOptions, assignment_link_metrics, gain_matrix, joint_aware_matching
from wt.round19.planner import joint_plan
from wt.governance_state_sync import approved_lifecycle_overlay_sha256
from wt.round28.audit import _direct_rates, _exclusive_score


ROOT = Path(__file__).resolve().parents[3]
ROUND = ROOT / "round_29"
OUTPUT = ROUND / "outputs" / "round29_cross_scale_dsp_20260830_1745"
FIGURES = ROUND / "figures" / "round29_cross_scale_dsp_20260830_1745"
CONFIG_PATH = ROUND / "configs" / "round29_cross_scale_dsp_config.json"
R28 = ROOT / "round_28"
R28_OUTPUT = R28 / "outputs" / "round28_planning_need_representation_20260830_1624"
EXPECTED_R28 = "9E3F10B893891B6E0E0415D886E00AEC234CD1376B3146E5704093A210384C91"
QUEUE_CAPACITY_BITS = float(MaritimeEnvConfig().queue_capacity_bits)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise RuntimeError(f"unexpected empty Round-29 output: {path.name}")
    fields = sorted({key for row in values for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(values)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if value in {"True", "False"}:
                    row[key] = value == "True"
                else:
                    try:
                        row[key] = float(value)
                    except (TypeError, ValueError):
                        row[key] = value
            values.append(row)
    return values


def _tree(path: Path) -> dict[str, str]:
    return {
        item.relative_to(path).as_posix(): sha256(item)
        for item in sorted(path.rglob("*")) if item.is_file() and "__pycache__" not in item.parts
    }


def verify_frozen_state() -> dict[str, Any]:
    """Verify every registered Round-28 evidence file without executing it."""
    lock_path = R28 / "ROUND_LOCK.json"
    if sha256(lock_path) != EXPECTED_R28:
        raise RuntimeError("Round-28 original lock changed")
    lock = _load(lock_path)
    for relative, digest in lock["configuration_hashes"].items():
        if sha256(R28 / relative) != digest:
            raise RuntimeError(f"Round-28 configuration changed: {relative}")
    for relative, digest in lock["source_tree"].items():
        if sha256(ROOT / "src" / "wt" / "round28" / relative) != digest:
            raise RuntimeError(f"Round-28 source changed: {relative}")
    amendment = _load(R28 / "ROUND_LOCK_AMENDMENT.json")
    if amendment.get("amendment_type") != "legal_round29_successor_governance_extension" or amendment.get("original_round28_lock_sha256") != EXPECTED_R28:
        raise RuntimeError("Round-28 legal successor amendment is invalid")
    allowed = {
        "src/wt/governance_state_sync.py": ROOT / "src" / "wt" / "governance_state_sync.py",
        "tests/test_governance_state_sync.py": ROOT / "tests" / "test_governance_state_sync.py",
        "tests/test_round28_planning_need_representation.py": ROOT / "tests" / "test_round28_planning_need_representation.py",
    }
    affected = amendment.get("affected_paths", {})
    if any(sha256(path) != approved_lifecycle_overlay_sha256(name, affected.get(name, {}).get("amended_sha256", "")) for name, path in allowed.items()):
        raise RuntimeError("Round-28 successor amendment hashes are invalid")
    prohibited = ("scientific_artifacts_modified", "model_checkpoints_modified", "dataset_modified", "metrics_modified", "fresh_dev_rerun", "untouched_eval_accessed", "original_round28_lock_modified")
    if any(amendment.get(name) is not False for name in prohibited):
        raise RuntimeError("Round-28 successor amendment changes protected evidence")
    for relative, digest in lock["output_hashes"].items():
        if sha256(R28_OUTPUT / relative) != digest:
            raise RuntimeError(f"Round-28 scientific output changed: {relative}")
    figure_dir = ROOT / lock["figure_directory"]
    for relative, digest in lock["figure_hashes"].items():
        if sha256(figure_dir / relative) != digest:
            raise RuntimeError(f"Round-28 figure changed: {relative}")
    if sha256(ROOT / lock["report"]) != lock["report_sha256"]:
        raise RuntimeError("Round-28 report changed")
    return {"passed": True, "round28_original_lock_sha256": EXPECTED_R28, "round28_output_hash_count": len(lock["output_hashes"]), "round28_figure_hash_count": len(lock["figure_hashes"])}


def _rank(values: Sequence[float]) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    order = np.argsort(data, kind="mergesort")
    result = np.empty(len(data), dtype=np.float64)
    cursor = 0
    while cursor < len(data):
        end = cursor + 1
        while end < len(data) and data[order[end]] == data[order[cursor]]:
            end += 1
        result[order[cursor:end]] = 0.5 * (cursor + 1 + end)
        cursor = end
    return result


def _auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    positive = np.asarray(labels, dtype=bool)
    if not np.any(positive) or np.all(positive):
        return 0.5
    ranks = _rank(scores)
    count_pos, count_neg = int(np.sum(positive)), int(np.sum(~positive))
    return float((np.sum(ranks[positive]) - count_pos * (count_pos + 1) / 2.0) / (count_pos * count_neg))


def _minmax(value: float, record: Mapping[str, Any], epsilon: float) -> float:
    return float(np.clip((value - float(record["train_min"])) / max(float(record["range"]), epsilon), 0.0, 1.0))


def _quantile(values: Sequence[float], fraction: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), fraction))


def _effect(left: Sequence[float], right: Sequence[float]) -> float:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if not len(a) or not len(b):
        return 0.0
    denominator = math.sqrt(0.5 * (float(np.var(a)) + float(np.var(b))))
    return float((float(np.mean(a)) - float(np.mean(b))) / max(denominator, 1e-12))


def _bootstrap(rows: Sequence[Mapping[str, Any]], function: Callable[[Sequence[Mapping[str, Any]]], float], *, seed: int, resamples: int) -> tuple[float, float]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["scenario_id"])].append(row)
    keys = sorted(groups)
    if not keys:
        return 0.0, 0.0
    generator = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(resamples):
        selected = generator.choice(keys, size=len(keys), replace=True)
        sample = [item for key in selected for item in groups[str(key)]]
        values.append(float(function(sample)))
    return _quantile(values, .025), _quantile(values, .975)


def _utility_class(value: float, tolerance: float) -> str:
    return "useful" if value > tolerance else "harmful" if value < -tolerance else "neutral"


def _variant_values(unserved_mass: float, total_normalized_queue: float, num_buoys: int, capacity_tension: float, norm_distance: float, v3: Mapping[str, Any], epsilon: float) -> dict[str, float]:
    v1 = unserved_mass / max(total_normalized_queue, epsilon)
    v2 = unserved_mass / max(float(num_buoys), epsilon)
    v3_norm = _minmax(capacity_tension, v3, epsilon)
    return {
        "DSP-V1": float(.5 * v1 + .5 * norm_distance),
        "DSP-V2": float(.5 * v2 + .5 * norm_distance),
        "DSP-V3": float(.5 * v3_norm + .5 * norm_distance),
        "backlog_v1_queue_share": float(v1),
        "backlog_v2_per_buoy_fraction": float(v2),
        "backlog_v3_capacity_tension_raw": float(capacity_tension),
        "backlog_v3_capacity_tension_norm": float(v3_norm),
    }


def _r28_replay(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, float], list[dict[str, Any]]]:
    rows = _read_csv(R28_OUTPUT / "planning_need_feature_table.csv")
    normalizer = _load(R28_OUTPUT / "planning_need_feature_normalization.json")
    epsilon = float(config["normalization"]["epsilon"])
    train = [row for row in rows if row["split"] == "train"]
    v3_values = [float(row["unassigned_queue_to_h5_current_capacity_ratio"]) for row in train]
    v3 = {"train_min": min(v3_values), "train_max": max(v3_values), "range": max(v3_values) - min(v3_values)}
    replay: list[dict[str, Any]] = []
    for row in rows:
        values = _variant_values(float(row["unserved_high_queue_mass"]), float(row["total_normalized_queue"]), int(row["num_buoys"]), float(row["unassigned_queue_to_h5_current_capacity_ratio"]), float(row["norm_distance_to_best_unassigned_high_value_buoy_m"]), v3, epsilon)
        original = .5 * float(row["norm_unserved_high_queue_mass"]) + .5 * float(row["norm_distance_to_best_unassigned_high_value_buoy_m"])
        replay.append({
            "state_id": row["state_id"], "scenario_id": row["scenario_id"], "split": row["split"], "timestep": int(row["timestep"]),
            "num_buoys": int(row["num_buoys"]), "eta": float(row["eta"]), "utility_payload": float(row["utility_payload"]), "utility_class": row["utility_class"],
            "original_dsp": float(original), "original_dsp_locked": float(row["deferred_service_pressure"]),
            "unserved_high_queue_mass": float(row["unserved_high_queue_mass"]), "total_normalized_queue": float(row["total_normalized_queue"]),
            "unserved_queue_bits": float(row["unserved_high_queue_mass"]) * QUEUE_CAPACITY_BITS,
            "total_queue_bits": float(row["total_normalized_queue"]) * QUEUE_CAPACITY_BITS,
            "current_aggregate_service_capacity_bits_per_step": float(row["total_normalized_queue"]) * QUEUE_CAPACITY_BITS / max(float(row["queue_to_current_capacity_ratio"]), epsilon),
            "distance_to_unserved_target_m": float(row["distance_to_best_unassigned_high_value_buoy_m"]),
            "norm_distance": float(row["norm_distance_to_best_unassigned_high_value_buoy_m"]),
            **values,
        })
    if any(abs(float(row["original_dsp"]) - float(row["original_dsp_locked"])) > 1e-12 for row in replay):
        raise RuntimeError("Round-29 did not reproduce frozen original DSP")
    thresholds = {name: _quantile([float(row[name]) for row in replay if row["split"] == "train"], .75) for name in ("DSP-V1", "DSP-V2", "DSP-V3")}
    return replay, normalizer, {"v3_scaler": v3, **thresholds}, rows


def _selection_rows(replay: Sequence[Mapping[str, Any]], thresholds: Mapping[str, float]) -> list[dict[str, Any]]:
    train = [row for row in replay if row["split"] == "train"]
    result: list[dict[str, Any]] = []
    for name, simplicity in (("DSP-V1", 3), ("DSP-V2", 2), ("DSP-V3", 1)):
        score = [float(row[name]) for row in train]
        useful = [row["utility_class"] == "useful" for row in train]
        useful_harmful = [row for row in train if row["utility_class"] != "neutral"]
        threshold = float(thresholds[name])
        activation = {k: sum(float(row[name]) >= threshold for row in train if int(row["num_buoys"]) == k) / max(sum(int(row["num_buoys"]) == k for row in train), 1) for k in (3, 5)}
        result.append({
            "variant": name, "source_split": "train", "threshold_train_p75": threshold,
            "useful_vs_nonuseful_auc": _auc(score, useful),
            "useful_vs_harmful_auc": _auc([float(row[name]) for row in useful_harmful], [row["utility_class"] == "useful" for row in useful_harmful]),
            "k3_activation_rate": activation[3], "k5_activation_rate": activation[5],
            "k3_k5_activation_balance": 1.0 - abs(activation[3] - activation[5]), "physical_simplicity_rank": simplicity,
        })
    return result


def _select_variant(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (-float(row["useful_vs_nonuseful_auc"]), -float(row["useful_vs_harmful_auc"]), -float(row["k3_k5_activation_balance"]), -int(row["physical_simplicity_rank"]), str(row["variant"])))
    return {"round": "round_29", "selection_source": "round28_train_only", "validation_utility_used": False, "confirmation_utility_used": False, "confirmation_primary_variant": ordered[0]["variant"], "selection_ranking": list(ordered), "selected_record": ordered[0]}


def _summary(values: Sequence[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(data)) if len(data) else 0.0, "median": float(np.median(data)) if len(data) else 0.0, "p25": _quantile(data, .25) if len(data) else 0.0, "p75": _quantile(data, .75) if len(data) else 0.0}


def _k_scale_audit(replay: Sequence[Mapping[str, Any]], original_threshold: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = ("unserved_high_queue_mass", "original_dsp", "total_queue_bits", "unserved_queue_bits", "num_buoys", "current_aggregate_service_capacity_bits_per_step", "distance_to_unserved_target_m")
    rows: list[dict[str, Any]] = []
    for split in ("all", "train", "validation"):
        population = list(replay) if split == "all" else [row for row in replay if row["split"] == split]
        for metric in metrics:
            groups = {k: [float(row[metric]) for row in population if int(row["num_buoys"]) == k] for k in (3, 5)}
            effect = _effect(groups[5], groups[3])
            for k in (3, 5):
                rows.append({"split": split, "metric": metric, "num_buoys": k, "count": len(groups[k]), **_summary(groups[k]), "effect_size_k5_minus_k3": effect})
        for metric, predicate in (("original_dsp_activation_rate", lambda row: float(row["original_dsp"]) >= original_threshold), ("planner_useful_rate", lambda row: row["utility_class"] == "useful"), ("planner_harmful_rate", lambda row: row["utility_class"] == "harmful")):
            groups = {k: [1.0 if predicate(row) else 0.0 for row in population if int(row["num_buoys"]) == k] for k in (3, 5)}
            effect = _effect(groups[5], groups[3])
            for k in (3, 5):
                rows.append({"split": split, "metric": metric, "num_buoys": k, "count": len(groups[k]), **_summary(groups[k]), "effect_size_k5_minus_k3": effect})
    decomposition: list[dict[str, Any]] = []
    for threshold_scope, threshold_map in (("formal_global_train_p75", {3: original_threshold, 5: original_threshold}), ("diagnostic_k_specific_train_p75", {k: _quantile([float(row["original_dsp"]) for row in replay if row["split"] == "train" and int(row["num_buoys"]) == k], .75) for k in (3, 5)})):
        for split in ("all", "train", "validation"):
            population = list(replay) if split == "all" else [row for row in replay if row["split"] == split]
            for k in (3, 5):
                group = [row for row in population if int(row["num_buoys"]) == k]
                active = [row for row in group if float(row["original_dsp"]) >= threshold_map[k]]
                decomposition.append({"threshold_scope": threshold_scope, "threshold": threshold_map[k], "split": split, "num_buoys": k, "population_count": len(group), "activation_count": len(active), "activation_rate": len(active) / max(len(group), 1), "useful_rate_active": sum(row["utility_class"] == "useful" for row in active) / max(len(active), 1), "harmful_rate_active": sum(row["utility_class"] == "harmful" for row in active) / max(len(active), 1), "mean_active_utility": float(np.mean([float(row["utility_payload"]) for row in active])) if active else 0.0, "median_active_utility": float(np.median([float(row["utility_payload"]) for row in active])) if active else 0.0})
    return rows, decomposition


@dataclass(frozen=True)
class ConfirmationScenario:
    scenario_id: str
    seed: int
    num_buoys: int
    eta: float
    queue_pressure: str
    geometry_class: str
    buoy_positions_m: tuple[tuple[float, float], ...]
    uav_positions_m: tuple[tuple[float, float], ...]
    initial_queue_bits: tuple[float, ...]

    def config(self) -> MaritimeEnvConfig:
        return MaritimeEnvConfig(num_uavs=2, num_buoys=self.num_buoys, task_semantics_version="v1", buoy_drift_mode="static", channel_variation_mode="static", episode_horizon=12, terminate_on_task_success=False, seed=self.seed, initial_uav_positions_m=self.uav_positions_m, initial_queue_bits_by_buoy=self.initial_queue_bits, arrival_mode="constant", constant_arrival_bits_per_step=4.5e5, multi_uav_min_separation_m=15.0)

    def options(self) -> SharedSpectrumOptions:
        return SharedSpectrumOptions("shared", self.eta, "legacy", .25)


def _confirmation_scenarios(config: Mapping[str, Any]) -> tuple[ConfirmationScenario, ...]:
    layouts = {
        "compact": ((60., 82.), (100., 108.), (140., 82.), (78., 134.), (124., 136.)),
        "spread": ((28., 42.), (100., 164.), (172., 42.), (36., 166.), (164., 166.)),
        "stress": ((86., 101.), (114., 101.), (100., 128.), (74., 128.), (126., 128.)),
        "offset": ((54., 96.), (104., 114.), (146., 86.), (76., 142.), (130., 132.)),
    }
    uavs = {"compact": ((88., 66.), (112., 66.)), "spread": ((38., 154.), (162., 32.)), "stress": ((91., 79.), (109., 79.)), "offset": ((80., 72.), (120., 68.))}
    pressures = (("low", .85e6), ("medium", 2.3e6), ("high", 4.0e6), ("mixed", 2.9e6))
    seed = int(config["confirmation"]["seed_start"])
    scenarios: list[ConfirmationScenario] = []
    for k in (3, 5):
        for eta in (.25, .5):
            for replica in range(10):
                geometry = tuple(layouts)[(replica + k + int(eta * 4)) % len(layouts)]
                pressure, base = pressures[(replica + k) % len(pressures)]
                generator = np.random.default_rng(seed)
                queue = np.clip(base * generator.uniform(.62, 1.18, size=k), 1.e5, QUEUE_CAPACITY_BITS)
                if pressure in {"high", "mixed"}:
                    queue[0] = min(QUEUE_CAPACITY_BITS, queue[0] * 1.18)
                scenarios.append(ConfirmationScenario(f"r29_confirm_k{k}_eta{str(eta).replace('.', '')}_{replica:02d}", seed, k, eta, pressure, geometry, tuple(layouts[geometry][:k]), tuple(uavs[geometry]), tuple(float(value) for value in queue)))
                seed += 1
    if len(scenarios) != 40 or {item.seed for item in scenarios} & set(range(20000, 20080)) or {item.seed for item in scenarios} & set(range(29000, 29016)):
        raise RuntimeError("Round-29 confirmation scenarios are not a new disjoint fixed set")
    return tuple(scenarios)


def _confirmation_manifest(config: Mapping[str, Any], scenarios: Sequence[ConfirmationScenario]) -> dict[str, Any]:
    return {"round": "round_29", "namespace": config["confirmation"]["namespace"], "scenario_count": len(scenarios), "cells": {f"K{k}_eta{eta}": sum(item.num_buoys == k and item.eta == eta for item in scenarios) for k in (3, 5) for eta in (.25, .5)}, "state_timesteps": config["confirmation"]["state_timesteps"], "selection_precedes_dsp_and_utility": True, "scenarios": [{**asdict(item), "num_uavs": 2, "spectrum_mode": "shared", "buoy_drift_mode": "static", "channel_variation_mode": "static", "episode_horizon": 12} for item in scenarios]}


def _restore(scenario: ConfirmationScenario, timestep: int) -> tuple[SharedSpectrumMultiUAVEnv, dict[str, Any]]:
    env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m, dtype=np.float64), options=scenario.options())
    _, info = env.reset(seed=scenario.seed)
    for _ in range(timestep):
        decision = joint_plan(env, info, horizon=1)
        if decision.future_trace_accessed:
            raise RuntimeError("confirmation state restoration used future trace")
        _, _, terminated, truncated, info = env.step(decision.action, matching_mode="joint_aware")
        if terminated or truncated:
            raise RuntimeError("confirmation state ended before registered sampling time")
    return env, info


def _signature(info: Mapping[str, Any], scenario: ConfirmationScenario, timestep: int) -> tuple[bytes, bytes, int, int]:
    return (np.asarray(info["uav_positions_m"], dtype=np.float64).tobytes(), np.asarray(info["raw_queue_bits"], dtype=np.float64).tobytes(), scenario.seed, timestep)


def _current_features(scenario: ConfirmationScenario, timestep: int, normalizer: Mapping[str, Any], v3: Mapping[str, Any], epsilon: float) -> dict[str, Any]:
    env, info = _restore(scenario, timestep)
    positions, queues, buoys = np.asarray(info["uav_positions_m"], dtype=np.float64), np.asarray(info["raw_queue_bits"], dtype=np.float64), np.asarray(info["buoy_positions_m"], dtype=np.float64)
    gains, direct = _direct_rates(positions, buoys, env)
    normalized = np.clip(queues / QUEUE_CAPACITY_BITS, 0., 1.)
    weighted = normalized[None, :] * direct
    top = np.argmax(weighted, axis=1)
    independent = float(np.sum(np.max(weighted, axis=1)))
    exclusive, _ = _exclusive_score(weighted)
    assignment, _, _, _ = joint_aware_matching(queues, gains, env.config, scenario.eta)
    _, _, _, _, selected_rates = assignment_link_metrics(assignment, gains, env.config, scenario.eta)
    unserved = np.ones(env.num_buoys, dtype=bool)
    unserved[assignment[assignment >= 0]] = False
    assigned_set = {int(value) for value in assignment if value >= 0}
    alternatives: list[float] = []
    for uav in range(env.num_uavs):
        choices = [buoy for buoy in range(env.num_buoys) if buoy not in assigned_set or buoy == int(assignment[uav])]
        options = [buoy for buoy in choices if buoy != int(assignment[uav])]
        if options:
            candidate = max(options, key=lambda buoy: (weighted[uav, buoy], -buoy))
            alternatives.append(float(np.linalg.norm(positions[uav] - buoys[candidate])))
        else:
            alternatives.append(0.)
    mass = float(np.sum(queues[unserved]) / QUEUE_CAPACITY_BITS)
    capacity = float(np.sum(selected_rates) * env.config.delta_t)
    distance = float(np.mean(alternatives))
    norm_mass = _minmax(mass, normalizer["features"]["unserved_high_queue_mass"], epsilon)
    norm_distance = _minmax(distance, normalizer["features"]["distance_to_best_unassigned_high_value_buoy_m"], epsilon)
    values = _variant_values(mass, float(np.sum(normalized)), env.num_buoys, float(np.sum(queues[unserved]) / max(5. * capacity, epsilon)), norm_distance, v3, epsilon)
    return {"state_id": f"{scenario.scenario_id}__t{timestep:02d}", "scenario_id": scenario.scenario_id, "seed": scenario.seed, "timestep": timestep, "num_buoys": env.num_buoys, "eta": scenario.eta, "queue_pressure": scenario.queue_pressure, "geometry_class": scenario.geometry_class, "unserved_high_queue_mass": mass, "total_normalized_queue": float(np.sum(normalized)), "unserved_queue_bits": float(np.sum(queues[unserved])), "total_queue_bits": float(np.sum(queues)), "current_aggregate_service_capacity_bits_per_step": capacity, "distance_to_unserved_target_m": distance, "assignment_regret": float(max(independent - exclusive, 0.) / max(abs(independent), epsilon)), "assignment_competition": float(top[0] == top[1]), "original_dsp": float(.5 * norm_mass + .5 * norm_distance), "norm_distance": norm_distance, **values}


def _branch(scenario: ConfirmationScenario, timestep: int, first_horizon: int, total_horizon: int) -> dict[str, Any]:
    env, info = _restore(scenario, timestep)
    signature = _signature(info, scenario, timestep)
    payload, episode_return = 0., 0.
    decision = joint_plan(env, info, horizon=first_horizon)
    if decision.future_trace_accessed:
        raise RuntimeError("confirmation planner used future trace")
    action = decision.action.copy()
    applied_first: np.ndarray | None = None
    for step in range(total_horizon):
        if step:
            decision = joint_plan(env, info, horizon=1)
            if decision.future_trace_accessed:
                raise RuntimeError("confirmation continuation used future trace")
            action = decision.action
        _, reward, terminated, truncated, info = env.step(action, matching_mode="joint_aware")
        if step == 0:
            applied_first = np.asarray(info["normalized_applied_action"], dtype=np.float64)
        payload += float(np.sum(np.asarray(info["transmitted_payload_bits"], dtype=np.float64)))
        episode_return += float(reward)
        if terminated or (truncated and step + 1 != total_horizon):
            raise RuntimeError("confirmation counterfactual terminated unexpectedly")
    return {"signature": signature, "payload_bits": payload, "episode_return": episode_return, "applied_first_action": applied_first}


def _confirmation_rows(scenarios: Sequence[ConfirmationScenario], config: Mapping[str, Any], normalizer: Mapping[str, Any], v3: Mapping[str, Any]) -> list[dict[str, Any]]:
    epsilon, tolerance = float(config["normalization"]["epsilon"]), float(config["counterfactual_protocol"]["numerical_utility_tolerance_bits"])
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for timestep in config["confirmation"]["state_timesteps"]:
            features = _current_features(scenario, int(timestep), normalizer, v3, epsilon)
            h5 = _branch(scenario, int(timestep), 5, 5)
            h1 = _branch(scenario, int(timestep), 1, 5)
            if h5["signature"] != h1["signature"]:
                raise RuntimeError("confirmation utility branches do not share an identical snapshot")
            utility = float(h5["payload_bits"] - h1["payload_bits"])
            rmse = float(np.sqrt(np.mean((np.asarray(h5["applied_first_action"]) - np.asarray(h1["applied_first_action"])) ** 2)))
            rows.append({**features, "planner_payload_bits": h5["payload_bits"], "reference_payload_bits": h1["payload_bits"], "utility_payload": utility, "utility_class": _utility_class(utility, tolerance), "same_snapshot_verified": True, "only_first_action_intervention": True, "continuation_policy": "frozen_causal_joint_h1", "h5_to_h1_first_action_rmse": rmse})
    if len(rows) != 160:
        raise RuntimeError("confirmation sampling did not cover all registered states")
    return rows


def _trigger_metric(rows: Sequence[Mapping[str, Any]], score: str, threshold: float, group_type: str, group_value: str, config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    active = [row for row in rows if float(row[score]) >= threshold]
    active_utility = [float(row["utility_payload"]) for row in active]
    ci = _bootstrap(rows, lambda sample: float(np.mean([float(item["utility_payload"]) for item in sample if float(item[score]) >= threshold])) if any(float(item[score]) >= threshold for item in sample) else 0., seed=seed, resamples=int(config["statistics"]["bootstrap_resamples"]))
    useful_total = sum(row["utility_class"] == "useful" for row in rows)
    harmful_total = sum(row["utility_class"] == "harmful" for row in rows)
    useful_active = sum(row["utility_class"] == "useful" for row in active)
    harmful_active = sum(row["utility_class"] == "harmful" for row in active)
    return {"score": score, "threshold": threshold, "threshold_scope": "single_train_p75", "group_type": group_type, "group_value": group_value, "population_count": len(rows), "activation_count": len(active), "activation_rate": len(active) / max(len(rows), 1), "active_scenario_count": len({row["scenario_id"] for row in active}), "precision_useful": useful_active / max(len(active), 1), "recall_useful": useful_active / max(useful_total, 1), "specificity": sum(row["utility_class"] != "useful" and row not in active for row in rows) / max(len(rows) - useful_total, 1), "harmful_rate_active": harmful_active / max(len(active), 1), "harmful_rate_population": harmful_total / max(len(rows), 1), "mean_active_utility": float(np.mean(active_utility)) if active else 0., "median_active_utility": float(np.median(active_utility)) if active else 0., "active_utility_ci95_low": ci[0], "active_utility_ci95_high": ci[1], "bootstrap_cluster": "scenario_id"}


def _confirmation_metrics(rows: Sequence[Mapping[str, Any]], selected: str, thresholds: Mapping[str, Any], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    candidates = (("original_dsp", float(thresholds["original_dsp_train_p75"])), (selected, float(thresholds[selected])))
    for index, (score, threshold) in enumerate(candidates):
        metrics.append(_trigger_metric(rows, score, threshold, "overall", "all", config, 2910 + index))
        for k in (3, 5):
            metrics.append(_trigger_metric([row for row in rows if int(row["num_buoys"]) == k], score, threshold, "num_buoys", str(k), config, 2920 + index * 10 + k))
        for eta in (.25, .5):
            metrics.append(_trigger_metric([row for row in rows if float(row["eta"]) == eta], score, threshold, "eta", str(eta), config, 2940 + index * 10 + int(eta * 100)))
    return metrics


def _interaction(rows: Sequence[Mapping[str, Any]], score: str, threshold: float, config: Mapping[str, Any]) -> dict[str, Any]:
    k3, k5 = ([row for row in rows if int(row["num_buoys"]) == k] for k in (3, 5))
    def mean_active(sample: Sequence[Mapping[str, Any]]) -> float:
        values = [float(row["utility_payload"]) for row in sample if float(row[score]) >= threshold]
        return float(np.mean(values)) if values else 0.
    low3, high3 = _bootstrap(k3, mean_active, seed=2963, resamples=int(config["statistics"]["bootstrap_resamples"]))
    low5, high5 = _bootstrap(k5, mean_active, seed=2965, resamples=int(config["statistics"]["bootstrap_resamples"]))
    groups3, groups5 = sorted({str(row["scenario_id"]) for row in k3}), sorted({str(row["scenario_id"]) for row in k5})
    g3, g5 = defaultdict(list), defaultdict(list)
    for row in k3: g3[str(row["scenario_id"])].append(row)
    for row in k5: g5[str(row["scenario_id"])].append(row)
    rng, samples = np.random.default_rng(2969), []
    for _ in range(int(config["statistics"]["bootstrap_resamples"])):
        a = [item for key in rng.choice(groups3, size=len(groups3), replace=True) for item in g3[str(key)]]
        b = [item for key in rng.choice(groups5, size=len(groups5), replace=True) for item in g5[str(key)]]
        samples.append(mean_active(b) - mean_active(a))
    return {"score": score, "threshold": threshold, "k3_mean_active_utility": mean_active(k3), "k3_ci95_low": low3, "k3_ci95_high": high3, "k5_mean_active_utility": mean_active(k5), "k5_ci95_low": low5, "k5_ci95_high": high5, "delta_utility_k5_minus_k3": mean_active(k5) - mean_active(k3), "delta_ci95_low": _quantile(samples, .025), "delta_ci95_high": _quantile(samples, .975), "bootstrap_cluster": "scenario_id"}


def _mechanism(rows: Sequence[Mapping[str, Any]], selected: str, threshold: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = ("backlog_v1_queue_share", "backlog_v3_capacity_tension_raw", "distance_to_unserved_target_m", "assignment_regret", "assignment_competition", "utility_payload")
    comparison: list[dict[str, Any]] = []
    for metric in metrics:
        groups = {k: [float(row[metric]) for row in rows if int(row["num_buoys"]) == k and float(row[selected]) >= threshold] for k in (3, 5)}
        for k in (3, 5):
            comparison.append({"metric": metric, "num_buoys": k, "active_count": len(groups[k]), **_summary(groups[k]), "effect_size_k5_minus_k3": _effect(groups[5], groups[3])})
    prevalence = []
    for k in (3, 5):
        group = [row for row in rows if int(row["num_buoys"]) == k]
        prevalence.append({"num_buoys": k, "population_count": len(group), "useful_count": sum(row["utility_class"] == "useful" for row in group), "harmful_count": sum(row["utility_class"] == "harmful" for row in group), "useful_state_prevalence": sum(row["utility_class"] == "useful" for row in group) / len(group), "harmful_state_prevalence": sum(row["utility_class"] == "harmful" for row in group) / len(group)})
    return comparison, prevalence


def _gate(metrics: Sequence[Mapping[str, Any]], interaction: Mapping[str, Any], prevalence: Sequence[Mapping[str, Any]], selected: str, config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    pick = lambda kind, value: next(row for row in metrics if row["score"] == selected and row["group_type"] == kind and row["group_value"] == value)
    overall, k3, k5, eta25, eta50 = pick("overall", "all"), pick("num_buoys", "3"), pick("num_buoys", "5"), pick("eta", "0.25"), pick("eta", "0.5")
    minimum_states, minimum_scenarios = int(config["statistics"]["minimum_active_states"]), int(config["statistics"]["minimum_active_scenarios"])
    coverage = overall["activation_count"] >= minimum_states and overall["active_scenario_count"] >= minimum_scenarios and k3["activation_count"] >= 4 and k5["activation_count"] >= 4
    overall_stable = overall["mean_active_utility"] > 0. and overall["active_utility_ci95_low"] > 0.
    k_positive = k3["mean_active_utility"] > 0. and k5["mean_active_utility"] > 0. and k3["active_utility_ci95_low"] >= 0. and k5["active_utility_ci95_low"] >= 0.
    eta_positive = eta25["mean_active_utility"] > 0. and eta50["mean_active_utility"] > 0.
    eta_reversal = (eta25["active_utility_ci95_high"] < 0. and eta50["active_utility_ci95_low"] > 0.) or (eta50["active_utility_ci95_high"] < 0. and eta25["active_utility_ci95_low"] > 0.)
    harmful_ok = overall["harmful_rate_active"] <= overall["harmful_rate_population"]
    original = next(row for row in metrics if row["score"] == "original_dsp" and row["group_type"] == "num_buoys" and row["group_value"] == "3")
    original_k5 = next(row for row in metrics if row["score"] == "original_dsp" and row["group_type"] == "num_buoys" and row["group_value"] == "5")
    original_reversal = original["mean_active_utility"] < 0. < original_k5["mean_active_utility"]
    normalization_eliminated_reversal = k3["mean_active_utility"] >= 0. and k5["mean_active_utility"] >= 0.
    prevalence_map = {int(row["num_buoys"]): float(row["useful_state_prevalence"]) for row in prevalence}
    physical_conditional = not normalization_eliminated_reversal and k5["active_utility_ci95_low"] > 0. and prevalence_map[5] > prevalence_map[3]
    cs_a = bool(overall_stable and k_positive and eta_positive and not eta_reversal and harmful_ok and coverage)
    cs_b = bool(not cs_a and overall_stable and physical_conditional and coverage)
    gate_name = "CS-A" if cs_a else "CS-B" if cs_b else "CS-C"
    gate = {"round": "round_29", "gate": gate_name, "confirmation_primary_variant": selected, "criteria": {"overall_stable": overall_stable, "k3_k5_nonnegative": k_positive, "eta_positive_direction": eta_positive, "eta_significant_reversal": eta_reversal, "harmful_rate_ok": harmful_ok, "coverage": coverage}, "original_k_sign_reversal": original_reversal, "normalization_eliminated_k_sign_reversal": normalization_eliminated_reversal, "scale_confounding_hypothesis_supported": bool(original_reversal and normalization_eliminated_reversal), "physical_k_dependent_planning_hypothesis_supported": physical_conditional, "cross_scale_trigger_confirmed": cs_a, "task_scale_conditional_trigger": cs_b, "training_performed": False}
    readiness = {"round": "round_29", "cross_scale_gate": gate_name, "minimal_trigger_model_training_ready": cs_a, "training_authorized_this_round": False, "round30_created": False, "recommendation": "R29-A_future_minimal_interpretable_trigger_study_only" if cs_a else "R29-B_scale_aware_planning_representation_research_only" if cs_b else "R29-C_stop_planner_trigger_representation_route"}
    return gate, readiness, {"k3": k3, "k5": k5, "eta25": eta25, "eta50": eta50, "overall": overall, "interaction": interaction}


def _figures(replay: Sequence[Mapping[str, Any]], selection: Sequence[Mapping[str, Any]], confirmation: Sequence[Mapping[str, Any]], metrics: Sequence[Mapping[str, Any]], interaction: Mapping[str, Any], prevalence: Sequence[Mapping[str, Any]], mechanism: Sequence[Mapping[str, Any]], selected: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    def save(number: int, name: str, draw: Callable[[Any], None]) -> None:
        fig, axis = plt.subplots(figsize=(6.4, 4.2)); draw(axis); fig.tight_layout()
        for suffix in ("png", "svg"): fig.savefig(FIGURES / f"{number:02d}_{name}.{suffix}", dpi=140)
        plt.close(fig)
    group = lambda data, field, k: [float(row[field]) for row in data if int(row["num_buoys"]) == k]
    save(1, "original_unserved_mass_k", lambda ax: (ax.boxplot([group(replay, "unserved_high_queue_mass", 3), group(replay, "unserved_high_queue_mass", 5)], labels=["K3", "K5"]), ax.set(ylabel="unserved high queue mass")))
    save(2, "original_dsp_k", lambda ax: (ax.boxplot([group(replay, "original_dsp", 3), group(replay, "original_dsp", 5)], labels=["K3", "K5"]), ax.set(ylabel="original DSP")))
    original_k = [next(row for row in metrics if row["score"] == "original_dsp" and row["group_type"] == "num_buoys" and row["group_value"] == str(k)) for k in (3, 5)]
    save(3, "original_activation_k", lambda ax: (ax.bar(["K3", "K5"], [float(row["activation_rate"]) for row in original_k], color="#4477aa"), ax.set(ylabel="activation rate")))
    save(4, "variant_train_auc", lambda ax: (ax.bar([row["variant"] for row in selection], [float(row["useful_vs_nonuseful_auc"]) for row in selection], color="#44aa77"), ax.set(ylabel="train useful AUC")))
    save(5, "variant_activation_balance", lambda ax: (ax.bar([row["variant"] for row in selection], [float(row["k3_k5_activation_balance"]) for row in selection], color="#aa4499"), ax.set(ylabel="K activation balance")))
    save(6, "selected_variant_k", lambda ax: (ax.boxplot([group(confirmation, selected, 3), group(confirmation, selected, 5)], labels=["K3", "K5"]), ax.set(ylabel=selected)))
    overall = [row for row in metrics if row["group_type"] == "overall"]
    save(7, "confirmation_overall_utility", lambda ax: (ax.bar([row["score"] for row in overall], [float(row["mean_active_utility"]) for row in overall], color="#4477aa"), ax.set(ylabel="active U5 payload bits")))
    selected_k = [next(row for row in metrics if row["score"] == selected and row["group_type"] == "num_buoys" and row["group_value"] == str(k)) for k in (3, 5)]
    save(8, "confirmation_k_utility", lambda ax: (ax.bar(["K3", "K5"], [float(row["mean_active_utility"]) for row in selected_k], color="#4477aa"), ax.set(ylabel="active U5 payload bits")))
    selected_eta = [next(row for row in metrics if row["score"] == selected and row["group_type"] == "eta" and row["group_value"] == str(eta)) for eta in (.25, .5)]
    save(9, "confirmation_eta_utility", lambda ax: (ax.bar([".25", ".5"], [float(row["mean_active_utility"]) for row in selected_eta], color="#ddcc77"), ax.set(xlabel="eta", ylabel="active U5 payload bits")))
    save(10, "original_vs_normalized_dsp", lambda ax: (ax.scatter([float(row["original_dsp"]) for row in confirmation], [float(row[selected]) for row in confirmation], c=["#44aa77" if row["utility_class"] == "useful" else "#cc6677" if row["utility_class"] == "harmful" else "#999999" for row in confirmation], s=18), ax.set(xlabel="original DSP", ylabel=selected)))
    save(11, "useful_prevalence_k", lambda ax: (ax.bar(["K3", "K5"], [float(next(row for row in prevalence if int(row["num_buoys"]) == k)["useful_state_prevalence"]) for k in (3, 5)], color="#44aa77"), ax.set(ylabel="useful-state prevalence")))
    mechanics = [row for row in mechanism if row["metric"] == "backlog_v1_queue_share"]
    save(12, "active_mechanism_comparison", lambda ax: (ax.bar(["K3", "K5"], [float(next(row for row in mechanics if int(row["num_buoys"]) == k)["mean"]) for k in (3, 5)], color="#aa4499"), ax.set(ylabel="active unserved queue share")))
    save(13, "cross_k_interaction", lambda ax: (ax.bar(["K5 - K3"], [float(interaction["delta_utility_k5_minus_k3"])], color="#4477aa"), ax.set(ylabel="active utility difference")))
    save(14, "harmful_rate", lambda ax: (ax.bar(["K3", "K5"], [float(row["harmful_rate_active"]) for row in selected_k], color="#cc6677"), ax.set(ylabel="active harmful rate")))
    save(15, "coverage", lambda ax: (ax.bar(["K3", "K5"], [float(row["activation_count"]) for row in selected_k], color="#44aa77"), ax.set(ylabel="active state count")))
    return sorted(path.name for path in FIGURES.iterdir() if path.suffix in {".png", ".svg"})


def _write_report(normalizer: Mapping[str, Any], selection: Mapping[str, Any], confirmation: Sequence[Mapping[str, Any]], gate: Mapping[str, Any], readiness: Mapping[str, Any], summary: Mapping[str, Any], interaction: Mapping[str, Any]) -> None:
    selected = str(selection["confirmation_primary_variant"])
    lines = ["# Round-29 Completion Report", "", f"1. Original DSP: {normalizer['features']['unserved_high_queue_mass']['train_min']:.6f}--{normalizer['features']['unserved_high_queue_mass']['train_max']:.6f} train-minmax unserved mass and frozen train-minmax distance, equal .5/.5.", "2. unserved_high_queue_mass is a raw sum over unserved buoy queues divided by one-buoy capacity; it has implicit K dependence.", f"3. Train-only confirmation primary variant: {selected}.", f"4. Confirmation scenarios / states: 40 / {len(confirmation)}.", f"5. Overall active utility + CI: {summary['overall']['mean_active_utility']:.6f} [{summary['overall']['active_utility_ci95_low']:.6f}, {summary['overall']['active_utility_ci95_high']:.6f}].", f"6. Activation / precision / recall / harmful: {summary['overall']['activation_rate']:.6f} / {summary['overall']['precision_useful']:.6f} / {summary['overall']['recall_useful']:.6f} / {summary['overall']['harmful_rate_active']:.6f}.", f"7. K3 utility + CI: {summary['k3']['mean_active_utility']:.6f} [{summary['k3']['active_utility_ci95_low']:.6f}, {summary['k3']['active_utility_ci95_high']:.6f}].", f"8. K5 utility + CI: {summary['k5']['mean_active_utility']:.6f} [{summary['k5']['active_utility_ci95_low']:.6f}, {summary['k5']['active_utility_ci95_high']:.6f}].", f"9. Cross-K interaction: {interaction['delta_utility_k5_minus_k3']:.6f} [{interaction['delta_ci95_low']:.6f}, {interaction['delta_ci95_high']:.6f}].", f"10. Eta=.25/.5 active utility: {summary['eta25']['mean_active_utility']:.6f} / {summary['eta50']['mean_active_utility']:.6f}.", f"11. Cross-Scale Gate: {gate['gate']}.", f"12. cross_scale_trigger_confirmed: {gate['cross_scale_trigger_confirmed']}.", f"13. task_scale_conditional_trigger: {gate['task_scale_conditional_trigger']}.", f"14. minimal_trigger_model_training_ready: {readiness['minimal_trigger_model_training_ready']}.", f"15. Recommendation: {readiness['recommendation']}.", "16. No neural/model/policy training, protected-data access, or Round-30 creation occurred."]
    (ROUND / "report" / "round29_completion_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_round29() -> dict[str, Any]:
    config, frozen = _load(CONFIG_PATH), verify_frozen_state()
    prereg_path = OUTPUT / "round29_preregistration_lock.json"
    confirmation_raw = OUTPUT / "round29_confirmation_raw_metrics.csv"
    resuming = prereg_path.is_file()
    if any(OUTPUT.iterdir()) and not resuming:
        raise RuntimeError("Round-29 contains an invalid partial pre-registration state")
    if confirmation_raw.exists():
        raise RuntimeError("Round-29 confirmation evidence already exists and is append-only")
    replay, normalizer, values, _ = _r28_replay(config)
    original_threshold = float(_load(R28_OUTPUT / "round28_trigger_thresholds.json")["composites"]["deferred_service_pressure"]["train_p75"])
    scale_rows, decomposition = _k_scale_audit(replay, original_threshold)
    selection_rows = _selection_rows(replay, values)
    selection = _select_variant(selection_rows)
    selected = str(selection["confirmation_primary_variant"])
    thresholds = {"round": "round_29", "fit_split": "round28_train", "validation_used_for_fit": False, "confirmation_used_for_fit": False, "formal_threshold_kind": "train_p75", "k_specific_threshold_excluded_from_gate": True, "original_dsp_train_p75": original_threshold, "DSP-V1": values["DSP-V1"], "DSP-V2": values["DSP-V2"], "DSP-V3": values["DSP-V3"]}
    scenarios = _confirmation_scenarios(config)
    preregistered = ("existing_dsp_scale_audit.md", "round29_dsp_variant_spec.json", "round29_trigger_thresholds.json", "round29_variant_selection_protocol.json", "selected_scale_normalized_dsp.json")
    if not resuming:
        existing = "# Existing DSP scale audit\n\nOriginal DSP is `0.5 * train_minmax(unserved_high_queue_mass) + 0.5 * train_minmax(distance_to_best_unassigned_high_value_buoy_m)`. `unserved_high_queue_mass` is `sum(queue_bits at currently unserved buoys) / per_buoy_queue_capacity_bits`: a raw unserved sum, not a fraction, mean, or capacity-normalized quantity. Its denominator is one-buoy capacity, so K enters implicitly through the count of unserved buoys. The spatial component uses only the frozen Round-28 train min/max.\n"
        (OUTPUT / "existing_dsp_scale_audit.md").write_text(existing, encoding="utf-8")
        _csv(OUTPUT / "dsp_K_scale_audit.csv", scale_rows); _csv(OUTPUT / "dsp_activation_decomposition.csv", decomposition); _csv(OUTPUT / "round28_replay_dsp_variants.csv", replay); _csv(OUTPUT / "round29_train_variant_selection.csv", selection_rows)
        _json(OUTPUT / "round29_dsp_variant_spec.json", {"round": "round_29", "variants": config["variants"], "frozen_original_distance_scaler": normalizer["features"]["distance_to_best_unassigned_high_value_buoy_m"], "v3_train_scaler": values["v3_scaler"], "weights_frozen": True, "current_observable_only": True})
        _json(OUTPUT / "round29_trigger_thresholds.json", thresholds)
        _json(OUTPUT / "round29_variant_selection_protocol.json", config["variant_selection"])
        _json(OUTPUT / "selected_scale_normalized_dsp.json", selection)
        _json(OUTPUT / "round29_preregistration_lock.json", {"round": "round_29", "status": "frozen_before_confirmation", "hashes": {name: sha256(OUTPUT / name) for name in preregistered}, "confirmation_outcomes_accessed": False})
        _json(OUTPUT / "round29_confirmation_manifest.json", _confirmation_manifest(config, scenarios))
        _json(OUTPUT / "round29_confirmation_manifest_lock.json", {"round": "round_29", "status": "frozen_before_utility", "manifest_sha256": sha256(OUTPUT / "round29_confirmation_manifest.json"), "preregistration_lock_sha256": sha256(prereg_path), "confirmation_outcomes_accessed": False})
    else:
        prereg = _load(prereg_path)
        if prereg.get("confirmation_outcomes_accessed") or prereg.get("hashes") != {name: sha256(OUTPUT / name) for name in preregistered}:
            raise RuntimeError("Round-29 pre-registration evidence changed before confirmation resume")
        if _load(OUTPUT / "selected_scale_normalized_dsp.json") != selection or _load(OUTPUT / "round29_trigger_thresholds.json") != thresholds:
            raise RuntimeError("Round-29 train-only selection changed before confirmation resume")
        manifest_lock = _load(OUTPUT / "round29_confirmation_manifest_lock.json")
        if manifest_lock.get("confirmation_outcomes_accessed") or manifest_lock.get("manifest_sha256") != sha256(OUTPUT / "round29_confirmation_manifest.json") or manifest_lock.get("preregistration_lock_sha256") != sha256(prereg_path):
            raise RuntimeError("Round-29 confirmation manifest changed before resume")
        selection_rows = _read_csv(OUTPUT / "round29_train_variant_selection.csv")
    confirmation = _confirmation_rows(scenarios, config, normalizer, values["v3_scaler"])
    metrics = _confirmation_metrics(confirmation, selected, thresholds, config)
    interaction = _interaction(confirmation, selected, float(thresholds[selected]), config)
    mechanism, prevalence = _mechanism(confirmation, selected, float(thresholds[selected]))
    eta = [row for row in metrics if row["score"] == selected and row["group_type"] == "eta"]
    gate, readiness, summary = _gate(metrics, interaction, prevalence, selected, config)
    _csv(OUTPUT / "round29_confirmation_raw_metrics.csv", confirmation); _csv(OUTPUT / "round29_confirmation_trigger_metrics.csv", metrics); _csv(OUTPUT / "cross_K_trigger_interaction.csv", [interaction]); _csv(OUTPUT / "cross_K_mechanism_audit.csv", mechanism); _csv(OUTPUT / "planner_useful_prevalence_K.csv", prevalence); _csv(OUTPUT / "eta_confirmation_audit.csv", eta)
    _json(OUTPUT / "cross_scale_planning_pressure_gate.json", gate); _json(OUTPUT / "round29_trigger_training_readiness.json", readiness); _json(OUTPUT / "round29_readiness_gate.json", {"round": "round_29", "gate": gate["gate"], "recommendation": readiness["recommendation"], "training_performed": False, "trigger_model_training_performed": False, "round30_created": False})
    files = _figures(replay, selection_rows, confirmation, metrics, interaction, prevalence, mechanism, selected)
    _write_report(normalizer, selection, confirmation, gate, readiness, summary, interaction)
    access = {"round": "round_29", "round28_train_rows_read": 192, "round28_validation_rows_read_exploratory": 64, "round20_development_test_reuse_count": 0, "round22_fresh_access_count": 0, "round24_fresh_access_count": 0, "untouched_eval_access_count": 0, "confirmation_scenario_count": 40, "confirmation_state_count": 160, "confirmation_outcome_access_count": 1, "fresh_new_rollout_count": 0, "new_scenarios_created": 40, "training_performed": False, "trigger_model_training_performed": False, "policy_checkpoint_inference_only": True, "planner_modified": False, "environment_modified": False, "round30_created": False}
    _json(OUTPUT / "access_audit.json", access)
    run = {"round": "round_29", "cross_scale_audit_performed": True, "confirmation_diagnostic_performed": True, "formal_sample_counts": {"round28_replay_states": 256, "confirmation_scenarios": 40, "confirmation_states": 160, "counterfactual_branches": 320}, "figure_file_count": len(files), "confirmation_outcome_access_count": 1, "frozen_state": frozen, "training_performed": False, "trigger_model_training_performed": False, "round30_created": False}
    _json(OUTPUT / "run_manifest.json", run)
    (ROUND / "logs" / "implementation_log.md").write_text("# Round-29 implementation log\n\nRead the locked Round-28 train/validation feature table only for scale diagnostics and train-only variant selection. Generated exactly the pre-registered independent K-balanced confirmation scenarios once, then ran same-snapshot H5-versus-H1 frozen-planner counterfactuals. No optimizer, model/policy training, checkpoint write, planner/environment modification, protected-data access, or Round-30 creation occurred.\n", encoding="utf-8")
    return {"gate": gate, "readiness": readiness, "selected": selection, "access": access, "figures": files}


def main() -> None:
    run_round29()


if __name__ == "__main__":
    main()
