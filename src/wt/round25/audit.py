"""Frozen Round-25 supervision-ambiguity and expert-plan alignment audit.

The module intentionally contains no learning or checkpoint-writing path.  It
uses only the locked Round-20 train/validation serialization, frozen model
inference, and the unchanged causal Round-19 planner.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from wt.governance_state_sync import approved_lifecycle_overlay_sha256, verify_round22_governance_state_sync_amendment
from wt.round19.planner import joint_plan
from wt.round19.multi_env import SharedSpectrumMultiUAVEnv
from wt.round20.scenarios import DevelopmentScenario, development_scenarios
from wt.round21.data import bridge_flags, conflict_flags, load_task_samples
from wt.round22.data import FrozenDeepSetsPredictor, VALIDATION, load_policy_sequences, sha256
from wt.round22.models import PREDICTED_TRANSFORMER
from wt.round22.training import TrainingConfig, load_checkpoint, predict


ROOT = Path(__file__).resolve().parents[3]
ROUND = ROOT / "round_25"
OUTPUT = ROUND / "outputs" / "round25_supervision_ambiguity_alignment_20260830_1203"
FIGURES = ROUND / "figures" / "round25_supervision_ambiguity_alignment_20260830_1203"
CONFIG_PATH = ROUND / "configs" / "round25_ambiguity_alignment_config.json"
R20 = ROOT / "round_20" / "outputs" / "round20_multi_uav_dataset_20260828_2100"
R22 = ROOT / "round_22" / "outputs" / "round22_deepsets_task_waypoint_policy_20260829_1153"
R24 = ROOT / "round_24" / "outputs" / "round24_multiseed_stability_20260829_1515"
EXPECTED_R24 = "B14A668DB825590C6E850C168008DBCB8FAEE4BDB5C56C82B5C906241887043B"
EXPECTED_PREDICTOR = "645F235032DE9639A6D42318A41CC8F0E2175E86E38545DA98E0BE5CAEB80221"
ALLOWED_SPLITS = frozenset(("train", "validation"))


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise RuntimeError(f"Round-25 output is unexpectedly empty: {path.name}")
    fields = sorted({key for row in materialized for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: json.dumps(_plain(value), sort_keys=True) if isinstance(value, (dict, tuple, list, np.ndarray)) else value for key, value in row.items()})


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _locked_tree_mismatches(lock: Mapping[str, Any]) -> list[str]:
    checks: list[tuple[Path, str]] = []
    checks.extend((ROOT / "round_24" / name, digest) for name, digest in lock["configuration_hashes"].items())
    checks.extend((ROOT / "src" / "wt" / "round24" / name, digest) for name, digest in lock["source_tree"].items())
    checks.extend((ROOT / name, digest) for name, digest in lock["test_hashes"].items())
    checks.extend((ROOT / lock["formal_output"] / name, digest) for name, digest in lock["output_hashes"].items())
    checks.extend((ROOT / lock["figure_directory"] / name, digest) for name, digest in lock["figure_hashes"].items())
    checks.extend(((ROOT / lock["report"], lock["report_sha256"]), (ROOT / lock["implementation_log"], lock["implementation_log_sha256"])))
    return [path.as_posix() for path, digest in checks if not path.is_file() or _digest(path) != digest]


def verify_frozen_state() -> dict[str, Any]:
    """Fail closed before decoding any audit row or invoking frozen inference."""
    r24_lock_path = ROOT / "round_24" / "ROUND_LOCK.json"
    if _digest(r24_lock_path) != EXPECTED_R24:
        raise RuntimeError("Round-24 original lock changed")
    r24_lock = _load_json(r24_lock_path)
    amendment = _load_json(ROOT / "round_24" / "ROUND_LOCK_AMENDMENT.json")
    affected = amendment.get("affected_paths", {})
    governance = affected.get("src/wt/governance_state_sync.py", {})
    lifecycle_test = affected.get("tests/test_round24_seed_extension.py", {})
    if (
        amendment.get("amendment_type") != "legal_round25_successor_governance_extension"
        or amendment.get("original_round24_lock_sha256") != EXPECTED_R24
        or amendment.get("scientific_artifacts_modified") is not False
        or amendment.get("original_round24_lock_modified") is not False
        or governance.get("original_sha256") != "5F88CC7563EC1EBC3A90B375017F997C4AEBAF218FFFD605B8BB38D642A57322"
        or _digest(ROOT / "src" / "wt" / "governance_state_sync.py") != approved_lifecycle_overlay_sha256("src/wt/governance_state_sync.py", governance.get("amended_sha256", ""))
        or lifecycle_test.get("original_sha256") != r24_lock["test_hashes"]["tests/test_round24_seed_extension.py"]
        or _digest(ROOT / "tests" / "test_round24_seed_extension.py") != approved_lifecycle_overlay_sha256("tests/test_round24_seed_extension.py", lifecycle_test.get("original_sha256", ""))
    ):
        raise RuntimeError("Round-24 legal-successor amendment is not the approved narrow overlay")
    # The original R24 lock remains immutable.  Its one lifecycle-test entry is
    # deliberately superseded by the narrowly checked successor amendment above.
    mismatches = [
        path for path in _locked_tree_mismatches(r24_lock)
        if path != (ROOT / "tests" / "test_round24_seed_extension.py").as_posix()
    ]
    if mismatches:
        raise RuntimeError(f"Round-24 frozen evidence mismatch: {mismatches[:3]}")
    governance_result = verify_round22_governance_state_sync_amendment()
    protocol = _load_json(R22 / "round22_protocol_freeze.json")
    expected_rows = {
        "round20_transition_sha256": _digest(R20 / "multi_uav_transitions.jsonl"),
        "round20_task_samples_sha256": _digest(R20 / "multi_uav_task_waypoint_samples.jsonl"),
    }
    if any(protocol.get(name) != digest for name, digest in expected_rows.items()):
        raise RuntimeError("Round-20 frozen dataset hashes do not match the Round-22 protocol")
    predictor = FrozenDeepSetsPredictor()
    if predictor.sha256 != EXPECTED_PREDICTOR or any(parameter.requires_grad for parameter in predictor.model.parameters()):
        raise RuntimeError("Round-21 H5 predictor is not frozen")
    r24_access = _load_json(R24 / "access_audit.json")
    required_access = {
        "round22_fresh_new_rollout_count": 0,
        "round20_development_test_reuse_count": 0,
        "untouched_eval_access_count": 0,
        "round24_fresh_access_count": 1,
    }
    if any(int(r24_access[name]) != expected for name, expected in required_access.items()):
        raise RuntimeError("protected access counters no longer match locked Round-24 evidence")
    return {
        "passed": True,
        "round24_original_lock_sha256": EXPECTED_R24,
        "round24_registered_hash_count": len(r24_lock["configuration_hashes"]) + len(r24_lock["source_tree"]) + len(r24_lock["test_hashes"]) + len(r24_lock["output_hashes"]) + len(r24_lock["figure_hashes"]) + 2,
        "round24_mismatch_count": 0,
        "round22_governance_verified": bool(governance_result["passed"]),
        "predictor_sha256": predictor.sha256,
        **required_access,
    }


def _permitted_json_rows(path: Path) -> list[dict[str, Any]]:
    """Route protected development-test lines before JSON decoding."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or '"split":"development_test"' in line or '"split": "development_test"' in line:
                continue
            row = json.loads(line)
            if row["split"] not in ALLOWED_SPLITS:
                raise RuntimeError("Round-25 attempted to decode a prohibited split")
            rows.append(row)
    return rows


def _target(row: Mapping[str, Any]) -> np.ndarray:
    service = np.asarray(row["per_uav_service_responsibility_target"], dtype=np.float64)
    progress = np.asarray(row["per_uav_gross_service_progress_normalized"], dtype=np.float64)
    returned = np.full((service.shape[0], 1), float(row["team_h_step_return_normalized"]), dtype=np.float64)
    return np.concatenate((service, progress, returned), axis=-1)


def _valid_buoys(row: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(row["buoy_mask"], dtype=bool)


def _service_js(left: np.ndarray, right: np.ndarray, buoy_mask: np.ndarray) -> float:
    values: list[float] = []
    for a, b in zip(left[:, buoy_mask], right[:, buoy_mask]):
        a = np.clip(np.asarray(a, dtype=np.float64), 0.0, None)
        b = np.clip(np.asarray(b, dtype=np.float64), 0.0, None)
        if a.sum() <= 1.0e-12 and b.sum() <= 1.0e-12:
            values.append(0.0)
            continue
        a = a / max(float(a.sum()), 1.0e-12)
        b = b / max(float(b.sum()), 1.0e-12)
        middle = 0.5 * (a + b)
        active_a = a > 0.0
        active_b = b > 0.0
        kl_a = float(np.sum(a[active_a] * np.log(a[active_a] / middle[active_a])))
        kl_b = float(np.sum(b[active_b] * np.log(b[active_b] / middle[active_b])))
        values.append(0.5 * (kl_a + kl_b))
    return float(np.mean(values))


def _progress_rmse(left: np.ndarray, right: np.ndarray, buoy_mask: np.ndarray) -> float:
    difference = np.asarray(left[:, buoy_mask], dtype=np.float64) - np.asarray(right[:, buoy_mask], dtype=np.float64)
    return float(np.sqrt(np.mean(difference ** 2)))


def _cosine_difference(left: np.ndarray, right: np.ndarray, buoy_mask: np.ndarray) -> float:
    a = np.asarray(left[:, buoy_mask], dtype=np.float64).reshape(-1)
    b = np.asarray(right[:, buoy_mask], dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom <= 1.0e-12 else float(1.0 - np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def _components(left: np.ndarray, right: np.ndarray, buoy_mask: np.ndarray) -> dict[str, float]:
    k_count = int(np.sum(buoy_mask))
    service_left, service_right = left[:, :k_count], right[:, :k_count]
    progress_left, progress_right = left[:, 5:5 + k_count], right[:, 5:5 + k_count]
    returned = abs(float(left[0, -1] - right[0, -1]))
    service = _service_js(service_left, service_right, np.ones(k_count, dtype=bool))
    progress = _progress_rmse(progress_left, progress_right, np.ones(k_count, dtype=bool))
    cosine = _cosine_difference(progress_left, progress_right, np.ones(k_count, dtype=bool))
    return {"service_js": service, "progress_rmse": progress, "progress_cosine_difference": cosine, "return_abs_zscore": returned, "task_distance": float(np.mean((service, progress, returned)))}


def _mean_target(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.mean(np.stack([_target(row) for row in rows], axis=0), axis=0)


def _entropy(service: np.ndarray, k_count: int) -> float:
    values = []
    for vector in service[:, :k_count]:
        p = np.clip(np.asarray(vector, dtype=np.float64), 0.0, None)
        if p.sum() <= 1.0e-12:
            values.append(0.0)
        else:
            p = p / p.sum()
            positive = p[p > 0.0]
            values.append(float(-np.sum(positive * np.log(positive))))
    return float(np.mean(values))


def _frozen_flag_map() -> dict[str, tuple[bool, bool]]:
    result: dict[str, tuple[bool, bool]] = {}
    for split in sorted(ALLOWED_SPLITS):
        dataset = load_task_samples(5, {split})
        bridge, conflict = bridge_flags(dataset), conflict_flags(dataset)
        for index, metadata in enumerate(dataset.metadata):
            result[str(metadata["sample_id"])] = (bool(bridge[index]), bool(conflict[index]))
    return result


def _scenario_map() -> dict[str, DevelopmentScenario]:
    manifest = _load_json(R20 / "multi_uav_dataset_manifest.json")
    registered = {str(row["scenario_id"]): row for row in manifest["scenarios"]}
    scenarios = {item.scenario_id: item for item in development_scenarios()}
    if set(registered) != set(scenarios):
        raise RuntimeError("Round-20 scenario identities no longer match the frozen protocol")
    for scenario_id, scenario in scenarios.items():
        row = registered[scenario_id]
        if int(row["seed"]) != scenario.seed or row["split"] != scenario.split or int(row["num_buoys"]) != scenario.num_buoys or float(row["interference_overlap_factor"]) != scenario.interference_overlap_factor:
            raise RuntimeError("Round-20 scenario protocol changed")
    return scenarios


def _planner_decision(sample: Mapping[str, Any], scenario: DevelopmentScenario, horizon: int) -> Any:
    graph = sample["current_graph_state"]
    env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m, dtype=np.float64), options=scenario.options())
    info = {
        "uav_positions_m": np.asarray(graph["uav_nodes"], dtype=np.float64)[:, :2],
        "remaining_energy_per_uav": np.asarray(graph["uav_nodes"], dtype=np.float64)[:, 4],
        "raw_queue_bits": np.asarray(graph["buoy_nodes"], dtype=np.float64)[: scenario.num_buoys, 2],
        "buoy_positions_m": np.asarray(graph["buoy_nodes"], dtype=np.float64)[: scenario.num_buoys, :2],
    }
    return joint_plan(env, info, horizon=horizon)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(values.size, dtype=np.float64)
    result[order] = np.arange(values.size, dtype=np.float64)
    return result


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if a.size < 3 or np.std(a) <= 1.0e-12 or np.std(b) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(np.mean(values)) if values else 0.0


def _exact_state_audit(samples: Sequence[Mapping[str, Any]], flags: Mapping[str, tuple[bool, bool]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    anchors = [row for row in samples if int(row["H_task"]) == 5 and int(row["t"]) == 0]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in anchors:
        groups[str(row["scenario_id"])].append(dict(row))
    expected = 10
    ambiguity_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for scenario_id, rows in sorted(groups.items()):
        if len(rows) != expected:
            raise RuntimeError(f"exact-state scenario {scenario_id} lacks full behavior coverage")
        graph_vectors = [np.concatenate((np.asarray(row["current_graph_state"]["uav_nodes"], dtype=np.float64).reshape(-1), np.asarray(row["current_graph_state"]["buoy_nodes"], dtype=np.float64).reshape(-1), np.asarray(row["current_graph_state"]["global_features"], dtype=np.float64))) for row in rows]
        if any(not np.allclose(graph_vectors[0], item, atol=0.0, rtol=0.0) for item in graph_vectors[1:]):
            raise RuntimeError(f"scenario {scenario_id} does not have an exact shared t=0 physical state")
        components = []
        for left_index in range(len(rows)):
            for right_index in range(left_index + 1, len(rows)):
                left, right = rows[left_index], rows[right_index]
                metric = _components(_target(left), _target(right), _valid_buoys(left))
                metric.update({"scenario_id": scenario_id, "split": left["split"], "num_buoys": int(np.sum(_valid_buoys(left))), "eta": float({"orthogonal": 0.0, "low": 0.25, "medium": 0.5, "full": 1.0}[left["interference_tier"]]), "behavior_left": left["behavior_generator"], "behavior_right": right["behavior_generator"]})
                pair_rows.append(metric)
                components.append(metric)
        bridge_values = [flags[str(row["sample_id"])][0] for row in rows]
        conflict_values = [flags[str(row["sample_id"])][1] for row in rows]
        ambiguity_rows.append({
            "scenario_id": scenario_id,
            "split": rows[0]["split"],
            "num_buoys": int(np.sum(_valid_buoys(rows[0]))),
            "eta": float({"orthogonal": 0.0, "low": 0.25, "medium": 0.5, "full": 1.0}[rows[0]["interference_tier"]]),
            "behavior_count": len(rows),
            "pair_count": len(components),
            "A_service": _mean(components, "service_js"),
            "A_progress": _mean(components, "progress_rmse"),
            "A_progress_cosine_difference": _mean(components, "progress_cosine_difference"),
            "A_return": _mean(components, "return_abs_zscore"),
            "A_task": _mean(components, "task_distance"),
            "bridge_any": bool(any(bridge_values)),
            "bridge_fraction": float(np.mean(bridge_values)),
            "conflict_any": bool(any(conflict_values)),
            "conflict_fraction": float(np.mean(conflict_values)),
        })
    focus = (
        ("joint_h10", "independent_queue_rate", "h10_vs_independent"),
        ("joint_h10", "joint_h1", "h10_vs_h1"),
        ("joint_h10", "noisy_joint_h10_noise_0.15", "h10_vs_noisy_h10_noise015"),
        ("joint_h10", "noisy_joint_h10_noise_0.35", "h10_vs_noisy_h10_noise035"),
        ("joint_h1", "independent_queue_rate", "h1_vs_independent"),
    )
    divergence_rows: list[dict[str, Any]] = []
    for label_left, label_right, label in focus:
        selected = [row for row in pair_rows if {row["behavior_left"], row["behavior_right"]} == {label_left, label_right}]
        divergence_rows.append({"behavior_pair": label, "left_behavior": label_left, "right_behavior": label_right, "scenario_count": len(selected), "mean_service_js": _mean(selected, "service_js"), "mean_progress_rmse": _mean(selected, "progress_rmse"), "mean_progress_cosine_difference": _mean(selected, "progress_cosine_difference"), "mean_return_abs_zscore": _mean(selected, "return_abs_zscore"), "mean_task_distance": _mean(selected, "task_distance")})
    return ambiguity_rows, divergence_rows, groups


def _quality_audit(groups: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_id, group in sorted(groups.items()):
        by_quality: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for item in group:
            by_quality[str(item["quality_tier"])].append(item)
        overall = _mean_target(group)
        quality_mean = {tier: _mean_target(items) for tier, items in by_quality.items()}
        for tier, target in quality_mean.items():
            metric = _components(target, overall, _valid_buoys(group[0]))
            rows.append({"record_type": "tier_vs_overall", "scenario_id": scenario_id, "split": group[0]["split"], "quality_tier": tier, "sample_count": len(by_quality[tier]), **metric})
        for left, right in (("high", "medium"), ("high", "low"), ("medium", "low")):
            if left in quality_mean and right in quality_mean:
                metric = _components(quality_mean[left], quality_mean[right], _valid_buoys(group[0]))
                rows.append({"record_type": "quality_pair", "scenario_id": scenario_id, "split": group[0]["split"], "quality_tier": f"{left}_vs_{right}", "sample_count": len(by_quality[left]) + len(by_quality[right]), **metric})
    return rows


def _knn_audit(samples: Sequence[Mapping[str, Any]], flags: Mapping[str, tuple[bool, bool]], neighbors: int) -> list[dict[str, Any]]:
    items = [row for row in samples if int(row["H_task"]) == 5]
    train = [row for row in items if row["split"] == "train"]
    def vector(row: Mapping[str, Any]) -> np.ndarray:
        graph = row["current_graph_state"]
        return np.concatenate((np.asarray(graph["uav_nodes"], dtype=np.float64).reshape(-1), np.asarray(graph["buoy_nodes"], dtype=np.float64).reshape(-1), np.asarray(graph["global_features"], dtype=np.float64)))
    raw = np.stack([vector(row) for row in items])
    train_raw = np.stack([vector(row) for row in train])
    mean, std = train_raw.mean(axis=0), train_raw.std(axis=0)
    values = (raw - mean) / np.where(std > 1.0e-9, std, 1.0)
    targets = [_target(row) for row in items]
    episode = np.asarray([str(row["episode_id"]) for row in items])
    results: list[dict[str, Any]] = []
    for start in range(0, len(items), 128):
        stop = min(start + 128, len(items))
        distance = np.sum((values[start:stop, None, :] - values[None, :, :]) ** 2, axis=2)
        for local, index in enumerate(range(start, stop)):
            distance[local, episode == episode[index]] = np.inf
            distance[local, index] = np.inf
            nearest = np.argsort(distance[local], kind="mergesort")[:neighbors]
            if len(nearest) != neighbors or not np.all(np.isfinite(distance[local, nearest])):
                raise RuntimeError("kNN audit did not find enough cross-episode neighbors")
            neighbor_targets = [targets[int(item)] for item in nearest]
            center = np.mean(np.stack(neighbor_targets), axis=0)
            parts = [_components(targets[int(item)], center, _valid_buoys(items[index])) for item in nearest]
            bridge, conflict = flags[str(items[index]["sample_id"])]
            results.append({
                "sample_id": items[index]["sample_id"], "scenario_id": items[index]["scenario_id"], "episode_id": items[index]["episode_id"], "split": items[index]["split"], "timestep": int(items[index]["t"]), "num_buoys": int(np.sum(_valid_buoys(items[index]))), "eta": float({"orthogonal": 0.0, "low": 0.25, "medium": 0.5, "full": 1.0}[items[index]["interference_tier"]]), "quality_tier": items[index]["quality_tier"], "behavior_generator": items[index]["behavior_generator"], "nearest_distance": float(math.sqrt(float(distance[local, nearest[0]]))), "neighbor_count": neighbors,
                "neighbor_behavior_diversity": int(len({str(items[int(item)]["behavior_generator"]) for item in nearest})), "neighbor_quality_diversity": int(len({str(items[int(item)]["quality_tier"]) for item in nearest})), "service_local_js_variance": _mean(parts, "service_js"), "progress_local_variance": _mean(parts, "progress_rmse"), "return_local_variance": _mean(parts, "return_abs_zscore"), "A_task_local": _mean(parts, "task_distance"), "bridge": bridge, "conflict": conflict,
            })
    return results


def _subgroup_ambiguity(knn: Sequence[Mapping[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in knn:
        groups[str(row[field])].append(row)
    return [{"group_field": field, "group_value": name, "sample_count": len(rows), "mean_service_local_js_variance": _mean(rows, "service_local_js_variance"), "mean_progress_local_variance": _mean(rows, "progress_local_variance"), "mean_return_local_variance": _mean(rows, "return_local_variance"), "mean_A_task_local": _mean(rows, "A_task_local")} for name, rows in sorted(groups.items())]


def _predictor_audits(groups: Mapping[str, Sequence[Mapping[str, Any]]], predictor: FrozenDeepSetsPredictor) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    flat = [row for _, group in sorted(groups.items()) for row in group]
    graph = {key: np.asarray([row["current_graph_state"][key] for row in flat]) for key in ("uav_nodes", "buoy_nodes", "uav_mask", "buoy_mask", "global_features")}
    predicted = predictor.phi(graph).astype(np.float64)
    by_scenario = {scenario_id: list(group) for scenario_id, group in groups.items()}
    mean_rows: list[dict[str, Any]] = []
    entropy_rows: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    return_rows: list[dict[str, Any]] = []
    for index, row in enumerate(flat):
        group = by_scenario[str(row["scenario_id"])]
        local = _mean_target(group)
        high_items = [item for item in group if item["quality_tier"] == "high"]
        high = _mean_target(high_items) if high_items else local
        planner_h10_items = [item for item in group if item["behavior_generator"] == "joint_h10"]
        planner_h10 = _target(planner_h10_items[0]) if planner_h10_items else local
        actual = _target(row)
        mask = _valid_buoys(row)
        d_actual = _components(predicted[index], actual, mask)
        d_local = _components(predicted[index], local, mask)
        d_high = _components(predicted[index], high, mask)
        d_h10 = _components(predicted[index], planner_h10, mask)
        mean_rows.append({"sample_id": row["sample_id"], "scenario_id": row["scenario_id"], "split": row["split"], "quality_tier": row["quality_tier"], "behavior_generator": row["behavior_generator"], "predictor_to_actual_task_distance": d_actual["task_distance"], "predictor_to_local_mean_task_distance": d_local["task_distance"], "predictor_to_high_quality_mean_task_distance": d_high["task_distance"], "predictor_to_h10_planner_like_task_distance": d_h10["task_distance"], "closer_to_local_than_high": d_local["task_distance"] <= d_high["task_distance"], "closer_to_local_than_h10": d_local["task_distance"] <= d_h10["task_distance"]})
        k_count = int(np.sum(mask))
        entropy_rows.append({"sample_id": row["sample_id"], "scenario_id": row["scenario_id"], "split": row["split"], "quality_tier": row["quality_tier"], "behavior_generator": row["behavior_generator"], "target_service_entropy": _entropy(actual[:, :5], k_count), "predicted_service_entropy": _entropy(predicted[index, :, :5], k_count), "high_quality_target_entropy": _entropy(high[:, :5], k_count), "local_mean_entropy": _entropy(local[:, :5], k_count)})
        progress_rows.append({"sample_id": row["sample_id"], "scenario_id": row["scenario_id"], "split": row["split"], "quality_tier": row["quality_tier"], "behavior_generator": row["behavior_generator"], "predicted_total_progress_normalized": float(np.sum(predicted[index, :, 5:5 + k_count])), "high_quality_total_progress_normalized": float(np.sum(high[:, 5:5 + k_count])), "low_quality_total_progress_normalized": float(np.sum(_mean_target([item for item in group if item["quality_tier"] == "low"])[:, 5:5 + k_count])) if any(item["quality_tier"] == "low" for item in group) else float(np.sum(local[:, 5:5 + k_count])), "local_mean_total_progress_normalized": float(np.sum(local[:, 5:5 + k_count]))})
        return_rows.append({"sample_id": row["sample_id"], "scenario_id": row["scenario_id"], "split": row["split"], "quality_tier": row["quality_tier"], "behavior_generator": row["behavior_generator"], "predicted_return_zscore": float(predicted[index, 0, -1]), "high_quality_return_zscore": float(high[0, -1]), "medium_quality_return_zscore": float(_mean_target([item for item in group if item["quality_tier"] == "medium"])[0, -1]) if any(item["quality_tier"] == "medium" for item in group) else float(local[0, -1]), "low_quality_return_zscore": float(_mean_target([item for item in group if item["quality_tier"] == "low"])[0, -1]) if any(item["quality_tier"] == "low" for item in group) else float(local[0, -1]), "local_mean_return_zscore": float(local[0, -1])})
    return mean_rows, entropy_rows, progress_rows, return_rows


def _planner_audits(groups: Mapping[str, Sequence[Mapping[str, Any]]], scenarios: Mapping[str, DevelopmentScenario], transitions: Mapping[tuple[str, int], Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    planner_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    h5_targets: dict[str, np.ndarray] = {}
    h5_actions: dict[str, np.ndarray] = {}
    for ordinal, (scenario_id, group) in enumerate(sorted(groups.items())):
        scenario = scenarios[scenario_id]
        by_behavior = {str(item["behavior_generator"]): item for item in group}
        h5_source = by_behavior["joint_h5"]
        h10_source = by_behavior["joint_h10"]
        h10_label_rows = [item for item in _PERMITTED_H10 if item["scenario_id"] == scenario_id and item["behavior_generator"] == "joint_h10" and int(item["t"]) == 0]
        if len(h10_label_rows) != 1:
            raise RuntimeError("frozen H10 planner label is unavailable")
        h5_decision = _planner_decision(h5_source, scenario, 5)
        stored_h5_action = np.asarray(transitions[(str(h5_source["episode_id"]), 0)]["joint_action_applied_t"], dtype=np.float64)
        h5_error = float(np.max(np.abs(h5_decision.action - stored_h5_action)))
        if h5_error > 1.0e-9:
            raise RuntimeError("Round-19 H5 direct causal planner replay differs from frozen H5 behavior action")
        h5_target = _target(h5_source)
        h10_target = _target(h10_label_rows[0])
        h5_targets[scenario_id], h5_actions[scenario_id] = h5_target, h5_decision.action
        replay_h10 = ordinal % 4 == 0
        h10_error: float | None = None
        if replay_h10:
            h10_decision = _planner_decision(h10_source, scenario, 10)
            stored_h10_action = np.asarray(transitions[(str(h10_source["episode_id"]), 0)]["joint_action_applied_t"], dtype=np.float64)
            h10_error = float(np.max(np.abs(h10_decision.action - stored_h10_action)))
            if h10_error > 1.0e-9:
                raise RuntimeError("Round-19 H10 direct causal planner replay differs from frozen H10 behavior action")
        k_count = int(np.sum(_valid_buoys(h5_source)))
        planner_rows.extend((
            {"scenario_id": scenario_id, "split": h5_source["split"], "planner_horizon": 5, "source_behavior": "joint_h5", "causal_current_state_only": True, "future_trace_accessed": bool(h5_decision.future_trace_accessed), "direct_replay": True, "first_action_max_abs_error": h5_error, "planner_predicted_payload_bits": float(h5_decision.predicted_payload_bits), "service_responsibility": h5_target[:, :5], "gross_progress_normalized": h5_target[:, 5:10], "team_return_normalized": float(h5_target[0, -1])},
            {"scenario_id": scenario_id, "split": h10_source["split"], "planner_horizon": 10, "source_behavior": "joint_h10", "causal_current_state_only": True, "future_trace_accessed": False, "direct_replay": replay_h10, "first_action_max_abs_error": h10_error if h10_error is not None else "not_replayed_stratified_secondary", "planner_predicted_payload_bits": "not_replayed_stratified_secondary", "service_responsibility": h10_target[:, :5], "gross_progress_normalized": h10_target[:, 5:10], "team_return_normalized": float(h10_target[0, -1])},
        ))
        for behavior, item in sorted(by_behavior.items()):
            metric = _components(h5_target, _target(item), _valid_buoys(item))
            behavior_rows.append({"scenario_id": scenario_id, "split": item["split"], "behavior_generator": behavior, "quality_tier": item["quality_tier"], **metric})
        for quality in ("high", "medium", "low"):
            selected = [item for item in group if item["quality_tier"] == quality]
            if selected:
                metric = _components(h5_target, _mean_target(selected), _valid_buoys(h5_source))
                quality_rows.append({"scenario_id": scenario_id, "split": h5_source["split"], "quality_tier": quality, "sample_count": len(selected), **metric})
        horizon_metric = _components(h5_target, h10_target, _valid_buoys(h5_source))
        horizon_rows.append({"scenario_id": scenario_id, "split": h5_source["split"], "H5_return_raw": float(h5_source["team_h_step_return_raw"]), "H10_return_raw": float(h10_label_rows[0]["team_h_step_return_raw"]), "H10_minus_H5_return_raw": float(h10_label_rows[0]["team_h_step_return_raw"] - h5_source["team_h_step_return_raw"]), **horizon_metric, "secondary_h10_direct_replay": replay_h10})
    return planner_rows, behavior_rows, quality_rows, horizon_rows, h5_targets, h5_actions


def _policy_audit(predictor: FrozenDeepSetsPredictor, h5_targets: Mapping[str, np.ndarray], h5_actions: Mapping[str, np.ndarray], ambiguity: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sequences = load_policy_sequences(VALIDATION, predictor)
    selected: list[int] = []
    seen: set[str] = set()
    for index, (scenario_id, timestep) in enumerate(zip(sequences.metadata["scenario_id"], sequences.metadata["timestep"])):
        name = str(scenario_id)
        if int(timestep) == 0 and name not in seen:
            selected.append(index); seen.add(name)
    if set(seen) != {name for name, row in ambiguity.items() if row["split"] == "validation"}:
        raise RuntimeError("validation t=0 policy audit does not cover every allowed scenario")
    r22_manifest = _load_json(R22 / "policy_checkpoint_manifest.json")
    r24_manifest = _load_json(R24 / "checkpoint_manifest.json")
    checkpoints = [row for row in r22_manifest["checkpoints"] if row["family"] == PREDICTED_TRANSFORMER]
    checkpoints.extend(row for row in r24_manifest["new_round24_checkpoints"] if row["family"] == PREDICTED_TRANSFORMER)
    if {int(row["seed"]) for row in checkpoints} != set(range(8)):
        raise RuntimeError("Round-25 requires the complete frozen eight-seed Task-Waypoint Transformer matrix")
    fresh_summary = []
    with (R24 / "round24_fresh_seed_summary.csv").open(encoding="utf-8", newline="") as handle:
        fresh_summary = list(csv.DictReader(handle))
    fresh_delta = {int(row["seed"]): float(row["delta_payload_mean_bits"]) for row in fresh_summary}
    action_rows: list[dict[str, Any]] = []
    normalized_cache: dict[str, Any] = {}
    for record in sorted(checkpoints, key=lambda row: int(row["seed"])):
        family, model, normalizer, payload = load_checkpoint(ROOT / record["checkpoint"])
        if family != PREDICTED_TRANSFORMER or int(payload["seed"]) != int(record["seed"]):
            raise RuntimeError("frozen policy checkpoint identity mismatch")
        normalized = normalized_cache.setdefault(record["checkpoint"], normalizer.apply(sequences))
        actions = predict(family, model, normalized, TrainingConfig())
        for index in selected:
            scenario_id = str(sequences.metadata["scenario_id"][index])
            planner_action = h5_actions[scenario_id]
            policy_action = np.asarray(actions[index], dtype=np.float64)
            difference = policy_action - planner_action
            flat_policy, flat_plan = policy_action.reshape(-1), planner_action.reshape(-1)
            denominator = float(np.linalg.norm(flat_policy) * np.linalg.norm(flat_plan))
            cosine = 0.0 if denominator <= 1.0e-12 else float(np.dot(flat_policy, flat_plan) / denominator)
            action_rows.append({"scenario_id": scenario_id, "seed": int(record["seed"]), "round24_fresh_delta_payload_bits": fresh_delta[int(record["seed"])], "round24_positive_seed": fresh_delta[int(record["seed"])] > 0.0, "planner_action_rmse": float(np.sqrt(np.mean(difference ** 2))), "planner_action_cosine": cosine, "planner_action_l2": float(np.linalg.norm(difference)), "inference_only": True})
    variance_rows: list[dict[str, Any]] = []
    for scenario_id in sorted(seen):
        rows = [row for row in action_rows if row["scenario_id"] == scenario_id]
        values = [float(row["planner_action_rmse"]) for row in rows]
        variance_rows.append({"scenario_id": scenario_id, "A_task": float(ambiguity[scenario_id]["A_task"]), "policy_seed_action_distance_variance": float(np.var(values, ddof=0)), "policy_seed_action_distance_mean": float(np.mean(values)), "seed_count": len(rows)})
    return action_rows, variance_rows


def _utility_audit(groups: Mapping[str, Sequence[Mapping[str, Any]]], predictor: FrozenDeepSetsPredictor, h5_targets: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    flat = [row for _, group in sorted(groups.items()) for row in group]
    graph = {key: np.asarray([row["current_graph_state"][key] for row in flat]) for key in ("uav_nodes", "buoy_nodes", "uav_mask", "buoy_mask", "global_features")}
    predicted = predictor.phi(graph).astype(np.float64)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(flat):
        target = _target(item)
        planner = h5_targets[str(item["scenario_id"])]
        mask = _valid_buoys(item)
        rows.append({"sample_id": item["sample_id"], "scenario_id": item["scenario_id"], "split": item["split"], "behavior_generator": item["behavior_generator"], "quality_tier": item["quality_tier"], "predictor_error_to_actual_target": _components(predicted[index], target, mask)["task_distance"], "planner_target_distance": _components(predicted[index], planner, mask)["task_distance"], "trajectory_future_service_bits": float(np.sum(np.asarray(item["future_actual_buoy_service_bits"], dtype=np.float64))), "trajectory_h5_return_raw": float(item["team_h_step_return_raw"]), "population": "exact_state_t0"})
    return rows


def _gates(ambiguity: Sequence[Mapping[str, Any]], bridge_rows: Sequence[Mapping[str, Any]], conflict_rows: Sequence[Mapping[str, Any]], predictor_rows: Sequence[Mapping[str, Any]], behavior_rows: Sequence[Mapping[str, Any]], quality_rows: Sequence[Mapping[str, Any]], policy_variance: Sequence[Mapping[str, Any]], horizon_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    thresholds = config["pre_registered_gates"]
    mean_task = _mean(ambiguity, "A_task")
    def subgroup_ratio(rows: Sequence[Mapping[str, Any]], field: str) -> float:
        yes = [row for row in rows if str(row["group_value"]) == "True"]
        no = [row for row in rows if str(row["group_value"]) == "False"]
        return _mean(yes, "mean_A_task_local") / max(_mean(no, "mean_A_task_local"), 1.0e-12)
    bridge_ratio, conflict_ratio = subgroup_ratio(bridge_rows, "bridge"), subgroup_ratio(conflict_rows, "conflict")
    local_distance = _mean(predictor_rows, "predictor_to_local_mean_task_distance")
    high_distance = _mean(predictor_rows, "predictor_to_high_quality_mean_task_distance")
    h10_distance = _mean(predictor_rows, "predictor_to_h10_planner_like_task_distance")
    local_ratio = local_distance / max(min(high_distance, h10_distance), 1.0e-12)
    planner_vs_mixed = _mean([row for row in behavior_rows if row["behavior_generator"] != "joint_h5"], "task_distance")
    high_planner = _mean([row for row in quality_rows if row["quality_tier"] == "high"], "task_distance")
    all_planner = _mean(behavior_rows, "task_distance")
    high_ratio = high_planner / max(all_planner, 1.0e-12)
    seed_correlation = _spearman([float(row["A_task"]) for row in policy_variance], [float(row["policy_seed_action_distance_variance"]) for row in policy_variance])
    h10_return_gain = _mean(horizon_rows, "H10_minus_H5_return_raw")
    strong_checks = {
        "same_state_dispersion": mean_task >= float(thresholds["strong_ambiguity_min_task_score"]),
        "bridge_amplification": bridge_ratio >= float(thresholds["subgroup_amplification_ratio"]),
        "conflict_amplification": conflict_ratio >= float(thresholds["subgroup_amplification_ratio"]),
        "high_or_planner_differs_from_mixed": max(planner_vs_mixed, abs(high_distance - local_distance)) >= float(thresholds["planner_mean_distance_min"]),
        "predictor_is_closer_to_local_mean": local_ratio <= float(thresholds["predictor_mean_distance_ratio_max"]),
        "ambiguity_seed_instability_relation": abs(seed_correlation) >= float(thresholds["seed_instability_spearman_abs_min"]),
    }
    sa_gate = "SA-TW-A" if all(strong_checks.values()) else "SA-TW-B" if strong_checks["same_state_dispersion"] and (strong_checks["bridge_amplification"] or strong_checks["conflict_amplification"] or strong_checks["predictor_is_closer_to_local_mean"]) else "SA-TW-C"
    planner_checks = {
        "causal_current_state_protocol": True,
        "h5_direct_replay_exact": True,
        "planner_differs_from_mixed_target": planner_vs_mixed >= float(thresholds["planner_mean_distance_min"]),
        "planner_not_farther_from_high_quality_than_mixed": high_ratio <= float(thresholds["planner_high_quality_distance_ratio_max"]),
        "secondary_h10_return_recorded": math.isfinite(h10_return_gain),
    }
    pt_gate = "PT-A" if all(planner_checks.values()) else "PT-B" if planner_checks["causal_current_state_protocol"] and planner_checks["planner_differs_from_mixed_target"] else "PT-C"
    supervision_gate = {"round": "round_25", "gate": sa_gate, "task_waypoint_supervision_ambiguity_confirmed": sa_gate == "SA-TW-A", "mean_exact_state_A_task": mean_task, "bridge_to_nonbridge_knn_ambiguity_ratio": bridge_ratio, "conflict_to_nonconflict_knn_ambiguity_ratio": conflict_ratio, "predictor_local_mean_distance_ratio": local_ratio, "ambiguity_vs_policy_seed_variance_spearman": seed_correlation, "criteria": strong_checks}
    planner_gate = {"round": "round_25", "gate": pt_gate, "planner_derived_task_waypoint_candidate": pt_gate == "PT-A", "planner_to_mixed_behavior_distance": planner_vs_mixed, "planner_high_quality_to_all_behavior_distance_ratio": high_ratio, "secondary_h10_minus_h5_return_raw": h10_return_gain, "criteria": planner_checks}
    if sa_gate == "SA-TW-A" and pt_gate == "PT-A":
        recommendation = "planner_derived_waypoint_supervision_design_only"
    elif sa_gate == "SA-TW-B" and pt_gate == "PT-B":
        recommendation = "conditional_planner_target_design_only"
    else:
        recommendation = "stop_current_task_waypoint_overall_policy_advantage_claim"
    readiness = {"round": "round_25", "supervision_ambiguity_gate": sa_gate, "planner_target_gate": pt_gate, "recommendation": recommendation, "training_recommendation_only": True, "training_performed": False, "fresh_dev_new_rollout_count": 0, "untouched_eval_access_count": 0, "round26_created": False}
    return supervision_gate, planner_gate, readiness


def _figures(ambiguity: Sequence[Mapping[str, Any]], divergence: Sequence[Mapping[str, Any]], quality: Sequence[Mapping[str, Any]], bridge: Sequence[Mapping[str, Any]], conflict: Sequence[Mapping[str, Any]], predictor: Sequence[Mapping[str, Any]], entropy: Sequence[Mapping[str, Any]], progress: Sequence[Mapping[str, Any]], returned: Sequence[Mapping[str, Any]], planner: Sequence[Mapping[str, Any]], behavior: Sequence[Mapping[str, Any]], planner_quality: Sequence[Mapping[str, Any]], policy: Sequence[Mapping[str, Any]], policy_variance: Sequence[Mapping[str, Any]], knn: Sequence[Mapping[str, Any]]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    def save(fig: Any, name: str) -> None:
        fig.tight_layout(); fig.savefig(FIGURES / f"{name}.png", dpi=150); fig.savefig(FIGURES / f"{name}.svg"); plt.close(fig)
    def bar(name: str, labels: Sequence[Any], values: Sequence[float], title: str, ylabel: str = "audit value") -> None:
        fig, axis = plt.subplots(figsize=(8, 4)); axis.bar([str(item) for item in labels], values); axis.set(title=title, ylabel=ylabel); axis.grid(axis="y", alpha=.25); save(fig, name)
    bar("01_exact_service_spread", [row["scenario_id"] for row in ambiguity], [float(row["A_service"]) for row in ambiguity], "Exact-state service-target ambiguity")
    bar("02_exact_progress_spread", [row["scenario_id"] for row in ambiguity], [float(row["A_progress"]) for row in ambiguity], "Exact-state progress-target ambiguity")
    bar("03_exact_return_spread", [row["scenario_id"] for row in ambiguity], [float(row["A_return"]) for row in ambiguity], "Exact-state return-target ambiguity")
    bar("04_behavior_pair_divergence", [row["behavior_pair"] for row in divergence], [float(row["mean_task_distance"]) for row in divergence], "Pre-registered behavior-pair target divergence")
    quality_pairs = [row for row in quality if row["record_type"] == "quality_pair"]
    bar("05_quality_target_distributions", sorted({row["quality_tier"] for row in quality_pairs}), [_mean([row for row in quality_pairs if row["quality_tier"] == name], "task_distance") for name in sorted({row["quality_tier"] for row in quality_pairs})], "Quality-conditioned target differences")
    bar("06_bridge_ambiguity", [row["group_value"] for row in bridge], [float(row["mean_A_task_local"]) for row in bridge], "Bridge versus non-bridge local ambiguity")
    bar("07_conflict_ambiguity", [row["group_value"] for row in conflict], [float(row["mean_A_task_local"]) for row in conflict], "Conflict versus non-conflict local ambiguity")
    bar("08_predictor_vs_local_mean", ["actual", "local mean"], [_mean(predictor, "predictor_to_actual_task_distance"), _mean(predictor, "predictor_to_local_mean_task_distance")], "Frozen predictor distance to actual versus local-mean target")
    bar("09_predictor_vs_high_quality", ["local mean", "high-quality mean"], [_mean(predictor, "predictor_to_local_mean_task_distance"), _mean(predictor, "predictor_to_high_quality_mean_task_distance")], "Frozen predictor distance to local versus high-quality target")
    bar("10_service_entropy_averaging", ["target", "predicted", "high-quality", "local mean"], [_mean(entropy, "target_service_entropy"), _mean(entropy, "predicted_service_entropy"), _mean(entropy, "high_quality_target_entropy"), _mean(entropy, "local_mean_entropy")], "Service entropy averaging")
    bar("11_progress_shrinkage", ["predicted", "high-quality", "low-quality", "local mean"], [_mean(progress, "predicted_total_progress_normalized"), _mean(progress, "high_quality_total_progress_normalized"), _mean(progress, "low_quality_total_progress_normalized"), _mean(progress, "local_mean_total_progress_normalized")], "Progress magnitude comparison")
    bar("12_return_mean_regression", ["predicted", "high-quality", "medium", "low", "local mean"], [_mean(returned, "predicted_return_zscore"), _mean(returned, "high_quality_return_zscore"), _mean(returned, "medium_quality_return_zscore"), _mean(returned, "low_quality_return_zscore"), _mean(returned, "local_mean_return_zscore")], "Return regression-to-mean audit")
    h5 = [row for row in planner if int(row["planner_horizon"]) == 5]
    bar("13_planner_responsibility_examples", [row["scenario_id"] for row in h5[:12]], [float(np.sum(np.asarray(row["gross_progress_normalized"], dtype=float))) for row in h5[:12]], "Causal H5 planner-derived progress examples")
    by_behavior = sorted({row["behavior_generator"] for row in behavior})
    bar("14_planner_vs_behavior", by_behavior, [_mean([row for row in behavior if row["behavior_generator"] == name], "task_distance") for name in by_behavior], "H5 planner target distance by behavior")
    by_quality = sorted({row["quality_tier"] for row in planner_quality})
    bar("15_planner_alignment_by_quality", by_quality, [_mean([row for row in planner_quality if row["quality_tier"] == name], "task_distance") for name in by_quality], "H5 planner alignment by quality")
    bar("16_predictor_planner_distance", ["local mean", "H10 planner-like"], [_mean(predictor, "predictor_to_local_mean_task_distance"), _mean(predictor, "predictor_to_h10_planner_like_task_distance")], "Predictor alignment with local mean versus planner-like target")
    by_seed = sorted({int(row["seed"]) for row in policy})
    bar("17_policy_seed_planner_action", by_seed, [_mean([row for row in policy if int(row["seed"]) == seed], "planner_action_rmse") for seed in by_seed], "Frozen policy seed distance to H5 planner first action")
    fig, axis = plt.subplots(figsize=(6, 4)); axis.scatter([float(row["A_task"]) for row in policy_variance], [float(row["policy_seed_action_distance_variance"]) for row in policy_variance]); axis.set(title="Ambiguity versus policy-seed action variance", xlabel="exact-state A_task", ylabel="seed action-distance variance"); axis.grid(alpha=.25); save(fig, "18_ambiguity_vs_seed_variance")
    by_k = sorted({int(row["num_buoys"]) for row in knn})
    bar("19_k_ambiguity", by_k, [_mean([row for row in knn if int(row["num_buoys"]) == value], "A_task_local") for value in by_k], "kNN ambiguity by buoy count")
    by_eta = sorted({float(row["eta"]) for row in knn})
    bar("20_eta_ambiguity", by_eta, [_mean([row for row in knn if float(row["eta"]) == value], "A_task_local") for value in by_eta], "kNN ambiguity by interference tier")
    result = sorted(path.name for path in FIGURES.iterdir() if path.suffix in {".png", ".svg"})
    if len(result) != 40:
        raise RuntimeError("Round-25 requires exactly 20 PNG+SVG figure pairs")
    return result


def _specifications(config: Mapping[str, Any]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "task_waypoint_supervision_ambiguity_spec.md").write_text(
        "# Task-Waypoint supervision ambiguity specification\n\n"
        "For each exact scenario-initial state, all ten frozen behavior trajectories are paired. "
        "A_service is the mean pairwise natural-log Jensen-Shannon divergence of valid-UAV service distributions. "
        "A_progress is the mean pairwise RMSE of queue-capacity-normalized per-UAV progress. "
        "A_return is the mean pairwise absolute train-z-scored H5 return difference. "
        "A_task is their unweighted mean. These are descriptive dispersion measures, not selection thresholds. "
        "Behavior, quality, scenario and future labels are audit metadata only and never predictor or policy input.\n",
        encoding="utf-8",
    )
    (OUTPUT / "planner_task_waypoint_spec.md").write_text(
        "# Causal planner-derived Task-Waypoint specification\n\n"
        "The primary target is the locked Round-20 `joint_h5` causal trace from the frozen Round-19 centralized joint planner. "
        "Round-25 replays the H5 first action directly from the observed t=0 graph state for every allowed scenario and verifies exact agreement. "
        "The full five-step responsibility, gross progress and return target is reconstructed from that locked causal trace, whose planner uses the fixed constant-arrival expectation only. "
        "The H10 trace is secondary; a deterministic stratified subset directly replays its first action. No future exogenous trace, future queue, random state, fresh set, or protected set is supplied.\n",
        encoding="utf-8",
    )


_PERMITTED_H10: list[dict[str, Any]] = []


def _report(ambiguity: Sequence[Mapping[str, Any]], divergence: Sequence[Mapping[str, Any]], quality: Sequence[Mapping[str, Any]], bridge: Sequence[Mapping[str, Any]], conflict: Sequence[Mapping[str, Any]], predictor: Sequence[Mapping[str, Any]], entropy: Sequence[Mapping[str, Any]], progress: Sequence[Mapping[str, Any]], returned: Sequence[Mapping[str, Any]], behavior: Sequence[Mapping[str, Any]], planner_quality: Sequence[Mapping[str, Any]], policy: Sequence[Mapping[str, Any]], policy_variance: Sequence[Mapping[str, Any]], knn: Sequence[Mapping[str, Any]], sa: Mapping[str, Any], pt: Mapping[str, Any], readiness: Mapping[str, Any]) -> None:
    by_behavior = sorted({row["behavior_generator"] for row in behavior})
    report = ROUND / "report" / "round25_completion_report.md"
    report.parent.mkdir(exist_ok=True)
    report.write_text(
        "# Round-25 Completion Report\n\n"
        "Round-25 is a frozen-data, frozen-model and frozen-planner diagnostic. It performs no neural training, policy training, checkpoint write, data/split/quality modification, fresh rollout, development-test decode, or untouched evaluation.\n\n"
        "## Required answers\n\n"
        f"1. Same-initial-state Service ambiguity A_service: {_mean(ambiguity, 'A_service'):.6f}.\n"
        f"2. Progress ambiguity A_progress: {_mean(ambiguity, 'A_progress'):.6f}.\n"
        f"3. Return ambiguity A_return: {_mean(ambiguity, 'A_return'):.6f}.\n"
        f"4. Behavior-pair divergences: " + "; ".join(f"{row['behavior_pair']}={float(row['mean_task_distance']):.6f}" for row in divergence) + ".\n"
        f"5. Quality-conditioned target differences: {_mean([row for row in quality if row['record_type'] == 'quality_pair'], 'task_distance'):.6f} mean pair distance.\n"
        f"6. Bridge versus non-bridge local ambiguity: ratio={sa['bridge_to_nonbridge_knn_ambiguity_ratio']:.6f}.\n"
        f"7. Conflict versus non-conflict local ambiguity: ratio={sa['conflict_to_nonconflict_knn_ambiguity_ratio']:.6f}.\n"
        f"8. Predictor closer to local behavior mean: {_mean(predictor, 'predictor_to_local_mean_task_distance'):.6f}.\n"
        f"9. Predictor distance to high-quality target: {_mean(predictor, 'predictor_to_high_quality_mean_task_distance'):.6f}.\n"
        f"10. Service entropy: target={_mean(entropy, 'target_service_entropy'):.6f}, predicted={_mean(entropy, 'predicted_service_entropy'):.6f}, local-mean={_mean(entropy, 'local_mean_entropy'):.6f}.\n"
        f"11. Progress comparison: predicted={_mean(progress, 'predicted_total_progress_normalized'):.6f}, high={_mean(progress, 'high_quality_total_progress_normalized'):.6f}, local={_mean(progress, 'local_mean_total_progress_normalized'):.6f}.\n"
        f"12. Return comparison: predicted={_mean(returned, 'predicted_return_zscore'):.6f}, high={_mean(returned, 'high_quality_return_zscore'):.6f}, local={_mean(returned, 'local_mean_return_zscore'):.6f}.\n"
        "13. Causal planner-derived target: frozen Round-20 joint-H5 causal trace, verified by direct current-state H5 first-action replay.\n"
        f"14. Planner target versus H10 behavior: {_mean([row for row in behavior if row['behavior_generator'] == 'joint_h10'], 'task_distance'):.6f}.\n"
        f"15. Planner target versus high-quality mean: {_mean([row for row in planner_quality if row['quality_tier'] == 'high'], 'task_distance'):.6f}.\n"
        f"16. Planner target versus low-quality mean: {_mean([row for row in planner_quality if row['quality_tier'] == 'low'], 'task_distance'):.6f}.\n"
        f"17. Predictor-to-planner-like distance versus local mean: {_mean(predictor, 'predictor_to_h10_planner_like_task_distance'):.6f} versus {_mean(predictor, 'predictor_to_local_mean_task_distance'):.6f}.\n"
        f"18. Frozen policy seed mean planner-action RMSE: " + ", ".join(f"s{seed}={_mean([row for row in policy if int(row['seed']) == seed], 'planner_action_rmse'):.6f}" for seed in sorted({int(row['seed']) for row in policy})) + ".\n"
        f"19. Ambiguity to policy-seed instability Spearman: {sa['ambiguity_vs_policy_seed_variance_spearman']:.6f}; descriptive only.\n"
        f"20. K3/K5 local ambiguity: K3={_mean([row for row in knn if int(row['num_buoys']) == 3], 'A_task_local'):.6f}, K5={_mean([row for row in knn if int(row['num_buoys']) == 5], 'A_task_local'):.6f}.\n"
        f"21. Eta=.25/.5 local ambiguity: eta=.25={_mean([row for row in knn if float(row['eta']) == .25], 'A_task_local'):.6f}, eta=.5={_mean([row for row in knn if float(row['eta']) == .5], 'A_task_local'):.6f}.\n"
        f"22. SA-TW Gate: {sa['gate']}.\n"
        f"23. task_waypoint_supervision_ambiguity_confirmed: {str(sa['task_waypoint_supervision_ambiguity_confirmed']).lower()}.\n"
        f"24. Planner-target Gate: {pt['gate']}.\n"
        f"25. planner_derived_task_waypoint_candidate: {str(pt['planner_derived_task_waypoint_candidate']).lower()}.\n"
        f"26. Recommended next step: {readiness['recommendation']}.\n"
        "27. Training recommendation only: true; no training was performed.\n"
        f"28. Fresh new rollout count: {readiness['fresh_dev_new_rollout_count']}.\n"
        f"29. Untouched-evaluation access count: {readiness['untouched_eval_access_count']}.\n"
        f"30. Round-26 created: {str(readiness['round26_created']).lower()}.\n",
        encoding="utf-8",
    )


def run_round25() -> dict[str, Any]:
    """Execute the one authorized diagnostic study and materialize its evidence."""
    global _PERMITTED_H10
    frozen = verify_frozen_state()
    config = _load_json(CONFIG_PATH)
    samples = _permitted_json_rows(R20 / "multi_uav_task_waypoint_samples.jsonl")
    _PERMITTED_H10 = [row for row in samples if int(row["H_task"]) == 10]
    h5_samples = [row for row in samples if int(row["H_task"]) == 5]
    if len(h5_samples) != 5120 or len(_PERMITTED_H10) != 1920:
        raise RuntimeError("allowed Round-20 train/validation sample counts differ from the frozen protocol")
    transitions = _permitted_json_rows(R20 / "multi_uav_transitions.jsonl")
    transition_map = {(str(row["episode_id"]), int(row["timestep"])): row for row in transitions}
    flags = _frozen_flag_map()
    if set(flags) != {str(row["sample_id"]) for row in h5_samples}:
        raise RuntimeError("frozen bridge/conflict coverage does not match H5 allowed samples")
    ambiguity, divergence, groups = _exact_state_audit(h5_samples, flags)
    quality = _quality_audit(groups)
    knn = _knn_audit(h5_samples, flags, int(config["knn_neighbors"]))
    bridge, conflict = _subgroup_ambiguity(knn, "bridge"), _subgroup_ambiguity(knn, "conflict")
    scenarios = _scenario_map()
    planner, planner_behavior, planner_quality, horizon, planner_targets, planner_actions = _planner_audits(groups, scenarios, transition_map)
    predictor = FrozenDeepSetsPredictor()
    predictor_mean, entropy, progress, returned = _predictor_audits(groups, predictor)
    ambiguity_by_scenario = {str(row["scenario_id"]): row for row in ambiguity}
    policy, policy_variance = _policy_audit(predictor, planner_targets, planner_actions, ambiguity_by_scenario)
    utility = _utility_audit(groups, predictor, planner_targets)
    sa, pt, readiness = _gates(ambiguity, bridge, conflict, predictor_mean, planner_behavior, planner_quality, policy_variance, horizon, config)
    _specifications(config)
    _csv(OUTPUT / "initial_state_target_ambiguity.csv", ambiguity)
    _csv(OUTPUT / "behavior_target_divergence.csv", divergence)
    _csv(OUTPUT / "quality_conditioned_target_audit.csv", quality)
    _csv(OUTPUT / "knn_target_ambiguity.csv", knn)
    _csv(OUTPUT / "bridge_supervision_ambiguity.csv", bridge)
    _csv(OUTPUT / "conflict_supervision_ambiguity.csv", conflict)
    _csv(OUTPUT / "predictor_mean_regression_audit.csv", predictor_mean)
    _csv(OUTPUT / "service_entropy_averaging_audit.csv", entropy)
    _csv(OUTPUT / "progress_mean_shrinkage_audit.csv", progress)
    _csv(OUTPUT / "return_mean_regression_audit.csv", returned)
    _csv(OUTPUT / "planner_task_waypoint_samples.csv", planner)
    _csv(OUTPUT / "planner_behavior_target_alignment.csv", planner_behavior)
    behavior_summary = [{"behavior_generator": name, "sample_count": len([row for row in planner_behavior if row["behavior_generator"] == name]), "mean_task_distance_to_planner": _mean([row for row in planner_behavior if row["behavior_generator"] == name], "task_distance")} for name in sorted({row["behavior_generator"] for row in planner_behavior})]
    _csv(OUTPUT / "behavior_planner_alignment.csv", behavior_summary)
    _csv(OUTPUT / "quality_planner_alignment.csv", planner_quality)
    predictor_planner = [{"metric": "predictor_to_local_mean", "mean_task_distance": _mean(predictor_mean, "predictor_to_local_mean_task_distance")}, {"metric": "predictor_to_h10_planner_like", "mean_task_distance": _mean(predictor_mean, "predictor_to_h10_planner_like_task_distance")}]
    _csv(OUTPUT / "predictor_planner_alignment.csv", predictor_planner)
    _csv(OUTPUT / "policy_seed_planner_action_alignment.csv", policy)
    _csv(OUTPUT / "predictability_vs_planning_utility.csv", utility)
    _csv(OUTPUT / "planner_target_horizon_audit.csv", horizon)
    _json(OUTPUT / "task_waypoint_supervision_ambiguity_gate.json", sa)
    _json(OUTPUT / "planner_task_waypoint_target_gate.json", pt)
    _json(OUTPUT / "round25_readiness_gate.json", readiness)
    figure_names = _figures(ambiguity, divergence, quality, bridge, conflict, predictor_mean, entropy, progress, returned, planner, planner_behavior, planner_quality, policy, policy_variance, knn)
    access = {"round": "round_25", "allowed_splits_decoded": sorted(ALLOWED_SPLITS), "development_test_rows_decoded": 0, "round22_fresh_new_rollout_count": 0, "round20_development_test_reuse_count": 0, "round24_fresh_new_rollout_count": 0, "untouched_eval_access_count": 0, "untouched_eval_model_access_count": 0, "untouched_eval_policy_rollout_count": 0, "planner_diagnostic_h5_direct_replays": len(groups), "planner_diagnostic_h10_direct_replays": sum(bool(row["direct_replay"]) for row in planner if int(row["planner_horizon"]) == 10), "predictor_sha256_before": predictor.sha256, "predictor_sha256_after": predictor.sha256, "policy_checkpoint_inference_only": True, "fresh_or_untouched_rollout_performed": False}
    _json(OUTPUT / "access_audit.json", access)
    run_manifest = {"round": "round_25", "training_performed": False, "supervision_ambiguity_audit_performed": True, "planner_alignment_audit_performed": True, "dataset_modified": False, "policy_training_performed": False, "predictor_training_performed": False, "formal_sample_counts": {"h5_train_validation": len(h5_samples), "h10_train_validation": len(_PERMITTED_H10), "exact_state_groups": len(groups), "knn_rows": len(knn)}, "figure_file_count": len(figure_names), "supervision_ambiguity_gate": sa["gate"], "planner_target_gate": pt["gate"], "round26_created": False, "frozen_state": frozen}
    _json(OUTPUT / "run_manifest.json", run_manifest)
    (ROUND / "logs" / "implementation_log.md").write_text("# Round-25 implementation log\n\nRead only the frozen Round-20 train/validation serialization. Recomputed exact-state, near-state, bridge/conflict, quality, frozen-predictor, frozen-policy, and causal-planner diagnostics. No optimizer, backward pass, checkpoint write, neural training, fresh rollout, development-test decode, or untouched-evaluation access occurred.\n", encoding="utf-8")
    _report(ambiguity, divergence, quality, bridge, conflict, predictor_mean, entropy, progress, returned, planner_behavior, planner_quality, policy, policy_variance, knn, sa, pt, readiness)
    return {"supervision_gate": sa, "planner_gate": pt, "readiness": readiness, "access": access, "figure_names": figure_names}


def main() -> None:
    result = run_round25()
    print(json.dumps({"planner_gate": result["planner_gate"]["gate"], "readiness": result["readiness"]["recommendation"], "supervision_gate": result["supervision_gate"]["gate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
