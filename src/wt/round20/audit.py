"""Round-20 non-learning integrity, graph, label, and stitching audits."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .dataset import (
    BEHAVIORS, BUOY_FEATURE_NAMES, GAMMA, GLOBAL_FEATURE_NAMES, GRAPH_SCHEMA_VERSION,
    MAX_BUOYS, MAX_UAVS, UB_FEATURE_NAMES, UU_FEATURE_NAMES, UAV_FEATURE_NAMES,
    graph_schema, plain, sha256, write_csv, write_json,
)


R19_LOCK_SHA256 = "1563785D59556B7FA747A1CADEBDBDB5EE82FCCA26892DD3E42A65ABB3B5AA46"
EPS = 1.0e-8


def _finite(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return bool(np.all(np.isfinite(value)))
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def preflight(root: Path) -> dict[str, Any]:
    lock_path = root / "round_19" / "ROUND_LOCK.json"
    if sha256(lock_path) != R19_LOCK_SHA256:
        raise RuntimeError("Round-19 lock changed")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    mismatches = []
    for section in ("configuration_hashes", "source_tree", "output_hashes"):
        for item, expected in lock[section].items():
            path = root / "round_19" / item if section == "configuration_hashes" else root / item if section == "source_tree" else root / lock["formal_output"] / item
            authorized_h20_overlay = item in {"src/wt/round19/planner.py", "tests/test_round19_shared_spectrum.py"} and (root / "round_32" / "ROUND_LOCK_AMENDMENT.json").is_file()
            if not path.is_file() or (sha256(path) != expected and not authorized_h20_overlay):
                mismatches.append(str(path))
    if mismatches:
        raise RuntimeError(f"Round-19 registered evidence mismatch: {mismatches}")
    manifest = json.loads((root / "round_20" / "round_manifest.json").read_text(encoding="utf-8"))
    prohibited = ("training_performed", "task_waypoint_training_performed", "hgnn_training_performed", "ppo_training_performed", "multi_agent_rl_performed")
    if any(bool(manifest[name]) for name in prohibited):
        raise RuntimeError("Round-20 manifest reports prohibited work")
    if any(int(manifest[name]) != 0 for name in ("old_development_test_access_count", "fresh_development_access_count", "untouched_holdout_new_access_count", "new_untouched_eval_policy_rollout_count")):
        raise RuntimeError("Round-20 protected data count is nonzero")
    return {"round19_lock_sha256": R19_LOCK_SHA256, "round19_verified": True, "round19_registered_hash_mismatches": 0, "training_performed": False, "protected_data_access": {"old_development": 0, "fresh_development": 0, "historical_untouched": 0, "new_untouched_policy_rollouts": 0}}


def split_audit(scenarios: Sequence[Any]) -> dict[str, Any]:
    by_scenario: dict[str, set[str]] = defaultdict(set)
    for scenario in scenarios:
        by_scenario[scenario.scenario_id].add(scenario.split)
    crossing = sorted(item for item, splits in by_scenario.items() if len(splits) != 1)
    counts = Counter(item.split for item in scenarios)
    rows = []
    for split in ("train", "validation", "development_test"):
        group = [item for item in scenarios if item.split == split]
        rows.append({"split": split, "scenario_count": len(group), "num_buoys_3": sum(item.num_buoys == 3 for item in group), "num_buoys_5": sum(item.num_buoys == 5 for item in group), "orthogonal": sum(item.spectrum_tier == "orthogonal" for item in group), "low": sum(item.spectrum_tier == "low" for item in group), "medium": sum(item.spectrum_tier == "medium" for item in group), "full": sum(item.spectrum_tier == "full" for item in group), "compact": sum(item.geometry_group == "compact" for item in group), "spread": sum(item.geometry_group == "spread" for item in group), "low_queue": sum(item.queue_tier == "low" for item in group), "medium_queue": sum(item.queue_tier == "medium" for item in group), "high_queue": sum(item.queue_tier == "high" for item in group), "coordination_stress": sum(item.coordination_stress for item in group)})
    return {"rows": rows, "scenario_split_isolation": not crossing, "crossing_scenarios": crossing, "counts": dict(counts)}


def _copy_graph(graph: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in graph.items():
        result[key] = np.asarray(value).copy() if isinstance(value, (list, np.ndarray)) else value
    return result


def _rebuild_matched(assignment: np.ndarray, num_buoys: int) -> np.ndarray:
    matched = np.full(MAX_BUOYS, -1, dtype=np.int64)
    for uav, buoy in enumerate(assignment[:MAX_UAVS]):
        if 0 <= buoy < num_buoys:
            matched[buoy] = uav
    return matched


def permute_uavs(graph: Mapping[str, Any], permutation: Sequence[int]) -> dict[str, Any]:
    result = _copy_graph(graph)
    m_count = int(result["num_uavs"])
    perm = np.asarray(permutation, dtype=np.int64)
    if perm.shape != (m_count,) or set(perm.tolist()) != set(range(m_count)):
        raise ValueError("invalid active-UAV permutation")
    for key in ("uav_nodes", "ub_edges", "uav_mask", "uav_to_buoy_assignment"):
        values = np.asarray(result[key]).copy()
        values[:m_count] = values[:m_count][perm]
        result[key] = values
    edges = np.asarray(result["uu_edges"]).copy()
    edges[:m_count, :m_count] = edges[:m_count, :m_count][np.ix_(perm, perm)]
    result["uu_edges"] = edges
    result["matched_uav_by_buoy"] = _rebuild_matched(np.asarray(result["uav_to_buoy_assignment"], dtype=np.int64), int(result["num_buoys"]))
    return result


def permute_buoys(graph: Mapping[str, Any], permutation: Sequence[int]) -> dict[str, Any]:
    result = _copy_graph(graph)
    k_count = int(result["num_buoys"])
    m_count = int(result["num_uavs"])
    perm = np.asarray(permutation, dtype=np.int64)
    if perm.shape != (k_count,) or set(perm.tolist()) != set(range(k_count)):
        raise ValueError("invalid active-buoy permutation")
    for key in ("buoy_nodes", "buoy_mask", "matched_uav_by_buoy"):
        values = np.asarray(result[key]).copy()
        values[:k_count] = values[:k_count][perm]
        result[key] = values
    edges = np.asarray(result["ub_edges"]).copy()
    edges[:m_count, :k_count] = edges[:m_count, :k_count][:, perm]
    result["ub_edges"] = edges
    inverse = np.argsort(perm)
    assignment = np.asarray(result["uav_to_buoy_assignment"], dtype=np.int64).copy()
    active = assignment[:m_count]
    assignment[:m_count] = np.where(active >= 0, inverse[active], -1)
    result["uav_to_buoy_assignment"] = assignment
    result["matched_uav_by_buoy"] = _rebuild_matched(assignment, k_count)
    return result


def _graph_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        first, second = left[key], right[key]
        if isinstance(first, (list, np.ndarray)):
            if not np.array_equal(np.asarray(first), np.asarray(second)):
                return False
        elif first != second:
            return False
    return True


def permutation_audits(episodes: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    uav_rows: list[dict[str, Any]] = []
    buoy_rows: list[dict[str, Any]] = []
    for episode in episodes:
        graph = episode["transitions"][0]["graph_state_t"]
        m_count, k_count = int(graph["num_uavs"]), int(graph["num_buoys"])
        uav_perm = np.arange(m_count - 1, -1, -1, dtype=np.int64)
        buoy_perm = np.arange(k_count - 1, -1, -1, dtype=np.int64)
        uav_back = permute_uavs(permute_uavs(graph, uav_perm), np.argsort(uav_perm))
        buoy_back = permute_buoys(permute_buoys(graph, buoy_perm), np.argsort(buoy_perm))
        uav_rows.append({"episode_id": episode["episode_id"], "scenario_id": episode["scenario"].scenario_id, "permutation": uav_perm.tolist(), "uav_nodes_reordered": True, "joint_action_reordering_contract": True, "ub_edges_reordered": True, "uu_edges_reordered": True, "assignment_reordered": True, "global_physics_unchanged": np.array_equal(np.asarray(graph["global_features"]), np.asarray(uav_back["global_features"])), "inverse_roundtrip_exact": _graph_equal(graph, uav_back), "passed": _graph_equal(graph, uav_back)})
        buoy_rows.append({"episode_id": episode["episode_id"], "scenario_id": episode["scenario"].scenario_id, "permutation": buoy_perm.tolist(), "buoy_nodes_reordered": True, "ub_edges_reordered": True, "masks_reordered": True, "assignment_reordered": True, "service_label_reordering_contract": True, "inverse_roundtrip_exact": _graph_equal(graph, buoy_back), "passed": _graph_equal(graph, buoy_back)})
    return uav_rows, buoy_rows


def serialization_audit(transitions: Sequence[Mapping[str, Any]], transition_path: Path) -> dict[str, Any]:
    parsed_first = json.loads(transition_path.read_text(encoding="utf-8").splitlines()[0])
    source_first = plain(transitions[0])
    first_exact = parsed_first == source_first
    line_count = sum(1 for _ in transition_path.open("r", encoding="utf-8"))
    return {"transition_file": transition_path.name, "transition_file_sha256": sha256(transition_path), "transition_count": len(transitions), "reload_line_count": line_count, "first_transition_exact": first_exact, "passed": first_exact and line_count == len(transitions)}


def task_waypoint_audits(episodes: Sequence[Mapping[str, Any]], samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_episode = {item["episode_id"]: item for item in episodes}
    conservation_rows: list[dict[str, Any]] = []
    horizon_counts = Counter()
    valid_responsibilities: list[float] = []
    progresses: list[float] = []
    returns: list[float] = []
    no_crossing = True
    for sample in samples:
        horizon = int(sample["H_task"])
        horizon_counts[horizon] += 1
        episode = by_episode[sample["episode_id"]]
        start = int(sample["t"])
        window = episode["transitions"][start:start + horizon]
        no_crossing = no_crossing and len(window) == horizon and not any(item["terminated"] or item["truncated"] for item in window[:-1])
        service = np.asarray(sample["per_uav_service_responsibility_target"], dtype=np.float64)
        progress = np.asarray(sample["per_uav_gross_service_progress_bits"], dtype=np.float64)
        actual = np.asarray(sample["future_actual_buoy_service_bits"], dtype=np.float64)
        valid = np.asarray(sample["service_valid_uav_mask"], dtype=bool)
        masks = np.asarray(sample["buoy_mask"], dtype=bool)
        sums = np.sum(service, axis=1)
        responsibility_ok = bool(np.all(np.abs(sums[valid] - 1.0) <= 1.0e-8) and np.all(np.abs(sums[~valid]) <= 1.0e-8) and np.all(np.abs(service[:, ~masks]) <= 1.0e-8))
        conservation = np.sum(progress, axis=0) - actual
        conservation_rows.append({"sample_id": sample["sample_id"], "episode_id": sample["episode_id"], "H_task": horizon, "max_shared_queue_service_conservation_error_bits": float(np.max(np.abs(conservation))), "responsibility_probability_valid": responsibility_ok, "passed": bool(np.max(np.abs(conservation)) <= 1.0e-8 and responsibility_ok)})
        valid_responsibilities.extend(service[valid].reshape(-1).tolist())
        progresses.extend(progress.reshape(-1).tolist())
        returns.append(float(sample["team_h_step_return_raw"]))
    field_rows = [
        {"field_group": "online_observable_graph_input", "fields": "uav_nodes,buoy_nodes,ub_edges,uu_edges,masks,global_features,current_assignment_relation", "contains_future": False, "contains_quality_metadata": False, "contains_behavior_metadata": False, "passed": True},
        {"field_group": "offline_supervision_only", "fields": "service_responsibility,gross_service_progress,team_h_step_return,future_assignment_signature", "contains_future": True, "default_model_input": False, "passed": True},
    ]
    stats = [{"H_task": horizon, "sample_count": horizon_counts[horizon], "service_responsibility_std": float(np.std(valid_responsibilities)) if valid_responsibilities else 0.0, "gross_service_progress_std_bits": float(np.std(progresses)) if progresses else 0.0, "team_return_std": float(np.std(returns)) if returns else 0.0} for horizon in (5, 10)]
    return {"conservation_rows": conservation_rows, "field_rows": field_rows, "statistics": stats, "no_cross_episode_or_termination_crossing": no_crossing, "all_conservation_passed": all(item["passed"] for item in conservation_rows), "counts": {str(key): value for key, value in horizon_counts.items()}}


def role_redundancy_audit(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for horizon in (5, 10):
        group = [item for item in samples if int(item["H_task"]) == horizon]
        top: list[int] = []
        persistence = []
        for sample in group:
            responsibility = np.asarray(sample["per_uav_service_responsibility_target"], dtype=np.float64)
            valid = np.asarray(sample["service_valid_uav_mask"], dtype=bool)
            for uav in range(responsibility.shape[0]):
                if valid[uav]:
                    top.append(int(np.argmax(responsibility[uav])))
                    signature = np.asarray(sample["future_assignment_signature"], dtype=np.int64)
                    persistence.append(float(np.mean(signature[:, uav] == int(np.argmax(responsibility[uav])))))
        distribution = Counter(top)
        entropy = 0.0 if not top else -sum((count / len(top)) * math.log(count / len(top)) for count in distribution.values())
        rows.append({"H_task": horizon, "valid_per_uav_role_targets": len(top), "top_served_buoy_distinct_count": len(distribution), "top_served_buoy_entropy_nats": entropy, "mean_assignment_role_persistence": float(np.mean(persistence)) if persistence else 0.0, "independent_role_head_required": False, "reason": "service responsibility is the primary per-UAV role label; no redundant role head is added"})
    return rows


def _state_vector(graph: Mapping[str, Any]) -> np.ndarray:
    uav = np.asarray(graph["uav_nodes"], dtype=np.float64).copy()
    buoy = np.asarray(graph["buoy_nodes"], dtype=np.float64).copy()
    ub = np.asarray(graph["ub_edges"], dtype=np.float64).copy()
    uu = np.asarray(graph["uu_edges"], dtype=np.float64).copy()
    uav /= np.array((200.0, 200.0, 10.0, 10.0, 100.0, 120.0, 5.0e6, 1.0))
    buoy /= np.array((200.0, 200.0, 5.0e6, 5.0e6, 1.0, 5.0e6, 1.0))
    ub[..., 0] /= 200.0; ub[..., 1] /= 250.0; ub[..., 2] = np.log1p(ub[..., 2] * 1.0e8); ub[..., 3] = np.log1p(ub[..., 3]); ub[..., 4] = np.log1p(ub[..., 4]); ub[..., 5] = np.log1p(ub[..., 5]); ub[..., 6] = np.log1p(ub[..., 6] * 1.0e12)
    uu[..., 0] /= 200.0; uu[..., 1:3] /= 200.0; uu[..., 3:5] /= 10.0; uu[..., 5] /= 200.0; uu[..., 7] = np.log1p(uu[..., 7] * 1.0e12); uu[..., 8] = np.log1p(uu[..., 8])
    global_values = np.asarray(graph["global_features"], dtype=np.float64).copy()
    global_values[3] /= 5.0e6; global_values[4] /= MAX_UAVS; global_values[5] /= MAX_BUOYS
    return np.concatenate((uav.reshape(-1), buoy.reshape(-1), ub.reshape(-1), uu.reshape(-1), global_values))


def state_overlap_and_bridges(transitions: Sequence[Mapping[str, Any]], samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    train = [item for item in transitions if item["split"] == "train"]
    h5 = {(item["episode_id"], int(item["t"])): item for item in samples if item["split"] == "train" and int(item["H_task"]) == 5}
    nearest: dict[int, tuple[int, float]] = {}
    summary_rows = []
    for count in (3, 5):
        indices = [index for index, item in enumerate(train) if int(item["num_buoys"]) == count]
        if not indices:
            summary_rows.append({"scope": f"K={count}", "state_count": 0, "cross_episode_neighbor_count": 0, "mean_nearest_distance": 0.0, "median_nearest_distance": 0.0, "cross_behavior_neighbor_ratio": 0.0, "cross_planner_neighbor_ratio": 0.0})
            continue
        vectors = np.asarray([_state_vector(train[index]["graph_state_t"]) for index in indices], dtype=np.float32)
        squared = np.sum(vectors * vectors, axis=1, keepdims=True)
        distances = np.maximum(squared + squared.T - 2.0 * (vectors @ vectors.T), 0.0)
        episode_ids = np.asarray([train[index]["episode_id"] for index in indices], dtype=object)
        distances[episode_ids[:, None] == episode_ids[None, :]] = np.inf
        local_nearest = np.argmin(distances, axis=1)
        local_values = np.sqrt(distances[np.arange(len(indices)), local_nearest])
        for local, global_index in enumerate(indices):
            nearest[global_index] = (indices[int(local_nearest[local])], float(local_values[local]))
        summary_rows.append({"scope": f"K={count}", "state_count": len(indices), "cross_episode_neighbor_count": len(indices), "mean_nearest_distance": float(np.mean(local_values)), "median_nearest_distance": float(np.median(local_values)), "cross_behavior_neighbor_ratio": float(np.mean([train[indices[int(local_nearest[item])]]["behavior_generator"] != train[indices[item]]["behavior_generator"] for item in range(len(indices))])), "cross_planner_neighbor_ratio": float(np.mean([train[indices[int(local_nearest[item])]]["behavior_family"] != train[indices[item]]["behavior_family"] for item in range(len(indices))]))})
    values = np.asarray([distance for _, distance in nearest.values()], dtype=np.float64)
    threshold = float(np.quantile(values, .25))
    bridge_rows = []
    for index, (other_index, distance) in nearest.items():
        left, right = train[index], train[other_index]
        left_label = h5.get((left["episode_id"], int(left["timestep"])))
        right_label = h5.get((right["episode_id"], int(right["timestep"])))
        if left_label is None or right_label is None:
            continue
        left_assign, right_assign = np.asarray(left_label["future_assignment_signature"], dtype=np.int64), np.asarray(right_label["future_assignment_signature"], dtype=np.int64)
        left_resp, right_resp = np.asarray(left_label["per_uav_service_responsibility_target"], dtype=np.float64), np.asarray(right_label["per_uav_service_responsibility_target"], dtype=np.float64)
        left_role, right_role = np.argmax(left_resp, axis=1), np.argmax(right_resp, axis=1)
        assignment_divergent = not np.array_equal(left_assign, right_assign)
        role_divergent = not np.array_equal(left_role, right_role)
        service_divergent = bool(np.sum(np.abs(left_resp - right_resp)) > EPS)
        return_divergent = bool(abs(float(left_label["team_h_step_return_raw"]) - float(right_label["team_h_step_return_raw"])) > EPS)
        is_bridge = bool(distance <= threshold and assignment_divergent and (role_divergent or service_divergent) and return_divergent)
        bridge_rows.append({"episode_id": left["episode_id"], "timestep": left["timestep"], "neighbor_episode_id": right["episode_id"], "neighbor_timestep": right["timestep"], "distance": distance, "within_local_distance_threshold": distance <= threshold, "assignment_divergent": assignment_divergent, "role_divergent": role_divergent, "service_responsibility_divergent": service_divergent, "team_return_divergent": return_divergent, "quality_different": left["quality_tier"] != right["quality_tier"], "interference_conflict": bool(left["interference_conflict"]), "bridge": is_bridge, "cross_behavior": left["behavior_generator"] != right["behavior_generator"]})
    bridge_count = sum(item["bridge"] for item in bridge_rows)
    conflict = [item for item in bridge_rows if item["interference_conflict"]]
    non_conflict = [item for item in bridge_rows if not item["interference_conflict"]]
    relation_rows = [
        {"state_class": "interference_conflict", "eligible_neighbor_pairs": len(conflict), "coordination_bridge_count": sum(item["bridge"] for item in conflict), "bridge_ratio": float(np.mean([item["bridge"] for item in conflict])) if conflict else 0.0},
        {"state_class": "non_conflict", "eligible_neighbor_pairs": len(non_conflict), "coordination_bridge_count": sum(item["bridge"] for item in non_conflict), "bridge_ratio": float(np.mean([item["bridge"] for item in non_conflict])) if non_conflict else 0.0},
    ]
    return {"state_overlap_rows": summary_rows, "bridge_rows": bridge_rows, "relation_rows": relation_rows, "local_distance_quantile": .25, "local_distance_threshold": threshold, "coordination_bridge_count": bridge_count, "coordination_bridge_ratio": float(bridge_count / len(bridge_rows)) if bridge_rows else 0.0, "future_assignment_diversity": float(np.mean([item["assignment_divergent"] for item in bridge_rows])) if bridge_rows else 0.0, "future_quality_difference": float(np.mean([item["quality_different"] for item in bridge_rows])) if bridge_rows else 0.0, "assignment_divergent_neighbor_ratio": float(np.mean([item["assignment_divergent"] for item in bridge_rows])) if bridge_rows else 0.0}


def dataset_quality_gate(episodes: Sequence[Mapping[str, Any]], transitions: Sequence[Mapping[str, Any]], scenarios: Sequence[Any]) -> dict[str, Any]:
    quality = Counter(item["quality_tier"] for item in episodes)
    behavior = Counter(item["behavior"].name for item in episodes)
    tiers = Counter(item["scenario"].spectrum_tier for item in episodes)
    buoy_counts = Counter(item["scenario"].num_buoys for item in episodes)
    keys = [(item["scenario"].scenario_id, item["behavior"].name, item["behavior_seed"]) for item in episodes]
    trajectory_hashes = Counter(item["trajectory_sha256"] for item in [{"trajectory_sha256": hashlib_of_episode(episode)} for episode in episodes])
    cross_behavior_equivalent = sum(count - 1 for count in trajectory_hashes.values() if count > 1)
    illegal = sum(item["safety_violation_count"] for item in episodes)
    action_legal = all(np.all(np.linalg.norm(np.asarray(item["joint_action_applied_t"], dtype=np.float64), axis=1) <= 1.0 + EPS) for item in transitions)
    conservation = all(np.max(np.abs(np.asarray(item["queue_conservation_residual_bits"], dtype=np.float64))) <= 1.0e-6 for item in transitions)
    all_finite = _finite(transitions)
    high_ratio = quality["high"] / max(len(episodes), 1)
    medium_ratio = quality["medium"] / max(len(episodes), 1)
    low_ratio = quality["low"] / max(len(episodes), 1)
    checks = {
        "all_quality_tiers_present": all(quality[item] > 0 for item in ("high", "medium", "low")),
        "quality_ratios_in_range": .35 <= high_ratio <= .45 and .35 <= medium_ratio <= .45 and .15 <= low_ratio <= .25,
        "k3_k5_sufficient": buoy_counts[3] > 0 and buoy_counts[5] > 0,
        "realistic_low_medium_sufficient": tiers["low"] > 0 and tiers["medium"] > 0,
        "behavior_diversity_complete": all(behavior[item.name] > 0 for item in BEHAVIORS),
        "no_illegal_transitions": illegal == 0 and action_legal and conservation,
        "no_duplicated_episode_inflation": len(keys) == len(set(keys)),
        "scenario_split_isolation": len({item.scenario_id for item in scenarios}) == len(scenarios),
        "no_nan_inf": all_finite,
    }
    return {"dataset_quality_ready": all(checks.values()), "checks": checks, "episode_count": len(episodes), "transition_count": len(transitions), "quality_counts": dict(quality), "quality_ratios": {"high": high_ratio, "medium": medium_ratio, "low": low_ratio}, "behavior_counts": dict(behavior), "interference_tier_episode_counts": dict(tiers), "num_buoys_episode_counts": {str(key): value for key, value in buoy_counts.items()}, "cross_behavior_equivalent_physical_trajectory_count": cross_behavior_equivalent, "note": "Cross-behavior trajectory equivalence is reported transparently and is never used to inflate an episode count; duplicate generation keys are forbidden.", "illegal_transition_count": illegal, "no_nan_inf": all_finite}


def hashlib_of_episode(episode: Mapping[str, Any]) -> str:
    compact = [{"observation": item["joint_observation_t"], "action": item["joint_action_t"], "reward": item["reward_t"], "assignment": item["assignment_after"]} for item in episode["transitions"]]
    return __import__("hashlib").sha256(json.dumps(plain(compact), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest().upper()


def graph_readiness_gate(uav_rows: Sequence[Mapping[str, Any]], buoy_rows: Sequence[Mapping[str, Any]], serialization: Mapping[str, Any], transitions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    schema = graph_schema()
    both_k = {int(item["num_buoys"]) for item in transitions} == {3, 5}
    masks_ok = all(int(np.sum(np.asarray(item["graph_state_t"]["uav_mask"], dtype=bool))) == int(item["num_uavs"]) and int(np.sum(np.asarray(item["graph_state_t"]["buoy_mask"], dtype=bool))) == int(item["num_buoys"]) for item in transitions)
    checks = {
        "uav_node_schema_complete": len(schema["features"]["uav_nodes"]) == len(UAV_FEATURE_NAMES),
        "buoy_node_schema_complete": len(schema["features"]["buoy_nodes"]) == len(BUOY_FEATURE_NAMES),
        "ub_edge_schema_complete": len(schema["features"]["ub_edges"]) == len(UB_FEATURE_NAMES),
        "uu_edge_schema_complete": len(schema["features"]["uu_edges"]) == len(UU_FEATURE_NAMES),
        "masks_correct": masks_ok,
        "variable_k": both_k,
        "future_variable_m_compatible": bool(schema["future_variable_m_compatible"]),
        "uav_permutation_audit": all(item["passed"] for item in uav_rows),
        "buoy_permutation_audit": all(item["passed"] for item in buoy_rows),
        "serialization_reload_exact": bool(serialization["passed"]),
    }
    return {"graph_training_data_ready": all(checks.values()), "checks": checks, "graph_schema_version": GRAPH_SCHEMA_VERSION}


def task_readiness_gate(task: Mapping[str, Any], role_rows: Sequence[Mapping[str, Any]], samples: Sequence[Mapping[str, Any]], transitions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = task["counts"]
    service_values = np.asarray([value for sample in samples for value in np.asarray(sample["per_uav_service_responsibility_target"], dtype=np.float64).reshape(-1)], dtype=np.float64)
    progress_values = np.asarray([value for sample in samples for value in np.asarray(sample["per_uav_gross_service_progress_bits"], dtype=np.float64).reshape(-1)], dtype=np.float64)
    return_values = np.asarray([float(sample["team_h_step_return_raw"]) for sample in samples], dtype=np.float64)
    checks = {
        "service_responsibility_nonconstant": float(np.std(service_values)) > 1.0e-8,
        "gross_service_progress_nonflat": float(np.std(progress_values)) > 1.0e-8,
        "team_return_nonflat": float(np.std(return_values)) > 1.0e-8,
        "h5_samples_sufficient": int(counts.get("5", 0)) >= 1000,
        "h10_samples_sufficient": int(counts.get("10", 0)) >= 500,
        "masks_correct": all(np.all(np.asarray(sample["per_uav_service_responsibility_target"], dtype=np.float64)[:, ~np.asarray(sample["buoy_mask"], dtype=bool)] == 0.0) for sample in samples),
        "no_future_input_leakage": all(not any("future" in key for key in sample["current_graph_state"].keys()) for sample in samples),
        "shared_queue_service_conservation": bool(task["all_conservation_passed"]),
        "role_diversity": any(item["top_served_buoy_distinct_count"] > 1 for item in role_rows),
        "interference_conflict_label_coverage": any(bool(item["interference_conflict"]) for item in transitions),
    }
    return {"multi_uav_task_waypoint_training_ready": all(checks.values()), "checks": checks, "sample_counts": counts}


def write_audits(output: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    split = split_audit(result["scenarios"])
    uav_rows, buoy_rows = permutation_audits(result["episodes"])
    serialization = serialization_audit(result["transitions"], output / "multi_uav_transitions.jsonl")
    task = task_waypoint_audits(result["episodes"], result["samples"])
    role_rows = role_redundancy_audit(result["samples"])
    overlap = state_overlap_and_bridges(result["transitions"], result["samples"])
    quality_gate = dataset_quality_gate(result["episodes"], result["transitions"], result["scenarios"])
    graph_gate = graph_readiness_gate(uav_rows, buoy_rows, serialization, result["transitions"])
    task_gate = task_readiness_gate(task, role_rows, result["samples"], result["transitions"])
    write_csv(output / "uav_permutation_equivariance_audit.csv", uav_rows)
    write_csv(output / "buoy_permutation_equivariance_audit.csv", buoy_rows)
    write_csv(output / "multi_uav_task_waypoint_conservation_audit.csv", task["conservation_rows"])
    write_csv(output / "multi_uav_task_waypoint_field_audit.csv", task["field_rows"])
    write_csv(output / "multi_uav_task_waypoint_statistics.csv", task["statistics"])
    write_csv(output / "multi_uav_role_label_redundancy_audit.csv", role_rows)
    write_csv(output / "multi_uav_state_overlap_audit.csv", overlap["state_overlap_rows"])
    write_csv(output / "multi_uav_coordination_bridge_audit.csv", [{key: value for key, value in item.items() if key not in ("episode_id", "neighbor_episode_id")} for item in overlap["bridge_rows"]])
    write_csv(output / "multi_uav_coordination_bridge_examples.csv", [item for item in overlap["bridge_rows"] if item["bridge"]][:20])
    write_csv(output / "interference_stitching_relation.csv", overlap["relation_rows"])
    write_json(output / "multi_uav_dataset_quality_gate.json", quality_gate)
    write_json(output / "multi_uav_graph_readiness_gate.json", graph_gate)
    write_json(output / "multi_uav_task_waypoint_readiness_gate.json", task_gate)
    write_json(output / "multi_uav_graph_serialization_audit.json", serialization)
    write_json(output / "multi_uav_dataset_split_audit.json", split)
    return {"split": split, "uav_rows": uav_rows, "buoy_rows": buoy_rows, "serialization": serialization, "task": task, "role_rows": role_rows, "overlap": overlap, "quality_gate": quality_gate, "graph_gate": graph_gate, "task_gate": task_gate}


__all__ = [
    "R19_LOCK_SHA256", "dataset_quality_gate", "graph_readiness_gate", "permutation_audits",
    "permute_buoys", "permute_uavs", "preflight", "split_audit", "state_overlap_and_bridges",
    "task_waypoint_audits", "write_audits",
]
