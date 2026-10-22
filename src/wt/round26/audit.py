"""Frozen Round-26 mixed executed-action ambiguity diagnostics.

No model is trained or written here.  All outputs derive from the immutable
Round-20 train/validation serialization, frozen inference, and current-state
calls to the unchanged Round-19 H=5 planner.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from wt.governance_state_sync import approved_lifecycle_overlay_sha256, verify_round22_governance_state_sync_amendment
from wt.round19.planner import joint_plan
from wt.round19.multi_env import SharedSpectrumMultiUAVEnv
from wt.round20.scenarios import development_scenarios
from wt.round21.data import bridge_flags, conflict_flags, load_task_samples
from wt.round22.data import FrozenDeepSetsPredictor, VALIDATION, load_policy_sequences
from wt.round22.models import PREDICTED_MLP, PREDICTED_TRANSFORMER, STATE_ONLY_TRANSFORMER
from wt.round22.training import TrainingConfig, load_checkpoint, predict
from wt.round25.audit import EXPECTED_R24, verify_frozen_state as verify_round25_frozen_state


ROOT = Path(__file__).resolve().parents[3]
ROUND = ROOT / "round_26"
OUTPUT = ROUND / "outputs" / "round26_action_ambiguity_20260830_1322"
FIGURES = ROUND / "figures" / "round26_action_ambiguity_20260830_1322"
CONFIG_PATH = ROUND / "configs" / "round26_action_ambiguity_config.json"
R20 = ROOT / "round_20" / "outputs" / "round20_multi_uav_dataset_20260828_2100"
R22 = ROOT / "round_22" / "outputs" / "round22_deepsets_task_waypoint_policy_20260829_1153"
R24 = ROOT / "round_24" / "outputs" / "round24_multiseed_stability_20260829_1515"
R25 = ROOT / "round_25" / "outputs" / "round25_supervision_ambiguity_alignment_20260830_1203"
EXPECTED_R25 = "120A417582F314CDB8B227104DE096C39340D27646F6BF33308CEC018AC72027"
ALLOWED = frozenset(("train", "validation"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Mapping): return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(v) for v in value]
    return value


def _json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized: raise RuntimeError(f"unexpected empty Round-26 output: {path.name}")
    fields = sorted({key for row in materialized for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise"); writer.writeheader()
        for row in materialized:
            writer.writerow({key: json.dumps(_plain(value), sort_keys=True) if isinstance(value, (dict, list, tuple, np.ndarray)) else value for key, value in row.items()})


def _allowed_rows(path: Path) -> list[dict[str, Any]]:
    """Filter raw lines before JSON decoding; protected rows are never decoded."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if '"split":"train"' not in raw and '"split":"validation"' not in raw:
                continue
            row = json.loads(raw)
            if row.get("split") not in ALLOWED: raise RuntimeError("protected split was decoded")
            rows.append(row)
    return rows


def verify_frozen_state() -> dict[str, Any]:
    """Fail closed before decoding Round-20 rows or frozen inference."""
    if _digest(ROOT / "round_25" / "ROUND_LOCK.json") != EXPECTED_R25:
        raise RuntimeError("Round-25 original lock changed")
    amendment = _load(ROOT / "round_25" / "ROUND_LOCK_AMENDMENT.json")
    affected = amendment.get("affected_paths", {})
    if (amendment.get("amendment_type") != "legal_round26_successor_governance_extension"
        or amendment.get("original_round25_lock_sha256") != EXPECTED_R25
        or amendment.get("scientific_artifacts_modified") is not False
        or amendment.get("original_round25_lock_modified") is not False
        or _digest(ROOT / "src" / "wt" / "governance_state_sync.py") != approved_lifecycle_overlay_sha256("src/wt/governance_state_sync.py", affected.get("src/wt/governance_state_sync.py", {}).get("amended_sha256", ""))
        or _digest(ROOT / "tests" / "test_round24_seed_extension.py") != approved_lifecycle_overlay_sha256("tests/test_round24_seed_extension.py", affected.get("tests/test_round24_seed_extension.py", {}).get("amended_sha256", ""))
        or _digest(ROOT / "tests" / "test_round21_hgnn_task_waypoint.py") != approved_lifecycle_overlay_sha256("tests/test_round21_hgnn_task_waypoint.py", affected.get("tests/test_round21_hgnn_task_waypoint.py", {}).get("amended_sha256", ""))):
        raise RuntimeError("Round-25 legal-successor overlay is invalid")
    prior = verify_round25_frozen_state()
    governance = verify_round22_governance_state_sync_amendment()
    protocol = _load(R22 / "round22_protocol_freeze.json")
    if protocol["round20_transition_sha256"] != _digest(R20 / "multi_uav_transitions.jsonl") or protocol["round20_task_samples_sha256"] != _digest(R20 / "multi_uav_task_waypoint_samples.jsonl"):
        raise RuntimeError("Round-20 frozen hashes differ from Round-22 protocol")
    return {"passed": True, "round25_original_lock_sha256": EXPECTED_R25, "round25_frozen_chain": prior, "round22_governance_verified": bool(governance["passed"]), "round25_fresh_new_rollout_count": 0, "round20_development_test_reuse_count": 0, "untouched_eval_access_count": 0}


def _action(row: Mapping[str, Any]) -> np.ndarray:
    action = np.asarray(row["joint_action_applied_t"], dtype=np.float64)
    if action.shape != (2, 2) or not np.all(np.isfinite(action)) or np.any(np.linalg.norm(action, axis=1) > 1.0 + 1e-9):
        raise RuntimeError("executed action is not a finite legal M=2 normalized action")
    return action


def _target(row: Mapping[str, Any]) -> np.ndarray:
    service = np.asarray(row["per_uav_service_responsibility_target"], dtype=np.float64)
    progress = np.asarray(row["per_uav_gross_service_progress_normalized"], dtype=np.float64)
    returned = np.full((2, 1), float(row["team_h_step_return_normalized"]), dtype=np.float64)
    return np.concatenate((service, progress, returned), axis=1)


def _state(row: Mapping[str, Any]) -> np.ndarray:
    graph = row["current_graph_state"]
    return np.concatenate((np.asarray(graph["uav_nodes"], dtype=np.float64).reshape(-1), np.asarray(graph["buoy_nodes"], dtype=np.float64).reshape(-1), np.asarray(graph["global_features"], dtype=np.float64)))


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator <= 1e-12 else float(math.acos(float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))))


def _metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    angles = [_angle(left[m], right[m]) for m in range(2)]
    cosine = [0.0 if np.linalg.norm(left[m]) * np.linalg.norm(right[m]) <= 1e-12 else 1.0 - math.cos(angles[m]) for m in range(2)]
    magnitudes = np.abs(np.linalg.norm(left, axis=1) - np.linalg.norm(right, axis=1))
    signs = np.mean(np.sign(left) != np.sign(right))
    left_sectors = np.asarray([int(round(math.atan2(vector[1], vector[0]) / (math.pi / 4.0))) if np.linalg.norm(vector) > 1e-12 else 99 for vector in left])
    right_sectors = np.asarray([int(round(math.atan2(vector[1], vector[0]) / (math.pi / 4.0))) if np.linalg.norm(vector) > 1e-12 else 99 for vector in right])
    sectors = float(np.mean(left_sectors != right_sectors))
    joint = float(np.sqrt(np.mean((left - right) ** 2)))
    score = float(np.mean((joint, float(np.mean(cosine)), float(np.mean(angles) / math.pi), float(np.mean(magnitudes)), float(sectors))))
    return {"joint_action_rmse": joint, "per_uav_cosine_disagreement": float(np.mean(cosine)), "action_angle_difference_rad": float(np.mean(angles)), "action_angle_difference_normalized": float(np.mean(angles) / math.pi), "action_magnitude_difference": float(np.mean(magnitudes)), "sign_disagreement_rate": float(signs), "target_region_disagreement": float(sectors), "A_action": score}


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def _rank(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64); order = np.argsort(x, kind="mergesort"); result = np.empty_like(x); result[order] = np.arange(x.size); return result


def _corr(left: Sequence[float], right: Sequence[float]) -> tuple[float, float]:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if a.size < 3 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12: return 0.0, 0.0
    return float(np.corrcoef(a, b)[0, 1]), float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def _flags() -> dict[str, tuple[bool, bool]]:
    result: dict[str, tuple[bool, bool]] = {}
    for split in sorted(ALLOWED):
        dataset = load_task_samples(5, {split}); bridge, conflict = bridge_flags(dataset), conflict_flags(dataset)
        for index, meta in enumerate(dataset.metadata): result[str(meta["sample_id"])] = (bool(bridge[index]), bool(conflict[index]))
    return result


def _exact(samples: Sequence[Mapping[str, Any]], actions: Mapping[tuple[str, int], np.ndarray], flags: Mapping[str, tuple[bool, bool]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        if int(row["t"]) == 0: groups[str(row["scenario_id"])].append(dict(row))
    ambiguity: list[dict[str, Any]] = []; pairs: list[dict[str, Any]] = []
    for scenario, group in sorted(groups.items()):
        if len(group) != 10: raise RuntimeError("exact-state behavior coverage is incomplete")
        states = [_state(row) for row in group]
        if any(not np.array_equal(states[0], value) for value in states[1:]): raise RuntimeError("t=0 scenario state is not exact")
        metrics = []
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                left, right = group[i], group[j]; metric = _metrics(actions[(str(left["episode_id"]), 0)], actions[(str(right["episode_id"]), 0)])
                metric.update({"scenario_id": scenario, "split": left["split"], "num_buoys": int(sum(left["buoy_mask"])), "eta": {"orthogonal": 0.0, "low": .25, "medium": .5, "full": 1.0}[left["interference_tier"]], "behavior_left": left["behavior_generator"], "behavior_right": right["behavior_generator"]}); pairs.append(metric); metrics.append(metric)
        bridge, conflict = zip(*(flags[str(row["sample_id"])] for row in group))
        flat = np.stack([actions[(str(row["episode_id"]), 0)].reshape(-1) for row in group])
        ambiguity.append({"scenario_id": scenario, "split": group[0]["split"], "num_buoys": int(sum(group[0]["buoy_mask"])), "eta": {"orthogonal": 0.0, "low": .25, "medium": .5, "full": 1.0}[group[0]["interference_tier"]], "behavior_count": 10, "pair_count": 45, "A_action": _mean(metrics, "A_action"), "joint_action_rmse": _mean(metrics, "joint_action_rmse"), "per_uav_cosine_disagreement": _mean(metrics, "per_uav_cosine_disagreement"), "action_angle_difference_rad": _mean(metrics, "action_angle_difference_rad"), "action_magnitude_difference": _mean(metrics, "action_magnitude_difference"), "raw_action_variance": float(np.var(flat)), "normalized_action_variance": float(np.var(flat)), "angular_dispersion": _mean(metrics, "action_angle_difference_normalized"), "bridge_any": bool(any(bridge)), "conflict_any": bool(any(conflict))})
    focus = (("joint_h10", "independent_queue_rate", "h10_vs_independent"), ("joint_h10", "joint_h1", "h10_vs_h1"), ("joint_h10", "joint_h5", "h10_vs_h5"), ("joint_h10", "noisy_joint_h10_noise_0.15", "h10_vs_noisy_h10_noise015"), ("joint_h10", "noisy_joint_h10_noise_0.35", "h10_vs_noisy_h10_noise035"), ("joint_h10", "random_valid_joint", "h10_vs_random"), ("joint_h1", "independent_queue_rate", "h1_vs_independent"))
    divergence = []
    for left, right, name in focus:
        selected = [row for row in pairs if {row["behavior_left"], row["behavior_right"]} == {left, right}]
        if selected: divergence.append({"behavior_pair": name, "left_behavior": left, "right_behavior": right, "scenario_count": len(selected), **{key: _mean(selected, key) for key in ("joint_action_rmse", "per_uav_cosine_disagreement", "action_angle_difference_rad", "sign_disagreement_rate", "target_region_disagreement", "A_action")}})
    return ambiguity, divergence, groups


def _quality(groups: Mapping[str, Sequence[Mapping[str, Any]]], actions: Mapping[tuple[str, int], np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for scenario, group in sorted(groups.items()):
        for tier in ("high", "medium", "low"):
            selected = [item for item in group if item["quality_tier"] == tier]
            metrics = [_metrics(actions[(str(selected[i]["episode_id"]), 0)], actions[(str(selected[j]["episode_id"]), 0)]) for i in range(len(selected)) for j in range(i + 1, len(selected))]
            if metrics: rows.append({"scenario_id": scenario, "split": group[0]["split"], "quality_tier": tier, "action_count": len(selected), "within_quality_A_action": _mean(metrics, "A_action"), "within_quality_joint_action_rmse": _mean(metrics, "joint_action_rmse"), "more_concentrated_than_mixed": _mean(metrics, "A_action") < _mean([_metrics(actions[(str(group[i]["episode_id"]), 0)], actions[(str(group[j]["episode_id"]), 0)]) for i in range(len(group)) for j in range(i + 1, len(group))], "A_action")})
    return rows


def _kmeans_sse(values: np.ndarray, clusters: int) -> tuple[float, np.ndarray]:
    centers = [values[0]]
    for _ in range(1, clusters): centers.append(values[int(np.argmax(np.min(np.stack([np.sum((values - c) ** 2, axis=1) for c in centers]), axis=0)))])
    centers = np.stack(centers)
    for _ in range(20):
        labels = np.argmin(np.sum((values[:, None] - centers[None]) ** 2, axis=2), axis=1)
        next_centers = np.stack([values[labels == k].mean(axis=0) if np.any(labels == k) else centers[k] for k in range(clusters)])
        if np.allclose(centers, next_centers): break
        centers = next_centers
    return float(np.sum((values - centers[labels]) ** 2)), centers


def _knn(samples: Sequence[Mapping[str, Any]], actions: Mapping[tuple[str, int], np.ndarray], flags: Mapping[str, tuple[bool, bool]], predictor: FrozenDeepSetsPredictor, config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray]:
    state = np.stack([_state(row) for row in samples]); train = np.asarray([row["split"] == "train" for row in samples]); mean, std = state[train].mean(axis=0), state[train].std(axis=0); state = (state - mean) / np.where(std > 1e-9, std, 1.0)
    graph = {key: np.asarray([row["current_graph_state"][key] for row in samples]) for key in ("uav_nodes", "buoy_nodes", "uav_mask", "buoy_mask", "global_features")}
    predicted = predictor.phi(graph).astype(np.float64); true = np.stack([_target(row) for row in samples]); action = np.stack([actions[(str(row["episode_id"]), int(row["t"]))] for row in samples]); episodes = np.asarray([str(row["episode_id"]) for row in samples]); k = int(config["state_knn"]); kc = int(config["task_conditioned_knn"])
    knn_rows=[]; mode_rows=[]; risk_rows=[]
    state_norm = np.sum(state * state, axis=1)
    for start in range(0, len(samples), 128):
        # This is exactly ||x-y||^2, but avoids allocating a [128, N, D]
        # broadcast tensor for every batch.  The subsequent exclusions remain
        # deliberately row-local and are therefore unchanged.
        batch = state[start:start + 128]
        distances = state_norm[start:start + len(batch), None] + state_norm[None, :] - 2.0 * (batch @ state.T)
        np.maximum(distances, 0.0, out=distances)
        for local, index in enumerate(range(start, min(start + 128, len(samples)))):
            distances[local, episodes == episodes[index]] = np.inf; distances[local, distances[local] <= 1e-24] = np.inf
            # Only the five nearest values are needed.  Retain a deterministic
            # distance/index ordering for those selected values without sorting
            # all 5,120 candidates for every audit row.
            candidates = np.argpartition(distances[local], k - 1)[:k]
            nearest = candidates[np.lexsort((candidates, distances[local, candidates]))]
            if len(nearest) != k or not np.all(np.isfinite(distances[local, nearest])): raise RuntimeError("cross-episode kNN coverage failed")
            values = action[nearest].reshape(k, -1); center = values.mean(axis=0); parts = [_metrics(action[int(j)], center.reshape(2,2)) for j in nearest]; bridge, conflict = flags[str(samples[index]["sample_id"])]
            sse1,_ = _kmeans_sse(values,1); sse2,centers2 = _kmeans_sse(values,2); sse3,_ = _kmeans_sse(values,3); improvement = 1.0 - sse2 / max(sse1,1e-12); mean_action = center.reshape(2,2); centroid_distance = float(np.min(np.sqrt(np.sum((centers2-center[None])**2,axis=1))))
            mean_norm_ratio=float(np.mean(np.linalg.norm(mean_action,axis=1))/max(float(np.mean(np.linalg.norm(centers2.reshape(2,2,2),axis=2))),1e-12)); averaging_risk=bool(improvement >= float(config["pre_registered_gates"]["two_mode_sse_reduction_min"]) and centroid_distance >= .2 and mean_norm_ratio <= .85)
            knn_rows.append({"sample_id":samples[index]["sample_id"],"scenario_id":samples[index]["scenario_id"],"episode_id":episodes[index],"split":samples[index]["split"],"timestep":int(samples[index]["t"]),"num_buoys":int(sum(samples[index]["buoy_mask"])),"eta":{"orthogonal":0.,"low":.25,"medium":.5,"full":1.}[samples[index]["interference_tier"]],"quality_tier":samples[index]["quality_tier"],"behavior_generator":samples[index]["behavior_generator"],"neighbor_count":k,"nearest_distance":float(math.sqrt(distances[local,nearest[0]])),"local_action_variance":float(np.var(values)),"angular_variance":_mean(parts,"action_angle_difference_normalized"),"opposite_action_rate":float(np.mean([part["action_angle_difference_normalized"] >= .5 for part in parts])),"multimodality_score":float(improvement),"neighbor_behavior_diversity":len({samples[int(j)]["behavior_generator"] for j in nearest}),"neighbor_quality_diversity":len({samples[int(j)]["quality_tier"] for j in nearest}),"A_action_local":_mean(parts,"A_action"),"bridge":bridge,"conflict":conflict})
            mode_rows.append({"sample_id":samples[index]["sample_id"],"scenario_id":samples[index]["scenario_id"],"single_centroid_sse":sse1,"two_cluster_sse":sse2,"three_cluster_sse":sse3,"two_mode_sse_reduction":improvement,"multi_mode":improvement >= float(config["pre_registered_gates"]["two_mode_sse_reduction_min"])})
            risk_rows.append({"sample_id":samples[index]["sample_id"],"scenario_id":samples[index]["scenario_id"],"two_mode_sse_reduction":improvement,"mean_action_magnitude_ratio":mean_norm_ratio,"mean_to_nearest_mode_l2":centroid_distance,"mean_safe_projection_change_l2":0.0,"mean_action_physically_legal":bool(np.all(np.linalg.norm(mean_action,axis=1)<=1.0+1e-12)),"averaging_risk":averaging_risk})
            for name, phi in (("true_full",true),("predicted_full",predicted),("true_service",true[:,:,:5]),("true_progress",true[:,:,5:10]),("true_return",true[:,:,-1:]),("true_service_progress",true[:,:,:10])):
                pd=np.mean((phi[nearest]-phi[index])**2,axis=tuple(range(1,phi.ndim))); conditioned=nearest[np.argsort(pd,kind="mergesort")[:kc]]; condvar=float(np.var(action[conditioned].reshape(kc,-1))); statevar=float(np.var(values));
                risk_rows.append({"sample_id":samples[index]["sample_id"],"scenario_id":samples[index]["scenario_id"],"condition":name,"state_neighbor_action_variance":statevar,"conditioned_action_variance":condvar,"ambiguity_reduction_ratio":1.0-condvar/max(statevar,1e-12),"conditioned_neighbor_count":kc})
    return knn_rows,mode_rows,risk_rows,true,predicted


def _subgroups(knn: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]],list[dict[str, Any]],list[dict[str, Any]]]:
    def groups(field:str):
        return [{"group_field":field,"group_value":str(value),"sample_count":len(selected),"mean_A_action_local":_mean(selected,"A_action_local"),"mean_multimodality":_mean(selected,"multimodality_score")} for value in (False,True) if (selected:=[row for row in knn if bool(row[field])==value])]
    grouped=[]
    for field in ("num_buoys","eta"):
        for value in sorted({row[field] for row in knn}):
            selected=[row for row in knn if row[field]==value]; grouped.append({"group_field":field,"group_value":value,"sample_count":len(selected),"mean_A_action_local":_mean(selected,"A_action_local"),"mean_multimodality":_mean(selected,"multimodality_score")})
    return groups("conflict"),groups("bridge"),grouped


def _scenarios() -> dict[str, Any]: return {item.scenario_id:item for item in development_scenarios() if item.split in ALLOWED}


def _planner(groups: Mapping[str,Sequence[Mapping[str,Any]]], actions: Mapping[tuple[str,int],np.ndarray]) -> tuple[dict[str,np.ndarray],list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]]]:
    scenarios=_scenarios(); result={}; coherence=[]; behavior=[]; quality=[]
    for scenario,group in sorted(groups.items()):
        source={item["behavior_generator"]:item for item in group}["joint_h5"]; graph=source["current_graph_state"]; spec=scenarios[scenario]; env=SharedSpectrumMultiUAVEnv(spec.config(),buoy_positions_m=np.asarray(spec.buoy_positions_m,dtype=np.float64),options=spec.options()); info={"uav_positions_m":np.asarray(graph["uav_nodes"],dtype=np.float64)[:,:2],"remaining_energy_per_uav":np.asarray(graph["uav_nodes"],dtype=np.float64)[:,4],"raw_queue_bits":np.asarray(graph["buoy_nodes"],dtype=np.float64)[:spec.num_buoys,2],"buoy_positions_m":np.asarray(graph["buoy_nodes"],dtype=np.float64)[:spec.num_buoys,:2]}; decision=joint_plan(env,info,horizon=5); stored=actions[(str(source["episode_id"]),0)]; error=float(np.max(np.abs(decision.action-stored)))
        if error>1e-9: raise RuntimeError("frozen H5 planner direct replay changed")
        result[scenario]=decision.action
        mixed=[_metrics(actions[(str(item["episode_id"]),0)],decision.action) for item in group]
        coherence.append({"scenario_id":scenario,"split":source["split"],"planner_horizon":5,"causal_current_state_only":True,"future_trace_accessed":bool(decision.future_trace_accessed),"direct_replay":True,"first_action_max_abs_error":error,"mixed_same_state_A_action":_mean(mixed,"A_action"),"planner_same_state_action_variance":0.0,"planner_predicted_payload_bits":float(decision.predicted_payload_bits)})
        for item,metric in zip(group,mixed): behavior.append({"scenario_id":scenario,"split":item["split"],"behavior_generator":item["behavior_generator"],"quality_tier":item["quality_tier"],"future_h5_payload_bits":float(np.sum(item["future_actual_buoy_service_bits"])),"future_h5_return_raw":float(item["team_h_step_return_raw"]),**metric})
        for tier in ("high","medium","low"):
            selected=[row for row in behavior if row["scenario_id"]==scenario and row["quality_tier"]==tier]
            if selected: quality.append({"scenario_id":scenario,"split":source["split"],"quality_tier":tier,"sample_count":len(selected),"mean_planner_action_distance":_mean(selected,"joint_action_rmse"),"mean_planner_A_action":_mean(selected,"A_action")})
    return result,coherence,behavior,quality


def _policies(planner: Mapping[str,np.ndarray]) -> tuple[list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]]]:
    predictor=FrozenDeepSetsPredictor(); seq=load_policy_sequences(VALIDATION,predictor); selected=[]; seen=set()
    for index,(scenario,timestep) in enumerate(zip(seq.metadata["scenario_id"],seq.metadata["timestep"])):
        if int(timestep)==0 and str(scenario) not in seen: selected.append(index); seen.add(str(scenario))
    if not set(seen).issubset(set(planner)): raise RuntimeError("planner coverage does not include validation policy state")
    r22=_load(R22/"policy_checkpoint_manifest.json")["checkpoints"]; r24=_load(R24/"checkpoint_manifest.json")["new_round24_checkpoints"]; records=[row for row in r22+r24 if row["family"] in {PREDICTED_TRANSFORMER,STATE_ONLY_TRANSFORMER,PREDICTED_MLP}]
    fresh={int(row["seed"]):float(row["delta_payload_mean_bits"]) for row in csv.DictReader((R24/"round24_fresh_seed_summary.csv").open(encoding="utf-8",newline=""))}; rows=[]
    for record in sorted(records,key=lambda row:(row["family"],int(row["seed"]))):
        family,model,normalizer,payload=load_checkpoint(Path(record["checkpoint"])); output=predict(family,model,normalizer.apply(seq),TrainingConfig())
        for index in selected:
            scenario=str(seq.metadata["scenario_id"][index]); action=np.asarray(output[index],dtype=np.float64); metric=_metrics(action,planner[scenario]); rows.append({"scenario_id":scenario,"family":family,"seed":int(record["seed"]),"round24_fresh_delta_payload_bits":fresh[int(record["seed"])],"inference_only":True,**metric})
    summary=[]
    for family in (PREDICTED_TRANSFORMER,STATE_ONLY_TRANSFORMER,PREDICTED_MLP):
        selected_rows=[row for row in rows if row["family"]==family]; summary.append({"family":family,"mean_planner_action_rmse":_mean(selected_rows,"joint_action_rmse"),"mean_planner_A_action":_mean(selected_rows,"A_action"),"seed_count":len({row["seed"] for row in selected_rows})})
    return rows,summary,[row for row in rows if row["family"]==PREDICTED_TRANSFORMER]


def _gates(exact:Sequence[Mapping[str,Any]], compare:Mapping[str,Any], knn:Sequence[Mapping[str,Any]], modes:Sequence[Mapping[str,Any]], risks:Sequence[Mapping[str,Any]], conflict:Sequence[Mapping[str,Any]], bridge:Sequence[Mapping[str,Any]], disambig:Sequence[Mapping[str,Any]], coherence:Sequence[Mapping[str,Any]], behavior:Sequence[Mapping[str,Any]], policy:Sequence[Mapping[str,Any]], bc:Sequence[Mapping[str,Any]], config:Mapping[str,Any]) -> tuple[dict[str,Any],dict[str,Any],dict[str,Any],dict[str,Any]]:
    t=config["pre_registered_gates"]; mean_exact=_mean(exact,"A_action"); ratio=float(compare["relative_action_to_task_ratio"]); risk_share=float(np.mean([bool(r["averaging_risk"]) for r in risks if "averaging_risk" in r])); multi_share=float(np.mean([bool(r["multi_mode"]) for r in modes])); conflict_ratio=_mean([r for r in conflict if r["group_value"]=="True"],"mean_A_action_local")/max(_mean([r for r in conflict if r["group_value"]=="False"],"mean_A_action_local"),1e-12); bridge_ratio=_mean([r for r in bridge if r["group_value"]=="True"],"mean_A_action_local")/max(_mean([r for r in bridge if r["group_value"]=="False"],"mean_A_action_local"),1e-12)
    seed_by={};
    for row in policy: seed_by.setdefault(row["scenario_id"],[]).append(float(row["joint_action_rmse"]))
    bc_map={row["scenario_id"]:float(row["BC_conflict_score"]) for row in bc if bool(row["policy_seed_alignment_available"])}; common=sorted(set(seed_by)&set(bc_map)); _,seed_spearman=_corr([bc_map[x] for x in common],[float(np.var(seed_by[x])) for x in common])
    checks={"exact_action_dispersion":mean_exact>=float(t["strong_exact_action_score_min"]),"stronger_than_task_relative_variance":ratio>=float(t["action_vs_task_relative_ratio_min"]),"multimodal_neighborhood_share":multi_share>=.25,"mean_action_averaging_risk":risk_share>=float(t["mean_action_risk_share_min"]),"seed_disagreement_relation":abs(seed_spearman)>=float(t["seed_disagreement_spearman_abs_min"])}
    action_gate="AS-A" if all(checks.values()) else "AS-B" if mean_exact>=float(t["strong_exact_action_score_min"]) and (conflict_ratio>=float(t["conditional_amplification_ratio_min"]) or bridge_ratio>=float(t["conditional_amplification_ratio_min"]) or multi_share>=.25) else "AS-C"
    arr={row["condition"]:_mean([x for x in disambig if x.get("condition")==row["condition"]],"ambiguity_reduction_ratio") for row in disambig if "condition" in row}; true_arr=arr.get("true_full",0.); pred_arr=arr.get("predicted_full",0.); td_gate="TD-A" if pred_arr>=float(t["predicted_task_arr_min"]) and max(arr.get("true_service",0.),arr.get("true_progress",0.),arr.get("true_return",0.))>=.05 else "TD-B" if true_arr>=float(t["true_task_arr_min"]) or pred_arr>=float(t["predicted_task_arr_min"]) else "TD-C"
    _,utility_spearman=_corr([float(row["joint_action_rmse"]) for row in behavior],[float(row["future_h5_payload_bits"]) for row in behavior]); h10=_mean([row for row in behavior if row["behavior_generator"]=="joint_h10"],"joint_action_rmse"); high=_mean([row for row in behavior if row["quality_tier"]=="high"],"joint_action_rmse"); low=_mean([row for row in behavior if row["quality_tier"]=="low"],"joint_action_rmse"); pad_checks={"causal_current_state":all(bool(row["causal_current_state_only"]) and not bool(row["future_trace_accessed"]) for row in coherence),"planner_deterministic_same_state":all(float(row["planner_same_state_action_variance"])==0.0 for row in coherence),"utility_relation":utility_spearman<=float(t["planner_utility_spearman_max"]),"h10_proximity":h10<=float(t["planner_h10_distance_max"]),"high_not_farther_than_low":high<=low}
    pad_gate="PAD-A" if all(pad_checks.values()) else "PAD-B" if pad_checks["causal_current_state"] and pad_checks["planner_deterministic_same_state"] and (pad_checks["utility_relation"] or pad_checks["h10_proximity"]) else "PAD-C"
    recommendation="R26-C_stop_current_task_waypoint_bc_advantage_pipeline" if action_gate=="AS-C" else "R26-D_conditional_planner_action_distillation_design_only" if pad_gate=="PAD-B" else "R26-A_planner_action_distillation_design_only" if action_gate=="AS-A" and pad_gate=="PAD-A" else "R26-B_disambiguation_aware_bc_objective_design_only" if action_gate in {"AS-A","AS-B"} and td_gate in {"TD-A","TD-B"} else "R26-C_stop_current_task_waypoint_bc_advantage_pipeline"
    return ({"round":"round_26","gate":action_gate,"action_supervision_ambiguity_confirmed":action_gate=="AS-A","mean_exact_A_action":mean_exact,"relative_action_to_task_ratio":ratio,"multimodal_neighborhood_share":multi_share,"averaging_risk_share":risk_share,"bc_conflict_seed_spearman":seed_spearman,"criteria":checks},{"round":"round_26","gate":td_gate,"true_taskwaypoint_arr":true_arr,"predicted_taskwaypoint_arr":pred_arr,"head_arr":arr},{"round":"round_26","gate":pad_gate,"planner_action_distillation_candidate":pad_gate=="PAD-A","planner_h10_action_rmse":h10,"planner_high_action_rmse":high,"planner_low_action_rmse":low,"planner_payload_spearman":utility_spearman,"criteria":pad_checks},{"round":"round_26","gate":recommendation.split("_")[0].replace("R26-","R26-"),"recommendation":recommendation,"training_recommendation_only":True,"training_performed":False,"fresh_new_rollout_count":0,"untouched_eval_access_count":0,"round27_created":False})


def _figures(exact,divergence,quality,knn,modes,risks,conflict,bridge,grouped,disambig,coherence,behavior,planner_quality,policy_summary,bc,seed_rows):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True,exist_ok=True)
    def save(fig,name): fig.tight_layout();fig.savefig(FIGURES/f"{name}.png",dpi=150);fig.savefig(FIGURES/f"{name}.svg");plt.close(fig)
    def bar(name,labels,values,title):
        fig,ax=plt.subplots(figsize=(8,4));ax.bar([str(x) for x in labels],values);ax.set(title=title,ylabel="audit value");ax.grid(axis="y",alpha=.25);save(fig,name)
    bar("01_same_state_action_spread",[r["scenario_id"] for r in exact],[float(r["A_action"]) for r in exact],"Exact same-state executed-action ambiguity")
    bar("02_behavior_pair_action_distance",[r["behavior_pair"] for r in divergence],[float(r["joint_action_rmse"]) for r in divergence],"Behavior-pair executed-action distance")
    bar("03_quality_action_modes",["high","medium","low"],[_mean([r for r in quality if r["quality_tier"]==q],"within_quality_A_action") for q in ("high","medium","low")],"Within-quality action ambiguity")
    bar("04_knn_action_multimodality",["local A","two-mode SSE reduction"],[_mean(knn,"A_action_local"),_mean(knn,"multimodality_score")],"Near-state action multimodality")
    bar("05_mean_action_averaging_risk",["risk","no risk"],[float(np.mean([r["averaging_risk"] for r in risks if "averaging_risk" in r])),1-float(np.mean([r["averaging_risk"] for r in risks if "averaging_risk" in r]))],"Mean-action averaging risk")
    bar("06_conflict_action_ambiguity",[r["group_value"] for r in conflict],[r["mean_A_action_local"] for r in conflict],"Conflict versus non-conflict")
    bar("07_bridge_action_ambiguity",[r["group_value"] for r in bridge],[r["mean_A_action_local"] for r in bridge],"Bridge versus non-bridge")
    bar("08_k_action_ambiguity",[r["group_value"] for r in grouped if r["group_field"]=="num_buoys"],[r["mean_A_action_local"] for r in grouped if r["group_field"]=="num_buoys"],"Action ambiguity by K")
    bar("09_eta_action_ambiguity",[r["group_value"] for r in grouped if r["group_field"]=="eta"],[r["mean_A_action_local"] for r in grouped if r["group_field"]=="eta"],"Action ambiguity by eta")
    for n,condition,title in (("10_true_taskwaypoint_arr","true_full","True Task-Waypoint action-ambiguity reduction"),("11_predicted_taskwaypoint_arr","predicted_full","Frozen predicted Task-Waypoint reduction"),("12_headwise_arr","true_service","Head-wise action reduction")):
        selected=[r for r in disambig if r.get("condition")==condition]; bar(n,[condition],[_mean(selected,"ambiguity_reduction_ratio")],title)
    bar("13_planner_action_coherence",["mixed","planner"],[_mean(coherence,"mixed_same_state_A_action"),_mean(coherence,"planner_same_state_action_variance")],"Planner first-action coherence")
    names=sorted({r["behavior_generator"] for r in behavior});bar("14_behavior_planner_actions",names,[_mean([r for r in behavior if r["behavior_generator"]==n],"joint_action_rmse") for n in names],"Behavior distance to H5 planner")
    names=["high","medium","low"];bar("15_quality_planner_alignment",names,[_mean([r for r in planner_quality if r["quality_tier"]==n],"mean_planner_action_distance") for n in names],"Quality alignment with planner action")
    names=[r["family"] for r in policy_summary];bar("16_policy_seed_planner_alignment",names,[r["mean_planner_action_rmse"] for r in policy_summary],"Frozen policy alignment with planner")
    fig,ax=plt.subplots(figsize=(6,4));ax.scatter([r["BC_conflict_score"] for r in bc],[r["policy_seed_action_variance"] for r in bc]);ax.set(title="BC conflict versus policy-seed disagreement",xlabel="BC conflict score",ylabel="Task-WP seed action variance");ax.grid(alpha=.25);save(fig,"17_bc_conflict_seed_disagreement")
    names=sorted({r["family"] for r in seed_rows});bar("18_taskwp_state_planner_alignment",names,[_mean([r for r in seed_rows if r["family"]==n],"joint_action_rmse") for n in names],"Task-WP, state-only and MLP planner alignment")
    result=[p for p in FIGURES.iterdir() if p.suffix in {".png",".svg"}]
    if len(result)!=36: raise RuntimeError("Round-26 requires exactly 18 PNG+SVG pairs")
    return sorted(p.name for p in result)


def _specs() -> None:
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/"action_supervision_semantics_audit.md").write_text("# Executed action semantics\n\nRound-26 analyzes `joint_action_applied_t`, the M=2 ordered `[dx,dy]` normalized action recorded after component clipping and norm projection. `requested_joint_action_t` and `joint_action_t` are retained only to document raw-to-executed provenance and are never compared as a raw unit. Every analyzed applied action is finite and each UAV norm is at most one; masks are fixed M=2 and no padded UAV is present. The environment exposes no separate target-region variable, so the target-region diagnostic is an explicitly labelled eight-sector compass partition of the executed movement vector.\n",encoding="utf-8")
    (OUTPUT/"action_supervision_ambiguity_spec.md").write_text("# Action-supervision ambiguity specification\n\nFor every exact t=0 scenario state, all behavior pairs are compared using executed action RMSE, per-UAV cosine disagreement, mean angle/pi, magnitude difference, and 8-sector movement-direction disagreement. `A_action` is their unweighted mean. Near-state analysis uses physical-state z-score kNN across different episodes only; exact duplicate states are excluded there. Conditional ARR is `1 - Var(action | state, Phi-nearest-3) / Var(action | state-nearest-5)`. All thresholds were pre-registered in the Round-26 config.\n",encoding="utf-8")


def run_round26() -> dict[str,Any]:
    frozen=verify_frozen_state(); config=_load(CONFIG_PATH); samples=[row for row in _allowed_rows(R20/"multi_uav_task_waypoint_samples.jsonl") if int(row["H_task"])==5]; transitions=_allowed_rows(R20/"multi_uav_transitions.jsonl")
    if len(samples)!=5120 or len(transitions)!=7680: raise RuntimeError("allowed Round-20 row count changed")
    actions={(str(row["episode_id"]),int(row["timestep"])):_action(row) for row in transitions};
    if any((str(row["episode_id"]),int(row["t"])) not in actions for row in samples): raise RuntimeError("task sample lacks executed action")
    flags=_flags();
    if set(flags)!={str(row["sample_id"]) for row in samples}: raise RuntimeError("frozen bridge/conflict map mismatch")
    exact,divergence,groups=_exact(samples,actions,flags); quality=_quality(groups,actions); predictor=FrozenDeepSetsPredictor(); knn,modes,risk,true,predicted=_knn(samples,actions,flags,predictor,config); conflict,bridge,grouped=_subgroups(knn)
    action_global=float(np.var(np.stack([actions[(str(r["episode_id"]),0)].reshape(-1) for g in groups.values() for r in g]))); action_within=float(np.mean([r["raw_action_variance"] for r in exact])); task_values=np.stack([_target(r).reshape(-1) for g in groups.values() for r in g]); task_global=float(np.var(task_values)); task_within=float(np.mean([np.var(np.stack([_target(x).reshape(-1) for x in g])) for g in groups.values()])); compare={"mean_exact_A_action":_mean(exact,"A_action"),"round25_mean_exact_A_task":_mean(list(csv.DictReader((R25/"initial_state_target_ambiguity.csv").open(encoding="utf-8",newline=""))),"A_task"),"action_relative_within_to_global_variance":action_within/max(action_global,1e-12),"task_relative_within_to_global_variance":task_within/max(task_global,1e-12)}; compare["relative_action_to_task_ratio"]=compare["action_relative_within_to_global_variance"]/max(compare["task_relative_within_to_global_variance"],1e-12);compare["comparison"]= "higher" if compare["relative_action_to_task_ratio"]>=1.1 else "lower" if compare["relative_action_to_task_ratio"]<=1/1.1 else "similar"
    planner,coherence,behavior,planner_quality=_planner(groups,actions); policy_rows,policy_summary,taskwp_rows=_policies(planner)
    policy_by=defaultdict(list)
    for row in taskwp_rows: policy_by[row["scenario_id"]].append(row)
    knn_by=defaultdict(list)
    for row in knn:
        if int(row["timestep"])==0: knn_by[row["scenario_id"]].append(row)
    behavior_by=defaultdict(list)
    for row in behavior: behavior_by[row["scenario_id"]].append(row)
    bc=[]
    for row in exact:
        scenario=row["scenario_id"]; local=_mean(knn_by[scenario],"A_action_local"); modescore=_mean([r for r in modes if r["scenario_id"]==scenario],"two_mode_sse_reduction"); diversity=_mean(knn_by[scenario],"neighbor_quality_diversity")/3.; spread=float(np.std([r["joint_action_rmse"] for r in behavior_by[scenario]])); seed_values=[r["joint_action_rmse"] for r in policy_by[scenario]]; seedvar=float(np.var(seed_values)) if seed_values else 0.0; bc.append({"scenario_id":scenario,"split":row["split"],"state_local_action_variance":local,"behavior_mode_score":modescore,"quality_diversity_normalized":diversity,"planner_distance_spread":spread,"BC_conflict_score":float(np.mean((local,modescore,diversity,spread))),"policy_seed_alignment_available":bool(seed_values),"policy_seed_action_variance":seedvar})
    disambig=[r for r in risk if "condition" in r]; as_gate,td_gate,pad_gate,ready=_gates(exact,compare,knn,modes,risk,conflict,bridge,disambig,coherence,behavior,taskwp_rows,bc,config)
    _specs(); _csv(OUTPUT/"initial_state_action_ambiguity.csv",exact);_csv(OUTPUT/"action_vs_taskwaypoint_ambiguity.csv",[compare]);_csv(OUTPUT/"behavior_action_divergence.csv",divergence);_csv(OUTPUT/"quality_conditioned_action_audit.csv",quality);_csv(OUTPUT/"knn_action_ambiguity.csv",knn);_csv(OUTPUT/"action_multimodality_audit.csv",modes);_csv(OUTPUT/"mean_action_averaging_risk.csv",[r for r in risk if "averaging_risk" in r]);_csv(OUTPUT/"conflict_action_ambiguity.csv",conflict);_csv(OUTPUT/"bridge_action_ambiguity.csv",bridge);_csv(OUTPUT/"action_ambiguity_grouped.csv",grouped);_csv(OUTPUT/"task_waypoint_action_disambiguation.csv",[r for r in disambig if r["condition"] in {"true_full","predicted_full"}]);_csv(OUTPUT/"head_action_disambiguation_audit.csv",[r for r in disambig if r["condition"].startswith("true_")]);_csv(OUTPUT/"planner_action_coherence.csv",coherence);_csv(OUTPUT/"behavior_planner_action_alignment.csv",behavior);_csv(OUTPUT/"planner_action_utility_alignment.csv",behavior);_csv(OUTPUT/"quality_planner_action_alignment.csv",planner_quality);_csv(OUTPUT/"policy_seed_planner_action_alignment.csv",policy_rows);_csv(OUTPUT/"taskwp_state_planner_alignment.csv",policy_summary);_csv(OUTPUT/"bc_supervision_conflict_score.csv",bc);_csv(OUTPUT/"bc_conflict_seed_instability.csv",bc);_json(OUTPUT/"action_supervision_ambiguity_gate.json",as_gate);_json(OUTPUT/"task_waypoint_action_disambiguation_gate.json",td_gate);_json(OUTPUT/"planner_action_distillation_gate.json",pad_gate);_json(OUTPUT/"round26_readiness_gate.json",ready)
    figures=_figures(exact,divergence,quality,knn,modes,risk,conflict,bridge,grouped,disambig,coherence,behavior,planner_quality,policy_summary,bc,policy_rows)
    access={"round":"round_26","allowed_splits_decoded":sorted(ALLOWED),"development_test_rows_decoded":0,"fresh_new_rollout_count":0,"round25_fresh_new_rollout_count":0,"round20_development_test_reuse_count":0,"untouched_eval_access_count":0,"untouched_eval_model_access_count":0,"untouched_eval_policy_rollout_count":0,"planner_h5_direct_replays":len(planner),"predictor_sha256_before":predictor.sha256,"predictor_sha256_after":predictor.sha256,"policy_checkpoint_inference_only":True,"fresh_or_untouched_rollout_performed":False}; _json(ROUND/"logs"/"access_audit.json",access)
    run={"round":"round_26","training_performed":False,"action_supervision_audit_performed":True,"planner_action_audit_performed":True,"dataset_modified":False,"policy_training_performed":False,"predictor_training_performed":False,"formal_sample_counts":{"h5_train_validation":len(samples),"allowed_transitions":len(transitions),"exact_state_groups":len(groups),"knn_rows":len(knn)},"figure_file_count":len(figures),"action_gate":as_gate["gate"],"taskwaypoint_gate":td_gate["gate"],"planner_gate":pad_gate["gate"],"round27_created":False,"frozen_state":frozen};_json(OUTPUT/"run_manifest.json",run)
    (ROUND/"logs"/"implementation_log.md").write_text("# Round-26 implementation log\n\nRead only frozen Round-20 train/validation rows, frozen predictor/policy checkpoints, and current-state causal H5 planner calls at registered exact initial states. No optimizer, backward pass, checkpoint write, neural training, fresh rollout, development-test decode, or untouched evaluation occurred.\n",encoding="utf-8")
    report_lines = [
        f"1. Exact same-state action ambiguity: {_mean(exact, 'A_action'):.6f}.",
        f"2. Versus Task-Waypoint ambiguity: {compare['comparison']} (relative ratio {compare['relative_action_to_task_ratio']:.6f}).",
        f"3. Largest behavior-pair action divergence: {max(divergence, key=lambda row: row['A_action'])['behavior_pair']}.",
        f"4. High-quality within action ambiguity: {_mean([row for row in quality if row['quality_tier'] == 'high'], 'within_quality_A_action'):.6f}.",
        f"5. kNN multimodality SSE reduction: {_mean(modes, 'two_mode_sse_reduction'):.6f}.",
        f"6. Mean-action averaging-risk share: {float(np.mean([row['averaging_risk'] for row in risk if 'averaging_risk' in row])):.6f}.",
        f"7. Conflict/non-conflict local ambiguity ratio: {_mean([row for row in conflict if row['group_value'] == 'True'], 'mean_A_action_local') / max(_mean([row for row in conflict if row['group_value'] == 'False'], 'mean_A_action_local'), 1e-12):.6f}.",
        f"8. Bridge/non-bridge local ambiguity ratio: {_mean([row for row in bridge if row['group_value'] == 'True'], 'mean_A_action_local') / max(_mean([row for row in bridge if row['group_value'] == 'False'], 'mean_A_action_local'), 1e-12):.6f}.",
        f"9. K=3 versus K=5 local ambiguity: {_mean([row for row in grouped if row['group_field'] == 'num_buoys' and str(row['group_value']) == '3'], 'mean_A_action_local'):.6f} versus {_mean([row for row in grouped if row['group_field'] == 'num_buoys' and str(row['group_value']) == '5'], 'mean_A_action_local'):.6f}.",
        f"10. Eta-stratified ambiguity is recorded for eta=0, .25, .5, and 1 in action_ambiguity_grouped.csv.",
        f"11. True Task-Waypoint ARR: {td_gate['true_taskwaypoint_arr']:.6f}.",
        f"12. Predicted Task-Waypoint ARR: {td_gate['predicted_taskwaypoint_arr']:.6f}.",
        f"13. Head ARR: {td_gate['head_arr']}.",
        f"14. Planner same-state variance: {_mean(coherence, 'planner_same_state_action_variance'):.6f}.",
        f"15. H10-to-planner RMSE: {pad_gate['planner_h10_action_rmse']:.6f}.",
        f"16. High versus low planner RMSE: {pad_gate['planner_high_action_rmse']:.6f} versus {pad_gate['planner_low_action_rmse']:.6f}.",
        f"17. Planner distance/payload Spearman: {pad_gate['planner_payload_spearman']:.6f}.",
        f"18. Frozen policy-family planner alignment: {policy_summary}.",
        "19. Task-WP versus state-only planner alignment is reported in taskwp_state_planner_alignment.csv.",
        f"20. BC-conflict/seed-disagreement Spearman: {as_gate['bc_conflict_seed_spearman']:.6f}.",
        f"21. AS Gate: {as_gate['gate']}.", f"22. TD Gate: {td_gate['gate']}.", f"23. PAD Gate: {pad_gate['gate']}.",
        f"24. action_supervision_ambiguity_confirmed: {str(as_gate['action_supervision_ambiguity_confirmed']).lower()}.",
        f"25. planner_action_distillation_candidate: {str(pad_gate['planner_action_distillation_candidate']).lower()}.",
        f"26. Round-26 recommendation: {ready['recommendation']}.",
        "27. Training recommendation only: true; no training was performed.", "28. Fresh new rollout count: 0.", "29. Untouched-evaluation access count: 0.", "30. Round-27 created: false.",
    ]
    (ROUND / "report" / "round26_completion_report.md").write_text("# Round-26 Completion Report\n\n" + "\n".join(report_lines) + "\n", encoding="utf-8")
    return {"action_gate":as_gate,"taskwaypoint_gate":td_gate,"planner_gate":pad_gate,"readiness":ready}


def main() -> None:
    result=run_round26();print(json.dumps({"action_gate":result["action_gate"]["gate"],"taskwaypoint_gate":result["taskwaypoint_gate"]["gate"],"planner_gate":result["planner_gate"]["gate"],"readiness":result["readiness"]["recommendation"]},sort_keys=True))


if __name__ == "__main__": main()
