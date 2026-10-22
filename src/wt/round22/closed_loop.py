"""Causal Round-22 validation and conditional fresh-development rollouts.

Only the three deployable policies can enter this module.  At every step Phi
is generated from the current graph by the frozen predictor; no future state,
true Task-Waypoint label, replay state, or historical action is supplied.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from wt.round19.multi_env import SharedSpectrumMultiUAVEnv
from wt.round20.dataset import build_graph_state
from wt.round20.scenarios import development_scenarios

from .data import FrozenDeepSetsPredictor, R22_OUTPUT, VALIDATION, load_policy_sequences, sha256
from .models import DEPLOYABLE_FAMILIES, PREDICTED_MLP, PREDICTED_TRANSFORMER, STATE_ONLY_TRANSFORMER, rollout_action
from .scenarios import FreshPolicyScenario
from .training import TrainingConfig, load_checkpoint


ETA = {"orthogonal": 0.0, "low": 0.25, "medium": 0.5, "full": 1.0}
FORMAL_DIR = R22_OUTPUT / "formal"


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Mapping): return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(item) for item in value]
    return value


def _json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_plain(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = [_plain(row) for row in rows]
    if not values: raise RuntimeError(f"required Round-22 output is empty: {path.name}")
    fields = sorted({key for row in values for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in values:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def _formal_runs() -> list[dict[str, Any]]:
    manifest = json.loads((R22_OUTPUT / "round22_formal_training_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("formal_training_completed") != "12/12": raise RuntimeError("closed-loop requires all 12 formal checkpoints")
    return list(manifest["runs"])


def _scenario_hash(scenario: Any) -> str:
    return hashlib.sha256(json.dumps(_plain(scenario.manifest_row()), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest().upper()


def _bridge_by_validation_scenario(predictor: FrozenDeepSetsPredictor) -> dict[str, bool]:
    sequences = load_policy_sequences(VALIDATION, predictor)
    return {str(scenario): bool(np.any(sequences.metadata["bridge"][sequences.metadata["scenario_id"] == scenario])) for scenario in np.unique(sequences.metadata["scenario_id"])}


def _model_inputs(history: list[tuple[dict[str, Any], np.ndarray]], normalizer, device: str) -> dict[str, torch.Tensor]:
    h, m, k = 10, 2, 5
    uav = np.zeros((1, h, m, 8), dtype=np.float32); buoy = np.zeros((1, h, k, 7), dtype=np.float32)
    uav_mask = np.zeros((1, h, m), dtype=bool); buoy_mask = np.zeros((1, h, k), dtype=bool)
    global_ = np.zeros((1, h, 6), dtype=np.float32); phi = np.zeros((1, h, m, 11), dtype=np.float32); attention = np.zeros((1, h), dtype=bool)
    for offset, (graph, condition) in enumerate(history[-h:]):
        index = h - min(len(history), h) + offset
        uav[0, index] = np.asarray(graph["uav_nodes"], dtype=np.float32); buoy[0, index] = np.asarray(graph["buoy_nodes"], dtype=np.float32)
        uav_mask[0, index] = np.asarray(graph["uav_mask"], dtype=bool); buoy_mask[0, index] = np.asarray(graph["buoy_mask"], dtype=bool)
        global_[0, index] = np.asarray(graph["global_features"], dtype=np.float32); phi[0, index] = condition; attention[0, index] = True
    uav = (uav - normalizer.means["uav_nodes"]) / normalizer.stds["uav_nodes"]
    buoy = (buoy - normalizer.means["buoy_nodes"]) / normalizer.stds["buoy_nodes"]
    global_ = (global_ - normalizer.means["global_features"]) / normalizer.stds["global_features"]
    values = {"uav_nodes": uav, "buoy_nodes": buoy, "uav_mask": uav_mask, "buoy_mask": buoy_mask, "global_features": global_, "attention_mask": attention, "phi": phi}
    return {key: torch.as_tensor(value, dtype=torch.bool if key.endswith("mask") else torch.float32, device=device) for key, value in values.items()}


def _current_phi(predictor: FrozenDeepSetsPredictor, graph: dict[str, Any]) -> np.ndarray:
    flat = {key: np.asarray(graph[key])[None, ...] for key in ("uav_nodes", "buoy_nodes", "uav_mask", "buoy_mask", "global_features")}
    return predictor.phi(flat)[0]


def _rollout(scenario: Any, family: str, seed: int, checkpoint: Path, predictor: FrozenDeepSetsPredictor, *, population: str, bridge: bool | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    loaded_family, model, normalizer, payload = load_checkpoint(checkpoint, "cpu")
    if loaded_family != family or family not in DEPLOYABLE_FAMILIES or not bool(payload["deployable"]):
        raise PermissionError("only registered deployable Round-22 policies may enter a causal rollout")
    env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m, dtype=np.float64), options=scenario.options())
    _, info = env.reset(seed=int(scenario.seed)); history: list[tuple[dict[str, Any], np.ndarray]] = []; steps: list[dict[str, Any]] = []
    total_return = 0.0; arrivals = 0.0; done = False
    while not done:
        graph = build_graph_state(info, env.config); phi = _current_phi(predictor, graph); history.append((graph, phi))
        inputs = _model_inputs(history, normalizer, "cpu")
        if family == STATE_ONLY_TRANSFORMER: inputs["phi"] = None
        action = rollout_action(family, model, **inputs).cpu().numpy()[0]
        requested = action.copy(); fallback = False
        try:
            _, reward, terminated, truncated, after = env.step(action, matching_mode="legacy")
        except RuntimeError as error:
            if "hard UAV separation projection did not converge" not in str(error): raise
            _, reward, terminated, truncated, after = env.step(np.zeros_like(action), matching_mode="legacy"); fallback = True
        total_return += float(reward); arrivals += float(np.sum(after["raw_arrival_bits"]))
        steps.append({"population": population, "scenario_id": scenario.scenario_id, "scenario_seed": int(scenario.seed), "family": family, "model_seed": seed, "timestep": int(after["time_step"]), "num_buoys": int(scenario.num_buoys), "eta": float(scenario.interference_overlap_factor), "interference_tier": str(scenario.spectrum_tier), "requested_action": requested, "applied_action": np.asarray(after["normalized_applied_action"], dtype=np.float64), "uav_positions_m": np.asarray(after["uav_positions_m"], dtype=np.float64), "raw_task_waypoint_phi": phi, "projection_activated": bool(np.any(np.asarray(after["safety_correction_m"], dtype=np.float64) > 1.0e-9)), "safety_projection_fallback": fallback, "min_uav_distance_m": float(after["min_uav_distance_m"]), "payload_bits_step": float(np.sum(after["transmitted_payload_bits"])), "interference_conflict": bool(after["interference_conflict"]), "uses_true_phi": False})
        info = after; done = bool(terminated or truncated)
    payload_by_buoy = np.asarray(info["cumulative_payload_bits_per_buoy"], dtype=np.float64); offered = float(np.sum(scenario.initial_queue_bits) + arrivals)
    fairness = float(payload_by_buoy.sum() ** 2 / (len(payload_by_buoy) * np.square(payload_by_buoy).sum())) if float(np.square(payload_by_buoy).sum()) > 0 else 0.0
    return {"population": population, "scenario_id": scenario.scenario_id, "scenario_seed": int(scenario.seed), "scenario_sha256": _scenario_hash(scenario), "family": family, "model_seed": seed, "num_buoys": int(scenario.num_buoys), "eta": float(scenario.interference_overlap_factor), "interference_tier": str(scenario.spectrum_tier), "queue_tier": str(scenario.queue_tier), "geometry_variant": str(getattr(scenario, "geometry_variant", getattr(scenario, "geometry_group", "registered"))), "bridge": "not_available_without_future_labels" if bridge is None else bool(bridge), "cumulative_payload_bits": float(info["cumulative_payload_bits"]), "completion": float(info["cumulative_payload_bits"]) / max(offered, 1.0), "residual_queue_bits": float(np.sum(info["raw_queue_bits"])), "episode_return": total_return, "jain_fairness": fairness, "movement_energy": float(np.sum(info["cumulative_energy_consumption_per_uav"])), "team_path_length_m": float(np.sum(info["cumulative_distance_m_per_uav"])), "safety_correction_rate": float(np.mean([row["projection_activated"] for row in steps])), "minimum_uav_distance_m": float(min(row["min_uav_distance_m"] for row in steps)), "interference_conflict": bool(any(row["interference_conflict"] for row in steps)), "uses_true_phi": False, "episode_length": len(steps)}, steps


def _paired_families(metrics: Sequence[dict[str, Any]], *, population: str, left_family: str, right_family: str) -> list[dict[str, Any]]:
    by_key = {(row["scenario_id"], int(row["model_seed"]), row["family"]): row for row in metrics}
    records: list[dict[str, Any]] = []
    for scenario, seed, _ in sorted({(row["scenario_id"], int(row["model_seed"]), row["family"]) for row in metrics if row["family"] == left_family}):
        left, right = by_key[(scenario, seed, left_family)], by_key[(scenario, seed, right_family)]
        records.append({"population": population, "scenario_id": scenario, "model_seed": seed, "num_buoys": left["num_buoys"], "eta": left["eta"], "interference_conflict": bool(left["interference_conflict"] or right["interference_conflict"]), "bridge": left["bridge"], **{f"delta_{key}": float(left[key]) - float(right[key]) for key in ("cumulative_payload_bits", "completion", "residual_queue_bits", "episode_return", "jain_fairness", "movement_energy", "safety_correction_rate", "minimum_uav_distance_m")}})
    return records


def _paired(metrics: Sequence[dict[str, Any]], *, population: str) -> list[dict[str, Any]]:
    return _paired_families(metrics, population=population, left_family=PREDICTED_TRANSFORMER, right_family=STATE_ONLY_TRANSFORMER)


def _bootstrap_cluster(values: list[dict[str, Any]], field: str, seed: int) -> tuple[float, float]:
    clusters: dict[str, list[float]] = {}
    for row in values: clusters.setdefault(str(row["scenario_id"]), []).append(float(row[field]))
    means = {key: float(np.mean(value)) for key, value in clusters.items()}
    keys = sorted(means)
    rng = np.random.default_rng(seed); draws = np.empty(2000, dtype=np.float64)
    for index in range(len(draws)): draws[index] = float(np.mean([means[keys[item]] for item in rng.integers(0, len(keys), len(keys))]))
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def _paired_summary(rows: list[dict[str, Any]], *, population: str, field: str, value: str, seed: int) -> dict[str, Any]:
    if not rows: return {"population": population, "group_field": field, "group_value": value, "pair_count": 0, "available": False}
    payload = np.asarray([row["delta_cumulative_payload_bits"] for row in rows], dtype=np.float64); lower, upper = _bootstrap_cluster(rows, "delta_cumulative_payload_bits", seed)
    result = {"population": population, "group_field": field, "group_value": value, "pair_count": len(rows), "available": True, "delta_payload_mean": float(payload.mean()), "delta_payload_median": float(np.median(payload)), "delta_payload_std": float(payload.std(ddof=1)) if len(payload) > 1 else 0.0, "delta_payload_ci95_lower": lower, "delta_payload_ci95_upper": upper, "payload_win_count": int(np.sum(payload > 1.0e-12)), "payload_tie_count": int(np.sum(np.abs(payload) <= 1.0e-12)), "payload_loss_count": int(np.sum(payload < -1.0e-12)), "payload_effect_size": float(payload.mean() / payload.std(ddof=1)) if len(payload) > 1 and payload.std(ddof=1) > 1.0e-12 else 0.0}
    for metric in ("completion", "residual_queue_bits", "episode_return", "jain_fairness", "movement_energy", "safety_correction_rate", "minimum_uav_distance_m"):
        result[f"delta_{metric}_mean"] = float(np.mean([row[f"delta_{metric}"] for row in rows]))
    return result


def _all_summaries(rows: list[dict[str, Any]], population: str) -> list[dict[str, Any]]:
    output = [_paired_summary(rows, population=population, field="overall", value="all", seed=22022)]
    for field in ("num_buoys", "eta", "interference_conflict", "bridge", "model_seed"):
        for value in sorted({str(row[field]) for row in rows}): output.append(_paired_summary([row for row in rows if str(row[field]) == value], population=population, field=field, value=value, seed=22022 + len(output)))
    return output


def _validation_gate(summary: list[dict[str, Any]]) -> dict[str, Any]:
    overall = next(row for row in summary if row["group_field"] == "overall")
    per_seed = [row for row in summary if row["group_field"] == "model_seed"]
    positive = sum(float(row["delta_payload_mean"]) > 0.0 for row in per_seed); negative = sum(float(row["delta_payload_mean"]) < 0.0 for row in per_seed)
    no_harm = overall["delta_completion_mean"] >= -1.0e-12 and overall["delta_residual_queue_bits_mean"] <= 1.0e-12 and overall["delta_safety_correction_rate_mean"] <= 1.0e-12
    if overall["delta_payload_mean"] > 0 and overall["delta_payload_ci95_lower"] > 0 and positive >= 2 and no_harm: gate = "V-A"
    elif overall["delta_payload_mean"] < 0 and overall["delta_payload_ci95_upper"] < 0 and negative >= 2: gate = "V-C"
    else: gate = "V-B"
    return {"gate": gate, "primary_metric": "actual_cumulative_payload_bits", "overall_delta_payload_mean": overall["delta_payload_mean"], "overall_delta_payload_ci95": [overall["delta_payload_ci95_lower"], overall["delta_payload_ci95_upper"]], "positive_seed_count": positive, "negative_seed_count": negative, "completion_not_degraded": overall["delta_completion_mean"] >= -1.0e-12, "residual_queue_not_degraded": overall["delta_residual_queue_bits_mean"] <= 1.0e-12, "safety_not_degraded": overall["delta_safety_correction_rate_mean"] <= 1.0e-12, "fresh_dev_permitted": gate in {"V-A", "V-B"}, "fresh_dev_forbidden": gate == "V-C", "bootstrap_unit": "scenario"}


def _validation_scenarios() -> tuple[Any, ...]:
    # This constructs registered source configurations only; it does not decode
    # any Round-20 output, let alone a protected outcome.
    return tuple(row for row in development_scenarios() if row.split == VALIDATION)


def _run_population(scenarios: Sequence[Any], runs: Sequence[dict[str, Any]], predictor: FrozenDeepSetsPredictor, *, population: str, bridge_map: Mapping[str, bool] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    deployable = [row for row in runs if row["family"] in DEPLOYABLE_FAMILIES]
    metrics: list[dict[str, Any]] = []; steps: list[dict[str, Any]] = []
    for scenario in scenarios:
        for run in deployable:
            bridge = None if bridge_map is None else bridge_map.get(str(scenario.scenario_id), False)
            episode, trajectory = _rollout(scenario, str(run["family"]), int(run["seed"]), Path(run["checkpoint"]), predictor, population=population, bridge=bridge)
            metrics.append(episode); steps.extend(trajectory)
    expected = len(scenarios) * 9
    if len(metrics) != expected or not all(row["uses_true_phi"] is False for row in metrics): raise RuntimeError("closed-loop coverage or causal Phi guard failed")
    return metrics, steps


def run_validation_closed_loop() -> dict[str, Any]:
    if not (R22_OUTPUT / "round22_offline_analysis_manifest.json").is_file(): raise RuntimeError("validation rollout is forbidden before offline analysis")
    runs = _formal_runs(); predictor = FrozenDeepSetsPredictor(); bridge_map = _bridge_by_validation_scenario(predictor)
    metrics, steps = _run_population(_validation_scenarios(), runs, predictor, population="validation", bridge_map=bridge_map)
    paired = _paired(metrics, population="validation"); summary = _all_summaries(paired, "validation"); gate = _validation_gate(summary)
    transformer_mlp = _paired_families(metrics, population="validation", left_family=PREDICTED_TRANSFORMER, right_family=PREDICTED_MLP)
    _csv(R22_OUTPUT / "validation_closed_loop_metrics.csv", metrics); _csv(R22_OUTPUT / "validation_closed_loop_steps.csv", steps); _csv(R22_OUTPUT / "validation_paired_comparison.csv", paired); _csv(R22_OUTPUT / "validation_paired_summary.csv", summary); _csv(R22_OUTPUT / "validation_transformer_vs_mlp_comparison.csv", transformer_mlp); _json(R22_OUTPUT / "round22_validation_policy_gate.json", gate)
    return gate


def _fresh_scenarios_from_manifest() -> tuple[FreshPolicyScenario, ...]:
    path = R22_OUTPUT / "fresh_multi_uav_policy_dev_manifest.json"; manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = "C8570659A88FE062CB7D199EA68468B91A8062C3E320E01F367BDB98CA2988B1"
    if sha256(path) != expected or manifest.get("namespace") != "round22_fresh_multi_uav_policy_dev_v1": raise RuntimeError("fresh-dev manifest is not the registered immutable file")
    fields = set(FreshPolicyScenario.__dataclass_fields__)
    rows = tuple(FreshPolicyScenario(**{key: row[key] for key in fields}) for row in manifest["scenarios"])
    if len(rows) != 60 or len({row.scenario_id for row in rows}) != 60: raise RuntimeError("registered fresh-dev scenario coverage changed")
    return rows


def _freeze_before_fresh(runs: Sequence[dict[str, Any]], predictor: FrozenDeepSetsPredictor) -> None:
    access_path = R22_OUTPUT / "access_audit.json"; access = json.loads(access_path.read_text(encoding="utf-8"))
    if int(access["fresh_dev_access_count"]) != 0: raise RuntimeError("fresh-dev is permitted exactly once")
    checkpoints = [{"family": row["family"], "seed": row["seed"], "checkpoint": row["checkpoint"], "sha256": sha256(Path(row["checkpoint"]))} for row in runs]
    if any(item["sha256"] != next(row["checkpoint_sha256"] for row in runs if row["checkpoint"] == item["checkpoint"]) for item in checkpoints): raise RuntimeError("a formal checkpoint changed before fresh-dev")
    _json(R22_OUTPUT / "round22_pre_fresh_freeze.json", {"all_formal_checkpoints_frozen": True, "selected_deployable_checkpoints_frozen": True, "predictor_sha256": predictor.sha256, "primary_metric": "actual_cumulative_payload_bits", "bootstrap_unit": "scenario", "subgroup_definitions_frozen": ["K", "eta", "interference_conflict", "validation bridge label"], "safety_matching_environment_frozen": True, "fresh_manifest_sha256": sha256(R22_OUTPUT / "fresh_multi_uav_policy_dev_manifest.json"), "checkpoints": checkpoints})
    access["fresh_dev_access_count"] = 1; access["fresh_dev_outcomes_read"] = True; _json(access_path, access)
    _json(R22_OUTPUT / "fresh_dev_access_audit.json", {"fresh_dev_access_count": 1, "access_is_single_use": True, "manifest_sha256_before": sha256(R22_OUTPUT / "fresh_multi_uav_policy_dev_manifest.json"), "outcomes_read": True, "round20_development_test_reuse_count": int(access["round20_development_test_reuse_count"]), "untouched_eval_model_access_count": int(access["untouched_eval_model_access_count"]), "untouched_eval_policy_rollout_count": int(access["untouched_eval_policy_rollout_count"])})


def run_fresh_development(validation_gate: Mapping[str, Any]) -> dict[str, Any] | None:
    if validation_gate["gate"] == "V-C":
        _json(R22_OUTPUT / "fresh_dev_access_audit.json", {"fresh_dev_access_count": 0, "outcomes_read": False, "reason": "validation_gate_V-C_forbids_fresh_development"}); return None
    if validation_gate["gate"] not in {"V-A", "V-B"}: raise RuntimeError("fresh development requires a completed V-A or V-B validation gate")
    runs = _formal_runs(); predictor = FrozenDeepSetsPredictor(); _freeze_before_fresh(runs, predictor)
    metrics, steps = _run_population(_fresh_scenarios_from_manifest(), runs, predictor, population="fresh_development", bridge_map=None)
    paired = _paired(metrics, population="fresh_development"); summary = _all_summaries(paired, "fresh_development")
    transformer_mlp = _paired_families(metrics, population="fresh_development", left_family=PREDICTED_TRANSFORMER, right_family=PREDICTED_MLP)
    _csv(R22_OUTPUT / "fresh_dev_closed_loop_metrics.csv", metrics); _csv(R22_OUTPUT / "fresh_dev_closed_loop_steps.csv", steps); _csv(R22_OUTPUT / "fresh_dev_paired_comparison.csv", paired); _csv(R22_OUTPUT / "fresh_dev_paired_summary.csv", summary); _csv(R22_OUTPUT / "fresh_dev_transformer_vs_mlp_comparison.csv", transformer_mlp)
    return {"metrics": metrics, "paired": paired, "summary": summary}


__all__ = ["run_fresh_development", "run_validation_closed_loop"]
