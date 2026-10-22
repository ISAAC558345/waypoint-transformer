"""Frozen Round-27 same-snapshot planner-action utility diagnostics.

This module trains nothing.  It compares exactly one H=5 planner first action
with one H=1 reference first action from the same registered Round-20 reset
snapshot, then applies the same deterministic H=1 continuation policy to both
branches through the actual frozen environment communication transition.
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
from wt.round19.multi_env import SharedSpectrumMultiUAVEnv, assignment_link_metrics, gain_matrix, joint_aware_matching
from wt.round19.planner import joint_plan
from wt.round20.scenarios import development_scenarios
from wt.round22.data import FrozenDeepSetsPredictor, VALIDATION, load_policy_sequences
from wt.round22.models import PREDICTED_TRANSFORMER
from wt.round22.training import TrainingConfig, load_checkpoint, predict
from wt.round26.audit import verify_frozen_state as verify_round26_frozen_state


ROOT = Path(__file__).resolve().parents[3]
ROUND = ROOT / "round_27"
OUTPUT = ROUND / "outputs" / "round27_conditional_planner_utility_20260830_1447"
FIGURES = ROUND / "figures" / "round27_conditional_planner_utility_20260830_1447"
CONFIG_PATH = ROUND / "configs" / "round27_conditional_planner_utility_config.json"
R20 = ROOT / "round_20" / "outputs" / "round20_multi_uav_dataset_20260828_2100"
R22 = ROOT / "round_22" / "outputs" / "round22_deepsets_task_waypoint_policy_20260829_1153"
R24 = ROOT / "round_24" / "outputs" / "round24_multiseed_stability_20260829_1515"
R26 = ROOT / "round_26" / "outputs" / "round26_action_ambiguity_20260830_1322"
EXPECTED_R26 = "606558C28019A0B4C49DB0CE1B09B10CCC49750CAEF8648B1648D3BC66666FEE"
ALLOWED = frozenset(("train", "validation"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Mapping): return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(item) for item in value]
    return value


def _json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized: raise RuntimeError(f"unexpected empty Round-27 output: {path.name}")
    fields = sorted({name for row in materialized for name in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in materialized:
            writer.writerow({name: json.dumps(_plain(value), sort_keys=True) if isinstance(value, (dict, list, tuple, np.ndarray)) else value for name, value in row.items()})


def _allowed_rows(path: Path) -> list[dict[str, Any]]:
    """Filter protected split lines before decoding any JSON field."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if '"split":"train"' not in raw and '"split":"validation"' not in raw:
                continue
            row = json.loads(raw)
            if row.get("split") not in ALLOWED:
                raise RuntimeError("protected Round-20 row was decoded")
            rows.append(row)
    return rows


def verify_frozen_state() -> dict[str, Any]:
    """Fail closed before any Round-20 state is decoded or replayed."""
    if _digest(ROUND.parent / "round_26" / "ROUND_LOCK.json") != EXPECTED_R26:
        raise RuntimeError("Round-26 original lock changed")
    amendment = _load(ROUND.parent / "round_26" / "ROUND_LOCK_AMENDMENT.json")
    affected = amendment.get("affected_paths", {})
    expected = {
        "src/wt/governance_state_sync.py": ROOT / "src" / "wt" / "governance_state_sync.py",
        "tests/test_round24_seed_extension.py": ROOT / "tests" / "test_round24_seed_extension.py",
        "tests/test_round21_hgnn_task_waypoint.py": ROOT / "tests" / "test_round21_hgnn_task_waypoint.py",
        "tests/test_governance_state_sync.py": ROOT / "tests" / "test_governance_state_sync.py",
    }
    if amendment.get("amendment_type") != "legal_round27_successor_governance_extension" or amendment.get("original_round26_lock_sha256") != EXPECTED_R26:
        raise RuntimeError("Round-26 legal-successor overlay is invalid")
    if any(_digest(path) != approved_lifecycle_overlay_sha256(name, affected.get(name, {}).get("amended_sha256", "")) for name, path in expected.items()):
        raise RuntimeError("Round-26 legal-successor overlay hashes are invalid")
    if any(amendment.get(name) is not False for name in ("scientific_artifacts_modified", "model_checkpoints_modified", "dataset_modified", "metrics_modified", "fresh_dev_rerun", "untouched_eval_accessed", "original_round26_lock_modified")):
        raise RuntimeError("Round-26 amendment modifies protected evidence")
    prior = verify_round26_frozen_state()
    governance = verify_round22_governance_state_sync_amendment()
    return {"passed": True, "round26_original_lock_sha256": EXPECTED_R26, "round26_frozen_chain": prior, "round22_governance_verified": bool(governance["passed"]), "fresh_new_rollout_count": 0, "round20_development_test_reuse_count": 0, "untouched_eval_access_count": 0}


def _scenario_map() -> dict[str, Any]:
    return {item.scenario_id: item for item in development_scenarios() if item.split in ALLOWED}


def _state_signature(info: Mapping[str, Any]) -> tuple[bytes, bytes, bytes, int]:
    return (np.asarray(info["uav_positions_m"], dtype=np.float64).tobytes(), np.asarray(info["raw_queue_bits"], dtype=np.float64).tobytes(), np.asarray(info["remaining_energy_per_uav"], dtype=np.float64).tobytes(), int(info["time_step"]))


def _initial_env(scenario: Any, graph: Mapping[str, Any]) -> tuple[SharedSpectrumMultiUAVEnv, dict[str, Any]]:
    env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m, dtype=np.float64), options=scenario.options())
    _, info = env.reset(seed=scenario.seed)
    positions = np.asarray(graph["uav_nodes"], dtype=np.float64)[: env.num_uavs, :2]
    queues = np.asarray(graph["buoy_nodes"], dtype=np.float64)[: env.num_buoys, 2]
    energy = np.asarray(graph["uav_nodes"], dtype=np.float64)[: env.num_uavs, 4]
    if not (np.array_equal(np.asarray(info["uav_positions_m"]), positions) and np.array_equal(np.asarray(info["raw_queue_bits"]), queues) and np.array_equal(np.asarray(info["remaining_energy_per_uav"]), energy) and int(info["time_step"]) == 0):
        raise RuntimeError("registered Round-20 t=0 graph is not the exact environment reset snapshot")
    return env, info


def _feature_row(scenario: Any, graph: Mapping[str, Any]) -> dict[str, Any]:
    env, info = _initial_env(scenario, graph)
    positions = np.asarray(info["uav_positions_m"], dtype=np.float64)
    queues = np.asarray(info["raw_queue_bits"], dtype=np.float64)
    buoys = np.asarray(info["buoy_positions_m"], dtype=np.float64)
    gains, _ = gain_matrix(positions, buoys, env.config)
    direct = float(env.config.total_bandwidth_hz) * np.log2(1.0 + float(env.config.buoy_tx_power_w) * gains / float(env.config.noise_power_w))
    weighted = queues[None, :] / float(env.config.queue_capacity_bits) * direct
    intent = np.argmax(weighted, axis=1)
    ordered = np.sort(weighted, axis=1)
    top, runner_up = ordered[:, -1], ordered[:, -2]
    competition = float(np.mean(np.clip(1.0 - (top - runner_up) / np.maximum(top, 1.0e-12), 0.0, 1.0)))
    assignment, _, _, _ = joint_aware_matching(queues, gains, env.config, float(scenario.interference_overlap_factor))
    _, _, _, isr, _ = assignment_link_metrics(assignment, gains, env.config, float(scenario.interference_overlap_factor))
    pair = float(np.linalg.norm(positions[0] - positions[1]))
    buoy_pairs = [float(np.linalg.norm(buoys[i] - buoys[j])) for i in range(env.num_buoys) for j in range(i + 1, env.num_buoys)]
    duplicate = bool(intent[0] == intent[1])
    assignment_conflict = bool(duplicate and assignment[0] >= 0 and assignment[1] >= 0 and assignment[0] != assignment[1])
    interference = bool(float(scenario.interference_overlap_factor) > 0.0 and float(np.max(isr)) >= float(env.options.conflict_isr_threshold))
    current_assignment = np.asarray(graph["uav_to_buoy_assignment"], dtype=np.int64)[: env.num_uavs]
    served_diversity = len({int(value) for value in current_assignment if value >= 0})
    return {"queue_pressure": float(np.mean(queues / float(env.config.queue_capacity_bits))), "queue_imbalance": float((np.max(queues) - np.min(queues)) / float(env.config.queue_capacity_bits)), "mean_direct_rate_bps": float(np.mean(direct)), "mean_queue_weighted_rate_bps": float(np.mean(weighted)), "max_current_isr": float(np.max(isr)), "uav_uav_distance_m": pair, "mean_uav_buoy_distance_m": float(np.mean(np.linalg.norm(buoys[None] - positions[:, None], axis=-1))), "safety_margin_m": pair - float(env.config.multi_uav_min_separation_m), "spatial_spread_m": float(np.mean(buoy_pairs)), "num_buoys": env.num_buoys, "eta": float(scenario.interference_overlap_factor), "current_served_buoy_diversity": served_diversity, "duplicate_intent_current": duplicate, "assignment_conflict_current": assignment_conflict, "interference_conflict_current": interference, "queue_competition_score": competition, "current_assignment": assignment.tolist(), "current_greedy_intent": intent.tolist()}


def _branch(scenario: Any, graph: Mapping[str, Any], *, first_horizon: int, total_horizon: int) -> dict[str, Any]:
    env, info = _initial_env(scenario, graph)
    signature = _state_signature(info)
    initial_queue = float(np.sum(np.asarray(info["raw_queue_bits"], dtype=np.float64)))
    decision = joint_plan(env, info, horizon=first_horizon)
    action = decision.action.copy()
    payload = 0.0
    episode_return = 0.0
    first_applied: np.ndarray | None = None
    for step in range(total_horizon):
        if step:
            action = joint_plan(env, info, horizon=1).action
        _, reward, terminated, truncated, after = env.step(action, matching_mode="joint_aware")
        if step == 0: first_applied = np.asarray(after["normalized_applied_action"], dtype=np.float64)
        payload += float(np.sum(np.asarray(after["transmitted_payload_bits"], dtype=np.float64)))
        episode_return += float(reward)
        info = after
        if terminated or truncated:
            raise RuntimeError("registered diagnostic horizon terminated unexpectedly")
    offered = initial_queue + total_horizon * float(env.num_buoys) * float(env.config.constant_arrival_bits_per_step)
    return {"snapshot_signature": signature, "requested_first_action": decision.action, "applied_first_action": first_applied, "payload_bits": payload, "completion": payload / max(offered, 1.0), "residual_queue_bits": float(np.sum(np.asarray(info["raw_queue_bits"], dtype=np.float64))), "episode_return": episode_return, "first_decision_future_trace_accessed": bool(decision.future_trace_accessed), "first_decision_predicted_payload_bits": float(decision.predicted_payload_bits), "first_horizon": first_horizon, "total_horizon": total_horizon}


def _utility_rows(pool: Sequence[Mapping[str, Any]], scenarios: Mapping[str, Any], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    protocol = config["counterfactual_protocol"]
    tolerance = float(protocol["numerical_utility_tolerance_bits"])
    rows: list[dict[str, Any]] = []
    for item in pool:
        scenario = scenarios[str(item["scenario_id"])]
        graph = item["graph_state_t"]
        feature = _feature_row(scenario, graph)
        h5 = _branch(scenario, graph, first_horizon=int(protocol["planner_first_horizon"]), total_horizon=int(protocol["primary_horizon"]))
        h1 = _branch(scenario, graph, first_horizon=int(protocol["reference_first_horizon"]), total_horizon=int(protocol["primary_horizon"]))
        h10 = _branch(scenario, graph, first_horizon=10, total_horizon=int(protocol["secondary_horizon"]))
        h1_10 = _branch(scenario, graph, first_horizon=int(protocol["reference_first_horizon"]), total_horizon=int(protocol["secondary_horizon"]))
        if not (h5["snapshot_signature"] == h1["snapshot_signature"] == h10["snapshot_signature"] == h1_10["snapshot_signature"]):
            raise RuntimeError("counterfactual branches do not share an exact reset snapshot")
        if h5["first_decision_future_trace_accessed"] or h1["first_decision_future_trace_accessed"] or h10["first_decision_future_trace_accessed"]:
            raise RuntimeError("frozen planner accessed future trace")
        utility = float(h5["payload_bits"] - h1["payload_bits"])
        label = "useful" if utility > tolerance else "harmful" if utility < -tolerance else "neutral"
        h5_applied, h1_applied = np.asarray(h5["applied_first_action"]), np.asarray(h1["applied_first_action"])
        rows.append({"state_id": f"{scenario.scenario_id}__t00", "scenario_id": scenario.scenario_id, "t": 0, "split": scenario.split, "utility_payload": utility, "utility_completion": float(h5["completion"] - h1["completion"]), "utility_queue": float(h1["residual_queue_bits"] - h5["residual_queue_bits"]), "utility_return": float(h5["episode_return"] - h1["episode_return"]), "utility_class": label, "planner_payload_bits": h5["payload_bits"], "reference_payload_bits": h1["payload_bits"], "planner_residual_queue_bits": h5["residual_queue_bits"], "reference_residual_queue_bits": h1["residual_queue_bits"], "planner_first_action": h5_applied, "reference_first_action": h1_applied, "planner_h5_to_h1_first_action_rmse": float(np.sqrt(np.mean((h5_applied - h1_applied) ** 2))), "same_snapshot_verified": True, "only_first_action_differs": True, "continuation_policy": "frozen_causal_joint_h1", "continuation_horizon": int(protocol["continuation_horizon"]), "secondary_h10_utility_payload": float(h10["payload_bits"] - h1_10["payload_bits"]), "secondary_h10_payload_bits": h10["payload_bits"], "secondary_h1_payload_bits": h1_10["payload_bits"], **feature})
    return rows


def _stats(rows: Sequence[Mapping[str, Any]], *, group_type: str, group_value: str) -> dict[str, Any]:
    values = np.asarray([float(row["utility_payload"]) for row in rows], dtype=np.float64)
    if not values.size: return {"group_type": group_type, "group_value": group_value, "state_count": 0, "mean_utility_payload": 0.0, "median_utility_payload": 0.0, "p10_utility_payload": 0.0, "p50_utility_payload": 0.0, "p90_utility_payload": 0.0, "useful_ratio": 0.0, "neutral_ratio": 0.0, "harmful_ratio": 0.0, "mean_positive_utility": 0.0, "mean_harmful_utility": 0.0}
    return {"group_type": group_type, "group_value": group_value, "state_count": int(values.size), "mean_utility_payload": float(np.mean(values)), "median_utility_payload": float(np.median(values)), "p10_utility_payload": float(np.percentile(values, 10)), "p50_utility_payload": float(np.percentile(values, 50)), "p90_utility_payload": float(np.percentile(values, 90)), "useful_ratio": float(np.mean(values > 1.0e-6)), "neutral_ratio": float(np.mean(np.abs(values) <= 1.0e-6)), "harmful_ratio": float(np.mean(values < -1.0e-6)), "mean_positive_utility": float(np.mean(values[values > 1.0e-6])) if np.any(values > 1.0e-6) else 0.0, "mean_harmful_utility": float(np.mean(values[values < -1.0e-6])) if np.any(values < -1.0e-6) else 0.0}


def _bootstrap(values: Sequence[float], *, seed: int, resamples: int) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0: return 0.0, 0.0
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(int(resamples), data.size), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _trigger_rows(rows: Sequence[dict[str, Any]], threshold: float) -> None:
    for row in rows:
        row["trigger_A_interference_conflict"] = bool(row["interference_conflict_current"])
        row["trigger_B_duplicate_intent"] = bool(row["duplicate_intent_current"])
        row["trigger_C_queue_interference"] = bool(row["interference_conflict_current"] and float(row["queue_competition_score"]) >= threshold)
        row["trigger_D_assignment_conflict"] = bool(row["assignment_conflict_current"])
        row["high_queue_pressure"] = bool(float(row["queue_competition_score"]) >= threshold)


def _trigger_metrics(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], threshold: float) -> list[dict[str, Any]]:
    gates, result = config["pre_registered_gates"], []
    for split in ("train", "validation"):
        population = [row for row in rows if row["split"] == split]
        for name in ("A_interference_conflict", "B_duplicate_intent", "C_queue_interference", "D_assignment_conflict"):
            field = f"trigger_{name}"
            active, inactive = [row for row in population if bool(row[field])], [row for row in population if not bool(row[field])]
            useful = [row for row in population if row["utility_class"] == "useful"]
            useful_active = [row for row in active if row["utility_class"] == "useful"]
            harmful = [row for row in active if row["utility_class"] == "harmful"]
            true_negative = [row for row in inactive if row["utility_class"] != "useful"]
            ci_low, ci_high = _bootstrap([float(row["utility_payload"]) for row in active], seed=int(gates["bootstrap_seed"]) + len(result), resamples=int(gates["bootstrap_resamples"]))
            result.append({"trigger": name, "split": split, "threshold_source": "train_p75" if name == "C_queue_interference" else "none", "queue_competition_threshold": threshold, "population_count": len(population), "activation_count": len(active), "activation_rate": len(active) / max(len(population), 1), "precision_useful": len(useful_active) / max(len(active), 1), "recall_useful": len(useful_active) / max(len(useful), 1), "specificity": len(true_negative) / max(len(inactive), 1), "average_active_utility_payload": float(np.mean([float(row["utility_payload"]) for row in active])) if active else 0.0, "average_inactive_utility_payload": float(np.mean([float(row["utility_payload"]) for row in inactive])) if inactive else 0.0, "active_utility_ci95_low": ci_low, "active_utility_ci95_high": ci_high, "harmful_action_rate_active": len(harmful) / max(len(active), 1), "oracle_trigger": False, "deployable": True})
    return result


def _corr(left: Sequence[float], right: Sequence[float]) -> tuple[float, float]:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if a.size < 3 or np.std(a) <= 1.0e-12 or np.std(b) <= 1.0e-12: return 0.0, 0.0
    order_a, order_b = np.argsort(np.argsort(a, kind="mergesort"), kind="mergesort"), np.argsort(np.argsort(b, kind="mergesort"), kind="mergesort")
    return float(np.corrcoef(a, b)[0, 1]), float(np.corrcoef(order_a, order_b)[0, 1])


def _policy_relations(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validation = [row for row in rows if row["split"] == "validation"]
    predictor = FrozenDeepSetsPredictor(); sequence = load_policy_sequences(VALIDATION, predictor)
    selected, seen = [], set()
    for index, (scenario, timestep) in enumerate(zip(sequence.metadata["scenario_id"], sequence.metadata["timestep"])):
        if int(timestep) == 0 and str(scenario) not in seen: selected.append(index); seen.add(str(scenario))
    if seen != {str(row["scenario_id"]) for row in validation}: raise RuntimeError("frozen policy validation states do not cover Round-27 state pool")
    manifest22, manifest24 = _load(R22 / "policy_checkpoint_manifest.json"), _load(R24 / "checkpoint_manifest.json")
    records = [row for row in manifest22["checkpoints"] + manifest24["new_round24_checkpoints"] if row["family"] == PREDICTED_TRANSFORMER]
    if {int(row["seed"]) for row in records} != set(range(8)): raise RuntimeError("Round-27 requires frozen Task-Waypoint seeds 0--7")
    by_scenario: dict[str, list[np.ndarray]] = defaultdict(list)
    for record in sorted(records, key=lambda row: int(row["seed"])):
        family, model, normalizer, payload = load_checkpoint(ROOT / record["checkpoint"])
        if family != PREDICTED_TRANSFORMER or int(payload["seed"]) != int(record["seed"]): raise RuntimeError("frozen Task-Waypoint checkpoint identity changed")
        actions = predict(family, model, normalizer.apply(sequence), TrainingConfig())
        for index in selected: by_scenario[str(sequence.metadata["scenario_id"][index])].append(np.asarray(actions[index], dtype=np.float64))
    paired = list(csv.DictReader((R24 / "eight_seed_validation_paired.csv").open(encoding="utf-8", newline="")))
    delta: dict[str, list[float]] = defaultdict(list)
    for row in paired:
        delta[str(row["scenario_id"])].append(float(row["delta_cumulative_payload_bits"]))
    task_rows, seed_rows = [], []
    for row in validation:
        scenario = str(row["scenario_id"]); actions = by_scenario[scenario]
        pairs = [float(np.sqrt(np.mean((actions[i] - actions[j]) ** 2))) for i in range(len(actions)) for j in range(i + 1, len(actions))]
        if len(actions) != 8 or scenario not in delta: raise RuntimeError("frozen validation policy evidence is incomplete")
        task_rows.append({"scenario_id": scenario, "split": "validation", "planner_utility_payload": float(row["utility_payload"]), "taskwp_minus_state_payload_bits_mean": float(np.mean(delta[scenario])), "taskwp_seed_action_disagreement_rmse": float(np.mean(pairs)), "seed_count": len(actions)})
        seed_rows.append({"scenario_id": scenario, "split": "validation", "planner_utility_payload": float(row["utility_payload"]), "taskwp_seed_action_disagreement_rmse": float(np.mean(pairs)), "seed_count": len(actions)})
    for target, value in ((task_rows, "taskwp_minus_state_payload_bits_mean"), (seed_rows, "taskwp_seed_action_disagreement_rmse")):
        pearson, spearman = _corr([float(row["planner_utility_payload"]) for row in target], [float(row[value]) for row in target])
        target.append({"scenario_id": "__summary__", "split": "validation", "planner_utility_payload": 0.0, value: 0.0, "seed_count": 8, "pearson": pearson, "spearman": spearman})
    return task_rows, seed_rows


def _figures(rows: Sequence[Mapping[str, Any]], grouped: Sequence[Mapping[str, Any]], metrics: Sequence[Mapping[str, Any]], oracle: Sequence[Mapping[str, Any]], behavior: Sequence[Mapping[str, Any]], task: Sequence[Mapping[str, Any]], seed: Sequence[Mapping[str, Any]]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    for path in FIGURES.iterdir():
        if path.is_file(): path.unlink()
    def save(fig: Any, name: str) -> None:
        fig.tight_layout(); fig.savefig(FIGURES / f"{name}.png", dpi=150); fig.savefig(FIGURES / f"{name}.svg"); plt.close(fig)
    def bar(name: str, labels: Sequence[str], values: Sequence[float], title: str, ylabel: str = "payload utility (bits)") -> None:
        fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(range(len(labels)), values); ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right"); ax.set(title=title, ylabel=ylabel); ax.axhline(0, color="black", linewidth=.8); ax.grid(axis="y", alpha=.25); save(fig, name)
    values = [float(row["utility_payload"]) for row in rows]; fig, ax = plt.subplots(figsize=(7,4)); ax.hist(values, bins=16); ax.axvline(0,color="black"); ax.set(title="Planner utility distribution",xlabel="payload utility (bits)",ylabel="states"); save(fig,"01_utility_distribution")
    labels=["useful","neutral","harmful"];bar("02_utility_classes",labels,[sum(r["utility_class"]==x for r in rows)/len(rows) for x in labels],"Useful / neutral / harmful ratio","ratio")
    for ordinal, group_type, title in ((3,"interference_conflict_current","Current interference-conflict utility"),(4,"duplicate_intent_current","Duplicate-intent utility"),(6,"assignment_conflict_current","Assignment-conflict utility"),(7,"num_buoys","K-stratified utility"),(8,"eta","Eta-stratified utility"),(13,"queue_pressure","Queue-pressure utility")):
        selected=[row for row in grouped if row["group_type"]==group_type]; bar(f"{ordinal:02d}_{group_type}",[str(r["group_value"]) for r in selected],[float(r["mean_utility_payload"]) for r in selected],title)
    fig,ax=plt.subplots(figsize=(7,4));ax.scatter([r["queue_competition_score"] for r in rows],values);ax.set(title="Queue competition versus utility",xlabel="queue competition",ylabel="payload utility (bits)");ax.axhline(0,color="black");save(fig,"05_queue_competition_utility")
    bar("09_oracle_routing",[r["split"] for r in oracle],[float(r["oracle_mean_payload_gain_bits"]) for r in oracle],"Oracle conditional routing upper bound")
    val=[r for r in metrics if r["split"]=="validation"];bar("10_trigger_precision_recall",[r["trigger"] for r in val],[r["precision_useful"] for r in val],"Validation trigger precision","precision")
    bar("11_trigger_active_utility",[r["trigger"] for r in val],[r["average_active_utility_payload"] for r in val],"Validation active-state utility")
    bar("12_trigger_harmful_rate",[r["trigger"] for r in val],[r["harmful_action_rate_active"] for r in val],"Validation trigger harmful rate","rate")
    names=sorted({r["behavior_generator"] for r in behavior});bar("14_behavior_alignment",names,[np.mean([float(r["planner_behavior_action_rmse"]) for r in behavior if r["behavior_generator"]==n]) for n in names],"Planner / behavior action alignment","RMSE")
    clean=[r for r in seed if r["scenario_id"]!="__summary__"];fig,ax=plt.subplots(figsize=(7,4));ax.scatter([r["planner_utility_payload"] for r in clean],[r["taskwp_seed_action_disagreement_rmse"] for r in clean]);ax.set(title="Planner utility versus seed disagreement",xlabel="planner utility",ylabel="seed action RMSE");save(fig,"15_utility_seed_disagreement")
    clean=[r for r in task if r["scenario_id"]!="__summary__"];fig,ax=plt.subplots(figsize=(7,4));ax.scatter([r["planner_utility_payload"] for r in clean],[r["taskwp_minus_state_payload_bits_mean"] for r in clean]);ax.set(title="Planner utility versus Task-WP value",xlabel="planner utility",ylabel="Task-WP minus state payload");save(fig,"16_utility_taskwp_relation")
    files=sorted(path.name for path in FIGURES.iterdir() if path.suffix in {".png",".svg"})
    if len(files)!=32: raise RuntimeError("Round-27 requires exactly 16 PNG+SVG pairs")
    return files


def run_round27() -> dict[str, Any]:
    frozen, config = verify_frozen_state(), _load(CONFIG_PATH)
    transitions = [row for row in _allowed_rows(R20 / "multi_uav_transitions.jsonl") if int(row["timestep"]) == 0]
    h5 = {str(row["scenario_id"]): row for row in transitions if row["behavior_generator"] == "joint_h5"}
    h1 = {str(row["scenario_id"]): row for row in transitions if row["behavior_generator"] == "joint_h1"}
    h10 = {str(row["scenario_id"]): row for row in transitions if row["behavior_generator"] == "joint_h10"}
    scenarios = _scenario_map()
    if set(h5) != set(scenarios) or set(h1) != set(scenarios) or set(h10) != set(scenarios) or len(scenarios) != 64:
        raise RuntimeError("registered Round-27 t=0 state pool is incomplete")
    bridge_info = {row["scenario_id"]: row for row in csv.DictReader((R26 / "initial_state_action_ambiguity.csv").open(encoding="utf-8", newline=""))}
    pool = [h5[name] for name in sorted(scenarios)]
    rows = _utility_rows(pool, scenarios, config)
    for row in rows:
        control_h5, control_h1 = np.asarray(h5[row["scenario_id"]]["joint_action_applied_t"], dtype=np.float64), np.asarray(h1[row["scenario_id"]]["joint_action_applied_t"], dtype=np.float64)
        if float(np.max(np.abs(np.asarray(row["planner_first_action"]) - control_h5))) > 1.0e-9 or float(np.max(np.abs(np.asarray(row["reference_first_action"]) - control_h1))) > 1.0e-9:
            raise RuntimeError("frozen planner action identity control changed")
        diagnostic = bridge_info[row["scenario_id"]]; row["diagnostic_bridge"] = diagnostic["bridge_any"] == "True"; row["diagnostic_conflict"] = diagnostic["conflict_any"] == "True"
    train = [row for row in rows if row["split"] == "train"]
    threshold = float(np.quantile([float(row["queue_competition_score"]) for row in train], float(config["trigger_rules"]["threshold_quantile"])))
    _trigger_rows(rows, threshold)
    grouped = [_stats(rows, group_type="overall", group_value="all"), _stats(train, group_type="split", group_value="train"), _stats([r for r in rows if r["split"]=="validation"], group_type="split", group_value="validation")]
    for field, name, values in (("interference_conflict_current","interference_conflict_current",(False,True)),("diagnostic_bridge","diagnostic_bridge",(False,True)),("duplicate_intent_current","duplicate_intent_current",(False,True)),("assignment_conflict_current","assignment_conflict_current",(False,True)),("high_queue_pressure","queue_pressure",(False,True))):
        grouped.extend(_stats([row for row in rows if bool(row[field])==value],group_type=name,group_value=str(value)) for value in values)
    for field in ("num_buoys","eta"):
        grouped.extend(_stats([row for row in rows if row[field]==value],group_type=field,group_value=str(value)) for value in sorted({row[field] for row in rows}))
    distribution=[]
    for split, population in (("all",rows),("train",train),("validation",[r for r in rows if r["split"]=="validation"])):
        distribution.append({"split":split,**_stats(population,group_type="distribution",group_value=split)})
    metrics = _trigger_metrics(rows, config, threshold)
    oracle=[]
    for split in ("train","validation","all"):
        population=rows if split=="all" else [row for row in rows if row["split"]==split]; gains=[max(float(row["utility_payload"]),0.0) for row in population]; low,high=_bootstrap(gains,seed=int(config["pre_registered_gates"]["bootstrap_seed"])+99+len(oracle),resamples=int(config["pre_registered_gates"]["bootstrap_resamples"])); oracle.append({"split":split,"state_count":len(population),"oracle_trigger":True,"deployable":False,"oracle_mean_payload_gain_bits":float(np.mean(gains)),"oracle_total_payload_gain_bits":float(np.sum(gains)),"oracle_ci95_low_bits":low,"oracle_ci95_high_bits":high})
    behavior=[]
    for row in rows:
        for behavior_name, control in (("joint_h5",h5[row["scenario_id"]]),("joint_h10",h10[row["scenario_id"]])):
            planner_action = np.asarray(row["planner_first_action"], dtype=np.float64)
            behavior_action = np.asarray(control["joint_action_applied_t"], dtype=np.float64)
            behavior.append({"scenario_id":row["scenario_id"],"split":row["split"],"behavior_generator":behavior_name,"planner_utility_payload":row["utility_payload"],"trigger_A_interference_conflict":row["trigger_A_interference_conflict"],"trigger_B_duplicate_intent":row["trigger_B_duplicate_intent"],"trigger_C_queue_interference":row["trigger_C_queue_interference"],"trigger_D_assignment_conflict":row["trigger_D_assignment_conflict"],"planner_behavior_action_rmse":float(np.sqrt(np.mean((planner_action - behavior_action) ** 2)))})
    task_rows, seed_rows = _policy_relations(rows)
    train_metrics=[row for row in metrics if row["split"]=="train"]; selected=max(train_metrics,key=lambda row:(float(row["average_active_utility_payload"]),-"ABCD".index(row["trigger"][0])))
    selected_name=selected["trigger"]; validation_metric=next(row for row in metrics if row["split"]=="validation" and row["trigger"]==selected_name)
    val_active=[row for row in rows if row["split"]=="validation" and bool(row[f"trigger_{selected_name}"])]
    stability_rows=[]
    for field, expected_values in (("num_buoys", (3, 5)), ("eta", (0.25, 0.5, 1.0))):
        for value in expected_values:
            active=[row for row in val_active if row[field] == value]
            stability_rows.append({"record_type":"validation_active_stratum","trigger":selected_name,"split":"validation","stratum_field":field,"stratum_value":value,"activation_count":len(active),"mean_active_utility_payload":float(np.mean([float(row["utility_payload"]) for row in active])) if active else 0.0,"harmful_action_rate_active":float(np.mean([row["utility_class"]=="harmful" for row in active])) if active else 0.0})
    gates=config["pre_registered_gates"]; oracle_validation=next(row for row in oracle if row["split"]=="validation")
    coverage=all(int(row["activation_count"]) > 0 for row in stability_rows)
    stable=bool(len(val_active)>=int(gates["minimum_validation_active_states"]) and float(validation_metric["average_active_utility_payload"])>float(gates["trigger_validation_mean_utility_min_bits"]) and float(validation_metric["active_utility_ci95_low"])>float(gates["trigger_validation_ci_lower_min_bits"]) and float(validation_metric["harmful_action_rate_active"])<=float(gates["trigger_harmful_rate_max"]) and coverage and all(float(row["mean_active_utility_payload"])>=float(gates["catastrophic_stratum_mean_utility_min_bits"]) for row in stability_rows))
    oracle_value=bool(float(oracle_validation["oracle_mean_payload_gain_bits"])>=float(gates["oracle_mean_payload_gain_min_bits"]))
    useful_ratio=float(next(row for row in distribution if row["split"]=="all")["useful_ratio"])
    cpa="CPA-A" if useful_ratio>=float(gates["nontrivial_useful_ratio_min"]) and oracle_value and stable else "CPA-B" if oracle_value else "CPA-C"
    trigger_gate="TR-A" if stable else "TR-B" if oracle_value and len(val_active)>0 else "TR-C"
    ready="R27-A_conditional_planner_action_distillation_training_recommendation_only" if cpa=="CPA-A" and trigger_gate=="TR-A" else "R27-B_continue_trigger_representation_study_no_training" if cpa=="CPA-B" or trigger_gate=="TR-B" else "R27-C_stop_current_task_waypoint_planner_distillation_route"
    cpa_gate={"round":"round_27","gate":cpa,"conditional_planner_action_distillation_ready":cpa=="CPA-A","planner_useful_ratio":useful_ratio,"oracle_validation_mean_payload_gain_bits":oracle_validation["oracle_mean_payload_gain_bits"],"selected_train_trigger":selected_name,"criteria":{"nontrivial_useful_prevalence":useful_ratio>=float(gates["nontrivial_useful_ratio_min"]),"oracle_headroom":oracle_value,"stable_causal_trigger":stable}}
    trigger_gate_json={"round":"round_27","gate":trigger_gate,"selected_by":"highest train active-state utility; fixed A/B/C/D tie order","selected_train_trigger":selected_name,"validation_metrics":validation_metric,"validation_k_eta_coverage":coverage,"validation_k_eta_stability":stability_rows,"train_queue_competition_p75":threshold,"deployable_features_only":True,"bridge_used_as_trigger":False,"trigger_stable":stable}
    readiness={"round":"round_27","gate":ready.split("_")[0],"recommendation":ready,"training_recommendation_only":True,"training_performed":False,"distillation_training_performed":False,"fresh_new_rollout_count":0,"untouched_eval_access_count":0,"round28_created":False}
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/"planner_utility_definition.md").write_text("# Planner utility definition\n\nPrimary utility is `U_5(s) = Payload_5(H5 first action, frozen causal H1 continuation) - Payload_5(H1 first action, identical frozen causal H1 continuation)`. Both actual environment branches reset the same registered scenario seed and preserve positions, queues, channel/static eta, arrivals, matching, safety, and horizon; only the requested first joint action differs. H=10 is a secondary diagnostic. Useful/neutral/harmful use the pre-registered 1e-6-bit numerical tolerance.\n",encoding="utf-8")
    (OUTPUT/"causal_trigger_feature_spec.md").write_text("# Causal trigger feature specification\n\nAll features are calculated from the current reset snapshot only: queue pressure/imbalance, direct and queue-weighted rates, joint-aware current ISR/matching, UAV and buoy geometry, safety margin, K, eta, current served-buoy diversity, greedy duplicate intent, and centralized assignment conflict. `interference_conflict_current` is current joint-aware max ISR >= 0.25 in shared eta>0. `duplicate_intent_current` compares current greedy queue-rate buoy indices. `assignment_conflict_current` is duplicate intent resolved to distinct current matching assignments. Bridge, behavior, quality, Task-Waypoint, future payload/rates/assignments, and planner action are excluded. Trigger C uses only the train P75 queue-competition threshold.\n",encoding="utf-8")
    (OUTPUT/"conditional_distillation_design.md").write_text("# Conditional distillation design (no training)\n\nDesign A would add a planner auxiliary action loss only when a frozen causal trigger is positive while retaining mixed behavior action supervision. Design B would replace the action target with the planner first action only when that trigger is positive. Neither is selected or trained in Round-27. Both require future authorization and must not use Task-Waypoint as the primary action-mode disambiguator because Round-26 closed TD-C.\n",encoding="utf-8")
    _csv(OUTPUT/"planner_state_utility_labels.csv",rows); _csv(OUTPUT/"planner_utility_by_group.csv",grouped); _csv(OUTPUT/"planner_utility_distribution.csv",distribution); _csv(OUTPUT/"oracle_conditional_routing_upper_bound.csv",oracle); _csv(OUTPUT/"causal_trigger_metrics.csv",metrics); _csv(OUTPUT/"trigger_payload_utility.csv",metrics); _csv(OUTPUT/"trigger_generalization_audit.csv",list(metrics)+stability_rows); _csv(OUTPUT/"conditional_planner_behavior_alignment.csv",behavior); _csv(OUTPUT/"planner_utility_taskwp_relation.csv",task_rows); _csv(OUTPUT/"planner_utility_seed_disagreement.csv",seed_rows); _json(OUTPUT/"conditional_planner_action_gate.json",cpa_gate); _json(OUTPUT/"causal_planning_trigger_gate.json",trigger_gate_json); _json(OUTPUT/"round27_readiness_gate.json",readiness)
    figures=_figures(rows,grouped,metrics,oracle,behavior,task_rows,seed_rows)
    access={"round":"round_27","allowed_splits_decoded":sorted(ALLOWED),"development_test_rows_decoded":0,"fresh_new_rollout_count":0,"round20_development_test_reuse_count":0,"round24_fresh_access_count":0,"untouched_eval_access_count":0,"counterfactual_diagnostic_replay_count":len(rows)*4,"new_scenarios_created":0,"policy_checkpoint_inference_only":True,"future_trigger_feature_access_count":0,"bridge_used_as_trigger":False,"training_performed":False,"distillation_training_performed":False}; _json(ROUND/"logs"/"access_audit.json",access)
    run={"round":"round_27","training_performed":False,"conditional_planner_utility_audit":True,"causal_trigger_design_performed":True,"distillation_training_performed":False,"dataset_modified":False,"policy_training_performed":False,"predictor_training_performed":False,"formal_sample_counts":{"state_pool":len(rows),"train":len(train),"validation":len(rows)-len(train),"counterfactual_branches":len(rows)*4},"figure_file_count":len(figures),"conditional_planner_action_gate":cpa,"causal_planning_trigger_gate":trigger_gate,"round28_created":False,"frozen_state":frozen}; _json(OUTPUT/"run_manifest.json",run)
    (ROUND/"logs"/"implementation_log.md").write_text("# Round-27 implementation log\n\nRead only frozen Round-20 train/validation t=0 scenario snapshots, frozen planners, frozen policy checkpoints, and prior locked diagnostic outputs. Counterfactual environment replays use only existing registered scenarios and are not fresh rollouts. No optimizer, backward pass, neural/classifier/policy/distillation training, checkpoint write, dataset modification, development-test decode, fresh outcome read, or untouched evaluation occurred.\n",encoding="utf-8")
    report=["# Round-27 Completion Report", "", f"1. Planner-useful ratio: {useful_ratio:.6f}.", f"2. Neutral ratio: {next(r for r in distribution if r['split']=='all')['neutral_ratio']:.6f}.", f"3. Harmful ratio: {next(r for r in distribution if r['split']=='all')['harmful_ratio']:.6f}.", f"4. Mean planner utility: {next(r for r in distribution if r['split']=='all')['mean_utility_payload']:.6f} bits.", f"5. Median utility: {next(r for r in distribution if r['split']=='all')['median_utility_payload']:.6f} bits.", f"6. Conflict utility is recorded in planner_utility_by_group.csv.", f"7. Duplicate-intent utility is recorded in planner_utility_by_group.csv.", f"8. Assignment-conflict utility is recorded in planner_utility_by_group.csv.", f"9. High-queue utility is recorded in planner_utility_by_group.csv.", f"10. K3/K5 utility is recorded in planner_utility_by_group.csv.", f"11. Eta=.25/.5 utility is recorded in planner_utility_by_group.csv.", f"12. Oracle conditional mean payload gain: {oracle_validation['oracle_mean_payload_gain_bits']:.6f} bits on validation.", f"13. Oracle validation CI: [{oracle_validation['oracle_ci95_low_bits']:.6f}, {oracle_validation['oracle_ci95_high_bits']:.6f}].", f"14. Trigger A validation metrics: {next(r for r in metrics if r['split']=='validation' and r['trigger']=='A_interference_conflict')}.", f"15. Trigger B validation metrics: {next(r for r in metrics if r['split']=='validation' and r['trigger']=='B_duplicate_intent')}.", f"16. Trigger C validation metrics: {next(r for r in metrics if r['split']=='validation' and r['trigger']=='C_queue_interference')}.", f"17. Trigger D validation metrics: {next(r for r in metrics if r['split']=='validation' and r['trigger']=='D_assignment_conflict')}.", f"18. Train-selected causal trigger: {selected_name}.", f"19. Validation positive-effect stability: {str(stable).lower()}.", f"20. Selected-trigger harmful rate: {validation_metric['harmful_action_rate_active']:.6f}.", "21. K/eta stability is recorded by group and gate criteria.", "22. Planner utility / Task-WP relation is in planner_utility_taskwp_relation.csv.", "23. Planner utility / seed disagreement is in planner_utility_seed_disagreement.csv.", f"24. CPA Gate: {cpa}.", f"25. Trigger Gate: {trigger_gate}.", f"26. conditional_planner_action_distillation_ready: {str(cpa=='CPA-A').lower()}.", f"27. Round-27 recommendation: {ready}.", "28. Training recommendation only; no training was performed.", "29. Fresh new rollout count: 0.", "30. Untouched-evaluation access count: 0.", "31. Round-28 created: false.", "32. Direct utility uses real frozen communication transitions, not action distance."]
    (ROUND/"report"/"round27_completion_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    return {"cpa":cpa_gate,"trigger":trigger_gate_json,"readiness":readiness}


def main() -> None:
    result=run_round27(); print(json.dumps({"cpa":result["cpa"]["gate"],"trigger":result["trigger"]["gate"],"readiness":result["readiness"]["recommendation"]},sort_keys=True))


if __name__ == "__main__": main()
