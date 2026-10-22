"""Frozen-checkpoint evaluation, seed-aware statistics, and Round-30 gates."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from .baselines import FrozenStateOnlyBC, IndependentMyopic, JointPlanner, RandomValid
from .evaluation import ppo_action, rollout
from .models import DeepSetsActorCritic
from .ppo import PPOAgent, load_agent
from .runner import CONFIG_PATH, FORMAL, OUTPUT, SEEDS, formal_config, sha256, write_csv, write_json
from .scenarios import PPOScenario, fresh_scenarios, validation_scenarios


PRIMARY = "cumulative_payload_bits"
COMPARATORS = ("random_valid", "independent_myopic", "state_only_bc_transformer", "joint_h1", "joint_h5", "joint_h10")
BOOTSTRAP_REPLICATES = 2000
_FROZEN_BC: FrozenStateOnlyBC | None = None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, Any]]:
    import csv
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_rows(rows: Iterable[Mapping[str, Any]]) -> bool:
    for row in rows:
        for value in row.values():
            if isinstance(value, (int, float, np.integer, np.floating)) and not np.isfinite(value):
                return False
    return True


def ppo_agents() -> dict[int, PPOAgent]:
    manifest = read_json(OUTPUT / "checkpoint_manifest.json")
    if manifest.get("fresh_access_count") != 0 or manifest.get("seeds") != list(SEEDS):
        raise RuntimeError("Round-30 checkpoints are not frozen before fresh access")
    config, agents = formal_config(), {}
    for item in manifest["checkpoints"]:
        path = Path(item["path"])
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise RuntimeError("Round-30 checkpoint hash mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload["seed"]) != int(item["seed"]) or int(payload["environment_steps"]) != int(item["step"]) or int(item["step"]) not in range(30720, 307201, 30720):
            raise RuntimeError("Round-30 selected checkpoint is outside the frozen validation schedule")
        agents[int(item["seed"])] = load_agent(payload, config, DeepSetsActorCritic())
    if tuple(sorted(agents)) != SEEDS:
        raise RuntimeError("Round-30 requires all five registered PPO checkpoints")
    return agents


def _baselines(scenario: PPOScenario) -> list[Any]:
    # Only the frozen BC comparator uses a Transformer, and it is not PPO input or training.
    global _FROZEN_BC
    if _FROZEN_BC is None: _FROZEN_BC = FrozenStateOnlyBC()
    return [RandomValid(scenario.seed * 101 + 17), IndependentMyopic(), JointPlanner(1), JointPlanner(5), JointPlanner(10), _FROZEN_BC]


def evaluate_population(population: str, scenarios: tuple[PPOScenario, ...], agents: Mapping[int, PPOAgent]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []; trajectories: list[dict[str, Any]] = []
    for scenario in scenarios:
        for baseline in _baselines(scenario):
            baseline.reset()
            metric, steps = rollout(scenario, population=population, policy_id=baseline.policy_id, model_seed=-1, action_fn=baseline.action, matching_mode=baseline.matching_mode)
            metrics.append(metric); trajectories.extend(steps)
        for seed, agent in sorted(agents.items()):
            metric, steps = rollout(scenario, population=population, policy_id="deepsets_ppo", model_seed=seed, action_fn=ppo_action(agent), matching_mode="legacy")
            metrics.append(metric); trajectories.extend(steps)
    if not finite_rows(metrics) or not finite_rows(trajectories):
        raise RuntimeError(f"Round-30 {population} evaluation produced NaN/Inf")
    expected = len(scenarios) * (len(COMPARATORS) + len(SEEDS))
    if len(metrics) != expected:
        raise RuntimeError("Round-30 evaluation is missing a registered policy/scenario result")
    return metrics, trajectories


def _load_or_evaluate_validation(agents: Mapping[int, PPOAgent]) -> list[dict[str, Any]]:
    all_path, trajectory_path = OUTPUT / "ppo_validation_all_policy_metrics.csv", OUTPUT / "ppo_validation_trajectory_steps.csv"
    if all_path.is_file() and trajectory_path.is_file(): return read_csv(all_path)
    metrics, trajectories = evaluate_population("validation", validation_scenarios(), agents)
    write_csv(all_path, metrics); write_csv(trajectory_path, trajectories)
    write_csv(OUTPUT / "ppo_validation_metrics.csv", [row for row in metrics if row["policy_id"] == "deepsets_ppo"])
    return metrics


def freeze_before_fresh(validation: list[Mapping[str, Any]]) -> dict[str, Any]:
    path = OUTPUT / "round30_pre_fresh_freeze.json"
    if path.is_file(): return read_json(path)
    required = (OUTPUT / "checkpoint_manifest.json", OUTPUT / "round30_protocol_freeze.json", OUTPUT / "ppo_validation_all_policy_metrics.csv")
    if not all(item.is_file() for item in required): raise RuntimeError("Round-30 must freeze checkpoints and validation before fresh")
    if (OUTPUT / "ppo_fresh_metrics.csv").exists() or (OUTPUT / "fresh_access_started.json").exists():
        raise RuntimeError("Round-30 fresh outcomes were already accessed")
    if len([row for row in validation if row["policy_id"] == "deepsets_ppo"]) != len(SEEDS) * len(validation_scenarios()):
        raise RuntimeError("Round-30 validation has not evaluated all five PPO seeds")
    payload = {
        "round": "round_30", "status": "frozen_before_single_fresh_access", "primary_metric": PRIMARY,
        "five_checkpoints_frozen": True, "all_baselines_frozen": list(COMPARATORS),
        "statistics": {"scenario_bootstrap": BOOTSTRAP_REPLICATES, "seed_cluster_bootstrap": BOOTSTRAP_REPLICATES, "hierarchical_seed_to_scenario_bootstrap": BOOTSTRAP_REPLICATES, "positive_seed_fraction": True, "median_seed_result": True, "leave_one_seed_out": True},
        "subgroups": {"num_buoys": [3, 5], "eta": [0.25, 0.5], "conflict_state": "frozen current-state interference_conflict outcome only; never deployment input"},
        "gates": {"learnability": "PPO-LA/B/C", "bc": "PBC-A/B/C", "LA_requires_validation_and_fresh_hierarchical_CI_lower_gt_zero": True},
        "fresh_access_count_before": 0, "fresh_outcomes_accessed": False,
        "hashes": {"formal_config": sha256(CONFIG_PATH), "protocol_freeze": sha256(OUTPUT / "round30_protocol_freeze.json"), "checkpoint_manifest": sha256(OUTPUT / "checkpoint_manifest.json"), "validation_all_policy_metrics": sha256(OUTPUT / "ppo_validation_all_policy_metrics.csv")},
    }
    write_json(path, payload); return payload


def _load_or_evaluate_fresh(agents: Mapping[int, PPOAgent]) -> list[dict[str, Any]]:
    all_path, trajectory_path = OUTPUT / "ppo_fresh_all_policy_metrics.csv", OUTPUT / "ppo_fresh_trajectory_steps.csv"
    if all_path.is_file() and trajectory_path.is_file(): return read_csv(all_path)
    freeze = OUTPUT / "round30_pre_fresh_freeze.json"
    if not freeze.is_file() or read_json(freeze).get("fresh_outcomes_accessed"):
        raise RuntimeError("Round-30 fresh evaluation requires the one-time pre-fresh freeze")
    started = OUTPUT / "fresh_access_started.json"
    if started.exists(): raise RuntimeError("Round-30 fresh access was interrupted; do not rerun it")
    write_json(started, {"round": "round_30", "fresh_access_count": 1, "reason": "single frozen post-training confirmation", "pre_fresh_freeze_sha256": sha256(freeze)})
    metrics, trajectories = evaluate_population("fresh", fresh_scenarios(), agents)
    write_csv(all_path, metrics); write_csv(trajectory_path, trajectories)
    write_csv(OUTPUT / "ppo_fresh_metrics.csv", [row for row in metrics if row["policy_id"] == "deepsets_ppo"])
    return metrics


def comparisons(metrics: Iterable[Mapping[str, Any]], population: str) -> list[dict[str, Any]]:
    rows = [row for row in metrics if row["population"] == population]
    index = {(str(row["scenario_id"]), str(row["policy_id"]), int(row["model_seed"])): row for row in rows}
    output: list[dict[str, Any]] = []
    for scenario in sorted({str(row["scenario_id"]) for row in rows}):
        for seed in SEEDS:
            ppo = index[(scenario, "deepsets_ppo", seed)]
            for comparator in COMPARATORS:
                baseline = index[(scenario, comparator, -1)]
                output.append({"population": population, "scenario_id": scenario, "scenario_seed": ppo["scenario_seed"], "model_seed": seed, "comparator": comparator, "num_buoys": ppo["num_buoys"], "eta": ppo["eta"], "queue_pressure": ppo["queue_pressure"], "geometry": ppo["geometry"], "initial_uav_configuration": ppo["initial_uav_configuration"], "interference_conflict": ppo["interference_conflict"], "ppo_cumulative_payload_bits": ppo["cumulative_payload_bits"], "baseline_cumulative_payload_bits": baseline["cumulative_payload_bits"], **{f"delta_{key}": float(ppo[key]) - float(baseline[key]) for key in ("cumulative_payload_bits", "completion", "residual_queue_bits", "episode_return", "jain_fairness", "team_path_length_m", "energy", "projection_rate", "mean_projection_magnitude", "minimum_uav_distance_m")}})
    return output


def _bootstrap(deltas: list[Mapping[str, Any]], population: str, comparator: str) -> dict[str, Any]:
    rows = [row for row in deltas if row["population"] == population and row["comparator"] == comparator]
    by_seed = {seed: [float(row["delta_cumulative_payload_bits"]) for row in rows if int(row["model_seed"]) == seed] for seed in SEEDS}
    if any(not values for values in by_seed.values()): raise RuntimeError("seed-aware comparison is incomplete")
    rng = np.random.default_rng(93000 + sum(ord(ch) for ch in population + comparator)); seed_means = np.asarray([np.mean(by_seed[seed]) for seed in SEEDS])
    pooled = np.asarray([value for values in by_seed.values() for value in values])
    scenario_matrix = np.asarray([by_seed[seed] for seed in SEEDS], dtype=np.float64)
    hierarchical = np.empty(BOOTSTRAP_REPLICATES); scenario_boot = np.empty(BOOTSTRAP_REPLICATES); seed_boot = np.empty(BOOTSTRAP_REPLICATES)
    scenarios = scenario_matrix.shape[1]
    for index in range(BOOTSTRAP_REPLICATES):
        selected_seeds = rng.integers(0, len(SEEDS), size=len(SEEDS)); selected_scenarios = rng.integers(0, scenarios, size=(len(SEEDS), scenarios))
        hierarchical[index] = np.mean([scenario_matrix[seed, selected_scenarios[row]].mean() for row, seed in enumerate(selected_seeds)])
        scenario_boot[index] = pooled[rng.integers(0, len(pooled), size=len(pooled))].mean()
        seed_boot[index] = seed_means[rng.integers(0, len(SEEDS), size=len(SEEDS))].mean()
    leave_one = {f"leave_out_seed_{seed}": float(np.mean([value for other, value in zip(SEEDS, seed_means) if other != seed])) for seed in SEEDS}
    return {"population": population, "comparator": comparator, "metric": PRIMARY, "paired_records": len(rows), "mean_delta_bits": float(pooled.mean()), "median_seed_delta_bits": float(np.median(seed_means)), "positive_seed_fraction": float(np.mean(seed_means > 0.0)), "seed_deltas_bits": seed_means.tolist(), "scenario_bootstrap_ci_low": float(np.quantile(scenario_boot, .025)), "scenario_bootstrap_ci_high": float(np.quantile(scenario_boot, .975)), "seed_cluster_bootstrap_ci_low": float(np.quantile(seed_boot, .025)), "seed_cluster_bootstrap_ci_high": float(np.quantile(seed_boot, .975)), "hierarchical_ci_low": float(np.quantile(hierarchical, .025)), "hierarchical_ci_high": float(np.quantile(hierarchical, .975)), "bootstrap_replicates": BOOTSTRAP_REPLICATES, **leave_one}


def uncertainty(deltas: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_bootstrap(deltas, population, comparator) for population in ("validation", "fresh") for comparator in COMPARATORS]


def _mean(rows: Iterable[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(np.mean(values)) if values else float("nan")


def robustness(deltas: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in deltas if row["comparator"] == "independent_myopic"]
    result: list[dict[str, Any]] = []
    for population in ("validation", "fresh"):
        selected = [row for row in rows if row["population"] == population]
        groups = [("overall", "all", selected), ("num_buoys", "K3", [row for row in selected if int(row["num_buoys"]) == 3]), ("num_buoys", "K5", [row for row in selected if int(row["num_buoys"]) == 5]), ("eta", "eta_025", [row for row in selected if float(row["eta"]) == .25]), ("eta", "eta_05", [row for row in selected if float(row["eta"]) == .5])]
        for axis, group, subset in groups:
            seed_means = [_mean([row for row in subset if int(row["model_seed"]) == seed], "delta_cumulative_payload_bits") for seed in SEEDS]
            mean_delta, mean_baseline = _mean(subset, "delta_cumulative_payload_bits"), _mean(subset, "baseline_cumulative_payload_bits")
            result.append({"population": population, "comparison": "ppo_minus_independent", "axis": axis, "group": group, "paired_records": len(subset), "mean_delta_payload_bits": mean_delta, "mean_independent_payload_bits": mean_baseline, "positive_seed_fraction": float(np.mean(np.asarray(seed_means) > 0.0)), "median_seed_delta_bits": float(np.median(seed_means)), "catastrophic_reversal": bool(mean_delta < -0.20 * max(abs(mean_baseline), 1.0))})
    return result


def conflict_analysis(metrics: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = list(metrics); result: list[dict[str, Any]] = []
    for population in ("validation", "fresh"):
        for comparator in ("state_only_bc_transformer", "joint_h1"):
            for label, conflict in (("conflict", True), ("non_conflict", False)):
                ppo = [row for row in rows if row["population"] == population and row["policy_id"] == "deepsets_ppo" and bool(row["interference_conflict"]) == conflict]
                base = [row for row in rows if row["population"] == population and row["policy_id"] == comparator and bool(row["interference_conflict"]) == conflict]
                if ppo and base:
                    ppo_mean, baseline_mean, delta = _mean(ppo, PRIMARY), _mean(base, PRIMARY), _mean(ppo, PRIMARY) - _mean(base, PRIMARY)
                else:
                    ppo_mean = baseline_mean = delta = "not_applicable_no_registered_episode"
                result.append({"population": population, "comparator": comparator, "current_state_conflict_group": label, "ppo_episode_records": len(ppo), "baseline_episode_records": len(base), "ppo_mean_payload_bits": ppo_mean, "baseline_mean_payload_bits": baseline_mean, "ppo_minus_baseline_bits": delta, "definition": "episode contains frozen environment interference_conflict; analysis only, never PPO input"})
    return result


def seed_stability(metrics: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    training = {int(row["seed"]): row for row in read_csv(OUTPUT / "ppo_training_summary.csv")}
    rows = list(metrics); result = []
    for seed in SEEDS:
        train = training[seed]
        val = [row for row in rows if row["population"] == "validation" and row["policy_id"] == "deepsets_ppo" and int(row["model_seed"]) == seed]
        fresh = [row for row in rows if row["population"] == "fresh" and row["policy_id"] == "deepsets_ppo" and int(row["model_seed"]) == seed]
        result.append({"seed": seed, "formal_environment_steps": int(train["formal_environment_steps"]), "early_training_payload_bits": float(train["early_training_payload_bits"]), "late_training_payload_bits": float(train["late_training_payload_bits"]), "late_gt_early": str(train["late_gt_early"]).lower() == "true", "best_validation_payload_bits": float(train["best_validation_payload_bits"]), "validation_payload_bits": _mean(val, PRIMARY), "fresh_payload_bits": _mean(fresh, PRIMARY), "validation_completion": _mean(val, "completion"), "fresh_completion": _mean(fresh, "completion"), "mean_entropy": float(train["mean_entropy"]), "mean_action_std": float(train["mean_action_std"]), "mean_kl": float(train["mean_kl"]), "mean_explained_variance": float(train["mean_explained_variance"]), "mean_projection_rate": float(train["mean_projection_rate"])})
    return result


def gates(uncertainty_rows: list[Mapping[str, Any]], stability: list[Mapping[str, Any]], all_metrics: list[Mapping[str, Any]], robust: list[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    lookup = {(row["population"], row["comparator"]): row for row in uncertainty_rows}
    vi, fi = lookup[("validation", "independent_myopic")], lookup[("fresh", "independent_myopic")]
    vb, fb = lookup[("validation", "state_only_bc_transformer")], lookup[("fresh", "state_only_bc_transformer")]
    late_positive = sum(bool(row["late_gt_early"]) for row in stability)
    validation_positive = sum(float(row["validation_payload_bits"]) > _mean([row for row in all_metrics if row["population"] == "validation" and row["policy_id"] == "independent_myopic"], PRIMARY) for row in stability)
    # Differential entropy of a narrow continuous Gaussian may legitimately be negative;
    # health means finite diagnostics and non-collapsed movement variance, not entropy > 0.
    health = all(np.isfinite(float(row[key])) for row in stability for key in ("mean_entropy", "mean_action_std", "mean_kl", "mean_explained_variance")) and all(float(row["mean_action_std"]) >= 0.05 for row in stability)
    ppo_projection = _mean([row for row in all_metrics if row["policy_id"] == "deepsets_ppo"], "projection_rate")
    independent_projection = _mean([row for row in all_metrics if row["policy_id"] == "independent_myopic"], "projection_rate")
    safety = ppo_projection <= independent_projection + .05
    robust_ok = not any(bool(row["catastrophic_reversal"]) for row in robust)
    learnable = late_positive >= 4 and validation_positive >= 4 and float(vi["hierarchical_ci_low"]) > 0.0 and float(fi["hierarchical_ci_low"]) > 0.0 and health and safety and robust_ok
    # LB needs an at-least-plausible PPO-versus-Independent effect.  A learning-curve
    # fluctuation alone cannot override paired validation and fresh intervals that
    # are both entirely below zero.
    conditional = (late_positive >= 3 or float(vi["mean_delta_bits"]) > 0.0) and health and (float(vi["hierarchical_ci_high"]) >= 0.0 or float(fi["hierarchical_ci_high"]) >= 0.0)
    label = "PPO-LA" if learnable else "PPO-LB" if conditional else "PPO-LC"
    ppo_gate = {"round": "round_30", "gate": label, "primary_metric": PRIMARY, "late_gt_early_seed_count": late_positive, "validation_ppo_gt_independent_seed_count": validation_positive, "validation_hierarchical_ci": [vi["hierarchical_ci_low"], vi["hierarchical_ci_high"]], "fresh_hierarchical_ci": [fi["hierarchical_ci_low"], fi["hierarchical_ci_high"]], "finite_health": health, "no_systematic_safety_decline": safety, "ppo_projection_rate": ppo_projection, "independent_projection_rate": independent_projection, "no_catastrophic_k_eta_reversal": robust_ok, "fresh_confirmation_required_and_used": True, "decision_rule": "LA requires all listed criteria including both validation and fresh CI lower bounds > 0"}
    bc_a = float(vb["hierarchical_ci_low"]) > 0.0 and float(fb["hierarchical_ci_low"]) > 0.0
    bc_c = float(vb["hierarchical_ci_high"]) < 0.0 or float(fb["hierarchical_ci_high"]) < 0.0
    bc_label = "PBC-A" if bc_a else "PBC-C" if bc_c else "PBC-B"
    bc_gate = {"round": "round_30", "gate": bc_label, "primary_metric": PRIMARY, "validation_hierarchical_ci": [vb["hierarchical_ci_low"], vb["hierarchical_ci_high"]], "fresh_hierarchical_ci": [fb["hierarchical_ci_low"], fb["hierarchical_ci_high"]], "positive_seed_fraction_validation": vb["positive_seed_fraction"], "positive_seed_fraction_fresh": fb["positive_seed_fraction"], "decision_rule": "PBC-A requires both lower bounds > 0; PBC-C when either upper bound < 0; otherwise PBC-B"}
    recommendation = "direct_rl_baseline_supported; future work may assess PPO auxiliaries only in a separately authorized round" if label == "PPO-LA" and bc_label == "PBC-A" else "continue_rl_baseline_diagnosis_or_optimization in a separately authorized round" if label in {"PPO-LA", "PPO-LB"} else "PPO_stability_diagnosis before adding architecture complexity in a separately authorized round"
    readiness = {"round": "round_30", "status": "completed_stop", "ppo_learnability_gate": label, "ppo_vs_bc_gate": bc_label, "conclusion": "Current frozen environment is demonstrated directly RL-learnable" if label == "PPO-LA" else "Current evidence does not establish direct RL learnability", "recommended_next_step": recommendation, "round31_created": False, "no_additional_training_or_fresh_access_authorized": True}
    return ppo_gate, bc_gate, readiness


def run_analysis() -> dict[str, Any]:
    """Evaluate frozen policies. Fresh scenarios are entered exactly once after all freezes."""
    agents = ppo_agents()
    validation = _load_or_evaluate_validation(agents)
    freeze_before_fresh(validation)
    fresh = _load_or_evaluate_fresh(agents)
    all_metrics = validation + fresh
    write_csv(OUTPUT / "ppo_fresh_seed_summary.csv", [{"seed": seed, "fresh_payload_bits": _mean([row for row in fresh if row["policy_id"] == "deepsets_ppo" and int(row["model_seed"]) == seed], PRIMARY), "fresh_completion": _mean([row for row in fresh if row["policy_id"] == "deepsets_ppo" and int(row["model_seed"]) == seed], "completion"), "fresh_residual_queue_bits": _mean([row for row in fresh if row["policy_id"] == "deepsets_ppo" and int(row["model_seed"]) == seed], "residual_queue_bits")} for seed in SEEDS])
    delta_rows, uncertainty_rows = comparisons(validation, "validation") + comparisons(fresh, "fresh"), None
    write_csv(OUTPUT / "ppo_baseline_comparison.csv", delta_rows)
    uncertainty_rows = uncertainty(delta_rows); write_csv(OUTPUT / "ppo_uncertainty_audit.csv", uncertainty_rows)
    robust = robustness(delta_rows); write_csv(OUTPUT / "ppo_K_eta_robustness.csv", robust)
    conflicts = conflict_analysis(all_metrics); write_csv(OUTPUT / "ppo_conflict_analysis.csv", conflicts)
    stability = seed_stability(all_metrics); write_csv(OUTPUT / "ppo_seed_stability.csv", stability)
    ppo_gate, bc_gate, readiness = gates(uncertainty_rows, stability, all_metrics, robust)
    write_json(OUTPUT / "ppo_learning_gate.json", ppo_gate); write_json(OUTPUT / "ppo_vs_bc_gate.json", bc_gate); write_json(OUTPUT / "round30_readiness_gate.json", readiness)
    access = {"round": "round_30", "round20_development_test_reuse_count": 0, "round22_fresh_access_count": 0, "round24_fresh_access_count": 0, "round29_confirmation_reuse_count": 0, "untouched_eval_access_count": 0, "untouched_eval_accessed": False, "round30_validation_outcome_access_count": 1, "round30_fresh_access_count": 1, "fresh_access_count": 1, "fresh_access_policy": "one frozen post-training confirmation; no rerun", "fresh_pre_freeze_sha256": sha256(OUTPUT / "round30_pre_fresh_freeze.json")}
    write_json(OUTPUT / "access_audit.json", access)
    run = read_json(OUTPUT / "run_manifest.json"); run.update({"status": "analysis_complete_fresh_access_once", "fresh_access_count": 1, "validation_outcome_access_count": 1, "untouched_eval_access_count": 0, "ppo_learnability_gate": ppo_gate["gate"], "ppo_vs_bc_gate": bc_gate["gate"], "round31_created": False}); write_json(OUTPUT / "run_manifest.json", run)
    return {"ppo_gate": ppo_gate, "bc_gate": bc_gate, "readiness": readiness, "access": access}


__all__ = ["PRIMARY", "run_analysis", "freeze_before_fresh", "ppo_agents"]
