"""Round-23 frozen-policy diagnostics; deliberately contains no training path."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from wt.governance_state_sync import verify_round22_governance_state_sync_amendment
from wt.round19.multi_env import SharedSpectrumMultiUAVEnv
from wt.round20.dataset import build_graph_state
from wt.round20.scenarios import development_scenarios
from wt.round22.closed_loop import _current_phi, _model_inputs, _scenario_hash
from wt.round22.data import FrozenDeepSetsPredictor, R22_OUTPUT, VALIDATION, load_policy_sequences
from wt.round22.models import PREDICTED_MLP, PREDICTED_TRANSFORMER, STATE_ONLY_TRANSFORMER, rollout_action
from wt.round22.training import TrainingConfig, load_checkpoint, predict, raw_rmse


ROOT = Path(__file__).resolve().parents[3]
ROUND = ROOT / "round_23"
OUTPUT = ROUND / "outputs" / "round23_seed_stability_audit_20260829_1436"
FIGURES = ROUND / "figures" / "round23_seed_stability_audit_20260829_1436"
CONFIG_PATH = ROUND / "configs" / "round23_audit_config.json"
R22_LOCK = ROOT / "round_22" / "ROUND_LOCK.json"
R22_AMENDMENT = ROOT / "round_22" / "ROUND_LOCK_AMENDMENT.json"
EXPECTED_R22_LOCK = "1D464A696ED8DE11E7AEDB14479F55ABB4F9F3E2621837E89CD854CD850190E4"
EXPECTED_PREDICTOR = "645F235032DE9639A6D42318A41CC8F0E2175E86E38545DA98E0BE5CAEB80221"
SEEDS = (0, 1, 2)
HEAD_SLICES = {"service": slice(0, 5), "progress": slice(5, 10), "return": slice(10, 11)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Mapping): return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(item) for item in value]
    return value


def _json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_plain(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"required Round-23 output is empty: {path.name}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(_plain(value), sort_keys=True) if isinstance(value, (dict, list, tuple, np.ndarray)) else value for key, value in row.items()})


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _f(value: Any, default: float = 0.0) -> float:
    return default if value in (None, "") else float(value)


def _b(value: Any) -> bool:
    return str(value).lower() == "true"


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([_f(row[field]) for row in rows])) if rows else float("nan")


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort"); result = np.empty(len(values), dtype=np.float64); index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]: end += 1
        result[order[index:end]] = 0.5 * (index + end - 1) + 1.0; index = end
    return result


def _corr(left: Sequence[float], right: Sequence[float], *, spearman: bool = False) -> float:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if len(a) < 2 or np.std(a) <= 1.0e-12 or np.std(b) <= 1.0e-12: return float("nan")
    if spearman: a, b = _rank(a), _rank(b)
    return float(np.corrcoef(a, b)[0, 1])


def _bootstrap(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    if len(values) == 0: return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def _scenario_bootstrap(rows: Sequence[Mapping[str, Any]], draws: int, seed: int) -> tuple[float, float]:
    by_scenario: dict[str, list[float]] = defaultdict(list)
    for row in rows: by_scenario[str(row["scenario_id"])].append(_f(row["delta_cumulative_payload_bits"]))
    return _bootstrap(np.asarray([np.mean(by_scenario[key]) for key in sorted(by_scenario)], dtype=np.float64), draws, seed)


def _formal_runs() -> list[dict[str, Any]]:
    payload = json.loads((R22_OUTPUT / "round22_formal_training_manifest.json").read_text(encoding="utf-8"))
    if payload.get("formal_training_completed") != "12/12" or len(payload.get("runs", [])) != 12:
        raise RuntimeError("Round-23 requires the locked 12/12 Round-22 formal matrix")
    return list(payload["runs"])


def verify_frozen_state() -> dict[str, Any]:
    """Verify all Round-22 evidence required before any Round-23 inference."""
    if sha256(R22_LOCK) != EXPECTED_R22_LOCK:
        raise RuntimeError("Round-22 original lock changed")
    amendment = json.loads(R22_AMENDMENT.read_text(encoding="utf-8"))
    if not amendment.get("resume_round23_creation_allowed"):
        raise RuntimeError("Round-23 creation is not authorized by the governance amendment")
    governance = verify_round22_governance_state_sync_amendment()
    runs = _formal_runs(); selected = json.loads((R22_OUTPUT / "selected_deepsets_task_waypoint_predictor.json").read_text(encoding="utf-8"))
    predictor_hash = sha256(ROOT / selected["checkpoint"])
    failures = [row["run_id"] for row in runs if sha256(ROOT / row["checkpoint"]) != row["checkpoint_sha256"] or sha256(ROOT / row["last_checkpoint"]) != row["last_checkpoint_sha256"]]
    access = json.loads((R22_OUTPUT / "access_audit.json").read_text(encoding="utf-8"))
    if predictor_hash != EXPECTED_PREDICTOR or failures or int(access["fresh_dev_access_count"]) != 1 or int(access["round20_development_test_reuse_count"]) != 0 or int(access["untouched_eval_model_access_count"]) != 0 or int(access["untouched_eval_policy_rollout_count"]) != 0:
        raise RuntimeError("Round-22 frozen access/checkpoint facts are inconsistent")
    return {"round22_original_lock_sha256": EXPECTED_R22_LOCK, "round22_governance_amendment_verified": bool(governance["passed"]), "predictor_sha256": predictor_hash, "predictor_hash_unchanged": True, "formal_best_last_checkpoint_failures": failures, "formal_policy_training": "12/12", "fresh_dev_access_count": 1, "round20_development_test_reuse_count": 0, "untouched_eval_model_access_count": 0, "untouched_eval_policy_rollout_count": 0, "fresh_dev_new_rollout_count": 0, "untouched_eval_access_count": 0, "training_performed": False, "policy_training_performed": False}


def _fresh_matrix(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    by_scenario_seed = {(str(row["scenario_id"]), int(row["model_seed"])): _f(row["delta_cumulative_payload_bits"]) for row in rows}
    scenarios = sorted({scenario for scenario, _ in by_scenario_seed})
    matrix: list[dict[str, Any]] = []
    by_seed: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        by_seed[seed] = np.asarray([by_scenario_seed[(scenario, seed)] for scenario in scenarios], dtype=np.float64)
    first = {str(row["scenario_id"]): row for row in rows if int(row["model_seed"]) == 0}
    for index, scenario in enumerate(scenarios):
        row = first[scenario]
        matrix.append({"scenario_id": scenario, "num_buoys": int(row["num_buoys"]), "eta": _f(row["eta"]), "interference_conflict": _b(row["interference_conflict"]), **{f"seed_{seed}_delta_payload_bits": float(by_seed[seed][index]) for seed in SEEDS}})
    return matrix, by_seed


def _bootstrap_audit(fresh_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matrix, by_seed = _fresh_matrix(fresh_rows)
    all_values = np.concatenate([by_seed[seed] for seed in SEEDS])
    original_low, original_high = _scenario_bootstrap(fresh_rows, 2000, 22022)
    seed_means = np.asarray([by_seed[seed].mean() for seed in SEEDS], dtype=np.float64)
    cluster_low, cluster_high = _bootstrap(seed_means, int(config["bootstrap_draws"]), int(config["bootstrap_seed"]))
    rng = np.random.default_rng(int(config["bootstrap_seed"]) + 1); draws = int(config["bootstrap_draws"]); hierarchical = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        selected = rng.integers(0, len(SEEDS), len(SEEDS)); values: list[float] = []
        for which in selected:
            sample = by_seed[SEEDS[int(which)]]
            values.extend(sample[rng.integers(0, len(sample), len(sample))].tolist())
        hierarchical[index] = float(np.mean(values))
    result = [
        {"method": "original_scenario_bootstrap", "excluded_seed": "", "mean_delta_payload_bits": float(all_values.mean()), "ci95_lower": original_low, "ci95_upper": original_high, "bootstrap_draws": 2000, "bootstrap_seed": 22022, "unit": "scenario mean across three model seeds", "round22_primary_metric_unchanged": True},
        {"method": "seed_cluster_bootstrap", "excluded_seed": "", "mean_delta_payload_bits": float(seed_means.mean()), "ci95_lower": cluster_low, "ci95_upper": cluster_high, "bootstrap_draws": draws, "bootstrap_seed": int(config["bootstrap_seed"]), "unit": "model-seed cluster"},
        {"method": "hierarchical_seed_then_scenario_bootstrap", "excluded_seed": "", "mean_delta_payload_bits": float(all_values.mean()), "ci95_lower": float(np.quantile(hierarchical, .025)), "ci95_upper": float(np.quantile(hierarchical, .975)), "bootstrap_draws": draws, "bootstrap_seed": int(config["bootstrap_seed"]) + 1, "unit": "model seed then within-seed scenario"},
    ]
    for seed in SEEDS:
        retained = np.concatenate([by_seed[other] for other in SEEDS if other != seed])
        low, high = _bootstrap(retained, draws, int(config["bootstrap_seed"]) + 10 + seed)
        result.append({"method": "leave_one_seed_out", "excluded_seed": seed, "mean_delta_payload_bits": float(retained.mean()), "ci95_lower": low, "ci95_upper": high, "bootstrap_draws": draws, "bootstrap_seed": int(config["bootstrap_seed"]) + 10 + seed, "unit": "all scenarios from retained model seeds"})
    positive = int(max(SEEDS, key=lambda seed: float(by_seed[seed].mean())))
    return result, {"matrix": matrix, "by_seed": by_seed, "positive_seed": positive, "negative_seeds": [seed for seed in SEEDS if seed != positive]}


def _tail_audit(by_seed: Mapping[int, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed, values in by_seed.items():
        ordered = np.sort(values); positive = values[values > 0.0]; negative = -values[values < 0.0]
        rows.append({"seed": seed, "scenario_count": len(values), "mean_delta_payload_bits": float(values.mean()), "median_delta_payload_bits": float(np.median(values)), "trimmed_mean_delta_payload_bits": float(np.mean(ordered[3:-3])), "p10": float(np.quantile(values, .10)), "p25": float(np.quantile(values, .25)), "p50": float(np.quantile(values, .50)), "p75": float(np.quantile(values, .75)), "p90": float(np.quantile(values, .90)), "min": float(values.min()), "max": float(values.max()), "top5_positive_contribution_share": float(np.sort(positive)[-5:].sum() / positive.sum()) if len(positive) else 0.0, "bottom5_negative_contribution_share": float(np.sort(negative)[-5:].sum() / negative.sum()) if len(negative) else 0.0, "primary_metric": "ordinary_mean_delta_payload_bits", "trimmed_mean_diagnostic_only": True})
    return rows


def _win_loss_audit(by_seed: Mapping[int, np.ndarray], tolerance: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed, values in by_seed.items():
        win, tie, loss = values > tolerance, np.abs(values) <= tolerance, values < -tolerance
        rows.append({"seed": seed, "tie_tolerance_bits": tolerance, "win_count": int(win.sum()), "tie_count": int(tie.sum()), "loss_count": int(loss.sum()), "win_rate": float(win.mean()), "mean_positive_delta_bits": float(values[win].mean()) if win.any() else 0.0, "mean_negative_delta_bits": float(values[loss].mean()) if loss.any() else 0.0, "median_delta_bits": float(np.median(values))})
    return rows


def _fresh_agreement(by_seed: Mapping[int, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        a, b = by_seed[left], by_seed[right]
        rows.append({"seed_pair": f"{left}-{right}", "pearson_correlation": _corr(a, b), "spearman_correlation": _corr(a, b, spearman=True), "sign_agreement": float(np.mean(np.sign(a) == np.sign(b))), "scenario_rank_agreement": _corr(a, b, spearman=True), "scenario_count": len(a)})
    return rows


def _by_family_seed(rows: Sequence[Mapping[str, Any]], family: str, seed: int) -> list[Mapping[str, Any]]:
    return [row for row in rows if str(row["family"]) == family and int(row["model_seed"]) == seed]


def _seed_level_summary(runs: Sequence[Mapping[str, Any]], predictions: Mapping[tuple[str, int], np.ndarray], sequences: Any) -> tuple[list[dict[str, Any]], dict[int, float], dict[int, float]]:
    histories: dict[tuple[str, int], list[dict[str, str]]] = {}
    for run in runs:
        history_path = ROOT / str(run["checkpoint"]).replace("best.pt", "training_history.csv")
        histories[(str(run["family"]), int(run["seed"]))] = _rows(history_path)
    validation_metrics = _rows(R22_OUTPUT / "validation_closed_loop_metrics.csv")
    fresh_metrics = _rows(R22_OUTPUT / "fresh_dev_closed_loop_metrics.csv")
    validation_pairs = _rows(R22_OUTPUT / "validation_paired_comparison.csv")
    fresh_pairs = _rows(R22_OUTPUT / "fresh_dev_paired_comparison.csv")
    result: list[dict[str, Any]] = []; offline_rmse: dict[int, float] = {}; fresh_delta: dict[int, float] = {}
    for run in runs:
        if str(run["family"]) != PREDICTED_TRANSFORMER: continue
        seed = int(run["seed"]); history = histories[(PREDICTED_TRANSFORMER, seed)]; prediction = predictions[(PREDICTED_TRANSFORMER, seed)]
        action_magnitudes = np.linalg.norm(prediction, axis=-1)[sequences.uav_mask[:, -1]]
        parameters = torch.load(ROOT / str(run["checkpoint"]), map_location="cpu", weights_only=False)["state_dict"]
        parameter_norm = float(math.sqrt(sum(float(torch.sum(value.detach().float() ** 2)) for value in parameters.values())))
        val = _by_family_seed(validation_metrics, PREDICTED_TRANSFORMER, seed); fresh = _by_family_seed(fresh_metrics, PREDICTED_TRANSFORMER, seed)
        val_pair = [row for row in validation_pairs if int(row["model_seed"]) == seed]; fresh_pair = [row for row in fresh_pairs if int(row["model_seed"]) == seed]
        offline_rmse[seed] = float(run["best_validation_raw_joint_action_rmse"]); fresh_delta[seed] = _mean(fresh_pair, "delta_cumulative_payload_bits")
        result.append({"seed": seed, "checkpoint": run["checkpoint"], "checkpoint_sha256": run["checkpoint_sha256"], "best_epoch": int(run["best_epoch"]), "train_loss_at_best_epoch": _f(next(row["train_action_mse"] for row in history if int(row["epoch"]) == int(run["best_epoch"]))), "validation_action_rmse": offline_rmse[seed], "parameter_l2_norm": parameter_norm, "gradient_norm_mean": _mean(history, "gradient_norm_preclip"), "gradient_norm_max": max(_f(row["gradient_norm_preclip"]) for row in history), "action_magnitude_mean": float(action_magnitudes.mean()), "action_magnitude_p50": float(np.median(action_magnitudes)), "action_magnitude_p95": float(np.quantile(action_magnitudes, .95)), "validation_payload_bits": _mean(val, "cumulative_payload_bits"), "validation_completion": _mean(val, "completion"), "validation_residual_queue_bits": _mean(val, "residual_queue_bits"), "validation_return": _mean(val, "episode_return"), "validation_fairness": _mean(val, "jain_fairness"), "validation_energy": _mean(val, "movement_energy"), "validation_projection_rate": _mean(val, "safety_correction_rate"), "validation_safety_correction": _mean(val, "safety_correction_rate"), "validation_payload_delta_bits": _mean(val_pair, "delta_cumulative_payload_bits"), "fresh_payload_bits": _mean(fresh, "cumulative_payload_bits"), "fresh_completion": _mean(fresh, "completion"), "fresh_residual_queue_bits": _mean(fresh, "residual_queue_bits"), "fresh_return": _mean(fresh, "episode_return"), "fresh_fairness": _mean(fresh, "jain_fairness"), "fresh_energy": _mean(fresh, "movement_energy"), "fresh_projection_rate": _mean(fresh, "safety_correction_rate"), "fresh_safety_correction": _mean(fresh, "safety_correction_rate"), "fresh_payload_delta_bits": fresh_delta[seed]})
    return result, offline_rmse, fresh_delta


def _offline_alignment(offline_rmse: Mapping[int, float], validation_pairs: Sequence[Mapping[str, Any]], fresh_pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    val_means = {seed: _mean([row for row in validation_pairs if int(row["model_seed"]) == seed], "delta_cumulative_payload_bits") for seed in SEEDS}
    fresh_means = {seed: _mean([row for row in fresh_pairs if int(row["model_seed"]) == seed], "delta_cumulative_payload_bits") for seed in SEEDS}
    x = [offline_rmse[seed] for seed in SEEDS]
    rows = [
        {"analysis_level": "seed", "outcome": "validation_payload_delta_bits", "n": 3, "pearson": _corr(x, [val_means[seed] for seed in SEEDS]), "spearman": _corr(x, [val_means[seed] for seed in SEEDS], spearman=True), "interpretation": "descriptive_only_n_equals_3"},
        {"analysis_level": "seed", "outcome": "fresh_payload_delta_bits", "n": 3, "pearson": _corr(x, [fresh_means[seed] for seed in SEEDS]), "spearman": _corr(x, [fresh_means[seed] for seed in SEEDS], spearman=True), "interpretation": "descriptive_only_n_equals_3"},
    ]
    offline = _rows(R22_OUTPUT / "offline_validation_metrics.csv"); paired = _rows(R22_OUTPUT / "validation_paired_comparison.csv")
    pair_lookup = {(str(row["scenario_id"]), int(row["model_seed"])): _f(row["delta_cumulative_payload_bits"]) for row in paired}
    for seed in SEEDS:
        by_scenario: dict[str, list[float]] = defaultdict(list)
        for row in offline:
            if str(row["family"]) == PREDICTED_TRANSFORMER and int(row["seed"]) == seed:
                by_scenario[str(row["scenario_id"])].append(_f(row["raw_action_rmse"]))
        keys = sorted(set(by_scenario) & {scenario for scenario, row_seed in pair_lookup if row_seed == seed})
        error = [float(np.mean(by_scenario[key])) for key in keys]; delta = [pair_lookup[(key, seed)] for key in keys]
        rows.append({"analysis_level": "scenario", "outcome": "validation_payload_delta_bits", "seed": seed, "n": len(keys), "pearson": _corr(error, delta), "spearman": _corr(error, delta, spearman=True), "interpretation": "paired_validation_scenario_descriptive"})
    return rows


def _grouped_action_error() -> list[dict[str, Any]]:
    offline = _rows(R22_OUTPUT / "offline_validation_metrics.csv")
    # Queue tier remains the registered Round-20 scenario attribute; it is not
    # inferred from any policy outcome.
    queue_tier = {str(row.scenario_id): str(row.queue_tier) for row in development_scenarios()}
    source: list[dict[str, Any]] = []
    for row in offline:
        if str(row["family"]) != PREDICTED_TRANSFORMER: continue
        source.append({**row, "eta": _f(row["eta"]), "high_queue": queue_tier[str(row["scenario_id"])] == "high", "non_conflict": not _b(row["conflict"]), "non_bridge": not _b(row["bridge"])})
    conditions: list[tuple[str, Any]] = [("interference_conflict", lambda r: _b(r["conflict"])), ("non_conflict", lambda r: r["non_conflict"]), ("coordination_bridge", lambda r: _b(r["bridge"])), ("non_bridge", lambda r: r["non_bridge"]), ("K3", lambda r: int(r["num_buoys"]) == 3), ("K5", lambda r: int(r["num_buoys"]) == 5), ("eta_0.25", lambda r: abs(_f(r["eta"]) - .25) < 1e-12), ("eta_0.5", lambda r: abs(_f(r["eta"]) - .5) < 1e-12), ("eta_1", lambda r: abs(_f(r["eta"]) - 1.0) < 1e-12), ("high_queue", lambda r: r["high_queue"])]
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for name, selector in conditions:
            subset = [row for row in source if int(row["seed"]) == seed and selector(row)]
            rows.append({"seed": seed, "group": name, "sample_count": len(subset), "action_rmse": _mean(subset, "raw_action_rmse"), "cosine": _mean(subset, "cosine"), "angular_error_deg": _mean(subset, "angular_error_deg"), "magnitude_error": _mean(subset, "magnitude_error"), "frozen_definition": name not in {"high_queue"} or "round20_queue_tier"})
    return rows


def _phi_variant(phi: np.ndarray, condition: str) -> np.ndarray:
    result = phi.copy()
    if condition == "normal": return result
    if condition == "zero_full": return np.zeros_like(result)
    if condition == "shuffle_full": return np.roll(result, 1, axis=2)
    if condition.startswith("zero_"):
        result[..., HEAD_SLICES[condition.removeprefix("zero_")]] = 0.0; return result
    if condition.startswith("shuffle_"):
        part = HEAD_SLICES[condition.removeprefix("shuffle_")]; result[..., part] = np.roll(result[..., part], 1, axis=2); return result
    raise ValueError(f"unknown Phi condition: {condition}")


def _action_quality(predicted: np.ndarray, sequences: Any) -> tuple[float, float]:
    valid = sequences.uav_mask[:, -1]
    target = sequences.actions; dot = (predicted * target).sum(axis=-1); denom = np.linalg.norm(predicted, axis=-1) * np.linalg.norm(target, axis=-1)
    cosine = np.divide(dot, denom, out=np.ones_like(dot), where=denom > 1.0e-8)
    magnitude = np.abs(np.linalg.norm(predicted, axis=-1) - np.linalg.norm(target, axis=-1))
    return float((cosine * valid).sum() / valid.sum()), float((magnitude * valid).sum() / valid.sum())


def _head_utilization(models: Mapping[tuple[str, int], tuple[Any, Any]], sequences: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Mapping[int, np.ndarray]]:
    conditions = ("normal", "zero_full", "shuffle_full", "zero_service", "zero_progress", "zero_return", "shuffle_service", "shuffle_progress", "shuffle_return")
    config = TrainingConfig(); detail: list[dict[str, Any]] = []; ratios: list[dict[str, Any]] = []; normal_predictions: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        model, normalizer = models[(PREDICTED_TRANSFORMER, seed)]; baseline: np.ndarray | None = None; baseline_rmse = 0.0; baseline_cosine = 0.0; baseline_magnitude = 0.0
        for condition in conditions:
            varied = sequences.replace(predicted_phi=_phi_variant(sequences.predicted_phi, condition)); predicted = predict(PREDICTED_TRANSFORMER, model, normalizer.apply(varied), config)
            rmse = raw_rmse(predicted, sequences.actions, sequences.uav_mask[:, -1]); cosine, magnitude = _action_quality(predicted, sequences)
            if baseline is None:
                baseline, baseline_rmse, baseline_cosine, baseline_magnitude = predicted, rmse, cosine, magnitude; normal_predictions[seed] = predicted
            valid = sequences.uav_mask[:, -1]
            sensitivity = float((np.linalg.norm(predicted - baseline, axis=-1) * valid).sum() / valid.sum())
            head = "full" if condition in {"normal", "zero_full", "shuffle_full"} else condition.split("_", 1)[1]
            detail.append({"seed": seed, "condition": condition, "head": head, "action_sensitivity": sensitivity, "raw_joint_action_rmse": rmse, "RMSE_change": rmse - baseline_rmse, "cosine_change": cosine - baseline_cosine, "magnitude_change": magnitude - baseline_magnitude, "uses_true_phi": False, "validation_only": True, "inference_only": True, "shuffle_scope": "within_current_validation_state_uav_axis" if condition.startswith("shuffle") else "not_applicable"})
        by_condition = {row["condition"]: row for row in detail if row["seed"] == seed}
        total = float(by_condition["zero_full"]["action_sensitivity"])
        ratios.append({"seed": seed, "S_service": by_condition["zero_service"]["action_sensitivity"], "S_progress": by_condition["zero_progress"]["action_sensitivity"], "S_return": by_condition["zero_return"]["action_sensitivity"], "S_total": total, "service_to_total": float(by_condition["zero_service"]["action_sensitivity"] / total) if total else 0.0, "progress_to_total": float(by_condition["zero_progress"]["action_sensitivity"] / total) if total else 0.0, "return_to_total": float(by_condition["zero_return"]["action_sensitivity"] / total) if total else 0.0})
    return detail, ratios, normal_predictions


def _ablation_phi(phi: np.ndarray, condition: str) -> np.ndarray:
    result = phi.copy()
    if condition == "normal": return result
    if condition == "zero_full": return np.zeros_like(result)
    if condition == "shuffle_full": return np.roll(result, 1, axis=0)
    # ``phi`` is [UAV, head], unlike the cached sequence tensor whose head is
    # also the final axis.  Always select the semantic head axis explicitly.
    result[..., HEAD_SLICES[condition.removeprefix("zero_")]] = 0.0
    return result


def _ablation_episode(scenario: Any, seed: int, model: Any, normalizer: Any, predictor: FrozenDeepSetsPredictor, condition: str) -> dict[str, Any]:
    """Validation-only causal rollout on the registered reset/configuration."""
    env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m, dtype=np.float64), options=scenario.options())
    _, info = env.reset(seed=int(scenario.seed)); history: list[tuple[dict[str, Any], np.ndarray]] = []
    total_return = 0.0; arrivals = 0.0; projection = 0; length = 0; minimum = float("inf"); conflicts = False; done = False
    while not done:
        graph = build_graph_state(info, env.config); phi = _ablation_phi(_current_phi(predictor, graph), condition); history.append((graph, phi))
        inputs = _model_inputs(history, normalizer, "cpu"); action = rollout_action(PREDICTED_TRANSFORMER, model, **inputs).cpu().numpy()[0]
        fallback = False
        try:
            _, reward, terminated, truncated, after = env.step(action, matching_mode="legacy")
        except RuntimeError as error:
            if "hard UAV separation projection did not converge" not in str(error): raise
            _, reward, terminated, truncated, after = env.step(np.zeros_like(action), matching_mode="legacy"); fallback = True
        total_return += float(reward); arrivals += float(np.sum(after["raw_arrival_bits"])); length += 1
        projection += int(bool(fallback or np.any(np.asarray(after["safety_correction_m"], dtype=np.float64) > 1.0e-9)))
        minimum = min(minimum, float(after["min_uav_distance_m"])); conflicts = conflicts or bool(after["interference_conflict"])
        info = after; done = bool(terminated or truncated)
    payload_per_buoy = np.asarray(info["cumulative_payload_bits_per_buoy"], dtype=np.float64); offered = float(np.sum(scenario.initial_queue_bits) + arrivals)
    return {"seed": seed, "condition": condition, "scenario_id": str(scenario.scenario_id), "scenario_sha256": _scenario_hash(scenario), "num_buoys": int(scenario.num_buoys), "eta": float(scenario.interference_overlap_factor), "interference_conflict": conflicts, "queue_tier": str(scenario.queue_tier), "geometry_variant": str(scenario.geometry_group), "cumulative_payload_bits": float(info["cumulative_payload_bits"]), "completion": float(info["cumulative_payload_bits"]) / max(offered, 1.0), "residual_queue_bits": float(np.sum(info["raw_queue_bits"])), "episode_return": total_return, "jain_fairness": float(payload_per_buoy.sum() ** 2 / (len(payload_per_buoy) * np.square(payload_per_buoy).sum())) if np.square(payload_per_buoy).sum() > 0 else 0.0, "movement_energy": float(np.sum(info["cumulative_energy_consumption_per_uav"])), "safety_correction_rate": float(projection / max(length, 1)), "minimum_uav_distance_m": minimum, "episode_length": length, "uses_true_phi": False, "validation_only": True, "inference_only": True, "fresh_dev_rollout": False}


def _validation_head_ablation(models: Mapping[tuple[str, int], tuple[Any, Any]], predictor: FrozenDeepSetsPredictor) -> list[dict[str, Any]]:
    scenarios = tuple(row for row in development_scenarios() if row.split == VALIDATION)
    conditions = ("normal", "zero_service", "zero_progress", "zero_return", "zero_full", "shuffle_full")
    raw: list[dict[str, Any]] = []
    for seed in SEEDS:
        model, normalizer = models[(PREDICTED_TRANSFORMER, seed)]
        for scenario in scenarios:
            for condition in conditions: raw.append(_ablation_episode(scenario, seed, model, normalizer, predictor, condition))
    normal = {(int(row["seed"]), str(row["scenario_id"])): row for row in raw if row["condition"] == "normal"}
    rows: list[dict[str, Any]] = []
    for row in raw:
        base = normal[(int(row["seed"]), str(row["scenario_id"]))]
        rows.append({**row, "delta_payload_bits_from_normal": _f(row["cumulative_payload_bits"]) - _f(base["cumulative_payload_bits"]), "delta_completion_from_normal": _f(row["completion"]) - _f(base["completion"]), "delta_residual_queue_bits_from_normal": _f(row["residual_queue_bits"]) - _f(base["residual_queue_bits"]), "delta_return_from_normal": _f(row["episode_return"]) - _f(base["episode_return"]), "delta_safety_correction_rate_from_normal": _f(row["safety_correction_rate"]) - _f(base["safety_correction_rate"]), "same_registered_validation_snapshot": True})
    return rows


def _head_gate(causal: Sequence[Mapping[str, Any]], head_detail: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload_rows: list[dict[str, Any]] = []; gates: dict[str, Any] = {}
    for head in ("service", "progress", "return"):
        condition = f"zero_{head}"; subset = [row for row in causal if str(row["condition"]) == condition]
        seed_means = {seed: _mean([row for row in subset if int(row["seed"]) == seed], "delta_payload_bits_from_normal") for seed in SEEDS}
        low, high = _scenario_bootstrap([{ "scenario_id": row["scenario_id"], "delta_cumulative_payload_bits": row["delta_payload_bits_from_normal"] } for row in subset], 4000, 23050 + len(payload_rows))
        beneficial = sum(value < 0.0 for value in seed_means.values())
        gate = "HV-A" if beneficial >= 2 and high < 0.0 else "HV-B" if beneficial > 0 and beneficial < 3 else "HV-C"
        gates[head] = {"gate": gate, "zero_head_payload_delta_by_seed": seed_means, "aggregate_payload_delta_mean": _mean(subset, "delta_payload_bits_from_normal"), "aggregate_ci95": [low, high], "beneficial_seed_count": beneficial, "classification_rule": "HV-A requires payload decline in at least two seeds and aggregate upper CI below zero; action sensitivity alone is insufficient"}
        for seed in SEEDS:
            sensitivity = next(_f(row["action_sensitivity"]) for row in head_detail if int(row["seed"]) == seed and str(row["condition"]) == condition)
            payload_rows.append({"head": head, "seed": seed, "normal_vs_zero_return": head == "return", "action_sensitivity": sensitivity, "payload_delta_bits_zero_minus_normal": seed_means[seed], "completion_delta": _mean([row for row in subset if int(row["seed"]) == seed], "delta_completion_from_normal"), "residual_queue_delta_bits": _mean([row for row in subset if int(row["seed"]) == seed], "delta_residual_queue_bits_from_normal"), "safety_delta": _mean([row for row in subset if int(row["seed"]) == seed], "delta_safety_correction_rate_from_normal"), "head_gate": gate})
    return {"round": "round_23", "heads": gates, "validation_only": True, "inference_only": True}, payload_rows


def _semantic_alignment(predictions: Mapping[int, np.ndarray], sequences: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    uav = sequences.uav_nodes[:, -1, :, :2]; buoy = sequences.buoy_nodes[:, -1, :, :2]; buoy_mask = sequences.buoy_mask[:, -1]
    phi = sequences.predicted_phi[:, -1]
    for seed in SEEDS:
        action = predictions[seed]
        for name, component in (("service", phi[..., :5]), ("progress", phi[..., 5:10])):
            values: list[float] = []
            for index in range(len(sequences)):
                for agent in range(action.shape[1]):
                    weights = component[index, agent] * buoy_mask[index]
                    direction = buoy[index] - uav[index, agent]
                    distance = np.linalg.norm(direction, axis=-1, keepdims=True)
                    target = (direction / np.maximum(distance, 1.0e-8) * weights[:, None]).sum(axis=0)
                    denominator = float(np.linalg.norm(action[index, agent]) * np.linalg.norm(target))
                    if denominator > 1.0e-8: values.append(float(np.dot(action[index, agent], target) / denominator))
            rows.append({"seed": seed, "semantic_head": name, "directional_consistency_cosine": float(np.mean(values)), "valid_uav_samples": len(values), "uses_predicted_current_phi": True, "uses_future_action": False})
    return rows


def _direction_regions(values: np.ndarray) -> np.ndarray:
    return np.floor(((np.arctan2(values[..., 1], values[..., 0]) + np.pi) / (2.0 * np.pi)) * 8.0).astype(int) % 8


def _action_disagreement(predictions: Mapping[tuple[str, int], np.ndarray], validation_steps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    safety: dict[tuple[int, str, int], bool] = {(int(row["model_seed"]), str(row["scenario_id"]), int(row["timestep"])): _b(row["projection_activated"]) for row in validation_steps if str(row["family"]) == PREDICTED_TRANSFORMER}
    for family, label in ((PREDICTED_TRANSFORMER, "task_waypoint_transformer"), (STATE_ONLY_TRANSFORMER, "state_only_transformer")):
        for left, right in ((0, 1), (0, 2), (1, 2)):
            a, b = predictions[(family, left)], predictions[(family, right)]
            dot = (a * b).sum(axis=-1); denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
            cosine = np.divide(dot, denom, out=np.ones_like(dot), where=denom > 1.0e-8)
            safety_pairs = [(value, safety.get((right, scenario, step))) for (seed, scenario, step), value in safety.items() if seed == left]
            rows.append({"family": label, "seed_pair": f"{left}-{right}", "euclidean_joint_action_disagreement": float(np.mean(np.linalg.norm(a - b, axis=-1))), "per_uav_cosine_disagreement": float(1.0 - np.mean(cosine)), "target_region_disagreement": float(np.mean(_direction_regions(a) != _direction_regions(b))), "safety_projection_disagreement": float(np.mean([first != second for first, second in safety_pairs if second is not None])) if family == PREDICTED_TRANSFORMER else "not_applicable", "inference_only": True})
    return rows


def _representation_batch(model: Any, batch: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Extract existing frozen components without changing the model forward path."""
    with torch.no_grad():
        state, _ = model.state_encoder(batch["uav_nodes"], batch["buoy_nodes"], batch["uav_mask"], batch["buoy_mask"], batch["global_features"])
        task = model._condition(batch["buoy_nodes"], batch["buoy_mask"], batch["predicted_phi"], state.shape[2])
        fused = state + task; bsz, steps, uavs, width = fused.shape
        shared = fused.permute(0, 2, 1, 3).reshape(bsz * uavs, steps, width)
        valid = batch["attention_mask"][:, None, :].expand(-1, uavs, -1).reshape(bsz * uavs, steps)
        causal = torch.tril(torch.ones((steps, steps), dtype=torch.bool, device=fused.device)); allowed = valid[:, :, None] & valid[:, None, :] & causal
        eye = torch.eye(steps, dtype=torch.bool, device=fused.device).unsqueeze(0); allowed = torch.where(valid[:, :, None], allowed, eye)
        encoded = model.transformer(shared, mask=(~allowed).repeat_interleave(model.schema.heads, dim=0)).reshape(bsz, uavs, steps, width).permute(0, 2, 1, 3)
        return {"state_token": state[:, -1].cpu().numpy(), "task_waypoint_token": task[:, -1].cpu().numpy(), "fused_token": fused[:, -1].cpu().numpy(), "final_temporal_representation": encoded[:, -1].cpu().numpy()}


def _effective_rank(values: np.ndarray) -> float:
    flat = values.reshape(-1, values.shape[-1]).astype(np.float64); centered = flat - flat.mean(axis=0, keepdims=True); singular = np.linalg.svd(centered, compute_uv=False)
    mass = singular / singular.sum() if singular.sum() > 1.0e-12 else singular
    return float(np.exp(-np.sum(np.where(mass > 0, mass * np.log(mass), 0.0)))) if len(mass) else 0.0


def _representation_audit(models: Mapping[tuple[str, int], tuple[Any, Any]], sequences: Any) -> list[dict[str, Any]]:
    collected: dict[int, dict[str, np.ndarray]] = {}
    for seed in SEEDS:
        model, normalizer = models[(PREDICTED_TRANSFORMER, seed)]; normalized = normalizer.apply(sequences); parts: dict[str, list[np.ndarray]] = defaultdict(list)
        for start in range(0, len(normalized), 256):
            batch = normalized.batch(range(start, min(len(normalized), start + 256)), "cpu")
            for name, value in _representation_batch(model, batch).items(): parts[name].append(value)
        collected[seed] = {name: np.concatenate(values, axis=0) for name, values in parts.items()}
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        entries = collected[seed]; state_norm = np.linalg.norm(entries["state_token"], axis=-1).mean(); task_norm = np.linalg.norm(entries["task_waypoint_token"], axis=-1).mean()
        for token, values in entries.items():
            rows.append({"record_type": "seed", "seed": seed, "token": token, "variance": float(np.var(values)), "effective_rank": _effective_rank(values), "token_norm": float(np.linalg.norm(values, axis=-1).mean()), "state_to_task_norm_ratio": float(state_norm / task_norm) if task_norm > 1.0e-12 else 0.0, "sample_count": int(values.shape[0] * values.shape[1]), "inference_only": True})
    for left, right in ((0, 1), (0, 2), (1, 2)):
        for token in collected[left]:
            a, b = collected[left][token], collected[right][token]; dot = (a * b).sum(axis=-1); denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
            cosine = np.divide(dot, denom, out=np.ones_like(dot), where=denom > 1.0e-8)
            rows.append({"record_type": "pair", "seed_pair": f"{left}-{right}", "token": token, "pairwise_cosine_similarity": float(np.mean(cosine)), "sample_count": int(cosine.size), "inference_only": True})
    return rows


def _seed_stability(predictions: Mapping[tuple[str, int], np.ndarray], validation_metrics: Sequence[Mapping[str, Any]], fresh_metrics: Sequence[Mapping[str, Any]], offline_rmse: Mapping[int, float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []; mlp_rows: list[dict[str, Any]] = []
    for family, label in ((PREDICTED_TRANSFORMER, "task_waypoint_transformer"), (STATE_ONLY_TRANSFORMER, "state_only_transformer"), (PREDICTED_MLP, "task_waypoint_mlp")):
        offline_values = [offline_rmse[seed] if family == PREDICTED_TRANSFORMER else _f(next(row["best_validation_raw_joint_action_rmse"] for row in _formal_runs() if str(row["family"]) == family and int(row["seed"]) == seed)) for seed in SEEDS]
        validation_values = [_mean(_by_family_seed(validation_metrics, family, seed), "cumulative_payload_bits") for seed in SEEDS]
        fresh_values = [_mean(_by_family_seed(fresh_metrics, family, seed), "cumulative_payload_bits") for seed in SEEDS]
        disagreements = []
        for left, right in ((0, 1), (0, 2), (1, 2)): disagreements.append(float(np.mean(np.linalg.norm(predictions[(family, left)] - predictions[(family, right)], axis=-1))))
        target = mlp_rows if family == PREDICTED_MLP else rows
        for metric, values in (("validation_offline_rmse", offline_values), ("validation_closed_loop_payload_bits", validation_values), ("fresh_frozen_payload_bits", fresh_values), ("pairwise_action_disagreement", disagreements)):
            target.append({"family": label, "metric": metric, "seed_values": values, "variance": float(np.var(values, ddof=1)), "mean": float(np.mean(values)), "n": len(values), "frozen_fresh_results_only": metric != "validation_offline_rmse"})
    return rows, mlp_rows


def _conditional_value(validation_pairs: Sequence[Mapping[str, Any]], fresh_pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    definitions = (("interference_conflict", lambda r: str(r["interference_conflict"])), ("coordination_bridge", lambda r: str(r["bridge"])), ("num_buoys", lambda r: str(r["num_buoys"])), ("eta", lambda r: str(r["eta"])))
    for population, source in (("validation", validation_pairs), ("fresh_frozen", fresh_pairs)):
        for field, value_fn in definitions:
            groups = sorted({value_fn(row) for row in source})
            for group in groups:
                for seed in SEEDS:
                    subset = [row for row in source if value_fn(row) == group and int(row["model_seed"]) == seed]
                    if subset: rows.append({"population": population, "frozen_group_definition": field, "group_value": group, "seed": seed, "scenario_count": len(subset), "delta_payload_bits": _mean(subset, "delta_cumulative_payload_bits"), "delta_completion": _mean(subset, "delta_completion"), "delta_residual_queue_bits": _mean(subset, "delta_residual_queue_bits")})
    return rows


def _interaction(rows: Sequence[Mapping[str, Any]], field: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for value in sorted({str(row[field]) for row in rows}):
        for seed in SEEDS:
            subset = [row for row in rows if str(row[field]) == value and int(row["model_seed"]) == seed]
            output.append({"group_field": field, "group_value": value, "seed": seed, "scenario_count": len(subset), "delta_payload_bits": _mean(subset, "delta_cumulative_payload_bits"), "delta_completion": _mean(subset, "delta_completion"), "delta_residual_queue_bits": _mean(subset, "delta_residual_queue_bits")})
    return output


def _predictor_reassessment() -> list[dict[str, Any]]:
    rows = _rows(R22_OUTPUT / "offline_validation_metrics.csv")
    left = {(str(row["sample_id"]), int(row["seed"])): row for row in rows if str(row["family"]) == PREDICTED_TRANSFORMER}; right = {(str(row["sample_id"]), int(row["seed"])): row for row in rows if str(row["family"]) == "oracle_task_waypoint_transformer"}
    pairs = [{"seed": seed, "scenario_id": value["scenario_id"], "conflict": value["conflict"], "bridge": value["bridge"], "num_buoys": value["num_buoys"], "eta": value["eta"], "predicted_minus_oracle_rmse": _f(value["raw_action_rmse"]) - _f(right[(sample, seed)]["raw_action_rmse"])} for (sample, seed), value in left.items()]
    result: list[dict[str, Any]] = []
    for field in ("overall", "conflict", "bridge", "num_buoys", "eta"):
        groups = ["all"] if field == "overall" else sorted({str(row[field]) for row in pairs})
        for group in groups:
            subset = pairs if field == "overall" else [row for row in pairs if str(row[field]) == group]
            result.append({"group_field": field, "group_value": group, "sample_count": len(subset), "predicted_minus_oracle_rmse": _mean(subset, "predicted_minus_oracle_rmse"), "small_gap_threshold_rmse": 0.02, "predictor_bottleneck_weakened": abs(_mean(subset, "predicted_minus_oracle_rmse")) < 0.02})
    return result


def _mechanism_and_gates(bootstrap: Sequence[Mapping[str, Any]], tail: Sequence[Mapping[str, Any]], head_gate: Mapping[str, Any], ratios: Sequence[Mapping[str, Any]], conditional: Sequence[Mapping[str, Any]], stability: Sequence[Mapping[str, Any]], mlp: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    hierarchical = next(row for row in bootstrap if row["method"] == "hierarchical_seed_then_scenario_bootstrap")
    leave_positive = min((row for row in bootstrap if row["method"] == "leave_one_seed_out"), key=lambda row: float(row["mean_delta_payload_bits"]))
    positive_seed = int(max(tail, key=lambda row: float(row["mean_delta_payload_bits"]))["seed"]); tail_row = next(row for row in tail if int(row["seed"]) == positive_seed)
    conditional_support = []
    for field in ("interference_conflict", "coordination_bridge"):
        values = [row for row in conditional if row["population"] == "validation" and row["frozen_group_definition"] == field and str(row["group_value"]) == "True"]
        conditional_support.append(sum(_f(row["delta_payload_bits"]) > 0.0 for row in values) >= 2)
    state_variance = next(row for row in stability if row["family"] == "state_only_transformer" and row["metric"] == "validation_closed_loop_payload_bits")["variance"]
    task_variance = next(row for row in stability if row["family"] == "task_waypoint_transformer" and row["metric"] == "validation_closed_loop_payload_bits")["variance"]
    mlp_variance = next(row for row in mlp if row["metric"] == "validation_closed_loop_payload_bits")["variance"]
    return_gate = str(head_gate["heads"]["return"]["gate"])
    evidence = {
        "round": "round_23",
        "ratings": {
            "predictor_bottleneck": {"rating": "weak", "support": "Oracle-versus-predicted offline gap is small in the locked Round-22 record and is reassessed by frozen subgroup rows."},
            "policy_initialization_instability": {"rating": "strong" if float(task_variance) > 1.5 * float(state_variance) else "moderate", "support": "Task-Waypoint policy payload seed variance is compared with matched State-only Transformer variance."},
            "return_head_over_dominance": {"rating": "moderate" if return_gate in {"HV-B", "HV-C"} else "weak", "support": "Return sensitivity and zero-Return causal payload result are reported separately."},
            "insufficient_service_utilization": {"rating": "moderate" if float(np.mean([_f(row["service_to_total"]) for row in ratios])) < .15 else "weak", "support": "Per-seed Service-to-total sensitivity ratios."},
            "insufficient_progress_utilization": {"rating": "moderate" if float(np.mean([_f(row["progress_to_total"]) for row in ratios])) < .20 else "weak", "support": "Per-seed Progress-to-total sensitivity ratios."},
            "temporal_transformer_instability": {"rating": "strong" if float(task_variance) > 1.5 * float(mlp_variance) else "moderate", "support": "Task-Waypoint Transformer and MLP seed payload variances."},
            "conditional_conflict_bridge_waypoint_value": {"rating": "moderate" if any(conditional_support) else "weak", "support": "Pre-registered validation conflict/bridge delta payload rows by seed."},
        },
        "round22_mechanism_classification_unchanged": "P-MU1 + P-MU4",
    }
    stab = "STAB-A" if float(hierarchical["ci95_lower"]) > 0.0 and sum(_f(row["mean_delta_payload_bits"]) > 0.0 for row in tail) >= 2 and float(tail_row["top5_positive_contribution_share"]) < .50 else "STAB-B" if any(conditional_support) else "STAB-C"
    stability_gate = {"gate": stab, "hierarchical_ci95": [hierarchical["ci95_lower"], hierarchical["ci95_upper"]], "positive_seed": positive_seed, "leave_positive_seed_out_delta_bits": leave_positive["mean_delta_payload_bits"], "positive_seed_tail_dominated": float(tail_row["top5_positive_contribution_share"]) >= .50, "cross_seed_conditional_conflict_or_bridge": any(conditional_support), "round22_mu_wp_gate_unchanged": "MU-WP-B"}
    task_mean = next(row for row in stability if row["family"] == "task_waypoint_transformer" and row["metric"] == "fresh_frozen_payload_bits")["mean"]
    mlp_mean = next(row for row in mlp if row["metric"] == "fresh_frozen_payload_bits")["mean"]
    temporal_gate = "TV-A" if float(task_mean) > float(mlp_mean) and float(task_variance) <= 1.25 * float(mlp_variance) else "TV-B" if float(task_mean) > float(mlp_mean) else "TV-C"
    temporal = {"gate": temporal_gate, "taskwp_transformer_fresh_payload_mean": task_mean, "taskwp_mlp_fresh_payload_mean": mlp_mean, "taskwp_transformer_validation_payload_variance": task_variance, "taskwp_mlp_validation_payload_variance": mlp_variance}
    readiness = "R23-A" if stab == "STAB-B" and any(conditional_support) else "R23-B" if stab == "STAB-C" and evidence["ratings"]["return_head_over_dominance"]["rating"] in {"strong", "moderate"} else "R23-C" if float(task_variance) <= 1.5 * float(state_variance) else "R23-D"
    return evidence, stability_gate, temporal, {"gate": readiness, "more_training_seeds_recommended": bool(readiness == "R23-A" and stab == "STAB-B"), "rationale": "recommend more seeds only if a cross-seed conditional mechanism is present; no seed is trained in Round-23", "round24_created": False}


def _save_figures(bootstrap: Sequence[Mapping[str, Any]], seed_summary: Sequence[Mapping[str, Any]], matrix: Sequence[Mapping[str, Any]], wins: Sequence[Mapping[str, Any]], tail: Sequence[Mapping[str, Any]], alignment: Sequence[Mapping[str, Any]], head_detail: Sequence[Mapping[str, Any]], causal: Sequence[Mapping[str, Any]], conditional: Sequence[Mapping[str, Any]], eta: Sequence[Mapping[str, Any]], counts: Sequence[Mapping[str, Any]], stability: Sequence[Mapping[str, Any]], representation: Sequence[Mapping[str, Any]]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    def save(figure: Any, stem: str) -> None:
        figure.tight_layout(); figure.savefig(FIGURES / f"{stem}.png", dpi=160); figure.savefig(FIGURES / f"{stem}.svg"); plt.close(figure)
    def bar(stem: str, labels: Sequence[str], values: Sequence[float], title: str, ylabel: str) -> None:
        figure, axis = plt.subplots(figsize=(7, 4)); axis.bar(labels, values); axis.axhline(0, color="black", lw=.8); axis.set(title=title, ylabel=ylabel); axis.grid(axis="y", alpha=.25); save(figure, stem)
    seeds = [str(row["seed"]) for row in seed_summary]
    bar("01_validation_delta_payload_by_seed", seeds, [float(row["validation_payload_delta_bits"]) for row in seed_summary], "Validation paired payload delta by Task-WP seed", "bits")
    bar("02_fresh_delta_payload_by_seed", seeds, [float(row["fresh_payload_delta_bits"]) for row in seed_summary], "Frozen fresh-development payload delta by Task-WP seed", "bits")
    array = np.asarray([[float(row[f"seed_{seed}_delta_payload_bits"]) for seed in SEEDS] for row in matrix]); figure, axis = plt.subplots(figsize=(8, 5)); image = axis.imshow(array, aspect="auto", cmap="coolwarm"); axis.set(title="Frozen fresh seed × scenario payload delta", xlabel="training seed", ylabel="fresh scenario"); axis.set_xticks(range(3), ["0", "1", "2"]); figure.colorbar(image, ax=axis, label="bits"); save(figure, "03_seed_scenario_heatmap")
    bar("04_seed_win_rate", [str(row["seed"]) for row in wins], [float(row["win_rate"]) for row in wins], "Fresh win rate over State-only", "fraction")
    figure, axis = plt.subplots(figsize=(7, 4)); axis.boxplot([np.asarray([float(row[f"seed_{seed}_delta_payload_bits"]) for row in matrix]) for seed in SEEDS], labels=["0", "1", "2"]); axis.set(title="Frozen fresh payload-delta distribution", xlabel="Task-WP seed", ylabel="bits"); axis.axhline(0, color="black", lw=.8); save(figure, "05_seed_delta_distribution")
    leave = [row for row in bootstrap if row["method"] == "leave_one_seed_out"]; bar("06_leave_one_seed_out", [f"remove {row['excluded_seed']}" for row in leave], [float(row["mean_delta_payload_bits"]) for row in leave], "Leave-one-seed-out fresh delta", "bits")
    boot = [row for row in bootstrap if row["method"] != "leave_one_seed_out"]; figure, axis = plt.subplots(figsize=(7, 4)); means = [float(row["mean_delta_payload_bits"]) for row in boot]; lower = [mean - float(row["ci95_lower"]) for mean, row in zip(means, boot)]; upper = [float(row["ci95_upper"]) - mean for mean, row in zip(means, boot)]; axis.errorbar(range(len(boot)), means, yerr=[lower, upper], fmt="o"); axis.set_xticks(range(len(boot)), ["scenario", "seed cluster", "hierarchical"], rotation=15); axis.axhline(0, color="black", lw=.8); axis.set(title="Fresh bootstrap uncertainty audit", ylabel="payload delta (bits)"); save(figure, "07_hierarchical_bootstrap")
    scatter = [row for row in alignment if row.get("analysis_level") == "seed" and row["outcome"] == "fresh_payload_delta_bits"]; figure, axis = plt.subplots(figsize=(6, 4)); axis.scatter([row["validation_action_rmse"] for row in seed_summary], [row["fresh_payload_delta_bits"] for row in seed_summary]); [axis.annotate(str(row["seed"]), (row["validation_action_rmse"], row["fresh_payload_delta_bits"])) for row in seed_summary]; axis.set(title=f"Offline RMSE vs fresh payload (n=3; r={scatter[0]['pearson']:.3f})", xlabel="validation action RMSE", ylabel="fresh delta (bits)"); save(figure, "08_offline_rmse_vs_payload")
    for number, head in ((9, "service"), (10, "progress"), (11, "return")):
        rows = [row for row in head_detail if row["condition"] == f"zero_{head}"]; bar(f"{number:02d}_{head}_sensitivity", [str(row["seed"]) for row in rows], [float(row["action_sensitivity"]) for row in rows], f"{head.title()} action sensitivity by seed", "mean action change")
    for number, head in ((12, "service"), (13, "progress"), (14, "return")):
        rows = [row for row in causal if row["condition"] == f"zero_{head}"]; values = [_mean([row for row in rows if int(row["seed"]) == seed], "delta_payload_bits_from_normal") for seed in SEEDS]; bar(f"{number:02d}_zero_{head}_causal_payload", [str(seed) for seed in SEEDS], values, f"Validation causal payload: zero {head}", "zero minus normal (bits)")
    for number, field, value, title in ((15, "interference_conflict", "True", "Conflict conditional value"), (16, "coordination_bridge", "True", "Bridge conditional value")):
        rows = [row for row in conditional if row["population"] == "validation" and row["frozen_group_definition"] == field and str(row["group_value"]) == value]; bar(f"{number:02d}_{field}_conditional_value", [str(seed) for seed in SEEDS], [_f(next(row["delta_payload_bits"] for row in rows if int(row["seed"]) == seed)) for seed in SEEDS], title, "Task-WP minus State-only (bits)")
    for number, source, title in ((17, eta, "Frozen fresh eta × seed interaction"), (18, counts, "Frozen fresh K × seed interaction")):
        groups = sorted({str(row["group_value"]) for row in source}); figure, axis = plt.subplots(figsize=(8, 4)); width = .22
        for index, seed in enumerate(SEEDS): axis.bar(np.arange(len(groups)) + (index - 1) * width, [_f(next(row["delta_payload_bits"] for row in source if str(row["group_value"]) == group and int(row["seed"]) == seed)) for group in groups], width=width, label=f"seed {seed}")
        axis.set_xticks(range(len(groups)), groups); axis.axhline(0, color="black", lw=.8); axis.set(title=title, ylabel="payload delta (bits)"); axis.legend(); save(figure, f"{number:02d}_{'eta' if number == 17 else 'K'}_seed_interaction")
    var_rows = [row for row in stability if row["metric"] == "validation_closed_loop_payload_bits"]; bar("19_taskwp_vs_state_seed_variance", [row["family"] for row in var_rows], [float(row["variance"]) for row in var_rows], "Validation payload seed variance", "variance (bits²)")
    pairs = [row for row in representation if row.get("record_type") == "pair" and row["token"] == "final_temporal_representation"]; figure, axis = plt.subplots(figsize=(5, 4)); sim = np.eye(3); 
    for row in pairs:
        left, right = (int(part) for part in str(row["seed_pair"]).split("-")); sim[left, right] = sim[right, left] = _f(row["pairwise_cosine_similarity"])
    image = axis.imshow(sim, vmin=-1, vmax=1, cmap="viridis"); axis.set(title="Final representation similarity across Task-WP seeds", xticks=range(3), yticks=range(3), xticklabels=["0", "1", "2"], yticklabels=["0", "1", "2"]); figure.colorbar(image, ax=axis); save(figure, "20_representation_similarity")
    return sorted(path.name for path in FIGURES.iterdir() if path.suffix in {".png", ".svg"})


REQUIRED_OUTPUTS = (
    "seed_level_policy_summary.csv", "seed_identity_audit.json", "hierarchical_seed_bootstrap.csv",
    "seed_scenario_delta_matrix.csv", "seed_win_loss_audit.csv", "seed_tail_contribution_audit.csv",
    "offline_closed_loop_alignment.csv", "seed_grouped_action_error.csv", "seed_head_utilization.csv",
    "head_utilization_ratio.csv", "validation_head_causal_ablation.csv", "task_waypoint_head_causal_gate.json",
    "return_head_benefit_audit.csv", "task_semantic_action_alignment.csv", "policy_seed_action_disagreement.csv",
    "taskwp_vs_state_seed_stability.csv", "mlp_seed_stability.csv", "conditional_waypoint_value_audit.csv",
    "seed_eta_interaction.csv", "seed_K_interaction.csv", "policy_representation_seed_audit.csv",
    "predictor_bottleneck_reassessment.csv", "round23_mechanism_evidence.json",
    "task_waypoint_policy_stability_gate.json", "temporal_policy_value_gate.json", "round23_readiness_gate.json",
    "access_audit.json", "run_manifest.json",
)


def _finite_files() -> None:
    """Fail closed rather than preserving a non-finite audit result."""
    for path in OUTPUT.glob("*.csv"):
        for row in _rows(path):
            for value in row.values():
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(numeric):
                    raise RuntimeError(f"non-finite Round-23 result in {path.name}")


def _completion_report(seed_summary: Sequence[Mapping[str, Any]], bootstrap: Sequence[Mapping[str, Any]], wins: Sequence[Mapping[str, Any]], tail: Sequence[Mapping[str, Any]], head_gate: Mapping[str, Any], stability_gate: Mapping[str, Any], temporal_gate: Mapping[str, Any], readiness: Mapping[str, Any], evidence: Mapping[str, Any]) -> None:
    positive = max(seed_summary, key=lambda row: _f(row["fresh_payload_delta_bits"]))
    negative = [int(row["seed"]) for row in seed_summary if int(row["seed"]) != int(positive["seed"])]
    positive_win = next(row for row in wins if int(row["seed"]) == int(positive["seed"]))
    positive_tail = next(row for row in tail if int(row["seed"]) == int(positive["seed"]))
    hierarchical = next(row for row in bootstrap if row["method"] == "hierarchical_seed_then_scenario_bootstrap")
    removed = next(row for row in bootstrap if row["method"] == "leave_one_seed_out" and int(row["excluded_seed"]) == int(positive["seed"]))
    report = ROUND / "report" / "round23_completion_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Round-23 Completion Report\n\n"
        "This is a frozen-checkpoint, inference-only audit.  It does not add a model seed, retrain a policy or predictor, run fresh-development rollouts, or access untouched evaluation.\n\n"
        "## Seed identity and uncertainty\n\n"
        f"- Positive fresh-development seed: **{positive['seed']}**; negative seeds: **{negative}**.\n"
        f"- Positive-seed win rate: {float(positive_win['win_rate']):.3f}; top-five positive contribution share: {float(positive_tail['top5_positive_contribution_share']):.3f}.\n"
        f"- Hierarchical seed-then-scenario bootstrap: {float(hierarchical['mean_delta_payload_bits']):.2f} bits, 95% CI [{float(hierarchical['ci95_lower']):.2f}, {float(hierarchical['ci95_upper']):.2f}].\n"
        f"- Removing the positive seed: {float(removed['mean_delta_payload_bits']):.2f} bits.\n\n"
        "## Causal heads and gates\n\n"
        f"- Service: {head_gate['heads']['service']['gate']}; Progress: {head_gate['heads']['progress']['gate']}; Return: {head_gate['heads']['return']['gate']}.\n"
        f"- Stability gate: **{stability_gate['gate']}**; temporal gate: **{temporal_gate['gate']}**; Round-23 readiness: **{readiness['gate']}**.\n"
        f"- More training seeds recommended: {str(bool(readiness['more_training_seeds_recommended'])).lower()} (recommendation only; no seed was added).\n\n"
        "## Mechanism evidence\n\n"
        + "\n".join(f"- {name}: **{item['rating']}** — {item['support']}" for name, item in evidence["ratings"].items())
        + "\n\nRound-22’s locked MU-WP-B gate and P-MU1 + P-MU4 classification are historical facts and remain unchanged. Final regression and source/output-lock verification are recorded in `ROUND_LOCK.json`.\n",
        encoding="utf-8",
    )


def run_audit() -> dict[str, Any]:
    """Produce all Round-23 evidence from frozen Round-22 artifacts only."""
    OUTPUT.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    configuration = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    frozen = verify_frozen_state()
    predictor = FrozenDeepSetsPredictor()
    if predictor.sha256 != EXPECTED_PREDICTOR:
        raise RuntimeError("frozen Task-Waypoint predictor hash changed before Round-23 audit")
    sequences = load_policy_sequences(VALIDATION, predictor)
    config = TrainingConfig()
    runs = _formal_runs()
    models: dict[tuple[str, int], tuple[Any, Any]] = {}
    predictions: dict[tuple[str, int], np.ndarray] = {}
    for run in runs:
        family, seed = str(run["family"]), int(run["seed"])
        if family == "oracle_task_waypoint_transformer":
            continue
        loaded_family, model, normalizer, payload = load_checkpoint(ROOT / str(run["checkpoint"]), config.device)
        if loaded_family != family or int(payload["seed"]) != seed:
            raise RuntimeError("frozen formal checkpoint identity changed")
        models[(family, seed)] = (model, normalizer)
        predictions[(family, seed)] = predict(family, model, normalizer.apply(sequences), config)
    required_models = {(family, seed) for family in (PREDICTED_TRANSFORMER, STATE_ONLY_TRANSFORMER, PREDICTED_MLP) for seed in SEEDS}
    if set(models) != required_models:
        raise RuntimeError("Round-23 requires all nine deployable frozen policy checkpoints")

    seed_summary, offline_rmse, _ = _seed_level_summary(runs, predictions, sequences)
    validation_pairs = _rows(R22_OUTPUT / "validation_paired_comparison.csv")
    fresh_pairs = _rows(R22_OUTPUT / "fresh_dev_paired_comparison.csv")
    validation_metrics = _rows(R22_OUTPUT / "validation_closed_loop_metrics.csv")
    fresh_metrics = _rows(R22_OUTPUT / "fresh_dev_closed_loop_metrics.csv")
    validation_steps = _rows(R22_OUTPUT / "validation_closed_loop_steps.csv")
    bootstrap, boot_context = _bootstrap_audit(fresh_pairs, configuration)
    matrix, by_seed = boot_context["matrix"], boot_context["by_seed"]
    wins = _win_loss_audit(by_seed, float(configuration["payload_tie_tolerance_bits"]))
    tail = _tail_audit(by_seed)
    agreement = _fresh_agreement(by_seed)
    alignment = _offline_alignment(offline_rmse, validation_pairs, fresh_pairs)
    for summary in seed_summary:
        alignment.append({"analysis_level": "seed_observation", "seed": summary["seed"], "validation_action_rmse": summary["validation_action_rmse"], "validation_payload_delta_bits": summary["validation_payload_delta_bits"], "fresh_payload_delta_bits": summary["fresh_payload_delta_bits"], "interpretation": "descriptive_only_n_equals_3"})
    grouped_action = _grouped_action_error()
    head_detail, ratios, normal_predictions = _head_utilization(models, sequences)
    causal = _validation_head_ablation(models, predictor)
    head_gate, return_benefit = _head_gate(causal, head_detail)
    semantic = _semantic_alignment(normal_predictions, sequences)
    disagreement = _action_disagreement(predictions, validation_steps)
    representation = _representation_audit(models, sequences)
    stability, mlp_stability = _seed_stability(predictions, validation_metrics, fresh_metrics, offline_rmse)
    conditional = _conditional_value(validation_pairs, fresh_pairs)
    eta_interaction, k_interaction = _interaction(fresh_pairs, "eta"), _interaction(fresh_pairs, "num_buoys")
    predictor_reassessment = _predictor_reassessment()
    evidence, stability_gate, temporal_gate, readiness = _mechanism_and_gates(bootstrap, tail, head_gate, ratios, conditional, stability, mlp_stability)

    outputs: dict[str, Any] = {
        "seed_level_policy_summary.csv": seed_summary,
        "hierarchical_seed_bootstrap.csv": bootstrap,
        "seed_scenario_delta_matrix.csv": matrix,
        "seed_win_loss_audit.csv": wins,
        "seed_tail_contribution_audit.csv": tail,
        "offline_closed_loop_alignment.csv": alignment,
        "seed_grouped_action_error.csv": grouped_action,
        "seed_head_utilization.csv": head_detail,
        "head_utilization_ratio.csv": ratios,
        "validation_head_causal_ablation.csv": causal,
        "return_head_benefit_audit.csv": return_benefit,
        "task_semantic_action_alignment.csv": semantic,
        "policy_seed_action_disagreement.csv": disagreement,
        "taskwp_vs_state_seed_stability.csv": stability,
        "mlp_seed_stability.csv": mlp_stability,
        "conditional_waypoint_value_audit.csv": conditional,
        "seed_eta_interaction.csv": eta_interaction,
        "seed_K_interaction.csv": k_interaction,
        "policy_representation_seed_audit.csv": representation,
        "predictor_bottleneck_reassessment.csv": predictor_reassessment,
    }
    for name, rows in outputs.items():
        _csv(OUTPUT / name, rows)
    positive_seed = int(boot_context["positive_seed"])
    identity = {"positive_seed_id": positive_seed, "negative_seed_ids": boot_context["negative_seeds"], "identity_rule": "maximum locked fresh-development mean paired payload delta", "seeds": [{key: row[key] for key in ("seed", "checkpoint", "checkpoint_sha256", "best_epoch", "validation_action_rmse", "validation_payload_delta_bits", "fresh_payload_delta_bits")} for row in seed_summary]}
    access = {**frozen, "round": "round_23", "fresh_dev_source": "Round-22 frozen raw outputs only", "fresh_dev_new_rollout_count": 0, "untouched_eval_access_count": 0, "untouched_eval_accessed": False, "validation_head_ablation_rollouts": len(causal), "validation_only_new_inference": True, "uses_true_phi": False}
    _json(OUTPUT / "seed_identity_audit.json", identity)
    _json(OUTPUT / "task_waypoint_head_causal_gate.json", head_gate)
    _json(OUTPUT / "round23_mechanism_evidence.json", evidence)
    _json(OUTPUT / "task_waypoint_policy_stability_gate.json", stability_gate)
    _json(OUTPUT / "temporal_policy_value_gate.json", temporal_gate)
    _json(OUTPUT / "round23_readiness_gate.json", readiness)
    _json(OUTPUT / "access_audit.json", access)
    figures = _save_figures(bootstrap, seed_summary, matrix, wins, tail, alignment, head_detail, causal, conditional, eta_interaction, k_interaction, stability, representation)
    if len(figures) != 40:
        raise RuntimeError(f"expected 20 PNG+SVG figure pairs, found {len(figures)} files")
    manifest = {"round": "round_23", "audit_completed": True, "formal_policy_training_performed": False, "predictor_training_performed": False, "new_policy_seeds": 0, "frozen_formal_checkpoint_count": len(models), "validation_sequence_count": len(sequences), "fresh_dev_new_rollout_count": 0, "untouched_eval_access_count": 0, "head_causal_validation_records": len(causal), "figure_file_count": len(figures), "required_output_count": len(REQUIRED_OUTPUTS), "seed_scenario_agreement": agreement, "round24_created": False}
    _json(OUTPUT / "run_manifest.json", manifest)
    missing = [name for name in REQUIRED_OUTPUTS if not (OUTPUT / name).is_file()]
    if missing:
        raise RuntimeError(f"Round-23 required artifacts missing: {missing}")
    _finite_files()
    logs = ROUND / "logs"; logs.mkdir(parents=True, exist_ok=True)
    (logs / "implementation_log.md").write_text("# Round-23 implementation log\n\nExecuted one frozen-artifact audit: nine deployable Round-22 checkpoints were loaded for CPU inference; fresh-development data were read only from frozen raw output CSVs; validation-only head ablations used registered validation scenarios. No policy or predictor was updated, and no fresh-development or untouched-evaluation rollout was run.\n", encoding="utf-8")
    _completion_report(seed_summary, bootstrap, wins, tail, head_gate, stability_gate, temporal_gate, readiness, evidence)
    return {"positive_seed": positive_seed, "figures": figures, "bootstrap": bootstrap, "stability_gate": stability_gate, "head_gate": head_gate, "readiness": readiness, "access": access}


def main() -> None:
    result = run_audit()
    print(json.dumps({"round": "round_23", "positive_seed": result["positive_seed"], "figure_file_count": len(result["figures"]), "stability_gate": result["stability_gate"]["gate"], "readiness": result["readiness"]["gate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
