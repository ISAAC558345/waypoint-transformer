"""Post-training analyses for the fixed Round-24 seed extension."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from wt.round20.scenarios import development_scenarios
from wt.round22.closed_loop import _bridge_by_validation_scenario, _rollout
from wt.round22.data import FrozenDeepSetsPredictor, VALIDATION, load_policy_sequences, sha256
from wt.round22.evaluation import offline_metrics
from wt.round22.models import PREDICTED_MLP, PREDICTED_TRANSFORMER, STATE_ONLY_TRANSFORMER
from wt.round22.training import TrainingConfig, load_checkpoint, predict

from .fresh import fresh_stability_scenarios
from .runner import (ALL_SEEDS, EXPECTED_PREDICTOR, FAMILIES, NEW_SEEDS, OLD_SEEDS,
                     OUTPUT, ROOT, R22_OUTPUT, _load_json, _plain, _r22_runs, _csv, _json,
                     verify_frozen_state)


FIGURES = ROOT / "round_24" / "figures" / "round24_multiseed_stability_20260829_1515"
ROUND = ROOT / "round_24"
TIE = 1.0


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _f(value: Any, default: float = 0.0) -> float:
    return default if value in (None, "") else float(value)


def _b(value: Any) -> bool:
    return str(value).lower() == "true"


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _all_runs() -> list[dict[str, Any]]:
    old = [row for row in _r22_runs() if row["family"] in FAMILIES and int(row["seed"]) in OLD_SEEDS]
    new_manifest = _load_json(OUTPUT / "round24_new_seed_training_manifest.json")
    new = list(new_manifest.get("runs", []))
    expected = {(family, seed) for family in FAMILIES for seed in ALL_SEEDS}
    actual = {(str(row["family"]), int(row["seed"])) for row in old + new}
    if new_manifest.get("formal_training_completed") != "15/15" or len(new) != 15 or actual != expected:
        raise RuntimeError("Round-24 requires the complete 24 deployable checkpoint matrix")
    for row in old + new:
        if sha256(_path(str(row["checkpoint"]))) != row["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint changed: {row['run_id']}")
    return old + new


def _models(runs: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], tuple[Any, Any]]:
    config = TrainingConfig(); result: dict[tuple[str, int], tuple[Any, Any]] = {}
    for run in runs:
        family, model, normalizer, payload = load_checkpoint(_path(str(run["checkpoint"])), config.device)
        if family != run["family"] or int(payload.get("seed", -1)) != int(run["seed"]):
            raise RuntimeError("checkpoint identity does not match the registered matrix")
        result[(family, int(run["seed"]))] = (model, normalizer)
    return result


def _pair(metrics: Sequence[Mapping[str, Any]], left: str, right: str, population: str) -> list[dict[str, Any]]:
    index = {(str(row["scenario_id"]), int(row["model_seed"]), str(row["family"])): row for row in metrics}
    records: list[dict[str, Any]] = []
    for scenario, seed, family in sorted(index):
        if family != left: continue
        left_row, right_row = index[(scenario, seed, left)], index[(scenario, seed, right)]
        fields = ("cumulative_payload_bits", "completion", "residual_queue_bits", "episode_return",
                  "jain_fairness", "movement_energy", "safety_correction_rate", "minimum_uav_distance_m")
        records.append({"population": population, "scenario_id": scenario, "model_seed": seed,
                        "num_buoys": int(left_row["num_buoys"]), "eta": float(left_row["eta"]),
                        "interference_conflict": bool(left_row["interference_conflict"] or right_row["interference_conflict"]),
                        "bridge": left_row["bridge"], "left_family": left, "right_family": right,
                        **{f"delta_{field}": float(left_row[field]) - float(right_row[field]) for field in fields}})
    expected = len({str(row["scenario_id"]) for row in metrics}) * 8
    if len(records) != expected:
        raise RuntimeError(f"incomplete {population} paired comparison: {len(records)} != {expected}")
    return records


def _seed_summary(rows: Sequence[Mapping[str, Any]], *, population: str, comparison: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for seed in ALL_SEEDS:
        values = np.asarray([_f(row["delta_cumulative_payload_bits"]) for row in rows if int(row["model_seed"]) == seed], dtype=np.float64)
        if not len(values): raise RuntimeError("missing seed in paired summary")
        positive = values[values > TIE]
        result.append({"population": population, "comparison": comparison, "seed": seed, "scenario_count": len(values),
                       "delta_payload_mean_bits": float(values.mean()), "delta_payload_median_bits": float(np.median(values)),
                       "delta_payload_std_bits": float(values.std(ddof=1)), "positive_scenario_count": int((values > TIE).sum()),
                       "negative_scenario_count": int((values < -TIE).sum()), "tie_scenario_count": int((np.abs(values) <= TIE).sum()),
                       "win_rate": float((values > TIE).mean()), "positive_seed": bool(values.mean() > 0.0),
                       "top5_positive_contribution_share": float(np.sort(positive)[-5:].sum() / positive.sum()) if len(positive) else 0.0,
                       "mean_delta_completion": float(np.mean([_f(row["delta_completion"]) for row in rows if int(row["model_seed"]) == seed])),
                       "mean_delta_residual_queue_bits": float(np.mean([_f(row["delta_residual_queue_bits"]) for row in rows if int(row["model_seed"]) == seed])),
                       "mean_delta_safety_correction_rate": float(np.mean([_f(row["delta_safety_correction_rate"]) for row in rows if int(row["model_seed"]) == seed]))})
    return result


def _bootstrap(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    result = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(np.quantile(result, .025)), float(np.quantile(result, .975))


def _hierarchical(by_seed: Mapping[int, np.ndarray], draws: int, seed: int) -> tuple[float, float]:
    # Overall/K/eta groups have an equal scenario count per seed.  The frozen
    # validation conflict diagnostic can legitimately have unequal realised
    # group counts after the fixed environment rollout.  Resampling each
    # selected seed from its own registered rows is the same hierarchical
    # estimand and handles both cases without discarding observations.
    if any(len(by_seed[item]) == 0 for item in ALL_SEEDS):
        raise RuntimeError("conditional bootstrap has an empty training-seed group")
    rng = np.random.default_rng(seed); sampled = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        chosen = rng.integers(0, len(ALL_SEEDS), size=len(ALL_SEEDS))
        means = []
        for index in chosen:
            values = by_seed[ALL_SEEDS[int(index)]]
            means.append(float(values[rng.integers(0, len(values), size=len(values))].mean()))
        sampled[draw] = float(np.mean(means))
    return float(np.quantile(sampled, .025)), float(np.quantile(sampled, .975))


def uncertainty(rows: Sequence[Mapping[str, Any]], population: str, seed: int) -> list[dict[str, Any]]:
    by_seed = {model_seed: np.asarray([_f(row["delta_cumulative_payload_bits"]) for row in rows if int(row["model_seed"]) == model_seed], dtype=np.float64) for model_seed in ALL_SEEDS}
    if any(len(value) == 0 for value in by_seed.values()): raise RuntimeError("uncertainty missing a model seed")
    seed_means = np.asarray([by_seed[item].mean() for item in ALL_SEEDS]); scenario_means = np.stack([by_seed[item] for item in ALL_SEEDS]).mean(axis=0)
    scenario_low, scenario_high = _bootstrap(scenario_means, 2000, seed)
    cluster_low, cluster_high = _bootstrap(seed_means, 10000, seed + 1)
    hierarchical_low, hierarchical_high = _hierarchical(by_seed, 10000, seed + 2)
    result = [
        {"population": population, "method": "scenario_bootstrap", "excluded_seed": "", "mean_delta_payload_bits": float(scenario_means.mean()), "ci95_lower": scenario_low, "ci95_upper": scenario_high, "bootstrap_draws": 2000, "bootstrap_seed": seed, "unit": "scenario mean across eight seeds"},
        {"population": population, "method": "seed_cluster_bootstrap", "excluded_seed": "", "mean_delta_payload_bits": float(seed_means.mean()), "ci95_lower": cluster_low, "ci95_upper": cluster_high, "bootstrap_draws": 10000, "bootstrap_seed": seed + 1, "unit": "training-seed cluster"},
        {"population": population, "method": "hierarchical_seed_then_scenario_bootstrap", "excluded_seed": "", "mean_delta_payload_bits": float(seed_means.mean()), "ci95_lower": hierarchical_low, "ci95_upper": hierarchical_high, "bootstrap_draws": 10000, "bootstrap_seed": seed + 2, "unit": "resample seed then scenarios within seed"},
        {"population": population, "method": "median_of_seeds", "excluded_seed": "", "mean_delta_payload_bits": float(np.median(seed_means)), "ci95_lower": "", "ci95_upper": "", "bootstrap_draws": 0, "bootstrap_seed": 0, "unit": "eight seed means"},
    ]
    for excluded in ALL_SEEDS:
        retained = np.concatenate([by_seed[item] for item in ALL_SEEDS if item != excluded])
        low, high = _bootstrap(retained, 10000, seed + 10 + excluded)
        result.append({"population": population, "method": "leave_one_seed_out", "excluded_seed": excluded, "mean_delta_payload_bits": float(retained.mean()), "ci95_lower": low, "ci95_upper": high, "bootstrap_draws": 10000, "bootstrap_seed": seed + 10 + excluded, "unit": "retained seven-seed scenarios"})
    return result


def _offline(runs: Sequence[Mapping[str, Any]], models: Mapping[tuple[str, int], tuple[Any, Any]], predictor: FrozenDeepSetsPredictor) -> list[dict[str, Any]]:
    config = TrainingConfig(); sequences = load_policy_sequences(VALIDATION, predictor); result: list[dict[str, Any]] = []
    for run in runs:
        family, seed = str(run["family"]), int(run["seed"]); model, normalizer = models[(family, seed)]
        prediction = predict(family, model, normalizer.apply(sequences), config)
        rows, _ = offline_metrics(family, seed, prediction, sequences)
        result.extend(rows)
    if len(result) != len(sequences) * 24: raise RuntimeError("eight-seed offline validation coverage failed")
    _csv(OUTPUT / "eight_seed_offline_validation.csv", result)
    return result


def _head_utilization(models: Mapping[tuple[str, int], tuple[Any, Any]], predictor: FrozenDeepSetsPredictor) -> list[dict[str, Any]]:
    """The small requested validation-only sensitivity check, no causal rollout."""
    config = TrainingConfig(); sequences = load_policy_sequences(VALIDATION, predictor); result: list[dict[str, Any]] = []
    for seed in NEW_SEEDS:
        model, normalizer = models[(PREDICTED_TRANSFORMER, seed)]
        normal = predict(PREDICTED_TRANSFORMER, model, normalizer.apply(sequences), config); valid = sequences.uav_mask[:, -1]
        for head, subset in (("service", slice(0, 5)), ("progress", slice(5, 10)), ("return", slice(10, 11))):
            phi = sequences.predicted_phi.copy(); phi[..., subset] = 0.0
            varied = predict(PREDICTED_TRANSFORMER, model, normalizer.apply(sequences.replace(predicted_phi=phi)), config)
            sensitivity = float((np.linalg.norm(varied - normal, axis=-1) * valid).sum() / valid.sum())
            result.append({"seed": seed, "head": head, "condition": f"zero_{head}", "action_sensitivity": sensitivity, "source": "round24_new_validation_inference", "uses_true_phi": False, "retraining_performed": False})
    old = [row for row in _rows(ROOT / "round_23" / "outputs" / "round23_seed_stability_audit_20260829_1436" / "seed_head_utilization.csv") if row["condition"] in {"zero_service", "zero_progress", "zero_return"}]
    for row in old:
        result.append({"seed": int(row["seed"]), "head": row["head"], "condition": row["condition"], "action_sensitivity": _f(row["action_sensitivity"]), "source": "round23_frozen_validation_inference", "uses_true_phi": False, "retraining_performed": False})
    if len(result) != 24: raise RuntimeError("eight-seed head sensitivity coverage failed")
    return sorted(result, key=lambda row: (int(row["seed"]), str(row["head"])))


def _validation_metrics(runs: Sequence[Mapping[str, Any]], predictor: FrozenDeepSetsPredictor) -> list[dict[str, Any]]:
    """Reuse frozen old raw rows; run exactly the 5x3 registered new rows."""
    source = _rows(R22_OUTPUT / "validation_closed_loop_metrics.csv")
    old = [{key: (_f(value) if key not in {"population", "scenario_id", "scenario_sha256", "family", "bridge", "interference_tier", "queue_tier", "geometry_variant"} and value not in {"", "True", "False"} else value) for key, value in row.items()} for row in source if row["family"] in FAMILIES and int(row["model_seed"]) in OLD_SEEDS]
    # Preserve booleans needed by pairing and group reporting.
    for row in old:
        for key in ("interference_conflict", "uses_true_phi"):
            row[key] = _b(row[key])
        for key in ("model_seed", "scenario_seed", "num_buoys", "episode_length"):
            row[key] = int(float(row[key]))
    bridge = _bridge_by_validation_scenario(predictor); scenarios = tuple(row for row in development_scenarios() if row.split == VALIDATION)
    new: list[dict[str, Any]] = []
    for run in runs:
        if int(run["seed"]) not in NEW_SEEDS: continue
        for scenario in scenarios:
            metric, _ = _rollout(scenario, str(run["family"]), int(run["seed"]), _path(str(run["checkpoint"])), predictor, population="validation", bridge=bridge.get(str(scenario.scenario_id), False))
            new.append(metric)
    metrics = old + new
    if len(metrics) != 16 * 24: raise RuntimeError("eight-seed validation rollout coverage failed")
    return metrics


def _fresh_metrics(runs: Sequence[Mapping[str, Any]], predictor: FrozenDeepSetsPredictor) -> list[dict[str, Any]]:
    raw_path = OUTPUT / "round24_fresh_raw_metrics.csv"
    if raw_path.is_file():
        rows = _rows(raw_path)
        if len(rows) != 80 * 24: raise RuntimeError("existing fresh output has invalid coverage")
        for row in rows:
            for key in ("model_seed", "scenario_seed", "num_buoys", "episode_length"):
                row[key] = int(float(row[key]))
            for key in ("interference_conflict", "uses_true_phi"):
                row[key] = _b(row[key])
        return rows
    train = _load_json(OUTPUT / "round24_new_seed_training_manifest.json")
    frozen_manifest = OUTPUT / "round24_fresh_manifest.json"
    if sha256(frozen_manifest) != train["fresh_manifest_sha256_pretraining"]:
        raise RuntimeError("fresh manifest changed after training began")
    frozen_validation = {name: sha256(OUTPUT / name) for name in ("eight_seed_offline_validation.csv", "eight_seed_validation_closed_loop_metrics.csv", "eight_seed_validation_paired.csv", "eight_seed_validation_stability.csv", "eight_seed_validation_uncertainty_pre_fresh.csv")}
    _json(OUTPUT / "round24_pre_fresh_freeze.json", {"all_24_deployable_checkpoints_frozen": True,
        "new_formal_training_completed": "15/15", "predictor_sha256": predictor.sha256,
        "fresh_manifest_sha256": sha256(frozen_manifest), "primary_metric": "actual_cumulative_payload_bits",
        "subgroups_frozen": ["K", "eta", "validation conflict", "validation bridge"],
        "validation_analysis_hashes": frozen_validation,
        "analysis_source_sha256": sha256(ROOT / "src" / "wt" / "round24" / "analysis.py"),
        "fresh_access_permitted_once": True})
    result: list[dict[str, Any]] = []
    for scenario in fresh_stability_scenarios():
        for run in runs:
            metric, _ = _rollout(scenario, str(run["family"]), int(run["seed"]), _path(str(run["checkpoint"])), predictor, population="round24_fresh_stability", bridge=None)
            result.append(metric)
    if len(result) != 80 * 24: raise RuntimeError("Round-24 fresh rollout coverage failed")
    _csv(raw_path, result)
    return result


def _conditional(rows: Sequence[Mapping[str, Any]], population: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for field in ("interference_conflict", "bridge"):
        if population == "round24_fresh_stability" and field == "bridge":
            result.append({"population": population, "condition": "bridge", "condition_value": "not_available_without_future_labels", "available": False, "seed": "", "scenario_count": 0, "delta_payload_mean_bits": "", "hierarchical_ci95_lower": "", "hierarchical_ci95_upper": "", "frozen_definition": True})
            continue
        for value in sorted({str(row[field]) for row in rows}):
            grouped = [row for row in rows if str(row[field]) == value]
            summaries = _seed_summary(grouped, population=population, comparison="task_waypoint_minus_state")
            by_seed = {int(row["seed"]): np.asarray([_f(item["delta_cumulative_payload_bits"]) for item in grouped if int(item["model_seed"]) == int(row["seed"])]) for row in summaries}
            low, high = _hierarchical(by_seed, 10000, 24100 + len(result))
            for item in summaries:
                result.append({"population": population, "condition": field, "condition_value": value, "available": True, "seed": item["seed"], "scenario_count": item["scenario_count"], "delta_payload_mean_bits": item["delta_payload_mean_bits"], "positive_seed": item["positive_seed"], "hierarchical_ci95_lower": low, "hierarchical_ci95_upper": high, "frozen_definition": True})
    return result


def _interaction(rows: Sequence[Mapping[str, Any]], field: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for value in sorted({str(row[field]) for row in rows}):
        for seed in ALL_SEEDS:
            values = np.asarray([_f(row["delta_cumulative_payload_bits"]) for row in rows if str(row[field]) == value and int(row["model_seed"]) == seed])
            result.append({"group_field": field, "group_value": value, "seed": seed, "scenario_count": len(values), "delta_payload_mean_bits": float(values.mean()), "delta_payload_median_bits": float(np.median(values)), "positive": bool(values.mean() > 0.0)})
    return result


def _variance(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for family in FAMILIES:
        seed_means = np.asarray([np.mean([_f(row["cumulative_payload_bits"]) for row in metrics if row["family"] == family and int(row["model_seed"]) == seed]) for seed in ALL_SEEDS])
        result.append({"family": family, "seed_count": 8, "fresh_payload_seed_mean": float(seed_means.mean()), "fresh_payload_seed_variance": float(seed_means.var(ddof=1)), "fresh_payload_seed_std": float(seed_means.std(ddof=1)), "initialization_sensitive_vs_state_only": bool(family == PREDICTED_TRANSFORMER and seed_means.var(ddof=1) > 0)})
    task = next(row for row in result if row["family"] == PREDICTED_TRANSFORMER)
    state = next(row for row in result if row["family"] == STATE_ONLY_TRANSFORMER)
    task["initialization_sensitive_vs_state_only"] = bool(task["fresh_payload_seed_variance"] > state["fresh_payload_seed_variance"])
    return result


def _correlation(left: Sequence[float], right: Sequence[float], rank: bool = False) -> float:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if rank:
        a = np.argsort(np.argsort(a, kind="stable"), kind="stable"); b = np.argsort(np.argsort(b, kind="stable"), kind="stable")
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and a.std() > 1e-12 and b.std() > 1e-12 else float("nan")


def _gate(fresh_summary: Sequence[Mapping[str, Any]], uncertainty_rows: Sequence[Mapping[str, Any]], fresh_pairs: Sequence[Mapping[str, Any]], conditional: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    hierarchy = next(row for row in uncertainty_rows if row["population"] == "round24_fresh_stability" and row["method"] == "hierarchical_seed_then_scenario_bootstrap")
    leave = [row for row in uncertainty_rows if row["population"] == "round24_fresh_stability" and row["method"] == "leave_one_seed_out"]
    seed_means = np.asarray([_f(row["delta_payload_mean_bits"]) for row in fresh_summary]); positive = int((seed_means > 0).sum())
    tail_dominated = any(_f(row["top5_positive_contribution_share"]) >= .5 for row in fresh_summary if _f(row["delta_payload_mean_bits"]) > 0)
    safety = {field: float(np.mean([_f(row[field]) for row in fresh_pairs])) for field in ("delta_completion", "delta_residual_queue_bits", "delta_safety_correction_rate")}
    no_harm = safety["delta_completion"] >= 0 and safety["delta_residual_queue_bits"] <= 0 and safety["delta_safety_correction_rate"] <= 0
    loo_stable = all(_f(row["mean_delta_payload_bits"]) >= 0.0 for row in leave)
    robust = float(np.median(seed_means)) > 0 and _f(hierarchy["ci95_lower"]) > 0 and positive >= 6 and loo_stable and not tail_dominated and no_harm
    no_advantage = float(np.median(seed_means)) <= 0 or positive <= 3 or _f(hierarchy["ci95_upper"]) <= 0 or tail_dominated
    gate = "MS-A" if robust else "MS-C" if no_advantage else "MS-B"
    available = [row for row in conditional if row["available"] and row["population"] == "validation"]
    conditional_ok = gate == "MS-B" and any(sum(_f(item["delta_payload_mean_bits"]) > 0 for item in available if item["condition"] == cond and item["condition_value"] == value) >= 6 and _f(next(item["hierarchical_ci95_lower"] for item in available if item["condition"] == cond and item["condition_value"] == value)) > 0 for cond, value in {(str(item["condition"]), str(item["condition_value"])) for item in available})
    stability = {"round": "round_24", "gate": gate, "multi_seed_task_waypoint_advantage": gate == "MS-A", "candidate_ready_for_untouched_eval": gate == "MS-A", "fresh_seed_mean_delta_payload_bits": float(seed_means.mean()), "fresh_seed_median_delta_payload_bits": float(np.median(seed_means)), "positive_seed_count": positive, "negative_seed_count": 8 - positive, "hierarchical_ci95": [_f(hierarchy["ci95_lower"]), _f(hierarchy["ci95_upper"])], "leave_one_seed_out_stable": loo_stable, "tail_dominated": tail_dominated, "completion_residual_safety_not_systematically_degraded": no_harm, "criteria": {"median_positive": float(np.median(seed_means)) > 0, "hierarchical_lower_positive": _f(hierarchy["ci95_lower"]) > 0, "positive_seeds_at_least_six": positive >= 6, "loo_not_negative": loo_stable, "not_tail_dominated": not tail_dominated, "no_systematic_harm": no_harm}}
    readiness = {"round": "round_24", "gate": "R24-A" if gate == "MS-A" else "R24-B" if conditional_ok else "R24-C", "conditional_waypoint_value_confirmed": conditional_ok, "candidate_ready_for_untouched_eval": gate == "MS-A", "next_step": "next_round_untouched_evaluation_only" if gate == "MS-A" else "coupling_aware_task_waypoint_training_objective_study" if conditional_ok else "policy_training_stability_redesign_or_stop_overall_claim", "more_training_seeds_recommended": False, "round25_created": False}
    return stability, readiness


def _figures(validation: Sequence[Mapping[str, Any]], fresh: Sequence[Mapping[str, Any]], fresh_summary: Sequence[Mapping[str, Any]], uncertainty_rows: Sequence[Mapping[str, Any]], conditional: Sequence[Mapping[str, Any]], eta: Sequence[Mapping[str, Any]], counts: Sequence[Mapping[str, Any]], temporal: Sequence[Mapping[str, Any]], variance: Sequence[Mapping[str, Any]], alignment: Sequence[Mapping[str, Any]], heads: Sequence[Mapping[str, Any]]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    def save(fig: Any, name: str) -> None:
        fig.tight_layout(); fig.savefig(FIGURES / f"{name}.png", dpi=150); fig.savefig(FIGURES / f"{name}.svg"); plt.close(fig)
    def bar(name: str, labels: Sequence[Any], values: Sequence[float], title: str) -> None:
        fig, axis = plt.subplots(figsize=(7, 4)); axis.bar([str(x) for x in labels], values); axis.axhline(0, color="black", lw=.8); axis.set(title=title, ylabel="payload delta (bits)"); axis.grid(axis="y", alpha=.25); save(fig, name)
    validation_summary = _seed_summary(validation, population="validation", comparison="task_minus_state")
    bar("01_validation_deltas", ALL_SEEDS, [_f(row["delta_payload_mean_bits"]) for row in validation_summary], "8-seed validation deltas")
    bar("02_fresh_deltas", ALL_SEEDS, [_f(row["delta_payload_mean_bits"]) for row in fresh_summary], "8-seed fresh deltas")
    bar("03_fresh_median_mean", ["median", "mean"], [float(np.median([_f(row["delta_payload_mean_bits"]) for row in fresh_summary])), float(np.mean([_f(row["delta_payload_mean_bits"]) for row in fresh_summary]))], "Fresh seed median and mean")
    boot = [row for row in uncertainty_rows if row["population"] == "round24_fresh_stability" and row["method"] in {"hierarchical_seed_then_scenario_bootstrap", "seed_cluster_bootstrap"}]
    for index, (name, title) in enumerate((("04_hierarchical_bootstrap", "Hierarchical bootstrap"), ("05_seed_cluster_bootstrap", "Seed-cluster bootstrap"))):
        row = boot[index]; fig, axis = plt.subplots(figsize=(5, 4)); mean = _f(row["mean_delta_payload_bits"]); axis.errorbar([0], [mean], yerr=[[mean - _f(row["ci95_lower"])], [_f(row["ci95_upper"]) - mean]], fmt="o"); axis.set(xlim=(-1, 1), xticks=[], title=title, ylabel="payload delta (bits)"); axis.axhline(0, color="black", lw=.8); save(fig, name)
    leave = [row for row in uncertainty_rows if row["population"] == "round24_fresh_stability" and row["method"] == "leave_one_seed_out"]
    bar("06_leave_one_seed_out", [f"-{row['excluded_seed']}" for row in leave], [_f(row["mean_delta_payload_bits"]) for row in leave], "Leave-one-seed-out fresh deltas")
    scenarios = sorted({str(row["scenario_id"]) for row in fresh}); matrix = np.asarray([[next(_f(row["delta_cumulative_payload_bits"]) for row in fresh if str(row["scenario_id"]) == scenario and int(row["model_seed"]) == seed) for seed in ALL_SEEDS] for scenario in scenarios]); fig, axis = plt.subplots(figsize=(8, 5)); image = axis.imshow(matrix, aspect="auto", cmap="coolwarm"); axis.set(title="8-seed fresh scenario delta matrix", xlabel="training seed", ylabel="fresh scenario", xticks=range(8), xticklabels=ALL_SEEDS); fig.colorbar(image, ax=axis); save(fig, "07_fresh_heatmap")
    bar("08_win_loss_by_seed", ALL_SEEDS, [row["win_rate"] for row in fresh_summary], "Fresh win rate by seed")
    bar("09_tail_contribution", ALL_SEEDS, [row["top5_positive_contribution_share"] for row in fresh_summary], "Top-five positive contribution share")
    for name, data, title in (("10_K_stability", counts, "K stability"), ("11_eta_stability", [row for row in eta if row["group_value"] in {"0.25", "0.5"}], "eta .25/.5 stability")):
        groups = sorted({str(row["group_value"]) for row in data}); fig, axis = plt.subplots(figsize=(8, 4)); width = .09
        for i, seed in enumerate(ALL_SEEDS): axis.bar(np.arange(len(groups)) + (i - 3.5) * width, [next(_f(row["delta_payload_mean_bits"]) for row in data if str(row["group_value"]) == group and int(row["seed"]) == seed) for group in groups], width=width, label=str(seed))
        axis.set(title=title, xticks=range(len(groups)), xticklabels=groups, ylabel="payload delta (bits)"); axis.axhline(0, color="black", lw=.8); axis.legend(ncol=4); save(fig, name)
    for name, condition, title in (("12_conflict_stability", "interference_conflict", "Conflict stability"), ("13_bridge_stability", "bridge", "Bridge stability")):
        data = [row for row in conditional if row["population"] == "validation" and row["condition"] == condition and row["available"] and str(row["condition_value"]) == "True"]; bar(name, [row["seed"] for row in data], [_f(row["delta_payload_mean_bits"]) for row in data], title)
    bar("14_transformer_vs_mlp", [row["seed"] for row in temporal if row.get("record_type") == "seed"], [_f(row["delta_payload_mean_bits"]) for row in temporal if row.get("record_type") == "seed"], "Transformer minus MLP")
    bar("15_seed_variance", [row["family"].replace("predicted_task_waypoint_", "") for row in variance], [_f(row["fresh_payload_seed_variance"]) for row in variance], "Fresh seed variance")
    points = [row for row in alignment if row.get("record_type") == "seed"]; fig, axis = plt.subplots(figsize=(6, 4)); axis.scatter([_f(row["offline_raw_action_rmse"]) for row in points], [_f(row["fresh_delta_payload_bits"]) for row in points]); [axis.annotate(str(row["seed"]), (_f(row["offline_raw_action_rmse"]), _f(row["fresh_delta_payload_bits"]))) for row in points]; axis.set(title="Offline RMSE vs fresh payload", xlabel="offline RMSE", ylabel="fresh delta (bits)"); save(fig, "16_offline_rmse_vs_payload")
    for_head = {head: [row for row in heads if row["head"] == head] for head in ("service", "progress", "return")}; fig, axis = plt.subplots(figsize=(8, 4)); width=.22
    for i, head in enumerate(for_head): axis.bar(np.arange(8)+(i-1)*width, [_f(row["action_sensitivity"]) for row in for_head[head]], width=width, label=head)
    axis.set(title="Head sensitivity across 8 seeds", xticks=range(8), xticklabels=ALL_SEEDS, ylabel="action sensitivity"); axis.legend(); save(fig, "17_head_sensitivity")
    result = sorted(path.name for path in FIGURES.iterdir() if path.suffix in {".png", ".svg"})
    if len(result) != 34: raise RuntimeError("Round-24 requires 17 PNG+SVG figure pairs")
    return result


def _completion_report(validation: Sequence[Mapping[str, Any]], fresh: Sequence[Mapping[str, Any]], uncertainty_rows: Sequence[Mapping[str, Any]], conditional: Sequence[Mapping[str, Any]], eta: Sequence[Mapping[str, Any]], counts: Sequence[Mapping[str, Any]], temporal: Sequence[Mapping[str, Any]], variance: Sequence[Mapping[str, Any]], heads: Sequence[Mapping[str, Any]], alignment: Sequence[Mapping[str, Any]], stability: Mapping[str, Any], readiness: Mapping[str, Any], access: Mapping[str, Any]) -> None:
    validation_values = ", ".join(f"{int(row['seed'])}: {_f(row['delta_payload_mean_bits']):.2f}" for row in validation)
    fresh_values = ", ".join(f"{int(row['seed'])}: {_f(row['delta_payload_mean_bits']):.2f}" for row in fresh)
    hierarchy = next(row for row in uncertainty_rows if row["population"] == "round24_fresh_stability" and row["method"] == "hierarchical_seed_then_scenario_bootstrap")
    cluster = next(row for row in uncertainty_rows if row["population"] == "round24_fresh_stability" and row["method"] == "seed_cluster_bootstrap")
    validation_hierarchy = next(row for row in uncertainty_rows if row["population"] == "validation" and row["method"] == "hierarchical_seed_then_scenario_bootstrap")
    loo = [row for row in uncertainty_rows if row["population"] == "round24_fresh_stability" and row["method"] == "leave_one_seed_out"]
    def grouped(data: Sequence[Mapping[str, Any]], field: str, value: str) -> str:
        entries = [row for row in data if str(row.get("group_field", row.get("condition", ""))) == field and str(row.get("group_value", row.get("condition_value", ""))) == value]
        return "; ".join(f"s{row['seed']}={_f(row['delta_payload_mean_bits']):.2f}" for row in entries)
    def validation_conditional(condition: str) -> str:
        entries = [row for row in conditional if row["population"] == "validation" and row["condition"] == condition and str(row["condition_value"]) == "True"]
        return "; ".join(f"s{row['seed']}={_f(row['delta_payload_mean_bits']):.2f}" for row in entries)
    conflict = validation_conditional("interference_conflict")
    bridge = validation_conditional("bridge")
    eta25, eta50 = grouped(eta, "eta", "0.25"), grouped(eta, "eta", "0.5")
    k3, k5 = grouped(counts, "num_buoys", "3"), grouped(counts, "num_buoys", "5")
    temporal_seed = [row for row in temporal if row.get("record_type") == "seed"]
    temporal_gate = next(row["gate"] for row in temporal if row.get("record_type") == "uncertainty")
    dominance = {seed: max((row for row in heads if int(row["seed"]) == seed), key=lambda row: _f(row["action_sensitivity"]))["head"] for seed in ALL_SEEDS}
    corr = next(row for row in alignment if row.get("record_type") == "correlation")
    loo_text = "; ".join("remove {}: {:.2f}".format(row["excluded_seed"], _f(row["mean_delta_payload_bits"])) for row in loo)
    temporal_text = ", ".join("{}: {:.2f}".format(row["seed"], _f(row["delta_payload_mean_bits"])) for row in temporal_seed)
    report = ROUND / "report" / "round24_completion_report.md"; report.parent.mkdir(exist_ok=True)
    report.write_text(
        "# Round-24 Completion Report\n\n"
        "Round-24 was a pre-registered pure seed extension. No algorithm, predictor, data split, environment, sampler, loss, architecture, optimizer, or learning-rate field changed.\n\n"
        "## Required answers\n\n"
        "1. New training seeds: 3, 4, 5, 6, 7 for each of the three frozen deployable families.\n"
        "2. New formal runs: 15/15 completed.\n"
        f"3. Eight validation seed payload deltas (bits): {validation_values}.\n"
        f"4. Validation positive-seed count: {sum(_f(row['delta_payload_mean_bits']) > 0 for row in validation)}.\n"
        f"5. Validation median seed delta: {float(np.median([_f(row['delta_payload_mean_bits']) for row in validation])):.2f} bits.\n"
        f"6. Validation hierarchical 95% CI: [{_f(validation_hierarchy['ci95_lower']):.2f}, {_f(validation_hierarchy['ci95_upper']):.2f}] bits.\n"
        f"7. Eight fresh seed payload deltas (bits): {fresh_values}.\n"
        f"8. Fresh positive-seed count: {stability['positive_seed_count']}.\n"
        f"9. Fresh median seed delta: {stability['fresh_seed_median_delta_payload_bits']:.2f} bits.\n"
        f"10. Fresh mean seed delta: {stability['fresh_seed_mean_delta_payload_bits']:.2f} bits.\n"
        f"11. Fresh hierarchical 95% CI: [{_f(hierarchy['ci95_lower']):.2f}, {_f(hierarchy['ci95_upper']):.2f}] bits.\n"
        f"12. Fresh seed-cluster 95% CI: [{_f(cluster['ci95_lower']):.2f}, {_f(cluster['ci95_upper']):.2f}] bits.\n"
        f"13. Leave-one-seed-out stable: {str(stability['leave_one_seed_out_stable']).lower()} ({loo_text}).\n"
        f"14. Single-seed/tail domination: {str(stability['tail_dominated']).lower()}.\n"
        f"15. Conflict eight-seed validation result: {conflict or 'not available'}.\n"
        f"16. Bridge eight-seed validation result: {bridge or 'not available'}; fresh bridge is unavailable without future labels.\n"
        f"17. Eta=.25 fresh result: {eta25}.\n"
        f"18. Eta=.5 fresh result: {eta50}.\n"
        f"19. K=3 fresh result: {k3}.\n"
        f"20. K=5 fresh result: {k5}.\n"
        f"21. Task-WP Transformer fresh seed variance: {_f(next(row['fresh_payload_seed_variance'] for row in variance if row['family'] == PREDICTED_TRANSFORMER)):.2f}.\n"
        f"22. State-only Transformer fresh seed variance: {_f(next(row['fresh_payload_seed_variance'] for row in variance if row['family'] == STATE_ONLY_TRANSFORMER)):.2f}.\n"
        f"23. Task-WP MLP fresh seed variance: {_f(next(row['fresh_payload_seed_variance'] for row in variance if row['family'] == PREDICTED_MLP)):.2f}.\n"
        f"24. Transformer vs MLP: {temporal_gate}; per-seed deltas: {temporal_text}.\n"
        f"25. Return dominance persists: {str(sum(value == 'return' for value in dominance.values()) >= 5).lower()} (dominant head by seed: {dominance}).\n"
        f"26. Offline RMSE-to-fresh-payload correlation (descriptive only): Pearson={_f(corr['pearson']):.3f}, Spearman={_f(corr['spearman']):.3f}.\n"
        f"27. MS Gate: {stability['gate']}.\n"
        f"28. conditional_waypoint_value_confirmed: {str(readiness['conditional_waypoint_value_confirmed']).lower()}.\n"
        f"29. candidate_ready_for_untouched_eval: {str(readiness['candidate_ready_for_untouched_eval']).lower()}.\n"
        f"30. More training seeds recommended: {str(readiness['more_training_seeds_recommended']).lower()}; no seed 8+ was trained.\n"
        f"31. Round-22 fresh new rollout count: {access['round22_fresh_new_rollout_count']}.\n"
        f"32. Round-20 development-test reuse count: {access['round20_development_test_reuse_count']}.\n"
        f"33. Untouched-evaluation access count: {access['untouched_eval_access_count']}.\n"
        f"34. Round-24 fresh access count: {access['round24_fresh_access_count']}.\n"
        "35. Round-25 created: false.\n\n"
        "Fresh bridge conditional analysis is explicitly marked unavailable because deriving that label would require future labels; validation bridge uses the frozen Round-20 definition.\n",
        encoding="utf-8")


def run_analysis() -> dict[str, Any]:
    frozen = verify_frozen_state(); runs = _all_runs(); predictor = FrozenDeepSetsPredictor()
    if predictor.sha256 != EXPECTED_PREDICTOR: raise RuntimeError("predictor changed before Round-24 post-training analysis")
    models = _models(runs); offline = _offline(runs, models, predictor); heads = _head_utilization(models, predictor)
    validation_metrics = _validation_metrics(runs, predictor); validation = _pair(validation_metrics, PREDICTED_TRANSFORMER, STATE_ONLY_TRANSFORMER, "validation")
    _csv(OUTPUT / "eight_seed_validation_closed_loop_metrics.csv", validation_metrics); _csv(OUTPUT / "eight_seed_validation_paired.csv", validation)
    validation_summary = _seed_summary(validation, population="validation", comparison="task_waypoint_transformer_minus_state_only")
    validation_uncertainty = uncertainty(validation, "validation", 24024)
    _csv(OUTPUT / "eight_seed_validation_stability.csv", validation_summary)
    _csv(OUTPUT / "eight_seed_validation_uncertainty_pre_fresh.csv", validation_uncertainty)
    fresh_access_before = (OUTPUT / "round24_fresh_raw_metrics.csv").is_file(); fresh_metrics = _fresh_metrics(runs, predictor); fresh = _pair(fresh_metrics, PREDICTED_TRANSFORMER, STATE_ONLY_TRANSFORMER, "round24_fresh_stability")
    _csv(OUTPUT / "round24_fresh_seed_summary.csv", _seed_summary(fresh, population="round24_fresh_stability", comparison="task_waypoint_transformer_minus_state_only"))
    fresh_summary = _seed_summary(fresh, population="round24_fresh_stability", comparison="task_waypoint_transformer_minus_state_only")
    uncertainties = validation_uncertainty + uncertainty(fresh, "round24_fresh_stability", 24044)
    _csv(OUTPUT / "eight_seed_uncertainty_audit.csv", uncertainties); _csv(OUTPUT / "round24_fresh_uncertainty.csv", [row for row in uncertainties if row["population"] == "round24_fresh_stability"])
    conditional = _conditional(validation, "validation") + _conditional(fresh, "round24_fresh_stability"); _csv(OUTPUT / "eight_seed_conditional_value.csv", conditional)
    eta, counts = _interaction(fresh, "eta"), _interaction(fresh, "num_buoys"); _csv(OUTPUT / "eight_seed_eta_stability.csv", eta); _csv(OUTPUT / "eight_seed_K_stability.csv", counts)
    temporal_pairs = _pair(fresh_metrics, PREDICTED_TRANSFORMER, PREDICTED_MLP, "round24_fresh_stability")
    temporal_summary = _seed_summary(temporal_pairs, population="round24_fresh_stability", comparison="task_waypoint_transformer_minus_mlp")
    temporal_u = uncertainty(temporal_pairs, "temporal", 24124); temporal_h = next(row for row in temporal_u if row["method"] == "hierarchical_seed_then_scenario_bootstrap")
    temporal_gate = "TV8-A" if sum(_f(row["delta_payload_mean_bits"]) > 0 for row in temporal_summary) >= 6 and _f(temporal_h["ci95_lower"]) > 0 else "TV8-B" if float(np.mean([_f(row["delta_payload_mean_bits"]) for row in temporal_summary])) > 0 else "TV8-C"
    temporal = [{"record_type": "seed", **row} for row in temporal_summary] + [{"record_type": "uncertainty", "gate": temporal_gate, **row} for row in temporal_u]; _csv(OUTPUT / "eight_seed_temporal_value.csv", temporal)
    variance = _variance(fresh_metrics); _csv(OUTPUT / "eight_seed_variance_comparison.csv", variance); _csv(OUTPUT / "eight_seed_head_utilization.csv", heads)
    rmse = {seed: float(np.mean([_f(row["raw_action_rmse"]) for row in offline if row["family"] == PREDICTED_TRANSFORMER and int(row["seed"]) == seed])) for seed in ALL_SEEDS}
    fresh_delta = {int(row["seed"]): _f(row["delta_payload_mean_bits"]) for row in fresh_summary}
    alignment = [{"record_type": "seed", "seed": seed, "offline_raw_action_rmse": rmse[seed], "fresh_delta_payload_bits": fresh_delta[seed], "interpretation": "descriptive_only_n_equals_8"} for seed in ALL_SEEDS] + [{"record_type": "correlation", "n": 8, "pearson": _correlation([rmse[s] for s in ALL_SEEDS], [fresh_delta[s] for s in ALL_SEEDS]), "spearman": _correlation([rmse[s] for s in ALL_SEEDS], [fresh_delta[s] for s in ALL_SEEDS], True), "interpretation": "descriptive_not_causal"}]
    _csv(OUTPUT / "eight_seed_offline_closed_loop_alignment.csv", alignment)
    stability, readiness = _gate(fresh_summary, uncertainties, fresh, conditional); _json(OUTPUT / "round24_seed_stability_gate.json", stability); _json(OUTPUT / "round24_readiness_gate.json", readiness)
    figures = _figures(validation, fresh, fresh_summary, uncertainties, conditional, eta, counts, temporal, variance, alignment, heads)
    access = {"round": "round_24", "round22_fresh_new_rollout_count": 0, "round20_development_test_reuse_count": 0, "untouched_eval_access_count": 0, "untouched_eval_model_access_count": 0, "untouched_eval_policy_rollout_count": 0, "round24_fresh_access_count": 1, "round24_fresh_raw_outcomes_read": True, "round24_fresh_already_accessed_before_this_stage": fresh_access_before, "predictor_sha256_before": EXPECTED_PREDICTOR, "predictor_sha256_after": predictor.sha256, "pre_analysis_frozen_state": frozen}
    _json(OUTPUT / "access_audit.json", access)
    manifest = {"round": "round_24", "formal_training_completed": "15/15", "all_deployable_checkpoint_count": 24, "old_seeds_retrained": [], "round24_fresh_access_count": 1, "fresh_scenario_count": 80, "fresh_episode_count": len(fresh_metrics), "validation_episode_count": len(validation_metrics), "figure_file_count": len(figures), "predictor_training_performed": False, "hgnn_training_performed": False, "ppo_training_performed": False, "round25_created": False, "stability_gate": stability["gate"], "readiness_gate": readiness["gate"]}
    _json(OUTPUT / "run_manifest.json", manifest)
    (ROUND / "logs").mkdir(exist_ok=True); (ROUND / "logs" / "implementation_log.md").write_text("# Round-24 implementation log\n\nPre-registered 80 fresh configurations before the fixed 15-run seed extension. Old 0--2 checkpoints were read-only. All Round-22 protocol fields were audited equal. After all 15 new checkpoints and validation analyses were frozen, the new fresh stability set was evaluated exactly once. No protected Round-22 fresh, Round-20 development-test, or untouched-evaluation rollout was made.\n", encoding="utf-8")
    _completion_report(validation_summary, fresh_summary, uncertainties, conditional, eta, counts, temporal, variance, heads, alignment, stability, readiness, access)
    return {"stability": stability, "readiness": readiness, "figures": figures, "access": access}


def main() -> None:
    result = run_analysis(); print(json.dumps({"gate": result["stability"]["gate"], "readiness": result["readiness"]["gate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
