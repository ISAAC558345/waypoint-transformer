"""Four fixed family smoke trainings; deliberately distinct from formal runs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .cache import materialize_predicted_task_waypoint_cache
from .data import R22_OUTPUT, FrozenDeepSetsPredictor, load_policy_sequences
from .evaluation import offline_metrics
from .models import POLICY_FAMILIES
from .training import TrainingConfig, load_checkpoint, predict, train_one


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def run_smokes(output: Path = R22_OUTPUT) -> dict[str, Any]:
    predictor = FrozenDeepSetsPredictor()
    before = predictor.sha256
    cache_manifest = output / "predicted_task_waypoint_cache_manifest.json"
    cache = json.loads(cache_manifest.read_text(encoding="utf-8")) if cache_manifest.exists() else materialize_predicted_task_waypoint_cache(predictor, output)
    if cache["predictor_sha256"] != before:
        raise RuntimeError("predicted Task-Waypoint cache provenance is not frozen predictor provenance")
    train, validation = load_policy_sequences("train", predictor), load_policy_sequences("validation", predictor)
    config = TrainingConfig(max_epochs=2, early_stopping_patience=2)
    rows: list[dict[str, Any]] = []; offline: list[dict[str, Any]] = []
    for family in POLICY_FAMILIES:
        path = output / "smoke" / family / "seed_0" / "best.pt"
        model, normalizer, history, record = train_one(family, 0, train, validation, config, path)
        normalized = normalizer.apply(validation)
        reference = predict(family, model, normalized, config)
        loaded_family, loaded, loaded_normalizer, payload = load_checkpoint(path)
        reloaded = predict(loaded_family, loaded, loaded_normalizer.apply(validation), config)
        metric_rows, _ = offline_metrics(family, 0, reference, validation)
        summary = {metric: float(np.mean([row[metric] for row in metric_rows])) for metric in ("normalized_mse", "raw_action_rmse", "mae", "cosine", "angular_error_deg", "magnitude_error", "projection_rate")}
        checks = {"forward_finite": bool(np.isfinite(reference).all()), "loss_finite": bool(np.isfinite([row["train_action_mse"] for row in history]).all()), "gradient_finite": bool(np.isfinite([row["gradient_norm_preclip"] for row in history]).all()), "joint_action_shape": tuple(reference.shape[1:]) == (2, 2), "checkpoint_reload_exact": bool(np.array_equal(reference, reloaded)), "predictor_no_grad": not any(parameter.requires_grad for parameter in predictor.model.parameters()), "predictor_sha_unchanged": predictor.sha256 == before, "no_nan_or_inf": bool(np.isfinite(reference).all())}
        if not all(checks.values()): raise RuntimeError(f"Round-22 smoke failure for {family}: {checks}")
        rows.append({**record, **checks, **summary, "smoke": True, "cache_predictor_sha256": cache["predictor_sha256"]})
        offline.extend(metric_rows)
    after = predictor.sha256
    result = {"smoke_run_count": len(rows), "families": list(POLICY_FAMILIES), "all_passed": len(rows) == 4 and all(row["checkpoint_reload_exact"] and row["predictor_sha_unchanged"] for row in rows), "predictor_sha256_before": before, "predictor_sha256_after": after, "fresh_dev_access_count": 0, "untouched_eval_model_access_count": 0, "untouched_eval_policy_rollout_count": 0}
    _csv(output / "smoke_training_summary.csv", rows); _csv(output / "smoke_offline_validation_metrics.csv", offline); _json(output / "smoke_manifest.json", result)
    return {"manifest": result, "rows": rows}


if __name__ == "__main__":
    print(json.dumps(run_smokes()["manifest"], sort_keys=True))
