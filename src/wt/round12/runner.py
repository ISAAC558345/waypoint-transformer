"""Round-12 policy smoke runner; it never opens fresh/test/holdout data."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contracts import PREDICTOR_SHA256, assert_predictor_frozen, load_frozen_predictor, sha256
from .data import load_round12_policy_dataset
from .policy_models import POLICY_FAMILIES, Round12PolicySchema
from .sequence import build_causal_sequence_split
from .trainer import Round12PolicyTrainer, Round12TrainingConfig, load_round12_checkpoint


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run_smoke(root: Path = Path.cwd(), *, device: str = "cpu") -> Path:
    """Run one short, reload-verified train/validation smoke for every family."""
    root = Path(root)
    dataset = load_round12_policy_dataset(root, limits={"train": 64, "validation": 48})
    train = build_causal_sequence_split(dataset.train)
    validation = build_causal_sequence_split(dataset.validation)
    schema = Round12PolicySchema(state_dim=dataset.state_dim, max_buoys=dataset.max_buoys)
    run_id = datetime.now(timezone.utc).strftime("round12_smoke_%Y%m%d_%H%M%S_%f")
    output = root / "round_12" / "outputs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    predictor = load_frozen_predictor(root, schema.state_dim, schema.max_buoys)
    predictor_hash_before = sha256(root / "round_11/checkpoints/H10/seed_2/best.pt")
    assert_predictor_frozen(predictor)
    results: dict[str, Any] = {}
    for family in POLICY_FAMILIES:
        trainer = Round12PolicyTrainer(
            family,
            schema,
            Round12TrainingConfig(
                seed=0, batch_size=8, max_epochs=2, patience=2, batches_per_epoch=1
            ),
            root=root,
            output_dir=output / family,
            device=device,
        )
        result = trainer.fit(train, validation)
        before, target = trainer.predict(validation.take(np.arange(8, dtype=np.int64)))
        reloaded, checkpoint = load_round12_checkpoint(
            Path(result.best_checkpoint), family=family, schema=schema, map_location=device
        )
        trainer.model = reloaded.to(trainer.device)
        after, reloaded_target = trainer.predict(validation.take(np.arange(8, dtype=np.int64)))
        finite = bool(np.all(np.isfinite(before)) and np.all(np.isfinite(after)))
        identical = bool(np.array_equal(before, after) and np.array_equal(target, reloaded_target))
        results[family] = {
            "forward_finite": finite,
            "loss_finite": all(np.isfinite(row["train_normalized_action_mse"]) for row in result.history),
            "gradient_finite": all(np.isfinite(row["gradient_norm_preclip_max"]) for row in result.history),
            "action_shape": list(before.shape),
            "checkpoint_saved": Path(result.best_checkpoint).is_file() and Path(result.last_checkpoint).is_file(),
            "checkpoint_reload_identical": identical,
            "best_validation_raw_action_rmse": result.best_validation_raw_action_rmse,
            "parameter_count": result.parameter_count,
            "oracle_offline_only": family == "oracle_task_waypoint_transformer",
            "checkpoint_schema": checkpoint["schema_version"],
        }
    assert_predictor_frozen(predictor)
    predictor_hash_after = sha256(root / "round_11/checkpoints/H10/seed_2/best.pt")
    if predictor_hash_before != PREDICTOR_SHA256 or predictor_hash_after != PREDICTOR_SHA256:
        raise RuntimeError("selected predictor SHA-256 changed during smoke")
    fresh = json.loads((root / "round_12/fresh_policy_dev_manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "run_id": run_id,
        "smoke": True,
        "families": results,
        "predictor_sha256_before": predictor_hash_before,
        "predictor_sha256_after": predictor_hash_after,
        "predictor_has_gradients": any(parameter.grad is not None for parameter in predictor.parameters()),
        "fresh_development_access_count": int(fresh["access_count"]),
        "old_development_test_access_count": 0,
        "untouched_holdout_access_count": 0,
        "no_nan_or_inf": all(value["forward_finite"] and value["loss_finite"] and value["gradient_finite"] for value in results.values()),
    }
    _write_json(output / "smoke_manifest.json", manifest)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run all four short policy smokes")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if not args.smoke:
        parser.error("Round-12 runner currently exposes only the required --smoke mode")
    print(run_smoke(args.root, device=args.device))


if __name__ == "__main__":
    main()
