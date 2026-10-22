"""Round-24 pre-registered multi-seed stability extension.

The only stochastic degrees of freedom introduced here are the registered
initialization/shuffle seeds 3--7.  All model and data handling is delegated
to the locked Round-22 implementation so this module cannot silently create a
different training protocol.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from wt.governance_state_sync import verify_round22_governance_state_sync_amendment
from wt.round22.data import (FrozenDeepSetsPredictor, R20_OUTPUT, R22_OUTPUT,
                             TRAIN, VALIDATION, load_policy_sequences, sha256)
from wt.round22.formal import _reload_audit
from wt.round22.models import (PREDICTED_MLP, PREDICTED_TRANSFORMER,
                               STATE_ONLY_TRANSFORMER)
from wt.round22.training import TrainingConfig, predict, train_one

from .fresh import FRESH_NAMESPACE, fresh_stability_scenarios


ROOT = Path(__file__).resolve().parents[3]
ROUND = ROOT / "round_24"
OUTPUT = ROUND / "outputs" / "round24_multiseed_stability_20260829_1515"
CONFIG_PATH = ROUND / "configs" / "round24_seed_extension_config.json"
R23_LOCK = ROOT / "round_23" / "ROUND_LOCK.json"
R23_AMENDMENT = ROOT / "round_23" / "ROUND_LOCK_AMENDMENT.json"
R22_TRAINING = R22_OUTPUT / "round22_formal_training_manifest.json"
R22_PROTOCOL = R22_OUTPUT / "round22_protocol_freeze.json"
EXPECTED_R23_LOCK = "948E89F9197E8204B147FA2F3B9543CD878D4380680254DCB1190E7E9015F22A"
EXPECTED_PREDICTOR = "645F235032DE9639A6D42318A41CC8F0E2175E86E38545DA98E0BE5CAEB80221"
FAMILIES = (PREDICTED_TRANSFORMER, STATE_ONLY_TRANSFORMER, PREDICTED_MLP)
OLD_SEEDS = (0, 1, 2)
NEW_SEEDS = (3, 4, 5, 6, 7)
ALL_SEEDS = OLD_SEEDS + NEW_SEEDS


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Mapping): return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [_plain(v) for v in value]
    return value


def _json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise RuntimeError(f"Round-24 required output is empty: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in materialized for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: json.dumps(_plain(value), sort_keys=True) if isinstance(value, (dict, tuple, list, np.ndarray)) else value for key, value in row.items()})


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _r22_runs() -> list[dict[str, Any]]:
    payload = _load_json(R22_TRAINING)
    runs = list(payload.get("runs", []))
    if payload.get("formal_training_completed") != "12/12" or len(runs) != 12:
        raise RuntimeError("Round-24 requires the immutable Round-22 12/12 matrix")
    return runs


def verify_frozen_state() -> dict[str, Any]:
    """Fail closed before any Round-24 checkpoint can be made."""
    if sha256(R23_LOCK) != EXPECTED_R23_LOCK:
        raise RuntimeError("Round-23 original lock changed")
    amendment = _load_json(R23_AMENDMENT)
    if (amendment.get("amendment_type") != "legal_round24_successor_governance_extension"
            or amendment.get("original_round23_lock_sha256") != EXPECTED_R23_LOCK
            or amendment.get("scientific_artifacts_modified") is not False):
        raise RuntimeError("Round-23 successor amendment is not the approved narrow lifecycle amendment")
    affected = amendment.get("affected_paths", {})
    governance_record = affected.get("src/wt/governance_state_sync.py", {})
    test_record = affected.get("tests/test_round23_audit.py", {})
    if (governance_record.get("original_sha256") != "281D3633A1FC706BF42DC81FA16E805755B168FB775C2EEA6F4377C50D1DD001"
            or sha256(ROOT / "src/wt/governance_state_sync.py") != governance_record.get("amended_sha256")
            or test_record.get("original_sha256") != "1D54D2B378DE4BC5633BE82C392C484D27C864F56CED7D294C0D784EACBD1C64"
            or sha256(ROOT / "tests/test_round23_audit.py") != test_record.get("amended_sha256")):
        raise RuntimeError("Round-23 successor amendment source/test hashes changed")
    governance = verify_round22_governance_state_sync_amendment()
    runs = _r22_runs()
    old = [row for row in runs if row["family"] in FAMILIES and int(row["seed"]) in OLD_SEEDS]
    if len(old) != 9:
        raise RuntimeError("Round-22 deployable 0--2 checkpoint matrix is incomplete")
    failures = [row["run_id"] for row in old if sha256(ROOT / row["checkpoint"]) != row["checkpoint_sha256"] or sha256(ROOT / row["last_checkpoint"]) != row["last_checkpoint_sha256"]]
    predictor = FrozenDeepSetsPredictor()
    if predictor.sha256 != EXPECTED_PREDICTOR or any(parameter.requires_grad for parameter in predictor.model.parameters()):
        raise RuntimeError("the H5 DeepSets predictor is not frozen")
    if failures:
        raise RuntimeError(f"Round-22 checkpoint hash mismatch: {failures}")
    access = _load_json(R22_OUTPUT / "access_audit.json")
    expected_access = {"fresh_dev_access_count": 1, "round20_development_test_reuse_count": 0,
                       "untouched_eval_model_access_count": 0, "untouched_eval_policy_rollout_count": 0}
    if any(int(access[key]) != value for key, value in expected_access.items()):
        raise RuntimeError("protected Round-22 access counts changed")
    return {"passed": True, "round23_original_lock_sha256": EXPECTED_R23_LOCK,
            "round22_governance_verified": bool(governance["passed"]),
            "predictor_sha256": predictor.sha256, "old_checkpoint_failures": failures,
            **expected_access}


def protocol_equivalence_audit() -> dict[str, Any]:
    """Record field-by-field equivalence of the reused scientific protocol."""
    frozen = _load_json(R22_PROTOCOL)
    source_config = _load_json(ROOT / frozen["training_config"])
    config = TrainingConfig()
    target = {"learning_rate": config.learning_rate, "weight_decay": config.weight_decay,
              "gradient_clip_norm": config.gradient_clip_norm, "batch_size": config.batch_size,
              "max_epochs": config.max_epochs, "early_stopping_patience": config.early_stopping_patience,
              "device": config.device, "h_task": 5, "context_length": 10,
              "train_split": TRAIN, "validation_split": VALIDATION,
              "checkpoint_selection": "validation_raw_joint_movement_action_rmse_only",
              "families": list(FAMILIES), "new_seeds": list(NEW_SEEDS)}
    source = {"learning_rate": source_config["optimizer"]["learning_rate"],
              "weight_decay": source_config["optimizer"]["weight_decay"],
              "gradient_clip_norm": source_config["optimizer"]["gradient_clip_norm"],
              "batch_size": source_config["optimizer"]["batch_size"],
              "max_epochs": source_config["optimizer"]["max_epochs"],
              "early_stopping_patience": source_config["optimizer"]["early_stopping_patience"],
              "h_task": source_config["dataset"]["h_task"],
              "context_length": source_config["architecture"]["context_length"],
              "train_split": frozen["train_split"], "validation_split": frozen["validation_split"],
              "checkpoint_selection": frozen["checkpoint_selection"]}
    fields = []
    for name, observed in source.items():
        target_value = target[name]
        fields.append({"field": name, "round22_value": observed, "round24_value": target_value,
                       "equal": observed == target_value, "scientific_field": True})
    fields.extend([
        {"field": "policy_architecture", "round22_value": source_config["architecture"],
         "round24_value": source_config["architecture"], "equal": True, "scientific_field": True},
        {"field": "frozen_predictor_sha256", "round22_value": frozen["predictor_sha256"],
         "round24_value": EXPECTED_PREDICTOR, "equal": frozen["predictor_sha256"] == EXPECTED_PREDICTOR, "scientific_field": True},
        {"field": "round20_transition_sha256", "round22_value": frozen["round20_transition_sha256"],
         "round24_value": sha256(R20_OUTPUT / "multi_uav_transitions.jsonl"), "equal": frozen["round20_transition_sha256"] == sha256(R20_OUTPUT / "multi_uav_transitions.jsonl"), "scientific_field": True},
        {"field": "round20_task_samples_sha256", "round22_value": frozen["round20_task_samples_sha256"],
         "round24_value": sha256(R20_OUTPUT / "multi_uav_task_waypoint_samples.jsonl"), "equal": frozen["round20_task_samples_sha256"] == sha256(R20_OUTPUT / "multi_uav_task_waypoint_samples.jsonl"), "scientific_field": True},
        {"field": "allowed_difference", "round22_value": [0, 1, 2], "round24_value": list(NEW_SEEDS),
         "equal": False, "scientific_field": False, "explanation": "pre-registered initialization/shuffle seed extension only"},
    ])
    audit = {"round": "round_24", "round22_protocol_sha256": sha256(R22_PROTOCOL),
             "source_training_config_sha256": sha256(ROOT / frozen["training_config"]),
             "all_scientific_fields_equal": all(bool(row["equal"]) for row in fields if row["scientific_field"]),
             "no_architecture_loss_sampler_or_optimizer_change": True, "fields": fields}
    if not audit["all_scientific_fields_equal"]:
        raise RuntimeError("Round-24 scientific protocol equivalence failed")
    _json(OUTPUT / "round24_protocol_equivalence_audit.json", audit)
    return audit


def preregister_fresh_set() -> dict[str, Any]:
    """Write the pre-training manifest without constructing an environment."""
    path = OUTPUT / "round24_fresh_manifest.json"
    scenarios = [row.manifest_row() for row in fresh_stability_scenarios()]
    historic_paths = {
        "round20_development": R20_OUTPUT / "multi_uav_dataset_manifest.json",
        "round22_fresh": R22_OUTPUT / "fresh_multi_uav_policy_dev_manifest.json",
        "round20_untouched": R20_OUTPUT / "round20_multi_uav_untouched_eval_manifest.json",
    }
    historic_ids: set[str] = set(); historic_seeds: set[int] = set(); historic_hashes: dict[str, str] = {}
    for label, source in historic_paths.items():
        payload = _load_json(source); records = payload.get("scenarios", [])
        historic_ids.update(str(row["scenario_id"]) for row in records)
        historic_seeds.update(int(row["seed"]) for row in records)
        historic_hashes[label] = sha256(source)
    ids, seeds = {str(row["scenario_id"]) for row in scenarios}, {int(row["seed"]) for row in scenarios}
    overlap_ids, overlap_seeds = sorted(ids & historic_ids), sorted(seeds & historic_seeds)
    if len(scenarios) != 80 or overlap_ids or overlap_seeds:
        raise RuntimeError("Round-24 fresh set is not exactly 80 and fully historical-set-disjoint")
    manifest = {"round": "round_24", "namespace": FRESH_NAMESPACE, "pre_training_registration": True,
                "rollout_outcomes_read": False, "fresh_access_count": 0, "scenario_count": 80,
                "base_seeds": list(range(24000, 24010)), "num_buoys": [3, 5], "eta": [0.0, .25, .5, 1.0],
                "protocol": {"episode_horizon": 12, "same_environment_as_round22": True},
                "historic_manifest_sha256": historic_hashes,
                "disjointness_audit": {"scenario_id_overlap": overlap_ids, "seed_overlap": overlap_seeds,
                                       "historical_scenario_count": len(historic_ids), "historical_seed_count": len(historic_seeds)},
                "bridge_label": "not_available_without_future_labels", "scenarios": scenarios}
    canonical = _plain(manifest)
    if path.is_file():
        previous = _load_json(path)
        if previous != canonical:
            raise RuntimeError("Round-24 fresh manifest changed after registration")
    else:
        _json(path, canonical)
    return canonical


def _new_dir(family: str, seed: int) -> Path:
    return OUTPUT / "formal" / family / f"seed_{seed}"


def _run_id(family: str, seed: int) -> str:
    return f"{family}__seed_{seed}"


def _existing_new_result(family: str, seed: int, train: Any, validation: Any, config: TrainingConfig) -> dict[str, Any] | None:
    directory = _new_dir(family, seed); checkpoint, last = directory / "best.pt", directory / "last.pt"
    if not (checkpoint.is_file() and last.is_file()): return None
    from wt.round22.training import load_checkpoint
    loaded_family, _, _, payload = load_checkpoint(checkpoint, config.device)
    if loaded_family != family or int(payload.get("seed", -1)) != seed:
        raise RuntimeError("existing Round-24 checkpoint identity mismatch")
    history = list(payload.get("history", []))
    if not history: raise RuntimeError("existing Round-24 checkpoint is missing history")
    audit = _reload_audit(checkpoint, train, validation, config)
    return {"run_id": _run_id(family, seed), "family": family, "seed": seed,
            "checkpoint": checkpoint.as_posix(), "checkpoint_sha256": sha256(checkpoint),
            "last_checkpoint": last.as_posix(), "last_checkpoint_sha256": sha256(last),
            "best_epoch": int(payload["best_epoch"]),
            "best_validation_raw_joint_action_rmse": float(payload["best_validation_raw_joint_action_rmse"]),
            "stopped_epoch": len(history), "parameter_count": sum(p.numel() for p in load_checkpoint(checkpoint)[1].parameters()),
            "deployable": bool(payload["deployable"]), "finite": bool(np.isfinite([row["train_action_mse"] for row in history]).all()),
            "resume_supported": bool(payload.get("resume_supported")), "resumed_existing_checkpoint": True, **audit}


def _write_training_progress(protocol: Mapping[str, Any], runs: list[dict[str, Any]], status: str) -> None:
    manifest = {"round": "round_24", "status": status, "formal_run_count": 15,
                "formal_training_completed": f"{len(runs)}/15", "families": list(FAMILIES),
                "new_seeds": list(NEW_SEEDS), "old_seeds_retrained": [],
                "predictor_sha256_before": protocol["predictor_sha256"],
                "predictor_sha256_after": sha256(Path(protocol["predictor_checkpoint"])),
                "dataset_transition_sha256": protocol["round20_transition_sha256"],
                "dataset_task_samples_sha256": protocol["round20_task_samples_sha256"],
                "protocol_equivalence_audit": (OUTPUT / "round24_protocol_equivalence_audit.json").as_posix(),
                "fresh_manifest_sha256_pretraining": sha256(OUTPUT / "round24_fresh_manifest.json"), "runs": runs}
    _json(OUTPUT / "round24_new_seed_training_manifest.json", manifest)
    if runs: _csv(OUTPUT / "round24_new_seed_training_summary.csv", runs)


def train_new_seeds() -> list[dict[str, Any]]:
    """Train exactly the registered 3x5 matrix; no pre-existing seed is legal."""
    frozen = verify_frozen_state(); protocol = protocol_equivalence_audit(); fresh = preregister_fresh_set()
    if fresh["fresh_access_count"] != 0 or fresh["rollout_outcomes_read"]:
        raise RuntimeError("new fresh outcomes are prohibited during training")
    predictor = FrozenDeepSetsPredictor()
    if predictor.sha256 != EXPECTED_PREDICTOR:
        raise RuntimeError("predictor changed before Round-24 training")
    config = TrainingConfig(); train = load_policy_sequences(TRAIN, predictor); validation = load_policy_sequences(VALIDATION, predictor)
    frozen_protocol = _load_json(R22_PROTOCOL)
    runs: list[dict[str, Any]] = []
    for family in FAMILIES:
        for seed in NEW_SEEDS:
            existing = _existing_new_result(family, seed, train, validation, config)
            if existing is not None:
                runs.append(existing); continue
            directory = _new_dir(family, seed); checkpoint = directory / "best.pt"
            model, normalizer, history, result = train_one(family, seed, train, validation, config, checkpoint)
            _csv(directory / "training_history.csv", history)
            expected = predict(family, model, normalizer.apply(validation), config)
            audit = _reload_audit(checkpoint, train, validation, config)
            if hashlib.sha256(expected.tobytes()).hexdigest().upper() != audit["prediction_sha256"]:
                raise RuntimeError("reloaded Round-24 checkpoint does not reproduce the saved policy")
            if not result["finite"] or predictor.sha256 != EXPECTED_PREDICTOR:
                raise RuntimeError(f"Round-24 run integrity failed: {_run_id(family, seed)}")
            runs.append({"run_id": _run_id(family, seed), **result, "resumed_existing_checkpoint": False,
                         "dataset_transition_sha256": frozen_protocol["round20_transition_sha256"],
                         "dataset_task_samples_sha256": frozen_protocol["round20_task_samples_sha256"],
                         "predictor_sha256": predictor.sha256, **audit})
            _write_training_progress(frozen_protocol, runs, "training_in_progress")
    expected_ids = {_run_id(family, seed) for family in FAMILIES for seed in NEW_SEEDS}
    if len(runs) != 15 or {row["run_id"] for row in runs} != expected_ids:
        raise RuntimeError("Round-24 15-run matrix is incomplete, duplicated, or unauthorized")
    if predictor.sha256 != EXPECTED_PREDICTOR:
        raise RuntimeError("predictor changed during Round-24 training")
    _write_training_progress(frozen_protocol, runs, "formal_training_complete")
    _json(OUTPUT / "checkpoint_manifest.json", {"round": "round_24", "old_round22_checkpoints": [row for row in _r22_runs() if row["family"] in FAMILIES and int(row["seed"]) in OLD_SEEDS],
        "new_round24_checkpoints": runs, "old_checkpoints_retrained": False,
        "new_seed_set": list(NEW_SEEDS), "predictor_sha256_before": EXPECTED_PREDICTOR,
        "predictor_sha256_after": predictor.sha256, "dataset_hash_unchanged": True,
        "frozen_pretraining_fresh_manifest_sha256": sha256(OUTPUT / "round24_fresh_manifest.json"),
        "frozen_state": frozen})
    return runs


def run_round24() -> dict[str, Any]:
    """Initial executable stage; analyses are deliberately unlocked only post-training."""
    runs = train_new_seeds()
    return {"round": "round_24", "formal_training_completed": f"{len(runs)}/15",
            "fresh_outcomes_accessed": False, "predictor_sha256": runs[0]["predictor_sha256"]}


def main() -> None:
    print(json.dumps(run_round24(), sort_keys=True))


if __name__ == "__main__":
    main()
