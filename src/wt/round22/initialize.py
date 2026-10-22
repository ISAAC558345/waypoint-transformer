"""Round-22 immutable-input audit, predictor selection and fresh registration."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .scenarios import fresh_policy_development_scenarios
from .governance import verify_round21_amendment_overlay


ROOT = Path(".")
ROUND = ROOT / "round_22"
R21 = ROOT / "round_21"
R20 = ROOT / "round_20"
R20_OUTPUT = R20 / "outputs" / "round20_multi_uav_dataset_20260828_2100"
OUTPUT = ROUND / "outputs" / "round22_deepsets_task_waypoint_policy_20260829_1153"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_round21_lock() -> dict[str, Any]:
    amendment = R21 / "ROUND_LOCK_AMENDMENT.json"
    if amendment.exists():
        return verify_round21_amendment_overlay()
    lock = json.loads((R21 / "ROUND_LOCK.json").read_text(encoding="utf-8"))
    manifest = json.loads((R21 / "round_manifest.json").read_text(encoding="utf-8"))
    checks: list[tuple[Path, str]] = []
    checks += [(R21 / name, value) for name, value in lock["configuration_hashes"].items()]
    checks += [(ROOT / name, value) for name, value in lock["source_tree"].items()]
    checks += [(ROOT / manifest["formal_output"] / name, value) for name, value in lock["output_hashes"].items()]
    checks += [(ROOT / manifest["figure_directory"] / name, value) for name, value in lock["figure_hashes"].items()]
    checks.append((ROOT / lock["report"], lock["report_sha256"]))
    mismatches = [{"path": path.as_posix(), "actual": sha256(path) if path.exists() else "MISSING", "expected": expected} for path, expected in checks if not path.exists() or sha256(path) != expected]
    readiness = lock["final_verification"]["readiness"]
    expected = {"task_waypoint_gate": "TWG-A", "graph_value_gate": "GV-C", "uu_edge_gate": "UU-C", "interference_feature_gate": "IF-C"}
    actual = {key: readiness[key] for key in expected}
    if mismatches or actual != expected or lock["formal_run_count"] != 18 or lock["development_test_access_count"] != 1 or lock["untouched_eval_model_access_count"] != 0:
        raise RuntimeError(f"Round-21 frozen-state conflict: {mismatches or actual}")
    return {"passed": True, "registered_hash_count": len(checks), "lock_sha256": sha256(R21 / "ROUND_LOCK.json"), "readiness": readiness}


def verify_round20_lock() -> dict[str, Any]:
    lock = json.loads((R20 / "ROUND_LOCK.json").read_text(encoding="utf-8"))
    manifest = json.loads((R20 / "round_manifest.json").read_text(encoding="utf-8"))
    groups = (("configuration_hashes", R20), ("source_tree", ROOT), ("output_hashes", ROOT / manifest["formal_output"]), ("figure_hashes", ROOT / manifest["figure_directory"]))
    mismatches = []
    count = 0
    for group, prefix in groups:
        for relative, expected in lock[group].items():
            count += 1
            path = prefix / relative
            actual = sha256(path) if path.exists() else "MISSING"
            if actual != expected:
                mismatches.append({"path": path.as_posix(), "actual": actual, "expected": expected})
    if mismatches:
        raise RuntimeError(f"Round-20 frozen dataset conflict: {mismatches[:3]}")
    return {"passed": True, "registered_hash_count": count, "lock_sha256": sha256(R20 / "ROUND_LOCK.json")}


def select_predictor() -> dict[str, Any]:
    summary = R21 / "outputs" / "round21_hgnn_task_waypoint_20260828_2142" / "task_waypoint_training_summary.csv"
    with summary.open(encoding="utf-8", newline="") as handle:
        candidates = [row for row in csv.DictReader(handle) if row["family"] == "deepsets" and int(row["horizon"]) == 5 and row["head_guard_passed"].lower() == "true"]
    if len(candidates) != 3:
        raise RuntimeError("Round-21 H5 DeepSets selection must have exactly three guarded validation candidates")
    chosen = min(candidates, key=lambda row: float(row["validation_total_loss"]))
    checkpoint = ROOT / chosen["checkpoint"]
    expected_sha = chosen["checkpoint_sha256"].upper()
    if sha256(checkpoint) != expected_sha:
        raise RuntimeError("selected frozen predictor hash does not match Round-21 manifest")
    graph_schema = R20_OUTPUT / "multi_uav_graph_schema_v1.json"
    dataset_manifest = R20_OUTPUT / "multi_uav_dataset_manifest.json"
    task_schema = R20_OUTPUT / "multi_uav_task_waypoint_schema_v1.json"
    return {
        "selection_source": "Round-21 validation-only total Task-Waypoint loss with finite-head guard",
        "candidate_count": 3,
        "candidate_validation_total_losses": [{"seed": int(row["seed"]), "validation_total_loss": float(row["validation_total_loss"]), "head_guard_passed": True} for row in sorted(candidates, key=lambda row: int(row["seed"]))],
        "family": "deepsets",
        "h_task": 5,
        "seed": int(chosen["seed"]),
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256_before_policy_training": expected_sha,
        "checkpoint_sha256_after_policy_training": expected_sha,
        "model_mode": "eval",
        "requires_grad": False,
        "optimizer_attached": False,
        "service_soft_ce": float(chosen["service_soft_ce"]),
        "progress_normalized_rmse": float(chosen["progress_normalized_rmse"]),
        "return_normalized_rmse": float(chosen["return_normalized_rmse"]),
        "validation_total_loss": float(chosen["validation_total_loss"]),
        "graph_schema_sha256": sha256(graph_schema),
        "task_schema_sha256": sha256(task_schema),
        "dataset_manifest_sha256": sha256(dataset_manifest),
    }


def register_fresh_development_set() -> dict[str, Any]:
    # Inputs here are manifest metadata only; no fresh environment/rollout is
    # constructed and no outcome field exists in the fresh manifest.
    r20_dev = json.loads((R20_OUTPUT / "multi_uav_dataset_manifest.json").read_text(encoding="utf-8"))
    r20_untouched = json.loads((R20_OUTPUT / "round20_multi_uav_untouched_eval_manifest.json").read_text(encoding="utf-8"))
    prior_seeds = {int(row["seed"]) for row in r20_dev["scenarios"]} | {int(row["seed"]) for row in r20_untouched["scenarios"]}
    prior_ids = {str(row["scenario_id"]) for row in r20_dev["scenarios"]} | {str(row["scenario_id"]) for row in r20_untouched["scenarios"]}
    fresh = fresh_policy_development_scenarios()
    fresh_seeds = {row.seed for row in fresh}
    fresh_ids = {row.scenario_id for row in fresh}
    overlap = {"seed_overlap": sorted(prior_seeds & fresh_seeds), "scenario_id_overlap": sorted(prior_ids & fresh_ids)}
    if overlap["seed_overlap"] or overlap["scenario_id_overlap"]:
        raise RuntimeError(f"fresh development registration is not disjoint: {overlap}")
    rows = [row.manifest_row() for row in fresh]
    tiers = {str(row["spectrum_tier"]) for row in rows}
    payload = {
        "namespace": "round22_fresh_multi_uav_policy_dev_v1",
        "registered_before_policy_training": True,
        "outcomes_read": False,
        "environment_constructed": False,
        "access_count": 0,
        "scenario_count": len(rows),
        "num_uavs": 2,
        "num_buoys": [3, 5],
        "interference_tiers": sorted(tiers),
        "realistic_tiers_required": ["low", "medium"],
        "disjoint_from": ["round20_train", "round20_validation", "round20_development_test", "round20_untouched_eval"],
        "disjointness_audit": overlap,
        "scenarios": rows,
    }
    write_json(OUTPUT / "fresh_multi_uav_policy_dev_manifest.json", payload)
    return {"manifest": (OUTPUT / "fresh_multi_uav_policy_dev_manifest.json").as_posix(), "manifest_sha256": sha256(OUTPUT / "fresh_multi_uav_policy_dev_manifest.json"), "scenario_count": len(rows), "access_count": 0, "outcomes_read": False, "disjointness_audit": overlap}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    r21 = verify_round21_lock()
    r20 = verify_round20_lock()
    predictor = select_predictor()
    fresh = register_fresh_development_set()
    write_json(OUTPUT / "selected_deepsets_task_waypoint_predictor.json", predictor)
    write_json(OUTPUT / "access_audit.json", {"round20_development_test_reuse_count": 0, "fresh_dev_access_count": 0, "fresh_dev_outcomes_read": False, "untouched_eval_accessed": False, "untouched_eval_model_access_count": 0, "untouched_eval_policy_rollout_count": 0, "predictor_checkpoint_sha256_before": predictor["checkpoint_sha256_before_policy_training"], "predictor_checkpoint_sha256_after": predictor["checkpoint_sha256_after_policy_training"]})
    write_json(OUTPUT / "run_manifest.json", {"round": "round_22", "status": "initialized", "round21_audit": r21, "round20_audit": r20, "selected_predictor": {key: predictor[key] for key in ("family", "h_task", "seed", "checkpoint", "checkpoint_sha256_before_policy_training")}, "fresh_development": fresh, "formal_run_count": 12, "training_started": False})
    (OUTPUT / "implementation_log.md").write_text("# Round-22 implementation log\n\nInitialized only: exact Round-21/20 lock audits passed, the DeepSets H=5 predictor was selected by validation-only loss, and the fresh policy-development manifest was registered without environment construction or outcome access.\n", encoding="utf-8")
    print(json.dumps({"round21": r21["passed"], "round20": r20["passed"], "predictor_seed": predictor["seed"], "fresh_scenarios": fresh["scenario_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
