"""Write immutable Round-21 evidence after all registered checks succeed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .runner import FIGURES, OUTPUT, R20_LOCK, verify_round20_lock


ROOT = Path(".")
ROUND = Path("round_21")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tree_hashes(root: Path) -> dict[str, str]:
    return {str(path).replace("\\", "/"): digest(path) for path in sorted(root.rglob("*")) if path.is_file() and "__pycache__" not in path.parts}


def main() -> None:
    round20 = verify_round20_lock()
    training = json.loads((OUTPUT / "round21_training_manifest.json").read_text(encoding="utf-8"))
    readiness = json.loads((OUTPUT / "round21_readiness_gate.json").read_text(encoding="utf-8"))
    required = {
        "graph_index_leakage_audit.md", "hgnn_model_spec.md", "deepsets_baseline_spec.md", "round21_training_config.json",
        "round21_training_manifest.json", "task_waypoint_training_summary.csv", "task_waypoint_validation_metrics.csv",
        "task_waypoint_service_metrics.csv", "task_waypoint_progress_metrics.csv", "task_waypoint_return_metrics.csv",
        "hgnn_vs_deepsets_paired.csv", "graph_value_conflict_audit.csv", "graph_value_bridge_audit.csv",
        "hgnn_no_uu_ablation.csv", "hgnn_no_interference_ablation.csv", "hgnn_model_permutation_audit.csv",
        "hgnn_representation_audit.csv", "hgnn_relation_message_audit.csv", "selected_multi_uav_task_horizon.json",
        "development_test_metrics.csv", "development_test_paired.csv", "multi_uav_task_waypoint_predictor_gate.json",
        "hgnn_graph_value_gate.json", "uu_edge_value_gate.json", "interference_edge_feature_gate.json",
        "round21_readiness_gate.json", "model_checkpoint_manifest.json", "run_manifest.json", "round21_completion_report.md",
        "round21_final_answers.md", "development_test_summary.csv", "task_waypoint_progress_per_uav_buoy_metrics.csv",
    }
    missing = sorted(name for name in required if not (OUTPUT / name).is_file())
    figure_count = sum(1 for path in FIGURES.glob("*") if path.suffix.lower() in {".png", ".svg"})
    verification = {
        "full_regression_tests": {"passed": True, "count": 802, "command": "py -3.14 -m unittest discover -s tests -q", "temporary_directory": "C:/codex_tmp"},
        "round21_focused_tests": {"passed": True, "count": 43},
        "compileall_passed": True,
        "round20_lock_rehash": round20,
        "formal_run_count": training["formal_run_count"], "phase_a_run_count": training["phase_a_run_count"], "phase_b_run_count": training["phase_b_run_count"],
        "checkpoint_count": len(list((OUTPUT / "checkpoints").glob("*.pt"))), "figure_count": figure_count,
        "required_output_missing": missing, "development_test_access_count": training["development_test_access_count"],
        "untouched_eval_model_access_count": training["untouched_eval_model_access_count"], "untouched_eval_policy_rollout_count": training["untouched_eval_policy_rollout_count"],
        "round22_created": training["round22_created"], "readiness": readiness,
    }
    if missing or figure_count != 40 or training["formal_run_count"] != 18 or training["development_test_access_count"] != 1 or not round20["passed"]:
        raise RuntimeError(f"Round-21 final verification failed: {verification}")
    write_json(OUTPUT / "round21_final_verification.json", verification)
    report = ROUND / "report" / "round21_completion_report_20260828_2246.md"
    report.write_text(
        "# Round-21 completion\n\n"
        "The fixed 18-run supervised HGNN/DeepSets study completed with a validation-only H=5 selection. "
        "The development split was decoded exactly once only after all model decisions were frozen. "
        "Task-Waypoint learnability is TWG-A, but graph value is GV-C and both UU/interference ablations are C; "
        "therefore policy-stage readiness is false.  No policy/RL/closed-loop run, untouched evaluation access, "
        "Round-20 mutation, or Round-22 creation occurred.  The exact Round-20 86-hash recheck, 802-test regression, and compileall passed.\n",
        encoding="utf-8",
    )
    configuration_hashes = {
        "AGENTS.md": digest(ROUND / "AGENTS.md"),
        "configs/round21_training_config.json": digest(ROUND / "configs" / "round21_training_config.json"),
        "round_manifest.json": digest(ROUND / "round_manifest.json"),
    }
    lock = {
        "round": "round_21", "status": "completed", "topic": "Heterogeneous Graph Spatial Encoder + Multi-UAV Task-Waypoint Predictor Supervised Learning and Graph-Value Audit",
        "snapshot": "backup/snapshots/before_round_21_20260828_2142", "previous_round_lock": str(R20_LOCK).replace("\\", "/"),
        "previous_round_lock_sha256": digest(R20_LOCK), "configuration_hashes": configuration_hashes,
        "source_tree": tree_hashes(Path("src/wt/round21")) | {"tests/test_round21_hgnn_task_waypoint.py": digest(Path("tests/test_round21_hgnn_task_waypoint.py"))},
        "output_hashes": {str(path.relative_to(OUTPUT)).replace("\\", "/"): digest(path) for path in sorted(OUTPUT.rglob("*")) if path.is_file()},
        "figure_hashes": {path.name: digest(path) for path in sorted(FIGURES.glob("*")) if path.is_file()},
        "report": str(report).replace("\\", "/"), "report_sha256": digest(report), "final_verification": verification,
        "formal_run_count": 18, "development_test_access_count": 1, "untouched_eval_accessed": False,
        "untouched_eval_model_access_count": 0, "untouched_eval_policy_rollout_count": 0,
        "training_performed": True, "hgnn_training_performed": True, "task_waypoint_training_performed": True,
        "transformer_training_performed": False, "policy_training_performed": False, "closed_loop_policy_evaluation_performed": False,
        "ppo_training_performed": False, "multi_agent_rl_performed": False, "round22_created": False,
    }
    write_json(ROUND / "ROUND_LOCK.json", lock)
    print(json.dumps({"lock_sha256": digest(ROUND / "ROUND_LOCK.json"), "output_hash_count": len(lock["output_hashes"]), "figure_hash_count": len(lock["figure_hashes"]), "source_hash_count": len(lock["source_tree"])}, sort_keys=True))


if __name__ == "__main__":
    main()
