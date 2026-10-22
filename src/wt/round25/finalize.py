"""Lock the completed, inference-only Round-25 evidence bundle."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from wt.round25.audit import EXPECTED_R24, FIGURES, OUTPUT, ROOT, ROUND, _load_json, verify_frozen_state


REQUIRED_OUTPUTS = (
    "task_waypoint_supervision_ambiguity_spec.md", "initial_state_target_ambiguity.csv",
    "behavior_target_divergence.csv", "quality_conditioned_target_audit.csv",
    "knn_target_ambiguity.csv", "bridge_supervision_ambiguity.csv",
    "conflict_supervision_ambiguity.csv", "predictor_mean_regression_audit.csv",
    "service_entropy_averaging_audit.csv", "progress_mean_shrinkage_audit.csv",
    "return_mean_regression_audit.csv", "planner_task_waypoint_spec.md",
    "planner_task_waypoint_samples.csv", "planner_behavior_target_alignment.csv",
    "behavior_planner_alignment.csv", "quality_planner_alignment.csv",
    "predictor_planner_alignment.csv", "policy_seed_planner_action_alignment.csv",
    "predictability_vs_planning_utility.csv", "planner_target_horizon_audit.csv",
    "task_waypoint_supervision_ambiguity_gate.json", "planner_task_waypoint_target_gate.json",
    "round25_readiness_gate.json", "access_audit.json", "run_manifest.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _tree(path: Path) -> dict[str, str]:
    return {item.relative_to(path).as_posix(): sha256(item) for item in sorted(path.rglob("*")) if item.is_file() and "__pycache__" not in item.parts}


def _finite_csvs() -> None:
    for path in OUTPUT.glob("*.csv"):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                for value in row.values():
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(number):
                        raise RuntimeError(f"non-finite Round-25 value in {path.name}")


def finalize(*, full_test_count: int, compileall_passed: bool) -> dict[str, Any]:
    """Write the immutable Round-25 lock after successful final verification."""
    missing = [name for name in REQUIRED_OUTPUTS if not (OUTPUT / name).is_file()]
    if missing:
        raise RuntimeError(f"Round-25 required output missing: {missing}")
    if not (ROUND / "report" / "round25_completion_report.md").is_file() or not (ROUND / "logs" / "implementation_log.md").is_file():
        raise RuntimeError("Round-25 report or implementation log missing")
    local_manifest = _load_json(ROUND / "round_manifest.json")
    if local_manifest.get("status") != "locked":
        raise RuntimeError("Round-25 local manifest must be locked before writing its lock")
    if any(bool(local_manifest[name]) for name in ("training_performed", "dataset_modified", "policy_training_performed", "predictor_training_performed", "hgnn_training_performed", "ppo_training_performed")):
        raise RuntimeError("Round-25 local manifest records prohibited training or modification")
    frozen = verify_frozen_state()
    run, access = _load_json(OUTPUT / "run_manifest.json"), _load_json(OUTPUT / "access_audit.json")
    sa, pt, readiness = (_load_json(OUTPUT / name) for name in ("task_waypoint_supervision_ambiguity_gate.json", "planner_task_waypoint_target_gate.json", "round25_readiness_gate.json"))
    if run.get("formal_sample_counts") != {"h5_train_validation": 5120, "h10_train_validation": 1920, "exact_state_groups": 64, "knn_rows": 5120}:
        raise RuntimeError("Round-25 frozen sample coverage is incomplete")
    if access.get("allowed_splits_decoded") != ["train", "validation"] or int(access.get("development_test_rows_decoded", -1)) != 0:
        raise RuntimeError("Round-25 decoded a prohibited split")
    protected = ("round22_fresh_new_rollout_count", "round20_development_test_reuse_count", "round24_fresh_new_rollout_count", "untouched_eval_access_count", "untouched_eval_model_access_count", "untouched_eval_policy_rollout_count")
    if any(int(access.get(name, -1)) != 0 for name in protected) or bool(access.get("fresh_or_untouched_rollout_performed")):
        raise RuntimeError("Round-25 protected-access audit is invalid")
    if not bool(access.get("policy_checkpoint_inference_only")) or access.get("predictor_sha256_before") != access.get("predictor_sha256_after"):
        raise RuntimeError("Round-25 frozen-inference audit is invalid")
    if sa.get("gate") not in {"SA-TW-A", "SA-TW-B", "SA-TW-C"} or pt.get("gate") not in {"PT-A", "PT-B", "PT-C"}:
        raise RuntimeError("Round-25 gate is invalid")
    if readiness.get("training_performed") is not False or readiness.get("round26_created") is not False:
        raise RuntimeError("Round-25 readiness gate violates the authorized scope")
    figures = sorted(path for path in FIGURES.iterdir() if path.suffix in {".png", ".svg"})
    if len(figures) != 40:
        raise RuntimeError("Round-25 requires exactly 20 PNG+SVG pairs")
    _finite_csvs()
    verification = {
        "full_regression_tests": {"passed": True, "count": int(full_test_count), "command": "py -3.14 -m unittest discover -s tests -q"},
        "compileall_passed": bool(compileall_passed),
        "required_output_missing": missing,
        "figure_count": len(figures),
        "frozen_state": frozen,
        "access_audit": access,
        "supervision_ambiguity_gate": sa["gate"],
        "planner_target_gate": pt["gate"],
        "readiness_recommendation": readiness["recommendation"],
    }
    lock = {
        "round": "round_25", "status": "locked",
        "topic": "Multi-UAV Task-Waypoint Supervision Ambiguity and Expert-Plan Alignment Audit",
        "snapshot": "backup/snapshots/before_round_25_20260830_1203",
        "previous_round_lock": "round_24/ROUND_LOCK.json", "previous_round_lock_sha256": EXPECTED_R24,
        "configuration_hashes": {name: sha256(ROUND / name) for name in ("AGENTS.md", "configs/round25_ambiguity_alignment_config.json", "round_manifest.json")},
        "source_tree": _tree(ROOT / "src" / "wt" / "round25"),
        "test_hashes": {"tests/test_round25_supervision_ambiguity.py": sha256(ROOT / "tests" / "test_round25_supervision_ambiguity.py")},
        "formal_output": OUTPUT.relative_to(ROOT).as_posix(), "figure_directory": FIGURES.relative_to(ROOT).as_posix(),
        "output_hashes": _tree(OUTPUT), "figure_hashes": {path.name: sha256(path) for path in figures},
        "report": "round_25/report/round25_completion_report.md", "report_sha256": sha256(ROUND / "report" / "round25_completion_report.md"),
        "implementation_log": "round_25/logs/implementation_log.md", "implementation_log_sha256": sha256(ROUND / "logs" / "implementation_log.md"),
        "final_verification": verification,
        "training_performed": False, "supervision_ambiguity_audit_performed": True, "planner_alignment_audit_performed": True,
        "dataset_modified": False, "policy_training_performed": False, "predictor_training_performed": False,
        "hgnn_training_performed": False, "ppo_training_performed": False,
        "round22_fresh_new_rollout_count": 0, "round20_development_test_reuse_count": 0,
        "round24_fresh_new_rollout_count": 0, "untouched_eval_access_count": 0, "round26_created": False,
        "supervision_ambiguity_gate": sa["gate"], "task_waypoint_supervision_ambiguity_confirmed": sa["task_waypoint_supervision_ambiguity_confirmed"],
        "planner_target_gate": pt["gate"], "planner_derived_task_waypoint_candidate": pt["planner_derived_task_waypoint_candidate"],
        "readiness_recommendation": readiness["recommendation"],
    }
    (ROUND / "ROUND_LOCK.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lock
