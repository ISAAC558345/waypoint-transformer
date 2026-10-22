"""Lock the completed, inference-only Round-26 evidence bundle."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from wt.round26.audit import EXPECTED_R25, FIGURES, OUTPUT, ROOT, ROUND, _load, verify_frozen_state


REQUIRED_OUTPUTS = (
    "action_supervision_semantics_audit.md", "action_supervision_ambiguity_spec.md",
    "initial_state_action_ambiguity.csv", "action_vs_taskwaypoint_ambiguity.csv",
    "behavior_action_divergence.csv", "quality_conditioned_action_audit.csv",
    "knn_action_ambiguity.csv", "action_multimodality_audit.csv",
    "mean_action_averaging_risk.csv", "conflict_action_ambiguity.csv",
    "bridge_action_ambiguity.csv", "action_ambiguity_grouped.csv",
    "task_waypoint_action_disambiguation.csv", "head_action_disambiguation_audit.csv",
    "planner_action_coherence.csv", "behavior_planner_action_alignment.csv",
    "planner_action_utility_alignment.csv", "quality_planner_action_alignment.csv",
    "policy_seed_planner_action_alignment.csv", "taskwp_state_planner_alignment.csv",
    "bc_supervision_conflict_score.csv", "bc_conflict_seed_instability.csv",
    "planner_action_distillation_gate.json", "action_supervision_ambiguity_gate.json",
    "task_waypoint_action_disambiguation_gate.json", "round26_readiness_gate.json",
    "run_manifest.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _tree(path: Path) -> dict[str, str]:
    return {
        item.relative_to(path).as_posix(): sha256(item)
        for item in sorted(path.rglob("*"))
        if item.is_file() and "__pycache__" not in item.parts
    }


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
                        raise RuntimeError(f"non-finite Round-26 value in {path.name}")


def finalize(*, full_test_count: int, compileall_passed: bool) -> dict[str, Any]:
    """Write the immutable Round-26 lock after successful final verification."""
    missing = [name for name in REQUIRED_OUTPUTS if not (OUTPUT / name).is_file()]
    actual = {path.name for path in OUTPUT.iterdir() if path.is_file()}
    if missing or actual != set(REQUIRED_OUTPUTS):
        raise RuntimeError(f"Round-26 output inventory mismatch: missing={missing}, extra={sorted(actual - set(REQUIRED_OUTPUTS))}")
    report = ROUND / "report" / "round26_completion_report.md"
    implementation_log = ROUND / "logs" / "implementation_log.md"
    access_log = ROUND / "logs" / "access_audit.json"
    if not report.is_file() or not implementation_log.is_file() or not access_log.is_file():
        raise RuntimeError("Round-26 report or log is missing")
    local = _load(ROUND / "round_manifest.json")
    if local.get("status") != "locked":
        raise RuntimeError("Round-26 local manifest must be locked before writing its lock")
    prohibited = ("training_performed", "dataset_modified", "policy_training_performed", "predictor_training_performed", "hgnn_training_performed", "ppo_training_performed")
    if any(bool(local.get(name)) for name in prohibited) or bool(local.get("round27_created")):
        raise RuntimeError("Round-26 local manifest records a prohibited action")
    frozen = verify_frozen_state()
    run, access = _load(OUTPUT / "run_manifest.json"), _load(access_log)
    action_gate, task_gate, planner_gate, readiness = (
        _load(OUTPUT / name) for name in (
            "action_supervision_ambiguity_gate.json", "task_waypoint_action_disambiguation_gate.json",
            "planner_action_distillation_gate.json", "round26_readiness_gate.json",
        )
    )
    expected_counts = {"h5_train_validation": 5120, "allowed_transitions": 7680, "exact_state_groups": 64, "knn_rows": 5120}
    if run.get("formal_sample_counts") != expected_counts:
        raise RuntimeError("Round-26 frozen sample coverage is incomplete")
    protected = (
        "fresh_new_rollout_count", "round20_development_test_reuse_count", "round25_fresh_new_rollout_count",
        "untouched_eval_access_count", "untouched_eval_model_access_count", "untouched_eval_policy_rollout_count",
    )
    if access.get("allowed_splits_decoded") != ["train", "validation"] or int(access.get("development_test_rows_decoded", -1)) != 0:
        raise RuntimeError("Round-26 decoded a prohibited split")
    if any(int(access.get(name, -1)) != 0 for name in protected) or bool(access.get("fresh_or_untouched_rollout_performed")):
        raise RuntimeError("Round-26 protected-access audit is invalid")
    if not bool(access.get("policy_checkpoint_inference_only")) or access.get("predictor_sha256_before") != access.get("predictor_sha256_after"):
        raise RuntimeError("Round-26 frozen-inference audit is invalid")
    if action_gate.get("gate") not in {"AS-A", "AS-B", "AS-C"} or task_gate.get("gate") not in {"TD-A", "TD-B", "TD-C"} or planner_gate.get("gate") not in {"PAD-A", "PAD-B", "PAD-C"}:
        raise RuntimeError("Round-26 gate is invalid")
    if readiness.get("training_performed") is not False or readiness.get("round27_created") is not False or readiness.get("training_recommendation_only") is not True:
        raise RuntimeError("Round-26 readiness gate violates the authorized scope")
    figures = sorted(path for path in FIGURES.iterdir() if path.suffix in {".png", ".svg"})
    if len(figures) != 36:
        raise RuntimeError("Round-26 requires exactly 18 PNG+SVG pairs")
    _finite_csvs()
    verification = {
        "full_regression_tests": {"passed": True, "count": int(full_test_count), "command": "py -3.14 -m unittest discover -s tests -q"},
        "compileall_passed": bool(compileall_passed), "required_output_missing": missing,
        "figure_count": len(figures), "frozen_state": frozen, "access_audit": access,
        "action_supervision_ambiguity_gate": action_gate["gate"],
        "task_waypoint_action_disambiguation_gate": task_gate["gate"],
        "planner_action_distillation_gate": planner_gate["gate"],
        "readiness_recommendation": readiness["recommendation"],
    }
    lock = {
        "round": "round_26", "status": "locked",
        "topic": "Mixed-Behavior Action-Supervision Ambiguity and Planner-Action Distillation Readiness Audit",
        "snapshot": "backup/snapshots/before_round_26_20260830_1322",
        "previous_round_lock": "round_25/ROUND_LOCK.json", "previous_round_lock_sha256": EXPECTED_R25,
        "configuration_hashes": {name: sha256(ROUND / name) for name in ("AGENTS.md", "configs/round26_action_ambiguity_config.json", "round_manifest.json")},
        "source_tree": _tree(ROOT / "src" / "wt" / "round26"),
        "test_hashes": {"tests/test_round26_action_supervision_ambiguity.py": sha256(ROOT / "tests" / "test_round26_action_supervision_ambiguity.py")},
        "formal_output": OUTPUT.relative_to(ROOT).as_posix(), "figure_directory": FIGURES.relative_to(ROOT).as_posix(),
        "output_hashes": _tree(OUTPUT), "figure_hashes": {path.name: sha256(path) for path in figures},
        "report": report.relative_to(ROOT).as_posix(), "report_sha256": sha256(report),
        "logs": {path.name: sha256(path) for path in (implementation_log, access_log)},
        "final_verification": verification,
        "training_performed": False, "action_supervision_audit_performed": True, "planner_action_audit_performed": True,
        "dataset_modified": False, "policy_training_performed": False, "predictor_training_performed": False,
        "hgnn_training_performed": False, "ppo_training_performed": False,
        "fresh_new_rollout_count": 0, "round20_development_test_reuse_count": 0,
        "round25_fresh_new_rollout_count": 0, "untouched_eval_access_count": 0, "round27_created": False,
        "action_supervision_ambiguity_gate": action_gate["gate"],
        "action_supervision_ambiguity_confirmed": action_gate["action_supervision_ambiguity_confirmed"],
        "task_waypoint_action_disambiguation_gate": task_gate["gate"],
        "planner_action_distillation_gate": planner_gate["gate"],
        "planner_action_distillation_candidate": planner_gate["planner_action_distillation_candidate"],
        "readiness_recommendation": readiness["recommendation"],
    }
    (ROUND / "ROUND_LOCK.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lock
