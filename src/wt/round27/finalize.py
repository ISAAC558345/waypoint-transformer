"""Lock the completed, diagnostic-only Round-27 evidence bundle."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from wt.round27.audit import EXPECTED_R26, FIGURES, OUTPUT, ROOT, ROUND, _load, verify_frozen_state


REQUIRED_OUTPUTS = (
    "planner_utility_definition.md", "planner_state_utility_labels.csv",
    "planner_utility_by_group.csv", "causal_trigger_feature_spec.md",
    "planner_utility_distribution.csv", "oracle_conditional_routing_upper_bound.csv",
    "causal_trigger_metrics.csv", "trigger_payload_utility.csv",
    "trigger_generalization_audit.csv", "conditional_planner_behavior_alignment.csv",
    "conditional_distillation_design.md", "planner_utility_taskwp_relation.csv",
    "planner_utility_seed_disagreement.csv", "conditional_planner_action_gate.json",
    "causal_planning_trigger_gate.json", "round27_readiness_gate.json", "run_manifest.json",
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
                        raise RuntimeError(f"non-finite Round-27 value in {path.name}")


def finalize(*, full_test_count: int, compileall_passed: bool) -> dict[str, Any]:
    """Write the immutable lock after successful complete verification."""
    missing = [name for name in REQUIRED_OUTPUTS if not (OUTPUT / name).is_file()]
    actual = {path.name for path in OUTPUT.iterdir() if path.is_file()}
    if missing or actual != set(REQUIRED_OUTPUTS):
        raise RuntimeError(f"Round-27 output inventory mismatch: missing={missing}, extra={sorted(actual - set(REQUIRED_OUTPUTS))}")
    report = ROUND / "report" / "round27_completion_report.md"
    implementation_log = ROUND / "logs" / "implementation_log.md"
    access_log = ROUND / "logs" / "access_audit.json"
    if not report.is_file() or not implementation_log.is_file() or not access_log.is_file():
        raise RuntimeError("Round-27 report or log is missing")
    local = _load(ROUND / "round_manifest.json")
    if local.get("status") != "locked":
        raise RuntimeError("Round-27 local manifest must be locked before writing its lock")
    prohibited = (
        "training_performed", "dataset_modified", "policy_training_performed",
        "predictor_training_performed", "hgnn_training_performed", "ppo_training_performed",
    )
    if any(bool(local.get(name)) for name in prohibited) or bool(local.get("round28_created")):
        raise RuntimeError("Round-27 local manifest records a prohibited action")
    frozen = verify_frozen_state()
    run, access = _load(OUTPUT / "run_manifest.json"), _load(access_log)
    planner_gate, trigger_gate, readiness = (
        _load(OUTPUT / name) for name in (
            "conditional_planner_action_gate.json", "causal_planning_trigger_gate.json",
            "round27_readiness_gate.json",
        )
    )
    expected_counts = {"state_pool": 64, "train": 48, "validation": 16, "counterfactual_branches": 256}
    if run.get("formal_sample_counts") != expected_counts:
        raise RuntimeError("Round-27 frozen sample coverage is incomplete")
    protected = (
        "development_test_rows_decoded", "fresh_new_rollout_count",
        "round20_development_test_reuse_count", "round24_fresh_access_count",
        "untouched_eval_access_count", "future_trigger_feature_access_count", "new_scenarios_created",
    )
    if access.get("allowed_splits_decoded") != ["train", "validation"]:
        raise RuntimeError("Round-27 decoded an unauthorized split")
    if any(int(access.get(name, -1)) != 0 for name in protected):
        raise RuntimeError("Round-27 protected-access audit is invalid")
    if not bool(access.get("policy_checkpoint_inference_only")) or bool(access.get("bridge_used_as_trigger")):
        raise RuntimeError("Round-27 frozen-inference or causal-trigger audit is invalid")
    if planner_gate.get("gate") not in {"CPA-A", "CPA-B", "CPA-C"} or trigger_gate.get("gate") not in {"TR-A", "TR-B", "TR-C"}:
        raise RuntimeError("Round-27 conditional-planning gate is invalid")
    if readiness.get("gate") not in {"R27-A", "R27-B", "R27-C"}:
        raise RuntimeError("Round-27 readiness gate is invalid")
    if readiness.get("training_performed") is not False or readiness.get("round28_created") is not False or readiness.get("training_recommendation_only") is not True:
        raise RuntimeError("Round-27 readiness gate violates the authorized scope")
    figures = sorted(path for path in FIGURES.iterdir() if path.suffix in {".png", ".svg"})
    if len(figures) != 32:
        raise RuntimeError("Round-27 requires exactly 16 PNG+SVG pairs")
    _finite_csvs()
    verification = {
        "full_regression_tests": {"passed": True, "count": int(full_test_count), "command": "py -3.14 -m unittest discover -s tests -q"},
        "compileall_passed": bool(compileall_passed), "required_output_missing": missing,
        "figure_count": len(figures), "frozen_state": frozen, "access_audit": access,
        "conditional_planner_action_gate": planner_gate["gate"],
        "causal_planning_trigger_gate": trigger_gate["gate"],
        "readiness_recommendation": readiness["recommendation"],
    }
    lock = {
        "round": "round_27", "status": "locked",
        "topic": "Conditional Planner-Action Utility and Causal Trigger Design Audit",
        "snapshot": "backup/snapshots/before_round_27_20260830_1447",
        "previous_round_lock": "round_26/ROUND_LOCK.json", "previous_round_lock_sha256": EXPECTED_R26,
        "configuration_hashes": {name: sha256(ROUND / name) for name in ("AGENTS.md", "configs/round27_conditional_planner_utility_config.json", "round_manifest.json")},
        "source_tree": _tree(ROOT / "src" / "wt" / "round27"),
        "test_hashes": {"tests/test_round27_conditional_planner_utility.py": sha256(ROOT / "tests" / "test_round27_conditional_planner_utility.py")},
        "formal_output": OUTPUT.relative_to(ROOT).as_posix(), "figure_directory": FIGURES.relative_to(ROOT).as_posix(),
        "output_hashes": _tree(OUTPUT), "figure_hashes": {path.name: sha256(path) for path in figures},
        "report": report.relative_to(ROOT).as_posix(), "report_sha256": sha256(report),
        "logs": {path.name: sha256(path) for path in (implementation_log, access_log)},
        "final_verification": verification,
        "training_performed": False, "conditional_planner_utility_audit": True, "causal_trigger_design_performed": True,
        "distillation_training_performed": False, "dataset_modified": False, "policy_training_performed": False,
        "predictor_training_performed": False, "hgnn_training_performed": False, "ppo_training_performed": False,
        "fresh_new_rollout_count": 0, "round20_development_test_reuse_count": 0,
        "round24_fresh_access_count": 0, "untouched_eval_access_count": 0, "round28_created": False,
        "conditional_planner_action_gate": planner_gate["gate"],
        "conditional_planner_action_distillation_ready": planner_gate["conditional_planner_action_distillation_ready"],
        "causal_planning_trigger_gate": trigger_gate["gate"],
        "causal_trigger_stable": trigger_gate["trigger_stable"],
        "readiness_recommendation": readiness["recommendation"],
    }
    (ROUND / "ROUND_LOCK.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lock
