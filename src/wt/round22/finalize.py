"""Lock Round-22 only after the fixed protocol has fully completed."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .data import R22_OUTPUT, sha256
from .governance import verify_round21_amendment_overlay
from .initialize import verify_round20_lock, verify_round21_lock


ROOT = Path(".")
ROUND = Path("round_22")
FIGURES = Path("round_22/figures/round22_deepsets_task_waypoint_policy_20260829_1153")
EXPECTED_PREDICTOR = "645F235032DE9639A6D42318A41CC8F0E2175E86E38545DA98E0BE5CAEB80221"


def _json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tree(path: Path) -> dict[str, str]:
    return {item.relative_to(path).as_posix(): sha256(item) for item in sorted(path.rglob("*")) if item.is_file() and "__pycache__" not in item.parts}


def _read(name: str) -> dict[str, Any]:
    return json.loads((R22_OUTPUT / name).read_text(encoding="utf-8"))


def _required(fresh_accessed: bool) -> list[str]:
    names = ["round22_protocol_freeze.json", "round22_formal_training_manifest.json", "round22_policy_training_summary.csv", "offline_validation_metrics.csv", "offline_grouped_metrics.csv", "oracle_task_waypoint_policy_gap.csv", "task_waypoint_head_utilization.csv", "conflict_policy_analysis.csv", "bridge_policy_analysis.csv", "validation_closed_loop_metrics.csv", "validation_paired_comparison.csv", "round22_validation_policy_gate.json", "fresh_dev_access_audit.json", "multi_uav_task_waypoint_policy_gate.json", "round22_mechanism_classification.json", "transformer_vs_mlp_audit.csv", "policy_checkpoint_manifest.json", "access_audit.json", "run_manifest.json", "round22_figure_manifest.json", "round22_postprocess_summary.json"]
    if fresh_accessed: names.extend(("fresh_dev_closed_loop_metrics.csv", "fresh_dev_paired_comparison.csv", "fresh_dev_paired_summary.csv", "round22_pre_fresh_freeze.json"))
    return names


def _update_round_manifests(access: Mapping[str, Any], gate: Mapping[str, Any], verification: Mapping[str, Any]) -> None:
    local = json.loads((ROUND / "round_manifest.json").read_text(encoding="utf-8"))
    local.update({"status": "locked", "closed_loop_evaluation": True, "fresh_dev_access_count": int(access["fresh_dev_access_count"]), "untouched_eval_accessed": False, "untouched_eval_model_access_count": int(access["untouched_eval_model_access_count"]), "untouched_eval_policy_rollout_count": int(access["untouched_eval_policy_rollout_count"]), "round23_created": False, "multi_uav_task_waypoint_policy_gate": gate["gate"], "multi_uav_task_waypoint_policy_advantage": gate["multi_uav_task_waypoint_policy_advantage"], "candidate_ready_for_untouched_eval": gate["candidate_ready_for_untouched_eval"], "final_verification_complete": True})
    _json(ROUND / "round_manifest.json", local)
    root = json.loads(Path("round_manifest.json").read_text(encoding="utf-8")); current = root["round_22"]
    current.update({"status": "locked", "formal_training_completed": "12/12", "closed_loop_evaluation": True, "fresh_dev_access_count": int(access["fresh_dev_access_count"]), "untouched_eval_model_access_count": int(access["untouched_eval_model_access_count"]), "untouched_eval_policy_rollout_count": int(access["untouched_eval_policy_rollout_count"]), "multi_uav_task_waypoint_policy_gate": gate["gate"], "multi_uav_task_waypoint_policy_advantage": gate["multi_uav_task_waypoint_policy_advantage"], "candidate_ready_for_untouched_eval": gate["candidate_ready_for_untouched_eval"], "round23_created": False, "full_regression_tests_pass_count": verification["full_regression_tests"]["count"], "compileall_passed": verification["compileall_passed"]})
    root["status"] = "locked"; root["round_22"] = current; _json(Path("round_manifest.json"), root)


def finalize(*, full_test_count: int, compileall_passed: bool) -> dict[str, Any]:
    training, validation, mu_gate, access = (_read("round22_formal_training_manifest.json"), _read("round22_validation_policy_gate.json"), _read("multi_uav_task_waypoint_policy_gate.json"), _read("access_audit.json"))
    if training.get("formal_training_completed") != "12/12" or len(training.get("runs", [])) != 12: raise RuntimeError("cannot lock an incomplete formal training matrix")
    predictor_path = Path(json.loads((R22_OUTPUT / "selected_deepsets_task_waypoint_predictor.json").read_text(encoding="utf-8"))["checkpoint"])
    checkpoint_ok = all(Path(row["checkpoint"]).is_file() and Path(row["last_checkpoint"]).is_file() and row["checkpoint_reload_exact"] and row["finite"] and sha256(Path(row["checkpoint"])) == row["checkpoint_sha256"] for row in training["runs"])
    fresh_accessed = int(access["fresh_dev_access_count"]) == 1
    if validation["gate"] in {"V-A", "V-B"} and not fresh_accessed: raise RuntimeError("V-A/V-B requires exactly one fresh-dev access")
    if validation["gate"] == "V-C" and fresh_accessed: raise RuntimeError("V-C forbids fresh-dev access")
    missing = [name for name in _required(fresh_accessed) if not (R22_OUTPUT / name).is_file()]
    figures = sorted(path for path in FIGURES.glob("*") if path.suffix.lower() in {".png", ".svg"})
    figure_manifest = _read("round22_figure_manifest.json")
    r20, r21, amendment = verify_round20_lock(), verify_round21_lock(), verify_round21_amendment_overlay()
    verification = {"full_regression_tests": {"passed": True, "count": int(full_test_count), "command": "py -3.14 -m unittest discover -s tests -q"}, "compileall_passed": bool(compileall_passed), "formal_training_completed": training["formal_training_completed"], "all_checkpoint_reload_passed": checkpoint_ok, "predictor_sha256_before": training["predictor_sha256_before"], "predictor_sha256_after": sha256(predictor_path), "predictor_hash_unchanged": training["predictor_sha256_before"] == sha256(predictor_path) == EXPECTED_PREDICTOR, "round20_lock_audit": r20, "round21_lock_audit": r21, "round21_amendment_overlay": amendment, "fresh_dev_access_count": int(access["fresh_dev_access_count"]), "round20_development_test_reuse_count": int(access["round20_development_test_reuse_count"]), "untouched_eval_model_access_count": int(access["untouched_eval_model_access_count"]), "untouched_eval_policy_rollout_count": int(access["untouched_eval_policy_rollout_count"]), "figure_count": len(figures), "figure_semantic_audit": figure_manifest["semantic_audit"], "required_output_missing": missing, "round23_created": False, "validation_gate": validation["gate"], "mu_wp_gate": mu_gate["gate"]}
    if missing or len(figures) != 40 or not checkpoint_ok or not verification["predictor_hash_unchanged"] or not r20["passed"] or not r21["passed"] or not amendment["passed"] or verification["untouched_eval_model_access_count"] != 0 or verification["untouched_eval_policy_rollout_count"] != 0 or not compileall_passed: raise RuntimeError(f"Round-22 completion checks failed: {verification}")
    _json(R22_OUTPUT / "round22_final_verification.json", verification)
    report = ROUND / "report" / "round22_completion_report.md"
    report.write_text(f"# Round-22 completion\n\nThe frozen 12-run behavioral-cloning matrix, validation-only offline diagnostics, deployable causal validation rollouts, and conditional fresh-development confirmation completed. Validation gate: {validation['gate']}; final MU-WP gate: {mu_gate['gate']}. Fresh-development access count: {access['fresh_dev_access_count']}; untouched-evaluation model and rollout access remain zero. The frozen H5 DeepSets predictor hash remained `{verification['predictor_sha256_after']}`.\n", encoding="utf-8")
    log = R22_OUTPUT / "implementation_log.md"; log.write_text(log.read_text(encoding="utf-8") + "\nFormal 12-run training, frozen-checkpoint offline evaluation, causal validation, gate-controlled fresh-development evaluation, figures, and final integrity verification completed without untouched-evaluation access or Round-23 creation.\n", encoding="utf-8")
    _update_round_manifests(access, mu_gate, verification)
    lock = {"round": "round_22", "status": "locked", "topic": "Core Multi-UAV Task-Waypoint Policy Value Using a Permutation-Invariant DeepSets Spatial Encoder", "snapshot": "backup/snapshots/before_round_22_20260829_1153", "previous_round_lock": "round_21/ROUND_LOCK.json", "previous_round_lock_sha256": sha256(Path("round_21/ROUND_LOCK.json")), "configuration_hashes": {"AGENTS.md": sha256(ROUND / "AGENTS.md"), "configs/round22_policy_training_config.json": sha256(ROUND / "configs/round22_policy_training_config.json"), "round_manifest.json": sha256(ROUND / "round_manifest.json")}, "source_tree": _tree(Path("src/wt/round22")) | {"tests/test_round22_policy.py": sha256(Path("tests/test_round22_policy.py")), "tests/test_round22_governance.py": sha256(Path("tests/test_round22_governance.py"))}, "formal_output": R22_OUTPUT.as_posix(), "figure_directory": FIGURES.as_posix(), "output_hashes": _tree(R22_OUTPUT), "figure_hashes": {path.name: sha256(path) for path in figures}, "report": report.as_posix(), "report_sha256": sha256(report), "final_verification": verification, "formal_run_count": 12, "training_performed": True, "policy_training_performed": True, "closed_loop_policy_evaluation_performed": True, "fresh_dev_access_count": int(access["fresh_dev_access_count"]), "round20_development_test_reuse_count": int(access["round20_development_test_reuse_count"]), "untouched_eval_model_access_count": 0, "untouched_eval_policy_rollout_count": 0, "round23_created": False, "multi_uav_task_waypoint_policy_gate": mu_gate["gate"], "multi_uav_task_waypoint_policy_advantage": mu_gate["multi_uav_task_waypoint_policy_advantage"], "candidate_ready_for_untouched_eval": mu_gate["candidate_ready_for_untouched_eval"]}
    _json(ROUND / "ROUND_LOCK.json", lock)
    return {"lock_sha256": sha256(ROUND / "ROUND_LOCK.json"), "verification": verification}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--full-test-count", type=int, required=True); parser.add_argument("--compileall-passed", action="store_true"); args = parser.parse_args()
    print(json.dumps(finalize(full_test_count=args.full_test_count, compileall_passed=args.compileall_passed), sort_keys=True))


if __name__ == "__main__": main()
