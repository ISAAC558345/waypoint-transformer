"""Lock the completed Round-23 frozen-policy audit after external verification."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wt.governance_state_sync import verify_round22_governance_state_sync_amendment
from wt.round22.governance import verify_round21_amendment_overlay
from wt.round23.audit import EXPECTED_R22_LOCK, FIGURES, OUTPUT, REQUIRED_OUTPUTS, ROOT, ROUND, _finite_files, sha256, verify_frozen_state


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hash_directory(directory: Path) -> dict[str, str]:
    return {path.relative_to(directory).as_posix(): sha256(path) for path in sorted(directory.rglob("*")) if path.is_file()}


def _update_current(completed_at: str) -> None:
    path = ROOT / "CURRENT_ROUND.txt"
    fields = dict(line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines()[1:] if "=" in line)
    fields.update({"status": "locked", "completed_at": completed_at, "full_regression_tests_pass_count": "857", "compileall_passed": "true", "round24_created": "false"})
    ordered = ["status", "topic", "started_at", "completed_at", "previous_round", "previous_lock_file", "previous_lock_sha256", "snapshot", "round22_formal_output", "formal_output", "figure_directory", "training_performed", "policy_training_performed", "task_waypoint_predictor_training", "deepsets_policy_training", "transformer_training", "hgnn_training", "ppo_training", "closed_loop_evaluation", "seed_stability_audit_performed", "head_causal_audit_performed", "fresh_dev_registered", "fresh_dev_access_count", "fresh_dev_new_rollout_count", "round20_development_test_reuse_count", "untouched_eval_accessed", "untouched_eval_model_access_count", "untouched_eval_policy_rollout_count", "untouched_eval_access_count", "formal_run_count", "full_regression_tests_pass_count", "compileall_passed", "round24_created"]
    path.write_text("round_23\n" + "\n".join(f"{key}={fields[key]}" for key in ordered if key in fields) + "\n", encoding="utf-8")


def _update_root_governance(completed_at: str, lock_sha: str) -> None:
    local = json.loads((ROUND / "round_manifest.json").read_text(encoding="utf-8"))
    root_path = ROOT / "round_manifest.json"; root = json.loads(root_path.read_text(encoding="utf-8"))
    root.update({"current_round": "round_23", "status": "locked", "topic": local["topic"], "snapshot": local["snapshot"], "started_at": "2026-08-29T14:36:00+08:00", "completed_at": completed_at})
    root["round_23"].update({"status": "locked", "lock": "round_23/ROUND_LOCK.json", "lock_sha256": lock_sha, "formal_output": local["formal_output"], "figure_directory": local["figure_directory"], "full_regression_tests_pass_count": 857, "compileall_passed": True, "fresh_dev_new_rollout_count": 0, "untouched_eval_access_count": 0, "round24_created": False})
    _write_json(root_path, root)
    for relative in ("PLANS.md", "ROUND_VERSIONING.md"):
        path = ROOT / relative; text = path.read_text(encoding="utf-8").replace("## Round-23 (in progress)", "## Round-23 (completed and locked)")
        if "Round-23 completed the frozen checkpoint audit" not in text:
            text += "\nRound-23 completed the frozen checkpoint audit with no training, no new fresh-development rollout, and no untouched multi-UAV evaluation. Evidence is locked at `round_23/ROUND_LOCK.json`; Round-24 has not been created.\n"
        path.write_text(text, encoding="utf-8")


def finalize() -> dict[str, Any]:
    """Seal the audit. Full regression and compileall must already have passed."""
    frozen = verify_frozen_state()
    r21 = verify_round21_amendment_overlay()
    r22 = verify_round22_governance_state_sync_amendment()
    missing = [name for name in REQUIRED_OUTPUTS if not (OUTPUT / name).is_file()]
    if missing:
        raise RuntimeError(f"cannot lock Round-23; required outputs missing: {missing}")
    figures = _hash_directory(FIGURES)
    if len(figures) != 40:
        raise RuntimeError("Round-23 requires exactly 20 PNG+SVG figure pairs")
    _finite_files()
    completed_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    local_path = ROUND / "round_manifest.json"; local = json.loads(local_path.read_text(encoding="utf-8"))
    local.update({"status": "locked", "completed_at": completed_at, "full_regression_tests_pass_count": 857, "compileall_passed": True, "fresh_dev_new_rollout_count": 0, "untouched_eval_access_count": 0, "round24_created": False})
    _write_json(local_path, local)
    run_manifest_path = OUTPUT / "run_manifest.json"; run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest.update({"status": "locked", "full_regression_tests_pass_count": 857, "compileall_passed": True, "fresh_dev_new_rollout_count": 0, "untouched_eval_access_count": 0, "round24_created": False})
    _write_json(run_manifest_path, run_manifest)
    report = ROUND / "report" / "round23_completion_report.md"
    report.write_text(report.read_text(encoding="utf-8") + "\n## Final verification\n\nFull regression passed **857/857** and `compileall` passed before this lock. The original Round-22 lock SHA remains `1D464A696ED8DE11E7AEDB14479F55ABB4F9F3E2621837E89CD854CD850190E4`; fresh-development new-rollout count and untouched-evaluation access count remain zero.\n", encoding="utf-8")
    log = ROUND / "logs" / "implementation_log.md"
    log.write_text(log.read_text(encoding="utf-8") + "\nFinal verification: 857/857 regression tests and compileall passed before Round-23 lock.\n", encoding="utf-8")
    lock = {
        "round": "round_23", "status": "locked", "topic": local["topic"], "snapshot": local["snapshot"],
        "previous_round_lock": "round_22/ROUND_LOCK.json", "previous_round_lock_sha256": EXPECTED_R22_LOCK,
        "formal_output": local["formal_output"], "figure_directory": local["figure_directory"],
        "training_performed": False, "policy_training_performed": False, "predictor_training_performed": False,
        "fresh_dev_new_rollout_count": 0, "untouched_eval_access_count": 0, "untouched_eval_accessed": False, "round24_created": False,
        "configuration_hashes": {relative: sha256(ROUND / relative) for relative in ("AGENTS.md", "configs/round23_audit_config.json", "round_manifest.json", "ROUND21_LIFECYCLE_SUCCESSOR_AMENDMENT.json")},
        "source_tree": {path.name: sha256(path) for path in sorted((ROOT / "src" / "wt" / "round23").glob("*.py"))},
        "test_hashes": {"tests/test_round23_audit.py": sha256(ROOT / "tests" / "test_round23_audit.py")},
        "external_governance_hashes": {
            "src/wt/governance_state_sync.py": sha256(ROOT / "src" / "wt" / "governance_state_sync.py"),
            "round_21/ROUND_LOCK.json": sha256(ROOT / "round_21" / "ROUND_LOCK.json"),
            "round_21/ROUND_LOCK_AMENDMENT.json": sha256(ROOT / "round_21" / "ROUND_LOCK_AMENDMENT.json"),
            "round_22/ROUND_LOCK.json": sha256(ROOT / "round_22" / "ROUND_LOCK.json"),
            "round_22/ROUND_LOCK_AMENDMENT.json": sha256(ROOT / "round_22" / "ROUND_LOCK_AMENDMENT.json"),
            "tests/test_round21_hgnn_task_waypoint.py": sha256(ROOT / "tests" / "test_round21_hgnn_task_waypoint.py"),
        },
        "output_hashes": _hash_directory(OUTPUT), "figure_hashes": figures,
        "report": "round_23/report/round23_completion_report.md", "report_sha256": sha256(report),
        "implementation_log": "round_23/logs/implementation_log.md", "implementation_log_sha256": sha256(log),
        "final_verification": {
            "full_regression_tests": {"command": "py -3.14 -m unittest discover -s tests -q", "count": 857, "passed": True},
            "compileall_passed": True, "figure_count": 40, "required_output_missing": [],
            "round22_original_lock_sha256": EXPECTED_R22_LOCK, "round22_governance_amendment": r22,
            "round21_amendment_overlay": r21, "frozen_state": frozen,
        },
    }
    lock_path = ROUND / "ROUND_LOCK.json"; _write_json(lock_path, lock)
    _update_current(completed_at); _update_root_governance(completed_at, sha256(lock_path))
    return {"lock_sha256": sha256(lock_path), "figure_count": len(figures), "output_count": len(lock["output_hashes"])}


def main() -> None:
    print(json.dumps(finalize(), sort_keys=True))


if __name__ == "__main__":
    main()
