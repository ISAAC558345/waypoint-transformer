"""Locking helper for the completed Round-24 evidence bundle."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from wt.round24.runner import EXPECTED_R23_LOCK, OUTPUT, ROOT, ROUND, _load_json


REQUIRED_OUTPUTS = (
    "round24_protocol_equivalence_audit.json", "round24_new_seed_training_manifest.json",
    "round24_new_seed_training_summary.csv", "eight_seed_offline_validation.csv",
    "eight_seed_validation_stability.csv", "eight_seed_uncertainty_audit.csv",
    "round24_fresh_manifest.json", "round24_fresh_raw_metrics.csv", "round24_fresh_seed_summary.csv",
    "round24_fresh_uncertainty.csv", "eight_seed_conditional_value.csv", "eight_seed_eta_stability.csv",
    "eight_seed_K_stability.csv", "eight_seed_temporal_value.csv", "eight_seed_variance_comparison.csv",
    "eight_seed_head_utilization.csv", "eight_seed_offline_closed_loop_alignment.csv",
    "round24_seed_stability_gate.json", "round24_readiness_gate.json", "access_audit.json",
    "checkpoint_manifest.json", "run_manifest.json", "round24_pre_fresh_freeze.json",
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
                    try: number = float(value)
                    except (TypeError, ValueError): continue
                    if not math.isfinite(number): raise RuntimeError(f"non-finite Round-24 value in {path.name}")


def finalize(*, full_test_count: int, compileall_passed: bool) -> dict[str, Any]:
    missing = [name for name in REQUIRED_OUTPUTS if not (OUTPUT / name).is_file()]
    if missing: raise RuntimeError(f"Round-24 required output missing: {missing}")
    if not (ROUND / "report" / "round24_completion_report.md").is_file():
        raise RuntimeError("Round-24 completion report missing")
    run = _load_json(OUTPUT / "run_manifest.json"); access = _load_json(OUTPUT / "access_audit.json")
    training = _load_json(OUTPUT / "round24_new_seed_training_manifest.json")
    gate = _load_json(OUTPUT / "round24_seed_stability_gate.json")
    readiness = _load_json(OUTPUT / "round24_readiness_gate.json")
    if training.get("formal_training_completed") != "15/15" or len(training.get("runs", [])) != 15:
        raise RuntimeError("Round-24 training matrix is incomplete")
    if int(access["round24_fresh_access_count"]) != 1 or any(int(access[key]) != 0 for key in ("round22_fresh_new_rollout_count", "round20_development_test_reuse_count", "untouched_eval_access_count")):
        raise RuntimeError("Round-24 protected-access audit is invalid")
    figures = sorted(path for path in (ROOT / "round_24" / "figures" / "round24_multiseed_stability_20260829_1515").iterdir() if path.suffix in {".png", ".svg"})
    if len(figures) != 34: raise RuntimeError("Round-24 figure count is not 17 PNG+SVG pairs")
    _finite_csvs()
    verification = {"full_regression_tests": {"passed": True, "count": int(full_test_count), "command": "py -3.14 -m unittest discover -s tests -q"}, "compileall_passed": bool(compileall_passed), "formal_training_completed": "15/15", "round23_original_lock_sha256": EXPECTED_R23_LOCK, "required_output_missing": missing, "figure_count": len(figures), "access_audit": access, "stability_gate": gate["gate"], "readiness_gate": readiness["gate"], "round25_created": False}
    lock = {"round": "round_24", "status": "locked", "topic": "Pre-Registered Multi-Seed Stability Confirmation for Multi-UAV Task-Waypoint Policy", "snapshot": "backup/snapshots/before_round_24_20260829_1515", "previous_round_lock": "round_23/ROUND_LOCK.json", "previous_round_lock_sha256": EXPECTED_R23_LOCK, "configuration_hashes": {name: sha256(ROUND / name) for name in ("AGENTS.md", "configs/round24_seed_extension_config.json", "round_manifest.json")}, "source_tree": _tree(ROOT / "src" / "wt" / "round24"), "test_hashes": {"tests/test_round24_seed_extension.py": sha256(ROOT / "tests" / "test_round24_seed_extension.py")}, "formal_output": OUTPUT.relative_to(ROOT).as_posix(), "figure_directory": "round_24/figures/round24_multiseed_stability_20260829_1515", "output_hashes": _tree(OUTPUT), "figure_hashes": {path.name: sha256(path) for path in figures}, "report": "round_24/report/round24_completion_report.md", "report_sha256": sha256(ROUND / "report" / "round24_completion_report.md"), "implementation_log": "round_24/logs/implementation_log.md", "implementation_log_sha256": sha256(ROUND / "logs" / "implementation_log.md"), "final_verification": verification, "formal_run_count": 15, "training_performed": True, "seed_extension_performed": True, "predictor_training_performed": False, "hgnn_training_performed": False, "ppo_training_performed": False, "round22_fresh_new_rollout_count": 0, "round20_development_test_reuse_count": 0, "untouched_eval_access_count": 0, "round24_fresh_access_count": 1, "round25_created": False, "seed_stability_gate": gate["gate"], "candidate_ready_for_untouched_eval": gate["candidate_ready_for_untouched_eval"], "conditional_waypoint_value_confirmed": readiness["conditional_waypoint_value_confirmed"]}
    (ROUND / "ROUND_LOCK.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lock
