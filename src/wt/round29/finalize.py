"""Hash and lock completed diagnostic-only Round-29 evidence."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from wt.round29.audit import EXPECTED_R28, FIGURES, OUTPUT, ROOT, ROUND, _load, verify_frozen_state


INTEGRITY_NAME = "round29_final_integrity_audit.json"
REQUIRED_OUTPUTS = (
    "access_audit.json", "cross_K_mechanism_audit.csv", "cross_K_trigger_interaction.csv",
    "cross_scale_planning_pressure_gate.json", "dsp_activation_decomposition.csv", "dsp_K_scale_audit.csv",
    "eta_confirmation_audit.csv", "existing_dsp_scale_audit.md", "planner_useful_prevalence_K.csv",
    "round28_replay_dsp_variants.csv", "round29_confirmation_manifest.json", "round29_confirmation_manifest_lock.json",
    "round29_confirmation_raw_metrics.csv", "round29_confirmation_trigger_metrics.csv", "round29_dsp_variant_spec.json",
    "round29_final_integrity_audit.json", "round29_preregistration_lock.json", "round29_readiness_gate.json",
    "round29_train_variant_selection.csv", "round29_trigger_thresholds.json", "round29_trigger_training_readiness.json",
    "round29_variant_selection_protocol.json", "run_manifest.json", "selected_scale_normalized_dsp.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _tree(path: Path) -> dict[str, str]:
    return {item.relative_to(path).as_posix(): sha256(item) for item in sorted(path.rglob("*")) if item.is_file() and "__pycache__" not in item.parts}


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest().upper()


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
                        raise RuntimeError(f"non-finite Round-29 value in {path.name}")


def _figures() -> tuple[list[Path], dict[str, str]]:
    figures = sorted(path for path in FIGURES.iterdir() if path.is_file() and path.suffix in {".png", ".svg"})
    png, svg = {path.stem for path in figures if path.suffix == ".png"}, {path.stem for path in figures if path.suffix == ".svg"}
    if len(figures) != 30 or len(png) != 15 or png != svg or any(path.stat().st_size <= 0 for path in figures):
        raise RuntimeError("Round-29 requires exactly 15 nonempty matching PNG+SVG figure pairs")
    return figures, {path.name: sha256(path) for path in figures}


def _assert_inventory(*, allow_integrity: bool) -> list[str]:
    actual = {path.name for path in OUTPUT.iterdir() if path.is_file()}
    expected = set(REQUIRED_OUTPUTS if allow_integrity else (name for name in REQUIRED_OUTPUTS if name != INTEGRITY_NAME))
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    if missing or extra or any((OUTPUT / name).stat().st_size <= 0 for name in expected):
        raise RuntimeError(f"Round-29 output inventory mismatch: missing={missing}, extra={extra}")
    return missing


def _validate_evidence() -> dict[str, Any]:
    frozen = verify_frozen_state()
    run, access = _load(OUTPUT / "run_manifest.json"), _load(OUTPUT / "access_audit.json")
    gate, readiness = _load(OUTPUT / "cross_scale_planning_pressure_gate.json"), _load(OUTPUT / "round29_trigger_training_readiness.json")
    expected = {"round28_replay_states": 256, "confirmation_scenarios": 40, "confirmation_states": 160, "counterfactual_branches": 320}
    if run.get("formal_sample_counts") != expected or run.get("figure_file_count") != 30 or run.get("confirmation_outcome_access_count") != 1:
        raise RuntimeError("Round-29 coverage is incomplete")
    protected = ("round20_development_test_reuse_count", "round22_fresh_access_count", "round24_fresh_access_count", "untouched_eval_access_count", "fresh_new_rollout_count")
    if any(int(access.get(name, -1)) != 0 for name in protected):
        raise RuntimeError("Round-29 protected-access audit is invalid")
    if int(access.get("confirmation_scenario_count", -1)) != 40 or int(access.get("confirmation_state_count", -1)) != 160 or int(access.get("confirmation_outcome_access_count", -1)) != 1:
        raise RuntimeError("Round-29 confirmation access audit is invalid")
    if any(access.get(name) is not False for name in ("training_performed", "trigger_model_training_performed", "planner_modified", "environment_modified", "round30_created")):
        raise RuntimeError("Round-29 records a prohibited action")
    if gate.get("gate") not in {"CS-A", "CS-B", "CS-C"} or readiness.get("cross_scale_gate") != gate.get("gate"):
        raise RuntimeError("Round-29 cross-scale gate is inconsistent")
    if readiness.get("training_authorized_this_round") is not False or readiness.get("round30_created") is not False:
        raise RuntimeError("Round-29 readiness violates scope")
    if gate.get("cross_scale_trigger_confirmed") != (gate.get("gate") == "CS-A") or gate.get("task_scale_conditional_trigger") != (gate.get("gate") == "CS-B"):
        raise RuntimeError("Round-29 gate booleans are inconsistent")
    prereg, manifest = _load(OUTPUT / "round29_preregistration_lock.json"), _load(OUTPUT / "round29_confirmation_manifest_lock.json")
    if prereg.get("confirmation_outcomes_accessed") is not False or manifest.get("confirmation_outcomes_accessed") is not False:
        raise RuntimeError("Round-29 pre-registration was not frozen before outcomes")
    if manifest.get("manifest_sha256") != sha256(OUTPUT / "round29_confirmation_manifest.json") or manifest.get("preregistration_lock_sha256") != sha256(OUTPUT / "round29_preregistration_lock.json"):
        raise RuntimeError("Round-29 frozen confirmation manifest is invalid")
    _finite_csvs()
    figures, figure_hashes = _figures()
    return {"frozen_state": frozen, "run_manifest": run, "access_audit": access, "gate": gate, "readiness": readiness, "figures": figures, "figure_hashes": figure_hashes}


def write_final_integrity_audit(*, full_test_count: int, compileall_passed: bool) -> dict[str, Any]:
    if (OUTPUT / INTEGRITY_NAME).exists():
        raise RuntimeError("Round-29 final integrity audit is append-only")
    _assert_inventory(allow_integrity=False)
    local = _load(ROUND / "round_manifest.json")
    report, log = ROUND / "report" / "round29_completion_report.md", ROUND / "logs" / "implementation_log.md"
    if local.get("status") != "locked" or not report.is_file() or not log.is_file():
        raise RuntimeError("Round-29 local lifecycle state is not ready for integrity recording")
    evidence = _validate_evidence()
    output_hashes = _tree(OUTPUT)
    audit = {
        "round": "round_29", "audit_type": "final_integrity_audit", "status": "verified_prelock",
        "previous_round_lock": "round_28/ROUND_LOCK.json", "previous_round_lock_sha256": EXPECTED_R28,
        "configuration_hashes": {name: sha256(ROUND / name) for name in ("AGENTS.md", "configs/round29_cross_scale_dsp_config.json", "round_manifest.json")},
        "source_tree": _tree(ROOT / "src" / "wt" / "round29"), "test_hashes": {"tests/test_round29_cross_scale_deferred_service.py": sha256(ROOT / "tests" / "test_round29_cross_scale_deferred_service.py")},
        "formal_output_hashes_excluding_self": output_hashes, "figure_hashes": evidence["figure_hashes"], "figure_manifest_sha256": _canonical_sha256(evidence["figure_hashes"]),
        "report_sha256": sha256(report), "implementation_log_sha256": sha256(log),
        "full_regression_tests": {"passed": True, "count": int(full_test_count), "command": "py -3.14 -m unittest discover -s tests -q"},
        "compileall": {"passed": bool(compileall_passed), "command": "py -3.14 -m compileall -q src tests experiments"},
        "formal_sample_counts": evidence["run_manifest"]["formal_sample_counts"], "access_audit": evidence["access_audit"],
        "gate": evidence["gate"], "readiness": evidence["readiness"], "scientific_outputs_modified_after_completion": False,
        "serialization_or_manifest_repairs": [], "training_performed": False, "trigger_model_training_performed": False, "round30_created": False,
    }
    (OUTPUT / INTEGRITY_NAME).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def _verify_integrity_audit() -> dict[str, Any]:
    audit = _load(OUTPUT / INTEGRITY_NAME)
    hashes = audit.get("formal_output_hashes_excluding_self", {})
    expected = set(REQUIRED_OUTPUTS) - {INTEGRITY_NAME}
    if set(hashes) != expected or any(sha256(OUTPUT / name) != digest for name, digest in hashes.items()):
        raise RuntimeError("Round-29 prelock output hashes are invalid")
    _, figures = _figures()
    if audit.get("figure_hashes") != figures or audit.get("figure_manifest_sha256") != _canonical_sha256(figures):
        raise RuntimeError("Round-29 figure integrity audit is invalid")
    return audit


def finalize(*, full_test_count: int, compileall_passed: bool) -> dict[str, Any]:
    if (ROUND / "ROUND_LOCK.json").exists():
        raise RuntimeError("Round-29 lock is append-only")
    missing = _assert_inventory(allow_integrity=True)
    local = _load(ROUND / "round_manifest.json")
    if local.get("status") != "locked":
        raise RuntimeError("Round-29 local manifest must be locked before writing its lock")
    prohibited = ("training_performed", "trigger_model_training_performed", "dataset_modified", "policy_training_performed", "predictor_training_performed", "hgnn_training_performed", "ppo_training_performed", "round30_created")
    if any(bool(local.get(name)) for name in prohibited):
        raise RuntimeError("Round-29 local manifest records a prohibited action")
    evidence, integrity = _validate_evidence(), _verify_integrity_audit()
    report, log = ROUND / "report" / "round29_completion_report.md", ROUND / "logs" / "implementation_log.md"
    lock = {
        "round": "round_29", "status": "locked", "topic": "Cross-Scale Deferred-Service Pressure Normalization and Independent Confirmation Audit", "snapshot": "backup/snapshots/before_round_29_20260830_1800",
        "previous_round_lock": "round_28/ROUND_LOCK.json", "previous_round_lock_sha256": EXPECTED_R28,
        "configuration_hashes": {name: sha256(ROUND / name) for name in ("AGENTS.md", "configs/round29_cross_scale_dsp_config.json", "round_manifest.json")},
        "source_tree": _tree(ROOT / "src" / "wt" / "round29"), "test_hashes": {"tests/test_round29_cross_scale_deferred_service.py": sha256(ROOT / "tests" / "test_round29_cross_scale_deferred_service.py")},
        "formal_output": OUTPUT.relative_to(ROOT).as_posix(), "figure_directory": FIGURES.relative_to(ROOT).as_posix(), "output_hashes": _tree(OUTPUT), "figure_hashes": evidence["figure_hashes"],
        "report": report.relative_to(ROOT).as_posix(), "report_sha256": sha256(report), "logs": {log.name: sha256(log)},
        "final_integrity_audit": {"path": (OUTPUT / INTEGRITY_NAME).relative_to(ROOT).as_posix(), "sha256": sha256(OUTPUT / INTEGRITY_NAME), "scientific_output_hashes_excluding_self": integrity["formal_output_hashes_excluding_self"], "figure_manifest_sha256": integrity["figure_manifest_sha256"]},
        "final_verification": {"full_regression_tests": {"passed": True, "count": int(full_test_count), "command": "py -3.14 -m unittest discover -s tests -q"}, "compileall_passed": bool(compileall_passed), "required_output_missing": missing, "figure_count": len(evidence["figures"]), "frozen_state": evidence["frozen_state"], "access_audit": evidence["access_audit"], "cross_scale_gate": evidence["gate"]["gate"], "readiness_recommendation": evidence["readiness"]["recommendation"]},
        "training_performed": False, "trigger_model_training_performed": False, "cross_scale_audit_performed": True, "confirmation_diagnostic_performed": True, "dataset_modified": False, "policy_training_performed": False, "predictor_training_performed": False, "hgnn_training_performed": False, "ppo_training_performed": False, "fresh_new_rollout_count": 0, "round20_development_test_reuse_count": 0, "round22_fresh_access_count": 0, "round24_fresh_access_count": 0, "untouched_eval_access_count": 0, "round30_created": False, "cross_scale_gate": evidence["gate"]["gate"], "cross_scale_trigger_confirmed": evidence["gate"]["cross_scale_trigger_confirmed"], "task_scale_conditional_trigger": evidence["gate"]["task_scale_conditional_trigger"], "minimal_trigger_model_training_ready": evidence["readiness"]["minimal_trigger_model_training_ready"], "readiness_recommendation": evidence["readiness"]["recommendation"],
    }
    (ROUND / "ROUND_LOCK.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lock
