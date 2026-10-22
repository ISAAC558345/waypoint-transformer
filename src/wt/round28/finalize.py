"""Hash and lock the completed diagnostic-only Round-28 evidence bundle."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from wt.round28.audit import EXPECTED_R27, FIGURES, OUTPUT, ROOT, ROUND, _load, verify_frozen_state


INTEGRITY_NAME = "round28_final_integrity_audit.json"
REQUIRED_OUTPUTS = (
    "access_audit.json",
    "composite_trigger_grouped_metrics.csv",
    "composite_trigger_metrics.csv",
    "feature_generalization_audit.csv",
    "planner_utility_state_corpus_manifest.json",
    "planning_composite_score_spec.md",
    "planning_feature_separability.csv",
    "planning_feature_utility_correlation.csv",
    "planning_need_feature_normalization.json",
    "planning_need_feature_spec.md",
    "planning_need_feature_table.csv",
    "planning_need_pca_audit.csv",
    "planning_need_representation_gate.json",
    "planning_pressure_taskwp_relation.csv",
    "planning_utility_seed_disagreement_v2.csv",
    "round27_trigger_revalidation.csv",
    "round28_final_integrity_audit.json",
    "round28_readiness_gate.json",
    "round28_state_sampling_protocol.json",
    "round28_trigger_thresholds.json",
    "run_manifest.json",
    "trigger_model_readiness_gate.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


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
                        raise RuntimeError(f"non-finite Round-28 value in {path.name}")


def _figures() -> tuple[list[Path], dict[str, str]]:
    figures = sorted(path for path in FIGURES.iterdir() if path.is_file() and path.suffix in {".png", ".svg"})
    png = {path.stem for path in figures if path.suffix == ".png"}
    svg = {path.stem for path in figures if path.suffix == ".svg"}
    if len(figures) != 38 or len(png) != 19 or png != svg or any(path.stat().st_size <= 0 for path in figures):
        raise RuntimeError("Round-28 requires exactly 19 nonempty matching PNG+SVG figure pairs")
    return figures, {path.name: sha256(path) for path in figures}


def _assert_output_inventory(*, allow_integrity: bool) -> list[str]:
    actual = {path.name for path in OUTPUT.iterdir() if path.is_file()}
    expected = set(REQUIRED_OUTPUTS if allow_integrity else (name for name in REQUIRED_OUTPUTS if name != INTEGRITY_NAME))
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    if missing or extra:
        raise RuntimeError(f"Round-28 output inventory mismatch: missing={missing}, extra={extra}")
    if any((OUTPUT / name).stat().st_size <= 0 for name in expected):
        raise RuntimeError("Round-28 output inventory contains an empty artifact")
    return missing


def _validate_evidence() -> dict[str, Any]:
    """Validate fixed evidence without replaying the scientific computation."""
    frozen = verify_frozen_state()
    run = _load(OUTPUT / "run_manifest.json")
    access = _load(OUTPUT / "access_audit.json")
    representation = _load(OUTPUT / "planning_need_representation_gate.json")
    trigger = _load(OUTPUT / "trigger_model_readiness_gate.json")
    readiness = _load(OUTPUT / "round28_readiness_gate.json")
    expected_counts = {"state_pool": 256, "train": 192, "validation": 64, "counterfactual_branches": 768}
    if run.get("formal_sample_counts") != expected_counts or run.get("figure_file_count") != 38:
        raise RuntimeError("Round-28 frozen sample coverage or figure count is incomplete")
    protected = (
        "development_test_rows_decoded", "fresh_new_rollout_count", "round20_development_test_reuse_count",
        "round24_fresh_access_count", "untouched_eval_access_count", "future_trigger_feature_access_count",
        "new_scenarios_created",
    )
    if access.get("allowed_splits_decoded") != ["train", "validation"]:
        raise RuntimeError("Round-28 decoded an unauthorized split")
    if any(int(access.get(name, -1)) != 0 for name in protected):
        raise RuntimeError("Round-28 protected-access audit is invalid")
    if access.get("source_behavior_generator") != "joint_h1" or int(access.get("stored_h1_state_rows_decoded", -1)) != 256:
        raise RuntimeError("Round-28 did not use the fixed Joint-H1 state corpus")
    if int(access.get("counterfactual_diagnostic_branch_count", -1)) != 768:
        raise RuntimeError("Round-28 counterfactual branch count is invalid")
    if not bool(access.get("policy_checkpoint_inference_only")) or bool(access.get("bridge_used_as_trigger")):
        raise RuntimeError("Round-28 frozen-inference or bridge restriction is invalid")
    if representation.get("gate") != "REP-B" or trigger.get("gate") != "REP-B" or readiness.get("gate") != "R28-B":
        raise RuntimeError("Round-28 recorded gates are inconsistent with the fixed completed evidence")
    if representation.get("causal_planning_need_separable") is not False or representation.get("minimal_trigger_model_training_ready") is not False:
        raise RuntimeError("Round-28 representation gate unexpectedly authorizes training")
    if trigger.get("minimal_trigger_model_training_ready") is not False:
        raise RuntimeError("Round-28 trigger readiness gate unexpectedly authorizes training")
    if any(gate.get(name) is not False for gate in (representation, readiness) for name in ("training_performed",)):
        raise RuntimeError("Round-28 gate records prohibited training")
    if readiness.get("trigger_model_training_performed") is not False or readiness.get("round29_created") is not False:
        raise RuntimeError("Round-28 readiness scope is invalid")
    _finite_csvs()
    figures, figure_hashes = _figures()
    return {
        "frozen_state": frozen,
        "run_manifest": run,
        "access_audit": access,
        "representation_gate": representation,
        "trigger_model_readiness_gate": trigger,
        "round28_readiness_gate": readiness,
        "figures": figures,
        "figure_hashes": figure_hashes,
    }


def write_final_integrity_audit(*, full_test_count: int, compileall_passed: bool) -> dict[str, Any]:
    """Write the one permitted post-computation integrity artifact before locking."""
    if (OUTPUT / INTEGRITY_NAME).exists():
        raise RuntimeError("Round-28 integrity audit already exists and is append-only")
    _assert_output_inventory(allow_integrity=False)
    report = ROUND / "report" / "round28_completion_report.md"
    implementation_log = ROUND / "logs" / "implementation_log.md"
    if not report.is_file() or not implementation_log.is_file() or report.stat().st_size <= 0 or implementation_log.stat().st_size <= 0:
        raise RuntimeError("Round-28 report or implementation log is missing or empty")
    local = _load(ROUND / "round_manifest.json")
    if local.get("status") != "locked":
        raise RuntimeError("Round-28 local manifest must be locked before writing its integrity audit")
    evidence = _validate_evidence()
    output_hashes = _tree(OUTPUT)
    if INTEGRITY_NAME in output_hashes:
        raise RuntimeError("Round-28 integrity audit must be written after its scientific evidence hashes")
    audit = {
        "round": "round_28",
        "audit_type": "final_integrity_audit",
        "status": "verified_prelock",
        "previous_round_lock": "round_27/ROUND_LOCK.json",
        "previous_round_lock_sha256": EXPECTED_R27,
        "configuration_hashes": {
            name: sha256(ROUND / name)
            for name in ("AGENTS.md", "configs/round28_planning_need_representation_config.json", "round_manifest.json")
        },
        "source_tree": _tree(ROOT / "src" / "wt" / "round28"),
        "test_hashes": {"tests/test_round28_planning_need_representation.py": sha256(ROOT / "tests" / "test_round28_planning_need_representation.py")},
        "formal_output_inventory_excluding_self": sorted(output_hashes),
        "formal_output_hashes_excluding_self": output_hashes,
        "figure_hashes": evidence["figure_hashes"],
        "figure_manifest_sha256": _canonical_sha256(evidence["figure_hashes"]),
        "report_sha256": sha256(report),
        "implementation_log_sha256": sha256(implementation_log),
        "full_regression_tests": {
            "passed": True,
            "count": int(full_test_count),
            "command": "py -3.14 -m unittest discover -s tests -q",
        },
        "compileall": {
            "passed": bool(compileall_passed),
            "command": "py -3.14 -m compileall -q src tests experiments",
        },
        "formal_sample_counts": evidence["run_manifest"]["formal_sample_counts"],
        "protected_access_counts": {
            name: evidence["access_audit"].get(name)
            for name in (
                "development_test_rows_decoded", "fresh_new_rollout_count", "round20_development_test_reuse_count",
                "round24_fresh_access_count", "untouched_eval_access_count", "future_trigger_feature_access_count",
                "new_scenarios_created",
            )
        },
        "gate_record": {
            "representation": evidence["representation_gate"]["gate"],
            "trigger_readiness": evidence["trigger_model_readiness_gate"]["gate"],
            "round_readiness": evidence["round28_readiness_gate"]["gate"],
            "causal_planning_need_separable": evidence["representation_gate"]["causal_planning_need_separable"],
            "minimal_trigger_model_training_ready": evidence["representation_gate"]["minimal_trigger_model_training_ready"],
            "recommendation": evidence["round28_readiness_gate"]["recommendation"],
        },
        "frozen_state": evidence["frozen_state"],
        "scientific_outputs_modified_after_completion": False,
        "serialization_or_manifest_repairs": [],
        "training_performed": False,
        "trigger_model_training_performed": False,
        "fresh_new_rollout_count": 0,
        "untouched_eval_access_count": 0,
        "round29_created": False,
    }
    (OUTPUT / INTEGRITY_NAME).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def _verify_integrity_audit() -> dict[str, Any]:
    audit = _load(OUTPUT / INTEGRITY_NAME)
    expected = set(REQUIRED_OUTPUTS) - {INTEGRITY_NAME}
    hashes = audit.get("formal_output_hashes_excluding_self", {})
    if set(hashes) != expected or any(sha256(OUTPUT / name) != value for name, value in hashes.items()):
        raise RuntimeError("Round-28 prelock scientific output hash audit is invalid")
    figures, figure_hashes = _figures()
    if audit.get("figure_hashes") != figure_hashes or audit.get("figure_manifest_sha256") != _canonical_sha256(figure_hashes):
        raise RuntimeError("Round-28 figure integrity audit is invalid")
    if audit.get("previous_round_lock_sha256") != EXPECTED_R27 or audit.get("scientific_outputs_modified_after_completion") is not False:
        raise RuntimeError("Round-28 final integrity audit has an invalid provenance record")
    return audit


def finalize(*, full_test_count: int, compileall_passed: bool) -> dict[str, Any]:
    """Write the immutable lock only after the final integrity audit succeeds."""
    if (ROUND / "ROUND_LOCK.json").exists():
        raise RuntimeError("Round-28 is already locked; its lock is append-only")
    missing = _assert_output_inventory(allow_integrity=True)
    report = ROUND / "report" / "round28_completion_report.md"
    implementation_log = ROUND / "logs" / "implementation_log.md"
    local = _load(ROUND / "round_manifest.json")
    if local.get("status") != "locked":
        raise RuntimeError("Round-28 local manifest must be locked before writing its lock")
    prohibited = (
        "training_performed", "trigger_model_training_performed", "dataset_modified", "policy_training_performed",
        "predictor_training_performed", "hgnn_training_performed", "ppo_training_performed",
    )
    if any(bool(local.get(name)) for name in prohibited) or bool(local.get("round29_created")):
        raise RuntimeError("Round-28 local manifest records a prohibited action")
    evidence = _validate_evidence()
    integrity = _verify_integrity_audit()
    lock = {
        "round": "round_28",
        "status": "locked",
        "topic": "Current-State Planning-Need Representation and Causal Separability Audit",
        "snapshot": "backup/snapshots/before_round_28_20260830_1624",
        "previous_round_lock": "round_27/ROUND_LOCK.json",
        "previous_round_lock_sha256": EXPECTED_R27,
        "configuration_hashes": {
            name: sha256(ROUND / name)
            for name in ("AGENTS.md", "configs/round28_planning_need_representation_config.json", "round_manifest.json")
        },
        "source_tree": _tree(ROOT / "src" / "wt" / "round28"),
        "test_hashes": {"tests/test_round28_planning_need_representation.py": sha256(ROOT / "tests" / "test_round28_planning_need_representation.py")},
        "formal_output": OUTPUT.relative_to(ROOT).as_posix(),
        "figure_directory": FIGURES.relative_to(ROOT).as_posix(),
        "output_hashes": _tree(OUTPUT),
        "figure_hashes": evidence["figure_hashes"],
        "report": report.relative_to(ROOT).as_posix(),
        "report_sha256": sha256(report),
        "logs": {implementation_log.name: sha256(implementation_log)},
        "final_integrity_audit": {
            "path": (OUTPUT / INTEGRITY_NAME).relative_to(ROOT).as_posix(),
            "sha256": sha256(OUTPUT / INTEGRITY_NAME),
            "scientific_output_hashes_excluding_self": integrity["formal_output_hashes_excluding_self"],
            "figure_manifest_sha256": integrity["figure_manifest_sha256"],
        },
        "final_verification": {
            "full_regression_tests": {"passed": True, "count": int(full_test_count), "command": "py -3.14 -m unittest discover -s tests -q"},
            "compileall_passed": bool(compileall_passed),
            "required_output_missing": missing,
            "figure_count": len(evidence["figures"]),
            "frozen_state": evidence["frozen_state"],
            "access_audit": evidence["access_audit"],
            "representation_gate": evidence["representation_gate"]["gate"],
            "trigger_model_readiness_gate": evidence["trigger_model_readiness_gate"]["gate"],
            "readiness_recommendation": evidence["round28_readiness_gate"]["recommendation"],
        },
        "training_performed": False,
        "trigger_model_training_performed": False,
        "planning_need_representation_audit": True,
        "planner_utility_replay_performed": True,
        "dataset_modified": False,
        "policy_training_performed": False,
        "predictor_training_performed": False,
        "hgnn_training_performed": False,
        "ppo_training_performed": False,
        "fresh_new_rollout_count": 0,
        "round20_development_test_reuse_count": 0,
        "round24_fresh_access_count": 0,
        "untouched_eval_access_count": 0,
        "round29_created": False,
        "representation_gate": evidence["representation_gate"]["gate"],
        "causal_planning_need_separable": evidence["representation_gate"]["causal_planning_need_separable"],
        "minimal_trigger_model_training_ready": evidence["representation_gate"]["minimal_trigger_model_training_ready"],
        "readiness_recommendation": evidence["round28_readiness_gate"]["recommendation"],
    }
    (ROUND / "ROUND_LOCK.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lock
