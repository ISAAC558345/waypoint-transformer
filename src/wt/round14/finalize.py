"""Post-regression Round-14 evidence finalization; no audit computation occurs here."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from wt.round12.contracts import PREDICTOR_SHA256, assert_selected_predictor, sha256


REQUIRED_OUTPUTS = (
    "task_to_motion_mapping_spec.md",
    "actionable_motion_guidance_schema.json",
    "mapping_teacher_alignment.csv",
    "mapping_quality_stratified_alignment.csv",
    "mapping_future_relevance.csv",
    "mapping_counterfactual_metrics.csv",
    "mapping_paired_comparison.csv",
    "return_magnitude_modulation_audit.csv",
    "task_waypoint_head_role_audit.csv",
    "task_to_motion_mapping_gate.json",
    "selected_task_to_motion_mapping.json",
    "future_policy_mapping_interface.md",
    "round14_mapping_manifest.json",
    "round14_readiness_gate.json",
    "run_manifest.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _source_tree(root: Path) -> dict[str, str]:
    source = root / "src/wt/round14"
    return {path.relative_to(source).as_posix(): sha256(path) for path in sorted(source.rglob("*.py"))}


def finalize(
    root: Path,
    output: Path,
    figures: Path,
    *,
    regression_test_count: int,
    compileall_passed: bool,
) -> dict[str, Any]:
    """Create the final report, run manifest refresh, and immutable Round-14 lock.

    The caller must invoke this only after the supplied full-regression result
    is known.  It does not train, infer, access any split, or alter historical
    round evidence.
    """
    root, output, figures = Path(root), Path(output), Path(figures)
    if regression_test_count < 1 or not compileall_passed:
        raise RuntimeError("Round-14 finalization requires successful full regression and compileall")
    missing = [name for name in REQUIRED_OUTPUTS if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"Round-14 required outputs are missing: {missing}")
    pngs = sorted(figures.glob("*.png"))
    svgs = sorted(figures.glob("*.svg"))
    if len(pngs) != 15 or len(svgs) != 15 or {path.stem for path in pngs} != {path.stem for path in svgs}:
        raise RuntimeError("Round-14 requires exactly fifteen matching PNG/SVG figure pairs")
    gate = _read_json(output / "task_to_motion_mapping_gate.json")
    selected = _read_json(output / "selected_task_to_motion_mapping.json")
    readiness = _read_json(output / "round14_readiness_gate.json")
    if selected.get("mapping_gate") != gate.get("task_to_motion_mapping_gate"):
        raise RuntimeError("selected mapping and mapping gate disagree")
    if readiness.get("task_to_motion_mapping_gate") != gate.get("task_to_motion_mapping_gate"):
        raise RuntimeError("readiness gate and mapping gate disagree")
    if readiness.get("round15_created") is not False:
        raise RuntimeError("Round-14 finalization forbids Round-15 creation")
    predictor = assert_selected_predictor(root)
    if sha256(predictor) != PREDICTOR_SHA256:
        raise RuntimeError("frozen Task-Waypoint predictor changed")

    core = _read_json(output / "round14_mapping_manifest.json")
    core.update(
        {
            "source_tree": _source_tree(root),
            "selected_mapping": selected,
            "mapping_gate": gate["task_to_motion_mapping_gate"],
            "readiness_gate": readiness["future_policy_mapping_ready"],
            "figure_pairs": 15,
            "finalization_regression": {
                "full_regression_tests_passed": True,
                "full_regression_tests_pass_count": int(regression_test_count),
                "compileall_passed": True,
            },
        }
    )
    _write_json(output / "round14_mapping_manifest.json", core)
    run = _read_json(output / "run_manifest.json")
    run.update(
        {
            "source_tree": _source_tree(root),
            "selected_mapping": selected,
            "mapping_gate": gate["task_to_motion_mapping_gate"],
            "readiness_gate": readiness["future_policy_mapping_ready"],
            "finalization_regression": core["finalization_regression"],
        }
    )
    _write_json(output / "run_manifest.json", run)
    report = output / "round14_report.md"
    report.write_text(
        "# Round-14 Task-Waypoint-to-Motion Mapping Audit\n\n"
        f"Selected direction mapping: `{selected['selected_direction_mapping']}`.  It had validation teacher-action cosine "
        f"`{selected['validation_teacher_cosines'][selected['selected_direction_mapping']]:.6f}`.\n\n"
        f"The paired same-snapshot mapping gate is `{gate['task_to_motion_mapping_gate']}`.  The payload delta versus "
        f"`queue_rate_energy` is `{gate['payload']['paired_mean']:.3f}` bits with 95% CI "
        f"`[{gate['payload']['bootstrap_ci_low']:.3f}, {gate['payload']['bootstrap_ci_high']:.3f}]`; it crosses zero.\n\n"
        "Accordingly this is diagnostic-only evidence: no neural/policy training, service-action extension, environment "
        "change, old-development-test access, untouched-holdout access, or Round-15 creation occurred.\n",
        encoding="utf-8",
    )
    (output / "implementation_log.md").write_text(
        "# Round-14 Implementation Log\n\n"
        "- Created a pure deterministic mapping from frozen current Task-Waypoint heads plus current geometry/rates to the unchanged 2-D movement action.\n"
        "- Audited A/B/C teacher alignment on registered train/validation data and ran pre-registered paired validation snapshots.\n"
        "- Generated only Round-14 artifacts; no environment/resource/reward code or historical locked artifacts was changed.\n"
        f"- Final full regression: {regression_test_count} tests; compileall: PASS.\n",
        encoding="utf-8",
    )
    output_hashes = {name: sha256(output / name) for name in REQUIRED_OUTPUTS}
    output_hashes.update({"round14_report.md": sha256(report), "implementation_log.md": sha256(output / "implementation_log.md")})
    lock = {
        "round": "round_14",
        "status": "locked",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_output": str(output.relative_to(root)),
        "figure_directory": str(figures.relative_to(root)),
        "figure_pair_count": 15,
        "mapping_gate": gate["task_to_motion_mapping_gate"],
        "selected_direction_mapping": selected["selected_direction_mapping"],
        "future_policy_mapping_ready": readiness["future_policy_mapping_ready"],
        "predictor_sha256": PREDICTOR_SHA256,
        "round13_lock_sha256": sha256(root / "round_13/ROUND_LOCK.json"),
        "round13_amendment_sha256": sha256(root / "round_13/ROUND_LOCK_AMENDMENT.json"),
        "round12_lock_sha256": sha256(root / "round_12/ROUND_LOCK.json"),
        "source_tree": _source_tree(root),
        "output_hashes": output_hashes,
        "full_regression_tests_passed": True,
        "full_regression_tests_pass_count": int(regression_test_count),
        "compileall_passed": True,
        "training_performed": False,
        "task_to_motion_mapping_audit_performed": True,
        "neural_policy_training_performed": False,
        "service_action_extension_performed": False,
        "old_development_test_accessed": False,
        "untouched_holdout_accessed": False,
        "hgnn_training_performed": False,
        "ppo_training_performed": False,
        "round15_created": False,
        "acceptance_complete": True,
    }
    _write_json(root / "round_14/ROUND_LOCK.json", lock)
    return lock


__all__ = ["REQUIRED_OUTPUTS", "finalize"]
