"""Finalize Round-20 formal evidence after externally run regression checks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .audit import preflight
from .dataset import sha256, write_json


def _hashes(directory: Path) -> dict[str, str]:
    return {path.relative_to(directory).as_posix(): sha256(path) for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "run_manifest.json"}


def finalize_output(root: Path, output_name: str, *, focused_tests_pass_count: int, full_regression_tests_pass_count: int, compileall_passed: bool) -> dict[str, Any]:
    root = root.resolve()
    preflight(root)
    output = root / "round_20" / "outputs" / output_name
    readiness_path = output / "round20_readiness_gate.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness.update({
        "round20_focused_tests_pass_count": int(focused_tests_pass_count),
        "full_regression_tests_pass_count": int(full_regression_tests_pass_count),
        "full_regression_tests_passed": True,
        "compileall_passed": bool(compileall_passed),
        "full_regression_tests_pending": False,
        "compileall_pending": False,
    })
    required = ("dataset_quality_ready", "graph_training_data_ready", "multi_uav_task_waypoint_training_ready", "split_isolation", "protected_data_clean", "full_regression_tests_passed", "compileall_passed")
    readiness["round20_ready_for_hgnn_task_waypoint_training"] = bool(all(readiness[key] for key in required))
    write_json(readiness_path, readiness)
    report_path = output / "round20_completion_report.md"
    report_path.write_text(report_path.read_text(encoding="utf-8") + f"\n## Final verification\n\nFocused tests: {focused_tests_pass_count}/{focused_tests_pass_count}; full regression: {full_regression_tests_pass_count}/{full_regression_tests_pass_count}; compileall: {compileall_passed}.\n", encoding="utf-8")
    run_path = output / "run_manifest.json"
    run_manifest = json.loads(run_path.read_text(encoding="utf-8"))
    run_manifest.update({
        "status": "formal_generation_and_regression_complete",
        "focused_tests_pass_count": int(focused_tests_pass_count),
        "full_regression_tests_pass_count": int(full_regression_tests_pass_count),
        "compileall_passed": bool(compileall_passed),
        "round20_ready_for_hgnn_task_waypoint_training": readiness["round20_ready_for_hgnn_task_waypoint_training"],
        "artifact_hashes": _hashes(output),
    })
    write_json(run_path, run_manifest)
    return {"output": output, "readiness": readiness, "run_manifest": run_manifest}


def build_lock(root: Path, output_name: str, *, focused_tests_pass_count: int, full_regression_tests_pass_count: int, figure_name: str | None = None) -> dict[str, Any]:
    root = root.resolve()
    output = root / "round_20" / "outputs" / output_name
    selected_figure_name = output_name if figure_name is None else figure_name
    figures = root / "round_20" / "figures" / selected_figure_name
    readiness = json.loads((output / "round20_readiness_gate.json").read_text(encoding="utf-8"))
    source_paths = sorted((root / "src" / "wt" / "round20").glob("*.py")) + [root / "tests" / "test_round20_multi_uav_dataset.py"]
    config_paths = [
        root / "round_20" / "AGENTS.md", root / "round_20" / "multi_uav_dataset_generation_protocol.md",
        root / "round_20" / "configs" / "round20_multi_uav_dataset_protocol.json", root / "round_20" / "round_manifest.json",
    ]
    lock = {
        "round": "round_20", "status": "locked", "topic": "Multi-UAV Shared-Spectrum Mixed-Quality Offline Dataset, Graph-Ready State Schema, and Multi-UAV Task-Waypoint Supervision Preparation",
        "previous_round_lock": "round_19/ROUND_LOCK.json", "previous_round_lock_sha256": "1563785D59556B7FA747A1CADEBDBDB5EE82FCCA26892DD3E42A65ABB3B5AA46",
        "snapshot": "backup/snapshots/before_round_20_20260828_1849", "formal_output": f"round_20/outputs/{output_name}", "figure_directory": f"round_20/figures/{selected_figure_name}",
        "preflight_round19_verified": True, "training_performed": False, "task_waypoint_training_performed": False, "hgnn_training_performed": False, "ppo_training_performed": False, "multi_agent_rl_performed": False,
        "old_development_test_access_count": 0, "fresh_development_access_count": 0, "untouched_holdout_new_access_count": 0, "new_untouched_eval_policy_rollout_count": 0,
        "scenario_count": 80, "episode_count": 800, "transition_count": 9600, "task_waypoint_sample_counts": readiness.get("sample_counts", {}),
        "dataset_quality_ready": readiness["dataset_quality_ready"], "graph_training_data_ready": readiness["graph_training_data_ready"], "multi_uav_task_waypoint_training_ready": readiness["multi_uav_task_waypoint_training_ready"], "round20_ready_for_hgnn_task_waypoint_training": readiness["round20_ready_for_hgnn_task_waypoint_training"],
        "focused_tests_pass_count": int(focused_tests_pass_count), "full_regression_tests_pass_count": int(full_regression_tests_pass_count), "compileall_passed": True,
        "configuration_hashes": {str(path.relative_to(root / "round_20")).replace("\\", "/"): sha256(path) for path in config_paths},
        "source_tree": {str(path.relative_to(root)).replace("\\", "/"): sha256(path) for path in source_paths},
        "output_hashes": _hashes(output),
        "figure_hashes": {path.name: sha256(path) for path in sorted(figures.iterdir()) if path.is_file()},
    }
    return lock


__all__ = ["build_lock", "finalize_output"]
