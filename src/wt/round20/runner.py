"""Formal Round-20 non-learning mixed-quality dataset entrypoint."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .audit import preflight, write_audits
from .dataset import episode_metrics, generate_dataset, reconstruct_formal_result, sha256, write_csv, write_json
from .reporting import figure_manifest, render_figures, write_completion_report
from .scenarios import untouched_evaluation_scenarios


FORMAL_STEM = "round20_multi_uav_dataset_20260828_1950"


def _artifact_hashes(directory: Path) -> dict[str, str]:
    return {path.relative_to(directory).as_posix(): sha256(path) for path in sorted(directory.rglob("*")) if path.is_file()}


def run(root: Path, *, output_name: str = FORMAL_STEM) -> dict[str, Any]:
    root = root.resolve()
    preflight_result = preflight(root)
    output = root / "round_20" / "outputs" / output_name
    figures = root / "round_20" / "figures" / output_name
    result = generate_dataset(root, output)
    audits = write_audits(output, result)
    untouched = {
        "namespace": "round20_multi_uav_untouched_eval_v1",
        "scenario_count": len(untouched_evaluation_scenarios()),
        "scenarios": list(untouched_evaluation_scenarios()),
        "untouched_eval_policy_rollout_count": 0,
        "outcomes_read": False,
        "training_or_validation_use": False,
    }
    write_json(output / "round20_multi_uav_untouched_eval_manifest.json", untouched)
    protocol = root / "round_20" / "multi_uav_dataset_generation_protocol.md"
    (output / "multi_uav_dataset_generation_protocol.md").write_text(protocol.read_text(encoding="utf-8"), encoding="utf-8")
    config = json.loads((root / "round_20" / "configs" / "round20_multi_uav_dataset_protocol.json").read_text(encoding="utf-8"))
    write_json(output / "run_config.json", config)
    readiness = {
        "dataset_quality_ready": audits["quality_gate"]["dataset_quality_ready"],
        "graph_training_data_ready": audits["graph_gate"]["graph_training_data_ready"],
        "multi_uav_task_waypoint_training_ready": audits["task_gate"]["multi_uav_task_waypoint_training_ready"],
        "split_isolation": audits["split"]["scenario_split_isolation"],
        "protected_data_clean": True,
        "new_untouched_eval_policy_rollout_count": 0,
        "historical_protected_access_counts": {"old_development": 0, "fresh_development": 0, "historical_untouched": 0},
        "full_regression_tests_pending": True,
        "compileall_pending": True,
    }
    readiness["round20_ready_for_hgnn_task_waypoint_training"] = bool(all(readiness[key] for key in ("dataset_quality_ready", "graph_training_data_ready", "multi_uav_task_waypoint_training_ready", "split_isolation", "protected_data_clean")))
    write_json(output / "round20_readiness_gate.json", readiness)
    render_figures(figures, result, audits)
    write_completion_report(output / "round20_completion_report.md", result, audits, readiness, figures)
    implementation_log = "Round-20 deterministic, non-learning construction.  Verified Round-19 lock; generated only pre-registered development trajectories, graph states, and H=5/H=10 supervision.  No neural training, protected-data access, or untouched policy rollout.\n"
    (output / "implementation_log.md").write_text(implementation_log, encoding="utf-8")
    manifest = {
        "round": "round_20", "output_name": output_name, "status": "formal_generation_complete_pending_regression",
        "preflight": preflight_result, "scenario_count": len(result["scenarios"]), "episode_count": len(result["episodes"]),
        "transition_count": len(result["transitions"]), "task_waypoint_sample_count": len(result["samples"]),
        "task_waypoint_counts": audits["task"]["counts"], "figure_count": len(figure_manifest(figures)),
        "figure_hashes": figure_manifest(figures), "dataset_quality_gate": audits["quality_gate"]["dataset_quality_ready"],
        "graph_readiness_gate": audits["graph_gate"]["graph_training_data_ready"],
        "task_waypoint_readiness_gate": audits["task_gate"]["multi_uav_task_waypoint_training_ready"],
        "round20_ready_for_hgnn_task_waypoint_training": readiness["round20_ready_for_hgnn_task_waypoint_training"],
        "untouched_eval_policy_rollout_count": 0, "artifact_hashes": _artifact_hashes(output),
    }
    write_json(output / "run_manifest.json", manifest)
    return {"output": output, "figures": figures, "result": result, "audits": audits, "readiness": readiness, "manifest": manifest}


def recover_formal_output(root: Path, *, output_name: str, figure_name: str) -> dict[str, Any]:
    """Complete reporting-only derivatives from already-written formal raw data."""
    root = root.resolve()
    preflight_result = preflight(root)
    output = root / "round_20" / "outputs" / output_name
    figures = root / "round_20" / "figures" / figure_name
    raw_inputs = {
        name: sha256(output / name)
        for name in ("multi_uav_dataset_manifest.json", "multi_uav_transitions.jsonl", "multi_uav_task_waypoint_samples.jsonl")
    }
    result = reconstruct_formal_result(output)
    write_csv(output / "multi_uav_episode_metrics.csv", episode_metrics(result["episodes"]))
    audits = write_audits(output, result)
    readiness = {
        "dataset_quality_ready": audits["quality_gate"]["dataset_quality_ready"],
        "graph_training_data_ready": audits["graph_gate"]["graph_training_data_ready"],
        "multi_uav_task_waypoint_training_ready": audits["task_gate"]["multi_uav_task_waypoint_training_ready"],
        "split_isolation": audits["split"]["scenario_split_isolation"],
        "protected_data_clean": True,
        "new_untouched_eval_policy_rollout_count": 0,
        "historical_protected_access_counts": {"old_development": 0, "fresh_development": 0, "historical_untouched": 0},
        "full_regression_tests_pending": True,
        "compileall_pending": True,
        "formal_raw_data_recovered_without_rollout": True,
        "formal_raw_input_hashes": raw_inputs,
    }
    readiness["round20_ready_for_hgnn_task_waypoint_training"] = bool(all(readiness[key] for key in ("dataset_quality_ready", "graph_training_data_ready", "multi_uav_task_waypoint_training_ready", "split_isolation", "protected_data_clean")))
    write_json(output / "round20_readiness_gate.json", readiness)
    render_figures(figures, result, audits)
    write_completion_report(output / "round20_completion_report.md", result, audits, readiness, figures)
    write_json(output / "recovery_provenance.json", {"recovery": "render/audit only; no environment, planner, or rollout invoked", "raw_inputs": raw_inputs, "safety_audit_numerical_tolerance_m": 1.0e-5})
    (output / "implementation_log.md").write_text("Formal raw trajectories completed before this reporting-only recovery. Existing JSONL files were deserialized, the unchanged 15 m safety invariant was audited with a 1e-5 m floating-point tolerance, derived audits were rewritten, and the full figure set was rendered. No environment, planner, neural model, or rollout was invoked.\n", encoding="utf-8")
    manifest = {
        "round": "round_20", "output_name": output_name, "status": "formal_generation_complete_pending_regression",
        "recovered_from_formal_raw_data": True, "preflight": preflight_result, "scenario_count": len(result["scenarios"]), "episode_count": len(result["episodes"]),
        "transition_count": len(result["transitions"]), "task_waypoint_sample_count": len(result["samples"]), "task_waypoint_counts": audits["task"]["counts"],
        "figure_directory": f"round_20/figures/{figure_name}", "figure_count": len(figure_manifest(figures)), "figure_hashes": figure_manifest(figures),
        "dataset_quality_gate": audits["quality_gate"]["dataset_quality_ready"], "graph_readiness_gate": audits["graph_gate"]["graph_training_data_ready"], "task_waypoint_readiness_gate": audits["task_gate"]["multi_uav_task_waypoint_training_ready"],
        "round20_ready_for_hgnn_task_waypoint_training": readiness["round20_ready_for_hgnn_task_waypoint_training"], "untouched_eval_policy_rollout_count": 0,
        "formal_raw_input_hashes": raw_inputs, "artifact_hashes": _artifact_hashes(output),
    }
    write_json(output / "run_manifest.json", manifest)
    return {"output": output, "figures": figures, "result": result, "audits": audits, "readiness": readiness, "manifest": manifest}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-name", default=FORMAL_STEM)
    args = parser.parse_args()
    completed = run(args.root, output_name=args.output_name)
    print(completed["output"])
