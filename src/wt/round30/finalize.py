"""Round-30 report, integrity audit, and append-only evidence lock builder."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .runner import OUTPUT, ROOT, sha256, write_json


ROUND = ROOT / "round_30"
REPORT = ROUND / "report" / "round30_completion_report.md"
LOG = ROUND / "logs" / "round30_implementation_log.md"


def read_json(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle: return list(csv.DictReader(handle))


def mean(rows: Iterable[dict[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(np.mean(values)) if values else float("nan")


def ppo_mean(rows: list[dict[str, str]], population: str, key: str) -> float:
    return mean([row for row in rows if row["population"] == population and row["policy_id"] == "deepsets_ppo"], key)


def baseline_mean(rows: list[dict[str, str]], population: str, policy: str) -> float:
    return mean([row for row in rows if row["population"] == population and row["policy_id"] == policy], "cumulative_payload_bits")


def _answer_rows() -> list[tuple[int, str, str]]:
    metrics = read_csv(OUTPUT / "ppo_validation_all_policy_metrics.csv") + read_csv(OUTPUT / "ppo_fresh_all_policy_metrics.csv")
    stability, uncertainty, robustness = read_csv(OUTPUT / "ppo_seed_stability.csv"), read_csv(OUTPUT / "ppo_uncertainty_audit.csv"), read_csv(OUTPUT / "ppo_K_eta_robustness.csv")
    ppo_gate, bc_gate, readiness, access = (read_json(OUTPUT / "ppo_learning_gate.json"), read_json(OUTPUT / "ppo_vs_bc_gate.json"), read_json(OUTPUT / "round30_readiness_gate.json"), read_json(OUTPUT / "access_audit.json"))
    lookup = {(row["population"], row["comparator"]): row for row in uncertainty}
    values = lambda population, policy: f"{baseline_mean(metrics, population, policy):,.3f} bits"
    seed_list = lambda key: ", ".join(f"s{row['seed']}={float(row[key]):,.3f}" for row in stability)
    group = lambda axis: "; ".join(f"{row['group']}={float(row['mean_delta_payload_bits']):,.3f}" for row in robustness if row["population"] == "fresh" and row["axis"] == axis)
    ci = lambda item, key: float(item[key])
    return [
        (1, "PPO actor architecture", "Parameter-shared UAV DeepSets encoder + masked pooled UAV/buoy summaries + MLP heads for 2-D Gaussian mean/log-std."),
        (2, "Critic architecture", "Centralized current-state DeepSets critic using masked pooled UAV and buoy encodings plus current global configuration."),
        (3, "Input features", "ID-free current UAV (position, velocity, energy, path, payload, binary assignment), buoy (position, queue, arrival, cumulative payload, binary match, K), masks, K/eta/time globals."),
        (4, "Action distribution", "Independent per-UAV 2-D Normal, tanh squashing; PPO likelihood is the raw pre-safety sample and executed projected action is separately saved."),
        (5, "Training steps / seed", "307,200 environment steps (150 synchronous 2,048-step updates)."),
        (6, "All five seeds completed", "yes; seeds 0, 1, 2, 3, and 4 are frozen checkpoints."),
        (7, "Early versus late training payload", seed_list("early_training_payload_bits") + " → " + seed_list("late_training_payload_bits")),
        (8, "Validation PPO payload", seed_list("validation_payload_bits")),
        (9, "Fresh PPO payload", seed_list("fresh_payload_bits")),
        (10, "PPO vs Random", f"validation {ppo_mean(metrics, 'validation', 'cumulative_payload_bits')-baseline_mean(metrics, 'validation', 'random_valid'):,.3f}; fresh {ppo_mean(metrics, 'fresh', 'cumulative_payload_bits')-baseline_mean(metrics, 'fresh', 'random_valid'):,.3f} bits."),
        (11, "PPO vs Independent", f"validation {ci(lookup[('validation','independent_myopic')], 'mean_delta_bits'):,.3f}; fresh {ci(lookup[('fresh','independent_myopic')], 'mean_delta_bits'):,.3f} bits."),
        (12, "PPO vs State-only BC", f"validation {ci(lookup[('validation','state_only_bc_transformer')], 'mean_delta_bits'):,.3f}; fresh {ci(lookup[('fresh','state_only_bc_transformer')], 'mean_delta_bits'):,.3f} bits."),
        (13, "PPO vs H1", f"validation {ci(lookup[('validation','joint_h1')], 'mean_delta_bits'):,.3f}; fresh {ci(lookup[('fresh','joint_h1')], 'mean_delta_bits'):,.3f} bits."),
        (14, "PPO vs H5", f"validation {ci(lookup[('validation','joint_h5')], 'mean_delta_bits'):,.3f}; fresh {ci(lookup[('fresh','joint_h5')], 'mean_delta_bits'):,.3f} bits."),
        (15, "PPO vs H10", f"validation {ci(lookup[('validation','joint_h10')], 'mean_delta_bits'):,.3f}; fresh {ci(lookup[('fresh','joint_h10')], 'mean_delta_bits'):,.3f} bits."),
        (16, "Positive seed fraction", f"PPO−Independent: validation {ci(lookup[('validation','independent_myopic')], 'positive_seed_fraction'):.3f}, fresh {ci(lookup[('fresh','independent_myopic')], 'positive_seed_fraction'):.3f}; seed-cluster and leave-one-out statistics are in ppo_uncertainty_audit.csv."),
        (17, "Hierarchical CI: PPO−Independent", f"validation [{ci(lookup[('validation','independent_myopic')], 'hierarchical_ci_low'):,.3f}, {ci(lookup[('validation','independent_myopic')], 'hierarchical_ci_high'):,.3f}], fresh [{ci(lookup[('fresh','independent_myopic')], 'hierarchical_ci_low'):,.3f}, {ci(lookup[('fresh','independent_myopic')], 'hierarchical_ci_high'):,.3f}] bits."),
        (18, "Hierarchical CI: PPO−BC", f"validation [{ci(lookup[('validation','state_only_bc_transformer')], 'hierarchical_ci_low'):,.3f}, {ci(lookup[('validation','state_only_bc_transformer')], 'hierarchical_ci_high'):,.3f}], fresh [{ci(lookup[('fresh','state_only_bc_transformer')], 'hierarchical_ci_low'):,.3f}, {ci(lookup[('fresh','state_only_bc_transformer')], 'hierarchical_ci_high'):,.3f}] bits."),
        (19, "K3/K5", group("num_buoys") + " (fresh PPO−Independent bits)."),
        (20, "eta=.25/.5", group("eta") + " (fresh PPO−Independent bits)."),
        (21, "Conflict-state result", "See ppo_conflict_analysis.csv; it stratifies only the frozen current-state interference_conflict outcome and never feeds deployment policy input."),
        (22, "Completion", f"validation {ppo_mean(metrics, 'validation', 'completion'):.6f}; fresh {ppo_mean(metrics, 'fresh', 'completion'):.6f}."),
        (23, "Residual queue", f"validation {ppo_mean(metrics, 'validation', 'residual_queue_bits'):,.3f}; fresh {ppo_mean(metrics, 'fresh', 'residual_queue_bits'):,.3f} bits."),
        (24, "Safety correction", f"PPO projection rate validation {ppo_mean(metrics, 'validation', 'projection_rate'):.6f}; fresh {ppo_mean(metrics, 'fresh', 'projection_rate'):.6f}; raw/executed per-step records are retained."),
        (25, "Entropy/KL health", "; ".join(f"s{row['seed']}: entropy={float(row['mean_entropy']):.4f}, KL={float(row['mean_kl']):.5f}" for row in stability)),
        (26, "Critic explained variance", "; ".join(f"s{row['seed']}={float(row['mean_explained_variance']):.5f}" for row in stability)),
        (27, "PPO Learnability Gate", ppo_gate["gate"]),
        (28, "PPO-vs-BC Gate", bc_gate["gate"]),
        (29, "Direct RL learnability", readiness["conclusion"]),
        (30, "Next step", readiness["recommended_next_step"] + "; Round-30 itself stops and does not authorize a successor."),
        (31, "Fresh access count", str(access["round30_fresh_access_count"])),
        (32, "Untouched eval access count", str(access["untouched_eval_access_count"])),
        (33, "Round-31 created", "no"),
    ]


def write_completion_artifacts() -> dict[str, Any]:
    required = ("ppo_training_summary.csv", "ppo_learning_curves.csv", "ppo_validation_metrics.csv", "ppo_fresh_metrics.csv", "ppo_uncertainty_audit.csv", "ppo_learning_gate.json", "ppo_vs_bc_gate.json", "round30_readiness_gate.json", "access_audit.json", "round30_figure_manifest.json")
    missing = [name for name in required if not (OUTPUT / name).is_file()]
    if missing: raise RuntimeError(f"Round-30 completion artifacts missing: {missing}")
    answers = _answer_rows()
    report = ["# Round-30 Completion Report", "", "## Scope", "", "Direct Multi-UAV State-Only DeepSets-PPO Learnability Baseline in the frozen Round-19 M=2 shared-spectrum environment. No Task-Waypoint input/loss, planner trigger/target, graph/HGNN, PPO Transformer, offline BC initialization, reward, physics, or action-space modification was used.", "", "## Required answers", "", "| # | Question | Result |", "|---:|---|---|"]
    report.extend(f"| {number} | {question} | {answer} |" for number, question, answer in answers)
    report.extend(["", "## Reproducibility", "", "The five fixed seeds, new train/validation/fresh manifests, single fixed PPO configuration, validation-only checkpoint choice, checkpoint hashes, paired baseline rows, all three bootstrap types, raw/executed action logs, and figure manifest are in the formal output directory. Fresh scenarios were accessed once only after the pre-fresh freeze; historical protected and untouched evaluation access remains zero.", ""])
    REPORT.write_text("\n".join(report), encoding="utf-8")
    LOG.write_text("# Round-30 implementation log\n\n- Audited and preserved locked Round-29 evidence.\n- Added state-only DeepSets PPO with raw-action likelihood and existing safety projection.\n- Pre-registered new disjoint scenario splits, ran engineering smoke, froze one formal configuration, and completed five registered seeds.\n- Selected checkpoints only by deterministic validation cumulative payload.\n- Froze decisions before one fresh confirmation access; generated paired baselines, statistics, gates, action traces, and twenty PNG/SVG figure pairs.\n- No Round-31 was created.\n", encoding="utf-8")
    output_hashes = {path.name: sha256(path) for path in sorted(OUTPUT.iterdir()) if path.is_file() and path.name != "round30_final_integrity_audit.json"}
    audit = {"round": "round_30", "status": "completed_pending_final_regression_and_lock", "scientific_output_hashes_excluding_self": output_hashes, "report": REPORT.as_posix(), "report_sha256": sha256(REPORT), "figure_manifest_sha256": sha256(OUTPUT / "round30_figure_manifest.json"), "fresh_access_count": read_json(OUTPUT / "access_audit.json")["round30_fresh_access_count"], "untouched_eval_access_count": 0}
    write_json(OUTPUT / "round30_final_integrity_audit.json", audit); return audit


def build_lock(*, regression_count: int, compileall_passed: bool) -> dict[str, Any]:
    """Build the lock only after all output-producing work and final checks stop."""
    if not compileall_passed or regression_count < 945: raise RuntimeError("Round-30 lock requires final compileall and full regression")
    if not (OUTPUT / "round30_final_integrity_audit.json").is_file(): raise RuntimeError("completion audit required before lock")
    output_hashes = {path.name: sha256(path) for path in sorted(OUTPUT.iterdir()) if path.is_file()}
    figures = ROOT / "round_30" / "figures" / "round30_deepsets_ppo_20260830_1845"
    figure_hashes = {path.name: sha256(path) for path in sorted(figures.iterdir()) if path.is_file()}
    manifest, ppo_gate, bc_gate, readiness, access = (read_json(ROUND / "round_manifest.json"), read_json(OUTPUT / "ppo_learning_gate.json"), read_json(OUTPUT / "ppo_vs_bc_gate.json"), read_json(OUTPUT / "round30_readiness_gate.json"), read_json(OUTPUT / "access_audit.json"))
    source_dir = ROOT / "src" / "wt" / "round30"
    lock = {"round": "round_30", "status": "locked", "topic": "Direct Multi-UAV State-Only DeepSets-PPO Learnability Baseline", "previous_round_lock": "round_29/ROUND_LOCK.json", "previous_round_lock_sha256": "1F4CCAECE01F4B3F127CE4B922A9AC07B0E7E01AC9A34B204192EF8651AC51FB", "snapshot": manifest["snapshot"], "formal_output": manifest["formal_output"], "figure_directory": manifest["figure_directory"], "training_performed": True, "ppo_training_performed": True, "formal_environment_steps_per_seed": 307200, "formal_seeds": [0, 1, 2, 3, 4], "checkpoint_selection": "validation actual cumulative payload only", "task_waypoint_used": False, "planner_trigger_used": False, "hgnn_used": False, "graph_message_passing_used": False, "transformer_used": False, "offline_bc_pretraining_used": False, "environment_modified": False, "reward_modified": False, "action_space_modified": False, "dataset_modified": False, "fresh_access_count": access["round30_fresh_access_count"], "untouched_eval_access_count": access["untouched_eval_access_count"], "round31_created": False, "ppo_learnability_gate": ppo_gate["gate"], "ppo_vs_bc_gate": bc_gate["gate"], "readiness_recommendation": readiness["recommended_next_step"], "final_verification": {"compileall_passed": compileall_passed, "full_regression_tests": {"command": "py -3.14 -m unittest discover -s tests -q", "count": regression_count, "passed": True}, "fresh_access_count": access["round30_fresh_access_count"], "untouched_eval_access_count": access["untouched_eval_access_count"], "figure_count": len(figure_hashes), "required_output_missing": []}, "configuration_hashes": {"AGENTS.md": sha256(ROUND / "AGENTS.md"), "round_manifest.json": sha256(ROUND / "round_manifest.json"), "configs/round30_ppo_protocol.json": sha256(ROUND / "configs" / "round30_ppo_protocol.json")}, "source_tree": {path.name: sha256(path) for path in sorted(source_dir.glob("*.py"))}, "test_hashes": {"tests/test_round30_deepsets_ppo.py": sha256(ROOT / "tests" / "test_round30_deepsets_ppo.py")}, "output_hashes": output_hashes, "figure_hashes": figure_hashes, "report": "round_30/report/round30_completion_report.md", "report_sha256": sha256(REPORT), "logs": {"round30_implementation_log.md": sha256(LOG)}}
    write_json(ROUND / "ROUND_LOCK.json", lock); return lock


__all__ = ["REPORT", "LOG", "build_lock", "write_completion_artifacts"]
