"""Frozen tag-signal audit. This runner does not train or merge."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from wt.benchmark_v2.phase6i_priority_tag_direct_service_reward.evaluate import preflight_l0
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import EVAL_ASSIGNMENT_SEED
from wt.benchmark_v2.phase6i_priority_tag_reward.smoke import file_sha256

from .counterfactual import run_counterfactuals
from .episode import run_episode
from .integrity import checkpoint_inventory, load_e10_rows, recompute_gate, row_inventory, source_commit_audit
from .metrics import exposure, gae_compare, training_logs
from .metrics import _ratios as reward_ratios
from .protocol import (
    CANDIDATE_OFFSETS,
    COUNTERFACTUAL_STEPS,
    DIAGNOSTIC_SCENARIOS,
    E10_RESULT_COMMIT,
    E10_SOURCE_COMMIT,
    GATE_BLOCKED,
    GATE_COMPLETE,
    INSTRUCTION_COMMIT,
    L3_SEEDS,
    OPPORTUNITY_PAYLOAD_BITS,
    OPPORTUNITY_TAG_BITS,
    OUTPUT,
    PATH_TOLERANCE_M,
    PERTURBATION,
    REPORT,
    SAFETY_CASES,
    SAFETY_DISTANCE_TOLERANCE_M,
    SCHEMA_VERSION,
    snapshot_indices,
)
from .report import render
from .safety import localize
from .tables import STEP_COLUMNS, sensitivity_summary, step_rows, write_csv


RELATED_TESTS = (
    "tests.test_priority_tag_signal_audit",
    "tests.test_priority_tag_direct_service_reward",
    "tests.test_priority_tag_prototype",
    "tests.test_priority_tag_eval_context",
    "tests.test_priority_tag_reward_attribution",
    "tests.test_priority_tag_attribution_full_replay",
)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        raise TypeError("a numpy snapshot entered the public summary")
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _environment() -> dict:
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[4] / "src")
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not current else src + os.pathsep + current
    return env


def _failures(text: str) -> list[str]:
    found = []
    for line in text.splitlines():
        if line.startswith("FAIL:") or line.startswith("ERROR:") or line.startswith("FAILED ("):
            found.append(line.strip())
    return found


def _execute(command: list[str], log_path: Path) -> dict:
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[4], env=_environment(), capture_output=True, text=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    return {
        "command": " ".join(command),
        "exit_code": int(completed.returncode),
        "failures": _failures(completed.stdout + "\n" + completed.stderr),
        "log": str(log_path),
    }


def _schema() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "written_before_trajectories": True,
        "diagnostic_scenarios": list(DIAGNOSTIC_SCENARIOS),
        "safety_cases": list(SAFETY_CASES),
        "snapshot_indices_for_horizon_48": list(snapshot_indices(48)),
        "counterfactual_steps": COUNTERFACTUAL_STEPS,
        "candidate_offsets": [list(item) for item in CANDIDATE_OFFSETS],
        "perturbation": PERTURBATION,
        "opportunity_tag_bits": OPPORTUNITY_TAG_BITS,
        "opportunity_payload_bits": OPPORTUNITY_PAYLOAD_BITS,
        "counter_distance_tolerance_m": SAFETY_DISTANCE_TOLERANCE_M,
        "projector_distance_tolerance_m": 1.0e-6,
        "bit_tolerance": 1.0,
        "path_tolerance_m": PATH_TOLERANCE_M,
        "assignment_seed": "7000000 + scenario_id",
        "training_assignment_seed_not_used": "321000000 + model_seed * 10000 + env_index",
        "historical_process_seed_role": "historical_inherited_process_context",
        "canonical_explicit_eval_seed": None,
        "policy_active_mask": "ExperimentWorker writes ones; it is not a learned gate",
        "l0_tag_input": "not fabricated",
        "gae_label": "diagnostic_estimate",
        "does_not_train": True,
    }


def _counterfactual_summary(rows: list[dict]) -> dict:
    snapshots = {(row["variant"], int(row["model_seed"]), int(row["scenario_id"]), int(row["step"])) for row in rows}
    reproduced = {
        key
        for key in snapshots
        if all(row["baseline_reproduced"] for row in rows if (row["variant"], int(row["model_seed"]), int(row["scenario_id"]), int(row["step"])) == key)
    }
    opportunities = [
        row
        for row in rows
        if row["short_horizon_opportunity"] and not (float(row["offset_x"]) == 0.0 and float(row["offset_y"]) == 0.0)
    ]
    if not rows or len(reproduced) == 0:
        statement = "基线未能复现，控制分析不能解释这些快照。"
    elif not opportunities:
        statement = "未观察到证据"
    else:
        statement = f"在预定短时域筛选中观察到 {len(opportunities)} 个候选行满足诊断阈值。"
    return {
        "rows": len(rows),
        "snapshots": len(snapshots),
        "snapshots_reproduced": len(reproduced),
        "invalid_snapshots": len(snapshots) - len(reproduced),
        "opportunity_rows": len(opportunities),
        "statement": statement,
    }


def _recommendation(summary: dict) -> str:
    if summary["gate"] == GATE_BLOCKED:
        return "必要核对没有成立。下一步只应先处理本报告列出的阻断项，不应改奖励系数或启动新实验。"
    exposure = summary.get("exposure", {}).get("episodes", [])
    sensitivity = [item for item in summary.get("sensitivity", {}).get("episodes", []) if item.get("actor_accepts_tag")]
    visible = sum(int(item["visible_tagged_uav_steps"]) for item in exposure)
    served = sum(int(item["positive_tag_service_steps"]) for item in exposure)
    changed = sum(int(item["executed_changed_steps"]) for item in sensitivity)
    inspected = sum(int(item["aligned_nonzero_tag_input_steps"]) for item in sensitivity)
    opportunities = int(summary.get("counterfactual", {}).get("opportunity_rows", 0))
    parts = [
        "若继续做实验，应单独预先声明，并以本次未回答的问题为对照，而不是直接调大系数。",
    ]
    if visible == 0 or served == 0:
        parts.append("这两条训练场景上，tag 的可见窗口或正服务覆盖不足。下一轮若改观测或任务关联，需要同一 FIFO 和同一奖励下的无标签对照，并事先扩大场景的规则，不能事后挑选。")
    if inspected and changed == 0:
        parts.append("冻结 L3 在已对齐且 tag 非零的快照上没有改变执行动作。下一轮若怀疑策略没用到该列，应设置保留该列与移除该列的对照，而不是只改变 bonus 系数。")
    if opportunities == 0:
        parts.append("预定的 4 步残差扰动没有给出兼顾 payload 和安全的 tagged 服务机会。这仍不能证明更长时域或不同动作不可控。")
    else:
        parts.append("短时域筛选看到了局部机会。这只说明这些冻结动作附近存在诊断意义上的差异，不能当作新策略已经更好。")
    parts.append("两处安全事件只有在重放匹配后才有证据标签；标签可以同时成立，不能收成单一原因。历史逐步 tag bonus 和训练当时的实现字节仍然缺失。")
    return "".join(parts)


def _safety_rows(cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        if not case["events"]:
            rows.append(
                {
                    "variant": case["case"]["variant"],
                    "model_seed": case["case"]["model_seed"],
                    "scenario_id": case["case"]["scenario_id"],
                    "matched_saved_rows": case["matched_saved_rows"],
                    "causal_interpretation": case["causal_interpretation"],
                    "step": "",
                    "labels": "",
                }
            )
            continue
        for event in case["events"]:
            previous = event.get("previous_step") or {}
            rows.append(
                {
                    "variant": event["variant"],
                    "model_seed": event["model_seed"],
                    "scenario_id": event["scenario_id"],
                    "matched_saved_rows": case["matched_saved_rows"],
                    "causal_interpretation": case["causal_interpretation"],
                    "step": event["step"],
                    "uav_i": event["uav_i"],
                    "uav_j": event["uav_j"],
                    "distance_m": event["distance_m"],
                    "previous_distance_m": previous.get("distance_m", ""),
                    "predicted_distance_m": event["predicted_distance_m"],
                    "threshold_m": event["threshold_m"],
                    "projection_status": event["projection_status"],
                    "constraint_failure": event["constraint_failure"],
                    "labels": ";".join(event["labels"]),
                }
            )
    return rows


def _publish(summary: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    public = _plain(summary)
    (OUTPUT / "summary.json").write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT.write_text(render(public), encoding="utf-8")


def main() -> dict:
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    schema = _schema()
    (OUTPUT / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    related = _execute(["py", "-3", "-m", "unittest", *RELATED_TESTS], OUTPUT / "related_tests.log")
    rows = load_e10_rows()
    inventory = row_inventory(rows)
    recomputed = recompute_gate(rows)
    checkpoints = checkpoint_inventory()
    source = source_commit_audit()
    saved_index = {(row["variant"], int(row["model_seed"]), int(row["scenario_id"])): row for row in rows}
    preflight_public = None
    reasons = []
    print(f"related_tests {related['exit_code']}", flush=True)
    if related["exit_code"] != 0:
        reasons.append("related tests failed")
    if inventory["problems"]:
        reasons.append("scenario inventory mismatch")
    if not recomputed["matches_saved_fail_gate"] or recomputed["interval_gaps"]:
        reasons.append("recomputed E10 gate does not match the saved failure")
    if any(not item["matches_reported_event"] for item in recomputed["safety_events"]):
        reasons.append("saved safety events do not match the reported pair")
    if checkpoints["problems"]:
        reasons.append("checkpoint hash mismatch")
    if int(source["git_exit_code"]) != 0 or not source["source_commit_is_instruction_only"]:
        reasons.append("E10 source commit could not be confirmed as instruction-only")
    try:
        raw_preflight = preflight_l0(saved_index[("L0", 0, 321101)])
        preflight_public = {
            "scenario_id": int(raw_preflight["scenario_id"]),
            "matches_saved": bool(raw_preflight["matches_saved"]),
            "max_bit_gap": float(raw_preflight["max_bit_gap"]),
            "path_gap_m": float(raw_preflight["path_gap_m"]),
        }
    except Exception as exc:
        reasons.append(f"preflight failed: {type(exc).__name__}: {exc}")
        preflight_public = None
    if preflight_public is not None and not preflight_public["matches_saved"]:
        reasons.append("L0 scenario 321101 preflight missed the saved row")
    print(f"preflight {preflight_public}", flush=True)
    episodes = []
    exposure_result = None
    sensitivity = None
    ratios = None
    gae = None
    logs = None
    counter_rows: list[dict] = []
    counter = None
    safety = None
    try:
        if not reasons:
            for scenario in DIAGNOSTIC_SCENARIOS:
                print(f"episode L0 {scenario}", flush=True)
                episodes.append(run_episode("L0", 0, int(scenario)))
                for seed in L3_SEEDS:
                    print(f"episode L3 {seed} {scenario}", flush=True)
                    episodes.append(run_episode("L3", int(seed), int(scenario)))
            if len(episodes) != 8 or any(not item["context_is_historical"] for item in episodes):
                reasons.append("diagnostic episodes or historical context are incomplete")
            else:
                print("aggregate", flush=True)
                exposure_result = exposure(episodes)
                sensitivity = sensitivity_summary(episodes)
                ratios = reward_ratios(episodes)
                gae = gae_compare(episodes)
                logs = training_logs()
                print("counterfactual", flush=True)
                for episode in episodes:
                    counter_rows.extend(run_counterfactuals(episode))
                counter = _counterfactual_summary(counter_rows)
                if int(counter["snapshots_reproduced"]) == 0:
                    reasons.append("no counterfactual snapshot reproduced its baseline")
                if any(int(item.get("alignment_failed_steps", 0)) == int(item.get("steps", 0)) for item in sensitivity["episodes"] if item.get("actor_accepts_tag")):
                    reasons.append("tag sensitivity did not align with the worker action")
                print("safety", flush=True)
                safety = [localize(saved_index, case) for case in SAFETY_CASES]
                write_csv(OUTPUT / "step_rows.csv", step_rows(episodes), STEP_COLUMNS)
                write_csv(
                    OUTPUT / "bucket_events.csv",
                    exposure_result["events"],
                    ["variant", "model_seed", "scenario_id", "buoy", "admission_step", "queued", "visible_step", "served_step"],
                )
                write_csv(
                    OUTPUT / "counterfactual_rows.csv",
                    counter_rows,
                    [
                        "variant",
                        "model_seed",
                        "scenario_id",
                        "step",
                        "offset_x",
                        "offset_y",
                        "steps",
                        "served_tagged_bits",
                        "payload_delta_bits",
                        "safety_violations",
                        "constraint_failures",
                        "projection_intervention_sum",
                        "clipped_steps",
                        "baseline_reproduced",
                        "tagged_gain_vs_baseline",
                        "payload_gain_vs_baseline",
                        "short_horizon_opportunity",
                    ],
                )
    except Exception as exc:
        traceback.print_exc()
        reasons.append(f"diagnostic failed: {type(exc).__name__}: {exc}")
    if safety is not None:
        write_csv(
            OUTPUT / "safety_events.csv",
            _safety_rows(safety),
            [
                "variant",
                "model_seed",
                "scenario_id",
                "matched_saved_rows",
                "causal_interpretation",
                "step",
                "uav_i",
                "uav_j",
                "distance_m",
                "previous_distance_m",
                "predicted_distance_m",
                "threshold_m",
                "projection_status",
                "constraint_failure",
                "labels",
            ],
        )
    print("repository_suite", flush=True)
    repository = _execute(["py", "-3", "-m", "unittest", "discover", "-s", "tests"], OUTPUT / "repository_tests.log")
    package = Path(__file__).resolve().parent
    digests = {path.name: file_sha256(path) for path in sorted(package.glob("*.py"))}
    gaps = [
        "E10 training checkpoint has no implementation digest; current source is not the training-time proof.",
        "Historical training logs do not record per-step bonus_tag.",
        "A four-step residual screen does not prove long-horizon controllability.",
        "The synthetic generation rate is not the actor visibility rate.",
        "L0 has no tag input, so it is an interface control rather than a tag-sensitivity baseline.",
        "policy_active_mask is identically one in ExperimentWorker.",
    ]
    if repository["exit_code"] != 0:
        gaps.append("The repository test suite reported failures. They are listed and were not deleted.")
    if safety is not None and any(not item["matched_saved_rows"] for item in safety):
        gaps.append("At least one registered safety replay did not match its saved E10 row, so that event was not given a causal label.")
    gate = GATE_BLOCKED if reasons else GATE_COMPLETE
    summary = {
        "gate": gate,
        "block_reasons": reasons,
        "merged": False,
        "trained": False,
        "e10_scientific_gate": "DIRECT_TAG_REWARD_BENCHMARK_FAIL",
        "instruction_commit": INSTRUCTION_COMMIT,
        "parent_result_commit": E10_RESULT_COMMIT,
        "e10_source_commit": E10_SOURCE_COMMIT,
        "inventory": inventory,
        "recomputed_gate": recomputed,
        "checkpoints": checkpoints,
        "source_audit": source,
        "preflight": preflight_public,
        "protocol": schema,
        "exposure": None if exposure_result is None else {"episodes": exposure_result["episodes"]},
        "sensitivity": sensitivity,
        "reward_signal": ratios,
        "gae": gae,
        "training_logs": logs,
        "counterfactual": counter,
        "safety": safety,
        "tests": {"related": related, "repository_suite": repository},
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": bool(torch.cuda.is_available()),
            "elapsed_seconds": float(time.perf_counter() - started),
            "assignment_seed_offset": int(EVAL_ASSIGNMENT_SEED),
            "deployment_verified": False,
        },
        "source_file_sha256": digests,
        "evidence_gaps": gaps,
        "recommendation": "",
    }
    summary["recommendation"] = _recommendation(summary)
    manifest = {
        "runtime": summary["runtime"],
        "historical_context_required": True,
        "canonical_explicit_eval_seed": None,
        "source_file_sha256": digests,
        "checkpoint_hashes": checkpoints["hashes"],
        "watched_hashes": checkpoints["watched"],
    }
    (OUTPUT / "runtime_manifest.json").write_text(json.dumps(_plain(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT / "evidence_gaps.json").write_text(json.dumps(gaps, ensure_ascii=False, indent=2), encoding="utf-8")
    _publish(summary)
    print(gate)
    return summary


if __name__ == "__main__":
    main()
