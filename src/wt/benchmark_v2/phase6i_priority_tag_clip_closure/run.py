"""Write the closure artifacts from the read-only audit."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.run import parse_unittest
from wt.benchmark_v2.phase6i_priority_tag_reward.smoke import file_sha256
from wt.benchmark_v2.phase6i_priority_tag_reward.train import source_commit

from .protocol import (
    E15_IMPLEMENTATION_COMMIT,
    E15_RESULT_COMMIT,
    GATE_BLOCKED,
    GATE_CLOSED,
    INSTRUCTION_COMMIT,
    OUTPUT,
    ROOT,
    SCHEMA_VERSION,
)
from .report import write_report
from .review import audit

MATRIX_FIELDS = (
    "model_seed", "reference", "tagged_mean_mbit", "tagged_ci_low_mbit", "tagged_ci_high_mbit",
    "payload_mean_mbit", "payload_ci_low_mbit", "payload_ci_floor_mbit", "sinr_mean_delta",
    "interference_mean_delta", "safety_diff_sum", "constraint_diff_sum", "leave_one_tagged",
    "leave_two_tagged", "reference_checks_met", "reason",
)
REGRESSION_FIELDS = (
    "model_seed", "reference", "scenario_id", "tagged_delta_bits", "payload_delta_bits",
    "tagged_regression", "payload_regression",
)
SAFETY_FIELDS = (
    "source_event_index", "source_sha256", "variant", "model_seed", "ppo_update", "env_index",
    "safety_violation", "constraint_failure", "env_t", "scenario_id", "geometry", "classification",
    "distance_before_recomputed_m", "distance_after_recomputed_m", "margin_after_recomputed_m",
    "distance_before_gap_m", "distance_after_gap_m", "margin_gap_m", "counter_flags",
    "projector_would_accept_separation", "predicted_distance_m", "predicted_distance_scope",
)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value).replace("\\", "/")
    if hasattr(value, "item"):
        return value.item()
    return value


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        return repr(float(value))
    if isinstance(value, (list, dict)):
        return json.dumps(_plain(value), ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in fields})


def _artifact_index(head: str) -> dict:
    names = (
        "results/phase6i_priority_tag_clip_benchmark/summary.json",
        "results/phase6i_priority_tag_clip_benchmark/scenario_rows.csv",
        "results/phase6i_priority_tag_clip_benchmark/safety_events.json",
        "results/phase6i_priority_tag_clip_benchmark/training_update_rows.csv",
        "results/phase6i_priority_tag_clip_benchmark/optimizer_step_rows.csv",
        "results/phase6i_priority_tag_clip_benchmark/checkpoint_manifest.json",
        "results/phase6i_priority_tag_clip_benchmark/runtime_manifest.json",
        "results/phase6i_priority_tag_clip_benchmark/source_manifest.json",
        "results/phase6i_priority_tag_clip_benchmark/direction_decision.json",
        "results/phase6i_priority_tag_optimizer_intervention/summary.json",
        "results/phase6i_priority_tag_optimizer_intervention/safety_events.json",
        "src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt",
    )
    return {
        "e15_result_commit": E15_RESULT_COMMIT,
        "e15_implementation_commit": E15_IMPLEMENTATION_COMMIT,
        "instruction_commit": INSTRUCTION_COMMIT,
        "implementation_commit": head,
        "files": {name: file_sha256(ROOT / name) for name in names},
        "adoption": {
            "formal_baseline": "L0",
            "full_j": "research_reference",
            "full_s": "research_reference",
            "pilot_checkpoints": "research_reference",
            "new_formal_loader": False,
        },
    }


def _public_summary(body: dict) -> dict:
    gate = GATE_CLOSED if body["consistent"] else GATE_BLOCKED
    safety = body["safety"]
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_gate": gate,
        "e15_gate": body["recomputed_e15_gate"],
        "e15_gate_unchanged": body["recomputed_e15_gate"] == "CLIP_GROUP_BENCHMARK_INCONCLUSIVE" and body["consistent"],
        "release_eligible": False,
        "merged": False,
        "adopted": False,
        "ppo_updates": body["budget"]["ppo_updates"],
        "adam_steps": body["budget"]["adam_steps"],
        "passing_seeds": int(body["score"]["passing_seeds"]),
        "score_max_abs_gap": body["score_max_abs_gap"],
        "regression_rows": len(body["regressions"]),
        "problems": body["problems"],
        "lineage_gates": body["lineage"]["gates"],
        "safety_review": {
            "records": safety["records"],
            "safety_violations": safety["safety_violations"],
            "constraint_failures": safety["constraint_failures"],
            "safety_by_variant": safety["safety_by_variant"],
            "margin_min_m": safety["margin_min_m"],
            "margin_max_m": safety["margin_max_m"],
            "boundary_count": safety["boundary_count"],
            "boundary_does_not_reduce_count": safety["boundary_does_not_reduce_count"],
            "e14_safety_events": safety["e14_safety_events"],
            "e14_geometry": safety["e14_geometry"],
            "constraint_review": {key: value for key, value in safety["constraint_review"].items() if key != "problems"},
            "training_counts_are_not_development_counts": True,
        },
        "criteria_matrix": body["criteria_matrix"],
        "formal_actor_sha256": body["formal"]["formal_actor_sha256"],
        "snapshot_resume": "weights_adam_process_rng_and_worker_assignment_rng_only",
        "test_comparison": None,
    }


def execute() -> dict:
    import subprocess

    head = source_commit()
    if head == INSTRUCTION_COMMIT:
        raise RuntimeError("the closure implementation commit is missing")
    ancestor = subprocess.run(
        ["git", "-c", "core.longpaths=true", "merge-base", "--is-ancestor", INSTRUCTION_COMMIT, "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("HEAD does not descend from the closure instruction")
    dirty = subprocess.run(
        ["git", "-c", "core.longpaths=true", "status", "--porcelain", "--", "src", "tests"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if dirty.stdout.strip():
        raise RuntimeError("src or tests changed after the closure implementation commit")
    body = audit()
    summary = _public_summary(body)
    summary["implementation_commit"] = head
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT / "criteria_matrix.csv", MATRIX_FIELDS, body["criteria_matrix"])
    _write_csv(OUTPUT / "scenario_regressions.csv", REGRESSION_FIELDS, body["regressions"])
    _write_csv(OUTPUT / "safety_event_review.csv", SAFETY_FIELDS, body["safety"]["rows"])
    _write_json(OUTPUT / "constraint_failure_review.json", summary["safety_review"]["constraint_review"])
    _write_json(OUTPUT / "artifact_index.json", _artifact_index(head))
    _write_json(
        OUTPUT / "formal_baseline_manifest.json",
        {
            "formal_actor_sha256": body["formal"]["formal_actor_sha256"],
            "matches_e15_manifest": body["formal"]["matches_e15_manifest"],
            "watched": body["formal"]["watched"],
            "adopted_model": "L0",
            "research_references": ["FULL_J", "FULL_S", "E14_PILOT"],
            "default_config_changed": False,
            "formal_checkpoint_replaced": False,
        },
    )
    _write_json(
        OUTPUT / "evidence_gaps.json",
        [
            "E14 的 4 条安全事件没有位置和距离，不能用 E15 回填。",
            "E15 事件没有保存 env.t 或 scenario_id。",
            "E15 定点快照不含环境、账本和 option，不能无缝续训。",
            "40 次边界分类不减少原计数，也不排除运动风险。",
            "predicted_distance_m 是投影器全局最小值，不是当前 UAV 对的预测距离。",
            "开发池已经被多轮使用，本次结案不是未见场景泛化。",
            "结案不证明所有裁剪方法无效，也不结束 payload 加紧急服务的整条研究。",
        ],
    )
    blockers = [
        "E15 保持 CLIP_GROUP_BENCHMARK_INCONCLUSIVE，FULL_J 和 FULL_S 不采用。",
        "完整历史测试套件仍有既有失败。",
        "继承分支不是正式采用的最小差异。",
        "release_eligible 保持 false。",
        "没有合并。",
    ]
    if summary["scientific_gate"] != GATE_CLOSED:
        blockers.insert(0, "结案核对不一致：" + "; ".join(body["problems"]))
    _write_json(OUTPUT / "release_blockers.json", blockers)
    _write_json(OUTPUT / "schema.json", {"schema_version": SCHEMA_VERSION, "e15_result_commit": E15_RESULT_COMMIT, "training": False})
    _write_json(OUTPUT / "summary.json", summary)
    write_report(summary)
    print(f"scientific_gate={summary['scientific_gate']} problems={len(body['problems'])} e15_gate={summary['e15_gate']}", flush=True)
    return summary


def _normalize(detail: str) -> str:
    text = str(detail)
    for root in (str(ROOT), str(ROOT).replace("\\", "/"), r"E:\srdp\wt_e16_baseline", "E:/srdp/wt_e16_baseline", r"E:\srdp\wt_e15_baseline", "E:/srdp/wt_e15_baseline"):
        text = text.replace(root, "<ROOT>")
    return text


def compare_logs(baseline_text: str, current_text: str) -> dict:
    baseline = parse_unittest(baseline_text)
    current = parse_unittest(current_text)
    baseline_ids = {(item["test_id"], item["kind"]): item for item in baseline["failures"]}
    new_items = []
    historical = []
    for item in current["failures"]:
        previous = baseline_ids.get((item["test_id"], item["kind"]))
        same = previous is not None and previous["exception"] == item["exception"] and _normalize(previous["detail"]) == _normalize(item["detail"])
        if same:
            historical.append(item)
        else:
            new_items.append(item)
    return {
        "baseline_ran": baseline["ran"],
        "repository_ran": current["ran"],
        "baseline_failure_count": len(baseline["failures"]),
        "repository_failure_count": len(current["failures"]),
        "historical_count": len(historical),
        "new_failure_count": len(new_items),
        "new_failures": new_items,
        "comparison_root": "E:/srdp/wt_e16_baseline",
    }


def attach_tests() -> dict:
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    comparison = compare_logs(
        (OUTPUT / "baseline_tests.log").read_text(encoding="utf-8", errors="replace"),
        (OUTPUT / "repository_tests.log").read_text(encoding="utf-8", errors="replace"),
    )
    related = (OUTPUT / "related_tests.log").read_text(encoding="utf-8", errors="replace")
    comparison["related_ok"] = related.rstrip().endswith("OK")
    summary["test_comparison"] = comparison
    if comparison["new_failure_count"] != 0 or not comparison["related_ok"]:
        summary["scientific_gate"] = GATE_BLOCKED
        summary["problems"] = list(summary.get("problems", [])) + ["new tests or the clean-suite comparison failed"]
    _write_json(OUTPUT / "test_comparison.json", comparison)
    _write_json(OUTPUT / "summary.json", summary)
    write_report(summary)
    print(f"new_failure_count={comparison['new_failure_count']} related_ok={comparison['related_ok']} gate={summary['scientific_gate']}", flush=True)
    return comparison


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "audit":
        execute()
        return
    if command == "tests":
        attach_tests()
        return
    raise SystemExit("usage: audit|tests")


if __name__ == "__main__":
    main()
