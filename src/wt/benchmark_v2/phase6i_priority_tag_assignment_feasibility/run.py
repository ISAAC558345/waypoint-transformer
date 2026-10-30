"""Write the assignment-feasibility artifacts after the implementation commit."""
from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.run import parse_unittest
from wt.benchmark_v2.phase6i_priority_tag_reward.smoke import file_sha256
from wt.benchmark_v2.phase6i_priority_tag_reward.train import source_commit

from .protocol import (
    E15_GATE,
    E16_GATE,
    E16_RESULT_COMMIT,
    GATE_BLOCKED,
    GATE_COMPLETE,
    INSTRUCTION_COMMIT,
    OUTPUT,
    PREPARE_COMMIT,
    ROOT,
    SCHEMA_VERSION,
)
from .report import write_report
from .session import diagnose

BASELINE_FIELDS = (
    "scenario_id", "steps", "payload_bits", "served_tagged_bits", "served_ordinary_bits",
    "safety_count", "constraint_count", "final_t", "decision_times",
)
CANDIDATE_FIELDS = (
    "scenario_id", "decision_time", "candidate_slot", "pool_index", "is_baseline", "assignment",
    "payload_bits", "payload_delta_bits", "known_tagged_served_bits", "known_tagged_delta_bits",
    "known_ordinary_served_bits", "future_tagged_served_bits", "future_ordinary_served_bits",
    "all_tagged_served_bits", "all_tagged_delta_bits", "all_tagged_descriptive_flag",
    "age_weighted_service", "age_weighted_delta", "dropped_tagged_bits", "dropped_ordinary_bits",
    "final_queue_bits", "final_age_mean", "final_age_max", "mean_sinr", "mean_sinr_delta",
    "mean_interference", "mean_interference_delta", "safety_count", "safety_delta",
    "constraint_count", "constraint_delta", "min_distance_m", "path_length_m",
    "attribution_gap_bits", "s0", "s1", "s2", "legal", "interpretable", "hindsight_diagnostic",
    "deployable_policy",
)
ATTRIBUTION_FIELDS = (
    "scenario_id", "decision_time", "candidate_slot", "is_baseline", "known_tagged_served_bits",
    "known_ordinary_served_bits", "future_tagged_served_bits", "future_ordinary_served_bits",
    "dropped_tagged_bits", "dropped_ordinary_bits", "attribution_gap_bits", "hindsight_diagnostic",
)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value).replace("\\", "/")
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            return value.item()
        except ValueError:
            return _plain(value.tolist())
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


def _git(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", "core.longpaths=true", *args], cwd=ROOT, capture_output=True, text=True, check=False)


def _require_committed_implementation() -> str:
    head = source_commit()
    if head == INSTRUCTION_COMMIT:
        raise RuntimeError("diagnosis refused to run at the instruction commit")
    if _git(["merge-base", "--is-ancestor", INSTRUCTION_COMMIT, "HEAD"]).returncode != 0:
        raise RuntimeError("HEAD does not descend from the E17 instruction")
    dirty = _git(["status", "--porcelain", "--", "src", "tests"]).stdout.strip()
    if dirty:
        raise RuntimeError("src or tests are dirty; commit the implementation before the diagnosis")
    return head


def _counts(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "s0": int(sum(1 for row in rows if row.get("s0"))),
        "s1": int(sum(1 for row in rows if row.get("s1"))),
        "s2": int(sum(1 for row in rows if row.get("s2"))),
        "s0_scenarios": sorted({int(row["scenario_id"]) for row in rows if row.get("s0")}),
        "s2_scenarios": sorted({int(row["scenario_id"]) for row in rows if row.get("s2")}),
    }


def _summary(body: dict, head: str) -> dict:
    interpreted = [item for item in body["snapshots"] if item.get("interpretation") == "interpreted"]
    gate = GATE_COMPLETE if body["opportunity_published"] and not body["problems"] else GATE_BLOCKED
    rows = []
    for row in body["candidates"]:
        copied = dict(row)
        copied["legal"] = True
        rows.append(copied)
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_gate": gate,
        "e15_gate": E15_GATE,
        "e16_gate": E16_GATE,
        "e15_gate_unchanged": True,
        "e16_gate_unchanged": True,
        "opportunity_evidence": body["opportunity_evidence"],
        "opportunity_published": bool(body["opportunity_published"]),
        "opportunity_counts": _counts(rows),
        "release_eligible": False,
        "merged": False,
        "adopted": False,
        "training_updates": 0,
        "parameter_updates": 0,
        "new_formal_loader": False,
        "alternative_b_activated": False,
        "hindsight_is_not_a_deployable_policy": True,
        "t0_negative_control": bool(body["t0_negative_control"]),
        "interpreted_snapshots": len(interpreted),
        "problems": list(body["problems"]),
        "budget": body["budget"],
        "formal_actor_sha256": body["formal_actor_sha256"],
        "implementation_commit": head,
        "instruction_commit": INSTRUCTION_COMMIT,
        "prepare_commit": PREPARE_COMMIT,
        "e16_result_commit": E16_RESULT_COMMIT,
        "process_seed": body["process_seed"],
        "process_seed_role": body["process_seed_role"],
        "deterministic_algorithms": body["deterministic_algorithms"],
        "torch_threads": body["torch_threads"],
        "candidates": rows,
    }


def _source_manifest() -> dict:
    names = [
        "docs/research/HTW_UrgencyAware_Next_Direction_Instructions_CN.md",
        "results/phase6i_priority_tag_assignment_feasibility/direction_decision.json",
        "results/phase6i_priority_tag_clip_closure/summary.json",
        "results/phase6i_priority_tag_clip_benchmark/summary.json",
        "src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt",
    ]
    package = ROOT / "src/wt/benchmark_v2/phase6i_priority_tag_assignment_feasibility"
    names.extend(sorted(str(path.relative_to(ROOT)).replace("\\", "/") for path in package.glob("*.py")))
    return {"files": {name: file_sha256(ROOT / name) for name in names}}


def execute() -> dict:
    head = _require_committed_implementation()
    body = diagnose()
    summary = _summary(body, head)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_json(OUTPUT / "summary.json", summary)
    _write_json(OUTPUT / "schema.json", {"schema_version": SCHEMA_VERSION, "training": False, "max_simulator_steps": 2544})
    _write_json(OUTPUT / "interface_identity.json", body["identity"])
    _write_json(OUTPUT / "snapshot_manifest.json", body["snapshots"])
    _write_csv(OUTPUT / "baseline_rows.csv", BASELINE_FIELDS, body["baselines"])
    _write_csv(OUTPUT / "candidate_rows.csv", CANDIDATE_FIELDS, summary["candidates"])
    _write_csv(OUTPUT / "attribution_rows.csv", ATTRIBUTION_FIELDS, summary["candidates"])
    _write_json(OUTPUT / "runtime_manifest.json", {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "deterministic_algorithms": bool(body["deterministic_algorithms"]),
        "torch_threads": body["torch_threads"],
        "process_seed": body["process_seed"],
        "process_seed_role": body["process_seed_role"],
    })
    _write_json(OUTPUT / "seed_manifest.json", {
        "env_reset_seed": "scenario_id",
        "assignment_rng_seed": "7000000 + scenario_id",
        "process_seed": 2,
        "process_seed_role": body["process_seed_role"],
        "critic_used": False,
        "model_seed": None,
        "training_updates": 0,
    })
    _write_json(OUTPUT / "source_manifest.json", _source_manifest())
    gaps = [
        "候选帽最多 5 个，不是池内全部合法关联，也不是全局上界。",
        "t=45 的短尾没有比较。",
        "只使用训练场景 321000-321007，没有读取开发、保留或测试场景。",
        "事后最优是 hindsight_diagnostic，不是可部署高层。",
        "本阶段不评估正式 Scheme C，也不说明正式高层已经失败。",
        "替代方案 B 没有启动。",
    ]
    if summary["problems"]:
        gaps.extend(summary["problems"])
    _write_json(OUTPUT / "evidence_gaps.json", gaps)
    blockers = [
        "release_eligible 保持 false。",
        "没有合并。",
        "诊断没有采用新策略，也没有新增正式加载路径。",
        "E15 保持 CLIP_GROUP_BENCHMARK_INCONCLUSIVE。",
        "E16 保持 CLIP_GROUP_LINE_CLOSED_UNADOPTED。",
        "完整历史测试套件的既有失败仍是发布阻塞项。",
    ]
    if summary["scientific_gate"] != GATE_COMPLETE:
        blockers.insert(0, "可行性核对未完成：" + "; ".join(summary["problems"]))
    _write_json(OUTPUT / "release_blockers.json", blockers)
    write_report(summary)
    print(
        f"scientific_gate={summary['scientific_gate']} opportunity={summary['opportunity_evidence']} "
        f"steps={summary['budget']['total_steps']} problems={len(summary['problems'])}",
        flush=True,
    )
    return summary


def _normalize(detail: str) -> str:
    text = str(detail)
    for root in (
        str(ROOT), str(ROOT).replace("\\", "/"),
        r"E:\srdp\wt_e17_baseline", "E:/srdp/wt_e17_baseline",
        r"E:\srdp\wt_e16_baseline", "E:/srdp/wt_e16_baseline",
    ):
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
        "comparison_root": "E:/srdp/wt_e17_baseline",
    }


def attach_tests(summary: dict | None = None) -> dict:
    if summary is None:
        summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    comparison = compare_logs(
        (OUTPUT / "baseline_tests.log").read_text(encoding="utf-8", errors="replace"),
        (OUTPUT / "repository_tests.log").read_text(encoding="utf-8", errors="replace"),
    )
    related = (OUTPUT / "related_tests.log").read_text(encoding="utf-8", errors="replace")
    comparison["related_ok"] = related.strip().endswith("OK")
    summary["test_comparison"] = comparison
    if comparison["new_failure_count"] != 0 or not comparison["related_ok"]:
        summary["scientific_gate"] = GATE_BLOCKED
        summary["problems"] = list(summary.get("problems", [])) + ["new test failures or related tests did not pass"]
        summary["opportunity_published"] = False
    _write_json(OUTPUT / "test_comparison.json", comparison)
    _write_json(OUTPUT / "summary.json", summary)
    write_report(summary)
    print(
        f"new_failure_count={comparison['new_failure_count']} related_ok={comparison['related_ok']} gate={summary['scientific_gate']}",
        flush=True,
    )
    return summary


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "tests":
        attach_tests()
        return
    execute()


if __name__ == "__main__":
    main()
