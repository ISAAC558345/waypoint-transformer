"""Finish the registered comparison after Stage A. Do not retune the gate."""
from __future__ import annotations

import csv
import json
import platform
import traceback

import numpy as np
import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import DEV_POOL
from wt.benchmark_v2.phase6i_priority_tag_direct_service_reward.evaluate import (
    evaluate_l3,
    evaluate_saved,
    load_l3,
    load_reference,
    preflight_l0,
)
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import WATCHED_PATHS
from wt.benchmark_v2.phase6i_priority_tag_reward.smoke import file_sha256, watched_hashes
from wt.benchmark_v2.phase6i_priority_tag_reward.train import source_commit
from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.reconcile import load_saved_rows, saved_index

from .criteria import choose_gate, score_l4
from .evaluate import evaluate_l4, load_l4
from .protocol import (
    E11_RESULT_COMMIT,
    FORMAL_ACTOR_SHA256,
    GATE_BLOCKED,
    GATE_FAIL,
    INSTRUCTION_COMMIT,
    OUTPUT,
    REPORT,
    ROOT,
    SCHEMA_VERSION,
    TRAIN_SEEDS,
    VARIANTS,
)
from .report import render_report
from .stage_a import _cell, _plain
from .train import assert_sources_committed, build_pair, train_group


SCENARIO_FIELDS = (
    "variant",
    "model_seed",
    "scenario_id",
    "steps",
    "payload_bits",
    "urgency_served_bits",
    "served_tagged_bits",
    "served_ordinary_bits",
    "dropped_tagged_bits",
    "dropped_ordinary_bits",
    "dropped_bits",
    "final_queue_bits",
    "final_tagged_queue_bits",
    "final_ordinary_queue_bits",
    "queue_age_mean",
    "queue_age_max",
    "sinr_mean",
    "interference_mean",
    "path_length_m",
    "projection_rate",
    "safety_violations",
    "constraint_failures",
    "return_base",
    "return_augmented",
    "bonus_sum",
    "bonus_age_sum",
    "bonus_tag_sum",
    "age_service_bits",
    "tag_increment_service_bits",
    "combined_urgency_served_bits",
    "served_age_mean_ordinary",
    "served_age_mean_tagged",
    "max_ledger_gap_bits",
    "env_reset_seed",
    "assignment_rng_seed",
    "process_seed_role",
    "model_seed_role",
)


def _write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=2), encoding="utf-8")


def _scenarios(rows: list[dict]) -> None:
    path = OUTPUT / "scenario_rows.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SCENARIO_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key, "")) for key in SCENARIO_FIELDS})


def parse_unittest(text: str) -> dict:
    """Collect real test IDs. The FAILED summary tail is not a test."""
    blocks = []
    current = None
    ran = None
    for line in text.splitlines():
        if line.startswith("FAIL:") or line.startswith("ERROR:"):
            if current is not None:
                blocks.append(current)
            kind, test_id = line.split(":", 1)
            current = {"kind": kind.strip(), "test_id": test_id.strip(), "body": []}
            continue
        if line.startswith("Ran ") and "test" in line:
            ran = int(line.split()[1])
            if current is not None:
                blocks.append(current)
                current = None
            continue
        if current is not None:
            current["body"].append(line)
    if current is not None:
        blocks.append(current)
    parsed = []
    for block in blocks:
        if block["test_id"].startswith("FAILED"):
            continue
        exception = ""
        detail = ""
        for line in block["body"]:
            stripped = line.strip()
            if not stripped or set(stripped) == {"="} or stripped.startswith("File ") or stripped.startswith("Traceback") or stripped.startswith("^"):
                continue
            name = stripped.split(":", 1)[0]
            if name.endswith("Error") or name.endswith("Exception") or name in {"AssertionError", "RuntimeError"}:
                exception = name
                detail = stripped
        parsed.append({"kind": block["kind"], "test_id": block["test_id"], "exception": exception, "detail": detail})
    return {"ran": ran, "failures": parsed}


def _normalize_detail(detail: str) -> str:
    roots = (
        str(ROOT),
        str(ROOT).replace("\\", "/"),
        r"E:\srdp\wt_e12_baseline",
        "E:/srdp/wt_e12_baseline",
    )
    text = str(detail)
    for root in roots:
        text = text.replace(root, "<ROOT>")
    return text


def compare_logs(baseline_text: str, current_text: str) -> dict:
    baseline = parse_unittest(baseline_text)
    current = parse_unittest(current_text)
    baseline_ids = {item["test_id"]: item for item in baseline["failures"]}
    historical = []
    new_items = []
    for item in current["failures"]:
        previous = baseline_ids.get(item["test_id"])
        same = previous is not None and previous["exception"] == item["exception"] and _normalize_detail(previous["detail"]) == _normalize_detail(item["detail"])
        if same:
            historical.append(item)
        else:
            new_items.append({**item, "baseline_exception": None if previous is None else previous["exception"], "baseline_detail": None if previous is None else previous["detail"]})
    return {
        "baseline_ran": baseline["ran"],
        "repository_ran": current["ran"],
        "baseline_failure_count": len(baseline["failures"]),
        "repository_failure_count": len(current["failures"]),
        "historical": historical,
        "new_failures": new_items,
        "new_failure_count": len(new_items),
    }


def _source_hashes() -> dict[str, str]:
    paths = sorted((ROOT / "src" / "wt" / "benchmark_v2" / "phase6i_priority_tag_queue_lookahead").glob("*.py"))
    paths.append(ROOT / "tests" / "test_priority_tag_queue_lookahead.py")
    return {str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path) for path in paths}


def _publish(summary: dict) -> None:
    summary["schema_version"] = SCHEMA_VERSION
    summary["instruction_commit"] = INSTRUCTION_COMMIT
    summary["e11_result_commit"] = E11_RESULT_COMMIT
    summary["merged"] = False
    summary["release_eligible"] = False
    _write(OUTPUT / "summary.json", summary)
    _write(OUTPUT / "release_blockers.json", {"release_eligible": False, "blockers": summary.get("release_blockers", [])})
    _write(OUTPUT / "evidence_gaps.json", {"gaps": summary.get("evidence_gaps", [])})
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render_report(summary), encoding="utf-8")


def _blocked(reason: str, stage: dict, tests: dict, extra: dict | None = None) -> dict:
    summary = {
        "scientific_gate": GATE_BLOCKED,
        "implementation_commit": source_commit(),
        "stage_a": stage,
        "tests": tests,
        "release_blockers": [reason, "完整测试仍有未解决失败或正式分支差异不是本实验的最小差异。"],
        "evidence_gaps": [
            "标签是 synthetic tag。",
            "开发池已被前期实验使用。",
            "E10 训练实现字节没有追溯摘要，本轮不回填。",
        ],
        "stop_reason": reason,
    }
    if extra:
        summary.update(extra)
    _publish(summary)
    return summary


def execute() -> dict:
    stage = json.loads((OUTPUT / "stage_a_summary.json").read_text(encoding="utf-8"))
    comparison = json.loads((OUTPUT / "test_comparison.json").read_text(encoding="utf-8"))
    tests = {
        "related_ok": bool(comparison.get("related_ok")),
        "new_failure_count": int(comparison.get("new_failure_count", 0)),
        "baseline_ran": comparison.get("baseline_ran"),
        "repository_ran": comparison.get("repository_ran"),
        "historical_ids": [item["test_id"] for item in comparison.get("historical", [])],
        "new_ids": [item["test_id"] for item in comparison.get("new_failures", [])],
    }
    if stage.get("feasibility_status") != "pass" or not stage.get("safety_matched_saved_rows") or not stage.get("formal_actor_hash_ok"):
        return _blocked("阶段 A 的可行性、安全复现或正式哈希没有通过。", stage, tests)
    if not tests["related_ok"] or tests["new_failure_count"] != 0:
        return _blocked("相关测试未通过，或完整测试出现基线中没有的失败。", stage, tests)
    implementation = source_commit()
    if implementation == INSTRUCTION_COMMIT:
        return _blocked("当前提交只含实验指令，训练实现还没有提交。", stage, tests)
    try:
        assert_sources_committed(implementation)
    except RuntimeError as exc:
        return _blocked(str(exc), stage, tests)
    source_hashes = _source_hashes()
    frozen_before = watched_hashes()
    if frozen_before["src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt"] != FORMAL_ACTOR_SHA256:
        return _blocked("正式低层 checkpoint 哈希与登记值不一致。", stage, tests)
    trained = []
    try:
        for seed in TRAIN_SEEDS:
            bundle = build_pair(int(seed))
            for variant in VARIANTS:
                trained.append(train_group(int(seed), variant, bundle, implementation, source_hashes))
    except Exception as exc:
        _write(OUTPUT / "training_error.json", {"error": str(exc), "trace": traceback.format_exc(), "completed": trained})
        return _blocked(f"训练在注册预算内停止：{exc}", stage, tests, {"completed_training": trained})
    frozen_after = watched_hashes()
    if frozen_before != frozen_after:
        return _blocked("训练前后冻结核心文件哈希不一致。", stage, tests, {"completed_training": trained})
    index = saved_index(load_saved_rows())
    preflight = preflight_l0(index[("L0", 0, 321101)])
    if not preflight["matches_saved"]:
        return _blocked("L0 预检没有对齐保存行。", stage, tests, {"preflight": preflight, "completed_training": trained})
    rows = []
    l0_actor = load_reference("L0", 0)
    for scenario in DEV_POOL:
        row, _status = evaluate_saved("L0", 0, int(scenario), index[("L0", 0, int(scenario))], l0_actor)
        row["process_seed_role"] = "historical_inherited_process_context"
        row["model_seed_role"] = "frozen_reference_seed"
        rows.append(row)
        print(f"L0 {int(scenario)} tagged={row['served_tagged_bits']:.1f}", flush=True)
    l3_actors = {int(seed): load_l3(int(seed)) for seed in TRAIN_SEEDS}
    l4_actors = {(variant, int(seed)): load_l4(variant, int(seed)) for variant in VARIANTS for seed in TRAIN_SEEDS}
    for seed in TRAIN_SEEDS:
        for scenario in DEV_POOL:
            row = evaluate_l3(l3_actors[int(seed)], int(seed), int(scenario))
            row["process_seed_role"] = "historical_inherited_process_context"
            row["model_seed_role"] = "frozen_reference_seed"
            rows.append(row)
        print(f"L3 seed {int(seed)} done", flush=True)
    for variant in VARIANTS:
        for seed in TRAIN_SEEDS:
            for scenario in DEV_POOL:
                row = evaluate_l4(l4_actors[(variant, int(seed))], variant, int(seed), int(scenario))
                rows.append(row)
            print(f"{variant} seed {int(seed)} done", flush=True)
    _scenarios(rows)
    score = score_l4(rows)
    gate = choose_gate(tests_ok=True, train_ok=True, hashes_ok=True, feasibility="pass", score=score)
    blockers = [
        "相对 origin/feature/maritime-wt-stage1a 的历史包含未合并实验，不是本实验的最小差异。",
        "完整测试中与干净基线相同的既有失败尚未解决。",
    ]
    if gate != "QUEUE_LOOKAHEAD_BENCHMARK_PASS":
        blockers.insert(0, f"科学 Gate 为 {gate}。")
    if gate == GATE_FAIL:
        blockers.append("失败的受控对照保留原样，不回写成奖励系数选择依据。")
    gaps = [
        "synthetic tag 在入队时已知，这不是对真实业务标记的声称。",
        "开发池 321096–321127 已经用于前期研究，结论不能外推。",
        "L3 对照是描述性历史参考，其训练来源没有实现摘要，不用于严格归因。",
        "E10 的 source_commit 只指向指令提交，本轮不补造当时的实现哈希。",
        "critic 没有接收前瞻列，因此本结果不是 critic 信息实验。",
        "requirements.txt 中的 numpy==1.21.6 与 torch==1.13.0 不是本次运行版本，没有降级。",
    ]
    if comparison.get("workdir_note"):
        gaps.append(str(comparison["workdir_note"]))
    manifest = {
        "instruction_commit": INSTRUCTION_COMMIT,
        "implementation_commit": implementation,
        "source_commit_at_runtime": source_commit(),
        "source_file_sha256": source_hashes,
        "frozen_before": frozen_before,
        "frozen_after": frozen_after,
        "formal_actor_sha256": FORMAL_ACTOR_SHA256,
        "watched_paths": list(WATCHED_PATHS),
        "checkpoints": trained,
        "preflight_l0_321101": preflight,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": bool(torch.cuda.is_available()),
            "platform": platform.platform(),
        },
        "rng_roles": {
            "training_model_seed": "0,1,2 reseeded before each group",
            "evaluation_process": "historical_inherited_process_context",
            "evaluation_assignment": "7000000 + scenario_id",
            "environment_reset": "scenario_id",
        },
    }
    _write(OUTPUT / "checkpoint_manifest.json", {"checkpoints": trained})
    _write(OUTPUT / "source_manifest.json", {"sources": source_hashes, "frozen": frozen_after})
    _write(OUTPUT / "runtime_manifest.json", manifest)
    summary = {
        "scientific_gate": gate,
        "implementation_commit": implementation,
        "stage_a": stage,
        "tests": tests,
        "score": score,
        "scenario_rows": len(rows),
        "training": trained,
        "preflight": preflight,
        "release_blockers": blockers,
        "evidence_gaps": gaps,
    }
    _publish(summary)
    print(gate, flush=True)
    return summary


if __name__ == "__main__":
    execute()
