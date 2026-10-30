"""Replay the frozen matched-mask list. The manifest is the only candidate source."""
from __future__ import annotations

import json
import platform
import random
import subprocess

import numpy as np
import torch

from wt.benchmark_v2.phase6i_priority_tag_assignment_feasibility.attribution import copy_buckets
from wt.benchmark_v2.phase6i_priority_tag_assignment_feasibility.criteria import pareto_slots, posthoc_best
from wt.benchmark_v2.phase6i_priority_tag_assignment_feasibility.session import (
    FrozenAssignmentSession,
    _candidate_row,
    _exogenous_match,
    _hold_metrics,
    _window_match,
    load_actor,
)
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import FORMAL_ACTOR
from wt.benchmark_v2.phase6i_priority_tag_reward.smoke import file_sha256

from .artifacts import write_csv, write_json
from .codec import load_capture
from .criteria import opportunity_evidence
from .detail import bucket_service_rows, detail_matches_aggregate
from .identity import contract_problems, frozen_source_problems, history_problems
from .prepare import require_committed_code
from .protocol import (
    BIT_TOL,
    E17_GATE,
    E17_OPPORTUNITY,
    EXPECTED_SNAPSHOTS,
    FORMAL_ACTOR_SHA256,
    GATE_BLOCKED,
    GATE_COMPLETE,
    HOLD_STEPS,
    HORIZON,
    INSTRUCTION_COMMIT,
    OUTPUT,
    PROCESS_SEED,
    PROCESS_SEED_ROLE,
    ROOT,
    SOURCE_COMMIT,
    StepBudget,
    registered_flags,
    snapshot_file,
)
from .report import write_report
from .selection import dedup_sha256, public_record, selection_sha256

CANDIDATE_FIELDS = (
    "scenario_id", "decision_time", "candidate_slot", "pool_index", "is_baseline", "assignment", "active_mask",
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
DETAIL_FIELDS = (
    "scenario_id", "decision_time", "candidate_slot", "is_baseline", "step_t", "buoy", "insert_index",
    "admission_step", "priority_tag", "known_at_snapshot", "bits_taken", "service_class",
)
SAFETY_FIELDS = (
    "scenario_id", "decision_time", "candidate_slot", "is_baseline", "step_t", "safety", "constraint",
    "min_distance_m", "source",
)
BASELINE_FIELDS = (
    "scenario_id", "decision_time", "reference_match", "repeat_match", "action_gap", "position_gap_m",
    "payload_gap_bits", "queue_gap_bits", "bucket_gap_bits",
)
MANIFEST_RELATIVE = "results/phase6i_priority_tag_assignment_coverage/candidate_selection_manifest.json"


def _git(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", "core.longpaths=true", *args], cwd=ROOT, capture_output=True, text=True, check=False)


def _require_frozen_manifest() -> tuple[str, dict]:
    head = require_committed_code()
    if _git(["rev-parse", "--verify", "--quiet", f"HEAD:{MANIFEST_RELATIVE}"]).returncode != 0:
        raise RuntimeError("selection manifest is not committed")
    dirty = _git(
        [
            "status", "--porcelain", "--", MANIFEST_RELATIVE,
            "results/phase6i_priority_tag_assignment_coverage/snapshots",
            "results/phase6i_priority_tag_assignment_coverage/prepare_status.json",
            "src", "tests",
        ]
    ).stdout.strip()
    if dirty:
        raise RuntimeError("selection inputs are dirty; replay refused to reselect")
    manifest = json.loads((OUTPUT / "candidate_selection_manifest.json").read_text(encoding="utf-8"))
    records = manifest.get("records") or []
    if selection_sha256(records) != manifest.get("selection_sha256"):
        raise RuntimeError("selection manifest hash does not match its records")
    if not manifest.get("replay_authorized") or len(records) != int(EXPECTED_SNAPSHOTS):
        raise RuntimeError("selection manifest is not an authorized 24-state list")
    return head, manifest


def _mask_problems(record: dict) -> list[str]:
    problems = []
    selected = record["selected"]
    expected = 1 + min(int(record["matched_alternative_count"]), 4)
    if len(selected) != expected or len(selected) > 5:
        problems.append("frozen selection length breaks the cap")
    if len({tuple(item["assignment"]) for item in selected}) != len(selected):
        problems.append("frozen selection contains a duplicate")
    if sum(1 for item in selected if item["is_baseline"]) != 1:
        problems.append("frozen selection lost the baseline flag")
    for item in selected:
        if [int(bit) for bit in item["active_mask"]] != [int(bit) for bit in record["baseline_mask"]]:
            problems.append("frozen selection contains a different activity mask")
    return problems


def _roll(session: FrozenAssignmentSession, loaded: dict, record: dict, assignment: np.ndarray, kind: str) -> tuple[list[dict], list[dict]]:
    session.restore(loaded["capture"])
    pool, index, drawn = session.draw()
    if int(index) != int(loaded["baseline_index"]) or not np.array_equal(drawn, np.asarray(loaded["baseline_assignment"])):
        raise RuntimeError("restored draw does not reproduce the reference sample")
    if public_record(pool, index, int(record["scenario_id"]), int(record["decision_time"])) != record:
        raise RuntimeError("recomputed matched-mask list does not match the frozen manifest")
    if dedup_sha256([int(item) for item in loaded["dedup_first_index"]]) != record["dedup_sha256"]:
        raise RuntimeError("restored dedup map does not match the frozen manifest")
    session.install(np.asarray(assignment, dtype=np.int64))
    session.hold_known_ids = {int(item) for item in loaded["capture"]["known"]["known_ids"]}
    rows = []
    details = []
    try:
        for _ in range(int(HOLD_STEPS)):
            if int(session.env.t) >= int(HORIZON):
                raise RuntimeError("hold crossed the episode horizon")
            before = copy_buckets(session.ledger)
            served_before = np.asarray(session.env.cumulative_served_bits, dtype=np.float64).copy()
            known = set(session.hold_known_ids)
            time_step = int(session.env.t)
            session.budget.add(kind)
            row = session.physical()
            served = np.asarray(session.env.cumulative_served_bits, dtype=np.float64) - served_before
            detail = bucket_service_rows(before, served, time_step, known)
            if not detail_matches_aggregate(detail, served, before, time_step, known, session._urgency):
                raise RuntimeError("bucket detail does not match the aggregate attribution")
            rows.append(row)
            details.extend(detail)
    finally:
        session.hold_known_ids = None
    return rows, details


def _stamp(record: dict, item: dict, rows: list[dict]) -> list[dict]:
    stamped = []
    for row in rows:
        stamped.append(
            {
                "scenario_id": int(record["scenario_id"]),
                "decision_time": int(record["decision_time"]),
                "candidate_slot": int(item["slot"]),
                "is_baseline": bool(item["is_baseline"]),
                **row,
            }
        )
    return stamped


def _safety(scenario: int, decision_time: int, slot: int, is_baseline: bool, rows: list[dict]) -> list[dict]:
    return [
        {
            "scenario_id": int(scenario),
            "decision_time": int(decision_time),
            "candidate_slot": int(slot),
            "is_baseline": bool(is_baseline),
            "step_t": int(row["t"]),
            "safety": int(row["safety"]),
            "constraint": int(row["constraint"]),
            "min_distance_m": float(row["min_distance_m"]),
            "source": "replay",
        }
        for row in rows
        if int(row["safety"]) or int(row["constraint"])
    ]


def _counts(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "s0": int(sum(1 for row in rows if row.get("s0"))),
        "s1": int(sum(1 for row in rows if row.get("s1"))),
        "s2": int(sum(1 for row in rows if row.get("s2"))),
        "s0_scenarios": sorted({int(row["scenario_id"]) for row in rows if row.get("s0")}),
        "s2_scenarios": sorted({int(row["scenario_id"]) for row in rows if row.get("s2")}),
    }


def _coverage(records: list[dict], statuses: list[dict]) -> dict:
    return {
        "snapshots": len(records),
        "snapshots_with_same_mask_alternative": int(sum(1 for record in records if int(record["matched_alternative_count"]) > 0)),
        "snapshots_without_same_mask_alternative": int(sum(1 for record in records if int(record["matched_alternative_count"]) == 0)),
        "selected_alternatives": int(sum(max(len(record["selected"]) - 1, 0) for record in records)),
        "matched_unique_alternatives": int(sum(int(record["matched_alternative_count"]) for record in records)),
        "snapshots_with_interpretable_alternative": int(sum(1 for item in statuses if int(item.get("interpretable_alternatives", 0)) >= 1)),
    }


def _finish(head: str, manifest: dict, problems: list[str], budget: StepBudget, candidates: list[dict], statuses: list[dict], implementation: str | None) -> dict:
    interpreted = [item for item in statuses if item.get("interpreted")]
    publish = not problems and len(interpreted) == int(EXPECTED_SNAPSHOTS)
    measured = opportunity_evidence(candidates, statuses) if publish else None
    return {
        "schema_version": "assignment_coverage_diag_v1",
        "scientific_gate": GATE_COMPLETE if publish else GATE_BLOCKED,
        "opportunity_evidence": measured if publish else None,
        "opportunity_published": bool(publish),
        "measured_opportunity": measured,
        "e17_scientific_gate": E17_GATE,
        "e17_opportunity_evidence": E17_OPPORTUNITY,
        "method_change": "activity-matched stratified sample replaced prefix truncation",
        "problems": problems,
        "budget": budget.as_dict(),
        "implementation_commit": implementation,
        "selection_commit": head,
        "instruction_commit": INSTRUCTION_COMMIT,
        "source_commit": SOURCE_COMMIT,
        "formal_actor_sha256": file_sha256(FORMAL_ACTOR),
        "selection_sha256": manifest.get("selection_sha256"),
        "reconciliation_ok": not problems,
        "coverage": _coverage(manifest.get("records") or [], statuses),
        "opportunity_counts": _counts(candidates),
        "t0_negative_control": not any("t=0" in item for item in problems) and publish,
        "process_seed": int(PROCESS_SEED),
        "process_seed_role": PROCESS_SEED_ROLE,
        "snapshot_status": statuses,
        **registered_flags(),
    }


def _source_manifest() -> dict:
    package = ROOT / "src" / "wt" / "benchmark_v2" / "phase6i_priority_tag_assignment_coverage"
    names = sorted(str(path.relative_to(ROOT)).replace("\\", "/") for path in package.glob("*.py"))
    names.append("tests/test_priority_tag_assignment_coverage.py")
    return {"files": {name: file_sha256(ROOT / name) for name in names if (ROOT / name).is_file()}}


def _write(summary: dict, candidates: list[dict], details: list[dict], safety: list[dict], baseline_rows: list[dict]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "summary.json", summary)
    write_json(OUTPUT / "schema.json", {"schema_version": "assignment_coverage_diag_v1", "training": False, "max_simulator_steps": 2544, "release_eligible": False})
    write_csv(OUTPUT / "candidate_rows.csv", CANDIDATE_FIELDS, candidates)
    write_csv(OUTPUT / "attribution_rows.csv", ATTRIBUTION_FIELDS, candidates)
    write_csv(OUTPUT / "bucket_service_rows.csv", DETAIL_FIELDS, details)
    write_csv(OUTPUT / "safety_events.csv", SAFETY_FIELDS, safety)
    write_csv(OUTPUT / "baseline_rows.csv", BASELINE_FIELDS, baseline_rows)
    write_json(OUTPUT / "snapshot_status.json", summary.get("snapshot_status") or [])
    write_json(OUTPUT / "runtime_manifest.json", {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "deterministic_algorithms": True,
        "torch_threads": int(torch.get_num_threads()),
        "process_seed": int(PROCESS_SEED),
        "process_seed_role": PROCESS_SEED_ROLE,
    })
    write_json(OUTPUT / "seed_manifest.json", {
        "env_reset_seed": "scenario_id",
        "assignment_rng_seed": "7000000 + scenario_id",
        "candidate_seed": "9300000 + 100*scenario_id + decision_time",
        "process_seed": int(PROCESS_SEED),
        "process_seed_role": PROCESS_SEED_ROLE,
        "model_seed": None,
        "training_updates": 0,
    })
    write_json(OUTPUT / "source_manifest.json", _source_manifest())
    gaps = [
        "同掩码分层样本不是联合池的穷尽搜索，也不是全局上界。",
        "更换哪些 UAV 参与可能带来的收益不在本样本里。",
        "t=45 的短尾没有比较。",
        "只使用训练场景 321000-321007。",
        "事后最优是 hindsight_diagnostic，不是可部署高层。",
        "本阶段不评估正式 Scheme C，也不改写 E17。",
        "没有用完的步数不能拿去加场景或加候选。",
    ]
    gaps.extend(summary.get("problems") or [])
    blockers = [
        "release_eligible 保持 false。",
        "没有合并。",
        "没有训练，也没有新的正式加载路径。",
        "E17 保持 ASSIGNMENT_FEASIBILITY_COMPLETE / NOT_OBSERVED_IN_REGISTERED_POOL。",
        "E15 保持 CLIP_GROUP_BENCHMARK_INCONCLUSIVE。",
        "E16 保持 CLIP_GROUP_LINE_CLOSED_UNADOPTED。",
        "完整历史测试套件的既有失败仍是发布阻塞项。",
    ]
    if summary["scientific_gate"] != GATE_COMPLETE:
        blockers.insert(0, "覆盖诊断未完成：" + "; ".join(summary.get("problems") or []))
    write_json(OUTPUT / "evidence_gaps.json", gaps)
    write_json(OUTPUT / "release_blockers.json", blockers)
    write_report(summary)


def _annotate(row: dict, item: dict) -> dict:
    row["active_mask"] = list(item["active_mask"])
    row["legal"] = True
    return row


def replay() -> dict:
    head, manifest = _require_frozen_manifest()
    problems = contract_problems() + history_problems() + frozen_source_problems()
    if file_sha256(FORMAL_ACTOR) != FORMAL_ACTOR_SHA256:
        problems.append("formal low-level checkpoint hash is not the registered SHA-256")
    budget = StepBudget(reference=int(manifest["reference_steps"]))
    budget.add("reference", 0)
    records = list(manifest["records"])
    candidates: list[dict] = []
    details: list[dict] = []
    safety: list[dict] = []
    baseline_rows: list[dict] = []
    statuses: list[dict] = []
    implementation = None
    status_path = OUTPUT / "prepare_status.json"
    if status_path.is_file():
        implementation = json.loads(status_path.read_text(encoding="utf-8")).get("implementation_commit")
    if not problems:
        random.seed(PROCESS_SEED)
        np.random.seed(PROCESS_SEED)
        torch.manual_seed(PROCESS_SEED)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        actor = load_actor()
        for record in records:
            mask_problems = _mask_problems(record)
            if mask_problems:
                problems.extend(f"scenario {record['scenario_id']} t={record['decision_time']} {item}" for item in mask_problems)
                break
            print(f"replay scenario {record['scenario_id']} t={record['decision_time']}", flush=True)
            loaded = load_capture(snapshot_file(int(record["scenario_id"]), int(record["decision_time"])))
            if int(record["decision_time"]) == 0 and float(loaded["capture"]["known"]["known_tagged_bits"]) > float(BIT_TOL):
                problems.append(f"scenario {record['scenario_id']} t=0 already has known tagged bits")
                break
            session = FrozenAssignmentSession(actor, int(record["scenario_id"]), budget)
            status = {
                "scenario_id": int(record["scenario_id"]),
                "decision_time": int(record["decision_time"]),
                "interpreted": False,
                "interpretable_alternatives": 0,
            }
            try:
                baseline_item = next(item for item in record["selected"] if item["is_baseline"])
                baseline_assignment = np.asarray(baseline_item["assignment"], dtype=np.int64)
                baseline_play, baseline_detail = _roll(session, loaded, record, baseline_assignment, "candidate")
                matched = _window_match(loaded["reference_window"], baseline_play)
                repeat_play, _repeat_detail = _roll(session, loaded, record, baseline_assignment, "repeat")
                repeated = _window_match(baseline_play, repeat_play)
                baseline_rows.append(
                    {
                        "scenario_id": int(record["scenario_id"]),
                        "decision_time": int(record["decision_time"]),
                        "reference_match": bool(matched["ok"]),
                        "repeat_match": bool(repeated["ok"]),
                        "action_gap": float(matched["action_gap"]),
                        "position_gap_m": float(matched["position_gap_m"]),
                        "payload_gap_bits": float(matched["payload_gap_bits"]),
                        "queue_gap_bits": float(matched["queue_gap_bits"]),
                        "bucket_gap_bits": float(matched["bucket_gap_bits"]),
                    }
                )
                start_payload = float(loaded["capture"]["env"]["payload_bits"])
                baseline_metrics = _hold_metrics(baseline_play, start_payload)
                baseline_candidate = _annotate(
                    _candidate_row(
                        int(record["scenario_id"]), int(record["decision_time"]), int(baseline_item["slot"]),
                        int(baseline_item["pool_index"]), baseline_assignment, baseline_metrics, baseline_metrics,
                        is_baseline=True, interpretable=bool(matched["ok"] and repeated["ok"]),
                    ),
                    baseline_item,
                )
                candidates.append(baseline_candidate)
                details.extend(_stamp(record, baseline_item, baseline_detail))
                safety.extend(_safety(int(record["scenario_id"]), int(record["decision_time"]), int(baseline_item["slot"]), True, baseline_play))
                if not matched["ok"] or not repeated["ok"]:
                    problems.append(f"scenario {record['scenario_id']} t={record['decision_time']} baseline replay mismatch")
                    break
                interpretable_alternatives = 0
                exogenous_failed = False
                for item in record["selected"]:
                    if item["is_baseline"]:
                        continue
                    branch_rows, branch_detail = _roll(session, loaded, record, np.asarray(item["assignment"], dtype=np.int64), "candidate")
                    exogenous = _exogenous_match(baseline_play, branch_rows)
                    row = _annotate(
                        _candidate_row(
                            int(record["scenario_id"]), int(record["decision_time"]), int(item["slot"]),
                            int(item["pool_index"]), item["assignment"], baseline_metrics, _hold_metrics(branch_rows, start_payload),
                            is_baseline=False, interpretable=bool(exogenous["ok"]),
                        ),
                        item,
                    )
                    candidates.append(row)
                    details.extend(_stamp(record, item, branch_detail))
                    safety.extend(_safety(int(record["scenario_id"]), int(record["decision_time"]), int(item["slot"]), False, branch_rows))
                    if not exogenous["ok"]:
                        exogenous_failed = True
                        problems.append(f"scenario {record['scenario_id']} t={record['decision_time']} exogenous arrivals diverged")
                        break
                    interpretable_alternatives += 1
                if exogenous_failed:
                    break
                if int(record["decision_time"]) == 0 and float(baseline_candidate["known_tagged_served_bits"]) > float(BIT_TOL):
                    problems.append(f"scenario {record['scenario_id']} t=0 served known tagged bits")
                    break
                for row in candidates:
                    if int(row["scenario_id"]) == int(record["scenario_id"]) and int(row["decision_time"]) == 0:
                        if float(row["known_tagged_served_bits"]) > float(BIT_TOL):
                            problems.append(f"scenario {record['scenario_id']} t=0 served known tagged bits")
                if problems:
                    break
                group = [
                    row for row in candidates
                    if int(row["scenario_id"]) == int(record["scenario_id"]) and int(row["decision_time"]) == int(record["decision_time"])
                ]
                status["interpreted"] = True
                status["interpretable_alternatives"] = int(interpretable_alternatives)
                status["posthoc_best"] = posthoc_best(group)
                status["pareto_slots"] = pareto_slots(group)
                if status["posthoc_best"] is not None:
                    status["posthoc_best"]["deployable_policy"] = False
            except RuntimeError as exc:
                problems.append(f"scenario {record['scenario_id']} t={record['decision_time']}: {exc}")
                break
            statuses.append(status)
    summary = _finish(head, manifest, problems, budget, candidates, statuses, implementation)
    _write(summary, candidates, details, safety, baseline_rows)
    print(
        f"scientific_gate={summary['scientific_gate']} opportunity={summary['opportunity_evidence']} "
        f"steps={summary['budget']['total_steps']} problems={len(summary['problems'])}",
        flush=True,
    )
    return summary
