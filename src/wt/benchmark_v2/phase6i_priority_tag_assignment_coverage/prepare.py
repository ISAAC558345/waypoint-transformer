"""Rebuild the eight frozen references and freeze the matched-mask list before any alternative replay."""
from __future__ import annotations

import random
import subprocess

import numpy as np
import torch

from wt.benchmark_v2.phase6e.generators import build_pools
from wt.benchmark_v2.phase6i_priority_tag_assignment_feasibility.candidates import pool_sha256
from wt.benchmark_v2.phase6i_priority_tag_assignment_feasibility.session import (
    FrozenAssignmentSession,
    _ledger_gap,
    _same,
    _summarize_episode,
    load_actor,
)
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import FORMAL_ACTOR
from wt.benchmark_v2.phase6i_priority_tag_reward.smoke import file_sha256
from wt.benchmark_v2.phase6i_priority_tag_reward.train import source_commit

from .artifacts import write_csv, write_json
from .codec import dump_capture, load_capture
from .coverage import (
    audit_saved_prefix,
    compare_episode,
    compare_snapshot,
    load_e17_baselines,
    load_e17_snapshots,
    pool_coverage_row,
)
from .identity import contract_problems, frozen_source_problems, history_problems
from .protocol import (
    DECISION_TIMES,
    E17_GATE,
    E17_OPPORTUNITY,
    EXCLUDED_TAIL_TIME,
    EXPECTED_SNAPSHOTS,
    FORMAL_ACTOR_SHA256,
    GATE_BLOCKED,
    HOLD_STEPS,
    HORIZON,
    INSTRUCTION_COMMIT,
    OUTPUT,
    PROCESS_SEED,
    PROCESS_SEED_ROLE,
    ROOT,
    SCENARIOS,
    SOURCE_COMMIT,
    StepBudget,
    registered_flags,
    snapshot_file,
)
from .report import write_report
from .selection import active_mask, dedup_sha256, public_record, selection_sha256

PREFIX_FIELDS = (
    "source", "scenario_id", "decision_time", "candidate_slot", "pool_index", "is_baseline",
    "assignment", "active_count", "active_mask", "matches_saved",
)
POOL_FIELDS = (
    "scenario_id", "decision_time", "pool_size", "unique_count", "duplicate_count", "illegal_count",
    "baseline_pool_index", "baseline_assignment", "baseline_active_count", "baseline_mask",
    "matched_alternative_count", "old_prefix_masks", "unique_active_count_hist", "unique_active_mask_hist",
    "per_uav_topk", "tagged_target_coverage",
)


def _git(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", "core.longpaths=true", *args], cwd=ROOT, capture_output=True, text=True, check=False)


def require_committed_code() -> str:
    head = source_commit()
    if head == INSTRUCTION_COMMIT:
        raise RuntimeError("coverage diagnosis refused to run at the instruction commit")
    if _git(["merge-base", "--is-ancestor", INSTRUCTION_COMMIT, "HEAD"]).returncode != 0:
        raise RuntimeError("HEAD does not descend from the E18 instruction")
    if _git(["status", "--porcelain", "--", "src", "tests"]).stdout.strip():
        raise RuntimeError("src or tests are dirty; commit the implementation before the diagnosis")
    return head


def _run_reference(session: FrozenAssignmentSession) -> tuple[list[dict], list[dict]]:
    session.reset_episode()
    rows = []
    snapshots = []
    allowed = set(DECISION_TIMES) | {int(EXCLUDED_TAIL_TIME)}
    while int(session.env.t) < int(HORIZON):
        if session.pending <= 0:
            if int(session.env.t) not in allowed:
                raise RuntimeError(f"unexpected assignment boundary at t={int(session.env.t)}")
            snap = session.capture()
            snap["step_index"] = len(rows)
            pools, packed, illegal = build_pools(session.env, (("G1_RATE", 5),))
            if int(illegal) != 0:
                raise RuntimeError("candidate pool contains an illegal assignment")
            pool, index, chosen = session.draw()
            if pool_sha256(pool) != pool_sha256(pools[("G1_RATE", 5)]):
                raise RuntimeError("pool changed between the coverage read and the draw")
            session.install(chosen)
            if int(snap["t"]) in DECISION_TIMES:
                snap["baseline_index"] = int(index)
                snap["baseline_assignment"] = chosen.copy()
                snap["pool"] = pool
                snap["illegal_count"] = int(illegal)
                snap["tops"] = np.asarray(packed["tops"][("G1_RATE", 5)]).copy()
                snapshots.append(snap)
        session.budget.add("reference")
        rows.append(session.physical())
    if [int(item["t"]) for item in snapshots] != list(DECISION_TIMES):
        raise RuntimeError("reference trajectory did not snapshot t=0, 15, and 30")
    if len(rows) != int(HORIZON):
        raise RuntimeError("reference trajectory length is not the registered horizon")
    return rows, snapshots


def _capture_problems(original: dict, restored: dict) -> list[str]:
    problems = []
    if int(original["t"]) != int(restored["t"]) or int(original["pending"]) != int(restored["pending"]):
        problems.append("restored time or pending hold does not match the capture")
    for name, value in original["env"].items():
        other = restored["env"][name]
        if isinstance(value, np.ndarray):
            if not np.array_equal(value, other):
                problems.append(f"restored env field {name} is not bit-exact")
        elif float(value) != float(other):
            problems.append(f"restored env field {name} changed")
    if _ledger_gap(original["ledger"], restored["ledger"]) > 0.0:
        problems.append("restored ledger is not the captured ledger")
    if not _same(original.get("option"), restored.get("option")):
        problems.append("restored option does not match the capture")
    if not _same(original["assignment_rng"], restored["assignment_rng"]) or not _same(original["env_rng"], restored["env_rng"]):
        problems.append("restored RNG state does not match the capture")
    if not _same(original["process"], restored["process"]):
        problems.append("restored process RNG does not match the capture")
    if not np.array_equal(original["last_movement"], restored["last_movement"]):
        problems.append("restored last movement does not match the capture")
    return problems


def _summary(head: str, problems: list[str], budget: StepBudget, digest: str | None, authorized: bool) -> dict:
    return {
        "schema_version": "assignment_coverage_diag_v1",
        "scientific_gate": GATE_BLOCKED,
        "opportunity_evidence": None,
        "opportunity_published": False,
        "measured_opportunity": None,
        "e17_scientific_gate": E17_GATE,
        "e17_opportunity_evidence": E17_OPPORTUNITY,
        "method_change": "activity-matched stratified sample replaced prefix truncation",
        "problems": problems,
        "budget": budget.as_dict(),
        "implementation_commit": head,
        "instruction_commit": INSTRUCTION_COMMIT,
        "source_commit": SOURCE_COMMIT,
        "formal_actor_sha256": file_sha256(FORMAL_ACTOR),
        "selection_sha256": digest,
        "reconciliation_ok": False,
        "replay_authorized": bool(authorized),
        "coverage": {},
        "opportunity_counts": {"rows": 0, "s0": 0, "s1": 0, "s2": 0, "s0_scenarios": [], "s2_scenarios": []},
        "t0_negative_control": False,
        "process_seed": int(PROCESS_SEED),
        "process_seed_role": PROCESS_SEED_ROLE,
        **registered_flags(),
    }


def prepare() -> dict:
    if (OUTPUT / "candidate_selection_manifest.json").exists():
        raise RuntimeError("selection manifest already exists; do not rebuild or reselect")
    head = require_committed_code()
    audit = audit_saved_prefix()
    problems = contract_problems() + history_problems() + frozen_source_problems() + list(audit["problems"])
    budget = StepBudget()
    prefix_rows = list(audit["rows"])
    if problems or file_sha256(FORMAL_ACTOR) != FORMAL_ACTOR_SHA256:
        if file_sha256(FORMAL_ACTOR) != FORMAL_ACTOR_SHA256:
            problems.append("formal low-level checkpoint hash is not the registered SHA-256")
        summary = _summary(head, problems, budget, None, False)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        write_csv(OUTPUT / "prefix_coverage_audit.csv", PREFIX_FIELDS, prefix_rows)
        write_json(OUTPUT / "summary.json", summary)
        write_json(OUTPUT / "prepare_status.json", {"replay_authorized": False, "problems": problems, "budget": budget.as_dict()})
        write_report(summary)
        print(f"scientific_gate={GATE_BLOCKED} problems={len(problems)} steps={budget.total}", flush=True)
        return summary
    random.seed(PROCESS_SEED)
    np.random.seed(PROCESS_SEED)
    torch.manual_seed(PROCESS_SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    actor = load_actor()
    baselines = load_e17_baselines()
    saved_snapshots = load_e17_snapshots()
    records = []
    coverage_rows = []
    reconciliation = []
    saved_payloads = []
    try:
        for scenario in SCENARIOS:
            print(f"reference scenario {int(scenario)}", flush=True)
            session = FrozenAssignmentSession(actor, int(scenario), budget)
            rows, snapshots = _run_reference(session)
            episode = _summarize_episode(int(scenario), rows)
            episode_problems = compare_episode(episode, baselines[int(scenario)])
            reconciliation.append({"scenario_id": int(scenario), "problems": episode_problems, "rebuilt": episode})
            problems.extend(episode_problems)
            if episode_problems:
                break
            for snap in snapshots:
                key = (int(scenario), int(snap["t"]))
                coverage = pool_coverage_row(
                    int(scenario),
                    int(snap["t"]),
                    snap["pool"],
                    snap["tops"],
                    int(snap["illegal_count"]),
                    snap["ledger"],
                    int(snap["baseline_index"]),
                )
                prefix = coverage.pop("prefix_selection")
                prefix["queue_bits"] = float(np.sum(snap["env"]["q_bits"]))
                prefix["known_tagged_bits"] = float(snap["known"]["known_tagged_bits"])
                prefix["known_ordinary_bits"] = float(snap["known"]["known_ordinary_bits"])
                prefix["fifo_blocked_bits"] = float(snap["known"]["fifo_blocked_bits"])
                prefix["known_bucket_count"] = len(snap["known"]["known_ids"])
                prefix["heads"] = snap["known"]["heads"]
                snap_problems = compare_snapshot(prefix, saved_snapshots[key])
                problems.extend(snap_problems)
                record = public_record(snap["pool"], int(snap["baseline_index"]), int(scenario), int(snap["t"]))
                if record["dedup_sha256"] != dedup_sha256(prefix["dedup_first_index"]):
                    problems.append(f"scenario {scenario} t={int(snap['t'])} dedup map drifted")
                start = int(snap["step_index"])
                payload = {
                    "scenario_id": int(scenario),
                    "decision_time": int(snap["t"]),
                    "capture": {
                        name: snap[name]
                        for name in ("env", "env_rng", "ledger", "option", "pending", "last_movement", "assignment_rng", "process", "t", "known")
                    },
                    "reference_window": rows[start : start + int(HOLD_STEPS)],
                    "baseline_index": int(snap["baseline_index"]),
                    "baseline_assignment": np.asarray(snap["baseline_assignment"], dtype=np.int64),
                    "dedup_first_index": list(prefix["dedup_first_index"]),
                    "restore_notes": {
                        "format": "json codec v1",
                        "arrays": "native-endian base64 ndarray with dtype and shape",
                        "tuples_tagged": True,
                        "includes_env_rng": True,
                        "includes_process_rng": True,
                        "includes_ledger": True,
                        "includes_option": True,
                        "e17_public_manifest_is_restorable": False,
                    },
                }
                path = snapshot_file(int(scenario), int(snap["t"]))
                dump_capture(path, payload)
                saved_payloads.append(path)
                coverage_rows.append(coverage)
                records.append(record)
                for slot, item in enumerate(prefix["selected"]):
                    assignment = [int(value) for value in item]
                    prefix_rows.append(
                        {
                            "source": "rebuilt_prefix",
                            "scenario_id": int(scenario),
                            "decision_time": int(snap["t"]),
                            "candidate_slot": int(slot),
                            "pool_index": int(prefix["first_index"][item]),
                            "is_baseline": bool(item == prefix["baseline_key"]),
                            "assignment": assignment,
                            "active_count": int(sum(value >= 0 for value in assignment)),
                            "active_mask": [int(bit) for bit in active_mask(assignment)],
                            "matches_saved": not snap_problems,
                        }
                    )
                if problems:
                    break
            if problems:
                break
    except (RuntimeError, TypeError, ValueError) as exc:
        problems.append(str(exc))
    if not problems and len(records) == int(EXPECTED_SNAPSHOTS):
        for path in saved_payloads:
            loaded = load_capture(path)
            probe = FrozenAssignmentSession(actor, int(loaded["scenario_id"]), budget)
            probe.restore(loaded["capture"])
            item_problems = _capture_problems(loaded["capture"], probe.capture())
            problems.extend(f"scenario {loaded['scenario_id']} t={loaded['decision_time']} {item}" for item in item_problems)
            if item_problems:
                break
    authorized = not problems and len(records) == int(EXPECTED_SNAPSHOTS) and budget.candidate == 0 and budget.repeat == 0
    digest = selection_sha256(records) if records else None
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "prefix_coverage_audit.csv", PREFIX_FIELDS, prefix_rows)
    write_csv(OUTPUT / "pool_coverage.csv", POOL_FIELDS, coverage_rows)
    write_json(OUTPUT / "baseline_reconciliation.json", {"ok": bool(authorized), "episodes": reconciliation, "problems": problems})
    write_json(
        OUTPUT / "candidate_selection_manifest.json",
        {
            "schema_version": "assignment_coverage_diag_v1",
            "rule": "activity_matched_stratified_sample",
            "alternative_cap": 4,
            "candidate_limit": 5,
            "candidate_seed_formula": "9300000 + 100*scenario_id + decision_time",
            "replay_authorized": bool(authorized),
            "reference_steps": int(budget.reference),
            "records": records,
            "selection_sha256": digest,
        },
    )
    write_json(
        OUTPUT / "snapshot_index.json",
        {
            "e17_public_manifest_is_restorable": False,
            "full_capture_saved": True,
            "files": [
                {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": file_sha256(path)}
                for path in saved_payloads
            ],
        },
    )
    status = {
        "replay_authorized": bool(authorized),
        "problems": problems,
        "budget": budget.as_dict(),
        "selection_sha256": digest,
        "implementation_commit": head,
        "records": len(records),
    }
    write_json(OUTPUT / "prepare_status.json", status)
    if not authorized:
        summary = _summary(head, problems, budget, digest, False)
        write_json(OUTPUT / "summary.json", summary)
        write_report(summary)
        print(f"scientific_gate={GATE_BLOCKED} problems={len(problems)} steps={budget.total}", flush=True)
        return summary
    print(f"prepare_ok selection_sha256={digest} reference_steps={budget.reference}", flush=True)
    return status
