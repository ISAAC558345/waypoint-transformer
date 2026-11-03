"""Collect sixteen natural references and freeze N, M, and T before any new replay."""
from __future__ import annotations

import platform
import random
import subprocess
import sys

import numpy as np
import torch

from wt.benchmark_v2.phase6e.generators import build_pools, feature_matrices
from wt.benchmark_v2.phase6i_priority_tag_assignment_coverage.artifacts import write_csv, write_json
from wt.benchmark_v2.phase6i_priority_tag_assignment_coverage.codec import dump_capture, load_capture
from wt.benchmark_v2.phase6i_priority_tag_assignment_feasibility.candidates import pool_sha256
from wt.benchmark_v2.phase6i_priority_tag_assignment_feasibility.protocol import (
    BIT_TOL,
    DECISION_TIMES,
    EXCLUDED_TAIL_TIME,
)
from wt.benchmark_v2.phase6i_priority_tag_assignment_feasibility.session import (
    FrozenAssignmentSession,
    _inventory,
    _ledger_gap,
    _same,
    load_actor,
)
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import FORMAL_ACTOR
from wt.benchmark_v2.phase6i_priority_tag_reward.smoke import file_sha256
from wt.benchmark_v2.phase6i_priority_tag_reward.train import source_commit

from .identity import contract_problems, frozen_source_problems, history_problems
from .manifest import freeze_record, selection_sha256
from .protocol import (
    DECISION_TIME,
    E15_GATE,
    E16_GATE,
    E17_GATE,
    E17_OPPORTUNITY,
    E18_GATE,
    E18_OPPORTUNITY,
    FORMAL_ACTOR_SHA256,
    GATE_BLOCKED,
    HOLD_STEPS,
    HORIZON,
    INSTRUCTION_COMMIT,
    OUTPUT,
    POOL_RULE,
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
from .rule import choice_sha256, select_registered

SCORE_FIELDS = (
    "scenario_id", "pool_index", "assignment", "active_mask", "P", "K", "score_m", "score_t",
)


def _git(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", "core.longpaths=true", *args], cwd=ROOT, capture_output=True, text=True, check=False)


def require_committed_code() -> str:
    head = source_commit()
    if head == INSTRUCTION_COMMIT:
        raise RuntimeError("fixed tag-rule diagnosis refused to run at the instruction commit")
    if _git(["merge-base", "--is-ancestor", INSTRUCTION_COMMIT, "HEAD"]).returncode != 0:
        raise RuntimeError("HEAD does not descend from the E19 instruction")
    if _git(["status", "--porcelain", "--", "src", "tests"]).stdout.strip():
        raise RuntimeError("src or tests are dirty; commit the implementation before the diagnosis")
    return head


def _reset_process() -> None:
    random.seed(int(PROCESS_SEED))
    np.random.seed(int(PROCESS_SEED))
    torch.manual_seed(int(PROCESS_SEED))


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


def _run_reference(session: FrozenAssignmentSession) -> dict:
    _reset_process()
    session.reset_episode()
    known = _inventory(session.ledger, int(session.env.t))
    if int(session.env.t) != 0 or float(known["known_tagged_bits"]) > float(BIT_TOL):
        raise RuntimeError(f"scenario {session.scenario_id} known tagged bits at reset exceed 1 bit")
    rows = []
    saved = None
    allowed = set(DECISION_TIMES) | {int(EXCLUDED_TAIL_TIME)}
    while int(session.env.t) < int(HORIZON):
        if session.pending <= 0:
            if int(session.env.t) not in allowed:
                raise RuntimeError(f"unexpected assignment boundary at t={int(session.env.t)}")
            snap = session.capture()
            step_index = len(rows)
            if int(snap["t"]) == 0 and float(snap["known"]["known_tagged_bits"]) > float(BIT_TOL):
                raise RuntimeError(f"scenario {session.scenario_id} known tagged bits at reset exceed 1 bit")
            if int(snap["t"]) == int(DECISION_TIME):
                features = feature_matrices(session.env)
                queue_gap = float(np.max(np.abs(features["queue"] - np.asarray(snap["env"]["q_bits"], dtype=np.float64))))
                if queue_gap > 0.0:
                    raise RuntimeError("feature queue is not the snapshot queue")
                pools, _packed, illegal = build_pools(session.env, POOL_RULE)
                if int(illegal) != 0:
                    raise RuntimeError("candidate pool contains an illegal assignment")
                pool, index, chosen = session.draw()
                if pool_sha256(pool) != pool_sha256(pools[POOL_RULE[0]]):
                    raise RuntimeError("pool changed between the static read and the draw")
                delta_t = float(session.env.config.delta_t)
                result = select_registered(
                    pool,
                    index,
                    features["rate"],
                    features["eta"],
                    features["queue"],
                    snap["ledger"]["buckets"],
                    horizon=float(HOLD_STEPS),
                    delta_t=delta_t,
                    decision_time=int(snap["t"]),
                )
                if result["natural"]["assignment"] != [int(item) for item in chosen]:
                    raise RuntimeError("natural choice is not the sampled assignment")
                saved = {
                    "snap": snap,
                    "step_index": int(step_index),
                    "baseline_index": int(index),
                    "baseline_assignment": np.asarray(chosen, dtype=np.int64).copy(),
                    "result": result,
                    "delta_t": delta_t,
                    "t0_known_tagged_bits": float(known["known_tagged_bits"]),
                }
                session.install(chosen)
            else:
                _pool, _index, chosen = session.draw()
                session.install(chosen)
        session.budget.add("reference")
        rows.append(session.physical())
    if saved is None:
        raise RuntimeError("reference did not reach the registered decision time")
    if len(rows) != int(HORIZON):
        raise RuntimeError("reference trajectory length is not the registered horizon")
    start = int(saved["step_index"])
    window = rows[start : start + int(HOLD_STEPS)]
    if len(window) != int(HOLD_STEPS) or int(window[0]["t"]) != int(DECISION_TIME):
        raise RuntimeError("registered window is not the 15 steps that start at t=15")
    saved["window"] = window
    return saved


def _blocked(head: str, problems: list[str], budget: StepBudget) -> dict:
    summary = {
        "schema_version": "fixed_tag_rule_diag_v1",
        "scientific_gate": GATE_BLOCKED,
        "mechanism_evidence": None,
        "mechanism_published": False,
        "failed_checks": [],
        "e17_scientific_gate": E17_GATE,
        "e17_opportunity_evidence": E17_OPPORTUNITY,
        "e18_scientific_gate": E18_GATE,
        "e18_opportunity_evidence": E18_OPPORTUNITY,
        "e15_scientific_gate": E15_GATE,
        "e16_scientific_gate": E16_GATE,
        "problems": problems,
        "budget": budget.as_dict(),
        "implementation_commit": head,
        "instruction_commit": INSTRUCTION_COMMIT,
        "source_commit": SOURCE_COMMIT,
        "formal_actor_sha256": file_sha256(FORMAL_ACTOR),
        "selection_sha256": None,
        "performance_pass": False,
        "process_seed": int(PROCESS_SEED),
        "process_seed_role": PROCESS_SEED_ROLE,
        **registered_flags(),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "summary.json", summary)
    write_json(OUTPUT / "prepare_status.json", {"replay_authorized": False, "problems": problems, "budget": budget.as_dict()})
    write_report(summary)
    print(f"scientific_gate={GATE_BLOCKED} problems={len(problems)} steps={budget.total}", flush=True)
    return summary


def prepare() -> dict:
    if (OUTPUT / "selection_manifest.json").exists():
        raise RuntimeError("selection manifest already exists; do not rebuild or reselect")
    head = require_committed_code()
    problems = contract_problems() + history_problems() + frozen_source_problems()
    budget = StepBudget()
    if problems or file_sha256(FORMAL_ACTOR) != FORMAL_ACTOR_SHA256:
        if file_sha256(FORMAL_ACTOR) != FORMAL_ACTOR_SHA256:
            problems.append("formal low-level checkpoint hash is not the registered SHA-256")
        return _blocked(head, problems, budget)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    _reset_process()
    actor = load_actor()
    records = []
    score_blocks = []
    score_rows = []
    saved_paths = []
    t0_rows = []
    try:
        for scenario in SCENARIOS:
            print(f"reference scenario {int(scenario)}", flush=True)
            session = FrozenAssignmentSession(actor, int(scenario), budget)
            saved = _run_reference(session)
            result = saved["result"]
            record = freeze_record(result, int(scenario), float(saved["delta_t"]))
            payload = {
                "scenario_id": int(scenario),
                "decision_time": int(DECISION_TIME),
                "capture": {
                    name: saved["snap"][name]
                    for name in ("env", "env_rng", "ledger", "option", "pending", "last_movement", "assignment_rng", "process", "t", "known")
                },
                "reference_window": saved["window"],
                "baseline_index": int(saved["baseline_index"]),
                "baseline_assignment": saved["baseline_assignment"],
                "delta_t": float(saved["delta_t"]),
                "t0_known_tagged_bits": float(saved["t0_known_tagged_bits"]),
            }
            path = snapshot_file(int(scenario))
            dump_capture(path, payload)
            saved_paths.append(path)
            records.append(record)
            score_blocks.append(
                {"scenario_id": int(scenario), "scores_sha256": record["scores_sha256"], "eligible": result["eligible"]}
            )
            for item in result["eligible"]:
                score_rows.append({"scenario_id": int(scenario), **item})
            t0_rows.append(
                {
                    "scenario_id": int(scenario),
                    "t0_known_tagged_bits": float(saved["t0_known_tagged_bits"]),
                    "within_1_bit": True,
                }
            )
    except (RuntimeError, TypeError, ValueError) as exc:
        problems.append(str(exc))
    if not problems and len(records) == len(SCENARIOS):
        for path in saved_paths:
            loaded = load_capture(path)
            probe = FrozenAssignmentSession(actor, int(loaded["scenario_id"]), budget)
            probe.restore(loaded["capture"])
            item_problems = _capture_problems(loaded["capture"], probe.capture())
            problems.extend(f"scenario {loaded['scenario_id']} {item}" for item in item_problems)
            if item_problems:
                break
    authorized = (
        not problems
        and len(records) == len(SCENARIOS)
        and budget.reference == len(SCENARIOS) * int(HORIZON)
        and budget.candidate == 0
        and budget.repeat == 0
    )
    digest = selection_sha256(records) if records else None
    score_digest = choice_sha256(score_blocks) if score_blocks else None
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(
        OUTPUT / "runtime_manifest.json",
        {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "implementation_commit": head,
            "instruction_commit": INSTRUCTION_COMMIT,
            "source_commit": SOURCE_COMMIT,
            "formal_actor_sha256": file_sha256(FORMAL_ACTOR),
            "torch_threads": 1,
            "deterministic_algorithms": True,
            "actor_action": "deterministic mean, then the frozen residual scale times tanh of that mean",
        },
    )
    write_json(
        OUTPUT / "seed_manifest.json",
        {
            "env_reset_seed": "scenario_id",
            "assignment_rng": "7000000 + scenario_id",
            "process_seed": int(PROCESS_SEED),
            "process_seed_role": PROCESS_SEED_ROLE,
            "process_seed_reset": "before actor load and before every independent reference reset",
            "selector_rng": None,
            "bootstrap_seed": 9400019,
            "bootstrap_draws": 10000,
            "bootstrap_unit": "scenario",
        },
    )
    write_json(
        OUTPUT / "evidence_gaps.json",
        {
            "preregistered": [
                "The proxy assumes the current rate persists after the geometric ETA. It is not a transmission, a joint-interference prediction, or a future-service guarantee.",
                "Only the pre-decision state at t=15 is scored. That focus follows the location of every E18 S0 row and does not test t=0, t=30, t=45, or a full closed loop.",
                "The sixteen scenarios are inside the training pool. They are not a project-wide unseen holdout.",
                "Scenarios with no known tags, no same-mask alternative, or no rule change stay in the denominator.",
                "The 0.001 weight is this stage's fixed heuristic. It is not a trained reward and not a searched optimum.",
                "A completed label is mechanism evidence. It is not an adoption standard.",
            ]
        },
    )
    write_csv(OUTPUT / "static_scores.csv", SCORE_FIELDS, score_rows)
    write_json(
        OUTPUT / "static_scores.json",
        {"scores_sha256": score_digest, "scenarios": score_blocks},
    )
    write_json(
        OUTPUT / "baseline_reconciliation.json",
        {"stage": "prepare", "ok": bool(authorized), "t0_checks": t0_rows, "problems": problems},
    )
    if authorized:
        write_json(
            OUTPUT / "selection_manifest.json",
            {
                "schema_version": "fixed_tag_rule_diag_v1",
                "rule": "max P for the masked control, max P plus 0.001 K for the tagged rule",
                "weight": 0.001,
                "decision_time": int(DECISION_TIME),
                "hold_steps": int(HOLD_STEPS),
                "replay_authorized": True,
                "reference_steps": int(budget.reference),
                "records": records,
                "selection_sha256": digest,
                "scores_sha256": score_digest,
                "implementation_commit": head,
            },
        )
    write_json(
        OUTPUT / "snapshot_index.json",
        {
            "full_capture_saved": True,
            "decision_time": int(DECISION_TIME),
            "files": [
                {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": file_sha256(path)}
                for path in saved_paths
            ],
        },
    )
    status = {
        "replay_authorized": bool(authorized),
        "problems": problems,
        "budget": budget.as_dict(),
        "selection_sha256": digest,
        "scores_sha256": score_digest,
        "implementation_commit": head,
        "records": len(records),
    }
    write_json(OUTPUT / "prepare_status.json", status)
    if not authorized:
        return _blocked(head, problems, budget)
    print(f"prepare_ok selection_sha256={digest} reference_steps={budget.reference}", flush=True)
    return status
