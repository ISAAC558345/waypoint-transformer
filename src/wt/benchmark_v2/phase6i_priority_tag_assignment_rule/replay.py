"""Replay the frozen N, M, and T list. Selection is not recomputed into a new choice."""
from __future__ import annotations

import json
import subprocess

import numpy as np

from wt.benchmark_v2.phase6e.generators import feature_matrices
from wt.benchmark_v2.phase6i_priority_tag_assignment_coverage.artifacts import write_csv, write_json
from wt.benchmark_v2.phase6i_priority_tag_assignment_coverage.codec import load_capture
from wt.benchmark_v2.phase6i_priority_tag_assignment_coverage.detail import bucket_service_rows, detail_matches_aggregate
from wt.benchmark_v2.phase6i_priority_tag_assignment_feasibility.attribution import copy_buckets
from wt.benchmark_v2.phase6i_priority_tag_assignment_feasibility.candidates import assignment_key
from wt.benchmark_v2.phase6i_priority_tag_assignment_feasibility.session import (
    FrozenAssignmentSession,
    _exogenous_match,
    _hold_metrics,
    _window_match,
    load_actor,
)
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import FORMAL_ACTOR
from wt.benchmark_v2.phase6i_priority_tag_reward.smoke import file_sha256

from .criteria import contrast_levels, mechanism_evidence
from .identity import contract_problems, frozen_source_problems, history_problems
from .manifest import freeze_record, selection_sha256
from .prepare import require_committed_code
from wt.benchmark_v2.phase6i_priority_tag_assignment_feasibility.protocol import BIT_TOL, BUCKET_TOL

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
    GATE_COMPLETE,
    HOLD_STEPS,
    HORIZON,
    INSTRUCTION_COMMIT,
    OUTPUT,
    PROCESS_SEED,
    PROCESS_SEED_ROLE,
    ROOT,
    SCENARIOS,
    SOURCE_COMMIT,
    TAG_WEIGHT,
    StepBudget,
    registered_flags,
    snapshot_file,
)
from .report import write_report
from .rule import choice_sha256, select_registered
from .stats import bootstrap_mean

STRATEGY_FIELDS = (
    "scenario_id", "role", "pool_index", "assignment", "reused_from", "executed",
    "payload_bits", "known_tagged_served_bits", "known_ordinary_served_bits",
    "future_tagged_served_bits", "future_ordinary_served_bits", "age_weighted_service",
    "dropped_tagged_bits", "dropped_ordinary_bits", "final_queue_bits", "final_age_mean",
    "final_age_max", "mean_sinr", "mean_interference", "safety_count", "constraint_count",
    "min_distance_m", "path_length_m", "attribution_gap_bits", "proxy_P", "proxy_K",
    "proxy_payload_gap_bits", "proxy_known_tag_gap_bits",
)
PAIRED_FIELDS = (
    "scenario_id", "tagged_differs_from_masked", "reused_m", "reused_t", "interpretable",
    "known_tagged_tm", "known_ordinary_tm", "future_tagged_tm", "future_ordinary_tm",
    "payload_tm", "dropped_tagged_tm", "dropped_ordinary_tm", "final_queue_tm",
    "age_tm", "safety_tm", "constraint_tm", "sinr_tm", "interference_tm", "path_tm",
    "s0_tm", "s1_tm", "s2_tm",
    "known_tagged_tn", "payload_tn", "safety_tn", "constraint_tn", "sinr_tn", "interference_tn",
    "s0_tn", "s1_tn", "s2_tn",
    "known_tagged_mn", "payload_mn", "safety_mn", "constraint_mn", "sinr_mn", "interference_mn",
    "s0_mn", "s1_mn", "s2_mn",
    "proxy_p_loss_tm", "bonus_share_t", "safety_t", "safety_m", "safety_n",
    "constraint_t", "constraint_m", "constraint_n",
)
SAFETY_FIELDS = (
    "scenario_id", "role", "step_t", "safety", "constraint", "min_distance_m", "source",
)
DETAIL_FIELDS = (
    "scenario_id", "role", "step_t", "buoy", "insert_index", "admission_step", "priority_tag",
    "known_at_snapshot", "bits_taken", "service_class",
)
BUDGET_FIELDS = (
    "scenario_id", "candidate_n", "candidate_m", "candidate_t", "repeat_n", "reused_m", "reused_t",
)
RECONCILE_FIELDS = (
    "scenario_id", "reference_match", "repeat_match", "exogenous_m", "exogenous_t",
    "action_gap", "position_gap_m", "payload_gap_bits", "queue_gap_bits", "bucket_gap_bits",
)
MANIFEST_RELATIVE = "results/phase6i_priority_tag_assignment_rule/selection_manifest.json"


def _git(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", "core.longpaths=true", *args], cwd=ROOT, capture_output=True, text=True, check=False)


def _require_frozen_inputs() -> tuple[str, dict, dict]:
    head = require_committed_code()
    if _git(["rev-parse", "--verify", "--quiet", f"HEAD:{MANIFEST_RELATIVE}"]).returncode != 0:
        raise RuntimeError("selection manifest is not committed")
    dirty = _git(
        [
            "status", "--porcelain", "--",
            MANIFEST_RELATIVE,
            "results/phase6i_priority_tag_assignment_rule/static_scores.json",
            "results/phase6i_priority_tag_assignment_rule/snapshot_index.json",
            "results/phase6i_priority_tag_assignment_rule/snapshots",
            "results/phase6i_priority_tag_assignment_rule/prepare_status.json",
            "src", "tests",
        ]
    ).stdout.strip()
    if dirty:
        raise RuntimeError("selection inputs are dirty; replay refused to reselect")
    manifest = json.loads((OUTPUT / "selection_manifest.json").read_text(encoding="utf-8"))
    records = manifest.get("records") or []
    if selection_sha256(records) != manifest.get("selection_sha256"):
        raise RuntimeError("selection manifest hash does not match its records")
    if not manifest.get("replay_authorized") or len(records) != len(SCENARIOS):
        raise RuntimeError("selection manifest is not an authorized 16-scenario list")
    if [int(item["scenario_id"]) for item in records] != list(SCENARIOS):
        raise RuntimeError("selection manifest scenario order changed")
    if float(manifest.get("weight")) != float(TAG_WEIGHT) or int(manifest.get("decision_time")) != int(DECISION_TIME):
        raise RuntimeError("selection manifest does not carry the registered rule")
    scores = json.loads((OUTPUT / "static_scores.json").read_text(encoding="utf-8"))
    if choice_sha256(scores.get("scenarios") or []) != scores.get("scores_sha256"):
        raise RuntimeError("static score list hash does not match its rows")
    if scores.get("scores_sha256") != manifest.get("scores_sha256"):
        raise RuntimeError("static score list does not match the selection manifest")
    index = json.loads((OUTPUT / "snapshot_index.json").read_text(encoding="utf-8"))
    for item in index.get("files") or []:
        path = ROOT / str(item["path"])
        if file_sha256(path) != item["sha256"]:
            raise RuntimeError(f"snapshot changed after it was indexed: {item['path']}")
    return head, manifest, scores


def _roll(session: FrozenAssignmentSession, loaded: dict, record: dict, assignment, kind: str) -> tuple[list[dict], list[dict]]:
    session.restore(loaded["capture"])
    pool, index, drawn = session.draw()
    if int(index) != int(loaded["baseline_index"]) or not np.array_equal(drawn, np.asarray(loaded["baseline_assignment"])):
        raise RuntimeError("restored draw does not reproduce the reference sample")
    features = feature_matrices(session.env)
    result = select_registered(
        pool,
        index,
        features["rate"],
        features["eta"],
        features["queue"],
        loaded["capture"]["ledger"]["buckets"],
        horizon=float(HOLD_STEPS),
        delta_t=float(loaded["delta_t"]),
        decision_time=int(loaded["decision_time"]),
    )
    recomputed = freeze_record(result, int(record["scenario_id"]), float(loaded["delta_t"]))
    if recomputed != record:
        raise RuntimeError("recomputed fixed rule does not match the frozen manifest")
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


def _served(metrics: dict) -> float:
    return float(
        metrics["known_tagged_served_bits"]
        + metrics["known_ordinary_served_bits"]
        + metrics["future_tagged_served_bits"]
        + metrics["future_ordinary_served_bits"]
    )


def _strategy_row(scenario: int, role: str, choice: dict, metrics: dict, reused_from: str) -> dict:
    proxy_p = float(choice["P"])
    proxy_k = float(choice["K"])
    return {
        "scenario_id": int(scenario),
        "role": role,
        "pool_index": int(choice["pool_index"]),
        "assignment": list(choice["assignment"]),
        "reused_from": reused_from,
        "executed": reused_from == "",
        "proxy_P": proxy_p,
        "proxy_K": proxy_k,
        "proxy_payload_gap_bits": _served(metrics) - proxy_p,
        "proxy_known_tag_gap_bits": float(metrics["known_tagged_served_bits"]) - proxy_k,
        **metrics,
    }


def _paired(scenario: int, record: dict, natural: dict, masked: dict, tagged: dict, reused_m: bool, reused_t: bool) -> dict:
    tm = contrast_levels(tagged, masked)
    tn = contrast_levels(tagged, natural)
    mn = contrast_levels(masked, natural)

    def delta(left: dict, right: dict, name: str) -> float:
        return float(left[name]) - float(right[name])

    proxy_t = float(record["tagged"]["P"])
    proxy_k = float(record["tagged"]["K"])
    return {
        "scenario_id": int(scenario),
        "tagged_differs_from_masked": bool(record["tagged_differs_from_masked"]),
        "reused_m": bool(reused_m),
        "reused_t": bool(reused_t),
        "interpretable": True,
        "known_tagged_tm": delta(tagged, masked, "known_tagged_served_bits"),
        "known_ordinary_tm": delta(tagged, masked, "known_ordinary_served_bits"),
        "future_tagged_tm": delta(tagged, masked, "future_tagged_served_bits"),
        "future_ordinary_tm": delta(tagged, masked, "future_ordinary_served_bits"),
        "payload_tm": delta(tagged, masked, "payload_bits"),
        "dropped_tagged_tm": delta(tagged, masked, "dropped_tagged_bits"),
        "dropped_ordinary_tm": delta(tagged, masked, "dropped_ordinary_bits"),
        "final_queue_tm": delta(tagged, masked, "final_queue_bits"),
        "age_tm": delta(tagged, masked, "age_weighted_service"),
        "safety_tm": delta(tagged, masked, "safety_count"),
        "constraint_tm": delta(tagged, masked, "constraint_count"),
        "sinr_tm": delta(tagged, masked, "mean_sinr"),
        "interference_tm": delta(tagged, masked, "mean_interference"),
        "path_tm": delta(tagged, masked, "path_length_m"),
        "s0_tm": bool(tm["s0"]),
        "s1_tm": bool(tm["s1"]),
        "s2_tm": bool(tm["s2"]),
        "known_tagged_tn": delta(tagged, natural, "known_tagged_served_bits"),
        "payload_tn": delta(tagged, natural, "payload_bits"),
        "safety_tn": delta(tagged, natural, "safety_count"),
        "constraint_tn": delta(tagged, natural, "constraint_count"),
        "sinr_tn": delta(tagged, natural, "mean_sinr"),
        "interference_tn": delta(tagged, natural, "mean_interference"),
        "s0_tn": bool(tn["s0"]),
        "s1_tn": bool(tn["s1"]),
        "s2_tn": bool(tn["s2"]),
        "known_tagged_mn": delta(masked, natural, "known_tagged_served_bits"),
        "payload_mn": delta(masked, natural, "payload_bits"),
        "safety_mn": delta(masked, natural, "safety_count"),
        "constraint_mn": delta(masked, natural, "constraint_count"),
        "sinr_mn": delta(masked, natural, "mean_sinr"),
        "interference_mn": delta(masked, natural, "mean_interference"),
        "s0_mn": bool(mn["s0"]),
        "s1_mn": bool(mn["s1"]),
        "s2_mn": bool(mn["s2"]),
        "proxy_p_loss_tm": proxy_t - float(record["masked"]["P"]),
        "bonus_share_t": (float(TAG_WEIGHT) * proxy_k / proxy_t) if proxy_t > 0.0 else 0.0,
        "safety_t": int(tagged["safety_count"]),
        "safety_m": int(masked["safety_count"]),
        "safety_n": int(natural["safety_count"]),
        "constraint_t": int(tagged["constraint_count"]),
        "constraint_m": int(masked["constraint_count"]),
        "constraint_n": int(natural["constraint_count"]),
    }


def _contrast_summary(rows: list[dict], prefix: str) -> dict:
    scenarios_s0 = [int(row["scenario_id"]) for row in rows if row.get(f"s0_{prefix}")]
    scenarios_s2 = [int(row["scenario_id"]) for row in rows if row.get(f"s2_{prefix}")]
    return {
        "s0": len(scenarios_s0),
        "s1": int(sum(1 for row in rows if row.get(f"s1_{prefix}"))),
        "s2": len(scenarios_s2),
        "s0_scenarios": scenarios_s0,
        "s2_scenarios": scenarios_s2,
    }


def _bootstrap(rows: list[dict]) -> dict:
    keys = (
        "known_tagged_tm", "payload_tm", "known_tagged_tn", "payload_tn", "known_tagged_mn", "payload_mn",
    )
    return {key: bootstrap_mean([row[key] for row in rows]) for key in keys}


def _safety(scenario: int, role: str, rows: list[dict]) -> list[dict]:
    return [
        {
            "scenario_id": int(scenario),
            "role": role,
            "step_t": int(row["t"]),
            "safety": int(row["safety"]),
            "constraint": int(row["constraint"]),
            "min_distance_m": float(row["min_distance_m"]),
            "source": "replay",
        }
        for row in rows
        if int(row["safety"]) or int(row["constraint"])
    ]


def _finish(head: str, manifest: dict, problems: list[str], budget: StepBudget, paired: list[dict], reuse: dict) -> dict:
    publish = not problems and len(paired) == len(SCENARIOS) and all(row.get("interpretable") for row in paired)
    judged = mechanism_evidence(paired) if publish else {"label": None, "published": False, "failed_checks": []}
    intervals = _bootstrap(paired) if publish else {}
    primary = intervals.get("known_tagged_tm") or {}
    payload = intervals.get("payload_tm") or {}
    uncertain = bool(publish and (primary.get("crosses_zero") or payload.get("crosses_zero") or float(primary.get("ci_high", 0.0)) < 0.0 or float(payload.get("ci_high", 0.0)) < 0.0))
    differs = int(sum(1 for row in paired if row.get("tagged_differs_from_masked")))
    return {
        "schema_version": "fixed_tag_rule_diag_v1",
        "scientific_gate": GATE_COMPLETE if publish else GATE_BLOCKED,
        "mechanism_evidence": judged["label"] if publish else None,
        "mechanism_published": bool(publish and judged["published"]),
        "failed_checks": list(judged["failed_checks"]) if publish else [],
        "e17_scientific_gate": E17_GATE,
        "e17_opportunity_evidence": E17_OPPORTUNITY,
        "e18_scientific_gate": E18_GATE,
        "e18_opportunity_evidence": E18_OPPORTUNITY,
        "e15_scientific_gate": E15_GATE,
        "e16_scientific_gate": E16_GATE,
        "problems": problems,
        "budget": budget.as_dict(),
        "implementation_commit": manifest.get("implementation_commit"),
        "selection_commit": head,
        "instruction_commit": INSTRUCTION_COMMIT,
        "source_commit": SOURCE_COMMIT,
        "formal_actor_sha256": file_sha256(FORMAL_ACTOR),
        "selection_sha256": manifest.get("selection_sha256"),
        "tagged_differs_count": differs,
        "contrasts": {
            "tm": _contrast_summary(paired, "tm"),
            "tn": _contrast_summary(paired, "tn"),
            "mn": _contrast_summary(paired, "mn"),
        },
        "bootstrap": intervals,
        "overall_gain_uncertain": uncertain,
        "reuse": reuse,
        "decision_time": int(DECISION_TIME),
        "scenarios": list(SCENARIOS),
        "weight": float(TAG_WEIGHT),
        "process_seed": int(PROCESS_SEED),
        "process_seed_role": PROCESS_SEED_ROLE,
        "not_an_e18_hindsight_policy": True,
        **registered_flags(),
    }


def replay() -> dict:
    head, manifest, scores = _require_frozen_inputs()
    problems = contract_problems() + history_problems() + frozen_source_problems()
    budget = StepBudget(reference=int(manifest["reference_steps"]))
    if problems or file_sha256(FORMAL_ACTOR) != FORMAL_ACTOR_SHA256:
        if file_sha256(FORMAL_ACTOR) != FORMAL_ACTOR_SHA256:
            problems.append("formal low-level checkpoint hash is not the registered SHA-256")
        summary = _finish(head, manifest, problems, budget, [], {"scenarios_reused_m": 0, "scenarios_reused_t": 0})
        write_json(OUTPUT / "summary.json", summary)
        write_report(summary)
        print(f"scientific_gate={GATE_BLOCKED} problems={len(problems)}", flush=True)
        return summary
    by_score = {int(item["scenario_id"]): item for item in scores["scenarios"]}
    actor = load_actor()
    strategies = []
    paired = []
    details = []
    safety = []
    budget_rows = []
    reconcile = []
    try:
        for record in manifest["records"]:
            scenario = int(record["scenario_id"])
            print(f"replay scenario {scenario}", flush=True)
            if by_score[scenario]["scores_sha256"] != record["scores_sha256"]:
                raise RuntimeError(f"scenario {scenario} score block does not match the manifest")
            loaded = load_capture(snapshot_file(scenario))
            if int(loaded["scenario_id"]) != scenario or int(loaded["decision_time"]) != int(DECISION_TIME):
                raise RuntimeError(f"scenario {scenario} snapshot is not the registered t=15 capture")
            if float(loaded["t0_known_tagged_bits"]) > float(BIT_TOL):
                raise RuntimeError(f"scenario {scenario} known tagged bits at reset exceed 1 bit")
            session = FrozenAssignmentSession(actor, scenario, budget)
            natural_rows, natural_details = _roll(session, loaded, record, record["natural"]["assignment"], "candidate")
            reference = _window_match(loaded["reference_window"], natural_rows)
            repeat_rows, _repeat_details = _roll(session, loaded, record, record["natural"]["assignment"], "repeat")
            repeat = _window_match(natural_rows, repeat_rows)
            reconcile.append(
                {
                    "scenario_id": scenario,
                    "reference_match": bool(reference["ok"]),
                    "repeat_match": bool(repeat["ok"]),
                    "exogenous_m": False,
                    "exogenous_t": False,
                    "action_gap": float(reference["action_gap"]),
                    "position_gap_m": float(reference["position_gap_m"]),
                    "payload_gap_bits": float(reference["payload_gap_bits"]),
                    "queue_gap_bits": float(reference["queue_gap_bits"]),
                    "bucket_gap_bits": float(reference["bucket_gap_bits"]),
                }
            )
            if not reference["ok"] or not repeat["ok"]:
                raise RuntimeError(f"scenario {scenario} natural replay does not match the reference window")
            if float(_hold_metrics(natural_rows, float(loaded["capture"]["env"]["payload_bits"]))["attribution_gap_bits"]) > float(BUCKET_TOL):
                raise RuntimeError(f"scenario {scenario} attribution gap exceeds 1e-6 bit")
            start_payload = float(loaded["capture"]["env"]["payload_bits"])
            natural_metrics = _hold_metrics(natural_rows, start_payload)
            cache = {assignment_key(record["natural"]["assignment"]): ("N", natural_metrics, natural_rows, natural_details)}
            safety.extend(_safety(scenario, "N", natural_rows))
            for item in natural_details:
                details.append({"scenario_id": scenario, "role": "N", **item})
            branch_steps = {"M": 0, "T": 0}
            reused = {}
            branch_metrics = {"N": natural_metrics}
            for role, choice in (("M", record["masked"]), ("T", record["tagged"])):
                key = assignment_key(choice["assignment"])
                if key in cache:
                    source_role, metrics, _rows, _detail = cache[key]
                    branch_metrics[role] = metrics
                    reused[role] = source_role
                    reconcile[-1][f"exogenous_{role.lower()}"] = True
                    continue
                rows, role_details = _roll(session, loaded, record, choice["assignment"], "candidate")
                exogenous = _exogenous_match(loaded["reference_window"], rows)
                reconcile[-1][f"exogenous_{role.lower()}"] = bool(exogenous["ok"])
                if not exogenous["ok"]:
                    raise RuntimeError(f"scenario {scenario} {role} exogenous stream does not match the reference")
                metrics = _hold_metrics(rows, start_payload)
                if float(metrics["attribution_gap_bits"]) > float(BUCKET_TOL):
                    raise RuntimeError(f"scenario {scenario} {role} attribution gap exceeds 1e-6 bit")
                cache[key] = (role, metrics, rows, role_details)
                branch_metrics[role] = metrics
                reused[role] = ""
                branch_steps[role] = int(HOLD_STEPS)
                safety.extend(_safety(scenario, role, rows))
                for item in role_details:
                    details.append({"scenario_id": scenario, "role": role, **item})
            for role, choice in (("N", record["natural"]), ("M", record["masked"]), ("T", record["tagged"])):
                strategies.append(
                    _strategy_row(scenario, role, choice, branch_metrics[role], "" if role == "N" else reused[role])
                )
            paired.append(
                _paired(
                    scenario,
                    record,
                    branch_metrics["N"],
                    branch_metrics["M"],
                    branch_metrics["T"],
                    bool(reused["M"]),
                    bool(reused["T"]),
                )
            )
            budget_rows.append(
                {
                    "scenario_id": scenario,
                    "candidate_n": int(HOLD_STEPS),
                    "candidate_m": int(branch_steps["M"]),
                    "candidate_t": int(branch_steps["T"]),
                    "repeat_n": int(HOLD_STEPS),
                    "reused_m": bool(reused["M"]),
                    "reused_t": bool(reused["T"]),
                }
            )
    except (RuntimeError, TypeError, ValueError, KeyError) as exc:
        problems.append(str(exc))
    reuse = {
        "scenarios_reused_m": int(sum(1 for row in budget_rows if row["reused_m"])),
        "scenarios_reused_t": int(sum(1 for row in budget_rows if row["reused_t"])),
        "logical_rows": int(len(strategies)),
    }
    summary = _finish(head, manifest, problems, budget, paired, reuse)
    write_csv(OUTPUT / "strategy_rows.csv", STRATEGY_FIELDS, strategies)
    write_csv(OUTPUT / "paired_differences.csv", PAIRED_FIELDS, paired)
    write_csv(OUTPUT / "bucket_attribution.csv", DETAIL_FIELDS, details)
    write_csv(OUTPUT / "safety_events.csv", SAFETY_FIELDS, safety)
    write_csv(OUTPUT / "budget_items.csv", BUDGET_FIELDS, budget_rows)
    write_csv(OUTPUT / "baseline_reconciliation.csv", RECONCILE_FIELDS, reconcile)
    write_json(
        OUTPUT / "baseline_reconciliation.json",
        {"stage": "replay", "ok": summary["scientific_gate"] == GATE_COMPLETE, "rows": reconcile, "problems": problems},
    )
    write_json(OUTPUT / "bootstrap.json", summary["bootstrap"])
    write_json(OUTPUT / "summary.json", summary)
    write_report(summary)
    print(
        f"scientific_gate={summary['scientific_gate']} mechanism={summary['mechanism_evidence']} steps={budget.total}",
        flush=True,
    )
    return summary
