"""Compare two explicit-context traces against each other and the saved row."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.phase6i_priority_tag.protocol import TOL_BITS
from wt.benchmark_v2.phase6i_priority_tag_replay_repro.compare import bits_within_tolerance, path_matches, saved_gaps

from .protocol import ACCOUNTING_TOLERANCE_BITS, GATE_BLOCKED, GATE_INCONCLUSIVE, GATE_PASS, PATH_TOLERANCE_M


def _max_abs(left, right) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))))


def step_row(left: dict, right: dict, index: int) -> dict:
    payload_gap = abs(float(left["payload"]) - float(right["payload"]))
    q_gap = _max_abs(left["q_bits"], right["q_bits"])
    served_gap = abs(float(left["served"]) - float(right["served"]))
    dropped_gap = abs(float(left["dropped"]) - float(right["dropped"]))
    snap_gap = abs(float(left["snap"]) - float(right["snap"]))
    path_gap = abs(float(left["path_increment"]) - float(right["path_increment"]))
    position_gap = _max_abs(left["position"], right["position"])
    within = bool(
        left["obs_sha256"] == right["obs_sha256"]
        and payload_gap <= float(ACCOUNTING_TOLERANCE_BITS)
        and q_gap <= float(ACCOUNTING_TOLERANCE_BITS)
        and served_gap <= float(ACCOUNTING_TOLERANCE_BITS)
        and dropped_gap <= float(ACCOUNTING_TOLERANCE_BITS)
        and snap_gap <= float(ACCOUNTING_TOLERANCE_BITS)
        and path_gap <= float(PATH_TOLERANCE_M)
        and position_gap <= float(PATH_TOLERANCE_M)
    )
    return {
        "step_index": int(index),
        "obs_equal": left["obs_sha256"] == right["obs_sha256"],
        "association_equal": left["association"] == right["association"],
        "mu_abs": _max_abs(left["mu"], right["mu"]),
        "desired_abs": _max_abs(left["desired"], right["desired"]),
        "projected_abs": _max_abs(left["projected"], right["projected"]),
        "executed_abs": _max_abs(left["executed"], right["executed"]),
        "position_abs": position_gap,
        "q_bits_abs": q_gap,
        "served_abs": served_gap,
        "dropped_abs": dropped_gap,
        "snap_abs": snap_gap,
        "payload_abs": payload_gap,
        "path_increment_abs": path_gap,
        "within_tolerance": within,
    }


def traces_consistent(left: list[dict], right: list[dict]) -> dict:
    if len(left) != len(right) or len(left) == 0:
        return {"consistent": False, "reason": "trace length differs", "max_payload_abs": None, "path_abs": None}
    rows = [step_row(lhs, rhs, index) for index, (lhs, rhs) in enumerate(zip(left, right))]
    path_abs = abs(sum(float(step["path_increment"]) for step in left) - sum(float(step["path_increment"]) for step in right))
    payload_abs = abs(float(left[-1]["payload"]) - float(right[-1]["payload"]))
    consistent = bool(all(row["within_tolerance"] for row in rows) and path_abs <= float(PATH_TOLERANCE_M) and payload_abs <= float(ACCOUNTING_TOLERANCE_BITS))
    return {
        "consistent": consistent,
        "reason": "" if consistent else "a step exceeds the existing tolerance",
        "rows": rows,
        "max_payload_abs": float(max(row["payload_abs"] for row in rows)),
        "path_abs": float(path_abs),
    }


def ledger_ok(trace: dict) -> bool:
    episode = trace["episode"]
    steps = trace["steps"]
    served = float(sum(float(step["served"]) for step in steps))
    split = abs(float(episode["served_ordinary_bits"]) + float(episode["served_tagged_bits"]) - served)
    payload_gap = abs(float(episode["payload_bits"]) - served)
    max_ledger = max(float(step["ledger_gap"]) for step in steps)
    limit = min(float(TOL_BITS), float(ACCOUNTING_TOLERANCE_BITS))
    return bool(split <= limit and payload_gap <= limit and max_ledger <= limit)


def matches_saved(trace: dict, saved: dict) -> dict:
    gaps = saved_gaps(trace["episode"], saved)
    return {"gaps": gaps, "matches": bool(bits_within_tolerance(gaps) and path_matches(gaps))}


def choose_gate(
    *,
    hashes_ok: bool,
    split_ok: bool,
    manifest_ok: bool,
    processes_consistent: bool,
    both_match_saved: bool,
    ledger_ok_both: bool,
) -> str:
    if not hashes_ok or not split_ok or not manifest_ok or not processes_consistent:
        return GATE_BLOCKED
    if both_match_saved and ledger_ok_both:
        return GATE_PASS
    if processes_consistent and not both_match_saved:
        return GATE_INCONCLUSIVE
    return GATE_BLOCKED
