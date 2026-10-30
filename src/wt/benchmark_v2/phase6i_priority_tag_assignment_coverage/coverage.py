"""Recount the E17 prefix and describe same-mask coverage from the rebuilt pool only."""
from __future__ import annotations

import csv
import json
from collections import Counter

import numpy as np

from wt.benchmark_v2.phase6i_priority_tag.protocol import DUST_BITS
from wt.benchmark_v2.phase6i_priority_tag_assignment_feasibility.candidates import select_registered

from .protocol import BIT_TOL, BUCKET_TOL, CANDIDATE_LIMIT, E17_OUTPUT
from .selection import active_mask, activity_census


def _load_rows(path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _assignment(text: str) -> list[int]:
    return [int(item) for item in json.loads(text)]


def audit_saved_prefix() -> dict:
    """Independent recount of the published E17 candidate file. It does not replay those rows."""
    rows = _load_rows(E17_OUTPUT / "candidate_rows.csv")
    baselines = [row for row in rows if row["is_baseline"] == "1"]
    alternatives = [row for row in rows if row["is_baseline"] == "0"]
    baseline_active = [sum(item >= 0 for item in _assignment(row["assignment"])) for row in baselines]
    alternative_active = [sum(item >= 0 for item in _assignment(row["assignment"])) for row in alternatives]
    alternative_indices = {int(row["pool_index"]) for row in alternatives}
    baseline_indices = [int(row["pool_index"]) for row in baselines]
    active_counter = Counter(alternative_active)
    problems = []
    if len(baselines) != 24 or len(alternatives) != 96:
        problems.append("E17 candidate file does not have 24 baselines and 96 prefix alternatives")
    if not baseline_active or min(baseline_active) != 2 or max(baseline_active) != 5:
        problems.append("E17 baseline activity is not the recorded range of 2 to 5 UAVs")
    if active_counter.get(0) != 24 or active_counter.get(1) != 72 or sum(active_counter.values()) != 96:
        problems.append("E17 prefix alternatives are not 24 idle and 72 single-UAV rows")
    if alternative_indices != {0, 1, 2, 3}:
        problems.append("E17 prefix alternatives did not come only from pool indices 0, 1, 2, and 3")
    if not baseline_indices or min(baseline_indices) != 260 or max(baseline_indices) != 5444:
        problems.append("E17 baseline pool indices are not the recorded range 260 to 5444")
    public = []
    for row in rows:
        assignment = _assignment(row["assignment"])
        public.append(
            {
                "source": "e17_saved",
                "scenario_id": int(row["scenario_id"]),
                "decision_time": int(row["decision_time"]),
                "candidate_slot": int(row["candidate_slot"]),
                "pool_index": int(row["pool_index"]),
                "is_baseline": row["is_baseline"] == "1",
                "assignment": assignment,
                "active_count": int(sum(item >= 0 for item in assignment)),
                "active_mask": [int(item) for item in active_mask(assignment)],
                "matches_saved": True,
            }
        )
    return {"problems": problems, "rows": public, "baseline_count": len(baselines), "alternative_count": len(alternatives)}


def load_e17_baselines() -> dict[int, dict]:
    found = {}
    for row in _load_rows(E17_OUTPUT / "baseline_rows.csv"):
        found[int(row["scenario_id"])] = row
    return found


def load_e17_snapshots() -> dict[tuple[int, int], dict]:
    payload = json.loads((E17_OUTPUT / "snapshot_manifest.json").read_text(encoding="utf-8"))
    return {(int(item["scenario_id"]), int(item["decision_time"])): item for item in payload}


def _gap(left, right) -> float:
    return abs(float(left) - float(right))


def compare_episode(rebuilt: dict, saved: dict) -> list[str]:
    problems = []
    scenario = int(rebuilt["scenario_id"])
    if int(saved["steps"]) != int(rebuilt["steps"]) or int(saved["final_t"]) != int(rebuilt["final_t"]):
        problems.append(f"scenario {scenario} reference length does not match E17")
    if json.loads(saved["decision_times"]) != list(rebuilt["decision_times"]):
        problems.append(f"scenario {scenario} decision times do not match E17")
    if int(saved["safety_count"]) != int(rebuilt["safety_count"]) or int(saved["constraint_count"]) != int(rebuilt["constraint_count"]):
        problems.append(f"scenario {scenario} safety or constraint totals do not match E17")
    for name in ("payload_bits", "served_tagged_bits", "served_ordinary_bits"):
        if _gap(saved[name], rebuilt[name]) > float(BIT_TOL):
            problems.append(f"scenario {scenario} {name} gap { _gap(saved[name], rebuilt[name]) } exceeds 1 bit")
    return problems


def compare_snapshot(rebuilt_prefix: dict, saved: dict) -> list[str]:
    problems = []
    label = f"scenario {int(saved['scenario_id'])} t={int(saved['decision_time'])}"
    if rebuilt_prefix["pool_sha256"] != saved["pool_sha256"]:
        problems.append(f"{label} pool hash does not match E17")
    if int(rebuilt_prefix["pool_size"]) != int(saved["pool_size"]) or int(rebuilt_prefix["unique_count"]) != int(saved["unique_count"]):
        problems.append(f"{label} pool size does not match E17")
    if int(rebuilt_prefix["duplicate_count"]) != int(saved["duplicate_count"]):
        problems.append(f"{label} duplicate count does not match E17")
    if int(rebuilt_prefix["baseline_index"]) != int(saved["baseline_pool_index"]):
        problems.append(f"{label} baseline pool index does not match E17")
    if list(rebuilt_prefix["baseline_key"]) != [int(item) for item in saved["baseline_assignment"]]:
        problems.append(f"{label} baseline assignment does not match E17")
    rebuilt_selected = [
        {
            "slot": int(slot),
            "pool_index": int(rebuilt_prefix["first_index"][key]),
            "assignment": [int(item) for item in key],
            "is_baseline": key == rebuilt_prefix["baseline_key"],
        }
        for slot, key in enumerate(rebuilt_prefix["selected"])
    ]
    saved_selected = [
        {
            "slot": int(item["slot"]),
            "pool_index": int(item["pool_index"]),
            "assignment": [int(value) for value in item["assignment"]],
            "is_baseline": bool(item["is_baseline"]),
        }
        for item in saved["selected"]
    ]
    if rebuilt_selected != saved_selected:
        problems.append(f"{label} old prefix selection does not match E17")
    for name in ("queue_bits", "known_tagged_bits", "known_ordinary_bits", "fifo_blocked_bits"):
        if _gap(saved[name], rebuilt_prefix[name]) > float(BIT_TOL):
            problems.append(f"{label} {name} gap exceeds 1 bit")
    if int(saved["known_bucket_count"]) != int(rebuilt_prefix["known_bucket_count"]):
        problems.append(f"{label} known bucket count does not match E17")
    saved_heads = saved.get("heads") or []
    rebuilt_heads = rebuilt_prefix["heads"]
    if len(saved_heads) != len(rebuilt_heads):
        problems.append(f"{label} FIFO head count does not match E17")
    else:
        for saved_head, rebuilt_head in zip(saved_heads, rebuilt_heads):
            for name in ("buoy", "insert_index", "priority_tag", "admission_step", "fifo_blocked_count"):
                if int(saved_head[name]) != int(rebuilt_head[name]):
                    problems.append(f"{label} FIFO head {name} does not match E17")
                    break
            if _gap(saved_head["bits_remaining"], rebuilt_head["bits_remaining"]) > float(BUCKET_TOL):
                problems.append(f"{label} FIFO head bits do not match E17")
            if _gap(saved_head["fifo_blocked_bits"], rebuilt_head["fifo_blocked_bits"]) > float(BUCKET_TOL):
                problems.append(f"{label} FIFO blocked bits do not match E17")
    return problems


def _tagged_buoys(ledger: dict, time_step: int) -> list[dict]:
    found = []
    for buoy, buckets in enumerate(ledger["buckets"]):
        bits = 0.0
        for bucket in buckets:
            if int(bucket["priority_tag"]) != 1:
                continue
            if int(bucket["admission_step"]) > int(time_step):
                continue
            if float(bucket["bits_remaining"]) <= float(DUST_BITS):
                continue
            bits += float(bucket["bits_remaining"])
        if bits > float(DUST_BITS):
            found.append({"buoy": int(buoy), "known_tagged_bits": float(bits)})
    return found


def _covers(assignment, buoy: int) -> bool:
    return any(int(item) == int(buoy) for item in assignment)


def pool_coverage_row(scenario: int, decision_time: int, pool, tops, illegal: int, ledger: dict, baseline_index: int) -> dict:
    """Coverage visible from the current queue and the legal pool. Replay results are not an input."""
    census = activity_census(pool, baseline_index)
    prefix = select_registered(pool, baseline_index, CANDIDATE_LIMIT)
    baseline = [int(item) for item in prefix["baseline_key"]]
    tagged = _tagged_buoys(ledger, decision_time)
    top_lists = np.asarray(tops, dtype=np.int64).tolist()
    tagged_rows = []
    for item in tagged:
        buoy = int(item["buoy"])
        visible = any(int(candidate) == buoy for row in top_lists for candidate in row)
        tagged_rows.append(
            {
                "buoy": buoy,
                "known_tagged_bits": float(item["known_tagged_bits"]),
                "in_any_uav_topk": bool(visible),
                "baseline_covers": bool(_covers(baseline, buoy)),
                "matched_alternatives_covering": int(sum(1 for key in census["matched_keys"] if _covers(key, buoy))),
            }
        )
    prefix_masks = ["".join(str(bit) for bit in active_mask(key)) for key in prefix["selected"]]
    return {
        "scenario_id": int(scenario),
        "decision_time": int(decision_time),
        "pool_size": int(prefix["pool_size"]),
        "unique_count": int(prefix["unique_count"]),
        "duplicate_count": int(prefix["duplicate_count"]),
        "illegal_count": int(illegal),
        "baseline_pool_index": int(baseline_index),
        "baseline_assignment": baseline,
        "baseline_active_count": int(sum(active_mask(baseline))),
        "baseline_mask": "".join(str(bit) for bit in census["baseline_mask"]),
        "matched_alternative_count": int(census["matched_alternative_count"]),
        "old_prefix_masks": prefix_masks,
        "unique_active_count_hist": census["unique_active_count_hist"],
        "unique_active_mask_hist": census["unique_active_mask_hist"],
        "per_uav_topk": top_lists,
        "tagged_target_coverage": tagged_rows,
        "prefix_selection": prefix,
    }
