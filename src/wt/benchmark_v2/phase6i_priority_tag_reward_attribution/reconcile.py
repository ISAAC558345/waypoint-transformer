"""Reconcile the saved evaluation CSV, summary intervals, and report counts."""
from __future__ import annotations

import csv
import json

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import DEV_POOL
from wt.benchmark_v2.phase6i_priority_tag_reward.criteria import merge_decision
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import TRAIN_SEEDS

from .protocol import SOURCE_REPORT, SOURCE_ROWS, SOURCE_SUMMARY, SUMMARY_MATCH_TOLERANCE


def load_saved_rows() -> list[dict]:
    with SOURCE_ROWS.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    parsed = []
    for row in rows:
        item = {"variant": str(row["variant"])}
        for key, value in row.items():
            if key == "variant":
                continue
            item[key] = float(value)
        item["model_seed"] = int(item["model_seed"])
        item["scenario_id"] = int(item["scenario_id"])
        parsed.append(item)
    return parsed


def saved_index(rows: list[dict]) -> dict:
    return {(str(row["variant"]), int(row["model_seed"]), int(row["scenario_id"])): row for row in rows}


def reconcile_saved() -> dict:
    rows = load_saved_rows()
    summary = json.loads(SOURCE_SUMMARY.read_text(encoding="utf-8"))
    report = SOURCE_REPORT.read_text(encoding="utf-8")
    fresh = merge_decision(rows)
    gaps = []
    count_problems = []
    expected = len(DEV_POOL) * (1 + 2 * len(TRAIN_SEEDS))
    if len(rows) != expected:
        count_problems.append(f"row count {len(rows)} != {expected}")
    for variant, seed in [("L0", 0)] + [(name, int(seed)) for name in ("L1", "L2") for seed in TRAIN_SEEDS]:
        found = [row for row in rows if row["variant"] == variant and int(row["model_seed"]) == int(seed)]
        if len(found) != len(DEV_POOL):
            count_problems.append(f"{variant} seed {seed} has {len(found)} scenarios")
        found_ids = sorted(int(row["scenario_id"]) for row in found)
        pool_ids = sorted(int(item) for item in DEV_POOL)
        if found_ids != pool_ids:
            count_problems.append(f"{variant} seed {seed} scenario ids differ from the development pool")
    for side in ("l2_seeds", "l1_seeds"):
        saved_rows = summary["decision"][side]
        fresh_rows = fresh[side]
        if len(saved_rows) != len(fresh_rows):
            count_problems.append(f"{side} length differs")
            continue
        for saved, new in zip(saved_rows, fresh_rows):
            if int(saved["model_seed"]) != int(new["model_seed"]):
                count_problems.append(f"{side} seed order differs")
            if bool(saved["passes_preregistered_criteria"]) != bool(new["passes_preregistered_criteria"]):
                count_problems.append(f"{saved['variant']} seed {saved['model_seed']} pass flag differs")
            for metric in ("urgency", "tagged", "payload"):
                for key in ("mean", "ci_low", "ci_high"):
                    gaps.append(abs(float(saved[metric][key]) - float(new[metric][key])))
            for key in (
                "payload_negative_scenarios",
                "safety_diff_sum",
                "constraint_diff_sum",
                "dropped_diff_mean",
                "sinr_diff_mean",
                "interference_diff_mean",
            ):
                gaps.append(abs(float(saved[key]) - float(new[key])))
    for key in ("l2_passing_seeds", "l1_passing_seeds"):
        if int(summary["decision"][key]) != int(fresh[key]):
            count_problems.append(f"{key} differs")
    if "PRIORITY_TAG_EXPERIMENT_NOT_MERGED" not in report:
        count_problems.append("report gate text is missing")
    if "L2 通过种子数：0" not in report or "L1 通过种子数：0" not in report:
        count_problems.append("report passing-seed sentence does not match the saved zero counts")
    max_gap = float(max(gaps) if gaps else 0.0)
    return {
        "row_count": len(rows),
        "expected_row_count": expected,
        "max_abs_gap": max_gap,
        "problems": count_problems,
        "matches": bool(not count_problems and max_gap <= float(SUMMARY_MATCH_TOLERANCE)),
        "saved_gate": summary.get("gate"),
        "saved_l2_passing_seeds": int(summary["decision"]["l2_passing_seeds"]),
        "saved_l1_passing_seeds": int(summary["decision"]["l1_passing_seeds"]),
        "recomputed_l2_passing_seeds": int(fresh["l2_passing_seeds"]),
        "recomputed_l1_passing_seeds": int(fresh["l1_passing_seeds"]),
    }
