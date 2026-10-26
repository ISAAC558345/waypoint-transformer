"""Write the Stage D7 rule-design tables. No environment rollout."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from .analyze import delta_for, fields_support_rules, sensitivity_rows, state_record
from .protocol import (
    CANDIDATES_PER_STATE,
    DELTA_PAYLOAD_MAX,
    D6_POOL_COUNT,
    EPS_BITS,
    FRACTIONS,
    GATE_READY,
    INSTRUCTION,
    INSTRUCTION_COMMIT,
    LAMBDA_H,
    OUTPUT,
    PRIMARY_SOURCE,
    PRIMARY_TIMES,
    REPORT,
    ROOT,
    choose_gate,
)
from .report import render_report
from .sources import SPECS, load_states

STATE_FIELDS = [
    "source",
    "scenario_id",
    "time_step",
    "horizon",
    "horizon_source",
    "label_family",
    "arrival_semantics",
    "state_path",
    "candidate_count",
    "pool_count",
    "pool_count_source",
    "state_hash",
    "pool_hash",
    "subset_hash",
    "pareto_count",
    "pareto_candidate_ids",
    "payload_tie_count",
    "payload_tie_fraction",
    "payload_span_bits",
    "best_payload_bits",
    "best_urgent_bits",
    "payload_id_candidate_id",
    "payload_id_assignment",
    "rule_p_candidate_id",
    "rule_p_assignment",
    "rule_p_payload_bits",
    "rule_p_urgent_bits",
    "rule_p_payload_regret_bits",
    "rule_p_urgent_gain_vs_payload_id_bits",
    "urgent_first_candidate_id",
    "urgent_first_assignment",
    "urgent_first_payload_bits",
    "urgent_first_urgent_bits",
    "urgent_first_payload_regret_bits",
    "urgent_first_urgent_gain_vs_rule_p_bits",
    "rule_p_matches_urgent_first",
    "descriptive_exchange_ratio",
    "field_gaps",
    "primary_rule_state",
]

SENSITIVITY_FIELDS = [
    "source",
    "scenario_id",
    "time_step",
    "horizon",
    "label_family",
    "fraction_of_payload_span",
    "delta_payload_max_bits",
    "budget_selected",
    "candidate_id",
    "assignment",
    "payload_bits",
    "urgent_service_bits",
    "payload_regret_bits",
    "urgent_gain_vs_rule_p_bits",
    "assignment_changed_vs_rule_p",
    "matches_urgent_first",
    "eligible_count",
    "pareto_count",
    "primary_sensitivity",
]

EXPECTED_GATES = {
    "D0": "HIGHLEVEL_URGENCY_LABEL_PILOT_INCONCLUSIVE",
    "D1": "T30_URGENCY_SIGNAL_SUPPORTED",
    "D5": "ARRIVAL_AWARE_URGENCY_LABEL_COMPLETE",
    "D6": "ARRIVAL_AWARE_LABEL_EXPANSION_INCONCLUSIVE",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-c", "core.longpaths=true", "log", "-1", "--format=%H", "--", path.relative_to(ROOT).as_posix()],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if not commit:
        raise RuntimeError(f"no commit for {path}")
    return commit


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _mean(values: list[float]) -> float:
    if not values:
        raise RuntimeError("mean of an empty group")
    return float(sum(values) / len(values))


def _block(rows: list[dict]) -> dict:
    horizons = sorted({int(row["horizon"]) for row in rows})
    pools = sorted({str(row["pool_count"]) for row in rows})
    exchanges = [float(row["descriptive_exchange_ratio"]) for row in rows if row["descriptive_exchange_ratio"] != ""]
    return {
        "states": int(len(rows)),
        "horizon": horizons[0] if len(horizons) == 1 else "mixed",
        "label_family": rows[0]["label_family"],
        "candidate_count": int(rows[0]["candidate_count"]),
        "horizon_source": rows[0]["horizon_source"],
        "pool_count_text": "summary 未记录" if pools == [""] else ", ".join(pools),
        "pareto_min": int(min(int(row["pareto_count"]) for row in rows)),
        "pareto_max": int(max(int(row["pareto_count"]) for row in rows)),
        "pareto_mean": _mean([float(row["pareto_count"]) for row in rows]),
        "payload_tie_fraction_mean": _mean([float(row["payload_tie_fraction"]) for row in rows]),
        "tiebreak_gain_mean": _mean([float(row["rule_p_urgent_gain_vs_payload_id_bits"]) for row in rows]),
        "material_tiebreak_states": int(sum(float(row["rule_p_urgent_gain_vs_payload_id_bits"]) > float(EPS_BITS) for row in rows)),
        "matches_urgent_first": int(sum(int(row["rule_p_matches_urgent_first"]) == 1 for row in rows)),
        "urgent_regret_mean": _mean([float(row["urgent_first_payload_regret_bits"]) for row in rows]),
        "urgent_regret_min": float(min(float(row["urgent_first_payload_regret_bits"]) for row in rows)),
        "urgent_regret_max": float(max(float(row["urgent_first_payload_regret_bits"]) for row in rows)),
        "material_regret_states": int(sum(float(row["urgent_first_payload_regret_bits"]) > float(EPS_BITS) for row in rows)),
        "urgent_gain_vs_rule_p_mean": _mean([float(row["urgent_first_urgent_gain_vs_rule_p_bits"]) for row in rows]),
        "exchange_count": int(len(exchanges)),
        "exchange_min": min(exchanges) if exchanges else None,
        "exchange_max": max(exchanges) if exchanges else None,
        "rows": rows,
    }


def _sensitivity_block(rows: list[dict]) -> list[dict]:
    grouped: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[float(row["fraction_of_payload_span"])].append(row)
    output = []
    for fraction in FRACTIONS:
        items = grouped.get(float(fraction), [])
        if not items:
            continue
        regrets = [float(item["payload_regret_bits"]) for item in items]
        gains = [float(item["urgent_gain_vs_rule_p_bits"]) for item in items]
        output.append(
            {
                "fraction": float(fraction),
                "states": int(len(items)),
                "changed_states": int(sum(int(item["assignment_changed_vs_rule_p"]) == 1 for item in items)),
                "matched_urgent_states": int(sum(int(item["matches_urgent_first"]) == 1 for item in items)),
                "regret_mean": _mean(regrets),
                "regret_min": float(min(regrets)),
                "regret_max": float(max(regrets)),
                "gain_mean": _mean(gains),
            }
        )
    return output


def _recommendation(records: list[dict]) -> dict:
    primary = [row for row in records if int(row["primary_rule_state"]) == 1]
    by_time = {time: [row for row in primary if int(row["time_step"]) == int(time)] for time in PRIMARY_TIMES}
    must_spend = False
    tiebreak_everywhere = True
    counts = {}
    for time, rows in by_time.items():
        spend = sum(float(row["urgent_first_payload_regret_bits"]) > float(EPS_BITS) for row in rows)
        tiebreak = sum(float(row["rule_p_urgent_gain_vs_payload_id_bits"]) > float(EPS_BITS) for row in rows)
        matched = sum(int(row["rule_p_matches_urgent_first"]) == 1 for row in rows)
        counts[str(time)] = {"states": len(rows), "material_regret_states": int(spend), "material_tiebreak_states": int(tiebreak), "matches_urgent_first": int(matched)}
        if len(rows) != 8 or spend > 0:
            must_spend = True
        if tiebreak != len(rows):
            tiebreak_everywhere = False
    worth_trial = bool((not must_spend) and tiebreak_everywhere)
    if worth_trial:
        detail = (
            "Rule P 在 t=15 和 t=30 的每个状态都取得了超过 1 bit 的 tie-break 增益，"
            "且抽样 urgent-first 仍在 payload 并列组内。可以做独立小规模双头试验，但不能据此构建正式全量数据集。"
        )
    else:
        regret_15 = _mean([float(row["urgent_first_payload_regret_bits"]) for row in by_time[15]])
        regret_30 = _mean([float(row["urgent_first_payload_regret_bits"]) for row in by_time[30]])
        detail = (
            f"D6 t=15 有 {counts['15']['material_regret_states']}/8 个状态、t=30 有 {counts['30']['material_regret_states']}/8 个状态的抽样 urgent-first 不在 1 bit payload 并列组内。"
            "这些主时刻的 payload 最高候选在 1 bit 内是唯一的，Rule P 相对只按候选 ID 打破平局的 urgent 增益为 0。"
            f"换成 urgent-first 的 payload regret 均值在 t=15 为 {regret_15:,.3f} bit，在 t=30 为 {regret_30:,.3f} bit。"
            "因此不开始双头试验，也不构建全量数据集。下一步只等待研究负责人确定 `delta_payload_max`，网格里的比例都不是预算。"
        )
    return {
        "worth_dual_head_trial": worth_trial,
        "worth_full_dataset": False,
        "wait_for_delta_payload_max": bool(not worth_trial),
        "by_time": counts,
        "detail": detail,
    }


def _sensitivity_complete(rows: list[dict], records: list[dict]) -> bool:
    primary_records = [row for row in records if int(row["primary_rule_state"]) == 1]
    if len(primary_records) != 16:
        return False
    if any(int(row["candidate_count"]) != int(CANDIDATES_PER_STATE) or int(row["horizon"]) != 15 for row in primary_records):
        return False
    primary = [row for row in rows if int(row["primary_sensitivity"]) == 1]
    if len(primary) != 16 * len(FRACTIONS):
        return False
    horizons = {int(row["horizon"]) for row in primary}
    if horizons != {15}:
        return False
    for record in primary_records:
        anchors = [
            row
            for row in primary
            if int(row["scenario_id"]) == int(record["scenario_id"])
            and int(row["time_step"]) == int(record["time_step"])
            and float(row["fraction_of_payload_span"]) == 0.0
        ]
        if len(anchors) != 1:
            return False
        anchor = anchors[0]
        if int(anchor["candidate_id"]) != int(record["rule_p_candidate_id"]):
            return False
        if int(anchor["budget_selected"]) != 0:
            return False
        if abs(float(anchor["delta_payload_max_bits"]) - delta_for(float(record["payload_span_bits"]), 0.0)) > 0.0:
            return False
    return True


def main() -> None:
    if not INSTRUCTION.is_file():
        raise RuntimeError("D7 instruction file is missing")
    batches, manifest = load_states()
    records = [state_record(batch) for batch in batches]
    sensitivity = [row for batch in batches for row in sensitivity_rows(batch)]
    for spec, item in zip(SPECS, manifest):
        result = ROOT / spec.result_dir
        item["commit"] = _commit(result / "labels.csv")
        item["labels_sha256"] = _sha256(result / "labels.csv")
        item["selected_sha256"] = _sha256(result / "selected_candidates.csv")
        item["summary_sha256"] = _sha256(result / "summary.json")
        item["schema_sha256"] = _sha256(result / "schema.json")
        item["report_sha256"] = _sha256(ROOT / spec.report)
        item["report_commit"] = _commit(ROOT / spec.report)
    sources_verified = all(item["summary_gate"] == EXPECTED_GATES[item["source"]] and item["commit"] for item in manifest)
    d6 = [row for row in records if row["source"] == "D6"]
    d6_ids = {int(row["scenario_id"]) for row in d6}
    split_ok = d6_ids == set(range(322016, 322024)) and all(int(row["pool_count"]) == int(D6_POOL_COUNT) for row in d6)
    split_ok = split_ok and {int(row["time_step"]) for row in d6} == {0, 15, 30, 45}
    sensitivity_ok = _sensitivity_complete(sensitivity, records)
    rules_ok = fields_support_rules(records)
    delta_unset = DELTA_PAYLOAD_MAX is None and LAMBDA_H is None and all(int(row["budget_selected"]) == 0 for row in sensitivity)
    gate = choose_gate(
        sources_verified=sources_verified,
        split_ok=split_ok,
        sensitivity_complete=sensitivity_ok,
        fields_support_rules=rules_ok,
        delta_unset=delta_unset,
    )
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(str(row["source"]), int(row["time_step"]))].append(row)
    blocks = {f"{source}:{time}": _block(rows) for (source, time), rows in grouped.items()}
    sensitivity_groups = {
        "D6:15": _sensitivity_block([row for row in sensitivity if row["source"] == "D6" and int(row["time_step"]) == 15]),
        "D6:30": _sensitivity_block([row for row in sensitivity if row["source"] == "D6" and int(row["time_step"]) == 30]),
        "D6:0": _sensitivity_block([row for row in sensitivity if row["source"] == "D6" and int(row["time_step"]) == 0]),
        "D6:45": _sensitivity_block([row for row in sensitivity if row["source"] == "D6" and int(row["time_step"]) == 45]),
        "D5:30": _sensitivity_block([row for row in sensitivity if row["source"] == "D5" and int(row["time_step"]) == 30]),
    }
    gap_groups = []
    for source in ("D0", "D1", "D5", "D6"):
        gaps = sorted({gap for row in records if row["source"] == source for gap in str(row["field_gaps"]).split("|") if gap})
        gap_groups.append({"source": source, "gaps": "、".join(gaps)})
    recommendation = _recommendation(records)
    state_path = OUTPUT / "state_rule_rows.csv"
    sensitivity_path = OUTPUT / "delta_sensitivity.csv"
    _write_csv(state_path, records, STATE_FIELDS)
    _write_csv(sensitivity_path, sensitivity, SENSITIVITY_FIELDS)
    summary = {
        "gate": gate,
        "reasons": [] if gate == GATE_READY else ["sources, split, sensitivity, fields, or the unset budget check failed"],
        "lambda_H": None,
        "delta_payload_max": None,
        "budget_selected": False,
        "rollouts_run": 0,
        "learned_policy": False,
        "trained_hgnn": False,
        "trained_mappo": False,
        "formal_system_modified": False,
        "held_out_splits_read": False,
        "horizons_pooled": False,
        "instruction_commit": INSTRUCTION_COMMIT,
        "eps_bits": float(EPS_BITS),
        "fractions": [float(item) for item in FRACTIONS],
        "candidates_per_state": int(CANDIDATES_PER_STATE),
        "d6_pool_count": int(D6_POOL_COUNT),
        "primary_source": PRIMARY_SOURCE,
        "primary_times": list(PRIMARY_TIMES),
        "primary_state_counts": {str(time): int(sum(int(row["primary_rule_state"]) == 1 and int(row["time_step"]) == int(time) for row in records)) for time in PRIMARY_TIMES},
        "sources_verified": bool(sources_verified),
        "split_ok": bool(split_ok),
        "sensitivity_complete": bool(sensitivity_ok),
        "fields_support_rules": bool(rules_ok),
        "sources": manifest,
        "field_gap_groups": gap_groups,
        "recommendation": recommendation,
        "blocks": blocks,
        "sensitivity": sensitivity_groups,
        "output_sha256": {
            "state_rule_rows": _sha256(state_path),
            "delta_sensitivity": _sha256(sensitivity_path),
        },
    }
    schema = {
        "meaning": "Offline per-state rules on frozen labels. No fraction is a selected payload-loss budget.",
        "rules": {
            "rule_p": "maximum real payload; urgent service breaks ties only inside the 1-bit payload group; a remaining tie uses the smaller candidate id",
            "rule_c": "keep candidates whose payload loss from the state maximum is within delta_payload_max after sub-bit gaps are treated as zero, then apply the same urgent top rule",
            "urgent_first": "maximum urgent service on the sampled subset, reported as a reference and not selected",
        },
        "fractions_of_each_state_payload_span": [float(item) for item in FRACTIONS],
        "budget_selected": False,
        "lambda_H": None,
        "delta_payload_max": None,
        "eps_bits": float(EPS_BITS),
        "horizons_pooled": False,
        "primary_data": "D6 real-arrival labels at t=15 and t=30",
        "sampled_pool": "128 of 13327 candidates; the Pareto count is not the full legal pool",
        "held_out_splits_read": False,
        "rollouts_run": 0,
        "instruction_commit": INSTRUCTION_COMMIT,
        "fields": {"state_rule_rows.csv": STATE_FIELDS, "delta_sensitivity.csv": SENSITIVITY_FIELDS},
    }
    summary_path = OUTPUT / "summary.json"
    schema_path = OUTPUT / "schema.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(render_report(summary), encoding="utf-8")
    print(gate, recommendation["worth_dual_head_trial"], recommendation["worth_full_dataset"])


if __name__ == "__main__":
    main()
