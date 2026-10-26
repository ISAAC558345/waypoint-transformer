"""Write the E3 form comparison from the committed E2 rows."""
from __future__ import annotations

import csv
import json

from .compare import compare_forms, crosscheck_form_a
from .protocol import (
    E2_COMMIT,
    GATE_BLOCKED,
    GATE_INCONCLUSIVE,
    INSTRUCTION_COMMIT,
    LAMBDA_SELECTED,
    OUTPUT,
    REPORT,
    RHO_SELECTED,
    choose_gate,
)
from .report import render_report
from .sources import load_sources


def _compact(stats: dict, scenario_id: int | None = None) -> dict:
    payload = stats["bonus_over_payload"]
    penalty = stats["bonus_over_penalty"]
    row = {
        "bonus_mean": stats["bonus_mean"],
        "bonus_p50": stats["bonus_p50"],
        "bonus_p95": stats["bonus_p95"],
        "bonus_p99": stats["bonus_p99"],
        "bonus_max": stats["bonus_max"],
        "bonus_sum": stats["bonus_sum"],
        "bonus_over_payload_mean": payload["mean"],
        "bonus_over_payload_p95": payload["p95"],
        "bonus_over_penalty_p50": penalty["p50"],
        "bonus_over_penalty_p95": penalty["p95"],
        "bonus_over_penalty_p99": penalty["p99"],
        "near_zero_penalty_steps": penalty["near_zero_denominator_steps"],
        "positive_bonus_on_near_zero_penalty": penalty["positive_bonus_on_near_zero_denominator"],
        "order_changed_vs_base": int(stats["order_changed_vs_base"]),
        "material_order_changed": int(stats["material_order_changed"]),
        "payload_order_changed": int(stats["payload_order_changed"]),
        "payload_material_order_changed": int(stats["payload_material_order_changed"]),
        "sign_flips": stats["sign_flips"],
        "payload_sum": stats["payload_sum"],
        "urgency_served_bits_sum": stats["urgency_served_bits_sum"],
        "penalty_sum": stats["penalty_sum"],
        "projection_penalty_sum": stats["projection_penalty_sum"],
        "residual_penalty_sum": stats["residual_penalty_sum"],
        "zero_payload_steps": stats["zero_payload_steps"],
        "positive_urgency_on_zero_payload_steps": stats["positive_urgency_on_zero_payload_steps"],
        "signal_retained_fraction": stats["signal_retained_fraction"],
        "meets_conservative_standard": int(stats["meets_conservative_standard"]),
        "signal_too_weak": int(stats["signal_too_weak"]),
    }
    if scenario_id is not None:
        row["scenario_id"] = int(scenario_id)
    return row


def _conclusion(summary: dict) -> str:
    if summary["gate"] == GATE_BLOCKED:
        return "E2 源文件缺失、哈希或 schema 不一致，或者奖励字段不能复核。本阶段停止，没有比较出可用奖励形式。"
    if summary["gate"] == GATE_INCONCLUSIVE:
        return (
            "Form A 的正权重和 Form B 的全部 lambda/rho 组合，都不能在八个场景上同时满足预注册保守标准。"
            "标准没有放宽。没有证据支持进入训练，也不挑选看起来较小的组合。"
        )
    listed = "、".join(
        f"Form {item['form']} lambda={item['lambda_u']} rho={item['rho'] if item['rho'] is not None else '无'}"
        for item in summary["usable_candidates"]
    )
    return f"以下组合满足预注册保守标准，但仍只是待批准候选，不是选定值：{listed}。"


def _write_csv(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _schema() -> dict:
    return {
        "schema_version": "offline_urgency_reward_form_v1",
        "source": "results/phase6i_priority_tag_reward_scale_e2 committed at 5210dd7611befceed37c897a998cc1c5bd2a76bd",
        "reran_environment": False,
        "resampled_tags": False,
        "trained": False,
        "form_a": "r_base + lambda_u * urgency_term",
        "form_b": "r_base + min(lambda_u * urgency_term, rho * payload_term)",
        "payload_term": "E2 step column payload_term, equal to served_bits/1e6; not cumulative payload_bits",
        "lambda_form_a": [0.0, 0.0005, 0.001, 0.002],
        "lambda_reference": 0.05,
        "lambda_form_b": [0.0005, 0.001, 0.002],
        "rho": [0.001, 0.0025, 0.005],
        "lambda_selected": None,
        "rho_selected": None,
        "conservative_standard": {
            "mean_bonus_over_payload_max": 0.01,
            "p95_bonus_over_penalty_max": 1.0,
            "payload_gap_tie": 1.0e-6,
            "signal_retained_min": 0.5,
        },
        "closed_loop_claim": False,
    }


def main() -> None:
    loaded = load_sources()
    summary = {
        "gate": GATE_BLOCKED,
        "reason": "",
        "instruction_commit": INSTRUCTION_COMMIT,
        "expected_e2_commit": E2_COMMIT,
        "source_ok": bool(loaded["source_ok"]),
        "source_commit_ok": bool(loaded["source_ok"] or (loaded["git_commit"] and all(value == E2_COMMIT for value in loaded["git_commit"].values()))),
        "source_clean": bool(not any("differs" in item for item in loaded["problems"])),
        "problems": loaded["problems"],
        "source_sha256": loaded["sha256"],
        "source_git_commit": loaded["git_commit"],
        "step_rows": len(loaded["steps"]),
        "episode_rows": len(loaded["episodes"]),
        "lambda_selected": LAMBDA_SELECTED,
        "rho_selected": RHO_SELECTED,
        "trained": False,
        "environment_rerun": False,
        "held_out_splits_read": False,
        "formal_system_modified": False,
        "form_a_matches_e2": None,
        "payload_sums_identical_across_forms": None,
        "payload_caps_hold": None,
        "usable_candidates": [],
        "combinations": [],
        "pooled_rows": [],
        "conclusion": "",
    }
    form_rows: list[dict] = []
    analysis_ok = False
    any_candidate = False
    if loaded["source_ok"]:
        comparison = compare_forms(loaded["steps"])
        checked = crosscheck_form_a(comparison, loaded["summary"])
        analysis_ok = bool(checked["ok"] and comparison["payload_sums_identical_across_forms"] and comparison["payload_caps_hold"])
        any_candidate = bool(comparison["any_candidate_meets_standard"]) if analysis_ok else False
        summary["form_a_matches_e2"] = bool(checked["ok"])
        summary["form_a_mismatches"] = checked["mismatches"]
        summary["payload_sums_identical_across_forms"] = bool(comparison["payload_sums_identical_across_forms"])
        summary["payload_caps_hold"] = bool(comparison["payload_caps_hold"])
        summary["standards"] = comparison["standards"]
        summary["usable_candidates"] = comparison["usable_candidates"]
        summary["combinations"] = [
            {
                "form": combo["form"],
                "lambda_u": combo["lambda_u"],
                "rho": combo["rho"],
                "role": combo["role"],
                "all_scenarios_meet_conservative_standard": combo["all_scenarios_meet_conservative_standard"],
                "signal_too_weak": combo["signal_too_weak"],
                "usable_candidate": combo["usable_candidate"],
                "cap_holds": combo["cap_holds"],
                "worst_scenario_id": combo["worst_scenario_id"],
                "pooled": _compact(combo["pooled"]),
                "scenarios": [_compact(item, int(item["scenario_id"])) for item in combo["per_scenario"]],
            }
            for combo in comparison["combinations"]
        ]
        summary["pooled_rows"] = []
        for combo in summary["combinations"]:
            pooled = dict(combo["pooled"])
            pooled.update(
                {
                    "form": combo["form"],
                    "lambda_u": combo["lambda_u"],
                    "rho": "" if combo["rho"] is None else combo["rho"],
                    "worst_scenario_id": combo["worst_scenario_id"],
                    "all_scenarios_meet_conservative_standard": int(combo["all_scenarios_meet_conservative_standard"]),
                    "signal_retained_fraction": pooled["signal_retained_fraction"],
                    "scenarios_with_base_material_reorder": int(sum(int(item["material_order_changed"]) for item in combo["scenarios"])),
                    "scenarios_with_payload_material_reorder": int(sum(int(item["payload_material_order_changed"]) for item in combo["scenarios"])),
                }
            )
            summary["pooled_rows"].append(pooled)
        form_rows = comparison["form_rows"]
        if not analysis_ok:
            summary["problems"] = list(summary["problems"]) + ["Form A 不能复现 E2，或 payload cap 没有保持"]
    summary["gate"] = choose_gate(source_ok=bool(loaded["source_ok"]), analysis_ok=analysis_ok, any_candidate_meets_standard=any_candidate)
    if summary["gate"] == GATE_BLOCKED:
        summary["reason"] = "E2 源文件缺失、哈希或 schema 不一致，或者奖励字段不能复核"
    elif summary["gate"] == GATE_INCONCLUSIVE:
        summary["reason"] = "没有任何奖励形式同时满足预注册保守标准"
    else:
        summary["reason"] = "至少一种形式满足预注册标准，但仍需负责人批准，不能当作最优值"
    summary["conclusion"] = _conclusion(summary)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT / "form_rows.csv", form_rows)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "schema.json").write_text(json.dumps(_schema(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(render_report(summary), encoding="utf-8")
    print(summary["gate"], summary["form_a_matches_e2"], len(form_rows), len(summary["usable_candidates"]))


if __name__ == "__main__":
    main()
