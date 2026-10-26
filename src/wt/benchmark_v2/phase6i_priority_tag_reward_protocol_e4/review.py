"""Assemble the protocol review from stored E2/E3 fields."""
from __future__ import annotations

from .protocol import (
    ACTION_SCALE,
    CANDIDATE_STATUS,
    CONDITIONAL_LAMBDA,
    CONDITIONAL_RHO,
    E2_COMMIT,
    E2_GATE,
    E3_COMMIT,
    E3_GATE,
    INSTRUCTION_COMMIT,
    LAMBDA_CANDIDATES,
    LAMBDA_PROJECT,
    LAMBDA_RESIDUAL,
    LAMBDA_SELECTED,
    MEAN_BONUS_OVER_PAYLOAD_MAX,
    OWNER_APPROVED,
    P95_BONUS_OVER_PENALTY_MAX,
    PAYLOAD_SCALE,
    PAYLOAD_TIE,
    RHO_CANDIDATES,
    RHO_SELECTED,
    SIGNAL_RETAINED_MIN,
    choose_gate,
)


def _same(left: object, right: float) -> bool:
    return left is not None and abs(float(left) - float(right)) < 1.0e-12


def _find(combinations: list[dict], form: str, lambda_u: float, rho: float | None) -> dict | None:
    for combo in combinations:
        if combo.get("form") != form or not _same(combo.get("lambda_u"), lambda_u):
            continue
        if rho is None and combo.get("rho") is None:
            return combo
        if rho is not None and _same(combo.get("rho"), rho):
            return combo
    return None


def _ids(combo: dict, key: str) -> list[int]:
    return [int(item["scenario_id"]) for item in combo.get("scenarios", []) if item.get(key)]


def _cite(combo: dict) -> dict:
    pooled = combo["pooled"]
    return {
        "form": combo["form"],
        "lambda_u": combo["lambda_u"],
        "rho": combo["rho"],
        "role": combo["role"],
        "bonus_mean": pooled["bonus_mean"],
        "bonus_p95": pooled["bonus_p95"],
        "bonus_max": pooled["bonus_max"],
        "bonus_over_payload_mean": pooled["bonus_over_payload_mean"],
        "bonus_over_payload_p95": pooled["bonus_over_payload_p95"],
        "bonus_over_penalty_p50": pooled["bonus_over_penalty_p50"],
        "bonus_over_penalty_p95": pooled["bonus_over_penalty_p95"],
        "bonus_over_penalty_p99": pooled["bonus_over_penalty_p99"],
        "signal_retained_fraction": pooled["signal_retained_fraction"],
        "positive_urgency_on_zero_payload_steps": pooled["positive_urgency_on_zero_payload_steps"],
        "base_material_scenarios": _ids(combo, "material_order_changed"),
        "payload_material_scenarios": _ids(combo, "payload_material_order_changed"),
        "all_scenarios_meet_conservative_standard": bool(combo["all_scenarios_meet_conservative_standard"]),
        "cap_holds": bool(combo["cap_holds"]),
        "usable_candidate": bool(combo["usable_candidate"]),
        "worst_scenario_id": combo["worst_scenario_id"],
    }


def _e2_lambda_rows(summary: dict) -> list[dict]:
    rows = []
    for item in summary.get("scale", {}).get("by_lambda", []):
        payload = item.get("bonus_over_payload", {})
        penalty = item.get("bonus_over_penalty", {})
        rows.append(
            {
                "lambda_u": item.get("lambda_u"),
                "bonus_mean": item.get("bonus_mean"),
                "bonus_p95": item.get("bonus_p95"),
                "bonus_over_payload_mean": payload.get("mean"),
                "bonus_over_payload_max": payload.get("max"),
                "bonus_over_penalty_p95": penalty.get("p95"),
                "episodes_with_material_reorder": item.get("episodes_with_material_reorder"),
                "supported_by_preregistered_bound": item.get("supported_by_preregistered_bound"),
                "selected": item.get("selected"),
            }
        )
    return rows


def build_review(loaded: dict) -> dict:
    problems = list(loaded["problems"])
    constants = loaded["reward_constants"]
    for name, expected in (
        ("LAMBDA_PROJECT", LAMBDA_PROJECT),
        ("LAMBDA_RESIDUAL", LAMBDA_RESIDUAL),
        ("PAYLOAD_SCALE", PAYLOAD_SCALE),
        ("ACTION_SCALE", ACTION_SCALE),
    ):
        if not _same(constants.get(name), expected):
            problems.append(f"formal constant mismatch {name}")
    e3 = loaded["e3_summary"]
    standards = e3.get("standards", {})
    for name, expected in (
        ("mean_bonus_over_payload_max", MEAN_BONUS_OVER_PAYLOAD_MAX),
        ("p95_bonus_over_penalty_max", P95_BONUS_OVER_PENALTY_MAX),
        ("payload_tie", PAYLOAD_TIE),
        ("signal_retained_min", SIGNAL_RETAINED_MIN),
    ):
        if not _same(standards.get(name), expected):
            problems.append(f"E3 standard changed or missing: {name}")
    schema_standard = loaded["e3_schema"].get("conservative_standard", {})
    if not _same(schema_standard.get("p95_bonus_over_penalty_max"), P95_BONUS_OVER_PENALTY_MAX):
        problems.append("E3 schema penalty threshold is not the preregistered value")
    if e3.get("usable_candidates") not in ([], None) and e3.get("usable_candidates") != []:
        problems.append("E3 records a usable candidate; this review will not promote it")
    if e3.get("lambda_selected") is not None or e3.get("rho_selected") is not None:
        problems.append("E3 already selected lambda or rho")
    if e3.get("form_a_matches_e2") is not True or e3.get("payload_caps_hold") is not True:
        problems.append("E3 cross-check flags are not both true")
    if int(e3.get("step_rows") or 0) != 384 or int(e3.get("episode_rows") or 0) != 8:
        problems.append("E3 row counts are not 384 steps and 8 episodes")
    combinations = list(e3.get("combinations") or [])
    cited = []
    for combo in combinations:
        try:
            cited.append(_cite(combo))
        except KeyError as exc:
            problems.append(f"E3 combination is missing {exc}")
            break
    conditional = _find(combinations, "B", CONDITIONAL_LAMBDA, CONDITIONAL_RHO)
    conditional_cite = None
    if conditional is None:
        problems.append("E3 is missing Form B lambda 0.0005 rho 0.001")
    else:
        conditional_cite = _cite(conditional)
        if conditional_cite["payload_material_scenarios"]:
            problems.append("conditional candidate has a payload-step material reorder in the stored E3 rows")
        if conditional_cite["all_scenarios_meet_conservative_standard"] or conditional_cite["usable_candidate"]:
            problems.append("stored E3 marks the conditional candidate as passing")
        if float(conditional_cite["bonus_over_penalty_p95"]) <= P95_BONUS_OVER_PENALTY_MAX:
            problems.append("stored penalty ratio no longer exceeds the preregistered threshold")
    grid_lambda = sorted({float(row["lambda_u"]) for row in cited if row["form"] == "B"})
    grid_rho = sorted({float(row["rho"]) for row in cited if row["form"] == "B" and row["rho"] is not None})
    if grid_lambda != list(LAMBDA_CANDIDATES) or grid_rho != list(RHO_CANDIDATES):
        problems.append("E3 Form B grid is not the preregistered lambda/rho set")
    sources_ok = not problems
    return {
        "gate": choose_gate(sources_ok=sources_ok, owner_approved=OWNER_APPROVED),
        "e2_gate": E2_GATE,
        "e3_gate": E3_GATE,
        "e3_gate_preserved": E3_GATE,
        "e3_gate_rewritten": False,
        "standards_loosened": False,
        "instruction_commit": INSTRUCTION_COMMIT,
        "e2_commit": E2_COMMIT,
        "e3_commit": E3_COMMIT,
        "problems": problems,
        "sources_ok": sources_ok,
        "e2_files": loaded["e2_files"],
        "e3_files": loaded["e3_files"],
        "e1_schema_file": loaded["e1_schema_file"],
        "formal_files": loaded["formal_files"],
        "formal_formula": "r_base = served_bits/1e6 - 0.05*c_project - 0.01*c_residual",
        "formal_formula_ok": bool(loaded["formal_formula_ok"]),
        "reward_constants": {
            "lambda_project": constants.get("LAMBDA_PROJECT"),
            "lambda_residual": constants.get("LAMBDA_RESIDUAL"),
            "payload_scale": constants.get("PAYLOAD_SCALE"),
            "action_scale": constants.get("ACTION_SCALE"),
        },
        "checkpoint": loaded["e2_summary"].get("checkpoint"),
        "checkpoint_sha256": loaded["e2_summary"].get("checkpoint_sha256"),
        "e2_lambda_rows": _e2_lambda_rows(loaded["e2_summary"]),
        "cited_forms": cited,
        "conditional_candidate": {
            "status": CANDIDATE_STATUS,
            "form": "B",
            "lambda_u": CONDITIONAL_LAMBDA,
            "rho": CONDITIONAL_RHO,
            "selected": False,
            "approved": False,
            "training_authorized": False,
            "passed_e3_standard": False,
            "evidence": conditional_cite,
        },
        "preferred_design": {
            "name": "B",
            "formula": "r_base + min(lambda_u * U_t, rho * payload_term)",
            "performance_claim": False,
            "reason": "The cap limits the urgency bonus to a fixed fraction of this step's successful payload term.",
        },
        "lambda_selected": LAMBDA_SELECTED,
        "rho_selected": RHO_SELECTED,
        "lambda_h": "NOT_SELECTED",
        "delta_payload_max": "NOT_REQUIRED_FOR_THIS_PROTOCOL",
        "high_level_rule": "payload-first; low-level bonus is not added to the high-level objective",
        "scheme_c": "frozen",
        "synthetic_schema": "synthetic_priority_bucket_v1",
        "owner_decisions": {
            "accept_stop_or_rollback_if_benchmark_payload_drops": None,
            "approve_lambda_0_0005_rho_0_001_as_only_first_lowlevel_candidate": None,
            "remain_untrained_unless_approved": True,
        },
        "environment_rerun": False,
        "tags_resampled": False,
        "trained": False,
        "formal_system_modified": False,
        "held_out_splits_read": False,
        "e5_started": False,
        "closed_loop_claim": False,
        "standards": {
            "mean_bonus_over_payload_max": MEAN_BONUS_OVER_PAYLOAD_MAX,
            "p95_bonus_over_penalty_max": P95_BONUS_OVER_PENALTY_MAX,
            "payload_tie": PAYLOAD_TIE,
            "signal_retained_min": SIGNAL_RETAINED_MIN,
        },
    }
