"""M1.4 attribution of saved calibration requests.

The production planner, repair rule and admission rule are unchanged. This
module only reads their scores and separates a one-step value into the
no-update state term and the two information premiums.
"""

from __future__ import annotations

import math

from wt.active_model_identification.belief import quadrature
from wt.active_model_identification.policies import _repair, pick_action, select_lookahead, tie_abs
from wt.active_model_identification.segment import expectation_pair, plan_snapshot, roll_action
from wt.active_model_identification.world import action_allowed

SNAPSHOTS = (
    ("b1_tight", "conflict_repair"),
    ("b3_window", "conflict_repair"),
    ("b1_loose", "conflict_repair"),
    ("b1_window", "single_score"),
    ("b4_window", "single_score"),
    ("b2_loose", "single_score"),
)
CONTINUATION_TRIGGERS = {"state_cost_without_update", "information_premium", "mixed"}


def action_label(action: tuple) -> str:
    return f"{action[0]}:{action[1]}"


def origin_of(table_choice: str) -> str:
    if table_choice == "calibrate:None":
        return "single_score"
    return "conflict_repair"


def duration_of(executed: str) -> tuple[float | None, str]:
    """A final executed calibrate occupied the whole interval.

    The return guard only replaces an action and does not switch back. The
    saved decision row does not store the seconds before that replacement.
    """
    if executed == "calibrate:None":
        return 30.0, "final_executed_action_is_calibrate"
    return None, "return_guard_replaced_partial_seconds_not_logged"


def classify_requests(rows: list[dict]) -> list[dict]:
    """Label every core calibration request. Return-guard replacement is separate."""
    partners = {}
    for row in rows:
        if row.get("branch") != "core" or row.get("method") != "lookahead_calibration_on":
            continue
        partners[(row["scenario_id"], str(row["epoch"]), str(row["uav"]))] = row
    labeled = []
    for row in rows:
        if row.get("branch") != "core" or row.get("method") != "lookahead_calibration_on":
            continue
        if row.get("requested") != "calibrate:None":
            continue
        origin = origin_of(str(row["table_choice"]))
        duration, basis = duration_of(str(row["executed"]))
        partner = partners.get((row["scenario_id"], str(row["epoch"]), str(1 - int(row["uav"]))))
        labeled.append(
            {
                "base": str(row["scenario_id"]).split("_", 1)[0],
                "condition": str(row["scenario_id"]).split("_", 1)[1],
                "scenario_id": row["scenario_id"],
                "epoch": int(row["epoch"]),
                "uav": int(row["uav"]),
                "table_choice": row["table_choice"],
                "requested": row["requested"],
                "executed": row["executed"],
                "conflict_partner": "" if partner is None or origin != "conflict_repair" else int(partner["uav"]),
                "partner_table_choice": "" if partner is None else partner["table_choice"],
                "partner_requested": "" if partner is None else partner["requested"],
                "Qcal": float(row["calibrate_q"]) if row["calibrate_q"] != "" else None,
                "best_execute_Q": float(row["best_execute_q"]) if row["best_execute_q"] != "" else None,
                "Q_gap": float(row["q_gap_calibrate_minus_execute"]) if row["q_gap_calibrate_minus_execute"] != "" else None,
                "intervention": str(row["intervened"]).lower() == "true",
                "origin": origin,
                "duration_s": duration,
                "duration_basis": basis,
            }
        )
    return labeled


def pair_rows(episodes: list[dict]) -> list[dict]:
    """One row per saved episode, keyed by scenario rather than by base alone."""
    rows = []
    for episode in episodes:
        rows.append(
            {
                "base": episode["base"],
                "condition": episode["condition"],
                "scenario_id": episode["scenario_id"],
                "method": episode["method"],
                "branch": episode["branch"],
                "utility": episode["utility"],
                "on_time_bits": episode["on_time_bits"],
                "energy_j": episode["energy_j"],
                "calibration_s": episode["calibration_s"],
                "returned": episode["returned"],
                "not_returned": episode["not_returned"],
                "not_landed": episode["not_landed"],
                "below_reserve": episode["below_reserve"],
                "energy_exhausted": episode["energy_exhausted"],
                "return_failures": episode["return_failures"],
            }
        )
    return rows


def choose_alternative(table, partner_action, world, belief, spec, factor, uav_index):
    """Best legal non-calibrate action compatible with the partner's joint target.

    This is not the globally best joint plan. A shared service buoy is excluded.
    An action the existing admission rule rejects is excluded, because the option
    guard would replace it before it runs.
    """
    tau = tie_abs(spec)
    theta = belief.quantile(float(spec["belief"]["guard_probability"]))
    conflict_buoy = partner_action[1] if partner_action[0] == "service" else None
    notes = []
    legal = []
    for score, action in table:
        label = action_label(action)
        if action[0] == "calibrate":
            notes.append({"action": label, "reason": "calibrate_not_an_alternative", "score": score})
            continue
        if conflict_buoy is not None and action == ("service", int(conflict_buoy)):
            notes.append({"action": label, "reason": "conflict_same_buoy", "score": score})
            continue
        allowed = action_allowed(world.uavs[uav_index], action, theta, spec, factor, world)
        if not allowed:
            notes.append({"action": label, "reason": "admission_rejected", "score": score})
            continue
        legal.append((score, action))
        notes.append({"action": label, "reason": "legal", "score": score})
    if not legal:
        return None, None, notes
    score, action = pick_action(legal, tau)
    return score, action, notes


def identity_terms(q_c_plus: float, q_c_minus: float, q_e_plus: float, q_e_minus: float) -> dict:
    state_cost = q_c_minus - q_e_minus
    info_c = q_c_plus - q_c_minus
    info_e = q_e_plus - q_e_minus
    gap = q_c_plus - q_e_plus
    rebuilt = state_cost + info_c - info_e
    return {
        "state_cost": state_cost,
        "info_premium_c": info_c,
        "info_premium_e": info_e,
        "info_gap": info_c - info_e,
        "gap_plus": gap,
        "rebuilt": rebuilt,
        "residual": gap - rebuilt,
    }


def judge_sign(gap: float, tie: float, residual: float, fine_gap: float | None) -> str:
    """The frozen tie is a tie. A gap inside the known 15-node residual needs the 31×15 check."""
    if abs(gap) <= tie:
        return "unresolved"
    if abs(gap) > residual:
        return "resolved"
    if fine_gap is None or abs(fine_gap) <= tie or (fine_gap > 0) != (gap > 0):
        return "unresolved"
    return "resolved"


def attribute(terms: dict, terminal_minus_gap: float, tie: float, sign_status: str) -> str:
    if sign_status != "resolved":
        return "sign_unresolved"
    if abs(terms["info_gap"]) <= tie and abs(terminal_minus_gap) <= tie:
        return "immediate_motion_cost"
    if abs(terms["info_gap"]) <= tie:
        return "state_cost_without_update"
    if abs(terms["state_cost"]) <= tie:
        return "information_premium"
    return "mixed"


def needs_continuation(labels: list[str]) -> bool:
    return any(label in CONTINUATION_TRIGGERS for label in labels)


def split_action(world, beliefs, spec, factor, uav_index, action, stats) -> dict:
    """Q with the interval assimilated and Q with that focus update removed.

    Both values reuse one physical roll per theta. The waiting UAV's energy is
    inside R30 for every action and cancels in a difference.
    """
    nodes = int(spec["belief"]["quadrature_nodes"])
    inner = int(spec["integral"]["segment_inner_nodes"])
    grid = quadrature(beliefs[uav_index], nodes)
    mean = beliefs[uav_index].mean()
    r30_parts = []
    plus_parts = []
    minus_parts = []
    propulsion_parts = []
    aux_parts = []
    radio_parts = []
    executed = {}
    representative = None
    representative_theta = None
    for theta, weight in grid:
        roll = roll_action(world, beliefs, action, uav_index, theta, spec, factor, stats)
        future_plus, future_minus = expectation_pair(roll, spec, inner, stats)
        r30_parts.append(weight * roll.immediate)
        plus_parts.append(weight * future_plus)
        minus_parts.append(weight * future_minus)
        propulsion_parts.append(weight * float(theta) * roll.z_j)
        aux_parts.append(weight * roll.aux_j)
        radio_parts.append(weight * roll.radio_j)
        label = action_label(roll.executed)
        executed[label] = executed.get(label, 0) + 1
        if representative is None or abs(float(theta) - mean) < abs(float(representative_theta) - mean):
            representative = roll
            representative_theta = float(theta)
    r30 = math.fsum(r30_parts)
    terminal_plus = math.fsum(plus_parts)
    terminal_minus = math.fsum(minus_parts)
    aux = math.fsum(aux_parts)
    radio = math.fsum(radio_parts)
    detail = plan_snapshot(representative, spec)
    return {
        "action": action_label(action),
        "q_plus": r30 + terminal_plus,
        "q_minus": r30 + terminal_minus,
        "r30": r30,
        "terminal_plus": terminal_plus,
        "terminal_minus": terminal_minus,
        "info_premium": terminal_plus - terminal_minus,
        "propulsion_j": math.fsum(propulsion_parts),
        "aux_j": aux,
        "radio_j": radio,
        "compute_j": aux - radio,
        "representative_theta": representative_theta,
        "executed_counts": executed,
        "detail": detail,
    }


def motion_notes(calibration: dict, alternative: dict, tie: float) -> list[str]:
    notes = []
    if calibration["propulsion_j"] + 1.0 < alternative["propulsion_j"] and calibration["r30"] >= alternative["r30"] - tie:
        notes.append("calibration_propulsion_below_alternative")
    end_gap = math.hypot(calibration["detail"]["x"] - alternative["detail"]["x"], calibration["detail"]["y"] - alternative["detail"]["y"])
    if end_gap > 1.0:
        notes.append("end_position_differs")
    if action_label(calibration["detail"]["executed"]) != "calibrate:None":
        notes.append("calibration_rollout_replaced")
    if abs(calibration["terminal_minus"] - alternative["terminal_minus"]) <= tie and abs(calibration["r30"] - alternative["r30"]) > tie:
        notes.append("terminal_without_update_agrees_gap_is_immediate")
    return notes


def decompose_snapshot(world, beliefs, spec, factor, stats, stored_actions) -> dict:
    """Replay the frozen planner on one saved pre-decision state."""
    actions, tables = select_lookahead("lookahead_calibration_on", world, beliefs, spec, factor, stats)
    tau = tie_abs(spec)
    chosen = [pick_action(table, tau) for table in tables]
    repaired = _repair(chosen, tables, tau)
    stored = [tuple(action) for action in stored_actions]
    repaired_tuples = [tuple(action) for action in repaired]
    same_service = (
        len(chosen) == 2
        and chosen[0][1][0] == "service"
        and chosen[0][1] == chosen[1][1]
    )
    weaker = None
    if same_service:
        if abs(chosen[0][0] - chosen[1][0]) <= tau + 1e-15:
            weaker = 1
        else:
            weaker = 0 if chosen[0][0] < chosen[1][0] else 1
    focuses = [index for index, action in enumerate(repaired_tuples) if action[0] == "calibrate"]
    return {
        "stored_match": repaired_tuples == stored,
        "table_choice": [action_label(item[1]) for item in chosen],
        "table_score": [item[0] for item in chosen],
        "joint_action": [action_label(action) for action in repaired_tuples],
        "stored_action": [action_label(action) for action in stored],
        "repair_applied": [action_label(item[1]) for item in chosen] != [action_label(action) for action in repaired_tuples],
        "weaker_uav": weaker,
        "focus_uavs": focuses,
        "tables": tables,
        "actions": repaired_tuples,
        "chosen": chosen,
    }
