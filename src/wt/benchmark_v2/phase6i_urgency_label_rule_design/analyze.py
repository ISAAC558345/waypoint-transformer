"""Apply Rule P and Rule C independently on each sampled state."""
from __future__ import annotations

from .protocol import EPS_BITS, FRACTIONS, PRIMARY_SOURCE, PRIMARY_TIMES
from .rules import eligible_count, pareto_front, payload_id_only, payload_loss, rule_c, rule_p, urgent_first
from .sources import StateBatch


def _ids(rows: list[dict]) -> str:
    return "|".join(str(int(row["candidate_id"])) for row in rows)


def _exchange(regret: float, gain: float) -> float | None:
    if float(regret) <= float(EPS_BITS) or float(gain) <= float(EPS_BITS):
        return None
    return float(regret) / float(gain)


def state_record(batch: StateBatch) -> dict:
    rows = batch.rows
    if len(rows) < 2:
        raise RuntimeError(f"{batch.source} {batch.scenario_id} t={batch.time_step} has fewer than two candidates")
    front = pareto_front(rows)
    best_payload = max(float(row["payload_bits"]) for row in rows)
    best_urgent = max(float(row["urgent_bits"]) for row in rows)
    span = float(best_payload - min(float(row["payload_bits"]) for row in rows))
    tied = [row for row in rows if payload_loss(row["payload_bits"], best_payload) == 0.0]
    identifier = payload_id_only(rows)
    chosen = rule_p(rows)
    urgent = urgent_first(rows)
    regret = payload_loss(urgent["payload_bits"], best_payload)
    gain_vs_id = float(chosen["urgent_bits"]) - float(identifier["urgent_bits"])
    gain_vs_rule = float(urgent["urgent_bits"]) - float(chosen["urgent_bits"])
    if gain_vs_id < -float(EPS_BITS):
        raise RuntimeError("Rule P urgent service is below the payload-id choice")
    if payload_loss(chosen["payload_bits"], best_payload) != 0.0:
        raise RuntimeError("Rule P left the 1-bit payload group")
    return {
        "source": batch.source,
        "scenario_id": int(batch.scenario_id),
        "time_step": int(batch.time_step),
        "horizon": int(batch.horizon),
        "horizon_source": batch.horizon_source,
        "label_family": batch.label_family,
        "arrival_semantics": batch.arrival_semantics,
        "state_path": batch.state_path,
        "candidate_count": int(len(rows)),
        "pool_count": "" if batch.pool_count is None else int(batch.pool_count),
        "pool_count_source": batch.pool_count_source,
        "state_hash": batch.state_hash,
        "pool_hash": batch.pool_hash,
        "subset_hash": batch.subset_hash,
        "pareto_count": int(len(front)),
        "pareto_candidate_ids": _ids(front),
        "payload_tie_count": int(len(tied)),
        "payload_tie_fraction": float(len(tied) / len(rows)),
        "payload_span_bits": float(span),
        "best_payload_bits": float(best_payload),
        "best_urgent_bits": float(best_urgent),
        "payload_id_candidate_id": int(identifier["candidate_id"]),
        "payload_id_assignment": identifier["assignment"],
        "rule_p_candidate_id": int(chosen["candidate_id"]),
        "rule_p_assignment": chosen["assignment"],
        "rule_p_payload_bits": float(chosen["payload_bits"]),
        "rule_p_urgent_bits": float(chosen["urgent_bits"]),
        "rule_p_payload_regret_bits": 0.0,
        "rule_p_urgent_gain_vs_payload_id_bits": float(gain_vs_id),
        "urgent_first_candidate_id": int(urgent["candidate_id"]),
        "urgent_first_assignment": urgent["assignment"],
        "urgent_first_payload_bits": float(urgent["payload_bits"]),
        "urgent_first_urgent_bits": float(urgent["urgent_bits"]),
        "urgent_first_payload_regret_bits": float(regret),
        "urgent_first_urgent_gain_vs_rule_p_bits": float(gain_vs_rule),
        "rule_p_matches_urgent_first": int(int(chosen["candidate_id"]) == int(urgent["candidate_id"])),
        "descriptive_exchange_ratio": "" if _exchange(regret, gain_vs_rule) is None else float(_exchange(regret, gain_vs_rule)),
        "field_gaps": "|".join(batch.field_gaps),
        "primary_rule_state": int(batch.primary_rule_state),
    }


def delta_for(span_bits: float, fraction: float) -> float:
    if float(span_bits) < 0.0:
        raise RuntimeError("payload span is negative")
    if float(fraction) < 0.0 or float(fraction) > 1.0:
        raise RuntimeError("payload span fraction is outside [0, 1]")
    return float(span_bits) * float(fraction)


def sensitivity_rows(batch: StateBatch) -> list[dict]:
    """Grid a symbolic fraction of this state's own payload span. No budget is selected."""
    if batch.label_family != "real_arrival":
        return []
    rows = batch.rows
    record = state_record(batch)
    front = pareto_front(rows)
    chosen_p = rule_p(rows)
    urgent = urgent_first(rows)
    span = float(record["payload_span_bits"])
    output = []
    for fraction in FRACTIONS:
        delta = delta_for(span, fraction)
        chosen = rule_c(rows, delta)
        output.append(
            {
                "source": batch.source,
                "scenario_id": int(batch.scenario_id),
                "time_step": int(batch.time_step),
                "horizon": int(batch.horizon),
                "label_family": batch.label_family,
                "fraction_of_payload_span": float(fraction),
                "delta_payload_max_bits": float(delta),
                "budget_selected": 0,
                "candidate_id": int(chosen["candidate_id"]),
                "assignment": chosen["assignment"],
                "payload_bits": float(chosen["payload_bits"]),
                "urgent_service_bits": float(chosen["urgent_bits"]),
                "payload_regret_bits": payload_loss(chosen["payload_bits"], record["best_payload_bits"]),
                "urgent_gain_vs_rule_p_bits": float(chosen["urgent_bits"]) - float(chosen_p["urgent_bits"]),
                "assignment_changed_vs_rule_p": int(int(chosen["candidate_id"]) != int(chosen_p["candidate_id"])),
                "matches_urgent_first": int(int(chosen["candidate_id"]) == int(urgent["candidate_id"])),
                "eligible_count": eligible_count(rows, delta),
                "pareto_count": int(len(front)),
                "primary_sensitivity": int(batch.source == PRIMARY_SOURCE and int(batch.time_step) in PRIMARY_TIMES),
            }
        )
    return output


def fields_support_rules(records: list[dict]) -> bool:
    primary = [row for row in records if int(row["primary_rule_state"]) == 1]
    if not primary:
        return False
    for row in primary:
        if int(row["candidate_count"]) < 2:
            return False
        if str(row["field_gaps"]):
            gaps = set(str(row["field_gaps"]).split("|"))
            blocking = {"real_payload_bits", "real_urgent_service_bits", "assignment"}
            if gaps & blocking:
                return False
        if row["label_family"] != "real_arrival":
            return False
        if int(row["horizon"]) != 15:
            return False
    return True
