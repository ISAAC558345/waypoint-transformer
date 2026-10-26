"""Write the Stage D3 pilot tables. Formal checkpoints are only read."""
from __future__ import annotations

import csv
import hashlib
import json
import traceback

from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import limit_threads
from wt.benchmark_v2.phase6ib_highlevel_dataset.jobs import _frozen_actor
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.protocol import OUTPUT as D0_OUTPUT
from wt.benchmark_v2.phase6i_urgency_highlevel_t30_pilot.protocol import OUTPUT as D1_OUTPUT
from wt.benchmark_v2.phase6i_urgency_utility_rule.protocol import OUTPUT as D2_OUTPUT

from .episode import BudgetStop, LabelClock, run_arm
from .protocol import (
    ADAPTER_CONFIG_SHA256,
    ARMS,
    DECISION_TIMES,
    DELTA_PAYLOAD_MAX,
    EPOCH6_CKPT,
    EPOCH6_SHA256,
    EPS_BITS,
    FROZEN_SHARD_COMMIT,
    GATE_BLOCKED,
    GENERATOR,
    INSTRUCTION_COMMIT,
    LAMBDA_H,
    MAPPO_SHA256,
    MAX_CANDIDATES,
    MAX_ESTIMATED_SECONDS,
    MAX_STATE_SECONDS,
    ORACLE_LABEL,
    OUTPUT,
    PILOT_SEED,
    PLANNED_LABEL_STATES,
    SCHEME_C_CKPT,
    SCHEME_C_SHA256,
    SELECTED_K,
    choose_gate,
    config_sha256,
    label_horizon,
    primary_checkpoint,
    scenarios,
)


SELECTED_FIELDS = [
    "scenario_id",
    "arm",
    "time_step",
    "candidate_id",
    "assignment",
    "sampling_score",
    "sampling_source",
    "utility_rank",
    "selection",
    "stratum",
    "filled",
    "pool_hash",
    "pool_count",
    "subset_hash",
    "label_horizon",
]
DECISION_FIELDS = [
    "scenario_id",
    "arm",
    "time_step",
    "label_horizon",
    "pool_count",
    "pool_hash",
    "subset_hash",
    "subset_count",
    "sampling_source",
    "state_hash",
    "payload_tie_count",
    "payload_first_candidate_id",
    "tiebreak_candidate_id",
    "payload_first_assignment",
    "tiebreak_assignment",
    "assignment_changed",
    "payload_first_payload_bits",
    "tiebreak_payload_bits",
    "payload_gap_bits",
    "payload_gap_within_one_bit",
    "payload_first_urgent_bits",
    "tiebreak_urgent_bits",
    "urgent_gain_bits",
    "executed_candidate_id",
    "executed_assignment",
    "executed_payload_regret_bits",
    "selection_reason",
    "label_reused",
    "max_label_ledger_gap_bits",
    "max_label_accounting_error_bits",
    "label_seconds",
]
EPISODE_FIELDS = [
    "scenario_id",
    "arm",
    "status",
    "final_t",
    "payload_bits",
    "served_bits_sum",
    "payload_accounting_gap_bits",
    "urgent_service_bits",
    "dropped_bits",
    "final_queue_bits",
    "final_queue_vector",
    "oldest_wait_max",
    "oldest_wait_mean",
    "mean_wait_max",
    "mean_wait_mean",
    "path_length_m",
    "mean_sinr",
    "mean_total_interference",
    "max_sinr_recompute_gap",
    "projector_trigger_steps",
    "projector_trigger_rate",
    "mean_projection_norm",
    "safety_violation_steps",
    "first_service_step",
    "positive_service_steps",
    "max_ledger_gap_bits",
    "max_accounting_error_bits",
    "seconds",
    "tiebreak_changed_decisions",
    "max_abs_choice_payload_gap_bits",
]
STEP_FIELDS = [
    "scenario_id",
    "arm",
    "time_step",
    "assignment",
    "served_bits",
    "urgency",
    "oldest_wait_age",
    "q_bits_before",
    "q_bits_after",
    "dropped_bits",
    "payload_increment_bits",
    "sinr",
    "total_interference",
    "projection_norm",
    "constraint_status",
    "ledger_gap_bits",
    "accounting_error_bits",
]


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _watched() -> dict:
    paths = {
        "d0_labels": D0_OUTPUT / "labels.csv",
        "d1_labels": D1_OUTPUT / "labels.csv",
        "d2_summary": D2_OUTPUT / "summary.json",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _csv(value):
    if isinstance(value, float):
        return f"{value:.16g}"
    return value


def _write_csv(path, rows, fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv(row.get(key)) for key in fields})
        handle.flush()


def _mean(values: list[float]) -> float:
    if not values:
        raise RuntimeError("mean of an empty sample")
    return float(sum(values) / len(values))


def _median(values: list[float]) -> float:
    ordered = sorted(float(item) for item in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float(0.5 * (ordered[mid - 1] + ordered[mid]))


def _pairs(episodes: list[dict]) -> list[dict]:
    rows = []
    for scenario in scenarios():
        base = next(item for item in episodes if int(item["scenario_id"]) == int(scenario) and item["arm"] == "payload_first")
        other = next(item for item in episodes if int(item["scenario_id"]) == int(scenario) and item["arm"] == "payload_first_urgency_tiebreak")
        rows.append(
            {
                "scenario_id": int(scenario),
                "payload_delta_bits": float(other["payload_bits"]) - float(base["payload_bits"]),
                "urgent_delta_bits": float(other["urgent_service_bits"]) - float(base["urgent_service_bits"]),
                "dropped_delta_bits": float(other["dropped_bits"]) - float(base["dropped_bits"]),
                "final_queue_delta_bits": float(other["final_queue_bits"]) - float(base["final_queue_bits"]),
                "oldest_wait_max_delta": float(other["oldest_wait_max"]) - float(base["oldest_wait_max"]),
                "path_length_delta_m": float(other["path_length_m"]) - float(base["path_length_m"]),
                "mean_sinr_delta": float(other["mean_sinr"]) - float(base["mean_sinr"]),
                "mean_interference_delta": float(other["mean_total_interference"]) - float(base["mean_total_interference"]),
                "projector_trigger_delta": int(other["projector_trigger_steps"]) - int(base["projector_trigger_steps"]),
                "safety_violation_delta": int(other["safety_violation_steps"]) - int(base["safety_violation_steps"]),
            }
        )
    return rows


def _direction(pairs: list[dict]) -> str:
    urgent = [float(row["urgent_delta_bits"]) for row in pairs]
    if all(abs(value) <= float(EPS_BITS) for value in urgent):
        return "flat"
    if all(value >= -float(EPS_BITS) for value in urgent) and any(value > float(EPS_BITS) for value in urgent):
        return "consistent_nonnegative"
    return "mixed"


def _mechanical(episodes: list[dict], decisions: list[dict]) -> tuple[bool, list[str]]:
    reasons = []
    if len(episodes) != 8 or any(int(item["final_t"]) != 48 or item["status"] != "ok" for item in episodes):
        reasons.append("one arm did not finish a clean 48-step episode")
    if any(int(item["payload_gap_within_one_bit"]) != 1 for item in decisions):
        reasons.append("a tie-break payload gap exceeded 1 bit")
    if any(abs(float(item["executed_payload_regret_bits"])) > float(EPS_BITS) for item in decisions):
        reasons.append("an executed candidate left the 1-bit payload group")
    if any(float(item["max_label_ledger_gap_bits"]) > float(EPS_BITS) or float(item["max_label_accounting_error_bits"]) > float(EPS_BITS) for item in decisions):
        reasons.append("a label ledger or payload account exceeded 1 bit")
    if any(float(item["max_ledger_gap_bits"]) > float(EPS_BITS) or float(item["max_accounting_error_bits"]) > float(EPS_BITS) for item in episodes):
        reasons.append("an episode ledger or payload account exceeded 1 bit")
    if any(int(item["subset_count"]) != int(MAX_CANDIDATES) for item in decisions):
        reasons.append("a decision did not keep 128 candidates")
    return not reasons, reasons


def _checkpoint_hashes() -> dict:
    found = {
        "mappo_v2_seed0": _sha256(primary_checkpoint()),
        "scheme_c": _sha256(SCHEME_C_CKPT),
        "epoch6": _sha256(EPOCH6_CKPT),
        "adapter_config": config_sha256(),
    }
    expected = {
        "mappo_v2_seed0": MAPPO_SHA256,
        "scheme_c": SCHEME_C_SHA256,
        "epoch6": EPOCH6_SHA256,
        "adapter_config": ADAPTER_CONFIG_SHA256,
    }
    if found != expected:
        raise RuntimeError("frozen checkpoint or adapter config hash changed")
    return found


def _recommend(pairs: list[dict], tiebreak_fraction: float, mechanical_ok: bool) -> dict:
    payload = [float(row["payload_delta_bits"]) for row in pairs]
    urgent = [float(row["urgent_delta_bits"]) for row in pairs]
    losses = sum(value < -float(EPS_BITS) for value in payload)
    gains = sum(value > float(EPS_BITS) for value in urgent)
    worth = bool(mechanical_ok and tiebreak_fraction >= 0.5 and gains >= 3 and losses == 0 and all(value >= -float(EPS_BITS) for value in urgent))
    return {
        "worth_larger_dev_eval": worth,
        "oracle_label_only": True,
        "learned_policy": False,
        "transfers_to_scheme_c": False,
        "closed_loop_payload_loss_scenarios": int(losses),
        "closed_loop_urgent_gain_scenarios": int(gains),
    }


def execute() -> dict:
    limit_threads()
    before = _watched()
    hashes = _checkpoint_hashes()
    actor = _frozen_actor()
    clock = LabelClock(PLANNED_LABEL_STATES, MAX_STATE_SECONDS, MAX_ESTIMATED_SECONDS)
    cache = {}
    episodes = []
    sink = {"selected": [], "flush": lambda: _write_csv(OUTPUT / "selected_candidates.csv", sink["selected"], SELECTED_FIELDS)}
    budget_ok = True
    failure = ""
    try:
        for scenario in scenarios():
            for arm in ARMS:
                print(f"arm {arm} scenario {scenario}", flush=True)
                episodes.append(run_arm(int(scenario), arm, actor, clock, cache, sink))
    except BudgetStop as exc:
        budget_ok = False
        failure = str(exc)
    except Exception as exc:
        budget_ok = False
        failure = f"{type(exc).__name__}: {exc}"
        failure_trace = traceback.format_exc()
    else:
        failure_trace = ""
    if "failure_trace" not in locals():
        failure_trace = ""
    decisions = [row for episode in episodes for row in episode["decisions"]]
    steps = [row for episode in episodes for row in episode["steps"]]
    pairs = _pairs(episodes) if len(episodes) == 8 else []
    mechanical_ok, reasons = _mechanical(episodes, decisions) if episodes else (False, ["no finished episode"])
    if failure:
        reasons.append(failure)
        mechanical_ok = False
    tie_rows = [row for row in decisions if row["arm"] == "payload_first_urgency_tiebreak"]
    tie_fraction = None if not tie_rows else float(sum(int(row["assignment_changed"]) for row in tie_rows) / len(tie_rows))
    direction = _direction(pairs) if pairs else "unavailable"
    after = _watched()
    if before != after:
        mechanical_ok = False
        reasons.append("D0, D1, or D2 result files changed during the pilot")
    gate = choose_gate(
        mechanical_ok=mechanical_ok and not failure,
        budget_ok=budget_ok and not failure,
        tiebreak_fraction=tie_fraction,
        urgent_direction=direction,
    )
    if failure:
        gate = GATE_BLOCKED
    payload_deltas = [float(row["payload_delta_bits"]) for row in pairs]
    urgent_deltas = [float(row["urgent_delta_bits"]) for row in pairs]
    recommendation = _recommend(pairs, 0.0 if tie_fraction is None else tie_fraction, mechanical_ok and budget_ok and not failure)
    return {
        "gate": gate,
        "reasons": reasons,
        "failure_trace": failure_trace,
        "oracle_label": ORACLE_LABEL,
        "learned_policy": False,
        "scheme_c_used_for_selection": False,
        "formal_system_modified": False,
        "trained_hgnn": False,
        "lambda_H": LAMBDA_H,
        "delta_payload_max": DELTA_PAYLOAD_MAX,
        "d2_gate_unchanged": "HIGHLEVEL_UTILITY_RULE_READY",
        "instruction_commit": INSTRUCTION_COMMIT,
        "frozen_shard_commit": FROZEN_SHARD_COMMIT,
        "pilot_seed": PILOT_SEED,
        "seed_sequence": "[603, scenario_id, time_step]",
        "scenarios": list(scenarios()),
        "decision_times": list(DECISION_TIMES),
        "label_horizons": {str(time_step): int(label_horizon(int(time_step))) for time_step in DECISION_TIMES},
        "candidates_per_state": int(MAX_CANDIDATES),
        "generator": GENERATOR,
        "selected_k": int(SELECTED_K),
        "planned_label_states": int(PLANNED_LABEL_STATES),
        "label_calls": int(clock.calls),
        "first_state_seconds": clock.first_seconds,
        "estimated_seconds": clock.estimate_seconds,
        "label_seconds": float(clock.label_seconds),
        "sampling_source_counts": {
            source: int(sum(row["sampling_source"] == source for row in decisions)) for source in ("stored_j15", "additive_g1_rate")
        },
        "tiebreak_assignment_change_count": int(sum(int(row["assignment_changed"]) for row in tie_rows)),
        "tiebreak_decision_count": int(len(tie_rows)),
        "tiebreak_fraction": tie_fraction,
        "urgent_direction": direction,
        "paired": pairs,
        "payload_delta_mean_bits": None if not payload_deltas else _mean(payload_deltas),
        "payload_delta_median_bits": None if not payload_deltas else _median(payload_deltas),
        "urgent_delta_mean_bits": None if not urgent_deltas else _mean(urgent_deltas),
        "urgent_delta_median_bits": None if not urgent_deltas else _median(urgent_deltas),
        "max_choice_payload_gap_bits": None if not decisions else float(max(abs(float(row["payload_gap_bits"])) for row in decisions)),
        "max_episode_ledger_gap_bits": None if not episodes else float(max(float(row["max_ledger_gap_bits"]) for row in episodes)),
        "max_episode_accounting_error_bits": None if not episodes else float(max(float(row["max_accounting_error_bits"]) for row in episodes)),
        "max_label_ledger_gap_bits": None if not decisions else float(max(float(row["max_label_ledger_gap_bits"]) for row in decisions)),
        "max_label_accounting_error_bits": None if not decisions else float(max(float(row["max_label_accounting_error_bits"]) for row in decisions)),
        "checkpoint_sha256": hashes,
        "watched_sha256": before,
        "watched_unchanged": before == after,
        "recommendation": recommendation,
        "episodes": [{key: value for key, value in episode.items() if key not in {"decisions", "steps"}} for episode in episodes],
        "selected_rows": sink["selected"],
        "decision_rows": decisions,
        "step_rows": steps,
    }


def write_results(summary: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT / "selected_candidates.csv", summary["selected_rows"], SELECTED_FIELDS)
    _write_csv(OUTPUT / "decision_rows.csv", summary["decision_rows"], DECISION_FIELDS)
    _write_csv(OUTPUT / "episode_rows.csv", summary["episodes"], EPISODE_FIELDS)
    _write_csv(OUTPUT / "step_rows.csv", summary["step_rows"], STEP_FIELDS)
    public = {key: value for key, value in summary.items() if key not in {"selected_rows", "decision_rows", "step_rows"}}
    (OUTPUT / "summary.json").write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")
    schema = {
        "oracle_label": True,
        "meaning": "Both rules read frozen-MAPPO rollout labels. This is not an HGNN policy and it does not replace Scheme C.",
        "arms": {
            "payload_first": "maximum label payload; smaller candidate id breaks a 1-bit tie; urgent service is ignored",
            "payload_first_urgency_tiebreak": "maximum urgent service among candidates within 1 bit of the maximum label payload; smaller candidate id breaks a 1-bit urgent tie",
        },
        "sampling": {
            "seed": PILOT_SEED,
            "seed_sequence": "[603, scenario_id, time_step]",
            "size": "32 stable top scores plus 96 utility-rank strata",
            "stored_j15": "used only when the live state hash matches the frozen train shard",
            "additive_g1_rate": "subset ranking for a diverged state or t=45, where no stored J15 exists; rule labels are still MAPPO rollouts",
            "saved_before_urgent_rollout": True,
        },
        "label_horizon": {"0": 15, "15": 15, "30": 15, "45": 3},
        "label_horizon_note": "3-step labels are not compared with 15-step labels",
        "eps_bits": float(EPS_BITS),
        "primary_metric": "48-step episode payload of tie-break minus payload-first",
        "lambda_H": LAMBDA_H,
        "delta_payload_max": DELTA_PAYLOAD_MAX,
        "held_out_splits_read": False,
        "generator": GENERATOR,
        "selected_k": int(SELECTED_K),
        "instruction_commit": INSTRUCTION_COMMIT,
        "fields": {
            "selected_candidates.csv": SELECTED_FIELDS,
            "decision_rows.csv": DECISION_FIELDS,
            "episode_rows.csv": EPISODE_FIELDS,
            "step_rows.csv": STEP_FIELDS,
        },
    }
    (OUTPUT / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> dict:
    summary = execute()
    write_results(summary)
    print(summary["gate"], summary["first_state_seconds"], summary["tiebreak_fraction"], summary["paired"], flush=True)
    if summary["failure_trace"]:
        print(summary["failure_trace"], flush=True)
    return summary


if __name__ == "__main__":
    main()
