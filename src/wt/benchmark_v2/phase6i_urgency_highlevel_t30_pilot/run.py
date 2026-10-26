"""Run the twelve-scenario t=30 urgency label pilot."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import limit_threads, parameter_token
from wt.benchmark_v2.phase6ib_highlevel_dataset.jobs import _frozen_actor
from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import EPS_BITS, GENERATOR, HOLD, LABEL_NAME, SELECTED_K
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.analyze import macro_rank_summary, state_rank_summary
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.candidates import load_candidates
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.protocol import (
    ADAPTER_CONFIG_SHA256,
    EPOCH6_SHA256,
    FROZEN_SHARD_COMMIT,
    MAPPO_SHA256,
    PAYLOAD_SCALE,
    SAMPLE_K,
    SCHEME_C_SHA256,
    STRATA,
    TOP_K,
)
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.replay import replay_scenario
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.run import (
    _public_state,
    _same_labels,
    _score_bundle,
    _write_csv,
    checkpoint_hashes,
    hashes_match,
)
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.sampling import select_candidates
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.shards import dataset_fingerprint

from .protocol import (
    CANDIDATES_PER_STATE,
    D0_OUTPUT,
    EXPECTED_SCENARIOS,
    GATE_BLOCKED,
    INSTRUCTION_COMMIT,
    LAMBDA_H,
    MATERIAL_NEGATIVE_REGRET_BITS,
    MAX_ESTIMATED_SECONDS,
    MAX_ROLLOUTS,
    MAX_STATE_SECONDS,
    MIN_STATES_ABOVE,
    OUTPUT,
    PILOT_SEED_T30,
    REVERSAL_ABOVE,
    SPEARMAN_BELOW,
    TIME_STEP,
    choose_gate,
    recommendation,
    t30_scenarios,
)


def _result_fingerprint(folder: Path) -> list[dict]:
    import hashlib

    rows = []
    if not folder.is_dir():
        return [{"missing": str(folder)}]
    root = folder.parents[1]
    for path in sorted(item for item in folder.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return rows


def _top1_regret(rows: list[dict]) -> dict:
    best_payload = max(float(row["payload_bits"]) for row in rows)
    urgent = max(rows, key=lambda row: (float(row["urgent_service_bits"]), float(row["payload_bits"]), -int(row["candidate_id"])))
    payload_top = max(rows, key=lambda row: (float(row["payload_bits"]), -int(row["candidate_id"])))
    regret = float(urgent["payload_bits"]) - float(best_payload)
    return {
        "payload_top_candidate_id": int(payload_top["candidate_id"]),
        "urgent_top_candidate_id": int(urgent["candidate_id"]),
        "top1_changed": int(payload_top["candidate_id"]) != int(urgent["candidate_id"]),
        "top1_payload_regret_bits": float(regret),
        "material_payload_gap": bool(abs(regret) > float(EPS_BITS)),
    }


def execute() -> dict:
    started = time.perf_counter()
    limit_threads()
    scenarios = t30_scenarios()
    if tuple(scenarios) != EXPECTED_SCENARIOS:
        raise RuntimeError("t=30 scenario tuple left the registry continuation")
    dataset_before = dataset_fingerprint(scenarios)
    d0_before = _result_fingerprint(D0_OUTPUT)
    hashes_before = checkpoint_hashes()
    reasons: list[str] = []
    replay_rows: list[dict] = []
    bundles: list[dict] = []
    labels: list[dict] = []
    mechanical = {
        "checkpoint": hashes_match(hashes_before),
        "replay": False,
        "ledger": False,
        "shards": False,
        "legal": False,
        "payload": False,
        "determinism": False,
        "dataset_unchanged": False,
        "d0_unchanged": False,
        "actor_unchanged": False,
    }
    budget_ok = False
    timing = {"first_state_seconds": None, "estimated_total_seconds": None, "label_seconds": 0.0}
    rank_states: list[dict] = []
    determinism_rollouts = 0
    determinism_gap = None
    actor_token = None
    if not mechanical["checkpoint"]:
        reasons.append("checkpoint or adapter config hash does not match the frozen value")
    else:
        try:
            actor = _frozen_actor()
            actor_token = parameter_token(actor)
            for scenario in scenarios:
                print(f"replay {scenario}", flush=True)
                replay_rows.append(replay_scenario(scenario))
            mechanical["replay"] = all(
                bool(state["state_hash_match"]) and bool(state["checksum_state_match"])
                for row in replay_rows
                for state in row["states"]
            )
            mechanical["ledger"] = all(
                float(row["ledger_gap_bits"]) <= float(EPS_BITS)
                and all(bool(state["ledger_restore_independent"]) for state in row["states"])
                for row in replay_rows
            )
            if not mechanical["replay"]:
                reasons.append("replayed state does not match the frozen state shard")
            if not mechanical["ledger"]:
                reasons.append("admission-age ledger failed conservation or restore")
            if mechanical["replay"] and mechanical["ledger"]:
                for row in replay_rows:
                    state = next(item for item in row["states"] if int(item["time_step"]) == int(TIME_STEP))
                    print(f"candidates {state['scenario_id']} t={state['time_step']}", flush=True)
                    table = load_candidates(
                        state["scenario_id"],
                        state["time_step"],
                        state["env_snap"],
                        {
                            "checksum_candidates": state["manifest_checksum_candidates"],
                            "candidate_count": state["manifest_candidate_count"],
                            "illegal": state["manifest_illegal"],
                        },
                    )
                    chosen = select_candidates(
                        table["candidate_id"],
                        table["j15_bits"],
                        table["utility_rank"],
                        scenario_id=int(state["scenario_id"]),
                        time_step=int(TIME_STEP),
                        seed=int(PILOT_SEED_T30),
                    )
                    repeated = select_candidates(
                        table["candidate_id"],
                        table["j15_bits"],
                        table["utility_rank"],
                        scenario_id=int(state["scenario_id"]),
                        time_step=int(TIME_STEP),
                        seed=int(PILOT_SEED_T30),
                    )
                    if [int(item["candidate_id"]) for item in chosen["rows"]] != [int(item["candidate_id"]) for item in repeated["rows"]]:
                        raise RuntimeError("candidate sample changed when seed 602 was repeated")
                    rows = []
                    for item in chosen["rows"]:
                        position = int(item["row_index"])
                        copied = dict(item)
                        copied["scenario_id"] = int(state["scenario_id"])
                        copied["time_step"] = int(TIME_STEP)
                        copied["assignment"] = [int(value) for value in table["assignment"][position]]
                        copied["tie_group_id"] = int(table["tie_group_id"][position])
                        rows.append(copied)
                    if len(rows) != int(CANDIDATES_PER_STATE):
                        raise RuntimeError(f"candidate sample size is {len(rows)} at scenario {state['scenario_id']}")
                    bundles.append({"state": state, "table": table, "rows": rows, "sample": chosen})
                mechanical["shards"] = True
                mechanical["legal"] = all(bool(bundle["table"]["pool_order_match"]) and int(bundle["table"]["illegal_count"]) == 0 for bundle in bundles)
                _write_selected(bundles)
                if not mechanical["legal"]:
                    reasons.append("candidate legality or pool order failed")
                else:
                    print(f"score state 1/{len(bundles)}", flush=True)
                    first_rows, first_seconds, problem = _score_bundle(bundles[0], actor)
                    labels.extend(first_rows)
                    timing["first_state_seconds"] = float(first_seconds)
                    timing["estimated_total_seconds"] = float(first_seconds * len(bundles))
                    timing["label_seconds"] = float(first_seconds)
                    if problem:
                        raise RuntimeError(problem)
                    if first_seconds > float(MAX_STATE_SECONDS) or timing["estimated_total_seconds"] > float(MAX_ESTIMATED_SECONDS):
                        reasons.append(
                            f"budget stop: one state {first_seconds:.1f}s, twelve-state estimate {timing['estimated_total_seconds']:.1f}s"
                        )
                    else:
                        budget_ok = True
                        for index, bundle in enumerate(bundles[1:], start=2):
                            print(f"score state {index}/{len(bundles)}", flush=True)
                            scored, elapsed, problem = _score_bundle(bundle, actor)
                            labels.extend(scored)
                            timing["label_seconds"] = float(timing["label_seconds"] + elapsed)
                            if problem:
                                raise RuntimeError(problem)
                        print("repeat first state", flush=True)
                        repeated_rows, _elapsed, problem = _score_bundle(bundles[0], actor)
                        if problem:
                            raise RuntimeError(problem)
                        determinism_rollouts = int(len(repeated_rows))
                        determinism_gap = _same_labels(first_rows, repeated_rows)
                        mechanical["determinism"] = bool(determinism_gap == 0.0)
                        mechanical["payload"] = True
                        if not mechanical["determinism"]:
                            reasons.append("repeating the first state changed payload or urgent-service labels")
        except Exception as exc:
            reasons.append(str(exc))
            mechanical["payload"] = False
    if labels and bundles and mechanical["payload"]:
        grouped: dict[int, list[dict]] = {}
        for row in labels:
            grouped.setdefault(int(row["scenario_id"]), []).append(row)
        for bundle in bundles:
            scenario = int(bundle["state"]["scenario_id"])
            rows = grouped.get(scenario, [])
            if len(rows) != len(bundle["rows"]):
                continue
            summary = state_rank_summary(rows)
            summary.update(_top1_regret(rows))
            summary["scenario_id"] = scenario
            summary["time_step"] = int(TIME_STEP)
            summary["oldest_age_start"] = rows[0]["oldest_age_start"]
            rank_states.append(summary)
    dataset_after = dataset_fingerprint(scenarios)
    d0_after = _result_fingerprint(D0_OUTPUT)
    hashes_after = checkpoint_hashes()
    mechanical["dataset_unchanged"] = dataset_before == dataset_after
    mechanical["d0_unchanged"] = d0_before == d0_after
    mechanical["actor_unchanged"] = actor_token is not None and parameter_token(_frozen_actor()) == actor_token
    if hashes_before != hashes_after:
        mechanical["checkpoint"] = False
        reasons.append("checkpoint hash changed during the pilot")
    if not mechanical["dataset_unchanged"]:
        reasons.append("working-tree dataset files changed during the pilot")
    if not mechanical["d0_unchanged"]:
        reasons.append("stage D0 result files changed during the pilot")
    if not mechanical["actor_unchanged"]:
        reasons.append("frozen MAPPO parameters changed during the pilot")
    coverage_complete = len(rank_states) == len(scenarios) and len(scenarios) == 12
    mechanical_ok = all(bool(value) for value in mechanical.values()) and coverage_complete
    if not coverage_complete and "rank coverage is short of 12 t=30 states" not in reasons and mechanical["payload"]:
        reasons.append("rank coverage is short of 12 t=30 states")
    macro = macro_rank_summary(rank_states) if rank_states else {
        "states": 0,
        "mean_spearman": None,
        "min_spearman": None,
        "mean_reversal_rate": None,
        "max_reversal_rate": None,
        "states_with_top1_change": None,
        "lambda_H": LAMBDA_H,
    }
    states_above = int(sum(float(row["reversal_rate"]) > float(REVERSAL_ABOVE) for row in rank_states))
    regrets = [float(row["top1_payload_regret_bits"]) for row in rank_states]
    mean_regret = None if not regrets else float(np.mean(regrets))
    min_regret = None if not regrets else float(np.min(regrets))
    gate = choose_gate(
        mechanical_ok=mechanical_ok,
        budget_ok=bool(budget_ok),
        coverage_complete=bool(coverage_complete and mechanical_ok),
        mean_spearman=macro.get("mean_spearman"),
        mean_reversal=macro.get("mean_reversal_rate"),
        states_above=states_above if coverage_complete else 0,
    )
    if not mechanical_ok or not budget_ok:
        gate = GATE_BLOCKED
    return {
        "gate": gate,
        "recommendation": recommendation(gate, mean_regret if coverage_complete else None),
        "reasons": reasons,
        "lambda_H": LAMBDA_H,
        "scenarios": [int(item) for item in scenarios],
        "time_step": int(TIME_STEP),
        "rollout_count": int(len(labels)),
        "max_rollouts": int(MAX_ROLLOUTS),
        "determinism_rollouts": int(determinism_rollouts),
        "determinism_abs_gap_bits": determinism_gap,
        "candidate_cap_per_state": int(CANDIDATES_PER_STATE),
        "timing": timing,
        "mechanical": mechanical,
        "budget_ok": bool(budget_ok),
        "coverage_complete": bool(coverage_complete),
        "states_with_reversal_above": states_above,
        "mean_top1_payload_regret_bits": mean_regret,
        "min_top1_payload_regret_bits": min_regret,
        "material_negative_regret_bits": float(MATERIAL_NEGATIVE_REGRET_BITS),
        "replay": [
            {
                **{key: value for key, value in row.items() if key != "states"},
                "states": [_public_state(state) for state in row["states"]],
            }
            for row in replay_rows
        ],
        "selection": [
            {
                "scenario_id": int(bundle["state"]["scenario_id"]),
                "time_step": int(TIME_STEP),
                "count": int(len(bundle["rows"])),
                "top_count": int(bundle["sample"]["top_count"]),
                "stratum_count": int(bundle["sample"]["stratum_count"]),
                "fill_count": int(bundle["sample"]["fill_count"]),
                "shortfall_unfilled": int(bundle["sample"]["shortfall_unfilled"]),
                "pool_hash": bundle["table"]["pool_hash"],
                "candidate_count": int(bundle["table"]["candidate_count"]),
                "seed": int(bundle["sample"]["seed"]),
            }
            for bundle in bundles
        ],
        "rank_by_state": rank_states,
        "rank_macro": macro,
        "payload_gap_max_bits": None if not labels else float(max(abs(float(row["payload_gap_bits"])) for row in labels)),
        "ledger_gap_max_bits": None if not labels else float(max(float(row["ledger_gap_bits"]) for row in labels)),
        "accounting_error_max_bits": None if not labels else float(max(float(row["accounting_error_bits"]) for row in labels)),
        "checkpoint_hashes": hashes_after,
        "frozen_shard_commit": FROZEN_SHARD_COMMIT,
        "instruction_commit": INSTRUCTION_COMMIT,
        "pilot_seed": int(PILOT_SEED_T30),
        "dataset_files": dataset_after,
        "seconds_total": float(time.perf_counter() - started),
        "labels": labels,
        "selected_rows": [row for bundle in bundles for row in bundle["rows"]],
    }


def _write_selected(bundles: list[dict]) -> None:
    rows = [row for bundle in bundles for row in bundle["rows"]]
    _write_csv(
        OUTPUT / "selected_candidates.csv",
        rows,
        ["scenario_id", "time_step", "candidate_id", "assignment", "j15_bits", "utility_rank", "tie_group_id", "selection", "stratum", "filled"],
    )


def write_results(summary: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected_fields = ["scenario_id", "time_step", "candidate_id", "assignment", "j15_bits", "utility_rank", "tie_group_id", "selection", "stratum", "filled"]
    label_fields = [
        "scenario_id",
        "time_step",
        "candidate_id",
        "payload_bits",
        "payload_term",
        "urgent_service_bits",
        "urgent_service_term",
        "payload_gap_bits",
        "ledger_gap_bits",
        "accounting_error_bits",
        "served_bits",
        "oldest_age_start",
        "generated_bits",
        "seconds",
    ]
    _write_csv(OUTPUT / "selected_candidates.csv", summary["selected_rows"], selected_fields)
    _write_csv(OUTPUT / "labels.csv", summary["labels"], label_fields)
    public = {key: value for key, value in summary.items() if key not in {"labels", "selected_rows"}}
    (OUTPUT / "summary.json").write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")
    schema = {
        "split": "train",
        "scenarios": summary["scenarios"],
        "time_step": int(TIME_STEP),
        "generator": GENERATOR,
        "k": int(SELECTED_K),
        "hold": int(HOLD),
        "label": LABEL_NAME,
        "pilot_seed": int(PILOT_SEED_T30),
        "seed_sequence": "[602, scenario_id, 30]",
        "top_k": int(TOP_K),
        "sample_k": int(SAMPLE_K),
        "strata": int(STRATA),
        "sampling": "32 stable top existing J15 labels, then 96 equal-count utility_rank quantiles; saved before urgent rollouts",
        "formulas": {
            "payload_term": "payload_increment_bits / 1e6",
            "urgent_service_term": "sum_t,k(urgency_k_before_service[t] * served_bits[k,t]) / 1e6",
            "utility": "payload_term + lambda_H * urgent_service_term",
            "top1_payload_regret_bits": "payload_bits of urgent-service top minus payload_bits of payload top",
        },
        "lambda_H": LAMBDA_H,
        "lambda_H_decided": False,
        "signal_thresholds": {
            "spearman_below": SPEARMAN_BELOW,
            "reversal_above": REVERSAL_ABOVE,
            "min_states_above": MIN_STATES_ABOVE,
            "material_negative_regret_bits": MATERIAL_NEGATIVE_REGRET_BITS,
            "payload_scale": PAYLOAD_SCALE,
            "max_state_seconds": MAX_STATE_SECONDS,
            "max_estimated_seconds": MAX_ESTIMATED_SECONDS,
        },
        "arrival_convention": "candidate label rollout zeros arrivals and arrivals_at on the scratch copy only",
        "state_source": "MAX_RATE_INDEPENDENT held for 15 steps and executed by frozen MAPPO-V2 seed 0",
        "ledger": "episode_admission_age_v1",
        "frozen_shard_commit": FROZEN_SHARD_COMMIT,
        "instruction_commit": INSTRUCTION_COMMIT,
        "checkpoint_sha256": summary["checkpoint_hashes"],
        "expected_checkpoint_sha256": {
            "mappo_v2_seed0": MAPPO_SHA256,
            "scheme_c": SCHEME_C_SHA256,
            "epoch6": EPOCH6_SHA256,
            "adapter_config": ADAPTER_CONFIG_SHA256,
        },
        "fields": {"selected_candidates.csv": selected_fields, "labels.csv": label_fields},
    }
    (OUTPUT / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> dict:
    summary = execute()
    write_results(summary)
    print(summary["gate"], summary["rollout_count"], summary["timing"], summary["mean_top1_payload_regret_bits"], flush=True)
    return summary


if __name__ == "__main__":
    main()
