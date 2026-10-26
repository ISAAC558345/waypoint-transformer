"""Run the four-scenario Stage D0 pilot and write its result files."""
from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import limit_threads, parameter_token
from wt.benchmark_v2.phase6ib_highlevel_dataset.jobs import _frozen_actor
from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import EPS_BITS, PRIMARY_TIMES

from .analyze import macro_rank_summary, state_rank_summary
from .candidates import load_candidates
from .labels import score_assignment
from .protocol import (
    ADAPTER_CONFIG_SHA256,
    EPOCH6_CKPT,
    EPOCH6_SHA256,
    FROZEN_SHARD_COMMIT,
    GATE_BLOCKED,
    GENERATOR,
    HOLD,
    INSTRUCTION_COMMIT,
    LABEL_NAME,
    LAMBDA_H,
    MAPPO_SHA256,
    MAX_CANDIDATES_PER_STATE,
    MAX_ROLLOUTS,
    MAX_STATE_SECONDS,
    OUTPUT,
    PAYLOAD_SCALE,
    PILOT_SEED,
    REPEAT_CANDIDATES_IF_SLOW,
    REPEAT_STATE_IF_UNDER_SECONDS,
    SAMPLE_K,
    SCHEME_C_CKPT,
    SCHEME_C_SHA256,
    SELECTED_K,
    STRATA,
    TOP_K,
    USEFUL_REVERSAL_ABOVE,
    USEFUL_SPEARMAN_BELOW,
    choose_gate,
    config_sha256,
    pilot_scenarios,
    primary_checkpoint,
)
from .replay import replay_scenario
from .sampling import select_candidates
from .shards import dataset_fingerprint


RECOMMENDATION = {
    "HIGHLEVEL_URGENCY_LABEL_PILOT_PASS": "值得构建完整高层紧急度数据集，但本阶段不构建、不训练 HGNN。",
    "HIGHLEVEL_URGENCY_LABEL_PILOT_INCONCLUSIVE": "payload 与紧急服务的排序差异尚未形成有用信号，先不构建完整高层紧急度数据集。",
    "HIGHLEVEL_URGENCY_LABEL_PILOT_BLOCKED": "状态、标签或预算校验未通过，先不构建完整高层紧急度数据集。",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_hashes() -> dict:
    files = {
        "mappo_v2_seed0": primary_checkpoint(),
        "scheme_c": SCHEME_C_CKPT,
        "epoch6": EPOCH6_CKPT,
    }
    record = {}
    for name, path in files.items():
        file = Path(path)
        record[name] = {
            "path": file.as_posix(),
            "exists": file.is_file(),
            "bytes": int(file.stat().st_size) if file.is_file() else 0,
            "sha256": _sha256_file(file) if file.is_file() else "",
        }
    record["adapter_config"] = {"sha256": config_sha256()}
    return record


def hashes_match(record: dict) -> bool:
    expected = {
        "mappo_v2_seed0": MAPPO_SHA256,
        "scheme_c": SCHEME_C_SHA256,
        "epoch6": EPOCH6_SHA256,
    }
    for name, digest in expected.items():
        if record[name]["sha256"] != digest:
            return False
    return record["adapter_config"]["sha256"] == ADAPTER_CONFIG_SHA256


def _public_state(state: dict) -> dict:
    hidden = {"env_snap", "ledger_snap"}
    return {key: value for key, value in state.items() if key not in hidden}


def _score_bundle(bundle: dict, actor, limit: int | None = None) -> tuple[list[dict], float, str | None]:
    started = time.perf_counter()
    rows = []
    problem = None
    items = bundle["rows"] if limit is None else bundle["rows"][: int(limit)]
    for item in items:
        scored = score_assignment(
            bundle["state"]["env_snap"],
            bundle["state"]["ledger_snap"],
            actor,
            np.asarray(item["assignment"], dtype=np.int64),
            scenario_id=int(bundle["state"]["scenario_id"]),
        )
        gap = abs(float(scored["payload_bits"]) - float(item["j15_bits"]))
        row = {
            "scenario_id": int(item["scenario_id"]),
            "time_step": int(item["time_step"]),
            "candidate_id": int(item["candidate_id"]),
            "payload_bits": float(scored["payload_bits"]),
            "payload_term": float(scored["payload_term"]),
            "urgent_service_bits": float(scored["urgent_service_bits"]),
            "urgent_service_term": float(scored["urgent_service_term"]),
            "payload_gap_bits": float(gap),
            "ledger_gap_bits": float(scored["ledger_gap_bits"]),
            "accounting_error_bits": float(scored["accounting_error_bits"]),
            "served_bits": scored["served_bits"],
            "oldest_age_start": scored["oldest_age_start"],
            "generated_bits": float(scored["generated_bits"]),
            "seconds": float(scored["seconds"]),
            "j15_bits": float(item["j15_bits"]),
        }
        rows.append(row)
        tol = float(bundle["state"]["tol_bits"])
        if gap > tol or float(scored["ledger_gap_bits"]) > tol or float(scored["accounting_error_bits"]) > tol:
            problem = (
                f"label check failed for scenario {row['scenario_id']} t={row['time_step']} "
                f"candidate {row['candidate_id']}: payload gap {gap}"
            )
            break
    return rows, float(time.perf_counter() - started), problem


def _same_labels(left: list[dict], right: list[dict]) -> float:
    if len(left) != len(right):
        return float("inf")
    worst = 0.0
    for first, second in zip(left, right):
        if int(first["candidate_id"]) != int(second["candidate_id"]):
            return float("inf")
        worst = max(worst, abs(float(first["payload_bits"]) - float(second["payload_bits"])))
        worst = max(worst, abs(float(first["urgent_service_bits"]) - float(second["urgent_service_bits"])))
    return float(worst)


def execute() -> dict:
    started = time.perf_counter()
    limit_threads()
    scenarios = pilot_scenarios()
    fingerprint_before = dataset_fingerprint(scenarios)
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
        "actor_unchanged": False,
    }
    budget_ok = False
    timing = {"first_state_seconds": None, "estimated_twelve_state_seconds": None, "label_seconds": 0.0}
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
                float(row["ledger_gap_bits"]) <= float(EPS_BITS) and all(bool(state["ledger_restore_independent"]) for state in row["states"])
                for row in replay_rows
            )
            if not mechanical["replay"]:
                reasons.append("replayed state does not match the frozen state shard")
            if not mechanical["ledger"]:
                reasons.append("admission-age ledger failed conservation or restore")
            if mechanical["replay"] and mechanical["ledger"]:
                for row in replay_rows:
                    for state in row["states"]:
                        print(f"candidates {state['scenario_id']} t={state['time_step']}", flush=True)
                        table = load_candidates(state["scenario_id"], state["time_step"], state["env_snap"], {
                            "checksum_candidates": state["manifest_checksum_candidates"],
                            "candidate_count": state["manifest_candidate_count"],
                            "illegal": state["manifest_illegal"],
                        })
                        chosen = select_candidates(
                            table["candidate_id"],
                            table["j15_bits"],
                            table["utility_rank"],
                            scenario_id=int(state["scenario_id"]),
                            time_step=int(state["time_step"]),
                        )
                        repeated = select_candidates(
                            table["candidate_id"],
                            table["j15_bits"],
                            table["utility_rank"],
                            scenario_id=int(state["scenario_id"]),
                            time_step=int(state["time_step"]),
                        )
                        if [int(item["candidate_id"]) for item in chosen["rows"]] != [int(item["candidate_id"]) for item in repeated["rows"]]:
                            raise RuntimeError("candidate sample changed when the pilot seed was repeated")
                        rows = []
                        for item in chosen["rows"]:
                            position = int(item["row_index"])
                            copied = dict(item)
                            copied["scenario_id"] = int(state["scenario_id"])
                            copied["time_step"] = int(state["time_step"])
                            copied["assignment"] = [int(value) for value in table["assignment"][position]]
                            copied["tie_group_id"] = int(table["tie_group_id"][position])
                            rows.append(copied)
                        if len(rows) > int(MAX_CANDIDATES_PER_STATE):
                            raise RuntimeError("candidate sample exceeds 128")
                        bundles.append({"state": state, "table": table, "rows": rows, "sample": chosen})
                mechanical["shards"] = True
                mechanical["legal"] = all(bool(bundle["table"]["pool_order_match"]) and int(bundle["table"]["illegal_count"]) == 0 for bundle in bundles)
                if not mechanical["legal"]:
                    reasons.append("candidate legality or pool order failed")
                if mechanical["legal"]:
                    print(f"score state 1/{len(bundles)}", flush=True)
                    first_rows, first_seconds, problem = _score_bundle(bundles[0], actor)
                    labels.extend(first_rows)
                    timing["first_state_seconds"] = float(first_seconds)
                    timing["estimated_twelve_state_seconds"] = float(first_seconds * len(bundles))
                    timing["label_seconds"] = float(first_seconds)
                    if problem:
                        raise RuntimeError(problem)
                    if first_seconds > float(MAX_STATE_SECONDS):
                        reasons.append(
                            f"one state took {first_seconds:.1f}s, above the pre-registered {MAX_STATE_SECONDS:.0f}s limit"
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
                        repeat_limit = None if first_seconds <= float(REPEAT_STATE_IF_UNDER_SECONDS) else int(REPEAT_CANDIDATES_IF_SLOW)
                        print("repeat first state", flush=True)
                        repeated_rows, _elapsed, problem = _score_bundle(bundles[0], actor, repeat_limit)
                        if problem:
                            raise RuntimeError(problem)
                        determinism_rollouts = int(len(repeated_rows))
                        determinism_gap = _same_labels(first_rows if repeat_limit is None else first_rows[:repeat_limit], repeated_rows)
                        mechanical["determinism"] = bool(determinism_gap <= float(EPS_BITS))
                        mechanical["payload"] = True
                        if not mechanical["determinism"]:
                            reasons.append("repeating the pilot seed changed payload or urgent-service labels")
        except Exception as exc:
            reasons.append(str(exc))
            mechanical["payload"] = False
    if labels and bundles and mechanical["payload"]:
        grouped: dict[tuple[int, int], list[dict]] = {}
        for row in labels:
            grouped.setdefault((int(row["scenario_id"]), int(row["time_step"])), []).append(row)
        for bundle in bundles:
            key = (int(bundle["state"]["scenario_id"]), int(bundle["state"]["time_step"]))
            rows = grouped.get(key, [])
            if len(rows) != len(bundle["rows"]):
                continue
            summary = state_rank_summary(rows)
            summary["scenario_id"] = key[0]
            summary["time_step"] = key[1]
            summary["oldest_age_start"] = rows[0]["oldest_age_start"]
            rank_states.append(summary)
    fingerprint_after = dataset_fingerprint(scenarios)
    hashes_after = checkpoint_hashes()
    mechanical["dataset_unchanged"] = fingerprint_before == fingerprint_after
    mechanical["actor_unchanged"] = actor_token is not None and parameter_token(_frozen_actor()) == actor_token
    if hashes_before != hashes_after:
        mechanical["checkpoint"] = False
        reasons.append("checkpoint hash changed during the pilot")
    if not mechanical["dataset_unchanged"]:
        reasons.append("working-tree dataset files changed during the pilot")
    if not mechanical["actor_unchanged"]:
        reasons.append("frozen MAPPO parameters changed during the pilot")
    mechanical_ok = all(bool(value) for key, value in mechanical.items() if key != "payload" or mechanical["payload"])
    # payload is false when scoring never starts; that must block.
    mechanical_ok = all(bool(value) for value in mechanical.values())
    coverage_complete = len(rank_states) == len(scenarios) * len(PRIMARY_TIMES) and len(scenarios) == 4
    if mechanical_ok and not coverage_complete:
        mechanical_ok = False
        reasons.append("rank coverage is short of 4 scenarios and 3 decision times")
    macro = macro_rank_summary(rank_states) if coverage_complete else {
        "states": int(len(rank_states)),
        "mean_spearman": None,
        "min_spearman": None,
        "mean_reversal_rate": None,
        "max_reversal_rate": None,
        "states_with_top1_change": None,
        "lambda_H": LAMBDA_H,
    }
    gate = choose_gate(
        mechanical_ok=mechanical_ok,
        budget_ok=bool(budget_ok or not bundles),
        mean_spearman=macro["mean_spearman"],
        mean_reversal=macro["mean_reversal_rate"],
    )
    if not bundles and mechanical_ok:
        gate = GATE_BLOCKED
    if reasons and gate != GATE_BLOCKED and not mechanical_ok:
        gate = GATE_BLOCKED
    if not mechanical_ok or not budget_ok:
        gate = GATE_BLOCKED
    return {
        "gate": gate,
        "recommendation": RECOMMENDATION[gate],
        "reasons": reasons,
        "lambda_H": LAMBDA_H,
        "scenarios": [int(item) for item in scenarios],
        "times": [int(item) for item in PRIMARY_TIMES],
        "rollout_count": int(len(labels)),
        "max_rollouts": int(MAX_ROLLOUTS),
        "determinism_rollouts": int(determinism_rollouts),
        "determinism_abs_gap_bits": determinism_gap,
        "candidate_cap_per_state": int(MAX_CANDIDATES_PER_STATE),
        "timing": timing,
        "mechanical": mechanical,
        "budget_ok": bool(budget_ok),
        "coverage_complete": bool(coverage_complete),
        "replay": [{**{key: value for key, value in row.items() if key != "states"}, "states": [_public_state(state) for state in row["states"]]} for row in replay_rows],
        "selection": [
            {
                "scenario_id": int(bundle["state"]["scenario_id"]),
                "time_step": int(bundle["state"]["time_step"]),
                "count": int(len(bundle["rows"])),
                "top_count": int(bundle["sample"]["top_count"]),
                "stratum_count": int(bundle["sample"]["stratum_count"]),
                "fill_count": int(bundle["sample"]["fill_count"]),
                "shortfall_unfilled": int(bundle["sample"]["shortfall_unfilled"]),
                "pool_hash": bundle["table"]["pool_hash"],
                "candidate_count": int(bundle["table"]["candidate_count"]),
                "strata": bundle["sample"]["strata"],
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
        "pilot_seed": int(PILOT_SEED),
        "dataset_files": fingerprint_after,
        "seconds_total": float(time.perf_counter() - started),
        "labels": labels,
        "selected_rows": [row for bundle in bundles for row in bundle["rows"]],
    }


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _csv_value(value):
    if isinstance(value, (list, tuple)):
        return "|".join(str(item) for item in value)
    if isinstance(value, float):
        return f"{value:.10g}"
    return value


def write_results(summary: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected_fields = [
        "scenario_id",
        "time_step",
        "candidate_id",
        "assignment",
        "j15_bits",
        "utility_rank",
        "tie_group_id",
        "selection",
        "stratum",
        "filled",
    ]
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
        "times": summary["times"],
        "generator": GENERATOR,
        "k": int(SELECTED_K),
        "hold": int(HOLD),
        "label": LABEL_NAME,
        "pilot_seed": int(PILOT_SEED),
        "top_k": int(TOP_K),
        "sample_k": int(SAMPLE_K),
        "strata": int(STRATA),
        "sampling": "32 stable top existing J15 labels, then 96 equal-count utility_rank quantiles from the remainder; short strata fill by candidate id",
        "formulas": {
            "payload_term": "payload_increment_bits / 1e6",
            "urgent_service_term": "sum_t,k(urgency_k_before_service[t] * served_bits[k,t]) / 1e6",
            "utility": "payload_term + lambda_H * urgent_service_term",
            "urgency": "clip(oldest_wait_age / 48, 0, 1) when q_bits > 0 else 0",
        },
        "lambda_H": LAMBDA_H,
        "lambda_H_decided": False,
        "signal_thresholds": {
            "useful_spearman_below": USEFUL_SPEARMAN_BELOW,
            "useful_reversal_above": USEFUL_REVERSAL_ABOVE,
            "payload_scale": PAYLOAD_SCALE,
        },
        "arrival_convention": "candidate label rollout zeros arrivals and arrivals_at on the scratch copy only",
        "state_source": "MAX_RATE_INDEPENDENT held for 15 steps and executed by frozen MAPPO-V2 seed 0",
        "ledger": "episode_admission_age_v1",
        "age_scale_steps": 48,
        "reset_origin": "reset_convention",
        "frozen_shard_commit": FROZEN_SHARD_COMMIT,
        "instruction_commit": INSTRUCTION_COMMIT,
        "checkpoint_sha256": summary["checkpoint_hashes"],
        "expected_checkpoint_sha256": {
            "mappo_v2_seed0": MAPPO_SHA256,
            "scheme_c": SCHEME_C_SHA256,
            "epoch6": EPOCH6_SHA256,
            "adapter_config": ADAPTER_CONFIG_SHA256,
        },
        "fields": {
            "selected_candidates.csv": selected_fields,
            "labels.csv": label_fields,
        },
    }
    (OUTPUT / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> dict:
    summary = execute()
    write_results(summary)
    print(summary["gate"], summary["rollout_count"], summary["timing"], flush=True)
    return summary


if __name__ == "__main__":
    main()
