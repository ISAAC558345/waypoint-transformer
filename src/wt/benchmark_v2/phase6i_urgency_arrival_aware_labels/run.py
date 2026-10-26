"""Compare zero-arrival J15 labels with real-arrival labels at t=30."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import limit_threads
from wt.benchmark_v2.phase6ib_highlevel_dataset.jobs import _frozen_actor
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.candidates import load_candidates
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.replay import replay_scenario
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.sampling import select_candidates
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.shards import dataset_fingerprint
from wt.benchmark_v2.phase6i_urgency_stage_a.ledger import AdmissionAgeLedger
from wt.benchmark_v2.phase6i_urgency_tiebreak_closedloop.protocol import (
    ADAPTER_CONFIG_SHA256,
    EPOCH6_CKPT,
    EPOCH6_SHA256,
    MAPPO_SHA256,
    SCHEME_C_CKPT,
    SCHEME_C_SHA256,
    OUTPUT as D3_OUTPUT,
    config_sha256,
    primary_checkpoint,
)
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.protocol import OUTPUT as D0_OUTPUT
from wt.benchmark_v2.phase6i_urgency_highlevel_t30_pilot.protocol import OUTPUT as D1_OUTPUT
from wt.benchmark_v2.phase6i_urgency_utility_rule.protocol import OUTPUT as D2_OUTPUT

from .analyze import quartile_labels, rank_state, stratum_summary
from .labels import score_label
from .protocol import (
    DECISION_TIME,
    EPS_BITS,
    FROZEN_SHARD_COMMIT,
    GATE_BLOCKED,
    GENERATOR,
    HORIZON,
    INSTRUCTION_COMMIT,
    LABEL_SEED,
    MAX_CANDIDATES,
    MAX_ROLLOUTS,
    MAX_SECONDS,
    MAX_STATE_SECONDS,
    OUTPUT,
    SELECTED_K,
    choose_gate,
    scenario_effect,
    scenarios,
    study_effect,
)

D4_OUTPUT = Path(__file__).resolve().parents[4] / "results" / "phase6i_urgency_tiebreak_single_intervention"
PLANNER_PATH = Path(__file__).resolve().parents[1] / "phase6iaa_formal_planner" / "planner.py"

SELECTED_FIELDS = (
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
    "idle_count",
    "urgency_start_assigned",
    "state_hash",
    "pool_hash",
    "subset_hash",
    "row_index",
)
LABEL_FIELDS = SELECTED_FIELDS + (
    "zero_payload_bits",
    "zero_urgent_service_bits",
    "zero_dropped_bits",
    "zero_final_queue_bits",
    "zero_served_bits",
    "zero_generated_bits",
    "zero_ledger_gap_bits",
    "zero_accounting_error_bits",
    "zero_payload_served_gap_bits",
    "zero_payload_gap_vs_j15_bits",
    "real_payload_bits",
    "real_urgent_service_bits",
    "real_dropped_bits",
    "real_final_queue_bits",
    "real_served_bits",
    "real_generated_bits",
    "real_ledger_gap_bits",
    "real_accounting_error_bits",
    "real_payload_served_gap_bits",
    "payload_delta_bits",
    "urgent_delta_bits",
    "dropped_delta_bits",
    "final_queue_delta_bits",
    "payload_quartile",
    "urgency_start_quartile",
)

NEXT_STEP = {
    "material_consistent": "下一步只提出 arrival-aware label 的小规模扩展方案，先不构建训练集，也不训练 HGNN。",
    "small": "停止高层紧急度标签路线。真实到达没有稳定地改变候选排序，保留 payload-first 正式高层，紧急度只作为报告指标复查。",
    "unstable": "停止高层紧急度标签路线。真实到达对排序的影响跨场景不稳定，不能据此采用新的标签方案或进入训练。",
    "budget": "不要扩大本次预算。按已测速度，四个状态的总时间会超过 20 分钟，本次不能判断标签方案。",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _subset_hash(assignments: list[np.ndarray]) -> str:
    raw = np.ascontiguousarray(np.vstack(assignments), dtype=np.int64).tobytes()
    return hashlib.sha256(raw).hexdigest()


def _join(values) -> str:
    return "|".join(str(int(item)) if float(item).is_integer() else f"{float(item):.8g}" for item in values)


def _assignment_text(values) -> str:
    return "|".join(str(int(item)) for item in np.asarray(values, dtype=np.int64).reshape(-1))


def _csv(value):
    if isinstance(value, float):
        return f"{value:.16g}"
    return value


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv(row.get(key, "")) for key in fields})
        handle.flush()


def _watched() -> dict:
    paths = {
        "d0_labels": D0_OUTPUT / "labels.csv",
        "d1_labels": D1_OUTPUT / "labels.csv",
        "d2_summary": D2_OUTPUT / "summary.json",
        "d3_summary": D3_OUTPUT / "summary.json",
        "d4_summary": D4_OUTPUT / "summary.json",
        "formal_planner": PLANNER_PATH,
    }
    return {name: _sha256(path) for name, path in paths.items()}


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


def _urgency_start(ledger_snap: dict, assignment: np.ndarray, q_bits: np.ndarray, tol_bits: float) -> float:
    ledger = AdmissionAgeLedger(int(np.asarray(q_bits).size))
    ledger.restore(ledger_snap)
    view = ledger.view(int(DECISION_TIME), np.asarray(q_bits, dtype=np.float64), float(tol_bits))
    urgency = np.asarray(view["urgency"], dtype=np.float64)
    total = 0.0
    for buoy in np.asarray(assignment, dtype=np.int64).reshape(-1):
        if int(buoy) >= 0:
            total += float(urgency[int(buoy)])
    return float(total)


def _select(scenario: int, state: dict) -> tuple[list[dict], dict]:
    table = load_candidates(int(scenario), int(DECISION_TIME), state["env_snap"], {
        "checksum_candidates": state["manifest_checksum_candidates"],
        "candidate_count": state["manifest_candidate_count"],
        "illegal": state["manifest_illegal"],
    })
    picked = select_candidates(
        table["candidate_id"],
        table["j15_bits"],
        table["utility_rank"],
        scenario_id=int(scenario),
        time_step=int(DECISION_TIME),
        seed=int(LABEL_SEED),
    )
    if int(picked["count"]) != int(MAX_CANDIDATES):
        raise RuntimeError(f"subset size is {picked['count']}, not {MAX_CANDIDATES}")
    q_bits = np.asarray(state["env_snap"]["q_bits"], dtype=np.float64)
    rows = []
    assignments = []
    for item in picked["rows"]:
        position = int(item["row_index"])
        assignment = np.asarray(table["assignment"][position], dtype=np.int64).copy()
        assignments.append(assignment)
        rows.append(
            {
                "scenario_id": int(scenario),
                "time_step": int(DECISION_TIME),
                "candidate_id": int(item["candidate_id"]),
                "assignment": _assignment_text(assignment),
                "assignment_vector": assignment,
                "j15_bits": float(item["j15_bits"]),
                "utility_rank": int(item["utility_rank"]),
                "tie_group_id": int(table["tie_group_id"][position]),
                "selection": str(item["selection"]),
                "stratum": int(item["stratum"]),
                "filled": int(item["filled"]),
                "idle_count": int(np.sum(assignment < 0)),
                "urgency_start_assigned": _urgency_start(state["ledger_snap"], assignment, q_bits, float(state["tol_bits"])),
                "state_hash": str(state["state_hash"]),
                "pool_hash": str(table["pool_hash"]),
                "row_index": position,
            }
        )
    digest = _subset_hash(assignments)
    for row in rows:
        row["subset_hash"] = digest
    return rows, {
        "pool_hash": str(table["pool_hash"]),
        "pool_count": int(table["candidate_count"]),
        "subset_hash": digest,
        "checksum_candidates": str(table["checksum_candidates"]),
    }


def _score_rows(rows: list[dict], state: dict, actor) -> list[dict]:
    scored = []
    for row in rows:
        zero = score_label(
            state["env_snap"],
            state["ledger_snap"],
            actor,
            row["assignment_vector"],
            scenario_id=int(row["scenario_id"]),
            zero_arrivals=True,
        )
        real = score_label(
            state["env_snap"],
            state["ledger_snap"],
            actor,
            row["assignment_vector"],
            scenario_id=int(row["scenario_id"]),
            zero_arrivals=False,
        )
        gap = float(zero["payload_bits"]) - float(row["j15_bits"])
        scored.append(
            {
                **{key: value for key, value in row.items() if key != "assignment_vector"},
                "zero_payload_bits": float(zero["payload_bits"]),
                "zero_urgent_service_bits": float(zero["urgent_service_bits"]),
                "zero_dropped_bits": float(zero["dropped_bits"]),
                "zero_final_queue_bits": float(zero["final_queue_bits"]),
                "zero_served_bits": _join(zero["served_bits"]),
                "zero_generated_bits": float(zero["generated_bits"]),
                "zero_ledger_gap_bits": float(zero["ledger_gap_bits"]),
                "zero_accounting_error_bits": float(zero["accounting_error_bits"]),
                "zero_payload_served_gap_bits": float(zero["payload_served_gap_bits"]),
                "zero_payload_gap_vs_j15_bits": float(gap),
                "real_payload_bits": float(real["payload_bits"]),
                "real_urgent_service_bits": float(real["urgent_service_bits"]),
                "real_dropped_bits": float(real["dropped_bits"]),
                "real_final_queue_bits": float(real["final_queue_bits"]),
                "real_served_bits": _join(real["served_bits"]),
                "real_generated_bits": float(real["generated_bits"]),
                "real_ledger_gap_bits": float(real["ledger_gap_bits"]),
                "real_accounting_error_bits": float(real["accounting_error_bits"]),
                "real_payload_served_gap_bits": float(real["payload_served_gap_bits"]),
                "payload_delta_bits": float(real["payload_bits"]) - float(zero["payload_bits"]),
                "urgent_delta_bits": float(real["urgent_service_bits"]) - float(zero["urgent_service_bits"]),
                "dropped_delta_bits": float(real["dropped_bits"]) - float(zero["dropped_bits"]),
                "final_queue_delta_bits": float(real["final_queue_bits"]) - float(zero["final_queue_bits"]),
            }
        )
        if abs(gap) > float(state["tol_bits"]):
            raise RuntimeError(
                f"zero-arrival payload does not reproduce j15 for scenario {row['scenario_id']} candidate {row['candidate_id']}: gap {gap}"
            )
    return scored


def _annotate(rows: list[dict]) -> None:
    payload_bands = quartile_labels([float(row["j15_bits"]) for row in rows])
    urgency_bands = quartile_labels([float(row["urgency_start_assigned"]) for row in rows])
    for row, payload_band, urgency_band in zip(rows, payload_bands, urgency_bands):
        row["payload_quartile"] = payload_band
        row["urgency_start_quartile"] = urgency_band


def _state_record(scenario: int, replay: dict) -> dict:
    state = next(item for item in replay["states"] if int(item["time_step"]) == int(DECISION_TIME))
    if not state["state_hash_match"] or not state["checksum_state_match"] or not state["ledger_restore_independent"]:
        raise RuntimeError(f"t=30 replay does not match the frozen shard for {scenario}")
    return state


def _problems(rows: list[dict], state: dict) -> list[str]:
    reasons = []
    tol = float(state["tol_bits"])
    if any(abs(float(row["zero_payload_gap_vs_j15_bits"])) > tol for row in rows):
        reasons.append("a zero-arrival payload label does not reproduce j15_bits")
    if any(float(row["zero_generated_bits"]) > tol for row in rows):
        reasons.append("a zero-arrival label still generated arrivals")
    generated = [float(row["real_generated_bits"]) for row in rows]
    if generated and max(generated) - min(generated) > tol:
        reasons.append("real-arrival copies did not share the exogenous arrival amount")
    for key in (
        "zero_ledger_gap_bits",
        "real_ledger_gap_bits",
        "zero_accounting_error_bits",
        "real_accounting_error_bits",
        "zero_payload_served_gap_bits",
        "real_payload_served_gap_bits",
    ):
        if any(float(row[key]) > tol for row in rows):
            reasons.append(f"{key} exceeded 1 bit")
            break
    if float(state["ledger_gap_bits"]) > tol:
        reasons.append("the replay ledger exceeded 1 bit")
    return reasons


def execute() -> dict:
    limit_threads()
    before = _watched()
    fingerprint_before = dataset_fingerprint(scenarios())
    hashes = _checkpoint_hashes()
    actor = _frozen_actor()
    selected_rows: list[dict] = []
    label_rows: list[dict] = []
    state_rows: list[dict] = []
    budget_ok = True
    crashed = False
    failure = ""
    failure_trace = ""
    first_seconds = None
    estimate = None
    rollouts = 0
    try:
        for scenario in scenarios():
            if first_seconds is not None:
                estimate = float(first_seconds) * len(scenarios())
                print(f"projected total {estimate:.3f}s", flush=True)
                if estimate > float(MAX_SECONDS):
                    budget_ok = False
                    failure = f"projected total {estimate:.3f}s exceeds {MAX_SECONDS:.0f}s"
                    break
            started = time.perf_counter()
            print(f"scenario {scenario} replay", flush=True)
            replay = replay_scenario(int(scenario))
            state = _state_record(int(scenario), replay)
            chosen, pool = _select(int(scenario), state)
            selected_rows.extend({key: row.get(key, "") for key in SELECTED_FIELDS} for row in chosen)
            _write_csv(OUTPUT / "selected_candidates.csv", selected_rows, SELECTED_FIELDS)
            print(f"scenario {scenario} scoring {len(chosen)} candidates", flush=True)
            scored = _score_rows(chosen, state, actor)
            _annotate(scored)
            label_rows.extend(scored)
            rollouts += 2 * len(scored)
            elapsed = time.perf_counter() - started
            print(f"scenario {scenario} finished in {elapsed:.3f}s", flush=True)
            if first_seconds is None:
                first_seconds = elapsed
                if elapsed > float(MAX_STATE_SECONDS):
                    budget_ok = False
                    failure = f"first state {elapsed:.3f}s exceeds {MAX_STATE_SECONDS:.0f}s"
                    break
            ranking = rank_state(scored)
            effect = scenario_effect(
                urgent_reversal=float(ranking["urgent_inversion"]["inversion_rate"]),
                payload_reversal=float(ranking["payload_inversion"]["inversion_rate"]),
                payload_top1_within_one_bit=bool(ranking["payload_top1_within_one_bit"]),
                urgent_top1_within_one_bit=bool(ranking["urgent_top1_within_one_bit"]),
            )
            q_bits = np.asarray(state["env_snap"]["q_bits"], dtype=np.float64)
            ledger = AdmissionAgeLedger(int(q_bits.size))
            ledger.restore(state["ledger_snap"])
            view = ledger.view(int(DECISION_TIME), q_bits, float(state["tol_bits"]))
            state_rows.append(
                {
                    "scenario_id": int(scenario),
                    "seconds": float(elapsed),
                    "state_hash": str(state["state_hash"]),
                    "pool_hash": pool["pool_hash"],
                    "pool_count": int(pool["pool_count"]),
                    "subset_hash": pool["subset_hash"],
                    "checksum_candidates": pool["checksum_candidates"],
                    "q_bits": [float(item) for item in q_bits],
                    "urgency": [float(item) for item in view["urgency"]],
                    "oldest_wait_age": [float(item) for item in view["oldest_wait_age"]],
                    "replay_ledger_gap_bits": float(replay["ledger_gap_bits"]),
                    "replay_accounting_error_bits": float(replay["accounting_error_bits"]),
                    "ranking": ranking,
                    "effect": effect,
                    "payload_strata": stratum_summary(scored, "payload_quartile"),
                    "urgency_start_strata": stratum_summary(scored, "urgency_start_quartile"),
                    "idle_strata": stratum_summary(scored, "idle_count"),
                    "problems": _problems(scored, state),
                }
            )
            if state_rows[-1]["problems"]:
                raise RuntimeError("; ".join(state_rows[-1]["problems"]))
    except Exception as exc:
        crashed = True
        budget_ok = False
        failure = f"{type(exc).__name__}: {exc}"
        failure_trace = traceback.format_exc()
    after = _watched()
    fingerprint_after = dataset_fingerprint(scenarios())
    reasons = []
    if failure:
        reasons.append(failure)
    if before != after:
        reasons.append("D0-D4 results or the formal planner file changed")
    if fingerprint_before != fingerprint_after:
        reasons.append("the frozen dataset files changed")
    if rollouts > int(MAX_ROLLOUTS):
        reasons.append("rollout count exceeded 1024")
    for row in state_rows:
        reasons.extend(str(item) for item in row["problems"])
    complete = len(state_rows) == len(scenarios()) and all(int(row["ranking"]["count"]) == int(MAX_CANDIDATES) for row in state_rows)
    mechanical_ok = complete and not reasons and not failure
    effect = study_effect([str(row["effect"]) for row in state_rows]) if complete else "unstable"
    if (not budget_ok) and (not crashed) and failure:
        effect_name = "budget"
    elif mechanical_ok:
        effect_name = effect
    else:
        effect_name = "unstable"
    gate = choose_gate(
        mechanical_ok=mechanical_ok,
        budget_ok=budget_ok and not failure,
        effect=effect_name if effect_name in {"material_consistent", "small", "unstable"} else "unstable",
    )
    if failure or not budget_ok:
        gate = GATE_BLOCKED
    next_step = NEXT_STEP.get(effect_name, NEXT_STEP["unstable"])
    if gate == GATE_BLOCKED and effect_name != "budget":
        next_step = "不要扩大实验，也不要训练。先修复完整性或预算失败，再决定标签方案。"
    return {
        "gate": gate,
        "reasons": reasons,
        "failure_trace": failure_trace,
        "effect": effect_name,
        "scenario_effects": [str(row["effect"]) for row in state_rows],
        "next_step": next_step,
        "learned_policy": False,
        "trained_hgnn": False,
        "trained_mappo": False,
        "formal_system_modified": False,
        "lambda_H": None,
        "delta_payload_max": None,
        "d4_gate_unchanged": "URGENCY_TIEBREAK_SINGLE_INTERVENTION_INCONCLUSIVE",
        "instruction_commit": INSTRUCTION_COMMIT,
        "frozen_shard_commit": FROZEN_SHARD_COMMIT,
        "label_seed": int(LABEL_SEED),
        "seed_sequence": "[604, scenario_id, 30]",
        "decision_time": int(DECISION_TIME),
        "horizon": int(HORIZON),
        "generator": GENERATOR,
        "selected_k": int(SELECTED_K),
        "candidates_per_state": int(MAX_CANDIDATES),
        "rollouts": int(rollouts),
        "max_rollouts": int(MAX_ROLLOUTS),
        "first_state_seconds": first_seconds,
        "estimated_seconds": estimate if estimate is not None else (None if first_seconds is None else float(first_seconds) * len(scenarios())),
        "max_seconds": float(MAX_SECONDS),
        "scenarios": list(scenarios()),
        "states": state_rows,
        "checkpoint_sha256": hashes,
        "watched_sha256": before,
        "watched_unchanged": before == after,
        "dataset_unchanged": fingerprint_before == fingerprint_after,
        "arrival_pairing": "arrivals_at is a deterministic function of time; zero-arrival disables it only on the scratch copy",
        "selected_rows": selected_rows,
        "label_rows": label_rows,
    }


def write_results(summary: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT / "selected_candidates.csv", summary["selected_rows"], SELECTED_FIELDS)
    _write_csv(OUTPUT / "labels.csv", summary["label_rows"], LABEL_FIELDS)
    public = {key: value for key, value in summary.items() if key not in {"selected_rows", "label_rows"}}
    (OUTPUT / "summary.json").write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")
    schema = {
        "meaning": "Each selected candidate receives one zero-arrival J15 label and one real-arrival 15-step label from the same t=30 snapshot.",
        "arms": {
            "zero_arrival": "scratch copy disables arrivals and arrivals_at; payload is checked against the frozen j15_bits",
            "real_arrival": "scratch copy keeps the formal arrivals_at(t); the result is an offline label and is not fed back into the observation",
        },
        "sampling": {
            "seed": int(LABEL_SEED),
            "seed_sequence": "[604, scenario_id, 30]",
            "size": "32 stable top J15 labels plus 96 utility-rank strata",
            "saved_before_rollout": True,
        },
        "horizon": int(HORIZON),
        "decision_time": int(DECISION_TIME),
        "eps_bits": float(EPS_BITS),
        "top1": "maximum label, with a 1-bit tie broken by the smaller candidate id",
        "inversion": "a pair counts only when both labels differ by more than 1 bit and their order disagrees",
        "effect_thresholds": {
            "material_urgent_reversal": 0.05,
            "small_reversal": 0.01,
        },
        "strata_are_descriptive": True,
        "lambda_H": None,
        "delta_payload_max": None,
        "held_out_splits_read": False,
        "instruction_commit": INSTRUCTION_COMMIT,
        "fields": {
            "selected_candidates.csv": list(SELECTED_FIELDS),
            "labels.csv": list(LABEL_FIELDS),
        },
    }
    (OUTPUT / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> dict:
    summary = execute()
    write_results(summary)
    print(summary["gate"], summary["effect"], summary["first_state_seconds"], summary["rollouts"], flush=True)
    if summary["failure_trace"]:
        print(summary["failure_trace"], flush=True)
    return summary


if __name__ == "__main__":
    main()
