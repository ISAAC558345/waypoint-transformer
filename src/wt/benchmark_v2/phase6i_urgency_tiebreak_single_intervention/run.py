"""Run the single-decision diagnosis and stop when the 20-minute estimate is exceeded."""
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

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import limit_threads
from wt.benchmark_v2.phase6iaa_formal_planner.planner import FormalPlanner
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.protocol import OUTPUT as D0_OUTPUT
from wt.benchmark_v2.phase6i_urgency_highlevel_t30_pilot.protocol import OUTPUT as D1_OUTPUT
from wt.benchmark_v2.phase6i_urgency_tiebreak_closedloop.protocol import (
    ADAPTER_CONFIG_SHA256,
    EPOCH6_CKPT,
    EPOCH6_SHA256,
    MAPPO_SHA256,
    SCHEME_C_CKPT,
    SCHEME_C_SHA256,
    config_sha256,
    primary_checkpoint,
)
from wt.benchmark_v2.phase6i_urgency_utility_rule.protocol import OUTPUT as D2_OUTPUT

from .episode import ScoreTap, run_episode
from .protocol import (
    CONTROL_ARM,
    D3_OUTPUT,
    EPS_BITS,
    GATE_BLOCKED,
    INSTRUCTION_COMMIT,
    INTERVENTION_TIMES,
    MAX_EPISODES,
    MAX_SECONDS,
    ORACLE_LABEL,
    OUTPUT,
    PILOT_SEED,
    arm_name,
    choose_gate,
    mechanism_name,
    payload_sign,
    scenarios,
)

PLANNER_PATH = Path(__file__).resolve().parents[1] / "phase6iaa_formal_planner" / "planner.py"
SCHEME_C_CALLS = 4 * 3 * 4
LABEL_ALLOWANCE_SECONDS = 4 * 15.0
DIAGNOSTIC_ALLOWANCE_SECONDS = 40.0
CHECKPOINTS = (15, 30, 45, 48)

EPISODE_FIELDS = (
    "scenario_id",
    "arm",
    "intervention_time",
    "status",
    "final_t",
    "payload_bits",
    "served_bits_sum",
    "payload_accounting_gap_bits",
    "urgent_service_bits",
    "generated_arrival_bits",
    "dropped_bits",
    "final_queue_bits",
    "oldest_wait_max",
    "oldest_wait_mean",
    "path_length_m",
    "mean_sinr",
    "mean_total_interference",
    "projector_trigger_steps",
    "mean_projection_norm",
    "safety_violation_steps",
    "in_radius_links",
    "first_service_radius_step",
    "positive_service_steps",
    "max_ledger_gap_bits",
    "max_accounting_error_bits",
    "seconds",
)
DECISION_FIELDS = (
    "scenario_id",
    "arm",
    "time_step",
    "intervention_time",
    "is_intervention",
    "state_hash",
    "pool_hash",
    "candidate_count",
    "subset_hash",
    "subset_count",
    "sampling_source",
    "scheme_c_assignment",
    "scheme_c_score_index",
    "scheme_c_score",
    "scheme_c_called",
    "payload_first_assignment",
    "tiebreak_assignment",
    "executed_assignment",
    "assignment_changed_vs_scheme_c",
    "assignment_changed_vs_payload_first",
    "label_horizon",
    "label_payload_gap_bits",
    "label_urgent_gain_bits",
    "label_gap_within_one_bit",
    "real_payload_first_bits",
    "real_tiebreak_payload_bits",
    "real_payload_gap_bits",
    "real_urgent_gain_bits",
    "real_generated_gap_bits",
    "arrival_pairing",
    "arrival_paired",
    "diagnostic_previous_action",
    "label_source",
    "max_label_ledger_gap_bits",
    "max_label_accounting_error_bits",
    "pool_seconds",
    "score_seconds",
    "selection_rule",
)
STEP_FIELDS = (
    "scenario_id",
    "arm",
    "intervention_time",
    "time_step",
    "assignment",
    "payload_increment_bits",
    "cumulative_payload_bits",
    "cumulative_payload_delta_vs_control_bits",
    "served_bits",
    "generated_arrival_bits",
    "admitted_arrival_bits",
    "dropped_bits",
    "q_bits_after",
    "urgency",
    "oldest_wait_age",
    "uav_x_m",
    "uav_y_m",
    "target_distance_m",
    "sinr",
    "total_interference",
    "projection_norm",
    "constraint_status",
    "ledger_gap_bits",
    "accounting_error_bits",
    "in_radius_links",
)

NEXT_STEP = {
    "initial_assignment": "停止扩大 tie-break。t=0 的单次插入带来真实 payload 损失，t=30 没有同样的损失；不要训练 HGNN，也不要把 1 bit 标签并列解释成 48 步 payload 不变。",
    "mid_episode": "停止扩大 tie-break。t=30 的单次插入已经改变剩余 18 步的真实 payload。",
    "both_clocks": "停止扩大 tie-break。t=0 和 t=30 的单次插入都降低真实 payload，不能把损失归因给单独一个决策点后继续扩展。",
    "no_payload_loss": "停止扩大 tie-break。两个单点插入都没有稳定的真实 payload 损失，这四个场景不足以支持推广该规则。",
    "unresolved": "停止继续研究 tie-break，不扩大数据集。四个场景上的方向不一致，不能识别稳定机制。",
    "unpaired_arrivals": "停止继续研究 tie-break。两份真实转移副本的外生到达量没有配上，不能把 payload 差解释成 assignment 的因果效应。",
    "budget": "不要扩大本次预算。按已测速度，四场景总时间会超过 20 分钟，本次不能识别机制。",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _watched() -> dict:
    paths = {
        "d0_labels": D0_OUTPUT / "labels.csv",
        "d1_labels": D1_OUTPUT / "labels.csv",
        "d2_summary": D2_OUTPUT / "summary.json",
        "d3_summary": D3_OUTPUT / "summary.json",
        "d3_decision_rows": D3_OUTPUT / "decision_rows.csv",
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


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _median(values: list[float]) -> float:
    ordered = sorted(float(item) for item in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float(0.5 * (ordered[mid - 1] + ordered[mid]))


def _probe(planner: FormalPlanner) -> tuple[float, float]:
    """Time two formal choices. The second call is the steady-state cost."""
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=int(scenarios()[0]))
    position = np.asarray(env.uav_positions, dtype=np.float64).copy()
    started = time.perf_counter()
    first = planner.select_assignment(env, position, None)
    first_seconds = time.perf_counter() - started
    if first.status != "selected":
        raise RuntimeError(f"scheme C probe did not select: {first.reason or first.status}")
    started = time.perf_counter()
    second = planner.select_assignment(env, position, None)
    second_seconds = time.perf_counter() - started
    if second.status != "selected":
        raise RuntimeError(f"scheme C probe did not select: {second.reason or second.status}")
    return float(first_seconds), float(second_seconds)


def _estimate(first_seconds: float, second_seconds: float) -> float:
    """Probe calls plus every planned Scheme C call, with allowance for labels and diagnostics."""
    return float(first_seconds + second_seconds * (SCHEME_C_CALLS + 1) + LABEL_ALLOWANCE_SECONDS + DIAGNOSTIC_ALLOWANCE_SECONDS)


def _lookup(rows: list[dict], scenario_id: int, arm: str) -> dict:
    found = [row for row in rows if int(row["scenario_id"]) == int(scenario_id) and row["arm"] == arm]
    if len(found) != 1:
        raise RuntimeError(f"expected one {arm} episode for scenario {scenario_id}")
    return found[0]


def _decisions_for(episode: dict) -> dict:
    return {int(row["time_step"]): row for row in episode["decisions"]}


def _attach_curves(episodes: list[dict]) -> list[dict]:
    """Pair each intervention with the one deterministic Scheme C episode."""
    pairs = []
    for scenario in scenarios():
        present = {item["arm"] for item in episodes if int(item["scenario_id"]) == int(scenario)}
        if CONTROL_ARM not in present:
            continue
        control = _lookup(episodes, int(scenario), CONTROL_ARM)
        control_steps = {int(row["time_step"]): row for row in control["steps"]}
        for intervention in INTERVENTION_TIMES:
            arm = arm_name(int(intervention))
            if arm not in present:
                continue
            other = _lookup(episodes, int(scenario), arm)
            curve = []
            fork = None
            for row in other["steps"]:
                base = control_steps[int(row["time_step"])]
                delta = float(row["cumulative_payload_bits"]) - float(base["cumulative_payload_bits"])
                row["cumulative_payload_delta_vs_control_bits"] = float(delta)
                curve.append(float(delta))
                if fork is None and abs(delta) > float(EPS_BITS):
                    fork = int(row["time_step"])
            for row in control["steps"]:
                row["cumulative_payload_delta_vs_control_bits"] = 0.0
            control_decisions = _decisions_for(control)
            other_decisions = _decisions_for(other)
            changed_times = [
                int(time_step)
                for time_step, row in other_decisions.items()
                if row["executed_assignment"] != control_decisions[int(time_step)]["executed_assignment"]
            ]
            checkpoints = {str(time_step): float(curve[int(time_step) - 1]) for time_step in CHECKPOINTS}
            pairs.append(
                {
                    "scenario_id": int(scenario),
                    "arm": arm,
                    "intervention_time": int(intervention),
                    "payload_delta_bits": float(other["payload_bits"]) - float(control["payload_bits"]),
                    "urgent_delta_bits": float(other["urgent_service_bits"]) - float(control["urgent_service_bits"]),
                    "dropped_delta_bits": float(other["dropped_bits"]) - float(control["dropped_bits"]),
                    "final_queue_delta_bits": float(other["final_queue_bits"]) - float(control["final_queue_bits"]),
                    "oldest_wait_max_delta": float(other["oldest_wait_max"]) - float(control["oldest_wait_max"]),
                    "path_length_delta_m": float(other["path_length_m"]) - float(control["path_length_m"]),
                    "mean_sinr_delta": float(other["mean_sinr"]) - float(control["mean_sinr"]),
                    "mean_interference_delta": float(other["mean_total_interference"]) - float(control["mean_total_interference"]),
                    "projector_trigger_delta": int(other["projector_trigger_steps"]) - int(control["projector_trigger_steps"]),
                    "safety_violation_delta": int(other["safety_violation_steps"]) - int(control["safety_violation_steps"]),
                    "positive_service_steps_delta": int(other["positive_service_steps"]) - int(control["positive_service_steps"]),
                    "assignment_fork_time": None if not changed_times else int(min(changed_times)),
                    "payload_fork_step": fork,
                    "later_assignment_times_changed": changed_times,
                    "cumulative_payload_delta_bits": curve,
                    "payload_delta_after_time": checkpoints,
                    "intervention_changed_assignment": int(other_decisions[int(intervention)]["assignment_changed_vs_scheme_c"]),
                    "label_payload_gap_bits": float(other_decisions[int(intervention)]["label_payload_gap_bits"]),
                    "label_urgent_gain_bits": float(other_decisions[int(intervention)]["label_urgent_gain_bits"]),
                    "real_payload_gap_bits": float(other_decisions[int(intervention)]["real_payload_gap_bits"]),
                    "real_urgent_gain_bits": float(other_decisions[int(intervention)]["real_urgent_gain_bits"]),
                    "label_source": other_decisions[int(intervention)]["label_source"],
                }
            )
    return pairs


def _integrity(episodes: list[dict], decisions: list[dict]) -> tuple[bool, list[str]]:
    reasons = []
    expected = len(scenarios()) * (1 + len(INTERVENTION_TIMES))
    if len(episodes) != expected or expected > int(MAX_EPISODES):
        reasons.append("the paired episode count is not the shared-control budget")
    if any(int(item["final_t"]) != 48 or item["status"] != "ok" or len(item["steps"]) != 48 or len(item["decisions"]) != 4 for item in episodes):
        reasons.append("an episode did not finish 48 accounted steps")
    if any(float(item["max_ledger_gap_bits"]) > float(EPS_BITS) or float(item["max_accounting_error_bits"]) > float(EPS_BITS) for item in episodes):
        reasons.append("an episode ledger or queue account exceeded 1 bit")
    if any(float(item["payload_accounting_gap_bits"]) > float(EPS_BITS) for item in episodes):
        reasons.append("episode payload and served bits differ by more than 1 bit")
    interventions = [row for row in decisions if int(row["is_intervention"]) == 1]
    if len(interventions) != len(scenarios()) * len(INTERVENTION_TIMES):
        reasons.append("an intervention decision is missing")
    for row in interventions:
        if int(row["label_gap_within_one_bit"]) != 1 or abs(float(row["label_payload_gap_bits"])) > float(EPS_BITS):
            reasons.append("a tie-break label payload gap exceeded 1 bit")
        if int(row["subset_count"]) != 128:
            reasons.append("a tie-break subset did not keep 128 candidates")
        if int(row["arrival_paired"]) != 1 or abs(float(row["real_generated_gap_bits"])) > float(EPS_BITS):
            reasons.append("real-transition copies did not share the exogenous arrival amount")
        if float(row["max_label_ledger_gap_bits"]) > float(EPS_BITS) or float(row["max_label_accounting_error_bits"]) > float(EPS_BITS):
            reasons.append("a label ledger or payload account exceeded 1 bit")
        if row["selection_rule"] != "payload_first_urgency_tiebreak":
            reasons.append("an intervention was not selected by the tie-break rule")
    for scenario in scenarios():
        present = [item for item in episodes if int(item["scenario_id"]) == int(scenario)]
        if len(present) != 3:
            continue
        control = _decisions_for(_lookup(episodes, int(scenario), CONTROL_ARM))
        early = _decisions_for(_lookup(episodes, int(scenario), arm_name(0)))
        middle = _decisions_for(_lookup(episodes, int(scenario), arm_name(30)))
        if early[0]["state_hash"] != control[0]["state_hash"]:
            reasons.append(f"scenario {scenario} intervention-0 does not share the t=0 state")
        for time_step in (0, 15, 30):
            if middle[time_step]["state_hash"] != control[time_step]["state_hash"]:
                reasons.append(f"scenario {scenario} intervention-30 diverged before t={time_step}")
        for time_step in (0, 15):
            if middle[time_step]["executed_assignment"] != control[time_step]["executed_assignment"]:
                reasons.append(f"scenario {scenario} intervention-30 changed the assignment before t=30")
    return not reasons, reasons


def _public_episode(episode: dict) -> dict:
    return {key: episode.get(key, "") for key in EPISODE_FIELDS}


def _public_decision(row: dict) -> dict:
    return {key: row.get(key, "") for key in DECISION_FIELDS}


def execute() -> dict:
    limit_threads()
    before = _watched()
    hashes = _checkpoint_hashes()
    selected = OUTPUT / "selected_candidates.csv"
    if selected.exists():
        selected.unlink()
    planner = FormalPlanner("scheme_c")
    tap = ScoreTap(planner)
    probe_first = 0.0
    probe_second = 0.0
    estimate = 0.0
    episodes = []
    budget_ok = True
    crashed = False
    failure = ""
    failure_trace = ""
    scenario_seconds = None
    try:
        probe_first, probe_second = _probe(planner)
        estimate = _estimate(probe_first, probe_second)
        print(f"scheme C probe {probe_first:.3f}s then {probe_second:.3f}s; estimate {estimate:.3f}s", flush=True)
        if estimate > float(MAX_SECONDS):
            budget_ok = False
            failure = f"estimated {estimate:.3f}s exceeds {MAX_SECONDS:.0f}s"
        else:
            first_scenario_seconds = None
            for scenario in scenarios():
                if first_scenario_seconds is not None:
                    projected = probe_first + probe_second + first_scenario_seconds * len(scenarios())
                    print(f"projected total from the first scenario {projected:.3f}s", flush=True)
                    if projected > float(MAX_SECONDS):
                        budget_ok = False
                        failure = f"after scenario {scenarios()[0]}, projected total {projected:.3f}s exceeds {MAX_SECONDS:.0f}s"
                        break
                scenario_started = time.perf_counter()
                for intervention in (None, 0, 30):
                    print(f"scenario {scenario} arm {arm_name(intervention)}", flush=True)
                    episodes.append(run_episode(int(scenario), intervention, planner, tap))
                elapsed = time.perf_counter() - scenario_started
                print(f"scenario {scenario} finished in {elapsed:.3f}s", flush=True)
                if first_scenario_seconds is None:
                    first_scenario_seconds = elapsed
                    scenario_seconds = elapsed
    except Exception as exc:
        crashed = True
        budget_ok = False
        failure = f"{type(exc).__name__}: {exc}"
        failure_trace = traceback.format_exc()
    decisions = [_public_decision(row) for episode in episodes for row in episode["decisions"]]
    pairs = _attach_curves(episodes) if episodes else []
    steps = [row for episode in episodes for row in episode["steps"]]
    for row in steps:
        row.setdefault("cumulative_payload_delta_vs_control_bits", "")
    mechanical_ok, reasons = _integrity(episodes, [row for episode in episodes for row in episode["decisions"]]) if episodes else (False, ["no finished episode"])
    if failure:
        reasons.append(failure)
        mechanical_ok = False
    after = _watched()
    if before != after:
        mechanical_ok = False
        reasons.append("D0-D3 results or the formal planner file changed")
    by_time = {}
    for intervention in INTERVENTION_TIMES:
        chosen = [row for row in pairs if int(row["intervention_time"]) == int(intervention)]
        by_time[str(intervention)] = chosen
    complete = len(by_time["0"]) == len(scenarios()) and len(by_time["30"]) == len(scenarios())
    if not complete:
        mechanical_ok = False
        if "the paired episode count is not the shared-control budget" not in reasons and episodes:
            reasons.append("both intervention clocks were not paired on every scenario")
    arrival_paired = bool(complete and all(int(row["arrival_paired"]) == 1 for row in decisions if str(row["is_intervention"]) == "1"))
    if complete and mechanical_ok and budget_ok and not failure:
        mechanism = mechanism_name(
            sign_0=payload_sign([float(row["payload_delta_bits"]) for row in by_time["0"]]),
            sign_30=payload_sign([float(row["payload_delta_bits"]) for row in by_time["30"]]),
            arrival_paired=arrival_paired,
        )
    elif (not budget_ok or failure.startswith("estimated") or "projected total" in failure) and not crashed:
        mechanism = "budget"
    else:
        mechanism = "unresolved"
    gate = choose_gate(mechanical_ok=mechanical_ok and not failure, budget_ok=budget_ok and not failure, mechanism=mechanism)
    if failure or not budget_ok:
        gate = GATE_BLOCKED
    next_step = NEXT_STEP.get(mechanism, NEXT_STEP["unresolved"])
    if gate == GATE_BLOCKED and mechanism != "budget":
        next_step = "不要扩大实验，也不要训练 HGNN。先修复完整性或预算失败，再决定是否重跑这一诊断。"
    return {
        "gate": gate,
        "reasons": reasons,
        "failure_trace": failure_trace,
        "mechanism": mechanism,
        "next_step": next_step,
        "oracle_label": ORACLE_LABEL,
        "learned_policy": False,
        "formal_system_modified": False,
        "trained_hgnn": False,
        "lambda_H": None,
        "delta_payload_max": None,
        "d2_gate_unchanged": "HIGHLEVEL_UTILITY_RULE_READY",
        "d3_gate_unchanged": "URGENCY_TIEBREAK_CLOSEDLOOP_PILOT_INCONCLUSIVE",
        "instruction_commit": INSTRUCTION_COMMIT,
        "pilot_seed": PILOT_SEED,
        "scenarios": list(scenarios()),
        "control_shared": True,
        "episode_budget": int(MAX_EPISODES),
        "episodes_run": int(len(episodes)),
        "control_sharing_note": "Scheme C is deterministic, so one control episode per scenario is paired with both interventions. This is 12 episodes inside the 16-episode cap.",
        "probe_first_seconds": float(probe_first),
        "probe_second_seconds": float(probe_second),
        "estimated_seconds": float(estimate),
        "max_seconds": float(MAX_SECONDS),
        "scenario_seconds": scenario_seconds,
        "scheme_c_calls_planned": int(SCHEME_C_CALLS),
        "arrival_pairing": "arrivals_at is a deterministic function of time; capture_state does not copy the reset RNG, and that RNG is not used by later arrivals",
        "diagnostic_previous_action": "zeroed_to_match_label_scorer",
        "sign_0": None if not complete else payload_sign([float(row["payload_delta_bits"]) for row in by_time["0"]]),
        "sign_30": None if not complete else payload_sign([float(row["payload_delta_bits"]) for row in by_time["30"]]),
        "pairs": pairs,
        "payload_delta_mean_bits": {
            str(intervention): None if not by_time[str(intervention)] else _mean([float(row["payload_delta_bits"]) for row in by_time[str(intervention)]])
            for intervention in INTERVENTION_TIMES
        },
        "payload_delta_median_bits": {
            str(intervention): None if not by_time[str(intervention)] else _median([float(row["payload_delta_bits"]) for row in by_time[str(intervention)]])
            for intervention in INTERVENTION_TIMES
        },
        "urgent_delta_mean_bits": {
            str(intervention): None if not by_time[str(intervention)] else _mean([float(row["urgent_delta_bits"]) for row in by_time[str(intervention)]])
            for intervention in INTERVENTION_TIMES
        },
        "checkpoint_sha256": hashes,
        "watched_sha256": before,
        "watched_unchanged": before == after,
        "episodes": [_public_episode(episode) for episode in episodes],
        "decision_rows": decisions,
        "step_rows": steps,
    }


def write_results(summary: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT / "episode_rows.csv", summary["episodes"], EPISODE_FIELDS)
    _write_csv(OUTPUT / "decision_rows.csv", summary["decision_rows"], DECISION_FIELDS)
    _write_csv(OUTPUT / "step_rows.csv", summary["step_rows"], STEP_FIELDS)
    public = {key: value for key, value in summary.items() if key not in {"decision_rows", "step_rows"}}
    (OUTPUT / "summary.json").write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")
    schema = {
        "oracle_label": True,
        "meaning": "One urgency tie-break is inserted into an otherwise formal Scheme C episode. The rule reads frozen-MAPPO labels and does not train or replace Scheme C.",
        "arms": {
            "scheme_c": "formal Scheme C at t=0, 15, 30, and 45",
            "intervention_0": "D3 urgency tie-break only at t=0; Scheme C at t=15, 30, and 45",
            "intervention_30": "Scheme C at t=0 and t=15; D3 urgency tie-break only at t=30; Scheme C at t=45",
        },
        "control_sharing": summary["control_sharing_note"],
        "t0_labels": "reused from the D3 decision log when the live state hash matches; rescored only on a mismatch",
        "t30_labels": "rescored on this arm's own Scheme C live state; D3 future states are not reused",
        "real_transition": "two restored copies keep arrivals and do not choose the candidate; previous action is zero so the comparison isolates arrivals from the zero-arrival label",
        "arrival_pairing": summary["arrival_pairing"],
        "cumulative_delta": "intervention cumulative payload minus the shared Scheme C control; checkpoints 15, 30, 45, and 48 are the totals after the step that reaches that time",
        "eps_bits": float(EPS_BITS),
        "lambda_H": None,
        "delta_payload_max": None,
        "held_out_splits_read": False,
        "instruction_commit": INSTRUCTION_COMMIT,
        "pilot_seed": PILOT_SEED,
        "fields": {
            "episode_rows.csv": list(EPISODE_FIELDS),
            "decision_rows.csv": list(DECISION_FIELDS),
            "step_rows.csv": list(STEP_FIELDS),
        },
    }
    (OUTPUT / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> dict:
    summary = execute()
    write_results(summary)
    print(summary["gate"], summary["mechanism"], summary["estimated_seconds"], summary["episodes_run"], flush=True)
    if summary["failure_trace"]:
        print(summary["failure_trace"], flush=True)
    return summary


if __name__ == "__main__":
    main()
