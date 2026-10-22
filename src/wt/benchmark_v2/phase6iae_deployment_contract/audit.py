"""Normal-path regression after the fault-handling change. Holdout stays closed."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from wt.benchmark_v2.phase6f_system_bringup.episode import run_active_episode
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import limit_threads
from wt.benchmark_v2.phase6iaa_formal_planner.faults import rollback_to_baseline
from wt.benchmark_v2.phase6iaa_formal_planner.protocol import MAPPO_SHA256, POLICY_FILE, read_policy, write_policy
from wt.benchmark_v2.phase6iae_deployment_contract.protocol import (
    BASELINE_SCENARIO,
    BASELINE_STOP,
    OUTPUT,
    PAYLOAD_TOLERANCE_BITS,
    REGRESSION_PAYLOAD_BITS,
    REGRESSION_SCENARIO,
    REGRESSION_T0_ASSIGNMENT,
    REGRESSION_T0_INDEX,
    assert_output_path,
    assign_gate,
    load_requirements,
    pending_requirements,
    status_flags,
)


def _scheme_faults(row: dict) -> list[str]:
    faults = []
    if not row["complete"] or row["fault_recovery"] is not None:
        faults.append("scheme episode did not finish cleanly")
    if int(row["scheme_c_calls"]) != 4 or int(row["payload_matches_served"]) != 1:
        faults.append("scheme call or accounting")
    if abs(float(row["payload_bits"]) - float(row["served_bits"])) > PAYLOAD_TOLERANCE_BITS:
        faults.append("scheme payload mismatch")
    if abs(float(row["payload_bits"]) - REGRESSION_PAYLOAD_BITS) > PAYLOAD_TOLERANCE_BITS:
        faults.append("scheme payload drifted")
    if int(row["illegal"]) != 0 or int(row["nan_inf"]) != 0 or int(row["accounting_failures"]) != 0 or int(row["anomaly_count"]) != 0:
        faults.append("scheme execution fault")
    first = row["decisions"][0]
    if int(first["score_index"]) != REGRESSION_T0_INDEX or tuple(first["assignment"]) != REGRESSION_T0_ASSIGNMENT:
        faults.append("scheme t0 choice drifted")
    if int(first["candidate_count"]) != 13327 or int(first["scheme_c_called"]) != 1:
        faults.append("scheme pool binding")
    return faults


def _baseline_faults(row: dict) -> list[str]:
    faults = []
    if not row["complete"] or row["fault_recovery"] is not None:
        faults.append("baseline episode did not finish cleanly")
    if int(row["scheme_c_calls"]) != 0 or int(row["payload_matches_served"]) != 1:
        faults.append("baseline call or accounting")
    if abs(float(row["payload_bits"]) - float(row["served_bits"])) > PAYLOAD_TOLERANCE_BITS:
        faults.append("baseline payload mismatch")
    if int(row["illegal"]) != 0 or int(row["nan_inf"]) != 0 or int(row["accounting_failures"]) != 0:
        faults.append("baseline execution fault")
    if int(row["decisions"][0]["candidate_count"]) != 13327 or int(row["decisions"][0]["scheme_c_called"]) != 0:
        faults.append("baseline pool binding")
    return faults


def run_audit() -> dict:
    limit_threads()
    document = load_requirements()
    pending = pending_requirements(document)
    official_before = POLICY_FILE.read_text(encoding="utf-8")
    faults: list[str] = []
    scheme_row = None
    baseline_row = None
    try:
        scheme_row = run_active_episode(REGRESSION_SCENARIO)
        faults.extend(_scheme_faults(scheme_row))
        if not faults:
            with tempfile.TemporaryDirectory() as folder:
                policy = Path(folder) / "active_policy.json"
                write_policy("scheme_c", policy)
                recovery = rollback_to_baseline(policy)
                confirmed = json.loads(policy.read_text(encoding="utf-8"))
                if read_policy(policy) != "g1_rate" or confirmed.get("generator") != "G1_RATE" or int(confirmed.get("k", -1)) != 6:
                    faults.append("baseline contract")
                elif recovery.get("mappo_sha256") != MAPPO_SHA256:
                    faults.append("baseline mappo hash")
                else:
                    baseline_row = run_active_episode(BASELINE_SCENARIO, stop_time=BASELINE_STOP, policy_path=str(policy))
                    faults.extend(_baseline_faults(baseline_row))
    except Exception as exc:
        faults.append(f"exception {type(exc).__name__}: {exc}")
    official_after = POLICY_FILE.read_text(encoding="utf-8")
    if official_before != official_after:
        faults.append("official policy changed")
    regression_ok = not faults and scheme_row is not None and baseline_row is not None
    gate = assign_gate(pending=len(pending), fault_failures=0 if regression_ok else 1, regression_ok=regression_ok)
    planner_ready = bool(regression_ok and read_policy() == "scheme_c")
    flags = status_flags(gate=gate, planner_ready=planner_ready)
    result = {
        "gate": gate,
        "faults": faults,
        "pending_confirmation": pending,
        "requirements_deployment_verified": document["deployment_verified"],
        "scheme_episode": None if scheme_row is None else _public(scheme_row),
        "baseline_episode": None if baseline_row is None else _public(baseline_row),
        "official_policy": read_policy(),
        "reserved_scenarios_executed": 0,
        "trained": False,
        "production_deployment_executed": False,
        **flags,
    }
    folder = assert_output_path(OUTPUT)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"gate": gate, "faults": faults, "planner_ready": flags["planner_ready"], "deployment_verified": flags["deployment_verified"]}, sort_keys=True), flush=True)
    return result


def _public(row: dict) -> dict:
    return {
        "scenario_id": int(row["scenario_id"]),
        "policy": row["policy"],
        "complete": bool(row["complete"]),
        "payload_bits": float(row["payload_bits"]),
        "served_bits": float(row["served_bits"]),
        "payload_matches_served": int(row["payload_matches_served"]),
        "scheme_c_calls": int(row["scheme_c_calls"]),
        "terminal_time": int(row["terminal_time"]),
        "decision_times": [int(item) for item in row["decision_times"]],
        "anomaly_count": int(row["anomaly_count"]),
        "illegal": int(row["illegal"]),
        "accounting_failures": int(row["accounting_failures"]),
        "nan_inf": int(row["nan_inf"]),
        "fault_recovery": row["fault_recovery"],
        "first_decision": {
            "score_index": int(row["decisions"][0]["score_index"]),
            "assignment": [int(item) for item in row["decisions"][0]["assignment"]],
            "candidate_count": int(row["decisions"][0]["candidate_count"]),
            "scheme_c_called": int(row["decisions"][0]["scheme_c_called"]),
        },
    }


if __name__ == "__main__":
    run_audit()
