"""Paired full episodes. Official active_policy.json is left unchanged unless scheme C faults."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from wt.benchmark_v2.phase6f_system_bringup.episode import run_active_episode
from wt.benchmark_v2.phase6f_system_bringup.protocol import CANDIDATE_K
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import limit_threads
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6iaa_formal_planner.protocol import (
    ACTIVE_CANDIDATE_K,
    POLICY_FILE,
    SCHEME_C_CKPT,
    SCHEME_C_SHA256,
    MAPPO_SHA256,
    read_policy,
    write_policy,
)
from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import GENERATOR, HOLD, SELECTED_K
from wt.benchmark_v2.phase6iab_planner_stability.protocol import (
    BIT_TIE,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    DECISION_TIMES,
    HORIZON,
    OUTPUT,
    SCENARIOS,
    SMOKE,
    TORCH_SEED_OFFSET,
    assert_output_path,
    assert_scenario,
    assign_gate,
)
from wt.benchmark_v2.phase6iy_candidate_adapter.protocol import config_sha256
from wt.l3b.config import L3BConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _precheck(official_bytes: bytes) -> list[str]:
    failures = []
    root = Path(__file__).resolve().parents[4]
    integration = root / "docs/archive/phase_reports/HTW_Phase6I_FormalPlannerSchemeCIntegration_CN.md"
    closed_loop = root / "docs/archive/phase_reports/HTW_Phase6I_ClosedLoopExecutionGainAudit_CN.md"
    if not integration.is_file() or "PLANNER_INTEGRATION_READY" not in integration.read_text(encoding="utf-8"):
        failures.append("integration report")
    if not closed_loop.is_file() or "CLOSED_LOOP_GAIN_INCONCLUSIVE" not in closed_loop.read_text(encoding="utf-8"):
        failures.append("closed-loop report")
    if not POLICY_FILE.is_file():
        return failures + ["official policy missing"]
    official = json.loads(official_bytes.decode("utf-8"))
    if official.get("high_level_policy") != "scheme_c" or official.get("generator") != "G1_RATE" or int(official.get("k", 0)) != 6:
        failures.append("official policy is not scheme C on G1_RATE k=6")
    if int(SELECTED_K) != 6 or GENERATOR != "G1_RATE" or int(ACTIVE_CANDIDATE_K) != 6 or int(HOLD) != 15:
        failures.append("candidate contract")
    if int(CANDIDATE_K) != 5:
        failures.append("historical k=5 constant changed")
    if _sha256(SCHEME_C_CKPT) != SCHEME_C_SHA256 or SCHEME_C_CKPT.name != "epoch_001.pt":
        failures.append("scheme C checkpoint")
    if _sha256(primary_checkpoint()) != MAPPO_SHA256:
        failures.append("mappo checkpoint")
    config = L3BConfig()
    if int(config.episode_horizon) != HORIZON or int(config.num_uavs) != 6 or int(config.num_buoys) != 12:
        failures.append("environment contract")
    planner_text = (root / "src" / "wt" / "benchmark_v2" / "phase6iaa_formal_planner" / "planner.py").read_text(encoding="utf-8")
    if "frozen_candidate_keys" not in planner_text or "load_primary" not in planner_text or '("G1_RATE", 5)' in planner_text:
        failures.append("planner call chain")
    if "env.reset(seed=int(scenario_id))" not in planner_text or "payload_bits" not in planner_text:
        failures.append("reset or payload accounting")
    return failures


def _execution_fault(row: dict) -> bool:
    return bool(
        not row["complete"]
        or int(row["illegal"]) > 0
        or int(row["accounting_failures"]) > 0
        or int(row["nan_inf"]) > 0
        or int(row["anomaly_count"]) > 0
        or int(row["payload_matches_served"]) != 1
        or float(row["max_conservation_error"]) > 1.0
    )


def _arm_problems(row: dict, expected_calls: int) -> list[str]:
    found = []
    if _execution_fault(row):
        found.append("execution")
    if int(row["scheme_c_calls"]) != int(expected_calls):
        found.append("call count")
    if [int(item) for item in row["decision_times"]] != list(DECISION_TIMES) or int(row["terminal_time"]) != HORIZON:
        found.append("termination")
    if int(row["stop_time"]) != HORIZON:
        found.append("horizon")
    return found


def _bootstrap(deltas: list[float]) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    data = np.asarray(deltas, dtype=np.float64)
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    count = int(data.size)
    for index in range(BOOTSTRAP_DRAWS):
        draws[index] = float(np.mean(data[rng.integers(0, count, count)]))
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"mean": float(np.mean(data)), "median": float(np.median(data)), "ci_low": float(low), "ci_high": float(high), "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED}


def _run_arm(scenario: int, policy_path: Path) -> dict:
    assert_scenario(int(scenario))
    torch.manual_seed(TORCH_SEED_OFFSET + int(scenario))
    row = run_active_episode(int(scenario), policy_path=str(policy_path))
    row["torch_seed"] = int(TORCH_SEED_OFFSET + int(scenario))
    row["policy_file"] = str(policy_path)
    return row


def _pair(scenario: int, scheme_path: Path, baseline_path: Path) -> dict:
    scheme = _run_arm(scenario, scheme_path)
    baseline = _run_arm(scenario, baseline_path)
    scheme_hash = scheme["decisions"][0]["pool_hash"] if scheme["decisions"] else ""
    baseline_hash = baseline["decisions"][0]["pool_hash"] if baseline["decisions"] else ""
    paired_rng = int(scheme["rng_unchanged"]) == 1 and int(baseline["rng_unchanged"]) == 1 and scheme["rng_after_reset"] == baseline["rng_after_reset"]
    return {
        "scenario_id": int(scenario),
        "reset_seed": int(scenario),
        "torch_seed": int(TORCH_SEED_OFFSET + int(scenario)),
        "delta_payload": float(scheme["payload_bits"] - baseline["payload_bits"]),
        "t0_pool_hash_match": int(scheme_hash != "" and scheme_hash == baseline_hash),
        "rng_strictly_paired": int(paired_rng),
        "scheme_c": scheme,
        "g1_rate": baseline,
    }


def _pair_problems(pair: dict) -> list[str]:
    found = []
    if int(pair["t0_pool_hash_match"]) != 1:
        found.append("t0 pool")
    found.extend(f"scheme_c {item}" for item in _arm_problems(pair["scheme_c"], 4))
    found.extend(f"g1_rate {item}" for item in _arm_problems(pair["g1_rate"], 0))
    return found


def _rollback(official_bytes: bytes) -> list[str]:
    write_policy("g1_rate")
    problems = []
    if read_policy() != "g1_rate":
        problems.append("policy did not change")
    current = json.loads(POLICY_FILE.read_text(encoding="utf-8"))
    if current.get("generator") != "G1_RATE" or int(current.get("k", 0)) != 6:
        problems.append("rollback is not G1_RATE k=6")
    if POLICY_FILE.read_bytes() == official_bytes:
        problems.append("official file was not updated")
    checked = run_active_episode(322048, stop_time=1)
    if checked["policy"] != "g1_rate" or int(checked["scheme_c_calls"]) != 0:
        problems.append("entry did not use the baseline")
    if not checked["decisions"] or int(checked["decisions"][0]["candidate_count"]) < 1:
        problems.append("baseline pool was empty")
    startup_mappo = _sha256(primary_checkpoint())
    if startup_mappo != MAPPO_SHA256:
        problems.append("rollback mappo hash")
    return problems


def _mean(rows: list[dict], arm: str, key: str) -> float:
    return float(np.mean([row[arm][key] for row in rows])) if rows else 0.0


def run_audit() -> dict:
    limit_threads()
    official_bytes = POLICY_FILE.read_bytes()
    failures = _precheck(official_bytes)
    folder = assert_output_path(OUTPUT)
    folder.mkdir(parents=True, exist_ok=True)
    scheme_path = assert_output_path(folder / "config_scheme_c.json")
    baseline_path = assert_output_path(folder / "config_g1_rate.json")
    if not failures:
        write_policy("scheme_c", scheme_path)
        write_policy("g1_rate", baseline_path)
        if POLICY_FILE.read_bytes() != official_bytes:
            failures.append("official config changed while copying")
        if read_policy(scheme_path) != "scheme_c" or read_policy(baseline_path) != "g1_rate":
            failures.append("experiment copies")
    pairs = []
    rolled_back = False
    rollback_problems: list[str] = []
    if not failures:
        for scenario in list(SMOKE) + [item for item in SCENARIOS if item not in SMOKE]:
            if scenario not in SMOKE and any(_pair_problems(pair) for pair in pairs):
                failures.append("smoke accounting")
                break
            try:
                pair = _pair(int(scenario), scheme_path, baseline_path)
            except Exception as exc:
                failures.append(f"episode {scenario}: {exc}")
                break
            pairs.append(pair)
            (folder / "episodes").mkdir(parents=True, exist_ok=True)
            (folder / "episodes" / f"scenario_{int(scenario)}.json").write_text(json.dumps(pair, indent=2), encoding="utf-8")
            print(
                f"scenario {scenario} delta={pair['delta_payload']:.3f} problems={_pair_problems(pair)}",
                flush=True,
            )
            if _execution_fault(pair["scheme_c"]):
                failures.append(f"scheme C execution {scenario}")
                rolled_back = True
                rollback_problems = _rollback(official_bytes)
                failures.extend(f"rollback {item}" for item in rollback_problems)
                break
            if _pair_problems(pair):
                failures.append(f"pair {scenario}")
                break
    if POLICY_FILE.read_bytes() != official_bytes and not rolled_back:
        failures.append("official config changed")
    if len(pairs) != len(SCENARIOS):
        failures.append("incomplete")
    clean = len(pairs) == len(SCENARIOS) and not failures
    deltas = [float(pair["delta_payload"]) for pair in pairs]
    stats = _bootstrap(deltas) if clean else {"mean": float(np.mean(deltas)) if deltas else 0.0, "median": float(np.median(deltas)) if deltas else 0.0, "ci_low": 0.0, "ci_high": 0.0, "draws": 0, "seed": BOOTSTRAP_SEED}
    gate = assign_gate(failures=len(failures), complete_pairs=len(pairs) if clean else 0, ci_low=float(stats["ci_low"]), ci_high=float(stats["ci_high"]))
    positive = int(sum(float(pair["delta_payload"]) > BIT_TIE for pair in pairs))
    negative = int(sum(float(pair["delta_payload"]) < -BIT_TIE for pair in pairs))
    official_now = json.loads(POLICY_FILE.read_text(encoding="utf-8"))
    result = {
        "gate": gate,
        "failures": failures,
        "rolled_back": int(rolled_back),
        "rollback_problems": rollback_problems,
        "official_policy": official_now.get("high_level_policy"),
        "official_k": int(official_now.get("k", 0)),
        "official_unchanged": int(POLICY_FILE.read_bytes() == official_bytes),
        "scheme_c_sha256": _sha256(SCHEME_C_CKPT),
        "mappo_sha256": _sha256(primary_checkpoint()),
        "adapter_config_sha256": config_sha256(),
        "pairs": int(len(pairs)),
        "scheme_mean_payload": _mean(pairs, "scheme_c", "payload_bits"),
        "scheme_median_payload": float(np.median([pair["scheme_c"]["payload_bits"] for pair in pairs])) if pairs else 0.0,
        "g1_mean_payload": _mean(pairs, "g1_rate", "payload_bits"),
        "g1_median_payload": float(np.median([pair["g1_rate"]["payload_bits"] for pair in pairs])) if pairs else 0.0,
        "paired": stats,
        "positive_scenarios": positive,
        "tie_scenarios": int(len(pairs) - positive - negative),
        "negative_scenarios": negative,
        "bit_tie": BIT_TIE,
        "rng_strictly_paired_scenarios": int(sum(int(pair["rng_strictly_paired"]) for pair in pairs)),
        "scheme_mean_blocks": _mean(pairs, "scheme_c", "completed_blocks"),
        "g1_mean_blocks": _mean(pairs, "g1_rate", "completed_blocks"),
        "scheme_mean_queue": _mean(pairs, "scheme_c", "queue_bits"),
        "g1_mean_queue": _mean(pairs, "g1_rate", "queue_bits"),
        "scheme_mean_seconds": _mean(pairs, "scheme_c", "seconds"),
        "g1_mean_seconds": _mean(pairs, "g1_rate", "seconds"),
        "scheme_mean_high_level_seconds": _mean(pairs, "scheme_c", "high_level_seconds"),
        "g1_mean_high_level_seconds": _mean(pairs, "g1_rate", "high_level_seconds"),
        "anomaly_episodes": int(sum(int(pair["scheme_c"]["anomaly_count"]) + int(pair["g1_rate"]["anomaly_count"]) > 0 for pair in pairs)),
        "closed_loop_gain_gate": "CLOSED_LOOP_GAIN_INCONCLUSIVE",
        "validation_is_development_set": 1,
        "reserved_scenarios_read": 0,
        "trained": False,
        "planner_ready": int(official_now.get("generator") == "G1_RATE" and int(official_now.get("k", 0)) == 6 and not rollback_problems),
        "deployment_verified": False,
        "rows": pairs,
    }
    (folder / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"ci": [stats["ci_low"], stats["ci_high"]], "failures": failures, "gate": gate}, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    run_audit()
