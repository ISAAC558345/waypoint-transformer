"""Interface, smoke, and limited episodes. Phase-6I-Z files are not rewritten."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6f_system_bringup.episode import _contamination, _fingerprint, run_active_episode
from wt.benchmark_v2.phase6f_system_bringup.protocol import CANDIDATE_K
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import limit_threads, parameter_token
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6iaa_formal_planner.planner import FormalPlanner, graph_from_env
from wt.benchmark_v2.phase6iaa_formal_planner.protocol import (
    ADAPTER_CONFIG_SHA256,
    EPOCH6_CKPT,
    EPOCH6_SHA256,
    EPISODE_SCENARIOS,
    MAPPO_SHA256,
    OUTPUT,
    SCHEME_C_CKPT,
    SCHEME_C_SHA256,
    SMOKE_SCENARIOS,
    assert_output_path,
    assign_gate,
    status_flags,
    write_policy,
)
from wt.benchmark_v2.phase6ic_hgnn_smoke.dataset import load_decision
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import GRAPH_KEYS
from wt.benchmark_v2.phase6iy_candidate_adapter.adapter import g1_choice, generate_frozen_pool
from wt.benchmark_v2.phase6iy_candidate_adapter.protocol import config_sha256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hashes() -> dict:
    return {
        "scheme_c": _sha256(SCHEME_C_CKPT),
        "epoch6": _sha256(EPOCH6_CKPT),
        "mappo": _sha256(primary_checkpoint()),
        "adapter_config": config_sha256(),
    }


def _expected(found: dict) -> list[str]:
    checks = {
        "scheme_c": SCHEME_C_SHA256,
        "epoch6": EPOCH6_SHA256,
        "mappo": MAPPO_SHA256,
        "adapter_config": ADAPTER_CONFIG_SHA256,
    }
    return [name for name, digest in checks.items() if found[name] != digest]


def _interface(planner: FormalPlanner) -> list[str]:
    failures = []
    if int(CANDIDATE_K) != 5:
        failures.append("historical k changed")
    if planner.policy != "scheme_c" or planner.scorer is None:
        return ["scheme C was not loaded"]
    env = L3BBitQueueEnvV2()
    env.reset(seed=322000)
    position_prev = np.asarray(env.uav_positions, dtype=np.float64).copy()
    before = _fingerprint(env, None)
    token = parameter_token(planner.actor)
    graph = graph_from_env(env, position_prev, None)
    saved = load_decision(322000, 0)
    for key in GRAPH_KEYS:
        if not np.allclose(graph[key], saved["graph"][key], atol=1.0e-5, rtol=0.0):
            failures.append(f"graph {key}")
    pool = generate_frozen_pool(322000, 0)
    selection = planner.select_assignment(env, position_prev, None)
    scored = planner.scorer.infer(saved["graph"], pool.assignment_matrix())
    bound = tuple(pool.assignment_at(scored.chosen_index))
    if selection.status != "selected" or selection.assignment != bound or list(selection.assignment) != scored.chosen_assignment:
        failures.append("scheme C choice is not bound to the frozen pool")
    if int(selection.candidate_count) != int(pool.assignment_matrix().shape[0]):
        failures.append("candidate count")
    if _contamination(before, _fingerprint(env, None)) != 0 or parameter_token(planner.actor) != token:
        failures.append("contamination")
    fresh = L3BBitQueueEnvV2()
    fresh.reset(seed=322000)
    baseline = FormalPlanner("g1_rate")
    baseline_choice = baseline.select_assignment(fresh, fresh.uav_positions.copy(), None)
    if baseline.scheme_c_calls != 0 or baseline_choice.assignment != tuple(g1_choice(322000, 0, pool)):
        failures.append("g1 rollback choice")
    return failures


def _episode_ok(row: dict, times: list[int], policy: str) -> bool:
    return bool(
        row["complete"]
        and row["policy"] == policy
        and row["decision_times"] == times
        and row["anomaly_count"] == 0
        and row["illegal"] == 0
        and row["accounting_failures"] == 0
        and row["nan_inf"] == 0
        and row["payload_matches_served"] == 1
    )


def run_audit() -> dict:
    limit_threads()
    before = _hashes()
    failures = [f"hash {name}" for name in _expected(before)]
    write_policy("scheme_c")
    interface_failures: list[str] = []
    smoke_rows = []
    episode_rows = []
    paired_rows = []
    scheme_calls = 0
    if not failures:
        planner = FormalPlanner("scheme_c")
        interface_failures = _interface(planner)
        failures.extend(interface_failures)
    if not failures:
        for scenario in SMOKE_SCENARIOS:
            row = run_active_episode(int(scenario), stop_time=15)
            smoke_rows.append(row)
            scheme_calls += int(row["scheme_c_calls"])
            print(f"smoke {scenario} complete={row['complete']} payload={row['payload_bits']:.3f}", flush=True)
            if not _episode_ok(row, [0], "scheme_c") or int(row["scheme_c_calls"]) != 1:
                failures.append(f"smoke {scenario}")
    if not failures:
        for scenario in EPISODE_SCENARIOS:
            row = run_active_episode(int(scenario))
            episode_rows.append(row)
            scheme_calls += int(row["scheme_c_calls"])
            print(f"episode {scenario} complete={row['complete']} payload={row['payload_bits']:.3f}", flush=True)
            if not _episode_ok(row, [0, 15, 30, 45], "scheme_c") or int(row["scheme_c_calls"]) != 4:
                failures.append(f"episode {scenario}")
    if not failures:
        write_policy("g1_rate")
        for scenario in EPISODE_SCENARIOS:
            row = run_active_episode(int(scenario))
            paired_rows.append(row)
            print(f"baseline {scenario} complete={row['complete']} payload={row['payload_bits']:.3f}", flush=True)
            if not _episode_ok(row, [0, 15, 30, 45], "g1_rate") or int(row["scheme_c_calls"]) != 0:
                failures.append(f"baseline {scenario}")
    after = _hashes()
    if before != after:
        failures.append("hash changed")
    gate = assign_gate(
        interface_ok=not interface_failures and not any(item.startswith("hash ") for item in failures),
        smoke_ok=len(smoke_rows) == len(SMOKE_SCENARIOS) and all(_episode_ok(row, [0], "scheme_c") for row in smoke_rows),
        episode_ok=len(episode_rows) == len(EPISODE_SCENARIOS) and all(_episode_ok(row, [0, 15, 30, 45], "scheme_c") for row in episode_rows),
        scheme_c_called=scheme_calls > 0,
        failures=len(failures),
    )
    flags = status_flags(gate)
    write_policy("scheme_c" if flags["planner_ready"] else "g1_rate")
    paired = []
    for left, right in zip(episode_rows, paired_rows):
        paired.append(
            {
                "scenario_id": int(left["scenario_id"]),
                "scheme_c_payload_bits": float(left["payload_bits"]),
                "g1_payload_bits": float(right["payload_bits"]),
                "delta_bits": float(left["payload_bits"] - right["payload_bits"]),
            }
        )
    result = {
        "gate": gate,
        "failures": failures,
        "startup_hashes": before,
        "hashes_unchanged": before == after,
        "historical_candidate_k": int(CANDIDATE_K),
        "active_candidate_k": 6,
        "high_level_policy": "scheme_c" if flags["planner_ready"] else "g1_rate",
        "scheme_c_calls": int(scheme_calls),
        "smoke": smoke_rows,
        "episodes": episode_rows,
        "baseline_episodes": paired_rows,
        "paired_payload": paired,
        "closed_loop_gain_gate": "CLOSED_LOOP_GAIN_INCONCLUSIVE",
        "closed_loop_gain_replaced": 0,
        "reserved_scenarios_read": 0,
        "trained": False,
        **flags,
    }
    folder = assert_output_path(OUTPUT)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"gate": gate, "failures": failures, "planner_ready": flags["planner_ready"], "deployment_verified": False}, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    run_audit()
