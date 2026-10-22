"""Paired scheme-C and G1_RATE execution. The official planner is not called."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import load_primary, parameter_token
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6ic_hgnn_smoke.dataset import load_decision
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import assert_open
from wt.benchmark_v2.phase6is_assignment_head.audit import _rss_bytes
from wt.benchmark_v2.phase6iw_inference_adapter.interface import SchemeCAdapter
from wt.benchmark_v2.phase6iy_candidate_adapter.adapter import g1_choice, generate_frozen_pool
from wt.benchmark_v2.phase6iy_candidate_adapter.protocol import config_sha256
from wt.benchmark_v2.phase6iz_closed_loop.protocol import (
    BIT_TIE,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    DECISION_TIME,
    EPOCH6_CKPT,
    HORIZON,
    OUTPUT,
    SCHEME_C_CKPT,
    SMOKE,
    VALIDATION_POOL,
    assert_output_path,
    assign_gate,
)
from wt.benchmark_v2.phase6iz_closed_loop.rollout import execute_assignment, restore_snapshot
from wt.l3b.config import L3BConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_sha256() -> str:
    payload = asdict(L3BConfig())
    payload["step_length_m"] = float(L3BConfig().step_length_m)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _bootstrap(deltas: list[float]) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    data = np.asarray(deltas, dtype=np.float64)
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    count = int(data.size)
    for index in range(BOOTSTRAP_DRAWS):
        draws[index] = float(np.mean(data[rng.integers(0, count, count)]))
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "ci_low": float(low),
        "ci_high": float(high),
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _failures(row: dict) -> list[str]:
    found = []
    if int(row["restore"]["rate_match"]) != 1 or int(row["restore"]["time_step"]) != DECISION_TIME:
        found.append("restore")
    for name in ("g1", "scheme_c"):
        arm = row[name]
        if int(arm["steps_executed"]) != HORIZON or int(arm["accounting_ok"]) != 1 or int(arm["payload_matches_sum"]) != 1:
            found.append(f"{name} accounting")
        if int(arm["target_switches"]) != 0 or int(arm["rng_unchanged"]) != 1:
            found.append(f"{name} control")
    if int(row["same_assignment"]) == 1 and abs(float(row["delta_bits"])) > BIT_TIE:
        found.append("paired mismatch")
    if not np.isfinite(float(row["delta_bits"])):
        found.append("nonfinite")
    return found


def _one(scenario: int, actor, scorer: SchemeCAdapter) -> dict:
    assert_open(int(scenario), "validation")
    started = time.perf_counter()
    pool = generate_frozen_pool(int(scenario), DECISION_TIME)
    sample = load_decision(int(scenario), DECISION_TIME)
    score_started = time.perf_counter()
    scored = scorer.infer(sample["graph"], pool.assignment_matrix())
    score_seconds = time.perf_counter() - score_started
    scheme = list(pool.assignment_at(scored.chosen_index))
    if scheme != scored.chosen_assignment:
        raise RuntimeError("scheme-C score index is not bound to the executed assignment")
    baseline = [int(item) for item in g1_choice(int(scenario), DECISION_TIME, pool)]
    arms = {}
    for name, assignment in (("g1", baseline), ("scheme_c", scheme)):
        env, restore = restore_snapshot(int(scenario), DECISION_TIME)
        torch.manual_seed(1_000_000 + int(scenario))
        run_started = time.perf_counter()
        result = execute_assignment(env, actor, np.asarray(assignment, dtype=np.int64))
        result["seconds"] = float(time.perf_counter() - run_started)
        result["restore"] = restore
        arms[name] = result
    delta = float(arms["scheme_c"]["delivered_bits"] - arms["g1"]["delivered_bits"])
    return {
        "scenario_id": int(scenario),
        "time_step": DECISION_TIME,
        "pool_count": int(pool.assignment_matrix().shape[0]),
        "state_hash": pool.state_hash,
        "same_assignment": int(scheme == baseline),
        "scheme_c_assignment": scheme,
        "g1_assignment": baseline,
        "scheme_score_seconds": float(score_seconds),
        "delta_bits": delta,
        "restore": arms["g1"]["restore"],
        "g1": {key: value for key, value in arms["g1"].items() if key != "restore"},
        "scheme_c": {key: value for key, value in arms["scheme_c"].items() if key != "restore"},
        "selection_seconds": float(time.perf_counter() - started),
    }


def run_audit() -> dict:
    before = {
        "scheme_c": _sha256(SCHEME_C_CKPT),
        "epoch6": _sha256(EPOCH6_CKPT),
        "mappo": _sha256(primary_checkpoint()),
        "adapter_config": config_sha256(),
        "environment": _config_sha256(),
    }
    actor = load_primary()
    token = parameter_token(actor)
    scorer = SchemeCAdapter()
    scorer.checkpoint_sha256 = before["scheme_c"]
    rows = []
    for scenario in SMOKE:
        rows.append(_one(int(scenario), actor, scorer))
        print(f"smoke {scenario} delta={rows[-1]['delta_bits']:.3f} failures={_failures(rows[-1])}", flush=True)
    smoke_failures = [item for row in rows for item in _failures(row)]
    if not smoke_failures:
        for scenario in VALIDATION_POOL:
            if int(scenario) in SMOKE:
                continue
            rows.append(_one(int(scenario), actor, scorer))
            print(f"scenario {scenario} delta={rows[-1]['delta_bits']:.3f}", flush=True)
    failures = [item for row in rows for item in _failures(row)]
    if parameter_token(actor) != token:
        failures.append("actor contamination")
    if len(rows) != len(VALIDATION_POOL):
        failures.append("incomplete")
    deltas = [float(row["delta_bits"]) for row in rows]
    stats = _bootstrap(deltas) if len(deltas) == len(VALIDATION_POOL) and not failures else {
        "mean": float(np.mean(deltas)) if deltas else 0.0,
        "median": float(np.median(deltas)) if deltas else 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0,
        "draws": 0,
        "seed": BOOTSTRAP_SEED,
    }
    after = {
        "scheme_c": _sha256(SCHEME_C_CKPT),
        "epoch6": _sha256(EPOCH6_CKPT),
        "mappo": _sha256(primary_checkpoint()),
        "adapter_config": config_sha256(),
        "environment": _config_sha256(),
    }
    if before != after:
        failures.append("hash changed")
    gate = assign_gate(failures=len(failures), ci_low=float(stats["ci_low"]), ci_high=float(stats["ci_high"]))
    positive = int(sum(float(row["delta_bits"]) > BIT_TIE for row in rows))
    negative = int(sum(float(row["delta_bits"]) < -BIT_TIE for row in rows))
    tied = int(len(rows) - positive - negative)
    result = {
        "gate": gate,
        "failures": failures,
        "checkpoint_hashes": before,
        "hashes_unchanged": before == after,
        "environment_config_sha256": before["environment"],
        "states": int(len(rows)),
        "same_assignment_states": int(sum(int(row["same_assignment"]) for row in rows)),
        "g1_mean_bits": float(np.mean([row["g1"]["delivered_bits"] for row in rows])) if rows else 0.0,
        "g1_median_bits": float(np.median([row["g1"]["delivered_bits"] for row in rows])) if rows else 0.0,
        "scheme_mean_bits": float(np.mean([row["scheme_c"]["delivered_bits"] for row in rows])) if rows else 0.0,
        "scheme_median_bits": float(np.median([row["scheme_c"]["delivered_bits"] for row in rows])) if rows else 0.0,
        "paired": stats,
        "positive_states": positive,
        "tie_states": tied,
        "negative_states": negative,
        "bit_tie": BIT_TIE,
        "arrival_policy": "environment arrivals_at; not the zero-arrival J15 label",
        "payload_definition": "sum of served_bits from apply_service_then_arrival",
        "planner_modified": 0,
        "reserved_scenarios_read": 0,
        "rss_bytes": _rss_bytes(),
        "planner_ready": False,
        "deployment_verified": False,
        "rows": rows,
    }
    folder = assert_output_path(OUTPUT)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.json").write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
    print(json.dumps({"gate": gate, "failures": failures, "ci": [stats["ci_low"], stats["ci_high"]]}, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    run_audit()
