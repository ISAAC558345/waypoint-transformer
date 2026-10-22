"""Frozen holdout comparison. The protocol file is verified before any test episode."""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from wt.benchmark_v2.phase6f_system_bringup.episode import run_active_episode
from wt.benchmark_v2.phase6f_system_bringup.protocol import CANDIDATE_K
from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import ROOT
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import limit_threads
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6iaa_formal_planner.protocol import POLICY_FILE, read_policy, write_policy
from wt.benchmark_v2.phase6ib_highlevel_dataset.hgnn_revision import FROZEN_DATASET, REVISION_DATASET
from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import GENERATOR, HOLD, SELECTED_K, TRAIN_POOL, VALIDATION_POOL
from wt.benchmark_v2.phase6iac_independent_holdout.protocol import (
    ADAPTER_CONFIG_SHA256,
    BIT_TIE,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_HIGH,
    BOOTSTRAP_LOW,
    BOOTSTRAP_SEED,
    CODE_FILES,
    DECISION_TIMES,
    ENVIRONMENT_SHA256,
    EPOCH6_CKPT,
    EPOCH6_SHA256,
    HORIZON,
    MAPPO_SHA256,
    OUTPUT,
    PROTOCOL_FILE,
    SCENARIOS,
    SCHEME_C_CKPT,
    SCHEME_C_SHA256,
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


def _environment_sha256() -> str:
    payload = asdict(L3BConfig())
    payload["step_length_m"] = float(L3BConfig().step_length_m)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _code_hashes() -> dict[str, str]:
    return {name: _sha256(ROOT / name) for name in CODE_FILES}


def _git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _shard_exists(scenario: int) -> bool:
    name = f"scenario_{int(scenario)}"
    return any((root / split / name).is_dir() for root in (FROZEN_DATASET, REVISION_DATASET) for split in ("train", "validation"))


def write_protocol() -> dict:
    """Record the frozen contract before the first holdout episode."""
    mismatches = []
    if _sha256(SCHEME_C_CKPT) != SCHEME_C_SHA256 or SCHEME_C_CKPT.name != "epoch_001.pt":
        mismatches.append("scheme C checkpoint")
    if _sha256(EPOCH6_CKPT) != EPOCH6_SHA256:
        mismatches.append("epoch 6 checkpoint")
    if _sha256(primary_checkpoint()) != MAPPO_SHA256:
        mismatches.append("mappo checkpoint")
    if config_sha256() != ADAPTER_CONFIG_SHA256:
        mismatches.append("candidate adapter")
    if _environment_sha256() != ENVIRONMENT_SHA256:
        mismatches.append("environment config")
    if mismatches:
        raise RuntimeError("protocol freeze stopped: " + ", ".join(mismatches))
    official = json.loads(POLICY_FILE.read_text(encoding="utf-8"))
    payload = {
        "high_level_checkpoint": str(SCHEME_C_CKPT),
        "scheme_c_sha256": SCHEME_C_SHA256,
        "epoch6_sha256": EPOCH6_SHA256,
        "mappo_checkpoint": str(primary_checkpoint()),
        "mappo_sha256": MAPPO_SHA256,
        "adapter_config_sha256": ADAPTER_CONFIG_SHA256,
        "environment_sha256": ENVIRONMENT_SHA256,
        "generator": GENERATOR,
        "k": int(SELECTED_K),
        "historical_formal_k": int(CANDIDATE_K),
        "hold": int(HOLD),
        "horizon": HORIZON,
        "decision_times": list(DECISION_TIMES),
        "candidate_builder": "frozen_candidate_keys, the live path inside the Phase-6I-Y adapter used by generate_frozen_pool",
        "episode_entry": "run_active_episode",
        "official_policy": official,
        "scenarios": list(SCENARIOS),
        "smoke": list(SMOKE),
        "reset_seed": "scenario_id",
        "torch_seed": f"{TORCH_SEED_OFFSET} + scenario_id",
        "seeds_per_scenario": 1,
        "bit_tie": BIT_TIE,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_percentiles": [BOOTSTRAP_LOW, BOOTSTRAP_HIGH],
        "payload_definition": "sum of served_bits over the episode",
        "code_sha256": _code_hashes(),
        "git_revision": _git_revision(),
        "train_pool_unchanged": list(TRAIN_POOL) == list(range(322000, 322048)),
        "validation_pool_unchanged": list(VALIDATION_POOL) == list(range(322048, 322064)),
    }
    path = assert_output_path(PROTOCOL_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _protocol_failures(payload: dict) -> list[str]:
    failures = []
    expected = {
        "scheme_c_sha256": SCHEME_C_SHA256,
        "epoch6_sha256": EPOCH6_SHA256,
        "mappo_sha256": MAPPO_SHA256,
        "adapter_config_sha256": ADAPTER_CONFIG_SHA256,
        "environment_sha256": ENVIRONMENT_SHA256,
        "k": 6,
        "hold": 15,
        "horizon": HORIZON,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bit_tie": BIT_TIE,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            failures.append(f"protocol {key}")
    if _sha256(SCHEME_C_CKPT) != payload.get("scheme_c_sha256") or _sha256(primary_checkpoint()) != payload.get("mappo_sha256"):
        failures.append("checkpoint hash drift")
    if _environment_sha256() != payload.get("environment_sha256") or config_sha256() != payload.get("adapter_config_sha256"):
        failures.append("config hash drift")
    if payload.get("code_sha256") != _code_hashes():
        failures.append("code hash drift")
    if payload.get("scenarios") != list(SCENARIOS) or payload.get("smoke") != list(SMOKE):
        failures.append("scenario list")
    root = ROOT
    for name in (
        "HTW_Phase6I_FormalPlannerSchemeCIntegration_CN.md",
        "HTW_Phase6I_FormalCandidatePoolAlignmentClosure_CN.md",
        "HTW_Phase6I_ClosedLoopExecutionGainAudit_CN.md",
        "HTW_Phase6I_PlannerStabilityValidation_CN.md",
    ):
        if not (root / "docs/archive/phase_reports" / name).is_file():
            failures.append(f"missing {name}")
    official = json.loads(POLICY_FILE.read_text(encoding="utf-8"))
    if official.get("high_level_policy") != "scheme_c" or int(official.get("k", 0)) != 6 or official.get("generator") != "G1_RATE":
        failures.append("official policy")
    if any(int(scenario) in TRAIN_POOL or int(scenario) in VALIDATION_POOL for scenario in SCENARIOS):
        failures.append("holdout joined an existing split")
    return failures


def _execution_fault(row: dict) -> bool:
    return bool(
        not row["complete"]
        or int(row["illegal"]) > 0
        or int(row["accounting_failures"]) > 0
        or int(row["nan_inf"]) > 0
        or int(row["anomaly_count"]) > 0
        or int(row["payload_matches_served"]) != 1
        or float(row["max_conservation_error"]) > BIT_TIE
    )


def _arm_problems(row: dict, expected_calls: int) -> list[str]:
    found = []
    if _execution_fault(row):
        found.append("execution")
    if int(row["scheme_c_calls"]) != int(expected_calls):
        found.append("call count")
    if [int(item) for item in row["decision_times"]] != list(DECISION_TIMES) or int(row["terminal_time"]) != HORIZON:
        found.append("termination")
    return found


def _bootstrap(deltas: list[float]) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    data = np.asarray(deltas, dtype=np.float64)
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    count = int(data.size)
    for index in range(BOOTSTRAP_DRAWS):
        draws[index] = float(np.mean(data[rng.integers(0, count, count)]))
    low, high = np.percentile(draws, [BOOTSTRAP_LOW, BOOTSTRAP_HIGH])
    return {"mean": float(np.mean(data)), "median": float(np.median(data)), "ci_low": float(low), "ci_high": float(high), "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED}


def _run_arm(scenario: int, policy_path: Path) -> dict:
    assert_scenario(int(scenario))
    torch.manual_seed(TORCH_SEED_OFFSET + int(scenario))
    row = run_active_episode(int(scenario), policy_path=str(policy_path), allow_holdout=True)
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


def _rollback() -> list[str]:
    write_policy("g1_rate")
    problems = []
    current = json.loads(POLICY_FILE.read_text(encoding="utf-8"))
    if read_policy() != "g1_rate" or current.get("generator") != "G1_RATE" or int(current.get("k", 0)) != 6:
        problems.append("rollback policy")
    checked = run_active_episode(322048, stop_time=1)
    if checked["policy"] != "g1_rate" or int(checked["scheme_c_calls"]) != 0 or not checked["decisions"]:
        problems.append("rollback entry")
    return problems


def _mean(rows: list[dict], arm: str, key: str) -> float:
    return float(np.mean([row[arm][key] for row in rows])) if rows else 0.0


def _validation_reference() -> dict:
    path = ROOT / "results/phase6iab_planner_stability" / "summary.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    return {
        "gate": saved["gate"],
        "mean_delta": saved["paired"]["mean"],
        "ci_low": saved["paired"]["ci_low"],
        "ci_high": saved["paired"]["ci_high"],
        "positive_scenarios": saved["positive_scenarios"],
        "tie_scenarios": saved["tie_scenarios"],
        "negative_scenarios": saved["negative_scenarios"],
        "scheme_mean_payload": saved["scheme_mean_payload"],
        "g1_mean_payload": saved["g1_mean_payload"],
    }


def run_audit() -> dict:
    limit_threads()
    if not PROTOCOL_FILE.is_file():
        raise RuntimeError("frozen protocol is missing; refuse to open the holdout")
    payload = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    official_bytes = POLICY_FILE.read_bytes()
    failures = _protocol_failures(payload)
    folder = assert_output_path(OUTPUT)
    scheme_path = assert_output_path(folder / "config_scheme_c.json")
    baseline_path = assert_output_path(folder / "config_g1_rate.json")
    pairs = []
    rolled_back = False
    rollback_problems: list[str] = []
    shard_count = 0
    if not failures:
        shard_count = int(sum(_shard_exists(int(scenario)) for scenario in SCENARIOS))
        write_policy("scheme_c", scheme_path)
        write_policy("g1_rate", baseline_path)
        if POLICY_FILE.read_bytes() != official_bytes:
            failures.append("official config changed while copying")
    if not failures:
        ordered = list(SMOKE) + [item for item in SCENARIOS if item not in SMOKE]
        for scenario in ordered:
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
            print(f"scenario {scenario} delta={pair['delta_payload']:.3f} problems={_pair_problems(pair)}", flush=True)
            if _execution_fault(pair["scheme_c"]):
                failures.append(f"scheme C execution {scenario}")
                rolled_back = True
                rollback_problems = _rollback()
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
        "saved_holdout_shards": shard_count,
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
        "scheme_mean_sinr": _mean(pairs, "scheme_c", "sinr_mean"),
        "g1_mean_sinr": _mean(pairs, "g1_rate", "sinr_mean"),
        "scheme_mean_interference": _mean(pairs, "scheme_c", "interference_mean"),
        "g1_mean_interference": _mean(pairs, "g1_rate", "interference_mean"),
        "scheme_mean_service_steps": _mean(pairs, "scheme_c", "service_steps_mean"),
        "g1_mean_service_steps": _mean(pairs, "g1_rate", "service_steps_mean"),
        "scheme_mean_retargets": _mean(pairs, "scheme_c", "decision_retargets"),
        "g1_mean_retargets": _mean(pairs, "g1_rate", "decision_retargets"),
        "scheme_mean_seconds": _mean(pairs, "scheme_c", "seconds"),
        "g1_mean_seconds": _mean(pairs, "g1_rate", "seconds"),
        "scheme_mean_high_level_seconds": _mean(pairs, "scheme_c", "high_level_seconds"),
        "g1_mean_high_level_seconds": _mean(pairs, "g1_rate", "high_level_seconds"),
        "anomaly_episodes": int(sum(int(pair["scheme_c"]["anomaly_count"]) + int(pair["g1_rate"]["anomaly_count"]) > 0 for pair in pairs)),
        "validation_reference": _validation_reference() if clean or pairs else {},
        "planner_ready": int(official_now.get("generator") == "G1_RATE" and int(official_now.get("k", 0)) == 6 and not rollback_problems),
        "deployment_verified": False,
        "trained": False,
        "rows": pairs,
    }
    (folder / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"ci": [stats["ci_low"], stats["ci_high"]], "failures": failures, "gate": gate}, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "freeze":
        print(json.dumps({"protocol": str(write_protocol() and PROTOCOL_FILE)}, sort_keys=True), flush=True)
    else:
        run_audit()
