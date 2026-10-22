"""Limited deployment evidence. This does not train or open the holdout."""
from __future__ import annotations

import ctypes
import hashlib
import json
import platform
import subprocess
import time
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from wt.benchmark_v2.phase6f_system_bringup.protocol import CANDIDATE_K
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import limit_threads, load_primary, parameter_token
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6iaa_formal_planner.planner import accept_scored_row
from wt.benchmark_v2.phase6iaa_formal_planner.protocol import (
    ACTIVE_CANDIDATE_K,
    POLICY_FILE,
    read_policy,
    write_policy,
)
from wt.benchmark_v2.phase6iad_deployment_audit.protocol import (
    ADAPTER_CONFIG_SHA256,
    CANDIDATE_COUNT,
    ENVIRONMENT_SHA256,
    EPOCH6_SHA256,
    FULL_DECISION_TIMES,
    FROZEN_CODE_SHA256,
    INTERFACE_TESTS,
    MAPPO_SHA256,
    OUTPUT,
    REGRESSION_SCENARIO,
    REPEAT_COUNT,
    REPEAT_SCENARIO,
    REPEAT_STOP,
    ROOT,
    SCHEME_C_SHA256,
    assert_output_path,
    assert_scenario,
    assign_gate,
    status_flags,
)
from wt.benchmark_v2.phase6is_assignment_head.protocol import EPOCH6_CKPT, SCHEME_C_CKPT
from wt.benchmark_v2.phase6iy_candidate_adapter.protocol import config_sha256
from wt.l3b.config import L3BConfig


class _MemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


_ACTOR_SECONDS: list[float] = []


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


def _git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _memory() -> dict:
    import gc

    gc.collect()
    counters = _MemoryCounters()
    counters.cb = ctypes.sizeof(_MemoryCounters)
    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype = ctypes.c_void_p
    info = ctypes.windll.psapi.GetProcessMemoryInfo
    info.argtypes = [ctypes.c_void_p, ctypes.POINTER(_MemoryCounters), ctypes.c_ulong]
    info.restype = ctypes.c_int
    ok = info(get_process(), ctypes.byref(counters), counters.cb)
    if not ok:
        return {"available": 0, "error": int(ctypes.get_last_error())}
    return {
        "available": 1,
        "working_set_bytes": int(counters.WorkingSetSize),
        "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
        "private_bytes": int(counters.PrivateUsage),
    }


def _stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean_seconds": float(np.mean(array)),
        "median_seconds": float(np.median(array)),
        "max_seconds": float(np.max(array)),
        "p95_seconds": float(np.percentile(array, 95)),
    }


def _install_actor_timer():
    import wt.benchmark_v2.phase6iaa_formal_planner.planner as planner

    original = planner._actor_move

    def _timed(env, actor, option, last_movement):
        started = time.perf_counter()
        result = original(env, actor, option, last_movement)
        _ACTOR_SECONDS.append(time.perf_counter() - started)
        return result

    planner._actor_move = _timed
    return planner, original


def _hashes() -> dict[str, str]:
    return {
        "scheme_c": _sha256(SCHEME_C_CKPT),
        "epoch6": _sha256(EPOCH6_CKPT),
        "mappo": _sha256(primary_checkpoint()),
        "adapter_config": config_sha256(),
        "environment": _environment_sha256(),
    }


def _expected_hash_faults(found: dict[str, str]) -> list[str]:
    checks = {
        "scheme_c": SCHEME_C_SHA256,
        "epoch6": EPOCH6_SHA256,
        "mappo": MAPPO_SHA256,
        "adapter_config": ADAPTER_CONFIG_SHA256,
        "environment": ENVIRONMENT_SHA256,
    }
    return [f"hash {name}" for name, digest in checks.items() if found[name] != digest]


def _code_hash_faults() -> list[str]:
    faults = []
    for name, digest in FROZEN_CODE_SHA256.items():
        if _sha256(ROOT / name) != digest:
            faults.append(f"code {name}")
    return faults


def _requirements() -> dict:
    text = (ROOT / "src/wt/benchmark_v2/phase6iaa_formal_planner/protocol.py").read_text(encoding="utf-8")
    named = ("deployment_latency_sla", "target_runtime_environment", "operations_specification")
    return {
        "deployment_latency_sla": "NOT_YET_SPECIFIED",
        "target_runtime_environment": "NOT_YET_SPECIFIED",
        "operations_specification": "NOT_YET_SPECIFIED",
        "formal_protocol_defines_requirements": all(name in text for name in named),
    }


def _missing_policy_result(path: Path) -> str:
    try:
        return str(read_policy(path))
    except Exception as exc:
        return type(exc).__name__


def _safety_probes() -> dict:
    missing = OUTPUT / "_missing_policy.json"
    try:
        torch.load(OUTPUT / "_missing_checkpoint.pt", map_location="cpu", weights_only=False)
        missing_checkpoint = "none"
    except Exception as exc:
        missing_checkpoint = type(exc).__name__
    try:
        assert_scenario(322064)
        holdout_refused = 0
    except RuntimeError:
        holdout_refused = 1
    return {
        "nan_score_rejected": int(accept_scored_row(np.asarray([1.0, np.nan]), np.asarray([True, True])) is None),
        "illegal_flag_rejected": int(accept_scored_row(np.asarray([1.0, 2.0]), np.asarray([True, False])) is None),
        "empty_score_rejected": int(accept_scored_row(np.asarray([]), np.asarray([])) is None),
        "missing_policy_file_result": _missing_policy_result(missing),
        "missing_checkpoint_exception": missing_checkpoint,
        "holdout_refused": holdout_refused,
        "historical_candidate_k": int(CANDIDATE_K),
        "active_candidate_k": int(ACTIVE_CANDIDATE_K),
    }


def _row_faults(row: dict, times: tuple[int, ...], policy: str, calls: int) -> list[str]:
    faults = []
    if not row["complete"] or int(row["terminal_time"]) != int(row["stop_time"]):
        faults.append("incomplete")
    if row["policy"] != policy:
        faults.append("policy")
    if [int(item) for item in row["decision_times"]] != [int(item) for item in times]:
        faults.append("decision times")
    if int(row["illegal"]) != 0 or int(row["nan_inf"]) != 0 or int(row["anomaly_count"]) != 0:
        faults.append("execution")
    if int(row["accounting_failures"]) != 0 or int(row["payload_matches_served"]) != 1:
        faults.append("accounting")
    if abs(float(row["payload_bits"]) - float(row["served_bits"])) > 1.0:
        faults.append("payload mismatch")
    if float(row["max_conservation_error"]) > 1.0:
        faults.append("conservation")
    if int(row["scheme_c_calls"]) != int(calls):
        faults.append("scheme calls")
    for decision in row["decisions"]:
        if int(decision["candidate_count"]) != int(CANDIDATE_COUNT):
            faults.append("candidate count")
        if int(decision["score_index"]) < 0 or len(decision["assignment"]) != 6:
            faults.append("score index")
        expected_call = 1 if policy == "scheme_c" else 0
        if int(decision["scheme_c_called"]) != expected_call:
            faults.append("score binding")
    return faults


def _public_row(row: dict) -> dict:
    decisions = [
        {
            "time": int(item["time"]),
            "assignment": [int(value) for value in item["assignment"]],
            "score_index": int(item["score_index"]),
            "candidate_count": int(item["candidate_count"]),
            "pool_seconds": float(item["pool_seconds"]),
            "score_seconds": float(item["score_seconds"]),
            "scheme_c_called": int(item["scheme_c_called"]),
            "idle_slots": int(item["idle_slots"]),
        }
        for item in row["decisions"]
    ]
    return {
        "scenario_id": int(row["scenario_id"]),
        "policy": row["policy"],
        "complete": bool(row["complete"]),
        "payload_bits": float(row["payload_bits"]),
        "served_bits": float(row["served_bits"]),
        "payload_matches_served": int(row["payload_matches_served"]),
        "terminal_time": int(row["terminal_time"]),
        "stop_time": int(row["stop_time"]),
        "decision_times": [int(item) for item in row["decision_times"]],
        "illegal": int(row["illegal"]),
        "accounting_failures": int(row["accounting_failures"]),
        "nan_inf": int(row["nan_inf"]),
        "anomaly_count": int(row["anomaly_count"]),
        "anomalies": row["anomalies"],
        "max_conservation_error": float(row["max_conservation_error"]),
        "scheme_c_calls": int(row["scheme_c_calls"]),
        "seconds": float(row["seconds"]),
        "high_level_seconds": float(row["high_level_seconds"]),
        "rng_unchanged": int(row["rng_unchanged"]),
        "projector_interventions": int(row["projector_interventions"]),
        "decisions": decisions,
    }


def _run_interface_tests() -> dict:
    suite = unittest.defaultTestLoader.loadTestsFromNames(list(INTERFACE_TESTS))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return {
        "tests": int(result.testsRun),
        "failures": int(len(result.failures)),
        "errors": int(len(result.errors)),
        "skipped": int(len(result.skipped)),
    }


def _actor_probe() -> dict:
    actor = load_primary()
    first = parameter_token(actor)
    second = parameter_token(load_primary())
    return {
        "loaded": 1,
        "eval_mode": int(not actor.training),
        "frozen_parameters": int(all(not parameter.requires_grad for parameter in actor.parameters())),
        "reload_token_match": int(first == second),
        "checkpoint": str(primary_checkpoint()),
    }


def _timing(row: dict, actor_seconds: list[float]) -> dict:
    return {
        "candidate_generation": _stats([float(item["pool_seconds"]) for item in row["decisions"]]),
        "scheme_c_scoring": _stats([float(item["score_seconds"]) for item in row["decisions"]]),
        "mappo_step": _stats(actor_seconds),
        "episode_seconds": float(row["seconds"]),
        "high_level_seconds": float(row["high_level_seconds"]),
        "latency_requirement": "NOT_YET_SPECIFIED",
        "latency_requirement_met": 0,
    }


def run_audit() -> dict:
    limit_threads()
    assert_scenario(REGRESSION_SCENARIO)
    assert_scenario(REPEAT_SCENARIO)
    samples = [{"label": "start", **_memory()}]
    faults: list[str] = []
    policy_mutated = False
    severe_scheme = False
    scheme_row = None
    baseline_row = None
    reload_row = None
    repeats = []
    scheme_actor: list[float] = []
    baseline_actor: list[float] = []
    reload_actor: list[float] = []
    before = _hashes()
    faults.extend(_expected_hash_faults(before))
    faults.extend(_code_hash_faults())
    requirements = _requirements()
    probes = _safety_probes()
    if probes["missing_checkpoint_exception"] == "none" or probes["holdout_refused"] != 1:
        faults.append("safety probe")
    if probes["nan_score_rejected"] != 1 or probes["illegal_flag_rejected"] != 1 or probes["empty_score_rejected"] != 1:
        faults.append("score rejection")
    interface = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    actor_probe = {}
    planner = None
    original = None
    try:
        if not faults:
            interface = _run_interface_tests()
            if interface["failures"] or interface["errors"]:
                faults.append("interface tests")
        if not faults:
            actor_probe = _actor_probe()
            if not actor_probe["eval_mode"] or not actor_probe["frozen_parameters"] or not actor_probe["reload_token_match"]:
                faults.append("mappo load")
        if not faults:
            planner, original = _install_actor_timer()
            from wt.benchmark_v2.phase6f_system_bringup.episode import run_active_episode

            if read_policy() != "scheme_c":
                faults.append("official policy was not scheme_c before the audit")
            else:
                _ACTOR_SECONDS.clear()
                scheme_row = run_active_episode(REGRESSION_SCENARIO)
                scheme_actor = list(_ACTOR_SECONDS)
                samples.append({"label": "scheme_c_episode", **_memory()})
                scheme_faults = _row_faults(scheme_row, FULL_DECISION_TIMES, "scheme_c", 4)
                if scheme_faults:
                    severe_scheme = True
                    faults.extend(f"scheme {item}" for item in scheme_faults)
                else:
                    write_policy("g1_rate")
                    policy_mutated = True
                    _ACTOR_SECONDS.clear()
                    baseline_row = run_active_episode(REGRESSION_SCENARIO)
                    baseline_actor = list(_ACTOR_SECONDS)
                    samples.append({"label": "g1_rate_episode", **_memory()})
                    baseline_faults = _row_faults(baseline_row, FULL_DECISION_TIMES, "g1_rate", 0)
                    if baseline_faults:
                        faults.extend(f"baseline {item}" for item in baseline_faults)
                    else:
                        write_policy("scheme_c")
                        _ACTOR_SECONDS.clear()
                        reload_row = run_active_episode(REPEAT_SCENARIO, stop_time=REPEAT_STOP)
                        reload_actor = list(_ACTOR_SECONDS)
                        samples.append({"label": "scheme_c_reload", **_memory()})
                        reload_faults = _row_faults(reload_row, (0,), "scheme_c", 1)
                        if reload_faults:
                            severe_scheme = True
                            faults.extend(f"reload {item}" for item in reload_faults)
                        else:
                            for index in range(REPEAT_COUNT - 1):
                                _ACTOR_SECONDS.clear()
                                row = run_active_episode(REPEAT_SCENARIO, stop_time=REPEAT_STOP)
                                samples.append({"label": f"repeat_{index + 2}", **_memory()})
                                row_faults = _row_faults(row, (0,), "scheme_c", 1)
                                repeats.append(_public_row(row))
                                if row_faults:
                                    severe_scheme = True
                                    faults.extend(f"repeat {item}" for item in row_faults)
                                    break
    except Exception as exc:
        severe_scheme = True
        faults.append(f"exception {type(exc).__name__}: {exc}")
    finally:
        if planner is not None and original is not None:
            planner._actor_move = original
        if severe_scheme:
            write_policy("g1_rate")
            policy_mutated = True
        elif policy_mutated:
            write_policy("scheme_c")
    after = _hashes()
    if before != after:
        faults.append("hash changed")
    official = read_policy()
    if severe_scheme and official != "g1_rate":
        faults.append("rollback policy")
    if not severe_scheme and policy_mutated and official != "scheme_c":
        faults.append("restore policy")
    samples.append({"label": "end", **_memory()})
    requirements_specified = all(requirements[name] != "NOT_YET_SPECIFIED" for name in ("deployment_latency_sla", "target_runtime_environment", "operations_specification"))
    gate = assign_gate(observed_faults=len(faults), requirements_specified=requirements_specified)
    planner_ready = bool(
        not severe_scheme
        and official == "scheme_c"
        and int(ACTIVE_CANDIDATE_K) == 6
        and scheme_row is not None
        and reload_row is not None
        and not faults
    )
    flags = status_flags(gate=gate, planner_ready=planner_ready)
    growth = []
    private = [int(item["private_bytes"]) for item in samples if item.get("available")]
    for left, right in zip(private, private[1:]):
        growth.append(int(right - left))
    result = {
        "gate": gate,
        "faults": faults,
        "requirements": requirements,
        "safety_probes": probes,
        "safety_residuals": [
            "a missing active_policy.json defaults to scheme_c",
            "an aborted episode does not itself switch the official policy",
            "exceptions other than RuntimeError leave the episode handler",
            "no service supervisor, log sink, or restart rule is defined",
        ],
        "interface_tests": interface,
        "actor_probe": actor_probe,
        "hashes_before": before,
        "hashes_after": after,
        "hashes_unchanged": before == after,
        "code_sha256_matches_phase6iac_freeze": not _code_hash_faults(),
        "git_revision": _git_revision(),
        "official_policy": official,
        "historical_candidate_k": int(CANDIDATE_K),
        "active_candidate_k": int(ACTIVE_CANDIDATE_K),
        "scheme_episode": None if scheme_row is None else _public_row(scheme_row),
        "baseline_episode": None if baseline_row is None else _public_row(baseline_row),
        "reload_episode": None if reload_row is None else _public_row(reload_row),
        "repeat_episodes": repeats,
        "timing": None if scheme_row is None else _timing(scheme_row, scheme_actor),
        "baseline_timing": None if baseline_row is None else _timing(baseline_row, baseline_actor),
        "reload_timing": None if reload_row is None else _timing(reload_row, reload_actor),
        "resources": {
            "platform": platform.platform(),
            "cpu_count": int(os_cpu_count()),
            "torch_threads": int(torch.get_num_threads()),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "samples": samples,
            "private_byte_deltas": growth,
        },
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


def os_cpu_count() -> int:
    import os

    return int(os.cpu_count() or 0)


if __name__ == "__main__":
    run_audit()
