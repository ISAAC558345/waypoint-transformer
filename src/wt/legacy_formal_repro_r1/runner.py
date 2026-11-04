"""Call the official formal entry on policy-file copies."""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6f_system_bringup.episode import run_active_episode
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6iaa_formal_planner.faults import file_sha256
from wt.benchmark_v2.phase6iaa_formal_planner.protocol import (
    EPOCH6_CKPT,
    POLICY_FILE,
    SCHEME_C_CKPT,
    write_policy,
)
from wt.legacy_formal_repro_r1.protocol import ARMS, HORIZON, PROCESS_SEED_OFFSET, SCENARIOS, STEP_BUDGET


def sha256_file(path: Path) -> str:
    return file_sha256(Path(path))


def prepare_policy_copy(policy: str, destination: Path) -> Path:
    """Write one arm's policy file. The official active policy path is refused."""
    target = Path(destination)
    if target.resolve() == POLICY_FILE.resolve():
        raise RuntimeError("R1 refused to write the official active policy file")
    if policy not in ARMS:
        raise RuntimeError(f"R1 refused policy {policy}")
    return write_policy(policy, target)


def model_identity() -> dict:
    """Hash the checkpoint files the formal planner reads. This does not load them again."""
    rows = {
        "scheme_c_checkpoint": str(SCHEME_C_CKPT),
        "scheme_c_sha256": sha256_file(SCHEME_C_CKPT),
        "epoch6_checkpoint": str(EPOCH6_CKPT),
        "epoch6_sha256": sha256_file(EPOCH6_CKPT),
        "mappo_checkpoint": str(primary_checkpoint()),
        "mappo_sha256": sha256_file(primary_checkpoint()),
    }
    return rows


def _seed_process(scenario_id: int) -> int:
    value = int(PROCESS_SEED_OFFSET + int(scenario_id))
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return value


def _record_steps(steps: list[dict]):
    original = L3BBitQueueEnvV2.step

    def wrapped(self, *args, **kwargs):
        time_index = int(self.t)
        info = original(self, *args, **kwargs)
        steps.append(
            {
                "time": time_index,
                "served_bits": float(info["payload_increment_bits"]),
                "cumulative_payload_bits": float(self.payload_bits),
            }
        )
        return info

    return original, wrapped


def run_arm(scenario_id: int, policy: str, policy_path: Path) -> dict:
    """One 48-step arm through run_active_episode. A failure is returned, not retried here."""
    steps: list[dict] = []
    original, wrapped = _record_steps(steps)
    process_seed = _seed_process(scenario_id)
    policy_before = hashlib.sha256(Path(policy_path).read_bytes()).hexdigest()
    started = time.perf_counter()
    error = None
    result = None
    L3BBitQueueEnvV2.step = wrapped
    try:
        result = run_active_episode(
            int(scenario_id),
            stop_time=int(HORIZON),
            policy_path=str(policy_path),
            allow_holdout=False,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        L3BBitQueueEnvV2.step = original
    elapsed = time.perf_counter() - started
    payload = None if result is None else float(result["payload_bits"])
    served = None if result is None else float(result["served_bits"])
    step_served = float(sum(float(row["served_bits"]) for row in steps))
    return {
        "scenario_id": int(scenario_id),
        "policy": policy,
        "policy_path": str(policy_path),
        "policy_sha256_before": policy_before,
        "policy_sha256_after": hashlib.sha256(Path(policy_path).read_bytes()).hexdigest(),
        "process_seed": int(process_seed),
        "process_seed_role": "random, numpy, and torch are reset before the planner is constructed",
        "reset_seed": int(scenario_id),
        "reset_seed_role": "environment reset seed inside run_active_episode",
        "stop_time": int(HORIZON),
        "terminal_time": 0 if result is None else int(result["terminal_time"]),
        "environment_steps": len(steps),
        "complete": False if result is None else bool(result["complete"]),
        "payload_bits": payload,
        "served_bits": served,
        "payload_matches_served": 0 if result is None else int(result["payload_matches_served"]),
        "step_served_bits": step_served,
        "step_served_matches_episode": bool(result is not None and abs(step_served - served) <= 1.0e-6),
        "decision_times": [] if result is None else [int(item) for item in result["decision_times"]],
        "decision_count": 0 if result is None else int(result["decision_count"]),
        "scheme_c_calls": 0 if result is None else int(result["scheme_c_calls"]),
        "anomaly_count": 0 if result is None else int(result["anomaly_count"]),
        "anomalies": [] if result is None else result["anomalies"],
        "fault_recovery": None if result is None else result["fault_recovery"],
        "seconds": float(elapsed if result is None else result["seconds"]),
        "wall_seconds": float(elapsed),
        "error": error,
        "steps": steps,
        "episode": result,
    }


def run_registered(config_dir: Path) -> dict:
    """Four arms, 192 environment steps. Stop once the budget is exhausted."""
    consumed = 0
    rows = []
    for scenario_id in SCENARIOS:
        for policy in ARMS:
            if consumed + HORIZON > STEP_BUDGET:
                raise RuntimeError(f"R1 stopped before exceeding the {STEP_BUDGET}-step budget")
            path = prepare_policy_copy(policy, Path(config_dir) / f"{scenario_id}_{policy}.json")
            row = run_arm(scenario_id, policy, path)
            consumed += int(row["environment_steps"])
            rows.append(row)
            if row["error"] or int(row["environment_steps"]) != HORIZON:
                return {"rows": rows, "environment_steps": consumed, "stopped_early": True}
    return {"rows": rows, "environment_steps": consumed, "stopped_early": False}


def pair_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[int, dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["scenario_id"]), {})[row["policy"]] = row
    pairs = []
    for scenario_id in SCENARIOS:
        arms = grouped.get(int(scenario_id), {})
        if "scheme_c" not in arms or "g1_rate" not in arms:
            continue
        scheme = arms["scheme_c"]
        baseline = arms["g1_rate"]
        pairs.append(
            {
                "scenario_id": int(scenario_id),
                "scheme_c_payload_bits": scheme["payload_bits"],
                "g1_rate_payload_bits": baseline["payload_bits"],
                "payload_difference_bits": None
                if scheme["payload_bits"] is None or baseline["payload_bits"] is None
                else float(scheme["payload_bits"] - baseline["payload_bits"]),
                "scheme_c_seconds": scheme["wall_seconds"],
                "g1_rate_seconds": baseline["wall_seconds"],
                "scheme_c_complete": scheme["complete"],
                "g1_rate_complete": baseline["complete"],
            }
        )
    return pairs


def public_row(row: dict) -> dict:
    """Drop the nested episode object. The saved fields already carry its contract."""
    saved = dict(row)
    saved.pop("episode", None)
    return saved


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
