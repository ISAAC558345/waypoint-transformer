"""Policy, checkpoint, and episode fault handling. Scoring is unchanged."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6e.generators import is_legal
from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import ROOT


class PolicyError(RuntimeError):
    """The active policy file is missing, unreadable, or illegal."""


class CheckpointError(RuntimeError):
    """A frozen checkpoint is missing or does not match its recorded hash."""


class RollbackError(RuntimeError):
    """Baseline rollback did not complete. No other policy is selected."""


def default_fault_log() -> Path:
    return ROOT / "results/phase6iae_deployment_contract" / "faults.jsonl"


def append_fault_log(record: dict, log_path: Path | None = None) -> Path:
    """Append one fault record. This does not select a policy."""
    target = default_fault_log() if log_path is None else Path(log_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    row = {"logged_at": datetime.now().isoformat(timespec="seconds"), **record}
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return target


def file_sha256(path: Path, *, log_path: Path | None = None) -> str:
    file = Path(path)
    if not file.is_file():
        message = f"checkpoint missing: {file}"
        append_fault_log({"event": "checkpoint_rejected", "reason": message, "stopped": True}, log_path)
        raise CheckpointError(message)
    digest = hashlib.sha256()
    with file.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint_file(path: Path, expected_sha: str, *, log_path: Path | None = None) -> str:
    actual = file_sha256(path, log_path=log_path)
    if actual != str(expected_sha):
        message = f"checkpoint hash mismatch: {path}"
        append_fault_log(
            {"event": "checkpoint_rejected", "reason": message, "expected": str(expected_sha), "actual": actual, "stopped": True},
            log_path,
        )
        raise CheckpointError(message)
    return actual


def verify_startup_hashes(record: dict, *, log_path: Path | None = None) -> None:
    """Reject a planner startup whose frozen weights differ from the recorded contract."""
    from wt.benchmark_v2.phase6iaa_formal_planner.protocol import MAPPO_SHA256, SCHEME_C_SHA256

    expected = {"scheme_c_sha256": SCHEME_C_SHA256, "mappo_sha256": MAPPO_SHA256}
    mismatched = [name for name, digest in expected.items() if str(record.get(name, "")) != digest]
    if mismatched:
        message = f"checkpoint hash mismatch: {', '.join(mismatched)}"
        append_fault_log({"event": "checkpoint_rejected", "reason": message, "stopped": True}, log_path)
        raise CheckpointError(message)


def option_is_retainable(option, assignment, pending: int, *, n_uavs: int = 6, n_buoys: int = 12) -> bool:
    """A held TaskOption may continue only while its own association is still the legal decision."""
    if option is None or int(pending) <= 0:
        return False
    if int(option.remaining_hold) != int(pending):
        return False
    association = np.asarray(option.association, dtype=np.int64)
    chosen = np.asarray(assignment, dtype=np.int64)
    targets = np.asarray(option.targets, dtype=np.int64)
    if association.shape != chosen.shape or not np.array_equal(association, chosen):
        return False
    if targets.shape != association.shape or not np.array_equal(targets, association):
        return False
    return bool(is_legal(association, int(n_uavs), int(n_buoys)))


def rollback_to_baseline(path: Path | None = None, *, expected_mappo_sha: str | None = None, log_path: Path | None = None) -> dict:
    """Write the explicit g1_rate file after the frozen MAPPO and k=6 checks pass."""
    from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
    from wt.benchmark_v2.phase6iaa_formal_planner.protocol import (
        ACTIVE_CANDIDATE_K,
        GENERATOR,
        HOLD,
        MAPPO_SHA256,
        POLICY_FILE,
        read_policy,
        write_policy,
    )

    target = POLICY_FILE if path is None else Path(path)
    expected = MAPPO_SHA256 if expected_mappo_sha is None else str(expected_mappo_sha)
    checkpoint = primary_checkpoint()
    if GENERATOR != "G1_RATE" or int(ACTIVE_CANDIDATE_K) != 6 or int(HOLD) != 15:
        message = "rollback stopped because the candidate contract is not G1_RATE k=6"
        append_fault_log({"event": "rollback_failed", "reason": message, "stopped": True}, log_path)
        raise RollbackError(message)
    try:
        actual = verify_checkpoint_file(checkpoint, expected, log_path=log_path)
    except CheckpointError as exc:
        append_fault_log({"event": "rollback_failed", "reason": str(exc), "stopped": True}, log_path)
        raise RollbackError(str(exc)) from exc
    write_policy("g1_rate", target)
    try:
        loaded = read_policy(target)
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, PolicyError) as exc:
        message = "rollback wrote g1_rate but the file did not verify"
        append_fault_log({"event": "rollback_failed", "reason": message, "stopped": True}, log_path)
        raise RollbackError(message) from exc
    if loaded != "g1_rate" or payload.get("generator") != "G1_RATE" or int(payload.get("k", -1)) != 6:
        message = "rollback did not stay on G1_RATE k=6"
        append_fault_log({"event": "rollback_failed", "reason": message, "stopped": True}, log_path)
        raise RollbackError(message)
    record = {
        "event": "rollback_completed",
        "rollback": "g1_rate",
        "generator": "G1_RATE",
        "k": 6,
        "hold": int(payload["hold"]),
        "mappo_sha256": actual,
        "policy_file": str(target),
        "stopped": True,
    }
    append_fault_log(record, log_path)
    return record


def handle_episode_fault(*, reason: str, scenario_id: int, time_index: int, policy_path: Path | None = None, log_path: Path | None = None) -> dict:
    """Stop the failed episode and recover only through the verified g1_rate rollback."""
    append_fault_log(
        {
            "event": "episode_fault",
            "reason": str(reason),
            "scenario_id": int(scenario_id),
            "time_index": int(time_index),
            "stopped": True,
        },
        log_path,
    )
    try:
        recovery = rollback_to_baseline(policy_path, log_path=log_path)
    except RollbackError:
        append_fault_log(
            {
                "event": "rollback_failed",
                "reason": str(reason),
                "scenario_id": int(scenario_id),
                "time_index": int(time_index),
                "stopped": True,
            },
            log_path,
        )
        raise
    recovery["reason"] = str(reason)
    recovery["scenario_id"] = int(scenario_id)
    recovery["time_index"] = int(time_index)
    return recovery
