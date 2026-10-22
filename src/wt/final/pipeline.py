"""FINAL-HIER closed-loop facade.

Pipeline (existing modules, not rewritten):
state -> DeepSets encoder -> Order-Free Decoder -> assignment
-> Task Option -> fly-toward -> InRangeZeroMotionGate -> env.step
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import _bootstrap  # noqa: F401

from wt.htw_phase100.high_level import decide, load_deepsets_util, make_option
from wt.htw_phase100.rule_executor import RuleExecutorAdapter
from wt.htw_phase103.gate import InRangeZeroMotionGate
from wt.htw_phase9.const import HOLD, IDLE_INDEX
from wt.l3b.env import L3BEnv, conflict_mask
from wt.l3b.policies import fly_toward

from .constants import DEFAULT_DEMO_SEED, EXPECTED_HIGH_SHA, FINAL_CKPT, SOURCE_CKPT, UNTOUCHED_SEEDS, sha256_bytes


def refuse_untouched(seed: int) -> None:
    if int(seed) in UNTOUCHED_SEEDS:
        raise RuntimeError(f"refused Phase-10.6 untouched seed {seed}; demo/final entry must not reuse consumed untouched data")


def resolve_checkpoint() -> Path:
    for path in (FINAL_CKPT, SOURCE_CKPT):
        if path.is_file():
            digest = sha256_bytes(path.read_bytes())
            if digest != EXPECTED_HIGH_SHA:
                raise RuntimeError(f"checkpoint SHA mismatch at {path}: {digest}")
            return path
    raise FileNotFoundError("frozen DeepSets-util seed0 checkpoint is missing")


def load_final_model(path: Path | None = None):
    ckpt = resolve_checkpoint() if path is None else Path(path)
    digest = sha256_bytes(ckpt.read_bytes())
    if digest != EXPECTED_HIGH_SHA:
        raise RuntimeError(f"refused non-frozen checkpoint SHA {digest}")
    model = load_deepsets_util(0, ckpt)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, digest


def run_final_hier(
    seed: int = DEFAULT_DEMO_SEED,
    *,
    gated: bool = True,
    model=None,
) -> dict[str, Any]:
    """Run one ordinary scenario. Never uses untouched seeds."""
    refuse_untouched(int(seed))
    if model is None:
        model, ckpt_sha = load_final_model()
    else:
        ckpt_sha = EXPECTED_HIGH_SHA
    env = L3BEnv()
    info = env.reset(seed=int(seed))
    last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
    last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
    pending = 0
    option = None
    rule = RuleExecutorAdapter()
    gate = InRangeZeroMotionGate()
    flight = 0.0
    assignments: list[list[int]] = []
    movements: list[list[list[float]]] = []
    holds: list[int] = []
    while not info.get("terminated", False):
        if pending <= 0:
            assignment = decide(model, env, info, last_arrival, last_positions, pending)
            pending = int(HOLD)
            option = make_option(env, info, assignment, pending)
        assert option is not None
        option.remaining_hold = int(pending)
        raw = np.asarray(rule.act(env, option), dtype=np.float64)
        expected_raw = np.asarray(fly_toward(env, option.association), dtype=np.float64)
        if not np.allclose(raw, expected_raw, atol=1.0e-12):
            raise RuntimeError("facade fly-toward diverged from RuleExecutorAdapter")
        if gated:
            movement, flags = gate.apply(env, option, raw)
        else:
            flags = np.zeros(env.config.num_uavs, dtype=bool)
            movement = raw.copy()
        pos_before = np.asarray(env.uav_positions, dtype=np.float64)
        after = env.step(movement, option.association)
        pending -= 1
        flight += float(np.sum(np.linalg.norm(np.asarray(after["uav_positions_m"], dtype=np.float64) - pos_before, axis=-1)))
        assignments.append(np.asarray(option.association, dtype=np.int64).tolist())
        movements.append(np.asarray(movement, dtype=np.float64).tolist())
        holds.append(int(option.remaining_hold))
        last_arrival = np.asarray(after["arrivals_blocks"], dtype=np.float32)
        last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
        info = after
    payload = float(info["cumulative_payload_bits"])
    blocks = payload / max(float(env.config.block_payload_bits), 1.0e-9)
    if not np.isfinite(payload) or not np.isfinite(flight):
        raise RuntimeError("NaN/Inf in FINAL-HIER demo rollout")
    return {
        "scenario": int(seed),
        "gated": bool(gated),
        "checkpoint_sha256": ckpt_sha,
        "assignment": assignments[-1] if assignments else [],
        "assignments": assignments,
        "uav_movement": movements,
        "task_option": {
            "hold": HOLD,
            "idle_index": IDLE_INDEX,
            "idle_env_code": -1,
            "remaining_hold_trace": holds,
        },
        "payload": payload,
        "completed_blocks": float(blocks),
        "queue": np.asarray(info["queue_blocks"], dtype=np.float64).tolist(),
        "flight_distance": flight,
        "legality": bool(info["exclusive_ok"]),
        "exclusive_ok": bool(info["exclusive_ok"]),
        "conflict_last": bool(np.any(conflict_mask(np.asarray(assignments[-1], dtype=np.int64), env.config.num_buoys))) if assignments else False,
        "n_steps": len(assignments),
        "untouched_used": False,
    }


__all__ = ["load_final_model", "refuse_untouched", "resolve_checkpoint", "run_final_hier"]
