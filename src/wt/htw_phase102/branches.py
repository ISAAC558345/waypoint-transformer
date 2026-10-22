"""Same-snapshot diagnostic interventions. Association frozen; no training."""
from __future__ import annotations

from typing import Any

import numpy as np

from wt.htw_phase101.rewards import auxiliary_reward
from wt.l3b.env import L3BEnv
from wt.l3b.policies import fly_toward
from wt.round30.ppo import PPOAgent

from .protocol import ATOL_REPLAY, MAX_BRANCH_STEPS
from .replay import _ppo_move
from .snapshot import restore_controller, restore_env


def _apply_move(env: L3BEnv, option, movement: np.ndarray):
    if not np.array_equal(option.association, option.targets):
        raise RuntimeError("branch refused rematching")
    return env.step(movement, option.association)


def run_branch(snap: dict[str, Any], *, kind: str, focus_uav: int, schema: dict[str, Any], agent: PPOAgent, recorded: list[dict[str, Any]] | None) -> dict[str, Any]:
    env = L3BEnv()
    restore_env(env, snap["env"])
    ctrl = restore_controller(snap["ctrl"])
    pending = int(ctrl["pending"])
    option = ctrl["option"]
    last_arrival = ctrl["last_arrival"]
    last_positions = ctrl["last_positions"]
    horizon_left = min(int(pending), int(MAX_BRANCH_STEPS), int(env.config.episode_horizon) - int(env.t))
    start_payload = float(env.payload_bits)
    start_rho = float(env.progress[focus_uav])
    tgt = int(option.association[focus_uav])
    blocks = 0
    rho_gain = 0.0
    out_of_range = 0
    lost = 0
    match_ok = True
    match_fail = None
    rates = []
    sinrs = []
    info = env.info()
    for step_i in range(horizon_left):
        if pending <= 0:
            break
        option.remaining_hold = int(pending)
        raw, move = _ppo_move(agent, schema, env, info, option, last_arrival, last_positions)
        if kind == "C1":
            move = np.asarray(move, dtype=np.float64).copy()
            move[int(focus_uav)] = 0.0
        elif kind == "C2":
            fly = fly_toward(env, option.association)
            move = np.asarray(move, dtype=np.float64).copy()
            move[int(focus_uav)] = fly[int(focus_uav)]
        elif kind != "C0":
            raise ValueError(kind)
        rho_b = float(env.progress[focus_uav])
        in_b = bool(tgt >= 0 and env._in_range(focus_uav, tgt))
        after = _apply_move(env, option, move)
        pending -= 1
        option.remaining_hold = int(pending)
        last_arrival = np.asarray(after["arrivals_blocks"], dtype=np.float32)
        last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
        info = after
        served = np.asarray(after["step_service_blocks"], dtype=np.float64)
        in_a = bool(tgt >= 0 and env._in_range(focus_uav, tgt))
        if tgt >= 0 and float(served[tgt]) > 0.0 and in_a:
            blocks += 1
        rho_a = float(after["service_progress"][focus_uav])
        if rho_a + 1.0e-12 < rho_b and not (tgt >= 0 and float(served[tgt]) > 0.0):
            lost += 1
        if in_b and not in_a:
            out_of_range += 1
        rho_gain += max(rho_a - rho_b, 0.0)
        rates.append(float(after["rate_bps"][focus_uav]))
        sinrs.append(float(after["sinr"][focus_uav]))
        if kind == "C0" and recorded is not None:
            rec = recorded[step_i]
            if int(rec["t_before"]) != int(after["time_step"]) - 1:
                match_ok = False
                match_fail = "time_mismatch"
                break
            if not np.allclose(after["uav_positions_m"], rec["pos_after"], rtol=0.0, atol=ATOL_REPLAY):
                match_ok = False
                match_fail = "position_mismatch"
                break
            if abs(float(after["cumulative_payload_bits"]) - float(rec["payload_after"])) > ATOL_REPLAY:
                match_ok = False
                match_fail = "payload_mismatch"
                break
        if after.get("terminated"):
            break
    return {
        "kind": kind,
        "steps": horizon_left,
        "team_payload_delta": float(env.payload_bits) - start_payload,
        "focus_blocks": int(blocks),
        "focus_rho_gain": float(rho_gain),
        "mean_rate": float(np.mean(rates)) if rates else 0.0,
        "mean_sinr": float(np.mean(sinrs)) if sinrs else 0.0,
        "leave_range": int(out_of_range),
        "progress_loss_events": int(lost),
        "c0_match_ok": bool(match_ok) if kind == "C0" else None,
        "c0_match_fail": match_fail,
        "start_rho": start_rho,
        "end_rho": float(env.progress[focus_uav]),
        "aux_end": float(auxiliary_reward(env, option)["r_aux"]),
    }


__all__ = ["run_branch"]
