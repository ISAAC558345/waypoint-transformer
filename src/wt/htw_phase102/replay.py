"""Instrumented B0/B1 replay. Wrapper only; env.step is unchanged."""
from __future__ import annotations

from typing import Any

import numpy as np

from wt.htw_phase100.high_level import decide, make_option
from wt.htw_phase100.rule_executor import RuleExecutorAdapter
from wt.htw_phase101.features import encode_obs, stack_feature_dicts
from wt.htw_phase101.rewards import auxiliary_reward, env_reward
from wt.htw_phase9.const import HOLD
from wt.htw_phase9.features import conflict_rate
from wt.l3b.env import L3BEnv, conflict_mask
from wt.round30.ppo import PPOAgent

from .snapshot import capture_controller, capture_env


def _dist(env: L3BEnv, uav: int, tgt: int) -> float:
    if tgt < 0 or tgt >= int(env.config.num_buoys):
        return float("nan")
    return float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[tgt]))


def _ppo_move(agent: PPOAgent, schema: dict[str, Any], env: L3BEnv, info: dict[str, Any], option, last_arrival, last_positions) -> tuple[np.ndarray, np.ndarray]:
    obs = stack_feature_dicts([encode_obs(env, info, option, schema, last_arrival, last_positions)])
    decision = agent.act(obs, deterministic=True)
    raw = np.asarray(decision["pre_tanh"][0], dtype=np.float64)
    move = np.asarray(decision["action"][0], dtype=np.float64)
    if move.shape != (env.config.num_uavs, 2):
        raise RuntimeError("B1 action is not (6,2); refused fly-toward fallback")
    return raw, move


def classify_rho(prog_before: float, prog_after: float, completed: bool, target_changed: bool, conflict: bool, out_of_range: bool, empty_queue: bool, terminated: bool) -> list[str]:
    labels: list[str] = []
    dropped = float(prog_before) > 1.0e-12 and float(prog_after) <= 1.0e-12
    if completed:
        labels.append("NORMAL_COMPLETION")
    if target_changed and dropped:
        labels.append("TARGET_CHANGE")
    if empty_queue:
        labels.append("EMPTY_QUEUE")
    if conflict:
        labels.append("INVALID_OR_CONFLICT")
    if out_of_range:
        labels.append("OUT_OF_RANGE")
    if terminated and float(prog_after) > 1.0e-12:
        labels.append("TERMINAL_UNFINISHED")
    if dropped and not completed and not target_changed:
        labels.append("OTHER_OR_UNRESOLVED")
    if not labels:
        labels.append("NONE")
    return labels


def replay_episode(high_model, seed: int, *, mode: str, schema: dict[str, Any] | None, agent: PPOAgent | None) -> dict[str, Any]:
    env = L3BEnv()
    info = env.reset(seed=int(seed))
    last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
    last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
    pending = 0
    option = None
    high_decisions = 0
    prev_assign = np.full(env.config.num_uavs, -2, dtype=np.int64)
    rule = RuleExecutorAdapter()
    events: list[dict[str, Any]] = []
    start_snaps: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    while not info.get("terminated", False):
        high_switched = False
        if pending <= 0:
            assignment = decide(high_model, env, info, last_arrival, last_positions, pending)
            pending = int(HOLD)
            option = make_option(env, info, assignment, pending)
            high_decisions += 1
            high_switched = True
        assert option is not None
        option.remaining_hold = int(pending)
        start_snaps.append(
            {
                "t": int(info["time_step"]),
                "env": capture_env(env),
                "ctrl": capture_controller(pending, option, last_arrival, last_positions, high_decisions),
            }
        )
        pos_before = np.asarray(env.uav_positions, dtype=np.float64).copy()
        prog_before = np.asarray(env.progress, dtype=np.float64).copy()
        queue_before = np.asarray(env.queue, dtype=np.float64).copy()
        served_before = np.asarray(env.served_k, dtype=np.int64).copy()
        payload_before = float(env.payload_bits)
        if mode == "B0":
            movement = rule.act(env, option)
            raw = movement.copy()
        elif mode == "B1":
            if agent is None or schema is None:
                raise RuntimeError("B1 requires frozen PPO")
            raw, movement = _ppo_move(agent, schema, env, info, option, last_arrival, last_positions)
        else:
            raise ValueError(mode)
        after = env.step(movement, option.association)
        pending -= 1
        option.remaining_hold = int(pending)
        pos_after = np.asarray(after["uav_positions_m"], dtype=np.float64)
        executed = (pos_after - pos_before) / max(float(env.config.step_length_m), 1.0e-9)
        served = np.asarray(after["step_service_blocks"], dtype=np.float64)
        blocked = conflict_mask(option.association, env.config.num_buoys)
        r_env = env_reward(payload_before, float(after["cumulative_payload_bits"]), env.config.block_payload_bits)
        aux = auxiliary_reward(env, option)
        for uav in range(env.config.num_uavs):
            tgt = int(option.association[uav])
            idle = tgt < 0
            q = float("nan") if idle else float(queue_before[tgt])
            in_before = False if idle else bool(np.linalg.norm(pos_before[uav] - env.buoy_positions[tgt]) <= env.config.service_radius_m)
            in_after = False if idle else bool(env._in_range(uav, tgt))
            completed = (not idle) and float(served[tgt]) > 0.0 and in_after
            target_changed = int(tgt) != int(served_before[uav]) or (int(prev_assign[uav]) not in (-2,) and int(prev_assign[uav]) != int(tgt))
            empty = (not idle) and float(queue_before[tgt]) <= 1.0e-9
            labels = classify_rho(
                float(prog_before[uav]),
                float(after["service_progress"][uav]),
                bool(completed),
                bool(target_changed),
                bool(blocked[uav]),
                (not idle) and (not in_after),
                bool(empty),
                bool(after["terminated"]),
            )
            events.append(
                {
                    "mode": mode,
                    "scenario": int(seed),
                    "t_before": int(info["time_step"]),
                    "t_after": int(after["time_step"]),
                    "uav": int(uav),
                    "target": int(tgt),
                    "idle": bool(idle),
                    "remaining_hold_before": int(pending) + 1,
                    "remaining_hold_after": int(pending),
                    "remaining_horizon_before": int(env.config.episode_horizon) - int(info["time_step"]),
                    "queue_target_before": q,
                    "queue_target_after": float("nan") if idle else float(after["queue_blocks"][tgt]),
                    "dist_before_m": _dist(env, uav, tgt) if False else (float("nan") if idle else float(np.linalg.norm(pos_before[uav] - env.buoy_positions[tgt]))),
                    "dist_after_m": float("nan") if idle else float(np.linalg.norm(pos_after[uav] - env.buoy_positions[tgt])),
                    "raw_ax": float(raw[uav, 0]),
                    "raw_ay": float(raw[uav, 1]),
                    "exec_ax": float(executed[uav, 0]),
                    "exec_ay": float(executed[uav, 1]),
                    "in_range_before": bool(in_before),
                    "in_range_after": bool(in_after),
                    "sinr": float(after["sinr"][uav]),
                    "rate_bps": float(after["rate_bps"][uav]),
                    "rho_before": float(prog_before[uav]),
                    "rho_after": float(after["service_progress"][uav]),
                    "completed_block": bool(completed),
                    "payload_increment": float(after["cumulative_payload_bits"]) - payload_before,
                    "team_payload_after": float(after["cumulative_payload_bits"]),
                    "high_switched": bool(high_switched),
                    "conflict": bool(blocked[uav]),
                    "exclusive_ok": bool(after["exclusive_ok"]),
                    "empty_queue_before": bool(empty),
                    "r_env": float(r_env) if uav == 0 else 0.0,
                    "r_aux": float(aux["r_aux"]) if uav == 0 else 0.0,
                    "aux_in_range_count": float(aux["in_range_count"]) if uav == 0 else 0.0,
                    "labels": "|".join(labels),
                }
            )
        trace.append(
            {
                "t_before": int(info["time_step"]),
                "pos_after": pos_after.copy(),
                "payload_after": float(after["cumulative_payload_bits"]),
                "progress_after": np.asarray(after["service_progress"], dtype=np.float64).copy(),
                "assignment": np.asarray(option.association, dtype=np.int64).copy(),
            }
        )
        prev_assign = np.asarray(option.association, dtype=np.int64).copy()
        last_arrival = np.asarray(after["arrivals_blocks"], dtype=np.float32)
        last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
        info = after
    return {"events": events, "start_snaps": start_snaps, "trace": trace, "payload": float(info["cumulative_payload_bits"]), "mode": mode, "scenario": int(seed)}


__all__ = ["_ppo_move", "classify_rho", "replay_episode"]
