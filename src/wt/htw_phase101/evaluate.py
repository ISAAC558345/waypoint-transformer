"""Paired B0/B1 design evaluation. No confirmation seeds."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from wt.htw_phase100.high_level import decide, make_option
from wt.htw_phase100.rule_executor import RuleExecutorAdapter
from wt.htw_phase9.const import HOLD
from wt.htw_phase9.features import conflict_rate
from wt.l3b.env import L3BEnv
from wt.l3b.policies import fly_toward
from wt.round30.ppo import PPOAgent

from .features import encode_obs, stack_feature_dicts


def _metrics_from_steps(steps: list[dict[str, Any]], env: L3BEnv) -> dict[str, Any]:
    m_count = int(env.config.num_uavs)
    first_in_range = np.full(m_count, -1, dtype=np.int64)
    first_complete = np.full(m_count, -1, dtype=np.int64)
    in_range_slots = 0
    active_slots = 0
    progress_resets = 0
    distance = 0.0
    idle = 0
    prev_pos = None
    prev_progress = None
    for row in steps:
        assign = np.asarray(row["assignment"], dtype=np.int64)
        idle += int(np.sum(assign < 0))
        if prev_pos is not None:
            distance += float(np.sum(np.linalg.norm(row["uav_positions_m"] - prev_pos, axis=-1)))
        prev_pos = row["uav_positions_m"]
        prog = np.asarray(row["service_progress"], dtype=np.float64)
        if prev_progress is not None:
            progress_resets += int(np.sum((prev_progress > 1.0e-12) & (prog <= 1.0e-12)))
        prev_progress = prog
        for uav in range(m_count):
            tgt = int(assign[uav])
            if tgt < 0:
                continue
            active_slots += 1
            if bool(row["in_range"][uav]):
                in_range_slots += 1
                if int(first_in_range[uav]) < 0:
                    first_in_range[uav] = int(row["t"])
            if bool(row["completed_block"][uav]) and int(first_complete[uav]) < 0:
                first_complete[uav] = int(row["t"])
    n_in = int(np.sum(first_in_range >= 0))
    n_done = int(np.sum(first_complete >= 0))
    payload = float(steps[-1]["payload"]) if steps else 0.0
    return {
        "payload_bits": payload,
        "env_return": payload / max(float(env.config.block_payload_bits), 1.0e-9),
        "completed_blocks": int(round(payload / max(float(env.config.block_payload_bits), 1.0e-9))),
        "residual_queue_blocks": float(steps[-1]["queue_sum"]) if steps else 0.0,
        "dropped_blocks": float(steps[-1]["dropped"]) if steps else 0.0,
        "in_range_active_frac": float(in_range_slots) / max(float(active_slots), 1.0),
        "first_in_range_t": first_in_range.tolist(),
        "first_complete_t": first_complete.tolist(),
        "n_uav_reached_range": n_in,
        "n_uav_never_range": int(m_count - n_in),
        "n_uav_completed_service": n_done,
        "n_uav_never_completed": int(m_count - n_done),
        "mean_first_in_range_success_only": float(np.mean(first_in_range[first_in_range >= 0])) if n_in else None,
        "mean_first_complete_success_only": float(np.mean(first_complete[first_complete >= 0])) if n_done else None,
        "progress_reset_count": int(progress_resets),
        "flight_distance_m": float(distance),
        "conflict_rate": float(steps[-1]["conflict_rate"]) if steps else 0.0,
        "exclusive_ok": bool(steps[-1]["exclusive_ok"]) if steps else True,
        "idle_fraction": float(idle) / max(float(len(steps) * m_count), 1.0),
        "high_decisions": int(steps[-1]["high_decisions"]) if steps else 0,
        "note": "times of -1 / None mean the event never occurred; idle is not a data-service success",
    }


def rollout_executor(
    high_model,
    seed: int,
    *,
    schema: dict[str, Any] | None,
    agent: PPOAgent | None,
    mode: str,
) -> dict[str, Any]:
    env = L3BEnv()
    info = env.reset(seed=int(seed))
    last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
    last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
    pending = 0
    option = None
    high_decisions = 0
    steps = []
    rule = RuleExecutorAdapter()
    while not info.get("terminated", False):
        if pending <= 0:
            assignment = decide(high_model, env, info, last_arrival, last_positions, pending)
            pending = int(HOLD)
            option = make_option(env, info, assignment, pending)
            high_decisions += 1
        assert option is not None
        option.remaining_hold = int(pending)
        if mode == "B0":
            movement = rule.act(env, option)
            raw = movement
        elif mode == "B1":
            if agent is None or schema is None:
                raise RuntimeError("B1 requires the six-UAV PPO agent")
            obs = stack_feature_dicts([encode_obs(env, info, option, schema, last_arrival, last_positions)])
            decision = agent.act(obs, deterministic=True)
            raw = np.asarray(decision["pre_tanh"][0], dtype=np.float64)
            movement = np.asarray(decision["action"][0], dtype=np.float64)
            if movement.shape != (env.config.num_uavs, 2):
                raise RuntimeError("B1 action is not (6,2); refused fly-toward fallback")
        else:
            raise ValueError(mode)
        after = env.step(movement, option.association)
        pending -= 1
        option.remaining_hold = int(pending)
        served = np.asarray(after["step_service_blocks"], dtype=np.float64)
        in_range = np.array([env._in_range(uav, int(option.association[uav])) for uav in range(env.config.num_uavs)], dtype=bool)
        completed = np.zeros(env.config.num_uavs, dtype=bool)
        for uav, buoy in enumerate(np.asarray(option.association, dtype=np.int64)):
            if int(buoy) >= 0 and float(served[int(buoy)]) > 0.0 and env._in_range(uav, int(buoy)):
                completed[uav] = True
        steps.append(
            {
                "t": int(after["time_step"]),
                "assignment": np.asarray(option.association, dtype=np.int64).copy(),
                "movement": np.asarray(movement, dtype=np.float64).copy(),
                "raw_pre_tanh": np.asarray(raw, dtype=np.float64).copy(),
                "uav_positions_m": np.asarray(after["uav_positions_m"], dtype=np.float64).copy(),
                "service_progress": np.asarray(after["service_progress"], dtype=np.float64).copy(),
                "in_range": in_range,
                "completed_block": completed,
                "payload": float(after["cumulative_payload_bits"]),
                "queue_sum": float(np.sum(after["queue_blocks"])),
                "dropped": float(after["dropped_blocks"]),
                "conflict_rate": conflict_rate(option.association, env.config.num_buoys),
                "exclusive_ok": bool(after["exclusive_ok"]),
                "high_decisions": int(high_decisions),
            }
        )
        last_arrival = np.asarray(after["arrivals_blocks"], dtype=np.float32)
        last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
        info = after
    metrics = _metrics_from_steps(steps, env)
    metrics["scenario"] = int(seed)
    metrics["mode"] = mode
    return metrics


def evaluate_pair(high_model, seeds: tuple[int, ...], *, schema, agent: PPOAgent) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        b0 = rollout_executor(high_model, int(seed), schema=schema, agent=None, mode="B0")
        b1 = rollout_executor(high_model, int(seed), schema=schema, agent=agent, mode="B1")
        row = {"scenario": int(seed)}
        for key, value in b0.items():
            if key in {"scenario", "mode"}:
                continue
            row[f"b0_{key}"] = value
        for key, value in b1.items():
            if key in {"scenario", "mode"}:
                continue
            row[f"b1_{key}"] = value
        row["delta_payload_bits"] = float(b1["payload_bits"]) - float(b0["payload_bits"])
        row["payload_ratio_b1_over_b0"] = float(b1["payload_bits"]) / max(float(b0["payload_bits"]), 1.0e-9)
        rows.append(row)
    return rows


__all__ = ["evaluate_pair", "rollout_executor"]
