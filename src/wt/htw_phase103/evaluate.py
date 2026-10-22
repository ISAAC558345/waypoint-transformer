"""Closed-loop B0–B3 evaluation with optional in-range zero-motion gate."""
from __future__ import annotations

from typing import Any

import numpy as np

from wt.htw_phase100.high_level import decide, make_option
from wt.htw_phase100.rule_executor import RuleExecutorAdapter
from wt.htw_phase102.replay import _ppo_move, classify_rho
from wt.htw_phase9.const import HOLD
from wt.l3b.env import L3BEnv, conflict_mask
from wt.round30.ppo import PPOAgent

from .gate import InRangeZeroMotionGate


def run_episode(high_model, seed: int, *, family: str, schema: dict[str, Any] | None, agent: PPOAgent | None, gated: bool) -> dict[str, Any]:
    env = L3BEnv()
    info = env.reset(seed=int(seed))
    last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
    last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
    pending = 0
    option = None
    high_decisions = 0
    prev_assign = np.full(env.config.num_uavs, -2, dtype=np.int64)
    rule = RuleExecutorAdapter()
    gate = InRangeZeroMotionGate()
    events: list[dict[str, Any]] = []
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
        pos_before = np.asarray(env.uav_positions, dtype=np.float64).copy()
        prog_before = np.asarray(env.progress, dtype=np.float64).copy()
        queue_before = np.asarray(env.queue, dtype=np.float64).copy()
        served_before = np.asarray(env.served_k, dtype=np.int64).copy()
        payload_before = float(env.payload_bits)
        if family in {"B0", "B3"}:
            raw = np.asarray(rule.act(env, option), dtype=np.float64)
            pre = raw.copy()
        elif family in {"B1", "B2"}:
            if agent is None or schema is None:
                raise RuntimeError(f"{family} requires frozen PPO")
            pre, raw = _ppo_move(agent, schema, env, info, option, last_arrival, last_positions)
            raw = np.asarray(raw, dtype=np.float64)
            pre = np.asarray(pre, dtype=np.float64)
        else:
            raise ValueError(family)
        in_before = np.array(
            [False if int(option.association[u]) < 0 else bool(env._in_range(u, int(option.association[u]))) for u in range(env.config.num_uavs)],
            dtype=bool,
        )
        if gated:
            movement, flags = gate.apply(env, option, raw)
        else:
            flags = np.zeros(env.config.num_uavs, dtype=bool)
            movement = raw.copy()
        after = env.step(movement, option.association)
        pending -= 1
        option.remaining_hold = int(pending)
        pos_after = np.asarray(after["uav_positions_m"], dtype=np.float64)
        executed = pos_after - pos_before
        served = np.asarray(after["step_service_blocks"], dtype=np.float64)
        blocked = conflict_mask(option.association, env.config.num_buoys)
        for uav in range(env.config.num_uavs):
            tgt = int(option.association[uav])
            idle = tgt < 0
            empty = (not idle) and float(queue_before[tgt]) <= 1.0e-9
            in_b = bool(in_before[uav])
            in_after = False if idle else bool(env._in_range(uav, tgt))
            completed = (not idle) and float(served[tgt]) > 0.0 and in_after
            target_changed = int(tgt) != int(served_before[uav]) or (int(prev_assign[uav]) not in (-2,) and int(prev_assign[uav]) != int(tgt))
            labels = classify_rho(float(prog_before[uav]), float(after["service_progress"][uav]), bool(completed), bool(target_changed), bool(blocked[uav]), (not idle) and (not in_after), bool(empty), bool(after["terminated"]))
            dist_b = float("nan") if idle else float(np.linalg.norm(pos_before[uav] - env.buoy_positions[tgt]))
            dist_a = float("nan") if idle else float(np.linalg.norm(pos_after[uav] - env.buoy_positions[tgt]))
            events.append(
                {
                    "family": family,
                    "gated": bool(gated),
                    "scenario": int(seed),
                    "t_before": int(info["time_step"]),
                    "uav": int(uav),
                    "target": int(tgt),
                    "idle": bool(idle),
                    "queue_before": float("nan") if idle else float(queue_before[tgt]),
                    "dist_before_m": dist_b,
                    "dist_after_m": dist_a,
                    "remaining_hold_before": int(pending) + 1,
                    "raw_ax": float(raw[uav, 0]),
                    "raw_ay": float(raw[uav, 1]),
                    "ppo_pre_ax": float(pre[uav, 0]),
                    "ppo_pre_ay": float(pre[uav, 1]),
                    "gate": bool(flags[uav]),
                    "gated_ax": float(movement[uav, 0]),
                    "gated_ay": float(movement[uav, 1]),
                    "exec_dx_m": float(executed[uav, 0]),
                    "exec_dy_m": float(executed[uav, 1]),
                    "in_range_before": bool(in_b),
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
                    "dropped_blocks": float(after["dropped_blocks"]),
                    "queue_sum": float(np.sum(after["queue_blocks"])),
                    "labels": "|".join(labels),
                }
            )
        prev_assign = np.asarray(option.association, dtype=np.int64).copy()
        last_arrival = np.asarray(after["arrivals_blocks"], dtype=np.float32)
        last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
        info = after
    return {"events": events, "payload": float(info["cumulative_payload_bits"]), "family": family, "scenario": int(seed)}


def episode_metrics(events: list[dict[str, Any]], family: str, seed: int) -> dict[str, Any]:
    rows = [e for e in events if e["family"] == family and int(e["scenario"]) == int(seed)]
    m_slots = [e for e in rows]
    uavs = sorted({(e["scenario"], e["uav"]) for e in rows})
    completed_u = {(e["scenario"], e["uav"]) for e in rows if e["completed_block"]}
    active_q = [e for e in rows if (not e["idle"]) and (not e["empty_queue_before"])]
    flight = float(sum(np.hypot(e["exec_dx_m"], e["exec_dy_m"]) for e in rows))
    n_complete = sum(1 for e in rows if e["completed_block"])
    n_switch_loss = 0
    lost_mass = 0.0
    terminal_rho = 0.0
    for e in rows:
        labs = str(e["labels"]).split("|")
        if "TARGET_CHANGE" in labs:
            n_switch_loss += 1
            lost_mass += float(e["rho_before"])
        if "TERMINAL_UNFINISHED" in labs:
            terminal_rho += float(e["rho_after"])
    streak = 0
    best_streak = 0
    last_key = None
    for e in sorted(rows, key=lambda r: (r["uav"], r["t_before"])):
        key = (e["uav"], e["target"])
        serving = (not e["idle"]) and e["in_range_after"] and (not e["empty_queue_before"])
        if serving and key == last_key:
            streak += 1
        elif serving:
            streak = 1
            last_key = key
        else:
            streak = 0
            last_key = None
        best_streak = max(best_streak, streak)
    low_rate_gated = sum(1 for e in rows if e["gate"] and e["in_range_after"] and (not e["completed_block"]) and float(e["rate_bps"]) < 1.0e5)
    return {
        "family": family,
        "scenario": int(seed),
        "payload_bits": float(rows[-1]["team_payload_after"]) if rows else 0.0,
        "env_return": (float(rows[-1]["team_payload_after"]) / 1.0e6) if rows else 0.0,
        "completed_blocks": int(n_complete),
        "n_uav_never_completed": int(len(uavs) - len(completed_u)),
        "n_uav_completed": int(len(completed_u)),
        "residual_queue_blocks": float(rows[-1]["queue_sum"]) if rows else 0.0,
        "dropped_blocks": float(rows[-1]["dropped_blocks"]) if rows else 0.0,
        "active_queued_in_range_frac": (sum(1 for e in active_q if e["in_range_after"]) / max(len(active_q), 1)),
        "n_active_queued_slots": len(active_q),
        "n_in_range_queued_slots": sum(1 for e in active_q if e["in_range_after"]),
        "max_consecutive_service": int(best_streak),
        "normal_completion_resets": int(sum(1 for e in rows if "NORMAL_COMPLETION" in str(e["labels"]))),
        "target_change_loss_events": int(n_switch_loss),
        "target_change_lost_rho": float(lost_mass),
        "terminal_unfinished_rho": float(terminal_rho),
        "flight_distance_m": flight,
        "gate_frac": float(np.mean([e["gate"] for e in rows])) if rows else 0.0,
        "idle_frac": float(np.mean([e["idle"] for e in rows])) if rows else 0.0,
        "exclusive_ok_last": bool(rows[-1]["exclusive_ok"]) if rows else True,
        "gated_zero_input_nonzero_exec": int(sum(1 for e in rows if e["gate"] and (abs(e["gated_ax"]) + abs(e["gated_ay"]) < 1.0e-12) and np.hypot(e["exec_dx_m"], e["exec_dy_m"]) > 1.0e-9)),
        "gated_in_range_no_block": int(sum(1 for e in rows if e["gate"] and not e["completed_block"])),
        "low_rate_gated_unfinished": int(low_rate_gated),
        "note": "design evaluation; unfinished times are not filled with 0",
    }


__all__ = ["episode_metrics", "run_episode"]
