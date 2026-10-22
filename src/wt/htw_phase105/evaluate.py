"""Closed-loop FINAL-HIER / baseline episodes. Gate and fly-toward are frozen."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from wt.htw_phase100.high_level import decide, make_option
from wt.htw_phase100.rule_executor import RuleExecutorAdapter
from wt.htw_phase102.replay import classify_rho
from wt.htw_phase103.evaluate import episode_metrics
from wt.htw_phase103.gate import InRangeZeroMotionGate
from wt.htw_phase9.const import HOLD
from wt.l3b.env import L3BEnv, conflict_mask


def run_episode(
    seed: int,
    *,
    family: str,
    assign_fn: Callable[[L3BEnv, dict[str, Any], np.ndarray, np.ndarray, int], np.ndarray],
) -> dict[str, Any]:
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
    sinr_sum = 0.0
    rate_sum = 0.0
    n_slots = 0
    while not info.get("terminated", False):
        high_switched = False
        if pending <= 0:
            assignment = np.asarray(assign_fn(env, info, last_arrival, last_positions, pending), dtype=np.int64)
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
        raw = np.asarray(rule.act(env, option), dtype=np.float64)
        in_before = np.array(
            [False if int(option.association[u]) < 0 else bool(env._in_range(u, int(option.association[u]))) for u in range(env.config.num_uavs)],
            dtype=bool,
        )
        movement, flags = gate.apply(env, option, raw)
        after = env.step(movement, option.association)
        pending -= 1
        option.remaining_hold = int(pending)
        pos_after = np.asarray(after["uav_positions_m"], dtype=np.float64)
        executed = pos_after - pos_before
        served = np.asarray(after["step_service_blocks"], dtype=np.float64)
        blocked = conflict_mask(option.association, env.config.num_buoys)
        n_slots += env.config.num_uavs
        sinr_sum += float(np.sum(after["sinr"]))
        rate_sum += float(np.sum(after["rate_bps"]))
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
                    "gated": True,
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
                    "ppo_pre_ax": float("nan"),
                    "ppo_pre_ay": float("nan"),
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
    metrics = episode_metrics(events, family, int(seed))
    metrics["high_decisions"] = int(high_decisions)
    metrics["mean_sinr"] = float(sinr_sum / max(n_slots, 1))
    metrics["mean_active_rate_bps"] = float(rate_sum / max(n_slots, 1))
    metrics["exclusive_ok_frac"] = float(np.mean([e["exclusive_ok"] for e in events])) if events else 1.0
    metrics["conflict_slot_frac"] = float(np.mean([e["conflict"] for e in events])) if events else 0.0
    metrics["note"] = "pre-final independent benchmark; not final untouched test"
    metrics["payload_bits"] = float(info["cumulative_payload_bits"])
    return metrics


def learned_assigner(model):
    def _fn(env, info, last_arrival, last_positions, pending):
        return decide(model, env, info, last_arrival, last_positions, pending)

    return _fn


def rule_assigner(fn):
    def _fn(env, info, last_arrival, last_positions, pending):
        _ = info, last_arrival, last_positions, pending
        return fn(env)

    return _fn


__all__ = ["learned_assigner", "rule_assigner", "run_episode"]
