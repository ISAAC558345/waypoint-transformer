"""Closed-loop traces for original 9.4.5 path vs TaskOption adapter path."""
from __future__ import annotations

from typing import Any

import numpy as np

from wt.htw_phase943.order_free_decoder import OrderFreeAssignmentModel
from wt.l3b.env import L3BEnv
from wt.l3b.policies import fly_toward

from .contract import TaskOption
from .high_level import decide, make_option
from .protocol import ATOL_MOVE, ATOL_PAYLOAD, ATOL_POS, ATOL_PROGRESS, ATOL_QUEUE, HOLD
from .rule_executor import RuleExecutorAdapter


def _service_flags(env: L3BEnv, assignment: np.ndarray, after: dict[str, Any]) -> dict[str, Any]:
    in_range = np.array([env._in_range(uav, int(assignment[uav])) for uav in range(env.config.num_uavs)], dtype=bool)
    served = np.asarray(after["step_service_blocks"], dtype=np.float64)
    uav_served = np.zeros(env.config.num_uavs, dtype=bool)
    for uav, buoy in enumerate(np.asarray(assignment, dtype=np.int64)):
        if int(buoy) >= 0 and float(served[int(buoy)]) > 0.0 and env._in_range(uav, int(buoy)):
            uav_served[uav] = True
    return {"in_range": in_range, "uav_completed_block": uav_served, "step_service_blocks": served}


def _snapshot(env: L3BEnv, info: dict[str, Any], assignment: np.ndarray, movement: np.ndarray, pending: int, after: dict[str, Any]) -> dict[str, Any]:
    flags = _service_flags(env, assignment, after)
    return {
        "t": int(after["time_step"]),
        "assignment": np.asarray(assignment, dtype=np.int64).copy(),
        "remaining_hold": int(pending),
        "movement": np.asarray(movement, dtype=np.float64).copy(),
        "uav_positions_m": np.asarray(after["uav_positions_m"], dtype=np.float64).copy(),
        "queue_blocks": np.asarray(after["queue_blocks"], dtype=np.float64).copy(),
        "service_progress": np.asarray(after["service_progress"], dtype=np.float64).copy(),
        "step_service_blocks": flags["step_service_blocks"].copy(),
        "payload": float(after["cumulative_payload_bits"]),
        "payload_increment": float(after["cumulative_payload_bits"]) - float(info["cumulative_payload_bits"]),
        "in_range": flags["in_range"].copy(),
        "uav_completed_block": flags["uav_completed_block"].copy(),
    }


def _episode_metrics(steps: list[dict[str, Any]], env: L3BEnv) -> dict[str, Any]:
    m = int(env.config.num_uavs)
    first_in_range = np.full(m, -1, dtype=np.int64)
    first_complete = np.full(m, -1, dtype=np.int64)
    progress_resets = 0
    distance = 0.0
    prev_pos = None
    prev_progress = None
    for row in steps:
        if prev_pos is not None:
            distance += float(np.sum(np.linalg.norm(row["uav_positions_m"] - prev_pos, axis=-1)))
        prev_pos = row["uav_positions_m"]
        prog = np.asarray(row["service_progress"], dtype=np.float64)
        if prev_progress is not None:
            progress_resets += int(np.sum((prev_progress > 1.0e-12) & (prog <= 1.0e-12)))
        prev_progress = prog
        for uav in range(m):
            tgt = int(row["assignment"][uav])
            if tgt < 0:
                continue
            if int(first_in_range[uav]) < 0 and bool(row["in_range"][uav]):
                first_in_range[uav] = int(row["t"])
            if int(first_complete[uav]) < 0 and bool(row["uav_completed_block"][uav]):
                first_complete[uav] = int(row["t"])
    completed_blocks = int(round(float(steps[-1]["payload"]) / float(env.config.block_payload_bits))) if steps else 0
    return {
        "first_in_range_t": first_in_range.tolist(),
        "first_complete_t": first_complete.tolist(),
        "incomplete_in_range": [int(first_in_range[i]) < 0 for i in range(m)],
        "incomplete_service": [int(first_complete[i]) < 0 for i in range(m)],
        "progress_reset_count": int(progress_resets),
        "completed_blocks": completed_blocks,
        "flight_distance_m": float(distance),
        "episode_payload_bits": float(steps[-1]["payload"]) if steps else 0.0,
        "note": "idle is not counted as a successful data service; -1 times mean never occurred",
    }


def run_path_a(model: OrderFreeAssignmentModel, seed: int) -> dict[str, Any]:
    env = L3BEnv()
    info = env.reset(seed=int(seed))
    last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
    last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
    pending = 0
    assignment = np.full(env.config.num_uavs, -1, dtype=np.int64)
    steps = []
    while not info.get("terminated", False):
        if pending <= 0:
            assignment = decide(model, env, info, last_arrival, last_positions, pending)
            pending = int(HOLD)
        movement = fly_toward(env, assignment)
        after = env.step(movement, assignment)
        pending -= 1
        steps.append(_snapshot(env, info, assignment, movement, pending, after))
        last_arrival = np.asarray(after["arrivals_blocks"], dtype=np.float32)
        last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
        info = after
    return {"steps": steps, "metrics": _episode_metrics(steps, env), "path": "A_original_fly_toward"}


def run_path_b(model: OrderFreeAssignmentModel, seed: int) -> dict[str, Any]:
    env = L3BEnv()
    info = env.reset(seed=int(seed))
    last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
    last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
    pending = 0
    option: TaskOption | None = None
    executor = RuleExecutorAdapter()
    steps = []
    while not info.get("terminated", False):
        if pending <= 0:
            assignment = decide(model, env, info, last_arrival, last_positions, pending)
            pending = int(HOLD)
            option = make_option(env, info, assignment, pending)
        assert option is not None
        option.remaining_hold = int(pending)
        movement = executor.act(env, option)
        after = env.step(movement, option.association)
        pending -= 1
        option.remaining_hold = int(pending)
        steps.append(_snapshot(env, info, option.association, movement, pending, after))
        last_arrival = np.asarray(after["arrivals_blocks"], dtype=np.float32)
        last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
        info = after
    return {"steps": steps, "metrics": _episode_metrics(steps, env), "path": "B_task_option_adapter"}


def compare_traces(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    sa, sb = a["steps"], b["steps"]
    equal_len = len(sa) == len(sb)
    mismatches = []
    n = min(len(sa), len(sb))
    for i in range(n):
        ra, rb = sa[i], sb[i]
        checks = {
            "assignment": np.array_equal(ra["assignment"], rb["assignment"]),
            "remaining_hold": int(ra["remaining_hold"]) == int(rb["remaining_hold"]),
            "movement": np.allclose(ra["movement"], rb["movement"], rtol=0.0, atol=ATOL_MOVE),
            "uav_positions_m": np.allclose(ra["uav_positions_m"], rb["uav_positions_m"], rtol=0.0, atol=ATOL_POS),
            "queue_blocks": np.allclose(ra["queue_blocks"], rb["queue_blocks"], rtol=0.0, atol=ATOL_QUEUE),
            "service_progress": np.allclose(ra["service_progress"], rb["service_progress"], rtol=0.0, atol=ATOL_PROGRESS),
            "step_service_blocks": np.array_equal(ra["step_service_blocks"], rb["step_service_blocks"]),
            "payload_increment": abs(float(ra["payload_increment"]) - float(rb["payload_increment"])) <= ATOL_PAYLOAD,
            "payload": abs(float(ra["payload"]) - float(rb["payload"])) <= ATOL_PAYLOAD,
        }
        failed = [key for key, ok in checks.items() if not ok]
        if failed:
            mismatches.append({"step": i, "t": ra["t"], "failed": failed})
    return {
        "equal_length": equal_len,
        "n_steps": n,
        "n_mismatches": len(mismatches),
        "mismatches_head": mismatches[:8],
        "stepwise_equivalent": bool(equal_len and not mismatches),
        "final_payload_a": float(sa[-1]["payload"]) if sa else float("nan"),
        "final_payload_b": float(sb[-1]["payload"]) if sb else float("nan"),
        "metrics_a": a["metrics"],
        "metrics_b": b["metrics"],
        "atol": {"move": ATOL_MOVE, "pos": ATOL_POS, "queue": ATOL_QUEUE, "progress": ATOL_PROGRESS, "payload": ATOL_PAYLOAD},
    }


__all__ = ["compare_traces", "run_path_a", "run_path_b"]
