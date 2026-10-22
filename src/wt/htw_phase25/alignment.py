"""Compare oracle waypoint positions against current-state channel/queue stats."""
from __future__ import annotations

from typing import Any

import numpy as np

from wt.htwc_phase1.contract import DEFAULT_HORIZON
from wt.htwc_phase1.oracle import oracle_planner_window
from wt.round18.multi_env import EPSILON, _channel_kwargs, rate_matrix
from wt.round19.multi_env import SharedSpectrumMultiUAVEnv
from wt.round19.planner import independent_queue_rate_action
from wt.round30.scenarios import PPOScenario
from wt.maritime.channel import compute_channel_metrics


def _target_buoy(position: np.ndarray, queues: np.ndarray, buoys: np.ndarray, config: Any) -> int:
    rates, _ranges = rate_matrix(position[None, :], buoys, config)
    eligible = queues > EPSILON
    if not np.any(eligible):
        return -1
    score = np.where(eligible, queues / float(config.queue_capacity_bits) * rates[0], -np.inf)
    return int(np.argmax(score))


def _metrics_at(position: np.ndarray, buoys: np.ndarray, config: Any) -> Any:
    return compute_channel_metrics(position, buoys, **_channel_kwargs(config))


def alignment_rows(scenarios: tuple[PPOScenario, ...], horizon: int = DEFAULT_HORIZON) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m), options=scenario.options())
        _, info = env.reset(seed=scenario.seed)
        for time in range(4):
            queues = np.asarray(info["raw_queue_bits"], dtype=np.float64)
            buoys = np.asarray(info["buoy_positions_m"], dtype=np.float64)
            current = np.asarray(info["uav_positions_m"], dtype=np.float64)
            _label, path = oracle_planner_window(env, info, horizon=horizon)
            waypoint = path[-1]
            for uav in range(env.num_uavs):
                target = _target_buoy(current[uav], queues, buoys, env.config)
                if target < 0:
                    continue
                now = _metrics_at(current[uav], buoys, env.config)
                way = _metrics_at(waypoint[uav], buoys, env.config)
                best_now = int(np.argmax(now.rates_bps))
                best_way = int(np.argmax(way.rates_bps))
                rows.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "time": time,
                        "uav": uav,
                        "target_buoy": target,
                        "queue_target": float(queues[target]),
                        "queue_max": float(np.max(queues)),
                        "queue_target_is_max": int(target == int(np.argmax(queues))),
                        "distance_current_target_m": float(now.horizontal_distances_m[target]),
                        "distance_waypoint_target_m": float(way.horizontal_distances_m[target]),
                        "channel_gain_current_target": float(now.channel_gains[target]),
                        "channel_gain_waypoint_target": float(way.channel_gains[target]),
                        "rate_current_target_bps": float(now.rates_bps[target]),
                        "rate_waypoint_target_bps": float(way.rates_bps[target]),
                        "rate_current_best_bps": float(now.rates_bps[best_now]),
                        "rate_waypoint_best_bps": float(way.rates_bps[best_way]),
                        "waypoint_closer_to_target": int(way.horizontal_distances_m[target] < now.horizontal_distances_m[target] - 1.0e-9),
                        "waypoint_higher_rate_to_target": int(way.rates_bps[target] > now.rates_bps[target] + 1.0e-9),
                        "waypoint_higher_gain_to_target": int(way.channel_gains[target] > now.channel_gains[target] + 1.0e-12),
                    }
                )
            action = independent_queue_rate_action(current, queues, buoys, env.config)
            try:
                _, _reward, terminated, truncated, info = env.step(action, matching_mode="legacy")
            except RuntimeError as error:
                if "hard UAV separation projection did not converge" not in str(error):
                    raise
                break
            if terminated or truncated:
                break
    return rows


def summarize_alignment(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise RuntimeError("no waypoint-communication rows")
    rate_delta = [row["rate_waypoint_target_bps"] - row["rate_current_target_bps"] for row in rows]
    gain_delta = [row["channel_gain_waypoint_target"] - row["channel_gain_current_target"] for row in rows]
    dist_delta = [row["distance_waypoint_target_m"] - row["distance_current_target_m"] for row in rows]
    return {
        "count": float(len(rows)),
        "frac_waypoint_closer_to_target": float(np.mean([row["waypoint_closer_to_target"] for row in rows])),
        "frac_waypoint_higher_rate_to_target": float(np.mean([row["waypoint_higher_rate_to_target"] for row in rows])),
        "frac_waypoint_higher_gain_to_target": float(np.mean([row["waypoint_higher_gain_to_target"] for row in rows])),
        "frac_target_is_max_queue": float(np.mean([row["queue_target_is_max"] for row in rows])),
        "mean_rate_delta_target_bps": float(np.mean(rate_delta)),
        "mean_gain_delta_target": float(np.mean(gain_delta)),
        "mean_distance_delta_target_m": float(np.mean(dist_delta)),
        "mean_rate_current_target_bps": float(np.mean([row["rate_current_target_bps"] for row in rows])),
        "mean_rate_waypoint_target_bps": float(np.mean([row["rate_waypoint_target_bps"] for row in rows])),
    }


__all__ = ["alignment_rows", "summarize_alignment"]
