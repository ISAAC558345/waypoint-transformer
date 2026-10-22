"""Deterministic evaluation on new Round-30 scenarios only."""
from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from wt.round19.multi_env import SharedSpectrumMultiUAVEnv

from .features import feature_dict, stack_feature_dicts
from .ppo import PPOAgent
from .scenarios import PPOScenario


ActionFn = Callable[[SharedSpectrumMultiUAVEnv, Mapping[str, Any]], np.ndarray]


def rollout(scenario: PPOScenario, *, population: str, policy_id: str, model_seed: int, action_fn: ActionFn, matching_mode: str = "legacy") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m, dtype=np.float64), options=scenario.options())
    _, info = env.reset(seed=scenario.seed)
    total_return, arrivals, done, steps = 0.0, 0.0, False, []
    while not done:
        before = np.asarray(info["uav_positions_m"], dtype=np.float64).copy()
        requested, fallback = np.asarray(action_fn(env, info), dtype=np.float64), False
        if requested.shape != env.action_shape or not np.all(np.isfinite(requested)):
            raise RuntimeError("Round-30 policy emitted an invalid action")
        try:
            _, reward, terminated, truncated, after = env.step(requested, matching_mode=matching_mode)
        except RuntimeError as error:
            if "hard UAV separation projection did not converge" not in str(error):
                raise
            requested = np.zeros_like(requested); fallback = True
            _, reward, terminated, truncated, after = env.step(requested, matching_mode=matching_mode)
        executed = (np.asarray(after["uav_positions_m"], dtype=np.float64) - before) / float(env.config.max_speed * env.config.delta_t)
        projection_magnitude = float(np.linalg.norm(executed - requested))
        projected = bool(projection_magnitude > 1.0e-8)
        total_return += float(reward); arrivals += float(np.sum(after["raw_arrival_bits"]))
        steps.append({
            "population": population, "scenario_id": scenario.scenario_id, "scenario_seed": scenario.seed,
            "policy_id": policy_id, "model_seed": model_seed, "timestep": int(after["time_step"]),
            "num_buoys": scenario.num_buoys, "eta": scenario.eta, "raw_sampled_action": requested,
            "executed_projected_action": executed, "projection_activated": projected,
            "projection_magnitude": projection_magnitude, "safety_projection_fallback": fallback,
            "reward": float(reward), "payload_bits_step": float(np.sum(after["transmitted_payload_bits"])),
            "interference_conflict": bool(after["interference_conflict"]),
            "min_uav_distance_m": float(after["min_uav_distance_m"]),
            "uav_positions_m": np.asarray(after["uav_positions_m"], dtype=np.float64),
        })
        info, done = after, bool(terminated or truncated)
    payload_by_buoy = np.asarray(info["cumulative_payload_bits_per_buoy"], dtype=np.float64)
    offered = float(sum(scenario.initial_queue_bits) + arrivals)
    fairness_denominator = float(len(payload_by_buoy) * np.square(payload_by_buoy).sum())
    return {
        "population": population, "scenario_id": scenario.scenario_id, "scenario_seed": scenario.seed,
        "policy_id": policy_id, "model_seed": model_seed, "num_buoys": scenario.num_buoys,
        "eta": scenario.eta, "queue_pressure": scenario.queue_pressure, "geometry": scenario.geometry,
        "initial_uav_configuration": scenario.initial_uav_configuration,
        "cumulative_payload_bits": float(info["cumulative_payload_bits"]),
        "completion": float(info["cumulative_payload_bits"]) / max(offered, 1.0),
        "residual_queue_bits": float(np.sum(info["raw_queue_bits"])), "episode_return": total_return,
        "jain_fairness": float(payload_by_buoy.sum() ** 2 / fairness_denominator) if fairness_denominator > 0.0 else 0.0,
        "team_path_length_m": float(np.sum(info["cumulative_distance_m_per_uav"])),
        "energy": float(np.sum(info["cumulative_energy_consumption_per_uav"])),
        "projection_rate": float(np.mean([row["projection_activated"] for row in steps])),
        "mean_projection_magnitude": float(np.mean([row["projection_magnitude"] for row in steps])),
        "minimum_uav_distance_m": float(min(row["min_uav_distance_m"] for row in steps)),
        "interference_conflict": bool(any(row["interference_conflict"] for row in steps)),
        "episode_length": len(steps), "matching_mode": matching_mode,
    }, steps


def ppo_action(agent: PPOAgent) -> ActionFn:
    def action(env: SharedSpectrumMultiUAVEnv, info: Mapping[str, Any]) -> np.ndarray:
        return agent.act(stack_feature_dicts([feature_dict(info, env.config)]), deterministic=True)["action"][0]
    return action


__all__ = ["ppo_action", "rollout"]
