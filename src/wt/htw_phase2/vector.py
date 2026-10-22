"""Low-level training vector: oracle TW in the actor, tracking aux in PPO rewards.

PPOAgent.update, GAE, clip, and the environment reward function are untouched.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from wt.htwc_phase1.validation import Conditioner, ConditionedVector
from wt.round30.features import MAX_UAVS
from wt.round30.scenarios import PPOScenario

from .tracking import auxiliary_reward, normalize_positions_m, tracking_distance, waypoint_from_uav_features


class HierarchicalTrackingVector(ConditionedVector):
    """Same frozen env.step as Round-30; PPO sees r_comm - lambda * d(p, W)."""

    def __init__(
        self,
        scenarios: tuple[PPOScenario, ...],
        count: int,
        seed: int,
        conditioner: Conditioner,
        tracking_lambda: float,
    ) -> None:
        self.tracking_lambda = float(tracking_lambda)
        self.last_track_dist = np.zeros(count, dtype=np.float32)
        self.last_track_current = np.zeros((count, MAX_UAVS, 2), dtype=np.float32)
        self.last_track_waypoint = np.zeros((count, MAX_UAVS, 2), dtype=np.float32)
        self.last_track_mask = np.zeros((count, MAX_UAVS), dtype=np.float32)
        self.last_step_scale = np.zeros((count, 2), dtype=np.float32)
        super().__init__(scenarios, count, seed, conditioner)

    def step(self, actions: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, float]]]:
        if actions.shape != (len(self.worlds), 2, 2):
            raise ValueError("vector PPO action batch has invalid shape")
        rewards, dones, completed, distances = [], [], [], []
        for index, (world, requested) in enumerate(zip(self.worlds, actions)):
            env, info = world["env"], world["info"]
            features = self.conditioner(info, env)
            waypoint = waypoint_from_uav_features(features["uav"])
            before = np.asarray(info["uav_positions_m"], dtype=np.float64).copy()
            self.last_track_current[index] = normalize_positions_m(before, env.config).astype(np.float32)
            self.last_track_waypoint[index] = np.asarray(waypoint, dtype=np.float32)
            self.last_track_mask[index] = np.asarray(features["uav_mask"], dtype=np.float32)
            self.last_step_scale[index] = np.array(
                (
                    2.0 * float(env.config.max_speed) * float(env.config.delta_t) / float(env.config.area_width),
                    2.0 * float(env.config.max_speed) * float(env.config.delta_t) / float(env.config.area_height),
                ),
                dtype=np.float32,
            )
            fallback = False
            try:
                _, comm_reward, terminated, truncated, after = env.step(requested, matching_mode="legacy")
            except RuntimeError as error:
                if "hard UAV separation projection did not converge" not in str(error):
                    raise
                requested, fallback = np.zeros_like(requested), True
                _, comm_reward, terminated, truncated, after = env.step(requested, matching_mode="legacy")
            executed = (np.asarray(after["uav_positions_m"], dtype=np.float64) - before) / float(env.config.max_speed * env.config.delta_t)
            projection = float(np.linalg.norm(executed - requested))
            done = bool(terminated or truncated)
            after_xy = normalize_positions_m(after["uav_positions_m"], env.config)
            distance = tracking_distance(after_xy, waypoint, features["uav_mask"])
            ppo_reward = auxiliary_reward(float(comm_reward), distance, self.tracking_lambda)
            world["return"] += float(comm_reward)
            world["arrivals"] += float(np.sum(after["raw_arrival_bits"]))
            world["projection_count"] += int(projection > 1.0e-8)
            world["projection_magnitude"] += projection
            world["minimum_distance"] = min(world["minimum_distance"], float(after["min_uav_distance_m"]))
            self.environment_steps += 1
            rewards.append(float(ppo_reward))
            distances.append(float(distance))
            dones.append(done)
            if done:
                scenario = world["scenario"]
                completed.append(
                    {
                        "payload_bits": float(after["cumulative_payload_bits"]),
                        "return": float(world["return"]),
                        "completion": float(after["cumulative_payload_bits"]) / max(float(sum(scenario.initial_queue_bits) + world["arrivals"]), 1.0),
                        "residual_queue_bits": float(np.sum(after["raw_queue_bits"])),
                        "projection_rate": float(world["projection_count"] / max(int(after["time_step"]), 1)),
                        "projection_magnitude": float(world["projection_magnitude"] / max(int(after["time_step"]), 1)),
                        "minimum_uav_distance_m": float(world["minimum_distance"]),
                        "fallback": float(fallback),
                        "finish_environment_step": float(self.environment_steps),
                        "last_tracking_distance": float(distance),
                    }
                )
                self.worlds[index] = self._new_world()
            else:
                world["info"] = after
        self.last_track_dist = np.asarray(distances, dtype=np.float32)
        return self.features(), np.asarray(rewards, dtype=np.float32), np.asarray(dones, dtype=bool), completed


__all__ = ["HierarchicalTrackingVector"]
