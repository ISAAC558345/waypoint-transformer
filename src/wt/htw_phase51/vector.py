"""Training vector: env r_comm plus controller-internal completion, no waypoint distance."""
from __future__ import annotations

from typing import Any

import numpy as np

from wt.htwc_phase1.validation import ConditionedVector
from wt.round30.features import feature_dict, stack_feature_dicts
from wt.round30.scenarios import PPOScenario

from .option import (
    COMPLETION_COEF,
    OPTION_HOLD,
    attach_option,
    completion_score,
    encode_option,
    oracle_assignment,
)


class OptionVector(ConditionedVector):
    def __init__(
        self,
        scenarios: tuple[PPOScenario, ...],
        count: int,
        seed: int,
        completion_coef: float = COMPLETION_COEF,
        hold: int = OPTION_HOLD,
    ) -> None:
        self.completion_coef = float(completion_coef)
        self.hold = int(hold)
        super().__init__(scenarios, count, seed, self._condition)

    def _ensure_option(self, world: dict[str, Any]) -> np.ndarray:
        ttl = int(world.get("option_ttl", 0))
        if ttl <= 0 or "option_assignment" not in world:
            world["option_assignment"] = oracle_assignment(world["info"], world["env"])
            world["option_ttl"] = self.hold
        return np.asarray(world["option_assignment"], dtype=np.int64)

    def _condition(self, info, env):
        for world in self.worlds:
            if world["info"] is info and world["env"] is env:
                option = encode_option(self._ensure_option(world))
                return attach_option(feature_dict(info, env.config), option)
        return attach_option(feature_dict(info, env.config), encode_option(oracle_assignment(info, env)))

    def features(self) -> dict[str, np.ndarray]:
        rows = []
        for world in self.worlds:
            option = encode_option(self._ensure_option(world))
            rows.append(attach_option(feature_dict(world["info"], world["env"].config), option))
        return stack_feature_dicts(rows)

    def step(self, actions: np.ndarray):
        if actions.shape != (len(self.worlds), 2, 2):
            raise ValueError("vector PPO action batch has invalid shape")
        rewards, dones, completed = [], [], []
        for index, (world, requested) in enumerate(zip(self.worlds, actions)):
            env, info = world["env"], world["info"]
            intended = self._ensure_option(world)
            before = np.asarray(info["uav_positions_m"], dtype=np.float64).copy()
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
            complete = completion_score(after["uav_to_buoy_assignment"], intended, env.num_uavs)
            ppo_reward = float(comm_reward) + self.completion_coef * complete
            world["return"] += float(comm_reward)
            world["arrivals"] += float(np.sum(after["raw_arrival_bits"]))
            world["projection_count"] += int(projection > 1.0e-8)
            world["projection_magnitude"] += projection
            world["minimum_distance"] = min(world["minimum_distance"], float(after["min_uav_distance_m"]))
            world["option_ttl"] = int(world.get("option_ttl", 1)) - 1
            self.environment_steps += 1
            rewards.append(float(ppo_reward))
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
                        "last_goal_completion": float(complete),
                    }
                )
                self.worlds[index] = self._new_world()
            else:
                world["info"] = after
        return self.features(), np.asarray(rewards, dtype=np.float32), np.asarray(dones, dtype=bool), completed


__all__ = ["OptionVector"]
