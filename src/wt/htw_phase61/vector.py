"""L2-A PPO vector. Environment reward is payload-block r_comm only."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from wt.l2a.env import L2AEnv
from wt.l2a.policies import myopic_assignment, oracle_option_assignment

from .features import (
    MAX_UAVS,
    attach_option,
    encode_option,
    feature_dict_l2a,
    in_range_completion,
    service_success,
    stack_feature_dicts,
)

AssignFn = Callable[[L2AEnv], np.ndarray]


class L2AVector:
    def __init__(
        self,
        seeds: tuple[int, ...],
        count: int,
        seed: int,
        *,
        use_option: bool,
    ) -> None:
        self.seeds, self.rng, self.environment_steps = seeds, np.random.default_rng(seed), 0
        self.use_option = bool(use_option)
        self.assign_fn: AssignFn = oracle_option_assignment if self.use_option else myopic_assignment
        self.worlds = [self._new_world() for _ in range(count)]

    def _new_world(self) -> dict[str, Any]:
        scenario_seed = int(self.seeds[int(self.rng.integers(len(self.seeds)))])
        env = L2AEnv()
        info = env.reset(seed=scenario_seed)
        return {
            "env": env,
            "info": info,
            "return": 0.0,
            "seed": scenario_seed,
            "movement_m": 0.0,
            "completions": [],
            "services": [],
        }

    def _features_for(self, env: L2AEnv, info: dict[str, Any]) -> dict[str, np.ndarray]:
        features = feature_dict_l2a(info, env)
        if not self.use_option:
            return features
        return attach_option(features, encode_option(self.assign_fn(env)))

    def features(self) -> dict[str, np.ndarray]:
        return stack_feature_dicts([self._features_for(world["env"], world["info"]) for world in self.worlds])

    def step(self, actions: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, float]]]:
        if actions.shape != (len(self.worlds), MAX_UAVS, 2):
            raise ValueError("Phase-6.1 PPO action batch has invalid shape")
        rewards, dones, completed = [], [], []
        for index, (world, requested) in enumerate(zip(self.worlds, actions)):
            env = world["env"]
            before_pos = np.asarray(env.uav_positions, dtype=np.float64).copy()
            before_payload = float(env.payload_bits)
            intended = self.assign_fn(env)
            after = env.step(requested, intended)
            delta_blocks = (float(after["cumulative_payload_bits"]) - before_payload) / float(env.config.block_payload_bits)
            moved = float(np.sum(np.linalg.norm(np.asarray(after["uav_positions_m"], dtype=np.float64) - before_pos, axis=-1)))
            complete = in_range_completion(env, intended)
            served = service_success(after, intended)
            world["return"] += float(delta_blocks)
            world["movement_m"] += moved
            world["completions"].append(complete)
            world["services"].append(served)
            self.environment_steps += 1
            rewards.append(float(delta_blocks))
            done = bool(after["terminated"])
            dones.append(done)
            if done:
                completed.append(
                    {
                        "payload_bits": float(after["cumulative_payload_bits"]),
                        "return": float(world["return"]),
                        "residual_queue_blocks": float(np.sum(after["queue_blocks"])),
                        "goal_completion_rate": float(np.mean(world["completions"])),
                        "service_success_rate": float(np.mean(world["services"])),
                        "movement_m": float(world["movement_m"]),
                        "dropped_blocks": float(after["dropped_blocks"]),
                    }
                )
                self.worlds[index] = self._new_world()
            else:
                world["info"] = after
        return self.features(), np.asarray(rewards, dtype=np.float32), np.asarray(dones, dtype=bool), completed


__all__ = ["L2AVector"]
