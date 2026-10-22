"""Goal-conditioned L2-A vector. Env r_comm is unchanged; extras are controller-only."""
from __future__ import annotations

from typing import Any

import numpy as np

from wt.htw_phase61.features import MAX_UAVS, attach_option, encode_option, feature_dict_l2a, in_range_completion, service_success, stack_feature_dicts
from wt.htw_phase61.vector import L2AVector
from wt.l2a.env import L2AEnv
from wt.l2a.policies import oracle_option_assignment

from .rewards import low_level_reward, mean_target_distance_slots


class GoalOptionVector(L2AVector):
    def __init__(
        self,
        seeds: tuple[int, ...],
        count: int,
        seed: int,
        *,
        lambda_c: float,
        lambda_d: float,
    ) -> None:
        self.lambda_c = float(lambda_c)
        self.lambda_d = float(lambda_d)
        super().__init__(seeds, count, seed, use_option=True)

    def _new_world(self) -> dict[str, Any]:
        world = super()._new_world()
        world["env_return"] = 0.0
        world["first_in_range"] = np.full(MAX_UAVS, -1, dtype=np.int64)
        world["first_service"] = np.full(MAX_UAVS, -1, dtype=np.int64)
        return world

    def step(self, actions: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, float]]]:
        if actions.shape != (len(self.worlds), MAX_UAVS, 2):
            raise ValueError("Phase-6.2 PPO action batch has invalid shape")
        rewards, dones, completed = [], [], []
        for index, (world, requested) in enumerate(zip(self.worlds, actions)):
            env: L2AEnv = world["env"]
            before_pos = np.asarray(env.uav_positions, dtype=np.float64).copy()
            before_payload = float(env.payload_bits)
            intended = oracle_option_assignment(env)
            after = env.step(requested, intended)
            r_comm = (float(after["cumulative_payload_bits"]) - before_payload) / float(env.config.block_payload_bits)
            complete = in_range_completion(env, intended)
            served = service_success(after, intended)
            dist_slots = mean_target_distance_slots(env, intended)
            r_low = low_level_reward(r_comm, complete, dist_slots, self.lambda_c, self.lambda_d)
            moved = float(np.sum(np.linalg.norm(np.asarray(after["uav_positions_m"], dtype=np.float64) - before_pos, axis=-1)))
            service = np.asarray(after["step_service_blocks"], dtype=np.float64)
            for uav, buoy in enumerate(np.asarray(intended, dtype=np.int64)):
                if int(buoy) < 0:
                    continue
                if int(world["first_in_range"][uav]) < 0 and env._in_range(uav, int(buoy)):
                    world["first_in_range"][uav] = int(after["time_step"])
                if int(world["first_service"][uav]) < 0 and float(service[int(buoy)]) > 0.0:
                    world["first_service"][uav] = int(after["time_step"])
            world["return"] += float(r_comm)
            world["env_return"] += float(r_comm)
            world["movement_m"] += moved
            world["completions"].append(complete)
            world["services"].append(served)
            self.environment_steps += 1
            rewards.append(float(r_low))
            done = bool(after["terminated"])
            dones.append(done)
            if done:
                horizon = float(env.config.episode_horizon)
                in_range_times = [float(value) if int(value) >= 0 else horizon for value in world["first_in_range"]]
                service_times = [float(value) if int(value) >= 0 else horizon for value in world["first_service"]]
                completed.append(
                    {
                        "payload_bits": float(after["cumulative_payload_bits"]),
                        "return": float(world["env_return"]),
                        "residual_queue_blocks": float(np.sum(after["queue_blocks"])),
                        "goal_completion_rate": float(np.mean(world["completions"])),
                        "service_success_rate": float(np.mean(world["services"])),
                        "movement_m": float(world["movement_m"]),
                        "dropped_blocks": float(after["dropped_blocks"]),
                        "time_to_in_range": float(np.mean(in_range_times)),
                        "time_to_service": float(np.mean(service_times)),
                    }
                )
                self.worlds[index] = self._new_world()
            else:
                world["info"] = after
        return self.features(), np.asarray(rewards, dtype=np.float32), np.asarray(dones, dtype=bool), completed


__all__ = ["GoalOptionVector"]
