"""Held high-level option vector. Does not update the frozen low-level weights."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from wt.htw_phase61.features import MAX_UAVS, attach_option, encode_option, in_range_completion, service_success, stack_feature_dicts
from wt.htw_phase62.rewards import mean_target_distance_slots
from wt.htw_phase63.high_ppo import assignment_to_indices, indices_to_assignment
from wt.l2a.env import L2AEnv, project_exclusive
from wt.l2a.policies import oracle_option_assignment
from wt.round30.ppo import PPOAgent

from .features import feature_dict_high

FeatureFn = Callable[..., dict[str, np.ndarray]]


def high_level_reward(mode: str, delta_blocks: float, service: float, delay_slots: float, episode_blocks: float, done: bool) -> float:
    if mode == "episode":
        return float(episode_blocks) if done else 0.0
    if mode == "delta":
        return float(delta_blocks)
    if mode == "delta_service":
        return float(delta_blocks) + 0.5 * float(service)
    if mode == "delta_service_delay":
        return float(delta_blocks) + 0.5 * float(service) - 0.05 * float(delay_slots)
    raise ValueError(f"unknown high-level reward mode {mode}")


class HeldHighVector:
    def __init__(
        self,
        seeds: tuple[int, ...],
        count: int,
        seed: int,
        low_agent: PPOAgent,
        *,
        hold: int,
        reward_mode: str,
        enhanced: bool,
    ) -> None:
        self.seeds = seeds
        self.rng = np.random.default_rng(seed)
        self.environment_steps = 0
        self.low_agent = low_agent
        self.hold = int(hold)
        self.reward_mode = str(reward_mode)
        self.enhanced = bool(enhanced)
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
            "accuracies": [],
            "last_arrival": np.zeros(env.config.num_buoys, dtype=np.float32),
        }

    def _row_features(self, world: dict[str, Any]) -> dict[str, np.ndarray]:
        return feature_dict_high(world["info"], world["env"], last_arrival=world["last_arrival"], enhanced=self.enhanced)

    def features(self) -> dict[str, np.ndarray]:
        rows = [self._row_features(world) for world in self.worlds]
        batch = stack_feature_dicts(rows)
        batch["_uav_positions"] = np.stack([np.asarray(world["env"].uav_positions, dtype=np.float64) for world in self.worlds], axis=0)
        batch["_buoy_positions"] = np.stack([np.asarray(world["env"].buoy_positions, dtype=np.float64) for world in self.worlds], axis=0)
        return batch

    def _low_move(self, env: L2AEnv, info: dict[str, Any], assignment: np.ndarray) -> np.ndarray:
        from wt.htw_phase61.features import feature_dict_l2a

        features = attach_option(feature_dict_l2a(info, env), encode_option(assignment))
        stacked = stack_feature_dicts([features])
        return self.low_agent.act(stacked, deterministic=True)["action"][0]

    def step(self, z_indices: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, float]]]:
        if z_indices.shape != (len(self.worlds), MAX_UAVS):
            raise ValueError("held high-level z batch has invalid shape")
        rewards, dones, completed = [], [], []
        executed = np.zeros_like(z_indices)
        for index, (world, z_row) in enumerate(zip(self.worlds, z_indices)):
            env: L2AEnv = world["env"]
            oracle = oracle_option_assignment(env)
            assignment = project_exclusive(indices_to_assignment(z_row), env.uav_positions, env.buoy_positions)
            executed[index] = assignment_to_indices(assignment)
            accuracy = float(np.mean(np.asarray(assignment, dtype=np.int64) == np.asarray(oracle, dtype=np.int64)))
            world["accuracies"].append(accuracy)
            delta = 0.0
            services: list[float] = []
            delays: list[float] = []
            done = False
            after = world["info"]
            for _ in range(self.hold):
                before_payload = float(env.payload_bits)
                move = self._low_move(env, world["info"], assignment)
                after = env.step(move, assignment)
                delta += (float(after["cumulative_payload_bits"]) - before_payload) / float(env.config.block_payload_bits)
                services.append(service_success(after, assignment))
                delays.append(mean_target_distance_slots(env, assignment))
                world["last_arrival"] = np.asarray(after["arrivals_blocks"], dtype=np.float32)
                world["return"] += (float(after["cumulative_payload_bits"]) - before_payload) / float(env.config.block_payload_bits)
                self.environment_steps += 1
                world["info"] = after
                if bool(after["terminated"]):
                    done = True
                    break
            service = float(np.mean(services)) if services else 0.0
            delay = float(np.mean(delays)) if delays else 0.0
            reward = high_level_reward(self.reward_mode, delta, service, delay, float(world["return"]), done)
            rewards.append(float(reward))
            dones.append(done)
            if done:
                completed.append(
                    {
                        "payload_bits": float(after["cumulative_payload_bits"]),
                        "return": float(world["return"]),
                        "option_selection_accuracy": float(np.mean(world["accuracies"])),
                    }
                )
                self.worlds[index] = self._new_world()
        next_features = self.features()
        next_features["_executed_z"] = executed
        return next_features, np.asarray(rewards, dtype=np.float32), np.asarray(dones, dtype=bool), completed


def collect_held(agent, vector: HeldHighVector):
    feature_rows = {key: [] for key in ("uav", "buoy", "uav_mask", "buoy_mask", "global")}
    records = {key: [] for key in ("z", "log_prob", "values", "rewards", "dones")}
    events: list[dict[str, float]] = []
    features = vector.features()
    for _ in range(agent.config.rollout_length):
        decision = agent.act(features)
        for key in feature_rows:
            feature_rows[key].append(features[key])
        next_features, rewards, dones, completed = vector.step(decision["z"])
        records["z"].append(next_features.get("_executed_z", decision["z"]))
        records["log_prob"].append(decision["log_prob"])
        records["values"].append(decision["value"])
        records["rewards"].append(rewards)
        records["dones"].append(dones)
        events.extend(completed)
        features = next_features
    storage = {key: np.stack(value, axis=0) for key, value in feature_rows.items()}
    storage.update(
        {
            "z": np.stack(records["z"], axis=0),
            "log_prob": np.stack(records["log_prob"], axis=0),
            "values": np.stack(records["values"], axis=0),
            "rewards": np.stack(records["rewards"], axis=0),
            "dones": np.stack(records["dones"], axis=0),
            "last_values": agent.act(features, deterministic=True)["value"],
        }
    )
    return storage, events, {"mean_raw_reward": float(storage["rewards"].mean())}


__all__ = ["HeldHighVector", "collect_held", "high_level_reward"]
