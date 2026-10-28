"""Rollout collection for the five-column actor. The critic state stays three columns wide."""
from __future__ import annotations

import numpy as np
import torch

from wt.benchmark_v2.phase6i_priority_tag_reward.rollout import critic_dim, critic_vector
from wt.lowlevel_v2.contracts import env_target

from .protocol import ACTOR_FEATURE_COUNT, HEAD_FEATURE_COUNT
from .worker import QueueLookaheadWorker


def collect(workers: list[QueueLookaheadWorker], horizon: int) -> tuple[dict[str, np.ndarray], dict[str, float], list[dict]]:
    n_env = len(workers)
    n_uav = int(workers[0].env.config.num_uavs)
    n_buoy = int(workers[0].env.config.num_buoys)
    keys = {
        "self": (horizon, n_env, n_uav, 5),
        "task": (horizon, n_env, n_uav, 8),
        "communication": (horizon, n_env, n_uav, 4),
        "neighbor_set": (horizon, n_env, n_uav, 5, 11),
        "d_nearest": (horizon, n_env, n_uav),
        "priority": (horizon, n_env, n_uav, ACTOR_FEATURE_COUNT),
        "global_state": (horizon, n_env, critic_dim(n_uav, n_buoy)),
        "pre_tanh": (horizon, n_env, n_uav, 2),
        "log_prob_uav": (horizon, n_env, n_uav),
        "policy_active_mask": (horizon, n_env, n_uav),
        "values": (horizon, n_env),
        "rewards": (horizon, n_env),
        "dones": (horizon, n_env),
    }
    storage = {key: np.zeros(shape, dtype=np.float32) for key, shape in keys.items()}
    served, tagged, bonus_age, bonus_tag, motion, projected, safety, failures = [], [], [], [], [], [], [], []
    uav_steps = 0
    valid_steps = 0
    backlog_steps = 0
    ordinary_backlog = 0
    nonzero_input = 0
    distinguishable = 0
    positive_tag = 0
    samples = []
    for t in range(horizon):
        for index, worker in enumerate(workers):
            record = worker.step()
            priority = np.asarray(record["priority"], dtype=np.float64)
            association = np.asarray(record["association"], dtype=np.int64)
            storage["self"][t, index] = record["obs"]["self"]
            storage["task"][t, index] = record["obs"]["task"]
            storage["communication"][t, index] = record["obs"]["communication"]
            storage["neighbor_set"][t, index] = record["obs"]["neighbor_set"]
            storage["d_nearest"][t, index] = record["obs"]["d_nearest"]
            storage["priority"][t, index] = priority
            storage["global_state"][t, index] = record["global_state"]
            storage["pre_tanh"][t, index] = record["pre_tanh"]
            storage["log_prob_uav"][t, index] = record["log_prob_uav"]
            storage["policy_active_mask"][t, index] = record["policy_active_mask"]
            storage["values"][t, index] = record["value"]
            storage["rewards"][t, index] = record["reward"]
            storage["dones"][t, index] = float(record["done"])
            served.append(float(record["served_bits"]))
            tagged.append(float(record["served_tagged_bits"]))
            bonus_age.append(float(record["bonus_age"]))
            bonus_tag.append(float(record["bonus_tag"]))
            motion.append(float(record["motion_penalty"]))
            projected.append(int(record["projected"]))
            safety.append(int(record["safety_violation"]))
            failures.append(int(record["constraint_failure"]))
            true_look = np.asarray(record["lookahead_true"], dtype=np.float64)
            if float(record["bonus_tag"]) > 0.0:
                positive_tag += 1
                without = np.float32(float(record["base_reward"]) + float(record["bonus_age"]))
                with_tag = np.float32(float(record["reward"]))
                if float(with_tag) != float(without):
                    distinguishable += 1
            for uav, token in enumerate(association):
                uav_steps += 1
                assigned = int(env_target(int(token), n_buoy)) >= 0
                backlog = float(true_look[uav, 0]) > 0.0
                head_tag = float(priority[uav, 1]) > 0.0
                valid_head = float(priority[uav, 2]) > 0.0
                if assigned:
                    valid_steps += 1
                if assigned and backlog:
                    backlog_steps += 1
                if assigned and valid_head and not head_tag and backlog:
                    ordinary_backlog += 1
                if float(np.max(np.abs(priority[uav, HEAD_FEATURE_COUNT:]))) > 0.0:
                    nonzero_input += 1
            if index == 0:
                samples.append(
                    {
                        "step": int(t),
                        "reward": float(record["reward"]),
                        "reward_float32": float(np.float32(record["reward"])),
                        "base_reward": float(record["base_reward"]),
                        "bonus_age": float(record["bonus_age"]),
                        "bonus_tag": float(record["bonus_tag"]),
                        "served_bits": float(record["served_bits"]),
                        "served_tagged_bits": float(record["served_tagged_bits"]),
                        "motion_penalty": float(record["motion_penalty"]),
                        "lookahead_input": np.asarray(priority[:, HEAD_FEATURE_COUNT:], dtype=np.float64).reshape(-1).tolist(),
                        "lookahead_true": np.asarray(true_look, dtype=np.float64).reshape(-1).tolist(),
                    }
                )
    last = np.zeros(n_env, dtype=np.float32)
    for index, worker in enumerate(workers):
        if storage["dones"][-1, index] > 0.5 or worker.option is None or worker.critic is None:
            continue
        priority = worker._priority()[:, : int(HEAD_FEATURE_COUNT)]
        with torch.no_grad():
            vector = critic_vector(worker.env, worker.option, worker.last_movement, priority)
            last[index] = float(worker.critic(torch.as_tensor(vector, dtype=torch.float32)).cpu())
    storage["last_values"] = last
    if not np.isfinite(storage["rewards"]).all():
        raise RuntimeError("NONFINITE")
    metrics = {
        "mean_served_bits": float(np.mean(served)),
        "mean_served_tagged_bits": float(np.mean(tagged)),
        "mean_bonus": float(np.mean(np.asarray(bonus_age) + np.asarray(bonus_tag))),
        "mean_bonus_age": float(np.mean(bonus_age)),
        "mean_bonus_tag": float(np.mean(bonus_tag)),
        "mean_motion_penalty": float(np.mean(motion)),
        "projection_rate": float(np.mean(projected)),
        "safety_violation_rate": float(np.mean(safety)),
        "constraint_failure_rate": float(np.mean(failures)),
        "uav_steps": int(uav_steps),
        "valid_association_uav_steps": int(valid_steps),
        "tagged_backlog_valid_uav_steps": int(backlog_steps),
        "ordinary_head_tagged_backlog_uav_steps": int(ordinary_backlog),
        "lookahead_nonzero_input_uav_steps": int(nonzero_input),
        "positive_tag_bonus_steps": int(positive_tag),
        "float32_distinguishable_tag_steps": int(distinguishable),
    }
    return storage, metrics, samples
