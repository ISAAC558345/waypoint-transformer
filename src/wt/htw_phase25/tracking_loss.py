"""PPO wrapper that adds a tracking term to the loss. Does not edit round30/ppo.py."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torch

from wt.round30.ppo import PPOAgent, compute_gae, tanh_log_prob
from wt.round30.ppo import _tensor

from wt.htw_phase2.vector import HierarchicalTrackingVector


def collect_tracked(agent: PPOAgent, vector: HierarchicalTrackingVector):
    feature_rows = {key: [] for key in ("uav", "buoy", "uav_mask", "buoy_mask", "global")}
    records = {key: [] for key in ("pre_tanh", "log_prob", "values", "rewards", "dones", "track_dist", "track_current", "track_waypoint", "track_mask", "step_scale")}
    events = []
    action_stds = []
    features = vector.features()
    for _ in range(agent.config.rollout_length):
        decision = agent.act(features)
        for key in feature_rows:
            feature_rows[key].append(features[key])
        next_features, rewards, dones, completed = vector.step(decision["action"])
        records["pre_tanh"].append(decision["pre_tanh"])
        records["log_prob"].append(decision["log_prob"])
        records["values"].append(decision["value"])
        records["rewards"].append(rewards)
        records["dones"].append(dones)
        records["track_dist"].append(np.asarray(vector.last_track_dist, dtype=np.float32))
        records["track_current"].append(np.asarray(vector.last_track_current, dtype=np.float32))
        records["track_waypoint"].append(np.asarray(vector.last_track_waypoint, dtype=np.float32))
        records["track_mask"].append(np.asarray(vector.last_track_mask, dtype=np.float32))
        records["step_scale"].append(np.asarray(vector.last_step_scale, dtype=np.float32))
        action_stds.extend(decision["action_std"].tolist())
        events.extend(completed)
        features = next_features
    storage = {key: np.stack(value, axis=0) for key, value in feature_rows.items()}
    storage.update(
        {
            "pre_tanh": np.stack(records["pre_tanh"], axis=0),
            "log_prob": np.stack(records["log_prob"], axis=0),
            "values": np.stack(records["values"], axis=0),
            "rewards": np.stack(records["rewards"], axis=0),
            "dones": np.stack(records["dones"], axis=0),
            "track_dist": np.stack(records["track_dist"], axis=0),
            "track_current": np.stack(records["track_current"], axis=0),
            "track_waypoint": np.stack(records["track_waypoint"], axis=0),
            "track_mask": np.stack(records["track_mask"], axis=0),
            "step_scale": np.stack(records["step_scale"], axis=0),
            "last_values": agent.act(features, deterministic=True)["value"],
        }
    )
    return storage, events, {"mean_action_std": float(np.mean(action_stds)), "mean_raw_reward": float(storage["rewards"].mean()), "mean_track_dist": float(storage["track_dist"].mean())}


class TrackingLossAgent(PPOAgent):
    """Same GAE/clip/Adam as PPOAgent; loss is L_PPO + beta * mean||p-W||."""

    def __init__(self, model, config, track_beta: float) -> None:
        super().__init__(model, config)
        self.track_beta = float(track_beta)

    def update(self, storage: Mapping[str, np.ndarray]) -> dict[str, float]:
        extra = {"track_dist", "track_current", "track_waypoint", "track_mask", "step_scale"}
        if self.track_beta <= 0.0:
            return super().update({key: value for key, value in storage.items() if key not in extra})
        steps, envs = storage["rewards"].shape
        advantages, returns = compute_gae(storage["rewards"], storage["values"], storage["dones"], storage["last_values"], self.config.gamma, self.config.gae_lambda)
        skip = {"last_values"}
        flat = {key: value.reshape((steps * envs,) + value.shape[2:]) if value.ndim >= 2 else value.reshape(steps * envs) for key, value in storage.items() if key not in skip}
        flat["advantages"], flat["returns"] = advantages.reshape(-1), returns.reshape(-1)
        flat["advantages"] = (flat["advantages"] - flat["advantages"].mean()) / max(float(flat["advantages"].std()), 1.0e-6)
        count, minibatch = steps * envs, steps * envs // self.config.num_minibatches
        records = []
        self.model.train()
        for _epoch in range(self.config.update_epochs):
            order = np.random.permutation(count)
            for start in range(0, count, minibatch):
                index = order[start:start + minibatch]
                features = _tensor({key: flat[key][index] for key in ("uav", "buoy", "uav_mask", "buoy_mask", "global")}, self.config.device)
                old_log_prob = torch.as_tensor(flat["log_prob"][index], dtype=torch.float32, device=self.config.device)
                action_z = torch.as_tensor(flat["pre_tanh"][index], dtype=torch.float32, device=self.config.device)
                advantage = torch.as_tensor(flat["advantages"][index], dtype=torch.float32, device=self.config.device)
                target = torch.as_tensor(flat["returns"][index], dtype=torch.float32, device=self.config.device)
                current = torch.as_tensor(flat["track_current"][index], dtype=torch.float32, device=self.config.device)
                waypoint = torch.as_tensor(flat["track_waypoint"][index], dtype=torch.float32, device=self.config.device)
                mask = torch.as_tensor(flat["track_mask"][index], dtype=torch.float32, device=self.config.device)
                scale = torch.as_tensor(flat["step_scale"][index], dtype=torch.float32, device=self.config.device)
                mu, log_std, value = self.model(**features)
                log_prob, entropy = tanh_log_prob(mu, log_std, action_z)
                ratio = (log_prob - old_log_prob).exp()
                clipped = ratio.clamp(1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps)
                policy_loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
                value_loss = 0.5 * (target - value).square().mean()
                action = torch.tanh(mu)
                predicted_xy = current + action * scale[:, None, :]
                delta = predicted_xy - waypoint
                dist = torch.linalg.norm(delta, dim=-1)
                track_loss = (dist * mask).sum() / mask.sum().clamp_min(1.0)
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy.mean() + self.track_beta * track_loss
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm).detach().cpu())
                self.optimizer.step()
                records.append({
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "value_loss": float(value_loss.detach().cpu()),
                    "entropy": float(entropy.mean().detach().cpu()),
                    "approx_kl": float((old_log_prob - log_prob).mean().detach().cpu()),
                    "clip_fraction": float(((ratio - 1.0).abs() > self.config.clip_eps).float().mean().detach().cpu()),
                    "gradient_norm_preclip": grad_norm,
                    "action_std": float(log_std.exp().mean().detach().cpu()),
                    "track_loss": float(track_loss.detach().cpu()),
                })
        predicted, target = storage["values"].reshape(-1), returns.reshape(-1)
        explained = 1.0 - float(np.var(target - predicted) / max(np.var(target), 1.0e-8))
        if not np.isfinite(explained) or not all(np.isfinite(value) for row in records for value in row.values()):
            raise RuntimeError("PPO update produced NaN/Inf")
        return {key: float(np.mean([row[key] for row in records])) for key in records[0]} | {"explained_variance": explained, "advantage_mean": float(advantages.mean()), "advantage_std": float(advantages.std())}


__all__ = ["TrackingLossAgent", "collect_tracked"]
