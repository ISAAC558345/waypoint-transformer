"""Urgency actor and critic. New input columns start at zero so the warm start matches L0."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import LOG_STD_INIT
from wt.lowlevel_v2.neighbor_encoder import NeighborSetEncoder
from wt.lowlevel_v2.observation import COMM_DIM, SELF_DIM
from wt.lowlevel_v2.residual_policy import CentralizedCritic, ResidualActor

from .protocol import ACTOR_IN_DIM, CRITIC_IN_DIM, TASK_DIM, URGENCY_TASK_EXTRA


class UrgencyActor(nn.Module):
    def __init__(self, hidden: int = 64, neighbor_out: int = 16, log_std_init: float = -1.0) -> None:
        super().__init__()
        self.neighbor = NeighborSetEncoder(out_dim=int(neighbor_out))
        task_dim = int(TASK_DIM) + int(URGENCY_TASK_EXTRA)
        in_dim = int(SELF_DIM + task_dim + COMM_DIM + neighbor_out + 1)
        if in_dim != int(ACTOR_IN_DIM):
            raise RuntimeError("urgency actor input width drifted")
        self.body = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.mu = nn.Linear(hidden, 2)
        self.log_std = nn.Parameter(torch.full((2,), float(log_std_init)))
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)

    def forward(self, self_feat, task_feat, comm_feat, neighbor_edges, d_nearest=None):
        neigh = self.neighbor(neighbor_edges)
        if d_nearest is None:
            extra = torch.zeros(*self_feat.shape[:-1], 1, dtype=self_feat.dtype, device=self_feat.device)
        else:
            extra = d_nearest if d_nearest.shape[-1:] == (1,) else d_nearest.unsqueeze(-1)
        hidden = self.body(torch.cat([self_feat, task_feat, comm_feat, neigh, extra], dim=-1))
        mu = self.mu(hidden)
        return mu, self.log_std.expand_as(mu)


def warm_start_from_frozen(actor: UrgencyActor, critic: CentralizedCritic, frozen_path: Path) -> list[dict]:
    """Copy matching tensors. New urgency columns are zero and are listed explicitly."""
    payload = torch.load(frozen_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "actor" not in payload or "critic" not in payload:
        raise RuntimeError("frozen checkpoint has no actor and critic states")
    report = []
    insert_at = int(SELF_DIM + TASK_DIM)
    report.extend(_copy_state(actor, payload["actor"], {"body.0.weight": ("insert", insert_at, int(URGENCY_TASK_EXTRA))}))
    report.extend(_copy_state(critic, payload["critic"], {"net.0.weight": ("prefix", 188)}))
    if int(actor.body[0].in_features) != int(ACTOR_IN_DIM) or int(critic.net[0].in_features) != int(CRITIC_IN_DIM):
        raise RuntimeError("warm start left an unexpected input width")
    return report


def save_checkpoint(path: Path, actor: UrgencyActor, critic: CentralizedCritic, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "actor_in_dim": int(actor.body[0].in_features),
            "critic_in_dim": int(critic.net[0].in_features),
            "task_dim": int(TASK_DIM) + int(URGENCY_TASK_EXTRA),
            **metadata,
        },
        path,
    )


def load_checkpoint(path: Path) -> tuple[UrgencyActor, CentralizedCritic, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("actor_in_dim", -1)) != int(ACTOR_IN_DIM) or int(payload.get("critic_in_dim", -1)) != int(CRITIC_IN_DIM):
        raise RuntimeError(
            f"checkpoint dimensions {payload.get('actor_in_dim')} / {payload.get('critic_in_dim')} "
            f"do not match {ACTOR_IN_DIM} / {CRITIC_IN_DIM}"
        )
    actor = UrgencyActor(log_std_init=float(LOG_STD_INIT))
    critic = CentralizedCritic(int(CRITIC_IN_DIM))
    actor.load_state_dict(payload["actor"], strict=True)
    critic.load_state_dict(payload["critic"], strict=True)
    actor.eval()
    critic.eval()
    return actor, critic, payload


def assert_old_checkpoint_still_loads(frozen_path: Path) -> ResidualActor:
    actor = ResidualActor(log_std_init=float(LOG_STD_INIT))
    payload = torch.load(frozen_path, map_location="cpu", weights_only=False)
    state = payload["actor"] if isinstance(payload, dict) and "actor" in payload else payload
    actor.load_state_dict(state, strict=True)
    if int(actor.body[0].in_features) != 34:
        raise RuntimeError("formal actor width changed")
    return actor


def _copy_state(module: nn.Module, old_state: dict, prefixes: dict[str, tuple[str, int]]) -> list[dict]:
    report = []
    destination = module.state_dict()
    for key, value in destination.items():
        if key not in old_state:
            report.append({"name": key, "action": "scratch", "shape": list(value.shape)})
            continue
        old = old_state[key]
        if tuple(old.shape) == tuple(value.shape):
            value.copy_(old)
            report.append({"name": key, "action": "copied", "shape": list(value.shape)})
            continue
        rule = prefixes.get(key)
        if rule is None or int(value.shape[0]) != int(old.shape[0]):
            raise RuntimeError(f"no declared initialization for {key}: {tuple(old.shape)} -> {tuple(value.shape)}")
        value.zero_()
        if rule[0] == "prefix":
            if int(old.shape[1]) != int(rule[1]):
                raise RuntimeError(f"prefix width for {key} is not {rule[1]}")
            value[:, : int(rule[1])].copy_(old)
            action = "copied_prefix_new_columns_zero"
            copied = int(rule[1])
        elif rule[0] == "insert":
            index = int(rule[1])
            width = int(rule[2])
            if int(old.shape[1]) + width != int(value.shape[1]):
                raise RuntimeError(f"insert does not account for {key}")
            value[:, :index].copy_(old[:, :index])
            value[:, index + width :].copy_(old[:, index:])
            action = "copied_with_inserted_zero_columns"
            copied = int(old.shape[1])
        else:
            raise RuntimeError(f"unknown initialization rule for {key}")
        report.append(
            {
                "name": key,
                "action": action,
                "copied_columns": copied,
                "new_columns": int(value.shape[1] - old.shape[1]),
                "shape": list(value.shape),
            }
        )
    module.load_state_dict(destination)
    return report
