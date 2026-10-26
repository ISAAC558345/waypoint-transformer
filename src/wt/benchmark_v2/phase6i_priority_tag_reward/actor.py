"""Experiment actor. The formal ResidualActor class and its checkpoint loader stay unchanged."""
from __future__ import annotations

import torch
from torch import nn

from wt.lowlevel_v2.neighbor_encoder import NeighborSetEncoder
from wt.lowlevel_v2.observation import COMM_DIM, SELF_DIM, TASK_DIM
from wt.lowlevel_v2.residual_policy import ResidualActor

from .protocol import FEATURE_NAMES, OBSERVATION_SCHEMA

ORIGINAL_IN_DIM = int(SELF_DIM + TASK_DIM + COMM_DIM + 16 + 1)
EXTRA_DIM = len(FEATURE_NAMES)
EXPERIMENT_IN_DIM = ORIGINAL_IN_DIM + EXTRA_DIM


class PriorityTagActor(nn.Module):
    """Same trunk as ResidualActor, with three assigned-head features appended."""

    def __init__(self, hidden: int = 64, neighbor_out: int = 16, log_std_init: float = -1.0) -> None:
        super().__init__()
        self.observation_schema = OBSERVATION_SCHEMA
        self.neighbor = NeighborSetEncoder(out_dim=int(neighbor_out))
        self.in_dim = int(SELF_DIM + TASK_DIM + COMM_DIM + neighbor_out + 1 + EXTRA_DIM)
        if int(self.in_dim) != int(EXPERIMENT_IN_DIM):
            raise RuntimeError("experiment actor input dimension drifted")
        self.body = nn.Sequential(nn.Linear(self.in_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.mu = nn.Linear(hidden, 2)
        self.log_std = nn.Parameter(torch.full((2,), float(log_std_init)))
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)

    def _pack(self, self_feat, task_feat, comm_feat, neighbor_edges, d_nearest, priority):
        neigh = self.neighbor(neighbor_edges)
        if d_nearest is None:
            extra = torch.zeros(*self_feat.shape[:-1], 1, dtype=self_feat.dtype, device=self_feat.device)
        else:
            extra = d_nearest if d_nearest.shape[-1:] == (1,) else d_nearest.unsqueeze(-1)
        if priority.shape[-1:] != (EXTRA_DIM,):
            raise RuntimeError("priority feature width does not match the experiment schema")
        return torch.cat([self_feat, task_feat, comm_feat, neigh, extra, priority], dim=-1)

    def forward(self, self_feat, task_feat, comm_feat, neighbor_edges, d_nearest=None, priority=None):
        if priority is None:
            raise RuntimeError("experiment actor refuses a missing priority feature")
        hidden = self.body(self._pack(self_feat, task_feat, comm_feat, neighbor_edges, d_nearest, priority))
        mu = self.mu(hidden)
        return mu, self.log_std.expand_as(mu)


def formal_actor_state(payload: dict | object) -> dict:
    if isinstance(payload, dict) and "actor" in payload and isinstance(payload["actor"], dict):
        return payload["actor"]
    if isinstance(payload, dict):
        return payload
    raise RuntimeError("formal checkpoint does not contain an actor state")


def load_formal_actor(path, log_std_init: float) -> ResidualActor:
    """Default path. A shape mismatch raises instead of dropping tensors."""
    actor = ResidualActor(log_std_init=float(log_std_init))
    blob = torch.load(path, map_location="cpu", weights_only=False)
    actor.load_state_dict(formal_actor_state(blob), strict=True)
    return actor


def transplant_legacy_actor(source: ResidualActor, target: PriorityTagActor) -> None:
    """Copy matching tensors and zero only the new input columns. This is not a partial load."""
    source_state = source.state_dict()
    target_state = target.state_dict()
    if tuple(source.mu.weight.shape) != tuple(target.mu.weight.shape):
        raise RuntimeError("action head shape is not the formal residual head")
    copied = []
    for key, value in source_state.items():
        if key not in target_state:
            raise RuntimeError(f"experiment actor is missing {key}")
        destination = target_state[key]
        if key == "body.0.weight":
            if int(value.shape[0]) != int(destination.shape[0]) or int(value.shape[1]) != int(ORIGINAL_IN_DIM):
                raise RuntimeError("formal input layer does not match the frozen actor")
            if int(destination.shape[1]) != int(EXPERIMENT_IN_DIM):
                raise RuntimeError("experiment input layer width is wrong")
            destination.zero_()
            destination[:, : int(value.shape[1])] = value
            copied.append(key)
            continue
        if tuple(value.shape) != tuple(destination.shape):
            raise RuntimeError(f"refusing shape mismatch for {key}")
        destination.copy_(value)
        copied.append(key)
    missing = [key for key in source_state if key not in copied]
    if missing:
        raise RuntimeError(f"transplant left formal tensors unused: {missing}")
    target.load_state_dict(target_state, strict=True)


def refuse_formal_state_on_experiment_actor(target: PriorityTagActor, source: ResidualActor) -> None:
    try:
        target.load_state_dict(source.state_dict(), strict=True)
    except RuntimeError:
        return
    raise RuntimeError("experiment actor accepted a formal state without an explicit transplant")
