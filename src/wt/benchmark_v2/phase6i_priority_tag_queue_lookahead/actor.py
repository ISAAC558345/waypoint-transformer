"""Five-column actor. The critic stays on the three-column L3 state."""
from __future__ import annotations

import torch
from torch import nn

from wt.benchmark_v2.phase6i_priority_tag_reward.actor import ORIGINAL_IN_DIM, PriorityTagActor
from wt.lowlevel_v2.neighbor_encoder import NeighborSetEncoder
from wt.lowlevel_v2.observation import COMM_DIM, SELF_DIM, TASK_DIM

from .protocol import ACTOR_FEATURE_COUNT


class LookaheadActor(nn.Module):
    """Residual trunk with the original head features and two queue fractions."""

    def __init__(self, hidden: int = 64, neighbor_out: int = 16, log_std_init: float = -1.0) -> None:
        super().__init__()
        self.neighbor = NeighborSetEncoder(out_dim=int(neighbor_out))
        self.in_dim = int(SELF_DIM + TASK_DIM + COMM_DIM + neighbor_out + 1 + ACTOR_FEATURE_COUNT)
        if int(self.in_dim) != int(ORIGINAL_IN_DIM) + int(ACTOR_FEATURE_COUNT):
            raise RuntimeError("lookahead actor input dimension drifted")
        self.body = nn.Sequential(nn.Linear(self.in_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.mu = nn.Linear(hidden, 2)
        self.log_std = nn.Parameter(torch.full((2,), float(log_std_init)))
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)

    def forward(self, self_feat, task_feat, comm_feat, neighbor_edges, d_nearest=None, priority=None):
        if priority is None:
            raise RuntimeError("lookahead actor refuses a missing priority feature")
        if int(priority.shape[-1]) != int(ACTOR_FEATURE_COUNT):
            raise RuntimeError("lookahead actor received the wrong feature width")
        neigh = self.neighbor(neighbor_edges)
        if d_nearest is None:
            extra = torch.zeros(*self_feat.shape[:-1], 1, dtype=self_feat.dtype, device=self_feat.device)
        else:
            extra = d_nearest if d_nearest.shape[-1:] == (1,) else d_nearest.unsqueeze(-1)
        packed = torch.cat([self_feat, task_feat, comm_feat, neigh, extra, priority], dim=-1)
        hidden = self.body(packed)
        mu = self.mu(hidden)
        return mu, self.log_std.expand_as(mu)


def transplant_head_actor(source: PriorityTagActor, target: LookaheadActor) -> None:
    """Copy a three-column initialization and zero only the two new input columns."""
    source_state = source.state_dict()
    target_state = target.state_dict()
    copied = []
    for key, value in source_state.items():
        if key not in target_state:
            raise RuntimeError(f"lookahead actor is missing {key}")
        destination = target_state[key]
        if key == "body.0.weight":
            if int(destination.shape[1]) != int(value.shape[1]) + 2:
                raise RuntimeError("lookahead input layer does not add exactly two columns")
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
        raise RuntimeError(f"transplant left head-actor tensors unused: {missing}")
    target.load_state_dict(target_state, strict=True)


def same_module(left: nn.Module, right: nn.Module) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    if tuple(left_state) != tuple(right_state):
        return False
    return all(torch.equal(left_state[key], right_state[key]) for key in left_state)
