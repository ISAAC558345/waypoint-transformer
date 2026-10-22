"""Deterministic frozen MAPPO-V2. The checkpoint is loaded and never updated."""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.evaluate import _actor_move
from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import LOG_STD_INIT
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import CHECKPOINT_DIR, PRIMARY_SEED, primary_checkpoint
from wt.lowlevel_v2.residual_policy import ResidualActor


def limit_threads() -> None:
    """Keep one process on one thread so scenario workers do not oversubscribe the CPU."""
    torch.set_num_threads(1)


def checkpoint_sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def actor_is_stateless(actor: ResidualActor) -> bool:
    names = {type(module).__name__ for module in actor.modules()}
    return names.isdisjoint({"LSTM", "GRU", "RNN", "LSTMCell", "GRUCell"})


def load_actor(seed: int) -> ResidualActor:
    actor = ResidualActor(log_std_init=float(LOG_STD_INIT))
    path = CHECKPOINT_DIR / f"phase6g_seed{int(seed)}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["actor"] if isinstance(payload, dict) and "actor" in payload else payload
    actor.load_state_dict(state, strict=True)
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    if not actor_is_stateless(actor):
        raise RuntimeError("MAPPO actor has recurrent state and no restore rule")
    return actor


def load_primary() -> ResidualActor:
    if int(PRIMARY_SEED) != 0:
        raise RuntimeError("primary seed rule was changed")
    return load_actor(PRIMARY_SEED)


def parameter_token(actor: ResidualActor) -> tuple[tuple[str, bytes], ...]:
    return tuple((name, tensor.detach().cpu().numpy().tobytes()) for name, tensor in actor.state_dict().items())


def deterministic_move(env, actor: ResidualActor, option, last_movement: np.ndarray) -> np.ndarray:
    if actor.training:
        raise RuntimeError("MAPPO actor left eval mode")
    movement, _projected, _norm = _actor_move(env, actor, option, last_movement)
    return movement
