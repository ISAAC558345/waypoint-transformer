"""Clip grouping. The total norm bound stays G for both rules."""
from __future__ import annotations

import torch
from torch.nn.utils import clip_grad_norm_

from .protocol import JOINT_BOUND_ABS, joint_limit, separate_limit


def _finite_grads(parameters) -> None:
    for parameter in parameters:
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError("NONFINITE")


def _norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        flat = parameter.grad.detach().float().reshape(-1)
        total += float(torch.dot(flat, flat).cpu())
    return float(total ** 0.5)


def _scale(pre_norm: float, limit: float) -> float:
    return float(min(1.0, float(limit) / (float(pre_norm) + 1.0e-6)))


def assert_disjoint(actor, critic) -> None:
    actor_ids = {id(parameter) for parameter in actor.parameters()}
    critic_ids = {id(parameter) for parameter in critic.parameters()}
    if actor_ids & critic_ids:
        raise RuntimeError("actor and critic parameters overlap")
    if not actor_ids or not critic_ids:
        raise RuntimeError("actor or critic has no parameters")


def apply_clip(actor, critic, rule: str) -> dict:
    """Clip in place. PILOT_S uses G/sqrt(2) on each group so the union stays within G."""
    if rule not in {"PILOT_J", "PILOT_S"}:
        raise RuntimeError(f"unknown clip rule {rule}")
    assert_disjoint(actor, critic)
    actor_params = list(actor.parameters())
    critic_params = list(critic.parameters())
    _finite_grads(actor_params + critic_params)
    actor_pre = _norm(actor_params)
    critic_pre = _norm(critic_params)
    joint_pre = _norm(actor_params + critic_params)
    limit = joint_limit()
    group_limit = separate_limit()
    if rule == "PILOT_J":
        returned = float(clip_grad_norm_(actor_params + critic_params, float(limit)).detach().cpu())
        if not torch.isfinite(torch.tensor(returned)):
            raise RuntimeError("NONFINITE")
        actor_scale = _scale(returned, limit)
        critic_scale = actor_scale
        actor_limit = float(limit)
        critic_limit = float(limit)
    else:
        returned_actor = float(clip_grad_norm_(actor_params, float(group_limit)).detach().cpu())
        returned_critic = float(clip_grad_norm_(critic_params, float(group_limit)).detach().cpu())
        if not torch.isfinite(torch.tensor(returned_actor)) or not torch.isfinite(torch.tensor(returned_critic)):
            raise RuntimeError("NONFINITE")
        actor_scale = _scale(returned_actor, group_limit)
        critic_scale = _scale(returned_critic, group_limit)
        actor_limit = float(group_limit)
        critic_limit = float(group_limit)
    actor_post = _norm(actor_params)
    critic_post = _norm(critic_params)
    joint_post = _norm(actor_params + critic_params)
    if joint_post > float(limit) + float(JOINT_BOUND_ABS):
        raise RuntimeError("clipped joint norm exceeds the registered total bound")
    return {
        "clip_rule": rule,
        "actor_pre_norm": actor_pre,
        "critic_pre_norm": critic_pre,
        "joint_pre_norm": joint_pre,
        "actor_limit": actor_limit,
        "critic_limit": critic_limit,
        "actor_scale": actor_scale,
        "critic_scale": critic_scale,
        "actor_post_norm": actor_post,
        "critic_post_norm": critic_post,
        "joint_post_norm": joint_post,
        "joint_bound": float(limit),
    }
