"""Discarded one-step copies. They never re-enter the real pilot."""
from __future__ import annotations

import copy
from pathlib import Path

import torch

from wt.benchmark_v2.phase6i_priority_tag_credit_audit.probe import capture_rng, restore_rng, same_rng
from wt.benchmark_v2.phase6i_priority_tag_reward.smoke import file_sha256
from wt.lowlevel_v2.phase1b.mappo import tanh_log_prob_uav

from .clip import apply_clip
from .protocol import REPEAT_ABS


def _pairs(actor, critic):
    return list(actor.parameters()) + list(critic.parameters())


def clone_pilot(actor, critic, optimizer):
    """A new module pair and a new Adam. No tensor storage is shared with the source."""
    actor_copy = copy.deepcopy(actor)
    critic_copy = copy.deepcopy(critic)
    cloned = torch.optim.Adam(
        _pairs(actor_copy, critic_copy),
        lr=float(optimizer.param_groups[0]["lr"]),
    )
    cloned.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for source, dest in zip(_pairs(actor, critic), _pairs(actor_copy, critic_copy)):
        if source.data_ptr() == dest.data_ptr():
            raise RuntimeError("shadow parameter shares storage with the source")
        if source.grad is None:
            dest.grad = None
            continue
        dest.grad = source.grad.detach().clone()
        if source.grad.data_ptr() == dest.grad.data_ptr():
            raise RuntimeError("shadow gradient shares storage with the source")
    for source, dest in zip(_pairs(actor, critic), _pairs(actor_copy, critic_copy)):
        source_state = optimizer.state.get(source, {})
        dest_state = cloned.state.get(dest, {})
        for key in ("exp_avg", "exp_avg_sq"):
            if key not in source_state:
                continue
            if key not in dest_state:
                raise RuntimeError("shadow Adam state did not copy a moment")
            if source_state[key].data_ptr() == dest_state[key].data_ptr():
                raise RuntimeError("shadow Adam moment shares storage with the source")
            if not torch.equal(source_state[key].detach().cpu(), dest_state[key].detach().cpu()):
                raise RuntimeError("shadow Adam moment value was not copied")
        if "step" in source_state and int(source_state["step"]) != int(dest_state["step"]):
            raise RuntimeError("shadow Adam step counter was not copied")
    return actor_copy, critic_copy, cloned


def _snapshot(module) -> list[tuple[int, torch.Tensor | None, torch.Tensor | None]]:
    rows = []
    for parameter in module.parameters():
        grad = None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        rows.append((parameter.data_ptr(), parameter.detach().cpu().clone(), grad))
    return rows


def _adam_snapshot(optimizer) -> list[dict]:
    rows = []
    for parameter in optimizer.param_groups[0]["params"]:
        state = optimizer.state.get(parameter, {})
        rows.append(
            {
                "ptr": parameter.data_ptr(),
                "exp_avg": None if "exp_avg" not in state else state["exp_avg"].detach().cpu().clone(),
                "exp_avg_sq": None if "exp_avg_sq" not in state else state["exp_avg_sq"].detach().cpu().clone(),
                "step": None if "step" not in state else int(state["step"]),
            }
        )
    return rows


def _unchanged(actor, critic, optimizer, actor_saved, critic_saved, adam_saved) -> None:
    for module, saved in ((actor, actor_saved), (critic, critic_saved)):
        current = _snapshot(module)
        if len(current) != len(saved):
            raise RuntimeError("shadow changed the source parameter list")
        for (ptr, value, grad), (old_ptr, old_value, old_grad) in zip(current, saved):
            if ptr != old_ptr or not torch.equal(value, old_value):
                raise RuntimeError("shadow changed a source parameter")
            if grad is None or old_grad is None:
                if grad is not None or old_grad is not None:
                    raise RuntimeError("shadow changed a source gradient")
            elif not torch.equal(grad, old_grad):
                raise RuntimeError("shadow changed a source gradient")
    current_adam = _adam_snapshot(optimizer)
    if len(current_adam) != len(adam_saved):
        raise RuntimeError("shadow changed the source Adam parameter mapping")
    for item, old in zip(current_adam, adam_saved):
        if item["ptr"] != old["ptr"] or item["step"] != old["step"]:
            raise RuntimeError("shadow changed the source Adam state")
        for key in ("exp_avg", "exp_avg_sq"):
            if item[key] is None or old[key] is None:
                if item[key] is not None or old[key] is not None:
                    raise RuntimeError("shadow changed a source Adam moment")
            elif not torch.equal(item[key], old[key]):
                raise RuntimeError("shadow changed a source Adam moment")


def _flat_delta(source, module) -> torch.Tensor:
    parts = []
    for before, parameter in zip(source.parameters(), module.parameters()):
        parts.append((parameter.detach().cpu().float() - before.detach().cpu().float()).reshape(-1))
    return torch.cat(parts)


def _norm(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(torch.linalg.vector_norm(value).cpu())


def _max_abs_modules(left, right) -> float:
    gap = 0.0
    for first, second in zip(left.parameters(), right.parameters()):
        gap = max(gap, float(torch.max(torch.abs(first.detach().cpu() - second.detach().cpu())).cpu()))
    return float(gap)


def _layer_gaps(source, left, right) -> list[dict]:
    left_named = dict(left.named_parameters())
    right_named = dict(right.named_parameters())
    rows = []
    for name, parameter in source.named_parameters():
        base = parameter.detach().cpu().float()
        gap = left_named[name].detach().cpu().float() - right_named[name].detach().cpu().float()
        rows.append({"name": name, "l2": _norm(gap), "source_abs_max": float(torch.max(torch.abs(base)).cpu())})
    return rows


def _new_column(module, source_weight: torch.Tensor) -> torch.Tensor:
    weight = dict(module.named_parameters())["body.0.weight"].detach().cpu().float()
    return weight[:, -2:] - source_weight.float()[:, -2:]


def _policy_gap(left, right, batch: dict) -> tuple[float, float]:
    with torch.no_grad():
        mu_left, log_left = left(batch["self"], batch["task"], batch["communication"], batch["neighbor_set"], batch["d_nearest"], batch["priority"])
        mu_right, log_right = right(batch["self"], batch["task"], batch["communication"], batch["neighbor_set"], batch["d_nearest"], batch["priority"])
        prob_left, _entropy_left = tanh_log_prob_uav(mu_left, log_left, batch["pre_tanh"])
        prob_right, _entropy_right = tanh_log_prob_uav(mu_right, log_right, batch["pre_tanh"])
    return _norm(mu_left.detach().cpu().float() - mu_right.detach().cpu().float()), float(torch.mean(torch.abs(prob_left.detach().cpu() - prob_right.detach().cpu())).cpu())


def _save_source(path: Path, actor, critic, optimizer) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    grads = {}
    for name, parameter in list(actor.named_parameters()) + list(critic.named_parameters()):
        grads[name] = None if parameter.grad is None else parameter.grad.detach().cpu().clone()
    torch.save(
        {
            "actor": {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()},
            "critic": {key: value.detach().cpu().clone() for key, value in critic.state_dict().items()},
            "adam": copy.deepcopy(optimizer.state_dict()),
            "grads": grads,
        },
        path,
    )
    return file_sha256(path)


def run_shadow(context: dict, destination: Path) -> dict:
    """Apply J, S, and J again to copies. Restore RNG and leave the source untouched."""
    actor = context["actor"]
    critic = context["critic"]
    optimizer = context["optimizer"]
    saved_rng = capture_rng()
    saved_rng["python"] = copy.deepcopy(saved_rng["python"])
    actor_saved = _snapshot(actor)
    critic_saved = _snapshot(critic)
    adam_saved = _adam_snapshot(optimizer)
    try:
        digest = _save_source(destination, actor, critic, optimizer)
        batch = {key: value.detach().clone() for key, value in context["batch"].items()}
        source_weight = dict(actor.named_parameters())["body.0.weight"].detach().cpu().clone()
        copies = [clone_pilot(actor, critic, optimizer) for _rule in range(3)]
        rules = ("PILOT_J", "PILOT_S", "PILOT_J")
        for (actor_copy, critic_copy, cloned), rule in zip(copies, rules):
            apply_clip(actor_copy, critic_copy, rule)
            cloned.step()
        repeat_gap = _max_abs_modules(copies[0][0], copies[2][0])
        if repeat_gap > float(REPEAT_ABS):
            raise RuntimeError("shadow J repeat did not reproduce the actor update")
        delta_j = _flat_delta(actor, copies[0][0])
        delta_s = _flat_delta(actor, copies[1][0])
        j_norm = _norm(delta_j)
        s_norm = _norm(delta_s)
        absolute = _norm(delta_s - delta_j)
        if j_norm == 0.0 or s_norm == 0.0:
            cosine = None
            ratio = None
        else:
            cosine = float(torch.dot(delta_j, delta_s).cpu() / (j_norm * s_norm))
            ratio = float(absolute / j_norm)
        column_j = _new_column(copies[0][0], source_weight)
        column_s = _new_column(copies[1][0], source_weight)
        mu_gap, log_gap = _policy_gap(copies[0][0], copies[1][0], batch)
        return {
            "ppo_update": int(context["ppo_update"]),
            "epoch": int(context["epoch"]),
            "minibatch": int(context["minibatch"]),
            "repeat_ok": True,
            "repeat_max_abs": float(repeat_gap),
            "actor_delta_j": j_norm,
            "actor_delta_s": s_norm,
            "actor_delta_abs_diff": absolute,
            "cosine": cosine,
            "r": ratio,
            "j_norm_zero": bool(j_norm == 0.0),
            "s_norm_zero": bool(s_norm == 0.0),
            "new_column_delta_j": _norm(column_j),
            "new_column_delta_s": _norm(column_s),
            "new_column_abs_diff": _norm(column_s - column_j),
            "policy_mu_l2": mu_gap,
            "policy_logprob_mean_abs": log_gap,
            "layer_gaps": _layer_gaps(actor, copies[0][0], copies[1][0]),
            "source_state_sha256": digest,
            "shadow_adam_steps": 3,
        }
    finally:
        restore_rng(saved_rng)
        if not same_rng(saved_rng):
            raise RuntimeError("shadow did not restore RNG")
        _unchanged(actor, critic, optimizer, actor_saved, critic_saved, adam_saved)
