"""Static GAE and gradient accounting. Parameters are never updated."""
from __future__ import annotations

import numpy as np
import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import PPO
from wt.lowlevel_v2.phase1b.mappo import tanh_log_prob_uav
from wt.round30.ppo import compute_gae


def _flat_count(rewards: np.ndarray) -> int:
    return int(rewards.shape[0] * rewards.shape[1])


def _float32_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, last_values: np.ndarray):
    advantages, returns = compute_gae(
        np.asarray(rewards, dtype=np.float32),
        np.asarray(values, dtype=np.float32),
        np.asarray(dones, dtype=np.float32),
        np.asarray(last_values, dtype=np.float32),
        float(PPO.gamma),
        float(PPO.gae_lambda),
    )
    return advantages, returns


def _float64_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, last_values: np.ndarray):
    """Diagnostic recurrence. Historical training did not use this dtype."""
    reward64 = np.asarray(rewards, dtype=np.float64)
    value64 = np.asarray(values, dtype=np.float64)
    done64 = np.asarray(dones, dtype=np.float64)
    advantages = np.zeros_like(reward64)
    next_advantage = np.zeros(reward64.shape[1], dtype=np.float64)
    next_value = np.asarray(last_values, dtype=np.float64)
    gamma = float(PPO.gamma)
    lam = float(PPO.gae_lambda)
    for index in range(reward64.shape[0] - 1, -1, -1):
        nonterminal = 1.0 - done64[index]
        delta = reward64[index] + gamma * next_value * nonterminal - value64[index]
        next_advantage = delta + gamma * lam * nonterminal * next_advantage
        advantages[index] = next_advantage
        next_value = value64[index]
    return advantages, advantages + value64


def _train_scale(advantages: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(advantages))
    scale = max(float(np.std(advantages)), 1.0e-6)
    return mean, scale


def _apply_scale(advantages: np.ndarray, mean: float, scale: float) -> np.ndarray:
    return (np.asarray(advantages, dtype=np.float64) - float(mean)) / float(scale)


def _subset(values: np.ndarray, mask: np.ndarray) -> dict:
    selected = np.asarray(values, dtype=np.float64).reshape(-1)[np.asarray(mask, dtype=bool).reshape(-1)]
    if int(selected.size) == 0:
        return {"count": 0, "mean_abs": None, "q50_abs": None, "q95_abs": None}
    absolute = np.abs(selected)
    return {
        "count": int(selected.size),
        "mean_abs": float(np.mean(absolute)),
        "q50_abs": float(np.quantile(absolute, 0.50)),
        "q95_abs": float(np.quantile(absolute, 0.95)),
    }


def decompose_advantages(base: np.ndarray, age: np.ndarray, tag: np.ndarray, values: np.ndarray, dones: np.ndarray, last_values: np.ndarray) -> dict:
    """Fixed critic values. Bonus streams do not perturb the stored value or bootstrap."""
    base64 = np.asarray(base, dtype=np.float64)
    age64 = np.asarray(age, dtype=np.float64)
    tag64 = np.asarray(tag, dtype=np.float64)
    full64 = base64 + age64 + tag64
    no_tag64 = base64 + age64
    full_f32, returns_f32 = _float32_gae(full64, values, dones, last_values)
    base_f32, _returns_base = _float32_gae(base64, values, dones, last_values)
    age_f32, _returns_age = _float32_gae(no_tag64, values, dones, last_values)
    delta_age = age_f32.astype(np.float64) - base_f32.astype(np.float64)
    delta_tag = full_f32.astype(np.float64) - age_f32.astype(np.float64)
    linear_gap = float(np.max(np.abs(full_f32.astype(np.float64) - (base_f32.astype(np.float64) + delta_age + delta_tag))))
    full_mean, full_scale = _train_scale(full_f32)
    age_mean, age_scale = _train_scale(age_f32)
    base_mean, base_scale = _train_scale(base_f32)
    shared_full = _apply_scale(full_f32, full_mean, full_scale)
    shared_base = _apply_scale(base_f32, full_mean, full_scale)
    shared_age = np.asarray(delta_age, dtype=np.float64) / float(full_scale)
    shared_tag = np.asarray(delta_tag, dtype=np.float64) / float(full_scale)
    shared_gap = float(np.max(np.abs(shared_full - (shared_base + shared_age + shared_tag))))
    own_full = _apply_scale(full_f32, full_mean, full_scale)
    own_no_tag = _apply_scale(age_f32, age_mean, age_scale)
    own_base = _apply_scale(base_f32, base_mean, base_scale)
    separate_gap = own_full - own_no_tag
    full64_adv, _full64_ret = _float64_gae(full64, values, dones, last_values)
    base64_adv, _base64_ret = _float64_gae(base64, values, dones, last_values)
    age64_adv, _age64_ret = _float64_gae(no_tag64, values, dones, last_values)
    diagnostic_gap = float(np.max(np.abs(full64_adv - (base64_adv + (age64_adv - base64_adv) + (full64_adv - age64_adv)))))
    positive = tag64 > 0.0
    zero_tag = tag64 == 0.0
    zero_reward = full64 == 0.0
    stored_gap = float(np.max(np.abs(np.asarray(full64, dtype=np.float32) - (np.asarray(base64, dtype=np.float32) + np.asarray(age64, dtype=np.float32) + np.asarray(tag64, dtype=np.float32)))))
    distinguishable = np.float32(full64) != np.float32(no_tag64)
    return {
        "advantages_full": full_f32,
        "advantages_base": base_f32,
        "advantages_no_tag": age_f32,
        "returns_full": returns_f32,
        "normalized_full": own_full.astype(np.float32),
        "normalized_no_tag_separate": own_no_tag.astype(np.float32),
        "normalized_base_separate": own_base.astype(np.float32),
        "normalized_no_tag_shared": _apply_scale(age_f32, full_mean, full_scale).astype(np.float32),
        "normalized_tag_component_shared": shared_tag.astype(np.float32),
        "normalized_base_shared": shared_base.astype(np.float32),
        "float32_linear_gap": linear_gap,
        "shared_scale_additive_gap": shared_gap,
        "float64_diagnostic_linear_gap": diagnostic_gap,
        "component_float32_sum_gap": stored_gap,
        "full_mean": full_mean,
        "full_scale": full_scale,
        "tag_advantage_all": _subset(delta_tag, np.ones_like(tag64, dtype=bool)),
        "tag_advantage_positive_bonus": _subset(delta_tag, positive),
        "tag_advantage_zero_bonus": _subset(delta_tag, zero_tag),
        "tag_advantage_zero_reward": _subset(delta_tag, zero_reward),
        "separate_renorm_difference_all": _subset(separate_gap, np.ones_like(tag64, dtype=bool)),
        "separate_renorm_difference_positive_bonus": _subset(separate_gap, positive),
        "separate_renorm_difference_zero_bonus": _subset(separate_gap, zero_tag),
        "shared_tag_component_positive_bonus": _subset(shared_tag, positive),
        "shared_tag_component_zero_bonus": _subset(shared_tag, zero_tag),
        "float32_distinguishable_steps": int(np.sum(distinguishable)),
        "positive_tag_steps": int(np.sum(positive)),
        "zero_tag_steps": int(np.sum(zero_tag)),
        "future_tag_advantage_on_zero_bonus": int(np.sum(zero_tag & (np.abs(delta_tag) > 1.0e-6))),
        "normalization_note": "shared_scale uses the full-advantage mean and population std; separate_renorm normalizes each stream by its own mean and std and is not additive",
    }


def _flatten_storage(storage: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    count = _flat_count(storage["rewards"])

    def flat(key):
        value = storage[key]
        return value.reshape((count,) + value.shape[2:])

    return {key: flat(key) for key in ("self", "task", "communication", "neighbor_set", "d_nearest", "priority", "global_state", "pre_tanh", "log_prob_uav", "policy_active_mask")}


def _as_parameters(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def _pack(parameters: list[torch.nn.Parameter], grads) -> torch.Tensor:
    pieces = []
    for parameter, grad in zip(parameters, grads):
        if grad is None:
            pieces.append(torch.zeros(parameter.numel(), dtype=torch.float32))
        else:
            pieces.append(grad.detach().reshape(-1).cpu().float())
    if not pieces:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(pieces)


def _norm(vector: torch.Tensor) -> float:
    if int(vector.numel()) == 0:
        return 0.0
    return float(torch.linalg.vector_norm(vector).cpu())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left_norm = _norm(left)
    right_norm = _norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return float(torch.dot(left, right).cpu() / (left_norm * right_norm))


def _difference_zero_count(parameters: list[torch.nn.Parameter], left, right) -> int:
    count = 0
    for parameter, left_grad, right_grad in zip(parameters, left, right):
        if left_grad is None and right_grad is None:
            count += 1
            continue
        left_tensor = torch.zeros_like(parameter) if left_grad is None else left_grad.detach()
        right_tensor = torch.zeros_like(parameter) if right_grad is None else right_grad.detach()
        if float(torch.linalg.vector_norm(left_tensor - right_tensor).cpu()) == 0.0:
            count += 1
    return int(count)


def _zero_count(parameters: list[torch.nn.Parameter], grads) -> int:
    count = 0
    for parameter, grad in zip(parameters, grads):
        if grad is None or float(torch.linalg.vector_norm(grad.detach()).cpu()) == 0.0:
            count += 1
        del parameter
    return int(count)


def _policy_forward(actor, flat: dict[str, np.ndarray], advantage: np.ndarray):
    self_t = torch.as_tensor(flat["self"], dtype=torch.float32)
    task_t = torch.as_tensor(flat["task"], dtype=torch.float32)
    comm_t = torch.as_tensor(flat["communication"], dtype=torch.float32)
    neigh_t = torch.as_tensor(flat["neighbor_set"], dtype=torch.float32)
    near_t = torch.as_tensor(flat["d_nearest"], dtype=torch.float32)
    priority_t = torch.as_tensor(flat["priority"], dtype=torch.float32)
    pre_t = torch.as_tensor(flat["pre_tanh"], dtype=torch.float32)
    old = torch.as_tensor(flat["log_prob_uav"], dtype=torch.float32)
    mask = torch.as_tensor(flat["policy_active_mask"], dtype=torch.float32)
    mu, log_std = actor(self_t, task_t, comm_t, neigh_t, near_t, priority_t)
    log_prob, entropy = tanh_log_prob_uav(mu, log_std, pre_t)
    advantage_t = torch.as_tensor(np.asarray(advantage, dtype=np.float32).reshape(-1), dtype=torch.float32).unsqueeze(-1)
    ratio = (log_prob - old).exp()
    clipped = ratio.clamp(1.0 - float(PPO.clip_eps), 1.0 + float(PPO.clip_eps))
    per_uav = -torch.minimum(ratio * advantage_t, clipped * advantage_t)
    denom = mask.sum().clamp_min(1.0)
    policy_loss = (per_uav * mask).sum() / denom
    entropy_term = (entropy * mask).sum() / denom
    return {
        "log_prob": log_prob,
        "old": old,
        "mask": mask,
        "ratio": ratio,
        "per_uav": per_uav,
        "policy_loss": policy_loss,
        "entropy_term": entropy_term,
        "log_std": log_std,
    }


def _grad_pack(loss: torch.Tensor, parameters: list[torch.nn.Parameter], *, retain: bool) -> torch.Tensor:
    grads = torch.autograd.grad(loss, parameters, retain_graph=retain, allow_unused=True)
    return _pack(parameters, grads), grads


def clip_scale(total_norm: float, max_norm: float | None = None) -> float:
    """Display the registered clip_grad_norm_ scale. Do not apply it."""
    limit = float(PPO.max_grad_norm if max_norm is None else max_norm)
    return float(min(1.0, limit / (float(total_norm) + 1.0e-6)))


def static_gradients(actor, critic, storage: dict[str, np.ndarray], components: dict) -> dict:
    """autograd.grad on one frozen batch. This does not write parameter gradients or take an optimizer step."""
    decomposed = decompose_advantages(
        components["base"],
        components["age"],
        components["tag"],
        storage["values"],
        storage["dones"],
        storage["last_values"],
    )
    flat = _flatten_storage(storage)
    actor_params = _as_parameters(actor)
    critic_params = _as_parameters(critic)
    before_actor = [parameter.detach().clone() for parameter in actor_params]
    before_critic = [parameter.detach().clone() for parameter in critic_params]
    streams = {
        "full": decomposed["normalized_full"],
        "no_tag_separate": decomposed["normalized_no_tag_separate"],
        "base_separate": decomposed["normalized_base_separate"],
        "no_tag_shared": decomposed["normalized_no_tag_shared"],
        "base_shared": decomposed["normalized_base_shared"],
    }
    packed = {}
    raw = {}
    logprob_gap = None
    ratio_gap = None
    per_uav_gap = None
    for name, advantage in streams.items():
        forward = _policy_forward(actor, flat, advantage)
        if logprob_gap is None:
            gap = (forward["log_prob"] - forward["old"]).detach()
            logprob_gap = float(torch.max(torch.abs(gap)).cpu())
            ratio_gap = float(torch.max(torch.abs(forward["ratio"].detach() - 1.0)).cpu())
        vector, grads = _grad_pack(forward["policy_loss"], actor_params, retain=name == "full")
        packed[name] = vector
        raw[name] = grads
        if name == "full":
            entropy_loss = -float(PPO.entropy_coef) * forward["entropy_term"]
            entropy_vector, _entropy_grads = _grad_pack(entropy_loss, actor_params, retain=True)
            packed["entropy"] = entropy_vector
            within = forward["per_uav"].detach()
            center = within[:, :1]
            within_gap = float(torch.max(torch.abs(within - center)).cpu())
            denom = forward["mask"].sum().clamp_min(1.0)
            piece_vectors = []
            width = int(forward["per_uav"].shape[-1])
            for uav in range(width):
                term = (forward["per_uav"][:, uav] * forward["mask"][:, uav]).sum() / denom
                piece_vectors.append(_grad_pack(term, actor_params, retain=uav + 1 < width)[0])
            summed = torch.stack(piece_vectors).sum(dim=0)
            per_uav_gap = float(torch.max(torch.abs(summed - vector)).cpu())
            forward["policy_loss"].detach()
        del forward
    value_target = torch.as_tensor(decomposed["returns_full"].reshape(-1), dtype=torch.float32)
    global_state = torch.as_tensor(flat["global_state"], dtype=torch.float32)
    value = critic(global_state)
    value_loss = 0.5 * (value_target - value).square().mean()
    critic_vector, critic_grads = _grad_pack(float(PPO.value_coef) * value_loss, critic_params, retain=False)
    actor_combined = packed["full"] + packed["entropy"]
    critic_norm = _norm(critic_vector)
    actor_norm = _norm(actor_combined)
    joint_norm = float(np.sqrt(actor_norm ** 2 + critic_norm ** 2))
    joint = clip_scale(joint_norm)
    new_column_norm = 0.0
    named = dict(actor.named_parameters())
    weight = named.get("body.0.weight")
    if weight is not None:
        index = next(position for position, parameter in enumerate(actor_params) if parameter is weight)
        column_grad = raw["full"][index]
        if column_grad is not None:
            new_column_norm = float(torch.linalg.vector_norm(column_grad.detach()[:, -2:]).cpu())
    tag_difference = packed["full"] - packed["no_tag_separate"]
    tag_difference_shared = packed["full"] - packed["no_tag_shared"]
    unchanged = all(torch.equal(parameter.detach(), saved) for parameter, saved in zip(actor_params, before_actor))
    unchanged = unchanged and all(torch.equal(parameter.detach(), saved) for parameter, saved in zip(critic_params, before_critic))
    mask = np.asarray(storage["policy_active_mask"], dtype=np.float32)
    return {
        "decomposed": decomposed,
        "logprob_max_abs": logprob_gap,
        "ratio_max_abs_from_one": ratio_gap,
        "per_uav_gradient_sum_max_abs": per_uav_gap,
        "within_step_policy_term_max_abs": within_gap,
        "policy_norm_full": _norm(packed["full"]),
        "policy_norm_no_tag_separate": _norm(packed["no_tag_separate"]),
        "policy_norm_base_separate": _norm(packed["base_separate"]),
        "policy_norm_no_tag_shared": _norm(packed["no_tag_shared"]),
        "policy_norm_base_shared": _norm(packed["base_shared"]),
        "tag_difference_separate_norm": _norm(tag_difference),
        "tag_difference_shared_norm": _norm(tag_difference_shared),
        "tag_difference_separate_cosine_vs_full": _cosine(tag_difference, packed["full"]),
        "tag_difference_separate_cosine_vs_base": _cosine(tag_difference, packed["base_separate"]),
        "tag_difference_shared_cosine_vs_full": _cosine(tag_difference_shared, packed["full"]),
        "entropy_norm": _norm(packed["entropy"]),
        "actor_policy_entropy_norm": actor_norm,
        "critic_weighted_norm": critic_norm,
        "joint_norm": joint_norm,
        "critic_squared_share": None if joint_norm == 0.0 else float((critic_norm ** 2) / (joint_norm ** 2)),
        "joint_clip_scale": joint,
        "separate_actor_clip_scale": clip_scale(actor_norm),
        "separate_critic_clip_scale": clip_scale(critic_norm),
        "new_input_column_grad_norm": new_column_norm,
        "policy_zero_parameter_count": _zero_count(actor_params, raw["full"]),
        "tag_difference_zero_parameter_count": _difference_zero_count(actor_params, raw["full"], raw["no_tag_separate"]),
        "mask_min": float(np.min(mask)),
        "mask_max": float(np.max(mask)),
        "parameters_unchanged": bool(unchanged),
        "clip_applied": False,
        "optimizer_step_called": False,
    }
