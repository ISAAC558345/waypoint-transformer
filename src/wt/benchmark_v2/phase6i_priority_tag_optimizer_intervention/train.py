"""Six 32-update pilots. Shadow steps are logged apart from the PPO budget."""
from __future__ import annotations

import numpy as np
import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import PPO
from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.rollout import collect
from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.train import _tensor_sha256, _workers, build_pair
from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.replay import _install, _restore
from wt.round30.ppo import seed_everything

from .agent import GroupedPilot
from .clip import assert_disjoint
from .protocol import OUTPUT, REWARD_ABS, SHADOW_UPDATES, UPDATES
from .shadow import run_shadow


def _same_modules(left, right) -> bool:
    left_rows = list(left.state_dict().items())
    right_rows = list(right.state_dict().items())
    if len(left_rows) != len(right_rows):
        return False
    for (left_key, left_value), (right_key, right_value) in zip(left_rows, right_rows):
        if left_key != right_key or not torch.equal(left_value.detach().cpu(), right_value.detach().cpu()):
            return False
    return True


def _watch(workers, holder: dict, events: list[dict], totals: dict) -> list:
    originals = []
    for worker in workers:
        method = worker.step

        def call(bound=method, env_index=int(worker.env_index)):
            record = bound()
            gap = abs(float(record["reward"]) - (float(record["base_reward"]) + float(record["bonus_age"]) + float(record["bonus_tag"])))
            if gap > float(REWARD_ABS):
                raise RuntimeError("reward identity failed during the pilot")
            safety = int(record["safety_violation"])
            failure = int(record["constraint_failure"])
            totals["safety_violation_count"] += safety
            totals["constraint_failure_count"] += failure
            totals["steps"] += 1
            if safety or failure:
                events.append(
                    {
                        "variant": holder["variant"],
                        "model_seed": int(holder["model_seed"]),
                        "ppo_update": int(holder["ppo_update"]),
                        "env_index": int(env_index),
                        "safety_violation": safety,
                        "constraint_failure": failure,
                    }
                )
            return record

        worker.step = call
        originals.append(method)
    return originals


def _restore_steps(workers, originals) -> None:
    for worker, method in zip(workers, originals):
        worker.step = method


def train_variant(model_seed: int, variant: str, actor, critic, pilot: GroupedPilot) -> dict:
    if variant not in {"PILOT_J", "PILOT_S"}:
        raise RuntimeError(f"unknown pilot variant {variant}")
    assert_disjoint(actor, critic)
    torch.set_num_threads(1)
    seed_everything(int(model_seed))
    rollout, original = _install()
    events: list[dict] = []
    totals = {"safety_violation_count": 0, "constraint_failure_count": 0, "steps": 0}
    update_rows = []
    step_rows = []
    shadow_rows = []
    first_rewards = None
    holder = {"variant": variant, "model_seed": int(model_seed), "ppo_update": 0}
    error = None
    try:
        workers = _workers(int(model_seed), actor, critic, use_lookahead=True)
        originals = _watch(workers, holder, events, totals)
        try:
            for update in range(1, int(UPDATES) + 1):
                holder["ppo_update"] = int(update)
                storage, metrics, _samples = collect(workers, int(PPO.rollout_length))
                if int(update) == 1:
                    first_rewards = np.asarray(storage["rewards"], dtype=np.float64).reshape(-1).copy()
                destination = OUTPUT / "shadow_states" / f"{variant.lower()}_seed{int(model_seed)}_update{int(update)}.pt"

                def before_clip(context, path=destination, update_number=int(update)):
                    row = run_shadow(context, path)
                    row["source_variant"] = variant
                    row["model_seed"] = int(model_seed)
                    row["ppo_update"] = int(update_number)
                    shadow_rows.append(row)

                summary, steps = pilot.update(
                    storage,
                    ppo_update=int(update),
                    model_seed=int(model_seed),
                    before_clip=before_clip if int(update) in SHADOW_UPDATES else None,
                    step_sink=step_rows,
                )
                ordinary = float(metrics["mean_served_bits"]) - float(metrics["mean_served_tagged_bits"])
                update_rows.append(
                    {
                        "variant": variant,
                        "model_seed": int(model_seed),
                        "ppo_update": int(update),
                        "mean_served_bits": float(metrics["mean_served_bits"]),
                        "mean_served_tagged_bits": float(metrics["mean_served_tagged_bits"]),
                        "mean_served_ordinary_bits": float(ordinary),
                        "mean_bonus_age": float(metrics["mean_bonus_age"]),
                        "mean_bonus_tag": float(metrics["mean_bonus_tag"]),
                        "mean_motion_penalty": float(metrics["mean_motion_penalty"]),
                        "projection_rate": float(metrics["projection_rate"]),
                        "positive_tag_bonus_steps": int(metrics["positive_tag_bonus_steps"]),
                        "lookahead_nonzero_input_uav_steps": int(metrics["lookahead_nonzero_input_uav_steps"]),
                        "safety_violation_count": int(sum(1 for item in events if item["ppo_update"] == update and item["safety_violation"])),
                        "constraint_failure_count": int(sum(1 for item in events if item["ppo_update"] == update and item["constraint_failure"])),
                        "explained_variance": float(summary["explained_variance"]),
                        "policy_loss": float(summary["policy_loss"]),
                        "value_loss": float(summary["value_loss"]),
                        "entropy": float(summary["entropy"]),
                        "approx_kl": float(summary["approx_kl"]),
                        "clip_fraction": float(summary["clip_fraction"]),
                        "actor_delta_norm": float(summary["actor_delta_norm"]),
                        "critic_delta_norm": float(summary["critic_delta_norm"]),
                        "joint_pre_norm": float(summary["joint_pre_norm"]),
                        "joint_post_norm": float(summary["joint_post_norm"]),
                        "optimizer_steps": int(summary["optimizer_steps"]),
                    }
                )
                for row in steps:
                    row["model_seed"] = int(model_seed)
                if update == 1 or update % 8 == 0:
                    print(
                        f"{variant} seed {int(model_seed)} update {update}/{int(UPDATES)} "
                        f"served={float(metrics['mean_served_bits']):.1f} tagged={float(metrics['mean_served_tagged_bits']):.1f} "
                        f"actor_delta={float(summary['actor_delta_norm']):.3e}",
                        flush=True,
                    )
        finally:
            _restore_steps(workers, originals)
    except Exception as exc:
        error = exc
    finally:
        _restore(rollout, original)
    finished = error is None and len(update_rows) == int(UPDATES)
    checkpoint_name = f"{variant.lower()}_seed{int(model_seed)}.pt" if finished else f"incomplete_{variant.lower()}_seed{int(model_seed)}.pt"
    checkpoint = OUTPUT / "checkpoints" / checkpoint_name
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "optimizer": pilot.optimizer.state_dict(),
            "variant": variant,
            "model_seed": int(model_seed),
            "ppo_updates": int(len(update_rows)),
            "use_lookahead": True,
            "name_role": "independent_pilot_checkpoint" if finished else "incomplete_pilot_checkpoint",
        },
        checkpoint,
    )
    payload = {
        "variant": variant,
        "model_seed": int(model_seed),
        "complete": bool(finished),
        "ppo_updates": int(len(update_rows)),
        "adam_steps": int(len(step_rows)),
        "shadow_adam_steps": int(sum(int(row["shadow_adam_steps"]) for row in shadow_rows)),
        "safety_violation_count": int(totals["safety_violation_count"]),
        "constraint_failure_count": int(totals["constraint_failure_count"]),
        "mean_served_bits": None if not update_rows else float(np.mean([row["mean_served_bits"] for row in update_rows])),
        "mean_served_tagged_bits": None if not update_rows else float(np.mean([row["mean_served_tagged_bits"] for row in update_rows])),
        "mean_approx_kl": None if not update_rows else float(np.mean([row["approx_kl"] for row in update_rows])),
        "max_approx_kl": None if not update_rows else float(np.max([row["approx_kl"] for row in update_rows])),
        "final_actor_sha256": _tensor_sha256(actor),
        "final_critic_sha256": _tensor_sha256(critic),
        "checkpoint": str(checkpoint.relative_to(OUTPUT.parent.parent)).replace("\\", "/"),
        "update_rows": update_rows,
        "step_rows": step_rows,
        "shadow_rows": shadow_rows,
        "events": events,
        "first_rewards": first_rewards,
        "error": None if error is None else f"{type(error).__name__}: {error}",
    }
    if error is not None:
        raise PilotStopped(payload["error"], payload) from error
    return payload


class PilotStopped(RuntimeError):
    def __init__(self, message: str, partial: dict) -> None:
        super().__init__(message)
        self.partial = partial


def train_seed(model_seed: int) -> list[dict]:
    bundle = build_pair(int(model_seed))
    pilots = {
        "PILOT_J": GroupedPilot(bundle["L4_Q"]["actor"], bundle["L4_Q"]["critic"], "PILOT_J"),
        "PILOT_S": GroupedPilot(bundle["L4_C"]["actor"], bundle["L4_C"]["critic"], "PILOT_S"),
    }
    if not _same_modules(bundle["L4_Q"]["actor"], bundle["L4_C"]["actor"]):
        raise RuntimeError("PILOT_J and PILOT_S actors do not share the registered initialization")
    if not _same_modules(bundle["L4_Q"]["critic"], bundle["L4_C"]["critic"]):
        raise RuntimeError("PILOT_J and PILOT_S critics do not share the registered initialization")
    init_record = {
        "model_seed": int(model_seed),
        "actor_init_sha256": bundle["actor_init_sha256"],
        "critic_init_sha256": bundle["critic_init_sha256"],
        "lookahead_columns_zero": bool(torch.count_nonzero(dict(bundle["L4_Q"]["actor"].named_parameters())["body.0.weight"][:, -2:]).cpu() == 0),
    }
    if not init_record["lookahead_columns_zero"]:
        raise RuntimeError("new input columns did not start at zero")
    runs = []
    try:
        for variant, group_name in (("PILOT_J", "L4_Q"), ("PILOT_S", "L4_C")):
            group = bundle[group_name]
            result = train_variant(int(model_seed), variant, group["actor"], group["critic"], pilots[variant])
            result["initialization"] = init_record
            runs.append(result)
    except PilotStopped as exc:
        exc.partial["initialization"] = init_record
        runs.append(exc.partial)
        exc.runs = runs
        raise
    return runs
