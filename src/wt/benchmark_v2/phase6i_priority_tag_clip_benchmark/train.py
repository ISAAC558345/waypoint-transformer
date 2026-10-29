"""Fresh 300-update runs. Pilot checkpoints are not resumed."""
from __future__ import annotations

import numpy as np
import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import PPO
from wt.benchmark_v2.phase6i_priority_tag_credit_audit.probe import capture_rng
from wt.benchmark_v2.phase6i_priority_tag_optimizer_intervention.agent import GroupedPilot
from wt.benchmark_v2.phase6i_priority_tag_optimizer_intervention.clip import assert_disjoint
from wt.benchmark_v2.phase6i_priority_tag_optimizer_intervention.protocol import REWARD_ABS
from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.rollout import collect
from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.train import _tensor_sha256, _workers, build_pair
from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.replay import _install, _restore
from wt.round30.ppo import seed_everything

from .protocol import CLIP_RULE, OUTPUT, SNAPSHOT_UPDATES, UPDATES
from .safety import event_from, install_geometry, restore_geometry


class RunStopped(RuntimeError):
    def __init__(self, message: str, partial: dict) -> None:
        super().__init__(message)
        self.partial = partial


def _same(left, right) -> bool:
    left_rows = list(left.state_dict().items())
    right_rows = list(right.state_dict().items())
    if len(left_rows) != len(right_rows):
        return False
    return all(key_l == key_r and torch.equal(value_l.detach().cpu(), value_r.detach().cpu()) for (key_l, value_l), (key_r, value_r) in zip(left_rows, right_rows))


def _snapshot(path, actor, critic, pilot, workers, update: int, variant: str, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "optimizer": pilot.optimizer.state_dict(),
            "process_rng": capture_rng(),
            "worker_sampling": [
                {"env_index": int(worker.env_index), "cursor": int(getattr(worker, "cursor", -1)), "rng_state": worker.rng.bit_generator.state}
                for worker in workers
            ],
            "ppo_update": int(update),
            "variant": variant,
            "model_seed": int(seed),
            "restore_scope": "weights_adam_process_rng_and_worker_assignment_rng",
            "omitted": ["environment", "ledger", "option", "pending_hold"],
            "seamless_resume": False,
        },
        path,
    )


def _watch_collect(workers, holder, events, totals):
    originals = []
    for worker in workers:
        method = worker.step

        def call(bound=method, current=worker):
            record = bound()
            gap = abs(float(record["reward"]) - (float(record["base_reward"]) + float(record["bonus_age"]) + float(record["bonus_tag"])))
            if gap > float(REWARD_ABS):
                raise RuntimeError("reward identity failed during the full-budget run")
            totals["safety_violation_count"] += int(record["safety_violation"])
            totals["constraint_failure_count"] += int(record["constraint_failure"])
            event = event_from(current, record, holder)
            if event is not None:
                events.append(event)
            return record

        worker.step = call
        originals.append(method)
    return originals


def train_variant_watched(model_seed: int, variant: str, actor, critic, pilot: GroupedPilot, first_reference=None) -> dict:
    """Train with per-step safety geometry. The update body stays on GroupedPilot."""
    rule = CLIP_RULE[variant]
    assert_disjoint(actor, critic)
    torch.set_num_threads(1)
    seed_everything(int(model_seed))
    rollout, original = _install()
    events: list[dict] = []
    totals = {"safety_violation_count": 0, "constraint_failure_count": 0}
    update_rows = []
    step_rows = []
    snapshots = []
    first_rewards = None
    holder = {"variant": variant, "model_seed": int(model_seed), "ppo_update": 0}
    error = None
    geometry = []
    watched = []
    workers = []
    try:
        workers = _workers(int(model_seed), actor, critic, use_lookahead=True)
        geometry = [install_geometry(worker) for worker in workers]
        watched = _watch_collect(workers, holder, events, totals)
        for update in range(1, int(UPDATES) + 1):
            before_safety = int(totals["safety_violation_count"])
            before_failure = int(totals["constraint_failure_count"])
            holder["ppo_update"] = int(update)
            storage, metrics, _samples = collect(workers, int(PPO.rollout_length))
            if int(update) == 1:
                first_rewards = np.asarray(storage["rewards"], dtype=np.float64).reshape(-1).copy()
                if first_reference is not None:
                    gap = float(np.max(np.abs(first_rewards - np.asarray(first_reference, dtype=np.float64).reshape(-1))))
                    if gap != 0.0:
                        raise RuntimeError(f"first rollout diverged before the clip rule, max abs {gap}")
            summary, steps = pilot.update(storage, ppo_update=int(update), model_seed=int(model_seed), step_sink=step_rows)
            for row in steps:
                row["variant"] = variant
            update_rows.append(
                {
                    "variant": variant,
                    "model_seed": int(model_seed),
                    "ppo_update": int(update),
                    "mean_served_bits": float(metrics["mean_served_bits"]),
                    "mean_served_tagged_bits": float(metrics["mean_served_tagged_bits"]),
                    "mean_served_ordinary_bits": float(metrics["mean_served_bits"]) - float(metrics["mean_served_tagged_bits"]),
                    "mean_bonus_age": float(metrics["mean_bonus_age"]),
                    "mean_bonus_tag": float(metrics["mean_bonus_tag"]),
                    "mean_motion_penalty": float(metrics["mean_motion_penalty"]),
                    "projection_rate": float(metrics["projection_rate"]),
                    "positive_tag_bonus_steps": int(metrics["positive_tag_bonus_steps"]),
                    "safety_violation_count": int(totals["safety_violation_count"] - before_safety),
                    "constraint_failure_count": int(totals["constraint_failure_count"] - before_failure),
                    "explained_variance": float(summary["explained_variance"]),
                    "approx_kl": float(summary["approx_kl"]),
                    "clip_fraction": float(summary["clip_fraction"]),
                    "actor_delta_norm": float(summary["actor_delta_norm"]),
                    "critic_delta_norm": float(summary["critic_delta_norm"]),
                    "joint_pre_norm": float(summary["joint_pre_norm"]),
                    "joint_post_norm": float(summary["joint_post_norm"]),
                    "optimizer_steps": int(summary["optimizer_steps"]),
                }
            )
            if int(update) in SNAPSHOT_UPDATES:
                path = OUTPUT / "snapshots" / f"{variant.lower()}_seed{int(model_seed)}_update{int(update)}.pt"
                _snapshot(path, actor, critic, pilot, workers, int(update), variant, int(model_seed))
                snapshots.append(str(path.relative_to(OUTPUT.parent.parent)).replace("\\", "/"))
            if update == 1 or update % 25 == 0:
                print(
                    f"{variant} seed {int(model_seed)} update {update}/{int(UPDATES)} "
                    f"served={float(metrics['mean_served_bits']):.1f} tagged={float(metrics['mean_served_tagged_bits']):.1f}",
                    flush=True,
                )
    except Exception as exc:
        error = exc
    finally:
        for worker, method in zip(workers, watched):
            worker.step = method
        for worker, originals in zip(workers, geometry):
            restore_geometry(worker, originals)
        _restore(rollout, original)
    finished = error is None and len(update_rows) == int(UPDATES)
    name = f"{variant.lower()}_seed{int(model_seed)}.pt" if finished else f"incomplete_{variant.lower()}_seed{int(model_seed)}.pt"
    checkpoint = OUTPUT / "checkpoints" / name
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "optimizer": pilot.optimizer.state_dict(),
            "variant": variant,
            "model_seed": int(model_seed),
            "ppo_updates": int(len(update_rows)),
            "clip_rule": rule,
            "use_lookahead": True,
            "initialization": "LEGACY_WARM_START_THEN_ZERO_LOOKAHEAD_COLUMNS",
            "warm_start_from_pilot": False,
            "seamless_resume": False,
            "restore_scope": "weights_and_adam_only",
        },
        checkpoint,
    )
    payload = {
        "variant": variant,
        "model_seed": int(model_seed),
        "complete": bool(finished),
        "ppo_updates": int(len(update_rows)),
        "adam_steps": int(len(step_rows)),
        "safety_violation_count": int(totals["safety_violation_count"]),
        "constraint_failure_count": int(totals["constraint_failure_count"]),
        "mean_served_bits": None if not update_rows else float(np.mean([row["mean_served_bits"] for row in update_rows])),
        "mean_served_tagged_bits": None if not update_rows else float(np.mean([row["mean_served_tagged_bits"] for row in update_rows])),
        "mean_approx_kl": None if not update_rows else float(np.mean([row["approx_kl"] for row in update_rows])),
        "max_approx_kl": None if not update_rows else float(np.max([row["approx_kl"] for row in update_rows])),
        "final_actor_sha256": _tensor_sha256(actor),
        "checkpoint": str(checkpoint.relative_to(OUTPUT.parent.parent)).replace("\\", "/"),
        "snapshots": snapshots,
        "update_rows": update_rows,
        "step_rows": step_rows,
        "events": events,
        "first_rewards": first_rewards,
        "error": None if error is None else f"{type(error).__name__}: {error}",
    }
    if error is not None:
        raise RunStopped(payload["error"], payload) from error
    return payload


def train_seed(model_seed: int) -> list[dict]:
    bundle = build_pair(int(model_seed))
    pilots = {
        "FULL_J": GroupedPilot(bundle["L4_Q"]["actor"], bundle["L4_Q"]["critic"], "PILOT_J"),
        "FULL_S": GroupedPilot(bundle["L4_C"]["actor"], bundle["L4_C"]["critic"], "PILOT_S"),
    }
    if not _same(bundle["L4_Q"]["actor"], bundle["L4_C"]["actor"]) or not _same(bundle["L4_Q"]["critic"], bundle["L4_C"]["critic"]):
        raise RuntimeError("FULL_J and FULL_S do not share the registered initialization")
    if int(torch.count_nonzero(dict(bundle["L4_Q"]["actor"].named_parameters())["body.0.weight"][:, -2:]).cpu()) != 0:
        raise RuntimeError("new input columns did not start at zero")
    init = {
        "model_seed": int(model_seed),
        "actor_init_sha256": bundle["actor_init_sha256"],
        "critic_init_sha256": bundle["critic_init_sha256"],
        "warm_start_from_pilot": False,
    }
    runs = []
    try:
        reference = None
        for variant, group_name in (("FULL_J", "L4_Q"), ("FULL_S", "L4_C")):
            result = train_variant_watched(
                int(model_seed),
                variant,
                bundle[group_name]["actor"],
                bundle[group_name]["critic"],
                pilots[variant],
                first_reference=reference,
            )
            result["initialization"] = init
            runs.append(result)
            reference = result["first_rewards"]
    except RunStopped as exc:
        exc.partial["initialization"] = init
        runs.append(exc.partial)
        exc.runs = runs
        raise
    return runs
