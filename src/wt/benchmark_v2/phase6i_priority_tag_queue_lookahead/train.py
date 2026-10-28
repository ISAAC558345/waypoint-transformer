"""Train L4_C and L4_Q from one shared initialization. L0 through L3 stay frozen."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time

import numpy as np
import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import LOG_STD_INIT, PPO
from wt.benchmark_v2.phase6i_priority_tag_direct_service_reward.protocol import LAMBDA_AGE, LAMBDA_TAG, RHO_AGE, RHO_TAG
from wt.benchmark_v2.phase6i_priority_tag_direct_service_reward.reward import validate_config
from wt.benchmark_v2.phase6i_priority_tag_reward.actor import PriorityTagActor, load_formal_actor, transplant_legacy_actor
from wt.benchmark_v2.phase6i_priority_tag_reward.mappo import ExperimentMAPPO
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import LEGACY_ACTOR, TAG_RATE, TAG_SEED
from wt.benchmark_v2.phase6i_priority_tag_reward.rollout import critic_dim
from wt.benchmark_v2.phase6i_priority_tag_reward.smoke import file_sha256
from wt.benchmark_v2.phase6i_priority_tag_reward.train import source_commit
from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.replay import _install, _restore
from wt.lowlevel_v2.residual_policy import CentralizedCritic
from wt.round30.ppo import seed_everything

from .actor import LookaheadActor, same_module, transplant_head_actor
from .protocol import (
    E11_RESULT_COMMIT,
    INSTRUCTION_COMMIT,
    OUTPUT,
    ROOT,
    SAMPLE_ENV_INDEX,
    SAMPLE_UPDATES,
    SCHEMA_VERSION,
    TRAIN_SEEDS,
    UPDATES,
)
from .rollout import collect
from .worker import QueueLookaheadWorker


def _tensor_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in module.state_dict().items():
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _optimizer_sha256(optimizer: torch.optim.Optimizer) -> str:
    digest = hashlib.sha256()
    for group in optimizer.param_groups:
        digest.update(str(group["lr"]).encode("utf-8"))
        digest.update(str(len(group["params"])).encode("utf-8"))
        digest.update(str(sorted(optimizer.state.keys())).__repr__().encode("utf-8"))
    return digest.hexdigest()


def build_pair(model_seed: int) -> dict:
    """One legacy transplant, then two actors and two critics with identical parameters."""
    if int(model_seed) not in TRAIN_SEEDS:
        raise RuntimeError(f"unexpected model seed {model_seed}")
    validate_config()
    torch.set_num_threads(1)
    seed_everything(int(model_seed))
    head = PriorityTagActor(log_std_init=float(LOG_STD_INIT))
    transplant_legacy_actor(load_formal_actor(LEGACY_ACTOR, float(LOG_STD_INIT)), head)
    control_actor = LookaheadActor(log_std_init=float(LOG_STD_INIT))
    transplant_head_actor(head, control_actor)
    treated_actor = LookaheadActor(log_std_init=float(LOG_STD_INIT))
    treated_actor.load_state_dict(control_actor.state_dict())
    control_critic = CentralizedCritic(critic_dim())
    treated_critic = CentralizedCritic(critic_dim())
    treated_critic.load_state_dict(control_critic.state_dict())
    if not same_module(control_actor, treated_actor) or not same_module(control_critic, treated_critic):
        raise RuntimeError("paired initialization diverged before training")
    control_agent = ExperimentMAPPO(control_actor, control_critic, PPO)
    treated_agent = ExperimentMAPPO(treated_actor, treated_critic, PPO)
    if _optimizer_sha256(control_agent.optimizer) != _optimizer_sha256(treated_agent.optimizer):
        raise RuntimeError("paired optimizers do not start equal")
    return {
        "L4_C": {"actor": control_actor, "critic": control_critic, "agent": control_agent},
        "L4_Q": {"actor": treated_actor, "critic": treated_critic, "agent": treated_agent},
        "actor_init_sha256": _tensor_sha256(control_actor),
        "critic_init_sha256": _tensor_sha256(control_critic),
        "optimizer_init_sha256": _optimizer_sha256(control_agent.optimizer),
        "legacy_actor_sha256": file_sha256(LEGACY_ACTOR),
    }


def _workers(model_seed: int, actor, critic, *, use_lookahead: bool) -> list[QueueLookaheadWorker]:
    workers = []
    for index in range(int(PPO.num_envs)):
        worker = QueueLookaheadWorker(
            int(model_seed),
            int(index),
            actor,
            critic,
            deterministic=False,
            scenario_id=None,
            use_lookahead=bool(use_lookahead),
        )
        worker.reset()
        workers.append(worker)
    return workers


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    return value


def train_group(model_seed: int, variant: str, bundle: dict, implementation_commit: str, source_hashes: dict) -> dict:
    if variant not in {"L4_C", "L4_Q"}:
        raise RuntimeError(f"unexpected lookahead variant {variant}")
    group = bundle[variant]
    torch.set_num_threads(1)
    seed_everything(int(model_seed))
    if _tensor_sha256(group["actor"]) != bundle["actor_init_sha256"]:
        raise RuntimeError("actor parameters changed before its own training start")
    rollout, original = _install()
    started = time.perf_counter()
    samples = []
    try:
        workers = _workers(int(model_seed), group["actor"], group["critic"], use_lookahead=variant == "L4_Q")
        history = []
        for update in range(int(UPDATES)):
            storage, metrics, step_samples = collect(workers, int(PPO.rollout_length))
            stats = group["agent"].update(storage)
            if not np.isfinite(stats["policy_loss"]) or not np.isfinite(stats["gradient_norm_preclip"]):
                raise RuntimeError("NONFINITE")
            row = {"model_seed": int(model_seed), "variant": variant, "update": update + 1, **stats, **metrics}
            history.append(_plain(row))
            if update + 1 in SAMPLE_UPDATES:
                for sample in step_samples:
                    if int(SAMPLE_ENV_INDEX) == 0:
                        samples.append({"update": update + 1, "env_index": 0, **sample})
            if (update + 1) % 10 == 0 or update == 0:
                print(
                    f"{variant} seed {int(model_seed)} update {update + 1}/{int(UPDATES)} "
                    f"reward={float(np.mean(storage['rewards'])):.4f} tagged={metrics['mean_served_tagged_bits']:.1f}",
                    flush=True,
                )
    finally:
        _restore(rollout, original)
    destination = OUTPUT / "checkpoints" / f"{variant.lower()}_seed{int(model_seed)}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "actor": group["actor"].state_dict(),
        "critic": group["critic"].state_dict(),
        "model_seed": int(model_seed),
        "updates": int(UPDATES),
        "variant": variant,
        "use_lookahead": variant == "L4_Q",
        "initialization": "LEGACY_WARM_START_THEN_ZERO_LOOKAHEAD_COLUMNS",
        "warm_start_from_trained_l3": False,
        "actor_init_sha256": bundle["actor_init_sha256"],
        "critic_init_sha256": bundle["critic_init_sha256"],
        "optimizer_init_sha256": bundle["optimizer_init_sha256"],
        "paired_with": "L4_Q" if variant == "L4_C" else "L4_C",
        "reward_config": {
            "lambda_age": float(LAMBDA_AGE),
            "rho_age": float(RHO_AGE),
            "lambda_tag": float(LAMBDA_TAG),
            "rho_tag": float(RHO_TAG),
            "formula": "r_base + min(0.0005*A, 0.00025*P) + min(0.0010*T, 0.00075*P)",
        },
        "synthetic_tag": True,
        "tag_seed": int(TAG_SEED),
        "tag_rate": float(TAG_RATE),
        "schema_version": SCHEMA_VERSION,
        "instruction_commit": INSTRUCTION_COMMIT,
        "implementation_commit": implementation_commit,
        "e11_result_commit": E11_RESULT_COMMIT,
        "source_commit": source_commit(),
        "source_file_sha256": source_hashes,
        "legacy_actor_sha256": bundle["legacy_actor_sha256"],
        "critic_input": "l3_three_head_features",
        "actor_input": "three_head_features_plus_two_lookahead_columns",
        "sample_updates": list(SAMPLE_UPDATES),
        "sample_env_index": int(SAMPLE_ENV_INDEX),
        "ppo": {
            "updates": int(UPDATES),
            "rollout_length": int(PPO.rollout_length),
            "num_envs": int(PPO.num_envs),
            "gamma": float(PPO.gamma),
            "gae_lambda": float(PPO.gae_lambda),
            "learning_rate": float(PPO.learning_rate),
        },
    }
    if payload["source_commit"] != implementation_commit:
        raise RuntimeError("training source commit does not match the implementation commit")
    torch.save(payload, destination)
    log_path = OUTPUT / "training_logs" / f"{variant.lower()}_seed{int(model_seed)}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(history), encoding="utf-8")
    sample_path = OUTPUT / "training_samples" / f"{variant.lower()}_seed{int(model_seed)}.json"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(json.dumps(samples), encoding="utf-8")
    return {
        "path": str(destination.relative_to(ROOT)).replace("\\", "/"),
        "sha256": file_sha256(destination),
        "log_path": str(log_path.relative_to(ROOT)).replace("\\", "/"),
        "sample_path": str(sample_path.relative_to(ROOT)).replace("\\", "/"),
        "variant": variant,
        "model_seed": int(model_seed),
        "updates": int(UPDATES),
        "elapsed_seconds": float(time.perf_counter() - started),
        "final_policy_loss": float(history[-1]["policy_loss"]),
        "final_value_loss": float(history[-1]["value_loss"]),
        "final_gradient_norm": float(history[-1]["gradient_norm_preclip"]),
        "mean_safety_violation_rate": float(np.mean([item["safety_violation_rate"] for item in history])),
        "actor_init_sha256": bundle["actor_init_sha256"],
        "implementation_commit": implementation_commit,
    }


def assert_sources_committed(implementation_commit: str) -> None:
    head = source_commit()
    if head != implementation_commit:
        raise RuntimeError("HEAD is not the recorded implementation commit")
    dirty = subprocess.run(
        ["git", "-c", "core.longpaths=true", "status", "--porcelain", "--", "src", "tests"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if dirty.returncode != 0 or dirty.stdout.strip():
        raise RuntimeError("uncommitted source or test changes are present; training stays stopped")
    ancestor = subprocess.run(
        ["git", "-c", "core.longpaths=true", "merge-base", "--is-ancestor", INSTRUCTION_COMMIT, head],
        cwd=ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("implementation commit does not descend from the E12 instruction")
