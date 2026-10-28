"""Train L3 only. L0 stays frozen and L1/L2 checkpoints are not rewritten."""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import LOG_STD_INIT, PPO
from wt.benchmark_v2.phase6i_priority_tag_reward.actor import PriorityTagActor, load_formal_actor, transplant_legacy_actor
from wt.benchmark_v2.phase6i_priority_tag_reward.mappo import ExperimentMAPPO
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import LEGACY_ACTOR, OBSERVATION_SCHEMA, FEATURE_NAMES, SCHEMA_VERSION, TAG_RATE, TAG_SEED, UPDATES
from wt.benchmark_v2.phase6i_priority_tag_reward.rollout import critic_dim
from wt.benchmark_v2.phase6i_priority_tag_reward.smoke import file_sha256
from wt.benchmark_v2.phase6i_priority_tag_reward.train import _collect, source_commit
from wt.lowlevel_v2.residual_policy import CentralizedCritic
from wt.round30.ppo import seed_everything

from .protocol import LAMBDA_TAG, OUTPUT, REWARD_SCHEMA, RHO_TAG, ROOT, TRAIN_SEEDS, VARIANT
from .reward import validate_config
from .worker import DirectTagWorker


def train_l3(model_seed: int) -> dict:
    if int(model_seed) not in TRAIN_SEEDS:
        raise RuntimeError(f"unexpected L3 seed {model_seed}")
    validate_config(LAMBDA_TAG, RHO_TAG)
    torch.set_num_threads(1)
    seed_everything(int(model_seed))
    actor = PriorityTagActor(log_std_init=float(LOG_STD_INIT))
    transplant_legacy_actor(load_formal_actor(LEGACY_ACTOR, LOG_STD_INIT), actor)
    critic = CentralizedCritic(critic_dim())
    agent = ExperimentMAPPO(actor, critic, PPO)
    workers = []
    for index in range(int(PPO.num_envs)):
        worker = DirectTagWorker(int(model_seed), index, actor, critic, deterministic=False)
        worker.reset()
        workers.append(worker)
    history = []
    started = time.perf_counter()
    for update in range(int(UPDATES)):
        storage, metrics = _collect(workers, int(PPO.rollout_length))
        stats = agent.update(storage)
        if not np.isfinite(stats["policy_loss"]) or not np.isfinite(stats["gradient_norm_preclip"]):
            raise RuntimeError("NONFINITE")
        row = {"model_seed": int(model_seed), "variant": VARIANT, "update": update + 1, **stats, **metrics}
        history.append(row)
        if (update + 1) % 10 == 0 or update == 0:
            print(
                f"L3 seed {int(model_seed)} update {update + 1}/{int(UPDATES)} "
                f"reward={float(np.mean(storage['rewards'])):.4f} served={metrics['mean_served_bits']:.1f}",
                flush=True,
            )
    destination = OUTPUT / "checkpoints" / f"l3_seed{int(model_seed)}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "model_seed": int(model_seed),
        "updates": int(UPDATES),
        "variant": VARIANT,
        "initialization": "LEGACY_WARM_START_EXPLICIT_ZERO_NEW_COLUMNS",
        "warm_start_from_l1_or_l2": False,
        "observation_schema": OBSERVATION_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "reward_schema": REWARD_SCHEMA,
        "reward_config": {
            "lambda_tag": float(LAMBDA_TAG),
            "rho_tag": float(RHO_TAG),
            "formula": "r_base + min(lambda_tag * served_tagged_bits/1e6, rho_tag * served_bits/1e6)",
            "urgency_in_bonus": False,
        },
        "synthetic_tag_protocol": SCHEMA_VERSION,
        "synthetic_tag": True,
        "tag_seed": int(TAG_SEED),
        "tag_rate": float(TAG_RATE),
        "source_commit": source_commit(),
        "legacy_actor_sha256": file_sha256(LEGACY_ACTOR),
    }
    torch.save(payload, destination)
    history_path = OUTPUT / "logs" / f"l3_seed{int(model_seed)}.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history), encoding="utf-8")
    return {
        "path": str(destination.relative_to(ROOT)).replace("\\", "/"),
        "sha256": file_sha256(destination),
        "log_path": str(history_path.relative_to(ROOT)).replace("\\", "/"),
        "log_sha256": file_sha256(history_path),
        "variant": VARIANT,
        "model_seed": int(model_seed),
        "updates": int(UPDATES),
        "elapsed_seconds": float(time.perf_counter() - started),
        "final_policy_loss": float(history[-1]["policy_loss"]),
        "final_value_loss": float(history[-1]["value_loss"]),
        "final_gradient_norm": float(history[-1]["gradient_norm_preclip"]),
        "mean_safety_violation_rate": float(np.mean([row["safety_violation_rate"] for row in history])),
        "mean_constraint_failure_rate": float(np.mean([row["constraint_failure_rate"] for row in history])),
        "source_commit": payload["source_commit"],
    }
