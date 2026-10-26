"""Paired development-pool evaluation. Reserved and test splits are not read."""
from __future__ import annotations

import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import DEV_POOL, LOG_STD_INIT
from wt.lowlevel_v2.residual_policy import CentralizedCritic

from .actor import PriorityTagActor, load_formal_actor
from .protocol import FORMAL_ACTOR, OBSERVATION_SCHEMA, OUTPUT, RHO, TRAIN_SEEDS
from .rollout import critic_dim, eval_worker, run_fixed_episode, variant_lambda


def _episode_row(variant: str, seed: int | None, scenario: int, rows: list[dict]) -> dict:
    return {
        "variant": variant,
        "model_seed": "" if seed is None else int(seed),
        "scenario_id": int(scenario),
        "steps": len(rows),
        "payload_bits": float(rows[-1]["payload"]),
        "urgency_served_bits": float(sum(row["urgency_served_bits"] for row in rows)),
        "served_tagged_bits": float(sum(row["served_tagged_bits"] for row in rows)),
        "served_ordinary_bits": float(sum(row["served_ordinary_bits"] for row in rows)),
        "dropped_tagged_bits": float(sum(row["dropped_tagged_bits"] for row in rows)),
        "dropped_ordinary_bits": float(sum(row["dropped_ordinary_bits"] for row in rows)),
        "dropped_bits": float(sum(row["dropped_ordinary_bits"] + row["dropped_tagged_bits"] for row in rows)),
        "final_queue_bits": float(rows[-1]["q_bits"]),
        "final_tagged_queue_bits": float(rows[-1]["remaining_tagged_bits"]),
        "queue_age_mean": float(sum(row["queue_age_mean"] for row in rows) / max(len(rows), 1)),
        "queue_age_max": float(max(row["queue_age_max"] for row in rows)),
        "sinr_mean": float(sum(row["sinr"] for row in rows) / max(len(rows), 1)),
        "interference_mean": float(sum(row["interference"] for row in rows) / max(len(rows), 1)),
        "path_length_m": float(sum(row["path_step"] for row in rows)),
        "projection_rate": float(sum(row["projected"] for row in rows) / max(len(rows), 1)),
        "safety_violations": float(sum(row["safety_violation"] for row in rows)),
        "constraint_failures": float(sum(row["constraint_failure"] for row in rows)),
        "return_base": float(sum(row["base_reward"] for row in rows)),
        "return_augmented": float(sum(row["reward"] for row in rows)),
        "bonus_sum": float(sum(row["bonus"] for row in rows)),
        "max_ledger_gap_bits": float(max(row["ledger_gap_bits"] for row in rows)),
    }


def _load_experiment(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("observation_schema") != OBSERVATION_SCHEMA:
        raise RuntimeError("experiment checkpoint has no observation schema")
    if "reward_config" not in blob:
        raise RuntimeError("experiment checkpoint has no reward config")
    actor = PriorityTagActor(log_std_init=float(LOG_STD_INIT))
    critic = CentralizedCritic(critic_dim())
    actor.load_state_dict(blob["actor"], strict=True)
    critic.load_state_dict(blob["critic"], strict=True)
    actor.eval()
    return actor


def evaluate_all() -> list[dict]:
    rows = []
    formal = load_formal_actor(FORMAL_ACTOR, LOG_STD_INIT)
    formal.eval()
    for scenario in DEV_POOL:
        worker = eval_worker(formal, int(scenario), lambda_u=0.0, use_ledger=True)
        episode = run_fixed_episode(worker, int(worker.env.config.episode_horizon))
        rows.append(_episode_row("L0", 0, int(scenario), episode))
    for variant in ("L1", "L2"):
        for seed in TRAIN_SEEDS:
            path = OUTPUT / "checkpoints" / f"{variant.lower()}_seed{int(seed)}.pt"
            actor = _load_experiment(path)
            if float(variant_lambda(variant)) < 0.0 or float(RHO) <= 0.0:
                raise RuntimeError("evaluation reward config drifted")
            for scenario in DEV_POOL:
                worker = eval_worker(actor, int(scenario), lambda_u=variant_lambda(variant), use_ledger=True)
                episode = run_fixed_episode(worker, int(worker.env.config.episode_horizon))
                rows.append(_episode_row(variant, int(seed), int(scenario), episode))
    return rows
