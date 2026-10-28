"""Paired development evaluation under the explicit historical runtime context."""
from __future__ import annotations

import numpy as np
import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import LOG_STD_INIT
from wt.benchmark_v2.phase6i_priority_tag.protocol import DUST_BITS
from wt.benchmark_v2.phase6i_priority_tag_attribution_full_replay.compare import episode_status
from wt.benchmark_v2.phase6i_priority_tag_attribution_full_replay.context import apply_historical_context
from wt.benchmark_v2.phase6i_priority_tag_attribution_full_replay.replay import replay_one
from wt.benchmark_v2.phase6i_priority_tag_reward.actor import load_formal_actor
from wt.benchmark_v2.phase6i_priority_tag_reward.evaluate import _episode_row, _load_experiment
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import EVAL_ASSIGNMENT_SEED, FORMAL_ACTOR, assert_dev_scenario
from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.replay import _blank_episode, _consume, _install, _mean_age, _restore, load_actor

from .protocol import OUTPUT, PREFLIGHT_SCENARIO, REWARD_SCHEMA
from .worker import DirectServiceWorker


def _annotate(row: dict, status: dict | None) -> dict:
    row["compared_to_saved"] = status is not None
    if status is None:
        row["matches_saved"] = False
        row["max_bit_gap"] = None
        row["path_gap_m"] = None
        return row
    row["matches_saved"] = bool(status["matches_saved"])
    row["max_bit_gap"] = float(status["max_bit_gap"])
    row["path_gap_m"] = float(status["path_gap_m"])
    return row


def evaluate_saved(variant: str, seed: int, scenario: int, saved: dict, actor) -> tuple[dict, dict]:
    assert_dev_scenario(int(scenario))
    row, _steps = replay_one(actor, variant, int(seed), int(scenario))
    status = episode_status(row, saved)
    return _annotate(row, status), status


def evaluate_l3(actor, seed: int, scenario: int) -> dict:
    assert_dev_scenario(int(scenario))
    apply_historical_context()
    rollout, original = _install()
    try:
        apply_historical_context()
        if bool(actor.training):
            raise RuntimeError("L3 actor is not in evaluation mode")
        worker = DirectServiceWorker(
            int(EVAL_ASSIGNMENT_SEED) + int(scenario),
            0,
            actor,
            None,
            deterministic=True,
            scenario_id=int(scenario),
        )
        worker.rng = np.random.default_rng(int(EVAL_ASSIGNMENT_SEED) + int(scenario))
        worker.reset()
        horizon = int(worker.env.config.episode_horizon)
        total = _blank_episode(int(worker.env.config.num_buoys))
        rows = []
        bonus_age = 0.0
        bonus_tag = 0.0
        for _ in range(horizon):
            stepped = worker.step()
            rows.append(stepped)
            bonus_age += float(stepped["bonus_age"])
            bonus_tag += float(stepped["bonus_tag"])
            _consume(total, worker.ledger.step_attribution)
        ordinary, tagged = worker.ledger.bits_by_tag()
        base = _episode_row("L3", int(seed), int(scenario), rows)
        tagged_bits = float(base["served_tagged_bits"])
        ordinary_bits = float(base["served_ordinary_bits"])
        generated = float(total["generated_bits"])
        row = {
            **base,
            "age_service_bits": float(total["age_service_bits"]),
            "tag_increment_service_bits": float(total["tag_increment_service_bits"]),
            "combined_urgency_served_bits": float(total["combined_urgency_served_bits"]),
            "bonus_age_sum": float(bonus_age),
            "bonus_tag_sum": float(bonus_tag),
            "served_age_mean_ordinary": _mean_age(total["ordinary_age_mass"], ordinary_bits),
            "served_age_mean_tagged": _mean_age(total["tagged_age_mass"], tagged_bits),
            "tag_draw_rate": float(total["generated_tagged_bits"] / generated) if generated > float(DUST_BITS) else 0.0,
            "final_ordinary_queue_bits": float(np.sum(ordinary)),
            "final_tagged_queue_bits": float(np.sum(tagged)),
            "env_reset_seed": int(scenario),
            "assignment_rng_seed": int(EVAL_ASSIGNMENT_SEED) + int(scenario),
        }
        del tagged_bits
        return _annotate(row, None)
    finally:
        _restore(rollout, original)


def load_reference(variant: str, seed: int):
    if variant == "L0":
        actor = load_formal_actor(FORMAL_ACTOR, LOG_STD_INIT)
    else:
        actor = load_actor(variant, int(seed))
    actor.eval()
    return actor


def assert_l3_checkpoint(seed: int) -> None:
    path = OUTPUT / "checkpoints" / f"l3_seed{int(seed)}.pt"
    blob = torch.load(path, map_location="cpu", weights_only=False)
    config = blob.get("reward_config") or {}
    if blob.get("reward_schema") != REWARD_SCHEMA or config.get("age_and_tag_are_separate") is not True:
        raise RuntimeError("L3 checkpoint does not record separate age and tag bonuses")
    if int(blob.get("model_seed")) != int(seed):
        raise RuntimeError("L3 checkpoint seed does not match the file")


def load_l3(seed: int):
    actor = _load_experiment(OUTPUT / "checkpoints" / f"l3_seed{int(seed)}.pt")
    actor.eval()
    return actor


def preflight_l0(saved: dict) -> dict:
    actor = load_reference("L0", 0)
    row, status = evaluate_saved("L0", 0, int(PREFLIGHT_SCENARIO), saved, actor)
    return {
        "scenario_id": int(PREFLIGHT_SCENARIO),
        "matches_saved": bool(status["matches_saved"]),
        "max_bit_gap": float(status["max_bit_gap"]),
        "path_gap_m": float(status["path_gap_m"]),
        "row": row,
    }
