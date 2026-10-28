"""Development evaluation for L4. Frozen L0 and L3 stay on their existing loaders."""
from __future__ import annotations

import numpy as np
import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import DEV_POOL, LOG_STD_INIT
from wt.benchmark_v2.phase6i_priority_tag.protocol import DUST_BITS
from wt.benchmark_v2.phase6i_priority_tag_attribution_full_replay.context import apply_historical_context
from wt.benchmark_v2.phase6i_priority_tag_direct_service_reward.evaluate import evaluate_l3, evaluate_saved, load_l3, load_reference
from wt.benchmark_v2.phase6i_priority_tag_reward.evaluate import _episode_row
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import EVAL_ASSIGNMENT_SEED, assert_dev_scenario
from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.replay import _blank_episode, _consume, _install, _mean_age, _restore

from .actor import LookaheadActor
from .protocol import OUTPUT
from .worker import QueueLookaheadWorker


def load_l4(variant: str, seed: int) -> LookaheadActor:
    path = OUTPUT / "checkpoints" / f"{variant.lower()}_seed{int(seed)}.pt"
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("variant") != variant or int(blob.get("model_seed")) != int(seed):
        raise RuntimeError("L4 checkpoint identity does not match the request")
    if "implementation_commit" not in blob or "instruction_commit" not in blob:
        raise RuntimeError("L4 checkpoint has no implementation trace")
    actor = LookaheadActor(log_std_init=float(LOG_STD_INIT))
    actor.load_state_dict(blob["actor"], strict=True)
    actor.eval()
    return actor


def evaluate_l4(actor, variant: str, seed: int, scenario: int) -> dict:
    assert_dev_scenario(int(scenario))
    apply_historical_context()
    rollout, original = _install()
    try:
        apply_historical_context()
        worker = QueueLookaheadWorker(
            int(EVAL_ASSIGNMENT_SEED) + int(scenario),
            0,
            actor,
            None,
            deterministic=True,
            scenario_id=int(scenario),
            use_lookahead=variant == "L4_Q",
        )
        worker.rng = np.random.default_rng(int(EVAL_ASSIGNMENT_SEED) + int(scenario))
        worker.reset()
        total = _blank_episode(int(worker.env.config.num_buoys))
        rows = []
        bonus_age = 0.0
        bonus_tag = 0.0
        for _ in range(int(worker.env.config.episode_horizon)):
            stepped = worker.step()
            rows.append(stepped)
            bonus_age += float(stepped["bonus_age"])
            bonus_tag += float(stepped["bonus_tag"])
            _consume(total, worker.ledger.step_attribution)
        ordinary, tagged = worker.ledger.bits_by_tag()
        base = _episode_row(variant, int(seed), int(scenario), rows)
        ordinary_bits = float(base["served_ordinary_bits"])
        tagged_bits = float(base["served_tagged_bits"])
        generated = float(total["generated_bits"])
        base.update(
            {
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
                "process_seed_role": "historical_inherited_process_context",
                "model_seed_role": "paired_training_seed",
            }
        )
        return base
    finally:
        _restore(rollout, original)


def evaluate_frozen_reference(variant: str, seed: int, scenario: int, saved: dict | None) -> dict:
    if variant == "L3":
        row = evaluate_l3(load_l3(int(seed)), int(seed), int(scenario))
    else:
        actor = load_reference(variant, int(seed))
        row, _status = evaluate_saved(variant, int(seed), int(scenario), saved, actor)
    row["process_seed_role"] = "historical_inherited_process_context"
    row["model_seed_role"] = "frozen_reference_seed"
    return row


def development_scenarios() -> tuple[int, ...]:
    return tuple(int(scenario) for scenario in DEV_POOL)
