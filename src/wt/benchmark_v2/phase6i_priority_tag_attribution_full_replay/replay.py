"""One deterministic episode with the historical context applied first."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.phase6i_priority_tag_reward.evaluate import _episode_row
from wt.benchmark_v2.phase6i_priority_tag_reward.rollout import eval_worker, variant_lambda
from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.replay import (
    _blank_episode,
    _consume,
    _install,
    _mean_age,
    _restore,
)
from wt.benchmark_v2.phase6i_priority_tag.protocol import DUST_BITS

from .context import apply_historical_context, assignment_seed, context_is_historical
from .protocol import (
    GLOBAL_SEED_ROLE,
    HISTORICAL_INHERITED_GLOBAL_SEED,
)


def _step_row(variant: str, seed: int, scenario: int, step_index: int, step: dict, attr: dict) -> dict:
    return {
        "variant": variant,
        "model_seed": int(seed),
        "scenario_id": int(scenario),
        "env_reset_seed": int(scenario),
        "assignment_rng_seed": assignment_seed(scenario),
        "process_rng_seed": int(HISTORICAL_INHERITED_GLOBAL_SEED),
        "process_rng_role": GLOBAL_SEED_ROLE,
        "step": int(step_index),
        "age_service_bits": float(attr["age_service_bits"]),
        "tag_increment_service_bits": float(attr["tag_increment_service_bits"]),
        "combined_urgency_served_bits": float(attr["combined_urgency_served_bits"]),
        "served_tagged_bits": float(attr["served_tagged_bits"]),
        "served_ordinary_bits": float(attr["served_ordinary_bits"]),
        "tagged_served_below_full_boost_bits": float(attr["tagged_served_below_full_boost_bits"]),
        "tagged_served_fully_saturated_bits": float(attr["tagged_served_fully_saturated_bits"]),
        "ledger_urgency_bits": float(attr["ledger_urgency_bits"]),
        "decomposition_gap_bits": float(attr["decomposition_gap_bits"]),
        "served_bits": float(step["served_bits"]),
        "dropped_bits": float(step["dropped_ordinary_bits"] + step["dropped_tagged_bits"]),
        "payload_bits": float(step["payload"]),
        "q_bits": float(step["q_bits"]),
        "path_step_m": float(step["path_step"]),
        "ledger_gap_bits": float(step["ledger_gap_bits"]),
    }


def replay_one(actor, variant: str, seed: int, scenario: int) -> tuple[dict, list[dict]]:
    """Replay one development episode from the explicit historical context."""
    applied = apply_historical_context()
    if not context_is_historical(applied):
        raise RuntimeError("historical context was not applied")
    if bool(actor.training):
        raise RuntimeError("actor is not in evaluation mode")
    rollout, original = _install()
    try:
        apply_historical_context()
        worker = eval_worker(actor, int(scenario), lambda_u=variant_lambda(variant), use_ledger=True)
        worker.reset()
        horizon = int(worker.env.config.episode_horizon)
        total = _blank_episode(int(worker.env.config.num_buoys))
        rows = []
        steps = []
        for index in range(horizon):
            if int(worker.env.t) >= horizon:
                break
            stepped = worker.step()
            attr = worker.ledger.step_attribution
            rows.append(stepped)
            steps.append(_step_row(variant, int(seed), int(scenario), index, stepped, attr))
            _consume(total, attr)
        if len(rows) != horizon:
            raise RuntimeError(f"replay returned {len(rows)} steps for horizon {horizon}")
        ordinary, tagged = worker.ledger.bits_by_tag()
        base = _episode_row(variant, int(seed), int(scenario), rows)
        tagged_bits = float(base["served_tagged_bits"])
        ordinary_bits = float(base["served_ordinary_bits"])
        split_gap = max(
            abs(float(total["served_tagged_bits"]) - tagged_bits),
            abs(float(total["served_ordinary_bits"]) - ordinary_bits),
        )
        generated = float(total["generated_bits"])
        row = {
            **base,
            "age_service_bits": float(total["age_service_bits"]),
            "age_service_ordinary_bits": float(total["age_service_ordinary_bits"]),
            "age_service_tagged_bits": float(total["age_service_tagged_bits"]),
            "tag_increment_service_bits": float(total["tag_increment_service_bits"]),
            "combined_urgency_served_bits": float(total["combined_urgency_served_bits"]),
            "recorded_urgency_served_bits": float(base["urgency_served_bits"]),
            "decomposition_gap_bits": float(max(total["decomposition_gap_bits"], split_gap)),
            "tagged_served_below_full_boost_bits": float(total["tagged_served_below_full_boost_bits"]),
            "tagged_served_fully_saturated_bits": float(total["tagged_served_fully_saturated_bits"]),
            "admitted_bits": float(total["admitted_bits"]),
            "admitted_tagged_bits": float(total["admitted_tagged_bits"]),
            "generated_bits": generated,
            "generated_tagged_bits": float(total["generated_tagged_bits"]),
            "tag_draw_rate": float(total["generated_tagged_bits"] / generated) if generated > float(DUST_BITS) else 0.0,
            "tagged_service_ratio": (
                float(tagged_bits / (tagged_bits + ordinary_bits)) if tagged_bits + ordinary_bits > float(DUST_BITS) else 0.0
            ),
            "served_age_mean_ordinary": _mean_age(total["ordinary_age_mass"], ordinary_bits),
            "served_age_max_ordinary": float(total["ordinary_age_max"]),
            "served_age_mean_tagged": _mean_age(total["tagged_age_mass"], tagged_bits),
            "served_age_max_tagged": float(total["tagged_age_max"]),
            "final_ordinary_queue_bits": float(np.sum(ordinary)),
            "final_tagged_queue_bits": float(np.sum(tagged)),
            "env_reset_seed": int(scenario),
            "assignment_rng_seed": assignment_seed(scenario),
            "process_rng_seed": int(HISTORICAL_INHERITED_GLOBAL_SEED),
            "process_rng_role": GLOBAL_SEED_ROLE,
            "_ordinary_by_buoy": total["served_ordinary_by_buoy"],
            "_tagged_by_buoy": total["served_tagged_by_buoy"],
        }
        return row, steps
    finally:
        _restore(rollout, original)
