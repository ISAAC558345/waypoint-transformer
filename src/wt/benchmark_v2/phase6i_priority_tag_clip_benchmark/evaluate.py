"""Development-pool evaluation. Both clip groups keep the queue lookahead on."""
from __future__ import annotations

import numpy as np
import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import DEV_POOL, LOG_STD_INIT
from wt.benchmark_v2.phase6i_priority_tag.protocol import DUST_BITS
from wt.benchmark_v2.phase6i_priority_tag_attribution_full_replay.context import apply_historical_context
from wt.benchmark_v2.phase6i_priority_tag_direct_service_reward.evaluate import evaluate_saved, load_reference, preflight_l0
from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.actor import LookaheadActor
from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.worker import QueueLookaheadWorker
from wt.benchmark_v2.phase6i_priority_tag_reward.evaluate import _episode_row
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import EVAL_ASSIGNMENT_SEED, assert_dev_scenario
from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.reconcile import load_saved_rows, saved_index
from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.replay import _blank_episode, _consume, _install, _mean_age, _restore

from .protocol import OUTPUT, UPDATES, VARIANTS
from .safety import event_from, install_geometry, restore_geometry


def development_scenarios() -> tuple[int, ...]:
    scenarios = tuple(int(scenario) for scenario in DEV_POOL)
    if scenarios[0] != 321096 or scenarios[-1] != 321127 or len(scenarios) != 32:
        raise RuntimeError("development pool drifted")
    return scenarios


def assert_checkpoint_identity(blob: dict, variant: str, seed: int) -> None:
    if variant not in VARIANTS:
        raise RuntimeError(f"unexpected full-budget variant {variant}")
    if blob.get("variant") != variant or int(blob.get("model_seed")) != int(seed):
        raise RuntimeError("full-budget checkpoint identity does not match the request")
    if int(blob.get("ppo_updates")) != int(UPDATES) or blob.get("warm_start_from_pilot") is not False:
        raise RuntimeError("refusing a checkpoint that is not a fresh 300-update run")
    if blob.get("seamless_resume") is not False:
        raise RuntimeError("a full-budget checkpoint must not claim seamless resume")


def load_full(variant: str, seed: int):
    path = OUTPUT / "checkpoints" / f"{variant.lower()}_seed{int(seed)}.pt"
    blob = torch.load(path, map_location="cpu", weights_only=False)
    assert_checkpoint_identity(blob, variant, int(seed))
    actor = LookaheadActor(log_std_init=float(LOG_STD_INIT))
    actor.load_state_dict(blob["actor"], strict=True)
    actor.eval()
    return actor


def evaluate_full(actor, variant: str, seed: int, scenario: int, events: list[dict]) -> dict:
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
            use_lookahead=True,
        )
        worker.rng = np.random.default_rng(int(EVAL_ASSIGNMENT_SEED) + int(scenario))
        originals = install_geometry(worker)
        holder = {"variant": variant, "model_seed": int(seed), "ppo_update": 0}
        try:
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
                event = event_from(worker, stepped, holder)
                if event is not None:
                    event["phase"] = "evaluation"
                    event["scenario_id"] = int(scenario)
                    events.append(event)
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
                    "use_lookahead": True,
                }
            )
            return base
        finally:
            restore_geometry(worker, originals)
    finally:
        _restore(rollout, original)


def run_preflight() -> dict:
    index = saved_index(load_saved_rows())
    result = preflight_l0(index[("L0", 0, 321101)])
    result["role"] = "frozen_l0_context_check"
    return result


def evaluate_pool(actors: dict[tuple[str, int], object]) -> tuple[list[dict], list[dict]]:
    rows = []
    events: list[dict] = []
    l0 = load_reference("L0", 0)
    index = saved_index(load_saved_rows())
    for scenario in development_scenarios():
        row, _status = evaluate_saved("L0", 0, int(scenario), index[("L0", 0, int(scenario))], l0)
        row["process_seed_role"] = "historical_inherited_process_context"
        row["model_seed_role"] = "frozen_reference_seed"
        rows.append(row)
        print(f"L0 {int(scenario)} tagged={float(row['served_tagged_bits']):.1f}", flush=True)
    for variant in VARIANTS:
        for seed in (0, 1, 2):
            actor = actors[(variant, int(seed))]
            for scenario in development_scenarios():
                rows.append(evaluate_full(actor, variant, int(seed), int(scenario), events))
            print(f"{variant} seed {int(seed)} evaluation done", flush=True)
    if len(rows) != 224:
        raise RuntimeError(f"evaluation produced {len(rows)} episodes, not 224")
    return rows, events
