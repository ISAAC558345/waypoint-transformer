"""Instrumented deterministic replay. Checkpoints, rewards, and actions stay frozen."""
from __future__ import annotations

import time

import numpy as np

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import DEV_POOL, LOG_STD_INIT
from wt.benchmark_v2.phase6i_priority_tag.follow import STEP_FIELDS, arrival_tags, snap_from_step
from wt.benchmark_v2.phase6i_priority_tag.protocol import DUST_BITS
from wt.benchmark_v2.phase6i_priority_tag_reward.actor import load_formal_actor
from wt.benchmark_v2.phase6i_priority_tag_reward.evaluate import _episode_row, _load_experiment
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import FORMAL_ACTOR, RHO, TRAIN_SEEDS
from wt.benchmark_v2.phase6i_priority_tag_reward.rollout import eval_worker, run_fixed_episode, variant_lambda

from .attribute import allocate_service, service_origin_split
from .protocol import ACCOUNTING_TOLERANCE_BITS, SOURCE_OUTPUT


COMPARED_FIELDS = (
    "payload_bits",
    "urgency_served_bits",
    "served_tagged_bits",
    "served_ordinary_bits",
    "dropped_tagged_bits",
    "dropped_ordinary_bits",
)
SUM_FIELDS = (
    "age_service_bits",
    "age_service_ordinary_bits",
    "age_service_tagged_bits",
    "tag_increment_service_bits",
    "combined_urgency_served_bits",
    "served_ordinary_bits",
    "served_tagged_bits",
    "tagged_served_below_full_boost_bits",
    "tagged_served_fully_saturated_bits",
    "ordinary_age_mass",
    "tagged_age_mass",
    "admitted_bits",
    "admitted_tagged_bits",
    "generated_bits",
    "generated_tagged_bits",
)


def follow_with_attribution(env, ledger, movement, association, scenario_id: int) -> dict:
    """Follow one formal step and record the pre-service FIFO allocation."""
    time_step = int(env.t)
    snap_before = np.asarray(env.cumulative_snap_bits, dtype=np.float64).copy()
    info = env.step(movement, association)
    missing = [key for key in STEP_FIELDS if key not in info]
    if missing:
        raise RuntimeError(f"formal step is missing {missing}")
    snap, snap_source = snap_from_step(env, snap_before, info)
    generated = np.asarray(info["generated_arrival_bits"], dtype=np.float64)
    attribution = allocate_service(ledger.buckets, info["served_bits"], time_step)
    tags = arrival_tags(int(scenario_id), time_step, generated)
    split = ledger.apply_transition(
        info["served_bits"],
        generated,
        info["admitted_arrival_bits"],
        info["dropped_bits_step"],
        snap,
        time_step,
        tags,
        float(env.tol_bits),
    )
    gap = ledger.assert_matches(env.q_bits, float(env.tol_bits))
    admitted_ordinary = float(np.sum(split["admitted_ordinary_bits"]))
    admitted_tagged = float(np.sum(split["admitted_tagged_bits"]))
    generated_tagged = float(sum(float(bits) for bits, tag in zip(generated, tags) if tag == 1))
    attribution["admitted_bits"] = admitted_ordinary + admitted_tagged
    attribution["admitted_tagged_bits"] = admitted_tagged
    attribution["generated_bits"] = float(np.sum(generated))
    attribution["generated_tagged_bits"] = generated_tagged
    attribution["ledger_urgency_bits"] = float(np.sum(split["urgency_served_bits"]))
    attribution["decomposition_gap_bits"] = float(
        attribution["combined_urgency_served_bits"] - attribution["ledger_urgency_bits"]
    )
    ledger.step_attribution = attribution
    return {
        "time_step": time_step,
        "info_keys": sorted(info.keys()),
        "snap_source": snap_source,
        "served_bits": float(np.sum(info["served_bits"])),
        "payload_increment_bits": float(info["payload_increment_bits"]),
        "generated_bits": float(np.sum(generated)),
        "admitted_bits": float(np.sum(info["admitted_arrival_bits"])),
        "dropped_bits": float(np.sum(info["dropped_bits_step"])),
        "snap_bits": float(np.sum(snap)),
        "served_ordinary_bits": float(np.sum(split["served_ordinary_bits"])),
        "served_tagged_bits": float(np.sum(split["served_tagged_bits"])),
        "dropped_ordinary_bits": float(np.sum(split["dropped_ordinary_bits"])),
        "dropped_tagged_bits": float(np.sum(split["dropped_tagged_bits"])),
        "admitted_ordinary_bits": admitted_ordinary,
        "admitted_tagged_bits": admitted_tagged,
        "urgency_served_bits": float(np.sum(split["urgency_served_bits"])),
        "ledger_gap_bits": float(np.max(np.abs(gap))),
        "q_bits": float(np.sum(env.q_bits)),
        "payload_bits": float(env.payload_bits),
        "tags": tags,
    }


def _install():
    import wt.benchmark_v2.phase6i_priority_tag_reward.rollout as rollout

    original = rollout.follow_step
    rollout.follow_step = follow_with_attribution
    return rollout, original


def _restore(rollout, original) -> None:
    rollout.follow_step = original


def _blank_episode(num_buoys: int) -> dict:
    blank = {key: 0.0 for key in SUM_FIELDS}
    blank["ordinary_age_max"] = 0.0
    blank["tagged_age_max"] = 0.0
    blank["decomposition_gap_bits"] = 0.0
    blank["served_ordinary_by_buoy"] = np.zeros(int(num_buoys), dtype=np.float64)
    blank["served_tagged_by_buoy"] = np.zeros(int(num_buoys), dtype=np.float64)
    return blank


def _consume(total: dict, step: dict) -> None:
    for key in SUM_FIELDS:
        total[key] += float(step[key])
    total["ordinary_age_max"] = max(float(total["ordinary_age_max"]), float(step["ordinary_age_max"]))
    total["tagged_age_max"] = max(float(total["tagged_age_max"]), float(step["tagged_age_max"]))
    total["decomposition_gap_bits"] = max(
        float(total["decomposition_gap_bits"]),
        abs(float(step["decomposition_gap_bits"])),
    )
    total["served_ordinary_by_buoy"] += np.asarray(step["served_ordinary_by_buoy"], dtype=np.float64)
    total["served_tagged_by_buoy"] += np.asarray(step["served_tagged_by_buoy"], dtype=np.float64)


def _mean_age(mass: float, bits: float) -> float:
    if float(bits) <= float(DUST_BITS):
        return 0.0
    return float(mass) / float(bits)


def replay_episode(actor, variant: str, seed: int, scenario: int, saved: dict | None) -> dict:
    rollout, original = _install()
    try:
        worker = eval_worker(actor, int(scenario), lambda_u=variant_lambda(variant), use_ledger=True)
        worker.reset()
        horizon = int(worker.env.config.episode_horizon)
        total = _blank_episode(int(worker.env.config.num_buoys))
        rows = []
        for _ in range(horizon):
            if int(worker.env.t) >= horizon:
                break
            rows.append(worker.step())
            _consume(total, worker.ledger.step_attribution)
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
            "generated_bits": float(total["generated_bits"]),
            "generated_tagged_bits": float(total["generated_tagged_bits"]),
            "tag_draw_rate": (
                float(total["generated_tagged_bits"] / total["generated_bits"])
                if total["generated_bits"] > float(DUST_BITS)
                else 0.0
            ),
            "tagged_service_ratio": (
                float(tagged_bits / (tagged_bits + ordinary_bits)) if tagged_bits + ordinary_bits > float(DUST_BITS) else 0.0
            ),
            "served_age_mean_ordinary": _mean_age(total["ordinary_age_mass"], ordinary_bits),
            "served_age_max_ordinary": float(total["ordinary_age_max"]),
            "served_age_mean_tagged": _mean_age(total["tagged_age_mass"], tagged_bits),
            "served_age_max_tagged": float(total["tagged_age_max"]),
            "final_ordinary_queue_bits": float(np.sum(ordinary)),
            "final_tagged_queue_bits": float(np.sum(tagged)),
            "matches_saved": True,
            "max_saved_gap_bits": 0.0,
        }
        if saved is not None:
            field_gaps = {
                key: abs(float(row[key]) - float(saved[key]))
                for key in COMPARED_FIELDS + ("path_length_m",)
            }
            row["_field_gaps"] = field_gaps
            row["max_saved_gap_bits"] = float(max(field_gaps[key] for key in COMPARED_FIELDS))
            row["matches_saved"] = bool(row["max_saved_gap_bits"] <= float(ACCOUNTING_TOLERANCE_BITS))
        row["decomposition_ok"] = bool(row["decomposition_gap_bits"] <= float(ACCOUNTING_TOLERANCE_BITS))
        row["_ordinary_by_buoy"] = total["served_ordinary_by_buoy"]
        row["_tagged_by_buoy"] = total["served_tagged_by_buoy"]
        return row
    finally:
        _restore(rollout, original)


def attach_origin_splits(rows: list[dict]) -> None:
    baseline = {int(row["scenario_id"]): row for row in rows if row["variant"] == "L0"}
    for row in rows:
        if row["variant"] == "L0":
            split = service_origin_split(
                row["_ordinary_by_buoy"],
                row["_tagged_by_buoy"],
                row["_ordinary_by_buoy"],
                row["_tagged_by_buoy"],
            )
        else:
            base = baseline[int(row["scenario_id"])]
            split = service_origin_split(
                base["_ordinary_by_buoy"],
                base["_tagged_by_buoy"],
                row["_ordinary_by_buoy"],
                row["_tagged_by_buoy"],
            )
        row.update(split)


def public_row(row: dict) -> dict:
    return {key: value for key, value in row.items() if not str(key).startswith("_")}


def uninstrumented_gaps(actor, variant: str, seed: int, scenario: int, saved: dict) -> dict[str, float]:
    """Repeat the saved evaluator with its original follow function."""
    worker = eval_worker(actor, int(scenario), lambda_u=variant_lambda(variant), use_ledger=True)
    rows = run_fixed_episode(worker, int(worker.env.config.episode_horizon))
    base = _episode_row(variant, int(seed), int(scenario), rows)
    return {
        key: abs(float(base[key]) - float(saved[key]))
        for key in COMPARED_FIELDS + ("path_length_m",)
    }


def load_actor(variant: str, seed: int):
    if variant == "L0":
        actor = load_formal_actor(FORMAL_ACTOR, LOG_STD_INIT)
        actor.eval()
        return actor
    path = SOURCE_OUTPUT / "checkpoints" / f"{variant.lower()}_seed{int(seed)}.pt"
    actor = _load_experiment(path)
    actor.eval()
    return actor


def replay_saved(saved_index: dict, *, scenarios=None, jobs=None) -> tuple[list[dict], dict]:
    """Replay registered development scenarios. Caller times the first episode."""
    selected = list(DEV_POOL if scenarios is None else scenarios)
    planned = [("L0", 0)] + [(variant, int(seed)) for variant in ("L1", "L2") for seed in TRAIN_SEEDS]
    if jobs is not None:
        planned = list(jobs)
    rows = []
    started = time.perf_counter()
    first_seconds = None
    actors = {}
    for variant, seed in planned:
        actors[(variant, int(seed))] = load_actor(variant, int(seed))
        if float(variant_lambda(variant)) < 0.0 or float(RHO) <= 0.0:
            raise RuntimeError("replay reward config drifted")
        for scenario in selected:
            key = (variant, int(seed), int(scenario))
            if key not in saved_index:
                raise RuntimeError(f"saved evaluation has no row for {key}")
            episode_started = time.perf_counter()
            row = replay_episode(actors[(variant, int(seed))], variant, int(seed), int(scenario), saved_index[key])
            if first_seconds is None:
                first_seconds = float(time.perf_counter() - episode_started)
            rows.append(row)
            print(
                f"{variant} seed {int(seed)} scenario {int(scenario)} gap={row['max_saved_gap_bits']:.6g} "
                f"age={row['age_service_bits']:.6g} tag_boost={row['tag_increment_service_bits']:.6g}",
                flush=True,
            )
            if not row["matches_saved"] or not row["decomposition_ok"]:
                return rows, {
                    "stopped_early": True,
                    "first_scenario_seconds": first_seconds,
                    "elapsed_seconds": float(time.perf_counter() - started),
                    "completed_episodes": len(rows),
                }
    if len({int(row["scenario_id"]) for row in rows if row["variant"] == "L0"}) == len(selected):
        attach_origin_splits(rows)
    return rows, {
        "stopped_early": False,
        "first_scenario_seconds": first_seconds,
        "elapsed_seconds": float(time.perf_counter() - started),
        "completed_episodes": len(rows),
    }
