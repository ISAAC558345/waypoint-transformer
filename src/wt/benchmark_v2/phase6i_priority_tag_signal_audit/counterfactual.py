"""Short residual perturbations on the preregistered snapshots."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.replay import _install, _restore

from .protocol import (
    CANDIDATE_OFFSETS,
    COUNTERFACTUAL_STEPS,
    OPPORTUNITY_PAYLOAD_BITS,
    OPPORTUNITY_TAG_BITS,
)
from .trace import restore_worker, take_step


def _roll(worker, offset: tuple[float, float], steps: int) -> dict:
    tagged = 0.0
    payload_before = float(worker.env.payload_bits)
    safety = 0
    constraints = 0
    projection = 0.0
    clipped = 0
    for _ in range(int(steps)):
        record = take_step(worker, np.asarray(offset, dtype=np.float64))
        tagged += float(record["served_tagged_bits"])
        safety += int(record["safety_violation"])
        constraints += int(record["constraint_failure"])
        projection += float(record["projection_intervention"])
        clipped += int(record["clipped"])
    return {
        "served_tagged_bits": float(tagged),
        "payload_delta_bits": float(worker.env.payload_bits) - payload_before,
        "safety_violations": int(safety),
        "constraint_failures": int(constraints),
        "projection_intervention_sum": float(projection),
        "clipped_steps": int(clipped),
        "positions": np.asarray(worker.env.uav_positions, dtype=np.float64).copy(),
    }


def is_opportunity(outcome: dict, baseline: dict, *, reproduced: bool) -> bool:
    tagged_gain = float(outcome["served_tagged_bits"]) - float(baseline["served_tagged_bits"])
    payload_gain = float(outcome["payload_delta_bits"]) - float(baseline["payload_delta_bits"])
    return bool(
        reproduced
        and tagged_gain > float(OPPORTUNITY_TAG_BITS)
        and payload_gain >= -float(OPPORTUNITY_PAYLOAD_BITS)
        and int(outcome["safety_violations"]) <= int(baseline["safety_violations"])
        and int(outcome["constraint_failures"]) <= int(baseline["constraint_failures"])
    )


def _same(left: dict, right: dict) -> bool:
    return bool(
        float(left["served_tagged_bits"]) == float(right["served_tagged_bits"])
        and float(left["payload_delta_bits"]) == float(right["payload_delta_bits"])
        and int(left["safety_violations"]) == int(right["safety_violations"])
        and np.array_equal(left["positions"], right["positions"])
    )


def run_counterfactuals(episode: dict) -> list[dict]:
    worker = episode["worker"]
    rollout, original = _install()
    rows = []
    try:
        for snap in episode["snapshots"]:
            restore_worker(worker, snap["capture"])
            baseline = _roll(worker, (0.0, 0.0), COUNTERFACTUAL_STEPS)
            restore_worker(worker, snap["capture"])
            repeat = _roll(worker, (0.0, 0.0), COUNTERFACTUAL_STEPS)
            reproduced = _same(baseline, repeat)
            for offset in CANDIDATE_OFFSETS:
                restore_worker(worker, snap["capture"])
                outcome = _roll(worker, offset, COUNTERFACTUAL_STEPS)
                tagged_gain = float(outcome["served_tagged_bits"]) - float(baseline["served_tagged_bits"])
                payload_gain = float(outcome["payload_delta_bits"]) - float(baseline["payload_delta_bits"])
                opportunity = is_opportunity(outcome, baseline, reproduced=reproduced)
                rows.append(
                    {
                        "variant": episode["variant"],
                        "model_seed": int(episode["model_seed"]),
                        "scenario_id": int(episode["scenario_id"]),
                        "step": int(snap["step"]),
                        "offset_x": float(offset[0]),
                        "offset_y": float(offset[1]),
                        "steps": int(COUNTERFACTUAL_STEPS),
                        "served_tagged_bits": float(outcome["served_tagged_bits"]),
                        "payload_delta_bits": float(outcome["payload_delta_bits"]),
                        "safety_violations": int(outcome["safety_violations"]),
                        "constraint_failures": int(outcome["constraint_failures"]),
                        "projection_intervention_sum": float(outcome["projection_intervention_sum"]),
                        "clipped_steps": int(outcome["clipped_steps"]),
                        "baseline_reproduced": bool(reproduced),
                        "tagged_gain_vs_baseline": tagged_gain,
                        "payload_gain_vs_baseline": payload_gain,
                        "short_horizon_opportunity": opportunity,
                    }
                )
    finally:
        _restore(rollout, original)
    return rows
