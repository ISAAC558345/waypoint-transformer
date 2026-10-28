"""Side-channel lookahead on the frozen E11 trajectories. No actor is retrained here."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.phase6i_priority_tag.ledger import TagBucket
from wt.benchmark_v2.phase6i_priority_tag.protocol import DUST_BITS
from wt.benchmark_v2.phase6i_priority_tag_reward.features import fifo_key
from wt.benchmark_v2.phase6i_priority_tag_signal_audit.episode import run_episode
from wt.benchmark_v2.phase6i_priority_tag_signal_audit.protocol import L3_SEEDS
from wt.lowlevel_v2.contracts import env_target

from .features import queue_fractions
from .protocol import DIAGNOSTIC_SCENARIOS


def _as_buckets(rows: list[dict]) -> list[TagBucket]:
    return [
        TagBucket(int(item["admission_step"]), "snapshot", int(item["priority_tag"]), float(item["bits_remaining"]), int(item["insert_index"]))
        for item in rows
    ]


def exposure(episodes: list[dict]) -> dict:
    rows = []
    events = []
    ordinary_head_backlog = 0
    for episode in episodes:
        all_steps = 0
        valid_steps = 0
        backlog_steps = 0
        nonzero_steps = 0
        head_hidden_backlog = 0
        tracked: dict[tuple[int, int, int], dict] = {}
        for step in episode["steps"]:
            priority = np.asarray(step["priority"], dtype=np.float64)
            association = np.asarray(step["association"], dtype=np.int64)
            num_buoys = len(step["before_buckets"])
            for uav, token in enumerate(association):
                all_steps += 1
                buoy = int(env_target(int(token), num_buoys))
                if buoy < 0:
                    fractions = (0.0, 0.0)
                else:
                    fractions = queue_fractions(_as_buckets(step["before_buckets"][buoy]), int(step["step"]))
                    valid_steps += 1
                head_tag = float(priority[uav, 1])
                valid = float(priority[uav, 2]) > 0.0
                backlog = fractions[0] > 0.0
                if buoy >= 0 and backlog:
                    backlog_steps += 1
                if fractions[0] > 0.0 or fractions[1] > 0.0:
                    nonzero_steps += 1
                hidden = bool(buoy >= 0 and valid and head_tag == 0.0 and backlog)
                if hidden:
                    head_hidden_backlog += 1
                    ordinary_head_backlog += 1
                rows.append(
                    {
                        "variant": episode["variant"],
                        "model_seed": int(episode["model_seed"]),
                        "scenario_id": int(episode["scenario_id"]),
                        "step": int(step["step"]),
                        "uav": int(uav),
                        "assigned_buoy": int(buoy),
                        "head_tag": float(head_tag),
                        "head_valid": float(priority[uav, 2]),
                        "tagged_backlog_fraction": float(fractions[0]),
                        "fifo_before_fraction": float(fractions[1]),
                        "ordinary_head_with_tagged_backlog": hidden,
                    }
                )
            for buoy, buckets in enumerate(step["before_buckets"]):
                alive = _as_buckets(buckets)
                tagged = [bucket for bucket in alive if int(bucket.priority_tag) == 1 and float(bucket.bits_remaining) > float(DUST_BITS)]
                assigned = any(int(env_target(int(token), num_buoys)) == buoy for token in association)
                head = None if not alive else min(alive, key=fifo_key)
                for bucket in tagged:
                    key = (int(buoy), int(bucket.admission_step), int(bucket.insert_index))
                    item = tracked.setdefault(key, {"lookahead_step": None, "head_step": None, "admission_step": int(bucket.admission_step)})
                    if assigned and item["lookahead_step"] is None:
                        item["lookahead_step"] = int(step["step"])
                    if assigned and head is not None and fifo_key(head) == fifo_key(bucket) and item["head_step"] is None:
                        item["head_step"] = int(step["step"])
        advances = []
        censored_lookahead = 0
        censored_head = 0
        for key, item in tracked.items():
            events.append(
                {
                    "variant": episode["variant"],
                    "model_seed": int(episode["model_seed"]),
                    "scenario_id": int(episode["scenario_id"]),
                    "buoy": int(key[0]),
                    "admission_step": int(item["admission_step"]),
                    "lookahead_step": "" if item["lookahead_step"] is None else int(item["lookahead_step"]),
                    "head_step": "" if item["head_step"] is None else int(item["head_step"]),
                }
            )
            if item["lookahead_step"] is None:
                censored_lookahead += 1
            if item["head_step"] is None:
                censored_head += 1
            if item["lookahead_step"] is not None and item["head_step"] is not None:
                advances.append(int(item["head_step"]) - int(item["lookahead_step"]))
        episode["exposure_summary"] = {
            "uav_steps": all_steps,
            "valid_association_uav_steps": valid_steps,
            "tagged_backlog_valid_uav_steps": backlog_steps,
            "nonzero_lookahead_uav_steps": nonzero_steps,
            "ordinary_head_with_tagged_backlog_uav_steps": head_hidden_backlog,
            "lookahead_before_head_observed": len(advances),
            "lookahead_advance_median": None if not advances else float(np.median(np.asarray(advances))),
            "censored_lookahead": censored_lookahead,
            "censored_head": censored_head,
        }
    return {
        "rows": rows,
        "events": events,
        "episodes": [episode["exposure_summary"] | {"variant": episode["variant"], "model_seed": int(episode["model_seed"]), "scenario_id": int(episode["scenario_id"])} for episode in episodes],
        "ordinary_head_with_tagged_backlog_uav_steps": int(ordinary_head_backlog),
        "feasible": bool(ordinary_head_backlog > 0),
    }


def run_feasibility() -> dict:
    episodes = []
    for scenario in DIAGNOSTIC_SCENARIOS:
        episodes.append(run_episode("L0", 0, int(scenario)))
        for seed in L3_SEEDS:
            episodes.append(run_episode("L3", int(seed), int(scenario)))
    result = exposure(episodes)
    result["trajectories"] = len(episodes)
    result["status"] = "pass" if result["feasible"] else "inconclusive"
    return result
