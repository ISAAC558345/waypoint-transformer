"""Flatten diagnostic episodes into scalar tables."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from wt.lowlevel_v2.contracts import env_target

from .trace import fifo_key_from, head_identity


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    return str(value)


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in columns})


def step_rows(episodes: list[dict]) -> list[dict]:
    rows = []
    for episode in episodes:
        for step in episode["steps"]:
            item = step.get("sensitivity")
            common = {
                "variant": episode["variant"],
                "model_seed": int(episode["model_seed"]),
                "scenario_id": int(episode["scenario_id"]),
                "step": int(step["step"]),
                "bonus_age": float(step["bonus_age"]),
                "bonus_tag": float(step["bonus_tag"]),
                "base_reward": float(step["base_reward"]),
                "payload_term": float(step["payload_term"]),
                "motion_penalty": float(step["motion_penalty"]),
                "reward": float(step["reward"]),
                "reward_float32": float(np.float32(step["reward"])),
                "safety_violation": int(step["safety_violation"]),
                "distance_m": float(step["distance_after_m"]),
                "projection_status": step["projection_status"],
                "sensitivity_mu_gap": "" if not isinstance(item, dict) else float(item["mu_gap"]),
                "sensitivity_executed_gap": "" if not isinstance(item, dict) else float(item["executed_gap"]),
                "sensitivity_aligned": (
                    ""
                    if not isinstance(item, dict)
                    else bool(float(item["worker_mu_gap"]) <= 1.0e-5 and float(item["worker_action_gap"]) <= 1.0e-5)
                ),
            }
            priority = np.asarray(step["priority"], dtype=np.float64)
            association = np.asarray(step["association"], dtype=np.int64)
            mask = np.asarray(step["policy_active_mask"], dtype=np.float64)
            pre = np.asarray(step["pre_tanh"], dtype=np.float64)
            desired = np.asarray(step["desired"], dtype=np.float64)
            action = np.asarray(step["action"], dtype=np.float64)
            for uav in range(int(priority.shape[0])):
                rows.append(
                    {
                        **common,
                        "grain": "uav",
                        "index": int(uav),
                        "feature_age": float(priority[uav, 0]),
                        "feature_tag": float(priority[uav, 1]),
                        "feature_valid": float(priority[uav, 2]),
                        "policy_active": float(mask[uav]),
                        "assigned_buoy": int(env_target(int(association[uav]), len(step["before_buckets"]))),
                        "pre_x": float(pre[uav, 0]),
                        "pre_y": float(pre[uav, 1]),
                        "desired_x": float(desired[uav, 0]),
                        "desired_y": float(desired[uav, 1]),
                        "action_x": float(action[uav, 0]),
                        "action_y": float(action[uav, 1]),
                    }
                )
            generated = np.asarray(step["generated"], dtype=np.float64)
            dropped = np.asarray(step["dropped"], dtype=np.float64)
            served_tagged = np.asarray(step["served_tagged_by_buoy"], dtype=np.float64)
            served_ordinary = np.asarray(step["served_ordinary_by_buoy"], dtype=np.float64)
            for buoy, before in enumerate(step["before_buckets"]):
                after = step["after_buckets"][buoy]
                before_head = head_identity(before)
                after_head = head_identity(after)
                head = None if before_head is None else min(before, key=fifo_key_from)
                tag = step["arrival_tags"][buoy]
                rows.append(
                    {
                        **common,
                        "grain": "buoy",
                        "index": int(buoy),
                        "head_tag": "" if head is None else int(head["priority_tag"]),
                        "head_bits": "" if head is None else float(head["bits_remaining"]),
                        "generated_bits": float(generated[buoy]),
                        "generated_tag": "" if tag is None else int(tag),
                        "dropped_bits": float(dropped[buoy]),
                        "served_tagged_bits": float(served_tagged[buoy]),
                        "served_ordinary_bits": float(served_ordinary[buoy]),
                        "became_head": int(before_head != after_head and after_head is not None),
                    }
                )
    return rows


STEP_COLUMNS = [
    "variant",
    "model_seed",
    "scenario_id",
    "step",
    "grain",
    "index",
    "feature_age",
    "feature_tag",
    "feature_valid",
    "policy_active",
    "assigned_buoy",
    "pre_x",
    "pre_y",
    "desired_x",
    "desired_y",
    "action_x",
    "action_y",
    "head_tag",
    "head_bits",
    "generated_bits",
    "generated_tag",
    "dropped_bits",
    "served_tagged_bits",
    "served_ordinary_bits",
    "became_head",
    "bonus_age",
    "bonus_tag",
    "base_reward",
    "payload_term",
    "motion_penalty",
    "reward",
    "reward_float32",
    "safety_violation",
    "distance_m",
    "projection_status",
    "sensitivity_mu_gap",
    "sensitivity_executed_gap",
    "sensitivity_aligned",
]


def sensitivity_summary(episodes: list[dict]) -> dict:
    rows = []
    for episode in episodes:
        if not episode["steps"][0]["actor_accepts_tag"]:
            rows.append(
                {
                    "variant": episode["variant"],
                    "model_seed": int(episode["model_seed"]),
                    "scenario_id": int(episode["scenario_id"]),
                    "actor_accepts_tag": False,
                    "note": "L0 uses ResidualActor. No tag column was zeroed.",
                }
            )
            continue
        usable = []
        aligned = 0
        failed = 0
        for step in episode["steps"]:
            item = step["sensitivity"]
            if item is None:
                continue
            step_aligned = bool(float(item["worker_mu_gap"]) <= 1.0e-5 and float(item["worker_action_gap"]) <= 1.0e-5)
            if step_aligned:
                aligned += 1
            else:
                failed += 1
            if item["tag_input_nonzero"] and step_aligned:
                usable.append(item)
        rows.append(
            {
                "variant": episode["variant"],
                "model_seed": int(episode["model_seed"]),
                "scenario_id": int(episode["scenario_id"]),
                "actor_accepts_tag": True,
                "steps": len(episode["steps"]),
                "aligned_steps": aligned,
                "alignment_failed_steps": failed,
                "aligned_nonzero_tag_input_steps": len(usable),
                "mu_changed_steps": int(sum(float(item["mu_gap"]) > 0.0 for item in usable)),
                "executed_changed_steps": int(sum(float(item["executed_gap"]) > 0.0 for item in usable)),
                "median_mu_gap": None if not usable else float(np.median([item["mu_gap"] for item in usable])),
                "max_mu_gap": None if not usable else float(np.max([item["mu_gap"] for item in usable])),
                "median_executed_gap": None if not usable else float(np.median([item["executed_gap"] for item in usable])),
            }
        )
    return {"episodes": rows}
