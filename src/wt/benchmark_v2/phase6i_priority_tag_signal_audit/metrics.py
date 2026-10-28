"""Exposure, offline GAE, and training-log summaries. No optimizer step."""
from __future__ import annotations

import json

import numpy as np

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import PPO
from wt.benchmark_v2.phase6i_priority_tag.protocol import DUST_BITS
from wt.lowlevel_v2.contracts import env_target
from wt.round30.ppo import compute_gae

from .protocol import E10_OUTPUT, TAG_COLUMN
from .trace import fifo_key_from, head_identity


def _drain(buckets: list[dict], amount: float) -> list[dict]:
    remaining = float(amount)
    touched = []
    for bucket in sorted(buckets, key=fifo_key_from):
        if remaining <= float(DUST_BITS):
            break
        take = min(float(bucket["bits_remaining"]), remaining)
        if take <= float(DUST_BITS):
            continue
        touched.append({**bucket, "taken": float(take)})
        remaining -= take
    return touched


def exposure(episodes: list[dict]) -> dict:
    """Numerators and denominators are counted separately. The 0.25 draw rate is not reused."""
    summaries = []
    events = []
    for episode in episodes:
        visible_uav = 0
        valid_uav = 0
        active_visible = 0
        uav_steps = 0
        head_tagged = 0
        occupied = 0
        service_steps = 0
        generated_tagged = 0
        queued_tagged = 0
        tracked: dict[tuple[int, int, int], dict] = {}
        mask_values = []
        for step in episode["steps"]:
            priority = np.asarray(step["priority"], dtype=np.float64)
            association = np.asarray(step["association"], dtype=np.int64)
            mask = np.asarray(step["policy_active_mask"], dtype=np.float64)
            num_buoys = len(step["before_buckets"])
            uav_steps += int(priority.shape[0])
            mask_values.extend(float(value) for value in mask)
            assigned = {}
            for uav, token in enumerate(association):
                buoy = int(env_target(int(token), int(num_buoys)))
                assigned.setdefault(buoy, []).append(int(uav))
                if float(priority[uav, 2]) > 0.0:
                    valid_uav += 1
                if float(priority[uav, int(TAG_COLUMN)]) > 0.0 and float(priority[uav, 2]) > 0.0:
                    visible_uav += 1
                    if float(mask[uav]) > 0.0:
                        active_visible += 1
            for buoy, buckets in enumerate(step["before_buckets"]):
                if buckets:
                    occupied += 1
                identity = head_identity(buckets)
                if identity is None:
                    continue
                head = min(buckets, key=fifo_key_from)
                if int(head["priority_tag"]) == 1:
                    head_tagged += 1
                    key = (int(buoy), int(identity[0]), int(identity[1]))
                    if key in tracked and tracked[key]["visible_step"] is None and buoy in assigned:
                        tracked[key]["visible_step"] = int(step["step"])
            served = np.asarray(step["served_tagged_by_buoy"], dtype=np.float64) + np.asarray(step["served_ordinary_by_buoy"], dtype=np.float64)
            for buoy, buckets in enumerate(step["before_buckets"]):
                for touched in _drain(buckets, float(served[buoy])):
                    if int(touched["priority_tag"]) != 1:
                        continue
                    key = (int(buoy), int(touched["admission_step"]), int(touched["insert_index"]))
                    if key in tracked and tracked[key]["served_step"] is None:
                        tracked[key]["served_step"] = int(step["step"])
            if float(step["served_tagged_bits"]) > float(DUST_BITS):
                service_steps += 1
            tags = step["arrival_tags"]
            for buoy, (bits, tag) in enumerate(zip(np.asarray(step["generated"], dtype=np.float64), tags)):
                if tag != 1 or float(bits) <= float(DUST_BITS):
                    continue
                generated_tagged += 1
                created = [
                    bucket
                    for bucket in step["after_buckets"][buoy]
                    if int(bucket["admission_step"]) == int(step["step"]) and int(bucket["priority_tag"]) == 1
                ]
                if not created:
                    events.append(
                        {
                            "variant": episode["variant"],
                            "model_seed": int(episode["model_seed"]),
                            "scenario_id": int(episode["scenario_id"]),
                            "buoy": int(buoy),
                            "admission_step": int(step["step"]),
                            "queued": False,
                            "visible_step": "",
                            "served_step": "",
                        }
                    )
                    continue
                queued_tagged += 1
                bucket = created[0]
                key = (int(buoy), int(bucket["admission_step"]), int(bucket["insert_index"]))
                tracked[key] = {"visible_step": None, "served_step": None, "admission_step": int(step["step"])}
        visible_delays = []
        served_delays = []
        never_visible = 0
        never_served = 0
        for key, item in tracked.items():
            row = {
                "variant": episode["variant"],
                "model_seed": int(episode["model_seed"]),
                "scenario_id": int(episode["scenario_id"]),
                "buoy": int(key[0]),
                "admission_step": int(item["admission_step"]),
                "queued": True,
                "visible_step": "" if item["visible_step"] is None else int(item["visible_step"]),
                "served_step": "" if item["served_step"] is None else int(item["served_step"]),
            }
            events.append(row)
            if item["visible_step"] is None:
                never_visible += 1
            else:
                visible_delays.append(int(item["visible_step"]) - int(item["admission_step"]))
            if item["served_step"] is None:
                never_served += 1
            else:
                served_delays.append(int(item["served_step"]) - int(item["admission_step"]))
        summaries.append(
            {
                "variant": episode["variant"],
                "model_seed": int(episode["model_seed"]),
                "scenario_id": int(episode["scenario_id"]),
                "uav_steps": uav_steps,
                "valid_assigned_uav_steps": valid_uav,
                "visible_tagged_uav_steps": visible_uav,
                "visible_and_policy_active_uav_steps": active_visible,
                "policy_active_mask_min": float(min(mask_values) if mask_values else 0.0),
                "policy_active_mask_max": float(max(mask_values) if mask_values else 0.0),
                "occupied_buoy_steps": occupied,
                "head_tagged_buoy_steps": head_tagged,
                "generated_tagged_buckets": generated_tagged,
                "queued_tagged_buckets": queued_tagged,
                "positive_tag_service_steps": service_steps,
                "steps": len(episode["steps"]),
                "visible_delay_observed": len(visible_delays),
                "visible_delay_median": None if not visible_delays else float(np.median(np.asarray(visible_delays))),
                "never_visible_queued_buckets": never_visible,
                "served_delay_observed": len(served_delays),
                "served_delay_median": None if not served_delays else float(np.median(np.asarray(served_delays))),
                "never_served_queued_buckets": never_served,
                "served_tagged_bits": float(sum(step["served_tagged_bits"] for step in episode["steps"])),
                "bonus_tag_sum": float(sum(step["bonus_tag"] for step in episode["steps"])),
            }
        )
    return {"episodes": summaries, "events": events}


def _ratios(episodes: list[dict]) -> dict:
    groups = []
    for episode in episodes:
        if episode["variant"] != "L3":
            continue
        age_payload = []
        tag_payload = []
        age_motion = []
        tag_motion = []
        zero_payload = 0
        zero_motion = 0
        distinguishable = 0
        positive_tag = 0
        for step in episode["steps"]:
            payload = float(step["payload_term"])
            motion = float(step["motion_penalty"])
            age = float(step["bonus_age"])
            tag = float(step["bonus_tag"])
            if payload == 0.0:
                zero_payload += 1
            else:
                age_payload.append(age / payload)
                tag_payload.append(tag / payload)
            if motion == 0.0:
                zero_motion += 1
            else:
                age_motion.append(age / motion)
                tag_motion.append(tag / motion)
            if tag > 0.0:
                positive_tag += 1
                stored_without = np.float32(float(step["base_reward"]) + age)
                stored_with = np.float32(float(step["base_reward"]) + age + tag)
                if float(stored_with) != float(stored_without):
                    distinguishable += 1
        groups.append(
            {
                "variant": "L3",
                "model_seed": int(episode["model_seed"]),
                "scenario_id": int(episode["scenario_id"]),
                "steps": len(episode["steps"]),
                "zero_payload_steps": zero_payload,
                "zero_motion_steps": zero_motion,
                "positive_tag_bonus_steps": positive_tag,
                "float32_distinguishable_tag_steps": distinguishable,
                "age_over_payload_median": None if not age_payload else float(np.median(age_payload)),
                "tag_over_payload_median": None if not tag_payload else float(np.median(tag_payload)),
                "age_over_motion_median": None if not age_motion else float(np.median(age_motion)),
                "tag_over_motion_median": None if not tag_motion else float(np.median(tag_motion)),
                "bonus_age_sum": float(sum(step["bonus_age"] for step in episode["steps"])),
                "bonus_tag_sum": float(sum(step["bonus_tag"] for step in episode["steps"])),
                "payload_term_sum": float(sum(step["payload_term"] for step in episode["steps"])),
                "motion_penalty_sum": float(sum(step["motion_penalty"] for step in episode["steps"])),
            }
        )
    return {"episodes": groups, "label": "diagnostic_estimate"}


def _normalize(advantages: np.ndarray) -> np.ndarray:
    values = np.asarray(advantages, dtype=np.float64).reshape(-1)
    return (values - float(values.mean())) / max(float(values.std()), 1.0e-6)


def gae_compare(episodes: list[dict]) -> dict:
    rows = []
    for episode in episodes:
        if episode["variant"] != "L3":
            rows.append(
                {
                    "variant": episode["variant"],
                    "model_seed": int(episode["model_seed"]),
                    "scenario_id": int(episode["scenario_id"]),
                    "computed": False,
                    "reason": "L0 has no experiment critic; its value trace is not used as an L3 baseline",
                }
            )
            continue
        steps = episode["steps"]
        base = np.asarray([step["base_reward"] for step in steps], dtype=np.float32).reshape(-1, 1)
        age = np.asarray([step["bonus_age"] for step in steps], dtype=np.float32).reshape(-1, 1)
        tag = np.asarray([step["bonus_tag"] for step in steps], dtype=np.float32).reshape(-1, 1)
        values = np.asarray([step["value"] for step in steps], dtype=np.float32).reshape(-1, 1)
        dones = np.asarray([1.0 if step["done"] else 0.0 for step in steps], dtype=np.float32).reshape(-1, 1)
        last = np.zeros((1,), dtype=np.float32)
        advantages = {}
        for name, reward in (("base", base), ("base_age", base + age), ("full", base + age + tag)):
            advantage, _returns = compute_gae(reward, values, dones, last, float(PPO.gamma), float(PPO.gae_lambda))
            advantages[name] = advantage.reshape(-1)
        full_norm = _normalize(advantages["full"])
        age_norm = _normalize(advantages["base_age"])
        base_norm = _normalize(advantages["base"])
        rows.append(
            {
                "variant": "L3",
                "model_seed": int(episode["model_seed"]),
                "scenario_id": int(episode["scenario_id"]),
                "computed": True,
                "label": "diagnostic_estimate",
                "gamma": float(PPO.gamma),
                "gae_lambda": float(PPO.gae_lambda),
                "terminal_bootstrap": "last_value is zero because the recorded final done flag is the episode boundary",
                "mean_abs_normalized_tag_effect": float(np.mean(np.abs(full_norm - age_norm))),
                "max_abs_normalized_tag_effect": float(np.max(np.abs(full_norm - age_norm))),
                "mean_abs_normalized_age_effect": float(np.mean(np.abs(age_norm - base_norm))),
                "mean_abs_raw_tag_effect": float(np.mean(np.abs(advantages["full"] - advantages["base_age"]))),
                "note": "The critic values stay on the frozen full-reward trajectory. This is not the advantage from the 300 training updates.",
            }
        )
    return {"episodes": rows}


def training_logs() -> dict:
    rows = []
    keys = (
        "approx_kl",
        "clip_fraction",
        "value_loss",
        "explained_variance",
        "action_std",
        "projection_rate",
        "gradient_norm_preclip",
        "safety_violation_rate",
    )
    for seed in (0, 1, 2):
        path = E10_OUTPUT / "training_logs" / f"l3_seed{int(seed)}.json"
        history = json.loads(path.read_text(encoding="utf-8"))
        summary = {"model_seed": int(seed), "updates": len(history), "missing": []}
        for key in keys:
            if key not in history[0]:
                summary["missing"].append(key)
                continue
            values = np.asarray([float(item[key]) for item in history], dtype=np.float64)
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_max"] = float(np.max(values))
        summary["mean_bonus_only"] = "mean_bonus" in history[0]
        summary["step_bonus_tag_recorded"] = "bonus_tag" in history[0]
        rows.append(summary)
    return {
        "seeds": rows,
        "historical_step_bonus_tag": "missing",
        "large_preclip_gradient_is_not_instability": True,
    }
