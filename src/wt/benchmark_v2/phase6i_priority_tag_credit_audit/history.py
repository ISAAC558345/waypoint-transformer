"""Read the frozen E12 logs. This module does not roll out or train."""
from __future__ import annotations

import csv
import json

import numpy as np
import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import DEV_POOL, PPO
from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.criteria import choose_gate, score_l4
from wt.benchmark_v2.phase6i_priority_tag_reward.smoke import file_sha256

from .protocol import (
    BIT_ABS,
    E12_GATE,
    E12_IMPLEMENTATION_COMMIT,
    E12_OUTPUT,
    E12_RESULT_COMMIT,
    E12_ROWS,
    FLOAT32_ABS,
    LOOKAHEAD_ABS,
    REWARD_ABS,
    ROOT,
    SAMPLE_ENV_INDEX,
    SAMPLE_STEPS,
    SAMPLE_UPDATES,
    SAMPLES_PER_GROUP,
    SHARED_SOURCES,
    TRAIN_SEEDS,
    UPDATES,
    VARIANTS,
    WINDOWS,
)


SUM_KEYS = (
    "uav_steps",
    "valid_association_uav_steps",
    "tagged_backlog_valid_uav_steps",
    "ordinary_head_tagged_backlog_uav_steps",
    "lookahead_nonzero_input_uav_steps",
    "positive_tag_bonus_steps",
    "float32_distinguishable_tag_steps",
)
MEAN_KEYS = (
    "mean_served_bits",
    "mean_served_tagged_bits",
    "mean_bonus_age",
    "mean_bonus_tag",
    "mean_motion_penalty",
    "advantage_mean",
    "advantage_std",
    "approx_kl",
    "clip_fraction",
    "value_loss",
    "explained_variance",
    "gradient_norm_preclip",
    "action_std",
    "projection_rate",
    "safety_violation_rate",
)
REQUIRED_LOG_KEYS = ("variant", "model_seed", "update", "policy_loss") + SUM_KEYS + MEAN_KEYS
GATE_FIELDS = (
    "served_tagged_bits",
    "payload_bits",
    "sinr_mean",
    "interference_mean",
    "safety_violations",
    "constraint_failures",
)


def _finite(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number))


def load_scenario_rows(path=E12_ROWS) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    parsed = []
    for row in rows:
        item = {
            "variant": str(row["variant"]),
            "model_seed": int(row["model_seed"]),
            "scenario_id": int(row["scenario_id"]),
        }
        for key in GATE_FIELDS:
            item[key] = float(row[key])
        parsed.append(item)
    return parsed


def _pair_ok(rows: list[dict]) -> dict:
    keys = [(row["variant"], int(row["model_seed"]), int(row["scenario_id"])) for row in rows]
    expected = []
    expected.extend(("L0", 0, int(scenario)) for scenario in DEV_POOL)
    for seed in TRAIN_SEEDS:
        for variant in ("L3", "L4_C", "L4_Q"):
            expected.extend((variant, int(seed), int(scenario)) for scenario in DEV_POOL)
    return {
        "rows": len(rows),
        "unique_keys": len(set(keys)),
        "duplicate_keys": int(len(keys) - len(set(keys))),
        "expected_keys": len(expected),
        "missing_keys": int(len(set(expected) - set(keys))),
        "unexpected_keys": int(len(set(keys) - set(expected))),
        "development_min": int(min(DEV_POOL)),
        "development_max": int(max(DEV_POOL)),
        "aligned": bool(set(keys) == set(expected) and len(keys) == len(set(keys))),
    }


def recompute_gate(rows: list[dict]) -> dict:
    score = score_l4(rows)
    gate = choose_gate(tests_ok=True, train_ok=True, hashes_ok=True, feasibility="pass", score=score)
    published = json.loads((E12_OUTPUT / "summary.json").read_text(encoding="utf-8"))
    means = []
    for item, saved in zip(score["seeds"], published["score"]["seeds"]):
        if int(item["model_seed"]) != int(saved["model_seed"]):
            raise RuntimeError("published E12 seed order does not match the recomputed score")
        means.append(
            {
                "model_seed": int(item["model_seed"]),
                "versus_l4_c_tagged_mean": float(item["versus_l4_c"]["tagged"]["mean"]),
                "versus_l4_c_ci_low": float(item["versus_l4_c"]["tagged"]["ci_low"]),
                "versus_l4_c_ci_high": float(item["versus_l4_c"]["tagged"]["ci_high"]),
                "versus_l0_tagged_mean": float(item["versus_l0"]["tagged"]["mean"]),
                "versus_l0_safety_diff_sum": float(item["versus_l0"]["safety_diff_sum"]),
                "versus_l4_c_safety_diff_sum": float(item["versus_l4_c"]["safety_diff_sum"]),
                "versus_l0_constraint_diff_sum": float(item["versus_l0"]["constraint_diff_sum"]),
                "versus_l4_c_constraint_diff_sum": float(item["versus_l4_c"]["constraint_diff_sum"]),
                "versus_l0_interference_mean": float(item["versus_l0"]["interference_mean"]),
                "published_tagged_mean": float(saved["versus_l4_c"]["tagged"]["mean"]),
                "mean_gap": float(item["versus_l4_c"]["tagged"]["mean"]) - float(saved["versus_l4_c"]["tagged"]["mean"]),
            }
        )
    return {
        "gate": gate,
        "matches_published_fail": bool(gate == E12_GATE == published["scientific_gate"]),
        "passing_seeds": int(score["passing_seeds"]),
        "published_passing_seeds": int(published["score"]["passing_seeds"]),
        "means": means,
        "negative_tagged_versus_l0": int(sum(float(item["versus_l0"]["tagged"]["mean"]) < 0.0 for item in score["seeds"])),
        "all_versus_l4_c_intervals_cross_zero": bool(
            all(float(item["versus_l4_c"]["tagged"]["ci_low"]) <= 0.0 <= float(item["versus_l4_c"]["tagged"]["ci_high"]) for item in score["seeds"])
        ),
    }


def _window_rows(rows: list[dict]) -> list[dict]:
    windows = list(WINDOWS) + [(1, int(UPDATES))]
    output = []
    for start, stop in windows:
        selected = [row for row in rows if start <= int(row["update"]) <= stop]
        item = {
            "variant": rows[0]["variant"],
            "model_seed": int(rows[0]["model_seed"]),
            "window": f"{int(start)}-{int(stop)}",
            "update_count": len(selected),
        }
        for key in SUM_KEYS:
            item[key] = int(sum(int(row[key]) for row in selected))
        for key in MEAN_KEYS:
            item[key] = float(np.mean([float(row[key]) for row in selected]))
        denominator = float(item["mean_served_bits"])
        item["ratio_kind"] = "ratio_of_two_means"
        item["ratio_of_means_bonus_tag_over_served_bits"] = None if denominator == 0.0 else float(item["mean_bonus_tag"]) / denominator
        item["ratio_denominator_is_zero"] = bool(denominator == 0.0)
        item["mean_weighting"] = "each update has the same env-step count, so the mean of per-update means equals the pooled mean"
        output.append(item)
    return output


def _audit_log(variant: str, seed: int) -> tuple[list[dict], list[str]]:
    path = E12_OUTPUT / "training_logs" / f"{variant.lower()}_seed{int(seed)}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    gaps = []
    if len(rows) != int(UPDATES):
        gaps.append(f"{variant} seed {seed} has {len(rows)} log rows")
    updates = [int(row["update"]) for row in rows]
    if updates != list(range(1, int(UPDATES) + 1)):
        gaps.append(f"{variant} seed {seed} update index is missing or duplicated")
    for row in rows:
        missing = [key for key in REQUIRED_LOG_KEYS if key not in row]
        if missing:
            gaps.append(f"{variant} seed {seed} update {row.get('update')} is missing {missing}")
            break
        if str(row["variant"]) != variant or int(row["model_seed"]) != int(seed):
            gaps.append(f"{variant} seed {seed} log identity drifted")
            break
        if any(not _finite(row[key]) for key in MEAN_KEYS) or not _finite(row["policy_loss"]):
            gaps.append(f"{variant} seed {seed} update {row['update']} has a nonfinite value")
            break
    return rows, gaps


def _audit_samples(variant: str, seed: int) -> tuple[dict, list[str]]:
    path = E12_OUTPUT / "training_samples" / f"{variant.lower()}_seed{int(seed)}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    gaps = []
    grouped: dict[int, list[dict]] = {int(update): [] for update in SAMPLE_UPDATES}
    identity = 0.0
    cap_violations = 0
    float_gap = 0.0
    lookahead_gap = 0.0
    zero_reward = 0
    positive = 0
    ratios = []
    zero_denominator = 0
    for row in rows:
        update = int(row["update"])
        if update not in grouped or int(row["env_index"]) != int(SAMPLE_ENV_INDEX):
            gaps.append(f"{variant} seed {seed} sample is outside update/env 0")
            continue
        grouped[update].append(row)
        reward = float(row["reward"])
        base = float(row["base_reward"])
        age = float(row["bonus_age"])
        tag = float(row["bonus_tag"])
        served = float(row["served_bits"])
        identity = max(identity, abs(reward - (base + age + tag)))
        identity = max(identity, abs(float(row["motion_penalty"]) - (served / 1.0e6 - base)))
        payload = served / 1.0e6
        tag_cap = 0.0010 * float(row["served_tagged_bits"]) / 1.0e6
        if served == 0.0 and (age != 0.0 or tag != 0.0):
            cap_violations += 1
        if age < -1.0e-12 or tag < -1.0e-12 or age - 0.00025 * payload > 1.0e-12 or tag - min(0.00075 * payload, tag_cap) > 1.0e-12:
            cap_violations += 1
        float_gap = max(float_gap, abs(float(np.float32(reward)) - float(row["reward_float32"])))
        lookahead = np.asarray(row["lookahead_input"], dtype=np.float64)
        if variant == "L4_C":
            lookahead_gap = max(lookahead_gap, float(np.max(np.abs(lookahead))))
        if reward == 0.0:
            zero_reward += 1
            zero_denominator += 1
        else:
            ratios.append(tag / reward)
        if tag > 0.0:
            positive += 1
    for update, items in grouped.items():
        steps = [int(item["step"]) for item in items]
        if steps != list(range(int(SAMPLE_STEPS))):
            gaps.append(f"{variant} seed {seed} update {update} does not have steps 0-{int(SAMPLE_STEPS) - 1}")
    coverage_env0 = float(len(rows)) / float(int(UPDATES) * int(SAMPLE_STEPS))
    coverage_all = float(len(rows)) / float(int(UPDATES) * int(PPO.rollout_length) * int(PPO.num_envs))
    return {
        "variant": variant,
        "model_seed": int(seed),
        "rows": len(rows),
        "expected_rows": int(SAMPLES_PER_GROUP),
        "reward_identity_max_abs": float(identity),
        "cap_violations": int(cap_violations),
        "float32_max_abs": float(float_gap),
        "lookahead_input_max_abs": float(lookahead_gap),
        "zero_reward_steps": int(zero_reward),
        "positive_tag_steps": int(positive),
        "per_step_ratio_kind": "saved_sample_tag_bonus_over_reward",
        "per_step_ratio_mean": None if not ratios else float(np.mean(ratios)),
        "per_step_ratio_count": len(ratios),
        "zero_reward_denominator_count": int(zero_denominator),
        "coverage_of_env0_steps": coverage_env0,
        "coverage_of_all_env_steps": coverage_all,
        "coverage_note": "these 144 rows are env 0 at updates 1, 150, and 300 only",
    }, gaps


def _audit_checkpoints() -> tuple[dict, list[str]]:
    gaps = []
    rows = []
    summary = json.loads((E12_OUTPUT / "summary.json").read_text(encoding="utf-8"))
    published = {(item["variant"], int(item["model_seed"])): item for item in summary["training"]}
    for variant in VARIANTS:
        for seed in TRAIN_SEEDS:
            path = E12_OUTPUT / "checkpoints" / f"{variant.lower()}_seed{int(seed)}.pt"
            blob = torch.load(path, map_location="cpu", weights_only=False)
            record = published[(variant, int(seed))]
            item = {
                "variant": variant,
                "model_seed": int(seed),
                "sha256": file_sha256(path),
                "published_sha256": record["sha256"],
                "actor_init_sha256": blob["actor_init_sha256"],
                "critic_init_sha256": blob["critic_init_sha256"],
                "implementation_commit": blob["implementation_commit"],
                "updates": int(blob["updates"]),
                "use_lookahead": bool(blob["use_lookahead"]),
            }
            if item["sha256"] != item["published_sha256"]:
                gaps.append(f"{variant} seed {seed} checkpoint hash drifted from the E12 summary")
            if item["implementation_commit"] != E12_IMPLEMENTATION_COMMIT:
                gaps.append(f"{variant} seed {seed} implementation commit is not the E12 implementation")
            if int(item["updates"]) != int(UPDATES):
                gaps.append(f"{variant} seed {seed} checkpoint update count drifted")
            if bool(item["use_lookahead"]) != (variant == "L4_Q"):
                gaps.append(f"{variant} seed {seed} lookahead flag drifted")
            rows.append(item)
            del blob
    actor_inits = {row["actor_init_sha256"] for row in rows}
    critic_by_seed = {}
    for row in rows:
        critic_by_seed.setdefault(int(row["model_seed"]), set()).add(row["critic_init_sha256"])
    return {
        "checkpoints": rows,
        "actor_init_unique": len(actor_inits),
        "actor_init_identical_across_seeds": len(actor_inits) == 1,
        "critic_init_matches_within_seed": all(len(values) == 1 for values in critic_by_seed.values()),
        "critic_init_differs_across_seeds": len({row["critic_init_sha256"] for row in rows}) == len(TRAIN_SEEDS),
        "result_commit": E12_RESULT_COMMIT,
    }, gaps


def _sample_seed_effect() -> dict:
    rewards = {}
    for variant in VARIANTS:
        for seed in TRAIN_SEEDS:
            rows = json.loads((E12_OUTPUT / "training_samples" / f"{variant.lower()}_seed{int(seed)}.json").read_text(encoding="utf-8"))
            rewards[(variant, int(seed))] = [float(row["reward"]) for row in rows if int(row["update"]) == 1]
    gaps = {}
    for variant in VARIANTS:
        left = np.asarray(rewards[(variant, 0)], dtype=np.float64)
        right = np.asarray(rewards[(variant, 1)], dtype=np.float64)
        gaps[variant] = float(np.max(np.abs(left - right)))
    return {
        "update1_env0_reward_max_abs_seed0_vs_seed1": gaps,
        "note": "a nonzero gap shows the recorded first rollout differed by seed; identical actor-init hashes are the legacy transplant",
    }


def shared_source_hashes() -> dict[str, str]:
    hashes = {}
    for relative in SHARED_SOURCES:
        path = ROOT / relative
        hashes[relative] = file_sha256(path)
    package = ROOT / "src" / "wt" / "benchmark_v2" / "phase6i_priority_tag_queue_lookahead"
    for path in sorted(package.glob("*.py")):
        hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = file_sha256(path)
    return hashes


def audit_history() -> dict:
    rows = load_scenario_rows()
    pairing = _pair_ok(rows)
    gate = recompute_gate(rows)
    gaps = []
    if not pairing["aligned"] or pairing["rows"] != 320:
        gaps.append("E12 scenario CSV pairing is not the registered 320-row development pool")
    if not gate["matches_published_fail"] or int(gate["passing_seeds"]) != 0:
        gaps.append("recomputed E12 gate is not QUEUE_LOOKAHEAD_BENCHMARK_FAIL with zero passing seeds")
    if any(abs(float(item["mean_gap"])) > 1.0e-6 for item in gate["means"]):
        gaps.append("recomputed tagged means do not match the frozen E12 summary")
    logs = []
    samples = []
    for variant in VARIANTS:
        for seed in TRAIN_SEEDS:
            log_rows, log_gaps = _audit_log(variant, seed)
            gaps.extend(log_gaps)
            if log_rows and not log_gaps:
                logs.extend(_window_rows(log_rows))
            sample, sample_gaps = _audit_samples(variant, seed)
            gaps.extend(sample_gaps)
            if int(sample["rows"]) != int(SAMPLES_PER_GROUP):
                gaps.append(f"{variant} seed {seed} does not have 144 saved samples")
            if float(sample["reward_identity_max_abs"]) > float(REWARD_ABS):
                gaps.append(f"{variant} seed {seed} reward identity exceeds the locked tolerance")
            if int(sample["cap_violations"]) != 0:
                gaps.append(f"{variant} seed {seed} sample bonus cap failed")
            if float(sample["float32_max_abs"]) > float(FLOAT32_ABS):
                gaps.append(f"{variant} seed {seed} float32 storage check failed")
            if variant == "L4_C" and float(sample["lookahead_input_max_abs"]) > float(LOOKAHEAD_ABS):
                gaps.append(f"{variant} seed {seed} control lookahead input is nonzero")
            samples.append(sample)
    checkpoints, checkpoint_gaps = _audit_checkpoints()
    gaps.extend(checkpoint_gaps)
    exposure = [row for row in logs if row["window"] == f"1-{int(UPDATES)}"]
    totals = {
        "positive_tag_bonus_steps": int(sum(int(row["positive_tag_bonus_steps"]) for row in exposure)),
        "float32_distinguishable_tag_steps": int(sum(int(row["float32_distinguishable_tag_steps"]) for row in exposure)),
        "l4_c_nonzero_input_uav_steps": int(sum(int(row["lookahead_nonzero_input_uav_steps"]) for row in exposure if row["variant"] == "L4_C")),
        "l4_q_nonzero_input_uav_steps": int(sum(int(row["lookahead_nonzero_input_uav_steps"]) for row in exposure if row["variant"] == "L4_Q")),
        "l4_q_tagged_backlog_valid_uav_steps": int(sum(int(row["tagged_backlog_valid_uav_steps"]) for row in exposure if row["variant"] == "L4_Q")),
        "groups": len(exposure),
        "updates": int(UPDATES) * len(exposure),
    }
    return {
        "ok": not gaps,
        "gaps": gaps,
        "pairing": pairing,
        "gate": gate,
        "log_rows": logs,
        "sample_rows": samples,
        "checkpoints": checkpoints,
        "seed_effect": _sample_seed_effect(),
        "exposure_totals": totals,
        "bit_tolerance": float(BIT_ABS),
        "source_hashes": shared_source_hashes(),
    }
