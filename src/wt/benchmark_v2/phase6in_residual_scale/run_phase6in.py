"""Scan residual scales on the frozen actor. Training starts only after a supported gain."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info, make_option
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import load_primary
from wt.benchmark_v2.phase6iaa_formal_planner.planner import FormalPlanner

from .action import assert_official_match, scale_limits
from .evaluate import evaluate_scale, paired_summary
from .protocol import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CANDIDATE_ALPHAS,
    CLEAR_LOSS_BITS,
    EVAL_POOL,
    OFFICIAL_ALPHA,
    OUTPUT,
    SAFETY_MARGIN,
    SCREEN_POOL,
    SOURCE_ACTOR,
    assert_phase_output,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows])) if rows else 0.0


def _stop_reason(rows: list[dict], baseline: list[dict]) -> str:
    if any(int(row["anomaly_count"]) > 0 or not row["complete"] for row in rows):
        return "blocked"
    if len(rows) < 2:
        return ""
    left = {int(row["scenario_id"]): row for row in baseline}
    shared = [row for row in rows if int(row["scenario_id"]) in left]
    if len(shared) < 2:
        return ""
    safety = float(np.mean([row["safety_violation_rate"] for row in shared]))
    base_safety = float(np.mean([left[int(row["scenario_id"])]["safety_violation_rate"] for row in shared]))
    if safety > base_safety + float(SAFETY_MARGIN):
        return "stopped_safety"
    if len(shared) < 4:
        return ""
    delta = np.asarray(
        [row["payload_bits"] - left[int(row["scenario_id"])]["payload_bits"] for row in shared],
        dtype=np.float64,
    )
    if float(delta.mean()) < -float(CLEAR_LOSS_BITS) and int(np.sum(delta < -1.0)) >= len(shared) - 1:
        return "stopped_payload"
    return ""


def _audit(actor, planner: FormalPlanner) -> dict:
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=int(SCREEN_POOL[0]))
    selection = planner.select_assignment(env, env.uav_positions.copy(), None)
    if selection.status != "selected" or selection.option is None:
        raise RuntimeError("mapping audit could not obtain a scheme-C assignment")
    option = make_option(env, np.asarray(selection.option.association, dtype=np.int64))
    last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    assert_official_match(env, actor, option, last)
    return {
        "execution_order": [
            "actor deterministic mean",
            "delta = alpha * tanh(mean)",
            "desired = clip_unit_step(fly_toward + delta)",
            "SafetyProjector",
            "zero the step on projector failure",
            "clip_unit_step(executed)",
            "env.step multiplies by step_length_m",
        ],
        "actor_output": "two unbounded world-frame residual logits; dimension 0 is +x and dimension 1 is +y",
        "alpha_role": "multiplies tanh(pre) before addition to fly_toward",
        "official_alpha": float(OFFICIAL_ALPHA),
        "phase6im_difference": "phase6im reinterpreted the same two outputs as a turn angle and a speed; this scan keeps the additive world-frame residual",
        "limits": {f"{alpha:.2f}": scale_limits(alpha) for alpha in CANDIDATE_ALPHAS},
        "official_match": True,
    }


def _aggregate(rows: list[dict]) -> dict:
    keys = [
        "payload_bits",
        "service_steps",
        "mean_first_entry_steps",
        "path_length_m",
        "sinr_mean",
        "interference_mean",
        "projection_rate",
        "correction_norm",
        "command_turn_deg",
        "executed_turn_deg",
        "heading_correction_deg",
        "speed_correction",
        "command_speed",
        "executed_speed",
        "residual_norm",
        "safety_violation_rate",
    ]
    return {key: _mean(rows, key) for key in keys} | {
        "scenarios": [int(row["scenario_id"]) for row in rows],
        "incomplete": int(sum(not row["complete"] for row in rows)),
    }


def _gain_supported(pair: dict) -> bool:
    concentrated = float(pair["top_positive_share"]) > 0.50 and float(pair["mean_without_best"]) <= 0.0
    return (
        float(pair["mean"]) > 0.0
        and float(pair["ci_low"]) > 0.0
        and float(pair["candidate_safety"]) <= float(pair["baseline_safety"]) + float(SAFETY_MARGIN)
        and int(pair["incomplete"]) == 0
        and not concentrated
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    before = _sha256(SOURCE_ACTOR)
    actor = load_primary()
    planner = FormalPlanner("scheme_c")
    planner.actor = actor
    audit = _audit(actor, planner)
    assert_phase_output(OUTPUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("official alpha 0.25 matches the frozen executor", flush=True)
    screen: dict[str, list[dict]] = {}
    status: dict[str, str] = {}
    baseline_rows = evaluate_scale(planner, float(OFFICIAL_ALPHA), SCREEN_POOL)
    screen[f"{OFFICIAL_ALPHA:.2f}"] = baseline_rows
    status[f"{OFFICIAL_ALPHA:.2f}"] = "screen_complete"
    for alpha in CANDIDATE_ALPHAS:
        if abs(float(alpha) - float(OFFICIAL_ALPHA)) < 1.0e-12:
            continue
        rows = []
        reason = ""
        for scenario in SCREEN_POOL:
            rows.append(evaluate_scale(planner, float(alpha), (int(scenario),))[0])
            reason = _stop_reason(rows, baseline_rows)
            if reason:
                print(f"alpha {alpha:.2f} {reason} after {len(rows)} screen scenarios", flush=True)
                break
        screen[f"{alpha:.2f}"] = rows
        status[f"{alpha:.2f}"] = reason or "screen_complete"
    full: dict[str, list[dict]] = {}
    promising = [
        alpha
        for alpha in CANDIDATE_ALPHAS
        if abs(float(alpha) - float(OFFICIAL_ALPHA)) > 1.0e-12 and status[f"{alpha:.2f}"] == "screen_complete"
    ]
    screen_pairs = {}
    for alpha in CANDIDATE_ALPHAS:
        if abs(float(alpha) - float(OFFICIAL_ALPHA)) < 1.0e-12:
            continue
        screen_pairs[f"{alpha:.2f}"] = paired_summary(
            baseline_rows,
            screen[f"{alpha:.2f}"],
            draws=int(BOOTSTRAP_DRAWS),
            seed=int(BOOTSTRAP_SEED),
        )
        pair = screen_pairs[f"{alpha:.2f}"]
        clear_loss = status[f"{alpha:.2f}"] == "screen_complete" and float(pair["mean"]) <= -float(CLEAR_LOSS_BITS)
        if clear_loss:
            status[f"{alpha:.2f}"] = "stopped_payload"
            promising = [item for item in promising if abs(float(item) - float(alpha)) > 1.0e-12]
    if promising:
        remain = tuple(scenario for scenario in EVAL_POOL if scenario not in SCREEN_POOL)
        full[f"{OFFICIAL_ALPHA:.2f}"] = baseline_rows + evaluate_scale(planner, float(OFFICIAL_ALPHA), remain)
        for alpha in promising:
            have = screen[f"{alpha:.2f}"]
            if len(have) == len(SCREEN_POOL):
                full[f"{alpha:.2f}"] = have + evaluate_scale(planner, float(alpha), remain)
            else:
                full[f"{alpha:.2f}"] = evaluate_scale(planner, float(alpha), EVAL_POOL)
    full_pairs = {
        key: paired_summary(full[f"{OFFICIAL_ALPHA:.2f}"], rows, draws=int(BOOTSTRAP_DRAWS), seed=int(BOOTSTRAP_SEED))
        for key, rows in full.items()
        if key != f"{OFFICIAL_ALPHA:.2f}"
    }
    supported = [key for key, pair in full_pairs.items() if _gain_supported(pair)]
    blocked = any(value == "blocked" for value in status.values())
    if blocked:
        outcome = "RESIDUAL_SCALE_AUDIT_BLOCKED"
    elif supported:
        outcome = "RESIDUAL_SCALE_GAIN_SUPPORTED"
    elif any(float(pair["mean"]) > 0.0 for pair in list(screen_pairs.values()) + list(full_pairs.values())):
        outcome = "RESIDUAL_SCALE_GAIN_INCONCLUSIVE"
    else:
        outcome = "RESIDUAL_SCALE_GAIN_NEGATIVE"
    summary = {
        "gates": ["RESIDUAL_MAPPING_AUDIT_COMPLETE", "RESIDUAL_SCALE_SCAN_COMPLETE", outcome],
        "training_started": False,
        "replaced_official_low_level": False,
        "source_sha256": before,
        "source_unchanged": int(_sha256(SOURCE_ACTOR) == before),
        "official_config": False,
        "screen_pool": list(SCREEN_POOL),
        "status": status,
        "screen_aggregate": {key: _aggregate(rows) for key, rows in screen.items()},
        "screen_paired": screen_pairs,
        "full_paired": full_pairs,
        "supported_alphas": supported,
    }
    assert_phase_output(OUTPUT / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert_phase_output(OUTPUT / "scenario_rows.json").write_text(
        json.dumps({"screen": screen, "full": full}),
        encoding="utf-8",
    )
    print(json.dumps({"gates": summary["gates"], "status": status, "supported": supported}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
